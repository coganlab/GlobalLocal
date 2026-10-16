#!/usr/bin/env python
"""
Entrypoint for the coding-axes analysis (docs/analysis_plans.md, subspace plan
§3.1-3.4, §4 and §6: the coding axes and their cosine, the subspace test, the
trajectories and the block-split separations). Sets the parameters and calls coding_axes_dcc.main(). Wrapped by
sbatch_coding_axes_dcc.sh; the per-run choices come from submit_coding_axes_dcc.sh
as environment variables (EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOWS_TMIN,
WINDOWS_TMAX, WINDOW_SIZE, STEP_SIZE, REMOVE_MEAN, RT_CONTROL, RT_MATCH_BINS,
ELECTRODES, ROIS, N_SPLITS,
N_PERM, N_NULL_SPLITS, N_BOOT, ALPHA, SEED, K_MAX, TRAJ_AXES, TRAJ_FIRST).
"""
import sys
import os
from types import SimpleNamespace
from datetime import datetime

# ---------------------------------------------------------------------------
# PATH SETUP (detect cluster vs local, mirror the other run_* scripts)
# ---------------------------------------------------------------------------
if os.path.exists("/hpc/home"):
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dcc_scripts.subspace.coding_axes_dcc import main


def _flag(name, default='0'):
    return os.environ.get(name, default) not in ('0', '', 'false', 'False')


# ---------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ---------------------------------------------------------------------------
LAB_ROOT = None                      # auto-resolved in main()
TASK = 'GlobalLocal'
ACC_TRIALS_ONLY = True

SUBJECTS = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077', 'D0090',
            'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116',
            'D0117', 'D0121', 'D0133', 'D0134', 'D0137', 'D0138', 'D0139A',
            'D0144', 'D0145', 'D0146']

# --- data source: 'real' (epoched data) or 'synthetic' (validate pipeline) ---
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'real')
SYNTHETIC_RHO = float(os.environ.get('SYNTHETIC_RHO', '0.4'))
SYNTHETIC_SFREQ = 256.0              # synthetic samples per second, as the real data

EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading).")

# --- the window grid -----------------------------------------------------------
# Sliding windows tile [WINDOWS_TMIN, WINDOWS_TMAX] s, relative to the event the
# epochs file is locked to (Stimulus_ or Response_). The A4
# cross-decoding uses 64-sample windows every 16 samples at 256 Hz (250 ms every
# 62.5 ms) starting from -1 s; WINDOWS_TMIN on that grid (a multiple of 62.5 ms)
# gives the same windows.
WINDOWS_TMIN = float(os.environ.get('WINDOWS_TMIN', '0.0'))
WINDOWS_TMAX = float(os.environ.get('WINDOWS_TMAX', '1.5'))
WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', '64'))     # samples per window
STEP_SIZE = int(os.environ.get('STEP_SIZE', '16'))         # samples between starts

# REMOVE_MEAN=1: subtract each participant's mean across its electrodes, per
# trial and window, before scoring (A4's remove_mean; the overall-activity
# control the plan says to run first, §3.3).
REMOVE_MEAN = _flag('REMOVE_MEAN')

# RT_CONTROL (§3.3), applied to the trials before scoring:
#   none   -- no control [default]
#   adjust -- regress each electrode's pooled within-cell HG~RT slope out of
#             every window (as RT_ADJUST_HG in the segregation run)
#   match  -- keep trials whose RT distributions match across the four
#             congruency x switch-type cells, per participant (A4's RT_MATCH=rt)
#   random -- match's control: as many trials per participant and cell, at random
# RT_MATCH_BINS: quantile RT bins per participant for match / random.
RT_CONTROL = os.environ.get('RT_CONTROL', 'none').strip().lower()
RT_MATCH_BINS = int(os.environ.get('RT_MATCH_BINS', '10'))
if RT_CONTROL not in ('none', 'adjust', 'match', 'random'):
    raise ValueError(f"RT_CONTROL must be none, adjust, match or random; got {RT_CONTROL!r}")

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS = os.environ.get('ROIS', 'lpfc')                       # config ROI names, or 'all'
from src.analysis.config.rois import select_rois
ROIS_DICT = select_rois(ROIS)

# --- splits, null, bootstrap ---
N_SPLITS = int(os.environ.get('N_SPLITS', '200'))           # split halves, real labels
N_PERM = int(os.environ.get('N_PERM', '1000'))              # shuffles per factor
N_NULL_SPLITS = int(os.environ.get('N_NULL_SPLITS', '20'))  # split halves per shuffle
N_BOOT = int(os.environ.get('N_BOOT', '2000'))              # participant resamples
ALPHA = float(os.environ.get('ALPHA', '0.05'))
SEED = int(os.environ.get('SEED', '0'))                     # every other seed derives from it

# --- subspace test (§3.4) ---
# K_MAX: subspace dimensions compared (k = 1..K_MAX) and tested for
# reproducibility; the primary k is the largest at which both variables'
# leading dimensions all beat their shuffle null. The plan reports k = 1-3.
K_MAX = int(os.environ.get('K_MAX', '3'))

# --- trajectories (§4) ---
# TRAJ_AXES: the windows (s) the trajectory axes c and s are averaged over,
# 'tmin:tmax' comma-separated, one figure each. The first is the main figure
# (0-1.5 s, the anatomy's window); the other two are the early-vs-late variant.
# TRAJ_FIRST: the axis Gram-Schmidt keeps whole (x); the other becomes y.
TRAJ_AXES = [tuple(float(v) for v in w.split(':'))
             for w in os.environ.get('TRAJ_AXES', '0.0:1.5,0.1:0.5,0.6:1.4').split(',')]
TRAJ_FIRST = os.environ.get('TRAJ_FIRST', 'congruency').strip().lower()
if TRAJ_FIRST not in ('congruency', 'switch'):
    raise ValueError(f"TRAJ_FIRST must be congruency or switch; got {TRAJ_FIRST!r}")

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else f'synthetic_rho{SYNTHETIC_RHO}'
_roi_tag = 'all_rois' if ROIS_DICT is None else '-'.join(ROIS_DICT)
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag, 'coding_axes',
                        f'windows_{WINDOWS_TMIN}to{WINDOWS_TMAX}s_w{WINDOW_SIZE}s{STEP_SIZE}'
                        f'_{ELECTRODES}_{_roi_tag}_mean_diff'
                        f'_splits{N_SPLITS}_perm{N_PERM}x{N_NULL_SPLITS}_seed{SEED}'
                        + ('_remove_mean' if REMOVE_MEAN else '')
                        + {'none': '', 'adjust': '_rt_adjust'}.get(
                            RT_CONTROL, f'_rt_{RT_CONTROL}{RT_MATCH_BINS}'))


def run_analysis():
    args = SimpleNamespace(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        data_source=DATA_SOURCE,
        synthetic_rho=SYNTHETIC_RHO,
        synthetic_sfreq=SYNTHETIC_SFREQ,
        epochs_root_file=EPOCHS_ROOT_FILE,
        windows_tmin=WINDOWS_TMIN,
        windows_tmax=WINDOWS_TMAX,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        remove_mean=REMOVE_MEAN,
        rt_control=RT_CONTROL,
        rt_match_bins=RT_MATCH_BINS,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        n_splits=N_SPLITS,
        n_perm=N_PERM,
        n_null_splits=N_NULL_SPLITS,
        n_boot=N_BOOT,
        alpha=ALPHA,
        seed=SEED,
        k_max=K_MAX,
        traj_axes=TRAJ_AXES,
        traj_first=TRAJ_FIRST,
        save_dir=SAVE_DIR,
    )

    print("=" * 70)
    print("CODING AXES, SUBSPACE TEST, TRAJECTORIES (subspace plan §3.1-3.4, §4)")
    print("=" * 70)
    print(f"Data source:      {DATA_SOURCE}" + (f" (rho={SYNTHETIC_RHO})" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Windows:          {WINDOW_SIZE} samples every {STEP_SIZE} over [{WINDOWS_TMIN}, {WINDOWS_TMAX}] s")
    print(f"Remove mean:      {REMOVE_MEAN}")
    print(f"RT control:       {RT_CONTROL}" + (f" ({RT_MATCH_BINS} bins)" if RT_CONTROL in ('match', 'random') else ""))
    print(f"Electrodes:       {ELECTRODES} | ROIs: {list(ROIS_DICT.keys()) if ROIS_DICT else 'all'}")
    print(f"n_splits:         {N_SPLITS} | null: {N_PERM} x {N_NULL_SPLITS} splits per factor")
    print(f"n_boot:           {N_BOOT} | alpha: {ALPHA} | seed: {SEED}")
    print(f"Subspace k_max:   {K_MAX}")
    print(f"Trajectory axes:  {TRAJ_AXES} s | first: {TRAJ_FIRST}")
    print(f"Save dir:         {SAVE_DIR}")
    print("=" * 70)

    try:
        main(args)
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_analysis()
