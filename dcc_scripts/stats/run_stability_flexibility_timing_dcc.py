#!/usr/bin/env python
"""
Entrypoint for A5 — relative onset of stability (LWPC) vs. flexibility (LWPS)
information: per-bin interaction time courses, 50%-of-peak onsets, and the
Ulrich-Miller jackknifed onset-difference test. Sets up input args and calls
stability_flexibility_timing_dcc.main().
Wrapped by sbatch_stability_flexibility_timing_dcc.sh for the cluster.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, STATISTIC,
ALPHA, SEED, SYNTHETIC_N_SUBJ, SYNTHETIC_STAB_ONSET, SYNTHETIC_FLEX_ONSET) so you
can rerun without editing Python.
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

from dcc_scripts.stats.stability_flexibility_timing_dcc import main

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

# --- data source: 'real' (epoched HG time courses) or 'synthetic' (dry run) ---
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'real')
# synthetic-only: the planted onsets (seconds). Swapping these plants the REVERSE
# ordering, which is the falsification check — the reported sign should flip.
SYNTHETIC_N_SUBJ = int(os.environ.get('SYNTHETIC_N_SUBJ', '12'))
SYNTHETIC_STAB_ONSET = float(os.environ.get('SYNTHETIC_STAB_ONSET', '0.20'))
SYNTHETIC_FLEX_ONSET = float(os.environ.get('SYNTHETIC_FLEX_ONSET', '0.40'))

# --- epochs / analysis window (real data only) ---
EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading).")

# A5 reads a RISING FLANK, so the window must contain the pre-onset baseline and
# enough post-stimulus time for both effects to peak — wider than the A1/A2
# window-mean default on purpose.
WINDOW_TMIN = float(os.environ.get('WINDOW_TMIN', '-0.2'))   # seconds post-stimulus
WINDOW_TMAX = float(os.environ.get('WINDOW_TMAX', '0.8'))

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None

# --- A5 hyperparameters ---
# 'mean' = grand-average the per-electrode d-o-d(t) (the information time course);
# 't'    = noise-normalized t across electrodes, often a cleaner rising flank.
STATISTIC = os.environ.get('STATISTIC', 'mean')
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
ALPHA = float(os.environ.get('ALPHA', '0.05'))
SEED = int(os.environ.get('SEED', '0'))

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else \
    f'synthetic_onsets{SYNTHETIC_STAB_ONSET}v{SYNTHETIC_FLEX_ONSET}'
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag,
                        f'timing_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_'
                        f'{ELECTRODES}_{STATISTIC}')


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = SimpleNamespace(
        timestamp=timestamp,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        data_source=DATA_SOURCE,
        synthetic_n_subj=SYNTHETIC_N_SUBJ,
        synthetic_stab_onset=SYNTHETIC_STAB_ONSET,
        synthetic_flex_onset=SYNTHETIC_FLEX_ONSET,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        statistic=STATISTIC,
        alpha=ALPHA,
        contrast_mode=CONTRAST_MODE,
        seed=SEED,
        save_dir=SAVE_DIR,
    )

    print("=" * 70)
    print("STABILITY vs FLEXIBILITY — A5 TIMING (relative onset)")
    print("=" * 70)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (planted onsets LWPC={SYNTHETIC_STAB_ONSET}s, "
             f"LWPS={SYNTHETIC_FLEX_ONSET}s)" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES}")
    print("-" * 70)
    print(f"statistic:        {STATISTIC} | alpha: {ALPHA} | seed: {SEED}")
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
