#!/usr/bin/env python
"""
Entrypoint for A4 — cross-decoding of the stability/flexibility subpopulations
(label transfer + within-block 2x2 + temporal generalization). Sets up input args
and calls stability_flexibility_cross_decoding_dcc.main().
Wrapped by sbatch_stability_flexibility_cross_decoding_dcc.sh for the cluster.

A4 runs on the ordinary decoding pipeline — the ROI LabeledArray pseudopopulation,
`Decoder.cv_cm_jim_window_shuffle` with a second label vector, its refit shuffle
null, and the usual cluster correction across time windows. The hyperparameters
below are therefore the SAME ones the main decoding job uses (window size, step,
splits, repeats), not a separate pseudo-trial scheme.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, SYNTHETIC_CODE, WINDOW_TMIN, WINDOW_TMAX,
ELECTRODES, ALPHA, WINDOW_SIZE, STEP_SIZE, N_SPLITS, N_REPEATS, FRAC_TRAIN,
N_PERM, MIN_GROUP_SIZE, ROI) so you can rerun without editing Python.
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

from dcc_scripts.decoding.stability_flexibility_cross_decoding_dcc import main

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
# synthetic-only: 'shared' (should cross-decode) or 'orthogonal' (should not).
SYNTHETIC_CODE = os.environ.get('SYNTHETIC_CODE', 'shared')

# --- epochs / analysis window (real data only) ---
EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading).")

WINDOW_TMIN = float(os.environ.get('WINDOW_TMIN', '0.0'))   # seconds post-stimulus
WINDOW_TMAX = float(os.environ.get('WINDOW_TMAX', '0.5'))

# --- electrode selection + A1 electrode definition ---
# Two independent choices, easy to conflate:
#   ELECTRODES  = which electrodes are LOADED into the decoded ROI array at all.
#                 'sig' keeps the baseline task-significant ones in the ROIs,
#                 'all' keeps every electrode in the ROIs. (With ROIS_DICT=None
#                 no ROI/significance filter is applied and every channel is used.)
#   the GROUPS   = how those loaded electrodes are then split for the decodes:
#                 both / S_only / F_only from the interaction labels, plus the
#                 unselected REFERENCE_GROUP over all of them.
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None
ALPHA = float(os.environ.get('ALPHA', '0.05'))

# Which route defines the S/F electrode labels:
#   'anova'        -- one ANOVA per electrode on window-mean HG over [WINDOW_TMIN,
#                     WINDOW_TMAX], computed in this job (self-contained default).
#   'power_traces' -- read the finished within-electrode WINDOWED ANOVA runs and
#                     their permutation cluster correction, so the decoded sets
#                     are exactly the electrodes the power-trace figures call
#                     significant. More sensitive to transient interactions the
#                     window mean dilutes; requires the run directories below.
ELECTRODE_DEFINITION = os.environ.get('ELECTRODE_DEFINITION', 'anova')

# power_traces route: either ONE run whose ANOVA carried all four interactions,
#   POWER_TRACES_RUN_DIR=/path/to/run
# or one directory per interaction,
#   POWER_TRACES_CPC=... POWER_TRACES_SPS=... POWER_TRACES_CPS=... POWER_TRACES_SPC=...
_pt_single = os.environ.get('POWER_TRACES_RUN_DIR')
_pt_per_group = {g: os.environ.get(f'POWER_TRACES_{g}')
                 for g in ('CPC', 'SPS', 'CPS', 'SPC')}
_pt_per_group = {g: p for g, p in _pt_per_group.items() if p}
POWER_TRACES_RUNS = _pt_per_group or _pt_single
# 'fdr_bh' (BH across electrodes — the family a test that COUNTS electrodes
# needs), 'cluster' (raw cluster p, matching the existing lab convention), or 'none'.
POWER_TRACES_CORRECTION = os.environ.get('POWER_TRACES_CORRECTION', 'fdr_bh')
POWER_TRACES_ROI = os.environ.get('POWER_TRACES_ROI')       # None = don't filter by ROI

# The unselected reference set decoded alongside both/S_only/F_only: every
# electrode in the decoded ROI array (so ELECTRODES above decides whether that
# means "all" or "all baseline-significant"). Set REFERENCE_GROUP='' to drop it.
REFERENCE_GROUP = os.environ.get('REFERENCE_GROUP', 'all')

# Temporal generalization costs n_windows^2 decodes per matrix, so it runs on a
# chosen subset of the groups. TEMPGEN_GROUPS='both,all' adds the unselected
# reference matrix; TEMPGEN_GROUPS='' skips the design entirely.
TEMPGEN_GROUPS = tuple(g.strip() for g in
                       os.environ.get('TEMPGEN_GROUPS', 'both').split(',') if g.strip())

# --- decoding hyperparameters (the ordinary pipeline's) ---
WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', '20'))      # samples per decoding window
STEP_SIZE = int(os.environ.get('STEP_SIZE', '10'))          # window stride, in samples
N_SPLITS = int(os.environ.get('N_SPLITS', '5'))             # CV folds (or resamples, see below)
N_REPEATS = int(os.environ.get('N_REPEATS', '10'))          # CV repeats
EXPLAINED_VARIANCE = float(os.environ.get('EXPLAINED_VARIANCE', '0.8'))
N_PERM = int(os.environ.get('N_PERM', '500'))               # cluster-test permutations
MIN_GROUP_SIZE = int(os.environ.get('MIN_GROUP_SIZE', '5'))  # min electrodes per group
SEED = int(os.environ.get('SEED', '0'))
ROI = os.environ.get('ROI', 'all')                          # which ROI's LabeledArray to decode

# Proportion of trials used for TRAINING in each split. Unset (the default) keeps
# StratifiedKFold, i.e. (N_SPLITS-1)/N_SPLITS. Set it to sweep the train/test
# proportion directly — StratifiedShuffleSplit is used instead and N_SPLITS then
# counts random resamples per repeat rather than folds.
#   FRAC_TRAIN=0.5 bash submit_stability_flexibility_cross_decoding_dcc.sh
FRAC_TRAIN = os.environ.get('FRAC_TRAIN')
FRAC_TRAIN = float(FRAC_TRAIN) if FRAC_TRAIN else None

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else f'synthetic_{SYNTHETIC_CODE}'
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag,
                        f'cross_decoding_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_{ELECTRODES}')


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = SimpleNamespace(
        timestamp=timestamp,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        data_source=DATA_SOURCE,
        synthetic_code=SYNTHETIC_CODE,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        alpha=ALPHA,
        electrode_definition=ELECTRODE_DEFINITION,
        power_traces_runs=POWER_TRACES_RUNS,
        power_traces_correction=POWER_TRACES_CORRECTION,
        power_traces_roi=POWER_TRACES_ROI,
        reference_group=REFERENCE_GROUP,
        tempgen_groups=TEMPGEN_GROUPS,
        roi=ROI,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        explained_variance=EXPLAINED_VARIANCE,
        frac_train=FRAC_TRAIN,
        n_perm=N_PERM,
        min_group_size=MIN_GROUP_SIZE,
        seed=SEED,
        save_dir=SAVE_DIR,
    )

    print("=" * 72)
    print("STABILITY vs FLEXIBILITY — A4 CROSS-DECODING")
    print("=" * 72)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (code={SYNTHETIC_CODE})" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES}")
    print("-" * 72)
    print(f"Elec definition:  {ELECTRODE_DEFINITION}"
          + (f" (runs={POWER_TRACES_RUNS}, correction={POWER_TRACES_CORRECTION}, "
             f"roi={POWER_TRACES_ROI})" if ELECTRODE_DEFINITION == 'power_traces' else ""))
    print(f"Reference group:  {REFERENCE_GROUP or '(none)'} "
          f"| temporal gen on: {list(TEMPGEN_GROUPS) or '(none)'}")
    print(f"alpha (A1):       {ALPHA} | ROI: {ROI}")
    print(f"window/step:      {WINDOW_SIZE}/{STEP_SIZE} samples "
          f"| n_splits: {N_SPLITS} | n_repeats: {N_REPEATS}")
    print("train fraction:   "
          + (f"{FRAC_TRAIN} (StratifiedShuffleSplit)" if FRAC_TRAIN
             else f"{(N_SPLITS - 1) / N_SPLITS:.2f} (StratifiedKFold default)"))
    print(f"n_perm:           {N_PERM} | min_group_size: {MIN_GROUP_SIZE} | seed: {SEED}")
    print(f"Save dir:         {SAVE_DIR}")
    print("=" * 72)

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
