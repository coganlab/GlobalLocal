#!/usr/bin/env python
"""
Entrypoint for A4 — cross-decoding of the stability/flexibility subpopulations
(pseudo-trials + label / set / temporal transfer). Sets up input args and calls
stability_flexibility_cross_decoding_dcc.main().
Wrapped by sbatch_stability_flexibility_cross_decoding_dcc.sh for the cluster.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, SYNTHETIC_CODE, WINDOW_TMIN, WINDOW_TMAX,
ELECTRODES, ALPHA, N_PER_CELL, N_PSEUDO, N_FOLDS, N_PERM, MIN_GROUP_SIZE) so you
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
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None
ALPHA = float(os.environ.get('ALPHA', '0.05'))

# --- A4 decoding hyperparameters ---
N_PER_CELL = int(os.environ.get('N_PER_CELL', '5'))         # trials averaged/pseudo-trial
N_PSEUDO = int(os.environ.get('N_PSEUDO', '80'))            # pseudo-trials per class-cell
N_FOLDS = int(os.environ.get('N_FOLDS', '5'))              # disjoint pseudo-trial folds
N_PERM = int(os.environ.get('N_PERM', '500'))             # label-permutation null draws
MIN_GROUP_SIZE = int(os.environ.get('MIN_GROUP_SIZE', '5'))  # min electrodes per group
SEED = int(os.environ.get('SEED', '0'))

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
        n_per_cell=N_PER_CELL,
        n_pseudo=N_PSEUDO,
        n_folds=N_FOLDS,
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
    print(f"alpha (A1):       {ALPHA}")
    print(f"n_per_cell:       {N_PER_CELL} | n_pseudo: {N_PSEUDO} | n_folds: {N_FOLDS}")
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
