#!/usr/bin/env python
"""
Entrypoint for the A1/A2 analysis: parametric ANOVA electrode definition (A1) +
overlap / conjunction inference (A2). Sets up input args and calls
stability_flexibility_anova_conjunction_dcc.main().
Wrapped by sbatch_stability_flexibility_anova_conjunction_dcc.sh for the cluster.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, ALPHA,
N_PERM_NULL, THRESHOLDS) so you can rerun without editing Python.
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

from dcc_scripts.stats.stability_flexibility_anova_conjunction_dcc import main

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

# --- epochs / analysis window ---
EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading).")

WINDOW_TMIN = float(os.environ.get('WINDOW_TMIN', '0.0'))   # seconds post-stimulus
WINDOW_TMAX = float(os.environ.get('WINDOW_TMAX', '0.5'))

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
# Restrict to ROIs by providing a dict (see the sibling segregation run script
# for the LPFC/occipital sets); None keeps every channel.
ROIS_DICT = None

# --- A1/A2 hyperparameters ---
ALPHA = float(os.environ.get('ALPHA', '0.05'))
N_PERM_NULL = int(os.environ.get('N_PERM_NULL', '10000'))  # A2 overlap-null perms
SEED = int(os.environ.get('SEED', '0'))
# also cross-check A1's ANOVA flags against the nonparametric permutation
# definition (slower — runs per_electrode_labels). Off by default.
CROSSCHECK_NONPARAMETRIC = os.environ.get('CROSSCHECK_NONPARAMETRIC', '0') in ('1', 'true', 'True')
N_PERM_LABEL = int(os.environ.get('N_PERM_LABEL', '2000'))  # only used by the crosscheck

# threshold sweep q-cutoffs (comma-separated env override)
_thr_env = os.environ.get('THRESHOLDS')
THRESHOLDS = ([float(x) for x in _thr_env.split(',')] if _thr_env
              else [0.01, 0.05, 0.10, 0.20, 0.35, 0.50])

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else f'synthetic_rho{SYNTHETIC_RHO}'
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag,
                        f'anova_conjunction_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_{ELECTRODES}')


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = SimpleNamespace(
        timestamp=timestamp,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        data_source=DATA_SOURCE,
        synthetic_rho=SYNTHETIC_RHO,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        alpha=ALPHA,
        n_perm_null=N_PERM_NULL,
        seed=SEED,
        crosscheck_nonparametric=CROSSCHECK_NONPARAMETRIC,
        n_perm_label=N_PERM_LABEL,
        thresholds=THRESHOLDS,
        save_dir=SAVE_DIR,
    )

    print("=" * 70)
    print("STABILITY vs FLEXIBILITY — A1 ANOVA DEFINITION + A2 CONJUNCTION")
    print("=" * 70)
    print(f"Data source:      {DATA_SOURCE}" + (f" (rho={SYNTHETIC_RHO})" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES} | ROIs: {list(ROIS_DICT.keys()) if ROIS_DICT else 'all'}")
    print("-" * 70)
    print(f"alpha:            {ALPHA}")
    print(f"n_perm_null:      {N_PERM_NULL} | seed: {SEED}")
    print(f"thresholds:       {THRESHOLDS}")
    print(f"crosscheck:       {CROSSCHECK_NONPARAMETRIC}"
          + (f" (n_perm_label={N_PERM_LABEL})" if CROSSCHECK_NONPARAMETRIC else ""))
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
