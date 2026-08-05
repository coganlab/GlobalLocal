#!/usr/bin/env python
"""
Entrypoint for A6 — brain-behavior correlation: the across-subject correlation of
neural selectivity vs. the behavioral LWPC/LWPS RT effect, and the preferred
within-subject single-trial mixed model, each with its cross-pairing specificity
control. Sets up input args and calls
stability_flexibility_brain_behavior_dcc.main().
Wrapped by sbatch_stability_flexibility_brain_behavior_dcc.sh for the cluster.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, ALPHA,
BEHAVIOR_CSV, BEHAVIOR_RT_COL, NEURAL_SUMMARY, RUN_TRIALWISE, SEED,
SYNTHETIC_N_SUBJ, SYNTHETIC_ACROSS_BETA, SYNTHETIC_WITHIN_BETA,
SYNTHETIC_CROSS_FRAC) so you can rerun without editing Python.
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

from dcc_scripts.stats.stability_flexibility_brain_behavior_dcc import main

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

# --- data source: 'real' (epochs + behavioral CSV) or 'synthetic' (dry run) ---
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'real')
# synthetic-only: the planted coupling strengths. CROSS_FRAC is how much of each
# link leaks into the WRONG pairing — set it to 1.0 to destroy specificity and
# confirm `specificity_ok` turns False.
SYNTHETIC_N_SUBJ = int(os.environ.get('SYNTHETIC_N_SUBJ', '16'))
SYNTHETIC_ACROSS_BETA = float(os.environ.get('SYNTHETIC_ACROSS_BETA', '1.2'))
SYNTHETIC_WITHIN_BETA = float(os.environ.get('SYNTHETIC_WITHIN_BETA', '0.6'))
SYNTHETIC_CROSS_FRAC = float(os.environ.get('SYNTHETIC_CROSS_FRAC', '0.25'))

# --- epochs / analysis window (real data only) ---
EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading).")

WINDOW_TMIN = float(os.environ.get('WINDOW_TMIN', '0.0'))   # seconds post-stimulus
WINDOW_TMAX = float(os.environ.get('WINDOW_TMAX', '0.5'))

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None

# --- A1 hyperparameter (the electrode definition A6 sits on) ---
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
FDR_CORRECTION = os.environ.get('FDR_CORRECTION', 'fdr_bh')
ALPHA = float(os.environ.get('ALPHA', '0.05'))

# --- A6 hyperparameters ---
# Raw trial-level behavior. Defaults to the repo-root combinedData.csv; on the
# cluster point BEHAVIOR_CSV at the Box copy.
BEHAVIOR_CSV = os.environ.get(
    'BEHAVIOR_CSV', os.path.join(project_root, 'combinedData.csv'))
BEHAVIOR_RT_COL = os.environ.get('BEHAVIOR_RT_COL', 'RT')
# which per-subject neural summary headlines the across-subject level:
# 'count' (n_S / n_F), 'frac' (proportion of the subject's electrodes), or
# 'effect' (mean interaction F). All three are computed; this picks the primary.
NEURAL_SUMMARY = os.environ.get('NEURAL_SUMMARY', 'count')
# the within-subject single-trial level needs per-trial RT in the epochs metadata;
# set RUN_TRIALWISE=0 to run the across-subject level only.
RUN_TRIALWISE = os.environ.get('RUN_TRIALWISE', '1') not in ('0', 'false', 'False')
SEED = int(os.environ.get('SEED', '0'))

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else \
    f'synthetic_cross{SYNTHETIC_CROSS_FRAC}'
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag,
                        f'brain_behavior_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_'
                        f'{ELECTRODES}_{NEURAL_SUMMARY}')


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
        synthetic_across_beta=SYNTHETIC_ACROSS_BETA,
        synthetic_within_beta=SYNTHETIC_WITHIN_BETA,
        synthetic_cross_frac=SYNTHETIC_CROSS_FRAC,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        alpha=ALPHA,
        contrast_mode=CONTRAST_MODE,
        fdr_correction=FDR_CORRECTION,
        behavior_csv=BEHAVIOR_CSV,
        behavior_rt_col=BEHAVIOR_RT_COL,
        neural_summary=NEURAL_SUMMARY,
        run_trialwise=RUN_TRIALWISE,
        seed=SEED,
        save_dir=SAVE_DIR,
    )

    print("=" * 78)
    print("STABILITY vs FLEXIBILITY — A6 BRAIN-BEHAVIOR")
    print("=" * 78)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (planted across_beta={SYNTHETIC_ACROSS_BETA}, "
             f"within_beta={SYNTHETIC_WITHIN_BETA}, "
             f"cross_frac={SYNTHETIC_CROSS_FRAC})"
             if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Behavior CSV:     {BEHAVIOR_CSV}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES}")
    print("-" * 78)
    print(f"contrast_mode:    {CONTRAST_MODE}")
    print(f"fdr_correction:   {FDR_CORRECTION}")
    print(f"alpha (A1):       {ALPHA}")
    print(f"neural summary:   {NEURAL_SUMMARY} | trial-level: {RUN_TRIALWISE} | "
          f"seed: {SEED}")
    print(f"Save dir:         {SAVE_DIR}")
    print("=" * 78)

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
