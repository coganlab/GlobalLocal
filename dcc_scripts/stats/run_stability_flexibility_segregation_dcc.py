#!/usr/bin/env python
"""
Entrypoint for the stability vs. flexibility segregation analysis.
Sets up input args and calls stability_flexibility_segregation_dcc.main().
Wrapped by sbatch_stability_flexibility_segregation_dcc.sh for cluster submission.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, ROIS,
N_SPLITS, N_PERM_CORR, N_PERM_LABEL, FDR_CORRECTION) so you can rerun without editing Python.
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

from dcc_scripts.stats.stability_flexibility_segregation_dcc import main

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

# --- window alignment with a power_traces electrode-definition run -----------
# When the 2x2 counts come from the cluster-corrected windowed ANOVA
# (`power_traces_conjunction`), this continuous correlation is the CONFOUND
# CONTROL for those counts, not a second headline: it is the only route that
# residualises on responsiveness and estimates the two sensitivities on disjoint
# trial halves, which are the two things the count test cannot control.
#
# It stays an independent estimate -- different estimator, different temporal
# model -- but a control that covers a different stretch of the epoch than the
# thing it is controlling is not a control at all. Setting
# ALIGN_TO_POWER_TRACES_RUN to a windowed-ANOVA run directory overrides
# WINDOW_TMIN/WINDOW_TMAX with the extent that run's windows actually tiled.
#
#   ALIGN_TO_POWER_TRACES_RUN=/path/to/within_elec_anova/<run_label> \
#       bash submit_stability_flexibility_segregation_dcc.sh
ALIGN_TO_POWER_TRACES_RUN = os.environ.get('ALIGN_TO_POWER_TRACES_RUN')
_ALIGNMENT_NOTE = None
if ALIGN_TO_POWER_TRACES_RUN:
    from src.analysis.stats.power_traces_conjunction import (
        analysis_window_from_run, describe_window_alignment)
    _ALIGNMENT_NOTE = describe_window_alignment(
        ALIGN_TO_POWER_TRACES_RUN, WINDOW_TMIN, WINDOW_TMAX)
    WINDOW_TMIN, WINDOW_TMAX = analysis_window_from_run(ALIGN_TO_POWER_TRACES_RUN)

# --- analysis options (default to the original behaviour) ---
# contrast_mode : 'condition'  -> stability=congruency(i vs c), flexibility=switchType(s vs r)
#                 'proportion' -> stability=incongruent_proportion, flexibility=switch_proportion
# effect_measure: 'cohens_d'   -> standardized mean diff on window-mean HG
#                 'cluster'     -> aggregate cluster-mass statistic on windowed HG time courses
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
EFFECT_MEASURE = os.environ.get('EFFECT_MEASURE', 'cluster')

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
# Comma-separated names from src.analysis.config.rois, or 'all' to keep every
# channel. This mirrors the ROIS interface of power_traces_conjunction_dcc.
ROIS = os.environ.get('ROIS', 'lpfc')
from src.analysis.config.rois import select_rois
ROIS_DICT = select_rois(ROIS)

# --- responsiveness (gain control). None -> mean|HG| fallback inside the analysis.
# Prefer passing a {electrode: baseline-vs-signal cluster stat} dict here.
RESPONSIVENESS = None

# --- analysis hyperparameters ---
N_SPLITS = int(os.environ.get('N_SPLITS', '200'))          # disjoint-half resamples
N_PERM_CORR = int(os.environ.get('N_PERM_CORR', '10000'))  # continuous-test perms
N_PERM_LABEL = int(os.environ.get('N_PERM_LABEL', '2000')) # per-electrode label perms
FDR_CORRECTION = os.environ.get('FDR_CORRECTION', 'fdr_bh')
ALPHA = float(os.environ.get('ALPHA', '0.05'))
MIN_ELEC = int(os.environ.get('MIN_ELEC', '3'))            # min electrodes/subject

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else f'synthetic_rho{SYNTHETIC_RHO}'
_roi_tag = 'all_rois' if ROIS_DICT is None else '-'.join(ROIS_DICT)
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag, 'segregation_results',
                        f'window_{WINDOW_TMIN}to{WINDOW_TMAX}s_{ELECTRODES}'
                        f'_{_roi_tag}_{CONTRAST_MODE}_{EFFECT_MEASURE}_{FDR_CORRECTION}')

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
        responsiveness=RESPONSIVENESS,
        contrast_mode=CONTRAST_MODE,
        effect_measure=EFFECT_MEASURE,
        n_splits=N_SPLITS,
        n_perm_corr=N_PERM_CORR,
        n_perm_label=N_PERM_LABEL,
        alpha=ALPHA,
        fdr_correction=FDR_CORRECTION,
        min_elec=MIN_ELEC,
        save_dir=SAVE_DIR,
    )

    print("=" * 70)
    print("STABILITY vs FLEXIBILITY SEGREGATION")
    print("=" * 70)
    print(f"Data source:      {DATA_SOURCE}" + (f" (rho={SYNTHETIC_RHO})" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    if _ALIGNMENT_NOTE:
        print("Window aligned to a power_traces electrode-definition run; this "
              "continuous correlation is the CONFOUND CONTROL for that run's "
              "2x2 counts (responsiveness residualisation + disjoint trial "
              "halves), not an independent headline result.")
        for _line in _ALIGNMENT_NOTE.splitlines():
            print(f"  {_line}")
    print(f"Electrodes:       {ELECTRODES} | ROIs: {list(ROIS_DICT.keys()) if ROIS_DICT else 'all'}")
    print(f"Contrast mode:    {CONTRAST_MODE}")
    print(f"Effect measure:   {EFFECT_MEASURE}")
    print("-" * 70)
    print(f"n_splits:         {N_SPLITS}")
    print(f"n_perm_corr:      {N_PERM_CORR}")
    print(f"n_perm_label:     {N_PERM_LABEL}")
    print(f"alpha:            {ALPHA} | min_elec: {MIN_ELEC}")
    print(f"fdr_correction:   {FDR_CORRECTION}")
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
