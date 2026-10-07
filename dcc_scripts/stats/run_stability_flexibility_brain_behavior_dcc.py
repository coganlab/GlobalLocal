#!/usr/bin/env python
"""
Entrypoint for A6 — brain-behavior correlation: the per-participant continuous
scores against behavioral LWPC/LWPS (raw and RT-adjusted, with reliabilities),
the label-based across-subject correlations, and the within-subject single-trial
mixed model, each with its cross-pairing specificity control. Sets up input args
and calls stability_flexibility_brain_behavior_dcc.main().
Wrapped by sbatch_stability_flexibility_brain_behavior_dcc.sh for the cluster.
How to run it and read the outputs: docs/a6_brain_behavior.md.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, ROIS, ALPHA,
BEHAVIOR_CSV, NEURAL_SUMMARY, RUN_TRIALWISE, MIN_ELEC,
PARTICIPANT_N_SPLITS, SEED, SYNTHETIC_N_SUBJ, SYNTHETIC_ACROSS_BETA,
SYNTHETIC_WITHIN_BETA, SYNTHETIC_CROSS_FRAC, SYNTHETIC_LINK,
SYNTHETIC_RT_COUPLING) so you can rerun without editing Python.
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
# synthetic-only, per-participant level: the planted across-participant
# brain-behavior correlation, and how many noise SDs of HG each SD of RT adds.
# SYNTHETIC_LINK=0 with SYNTHETIC_RT_COUPLING>0 is the RT confound on its own:
# the raw correlation comes out positive, the RT-adjusted one should not.
SYNTHETIC_LINK = float(os.environ.get('SYNTHETIC_LINK', '0.6'))
SYNTHETIC_RT_COUPLING = float(os.environ.get('SYNTHETIC_RT_COUPLING', '0.3'))

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
# Comma-separated names from src.analysis.config.rois, or 'all' for every channel
# (as in the segregation runner). ELECTRODES only takes effect when ROIS names a
# region: with 'all', resolve_electrodes_to_keep keeps every channel, which is why
# the archived run (4412 electrodes) kept every channel despite ELECTRODES=sig.
ROIS = os.environ.get('ROIS', 'lpfc')
from src.analysis.config.rois import select_rois
ROIS_DICT = select_rois(ROIS)

# --- A1 hyperparameter (the electrode definition A6 sits on) ---
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
FDR_CORRECTION = os.environ.get('FDR_CORRECTION', 'fdr_bh')
ALPHA = float(os.environ.get('ALPHA', '0.05'))

# --- A6 hyperparameters ---
# Behavioral LWPC / LWPS come from the subject-level effects table: the
# LWPC_effect / LWPS_effect of its key_RT_mean rows, one per subject.
BEHAVIOR_CSV = os.environ.get(
    'BEHAVIOR_CSV', os.path.join(project_root, 'src', 'config',
                                 'ieeg_behavioral_subject_level_effects.csv'))
# which label-based neural summary is starred at level (2):
# 'count' (n_S / n_F), 'frac' (proportion of the subject's electrodes), or
# 'effect' (mean interaction F). All three are computed.
NEURAL_SUMMARY = os.environ.get('NEURAL_SUMMARY', 'count')
# the within-subject single-trial level needs per-trial RT in the epochs metadata;
# set RUN_TRIALWISE=0 to skip it.
RUN_TRIALWISE = os.environ.get('RUN_TRIALWISE', '1') not in ('0', 'false', 'False')
# per-participant level: participants with fewer usable electrodes get no neural
# score, and PARTICIPANT_N_SPLITS shared trial splits give the reliabilities.
MIN_ELEC = int(os.environ.get('MIN_ELEC', '3'))
PARTICIPANT_N_SPLITS = int(os.environ.get('PARTICIPANT_N_SPLITS', '200'))
SEED = int(os.environ.get('SEED', '0'))

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else \
    f'synthetic_cross{SYNTHETIC_CROSS_FRAC}_link{SYNTHETIC_LINK}_rt{SYNTHETIC_RT_COUPLING}'
_roi_tag = 'all_rois' if ROIS_DICT is None else '-'.join(ROIS_DICT)
SAVE_DIR = os.path.join(current_script_dir, 'results', _tag,
                        f'brain_behavior_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_'
                        f'{ELECTRODES}_{_roi_tag}_{NEURAL_SUMMARY}')


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
        synthetic_link=SYNTHETIC_LINK,
        synthetic_rt_coupling=SYNTHETIC_RT_COUPLING,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        alpha=ALPHA,
        contrast_mode=CONTRAST_MODE,
        fdr_correction=FDR_CORRECTION,
        behavior_csv=BEHAVIOR_CSV,
        neural_summary=NEURAL_SUMMARY,
        run_trialwise=RUN_TRIALWISE,
        min_elec=MIN_ELEC,
        participant_n_splits=PARTICIPANT_N_SPLITS,
        seed=SEED,
        save_dir=SAVE_DIR,
    )

    print("=" * 78)
    print("STABILITY vs FLEXIBILITY — A6 BRAIN-BEHAVIOR")
    print("=" * 78)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (planted across_beta={SYNTHETIC_ACROSS_BETA}, "
             f"within_beta={SYNTHETIC_WITHIN_BETA}, "
             f"cross_frac={SYNTHETIC_CROSS_FRAC}, link={SYNTHETIC_LINK}, "
             f"rt_coupling={SYNTHETIC_RT_COUPLING})"
             if DATA_SOURCE == 'synthetic' else ""))
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Behavior CSV:     {BEHAVIOR_CSV} (subject-level LWPC_effect / LWPS_effect, RT)")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES} | ROIs: "
          f"{list(ROIS_DICT.keys()) if ROIS_DICT else 'all'}")
    print("-" * 78)
    print(f"contrast_mode:    {CONTRAST_MODE}")
    print(f"fdr_correction:   {FDR_CORRECTION}")
    print(f"alpha (A1):       {ALPHA}")
    print(f"per-participant:  min_elec={MIN_ELEC} | shared splits={PARTICIPANT_N_SPLITS}")
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
