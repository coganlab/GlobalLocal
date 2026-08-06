#!/usr/bin/env python
"""
Entrypoint for the conjunction on `power_traces`-defined electrodes.
Sets up input args and calls power_traces_conjunction_dcc.main().
Wrapped by sbatch_power_traces_conjunction_dcc.sh for cluster submission.

The electrode definition is READ BACK from a finished within-electrode windowed
ANOVA + cluster-correction run (`dcc_scripts/power/run_power_traces_dcc.py` with
ANOVA_UNIT='electrode'), so this job does not re-fit anything unless the optional
continuous confound control is enabled.

Everything is overridable from the submit script via environment variables:

    PT_RUN            one run directory whose ANOVA held all four factors
                      (stimulus_experiment_conditions)
    PT_RUN_CPC/_SPS/_CPS/_SPC
                      or four separate two-factor runs (CPC and SPS required)
    ROIS              comma-separated ROI names, or 'all' to pool
    ALPHA, CORRECTION, EFFECT_MODE, N_PERM_NULL, THRESHOLDS, REQUIRE_ALL, USE_NPZ
    RUN_CONTINUOUS    1 to also run the continuous confound control (needs epochs)
    EPOCHS_ROOT_FILE, N_SPLITS, N_PERM_CORR, MIN_ELEC, EFFECT_MEASURES, ELECTRODES
    DATA_SOURCE       'real' (default) or 'synthetic' (fabricates the runs on disk)
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

from dcc_scripts.stats.power_traces_conjunction_dcc import main

# ---------------------------------------------------------------------------
# DATA SOURCE
# ---------------------------------------------------------------------------
# 'real'      -> read the run directories named below
# 'synthetic' -> fabricate four runs on disk with a KNOWN planted overlap, so the
#                read -> label -> conjunction path can be validated in seconds.
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'real')
SYNTHETIC_OVERLAP = float(os.environ.get('SYNTHETIC_OVERLAP', '0.6'))
SYNTHETIC_BASE_RATE = float(os.environ.get('SYNTHETIC_BASE_RATE', '0.25'))
SYNTHETIC_RHO = float(os.environ.get('SYNTHETIC_RHO', '0.4'))   # continuous control

# ---------------------------------------------------------------------------
# WHICH power_traces RUN(S) DEFINE THE ELECTRODES
# ---------------------------------------------------------------------------
# A run directory is the one holding `summary.csv` + `run_config.json`, i.e.
#   <dcc_scripts/power/figs>/<EPOCHS_ROOT_FILE>/anova_within_electrode/<conditions_save_name>
# e.g. .../anova_within_electrode/stimulus_experiment_conditions_24_subjects
PT_RUN = os.environ.get('PT_RUN')            # single 4-factor run, or None

# ... or four separate two-factor runs. Only CPC and SPS are required; the two
# cross runs are the specificity controls and may be omitted.
PT_RUN_DIRS = {
    'CPC': os.environ.get('PT_RUN_CPC'),     # congruency x incongruentProportion (LWPC)
    'SPS': os.environ.get('PT_RUN_SPS'),     # switchType x switchProportion      (LWPS)
    'CPS': os.environ.get('PT_RUN_CPS'),     # congruency x switchProportion      (cross)
    'SPC': os.environ.get('PT_RUN_SPC'),     # switchType x incongruentProportion (cross)
}
PT_RUN_DIRS = {k: v for k, v in PT_RUN_DIRS.items() if v}

# ---------------------------------------------------------------------------
# LABEL / CONJUNCTION PARAMETERS
# ---------------------------------------------------------------------------
# ROIs to analyse. The labels are per-ROI, and so is the electrode universe --
# note a subject with no electrodes in the ROI drops out of the CMH entirely.
# 'all' runs ONE analysis pooled over every ROI in the run.
ROIS = os.environ.get('ROIS', 'lpfc')
ROIS = 'all' if ROIS.strip() == 'all' else [r.strip() for r in ROIS.split(',') if r.strip()]

# correction across ELECTRODES, which is the family a count test needs:
#   'fdr_bh'  BH within (roi, effect)                     [default, recommended]
#   'cluster' raw cluster p < alpha, no across-electrode correction -- the
#             like-for-like port of what load_significant_electrodes does today
#   'none'    any surviving cluster counts
CORRECTION = os.environ.get('CORRECTION', 'fdr_bh')
ALPHA = float(os.environ.get('ALPHA', '0.05'))
# 'interaction': LWPC x LWPS; 'main': congruency x switch-type main effects.
EFFECT_MODE = os.environ.get('EFFECT_MODE', 'interaction')
if EFFECT_MODE not in ('interaction', 'main'):
    raise ValueError("EFFECT_MODE must be 'interaction' or 'main'")

# Keep only electrodes present in EVERY requested run. Different runs can end up
# with different electrode sets (min_trials_per_cell), and a 2x2 built over
# inconsistent denominators is not interpretable.
REQUIRE_ALL = os.environ.get('REQUIRE_ALL', '1') not in ('0', 'false', 'False')

# Only consulted for legacy runs whose summary.csv predates `best_cluster_p`:
# recompute a graded cluster p for every electrode from the saved .npz null.
USE_NPZ = os.environ.get('USE_NPZ', '1') not in ('0', 'false', 'False')

N_PERM_NULL = int(os.environ.get('N_PERM_NULL', '10000'))
THRESHOLDS = [float(t) for t in
              os.environ.get('THRESHOLDS', '0.01,0.025,0.05,0.10').split(',')]
SEED = int(os.environ.get('SEED', '0'))

# ---------------------------------------------------------------------------
# CONTINUOUS CONFOUND CONTROL (optional; the only part that loads epochs)
# ---------------------------------------------------------------------------
# The count test above cannot correct for (a) per-electrode detection power or
# (b) shared trial noise, and both bias it toward "shared". This re-estimates the
# two sensitivities on DISJOINT trial halves, residualised on responsiveness,
# over the SAME window the run tiled, under all three effect measures. It is the
# control on the counts, not a second headline (guide §5.1).
RUN_CONTINUOUS = os.environ.get('RUN_CONTINUOUS', '0') not in ('0', 'false', 'False')

TASK = 'GlobalLocal'
LAB_ROOT = None                      # auto-resolved in main()
ACC_TRIALS_ONLY = True
SUBJECTS = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077', 'D0090',
            'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116',
            'D0117', 'D0121', 'D0133', 'D0134', 'D0137', 'D0138', 'D0139A',
            'D0144', 'D0145', 'D0146']

EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if RUN_CONTINUOUS and DATA_SOURCE == 'real' and EPOCHS_ROOT_FILE is None:
    raise ValueError("RUN_CONTINUOUS=1 needs EPOCHS_ROOT_FILE (the continuous "
                     "control re-estimates from single trials). Set it via "
                     "sbatch --export=ALL,EPOCHS_ROOT_FILE=... , or leave "
                     "RUN_CONTINUOUS=0 to run the count test alone.")

ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
# ROIS_DICT gates which channels are assembled for the continuous control. It has
# no effect on the counts -- those come from the run's own ROI assignment.
ROIS_DICT = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul",
             "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont",
             "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup",
             "S_front_inf", "S_front_middle", "S_front_sup"],
    'occ':  ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle",
             "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual",
             "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus",
             "S_oc_sup_and_transversal", "S_occipital_ant"],
}
RESPONSIVENESS = None      # prefer a {electrode: baseline-vs-signal cluster stat}
N_SPLITS = int(os.environ.get('N_SPLITS', '200'))
N_PERM_CORR = int(os.environ.get('N_PERM_CORR', '10000'))
MIN_ELEC = int(os.environ.get('MIN_ELEC', '3'))
EFFECT_MEASURES = [m.strip() for m in
                   os.environ.get('EFFECT_MEASURES', 'peak_t,cluster,cohens_d').split(',')
                   if m.strip()]

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
if DATA_SOURCE == 'synthetic':
    _tag = f'synthetic_overlap{SYNTHETIC_OVERLAP}_base{SYNTHETIC_BASE_RATE}'
else:
    _ref = PT_RUN if PT_RUN else (PT_RUN_DIRS.get('CPC') or 'unknown_run')
    _tag = os.path.basename(os.path.normpath(_ref))
SAVE_DIR = os.path.join(current_script_dir, 'results', EPOCHS_ROOT_FILE, 'power_traces_conjunction_results',
                        _tag, f'{CORRECTION}_alpha{ALPHA}')
if EFFECT_MODE == 'main':
    # Keep existing interaction output paths stable and prevent main-effect runs
    # from overwriting them.
    SAVE_DIR = os.path.join(SAVE_DIR, 'main_effects')


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = SimpleNamespace(
        timestamp=timestamp,
        data_source=DATA_SOURCE,
        synthetic_overlap=SYNTHETIC_OVERLAP,
        synthetic_base_rate=SYNTHETIC_BASE_RATE,
        synthetic_rho=SYNTHETIC_RHO,
        run_dir=PT_RUN,
        run_dirs=PT_RUN_DIRS,
        rois=ROIS,
        alpha=ALPHA,
        correction=CORRECTION,
        effect_mode=EFFECT_MODE,
        require_all=REQUIRE_ALL,
        use_npz=USE_NPZ,
        n_perm_null=N_PERM_NULL,
        thresholds=THRESHOLDS,
        seed=SEED,
        # continuous confound control
        run_continuous=RUN_CONTINUOUS,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        epochs_root_file=EPOCHS_ROOT_FILE,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        responsiveness=RESPONSIVENESS,
        n_splits=N_SPLITS,
        n_perm_corr=N_PERM_CORR,
        min_elec=MIN_ELEC,
        effect_measures=EFFECT_MEASURES,
        save_dir=SAVE_DIR,
    )

    print("=" * 72)
    print("CONJUNCTION ON power_traces CLUSTER-CORRECTED ELECTRODES")
    print("=" * 72)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (overlap={SYNTHETIC_OVERLAP}, base={SYNTHETIC_BASE_RATE})"
             if DATA_SOURCE == 'synthetic' else ""))
    if DATA_SOURCE == 'real':
        if PT_RUN:
            print(f"Run (4-factor):   {PT_RUN}")
        for g, p in sorted(PT_RUN_DIRS.items()):
            print(f"Run {g}:           {p}")
    print(f"ROIs:             {ROIS}")
    print(f"Correction:       {CORRECTION} | alpha: {ALPHA}")
    print(f"Effect mode:      {EFFECT_MODE}")
    print(f"require_all:      {REQUIRE_ALL} | use_npz (legacy runs): {USE_NPZ}")
    print(f"n_perm_null:      {N_PERM_NULL} | thresholds: {THRESHOLDS}")
    print("-" * 72)
    print(f"Continuous control: {RUN_CONTINUOUS}")
    if RUN_CONTINUOUS:
        print(f"  Epochs file:    {EPOCHS_ROOT_FILE}")
        print(f"  Electrodes:     {ELECTRODES} | ROIs: {list(ROIS_DICT.keys()) if ROIS_DICT else 'all'}")
        print(f"  Effect measures:{EFFECT_MEASURES}")
        print(f"  n_splits:       {N_SPLITS} | n_perm_corr: {N_PERM_CORR} | min_elec: {MIN_ELEC}")
        print("  Window is taken from the run's run_config.json, NOT from "
              "WINDOW_TMIN/TMAX -- a control that covers a different stretch of "
              "the epoch than the counts is not a control.")
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
