#!/usr/bin/env python
"""
Entrypoint for A3 — anatomy of the stability/flexibility subpopulations
(brain maps + ROI histograms + coverage-conditioned enrichment test). Sets up
input args and calls stability_flexibility_anatomy_dcc.main().
Wrapped by sbatch_stability_flexibility_anatomy_dcc.sh for the cluster.

Most knobs can be overridden from the submit script via environment variables
(EPOCHS_ROOT_FILE, DATA_SOURCE, WINDOW_TMIN, WINDOW_TMAX, ELECTRODES, ALPHA,
MIN_SUBJECTS, N_PERM, SYNTHETIC_ENRICHMENT) so you can rerun without editing
Python.

Choosing the electrode definition
---------------------------------
LABEL_SOURCE=a1            (default) fit the A1 window-mean interaction ANOVA on
                           the epoched data. Needs EPOCHS_ROOT_FILE.
LABEL_SOURCE=power_traces  read finished within-electrode windowed-ANOVA runs
                           (cluster-corrected) instead. Needs the run dirs:
                             PT_RUN_DIR         one 4-factor run with all four
                                                interactions, OR
                             PT_RUN_CPC/PT_RUN_SPS (+ optional PT_RUN_CPS/
                                                PT_RUN_SPC) for separate runs.
                           Optional: PT_CORRECTION (fdr_bh|cluster|none),
                           PT_ALPHA, PT_ROI (the ANOVA's ROI, e.g. lpfc).
                           No epoched data is loaded on this route.

Restricting the anatomy
-----------------------
ROI_FILTER=lpfc   keep only electrodes whose Destrieux label falls in the `lpfc`
                  group of src/analysis/config/rois.py (any key of that dict, or
                  a comma-separated list). Unset = whole brain.
ANAT_LEVEL        auto (default) | group | destrieux. Which anatomical level the
                  histogram + enrichment test run on. `auto` switches to the raw
                  Destrieux labels whenever the analysis sits inside one ROI
                  group, where the coarse label is constant and uninformative.
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

from dcc_scripts.stats.stability_flexibility_anatomy_dcc import main

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

# --- data source: 'real' (epoched data + ROI atlas) or 'synthetic' (dry run) ---
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'real')
# synthetic-only: strength of the planted group×ROI association (0 = null).
SYNTHETIC_ENRICHMENT = float(os.environ.get('SYNTHETIC_ENRICHMENT', '0.6'))

# --- electrode definition: 'a1' (window-mean ANOVA here) or 'power_traces'
#     (finished cluster-corrected within-electrode ANOVA runs) ---
LABEL_SOURCE = os.environ.get('LABEL_SOURCE', 'a1')

# power_traces route: where the finished runs live. Either ONE run whose ANOVA
# had all four interaction terms, or one run per interaction.
PT_RUN_DIR = os.environ.get('PT_RUN_DIR')
PT_RUNS = {k: os.environ.get(f'PT_RUN_{k}') for k in ('CPC', 'SPS', 'CPS', 'SPC')}
PT_RUNS = {k: v for k, v in PT_RUNS.items() if v}
if LABEL_SOURCE == 'power_traces' and DATA_SOURCE != 'synthetic':
    if PT_RUN_DIR:
        PT_RUNS = PT_RUN_DIR                     # single 4-factor run
    elif not {'CPC', 'SPS'} <= set(PT_RUNS):
        raise ValueError(
            "LABEL_SOURCE=power_traces needs the finished ANOVA run directories: "
            "either PT_RUN_DIR=<one 4-factor run> or both PT_RUN_CPC=... and "
            "PT_RUN_SPS=... (optionally PT_RUN_CPS / PT_RUN_SPC for the cross "
            "controls). These are the dirs written by run_power_traces_dcc.py, "
            "i.e. the ones containing summary.csv.")
else:
    PT_RUNS = PT_RUN_DIR or PT_RUNS or None

# BH across electrodes within (roi, effect) — the right family for a test that
# counts electrodes. 'cluster' reproduces the raw lab convention instead.
PT_CORRECTION = os.environ.get('PT_CORRECTION', 'fdr_bh')
PT_ALPHA = os.environ.get('PT_ALPHA')            # defaults to ALPHA below
PT_ROI = os.environ.get('PT_ROI')                # the ANOVA run's ROI, e.g. 'lpfc'

# --- epochs / analysis window (A1 route on real data only) ---
EPOCHS_ROOT_FILE = os.environ.get('EPOCHS_ROOT_FILE')
if DATA_SOURCE == 'real' and LABEL_SOURCE == 'a1' and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading, "
                     "or LABEL_SOURCE=power_traces to read finished ANOVA runs).")

WINDOW_TMIN = float(os.environ.get('WINDOW_TMIN', '0.0'))   # seconds post-stimulus
WINDOW_TMAX = float(os.environ.get('WINDOW_TMAX', '0.5'))

# --- electrode selection ---
ELECTRODES = os.environ.get('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None

# --- A1 hyperparameter (the electrode definition A3 sits on) ---
ALPHA = float(os.environ.get('ALPHA', '0.05'))

# --- A3 hyperparameters ---
# keep only ROIs sampled in >= MIN_SUBJECTS subjects (the coverage condition).
MIN_SUBJECTS = int(os.environ.get('MIN_SUBJECTS', '3'))
N_PERM = int(os.environ.get('N_PERM', '10000'))            # within-subject perms
SEED = int(os.environ.get('SEED', '0'))
# where the electrodes-to-ROIs atlas json is cached (real data only).
ROI_DICT_DIR = os.environ.get('ROI_DICT_DIR')

# --- anatomical scope ---
# Restrict the anatomy to one (or several, comma-separated) ROI groups from
# src/analysis/config/rois.py, e.g. 'lpfc'. Empty = whole brain.
_roi_filter = os.environ.get('ROI_FILTER', '').strip()
ROI_FILTER = None
if _roi_filter:
    parts = [p.strip() for p in _roi_filter.split(',') if p.strip()]
    ROI_FILTER = parts[0] if len(parts) == 1 else parts
# 'auto' | 'group' | 'destrieux' — which level the histogram + test run on.
ANAT_LEVEL = os.environ.get('ANAT_LEVEL', 'auto')
# Cap the Destrieux histogram at the N most-populated labels (blank = all).
_top_n = os.environ.get('HIST_TOP_N', '').strip()
HIST_TOP_N = int(_top_n) if _top_n else None

# --- brain figure ---
MAKE_BRAIN = os.environ.get('MAKE_BRAIN', '1') not in ('0', 'false', 'False')
BRAIN_HEMI = os.environ.get('BRAIN_HEMI', 'both')   # 'both' | 'lh' | 'rh' | 'split'

# --- output ---
_tag = EPOCHS_ROOT_FILE if EPOCHS_ROOT_FILE else (
    'power_traces' if LABEL_SOURCE == 'power_traces'
    else f'synthetic_enr{SYNTHETIC_ENRICHMENT}')
_scope = ROI_FILTER if isinstance(ROI_FILTER, str) else (
    '-'.join(ROI_FILTER) if ROI_FILTER else 'wholebrain')
SAVE_DIR = os.path.join(
    current_script_dir, 'results', _tag,
    f'anatomy_{LABEL_SOURCE}_{_scope}_window_{WINDOW_TMIN}to{WINDOW_TMAX}s_{ELECTRODES}')


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = SimpleNamespace(
        timestamp=timestamp,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        task=TASK,
        acc_trials_only=ACC_TRIALS_ONLY,
        data_source=DATA_SOURCE,
        synthetic_enrichment=SYNTHETIC_ENRICHMENT,
        label_source=LABEL_SOURCE,
        pt_runs=PT_RUNS,
        pt_correction=PT_CORRECTION,
        pt_alpha=float(PT_ALPHA) if PT_ALPHA else None,
        pt_roi=PT_ROI,
        epochs_root_file=EPOCHS_ROOT_FILE,
        window_tmin=WINDOW_TMIN,
        window_tmax=WINDOW_TMAX,
        electrodes=ELECTRODES,
        rois_dict=ROIS_DICT,
        roi_filter=ROI_FILTER,
        anat_level=ANAT_LEVEL,
        hist_top_n=HIST_TOP_N,
        make_brain=MAKE_BRAIN,
        brain_hemi=BRAIN_HEMI,
        alpha=ALPHA,
        min_subjects=MIN_SUBJECTS,
        n_perm=N_PERM,
        seed=SEED,
        roi_dict_dir=ROI_DICT_DIR,
        save_dir=SAVE_DIR,
    )

    print("=" * 70)
    print("STABILITY vs FLEXIBILITY — A3 ANATOMY (coverage-conditioned)")
    print("=" * 70)
    print(f"Data source:      {DATA_SOURCE}"
          + (f" (planted enrichment={SYNTHETIC_ENRICHMENT})" if DATA_SOURCE == 'synthetic' else ""))
    print(f"Label source:     {LABEL_SOURCE}")
    if LABEL_SOURCE == 'power_traces':
        print(f"  power_traces runs:  {PT_RUNS}")
        print(f"  correction/alpha:   {PT_CORRECTION} / {PT_ALPHA or ALPHA}"
              f"  (ANOVA roi: {PT_ROI})")
    print(f"Subjects:         {SUBJECTS}")
    print(f"Task:             {TASK}")
    print(f"Epochs file:      {EPOCHS_ROOT_FILE}")
    print(f"Analysis window:  [{WINDOW_TMIN}, {WINDOW_TMAX}] s")
    print(f"Electrodes:       {ELECTRODES}")
    print(f"ROI filter:       {ROI_FILTER or 'none (whole brain)'}")
    print(f"Anatomical level: {ANAT_LEVEL}")
    print(f"Brain figure:     {'yes' if MAKE_BRAIN else 'no'} (hemi={BRAIN_HEMI})")
    print("-" * 70)
    print(f"alpha (A1):       {ALPHA}")
    print(f"min_subjects:     {MIN_SUBJECTS} | n_perm: {N_PERM} | seed: {SEED}")
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
