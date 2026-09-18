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

Choosing the arm
----------------
ARM=categorical  (default) binary S/F labels -> group×ROI enrichment.
ARM=continuous   plan §5–§7: per-electrode LWPC/LWPS SCORES -> anatomy. No
                 electrode threshold anywhere; tests delta = lwpc - lwps against
                 ROI and against MNI coordinates, with the noise ceiling and the
                 leave-one-subject-out sweep. Score input, cheapest first:
                   SCORES_CSV      a finished segregation run's electrodes.csv
                   PER_SPLIT_CSV   its per_split.csv — needed for the §5.4
                                   ceiling and the min_elec sweep
                 With neither, the scores are computed here from the epoched data
                 (needs EPOCHS_ROOT_FILE; N_SPLITS controls the cost) and written
                 out so the next run can take the CSV route.
                 USE_COORDS=0 skips the coordinate/centroid panels (they need the
                 recon files).
ARM=both         categorical, then continuous into a `continuous/` subdir.

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


def _env(name, default=None):
    """`os.environ[name]`, stripped, with blank treated as unset.

    sbatch --export always defines the variables it is handed, so a knob the
    submit script left empty arrives as '' rather than missing. '' is falsy but
    it is NOT None, so a plain os.environ.get() sends an empty string down into
    the analysis — os.path.join('', name) then silently becomes a bare relative
    path, and an `is None` guard never fires. Normalise once, here.
    """
    value = os.environ.get(name)
    value = value.strip() if value is not None else None
    return value if value else default


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
DATA_SOURCE = _env('DATA_SOURCE', 'real')
# synthetic-only: strength of the planted group×ROI association (0 = null).
SYNTHETIC_ENRICHMENT = float(_env('SYNTHETIC_ENRICHMENT', '0.6'))

# --- which arm: 'categorical' (S/F groups), 'continuous' (LWPC/LWPS scores,
#     plan §5–§7) or 'both' ---
ARM = _env('ARM', 'categorical')

# continuous arm: where the scores come from. Point these at a finished
# stability_flexibility_segregation run to skip re-scoring (hours -> seconds).
SCORES_CSV = _env('SCORES_CSV')          # its electrodes.csv
PER_SPLIT_CSV = _env('PER_SPLIT_CSV')    # its per_split.csv (ceiling)
N_SPLITS = int(_env('N_SPLITS', '200'))  # only when scoring here
USE_COORDS = _env('USE_COORDS', '1') not in ('0', 'false', 'False')

# --- electrode definition: 'a1' (window-mean ANOVA here) or 'power_traces'
#     (finished cluster-corrected within-electrode ANOVA runs) ---
LABEL_SOURCE = _env('LABEL_SOURCE', 'a1')

# power_traces route: where the finished runs live. Either ONE run whose ANOVA
# had all four interaction terms, or one run per interaction.
PT_RUN_DIR = _env('PT_RUN_DIR')
PT_RUNS = {k: _env(f'PT_RUN_{k}') for k in ('CPC', 'SPS', 'CPS', 'SPC')}
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
PT_CORRECTION = _env('PT_CORRECTION', 'fdr_bh')
PT_ALPHA = _env('PT_ALPHA')            # defaults to ALPHA below
PT_ROI = _env('PT_ROI')                # the ANOVA run's ROI, e.g. 'lpfc'

# --- epochs / analysis window (A1 route on real data only) ---
EPOCHS_ROOT_FILE = _env('EPOCHS_ROOT_FILE')
# The continuous arm reading finished score CSVs needs no epoched data either.
_needs_epochs = not (ARM == 'continuous' and SCORES_CSV)
if DATA_SOURCE == 'real' and LABEL_SOURCE == 'a1' and _needs_epochs \
        and EPOCHS_ROOT_FILE is None:
    raise ValueError("EPOCHS_ROOT_FILE environment variable not set. "
                     "Set it via sbatch --export=ALL,EPOCHS_ROOT_FILE=... "
                     "(or run with DATA_SOURCE=synthetic to skip data loading, "
                     "or LABEL_SOURCE=power_traces to read finished ANOVA runs).")

WINDOW_TMIN = float(_env('WINDOW_TMIN', '0.0'))   # seconds post-stimulus
WINDOW_TMAX = float(_env('WINDOW_TMAX', '0.5'))

# --- electrode selection ---
ELECTRODES = _env('ELECTRODES', 'all')            # 'all' or 'sig'
ROIS_DICT = None

# --- A1 hyperparameter (the electrode definition A3 sits on) ---
CONTRAST_MODE = _env('CONTRAST_MODE', 'proportion')
FDR_CORRECTION = _env('FDR_CORRECTION', 'fdr_bh')
ALPHA = float(_env('ALPHA', '0.05'))

# --- A3 hyperparameters ---
# keep only ROIs sampled in >= MIN_SUBJECTS subjects (the coverage condition).
MIN_SUBJECTS = int(_env('MIN_SUBJECTS', '3'))
N_PERM = int(_env('N_PERM', '10000'))            # within-subject perms
SEED = int(_env('SEED', '0'))
# where the electrodes-to-ROIs atlas json is cached (real data only).
# None (not '') means "look in src/analysis/config" — an empty string would
# instead resolve the json against the job's cwd and fail there.
ROI_DICT_DIR = _env('ROI_DICT_DIR')

# --- anatomical scope ---
# Restrict the anatomy to one (or several, comma-separated) ROI groups from
# src/analysis/config/rois.py, e.g. 'lpfc'. Empty = whole brain.
_roi_filter = _env('ROI_FILTER', '')
ROI_FILTER = None
if _roi_filter:
    parts = [p.strip() for p in _roi_filter.split(',') if p.strip()]
    ROI_FILTER = parts[0] if len(parts) == 1 else parts
# 'auto' | 'group' | 'destrieux' — which level the histogram + test run on.
ANAT_LEVEL = _env('ANAT_LEVEL', 'auto')
# Cap the Destrieux histogram at the N most-populated labels (blank = all).
_top_n = _env('HIST_TOP_N')
HIST_TOP_N = int(_top_n) if _top_n else None

# --- brain figure ---
MAKE_BRAIN = _env('MAKE_BRAIN', '1') not in ('0', 'false', 'False')
BRAIN_HEMI = _env('BRAIN_HEMI', 'both')   # 'both' | 'lh' | 'rh' | 'split'
# Per-panel camera zoom; <1 zooms out. Blank uses the renderer's default, which
# already zooms the two 'split' panels out far enough to keep the hemispheres
# apart -- set it to push them further apart (0.6) or fill the panels (1.0).
_brain_zoom = _env('BRAIN_ZOOM')
BRAIN_ZOOM = float(_brain_zoom) if _brain_zoom else None

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
        arm=ARM,
        scores_csv=SCORES_CSV,
        per_split_csv=PER_SPLIT_CSV,
        n_splits=N_SPLITS,
        use_coords=USE_COORDS,
        responsiveness=None,
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
        brain_zoom=BRAIN_ZOOM,
        alpha=ALPHA,
        contrast_mode=CONTRAST_MODE,
        fdr_correction=FDR_CORRECTION,
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
    print(f"Arm:              {ARM}")
    if ARM in ('continuous', 'both'):
        print(f"  scores:             {SCORES_CSV or f'computed here ({N_SPLITS} splits)'}")
        print(f"  per-split (ceiling): {PER_SPLIT_CSV or ('computed here' if not SCORES_CSV else 'MISSING')}")
        print(f"  MNI coordinates:     {'yes' if USE_COORDS else 'no'}")
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
    print(f"Brain figure:     {'yes' if MAKE_BRAIN else 'no'} (hemi={BRAIN_HEMI}, "
          f"zoom={BRAIN_ZOOM if BRAIN_ZOOM is not None else 'default'})")
    print("-" * 70)
    print(f"contrast_mode:    {CONTRAST_MODE}")
    print(f"fdr_correction:   {FDR_CORRECTION}")
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
