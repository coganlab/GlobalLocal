#!/bin/bash
# Submit A3 — anatomy of the stability/flexibility subpopulations: brain maps +
# ROI histograms + coverage-conditioned enrichment test.
#
# Usage:
#   bash submit_stability_flexibility_anatomy_dcc.sh                       # real data, A1 electrodes
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_anatomy_dcc.sh # dry-run
#   ARM=continuous \
#     SCORES_CSV="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/<epochs_root>/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh/electrodes.csv" \
#     PER_SPLIT_CSV="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/<epochs_root>/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh/per_split.csv" \
#     bash submit_stability_flexibility_anatomy_dcc.sh                 # reuse scores
#
#   # the all-lPFC N4 run with every follow-up (§15, §16, §19; inputs below):
#   ARM=continuous ROI_FILTER=lpfc ANAT_LEVEL=destrieux N_PERM=10000 \
#     bash submit_stability_flexibility_anatomy_dcc.sh
#   FOLLOWUPS=none ... skips the follow-ups; FOLLOWUPS=19 runs only §19's.
#
#   # power_traces (cluster-corrected) electrodes, lpfc only, counted by raw
#   # Destrieux label:
#   LABEL_SOURCE=power_traces ROI_FILTER=lpfc PT_ROI=lpfc \
#     PT_RUN_DIR="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs/<epochs_root>/anova_within_electrode/stimulus_experiment_conditions_24_subjects" \
#     bash submit_stability_flexibility_anatomy_dcc.sh

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# Only used by LABEL_SOURCE=a1 — the power_traces route reads finished runs.
# ---------------------------------------------------------------------------
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# ---------------------------------------------------------------------------
WINDOW_TMIN=0.0
WINDOW_TMAX=1.5
ELECTRODES=sig            # 'all' or 'sig'

# Data source: 'real' loads epoched data + the ROI atlas; 'synthetic' validates
# the whole path with a ground-truth electrode->ROI map.
DATA_SOURCE=${DATA_SOURCE:-real}
# synthetic-only knob: 0.0 = null (no group×ROI association), 0.6 = planted.
SYNTHETIC_ENRICHMENT=${SYNTHETIC_ENRICHMENT:-0.6}

# ---------------------------------------------------------------------------
# Analysis arm and continuous-score inputs.
#   categorical : binary stability/flexibility groups -> ROI enrichment.
#   continuous  : threshold-free LWPC/LWPS scores -> ROI/coordinate analyses.
#   both        : run categorical and continuous analyses.
#
# For a fast continuous run, point SCORES_CSV and PER_SPLIT_CSV at a completed
# segregation run, for example:
#   /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/
#     <epochs_root>/segregation_results/
#     window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh/electrodes.csv
#   /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/
#     <epochs_root>/segregation_results/
#     window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh/per_split.csv
# Leave them blank to calculate the scores from EPOCHS_ROOT_FILE instead.
# PER_SPLIT_CSV enables the noise-ceiling and minimum-electrode sweep.
# N_SPLITS is used only when calculating scores here; USE_COORDS=0 skips the
# reconstruction-dependent coordinate and centroid panels.
# ---------------------------------------------------------------------------
# A MAIN_EFFECTS=1 segregation run (directory ends in _main_effects) also gets
# the main-effect anatomy: dm = congruency - switch, and Tests 1 and 2 of
# docs/analysis_plans.md#closing-figure-plan. Drop the suffix for the archived LWPC/LWPS-only run.
SEG_RUN="/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects"

ARM=${ARM:-continuous}
SCORES_CSV=${SCORES_CSV:-"$SEG_RUN/electrodes.csv"}
PER_SPLIT_CSV=${PER_SPLIT_CSV:-"$SEG_RUN/per_split.csv"}
N_SPLITS=${N_SPLITS:-200}
USE_COORDS=${USE_COORDS:-1}

# ---------------------------------------------------------------------------
# N4 follow-ups (continuous arm), run on this job's own outputs once its
# summary is written. FOLLOWUPS lists which; 'none' skips them all.
#   15  n4_section15_followups.py              -> continuous/section15/
#   16  n4_section16_followups.py sections 1-5 -> continuous/section16/
#   19  local similarity on shared splits (LONG_DF_CSV) and the RT row of the
#       overlap controls (RT_COUPLING_CSV), in continuous/ with the rest of §19,
#       and the RT-adjusted companion (RT_LONG_DF_CSV) -> continuous/section19_rt_adjusted/
# The two long tables need trial ids. The segregation submitter's scatter-only
# route writes them in minutes (N4 doc §19.5):
#   RT_ADJUST_HG=0 SCATTER_N_SPLITS=0 SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh
#   RT_ADJUST_HG=1 SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh
# A file that is not there skips only its part. FOLLOWUPS contains commas, so
# it is exported (it reaches the job through --export=ALL) instead of going in
# the --export list, which splits on commas.
# ---------------------------------------------------------------------------
export FOLLOWUPS=${FOLLOWUPS:-15,16,19}
SEG_RESULTS=$(dirname "$SEG_RUN")
LONG_DF_CSV=${LONG_DF_CSV:-"$SEG_RESULTS/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_scatter_only_splits0/long_df.csv"}
RT_SEG_RUN="$SEG_RESULTS/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_rt_adjusted_scatter_only_splits200"
RT_LONG_DF_CSV=${RT_LONG_DF_CSV:-"$RT_SEG_RUN/long_df.csv"}
RT_COUPLING_CSV=${RT_COUPLING_CSV:-"$RT_SEG_RUN/rt_adjustment_slopes.csv"}
SUBSET_SCORES_CSV=${SUBSET_SCORES_CSV:-}   # §15 section 11: a subset run's scores_with_anatomy.csv
SHARED_N_SPLITS=${SHARED_N_SPLITS:-200}    # splits for the shared-split rescoring

# ---------------------------------------------------------------------------
# Electrode definition.
#   a1            : fit the window-mean interaction ANOVA here (needs epochs).
#   power_traces  : read finished within-electrode windowed-ANOVA runs
#                   (cluster-corrected). Set PT_RUN_DIR (one 4-factor run) or
#                   PT_RUN_CPC + PT_RUN_SPS (+ PT_RUN_CPS / PT_RUN_SPC).
# ---------------------------------------------------------------------------
LABEL_SOURCE=${LABEL_SOURCE:-a1}
PT_RUN_DIR=${PT_RUN_DIR:-}
PT_RUN_CPC=${PT_RUN_CPC:-}
PT_RUN_SPS=${PT_RUN_SPS:-}
PT_RUN_CPS=${PT_RUN_CPS:-}
PT_RUN_SPC=${PT_RUN_SPC:-}
PT_CORRECTION=${PT_CORRECTION:-fdr_bh}   # fdr_bh | cluster | none
PT_ALPHA=${PT_ALPHA:-}                   # defaults to ALPHA
PT_ROI=${PT_ROI:-}                       # the ANOVA run's ROI, e.g. lpfc

# ---------------------------------------------------------------------------
# Anatomical scope.
#   ROI_FILTER : keep only electrodes in this ROI group of config/rois.py
#                (e.g. lpfc; comma-separate for several). Empty = whole brain.
#   ANAT_LEVEL : auto | group | destrieux — level for the histogram + test.
#                'auto' uses raw Destrieux labels once restricted to one group.
# ---------------------------------------------------------------------------
ROI_FILTER=${ROI_FILTER:-'lpfc'}
ANAT_LEVEL=${ANAT_LEVEL:-auto}
HIST_TOP_N=${HIST_TOP_N:-}               # cap the Destrieux histogram at N labels

# Brain figure (needs mne + pyvista + the recon templates; falls back to the
# ROI histogram when they're missing).
MAKE_BRAIN=${MAKE_BRAIN:-1}
BRAIN_HEMI=${BRAIN_HEMI:-split}   # both | lh | rh | split
# Per-panel camera zoom; <1 zooms out. Blank keeps the renderer's default, which
# already separates the two 'split' panels; lower it (e.g. 0.6) to push the
# hemispheres further apart, raise it to 1 to fill each panel.
BRAIN_ZOOM=${BRAIN_ZOOM:-}

# A1 electrode definition + A3 hyperparameters.
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-fdr_bh}     # fdr_bh or none (LABEL_SOURCE=a1)
ALPHA=${ALPHA:-0.05}
MIN_SUBJECTS=${MIN_SUBJECTS:-3}      # keep ROIs sampled in >= this many subjects
N_PERM=${N_PERM:-10000}             # within-subject permutations for the null
SEED=${SEED:-0}
# Optional directory containing a precomputed electrodes-to-ROI atlas JSON.
ROI_DICT_DIR=${ROI_DICT_DIR:-"/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/config"}

mkdir -p out

echo "Submitting stability/flexibility A3 anatomy (source=$DATA_SOURCE, arm=$ARM, labels=$LABEL_SOURCE, roi=${ROI_FILTER:-wholebrain}, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION, followups=$FOLLOWUPS)"
sbatch --job-name="sf_anatomy_${LABEL_SOURCE}_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_ENRICHMENT="$SYNTHETIC_ENRICHMENT",ARM="$ARM",SCORES_CSV="$SCORES_CSV",PER_SPLIT_CSV="$PER_SPLIT_CSV",N_SPLITS="$N_SPLITS",USE_COORDS="$USE_COORDS",LONG_DF_CSV="$LONG_DF_CSV",RT_LONG_DF_CSV="$RT_LONG_DF_CSV",RT_COUPLING_CSV="$RT_COUPLING_CSV",SUBSET_SCORES_CSV="$SUBSET_SCORES_CSV",SHARED_N_SPLITS="$SHARED_N_SPLITS",LABEL_SOURCE="$LABEL_SOURCE",PT_RUN_DIR="$PT_RUN_DIR",PT_RUN_CPC="$PT_RUN_CPC",PT_RUN_SPS="$PT_RUN_SPS",PT_RUN_CPS="$PT_RUN_CPS",PT_RUN_SPC="$PT_RUN_SPC",PT_CORRECTION="$PT_CORRECTION",PT_ALPHA="$PT_ALPHA",PT_ROI="$PT_ROI",ROI_FILTER="$ROI_FILTER",ANAT_LEVEL="$ANAT_LEVEL",HIST_TOP_N="$HIST_TOP_N",MAKE_BRAIN="$MAKE_BRAIN",BRAIN_HEMI="$BRAIN_HEMI",BRAIN_ZOOM="$BRAIN_ZOOM",ALPHA="$ALPHA",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION",MIN_SUBJECTS="$MIN_SUBJECTS",N_PERM="$N_PERM",SEED="$SEED",ROI_DICT_DIR="$ROI_DICT_DIR" \
    sbatch_stability_flexibility_anatomy_dcc.sh
