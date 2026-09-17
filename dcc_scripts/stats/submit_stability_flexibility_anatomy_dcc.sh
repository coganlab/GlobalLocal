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
ARM=${ARM:-categorical}
SCORES_CSV=${SCORES_CSV:-}
PER_SPLIT_CSV=${PER_SPLIT_CSV:-}
N_SPLITS=${N_SPLITS:-200}
USE_COORDS=${USE_COORDS:-1}

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
ROI_FILTER=${ROI_FILTER:'lpfc'}
ANAT_LEVEL=${ANAT_LEVEL:-auto}
HIST_TOP_N=${HIST_TOP_N:-}               # cap the Destrieux histogram at N labels

# Brain figure (needs mne + pyvista + the recon templates; falls back to the
# ROI histogram when they're missing).
MAKE_BRAIN=${MAKE_BRAIN:-1}
BRAIN_HEMI=${BRAIN_HEMI:-both}

# A1 electrode definition + A3 hyperparameters.
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-fdr_bh}     # fdr_bh or none (LABEL_SOURCE=a1)
ALPHA=${ALPHA:-0.05}
MIN_SUBJECTS=${MIN_SUBJECTS:-3}      # keep ROIs sampled in >= this many subjects
N_PERM=${N_PERM:-10000}             # within-subject permutations for the null
SEED=${SEED:-0}
# Optional directory containing a precomputed electrodes-to-ROI atlas JSON.
ROI_DICT_DIR=${ROI_DICT_DIR:'/hpc/home/jz421/coganlab/jz421/GlobalLocal/src/analysis/config/'}

mkdir -p out

echo "Submitting stability/flexibility A3 anatomy (source=$DATA_SOURCE, arm=$ARM, labels=$LABEL_SOURCE, roi=${ROI_FILTER:-wholebrain}, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION)"
sbatch --job-name="sf_anatomy_${LABEL_SOURCE}_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_ENRICHMENT="$SYNTHETIC_ENRICHMENT",ARM="$ARM",SCORES_CSV="$SCORES_CSV",PER_SPLIT_CSV="$PER_SPLIT_CSV",N_SPLITS="$N_SPLITS",USE_COORDS="$USE_COORDS",LABEL_SOURCE="$LABEL_SOURCE",PT_RUN_DIR="$PT_RUN_DIR",PT_RUN_CPC="$PT_RUN_CPC",PT_RUN_SPS="$PT_RUN_SPS",PT_RUN_CPS="$PT_RUN_CPS",PT_RUN_SPC="$PT_RUN_SPC",PT_CORRECTION="$PT_CORRECTION",PT_ALPHA="$PT_ALPHA",PT_ROI="$PT_ROI",ROI_FILTER="$ROI_FILTER",ANAT_LEVEL="$ANAT_LEVEL",HIST_TOP_N="$HIST_TOP_N",MAKE_BRAIN="$MAKE_BRAIN",BRAIN_HEMI="$BRAIN_HEMI",ALPHA="$ALPHA",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION",MIN_SUBJECTS="$MIN_SUBJECTS",N_PERM="$N_PERM",SEED="$SEED",ROI_DICT_DIR="$ROI_DICT_DIR" \
    sbatch_stability_flexibility_anatomy_dcc.sh
