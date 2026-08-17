#!/bin/bash
# Submit the A1/A2 analysis: parametric ANOVA electrode definition (A1) +
# overlap / conjunction inference (A2).
#
# Usage:
#   bash submit_stability_flexibility_anova_conjunction_dcc.sh            # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_anova_conjunction_dcc.sh   # dry-run

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# ---------------------------------------------------------------------------
WINDOW_TMIN=1
WINDOW_TMAX=1.5
ELECTRODES=sig            # 'all' or 'sig'
ROIS=${ROIS:-lpfc}        # comma-separated config ROI names, or 'all'

# Data source: 'real' loads epoched data; 'synthetic' validates the pipeline.
DATA_SOURCE=${DATA_SOURCE:-real}

# A1/A2 hyperparameters (lower N_PERM_NULL for a quick test run).
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-none}     # fdr_bh or none
ALPHA=${ALPHA:-0.05}
N_PERM_NULL=${N_PERM_NULL:-10000}
THRESHOLDS=${THRESHOLDS:-0.01,0.05,0.10,0.20,0.35,0.50}
# Set CROSSCHECK_NONPARAMETRIC=1 to also compare A1's ANOVA flags to the
# nonparametric permutation definition (slower).
CROSSCHECK_NONPARAMETRIC=${CROSSCHECK_NONPARAMETRIC:-0}

mkdir -p out

echo "Submitting stability/flexibility A1/A2 ANOVA+conjunction (source=$DATA_SOURCE, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION)"
sbatch --job-name="sf_anova_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",ROIS="$ROIS",DATA_SOURCE="$DATA_SOURCE",ALPHA="$ALPHA",N_PERM_NULL="$N_PERM_NULL",THRESHOLDS="$THRESHOLDS",CROSSCHECK_NONPARAMETRIC="$CROSSCHECK_NONPARAMETRIC",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION" \
    sbatch_stability_flexibility_anova_conjunction_dcc.sh
