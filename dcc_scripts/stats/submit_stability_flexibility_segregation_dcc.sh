#!/bin/bash
# Submit the stability vs. flexibility segregation analysis.
#
# Usage:
#   bash submit_stability_flexibility_segregation_dcc.sh            # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_segregation_dcc.sh   # dry-run

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# ---------------------------------------------------------------------------
WINDOW_TMIN=0.0
WINDOW_TMAX=1.5
ELECTRODES=sig            # 'all' or 'sig'
# NOTE: sbatch --export separates VAR=VALUE pairs with commas, so a
# comma-containing value cannot be passed in that list -- it would be truncated
# at the first comma. ROIS is therefore `export`ed here and reaches the job
# through --export=ALL instead.
export ROIS=${ROIS:-lpfc} # comma-separated config ROI names, or 'all'

# Data source: 'real' loads epoched data; 'synthetic' validates the pipeline.
DATA_SOURCE=${DATA_SOURCE:-real}

# Contrast/electrode-label options.
CONTRAST_MODE=${CONTRAST_MODE:-condition}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-none}     # fdr_bh or none
EFFECT_MEASURE=${EFFECT_MEASURE:-cohens_d}    # cohens_d | cluster | peak_t
ALPHA=${ALPHA:-0.05}
MIN_ELEC=${MIN_ELEC:-3}

# Permutation counts (lower these for a quick test run).
# N_SPLITS=${N_SPLITS:-200}
# N_PERM_CORR=${N_PERM_CORR:-1000}
# N_PERM_LABEL=${N_PERM_LABEL:-1000}

N_SPLITS=${N_SPLITS:-50}
N_PERM_CORR=${N_PERM_CORR:-100}
N_PERM_LABEL=${N_PERM_LABEL:-100}
mkdir -p out

echo "Submitting stability/flexibility segregation (source=$DATA_SOURCE, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION)"
sbatch --job-name="segreg_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",N_SPLITS="$N_SPLITS",N_PERM_CORR="$N_PERM_CORR",N_PERM_LABEL="$N_PERM_LABEL",CONTRAST_MODE="$CONTRAST_MODE",EFFECT_MEASURE="$EFFECT_MEASURE",FDR_CORRECTION="$FDR_CORRECTION",ALPHA="$ALPHA",MIN_ELEC="$MIN_ELEC" \
    sbatch_stability_flexibility_segregation_dcc.sh
