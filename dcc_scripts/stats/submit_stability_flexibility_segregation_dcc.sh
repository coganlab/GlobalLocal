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
WINDOW_TMAX=0.5
ELECTRODES=all            # 'all' or 'sig'

# Data source: 'real' loads epoched data; 'synthetic' validates the pipeline.
DATA_SOURCE=${DATA_SOURCE:-real}

# Permutation counts (lower these for a quick test run).
N_SPLITS=${N_SPLITS:-200}
N_PERM_CORR=${N_PERM_CORR:-10000}
N_PERM_LABEL=${N_PERM_LABEL:-2000}

mkdir -p out

echo "Submitting stability/flexibility segregation (source=$DATA_SOURCE)"
sbatch --job-name="segreg_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",N_SPLITS="$N_SPLITS",N_PERM_CORR="$N_PERM_CORR",N_PERM_LABEL="$N_PERM_LABEL" \
    sbatch_stability_flexibility_segregation_dcc.sh
