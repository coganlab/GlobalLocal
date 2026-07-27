#!/bin/bash
# Submit A4 — cross-decoding of the stability/flexibility subpopulations:
# pseudo-trials + label / set / temporal transfer.
#
# Usage:
#   bash submit_stability_flexibility_cross_decoding_dcc.sh                        # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_cross_decoding_dcc.sh  # dry-run
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal bash submit_..._dcc.sh         # null code

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

# Data source: 'real' loads epoched data; 'synthetic' validates the whole path
# with a ground-truth pseudopopulation. SYNTHETIC_CODE picks the planted truth:
#   shared     -> stability & flexibility on one axis (should cross-decode)
#   orthogonal -> distinct axes (should NOT cross-decode, though each is decodable)
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_CODE=${SYNTHETIC_CODE:-shared}

# A1 electrode definition + A4 decoding hyperparameters.
ALPHA=${ALPHA:-0.05}
N_PER_CELL=${N_PER_CELL:-5}          # trials averaged per electrode per pseudo-trial
N_PSEUDO=${N_PSEUDO:-80}             # pseudo-trials per class-labelled cell
N_FOLDS=${N_FOLDS:-5}               # disjoint pseudo-trial folds
N_PERM=${N_PERM:-500}              # label-permutation null draws
MIN_GROUP_SIZE=${MIN_GROUP_SIZE:-5}  # skip electrode groups smaller than this

mkdir -p out

echo "Submitting stability/flexibility A4 cross-decoding (source=$DATA_SOURCE)"
sbatch --job-name="sf_xdecode_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",ALPHA="$ALPHA",N_PER_CELL="$N_PER_CELL",N_PSEUDO="$N_PSEUDO",N_FOLDS="$N_FOLDS",N_PERM="$N_PERM",MIN_GROUP_SIZE="$MIN_GROUP_SIZE" \
    sbatch_stability_flexibility_cross_decoding_dcc.sh
