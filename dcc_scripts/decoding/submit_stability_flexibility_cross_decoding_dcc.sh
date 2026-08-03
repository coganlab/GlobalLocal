#!/bin/bash
# Submit A4 — cross-decoding of the stability/flexibility subpopulations:
# label transfer + within-block 2x2 + temporal generalization.
#
# A4 runs on the ORDINARY decoding pipeline (ROI LabeledArray pseudopopulation,
# cross-validated folds, refit shuffle null, cluster correction over windows);
# the only addition is a second label vector, so the hyperparameters below are
# the same ones the main decoding job uses.
#
# Usage:
#   bash submit_stability_flexibility_cross_decoding_dcc.sh                        # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_cross_decoding_dcc.sh  # dry-run
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal bash submit_..._dcc.sh         # null code
#   FRAC_TRAIN=0.5 bash submit_stability_flexibility_cross_decoding_dcc.sh         # set the
#                                                                                 # train/test split

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

# A1 electrode definition.
ALPHA=${ALPHA:-0.05}
ROI=${ROI:-all}                      # which ROI's LabeledArray to decode

# Decoding hyperparameters (the ordinary pipeline's).
WINDOW_SIZE=${WINDOW_SIZE:-20}       # decoding window, in samples
STEP_SIZE=${STEP_SIZE:-10}           # window stride, in samples
N_SPLITS=${N_SPLITS:-5}              # CV folds (or resamples per repeat, see FRAC_TRAIN)
N_REPEATS=${N_REPEATS:-10}           # CV repeats
EXPLAINED_VARIANCE=${EXPLAINED_VARIANCE:-0.8}
N_PERM=${N_PERM:-500}                # permutations for the cluster test over windows
MIN_GROUP_SIZE=${MIN_GROUP_SIZE:-5}  # skip electrode groups smaller than this

# Proportion of trials used for TRAINING in each split. Leave empty to keep
# StratifiedKFold at (N_SPLITS-1)/N_SPLITS; set it to sweep the proportion
# directly (StratifiedShuffleSplit), e.g. FRAC_TRAIN=0.5.
FRAC_TRAIN=${FRAC_TRAIN:-}

mkdir -p out

echo "Submitting stability/flexibility A4 cross-decoding (source=$DATA_SOURCE)"
sbatch --job-name="sf_xdecode_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",ALPHA="$ALPHA",ROI="$ROI",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",N_SPLITS="$N_SPLITS",N_REPEATS="$N_REPEATS",EXPLAINED_VARIANCE="$EXPLAINED_VARIANCE",FRAC_TRAIN="$FRAC_TRAIN",N_PERM="$N_PERM",MIN_GROUP_SIZE="$MIN_GROUP_SIZE" \
    sbatch_stability_flexibility_cross_decoding_dcc.sh
