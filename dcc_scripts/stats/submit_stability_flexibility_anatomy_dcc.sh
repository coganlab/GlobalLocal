#!/bin/bash
# Submit A3 — anatomy of the stability/flexibility subpopulations: brain maps +
# ROI histograms + coverage-conditioned enrichment test.
#
# Usage:
#   bash submit_stability_flexibility_anatomy_dcc.sh                       # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_anatomy_dcc.sh # dry-run

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

# Data source: 'real' loads epoched data + the ROI atlas; 'synthetic' validates
# the whole path with a ground-truth electrode->ROI map.
DATA_SOURCE=${DATA_SOURCE:-real}
# synthetic-only knob: 0.0 = null (no group×ROI association), 0.6 = planted.
SYNTHETIC_ENRICHMENT=${SYNTHETIC_ENRICHMENT:-0.6}

# A1 electrode definition + A3 hyperparameters.
ALPHA=${ALPHA:-0.05}
MIN_SUBJECTS=${MIN_SUBJECTS:-3}      # keep ROIs sampled in >= this many subjects
N_PERM=${N_PERM:-10000}             # within-subject permutations for the null
REQUIRE_SIGN=${REQUIRE_SIGN:-0}

mkdir -p out

echo "Submitting stability/flexibility A3 anatomy (source=$DATA_SOURCE)"
sbatch --job-name="sf_anatomy_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_ENRICHMENT="$SYNTHETIC_ENRICHMENT",ALPHA="$ALPHA",MIN_SUBJECTS="$MIN_SUBJECTS",N_PERM="$N_PERM",REQUIRE_SIGN="$REQUIRE_SIGN" \
    sbatch_stability_flexibility_anatomy_dcc.sh
