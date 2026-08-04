#!/bin/bash
# Submit A5 — relative onset of stability (LWPC) vs. flexibility (LWPS):
# per-bin interaction time courses, 50%-of-peak onsets, and the Ulrich-Miller
# jackknifed onset-difference test.
#
# Usage:
#   bash submit_stability_flexibility_timing_dcc.sh                       # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_timing_dcc.sh # dry-run
#
# Falsification dry-run — plant the REVERSE ordering and confirm the reported
# sign flips (the method reads the order you planted, not a fixed prejudice):
#   DATA_SOURCE=synthetic SYNTHETIC_STAB_ONSET=0.40 SYNTHETIC_FLEX_ONSET=0.20 \
#       bash submit_stability_flexibility_timing_dcc.sh

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# A5 reads a RISING FLANK, so this window is wider than the A1/A2 window-mean
# default: it must include the pre-onset baseline and enough post-stimulus time
# for BOTH interactions to reach their peak (the 50%-of-peak threshold is
# meaningless if an effect is still climbing at WINDOW_TMAX).
# ---------------------------------------------------------------------------
WINDOW_TMIN=${WINDOW_TMIN:--1.0}
WINDOW_TMAX=${WINDOW_TMAX:-1.5}
ELECTRODES=${ELECTRODES:-sig}       # 'all' or 'sig'

# Data source: 'real' loads epoched HG time courses; 'synthetic' validates the
# whole path against a planted onset ordering.
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_N_SUBJ=${SYNTHETIC_N_SUBJ:-12}
SYNTHETIC_STAB_ONSET=${SYNTHETIC_STAB_ONSET:-0.20}
SYNTHETIC_FLEX_ONSET=${SYNTHETIC_FLEX_ONSET:-0.40}

# A5 hyperparameters.
# STATISTIC: 'mean' = grand-average d-o-d(t); 't' = t across electrodes (noise-
# normalized, often a cleaner flank). Report both if they disagree.
STATISTIC=${STATISTIC:-mean}
ALPHA=${ALPHA:-0.05}
SEED=${SEED:-0}

mkdir -p out

echo "Submitting stability/flexibility A5 timing (source=$DATA_SOURCE)"
sbatch --job-name="sf_timing_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_N_SUBJ="$SYNTHETIC_N_SUBJ",SYNTHETIC_STAB_ONSET="$SYNTHETIC_STAB_ONSET",SYNTHETIC_FLEX_ONSET="$SYNTHETIC_FLEX_ONSET",STATISTIC="$STATISTIC",ALPHA="$ALPHA",SEED="$SEED" \
    sbatch_stability_flexibility_timing_dcc.sh
