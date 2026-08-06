#!/bin/bash
# Submit A6 — brain-behavior correlation: does the A1 neural selectivity predict
# the actual behavioral control adjustment? Across subjects (underpowered) and
# within subject on single trials (preferred), each with its cross-pairing
# specificity control.
#
# Usage:
#   bash submit_stability_flexibility_brain_behavior_dcc.sh                       # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_brain_behavior_dcc.sh # dry-run
#
# Falsification dry-run — destroy the specificity (each neural group drives BOTH
# adjustments equally) and confirm `specificity_ok` stops holding:
#   DATA_SOURCE=synthetic SYNTHETIC_CROSS_FRAC=1.0 \
#       bash submit_stability_flexibility_brain_behavior_dcc.sh

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# ---------------------------------------------------------------------------
# Raw trial-level behavior. Defaults to the repo-root combinedData.csv; point it
# at the Box copy on the cluster if that is where yours lives.
# ---------------------------------------------------------------------------
BEHAVIOR_CSV=${BEHAVIOR_CSV:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/combinedData.csv"}
BEHAVIOR_RT_COL=${BEHAVIOR_RT_COL:-RT}

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# ---------------------------------------------------------------------------
WINDOW_TMIN=${WINDOW_TMIN:-0.0}
WINDOW_TMAX=${WINDOW_TMAX:-1.5}
ELECTRODES=${ELECTRODES:-sig}       # 'all' or 'sig'

# Data source: 'real' loads epoched data + the behavioral CSV; 'synthetic' plants
# a matched coupling stronger than its cross control and validates the whole path.
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_N_SUBJ=${SYNTHETIC_N_SUBJ:-16}
SYNTHETIC_ACROSS_BETA=${SYNTHETIC_ACROSS_BETA:-1.2}
SYNTHETIC_WITHIN_BETA=${SYNTHETIC_WITHIN_BETA:-0.6}
SYNTHETIC_CROSS_FRAC=${SYNTHETIC_CROSS_FRAC:-0.25}

# A1 electrode definition + A6 hyperparameters.
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-none}     # fdr_bh or none
ALPHA=${ALPHA:-0.05}
# which per-subject neural summary headlines the across-subject level:
# 'count' | 'frac' | 'effect' (all three are computed; this picks the primary).
NEURAL_SUMMARY=${NEURAL_SUMMARY:-count}
# the within-subject single-trial level needs per-trial RT in the epochs metadata;
# set to 0 for the across-subject level only.
RUN_TRIALWISE=${RUN_TRIALWISE:-1}
SEED=${SEED:-0}

mkdir -p out

echo "Submitting stability/flexibility A6 brain-behavior (source=$DATA_SOURCE, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION)"
sbatch --job-name="sf_brainbehav_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",BEHAVIOR_CSV="$BEHAVIOR_CSV",BEHAVIOR_RT_COL="$BEHAVIOR_RT_COL",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_N_SUBJ="$SYNTHETIC_N_SUBJ",SYNTHETIC_ACROSS_BETA="$SYNTHETIC_ACROSS_BETA",SYNTHETIC_WITHIN_BETA="$SYNTHETIC_WITHIN_BETA",SYNTHETIC_CROSS_FRAC="$SYNTHETIC_CROSS_FRAC",ALPHA="$ALPHA",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION",NEURAL_SUMMARY="$NEURAL_SUMMARY",RUN_TRIALWISE="$RUN_TRIALWISE",SEED="$SEED" \
    sbatch_stability_flexibility_brain_behavior_dcc.sh
