#!/bin/bash
# Submit A6 — brain-behavior correlation: does lPFC adaptation track behavioral
# adaptation? Per-participant continuous scores (raw and RT-adjusted, with
# reliabilities: the across-participant result), the label-based across-subject
# summaries (comparison only), and the within-subject single-trial mixed models.
# How to run it and read the outputs: docs/a6_brain_behavior.md.
#
# Usage:
#   bash submit_stability_flexibility_brain_behavior_dcc.sh                       # real data
#   DATA_SOURCE=synthetic bash submit_stability_flexibility_brain_behavior_dcc.sh # dry-run
#
# Falsification dry-run — destroy the specificity (each neural group drives BOTH
# adjustments equally) and confirm `specificity_ok` stops holding:
#   DATA_SOURCE=synthetic SYNTHETIC_CROSS_FRAC=1.0 \
#       bash submit_stability_flexibility_brain_behavior_dcc.sh
#
# RT-confound dry-run — no brain-behavior link at all, only HG that tracks RT.
# The RAW per-participant r comes out positive; the RT-ADJUSTED one should not:
#   DATA_SOURCE=synthetic SYNTHETIC_LINK=0 SYNTHETIC_RT_COUPLING=0.4 \
#       bash submit_stability_flexibility_brain_behavior_dcc.sh

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk. The
# active one is the file the N4 segregation / anatomy scores use, so each
# participant's neural score is the mean of those same electrode scores.
# ---------------------------------------------------------------------------
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE=${EPOCHS_ROOT_FILE:-"Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"}

# ---------------------------------------------------------------------------
# Behavior is scored from the epochs metadata (the same trials as the HG). The
# raw trial-level CSV is only a cross-check; a missing file skips the check.
# ---------------------------------------------------------------------------
BEHAVIOR_CSV=${BEHAVIOR_CSV:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/combinedData.csv"}
BEHAVIOR_RT_COL=${BEHAVIOR_RT_COL:-RT}

# ---------------------------------------------------------------------------
# Analysis window (seconds relative to stimulus onset) and electrode set.
# ELECTRODES only filters when ROIS names a region ('all' keeps every channel).
# ---------------------------------------------------------------------------
WINDOW_TMIN=${WINDOW_TMIN:-0.0}
WINDOW_TMAX=${WINDOW_TMAX:-1.5}
ELECTRODES=${ELECTRODES:-sig}       # 'all' or 'sig'
ROIS=${ROIS:-lpfc}                  # comma-separated names from src/analysis/config/rois.py, or 'all'

# Data source: 'real' loads epoched data; 'synthetic' plants known effects and
# validates the whole path.
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_N_SUBJ=${SYNTHETIC_N_SUBJ:-16}
SYNTHETIC_ACROSS_BETA=${SYNTHETIC_ACROSS_BETA:-1.2}
SYNTHETIC_WITHIN_BETA=${SYNTHETIC_WITHIN_BETA:-0.6}
SYNTHETIC_CROSS_FRAC=${SYNTHETIC_CROSS_FRAC:-0.25}
SYNTHETIC_LINK=${SYNTHETIC_LINK:-0.6}               # planted across-participant r
SYNTHETIC_RT_COUPLING=${SYNTHETIC_RT_COUPLING:-0.3} # HG noise SDs per RT SD

# A1 electrode definition + A6 hyperparameters.
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
FDR_CORRECTION=${FDR_CORRECTION:-none}     # fdr_bh or none
ALPHA=${ALPHA:-0.05}
# which label-based summary is starred at level (2):
# 'count' | 'frac' | 'effect' (all three are computed).
NEURAL_SUMMARY=${NEURAL_SUMMARY:-count}
# the within-subject single-trial level needs per-trial RT in the epochs metadata;
# set to 0 to skip it.
RUN_TRIALWISE=${RUN_TRIALWISE:-1}
# per-participant level: minimum usable electrodes per participant, and the number
# of shared trial splits behind the reliabilities.
MIN_ELEC=${MIN_ELEC:-3}
PARTICIPANT_N_SPLITS=${PARTICIPANT_N_SPLITS:-200}
SEED=${SEED:-0}

mkdir -p out

echo "Submitting stability/flexibility A6 brain-behavior (source=$DATA_SOURCE, electrodes=$ELECTRODES, rois=$ROIS, contrast=$CONTRAST_MODE, fdr=$FDR_CORRECTION)"
sbatch --job-name="sf_brainbehav_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",BEHAVIOR_CSV="$BEHAVIOR_CSV",BEHAVIOR_RT_COL="$BEHAVIOR_RT_COL",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",ROIS="$ROIS",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_N_SUBJ="$SYNTHETIC_N_SUBJ",SYNTHETIC_ACROSS_BETA="$SYNTHETIC_ACROSS_BETA",SYNTHETIC_WITHIN_BETA="$SYNTHETIC_WITHIN_BETA",SYNTHETIC_CROSS_FRAC="$SYNTHETIC_CROSS_FRAC",SYNTHETIC_LINK="$SYNTHETIC_LINK",SYNTHETIC_RT_COUPLING="$SYNTHETIC_RT_COUPLING",ALPHA="$ALPHA",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION",NEURAL_SUMMARY="$NEURAL_SUMMARY",RUN_TRIALWISE="$RUN_TRIALWISE",MIN_ELEC="$MIN_ELEC",PARTICIPANT_N_SPLITS="$PARTICIPANT_N_SPLITS",SEED="$SEED" \
    sbatch_stability_flexibility_brain_behavior_dcc.sh
