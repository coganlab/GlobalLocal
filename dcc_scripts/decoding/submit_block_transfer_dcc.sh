#!/bin/bash
# Submit N3b: block-transfer cross-decoding (docs/n3b_block_transfer.md).
#
# Trains congruency / switch type in one block level and tests it in the other
# (four designs, uncentered and centered) on EVERY electrode of one ROI. There are no
# electrode groups and no ANOVA / CSV / power-trace step.
#   ELECTRODES=sig  electrodes whose high-gamma responds to the stimulus (vs its
#                   pre-stimulus baseline); read from
#                   sig_chans_<subject>_<EPOCHS_ROOT_FILE>.json, so EPOCHS_ROOT_FILE
#                   also decides which significance file is used
#   ELECTRODES=all  every electrode in the ROI
#
#   bash submit_block_transfer_dcc.sh                        # lpfc, significant electrodes
#   ROI=acc bash submit_block_transfer_dcc.sh                # another region (config/rois.py)
#   ELECTRODES=all bash submit_block_transfer_dcc.sh         # every electrode
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=block_specific bash submit_block_transfer_dcc.sh
#                                    # planted answer: X1 must fail, X3 must transfer
#
# Results: results/<EPOCHS_ROOT_FILE>/block_transfer_<ROI>_<ELECTRODES>_w<W>s<S>/
# pooled_design_conditions/ -> summary.txt first.

EPOCHS_ROOT_FILE=${EPOCHS_ROOT_FILE:-"Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"}
ROI=${ROI:-lpfc}
ELECTRODES=${ELECTRODES:-sig}
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_CODE=${SYNTHETIC_CODE:-shared}   # synthetic only: shared (block-invariant) | block_specific

WINDOW_SIZE=${WINDOW_SIZE:-64}             # decoding window, in samples (256 Hz)
STEP_SIZE=${STEP_SIZE:-16}                 # window stride, in samples
N_SPLITS=${N_SPLITS:-5}                    # folds, cut inside the training level only
N_REPEATS=${N_REPEATS:-10}                 # balanced resamples, each with its own folds
N_PERM=${N_PERM:-500}                      # permutations for the cluster tests
SEED=${SEED:-0}

mkdir -p out
echo "Submitting N3b block transfer: source=$DATA_SOURCE roi=$ROI electrodes=$ELECTRODES"
sbatch --job-name="n3b_${DATA_SOURCE}_${ROI}_${ELECTRODES}" \
    --export=ALL,ANALYSIS=block_transfer,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ROI="$ROI",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",N_SPLITS="$N_SPLITS",N_REPEATS="$N_REPEATS",N_PERM="$N_PERM",SEED="$SEED" \
    sbatch_stability_flexibility_cross_decoding_dcc.sh
