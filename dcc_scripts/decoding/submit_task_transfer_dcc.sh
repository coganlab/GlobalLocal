#!/bin/bash
# Submit the task-transfer positive controls (docs/cross_decoding_controls.md §3.5).
#
# Same job as N3b (submit_block_transfer_dcc.sh) with a trial-level factor in place
# of the block: a contrast is trained in one level and tested in the other, uncentered
# and centered, on EVERY electrode of one ROI. No electrode groups, no ANOVA step.
#   T1  task (global vs local): congruent trials -> incongruent trials   (the clean control)
#   T2  task: repeat trials -> switch trials       (previous-task confound: on a switch
#                                                   trial the previous task was the other one)
#   T3  congruency: global-task trials -> local-task trials   (A4's contrast, own effect size)
#   T4  switch type: global-task trials -> local-task trials  (same previous-task confound)
#
# Run it on the electrodes of the cross-decode it controls. The A4 csv route decodes
# every electrode of the ROI, so pair that with ELECTRODES=all, and keep
# EPOCHS_ROOT_FILE the same as that run's.
#
#   bash submit_task_transfer_dcc.sh                         # lpfc, significant electrodes
#   ELECTRODES=all bash submit_task_transfer_dcc.sh          # every electrode (matches A4 csv)
#   ROI=occ bash submit_task_transfer_dcc.sh                 # another region (config/rois.py)
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=congruency_specific N_REPEATS=5 WINDOW_SIZE=16 STEP_SIZE=8 \
#       bash submit_task_transfer_dcc.sh    # planted answer: T1 must fail, T2 must transfer
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=carryover N_REPEATS=5 WINDOW_SIZE=16 STEP_SIZE=8 \
#       bash submit_task_transfer_dcc.sh    # planted previous-task carryover: T2/T4 drop, T1 doesn't
# (synthetic epochs are 32 samples long, so the 64-sample default window does not fit)
#
# Results: results/<EPOCHS_ROOT_FILE>/task_transfer_<ROI>_<ELECTRODES>_w<W>s<S>/
# pooled_design_conditions/ -> summary.txt first.

EPOCHS_ROOT_FILE=${EPOCHS_ROOT_FILE:-"Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"}
ROI=${ROI:-lpfc}
ELECTRODES=${ELECTRODES:-sig}
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_CODE=${SYNTHETIC_CODE:-shared}   # synthetic only: shared | congruency_specific | carryover

WINDOW_SIZE=${WINDOW_SIZE:-64}             # decoding window, in samples (256 Hz)
STEP_SIZE=${STEP_SIZE:-16}                 # window stride, in samples
N_SPLITS=${N_SPLITS:-5}                    # folds, cut inside the training level only
N_REPEATS=${N_REPEATS:-10}                 # balanced resamples, each with its own folds
N_PERM=${N_PERM:-500}                      # permutations for the cluster tests
SEED=${SEED:-0}

mkdir -p out
echo "Submitting task-transfer controls: source=$DATA_SOURCE roi=$ROI electrodes=$ELECTRODES"
sbatch --job-name="tasktransfer_${DATA_SOURCE}_${ROI}_${ELECTRODES}" \
    --export=ALL,ANALYSIS=task_transfer,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ROI="$ROI",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",N_SPLITS="$N_SPLITS",N_REPEATS="$N_REPEATS",N_PERM="$N_PERM",SEED="$SEED" \
    sbatch_stability_flexibility_cross_decoding_dcc.sh
