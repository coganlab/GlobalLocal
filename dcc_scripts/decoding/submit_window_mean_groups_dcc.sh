#!/bin/bash
# Launcher: window-mean decoding in each ROI's task-significant, non-significant
# and all electrodes, at a matched electrode count, with membership-permutation
# tests between the groups (Holm). One job per condition set.
#
#   bash submit_window_mean_groups_dcc.sh
#   ROIS=lpfc,dlpfc WINDOW_MEAN_N_PERM=2000 bash submit_window_mean_groups_dcc.sh
#   CONDITIONS="stimulus_lwpc_block_balanced_conditions" bash submit_window_mean_groups_dcc.sh
#
# Knobs (environment): WINDOW_MEAN_TMIN / WINDOW_MEAN_TMAX (0 / 1.5 s),
# WINDOW_MEAN_BOOTSTRAPS (20), WINDOW_MEAN_N_PERM (1000), WINDOW_MEAN_N_MATCH
# (default: the smallest group). Docs: docs/decoding.md §3.6.

if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_congruency_conditions
        stimulus_switch_type_conditions
    )
fi

mkdir -p out
for COND in "${CONDITION_LIST[@]}"; do
    echo "Submitting window-mean group decoding: $COND (rois=${ROIS:-run script default})"
    sbatch --job-name="dec_wmgroups_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",WINDOW_MEAN_GROUPS=true \
        sbatch_decoding_dcc.sh
done
