#!/bin/bash
# Launcher: window-mean decoding in each ROI's task-significant, non-significant
# and all electrodes, at a matched electrode count (or each group whole, with
# WINDOW_MEAN_MATCH_ELECTRODES=false), with membership-permutation tests between
# the groups (Holm). One job per condition set.
#
#   bash submit_window_mean_groups_dcc.sh
#   ROIS=lpfc,dlpfc WINDOW_MEAN_N_PERM=2000 bash submit_window_mean_groups_dcc.sh
#   CONDITIONS="stimulus_lwpc_block_balanced_conditions" bash submit_window_mean_groups_dcc.sh
#   WINDOW_MEAN_MATCH_ELECTRODES=false bash submit_window_mean_groups_dcc.sh   # no size matching
#   WINDOW_MEAN_N_PERM=0 bash submit_window_mean_groups_dcc.sh   # bars and curve, no tests
#
# Set the parameters below; the same names in the environment override them.
# Docs: docs/decoding.md §3.6.

if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_congruency_conditions
        stimulus_switch_type_conditions
    )
fi

# Epochs dataset to decode.
EPOCHS_ROOT_FILE="${EPOCHS_ROOT_FILE:-Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20}"

# Comma-separated ROI names from src/analysis/config/rois.py (e.g. lpfc,dlpfc).
# Empty uses ROIS_DICT in run_decoding_dcc.py (dlpfc).
ROIS="${ROIS:-lpfc,dlpfc}"

# Each electrode's HG is averaged over TMIN..TMAX (s), one feature per electrode.
WINDOW_MEAN_TMIN="${WINDOW_MEAN_TMIN:-0.0}"
WINDOW_MEAN_TMAX="${WINDOW_MEAN_TMAX:-1.5}"
# Pseudo-trial draws.
WINDOW_MEAN_BOOTSTRAPS="${WINDOW_MEAN_BOOTSTRAPS:-20}"
# Membership permutations for each pair of groups.
WINDOW_MEAN_N_PERM="${WINDOW_MEAN_N_PERM:-1000}"
# true decodes every group at the same electrode count (WINDOW_MEAN_N_MATCH);
# false decodes each group on all its electrodes. Results go to a tag ending in
# _unmatched, so both versions can sit side by side.
WINDOW_MEAN_MATCH_ELECTRODES="${WINDOW_MEAN_MATCH_ELECTRODES:-false}"
# Electrodes decoded per group. Empty uses the smallest group's size. Must be
# empty when WINDOW_MEAN_MATCH_ELECTRODES is false.
WINDOW_MEAN_N_MATCH="${WINDOW_MEAN_N_MATCH:-}"
# Accuracy-by-electrode-count curve: random subsets per count and bootstrap
# (counts 1, 2, 4, ... and every 20 up to each group's size). 0 skips it.
WINDOW_MEAN_CURVE_SUBSETS="${WINDOW_MEAN_CURVE_SUBSETS:-10}"

# Same truthy values as run_decoding_dcc.py's _env_bool.
case "${WINDOW_MEAN_MATCH_ELECTRODES,,}" in
    1|true|yes|on) MATCH_DESC="n_match=${WINDOW_MEAN_N_MATCH:-smallest group}" ;;
    *)
        if [ -n "$WINDOW_MEAN_N_MATCH" ]; then
            echo "WINDOW_MEAN_N_MATCH=$WINDOW_MEAN_N_MATCH needs WINDOW_MEAN_MATCH_ELECTRODES=true" >&2
            exit 1
        fi
        MATCH_DESC="unmatched (each group on all its electrodes)" ;;
esac

# Exported rather than listed in --export, which splits on commas (ROIS has
# them); --export=ALL passes them to the job.
export EPOCHS_ROOT_FILE ROIS WINDOW_MEAN_TMIN WINDOW_MEAN_TMAX \
    WINDOW_MEAN_BOOTSTRAPS WINDOW_MEAN_N_PERM WINDOW_MEAN_MATCH_ELECTRODES WINDOW_MEAN_N_MATCH \
    WINDOW_MEAN_CURVE_SUBSETS

echo "Window-mean groups: epochs=$EPOCHS_ROOT_FILE"
echo "  rois=${ROIS:-run script default}" \
     "window=${WINDOW_MEAN_TMIN}-${WINDOW_MEAN_TMAX}s" \
     "bootstraps=$WINDOW_MEAN_BOOTSTRAPS n_perm=$WINDOW_MEAN_N_PERM" \
     "$MATCH_DESC curve_subsets=$WINDOW_MEAN_CURVE_SUBSETS"

mkdir -p out
for COND in "${CONDITION_LIST[@]}"; do
    echo "Submitting window-mean group decoding: $COND"
    sbatch --job-name="dec_wmgroups_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",WINDOW_MEAN_GROUPS=true \
        sbatch_decoding_dcc.sh
done
