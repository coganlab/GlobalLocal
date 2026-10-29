#!/bin/bash
# Rerun the paper's analyses on one participant set from
# src/analysis/config/subjects.py:
#   F3       power traces: windowed ANOVA on the block-balanced sets, task-responsive lPFC
#   F4, S5b  within-block decoding on the block-balanced sets, task-responsive electrodes
#   F5 a-c   all-lPFC segregation scores -> continuous anatomy with the §15/16/19 follow-ups
#   F5 d-e   coding axes (all dlPFC+lPFC) with the S5 controls (remove_mean,
#            rt_adjust, RT-matched vs random subsets over five seeds)
# Settings are the ones the paper's all24 runs recorded. Every run writes to a
# folder ending in _subjects-<set>, so the all24 results are left alone.
#
# Usage, with the ieeg env active (sbatch hands the job this shell's conda):
#   SUBJECT_SET=good bash dcc_scripts/submit_subject_set_rerun.sh
#
# Submits 23 jobs: power 2, decoding 4, segregation 3, anatomy 1 (held until
# the three segregation jobs succeed), coding axes 13. The log, with the git
# commit and every job id, goes to dcc_scripts/out/.
set -eo pipefail
command -v conda >/dev/null || { echo "Activate the ieeg conda env first." >&2; exit 1; }

export SUBJECT_SET=${SUBJECT_SET:-good}
DCC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DCC/out"
LOG="$DCC/out/subject_set_rerun_${SUBJECT_SET}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "SUBJECT_SET=$SUBJECT_SET  commit=$(git -C "$DCC" rev-parse HEAD)  (uncommitted changes: $(git -C "$DCC" status --porcelain | wc -l))"

# Run one submit script, echo its output, and set LAST_IDS to the job ids it printed.
submit() {
    local out
    out=$("$@")
    echo "$out"
    LAST_IDS=$(grep -o 'Submitted batch job [0-9]*' <<< "$out" | awk '{print $4}' | paste -sd:)
}

echo "=== F3 power traces"
# Only these two sets: the cross-effect labels share their conditions objects
# and would race on the same output paths (see the power submit script).
cd "$DCC/power"
submit env CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
    ELECTRODES=sig ROIS=lpfc bash submit_specific_conditions_power_traces_dcc.sh

echo "=== F4 decoding (and the S5b cross-effects)"
cd "$DCC/decoding"
submit env CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions stimulus_congruency_by_switch_prop_block_balanced_conditions stimulus_switch_type_by_inc_prop_block_balanced_conditions" \
    ELECTRODES=sig FEATURE_MODE=channels_x_time bash submit_specific_conditions_decoding_dcc.sh

echo "=== F5 a-c segregation scores, then anatomy"
cd "$DCC/stats"
seg_env=(env ROIS=lpfc MAIN_EFFECTS=1 RT_ADJUST_HG=0 SHARED_SPLIT=0 EFFECT_MEASURE=cohens_d N_SPLITS=200)
submit "${seg_env[@]}" SCATTER_ONLY=0 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$LAST_IDS
# The two long tables with trial ids that the §19 follow-ups read.
submit "${seg_env[@]}" SCATTER_ONLY=1 SCATTER_N_SPLITS=0 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$seg_ids:$LAST_IDS
submit "${seg_env[@]}" SCATTER_ONLY=1 SCATTER_N_SPLITS=200 RT_ADJUST_HG=1 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$seg_ids:$LAST_IDS
submit env DEPENDENCY="afterok:$seg_ids" ROI=lpfc ELECTRODES=all ARM=continuous \
    ANAT_LEVEL=destrieux N_PERM=10000 bash submit_stability_flexibility_anatomy_dcc.sh

echo "=== F5 d-e coding axes and the S5 controls"
cd "$DCC/subspace"
coding_axes() {
    submit env ROIS=dlpfc,lpfc ELECTRODES=all EFFECT=main REMOVE_MEAN=0 RT_CONTROL=none SEED=0 \
        "$@" bash submit_coding_axes_dcc.sh
}
coding_axes
coding_axes REMOVE_MEAN=1
coding_axes RT_CONTROL=adjust
for seed in 0 1 2 3 4; do
    coding_axes RT_CONTROL=match SEED=$seed
    coding_axes RT_CONTROL=random SEED=$seed
done

echo "Log: $LOG"
