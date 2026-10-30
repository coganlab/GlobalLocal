#!/bin/bash
# The supplement and reviewer runs on one participant set from
# src/analysis/config/subjects.py, after submit_subject_set_rerun.sh has done
# the main-text runs (F3-F5 and the stimulus-locked coding axes):
#   S-F2   window-mean decoding of sig / nonsig / all lPFC and DLPFC: matched-count
#          bars with membership permutations, and the electrode-count curves
#   S-F3   power traces, task-responsive DLPFC
#   S-F4   within-block decoding on all lPFC electrodes
#   S-F5   DLPFC segregation scores -> continuous anatomy
#   S-F6   coding axes response-locked (main, remove_mean, rt_adjust, RT-matched
#          vs random seed 0) and occipital (both frames), with the
#          shuffled-blocks null on the main and control runs
#   reviewers  Figure 5 with the top 20 % LWPC / LWPS electrodes (lPFC now,
#          DLPFC after its anatomy); the every-trial score selection, whose
#          power traces are submitted when it finishes (see the end)
# Settings are the all24 runs' recorded ones. Every run writes to a folder
# ending in _subjects-<set>.
#
# Usage, with the ieeg env active:
#   SUBJECT_SET=good bash dcc_scripts/submit_subject_set_supplement.sh
set -eo pipefail
command -v conda >/dev/null || { echo "Activate the ieeg conda env first." >&2; exit 1; }

export SUBJECT_SET=${SUBJECT_SET:-good}
DCC="$(cd "$(dirname "$0")" && pwd)"
# The output-folder suffix of this set ('' for the default), from the config itself.
SUFFIX=$(cd "$DCC/.." && python -c "from src.analysis.config.subjects import resolve_subject_set as r; print(r()[1])")
EPOCHS=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
EPOCHS_RESP=${EPOCHS/Stimulus_/Response_}
mkdir -p "$DCC/out"
LOG="$DCC/out/subject_set_supplement_${SUBJECT_SET}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "SUBJECT_SET=$SUBJECT_SET (suffix '$SUFFIX')  commit=$(git -C "$DCC" rev-parse HEAD)  (uncommitted changes: $(git -C "$DCC" status --porcelain | wc -l))"

# Run one submit script, echo its output, and set LAST_IDS to the job ids it printed.
submit() {
    local out
    out=$("$@")
    echo "$out"
    LAST_IDS=$(grep -oE 'Submitted batch job [0-9]+|^[0-9]+$' <<< "$out" | grep -oE '[0-9]+' | paste -sd:)
}

echo "=== S-F2 window-mean decoding: matched bars, then the electrode-count curves"
cd "$DCC/decoding"
submit env ROIS=lpfc,dlpfc WINDOW_MEAN_MATCH_ELECTRODES=true WINDOW_MEAN_N_PERM=1000 \
    bash submit_window_mean_groups_dcc.sh
submit env ROIS=lpfc,dlpfc WINDOW_MEAN_MATCH_ELECTRODES=false WINDOW_MEAN_N_PERM=0 \
    bash submit_window_mean_groups_dcc.sh

echo "=== S-F3 power traces, task-responsive DLPFC"
cd "$DCC/power"
submit env CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
    ELECTRODES=sig ROIS=dlpfc bash submit_specific_conditions_power_traces_dcc.sh

echo "=== S-F4 within-block decoding, all lPFC electrodes"
cd "$DCC/decoding"
submit env CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
    ELECTRODES=all FEATURE_MODE=channels_x_time bash submit_specific_conditions_decoding_dcc.sh

echo "=== S-F5 DLPFC segregation scores, then anatomy"
cd "$DCC/stats"
seg_env=(env ROIS=dlpfc MAIN_EFFECTS=1 RT_ADJUST_HG=0 SHARED_SPLIT=0 EFFECT_MEASURE=cohens_d N_SPLITS=200)
submit "${seg_env[@]}" SCATTER_ONLY=0 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$LAST_IDS
submit "${seg_env[@]}" SCATTER_ONLY=1 SCATTER_N_SPLITS=0 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$seg_ids:$LAST_IDS
submit "${seg_env[@]}" SCATTER_ONLY=1 SCATTER_N_SPLITS=200 RT_ADJUST_HG=1 bash submit_stability_flexibility_segregation_dcc.sh
seg_ids=$seg_ids:$LAST_IDS
submit env DEPENDENCY="afterok:$seg_ids" ROI=dlpfc ELECTRODES=all ARM=continuous \
    ANAT_LEVEL=destrieux N_PERM=10000 bash submit_stability_flexibility_anatomy_dcc.sh
dlpfc_anatomy_id=$LAST_IDS

echo "=== reviewers: Figure 5 with the top 20 % LWPC and LWPS electrodes"
RES="$DCC/stats/results/$EPOCHS"
for ROI in lpfc dlpfc; do
    dep=()
    [ "$ROI" = dlpfc ] && dep=(--dependency="afterok:$dlpfc_anatomy_id")
    submit sbatch "${dep[@]}" --job-name="fig5_top_${ROI}_all${SUFFIX}" sbatch_fig5_top_sets_dcc.sh \
        --anatomy-dir "$RES/anatomy_a1_${ROI}_window_0.0to1.5s_all${SUFFIX}/continuous" \
        --seg-dir "$RES/segregation_results/window_0.0to1.5s_all_${ROI}_proportion_cohens_d_fdr_bh_main_effects${SUFFIX}" \
        --brain
done

echo "=== S-F6 coding axes: response-locked and occipital"
cd "$DCC/subspace"
coding_axes() {
    submit env ROIS=dlpfc,lpfc ELECTRODES=all EFFECT=main REMOVE_MEAN=0 RT_CONTROL=none SEED=0 \
        "$@" bash submit_coding_axes_dcc.sh
}
resp=(EPOCHS_ROOT_FILE="$EPOCHS_RESP" WINDOWS_TMIN=-1.0 TRAJ_AXES=-1.0:0.0,-1.0:-0.5,-0.5:0.0)
coding_axes "${resp[@]}" N_BLOCK_PERM=500
coding_axes "${resp[@]}" N_BLOCK_PERM=500 REMOVE_MEAN=1
coding_axes "${resp[@]}" N_BLOCK_PERM=500 RT_CONTROL=adjust
coding_axes "${resp[@]}" RT_CONTROL=match
coding_axes "${resp[@]}" RT_CONTROL=random
coding_axes ROIS=occ
coding_axes "${resp[@]}" ROIS=occ

echo "=== reviewers: every-trial score selection (its power traces follow; see below)"
cd "$DCC/decoding"
submit sbatch --parsable sbatch_score_electrode_selection_dcc.sh --rois lpfc --frac-select 1
echo "SELECTION_JOB=$LAST_IDS"
echo "SELECTION_JSON=$DCC/decoding/figs/$EPOCHS/score_selections/lpfc_all_sel1_top0.2_positive_0to1.5s_seed0$SUFFIX/score_selection.json"
echo "This cluster's sbatch ignores SBATCH_DEPENDENCY, so submit the score-set power"
echo "traces once the selection job has finished:"
echo "  SUBJECT_SET=$SUBJECT_SET bash dcc_scripts/submit_score_set_power_traces.sh <SELECTION_JSON>"
echo "Log: $LOG"
