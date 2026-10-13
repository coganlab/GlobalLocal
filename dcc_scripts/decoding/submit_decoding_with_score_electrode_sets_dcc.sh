#!/bin/bash
# Launcher: decode each electrode set of a score-selection file on the trials
# the selection did not use. One job per condition set; each job runs the
# battery once per electrode set, into elecset_<name>/.
#
# Make the file first (minutes):
#   sbatch sbatch_score_electrode_selection_dcc.sh --rois lpfc --sign positive
# then:
#   ELECTRODE_SETS_JSON=figs/<epochs>/score_selections/<run>/score_selection.json \
#       bash submit_decoding_with_score_electrode_sets_dcc.sh
#   ELECTRODE_SETS=lwpc_only,lwps_only CONDITIONS="stimulus_lwpc_block_balanced_conditions" \
#       ELECTRODE_SETS_JSON=... bash submit_decoding_with_score_electrode_sets_dcc.sh
#
# ROIS must be ROIs the file selected in (the job refuses others).
# Docs: docs/decoding.md §3.7.

if [ -z "$ELECTRODE_SETS_JSON" ]; then
    echo "Set ELECTRODE_SETS_JSON to a score_selection.json" >&2
    exit 1
fi
export ELECTRODE_SETS_JSON="$(realpath "$ELECTRODE_SETS_JSON")"
# comma-separated, so it travels through --export=ALL, not the --export list
export ELECTRODE_SETS="${ELECTRODE_SETS:-}"

if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_congruency_conditions
        stimulus_switch_type_conditions
        stimulus_lwpc_block_balanced_conditions
        stimulus_lwps_block_balanced_conditions
    )
fi

mkdir -p out
for COND in "${CONDITION_LIST[@]}"; do
    echo "Submitting score-selected decoding: decode=$COND sets=${ELECTRODE_SETS:-all} file=$ELECTRODE_SETS_JSON"
    sbatch --job-name="dec_scoresel_${COND}" --export=ALL,CONDITION_NAME="$COND" \
        sbatch_decoding_dcc.sh
done
