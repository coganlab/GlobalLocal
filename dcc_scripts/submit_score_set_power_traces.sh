#!/bin/bash
# Power traces of every electrode set in one score selection
# (sbatch_score_electrode_selection_dcc.sh), lPFC, on the LWPC and LWPS
# block-balanced sets: 7 sets x 2 = 14 jobs. The other two block-balanced
# labels (congruency_by_switch_prop, switch_type_by_inc_prop) fit the same
# ANOVA on the same cells, so they would only repeat these (2026-10-22: byte-
# identical results). Run after the selection job has finished: this cluster's
# sbatch ignores SBATCH_DEPENDENCY.
#
#   SUBJECT_SET=good bash dcc_scripts/submit_score_set_power_traces.sh <score_selection.json>
set -eo pipefail
SEL=${1:?give the score_selection.json}
[ -f "$SEL" ] || { echo "no such file: $SEL" >&2; exit 1; }
export SUBJECT_SET=${SUBJECT_SET:-all24}
cd "$(dirname "$0")/power"
for SET in lwpc lwps lwpc_only lwps_only adaptation_overlap congruency switch_type; do
    ROIS=lpfc ELECTRODE_SETS_JSON="$SEL" ELECTRODE_SET=$SET ELECTRODES=all \
        CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
        bash submit_specific_conditions_power_traces_dcc.sh
done
