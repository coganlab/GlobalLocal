#!/bin/bash
# Higher-level launcher: define significant electrodes on a HELD-OUT trial split,
# then decode on the disjoint remaining trials -- in one job, with no circularity
# between electrode selection and decoding accuracy.
#
# This is the "generate significant electrodes -> run decoding" pipeline done the
# bulletproof way (docs/analysis_guide.md §12.1 principle 1; §21.2 documents
# this launcher). It reuses the ordinary decoding job (sbatch_decoding_dcc.sh ->
# run_decoding_dcc.py -> decoding_dcc.py) but flips on the disjoint
# electrode-definition / decoding split via environment variables:
#
#   1. each subject's trials are split into a definition set (P_def) and a decode
#      set (P_dec), stratified so both stay balanced on condition/block,
#   2. responsive electrodes are (re)selected on P_def only
#      (apply_electrode_definition_split -> select_responsive_channels, FDR across
#      channels),
#   3. the bootstrap decoder scores P_dec only, restricted to those electrodes.
#
# The selection never sees the trials the accuracy is computed on, so it cannot
# inflate it. VALIDATE on one subject before a full re-run (the split is I/O glue;
# the primitives are unit-tested in tests/analysis/decoding/test_trial_splitting.py).
#
# Usage:
#   bash submit_decoding_with_electrode_definition_split_dcc.sh
#   FRAC_DEF=0.6 SEED=1 bash submit_decoding_with_electrode_definition_split_dcc.sh
#   CONDITIONS="stimulus_congruency_by_switch_prop_block_balanced_conditions" \
#       bash submit_decoding_with_electrode_definition_split_dcc.sh

# ---------------------------------------------------------------------------
# Conditions to decode (space-separated). Override with CONDITIONS="a b c".
# ---------------------------------------------------------------------------
if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_congruency_by_switch_prop_block_balanced_conditions
        stimulus_switch_type_by_inc_prop_block_balanced_conditions
    )
fi

# ---------------------------------------------------------------------------
# Electrode-definition split parameters (all overridable from the environment).
#   FRAC_DEF  fraction of each subject's trials used to DEFINE electrodes
#   STRATA    comma-separated metadata columns to stratify the split on
#   SEED      RNG seed for the split (reproducible)
#   ALPHA     FDR q-value for the held-out responsiveness selector
# ---------------------------------------------------------------------------
FRAC_DEF=${FRAC_DEF:-0.5}
STRATA=${STRATA:-congruency,switchType,blockType}
SEED=${SEED:-0}
ALPHA=${ALPHA:-0.05}

# NOTE: sbatch --export separates VAR=VALUE pairs with commas, so this
# comma-separated value cannot be passed in that list -- it would be truncated
# to a single stratum. Export it under its job-side name and let --export=ALL
# carry it instead.
export ELECTRODE_DEFINITION_SPLIT_STRATA="$STRATA"

mkdir -p out

for COND in "${CONDITION_LIST[@]}"; do
    echo "Submitting def-split decoding: $COND (frac_def=$FRAC_DEF, seed=$SEED, alpha=$ALPHA)"
    sbatch --job-name="dec_defsplit_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",\
ELECTRODE_DEFINITION_SPLIT=true,\
ELECTRODE_DEFINITION_SPLIT_FRAC="$FRAC_DEF",\
ELECTRODE_DEFINITION_SPLIT_SEED="$SEED",\
ELECTRODE_DEFINITION_SPLIT_ALPHA="$ALPHA" \
        sbatch_decoding_dcc.sh
done
