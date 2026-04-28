#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    # "stimulus_lwpc_block_balanced_conditions"
    # "stimulus_lwps_block_balanced_conditions"
    "stimulus_congruency_by_switch_prop_block_balanced_conditions"
    "stimulus_switch_type_by_inc_prop_block_balanced_conditions"
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_NAME="$COND" sbatch_decoding_dcc.sh
    # sleep 2
done