#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    stimulus_block_pairwise_conditions
    stimulus_block_multiclass_conditions
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_NAME="$COND" sbatch_decoding_dcc.sh
    # sleep 2
done