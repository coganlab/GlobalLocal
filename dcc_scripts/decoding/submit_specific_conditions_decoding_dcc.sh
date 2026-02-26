#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    "stimulus_congruency_blockA_conditions"
    # "stimulus_congruency_blockB_conditions"
    # add more condition names here
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_NAME="$COND" sbatch_decoding_dcc.sh
    sleep 15
done