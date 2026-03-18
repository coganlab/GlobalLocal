#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    # "stimulus_congruency_blockC_conditions"
    "stimulus_err_corr_conditions"

    # add more condition names here
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_NAME="$COND" sbatch_decoding_dcc.sh
    # sleep 2
done