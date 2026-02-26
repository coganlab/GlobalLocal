#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    "stimulus_congruency_blockA_conditions"
    "stimulus_congruency_blockB_conditions"
    "stimulus_congruency_blockC_conditions"
    "stimulus_congruency_blockD_conditions"
    "stimulus_switchType_blockA_conditions"
    "stimulus_switchType_blockB_conditions"
    "stimulus_switchType_blockC_conditions"
    "stimulus_switchType_blockD_conditions"
    # add more condition names here
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_NAME="$COND" sbatch_decoding_dcc.sh
    sleep 5
done