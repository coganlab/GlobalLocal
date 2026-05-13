#!/bin/bash
# Submit power trace jobs for multiple conditions

CONDITIONS=(
    stimulus_lwpc_conditions
    # stimulus_lwps_conditions
    # stimulus_congruency_by_switch_proportion_conditions
    # stimulus_switch_type_by_incongruent_proportion_conditions
)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" --export=ALL,CONDITION_LABEL="$COND" sbatch_power_traces_dcc.sh
    # sleep 2
done