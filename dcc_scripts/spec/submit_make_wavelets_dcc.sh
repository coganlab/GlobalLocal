#!/bin/bash
# Submit power trace jobs for multiple conditions

# CONDITIONS=(
#     stimulus_lwpc_conditions
#     stimulus_lwps_conditions
#     stimulus_congruency_by_switch_proportion_conditions
#     stimulus_switch_type_by_incongruent_proportion_conditions
# )

SUBJECTS=("D0057" "D0059" "D0063" "D0065" "D0069" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110" "D0116" "D0117" "D0121" "D0133" "D0134" "D0137" "D0138" "D0139A" "D0144" "D0145" "D0146")

CONDITIONS=(
    stimulus_congruency_conditions
)

# Create output directory if needed
mkdir -p out

for SUBJ in "${SUBJECTS[@]}"; do
    echo "Submitting: $SUBJ"
    for COND in "${CONDITIONS[@]}"; do
        echo "Submitting: $COND"
        sbatch --job-name="wav_${SUBJ}_${COND}" \
            --export=ALL,CONDITION_LABEL="$COND",SUBJECT_ID="$SUBJ" \
            sbatch_make_wavelets_dcc.sh
        # sleep 2
    done
done