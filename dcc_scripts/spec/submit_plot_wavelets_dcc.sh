#!/bin/bash
# Submit plot wavelet jobs for multiple subjects and conditions

SUBJECTS=("D0057" "D0059" "D0063" "D0065" "D0069" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110" "D0116" "D0117" "D0121" "D0133" "D0134" "D0137" "D0138" "D0139A" "D0144" "D0145" "D0146")

CONDITIONS=(
    stimulus_congruency_conditions
)

# Create output directory if needed (Slurm won't create it for the log files)
mkdir -p out/plot_wavelets

for SUBJ in "${SUBJECTS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
        echo "Submitting: $SUBJ $COND"
        sbatch --job-name="plot_${SUBJ}_${COND}" \
            --export=ALL,CONDITION_LABEL="$COND",SUBJECT_ID="$SUBJ" \
            sbatch_plot_wavelets_dcc.sh
    done
done