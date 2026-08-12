#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    stimulus_block_pairwise_conditions
    # stimulus_block_multiclass_conditions
)

# Optional selection from stats/results/anova_conjunction_windows/anova_labels.csv.
# Example (0--0.5 s, raw/no BH, LWPC; the CSV must have been generated for that window):
#   ANOVA_LABELS_CSV=/path/to/anova_labels.csv ANOVA_LABEL_EFFECT=lwpc \
#   ANOVA_LABEL_CORRECTION=none bash submit_specific_conditions_decoding_dcc.sh
ANOVA_LABELS_CSV=${ANOVA_LABELS_CSV:-}
ANOVA_LABEL_EFFECT=${ANOVA_LABEL_EFFECT:-lwpc}
ANOVA_LABEL_CORRECTION=${ANOVA_LABEL_CORRECTION:-flags} # flags | none | fdr_bh
ANOVA_LABEL_ALPHA=${ANOVA_LABEL_ALPHA:-0.05}
ANOVA_LABEL_ROI=${ANOVA_LABEL_ROI:-}

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
        sbatch_decoding_dcc.sh
    # sleep 2
done
