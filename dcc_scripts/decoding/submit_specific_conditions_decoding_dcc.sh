#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    stimulus_block_pairwise_conditions
    # stimulus_block_multiclass_conditions
)

# Optional selections from stats/results/anova_conjunction_windows/anova_labels.csv.
# Add as many CSVs (or result directories containing anova_labels.csv) as needed;
# one job is submitted for every condition x CSV combination. Leave the array
# with one empty entry to run without saved-ANOVA electrode selection.
#
# Example entries:
# ANOVA_LABELS_CSVS=(
#     /path/to/window_0.0to0.5s/anova_labels.csv
#     /path/to/window_0.5to1.0s/anova_labels.csv
# )
#
# For backward compatibility, setting ANOVA_LABELS_CSV in the environment uses
# that single path instead of this list. For example (0--0.5 s, raw/no BH, LWPC):
#   ANOVA_LABELS_CSV=/path/to/anova_labels.csv ANOVA_LABEL_EFFECT=lwpc \
#   ANOVA_LABEL_CORRECTION=none bash submit_specific_conditions_decoding_dcc.sh
# Use ANOVA_LABEL_EFFECT=both to require significance for both stability (LWPC)
# and flexibility (LWPS).
ANOVA_LABELS_CSVS=(
    ""
)
if [[ -n "${ANOVA_LABELS_CSV:-}" ]]; then
    ANOVA_LABELS_CSVS=("$ANOVA_LABELS_CSV")
fi
ANOVA_LABEL_EFFECT="${ANOVA_LABEL_EFFECT:-lwpc}"
ANOVA_LABEL_CORRECTION="${ANOVA_LABEL_CORRECTION:-flags}" # flags | none | fdr_bh
ANOVA_LABEL_ALPHA="${ANOVA_LABEL_ALPHA:-0.05}"
ANOVA_LABEL_ROI="${ANOVA_LABEL_ROI:-}"

# Create output directory if needed
mkdir -p out

for CSV_INDEX in "${!ANOVA_LABELS_CSVS[@]}"; do
    ANOVA_LABELS_CSV="${ANOVA_LABELS_CSVS[$CSV_INDEX]}"
    for COND in "${CONDITIONS[@]}"; do
        echo "Submitting: condition=$COND anova_labels=${ANOVA_LABELS_CSV:-none}"
        sbatch --job-name="dec_a${CSV_INDEX}_${COND}" \
            --export=ALL,CONDITION_NAME="$COND",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
            sbatch_decoding_dcc.sh
        # sleep 2
    done
done
