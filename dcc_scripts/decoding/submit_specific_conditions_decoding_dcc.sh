#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    stimulus_lwpc_block_balanced_conditions
    stimulus_lwps_block_balanced_conditions
    stimulus_congruency_by_switch_proportion_block_balanced_conditions
    stimulus_switch_type_by_incongruent_proportion_block_balanced_conditions
)

# Optional selection from stats/results/anova_conjunction_windows/anova_labels.csv.
# Example (0--0.5 s, raw/no BH, LWPC; the CSV must have been generated for that window):
#   ANOVA_LABELS_CSV=/path/to/anova_labels.csv ANOVA_LABEL_EFFECT=lwpc \
#   ANOVA_LABEL_CORRECTION=none bash submit_specific_conditions_decoding_dcc.sh
ANOVA_LABELS_CSVS=(
    "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"

    "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_condition_fdr_bh/anova_labels.csv"

    "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/another_result/anova_labels.csv"
)

ANOVA_LABEL_EFFECT=${ANOVA_LABEL_EFFECT:-lwpc}
ANOVA_LABEL_CORRECTION=${ANOVA_LABEL_CORRECTION:-flags} # flags | none | fdr_bh
ANOVA_LABEL_ALPHA=${ANOVA_LABEL_ALPHA:-0.05}
ANOVA_LABEL_ROI=${ANOVA_LABEL_ROI:-lpfc}

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
        sbatch_decoding_dcc.sh
    # sleep 2
done
