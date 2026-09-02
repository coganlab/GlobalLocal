#!/bin/bash
# Submit decoding jobs for multiple conditions

CONDITIONS=(
    stimulus_lwpc_block_balanced_conditions
    stimulus_lwps_block_balanced_conditions
    stimulus_congruency_by_switch_prop_block_balanced_conditions
    stimulus_switch_type_by_inc_prop_block_balanced_conditions
)

# Override this in the environment when decoding a different epochs dataset.
EPOCHS_ROOT_FILE="${EPOCHS_ROOT_FILE:-Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20}"

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

ANOVA_LABELS_CSVS=(
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
    
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_condition_none/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_condition_none/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"
#     "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"

#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_proportion_none/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_proportion_none/anova_labels.csv"
#     # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_proportion_none/anova_labels.csv"
#     "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_proportion_none/anova_labels.csv"

      "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"

)
# ANOVA_LABELS_CSVS=("")

if [[ -n "${ANOVA_LABELS_CSV:-}" ]]; then
    ANOVA_LABELS_CSVS=("$ANOVA_LABELS_CSV")
fi
# Submit every saved-label population by default.
# ANOVA_LABEL_EFFECTS=(
#     both lwpc lwps congruency switch_type
#     lwpc_only lwps_only congruency_only switch_type_only
# )

ANOVA_LABEL_EFFECTS=(
    both congruency_only switch_type_only
)

if [[ -z "${ANOVA_LABELS_CSVS[0]:-}" ]]; then
    # Dummy value: ignored when no saved ANOVA-label CSV is supplied.
    # Keeping one value ensures each condition is submitted only once.
    ANOVA_LABEL_EFFECTS=("lwpc")
elif [[ -n "${ANOVA_LABEL_EFFECT:-}" ]]; then
    # Optional override when using a real saved-label CSV.
    ANOVA_LABEL_EFFECTS=("$ANOVA_LABEL_EFFECT")
fi

ANOVA_LABEL_CORRECTION="${ANOVA_LABEL_CORRECTION:-flags}" # flags | none | fdr_bh
ANOVA_LABEL_ALPHA="${ANOVA_LABEL_ALPHA:-0.05}"
ANOVA_LABEL_ROI="${ANOVA_LABEL_ROI:-}"

# Create output directory if needed
mkdir -p out

for CSV_INDEX in "${!ANOVA_LABELS_CSVS[@]}"; do
    ANOVA_LABELS_CSV="${ANOVA_LABELS_CSVS[$CSV_INDEX]}"
    for EFFECT_INDEX in "${!ANOVA_LABEL_EFFECTS[@]}"; do
        ANOVA_LABEL_EFFECT="${ANOVA_LABEL_EFFECTS[$EFFECT_INDEX]}"
        for COND in "${CONDITIONS[@]}"; do
            echo "Submitting: condition=$COND anova_labels=${ANOVA_LABELS_CSV:-none} effect=$ANOVA_LABEL_EFFECT"
            sbatch --job-name="dec_a${CSV_INDEX}e${EFFECT_INDEX}_${COND}" \
                --export=ALL,CONDITION_NAME="$COND",EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
                sbatch_decoding_dcc.sh
            # sleep 2
        done
    done
done
