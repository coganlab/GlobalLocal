#!/bin/bash
# Submit power trace jobs for multiple conditions

CONDITIONS=(
    stimulus_lwpc_conditions
    stimulus_lwps_conditions
    stimulus_congruency_by_switch_proportion_conditions
    stimulus_switch_type_by_incongruent_proportion_conditions
)


# CONDITIONS=(
#     stimulus_experiment_conditions
# )

# Epochs file selection
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_4.0-8.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_4.0-8.0_Hz_padLength_1.5s_bandpass_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_13.0-30.0_Hz_padLength_1.5s_bandpass_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# anova stats
ANOVA_UNIT='roi'  # whether to do the stats in terms of 'roi' (across electrodes) or 'electrode' (within_electrodes)

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
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_fdr_bh/anova_labels.csv"
    
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_condition_none/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_condition_none/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"
    "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"

    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to0.5s_sig_lpfc_proportion_none/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.5to1.0s_sig_lpfc_proportion_none/anova_labels.csv"
    # "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_1.0to1.5s_sig_lpfc_proportion_none/anova_labels.csv"
    "/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_proportion_none/anova_labels.csv"

)

if [[ -n "${ANOVA_LABELS_CSV:-}" ]]; then
    ANOVA_LABELS_CSVS=("$ANOVA_LABELS_CSV")
fi

ANOVA_LABEL_EFFECT=${ANOVA_LABEL_EFFECT:-both}
ANOVA_LABEL_CORRECTION=${ANOVA_LABEL_CORRECTION:-flags} # flags | none | fdr_bh
ANOVA_LABEL_ALPHA=${ANOVA_LABEL_ALPHA:-0.05}
ANOVA_LABEL_ROI=${ANOVA_LABEL_ROI:-}                    # e.g. lpfc; blank = all

# Create output directory if needed
mkdir -p out

for CSV_INDEX in "${!ANOVA_LABELS_CSVS[@]}"; do
    ANOVA_LABELS_CSV="${ANOVA_LABELS_CSVS[$CSV_INDEX]}"
    for COND in "${CONDITIONS[@]}"; do
        echo "Submitting: condition=$COND anova_labels=${ANOVA_LABELS_CSV:-none}"
        sbatch --job-name="pwr_a${CSV_INDEX}_${COND}" \
            --export=ALL,CONDITION_LABEL="$COND",EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ANOVA_UNIT="$ANOVA_UNIT",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
            sbatch_power_traces_dcc.sh
        # sleep 2
    done
done





