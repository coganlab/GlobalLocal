#!/bin/bash
# Submit power trace jobs for multiple conditions

# CONDITIONS=(
#     stimulus_lwpc_conditions
#     stimulus_lwps_conditions
#     stimulus_congruency_by_switch_proportion_conditions
#     stimulus_switch_type_by_incongruent_proportion_conditions
# )


CONDITIONS=(
    stimulus_congruency_conditions
)

# Epochs file selection
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_4.0-8.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_4.0-8.0_Hz_padLength_1.5s_bandpass_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_13.0-30.0_Hz_padLength_1.5s_bandpass_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# anova stats
ANOVA_UNIT='electrode'  # whether to do the stats in terms of 'roi' (across electrodes) or 'electrode' (within_electrodes)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do
    echo "Submitting: $COND"
    sbatch --job-name="dec_${COND}" \
        --export=ALL,CONDITION_LABEL="$COND",EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ANOVA_UNIT="$ANOVA_UNIT" \
        sbatch_power_traces_dcc.sh
    # sleep 2
done