#!/bin/bash
# Submit selected LPFC electrode populations on one combined brain.
#
# Run from dcc_scripts/vis:
#     bash submit_plot_sig_electrodes_dcc.sh

# Example 1: complete congruency population, including "both"
# PLOT_SETS=${PLOT_SETS:-congruency_labels}

# Example 2: mutually exclusive A1 groups
PLOT_SETS=${PLOT_SETS:-congruency_only_labels,switch_type_only_labels,both_labels}

export POWER_FIGS_BASE="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs"

export ANOVA_EPOCHS_ROOT="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

export ANOVA_UNIT="electrode"

# Optional alternative electrode definition: the SOURCE A1 labels table. Set
# this when PLOT_SETS contains congruency_labels, switch_type_labels, an
# *_only_labels set, or both_labels. Do not point it at an
# anova_label_selections figures folder.
export ANOVA_LABELS_CSV="/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"

mkdir -p out

echo "Submitting LPFC plot sets: $PLOT_SETS"
echo "A1 labels source: $ANOVA_LABELS_CSV"

sbatch --job-name="plot_lpfc_sets" \
    --export=ALL,PLOT_SET_LABEL="lpfc_power_trace_sets",PLOT_SETS="$PLOT_SETS",POWER_FIGS_BASE="$POWER_FIGS_BASE",ANOVA_EPOCHS_ROOT="$ANOVA_EPOCHS_ROOT",ANOVA_UNIT="$ANOVA_UNIT",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV" \
    sbatch_plot_sig_electrodes_dcc.sh
