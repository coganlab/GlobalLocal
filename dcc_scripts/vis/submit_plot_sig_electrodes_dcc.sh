#!/bin/bash
# Submit significant-electrode brain-plotting jobs, one per named condition set.
#
# WHICH comparisons get plotted is chosen here: just list the labels you want
# from the PLOT_CONDITION_SETS registry in condition_plot_specs.py. Each label
# is submitted as its own job. Swapping in a different comparison = edit this
# list (or add a registry entry) -- no need to touch the python.
#
# Run from the dcc_scripts/vis directory so the sbatch relative out/ paths
# resolve:
#     bash submit_plot_sig_electrodes_dcc.sh

# Populations to overlay (must exist in condition_plot_specs.ELECTRODE_PLOT_SETS).
PLOT_SETS=${PLOT_SETS:-all_lpfc,task_relevant_lpfc,congruency_only,switch_type_only,both}

# Set this one epochs root directly. It identifies the baseline sig-channel files
# and the within-electrode power-trace ANOVA directory. anova_run() builds:
#   $POWER_FIGS_BASE/$ANOVA_EPOCHS_ROOT/anova_within_$ANOVA_UNIT/<label>_<N>_subjects
export POWER_FIGS_BASE="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs"
export ANOVA_EPOCHS_ROOT="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
export ANOVA_UNIT="electrode"

mkdir -p out

echo "Submitting LPFC plot sets: $PLOT_SETS"
sbatch --job-name="plot_lpfc_sets" \
    --export=ALL,PLOT_SET_LABEL="lpfc_power_trace_sets",PLOT_SETS="$PLOT_SETS",POWER_FIGS_BASE="$POWER_FIGS_BASE",ANOVA_EPOCHS_ROOT="$ANOVA_EPOCHS_ROOT",ANOVA_UNIT="$ANOVA_UNIT" \
    sbatch_plot_sig_electrodes_dcc.sh
