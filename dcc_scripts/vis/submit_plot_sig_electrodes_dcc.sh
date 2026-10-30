#!/bin/bash
# Submit selected LPFC electrode populations on one combined brain.
#
# Run from dcc_scripts/vis:
#     bash submit_plot_sig_electrodes_dcc.sh

# NOTE: sbatch --export uses commas to separate VAR=VALUE pairs, so a
# comma-containing value cannot be passed there -- "PLOT_SETS=a,b,c" is parsed
# as PLOT_SETS=a plus two bogus variable names, silently dropping b and c.
# Every variable below is therefore plain `export`ed into this shell and picked
# up by the job through --export=ALL.

# export PLOT_SETS="all_lpfc,task_relevant_lpfc,congruency_only_labels,switch_type_only_labels,both_labels"
# export PLOT_SETS="congruency_only_labels,switch_type_only_labels,both_labels"
# export PLOT_SETS="task_relevant_lpfc"
# export PLOT_SETS="congruency_only_labels"
# export PLOT_SETS="all_lpfc,task_relevant_lpfc"

# Fig 2: every localized contact (bad ones included), then analyzed LPFC, then
# task-significant analyzed LPFC. PLOT_SETS order is draw order, so later sets
# draw on top.
export PLOT_SETS="all_recon_elecs,all_lpfc_elecs,sig_lpfc_elecs"

export PLOT_SET_LABEL="lpfc_power_trace_sets"

export POWER_FIGS_BASE="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs"

# The epochs file every paper analysis uses (power traces, decoding, anatomy,
# coding axes): its sig_chans files define the task-significant electrodes, so
# Fig 2's counts match the analyses' (398 lPFC / 171 task-significant in all24).
# Until 2026-10-30 this pointed at the "..._0.5sec_within-1.0-0.0sec_base..."
# file, which gave 402 / 174.
export ANOVA_EPOCHS_ROOT="Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"

# Subject set (src/analysis/config/subjects.py); a set other than all24 writes
# to figs_subjects-<set>/.  e.g.  SUBJECT_SET=good bash submit_plot_sig_electrodes_dcc.sh
export SUBJECT_SET=${SUBJECT_SET:-all24}

export ANOVA_UNIT="electrode"

# Optional alternative electrode definition: the SOURCE A1 labels table. Set
# this when PLOT_SETS contains congruency_labels, switch_type_labels, an
# *_only_labels set, or both_labels. Do not point it at an
# anova_label_selections figures folder.
export ANOVA_LABELS_CSV="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/$ANOVA_EPOCHS_ROOT/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none/anova_labels.csv"

mkdir -p out

echo "Submitting LPFC plot sets: $PLOT_SETS (subject set $SUBJECT_SET)"
echo "A1 labels source: $ANOVA_LABELS_CSV"

sbatch --job-name="plot_lpfc_sets" \
    --export=ALL \
    sbatch_plot_sig_electrodes_dcc.sh
