#!/bin/bash
# Submit power trace jobs for multiple conditions

# The pooled sets below define each cell as a raw union of BIDS events, so the
# cells being contrasted hold 3:1 vs 1:3 mixtures of the OTHER proportion block
# and a block-level offset enters the contrast at every timepoint. Kept for
# reference; prefer the block-balanced sets.
CONDITIONS=(
    stimulus_lwpc_conditions
    stimulus_lwps_conditions
    stimulus_congruency_by_switch_proportion_conditions
    stimulus_switch_type_by_incongruent_proportion_conditions
)

# One cell per block type + full-factorial ANOVA, so the 2-way terms are
# equal-weight contrasts. Two jobs cover all four interactions: the lwpc set
# emits congruency x incProportion AND congruency x switchProportion, the lwps
# set emits switchType x switchProportion AND switchType x incProportion.
# Do NOT also list the *_by_switch_prop_/_by_inc_prop_block_balanced labels --
# they share a conditions_obj with these two, so they resolve to the same save
# name and the jobs would race on one output path.
# CONDITIONS=(
#     stimulus_lwpc_block_balanced_conditions
#     stimulus_lwps_block_balanced_conditions
# )

# Epochs file selection

# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_3.0s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"

# anova stats
ANOVA_UNIT='electrode'  # whether to do the stats in terms of 'roi' (across electrodes) or 'electrode' (within_electrodes)

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
# ANOVA_LABELS_CSVS=(
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

# )

if [[ -n "${ANOVA_LABELS_CSV:-}" ]]; then
    ANOVA_LABELS_CSVS=("$ANOVA_LABELS_CSV")
fi

# With every path above commented out the array is EMPTY, and a `for` over an
# empty array runs zero times -- so nothing gets submitted at all. Fall back to a
# single empty entry, which run_power_traces_dcc.py reads as "no saved-ANOVA
# selection": it keeps whatever ELECTRODES selects (all / sig) and writes to the
# plain figs/<epochs>/anova_within_<unit>/ path instead of an
# anova_label_selections/ subdirectory.
if [[ ${#ANOVA_LABELS_CSVS[@]} -eq 0 ]]; then
    ANOVA_LABELS_CSVS=("")
fi

# Submit every saved-label population by default. Set ANOVA_LABEL_EFFECT to
# retain the previous single-effect behavior for one-off/environment-driven runs.
ANOVA_LABEL_EFFECTS=(
    both lwpc lwps congruency switch_type
    lwpc_only lwps_only congruency_only switch_type_only
)
if [[ -n "${ANOVA_LABEL_EFFECT:-}" ]]; then
    ANOVA_LABEL_EFFECTS=("$ANOVA_LABEL_EFFECT")
fi
ANOVA_LABEL_CORRECTION=${ANOVA_LABEL_CORRECTION:-flags} # flags | none | fdr_bh
ANOVA_LABEL_ALPHA=${ANOVA_LABEL_ALPHA:-0.05}
ANOVA_LABEL_ROI=${ANOVA_LABEL_ROI:-}                    # e.g. lpfc; blank = all

# Which electrodes to start from, before any saved-ANOVA selection:
#   all = every electrode in the ROI, sig = those significant vs baseline.
ELECTRODES=${ELECTRODES:-sig}

# which electrodes to exclude
#
# NOTE: sbatch --export separates VAR=VALUE pairs with commas, so this
# comma-separated list cannot be passed in that list -- it would be truncated
# after the first electrode, silently leaving the rest in the analysis. Export
# it here and let --export=ALL carry it instead.
# export EXCLUDE_ELECTRODES="${EXCLUDE_ELECTRODES:-D0121:LFMI9,D0121:LFMI8}"
export EXCLUDE_ELECTRODES="${EXCLUDE_ELECTRODES:-}"

# Create output directory if needed
mkdir -p out

for CSV_INDEX in "${!ANOVA_LABELS_CSVS[@]}"; do
    ANOVA_LABELS_CSV="${ANOVA_LABELS_CSVS[$CSV_INDEX]}"

    # ANOVA_LABEL_EFFECT picks a subpopulation *inside* a labels CSV. With no CSV
    # there is no subpopulation to pick, so looping the nine effects would submit
    # nine identical jobs all writing the same output directory. Run once instead.
    if [[ -z "$ANOVA_LABELS_CSV" ]]; then
        EFFECTS_THIS_CSV=("all_${ELECTRODES}_elecs")
    else
        EFFECTS_THIS_CSV=("${ANOVA_LABEL_EFFECTS[@]}")
    fi

    for EFFECT_INDEX in "${!EFFECTS_THIS_CSV[@]}"; do
        ANOVA_LABEL_EFFECT="${EFFECTS_THIS_CSV[$EFFECT_INDEX]}"
        for COND in "${CONDITIONS[@]}"; do
            echo "Submitting: condition=$COND electrodes=$ELECTRODES anova_labels=${ANOVA_LABELS_CSV:-none} effect=$ANOVA_LABEL_EFFECT"
            sbatch --job-name="pwr_a${CSV_INDEX}e${EFFECT_INDEX}_${COND}" \
                --export=ALL,CONDITION_LABEL="$COND",EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",ANOVA_UNIT="$ANOVA_UNIT",ELECTRODES="$ELECTRODES",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_CORRECTION="$ANOVA_LABEL_CORRECTION",ANOVA_LABEL_ALPHA="$ANOVA_LABEL_ALPHA",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI" \
                sbatch_power_traces_dcc.sh
            # sleep 2
        done
    done
done




