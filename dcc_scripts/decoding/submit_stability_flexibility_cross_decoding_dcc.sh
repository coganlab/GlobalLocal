#!/bin/bash
# Submit A4 — cross-decoding of the stability/flexibility subpopulations:
# label transfer + within-block 2x2 + temporal generalization.
#
# A4 runs on the ORDINARY decoding pipeline (ROI LabeledArray pseudopopulation,
# cross-validated folds, refit shuffle null, cluster correction over windows);
# the only addition is a second label vector, so the hyperparameters below are
# the same ones the main decoding job uses.
#
# Every variable below can be overridden from the environment, so you never have
# to edit this file to change a run:
#
#   bash submit_stability_flexibility_cross_decoding_dcc.sh                    # real data, defaults
#   ROI=acc bash submit_stability_flexibility_cross_decoding_dcc.sh            # a different region
#   ELECTRODES=all bash submit_..._dcc.sh                                      # every ROI electrode
#   DATA_SOURCE=synthetic bash submit_..._dcc.sh                               # ground-truth dry run
#   DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal bash submit_..._dcc.sh     # the null code
#   TEMPGEN_GROUPS=both,all bash submit_..._dcc.sh                             # + unselected tempgen
#   FRAC_TRAIN=0.5 bash submit_..._dcc.sh                                      # set the train/test split
#   ANOVA_LABELS_CSV=/path/to/anova_labels.csv bash submit_..._dcc.sh          # one saved A1 run
#   ELECTRODE_DEFINITION=power_traces POWER_TRACES_RUN_DIR=/path/to/run \
#       bash submit_..._dcc.sh                                                 # define electrodes from
#                                                                              # the power-trace runs
#
# See docs/analysis_guide.md §17 for what each knob does and how to read the output.

# ---------------------------------------------------------------------------
# Data in: epochs file (high-gamma, rescaled) and the condition set.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE=${EPOCHS_ROOT_FILE:-"Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"}

# Default to the full crossed condition set. Keep this non-empty because the
# Python entrypoint treats a blank CONDITIONS as the default, but exporting an
# explicit value makes the submitted job log self-documenting.
# Unlike ordinary decoding, every cross-decoding condition set must carry both
# congruency and switchType on the same trials.  CONDITIONS may contain one or
# more compatible registry names, separated by spaces; submit one job per name.
# The full crossed set is the cross-decoding counterpart to the ordinary
# launcher's block-balanced condition battery.
if [[ -n "${CONDITIONS:-}" ]]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(stimulus_experiment_conditions)
fi

# Data source: 'real' loads epoched data; 'synthetic' validates the whole path
# with a ground-truth pseudopopulation. SYNTHETIC_CODE picks the planted truth:
#   shared     -> stability & flexibility on one axis (should cross-decode)
#   orthogonal -> distinct axes (should NOT cross-decode, though each is decodable)
DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_CODE=${SYNTHETIC_CODE:-shared}

# ---------------------------------------------------------------------------
# Which electrodes. Three separate choices:
#   ROI              which region (a key of src/analysis/config/rois.py)
#   ELECTRODES       which of that region's electrodes get loaded at all
#   REFERENCE_GROUP  the unselected group decoded alongside both/S_only/F_only
# ---------------------------------------------------------------------------
ROI=${ROI:-lpfc}
ELECTRODES=${ELECTRODES:-sig}            # 'sig' (baseline task-significant) or 'all'
REFERENCE_GROUP=${REFERENCE_GROUP:-all}  # '' to drop it
MIN_GROUP_SIZE=${MIN_GROUP_SIZE:-5}      # skip electrode groups smaller than this

# ---------------------------------------------------------------------------
# How the S/F electrode groups are defined.
#   anova         one ANOVA per electrode on the window-mean HG over
#                 [WINDOW_TMIN, WINDOW_TMAX], computed in this job
#   power_traces  read the finished within-electrode windowed-ANOVA runs and
#                 their cluster correction (needs the run directories)
#   csv           reuse S/F flags from an existing A1 anova_labels.csv
# ---------------------------------------------------------------------------
if [[ -z "${ELECTRODE_DEFINITION:-}" ]]; then
    # Real submissions mirror the normal decoder's saved-label selection. The
    # synthetic validation has planted labels and must remain self-contained.
    [[ "$DATA_SOURCE" == synthetic ]] && ELECTRODE_DEFINITION=anova || ELECTRODE_DEFINITION=csv
fi

# Keep this list in step with submit_specific_conditions_decoding_dcc.sh. Paths
# may name either anova_labels.csv itself or its result directory. As in the
# ordinary launcher, ANOVA_LABELS_CSV overrides the list with one saved run.
ANOVA_LABELS_CSVS=(
    # Add additional saved A1 runs here to submit the same condition battery for
    # each definition window/correction.
    /hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_proportion_none
    /hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none

)
if [[ -n "${ANOVA_LABELS_CSV:-}" ]]; then
    ANOVA_LABELS_CSVS=("$ANOVA_LABELS_CSV")
fi
ANOVA_LABEL_EFFECTS=(
    both lwpc lwps congruency switch_type
    lwpc_only lwps_only congruency_only switch_type_only
)
if [[ -n "${ANOVA_LABEL_EFFECT:-}" ]]; then
    ANOVA_LABEL_EFFECTS=("$ANOVA_LABEL_EFFECT")
fi
ANOVA_LABEL_CORRECTION=${ANOVA_LABEL_CORRECTION:-flags} # flags | none | fdr_bh
ANOVA_LABEL_ALPHA=${ANOVA_LABEL_ALPHA:-0.05}
ANOVA_LABEL_ROI=${ANOVA_LABEL_ROI:-} # only set when the CSV itself has an roi column
CONTRAST_MODE=${CONTRAST_MODE:-proportion}   # proportion=LWPC/LWPS interactions; condition=congruency/switch main effects
# For CSV runs, use the ordinary launcher's ANOVA_LABEL_CORRECTION and
# ANOVA_LABEL_ALPHA names. Explicit FDR_CORRECTION/ALPHA still take precedence.
FDR_CORRECTION=${FDR_CORRECTION:-$ANOVA_LABEL_CORRECTION}
WINDOW_TMIN=${WINDOW_TMIN:-0.0}          # seconds relative to stimulus onset
WINDOW_TMAX=${WINDOW_TMAX:-0.5}
ALPHA=${ALPHA:-$ANOVA_LABEL_ALPHA}
# Optional circularity guard for ELECTRODE_DEFINITION=anova. The ANOVA is fit
# over WINDOW_TMIN..WINDOW_TMAX on this fraction of physical trials; decoding
# and temporal generalization use only the complementary trials.
ELECTRODE_SELECTION_SPLIT=${ELECTRODE_SELECTION_SPLIT:-false}
ELECTRODE_SELECTION_FRAC=${ELECTRODE_SELECTION_FRAC:-0.3}
ELECTRODE_SELECTION_SEED=${ELECTRODE_SELECTION_SEED:-0}

POWER_TRACES_RUN_DIR=${POWER_TRACES_RUN_DIR:-}   # one run with all four interactions
POWER_TRACES_CPC=${POWER_TRACES_CPC:-}           # ...or one directory per interaction
POWER_TRACES_SPS=${POWER_TRACES_SPS:-}
POWER_TRACES_CPS=${POWER_TRACES_CPS:-}
POWER_TRACES_SPC=${POWER_TRACES_SPC:-}
POWER_TRACES_CORRECTION=${POWER_TRACES_CORRECTION:-fdr_bh}  # fdr_bh | cluster | none
POWER_TRACES_ROI=${POWER_TRACES_ROI:-}

# ---------------------------------------------------------------------------
# Decoding hyperparameters (the ordinary pipeline's).
# ---------------------------------------------------------------------------
WINDOW_SIZE=${WINDOW_SIZE:-64}       # decoding window, in samples
STEP_SIZE=${STEP_SIZE:-16}           # window stride, in samples
SAMPLING_RATE=${SAMPLING_RATE:-256}
FIRST_TIME_POINT=${FIRST_TIME_POINT:--1.0}
N_SPLITS=${N_SPLITS:-5}              # CV folds (or resamples per repeat, see FRAC_TRAIN)
N_REPEATS=${N_REPEATS:-10}           # CV repeats — the main runtime lever
EXPLAINED_VARIANCE=${EXPLAINED_VARIANCE:-0.8}
N_PERM=${N_PERM:-500}                # permutations for the cluster test over windows
SEED=${SEED:-0}

# Temporal generalization costs n_windows^2 decodes per matrix, so it runs only
# on these groups. 'both,all' adds the unselected reference matrix; '' skips it.
# Use `-` rather than `:-`: unset -> default "both", explicitly empty -> disable.
TEMPGEN_GROUPS=${TEMPGEN_GROUPS-both}
# Optional single requested transfer. Same labels = ordinary within-contrast
# decoding; different labels = cross-decoding. Leave both blank for the full
# backward-compatible bidirectional battery.
TRAIN_LABEL=${TRAIN_LABEL:-}
TEST_LABEL=${TEST_LABEL:-}

# Proportion of trials used for TRAINING in each split. Leave empty to keep
# StratifiedKFold at (N_SPLITS-1)/N_SPLITS; set it to sweep the proportion
# directly (StratifiedShuffleSplit), e.g. FRAC_TRAIN=0.5.
FRAC_TRAIN=${FRAC_TRAIN:-}

mkdir -p out

for CSV_INDEX in "${!ANOVA_LABELS_CSVS[@]}"; do
    ANOVA_LABELS_CSV=${ANOVA_LABELS_CSVS[$CSV_INDEX]}
    for EFFECT_INDEX in "${!ANOVA_LABEL_EFFECTS[@]}"; do
        ANOVA_LABEL_EFFECT=${ANOVA_LABEL_EFFECTS[$EFFECT_INDEX]}
        for COND in "${CONDITION_LIST[@]}"; do
            echo "Submitting stability/flexibility A4 cross-decoding"
            echo "  condition=$COND  anova_labels=${ANOVA_LABELS_CSV:-none}  effect=$ANOVA_LABEL_EFFECT"
            echo "  source=$DATA_SOURCE  roi=$ROI  electrodes=$ELECTRODES  definition=$ELECTRODE_DEFINITION  contrast=$CONTRAST_MODE  correction=$FDR_CORRECTION"
            sbatch --job-name="sf_xdec_a${CSV_INDEX}e${EFFECT_INDEX}_${DATA_SOURCE}_${ROI}" \
                --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",CONDITIONS="$COND",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",ALPHA="$ALPHA",CONTRAST_MODE="$CONTRAST_MODE",FDR_CORRECTION="$FDR_CORRECTION",ELECTRODE_SELECTION_SPLIT="$ELECTRODE_SELECTION_SPLIT",ELECTRODE_SELECTION_FRAC="$ELECTRODE_SELECTION_FRAC",ELECTRODE_SELECTION_SEED="$ELECTRODE_SELECTION_SEED",ROI="$ROI",ELECTRODE_DEFINITION="$ELECTRODE_DEFINITION",ANOVA_LABELS_CSV="$ANOVA_LABELS_CSV",ANOVA_LABEL_EFFECT="$ANOVA_LABEL_EFFECT",ANOVA_LABEL_ROI="$ANOVA_LABEL_ROI",POWER_TRACES_RUN_DIR="$POWER_TRACES_RUN_DIR",POWER_TRACES_CPC="$POWER_TRACES_CPC",POWER_TRACES_SPS="$POWER_TRACES_SPS",POWER_TRACES_CPS="$POWER_TRACES_CPS",POWER_TRACES_SPC="$POWER_TRACES_SPC",POWER_TRACES_CORRECTION="$POWER_TRACES_CORRECTION",POWER_TRACES_ROI="$POWER_TRACES_ROI",REFERENCE_GROUP="$REFERENCE_GROUP",TEMPGEN_GROUPS="$TEMPGEN_GROUPS",TRAIN_LABEL="$TRAIN_LABEL",TEST_LABEL="$TEST_LABEL",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",SAMPLING_RATE="$SAMPLING_RATE",FIRST_TIME_POINT="$FIRST_TIME_POINT",N_SPLITS="$N_SPLITS",N_REPEATS="$N_REPEATS",EXPLAINED_VARIANCE="$EXPLAINED_VARIANCE",FRAC_TRAIN="$FRAC_TRAIN",N_PERM="$N_PERM",MIN_GROUP_SIZE="$MIN_GROUP_SIZE",SEED="$SEED" \
                sbatch_stability_flexibility_cross_decoding_dcc.sh
        done
    done
done
