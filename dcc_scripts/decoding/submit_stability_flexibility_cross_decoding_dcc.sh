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
#   ELECTRODE_DEFINITION=power_traces POWER_TRACES_RUN_DIR=/path/to/run \
#       bash submit_..._dcc.sh                                                 # define electrodes from
#                                                                              # the power-trace runs
#
# See docs/analysis_guide.md §17 for what each knob does and how to read the output.

# ---------------------------------------------------------------------------
# Data in: epochs file (high-gamma, rescaled) and the condition set.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE=${EPOCHS_ROOT_FILE:-"Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"}

# A4 needs the FULL 2x2x2x2 (congruency x inc-proportion x switchType x
# switch-proportion) — it decodes one contrast, scores the other, and splits
# each by a block factor.
CONDITIONS=${CONDITIONS:-stimulus_experiment_conditions}

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
# ---------------------------------------------------------------------------
ELECTRODE_DEFINITION=${ELECTRODE_DEFINITION:-anova}
WINDOW_TMIN=${WINDOW_TMIN:-0.0}          # seconds relative to stimulus onset
WINDOW_TMAX=${WINDOW_TMAX:-0.5}
ALPHA=${ALPHA:-0.05}

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
WINDOW_SIZE=${WINDOW_SIZE:-20}       # decoding window, in samples
STEP_SIZE=${STEP_SIZE:-10}           # window stride, in samples
N_SPLITS=${N_SPLITS:-5}              # CV folds (or resamples per repeat, see FRAC_TRAIN)
N_REPEATS=${N_REPEATS:-10}           # CV repeats — the main runtime lever
EXPLAINED_VARIANCE=${EXPLAINED_VARIANCE:-0.8}
N_PERM=${N_PERM:-500}                # permutations for the cluster test over windows
SEED=${SEED:-0}

# Temporal generalization costs n_windows^2 decodes per matrix, so it runs only
# on these groups. 'both,all' adds the unselected reference matrix; '' skips it.
TEMPGEN_GROUPS=${TEMPGEN_GROUPS:-both}

# Proportion of trials used for TRAINING in each split. Leave empty to keep
# StratifiedKFold at (N_SPLITS-1)/N_SPLITS; set it to sweep the proportion
# directly (StratifiedShuffleSplit), e.g. FRAC_TRAIN=0.5.
FRAC_TRAIN=${FRAC_TRAIN:-}

mkdir -p out

echo "Submitting stability/flexibility A4 cross-decoding"
echo "  source=$DATA_SOURCE  roi=$ROI  electrodes=$ELECTRODES  definition=$ELECTRODE_DEFINITION"
sbatch --job-name="sf_xdecode_${DATA_SOURCE}_${ROI}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",CONDITIONS="$CONDITIONS",WINDOW_TMIN="$WINDOW_TMIN",WINDOW_TMAX="$WINDOW_TMAX",ELECTRODES="$ELECTRODES",DATA_SOURCE="$DATA_SOURCE",SYNTHETIC_CODE="$SYNTHETIC_CODE",ALPHA="$ALPHA",ROI="$ROI",ELECTRODE_DEFINITION="$ELECTRODE_DEFINITION",POWER_TRACES_RUN_DIR="$POWER_TRACES_RUN_DIR",POWER_TRACES_CPC="$POWER_TRACES_CPC",POWER_TRACES_SPS="$POWER_TRACES_SPS",POWER_TRACES_CPS="$POWER_TRACES_CPS",POWER_TRACES_SPC="$POWER_TRACES_SPC",POWER_TRACES_CORRECTION="$POWER_TRACES_CORRECTION",POWER_TRACES_ROI="$POWER_TRACES_ROI",REFERENCE_GROUP="$REFERENCE_GROUP",TEMPGEN_GROUPS="$TEMPGEN_GROUPS",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",N_SPLITS="$N_SPLITS",N_REPEATS="$N_REPEATS",EXPLAINED_VARIANCE="$EXPLAINED_VARIANCE",FRAC_TRAIN="$FRAC_TRAIN",N_PERM="$N_PERM",MIN_GROUP_SIZE="$MIN_GROUP_SIZE",SEED="$SEED" \
    sbatch_stability_flexibility_cross_decoding_dcc.sh
