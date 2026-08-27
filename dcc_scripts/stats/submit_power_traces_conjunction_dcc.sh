#!/bin/bash
# Submit the conjunction on power_traces-defined electrodes.
#
# The electrode definition is READ BACK from a finished within-electrode windowed
# ANOVA + cluster-correction run, so the count test costs seconds and needs no
# epoched data. Only the optional continuous confound control loads epochs.
#
# Usage:
#   DATA_SOURCE=synthetic bash submit_power_traces_conjunction_dcc.sh   # dry run
#   bash submit_power_traces_conjunction_dcc.sh                         # real
#   RUN_CONTINUOUS=1 bash submit_power_traces_conjunction_dcc.sh        # + control
#
# Run it first with DATA_SOURCE=synthetic: that fabricates four runs with a KNOWN
# planted overlap (SYNTHETIC_OVERLAP vs SYNTHETIC_BASE_RATE), so a wrong answer
# there is a plumbing bug, not a data finding. SYNTHETIC_OVERLAP=0.25
# SYNTHETIC_BASE_RATE=0.25 is the null and must come back with MH OR ~ 1.

# ---------------------------------------------------------------------------
# The power_traces run(s) that define the electrodes.
#
# A run directory is the one holding summary.csv + run_config.json:
#   <dcc_scripts/power/figs>/<EPOCHS_ROOT_FILE>/anova_within_electrode/<conditions_save_name>
# Produce one with dcc_scripts/power/submit_specific_conditions_power_traces_dcc.sh
# (ANOVA_UNIT=electrode).
# ---------------------------------------------------------------------------
FIGS_ROOT="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs"
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_ind_equal_var_False_nan_policy_omit"

# Option A (preferred): ONE run whose ANOVA held all four factors. Every
# interaction — LWPC, LWPS and both cross controls — is read from its summary.csv,
# so all four are defined on exactly the same electrodes and trials.
# PT_RUN="${PT_RUN:-$FIGS_ROOT/$EPOCHS_ROOT_FILE/anova_within_electrode/stimulus_experiment_conditions_24_subjects}"

# Option B: four separate two-factor runs. Uncomment and unset PT_RUN above.
# Only CPC and SPS are required; the cross runs are the specificity controls.
# PT_RUN=""
PT_RUN_CPC="$FIGS_ROOT/$EPOCHS_ROOT_FILE/anova_within_electrode/stimulus_lwpc_conditions_24_subjects"
PT_RUN_SPS="$FIGS_ROOT/$EPOCHS_ROOT_FILE/anova_within_electrode/stimulus_lwps_conditions_24_subjects"
PT_RUN_CPS="$FIGS_ROOT/$EPOCHS_ROOT_FILE/anova_within_electrode/stimulus_congruency_by_switch_proportion_conditions_24_subjects"
PT_RUN_SPC="$FIGS_ROOT/$EPOCHS_ROOT_FILE/anova_within_electrode/stimulus_switch_type_by_incongruent_proportion_conditions_24_subjects"

# ---------------------------------------------------------------------------
# Label / conjunction knobs.
# ---------------------------------------------------------------------------
# NOTE: sbatch --export separates VAR=VALUE pairs with commas, so a
# comma-containing value cannot be passed there -- "THRESHOLDS=0.01,0.05" is
# parsed as THRESHOLDS=0.01 plus a bogus variable name, silently collapsing the
# sweep to its first point. Comma-capable variables are therefore `export`ed
# here and reach the job through --export=ALL instead.
export ROIS=${ROIS:-lpfc}          # comma-separated, or 'all' to pool over ROIs
CORRECTION=${CORRECTION:-cluster}   # fdr_bh | cluster | none
EFFECT_MODE=${EFFECT_MODE:-main} # interaction (LWPC/LWPS) | main (congruency/switchType)
ALPHA=${ALPHA:-0.05}
N_PERM_NULL=${N_PERM_NULL:-10000}
export THRESHOLDS=${THRESHOLDS:-0.01,0.025,0.05,0.10}
REQUIRE_ALL=${REQUIRE_ALL:-1}

# ---------------------------------------------------------------------------
# Continuous confound control (loads epochs; off by default).
# The count test cannot correct for per-electrode detection power or shared trial
# noise, and both inflate the 'both' cell. Turn this on before reporting a
# shared-core claim. Its window is taken from the run, not set here.
# ---------------------------------------------------------------------------
RUN_CONTINUOUS=${RUN_CONTINUOUS:-1}
ELECTRODES=${ELECTRODES:-sig}      # 'all' or 'sig'
export EFFECT_MEASURES=${EFFECT_MEASURES:-peak_t,cluster,cohens_d}
N_SPLITS=${N_SPLITS:-50}
N_PERM_CORR=${N_PERM_CORR:-1000}
MIN_ELEC=${MIN_ELEC:-3}

DATA_SOURCE=${DATA_SOURCE:-real}
SYNTHETIC_OVERLAP=${SYNTHETIC_OVERLAP:-0.6}
SYNTHETIC_BASE_RATE=${SYNTHETIC_BASE_RATE:-0.25}

mkdir -p out

echo "Submitting power_traces conjunction (source=$DATA_SOURCE, effects=$EFFECT_MODE, rois=$ROIS, correction=$CORRECTION)"
sbatch --job-name="pt_conj_${DATA_SOURCE}" \
    --export=ALL,DATA_SOURCE="$DATA_SOURCE",PT_RUN="$PT_RUN",PT_RUN_CPC="$PT_RUN_CPC",PT_RUN_SPS="$PT_RUN_SPS",PT_RUN_CPS="$PT_RUN_CPS",PT_RUN_SPC="$PT_RUN_SPC",EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",CORRECTION="$CORRECTION",EFFECT_MODE="$EFFECT_MODE",ALPHA="$ALPHA",N_PERM_NULL="$N_PERM_NULL",REQUIRE_ALL="$REQUIRE_ALL",RUN_CONTINUOUS="$RUN_CONTINUOUS",ELECTRODES="$ELECTRODES",N_SPLITS="$N_SPLITS",N_PERM_CORR="$N_PERM_CORR",MIN_ELEC="$MIN_ELEC",SYNTHETIC_OVERLAP="$SYNTHETIC_OVERLAP",SYNTHETIC_BASE_RATE="$SYNTHETIC_BASE_RATE" \
    sbatch_power_traces_conjunction_dcc.sh
