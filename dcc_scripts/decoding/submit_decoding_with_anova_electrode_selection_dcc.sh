#!/bin/bash
# Launcher: define electrodes with the POWER-TRACES cluster-corrected ANOVA on a
# held-out slice of trials, then decode on the disjoint remainder -- once per
# electrode set, in one job.
#
# How this differs from submit_decoding_with_electrode_definition_split_dcc.sh
# ---------------------------------------------------------------------------
# That script does the same TRIAL SPLIT, but its selector is a task-
# responsiveness t-test (window vs. baseline, FDR across channels) and it
# produces exactly ONE electrode set. This script:
#
#   * selects with `run_within_electrode_windowed_anova_cluster_correction` --
#     the same within-electrode windowed ANOVA + permutation cluster correction
#     the power-traces figures use -- so "LWPC electrodes" here means exactly
#     what it means in the power-trace figure;
#   * runs that selection once per condition set (LWPC and LWPS by default) and
#     builds the SET ALGEBRA over the results: lwpc, lwps, lwpc_only, lwps_only,
#     overlap, union;
#   * runs the decoding battery once per set, each into its own
#     `elecset_<name>/` subdirectory, with the decoded condition and the
#     electrode set in every figure title and file name.
#
# The comparison it is built for: do LWPC-only and LWPS-only electrodes decode
# their own contrast but not the other's (segregated subpopulations), or does
# the overlap set carry both (a shared code)?
#
# Splitting on trials is what makes the "define on X, decode X" cells readable
# at all: the decoder never sees a selection trial. The contrast-level guard
# still applies on top -- see docs/analysis_guide.md §14.1 and §21.
#
# Usage:
#   bash submit_decoding_with_anova_electrode_selection_dcc.sh
#   FRAC_SELECT=0.3 N_PERM=500 bash submit_decoding_with_anova_electrode_selection_dcc.sh
#   SETS=lwpc_only,lwps_only,overlap bash submit_decoding_with_anova_electrode_selection_dcc.sh
#   CONDITIONS="stimulus_lwpc_conditions stimulus_lwps_conditions" \
#       bash submit_decoding_with_anova_electrode_selection_dcc.sh

# ---------------------------------------------------------------------------
# Conditions to DECODE (space-separated). One job each.
# ---------------------------------------------------------------------------
if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_lwpc_conditions
        stimulus_lwps_conditions
    )
fi

# ---------------------------------------------------------------------------
# Selection parameters.
#   FRAC_SELECT  fraction of each subject's trials spent DEFINING electrodes
#                (0.3 -> 30% select / 70% decode)
#   SEL_LABELS   comma-separated condition_registry keys whose ANOVA defines the
#                electrode sets. Each needs 'anova_factors' in the registry.
#   SETS         comma-separated electrode sets to decode; empty -> all
#                (each label, each <label>_only, overlap, union)
#   EFFECT       'interaction' (the LWPC / LWPS term), 'any', or an explicit
#                statsmodels effect name
#   N_PERM       permutations per electrode -- the cost driver
#   ALPHA        FDR q (or raw cluster p when USE_FDR=false)
#   STRATA       metadata columns to stratify the trial split on. Must be real
#                metadata columns: congruency, task_sequence, block_type, ...
# ---------------------------------------------------------------------------
FRAC_SELECT=${FRAC_SELECT:-0.3}
SEL_LABELS=${SEL_LABELS:-stimulus_lwpc_conditions,stimulus_lwps_conditions}
SETS=${SETS:-}
EFFECT=${EFFECT:-interaction}
N_PERM=${N_PERM:-200}
SEED=${SEED:-0}
ALPHA=${ALPHA:-0.05}
USE_FDR=${USE_FDR:-true}
MIN_TRIALS_PER_CELL=${MIN_TRIALS_PER_CELL:-2}
STRATA=${STRATA:-congruency,task_sequence,block_type}

mkdir -p out

for COND in "${CONDITION_LIST[@]}"; do
    echo "Submitting ANOVA-selected decoding: decode=$COND select_on=$SEL_LABELS " \
         "(frac_select=$FRAC_SELECT, effect=$EFFECT, n_perm=$N_PERM, sets=${SETS:-all})"
    sbatch --job-name="dec_anovasel_${COND}" \
        --export=ALL,CONDITION_NAME="$COND",\
ANOVA_ELECTRODE_SELECTION=true,\
ELECTRODE_SELECTION_FRAC="$FRAC_SELECT",\
ELECTRODE_SELECTION_LABELS="$SEL_LABELS",\
ELECTRODE_SELECTION_SETS="$SETS",\
ELECTRODE_SELECTION_EFFECT="$EFFECT",\
ELECTRODE_SELECTION_N_PERM="$N_PERM",\
ELECTRODE_SELECTION_SEED="$SEED",\
ELECTRODE_SELECTION_ALPHA="$ALPHA",\
ELECTRODE_SELECTION_USE_FDR="$USE_FDR",\
ELECTRODE_SELECTION_MIN_TRIALS_PER_CELL="$MIN_TRIALS_PER_CELL",\
ELECTRODE_SELECTION_STRATA="$STRATA" \
        sbatch_decoding_dcc.sh
done
