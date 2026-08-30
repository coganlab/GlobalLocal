#!/bin/bash
# Launcher: decode from electrodes that are COUPLED by high-gamma envelope
# correlation (the PAC path's significant pairs), against matched sets of
# non-coupled electrodes.
#
# The question
# ------------
# Ruoxi's `high_corr_<pairtype>_<condition>_<subject>.csv` files flag every
# channel pair with significantly high gamma-envelope correlation. An electrode
# is "coupled" if it takes part in at least one such pair. Do occ electrodes
# coupled to LPFC carry more decodable task / big-letter / small-letter
# information than occ electrodes that are not — and symmetrically for the LPFC
# side?
#
# Why the control side is MANY sets, not one
# ------------------------------------------
# The decoding bootstrap resamples TRIALS, not electrodes
# (labeled_array_utils.make_bootstrapped_labeled_arrays_for_roi): every
# electrode in a set is in every bootstrap, so accuracy tracks set size. There
# are far more non-coupled electrodes than coupled ones, so an unmatched
# comparison would just be measuring the larger set. This launcher therefore
# runs the battery on the coupled set plus N_DRAWS independent n-matched draws
# of non-coupled electrodes, and the comparison asks where the coupled trace
# falls in the distribution over draws.
#
# Run the counting pass FIRST
# ---------------------------
# Coupled pairs are sparse. Before spending cluster time, check that anything
# survives the bipolar->monopolar expansion and the decoding-pool intersection:
#
#   python dcc_scripts/decoding/report_coupling_counts.py \
#       --csv_dir "$COUPLING_CSV_DIR" --condition Stimulus \
#       --pair_types lpfc-occ lpfc-lpfc occ-occ
#
# Usage:
#   COUPLING_CSV_DIR=/path/to/high_corr_stats \
#       bash submit_decoding_with_coupling_electrode_sets_dcc.sh
#   PAIR_TYPES="lpfc-occ" N_DRAWS=50 \
#       bash submit_decoding_with_coupling_electrode_sets_dcc.sh
#   MATCH_LEVEL=within_subject MATCH_DEGREE=true \
#       bash submit_decoding_with_coupling_electrode_sets_dcc.sh

# ---------------------------------------------------------------------------
# What to DECODE. One job per (condition, pair type).
# ---------------------------------------------------------------------------
if [ -n "$CONDITIONS" ]; then
    read -r -a CONDITION_LIST <<< "$CONDITIONS"
else
    CONDITION_LIST=(
        stimulus_task_conditions
        stimulus_big_letter_conditions
        # stimulus_small_letter_conditions
    )
fi

# ---------------------------------------------------------------------------
# Which coupling pair types to run. Each decides which ROIs get decoded, so
# ROIS below must contain them.
#   lpfc-occ   the theory-driven one: cross-region coupling
#   lpfc-lpfc  within-region control
#   occ-occ    within-region control
# ---------------------------------------------------------------------------
if [ -n "$PAIR_TYPES" ]; then
    read -r -a PAIR_TYPE_LIST <<< "$PAIR_TYPES"
else
    # PAIR_TYPE_LIST=(lpfc-occ lpfc-lpfc occ-occ)
    PAIR_TYPE_LIST=(lpfc-occ)

fi

# ---------------------------------------------------------------------------
# Selection parameters.
#   COUPLING_CSV_DIR   where the high_corr_*.csv files live. Unset -> the
#                      standard derivatives path under LAB_root.
#   COUPLING_CONDITION the condition token in the CSV FILENAMES that defines
#                      coupling. Keep it 'Stimulus' (all trials): defining
#                      coupling on the same split you decode is circular.
#                      NOTE this does NOT make the definition orthogonal to
#                      the decodes -- pooled across-trial correlation partly
#                      measures shared condition tuning, which is exactly what
#                      makes a condition decodable. See analysis_guide.md 22.3
#                      'The signal-correlation confound' before interpreting a
#                      positive result.
#   N_DRAWS            matched control draws. Cost driver -- one full decoding
#                      battery each -- and it floors the exceedance p-value at
#                      1/(N+1), so 20 draws cannot go below p = 0.048.
#   MATCH_LEVEL        global | within_subject
#   MATCH_DEGREE       match controls on how many partners each electrode was
#                      tested against ("coupled" = significant in >=1 pair, so
#                      a high-degree electrode qualifies more easily)
# ---------------------------------------------------------------------------
COUPLING_CSV_DIR=${COUPLING_CSV_DIR:-}
COUPLING_CONDITION=${COUPLING_CONDITION:-Stimulus}
N_DRAWS=${N_DRAWS:-20}
SEED=${SEED:-0}
MATCH_LEVEL=${MATCH_LEVEL:-global}
MATCH_DEGREE=${MATCH_DEGREE:-false}

# lpfc and occ must both be decodable for the cross-region pair type.
#
# NOTE: sbatch --export separates VAR=VALUE pairs with commas, so "ROIS=lpfc,occ"
# cannot be passed in that list -- it is parsed as ROIS=lpfc plus a bogus
# variable name, silently dropping occ and leaving a cross-region pair type
# decoding only lpfc. Export it here and let --export=ALL carry it instead.
export ROIS=${ROIS:-lpfc,occ}

mkdir -p out

for PAIR in "${PAIR_TYPE_LIST[@]}"; do
    for COND in "${CONDITION_LIST[@]}"; do
        echo "Submitting coupling decoding: decode=$COND pair_type=$PAIR " \
             "(coupling_condition=$COUPLING_CONDITION, draws=$N_DRAWS, " \
             "match=$MATCH_LEVEL, degree=$MATCH_DEGREE)"
        sbatch --job-name="dec_coup_${PAIR}_${COND}" \
            --export=ALL,CONDITION_NAME="$COND",\
COUPLING_ELECTRODE_SELECTION=true,\
COUPLING_CSV_DIR="$COUPLING_CSV_DIR",\
COUPLING_PAIR_TYPE="$PAIR",\
COUPLING_CONDITION="$COUPLING_CONDITION",\
COUPLING_N_DRAWS="$N_DRAWS",\
COUPLING_SEED="$SEED",\
COUPLING_MATCH_LEVEL="$MATCH_LEVEL",\
COUPLING_MATCH_DEGREE="$MATCH_DEGREE" \
            sbatch_decoding_dcc.sh
    done
done
