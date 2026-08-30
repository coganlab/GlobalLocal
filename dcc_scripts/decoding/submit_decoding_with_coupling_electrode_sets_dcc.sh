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
#   DRAWS_PER_JOB      how many draws each SLURM job decodes (see below).
#   MATCH_LEVEL        global | within_subject
#   MATCH_DEGREE       match controls on how many partners each electrode was
#                      tested against ("coupled" = significant in >=1 pair, so
#                      a high-degree electrode qualifies more easily)
#
# Why the draws are fanned out across jobs
# ----------------------------------------
# One decoding battery took ~6h wall clock in the runs this was written for, and
# sbatch_decoding_dcc.sh allows 48h. A single job doing the coupled set plus 20
# draws needs ~128h and is killed a third of the way through, after which the
# comparison never runs. So the draws are split into chunks of DRAWS_PER_JOB,
# submitted as independent jobs sharing one save_dir, and a final job -- held by
# --dependency=afterok until they all succeed -- runs the comparison over the
# results on disk.
#
# This is a scheduling change only. Draw k is seeded `SEED + k` and named from k
# alone, so the fanned-out sets are byte-identical to the ones a single job would
# have built, and the comparison is the same comparison over the same 20 draws.
#
# Only the FIRST chunk decodes the coupled set (COUPLING_DECODE_COUPLED). The
# coupled set is the same electrodes in every job, so decoding it in each would
# buy nothing and cost a full battery per job.
#
# Set DRAWS_PER_JOB to a number >= N_DRAWS to go back to one job for everything.
# ---------------------------------------------------------------------------
COUPLING_CSV_DIR=${COUPLING_CSV_DIR:-}
COUPLING_CONDITION=${COUPLING_CONDITION:-Stimulus}
N_DRAWS=${N_DRAWS:-20}
DRAWS_PER_JOB=${DRAWS_PER_JOB:-4}
SEED=${SEED:-0}
MATCH_LEVEL=${MATCH_LEVEL:-global}
MATCH_DEGREE=${MATCH_DEGREE:-false}

if [ "$DRAWS_PER_JOB" -lt 1 ]; then
    echo "DRAWS_PER_JOB must be >= 1, got $DRAWS_PER_JOB" >&2
    exit 1
fi

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
             "per_job=$DRAWS_PER_JOB, match=$MATCH_LEVEL, degree=$MATCH_DEGREE)"

        CHUNK_JOB_IDS=()
        FIRST_CHUNK=true
        START=0
        while [ "$START" -lt "$N_DRAWS" ]; do
            END=$((START + DRAWS_PER_JOB - 1))
            if [ "$END" -ge "$N_DRAWS" ]; then END=$((N_DRAWS - 1)); fi

            # The coupled set rides along with the first chunk only.
            if [ "$FIRST_CHUNK" = true ]; then
                DECODE_COUPLED=true
            else
                DECODE_COUPLED=false
            fi

            JOB_ID=$(sbatch --parsable \
                --job-name="dec_coup_${PAIR}_${COND}_d${START}-${END}" \
                --export=ALL,CONDITION_NAME="$COND",\
COUPLING_ELECTRODE_SELECTION=true,\
COUPLING_CSV_DIR="$COUPLING_CSV_DIR",\
COUPLING_PAIR_TYPE="$PAIR",\
COUPLING_CONDITION="$COUPLING_CONDITION",\
COUPLING_N_DRAWS="$N_DRAWS",\
COUPLING_SEED="$SEED",\
COUPLING_MATCH_LEVEL="$MATCH_LEVEL",\
COUPLING_MATCH_DEGREE="$MATCH_DEGREE",\
COUPLING_DRAW_RANGE="${START}-${END}",\
COUPLING_DECODE_COUPLED="$DECODE_COUPLED" \
                sbatch_decoding_dcc.sh)
            echo "  chunk draws ${START}-${END} (coupled=$DECODE_COUPLED) -> job $JOB_ID"
            CHUNK_JOB_IDS+=("$JOB_ID")

            FIRST_CHUNK=false
            START=$((END + 1))
        done

        # One chunk covered everything AND carried the coupled set, so that job
        # already ran the comparison itself. Nothing left to aggregate.
        if [ "${#CHUNK_JOB_IDS[@]}" -eq 1 ]; then
            echo "  single chunk — the comparison runs inside job ${CHUNK_JOB_IDS[0]}"
            continue
        fi

        # afterok, not afterany: aggregating over a partial set of draws would
        # quietly shrink the null distribution rather than fail.
        DEPS=$(IFS=:; echo "${CHUNK_JOB_IDS[*]}")
        AGG_ID=$(sbatch --parsable \
            --job-name="cmp_coup_${PAIR}_${COND}" \
            --dependency=afterok:"$DEPS" \
            --export=ALL,CONDITION_NAME="$COND",\
COUPLING_PAIR_TYPE="$PAIR",\
COUPLING_N_DRAWS="$N_DRAWS",\
COUPLING_EXPECT_N_DRAWS="$N_DRAWS" \
            sbatch_coupling_comparison_dcc.sh)
        echo "  comparison -> job $AGG_ID (waits on ${#CHUNK_JOB_IDS[@]} chunks)"
    done
done
