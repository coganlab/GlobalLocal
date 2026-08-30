#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 5
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Aggregation half of a fanned-out coupling run.
#
# submit_decoding_with_coupling_electrode_sets_dcc.sh splits the draws across
# several decoding jobs; this one runs afterwards (held by --dependency=afterok)
# and does the coupled-vs-uncoupled comparison over the MASTER_RESULTS pickles
# they left on disk.
#
# It only reads saved results and runs the cluster permutations, so it needs a
# fraction of the decoding jobs' memory and time -- no epochs are loaded.
#
# --from_run_config imports run_decoding_dcc for SAVE_DIR / ROIS / CONDITION_LABEL
# rather than restating them, so this job cannot drift from the jobs it is
# aggregating. It therefore needs the same CONDITION_NAME the decoding jobs had.
#
# COUPLING_EXPECT_N_DRAWS makes a chunk that died an error instead of a quietly
# smaller null distribution.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

echo "Aggregating coupling comparison for condition: $CONDITION_NAME"

EXPECT_ARG=""
if [ -n "$COUPLING_EXPECT_N_DRAWS" ]; then
    EXPECT_ARG="--expect_n_draws $COUPLING_EXPECT_N_DRAWS"
fi

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_coupling_comparison_dcc.py \
    --from_run_config \
    $EXPECT_ARG
