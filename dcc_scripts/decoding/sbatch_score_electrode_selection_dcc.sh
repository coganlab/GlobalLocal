#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Top-fraction electrode sets on held-out trials (score_electrode_selection_dcc.py).
# Arguments are passed through, e.g.
#   sbatch sbatch_score_electrode_selection_dcc.sh --rois lpfc --sign positive
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/score_electrode_selection_dcc.py "$@"
