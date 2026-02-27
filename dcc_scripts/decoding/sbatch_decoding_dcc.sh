#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 20
#SBATCH --mem=150G
#SBATCH --time=120:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

echo "Running condition: $CONDITION_NAME"
python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_decoding_dcc.py