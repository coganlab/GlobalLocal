#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=16:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg  # needs scipy + statsmodels + scikit-learn + ieeg

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py
