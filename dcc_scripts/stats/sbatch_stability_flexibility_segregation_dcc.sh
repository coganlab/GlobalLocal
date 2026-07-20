#!/bin/bash
#SBATCH --output=out/slurm_%j.out
#SBATCH -e out/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg  # make sure this env has scipy + statsmodels

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/run_stability_flexibility_segregation_dcc.py
