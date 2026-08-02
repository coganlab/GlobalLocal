#!/bin/bash
#SBATCH --output=out/slurm_%j.out
#SBATCH -e out/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg  # needs scipy + statsmodels + mne

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/run_stability_flexibility_timing_dcc.py
