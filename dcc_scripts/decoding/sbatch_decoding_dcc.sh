#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=100G
#SBATCH --time=48:00:00  # 48 hours (adjust based on your needs)

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/decoding/run_decoding_dcc.py
