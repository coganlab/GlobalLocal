#!/bin/bash
#SBATCH --output=out/make_wavelets/slurm_%j.out
#SBATCH -e out/make_wavelets/slurm_%j.err
#SBATCH -p common,scavenger
#SBATCH -c 25
#SBATCH --mem=150G

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/run_make_wavelets_dcc.py