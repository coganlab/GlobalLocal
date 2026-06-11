#!/bin/bash
#SBATCH --output=out/plot_wavelets/slurm_%j.out
#SBATCH -e out/plot_wavelets/slurm_%j.err
#SBATCH -p common,scavenger
#SBATCH -c 4
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/run_plot_wavelets_dcc.py