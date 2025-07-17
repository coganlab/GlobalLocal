#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger
#SBATCH -c 25
#SBATCH --mem=150G

subject=$1

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

#to be able to search for src file
export PYTHONPATH=/hpc/home/$USER/coganlab/$USER/GlobalLocal

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/plot_wavelets_dcc.py --subject ${subject}