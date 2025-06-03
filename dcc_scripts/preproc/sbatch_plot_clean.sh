#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 25
#SBATCH --mem=32G

subject=$1

conda activate ieeg # make sure this works

python /hpc/home/jz421/coganlab/GlobalLocal/src/analysis/preproc/plot_clean.py --subject ${subject}


