#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 25
#SBATCH --mem=100G

subject=$1

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/theta_connect_plot.py \
    "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/Coherence_data/coherence_D0057_['ctx_rh_G_front_middle', 'ctx_rh_G_front_sup', 'ctx_rh_S_front_sup']_summary.csv" \
    --o "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/theta_coh_fig/coherence_D0057_['ctx_rh_G_front_middle', 'ctx_rh_G_front_sup', 'ctx_rh_S_front_sup']_summary.png"