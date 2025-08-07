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
    "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/Coherence_data/coherence_D0100_['ctx_lh_G_occipital_middle', 'ctx_lh_S_oc_middle_and_Lunatus', 'ctx_lh_Pole_occipital', 'ctx_lh_S_oc_sup_and_transversal']_Response_summary.csv" \
    --o "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/theta_coh_fig/coherence_D0100_['ctx_lh_G_occipital_middle', 'ctx_lh_S_oc_middle_and_Lunatus', 'ctx_lh_Pole_occipital', 'ctx_lh_S_oc_sup_and_transversal']_Response_summary.png"