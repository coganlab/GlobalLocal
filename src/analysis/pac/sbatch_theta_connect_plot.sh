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
    "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow/coherence_D0063_acc_stimulus_c_(-0.5, 0.0)_summary.csv" \
    --o "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow_plots/coherence_D0063_acc_stimulus_c_(-0.5, 0.0)_summary.png"