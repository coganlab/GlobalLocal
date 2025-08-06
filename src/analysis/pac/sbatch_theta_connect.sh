#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G

subject=$1

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/theta_connect.py \
    --subject ${subject} \
    --bids_root /cwork/rl330/BIDS-1.1_GlobalLocal/BIDS \
    --event Response \
    --rois ctx_rh_G_oc-temp_lat-fusifor \
    --roi_json /hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/subjects_electrodestoROIs_dict.json