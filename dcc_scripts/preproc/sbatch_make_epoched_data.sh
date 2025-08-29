#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 25
#SBATCH --mem=150G

subject=$1
task=$2
times=$3
within_base_times=$4
baseline_event=$5
base_times_length=$6
pad_length=$7
LAB_root=$8
channels=$9
dec_factor=$10
mark_outliers_as_nan=$11
outliers=$12
passband=$13

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

# this will always use the default stat func of scipy.stats.ttest_ind
python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/preproc/make_epoched_data.py --LAB_root /cwork/rl330