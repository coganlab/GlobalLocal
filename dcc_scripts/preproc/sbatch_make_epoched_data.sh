#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 10
#SBATCH --mem=60G

subject=$1
task=$2
times=$3
within_base_times=$4
baseline_event=$5
base_times_length=$6
pad_length=$7
LAB_root=$8
channels=$9
dec_factor=${10}
outlier_policy=${11}
outliers=${12}
threshold_percent=${13}
passband=${14}

# can't pass in stat_func because it's a function, sadly...manually set this in make_epoched_data_dcc.py

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

# this will always use the default stat func of scipy.stats.ttest_ind
python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/preproc/make_epoched_data.py \
    --subjects "${subject}" \
    --task "${task}" \
    --times ${times} \
    --within_base_times ${within_base_times} \
    --baseline_event "${baseline_event}" \
    --base_times_length ${base_times_length} \
    --pad_length ${pad_length} \
    --LAB_root "${LAB_root}" \
    --channels "${channels}" \
    --dec_factor ${dec_factor} \
    --outlier_policy "${outlier_policy}" \
    --outliers ${outliers} \
    --threshold_percent ${threshold_percent} \
    --passband ${passband}