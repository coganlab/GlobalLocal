#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00



default_subjA=("D0057")
default_subjB=("D0057")


if [ $# -ge 1 ]; then
    
    IFS=' ' read -r -a subjA <<< "${1}"
else
    subjA=("${default_subjA[@]}")
fi

regionA=${2:-lpfc}
condA=${3:-stimulus_c}
tstartA=${4:-0.0}
tendA=${5:-0.5}

if [ $# -ge 6 ]; then

    IFS=' ' read -r -a subjB <<< "${6}"
else
    subjB=("${default_subjB[@]}")
fi

regionB=${7:-lpfc}
condB=${8:-stimulus_i}
tstartB=${9:-0.0}
tendB=${10:-0.5}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg # make sure this works

subjA_args=""
for subj in "${subjA[@]}"; do
    subjA_args="$subjA_args --subjA $subj"
done

subjB_args=""
for subj in "${subjB[@]}"; do
    subjB_args="$subjB_args --subjB $subj"
done


python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_test.py \
    $subjA_args --regionA "$regionA" --condA "$condA" --tstartA "$tstartA" --tendA "$tendA" \
    $subjB_args --regionB "$regionB" --condB "$condB" --tstartB "$tstartB" --tendB "$tendB"
