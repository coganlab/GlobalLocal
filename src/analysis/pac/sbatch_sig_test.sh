#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00


subjA=${1:-D0063}
regionA=${2:-lpfc}
condA=${3:-stimulus_r}
tstartA=${4:--0.5}
tendA=${5:-0.0}

subjB=${6:-D0063}
regionB=${7:-lpfc}
condB=${8:-stimulus_s}
tstartB=${9:--0.5}
tendB=${10:-0.0}



source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg # make sure this works


python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_test.py \
    --subjA "$subjA" --regionA "$regionA" --condA "$condA" --tstartA "$tstartA" --tendA "$tendA" \
    --subjB "$subjB" --regionB "$regionB" --condB "$condB" --tstartB "$tstartB" --tendB "$tendB"
