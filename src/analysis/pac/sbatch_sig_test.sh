#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00

set -euo pipefail
default_subj=("D0090")

if [ $# -ge 1 ]; then
    IFS=' ' read -r -a subj <<< "${1}"
else
    subj=("${default_subj[@]}")
fi

region=${2:-lpfc}
condA=${3:-stimulus_c}
condB=${4:-stimulus_i}
perm_trials_flag=${5:-true}
n_perm=${6:-200}

echo "=========================================="
echo "Time Window Comparison Analysis (condA+condB combined)"
echo "=========================================="
echo "Subjects: ${subj[*]}"
echo "Region: $region"
echo "Condition A: $condA"
echo "Condition B: $condB"
echo "Permutation test: $perm_trials_flag (n_perm=$n_perm)"
echo "=========================================="

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -d "$HOME/miniconda3/etc/profile.d" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/anaconda3/etc/profile.d" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Warning: conda not found; proceeding without activation."
fi

conda activate ieeg

PYTHON_SCRIPT="/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_test.py"

cmd=(python "$PYTHON_SCRIPT")
cmd+=( --subj "${subj[@]}" )
cmd+=( --region "$region" )
cmd+=( --condA "$condA" )
cmd+=( --condB "$condB" )

if [ "$perm_trials_flag" = "true" ] || [ "$perm_trials_flag" = "1" ]; then
    cmd+=( --perm_trials --n_perm "$n_perm" )
fi

echo ""
echo "Executing command:"
printf '%q ' "${cmd[@]}"
echo -e "\n"

"${cmd[@]}"
