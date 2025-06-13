#!/bin/bash
#SBATCH --output=logs/count_trials_%j.out
#SBATCH --error=logs/count_trials_%j.err
#SBATCH -p common,scavenger
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=15:00

subject=$1

echo "--- Job task started for subject: $subject on host: $(hostname) ---"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH=/hpc/home/$USER/coganlab/$USER/GlobalLocal

python count_trials.py --subject ${subject}

echo "--- Job task finished for subject: $subject ---"