#!/bin/bash
#SBATCH --job-name=CountTrials
#SBATCH --output=logs/count_trials_%j.out
#SBATCH --error=logs/count_trials_%j.err
#SBATCH -p common,scavenger
#SBATCH -c 2                 
#SBATCH --mem=64G               
#SBATCH --time=10:00           

# This line takes the subject ID from your submit script
subject=$1

echo "--- Job Started for subject $subject on host: $(hostname) ---"

# Set up the environment (we know this works)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH=/hpc/home/$USER/coganlab/$USER/GlobalLocal

echo "--- Environment set up. Starting Python script. ---"

# Run your actual counting script
python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/count_trials_dcc.py --subject ${subject}

echo "--- Python script finished. Job complete. ---"