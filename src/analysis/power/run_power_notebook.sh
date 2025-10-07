#!/bin/bash
#SBATCH --job-name=run_power_traces
#SBATCH --time=02:00:00
#SBATCH --mem=50G

#SBATCH --output=power_traces_%j.out
#SBATCH --error=power_traces_%j.err

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

echo "Executing Jupyter Notebook..."
# --- Use the correct path to your project on the cluster ---
jupyter nbconvert --to notebook --execute /hpc/group/coganlab/etb28/GlobalLocal/src/analysis/power/power_traces.ipynb

echo "Job finished."