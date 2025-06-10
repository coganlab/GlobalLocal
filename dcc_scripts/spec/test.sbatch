#!/bin/bash
#SBATCH --job-name=CondaTest
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH -p common
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=2:00

echo "--- TEST JOB STARTED ---"

echo "Attempting to source conda..."
source $(conda info --base)/etc/profile.d/conda.sh
echo "Conda sourced successfully."

echo "Attempting to activate ieeg environment..."
conda activate ieeg
echo "Conda environment activated successfully. Current env is: $CONDA_DEFAULT_ENV"

echo "--- TEST JOB FINISHED ---"