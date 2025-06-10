#!/bin/bash
#SBATCH --job-name=HelloWorldTest
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH -p common
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --time=2:00

echo "--- TEST JOB HAS STARTED ---"
echo "Running on host: $(hostname)"
echo "My user is: $(whoami)"
echo "--- TEST JOB IS FINISHING ---"