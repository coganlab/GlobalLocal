#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH --time=120:00:00  # 48 hours (adjust based on your needs)

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

# 4. Define the EXACT python executable you want to use
THE_RIGHT_PYTHON="/hpc/home/etb28/miniconda3/envs/ieeg/bin/python"

# 5. Define the script to run
THE_SCRIPT="/hpc/home/etb28/coganlab/etb28/GlobalLocal/dcc_scripts/power/run_power_traces_dcc.py"
#THE_SCRIPT="/hpc/home/etb28/coganlab/etb28/GlobalLocal/dcc_scripts/power/check_env.py"

# 6. Run the script using that specific python in UNBUFFERED mode
echo "--- sbatch: Running script with: $THE_RIGHT_PYTHON -u ---"

# The -u flag is the important part
$THE_RIGHT_PYTHON -u $THE_SCRIPT

#python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/run_power_traces_dcc.py
