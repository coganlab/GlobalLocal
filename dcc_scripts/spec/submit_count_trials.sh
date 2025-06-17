#!/bin/bash

# Create directories for logs and results
mkdir -p trial_counts
mkdir -p logs

# Get a list of all subjects from the BIDS directory
BIDS_DIR="/cwork/etb28/BIDS-1.1_GlobalLocal"
subjects_list=($(ls -d $BIDS_DIR/BIDS/sub-*/ | xargs -n 1 basename | sed 's/sub-//'))

# Count the number of subjects to create the job array range
num_subjects=${#subjects_list[@]}
array_limit=$((num_subjects - 1))

echo "Found ${num_subjects} subjects. Submitting one job array with ${num_subjects} tasks..."

# Submit the job array to SLURM
# The --array flag tells SLURM to run this N times
# The sbatch script and the subject list are passed as arguments
sbatch --array=0-$array_limit sbatch_count_trials.sh "${subjects_list[@]}"