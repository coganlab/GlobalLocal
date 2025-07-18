#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean


subjects=("D0057")

for subject in "${subjects[@]}"
do
    sbatch /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sbatch_get_channels_detail.sh $subject
done