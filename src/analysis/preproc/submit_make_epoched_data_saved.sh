#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean

# subjects=("D0057" "D0059" "D0063" "D0065" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110")

subjects=("D0057")

for subject in "${subjects[@]}"
do
    sbatch /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/preproc/sbatch_make_epoched_data_saved.sh $subject
done