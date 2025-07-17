#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean

#no D0094, D0103 because already did it
#subjects=("D0057" "D0059" "D0063" "D0065" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0107A" "D0110" "D0117")
subjects=("D0059")

for subject in "${subjects[@]}"
do
    sbatch /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/sbatch_plot_wavelets.sh $subject
done