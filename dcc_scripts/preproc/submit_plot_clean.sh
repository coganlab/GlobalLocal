#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean

# subjects=("D0057" "D0059" "D0063" "D0065" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110")

# subjects=("D0130" "D0133" "D0134" "D0137")
subjects=("D0137")

for subject in "${subjects[@]}"
do
    sbatch /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/preproc/sbatch_plot_clean.sh $subject
done