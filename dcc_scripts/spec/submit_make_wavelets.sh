#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean
#no D0063 cause I already did that one
subjects=("D0057" "D0059" "D0065" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110" "D0117")
#subjects=("D0063")

analysis_type="multitaper"

for subject in "${subjects[@]}"
do
    sbatch /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/spec/sbatch_make_wavelets.sh $subject $analysis_type
done
