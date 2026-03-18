#!/bin/bash

# loop over subjects and submit the sbatch script for plot clean
subjects=("D0057" "D0059" "D0065" "D0063" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110" "D0116" "D0117" "D0121" "D0130" "D0133" "D0134" "D0137")
# subjects=("D0130" "D0133" "D0134" "D0137")
# subjects=("D0065")

# Define all arguments for the Python script
TASK="GlobalLocal"
TIMES="-1 1.5"
WITHIN_BASE_TIMES="-1 0"
BASELINE_EVENT="Stimulus"
BASE_TIMES_LENGTH=0.5
PAD_LENGTH=1.5 # this needs to at least be 1s for theta to avoid edge artifacts. Can be lower for higher frequencies (e.g., 0.5s for high gamma). 
LAB_ROOT="/cwork/jz421"
CHANNELS="None"
DEC_FACTOR=8
OUTLIER_POLICY="ignore"
OUTLIERS=10
THRESHOLD_PERCENT=5.0
PASSBAND="4 8"
FILTER_METHOD="bandpass" # choose either filterbank_hilbert or bandpass for now.
METHOD='fir' # the filter method for bandpass, if you choose to use bandpass. Can be fir or iir. Irrelevant if you use filterbank_hilbert.
FIR_DESIGN='firwin' # the fir design if you choose to use fir for bandpass. Irrelevant if you use filterbank_hilbert or you use bandpass with an iir filter.
SBATCH_SCRIPT_PATH="/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/preproc/sbatch_make_epoched_data.sh"

# This loop submits a separate job for each subject
echo "Submitting ${#subjects[@]} jobs..."
for subject in "${subjects[@]}"
do
    echo "-> Submitting job for subject: $subject"
    sbatch "$SBATCH_SCRIPT_PATH" \
        "$subject" \
        "$TASK" \
        "$TIMES" \
        "$WITHIN_BASE_TIMES" \
        "$BASELINE_EVENT" \
        "$BASE_TIMES_LENGTH" \
        "$PAD_LENGTH" \
        "$LAB_ROOT" \
        "$CHANNELS" \
        "$DEC_FACTOR" \
        "$OUTLIER_POLICY" \
        "$OUTLIERS" \
        "$THRESHOLD_PERCENT" \
        "$PASSBAND" \
        "$FILTER_METHOD" \
        "$METHOD" \
        "$FIR_DESIGN"
done

echo "All jobs submitted."