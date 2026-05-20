#!/bin/bash
# Submit one decoding job per held-out subject (LOO).

CONDITIONS=(
    "stimulus_congruency_by_switch_prop_block_balanced_conditions"
    "stimulus_switch_type_by_inc_prop_block_balanced_conditions"
)

LEAVE_OUT_SUBJECTS=(D0077 D0090 D0094 D0100 D0102 D0103 D0107A D0110 D0116 D0117 D0121 D0130 D0133 D0134)

# Create output directory if needed
mkdir -p out

for COND in "${CONDITIONS[@]}"; do

    for SUBJ in "${LEAVE_OUT_SUBJECTS[@]}"; do
        echo "Submitting LOO: leaving out $SUBJ (condition=$COND)"
        sbatch --job-name="dec_loo-${SUBJ}" \
            --export=ALL,CONDITION_NAME="$COND",LEAVE_OUT="$SUBJ" \
            sbatch_decoding_dcc.sh
    done

done


mkdir -p out

