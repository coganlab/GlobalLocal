#!/bin/bash
# Figure 5 with the top LWPC and LWPS electrodes (fig5_top_sets.py) for the four
# anatomy runs: all and task-significant electrodes, in lPFC and DLPFC. One job
# per run (2 CPUs, 32 GB, minutes). Run from dcc_scripts/stats; extra arguments
# (e.g. --frac 0.1) are passed to every job.
EPOCHS_ROOT_FILE="${EPOCHS_ROOT_FILE:-Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20}"
RESULTS="$PWD/results/$EPOCHS_ROOT_FILE"

mkdir -p out
for ELECS in all sig; do
    for ROI in lpfc dlpfc; do
        sbatch --job-name="fig5_top_${ROI}_${ELECS}" sbatch_fig5_top_sets_dcc.sh \
            --anatomy-dir "$RESULTS/anatomy_a1_${ROI}_window_0.0to1.5s_${ELECS}/continuous" \
            --seg-dir "$RESULTS/segregation_results/window_0.0to1.5s_${ELECS}_${ROI}_proportion_cohens_d_fdr_bh_main_effects" \
            --brain "$@"
    done
done
