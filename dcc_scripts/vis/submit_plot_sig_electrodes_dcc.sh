#!/bin/bash
# Submit significant-electrode brain-plotting jobs, one per named condition set.
#
# WHICH comparisons get plotted is chosen here: just list the labels you want
# from the PLOT_CONDITION_SETS registry in condition_plot_specs.py. Each label
# is submitted as its own job. Swapping in a different comparison = edit this
# list (or add a registry entry) -- no need to touch the python.
#
# Run from the dcc_scripts/vis directory so the sbatch relative out/ paths
# resolve:
#     bash submit_plot_sig_electrodes_dcc.sh

# Labels to plot (must exist in condition_plot_specs.PLOT_CONDITION_SETS).
PLOT_SETS=(
    lwpc_vs_lwps
    # congruency_vs_switchType
    # stim_vs_response
)

# Base directory holding your within-electrode ANOVA runs. Exported so
# condition_plot_specs.ANOVA_RUNS_BASE picks it up without editing the file.
export ANOVA_RUNS_BASE="/path/to/within_elec_anova_runs"   # <- EDIT ME

mkdir -p out

for LABEL in "${PLOT_SETS[@]}"; do
    echo "Submitting plot set: $LABEL"
    sbatch --job-name="plot_${LABEL}" \
        --export=ALL,PLOT_SET_LABEL="$LABEL",ANOVA_RUNS_BASE="$ANOVA_RUNS_BASE" \
        sbatch_plot_sig_electrodes_dcc.sh
done
