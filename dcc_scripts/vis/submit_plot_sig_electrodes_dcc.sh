#!/bin/bash
# Submit the significant-electrode brain plotting job.
#
# Which conditions get plotted (and in what colors), the subjects, ROIs, and
# output directory are all configured at the top of run_plot_sig_electrodes_dcc.py.
# Edit that file, then run:
#
#     bash submit_plot_sig_electrodes_dcc.sh
#
# from the dcc_scripts/vis directory so the sbatch relative out/ paths resolve.

mkdir -p out

sbatch --job-name="plot_sig_elecs" sbatch_plot_sig_electrodes_dcc.sh
