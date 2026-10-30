#!/bin/bash
#SBATCH --output=out/slurm_%j.out
#SBATCH -e out/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

# Brain surface rendering needs a display. xvfb-run gives us a virtual one so
# PyVista can render off-screen on a headless compute node. plot_sig_electrodes_dcc.py
# also sets PYVISTA_OFF_SCREEN=true as a fallback.
# -n: start the free-display search at a number set by the job ID. With the
# default (99), two jobs starting together on one node both took :99, and the
# first to finish killed the other's display (2026-10-30).
xvfb-run -a -n $((100 + SLURM_JOB_ID % 900)) python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/vis/run_plot_sig_electrodes_dcc.py
