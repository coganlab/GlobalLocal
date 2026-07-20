#!/bin/bash
#SBATCH --output=out/slurm_%j.out
#SBATCH -e out/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

# Brain surface rendering needs a display. xvfb-run gives us a virtual one so
# PyVista can render off-screen on a headless compute node. plot_sig_electrodes_dcc.py
# also sets PYVISTA_OFF_SCREEN=true as a fallback.
xvfb-run -a python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/vis/run_plot_sig_electrodes_dcc.py
