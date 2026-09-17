#!/bin/bash
#SBATCH --output=out/slurm_%j.out
#SBATCH -e out/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg  # needs scipy + statsmodels + mne (ROI atlas / brain figure)

# The brain figure renders electrodes on fsaverage through PyVista, which needs a
# display; xvfb-run supplies a virtual one on a headless compute node (same as
# dcc_scripts/vis/sbatch_plot_sig_electrodes_dcc.sh). Without it the surface
# render fails and the job falls back to the ROI histogram.
#
# Do NOT export PYVISTA_OFF_SCREEN here. With a DISPLAY from xvfb-run, off-screen
# mode leaves the Qt window unrealized, so its OpenGL context is never current
# and the screenshot dies with "RenderWindowUnavailable: Render window is not
# current" -- which is exactly why this job fell back to the by-ROI figure while
# the vis job (which sets the flag only when DISPLAY is missing) rendered fine.
# The renderer now picks the mode itself from DISPLAY.
xvfb-run -a python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py
