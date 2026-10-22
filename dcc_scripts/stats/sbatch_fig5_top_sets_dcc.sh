#!/bin/bash
#SBATCH --output=out/slurm_%j_%x.out
#SBATCH -e out/slurm_%j_%x.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Figure 5 with the top LWPC and LWPS electrodes picked out (fig5_top_sets.py),
# from a finished anatomy run. Arguments are passed through, e.g.
#   sbatch sbatch_fig5_top_sets_dcc.sh --anatomy-dir <run>/continuous --seg-dir <seg run> --brain
# xvfb-run gives the brain render a display, as in sbatch_stability_flexibility_anatomy_dcc.sh.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg

xvfb-run -a python /hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/fig5_top_sets.py "$@"
