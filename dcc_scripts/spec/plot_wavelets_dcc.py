## plot_wavelets dcc version

import os
import sys

# Get the absolute path to the directory containing the current script
try:
    # This will work if running as a .py script
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # headless backend for the cluster
import matplotlib.pyplot as plt

from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, get_bad_chans
from src.analysis.spec.wavelet_functions import load_wavelets, load_multitaper
from src.analysis.config.condition_registry import get_conditions_obj

def main(args):

    # ------------------------------------------------------------------
    # 1. Resolve LAB_root
    # ------------------------------------------------------------------
    if args.LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)

        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage",
                                    "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                LAB_root = os.path.join(HOME, "CoganLab")
    else:
        LAB_root = args.LAB_root

    # ------------------------------------------------------------------
    # 2. Resolve conditions from the registry, driven by condition_label.
    #    These condition names match the filenames written by
    #    make_wavelets_dcc.py.
    # ------------------------------------------------------------------
    conditions = get_conditions_obj(args.condition_label)
    condition_names = list(conditions.keys())

    layout = get_data(args.task, root=LAB_root)

    # ------------------------------------------------------------------
    # 3. Pick the loader and plot settings for this spec type
    # ------------------------------------------------------------------
    if args.spec_type == 'wavelet':
        load_func = load_wavelets
        plot_kwargs = {}  # chan_grid default (log) y-scale
    elif args.spec_type == 'multitaper':
        load_func = load_multitaper
        plot_kwargs = {'yscale': 'linear'}
    else:
        raise ValueError(f"Unknown spec_type: {args.spec_type}")

    spec_dir = os.path.join(layout.root, 'derivatives', 'spec', args.spec_type)
    fig_path = os.path.join(spec_dir, 'figs')
    os.makedirs(fig_path, exist_ok=True)

    suffix = 'rescaled' if args.rescaled else 'uncorrected'

    # ------------------------------------------------------------------
    # 4. Load and plot each condition
    # ------------------------------------------------------------------
    for condition_name in condition_names:
        tfr_file = os.path.join(spec_dir, args.subject_id,
                                f'{condition_name}_{suffix}-tfr.h5')
        if not os.path.exists(tfr_file):
            print(f"File not found, skipping: {tfr_file}")
            continue

        spec = load_func(args.subject_id, layout, condition_name, args.rescaled)

        # Mark bad channels from the source data files. Only the rescaled
        # TFRs carry the source-file pointer in subject_info.
        subject_info = spec.info.get('subject_info') or {}
        files = subject_info.get('files')
        if files:
            info_file = os.path.join(layout.root, files[0])
            all_bad = get_bad_chans(info_file)
            spec.info.update(bads=[b for b in all_bad if b in spec.ch_names])
        else:
            print(f"No source file info in {condition_name} TFR; "
                  "skipping bad-channel update")

        figs = chan_grid(spec, size=(20, 10), vmin=-2, vmax=2,
                         cmap=parula_map, show=False, **plot_kwargs)

        for i, f in enumerate(figs):
            fig_name = (f'{args.subject_id}_{condition_name}_{suffix}_'
                        f'{args.spec_type}_{i + 1}.svg')
            f.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')
            print("Saved figure:", fig_name)

        plt.close('all')  # free figure memory before the next condition


if __name__ == "__main__":
    if len(sys.argv) == 1:
        pass
    else:
        print("This script should be called via run_plot_wavelets_dcc.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)