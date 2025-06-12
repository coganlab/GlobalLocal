import mne.time_frequency
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, update, get_bad_chans
import os
import matplotlib.pyplot as plt
from wavelet_functions import load_wavelets

def main(subject_id):
    # Set up paths
    HOME = os.path.expanduser("~")
    USER = os.path.basename(HOME)
    task = 'GlobalLocal'

    # get box directory depending on OS
    LAB_root = os.path.join("/cwork", USER)
    
    # Load Data
    layout = get_data(task, root=LAB_root)

    rescaled=True

    # define output_names that you want to plot wavelets for
    output_names = ['ErrorTrials', 'CorrectTrials']

    layout = get_data("GlobalLocal", root=LAB_root)

    fig_path = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', 'figs')
    for output_name in output_names:
        spec = load_wavelets(subject_id, layout, output_name, rescaled)
            
        info_file = os.path.join(layout.root, spec.info['subject_info']['files'][0])

        # Check channels for outliers and remove them
        all_bad = get_bad_chans(info_file)
        spec.info.update(bads=[b for b in all_bad if b in spec.ch_names])

        # Plotting
        figs = chan_grid(spec, size=(20, 10), vmin=-2, vmax=2, cmap=parula_map, show=False)

        for i, f in enumerate(figs):
            if rescaled:
                fig_name = f'{subject_id}_{output_name}_rescaled_{i+1}.jpg'
            else:
                fig_name = f'{subject_id}_{output_name}_uncorrected_{i+1}.jpg'

            fig_pathname = os.path.join(fig_path, fig_name)
            f.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot wavelets for a given subject.")
    parser.add_argument('--subject', type=str, required=True, 
                        help='The subject ID to processs')
    args = parser.parse_args()
    main(args.subject)