import mne.time_frequency
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from ieeg.io import get_data, update, get_bad_chans
import os
import matplotlib.pyplot as plt
from wavelet_functions import load_wavelets

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

fig_path = os.path.join(layout.root, 'derivatives', 'clean', subject)

for output_name in output_names:
    spec = load_wavelets(subject, layout, output_name, rescaled)
            
    info_file = os.path.join(layout.root, spec.info['subject_info']['files'][0])

    # Check channels for outliers and remove them
    all_bad = get_bad_chans(info_file)
    spec.info.update(bads=[b for b in all_bad if b in spec.ch_names])

    # Plotting
    figs = chan_grid(spec, size=(20, 10), vmin=-2, vmax=2, cmap=parula_map, show=False)

    for i, f in enumerate(figs):
        if rescaled:
            fig_name = f'{subject}_{output_name}_rescaled_{i+1}.jpg'
        else:
            fig_name = f'{subject}_{output_name}_uncorrected_{i+1}.jpg'

        fig_pathname = os.path.join(fig_path, fig_name)
        f.savefig(fig_pathname, bbox_inches='tight')
        print("Saved figure:", fig_name)


# Description: Check channels for outliers and remove them


fig_path = os.path.join(layout.root, 'derivatives', 'clean', subject)
for output_name in output_names:
    filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', subject, f'{output_name}-tfr.h5')
    print("Filename:", filename)
    # spec = mne.time_frequency.read_tfrs(filename)[0]

    # try doing this block to get spec instead of the above line where we grab it as the 0th index of read_tfrs
    tfr_result = mne.time_frequency.read_tfrs(filename)
    if isinstance(tfr_result, list):
        # If it's a list, pick the first TFR object. UHHH I THINK I SHOULD AVERAGE THIS ACROSS TRIALS 2/28
        spec = tfr_result[0]
    else:
        # Otherwise it's already just a single AverageTFR object
        spec = tfr_result
            
    info_file = os.path.join(layout.root, spec.info['subject_info']['files'][0])

    # Check channels for outliers and remove them
    all_bad = get_bad_chans(info_file)
    spec.info.update(bads=[b for b in all_bad if b in spec.ch_names])

    # Plotting
    figs = chan_grid(spec, size=(20, 10), vmin=-2, vmax=2, cmap=parula_map, show=False)
    for i, f in enumerate(figs):
        fig_name = f'{subject}_{output_name}_{i+1}.jpg'
        fig_pathname = os.path.join(fig_path, fig_name)
        f.savefig(fig_pathname, bbox_inches='tight')
        print("Saved figure:", fig_name)