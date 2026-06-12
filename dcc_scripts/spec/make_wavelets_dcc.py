## make_wavelets dcc version
#comment to commit

import os
import sys
import argparse

print(sys.path)

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
try:
    # This will work if running as a .py script
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    # os.getcwd() often gives the directory of the notebook,
    # or the directory from which the Jupyter server was started.
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale

from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from src.analysis.utils.general_utils import get_good_data
from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets, get_uncorrected_multitaper
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
    # 2. Resolve conditions / comparisons / subtraction_pairs / anova
    #    config from the registry, driven by condition_label.
    # ------------------------------------------------------------------
    condition_label = args.condition_label
    conditions = get_conditions_obj(condition_label)
    condition_names = list(conditions.keys())


    # Load Data using the first 'get_data' function
    layout = get_data(args.task, root=LAB_root)

    # Use the second 'get_good_data' function, with the passed-in args.subject_id
    print(f"Loading good data for subject: {args.subject_id}")
    good = get_good_data(args.subject_id, layout)

    baseline_times = args.baseline_times
    signal_times = args.signal_times

    if args.spec_type == 'wavelet':

        ## epoching and trial outlier removal
        save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', args.subject_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Use the 'args.subject_id' variable 
        base = get_uncorrected_wavelets(args.subject_id, layout, events=["Stimulus"], times=baseline_times)

        # make signal wavelets
        for condition_name in condition_names:
            events = conditions.get(condition_name).get('BIDS_events')
            spec = get_uncorrected_wavelets(args.subject_id, layout, events, signal_times)

            spec_rescaled = rescale(spec, base, copy=True, mode='ratio').average(
                lambda x: np.nanmean(x, axis=0), copy=True)
            spec_rescaled._data = np.log10(spec_rescaled._data) * 20 # convert to dB
            fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
            spec_rescaled.info['temp'] = {'files': tuple(fnames)}
            spec_rescaled.info['bads'] = good.info['bads']

            rescaled_filename = os.path.join(save_dir, f'{condition_name}_rescaled-tfr.h5')
            uncorrected_filename = os.path.join(save_dir, f'{condition_name}_uncorrected-tfr.h5')

            mne.time_frequency.write_tfrs(rescaled_filename, spec_rescaled, overwrite=True)
            mne.time_frequency.write_tfrs(uncorrected_filename, spec, overwrite=True)


    elif args.spec_type == 'multitaper':

        # Define multitaper parameters
        freqs = args.freqs
        n_cycles = args.n_cycles 
        time_bandwidth = args.time_bandwidth 
        return_itc = args.return_itc

        # Set the save directory for multitaper results
        save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'multitaper', args.subject_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        base = get_uncorrected_multitaper(
            args.subject_id, layout, events=["Stimulus"], times=baseline_times,
            freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, return_itc=return_itc, average=False
        )

        # Generate and save multitaper spectrogram for each condition
        for condition_name in condition_names:
            events = conditions.get(condition_name).get('BIDS_events')

            # Call the multitaper function
            spec = get_uncorrected_multitaper(
                args.subject_id, layout, events, signal_times, 
                freqs=freqs, n_cycles=n_cycles, 
                time_bandwidth=time_bandwidth, return_itc=return_itc, average=False
            )
            spec_rescaled = rescale(spec, base, copy=True, mode='ratio').average(
                lambda x: np.nanmean(x, axis=0), copy=True)
            spec_rescaled._data = np.log10(spec_rescaled._data) * 20 # convert to dB
            fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
            spec_rescaled.info['temp'] = {'files': tuple(fnames)}
            spec_rescaled.info['bads'] = good.info['bads']

            rescaled_filename = os.path.join(save_dir, f'{condition_name}_rescaled-tfr.h5')
            uncorrected_filename = os.path.join(save_dir, f'{condition_name}_uncorrected-tfr.h5')

            spec_rescaled.save(rescaled_filename, overwrite=True)
            spec.save(uncorrected_filename, overwrite=True)
            print(f"Saved multitaper spectrogram to: {rescaled_filename}")
            print(f"Saved multitaper spectrogram to: {uncorrected_filename}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        pass
    else:
        print("This script should be called via run_make_wavelets_dcc.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)
