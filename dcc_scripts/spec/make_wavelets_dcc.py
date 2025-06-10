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
from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets

def main(subject_id):
    """Main function to run wavelet analysis for a single subject."""

    # Set up paths
    HOME = os.path.expanduser("~")
    USER = os.path.basename(HOME)
    task = 'GlobalLocal'

    # get box directory depending on OS
    LAB_root = os.path.join("/cwork", USER)

    # Load Data using the first 'get_data' function
    layout = get_data(task, root=LAB_root)

    output_names_and_events_dict = {}
    output_names_and_events_dict['ErrorTrials'] = ["Accuracy/0.0"] 
    output_names_and_events_dict['CorrectTrials'] = ["Accuracy/1.0"]

    baseline_times = [-0.5, 0]
    signal_times = [-0.5, 1.5]

    # Use the second 'get_good_data' function, with the passed-in subject_id
    print(f"Loading good data for subject: {subject_id}")
    good = get_good_data(subject_id, layout)

    ## epoching and trial outlier removal
    save_dir = os.path.join(layout.root, 'derivatives', 'clean', subject_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use the 'subject_id' variable 
    base = get_uncorrected_wavelets(subject_id, layout, events=["Stimulus"], times=baseline_times)

    # make signal wavelets
    for output_name, events in output_names_and_events_dict.items():
        spec = get_uncorrected_wavelets(subject_id, layout, events, signal_times)
        spec_rescaled = rescale(spec, base, copy=True, mode='ratio').average(
            lambda x: np.nanmean(x, axis=0), copy=True)
        spec_rescaled._data = np.log10(spec_rescaled._data) * 20 # convert to dB
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec_rescaled.info['subject_info']['files'] = tuple(fnames)
        spec_rescaled.info['bads'] = good.info['bads']

        rescaled_filename = os.path.join(save_dir, f'{output_name}_rescaled-tfr.h5')
        uncorrected_filename = os.path.join(save_dir, f'{output_name}_uncorrected-tfr.h5')

        mne.time_frequency.write_tfrs(rescaled_filename, spec_rescaled, overwrite=True)
        mne.time_frequency.write_tfrs(uncorrected_filename, spec, overwrite=True)
        
        spec_rescaled.save(os.path.join(save_dir, f'{output_name}_rescaled-avg.fif'), overwrite=True)
        spec.save(os.path.join(save_dir, f'{output_name}_uncorrected-avg.fif'), overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make wavelets for a given subject.")
    parser.add_argument('--subject', type=str, required=True, 
                        help='The subject ID to processsqueue')
    args = parser.parse_args()
    main(args.subject)
