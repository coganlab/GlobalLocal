## make_wavelets dcc version
#comment to commit

import sys
import os
import numpy as np
import pandas as pd
print(sys.path)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from utils import get_good_data
from wavelet_functions import get_uncorrected_wavelets

# Set up paths
HOME = os.path.expanduser("~")
task = 'GlobalLocal'

# get box directory depending on OS
LAB_root = os.path.join(HOME, "coganlab", "Data")
    
# Load Data
layout = get_data(task, root=LAB_root)

# let's make a dictionary where the keys are output names and the values are lists of corresponding events
output_names_and_events_dict = {}
output_names_and_events_dict['ErrorTrials'] = ["Accuracy/0.0"]
output_names_and_events_dict['CorrectTrials'] = ["Accuracy/1.0"]

baseline_times = [-0.5, 0]
signal_times = [-0.5, 1.5]

subjects = ["D0116"]

for sub in subjects:
    # load in good data so we can use it for gettin bad channels and getting filenames
    good = get_good_data(sub, layout)

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'clean', sub)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # make stimulus baseline EpochsTFR
    base = get_uncorrected_wavelets(sub, layout, events=["Stimulus"], times=baseline_times)

    # make signal wavelets
    for output_name, events in output_names_and_events_dict.items():
        spec = get_uncorrected_wavelets(sub, layout, events, signal_times)
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