# %% [markdown]
# 
# # Example of High Gamma Filter
# 
# Below is a code sample for extracting high gamma power from a raw data file, followed by permutation cluster stats on that high gamma power data
# 

# %% [markdown]
# ### working version 12/1/23

# %% [markdown]
# ### try gregs suggestion of using make_data_same to destroy the fixation cross

# %% [markdown]
# use window stats with perm testing (0 to 0.5, 0.5 to 1, 0 to 1 sec relative to stim onset)

# %%
import sys
import os
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import pandas as pd
import json
from statsmodels.stats.multitest import multipletests
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma # replace with naplib filterbank hilbert
from ieeg.calc.scaling import rescale
import mne
import numpy as np
#from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.fast import mean_diff
from ieeg.viz.mri import gen_labels
import matplotlib.pyplot as plt
from mne.utils import fill_doc, verbose
import random
from contextlib import redirect_stdout

print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...
import pickle
from scipy.stats import ttest_ind
from functools import partial
#from src.analysis.utils.general_utils import calculate_RTs, save_channels_to_file, save_sig_chans, load_sig_chans

from naplib.preprocessing import filterbank_hilbert as fb_hilb
from tqdm import tqdm

# Directory where your .npy files are saved
npy_directory = r'C:\Users\luoruoxi\Box\CoganLab\D_Data\GlobalLocal\accArrays'  # Replace with your directory path

# Dictionary to hold the data
acc_array = {}

# Iterate over each file in the directory
for file in os.listdir(npy_directory):
    if file.endswith('.npy'):
        # Construct the full file path
        file_path = os.path.join(npy_directory, file)
        # Load the numpy array from the file
        acc_array[file.split('_')[0]] = np.load(file_path)

# Now you have a dictionary where each key is the subject ID
# and the value is the numpy array of accuracies for that subject.

combined_data = pd.read_csv(r'C:\Users\luoruoxi\Box\CoganLab\D_Data\GlobalLocal\combinedData.csv')

# %% [markdown]
# define subjects

# %%

# %% [markdown]
# use time point cluster stats for determining stimulus significance (old method as of 2/13/24)
# 
# updated this one 2/29, once it's tested and works, then turn into a function and delete other cells below
# %%

def extract_amplitude_and_phase_and_freqs(data, fs=None,
            passband: tuple[int, int] = (70, 150), copy: bool = True,
            n_jobs=-1, verbose: bool = True):
    """
    Extract gamma band envelope, phase, and center frequencies from data.
    Supports both numpy arrays and MNE Epochs/Raw.
    """
    if hasattr(data, 'get_data'):
        if fs is None:
            fs = data.info['sfreq']
        in_data = data.get_data()
    else:
        if fs is None:
            raise ValueError("fs must be provided if data is not a Signal")
        in_data = data.copy() if copy else data

    passband = list(passband)
    env = np.zeros(in_data.shape)
    phase = np.zeros(in_data.shape)

    if in_data.ndim == 3:
        for idx in range(in_data.shape[0]):
            trial_data = in_data[idx]  # shape: (channels, times)
            x_phase, x_envelope, freqs = fb_hilb(trial_data.T, fs, passband, n_jobs)
            phase[idx] = np.sum(x_phase, axis=-1).T
            env[idx]   = np.sum(x_envelope, axis=-1).T
    elif in_data.ndim == 2:
        x_phase, x_envelope, freqs = fb_hilb(in_data.T, fs, passband, n_jobs)
        phase = np.sum(x_phase, axis=-1).T
        env   = np.sum(x_envelope, axis=-1).T
    else:
        raise ValueError(f"Unsupported data dimensions: {in_data.ndim}")

    return env, phase, freqs

def trial_ieeg_rand_offset(raw: mne.io.Raw, event: str | list[str, ...], within_times: tuple[float,float], times_length: float, pad_length: float,
               verbose=None, **kwargs) -> mne.Epochs:
    """Epochs data from a mne Raw iEEG instance.

    Takes a mne Raw instance and randomly epochs the data around a specified event, for each instance of the event,
    for a duration of times_length, within a range of within_times.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to epoch.
    event : str
        The event to epoch around.
    within_times : tuple[float, float]
        The time window within which to randomly select intervals for each event.
    times_length : float,
        The length of the time intervals to randomly select within `within_times`.
    pad_length : float,
        The length to pad each time interval. Will be removed later.
    %(picks_all)s
    %(reject_epochs)s
    %(flat)s
    %(decim)s
    %(epochs_reject_tmin_tmax)s
    %(detrend_epochs)s
    %(proj_epochs)s
    %(on_missing_epochs)s
    %(verbose)s

    Returns
    -------
    mne.Epochs
        The epoched data.
    """

    sfreq = raw.info['sfreq'] #raw.info in function

    # get padded within times and times_length
    within_times_padded = [within_times[0] - pad_length, within_times[1] + pad_length]
    times_length_padded = times_length + 2 * pad_length

    # Convert times to samples
    within_times_samples = [int(t * sfreq) for t in within_times_padded]
    times_length_samples = int((times_length_padded) * sfreq)

    # Shift the indices to be positive
    shift = abs(within_times_samples[0])
    within_times_samples_pos = [s + shift for s in within_times_samples]

    trials = trial_ieeg(raw, event, within_times_padded, preload=True, reject_by_annotation=False)

    rand_offset_data = []

    # Randomly select subsets for each trial
    for trial in trials.get_data():
        start_sample = random.randint(within_times_samples_pos[0], within_times_samples_pos[1] - times_length_samples)
        end_sample = start_sample + times_length_samples
        rand_offset_data.append(trial[:, start_sample:end_sample+1]) #across all channels, grab this time subset

    # Reassign data to rand_offset_trials and adjust the times in rand_offset_trials
    new_tmin = within_times_padded[0]
    new_tmax = new_tmin + times_length_padded
    rand_offset_trials = trial_ieeg(raw, event, [new_tmin, new_tmax], preload=True, reject_by_annotation=False)
    rand_offset_trials._data = np.array(rand_offset_data)

    return rand_offset_trials

# Define a function to shuffle an array
def shuffle_array(arr):
    arr = np.random.shuffle(arr)
    return arr

def epoch_and_save(sub, task='GlobalLocal', times=(-1, 1.5), within_base_times=(-1, 0), base_times_length=0.5, 
baseline_event="Stimulus", pad_length = 0.5, LAB_root=None, channels=None, dec_factor=8, outliers=10, passband=(70,150)):
    '''
    save epoched data. 
    '''

    if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                    "CoganLab")

    layout = get_data(task, root=LAB_root)
    print(f"Using raw_from_layout to load subject: {sub}")
    try:
        filt = raw_from_layout(
            layout.derivatives['derivatives/clean'],
            subject=sub,
            extension='.edf',
            desc='clean',
            preload=False
        )
        print(f"Successfully loaded subject {sub} using raw_from_layout")
    except Exception as e:
        print(f"Failed to load subject {sub} with raw_from_layout")
        print(f"Error: {e}")
        raise
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    good = crop_empty_data(filt)
    # %%

    print(f"good channels before dropping bads: {len(good.ch_names)}")
    print(f"filt channels before dropping bads: {len(filt.ch_names)}")

    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    print("Bad channels in 'good':", good.info['bads'])

    filt.drop_channels(good.info['bads'])  # this has to come first cuz if you drop from good first, then good.info['bads'] is just empty
    good.drop_channels(good.info['bads'])

    print("Bad channels in 'good' after dropping once:", good.info['bads'])

    print(f"good channels after dropping bads: {len(good.ch_names)}")
    print(f"filt channels after dropping bads: {len(filt.ch_names)}")

    good.load_data()

    # If channels is None, use all channels
    if channels is None:
        channels = good.ch_names
    else:
        # Validate the provided channels
        invalid_channels = [ch for ch in channels if ch not in good.ch_names]
        if invalid_channels:
            raise ValueError(
                f"The following channels are not valid: {invalid_channels}")

        # Use only the specified channels
        good.pick_channels(channels)

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)
    # within_times_duration = abs(within_base_times[1] - within_base_times[0]) #grab the duration as a string for naming
    pad_length_string = f"{pad_length}s" # define pad_length as a string so can use it as input to crop_pad

    for event in ["Stimulus", "Response"]:
        #output_name_event = f'{event}_{output_name_base}'
        trials_ev = trial_ieeg(
                good, event,
                [times[0] - pad_length, times[1] + pad_length],
                preload=True, reject_by_annotation=False
            )
        print("trial before crop_pad: ", trials_ev.tmin, trials_ev.tmax)
        crop_pad(trials_ev, pad_length_string) # crop the trials to remove the padding
        print("trial after crop_pad: ", trials_ev.tmin, trials_ev.tmax)

        # Save
        trials_ev.save(f'{save_dir}/{sub}_{event}_ev1-epo.fif', overwrite=True)
        print(f"Saved {event} trials for subject {sub} to {save_dir}/{sub}_{event}_ev1-epo.fif")


def main(subjects=None, task='GlobalLocal', times=(-1, 1.5),
         within_base_times=(-1, 0), base_times_length=0.5, pad_length=0.5, LAB_root=None, channels=None, dec_factor=8, outliers=10, passband=(70,150)):
    """
    Main function to bandpass filter and compute time permutation cluster stats and task-significant electrodes for chosen subjects.
    """
    if subjects is None:
        #subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D116', 'D117', 'D121']
        subjects = ['D0116']# use one subject at a time to avoid the permission error
    for sub in subjects:
        epoch_and_save(sub=sub, task=task, times=times,
                          within_base_times=within_base_times, base_times_length=base_times_length,
                          pad_length=pad_length, LAB_root=LAB_root, channels=channels,
                          dec_factor=dec_factor, outliers=outliers, passband=passband)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process subjects and plot bandpass-filtered data, compute time permutation cluster matrix of electrodes by time, and find task-significant electrodes.")
    parser.add_argument('--subjects', nargs='+', default=None, help='List of subjects to process. If not provided, all subjects will be processed.')
    parser.add_argument('--task', type=str, default='GlobalLocal', help='Task to process. Default is GlobalLocal.')
    parser.add_argument('--times', type=float, nargs=2, default=(-1, 1.5), help='Time window for event processing. Default is (-1, 1.5).')
    parser.add_argument('--within_base_times', type=float, nargs=2, default=(-1, 0), help='Time window for baseline processing. Default is (-1, 0).')
    parser.add_argument('--baseline_event', type=str, default='Stimulus', help='Event to use for baseline. Default is Stimulus.')
    parser.add_argument('--base_times_length', type=float, default=0.5, help='Length of the time intervals to randomly select within `within_base_times`. Default is 0.5.')
    parser.add_argument('--pad_length', type=float, default=0.5, help='Length to pad each time interval. Will be removed later. Default is 0.5.')
    parser.add_argument('--LAB_root', type=str, default=None, help='Root directory for the lab. Will be determined based on OS if not provided. Default is None.')
    parser.add_argument('--channels', type=str, default=None, help='Channels to plot and get stats for. Default is all channels.')
    parser.add_argument('--dec_factor', type=int, default=8, help='Decimation factor. Default is 8.')
    parser.add_argument('--outliers', type=int, default=10, help='How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.')
    parser.add_argument('--passband', type=float, nargs=2, default=(70,150), help='Frequency range for the frequency band of interest. Default is (70, 150).')
    args=parser.parse_args()

    print("--------- PARSED ARGUMENTS ---------")
    print(f"args.passband: {args.passband} (type: {type(args.passband)})")
    print(f"args.subjects: {args.subjects} (type: {type(args.subjects)})")

    main(subjects=args.subjects, 
        task=args.task, 
        times=args.times, 
        within_base_times=args.within_base_times, 
        base_times_length=args.base_times_length, 
        pad_length=args.pad_length, 
        LAB_root=args.LAB_root, 
        channels=args.channels, 
        dec_factor=args.dec_factor, 
        outliers=args.outliers, 
        passband=args.passband)