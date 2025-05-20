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
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

import pandas as pd
import json
from statsmodels.stats.multitest import multipletests
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import os
import numpy as np
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster
from ieeg.viz.mri import gen_labels
from utils import calculate_RTs, save_channels_to_file, save_sig_chans, load_sig_chans, plot_HG_and_stats
import matplotlib.pyplot as plt
from mne.utils import fill_doc, verbose
import random
from contextlib import redirect_stdout
import sys
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...
import pickle

# Directory where your .npy files are saved
npy_directory = r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\accArrays'  # Replace with your directory path

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
        
combined_data = pd.read_csv(r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\combinedData.csv')

# %% [markdown]
# define subjects

# %%
# subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110']
# subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103']
subjects = ['D0057']

# %% [markdown]
# use time point cluster stats for determining stimulus significance (old method as of 2/13/24)
# 
# updated this one 2/29, once it's tested and works, then turn into a function and delete other cells below
# %%

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

def plot_HG_and_stats(sub, task='GlobalLocal', times=(-1, 1.5),
                      within_base_times=(-1, 0), base_times_length=0.5, pad_length = 0.5, LAB_root=None, channels=None, dec_factor=10, outliers=10, passband=(70,150)):
    """
    Plot high gamma (HG) and statistics for a given subject and task using specified event.

    Parameters:
    - sub (str): The subject identifier.
    - task (str): The task identifier.
    - times (tuple, optional): A tuple indicating the start and end times for event processing. Defaults to (-1, 1.5).
    - times (tuple [float, float]): The time window to epoch around the event.
    - within_base_times (tuple [float, float]): The time window within which to randomly select intervals for each event, for baseline.
    - base_times_length (float): The length of the time intervals to randomly select within `within_base_times`. 
    - pad_length (float): The length to pad each time interval. Will be removed later.
    - LAB_root (str, optional): The root directory for the lab. Will be determined based on OS if not provided. Defaults to None.
    - channels (list of strings, optional): The channels to plot and get stats for. Default is all channels.
    - decimation_factor (int, optional): The factor by which to subsample the data. Default is 10, so should be 2048 Hz down to 204.8 Hz.
    - outliers (int, optional): How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.
    - passband (tuple, optional): The frequency range for the frequency band of interest. Default is (70, 150).

    This function will process the provided event for a given subject and task.
    High gamma (HG) will be computed, and statistics will be calculated and plotted.
    The results will be saved to output files.
    """

    if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                    "CoganLab")

    layout = get_data(task, root=LAB_root)
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False)
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
    within_times_duration = abs(within_base_times[1] - within_base_times[0]) #grab the duration as a string for naming

    # Create a baseline EpochsTFR using the stimulus event. For each trial, will randomly grab a segment of duration base_times_length from the within_base_times range. This offsets the fix cross. 6/15.
    trials = trial_ieeg_rand_offset(good, "Stimulus", within_base_times, base_times_length, pad_length, preload=True)
    outliers_to_nan(trials, outliers=outliers)
    HG_base = gamma.extract(trials, passband=passband, copy=False, n_jobs=1)
    pad_length_string = f"{pad_length}s" # define pad_length as a string so can use it as input to crop_pad
    crop_pad(HG_base, pad_length_string) # need to change this if pad length changes
    HG_base.decimate(dec_factor)
    
    # Square the data to get power from amplitude
    HG_base_power = HG_base.copy()
    HG_base_power._data = HG_base._data ** 2  # Square amplitude to get power

    output_name_base = f"{base_times_length}sec_within{within_times_duration}sec_randoffset_preStimulusBase_decFactor_{dec_factor}_outliers_{outliers}_passband_{passband[0]}-{passband[1]}_padLength_{pad_length}s"

    for event in ["Stimulus", "Response"]:
        output_name_event = f'{event}_{output_name_base}'
        times_adj = [times[0] - pad_length, times[1] + pad_length]
        trials = trial_ieeg(good, event, times_adj, preload=True,
                            reject_by_annotation=False)

        outliers_to_nan(trials, outliers=outliers)
        HG_ev1 = gamma.extract(trials, passband=passband, copy=True, n_jobs=1)
        print("HG_ev1 before crop_pad: ", HG_ev1.tmin, HG_ev1.tmax)
        crop_pad(HG_ev1, pad_length_string) #change this if pad length changes
        print("HG_ev1 after crop_pad: ", HG_ev1.tmin, HG_ev1.tmax)

        HG_ev1.decimate(dec_factor)

        # Square the data to get power from amplitude
        HG_ev1_power = HG_ev1.copy()
        HG_ev1_power._data = HG_ev1._data ** 2 # Square amplitude to get power

        # get the rescaled amplitude
        HG_ev1_rescaled = rescale(HG_ev1, HG_base, copy=True, mode='zscore')

        # get the rescaled power
        HG_ev1_power_rescaled = rescale(HG_ev1_power, HG_base_power, copy=True, mode='zscore')

        # get the evoke and evoke rescaled amplitude
        HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0)) #axis=0 should be set for actually running this, the axis=2 is just for drift testing.
        HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

        # get the evoke and evoke power rescaled amplitude
        HG_ev1_evoke_power = HG_ev1_power.average(method=lambda x: np.nanmean(x, axis=0)) #axis=0 should be set for actually running this, the axis=2 is just for drift testing.
        HG_ev1_evoke_power_rescaled = HG_ev1_power_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

        # Save HG_ev1
        HG_ev1.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1-epo.fif', overwrite=True)
        HG_ev1_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power-epo.fif', overwrite=True)

        # Save HG_base (the shuffled version)
        HG_base.save(f'{save_dir}/{sub}_{output_name_event}_HG_base-epo.fif', overwrite=True)
        HG_base_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_base_power-epo.fif', overwrite=True)

        # Save HG_ev1_rescaled
        HG_ev1_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_rescaled-epo.fif', overwrite=True)
        HG_ev1_power_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power_rescaled-epo.fif', overwrite=True)

        # Save HG_ev1_evoke
        HG_ev1_evoke.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke-epo.fif', overwrite=True)
        HG_ev1_evoke_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power-epo.fif', overwrite=True)
        
        # Save HG_ev1_evoke_rescaled
        HG_ev1_evoke_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_rescaled-epo.fif', overwrite=True)
        HG_ev1_evoke_power_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power_rescaled-epo.fif', overwrite=True)

        ###
        print(f"Shape of HG_ev1._data: {HG_ev1._data.shape}")
        print(f"Shape of HG_base._data: {HG_base._data.shape}")
        
        # oh this changed and returns both the significant clusters matrix and the p values now
        mat = time_perm_cluster(HG_ev1._data, HG_base._data, 0.05, n_jobs=6, ignore_adjacency=1)[0]

        #save channels with their indices 
        save_channels_to_file(channels, sub, task, save_dir)

        # save significant channels to a json
        save_sig_chans(f'{output_name_event}', mat, channels, sub, save_dir)
        
        # Assuming `mat` is your array and `save_dir` is the directory where you want to save it
        mat_save_path = os.path.join(save_dir, f'{output_name_event}_mat.npy')

        # Save the mat array
        np.save(mat_save_path, mat)

        # Plot the matrix as one figure. This broken for some reason 9/28.
        # fig, ax = plt.subplots()
        # cax = ax.imshow(mat, aspect='auto', cmap='viridis')
        # ax.set_title(f'Statistical Matrix for {sub}_{output_name_event}')
        # ax.set_xlabel('Samples (204.8 Hz after decimation, 0 is -1 sec)')
        # ax.set_ylabel('Channels')
        # plt.show()

        # # Save the figure
        # plot_save_path = os.path.join(save_dir, f'{output_name_event}_stats.png')
        # fig.savefig(plot_save_path, dpi=300)




# %% [markdown]
# 5/2 do this for each subject, for stimulus and response

# %% [markdown]
# 8/5 run this after tobias greg meeting

# %%
times = (-1,1.5)
within_base_times = (-1,0)
base_times_length = 0.5
pad_length = 0.5

for sub in subjects:
    plot_HG_and_stats(sub=sub, task='GlobalLocal', times=times,
                      within_base_times=within_base_times, base_times_length = 0.5, pad_length = 0.5, dec_factor=8, outliers=10, passband=(70,150))

































# %% [markdown]
# plot the time perm cluster results outside of the function cuz it seems broken right now 6/6

# %%
LAB_root = None

task='GlobalLocal'
output_name_event = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10"
if LAB_root is None:
    HOME = os.path.expanduser("~")
    if os.name == 'nt':  # windows
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    else:  # mac
        LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                "CoganLab")

layout = get_data(task, root=LAB_root)

for sub in subjects:
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False)
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)
    # Load the matrix
    mat_save_path = os.path.join(save_dir, f'{output_name_event}_mat.npy')
    mat = np.load(mat_save_path)
    # Plot the matrix as one figure
    fig, ax = plt.subplots()
    cax = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_title(f'Statistical Matrix for {sub}_{output_name_event}')
    ax.set_xlabel('Samples (204.8 Hz after decimation, 0 is -1 sec)')
    ax.set_ylabel('Channels')
    # fig.colorbar(cax)
    # plt.show()

    # Save the figure
    plot_save_path = os.path.join(save_dir, f'{output_name_event}_stats.png')
    fig.savefig(plot_save_path, dpi=300)

# %%


# Set global font size
plt.rcParams.update({'font.size': 16})

# Function to plot the presence of NaNs across trials and channels
def plot_nan_matrix(epochs, title, save_path):
    data = epochs.get_data()
    nan_matrix = np.isnan(data).any(axis=2).astype(int)  # Mark NaNs as 0, valid as 1
    nan_matrix = 1 - nan_matrix  # Invert to have 1 for valid and 0 for NaNs
    nan_matrix = nan_matrix.T  # Transpose to have channels as rows and trials as columns

    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(nan_matrix, aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Channels')
    fig.colorbar(cax, ax=ax, orientation='vertical', label='Data Presence (1: Valid, 0: NaN)')
    
    plt.savefig(save_path)
    plt.close()

# Function to plot the number of trial outliers per channel
def plot_trial_outlier_count_per_channel(epochs, title, save_path):
    data = epochs.get_data()
    trial_outlier_count = np.isnan(data).any(axis=2).sum(axis=0)  # Count trial outliers per channel
    channel_names = epochs.ch_names

    # Create a dictionary to store the number of trial outliers for each channel
    trial_outlier_count_dict = dict(zip(channel_names, trial_outlier_count))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(np.arange(len(channel_names)), trial_outlier_count)
    ax.set_title(title)
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Number of Trial Outliers')
    # ax.set_xticks(np.arange(len(channel_names)))
    # ax.set_xticklabels(np.arange(len(channel_names)), rotation=90)
    
    plt.savefig(save_path)
    plt.close()

    return trial_outlier_count_dict

def process_and_plot_for_subject(subject, root_dir, outlier_thresholds, sfreq=256):
    outlier_counts = {}

    for threshold in outlier_thresholds:
        if threshold == 10:
            filename = rf"{subject}_Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_HG_ev1_rescaled-epo.fif"
        else:
            filename = rf"{subject}_Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_{threshold}_HG_ev1_rescaled-epo.fif"
        
        filepath = os.path.join(root_dir, filename)
        epochs = mne.read_epochs(filepath)

        # Extract data and compute the means
        data = epochs.get_data()
        print(f'This many trials left for {subject} after {threshold} stdev outlier threshold: {data.shape[0]}')

        avg_data = np.nanmean(data, axis=(0, 1))

        # Create time axes for each signal based on their respective sampling rates
        time = np.arange(avg_data.shape[0]) / sfreq
        time = time - 1

        # Plot the averaged data
        plt.figure(figsize=(10, 5))
        plt.plot(time, avg_data, label=f'HG rescaled with Rand Offset Baseline, {threshold} stdev trial outlier thresh')
        plt.xlabel('Time from stim onset (s)')
        plt.ylabel('Z-Score')
        plt.title(f'{subject} high gamma signal rescaled')
        plt.legend(fontsize=14)

        # Save the plot
        plot_filename = f'{subject}_{threshold}stdev_trial_outlier_HG_ev1_rescaled_comparison.png'
        plot_filepath = os.path.join(root_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()  # Close the figure to free up memory

        # Plot NaN matrix
        nan_matrix_title = f'{subject} Trial Outlier Matrix ({threshold} stdev outlier threshold)'
        nan_matrix_filepath = os.path.join(root_dir, f'{subject}_{threshold}stdev_trial_outlier_matrix.png')
        plot_nan_matrix(epochs, nan_matrix_title, nan_matrix_filepath)

        # Plot trial outlier count per channel and get the trial outlier count dictionary
        trial_outlier_count_title = f'{subject} Trial Outlier Count Per Channel ({threshold} stdev outlier threshold)'
        trial_outlier_count_filepath = os.path.join(root_dir, f'{subject}_{threshold}stdev_trial_outlier_count_per_channel.png')
        trial_outlier_count_dict = plot_trial_outlier_count_per_channel(epochs, trial_outlier_count_title, trial_outlier_count_filepath)

        # Store the trial outlier count dictionary for this subject and threshold
        outlier_counts[f'{threshold}_stdev'] = trial_outlier_count_dict

    # Save the trial outlier count dictionary to a pickle file
    pickle_filepath = os.path.join(root_dir, f'{subject}_trial_outlier_counts.pkl')
    with open(pickle_filepath, 'wb') as pickle_file:
        pickle.dump(outlier_counts, pickle_file)

    return outlier_counts

outlier_thresholds = [2, 8, 10]

# Load your data and process each subject
for sub in subjects:
    root_dir = rf"C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\freqFilt\figs\{sub}"
    outlier_counts = process_and_plot_for_subject(sub, root_dir, outlier_thresholds)
    
    # Print the outlier counts for each subject
    print(f"Outlier counts for subject {sub}:")
    for threshold, counts in outlier_counts.items():
        print(f"{threshold}: {counts}")


# %% [markdown]
# read in the trial outlier counts per channel

# %%
import pickle

# Function to read and print the trial outlier counts from a pickle file
def read_trial_outlier_counts(subject, root_dir):
    pickle_filepath = os.path.join(root_dir, f'{subject}_trial_outlier_counts.pkl')
    with open(pickle_filepath, 'rb') as pickle_file:
        outlier_counts = pickle.load(pickle_file)
    return outlier_counts

# Specify the subject you want to read the pickle file for
subject = 'D0057'
root_dir = rf"C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\freqFilt\figs\{subject}"

# Read the trial outlier counts
outlier_counts = read_trial_outlier_counts(subject, root_dir)

# Print the outlier counts for the specified subject
print(f"Outlier counts for subject {subject}:")
for threshold, counts in outlier_counts.items():
    print(f"{threshold}: {counts}")


# %% [markdown]
# time shuffled vs non shuffled baseline HG base plotting

# %%
import numpy as np
import matplotlib.pyplot as plt
import mne

# Set global font size
plt.rcParams.update({'font.size': 16})

# Load your data
for sub in subjects:
    root_dir = rf"C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\freqFilt\figs\{sub}"
    randoffset_base_filename = rf"{sub}_Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_HG_base-epo.fif"
    randoffset_base_filepath = os.path.join(root_dir, randoffset_base_filename)
    randoffset_base = mne.read_epochs(randoffset_base_filepath)
    old_base_filename = rf"{sub}_Stimulus_1sec_preStimulusBase_decFactor_10_HG_base-epo.fif"
    old_base_filepath = os.path.join(root_dir, old_base_filename)
    old_base = mne.read_epochs(old_base_filepath)

    # Extract data and compute the means
    randoffset_base_data = randoffset_base.get_data()
    old_base_data = old_base.get_data()

    shuffled_avg = np.nanmean(randoffset_base_data, axis=(0, 1))
    unshuffled_avg = np.nanmean(old_base_data, axis=(0, 1))

    # Sampling rates
    sfreq_shuffled = 256  # Hz for shuffled baseline
    sfreq_unshuffled = 204.8  # Hz for unshuffled baseline

    # Create time axes for each signal based on their respective sampling rates
    time_shuffled = np.arange(shuffled_avg.shape[0]) / sfreq_shuffled
    time_unshuffled = np.arange(unshuffled_avg.shape[0]) / sfreq_unshuffled

    # Find the maximum time to ensure both signals can be plotted over the same time scale
    max_time = max(time_shuffled[-1], time_unshuffled[-1])

    # Pad the shorter time axis with NaNs
    if time_shuffled[-1] < max_time:
        extra_time = np.arange(time_shuffled[-1], max_time, 1/sfreq_shuffled)
        time_shuffled = np.concatenate((time_shuffled, extra_time))
        shuffled_avg = np.pad(shuffled_avg, (0, len(extra_time)), mode='constant', constant_values=np.nan)
    else:
        extra_time = np.arange(time_unshuffled[-1], max_time, 1/sfreq_unshuffled)
        time_unshuffled = np.concatenate((time_unshuffled, extra_time))
        unshuffled_avg = np.pad(unshuffled_avg, (0, len(extra_time)), mode='constant', constant_values=np.nan)

    # Plot the averaged data
    time_shuffled = time_shuffled - 1
    time_unshuffled = time_unshuffled - 1 # adjust for baseline starting 1 sec before stim onset
    plt.figure(figsize=(10, 5))
    plt.plot(time_shuffled, shuffled_avg, label='Trial+Channel Avg Rand Offset Baseline (0.5 sec within [-1,0])')
    plt.plot(time_unshuffled, unshuffled_avg, label='Trial+Channel Avg Unshuffled Baseline')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'{sub} high gamma baseline')
    plt.legend(fontsize=14)

    # Save the plot
    plot_filename = f'{sub}_randoffset_unshuffled_base_HG_base_comparison.png'
    plot_filepath = os.path.join(root_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()  # Close the figure to free up memory

# %% [markdown]
# plot evoked old baseline and new baseline hg ev1 rescaled

# %%
import numpy as np
import matplotlib.pyplot as plt
import mne
import os

# Set global font size
plt.rcParams.update({'font.size': 16})

# Load your data
for sub in subjects:
    root_dir = rf"C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\freqFilt\figs\{sub}"
    randoffset_base_filename = rf"{sub}_Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_HG_ev1_rescaled-epo.fif"
    randoffset_base_filepath = os.path.join(root_dir, randoffset_base_filename)
    randoffset_base = mne.read_epochs(randoffset_base_filepath)
    old_base_filename = rf"{sub}_Stimulus_1sec_preStimulusBase_decFactor_10_HG_ev1_rescaled-epo.fif"
    old_base_filepath = os.path.join(root_dir, old_base_filename)
    old_base = mne.read_epochs(old_base_filepath)

    # Extract data and compute the nan-mean
    randoffset_base_data = randoffset_base.get_data()
    old_base_data = old_base.get_data()

    randoffset_base_data = np.nanmean(randoffset_base_data, axis=0)
    old_base_data = np.nanmean(old_base_data, axis=0)

    # Create Evoked objects
    randoffset_evoked = mne.EvokedArray(randoffset_base_data, randoffset_base.info, tmin=randoffset_base.times[0])
    old_base_evoked = mne.EvokedArray(old_base_data, old_base.info, tmin=old_base.times[0])

    # Plot the evoked data for randoffset_evoked
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    randoffset_evoked.plot(axes=ax1, show=False, time_unit='s', spatial_colors=True)
    ax1.set_xlabel('Time from stim onset (s)')
    ax1.set_ylabel('Z-score')
    ax1.set_title(f'{sub} High Gamma Signal Rescaled Evoked with Rand Offset Baseline')
    ax1.legend(['Trial Avg HG rescaled evoked with Rand Offset Baseline'], fontsize=14)
    plot_filename1 = f'{sub}_randoffset_HG_ev1_rescaled_evoked.png'
    plot_filepath1 = os.path.join(root_dir, plot_filename1)
    fig1.savefig(plot_filepath1)
    plt.close(fig1)  # Close the figure to free up memory

    # Plot the evoked data for old_base_evoked
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    old_base_evoked.plot(axes=ax2, show=False, time_unit='s', spatial_colors=True)
    ax2.set_xlabel('Time from stim onset (s)')
    ax2.set_ylabel('Z-score')
    ax2.set_title(f'{sub} High Gamma Signal Rescaled Evoked with Unshuffled Baseline')
    ax2.legend(['Trial Avg HG rescaled evoked with Unshuffled Baseline'], fontsize=14)
    plot_filename2 = f'{sub}_unshuffled_HG_ev1_rescaled_evoked.png'
    plot_filepath2 = os.path.join(root_dir, plot_filename2)
    fig2.savefig(plot_filepath2)
    plt.close(fig2)  # Close the figure to free up memory


# %% [markdown]
# turn this into a loop over all three time windows and all 12 subjects

# %%
from PIL import Image, ImageChops

def trim_whitespace(image):
    """
    Trims the whitespace from an image.
    """
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    return image  # If no change

def plot_channels_on_grid_windows(evoke_data, std_err_data, channels_subset, time_windows, sig_chans, sample_rate, plot_x_dim=6, plot_y_dim=6):
    """
    Plots evoked EEG/MEG data for a subset of channels on a grid, overlaying significance markers for specified time windows.

    Parameters:
    - evoke_data: mne.Evoked object
        The evoked data to be plotted. This object contains the averaged EEG/MEG data over epochs.
    - std_err_data: 
        The standard error of the evoked data to be plotted
    - channels_subset: list of str
        A list of channel names to be plotted. Each channel name must correspond to a channel in `evoke_data`.
    - time_windows: dict
        A dictionary where keys are strings representing the names of the time windows of interest, and values are tuples
        indicating the start and end indices (in samples) of these windows.
    - sig_chans: dict
        A dictionary where keys are the names of the time windows (matching those in `time_windows`) and values are lists
        of channel names (str) that are significant within those windows.
    - sample_rate: float
        The sampling rate of the data, in Hz. Used to convert sample indices in `time_windows` to time in seconds.
    - plot_x_dim: int, optional (default=6)
        The number of columns in the grid layout for plotting the channels.
    - plot_y_dim: int, optional (default=6)
        The number of rows in the grid layout for plotting the channels.

    Returns:
    - fig: matplotlib.figure.Figure object
        The figure object containing the grid of plots. Each plot shows the evoked data for a channel, with significance
        markers overlaid for the specified time windows.
    """
    fig, axes = plt.subplots(plot_y_dim, plot_x_dim, figsize=(20, 12))  # Adjusted to match your desired layout
    fig.suptitle("Channels with Significance Overlay for Different Time Windows")
    axes_flat = axes.flatten()

    # Define colors for each time window
    colors = ['red', 'green', 'blue']
    window_names = list(time_windows.keys())

    for channel, ax in zip(channels_subset, axes_flat):
        stderr = stderr_data.data[channel_to_index[channel], :]
        # Plot the channel data with times in seconds
        ax.plot(evoke_data.times, evoke_data.data[channel_to_index[channel], :])
         # Add the standard error shading
        ax.fill_between(evoke_data.times, evoke_data.data[channel_to_index[channel], :] - stderr, evoke_data.data[channel_to_index[channel], :] + stderr, alpha=0.2)

        max_y_value = np.max(evoke_data.data[channel_to_index[channel], :])  # Find max y-value for significance lines
        # Assuming the epochs start 1 second before the stimulus/event
        epoch_start_time = -1  # Start time of epochs in seconds

        for window_index, window_name in enumerate(window_names):
            if channel in sig_chans[window_name]:
                start_idx, end_idx = time_windows[window_name]
                # Convert sample indices to times in seconds
                start_time = (start_idx / sample_rate) + epoch_start_time
                end_time = (end_idx / sample_rate) + epoch_start_time
                # Determine y-position for the significance line, adjusting to avoid overlap
                y_position = max_y_value - (window_index * 0.02 * max_y_value)  # Adjust overlap offset here

                # Cycle through colors for each time window
                color = colors[window_index % len(colors)]
                ax.hlines(y=y_position, xmin=start_time, xmax=end_time, color=color, linewidth=2, label=f"{window_name}: {color}")

        ax.set_title(channel)

    # Create a legend for the first subplot (if desired) to explain the colors
    if len(axes_flat) > 0 and len(window_names) > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title="Time Windows & Colors")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the legend
    return fig

sig_chans = {}

for sub in subjects:
    task = 'GlobalLocal'
    LAB_root = None
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                    "CoganLab")

    layout = get_data(task, root=LAB_root)
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False)
    sample_rate = filt.info['sfreq'] # get sampling rate, should be 2048 Hz
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_windows = {
        "Stimulus_fixationCrossBase_0.2sec_window_0to0.5": (sample_rate,1.5*sample_rate), #actually grab from 1 to 1.5 because the epochs start at -1 second before stim onset
        "Stimulus_fixationCrossBase_0.2sec_window_0.5to1": (1.5*sample_rate,2*sample_rate),
        "Stimulus_fixationCrossBase_0.2sec_window_0to1": (sample_rate,2*sample_rate)
    }

    for window in time_windows:
        output_name = window

        # Define file paths
        HG_ev1_file = f'{save_dir}/{sub}_{output_name}_HG_ev1-epo.fif'
        HG_base_file = f'{save_dir}/{sub}_{output_name}_HG_base-epo.fif'
        HG_ev1_rescaled_file = f'{save_dir}/{sub}_{output_name}_HG_ev1_rescaled-epo.fif'

        # Load the epochs and evoked objects
        HG_ev1 = mne.read_epochs(HG_ev1_file)
        HG_base = mne.read_epochs(HG_base_file)
        HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
        HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0))
        HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))
        HG_ev1_evoke_stderr = HG_ev1.standard_error()
        HG_ev1_evoke_rescaled_stderr = HG_ev1_rescaled.standard_error()

        channels = [] # load in all channels
        channel_to_index = {}
        channel_file = os.path.join(save_dir, f'channels_{sub}_GlobalLocal.txt') 
        with open(channel_file, 'r') as f:
            for line in f:
                index, channel = line.strip().split(': ')
                channels.append(channel)
                channel_to_index[channel] = int(index)

        sig_chans_filename = os.path.join(save_dir, f'sig_chans_{sub}_{output_name}.json') # load in sig channels
        sig_chans[window] = load_sig_chans(sig_chans_filename)

    # now plot 6x6 grid of 36 channels on one plot, for the z-scored signal
    plot_x_dim = 6
    plot_y_dim = 6
    channels_per_fig = plot_x_dim * plot_y_dim

    # Iterate over all channels in chunks and plot them with z-scored signal
    for i in range(0, len(channels), channels_per_fig):
        channels_subset = channels[i:i+channels_per_fig]
        fig = plot_channels_on_grid_windows(HG_ev1_evoke_rescaled, HG_ev1_evoke_rescaled_stderr, channels_subset, time_windows, sig_chans, sample_rate, plot_x_dim, plot_y_dim)
        combined_plot_path_rescaled = os.path.join(save_dir, f'{sub}_zscore_{output_name}_combinedChannelTracesAndWindowsSignificance_Page_{i//channels_per_fig + 1}.png')
        fig.savefig(combined_plot_path_rescaled)
        plt.close(fig)

    for i in range(0, len(channels), channels_per_fig):
        channels_subset = channels[i:i+channels_per_fig]
        fig = plot_channels_on_grid_windows(HG_ev1_evoke, HG_ev1_evoke_stderr, channels_subset, time_windows, sig_chans, sample_rate, plot_x_dim, plot_y_dim)
        combined_plot_path_rescaled = os.path.join(save_dir, f'{sub}_raw_{output_name}_combinedChannelTracesAndWindowsSignificance_Page_{i//channels_per_fig + 1}.png')
        fig.savefig(combined_plot_path_rescaled)
        plt.close(fig)

# %% [markdown]
# this below code is for when using the time perm cluster stats to determine significance timepoint by timepoint  
# it will plot the z-scored trace and the raw trace for each subject

# %%
def plot_channels_on_grid_time_perm_cluster(evoke_data, std_err_data, channels_subset, mat, sample_rate=2048, dec_factor=8, plot_x_dim=6, plot_y_dim=6):
    """
    Plots evoked EEG/MEG data for a subset of channels on a grid, overlaying significance markers for specified time windows.

    Parameters:
    - evoke_data: mne.Evoked object
        The evoked data to be plotted. This object contains the averaged EEG/MEG data over epochs.
    - std_err_data: 
        The standard error of the evoked data to be plotted
    - channels_subset: list of str
        A list of channel names to be plotted. Each channel name must correspond to a channel in `evoke_data`.
    - mat: numpy.array
        A binary matrix (same shape as evoke_data) indicating significant data points (1 for significant, 0 for non-significant).
    - sample_rate: float
        The sampling rate of the data, in Hz. Used to convert sample indices in `time_windows` to time in seconds.
    - dec_factor: int
        the decimation factor by which to downsample the sampling rate.
    - plot_x_dim: int, optional (default=6)
        The number of columns in the grid layout for plotting the channels.
    - plot_y_dim: int, optional (default=6)
        The number of rows in the grid layout for plotting the channels.

    Returns:
    - fig: matplotlib.figure.Figure object
        The figure object containing the grid of plots. Each plot shows the evoked data for a channel, with significance
        markers overlaid for the specified time windows.
    """
    fig, axes = plt.subplots(plot_x_dim, plot_y_dim, figsize=(20, 12))
    fig.suptitle("Channels with Significance Overlay")
    axes_flat = axes.flatten()

    for channel, ax in zip(channels_subset, axes_flat):
        stderr = std_err_data.data[channel_to_index[channel], :]
        time_in_seconds = np.arange(0, len(mat[channel_to_index[channel]])) / (sample_rate / dec_factor)  # Should be 2048 Hz sample rate
        sig_data_in_seconds = np.array(mat[channel_to_index[channel]])
        ax.plot(evoke_data.times, evoke_data.data[channel_to_index[channel], :])
         # Add the standard error shading
        ax.fill_between(evoke_data.times, evoke_data.data[channel_to_index[channel], :] - stderr, evoke_data.data[channel_to_index[channel], :] + stderr, alpha=0.2)

        # Find the maximum y-value for the current channel
        max_y_value = np.max(evoke_data.data[channel_to_index[channel], :])

        # Overlay significance as a horizontal line at the max y-value
        significant_points = np.where(sig_data_in_seconds == 1)[0]
        for point in significant_points:
            ax.hlines(y=max_y_value, xmin=time_in_seconds[point]-1, xmax=time_in_seconds[point] + 0.005 - 1, color='red', linewidth=1) # subtract 1 cuz the sig time is from 0 to 2.5, while the high gamma time is from -1 to 1.5

        ax.set_title(channel)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig

plot_x_dim = 6
plot_y_dim = 6
channels_per_fig = plot_x_dim * plot_y_dim

sig_chans = {}

for sub in subjects:
    task = 'GlobalLocal'
    LAB_root = None
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                    "CoganLab")

    layout = get_data(task, root=LAB_root)
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False)
    sample_rate = filt.info['sfreq'] # get sampling rate, should be 2048 Hz
    dec_factor = 8 # set this
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_name = 'Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8'

    # Define file paths
    HG_ev1_file = f'{save_dir}/{sub}_{output_name}_HG_ev1-epo.fif'
    HG_base_file = f'{save_dir}/{sub}_{output_name}_HG_base-epo.fif'
    HG_ev1_rescaled_file = f'{save_dir}/{sub}_{output_name}_HG_ev1_rescaled-epo.fif'

    # Load the epochs and evoked objects
    HG_ev1 = mne.read_epochs(HG_ev1_file)
    HG_base = mne.read_epochs(HG_base_file)
    HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
    HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0))
    HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))
    HG_ev1_evoke_stderr = HG_ev1.standard_error()
    HG_ev1_evoke_rescaled_stderr = HG_ev1_rescaled.standard_error()

    mat_save_path = os.path.join(save_dir, f'{output_name}_mat.npy')
    mat = np.load(mat_save_path)

    channels = [] # load in all channels
    channel_to_index = {}
    channel_file = os.path.join(save_dir, f'channels_{sub}_GlobalLocal.txt') 
    with open(channel_file, 'r') as f:
        for line in f:
            index, channel = line.strip().split(': ')
            channels.append(channel)
            channel_to_index[channel] = int(index)
    
    # Iterate over all channels in chunks of channels_per_fig (plot_x_dim * plot_y_dim) and plot them
    for i in range(0, len(channels), channels_per_fig):
        channels_subset = channels[i:i+channels_per_fig]
        fig = plot_channels_on_grid_time_perm_cluster(HG_ev1_evoke_rescaled, HG_ev1_evoke_rescaled_stderr, channels_subset, mat, sample_rate=sample_rate, dec_factor=dec_factor, plot_x_dim=plot_x_dim, plot_y_dim=plot_y_dim)
        combined_plot_path = os.path.join(save_dir, f'{sub}_zscore_{output_name}_channel_traces_page_{i//channels_per_fig + 1}.png')
        fig.savefig(combined_plot_path)
        plt.close(fig)

        # Iterate over all channels in chunks of channels_per_fig (plot_x_dim * plot_y_dim) and plot them
    for i in range(0, len(channels), channels_per_fig):
        channels_subset = channels[i:i+channels_per_fig]
        fig = plot_channels_on_grid_time_perm_cluster(HG_ev1_evoke, HG_ev1_evoke_stderr, channels_subset, mat, sample_rate=sample_rate, dec_factor=dec_factor, plot_x_dim=plot_x_dim, plot_y_dim=plot_y_dim)
        combined_plot_path = os.path.join(save_dir, f'{sub}_raw_{output_name}_channel_traces_page_{i//channels_per_fig + 1}.png')
        fig.savefig(combined_plot_path)
        plt.close(fig)
