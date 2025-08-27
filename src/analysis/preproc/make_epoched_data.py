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
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import numpy as np
from ieeg.calc.reshape import make_data_same
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
from src.analysis.utils.general_utils import calculate_RTs, save_channels_to_file, save_sig_chans, load_sig_chans

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

def bandpass_and_epoch_and_find_task_significant_electrodes(sub, task='GlobalLocal', times=(-1, 1.5),
                      within_base_times=(-1, 0), base_times_length=0.5, baseline_event="Stimulus", pad_length = 0.5, LAB_root=None, channels=None, dec_factor=8, mark_outliers_as_nan=True, outliers=10, passband=(70,150), stat_func=partial(ttest_ind, equal_var=False)):
    """
    Bandpass the filtered data, epoch around Stimulus and Response onsets, and find electrodes with significantly different activity from baseline for a given subject.

    Parameters:
    - sub (str): The subject identifier.
    - task (str): The task identifier.
    - times (tuple, optional): A tuple indicating the start and end times for event processing. Defaults to (-1, 1.5).
    - times (tuple [float, float]): The time window to epoch around the event.
    - within_base_times (tuple [float, float]): The time window within which to randomly select intervals for each event, for baseline.
    - base_times_length (float): The length of the time intervals to randomly select within `within_base_times`. 
    - baseline_event (str): The event to use for baseline. Use "experimentStart" for beginning of experiment, or use "Stimulus" for pre-stimulus. Not sure if "experimentStart" actually exists, check how to make this. 
    - pad_length (float): The length to pad each time interval. Will be removed later.
    - LAB_root (str, optional): The root directory for the lab. Will be determined based on OS if not provided. Defaults to None.
    - channels (list of strings, optional): The channels to plot and get stats for. Default is all channels.
    - decimation_factor (int, optional): The factor by which to subsample the data. Default is 10, so should be 2048 Hz down to 204.8 Hz.
    - mark_outliers_as_nan (bool, optional): Whether to mark outliers as NaN. Defaults to True.
    - outliers (int, optional): How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.
    - passband (tuple, optional): The frequency range for the frequency band of interest. Default is (70, 150).
    - stat_func (function, optional): The statistical function to use for significance testing. Default is ttest_ind(equal_var=False).
    
    This function will process the provided event for a given subject and task.
    Bandpassed and epoched data will be computed, and statistics will be calculated and plotted.
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
    # within_times_duration = abs(within_base_times[1] - within_base_times[0]) #grab the duration as a string for naming

    # debugging 6/17/25
    # Create a baseline EpochsTFR using the baseline event
    if baseline_event == "experimentStart":
        # Adjust the time window
        within_base_times_adj = [within_base_times[0] - pad_length, within_base_times[1] + pad_length]
        trials = trial_ieeg(good, baseline_event, within_base_times_adj, preload=True)
    else:
        trials = trial_ieeg_rand_offset(good, baseline_event, within_base_times, base_times_length, pad_length, preload=True)
    
    if mark_outliers_as_nan:
        outliers_to_nan(trials, outliers=outliers)
    HG_base = gamma.extract(trials, passband=passband, copy=False, n_jobs=1)
    pad_length_string = f"{pad_length}s" # define pad_length as a string so can use it as input to crop_pad
    crop_pad(HG_base, pad_length_string) # need to change this if pad length changes
    HG_base.decimate(dec_factor)
    
    # Square the data to get power from amplitude
    HG_base_power = HG_base.copy()
    HG_base_power._data = HG_base._data ** 2  # Square amplitude to get power
    
    if isinstance(stat_func, partial):
        base_func_name = stat_func.func.__name__
        # Create a descriptive name like "ttest_ind_equal_var_False"
        keywords_str = "_".join(f"{k}_{v}" for k, v in sorted(stat_func.keywords.items()))
        if keywords_str: # If there are keywords like equal_var
            stat_func_for_filename = f"{base_func_name}_{keywords_str}"
        else: # If partial was used without keywords (less likely here)
            stat_func_for_filename = base_func_name
    elif hasattr(stat_func, '__name__'): # For regular functions
        stat_func_for_filename = stat_func.__name__
    elif isinstance(stat_func, str): # If a string was somehow passed (e.g., from a less robust CLI)
        # Sanitize or use the string directly if it's simple.
        # For safety, you might want to ensure it's a valid filename component.
        stat_func_for_filename = stat_func.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    else:
        stat_func_for_filename = "custom_stat_func" # Fallback
    # need to adapt this to just have a randoffset variable instead of hard coding the output_name_base
    if mark_outliers_as_nan:
        if baseline_event == "experimentStart" or baseline_event == 'experimentStart':
            output_name_base = f"{base_times_length}sec_within{within_base_times[0]}-{within_base_times[1]}sec_{baseline_event}Base_decFactor_{dec_factor}_markOutliersAsNaN_{mark_outliers_as_nan}_outliers_{outliers}_passband_{passband[0]}-{passband[1]}_padLength_{pad_length}s_stat_func_{stat_func_for_filename}"
        else:
            output_name_base = f"{base_times_length}sec_within{within_base_times[0]}-{within_base_times[1]}sec_randoffset_{baseline_event}Base_decFactor_{dec_factor}_markOutliersAsNaN_{mark_outliers_as_nan}_outliers_{outliers}_passband_{passband[0]}-{passband[1]}_padLength_{pad_length}s_stat_func_{stat_func_for_filename}"
    else:
        if baseline_event == "experimentStart" or baseline_event == 'experimentStart':
            output_name_base = f"{base_times_length}sec_within{within_base_times[0]}-{within_base_times[1]}sec_{baseline_event}Base_decFactor_{dec_factor}_markOutliersAsNaN_{mark_outliers_as_nan}_passband_{passband[0]}-{passband[1]}_padLength_{pad_length}s_stat_func_{stat_func_for_filename}"
        else:
            output_name_base = f"{base_times_length}sec_within{within_base_times[0]}-{within_base_times[1]}sec_randoffset_{baseline_event}Base_decFactor_{dec_factor}_markOutliersAsNaN_{mark_outliers_as_nan}_passband_{passband[0]}-{passband[1]}_padLength_{pad_length}s_stat_func_{stat_func_for_filename}"
    
    for event in ["Stimulus", "Response"]:
        output_name_event = f'{event}_{output_name_base}'
        times_adj = [times[0] - pad_length, times[1] + pad_length]
        trials = trial_ieeg(good, event, times_adj, preload=True,
                            reject_by_annotation=False)

        if mark_outliers_as_nan:
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
        HG_ev1_evoke.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke-ave.fif', overwrite=True)
        HG_ev1_evoke_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power-ave.fif', overwrite=True)
        
        # Save HG_ev1_evoke_rescaled
        HG_ev1_evoke_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_rescaled-ave.fif', overwrite=True)
        HG_ev1_evoke_power_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power_rescaled-ave.fif', overwrite=True)

        ###
        print(f"Shape of HG_ev1._data: {HG_ev1._data.shape}")
        print(f"Shape of HG_base._data: {HG_base._data.shape}")
        
        # oh this changed and returns both the significant clusters matrix and the p values now
        mat = time_perm_cluster(HG_ev1._data, HG_base._data, 0.05, n_jobs=6, ignore_adjacency=1, stat_func=stat_func)[0]

        #save channels with their indices 
        save_channels_to_file(channels, sub, task, save_dir)

        # save significant channels to a json
        save_sig_chans(f'{output_name_event}', mat, channels, sub, save_dir)
        
        # Assuming `mat` is your array and `save_dir` is the directory where you want to save it
        mat_save_path = os.path.join(save_dir, f'{output_name_event}_mat.npy')

        # Save the mat array
        np.save(mat_save_path, mat)

# %%

def main(subjects=None, task='GlobalLocal', times=(-1, 1.5),
         within_base_times=(-1, 0), base_times_length=0.5, pad_length=0.5, LAB_root=None, channels=None, dec_factor=8, mark_outliers_as_nan=True, outliers=10, passband=(70,150), stat_func=partial(ttest_ind, equal_var=False)):
    """
    Main function to bandpass filter and compute time permutation cluster stats and task-significant electrodes for chosen subjects.
    """
    if subjects is None:
        subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121']

    for sub in subjects:
        bandpass_and_epoch_and_find_task_significant_electrodes(sub=sub, task=task, times=times,
                          within_base_times=within_base_times, base_times_length=base_times_length,
                          pad_length=pad_length, LAB_root=LAB_root, channels=channels,
                          dec_factor=dec_factor, outliers=outliers, passband=passband, stat_func=stat_func)
        
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
    parser.add_argument('--mark_outliers_as_nan', type=bool, default=True, help='Whether to mark outliers as NaN. Default is True.')
    parser.add_argument('--outliers', type=int, default=10, help='How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.')
    parser.add_argument('--passband', type=float, nargs=2, default=(70,150), help='Frequency range for the frequency band of interest. Default is (70, 150).')
    parser.add_argument('--stat_func', default=partial(ttest_ind, equal_var=False), help='Statistical function to use for significance testing. Default is ttest_ind(equal_var=False).')
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
        mark_outliers_as_nan=args.mark_outliers_as_nan,
        outliers=args.outliers, 
        passband=args.passband,
        stat_func=args.stat_func)