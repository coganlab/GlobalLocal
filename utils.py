import mne
import json
import numpy as np
import os
import pandas as pd
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
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def make_subjects_electrodestoROIs_dict(subjects):
    '''
    makes mappings for each electrode to its roi
    subjects: list of strings of subject numbers
    '''
    
    # Initialize the outer dictionary.
    subjects_electrodestoROIs_dict = {}

    for sub in subjects:
        print(sub)
        task = 'GlobalLocal'
        LAB_root = None
        channels = None

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

        good.info['bads'] = channel_outlier_marker(good, 3, 2)

        # Drop the trigger channel if it exists 9/30
        if 'Trigger' in good.ch_names:
            good.drop_channels('Trigger')

        filt.drop_channels(good.info['bads'])  # this has to come first cuz if you drop from good first, then good.info['bads'] is just empty
        good.drop_channels(good.info['bads'])

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

        default_dict = gen_labels(good.info)
        
        # Create rawROI_dict for the subject
        rawROI_dict = defaultdict(list)
        for key, value in default_dict.items():
            rawROI_dict[value].append(key)
        rawROI_dict = dict(rawROI_dict)

        # Filter out keys containing "White-Matter"
        filtROI_dict = {key: value for key, value in rawROI_dict.items() if "White-Matter" not in key}

        # Store the dictionaries in the subjects dictionary
        subjects_electrodestoROIs_dict[sub] = {
            'default_dict': dict(default_dict),
            'rawROI_dict': dict(rawROI_dict),
            'filtROI_dict': dict(filtROI_dict)
        }


    # # Save to a JSON file. Uncomment when actually running.
    filename = 'subjects_electrodestoROIs_dict.json'
    with open(filename, 'w') as file:
        json.dump(subjects_electrodestoROIs_dict, file, indent=4)

    print(f"Saved subjects_dict to {filename}")


def load_subjects_electrodestoROIs_dict(filename='subjects_electrodestoROIs_dict.json'):
    """
    Attempts to load the subjects' electrode to ROI dictionary from a JSON file.
    Returns the dictionary if successful, None otherwise.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        print(f"Loaded data from {filename}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load {filename}: {e}")
        return None
    
def make_or_load_subjects_electrodes_to_rois_dict(filename, subjects):
    """
    Ensure the subjects' electrodes to ROIs dictionary is available.
    If the dictionary doesn't exist, it is created and then loaded.

    Parameters:
    filename: The name of the file where the dictionary is stored.
    subjects: List of subjects, required if the dictionary needs to be created.

    Returns:
    A dictionary mapping subjects to their electrodes and associated ROIs.
    """
    print("Attempting to load the subjects' electrodes-to-ROIs dictionary...")
    subjects_electrodestoROIs_dict = load_subjects_electrodestoROIs_dict(filename)

    if subjects_electrodestoROIs_dict is None:
        print("No dictionary found. Looks like it's our lucky day to create one!")
        make_subjects_electrodestoROIs_dict(subjects)
        subjects_electrodestoROIs_dict = load_subjects_electrodestoROIs_dict(filename)
        print("Dictionary created and loaded successfully. Let's roll!")

    else:
        print("Dictionary loaded successfully. Ready to proceed!")

    return subjects_electrodestoROIs_dict

def load_mne_objects(sub, epochs_root_file, task, just_HG_ev1_rescaled=False, LAB_root=None):
    """
    Load MNE objects for a given subject and output name, with an option to load only rescaled high gamma epochs.

    Parameters:
    - sub (str): Subject identifier.
    - epochs_root_file (str): Name of the original epochs object that we will be indexing using our conditions. Use Stimulus_1sec_preStimulusBase_decFactor_10 for now.
    - task (str): Task identifier.
    - just_HG_ev1_rescaled (bool): If True, only the rescaled high gamma epochs are loaded.
    - LAB_root (str, optional): Root directory for the lab. If None, it will be determined based on the OS.

    Returns:
    A dictionary containing loaded MNE objects.
    """

    # Determine LAB_root based on the operating system
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        LAB_root = os.path.join(HOME, "Box", "CoganLab") if os.name == 'nt' else os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

    # Get data layout
    layout = get_data(task, root=LAB_root)
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the return dictionary
    mne_objects = {}

    if just_HG_ev1_rescaled:
        # Define path and load only the rescaled high gamma epochs
        HG_ev1_rescaled_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_ev1_rescaled-epo.fif'
        HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
        mne_objects['HG_ev1_rescaled'] = HG_ev1_rescaled

        HG_ev1_power_rescaled_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_ev1_power_rescaled-epo.fif'
        HG_ev1_power_rescaled = mne.read_epochs(HG_ev1_power_rescaled_file)
        mne_objects['HG_ev1_power_rescaled'] = HG_ev1_power_rescaled

    else:
        # Define file paths
        HG_ev1_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_ev1-epo.fif'
        HG_base_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_base-epo.fif'
        HG_ev1_rescaled_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_ev1_rescaled-epo.fif'
        HG_ev1_power_rescaled_file = f'{save_dir}/{sub}_{epochs_root_file}_HG_ev1_power_rescaled-epo.fif'

        # Load the objects
        HG_ev1 = mne.read_epochs(HG_ev1_file)
        HG_base = mne.read_epochs(HG_base_file)
        HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0))
        HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
        HG_ev1_power_rescaled = mne.read_epochs(HG_ev1_power_rescaled_file)
        HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

        mne_objects['HG_ev1'] = HG_ev1
        mne_objects['HG_base'] = HG_base
        mne_objects['HG_ev1_evoke'] = HG_ev1_evoke
        mne_objects['HG_ev1_rescaled'] = HG_ev1_rescaled
        mne_objects['HG_ev1_power_rescaled'] = HG_ev1_power_rescaled
        mne_objects['HG_ev1_evoke_rescaled'] = HG_ev1_evoke_rescaled

    return mne_objects

def create_subjects_mne_objects_dict(subjects, epochs_root_file, conditions, task, just_HG_ev1_rescaled=False, LAB_root=None, acc_trials_only=True):
    """
    Adjusted to handle multiple conditions per output name, with multiple condition columns.

    Parameters:
    - subjects: List of subject IDs.
    - output_names_conditions: Dictionary where keys are output names and values are dictionaries
        of condition column names and their required values.
    - task: Task identifier.
    - combined_data: DataFrame with combined behavioral and trial information.
    - acc_array: dict of numpy arrays of 0 for incorrect and 1 for correct trials for each subject
    - LAB_root: Root directory for data (optional).
    """
    subjects_mne_objects = {}

    for sub in subjects:
        print(f"Loading data for subject: {sub}")
        sub_mne_objects = {}

        mne_objects = load_mne_objects(sub, epochs_root_file, task, just_HG_ev1_rescaled=just_HG_ev1_rescaled, LAB_root=LAB_root)
        for mne_object in mne_objects.keys():
            if acc_trials_only == True:
                mne_objects[mne_object] = mne_objects[mne_object]["Accuracy1.0"] # this needs to be done for all the epochs objects I think. So loop over them. Unless it's set to just_HG_ev1_rescaled.

            for condition_name, condition_parameters in conditions.items():
                print(f"  Loading condition: {condition_name} with parameters: {condition_parameters}")
                # Get BIDS events from the conditions, and remove it so it doesn't complicate future analyses.
                bids_events = condition_parameters.get("BIDS_events")
                if bids_events is None:
                    print(f"Warning: condition {condition_name} is missing 'BIDS_events'. Fix this!")
                # if multiple bids events are part of this condition, concatenate their epochs. Otherwise just grab epochs.
                if isinstance(bids_events, list):
                    combined_epochs = []
                    for event in bids_events:
                        partial_event_epochs = mne_objects[mne_object][event]
                        combined_epochs.append(partial_event_epochs)
                    event_epochs = mne.concatenate_epochs(combined_epochs)
                else:
                    event_epochs = mne_objects[mne_object][bids_events]

                sub_mne_objects[condition_name] = {}
                sub_mne_objects[condition_name][mne_object] = event_epochs
            subjects_mne_objects[sub] = sub_mne_objects

    return subjects_mne_objects
    
def load_acc_arrays(npy_directory, skip_subjects=None):
    """
    Loads accuracy arrays from .npy files for each subject within a specified directory, 
    skipping specified subjects.

    Parameters:
        npy_directory (str): The path to the directory containing the .npy files.
        skip_subjects (list): A list of subject IDs to skip when loading arrays.

    Returns:
        dict: A dictionary where keys are subject IDs and values are numpy arrays loaded from .npy files.
    """
    acc_array = {}

    try:
        # Iterate over each file in the directory
        for file in os.listdir(npy_directory):
            if file.endswith('.npy'):
                subject_id = file.split('_')[0]  # Extract subject ID from the file name
                if subject_id not in skip_subjects:
                    # Construct the full file path
                    file_path = os.path.join(npy_directory, file)
                    # Load the numpy array from the file
                    acc_array[subject_id] = np.load(file_path)
    except Exception as e:
        print(f"Error occurred: {e}")

    return acc_array


def calculate_RTs(raw):
    annotations = raw.annotations
    reaction_times = []
    skipped = []

    for i in range(len(annotations) - 1):
        current_annotation = annotations[i]
        next_annotation = annotations[i + 1]
        if 'Stimulus' in current_annotation['description']:
            if 'Response' in next_annotation['description']:
                reaction_time = next_annotation['onset'] - current_annotation['onset']
                reaction_times.append(reaction_time)
            else:
                skipped.append(i)

    return reaction_times, skipped


def save_sig_chans(epochs_root_file, mask, channels, subject, save_path):
    # Get the indices of the channels that are significant at any time point
    significant_indices = np.any(mask, axis=1)
    
    # Convert indices to channel names (optional)
    sig_chans = [channels[i] for i in np.where(significant_indices)[0]]
    
    # Create a dictionary to store the data
    data = {
        "subject": subject,
        "sig_chans": sig_chans
    }
    
    # Define the filename
    filename = os.path.join(save_path, f'sig_chans_{subject}_{epochs_root_file}.json')
    
    # Save the dictionary as a JSON file
    with open(filename, 'w') as file:
        json.dump(data, file)
    
    print(f'Saved significant channels for subject {subject} and epochs root file {epochs_root_file} to {filename}')


def load_sig_chans(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # You can access the subject and significant channels directly from the dictionary
    subject = data['subject']
    sig_chans = data['sig_chans']

    print(f'Loaded significant channels for subject {subject}')
    return sig_chans


def channel_names_to_indices(sig_chans, channels):
    indices = [channels.index(chan_name) for chan_name in sig_chans if chan_name in channels]
    return indices

# untested code 8/21/23
def save_channels_to_file(channels, subject, task, save_dir):
    """
    Save each channel name and its corresponding index to a text file.
    
    Parameters:
    - channels (list): The list of channel names.
    - subject (str): The subject identifier.
    - task (str): The task identifier.
    - save_dir (str): The directory where the text file should be saved.
    """
    channel_text_filename = os.path.join(save_dir, f'channels_{subject}_{task}.txt')
    with open(channel_text_filename, 'w') as channel_file:
        for i, channel_name in enumerate(channels):
            channel_file.write(f"{i}: {channel_name}\n")
    
    print(f'Saved channel names and indices to {channel_text_filename}')


def filter_and_average_epochs(epochs, start_idx, end_idx):
    """
    Calculates trial averages for accurate trials and time averages with inaccurate trials marked as NaNs.
    This assumes you've already indexed accurate trials in subject_mne_objects.

    Parameters:
    - epochs: MNE Epochs object with accuracy metadata.
    - start_idx: Start index for time averaging.
    - end_idx: End index for time averaging.

    Returns:
    - trial_avg_data: Trial-averaged data across accurate trials.
    - time_avg_data: Time-averaged data with inaccurate trials marked as NaNs.
    """
    epochs_data = epochs.get_data().copy()

    # Calculate trial average for accurate trials
    trial_avg_data = np.nanmean(epochs_data, axis=0)

    # Calculate trial standard deviation for accurate trials
    trial_std_data = np.nanstd(epochs_data, axis=0)

    # Calculate time average within the specified window
    time_avg_data = np.nanmean(epochs_data[:, :, start_idx:end_idx], axis=2)

    return trial_avg_data, trial_std_data, time_avg_data


def permutation_test(data_timeavg_output_0, data_timeavg_output_1, n_permutations=10000, one_tailed=False):
    """
    Perform a permutation test to compare two conditions.

    Parameters:
    - data_timeavg_output_0: Numpy array for condition 0.
    - data_timeavg_output_1: Numpy array for condition 1.
    - n_permutations: Number of permutations to perform.
    - one_tailed: Boolean indicating if the test should be one-tailed. False by default.

    Returns:
    - p_value: P-value assessing the significance of the observed difference.
    """
    # Calculate the observed difference in means between the two conditions
    observed_diff = np.nanmean(data_timeavg_output_0) - np.nanmean(data_timeavg_output_1)
    
    # Combine the data from both conditions
    combined_data = np.hstack([data_timeavg_output_0, data_timeavg_output_1])
    
    # Initialize a variable to count how many times the permuted difference exceeds the observed difference
    count_extreme_values = 0
    
    for _ in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined_data) #this shuffles in the 0th dimension (trials)
        
        # Split the shuffled data back into two new groups
        permuted_0 = combined_data[:len(data_timeavg_output_0)]
        permuted_1 = combined_data[len(data_timeavg_output_0):]
        
        # Calculate the mean difference for this permutation
        permuted_diff = np.nanmean(permuted_0) - np.nanmean(permuted_1)
        
        # Check if the permuted difference is as extreme as the observed difference
        # For a one-tailed test, only count when permuted_diff is greater than observed_diff
        if one_tailed:
            if permuted_diff > observed_diff:
                count_extreme_values += 1
        else:
            if abs(permuted_diff) >= abs(observed_diff):
                count_extreme_values += 1
    
    # Calculate the p-value
    p_value = count_extreme_values / n_permutations
    
    return p_value


def perform_permutation_test_within_electrodes(data_0_list, data_1_list, n_permutations=10000, one_tailed=False):
    """
    Perform a permutation test for each electrode comparing two conditions across subjects.
    
    Parameters:
    - data_0_list: List of subject arrays from condition 0, each array is trials x electrodes.
    - data_1_list: List of subject arrays from condition 1, each array is trials x electrodes.
    - n_permutations: Number of permutations for the test.
    
    Returns:
    - p_values: A list of p-values for each electrode, across all subjects.
    """
    p_values = []

    # Ensure there is a corresponding condition 1 array for each condition 0 array
    if len(data_0_list) != len(data_1_list):
        raise ValueError("Mismatch in number of subjects between conditions")

    # Iterate through each subject's data arrays
    for idx, (data_0, data_1) in enumerate(zip(data_0_list, data_1_list)):
        print(f"Subject {idx} - Condition 0 shape: {data_0.shape}, Condition 1 shape: {data_1.shape}")

        # Check for matching electrode counts between conditions within a subject
        if data_0.shape[1] != data_1.shape[1]:
            raise ValueError(f"Electrode count mismatch in subject {idx}")

        n_electrodes_this_sub = data_0.shape[1]  # Number of electrodes for this subject

        # Perform the permutation test for each electrode in this subject
        for electrode_idx in range(n_electrodes_this_sub):  # Fix: use range(n_electrodes) to iterate correctly
            p_value = permutation_test(data_0[:, electrode_idx], data_1[:, electrode_idx], n_permutations, one_tailed)
            p_values.append(p_value)

    return p_values


def perform_permutation_test_across_electrodes(data_0_list, data_1_list, n_permutations=10000, one_tailed=False):
    """
    Perform a permutation test across electrodes comparing two conditions.
    
    Parameters:
    - data_0_list: List of arrays from condition 0, each array is trials x electrodes.
    - data_1_list: List of arrays from condition 1, each array is trials x electrodes.
    - n_permutations: Number of permutations for the test.
    
    Returns:
    - p_value: P-value from the permutation test.
    """
    # Aggregate data across electrodes
    data_0_aggregated = np.concatenate([np.nanmean(data, axis=0) for data in data_0_list])  # Average across trials to get a single value per electrode
    data_1_aggregated = np.concatenate([np.nanmean(data, axis=0) for data in data_1_list])  # though should I do avg across electrodes instead..?? Uhhhh. No, I think.
    
    # Perform the permutation test
    p_value = permutation_test(data_0_aggregated, data_1_aggregated, n_permutations, one_tailed)
    
    return p_value


def add_accuracy_to_epochs(epochs, accuracy_array):
    """
    Adds accuracy data from accuracy_array to the metadata of epochs.
    Assumes the order of trials in accuracy_array matches the order in epochs.
    """
    if epochs.metadata is None:
        # Create a new DataFrame if no metadata exists
        epochs.metadata = pd.DataFrame(index=range(len(epochs)))
    
    # Ensure the accuracy_array length matches the number of epochs
    assert len(accuracy_array) == len(epochs), "Mismatch in number of trials and accuracy data length."
    
    # Add the accuracy array as a new column in the metadata
    epochs.metadata['accuracy'] = accuracy_array

    # Reset the index to ensure it's sequential starting from 0
    epochs.metadata.reset_index(drop=True, inplace=True)
    
    return epochs


def save_sig_chans_with_reject(output_name, reject, channels, subject, save_dir):
    # Determine which channels are significant based on the reject array
    significant_indices = np.where(reject)[0]
    
    # Convert significant indices to channel names
    sig_chans = [channels[i] for i in significant_indices]
    
    # Create a dictionary to store the data
    data = {
        "subject": subject,
        "sig_chans": sig_chans
    }
    
    # Define the filename
    filename = os.path.join(save_dir, f'sig_chans_{subject}_{output_name}.json')
    
    # Save the dictionary as a JSON file
    with open(filename, 'w') as file:
        json.dump(data, file)
    
    print(f'Saved significant channels for subject {subject} and {output_name} to {filename}')

def create_subjects_mne_objects_dict(subjects, epochs_root_file, conditions, task, just_HG_ev1_rescaled=False, LAB_root=None, acc_trials_only=True):
    """
    Adjusted to handle multiple conditions per output name, with multiple condition columns.

    Parameters:
    - subjects: List of subject IDs.
    - output_names_conditions: Dictionary where keys are output names and values are dictionaries
        of condition column names and their required values.
    - task: Task identifier.
    - combined_data: DataFrame with combined behavioral and trial information.
    - acc_array: dict of numpy arrays of 0 for incorrect and 1 for correct trials for each subject
    - LAB_root: Root directory for data (optional).
    """
    subjects_mne_objects = {}

    for sub in subjects:
        print(f"Loading data for subject: {sub}")
        sub_mne_objects = {}

        mne_objects = load_mne_objects(sub, epochs_root_file, task, just_HG_ev1_rescaled=just_HG_ev1_rescaled, LAB_root=LAB_root)
        for mne_object in mne_objects.keys():
            if acc_trials_only == True:
                mne_objects[mne_object] = mne_objects[mne_object]["Accuracy1.0"] # this needs to be done for all the epochs objects I think. So loop over them. Unless it's set to just_HG_ev1_rescaled.

            for condition_name, condition_parameters in conditions.items():
                print(f"  Loading condition: {condition_name} with parameters: {condition_parameters}")
                # Get BIDS events from the conditions, and remove it so it doesn't complicate future analyses.
                bids_events = condition_parameters.get("BIDS_events")
                if bids_events is None:
                    print(f"Warning: condition {condition_name} is missing 'BIDS_events'. Fix this!")
                # if multiple bids events are part of this condition, concatenate their epochs. Otherwise just grab epochs.
                if isinstance(bids_events, list):
                    combined_epochs = []
                    for event in bids_events:
                        partial_event_epochs = mne_objects[mne_object][event]
                        combined_epochs.append(partial_event_epochs)
                    event_epochs = mne.concatenate_epochs(combined_epochs)
                else:
                    event_epochs = mne_objects[mne_object][bids_events]

                sub_mne_objects[condition_name] = {}
                sub_mne_objects[condition_name][mne_object] = event_epochs
            subjects_mne_objects[sub] = sub_mne_objects

    return subjects_mne_objects

def initialize_output_data(rois, condition_names):
    """
    Initialize dictionaries for storing data across different conditions and ROIs.
    """
    return {condition_name: {roi: [] for roi in rois} for condition_name in condition_names}
def process_data_for_roi(subjects_mne_objects, condition_names, rois, subjects, sig_electrodes_per_subject_roi, time_indices):
    """
    Process data by ROI, calculating averages for different time windows for either the first two outputs or all outputs, depending on the analysis purpose.
    """

    # Initialize data structures for trial averages, trial standard deviations, and time averages
    data_trialAvg_lists = initialize_output_data(rois, condition_names)
    data_trialStd_lists = initialize_output_data(rois, condition_names)
    data_timeAvg_lists = {suffix: initialize_output_data(rois, condition_names) for suffix in ['firstHalfSecond', 'secondHalfSecond', 'fullSecond']}
    overall_electrode_mapping = []
    electrode_mapping_per_roi = {roi: [] for roi in rois}  # Reinitialize for each processing run

    for sub in subjects:
        for roi in rois:
            sig_electrodes = sig_electrodes_per_subject_roi[roi].get(sub, [])
            print(f"Subject: {sub}, ROI: {roi}, Num of Sig Electrodes: {len(sig_electrodes)}")  # Debug print

            if not sig_electrodes:
                continue

            for condition_name in condition_names:
                epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick_channels(sig_electrodes)
                # Append mapping information for use in ANOVA.
                for electrode in sig_electrodes:
                    index = len(overall_electrode_mapping)
                    overall_electrode_mapping.append((sub, roi, electrode, index))
                    index_roi = len(electrode_mapping_per_roi[roi])
                    electrode_mapping_per_roi[roi].append((sub, electrode, index_roi))

                # Compute trial averages and standard deviations once per output per subject per ROI
                trial_avg, trial_std, _ = filter_and_average_epochs(epochs, start_idx=None, end_idx=None)
                data_trialAvg_lists[condition_name][roi].append(trial_avg)
                data_trialStd_lists[condition_name][roi].append(trial_std)

                # compute time average for each output per subject per roi for each time window. But why don't we look at standard deviation? 4/30
                for suffix, (start_idx, end_idx) in time_indices.items():
                    _, _, time_avg = filter_and_average_epochs(epochs, start_idx, end_idx)
                    data_timeAvg_lists[suffix][condition_name][roi].append(time_avg)

    return data_trialAvg_lists, data_trialStd_lists, data_timeAvg_lists, overall_electrode_mapping, electrode_mapping_per_roi

def concatenate_data(data_lists, rois, condition_names):
    """
    Concatenate data across subjects for each ROI and condition.
    """
    concatenated_data = {condition_name: {roi: np.concatenate(data_lists[condition_name][roi], axis=0) for roi in rois} for condition_name in condition_names}
    return concatenated_data
def calculate_mean_and_sem(concatenated_data, rois, condition_names):
    """
    Calculate mean and SEM across electrodes for all time windows and rois
    """
    mean_and_sem = {roi: {condition_name: {} for condition_name in condition_names} for roi in rois}
    for roi in rois:
        for condition_name in condition_names:
            trial_data = concatenated_data[condition_name][roi]
            mean = np.nanmean(trial_data, axis=0)
            sem = np.std(trial_data, axis=0, ddof=1) / np.sqrt(trial_data.shape[0])
            mean_and_sem[roi][condition_name] = {'mean': mean, 'sem': sem}
    return mean_and_sem

def calculate_time_perm_cluster_for_each_roi(concatenated_data, rois, output_names, alpha=0.05, n_jobs=6):
    """
    Perform time permutation cluster tests between the first two outputs for each ROI.
    Assumes that there are at least two output conditions to compare.
    """
    time_perm_cluster_results = {}
    for roi in rois:
        time_perm_cluster_results[roi] = time_perm_cluster(
            concatenated_data[output_names[0]][roi],
            concatenated_data[output_names[1]][roi], alpha, n_jobs=n_jobs
        )
    return time_perm_cluster_results

def extract_significant_effects(anova_table):
    """
    Extract significant effects and their p-values from the ANOVA results table,
    removing 'C(...)' from effect names and formatting them neatly.
    """
    significant_effects = []
    for effect in anova_table.index:
        p_value = anova_table.loc[effect, 'PR(>F)']
        if p_value < 0.05:
            # Remove 'C(' and ')' from the effect names
            formatted_effect = effect.replace('C(', '').replace(')', '')
            significant_effects.append((formatted_effect, p_value))
    return significant_effects


def convert_dataframe_to_serializable_format(df):
    """
    Convert a pandas DataFrame to a serializable format that can be used with json.dump.
    """
    return df.to_dict(orient='records')


def perform_modular_anova(df, time_window, output_names_conditions, save_dir, save_name):
    # Filter for a specific time window (I should probably make this not have a time_window input and just loop over all time windows like the within electrode code does)
    df_filtered = df[df['TimeWindow'] == time_window]

    # Dynamically construct the model formula based on condition keys
    condition_keys = [key for key in output_names_conditions[next(iter(output_names_conditions))].keys()]
    formula_terms = ' + '.join([f'C({key})' for key in condition_keys])
    interaction_terms = ' * '.join([f'C({key})' for key in condition_keys])
    formula = f'MeanActivity ~ {formula_terms} + {interaction_terms}'

    # Define the model
    model = ols(formula, data=df_filtered).fit()

    # Perform the ANOVA
    anova_results = anova_lm(model, typ=2)

    # Define the full path for the results file
    results_file_path = os.path.join(save_dir, save_name)

    # Save the ANOVA results to a text file
    with open(results_file_path, 'w') as file:
        file.write(anova_results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"ANOVA results saved to: {results_file_path}")

    # Print the results
    print(anova_results)

    return anova_results

def perform_modular_anova_all_time_windows(df, output_names_conditions, save_dir, save_name_prefix):
    # Dynamically construct the model formula based on condition keys and include TimeWindow
    condition_keys = [key for key in output_names_conditions[next(iter(output_names_conditions))].keys()]
    formula_terms = ' + '.join([f'C({key})' for key in condition_keys] + ['C(TimeWindow)'])
    interaction_terms = ' * '.join([f'C({key})' for key in condition_keys] + ['C(TimeWindow)'])
    formula = f'MeanActivity ~ {formula_terms} + {interaction_terms}'

    # Define the model
    model = ols(formula, data=df).fit()

    # Perform the ANOVA
    anova_results = anova_lm(model, typ=2)

    # Define the base part of the results file name
    results_file_path = os.path.join(save_dir, f"{save_name_prefix}_ANOVAacrossElectrodes_allTimeWindows.txt")

    # Save the ANOVA results to a text file
    with open(results_file_path, 'w') as file:
        file.write(anova_results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"ANOVA results for all time windows saved to: {results_file_path}")

    # Print the results
    print(anova_results)

    return anova_results


def make_plotting_parameters():
    # add the other conditions and give them condition names and colors too
    plotting_parameters = {
        'Stimulus_r25and75_fixationCrossBase_1sec_mirror': {
            'condition_name': 'repeat',
            'color': 'red',
            "line_style": "-"
        },
        'Stimulus_s25and75_fixationCrossBase_1sec_mirror': {
            'condition_name': 'switch',
            'color': 'green',
            "line_style": "-"
        },
        'Stimulus_c25and75_fixationCrossBase_1sec_mirror': {
            'condition_name': 'congruent',
            'color': 'blue',
            "line_style": "-"
        },
        'Stimulus_i25and75_fixationCrossBase_1sec_mirror': {
            'condition_name': 'incongruent',
            'color': 'orange',
            "line_style": "-"
        },
        "Stimulus_ir_fixationCrossBase_1sec_mirror": {
            "condition_name": "IR",
            "color": "blue",
            "line_style": "-"
        },
        "Stimulus_is_fixationCrossBase_1sec_mirror": {
            "condition_name": "IS",
            "color": "blue",
            "line_style": "--"
        },
        "Stimulus_cr_fixationCrossBase_1sec_mirror": {
            "condition_name": "CR",
            "color": "red",
            "line_style": "-"
        },
        "Stimulus_cs_fixationCrossBase_1sec_mirror": {
            "condition_name": "CS",
            "color": "red",
            "line_style": "--"
        },
        "Stimulus_c25_fixationCrossBase_1sec_mirror": {
            "condition_name": "c25",
            "color": "red",
            "line_style": "--"
        },
        "Stimulus_c75_fixationCrossBase_1sec_mirror": {
            "condition_name": "c75",
            "color": "red",
            "line_style": "-"
        },
        "Stimulus_i25_fixationCrossBase_1sec_mirror": {
            "condition_name": "i25",
            "color": "blue",
            "line_style": "--"
        },
        "Stimulus_i75_fixationCrossBase_1sec_mirror": {
            "condition_name": "i75",
            "color": "blue",
            "line_style": "-"
        },
        "Stimulus_s25_fixationCrossBase_1sec_mirror": {
            "condition_name": "s25",
            "color": "green",
            "line_style": "--"
        },
        "Stimulus_s75_fixationCrossBase_1sec_mirror": {
            "condition_name": "s75",
            "color": "green",
            "line_style": "-"
        },
        "Stimulus_r25_fixationCrossBase_1sec_mirror": {
            "condition_name": "r25",
            "color": "pink",
            "line_style": "--"
        },
        "Stimulus_r75_fixationCrossBase_1sec_mirror": {
            "condition_name": "r75",
            "color": "pink",
            "line_style": "-"
        },

    }

    # Save the dictionary to a file
    with open('plotting_parameters.json', 'w') as file:
        json.dump(plotting_parameters, file, indent=4)


def plot_significance(ax, times, sig_effects, y_offset=0.1):
    """
    Plot significance bars for the effects on top of the existing axes, adjusted for time windows.

    Parameters:
    - ax: The matplotlib Axes object to plot on.
    - times: Array of time points for the x-axis.
    - sig_effects: Dictionary with time windows as keys and lists of tuples (effect, p-value) as values.
    - y_offset: The vertical offset between different time window significance bars.
    """
    y_pos_base = ax.get_ylim()[1]  # Get the top y-axis limit to place significance bars

    time_windows = {
        'FirstHalfSecond': (0, 0.5),
        'SecondHalfSecond': (0.5, 1),
        'FullSecond': (0, 1)
    }

    window_offsets = {window: 0 for window in time_windows}  # Initialize offsets for each time window

    # Sort time windows to ensure 'FullSecond' bars are plotted last (on top)
    for time_window, effects in sorted(sig_effects.items(), key=lambda x: x[0] == 'FullSecond'):
        base_y_pos = y_pos_base + y_offset * list(time_windows).index(time_window)
        for effect, p_value in effects:
            start_time, end_time = time_windows[time_window]
            # Adjust y_pos based on how many bars have already been plotted in this window
            y_pos = base_y_pos + y_offset * window_offsets[time_window]

            # Update the color selection logic as per your requirement
            color = 'black'  # Default color for unmatched conditions
                        
            if 'congruency' in effect:
                color = 'red'
            elif 'congruencyProportion' in effect:
                color = 'green'
            elif 'switchType' in effect:
                color = 'blue'
            elif 'switchProportion' in effect:
                color = 'yellow'
            elif 'congruency:congruencyProportion' in effect:
                color = 'purple'
            elif 'switchType:switchProportion' in effect:
                color = 'yellowgreen'
            elif 'congruency:switchType' in effect:
                color = 'black'

            num_asterisks = '*' * (1 if p_value < 0.05 else 2 if p_value < 0.01 else 3)
            ax.plot([start_time, end_time], [y_pos, y_pos], color=color, lw=4)
            ax.text((start_time + end_time) / 2, y_pos, num_asterisks, ha='center', va='bottom', color=color)

            window_offsets[time_window] += 1  # Increment the offset for this time window



def map_block_type(row):
    '''
    maps blockType from behavioral csv to congruencyProportion and switchProportion
    '''
    if row['blockType'] == 'A':
        return pd.Series(['25%', '25%'])
    elif row['blockType'] == 'B':
        return pd.Series(['25%', '75%'])
    elif row['blockType'] == 'C':
        return pd.Series(['75%', '25%'])
    elif row['blockType'] == 'D':
        return pd.Series(['75%', '75%'])
    else:
        return pd.Series([None, None])


def get_sig_chans(sub, task, epochs_root_file, LAB_root=None):
    """
    Retrieves the significant channels for a given subject and task from a stored JSON file.
    
    Parameters:
        sub (str): Subject ID for which significant channels are retrieved.
        task (str): The specific task for which data is being processed.
        epochs_root_file (str): The root name for the epochs that the sig chans were defined from.
        LAB_root (str, optional): The root directory where the data is stored. If None, determines the path based on the OS.

    Returns:
        dict: A dictionary containing significant channels loaded from the JSON file.
    """
    # Determine LAB_root based on the operating system
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        LAB_root = os.path.join(HOME, "Box", "CoganLab") if os.name == 'nt' else os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

    # Get data layout
    layout = get_data(task, root=LAB_root)
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)

    stim_filename = os.path.join(save_dir, f'sig_chans_{sub}_{epochs_root_file}.json')
    stim_sig_chans = load_sig_chans(stim_filename)
    return stim_sig_chans

def get_sig_chans_per_subject(subjects, epochs_root_file, task='GlobalLocal', LAB_root=None):
    """
    Retrieves significant channels for a list of subjects for a specified task.
    
    Parameters:
        subjects (list of str): List of subject IDs to process.
        task (str, optional): The specific task for which data is being processed. Defaults to 'GlobalLocal'.
        LAB_root (str, optional): The root directory where the data is stored. If None, determines the path based on the OS.

    Returns:
        dict: A dictionary where keys are subject IDs and values are dictionaries of significant channels for each subject.
    """
    # Initialize an empty dictionary to store significant channels per subject
    sig_chans_per_subject = {}

    # Populate the dictionary using get_sig_chans for each subject
    for sub in subjects:
        sig_chans_per_subject[sub] = get_sig_chans(sub, task, epochs_root_file, LAB_root)

    return sig_chans_per_subject


def filter_electrodes_by_roi(subjects_electrodes_dict, sig_chans_per_subject, roi_list):
    """
    Filters electrodes based on specified ROIs and returns significant electrodes for each subject.

    Args:
    subjects_electrodes_dict (dict): A dictionary with subjects as keys and electrode-to-ROI mappings as values.
    sig_chans_per_subject (dict): A dictionary with subjects as keys and lists of significant channels as values.
    roi_list (list): A list of ROIs to filter electrodes.

    Returns:
    dict: A dictionary with subjects as keys and lists of significant electrodes in specified ROIs as values.
    """
    filtered_electrodes_per_subject = {}

    for sub, electrodes_dict in subjects_electrodes_dict.items():
        filtered = {key: value for key, value in electrodes_dict['filtROI_dict'].items() 
                    if any(roi in key for roi in roi_list)}

        # Aggregate electrodes into a list for each subject
        filtered_electrodes = []
        for electrodes in filtered.values():
            filtered_electrodes.extend(electrodes)

        filtered_electrodes_per_subject[sub] = filtered_electrodes
        print(f'For subject {sub}, {", ".join(roi_list)} electrodes are: {filtered_electrodes}')

    # Now filter for significant electrodes
    sig_filtered_electrodes_per_subject = {}

    for sub, filtered_electrodes in filtered_electrodes_per_subject.items():
        # Retrieve the list of significant channels for the subject
        sig_chans = sig_chans_per_subject.get(sub, [])

        # Find the intersection of filtered electrodes and significant channels for the subject
        sig_filtered_electrodes = [elec for elec in filtered_electrodes if elec in sig_chans]

        # Store the significant filtered electrodes for the subject
        sig_filtered_electrodes_per_subject[sub] = sig_filtered_electrodes
        print(f"Subject {sub} significant {', '.join(roi_list)} electrodes: {sig_filtered_electrodes}")

    return filtered_electrodes_per_subject, sig_filtered_electrodes_per_subject


def make_sig_electrodes_per_subject_and_roi_dict(rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject):
    """
    Processes electrodes by ROI and filters significant electrodes.

    Parameters:
    - rois_dict: A dictionary mapping each region of interest (ROI) to a list of brain regions.
    - subjects_electrodestoROIs_dict: A dictionary mapping subjects to their electrode-to-ROI assignments.
    - sig_chans_per_subject: A dictionary indicating significant channels per subject.

    Returns:
    - A tuple of two dictionaries:
      1. electrodes_per_subject_roi: Electrodes per subject for each ROI.
      2. sig_electrodes_per_subject_roi: Significant electrodes per subject for each ROI.
    """
    electrodes_per_subject_roi = {}
    sig_electrodes_per_subject_roi = {}

    for roi_name, roi_regions in rois_dict.items():
        # Apply the filter_electrodes_by_roi function for each set of ROI regions
        electrodes_per_subject, sig_electrodes_per_subject = filter_electrodes_by_roi(
            subjects_electrodestoROIs_dict, sig_chans_per_subject, roi_regions)
        
        # Store the results in the respective dictionaries
        electrodes_per_subject_roi[roi_name] = electrodes_per_subject
        sig_electrodes_per_subject_roi[roi_name] = sig_electrodes_per_subject

    return electrodes_per_subject_roi, sig_electrodes_per_subject_roi



def calculate_total_electrodes(sig_electrodes_per_subject_roi, electrodes_per_subject_roi):
    """
    Calculates the total number of significant and total electrodes for each ROI across all subjects.

    Parameters:
    - sig_electrodes_per_subject_roi: A dictionary containing significant electrodes per subject for each ROI.
    - electrodes_per_subject_roi: A dictionary containfing all electrodes per subject for each ROI.

    Returns:
    - A dictionary containing the counts of significant and total electrodes for each ROI.
    """
    total_electrodes_info = {}

    for roi in sig_electrodes_per_subject_roi:
        # Calculate total significant electrodes for the current ROI
        total_sig_entries = sum(len(sig_electrodes_per_subject_roi[roi][sub]) for sub in sig_electrodes_per_subject_roi[roi])
        # Calculate total electrodes for the current ROI
        total_entries = sum(len(electrodes_per_subject_roi[roi][sub]) for sub in electrodes_per_subject_roi[roi])

        # Store the results in the dictionary
        total_electrodes_info[roi] = {
            'total_significant_electrodes': total_sig_entries,
            'total_electrodes': total_entries
        }

    return total_electrodes_info


def plot_HG_and_stats(sub, task, output_name, events=None, times=(-1, 1.5),
                      base_times=(-0.5, 0), LAB_root=None, channels=None):
    """
    Plot high gamma (HG) and statistics for a given subject and task using specified event.

    Parameters:
    - sub (str): The subject identifier.
    - task (str): The task identifier.
    - output_name (str): The name for the output files.
    - events (list of strings, optional): Event names to process. Defaults to None.
    - times (tuple, optional): A tuple indicating the start and end times for processing. Defaults to (-1, 1.5).
    - base_times (tuple, optional): A tuple indicating the start and end base times for processing. Defaults to (-0.5, 0).
    - LAB_root (str, optional): The root directory for the lab. Will be determined based on OS if not provided. Defaults to None.
    - channels (list of strings, optional): The channels to plot and get stats for. Default is all channels.
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

    # Create a baseline EpochsTFR using the stimulus event

    adjusted_base_times = [base_times[0] - 0.5, base_times[1] + 0.5]
    trials = trial_ieeg(good, "Stimulus", adjusted_base_times, preload=True)
    outliers_to_nan(trials, outliers=10)
    HG_base = gamma.extract(trials, copy=False, n_jobs=1)
    crop_pad(HG_base, "0.5s")

    all_epochs_list = []

    for event in events:
    # Epoching and HG extraction for each specified event. Then concatenate all trials epochs objects together (do Stimulus/c25 and Stimulus/c75 for example, and combine to get all congruent trials)
        times_adj = [times[0] - 0.5, times[1] + 0.5]
        trials = trial_ieeg(good, event, times_adj, preload=True,
                            reject_by_annotation=False)
        all_epochs_list.append(trials)

    # Concatenate all trials
    all_trials = mne.concatenate_epochs(all_epochs_list)

    outliers_to_nan(all_trials, outliers=10)
    HG_ev1 = gamma.extract(all_trials, copy=True, n_jobs=1)
    print("HG_ev1 before crop_pad: ", HG_ev1.tmin, HG_ev1.tmax)
    crop_pad(HG_ev1, "0.5s")
    print("HG_ev1 after crop_pad: ", HG_ev1.tmin, HG_ev1.tmax)

    HG_ev1_rescaled = rescale(HG_ev1, HG_base, copy=True, mode='zscore')

    HG_base.decimate(2)
    HG_ev1.decimate(2)

    HG_ev1_avgOverTime = np.nanmean(HG_ev1.get_data(), axis=2)
    HG_ev1_rescaled_avgOverTime = np.nanmean(HG_ev1_rescaled.get_data(), axis=2)

    HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0)) #axis=0 should be set for actually running this, the axis=2 is just for drift testing.
    HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

    HG_ev1_evoke_stderr = HG_ev1.standard_error()
    HG_ev1_evoke_rescaled_stderr = HG_ev1_rescaled.standard_error()

    # if event == "Stimulus":
    #     print('plotting stimulus')
    #     fig = HG_ev1_evoke_rescaled.plot(unit=False, scalings=dict(sEEG=1)) #this line is not finishing...
    #     print('plotted')
    #     # for ax in fig.axes:
    #     #     ax.axvline(x=avg_RT, color='r', linestyle='--')
    #     print('about to save')
    #     fig.savefig(save_dir + '_HG_ev1_Stimulus_zscore.png')
    #     print('saved')
    # else:
    #     print('about to plot if not stimulus')
    #     fig = HG_ev1_evoke_rescaled.plot(unit=False, scalings=dict(sEEG=1))
    #     print('plotted non stimulus')
    #     fig.savefig(save_dir + f'_HG_ev1_{output_name}_zscore.png')

    # Save HG_ev1
    HG_ev1.save(f'{save_dir}/{sub}_{output_name}_HG_ev1-epo.fif', overwrite=True)

    # Save HG_base
    HG_base.save(f'{save_dir}/{sub}_{output_name}_HG_base-epo.fif', overwrite=True)

    # Save HG_ev1_rescaled
    HG_ev1_rescaled.save(f'{save_dir}/{sub}_{output_name}_HG_ev1_rescaled-epo.fif', overwrite=True)

    # Save HG_ev1_evoke
    HG_ev1_evoke.save(f'{save_dir}/{sub}_{output_name}_HG_ev1_evoke-epo.fif', overwrite=True)

    # Save HG_ev1_evoke_rescaled
    HG_ev1_evoke_rescaled.save(f'{save_dir}/{sub}_{output_name}_HG_ev1_evoke_rescaled-epo.fif', overwrite=True)

    ###
    print(f"Shape of HG_ev1._data: {HG_ev1._data.shape}")
    print(f"Shape of HG_base._data: {HG_base._data.shape}")

    sig1 = HG_ev1._data
    sig2 = HG_base._data
    sig3 = make_data_same(sig2, (sig2.shape[0],sig2.shape[1],sig2.shape[2]+1)) # originally we want to make the baseline the same shape as the signal. We still want to do that, but first, we'll make it bigger to reflect it once, then back to normal to randomly offset it and remove fixation cross effects.
    sig4 = make_data_same(sig3, sig2.shape) #here we do the random offset, we know that sig3 is bigger than sig1 by 1 in the time dimension so it will get randomly sliced.
    sig5 = make_data_same(sig4, sig1.shape) #and now sig4 should be sig2 but with a random offset, and we can then set it equal to sig1's shape like the original plan.
    print(f"Shape of sig1: {sig1.shape}")
    print(f"Shape of sig2: {sig2.shape}")
    print(f"Shape of sig3: {sig3.shape}")
    print(f"Shape of sig4: {sig4.shape}")
    print(f"Shape of sig5: {sig5.shape}")

    sig2 = sig5

    mat = time_perm_cluster(sig1, sig2, 0.05, n_jobs=6, ignore_adjacency=1)
    fig = plt.figure()
    plt.imshow(mat, aspect='auto')
    fig.savefig(save_dir + f'_{output_name}_stats.png', dpi=300)

    channels = good.ch_names

    #save channels with their indices 
    save_channels_to_file(channels, sub, task, save_dir)

    # save significant channels to a json
    save_sig_chans(f'{output_name}', mat, channels, sub, save_dir)

    # sig_chans_filename = os.path.join(save_dir, f'sig_chans_{sub}_{output_name}.json')
    # sig_chans = load_sig_chans(sig_chans_filename)

    # Assuming `mat` is your array and `save_dir` is the directory where you want to save it
    mat_save_path = os.path.join(save_dir, f'{output_name}_mat.npy')

    # Save the mat array
    np.save(mat_save_path, mat)


def check_sampling_rates(subjects_mne_objects, expected_sampling_rate=256):
    # This dictionary will store subjects with their sampling rates
    subject_sampling_rates = {}

    # Iterate through each subject and their corresponding data
    for subject, data in subjects_mne_objects.items():
        # Get the first epochs object from the dictionary
        if data:
            first_condition = list(data.keys())[0]
            mne_objects = data[first_condition]
            first_object = list(mne_objects.keys())[0]
            first_epochs = data[first_condition][first_object]
            actual_sampling_rate = first_epochs.info['sfreq']
            
            # Store the sampling rate in the dictionary
            subject_sampling_rates[subject] = actual_sampling_rate
    
    # Print the results
    for subject, rate in subject_sampling_rates.items():
        if rate != expected_sampling_rate:
            print(f"Subject {subject} has a different sampling rate: {rate} Hz.")
        else:
            print(f"Subject {subject} has the expected sampling rate: {rate} Hz.")
    
    return subject_sampling_rates

# Function to read and print the trial outlier counts from a pickle file
def read_trial_outlier_counts(subject, root_dir):
    pickle_filepath = os.path.join(root_dir, subject, f'{subject}_trial_outlier_counts.pkl')
    with open(pickle_filepath, 'rb') as pickle_file:
        outlier_counts = pickle.load(pickle_file)
    return outlier_counts


def prepare_data_for_temporal_dataset(subjects_mne_objects, condition_names, rois, subjects, sig_electrodes_per_subject_roi):
    # returns dat
    dat = {}
    overall_electrode_mapping = []
    electrode_mapping_per_roi = {roi: [] for roi in rois}  # Reinitialize for each processing run
    print('subjects: ', subjects)
    for roi in rois:
        dat[roi] = {}  # make a dict for each roi
        dat[roi]['channel_names'] = []  # initialize a list to hold channel names
        dat[roi]['channel_rois'] = [] # initialize a list to hold what roi each channel is a part of
        dat[roi]['cond_names'] = {}  # initialize dict where keys are condition names and values are integer indices
        dat[roi]['cond_idx'] = np.array([], dtype=int)  # initialize an empty 1D array for condition indices for each trial
        dat[roi]['sub_idx'] = np.array([], dtype=int) # initialize an empty 1D array for subject for each trial
        dat[roi]['times'] = np.array([])  # initialize an empty 1D array for time points

        # Determine all unique channels across subjects for this ROI, maintaining order
        all_channels = []
        for sub in subjects:
            sig_electrodes = sig_electrodes_per_subject_roi[roi].get(sub, [])
            sub_channel_names = [sub + '-' + sig_electrode for sig_electrode in sig_electrodes]
            for chan in sub_channel_names:
                if chan not in all_channels:
                    all_channels.append(chan)
                    dat[roi]['channel_rois'].append(roi) # append roi for each channel
        dat[roi]['channel_names'] = all_channels
        num_channels = len(all_channels)
        print('num channels: ', num_channels)

        # Initialize the data array with the number of trials and total channels
        dat[roi]['data'] = np.empty((0, num_channels, 0))  # initialize an empty 3D array for trials x channels x time points
        total_roi_trials = 0
        for sub in subjects:
            total_sub_trials = 0  # Initialize counter for total trials across all conditions
            sig_electrodes = sig_electrodes_per_subject_roi[roi].get(sub, [])
            sub_channel_names = [sub + '-' + sig_electrode for sig_electrode in sig_electrodes]
            if not sig_electrodes:
                continue

            cond_idx = 0  # the example uses indexing from 1, but let's start from 0 because python
            for condition_name in condition_names:
                print(f'Processing {sub} for {condition_name} in {roi}')
                epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(sig_electrodes)
                dat[roi]['cond_names'][condition_name] = cond_idx

                epochs_data = epochs.get_data(copy=True)
                num_trials, num_sub_channels, num_timepoints = epochs_data.shape

                print(f'Number of trials for {sub} in {condition_name}: {num_trials}')
                total_sub_trials += num_trials
                total_roi_trials += num_trials

                # Initialize data array time dimension if it is empty
                if dat[roi]['data'].shape[2] == 0:
                    dat[roi]['data'] = np.empty((0, num_channels, num_timepoints))
                    dat[roi]['times'] = epochs.times

                # Create an array filled with NaNs for the current subject's data
                sub_data = np.full((num_trials, num_channels, num_timepoints), np.nan)

                # Find the indices for the subject's channels in the total list of channels
                channel_indices = [all_channels.index(chan) for chan in sub_channel_names]
                print('sub: ', sub)
                print("channel indices: ", channel_indices)
                
                # Place the subject's data in the correct indices
                sub_data[:, channel_indices, :] = epochs_data

                # Concatenate the new data along the first axis (trials)
                dat[roi]['data'] = np.concatenate((dat[roi]['data'], sub_data), axis=0)

                # Extend the cond_idx array
                dat[roi]['cond_idx'] = np.concatenate((dat[roi]['cond_idx'], np.full(num_trials, cond_idx)))

                # extend the sub_idx array by this number of trials with this subject
                dat[roi]['sub_idx'] = np.concatenate((dat[roi]['sub_idx'], np.full(num_trials, sub)))
                cond_idx += 1  # increment cond_idx

            print(f'Total number of trials for {sub} across all conditions: {total_sub_trials}')
        print(f'total number of trials in {roi} is {total_roi_trials}')
    return dat

def get_good_data(sub, layout):
    '''
    Load and further preprocess the line-noise filtered EEG data for a given subject. 
    AKA drop bad channels, set re-referencing scheme

    Parameters:
    -----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.

    Returns:
    --------
    good : mne.io.Raw
        The preprocessed raw EEG data.

    Examples:
    ---------
    >>> sub = 'sub-01'
    >>> good = get_good_data(sub, layout)
    >>> isinstance(good, mne.io.Raw)
    True
    '''
    # Load the data
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                           extension='.edf', desc='clean', preload=False)  # Get line-noise filtered data
    print(filt)

    # Crop raw data to minimize processing time
    good = crop_empty_data(filt)

    # Mark and drop bad channels
    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
    good.load_data()

    # Set EEG reference
    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    # # Plot the data for inspection
    # good.plot()

    return good