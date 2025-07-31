import sys
import os

print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

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

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import mne
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from ieeg.calc.mat import LabeledArray

def get_max_trials_per_condition(
    subjects_mne_objects, condition_names, subjects,
    electrodes_per_subject_roi, roi, obs_axs
):
    """
    Find the maximum number of trials per condition across all subjects for a given ROI,
    and identify which subject(s) have that maximum number of trials.

    Args:
        subjects_mne_objects (dict): Dictionary of MNE epoch objects, structured as
                                     {subject_id: {condition_name: mne.Epochs}}.
        condition_names (list): List of strings representing the condition names.
        subjects (list): List of subject identifiers.
        electrodes_per_subject_roi (dict): Dictionary mapping ROIs to subjects
                                               and their corresponding significant electrodes.
                                               Structure: {roi_name: {subject_id: [electrode_list]}}.
        roi (str): The Region of Interest (ROI) name to process.
        obs_axs (int): The axis index in the MNE epochs data array that corresponds to trials.

    Returns:
        tuple: A tuple containing two dictionaries:
            - max_trials_per_condition (dict): {condition_name: max_trial_count}
            - max_trials_subject_per_condition (dict): {condition_name: [subject_id_list]}
    """
    max_trials_per_condition = {condition: 0 for condition in condition_names}
    max_trials_subject_per_condition = {condition: [] for condition in condition_names}

    for sub in subjects:
        electrodes = electrodes_per_subject_roi.get(roi, {}).get(sub, [])
        if not electrodes:
            continue
        for condition_name in condition_names:
            # Check if the subject has data for this condition
            if condition_name not in subjects_mne_objects[sub]:
                continue
            epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(electrodes)
            epochs_data = epochs.get_data(copy=True)
            n_trials = epochs_data.shape[obs_axs]
            if n_trials > max_trials_per_condition[condition_name]:
                max_trials_per_condition[condition_name] = n_trials
                max_trials_subject_per_condition[condition_name] = [sub]
            elif n_trials == max_trials_per_condition[condition_name]:
                max_trials_subject_per_condition[condition_name].append(sub)
    return max_trials_per_condition, max_trials_subject_per_condition

def make_subject_labeled_array(
    sub, subjects_mne_objects, condition_names, electrodes_per_subject_roi,
    roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, frequency_axs=None, rng=None
):
    """
    Process data for a single subject in a given ROI to create a LabeledArray.

    This function performs the following steps:
    1. Retrieves specified electrodes in the specified ROI for the subject.
    2. For each condition:
        a. Extracts epoch data for specified electrodes.
        b. Randomizes trial order.
        c. Pads data with NaNs to match the `max_trials_per_condition` for that condition.
    3. Collects processed data for all conditions into a dictionary.
    4. Creates a LabeledArray from this dictionary, assigning channel and time labels, and potentially frequency if there is a frequency dimension (for tfr objects).
       The LabeledArray will have a new leading dimension for conditions.

    Args:
        sub (str): Subject identifier.
        subjects_mne_objects (dict): {subject_id: {condition_name: mne.Epochs}}.
        condition_names (list): List of condition names.
        electrodes_per_subject_roi (dict): {roi_name: {subject_id: [electrode_list]}}.
        roi (str): The ROI name to process.
        max_trials_per_condition (dict): {condition_name: max_trial_count} for padding.
        obs_axs (int): Original trials axis index in epoch data (e.g., 0 for (trials, chans, time)).
        chans_axs (int): Original channels axis index in epoch data (e.g., 1 for (trials, chans, time)).
        time_axs (int): Original time axis index in epoch data (e.g., 2 for (trials, chans, time)).
        frequency_axs (int): Original frequency axis index in epoch data (e.g., 3 for (trials, chans, time, frequency)). Use for tfr objects.
        rng (np.random.RandomState): NumPy random number generator instance for shuffling.

    Returns:
        LabeledArray or None: A LabeledArray for the subject with dimensions
                              (Conditions, Trials, Channels, Timepoints) OR (Conditions, Trials, Channels, Timepoints, Frequencies)
                              Returns None if the subject has no significant electrodes for the ROI.
    """
    electrodes = electrodes_per_subject_roi.get(roi, {}).get(sub, [])
    if not electrodes:
        return None

    subject_nested_dict = {}

    # Get channel names for this subject's ROI
    sub_channel_names = [f"{sub}-{electrode}" for electrode in electrodes]

    # Loop through each condition
    for condition_name in condition_names:
        # Extract the epoch data for the current condition and subject
        epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(electrodes)
        epochs_data = epochs.get_data(copy=True)

        # Randomize the trial order
        n_trials = epochs_data.shape[obs_axs]
        print(f'in roi {roi}, subject {sub} has {n_trials} trials for condition {condition_name}')
        trial_indices = np.arange(n_trials)
        rng.shuffle(trial_indices)
        epochs_data = epochs_data.take(trial_indices, axis=obs_axs)

        # Get the target number of trials for padding
        max_trials = max_trials_per_condition[condition_name]

        # Pad with NaNs if necessary
        if n_trials < max_trials:
            padded_shape = list(epochs_data.shape)
            padded_shape[obs_axs] = max_trials
            padded_data = np.full(padded_shape, np.nan)
            indexer = [slice(None)] * epochs_data.ndim
            indexer[obs_axs] = slice(0, n_trials)
            padded_data[tuple(indexer)] = epochs_data
        else:
            padded_data = epochs_data

        subject_nested_dict[condition_name] = padded_data

    # Get time labels
    times = epochs.times
    str_times = [str(time) for time in times]
    np_array_str_times = np.array(str_times)

    # Create a LabeledArray for the subject
    # TODO: add freqs axs as optional input here
    subject_labeled_array = create_subject_labeled_array_from_dict(
        subject_nested_dict, sub_channel_names, np_array_str_times, chans_axs, time_axs
    )

    # Print the shape and time axis labels
    print(f"Subject {sub}, ROI {roi}, LabeledArray shape: {subject_labeled_array.shape}")
    time_axis_size = subject_labeled_array.shape[time_axs+1] # Adjusted time axis index - uhhh what is this for..?

    return subject_labeled_array

def create_subject_labeled_array_from_dict(
    subject_nested_dict, sub_channel_names, np_array_str_times, chans_axs, time_axs
):
    """
    Create a LabeledArray for a subject from a dictionary of condition data.

    The input dictionary `subject_nested_dict` is expected to have condition names
    as keys and NumPy arrays (trials, channels, timepoints) OR (trials, channels, timepoints, frequencies) as values.
    This function creates a LabeledArray where the first dimension is 'conditions'.
    Labels for channel and time axes are assigned by adjusting their original
    axis indices (`chans_axs`, `time_axs`) by +1 to account for the new 'conditions' axis. 
    If the data has a frequency dimension, the frequency axis is also adjusted by +1.

    Args:
        subject_nested_dict (dict): {condition_name: np.ndarray(trials, channels, timepoints) OR (trials, channels, timepoints, frequencies)}.
        sub_channel_names (list): List of strings for channel labels.
        np_array_str_times (np.ndarray): Array of strings for time labels.
        chans_axs (int): Original channels axis index in the per-condition data arrays
                         (e.g., 1 if data is trials, channels, time).
        time_axs (int): Original time axis index in the per-condition data arrays
                        (e.g., 2 if data is trials, channels, time).
        frequency_axs (int): Original frequency axis index in the per-condition data arrays
                             (e.g., 3 if data is trials, channels, time, frequency).

    Returns:
        LabeledArray: A LabeledArray object with dimensions (Conditions, Trials, Channels, Timepoints) OR (Conditions, Trials, Channels, Timepoints, Frequencies)
                      and corresponding labels.
    """
    subject_labeled_array = LabeledArray.from_dict(subject_nested_dict)
    # Adjust axes indices due to the added conditions axis
    subject_labeled_array.labels[chans_axs + 1].values = sub_channel_names  # Channels axis
    subject_labeled_array.labels[time_axs + 1].values = np_array_str_times  # Time axis
    if frequency_axs is not None:
        subject_labeled_array.labels[frequency_axs + 1].values = np_array_str_freqs  # Frequency axis
    return subject_labeled_array

def concatenate_subject_labeled_arrays(
    roi_labeled_array, subject_labeled_array, concatenation_axis
):
    """
    Concatenate a subject's LabeledArray to an ROI's LabeledArray along the specified axis.

    The concatenation axis index is adjusted by +1 to account for the
    leading 'conditions' dimension in the LabeledArrays. If `roi_labeled_array`
    is None (i.e., this is the first subject for the ROI), the `subject_labeled_array`
    is returned directly.

    Args:
        roi_labeled_array (LabeledArray or None): The LabeledArray for the ROI accumulated so far.
                                                  Can be None if this is the first subject.
        subject_labeled_array (LabeledArray): The LabeledArray for the current subject.
        concatenation_axis (int): The axis along which to concatenate the subject's LabeledArray
                                  to the ROI's LabeledArray. This will be adjusted to `concatenation_axis + 1`
                                  for concatenation.

    Returns:
        LabeledArray: The updated LabeledArray for the ROI, with the new subject's
                      data concatenated along the (adjusted) concatenation axis.
    """
    concatenation_axis += 1  # Adjusted concatenation axis index
    if roi_labeled_array is None:
        return subject_labeled_array
    else:
        return roi_labeled_array.concatenate(subject_labeled_array, axis=concatenation_axis)

def put_data_in_labeled_array_per_roi_subject(
    subjects_mne_objects, condition_names, rois, subjects,
    electrodes_per_subject_roi, obs_axs=0, chans_axs=1, time_axs=2,
    concatenation_axis=1, frequency_axs=None,
    random_state=None
):
    """
    Organize the MNE data into separate LabeledArrays for each ROI and subject,
    with randomized trial ordering within each subject before concatenation.
    Concatenates subject data along the channels axis.

    Parameters:
    - subjects_mne_objects: Dictionary of MNE objects, structured as {subject: {condition: MNE epoch objects}}
    - condition_names: List of condition names.
    - rois: List of region of interest (ROI) names.
    - subjects: List of subjects.
    - electrodes_per_subject_roi: Dictionary mapping ROIs to subjects and their corresponding electrodes.
    - obs_axs: The trials dimension (ignoring the conditions dimension for now)
    - chans_axs: The channels dimension
    - time_axs: The time dimension
    - concatenation_axis: The axis along which to concatenate the subject's LabeledArray
                          to the ROI's LabeledArray. This will be adjusted to `concatenation_axis + 1`
                          for concatenation.
    - frequency_axs: The frequency dimension
    - random_state: Optional; an integer seed, NumPy RandomState, or None for random shuffling.

    Returns:
    - roi_labeled_arrays: Dictionary of LabeledArrays for each ROI.
                          Each LabeledArray has dimensions: [Conditions, Trials, Channels, Timepoints]
    """
    # Set up the random state
    rng = np.random.RandomState(random_state)
    roi_labeled_arrays = {}

    # Loop through each ROI
    for roi in rois:
        # First pass: Find the max number of trials per condition across all subjects
        max_trials_per_condition, max_trials_subject_per_condition = get_max_trials_per_condition(
            subjects_mne_objects, condition_names, subjects,
            electrodes_per_subject_roi, roi, obs_axs
        )

        # Print out the subjects with maximum trials
        print(f"ROI '{roi}': Maximum trials per condition:")
        for condition_name in condition_names:
            max_trials = max_trials_per_condition[condition_name]
            subjects_with_max_trials = max_trials_subject_per_condition[condition_name]
            print(f"  Condition '{condition_name}': {max_trials} trials from subjects {subjects_with_max_trials}")
            
        # Initialize the ROI LabeledArray
        roi_labeled_array = None

        # Second pass: Process each subject's data
        for sub in subjects:
            subject_labeled_array = make_subject_labeled_array(
                sub, subjects_mne_objects, condition_names, electrodes_per_subject_roi,
                roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, rng, frequency_axs
            )
            if subject_labeled_array is None:
                continue  # Skip if subject has no data for this ROI

            # Concatenate subject's data into the ROI LabeledArray
            roi_labeled_array = concatenate_subject_labeled_arrays(
                roi_labeled_array, subject_labeled_array, chans_axs
            )

        # Add the concatenated LabeledArray to the ROI dictionary
        if roi_labeled_array is not None:
            roi_labeled_arrays[roi] = roi_labeled_array

    return roi_labeled_arrays

def remove_nans_from_labeled_array(labeled_array, obs_axs=0, chans_axs=1, time_axs=2, frequency_axs=None):
    """
    Remove trials that have NaN values from a LabeledArray and identify conditions with no valid trials.

    Parameters:
    - labeled_array: A LabeledArray with conditions, trials, channels, and timepoints dimensions.
    - obs_axs: The trials dimension 
    - chans_axs: The channels dimension
    - time_axs: The time dimension
    - frequency_axs: The frequency dimension

    Returns:
    - labeled_array_no_nans: A LabeledArray with only trials that have no NaN values.
    - conditions_with_no_valid_trials: List of condition names with no valid trials after NaN removal.
    """

    # Initialize a dictionary to store data without NaNs for each condition
    reshaped_data_dict = {}
    # Initialize a list to keep track of conditions with no valid trials
    conditions_with_no_valid_trials = []

    # Extract the condition labels (which are in labeled_array.labels[0])
    condition_names = labeled_array.labels[0]  # Ensure we get the condition names

    # Loop over each condition
    for condition_name in condition_names:
        # Extract the data for the current condition
        condition_data = labeled_array[condition_name]  # Shape: (Trials, Channels, Timepoints)

        # Find the indices of trials that do not contain NaNs
        # Reduce over channel, time, and frequency (if it exists) axes to check if any NaN exists in a trial
        if frequency_axs is not None:
            valid_trial_indices = ~np.isnan(condition_data).any(axis=(chans_axs, time_axs, frequency_axs))
            # valid_trial_indices is a boolean array of shape (Trials,)
            # Select only the valid trials
            condition_data_clean = condition_data[valid_trial_indices, :, :, :]
        else:
            valid_trial_indices = ~np.isnan(condition_data).any(axis=(chans_axs, time_axs))
            condition_data_clean = condition_data[valid_trial_indices, :, :]

        # Check if there are valid trials
        if condition_data_clean.shape[obs_axs] > 0:
            # Store the processed data for this condition
            reshaped_data_dict[condition_name] = condition_data_clean
        else:
            print(f"No valid trials for condition '{condition_name}' after removing NaNs.")
            conditions_with_no_valid_trials.append(condition_name)

    # Optionally, print the conditions with no valid trials
    if conditions_with_no_valid_trials:
        print("Conditions with no valid trials after NaN removal:", conditions_with_no_valid_trials)

    # Proceed to create the LabeledArray if there are any valid conditions
    if len(reshaped_data_dict) == 0:
        raise ValueError("All conditions have no valid trials after removing NaNs.")

    # Create a new LabeledArray from the reshaped data dictionary
    labeled_array_no_nans = LabeledArray.from_dict(reshaped_data_dict)

    return labeled_array_no_nans, conditions_with_no_valid_trials


def remove_nans_from_all_roi_labeled_arrays(roi_labeled_arrays, obs_axs=0, chans_axs=1, time_axs=2, frequency_axs=None):
    """
    Loop through all ROIs and apply the NaN removal function to each LabeledArray.

    Parameters:
    - roi_labeled_arrays: Dictionary of LabeledArrays for each ROI.
    - obs_axs: The trials dimension.
    - chans_axs: The channels dimension.
    - time_axs: The time dimension.
    - frequency_axs: The frequency dimension

    Returns:
    - roi_labeled_arrays_no_nans: Dictionary where keys are ROIs and values are LabeledArrays with NaNs removed.
    - conditions_with_no_valid_trials_per_roi: Dictionary where keys are ROIs and values are lists of condition names with no valid trials.
    """
    roi_labeled_arrays_no_nans = {}
    conditions_with_no_valid_trials_per_roi = {}

    # Loop through each ROI in the dictionary
    for roi, labeled_array in roi_labeled_arrays.items():
        # Apply the NaN removal function to the current labeled array
        labeled_array_no_nans, conditions_with_no_valid_trials = remove_nans_from_labeled_array(
            labeled_array, obs_axs=obs_axs, chans_axs=chans_axs, time_axs=time_axs, frequency_axs=frequency_axs
        )

        # Store the reshaped data for this ROI
        roi_labeled_arrays_no_nans[roi] = labeled_array_no_nans

        # Store the conditions with no valid trials for this ROI
        if conditions_with_no_valid_trials:
            conditions_with_no_valid_trials_per_roi[roi] = conditions_with_no_valid_trials

    return roi_labeled_arrays_no_nans, conditions_with_no_valid_trials_per_roi

def concatenate_conditions_by_string(roi_labeled_arrays, roi, strings_to_find, obs_axs=0):
    """
    Concatenate trials across condition names that contain specific strings.
    Assign labels based on the groupings of the conditions.

    Parameters:
    - roi_labeled_arrays: Dictionary of LabeledArrays for each ROI.
    - roi: The specific ROI to process.
    - strings_to_find: List of strings or list of lists of strings to search for in condition names.
                      If a list of strings is provided, each string is treated as its own condition group.
                      If a list of lists is provided, each sublist represents a group of conditions.
    - obs_axs (int) : The trials dimension. Concatenation will happen along this axis. This is the 1st dimension (not 0th) because conditions in the labeled array is the 0th. But we will subtract 1 if not considering conditions as a dimension (looping over conditions)
    
    Returns:
    - concatenated_data: The concatenated trials by (channels, timepoints, or whatever your other dimensions are) across the matching conditions.
    - labels: A numpy array of labels (0, 1, 2, ...) corresponding to each group of conditions.
    - cats: Dictionary of {condition_name: index} for decoding.
    """
    concatenated_data = []
    labels = []
    cats = {}

    # Track current label index
    current_label = 0

    # Normalize strings_to_find so each entry is a list (whether it's a string or a list of strings)
    if isinstance(strings_to_find, list) and all(isinstance(s, str) for s in strings_to_find):
        # If it's a flat list of strings, convert each string into its own single-item list
        strings_to_find = [[s] for s in strings_to_find]

    # Iterate over each group (whether it's a single string or a list of strings)
    for string_group in strings_to_find:
        # Find condition names that match any of the strings in the current string_group
        matching_conditions = [cond for cond in roi_labeled_arrays[roi].keys() if any(s in cond for s in string_group)]

        if not matching_conditions:
            continue

        # Concatenate data for all matching conditions
        data_to_concatenate = []
        for cond in matching_conditions:
            # Extract data for the current condition
            data = roi_labeled_arrays[roi][cond]  # Shape: (trials, channels, timepoints)
            data_to_concatenate.append(data)
            
            # Update labels for the current condition group
            labels.extend([current_label] * data.shape[0])
        
        # Check if we have data to concatenate for this condition group
        if data_to_concatenate:
            concatenated_data.append(np.concatenate(data_to_concatenate, axis=obs_axs))

        # Assign current label to the condition group (based on the first string in the group for reference)
        cats[tuple(string_group)] = current_label
        current_label += 1

    # Ensure there is data to concatenate
    if not concatenated_data:
        raise ValueError(f"No matching conditions found for ROI: {roi} and strings: {strings_to_find}")

    # Concatenate all condition data along the trials axis
    concatenated_data = np.concatenate(concatenated_data, axis=obs_axs)
    
    return concatenated_data, np.array(labels), cats

def get_data_in_time_range(labeled_array, time_range, time_axs=-1):
    """
    Extract data from a LabeledArray where the time points fall within a given range, using the LabeledArray `take` function.
    
    Parameters:
    - labeled_array: The LabeledArray containing time points as labels.
    - time_range: A tuple (start_time, end_time) representing the range of time.
    - time_axs: The time dimension.
    
    Returns:
    - filtered_data: A LabeledArray containing only the data where the timepoints are within the specified range.
    """
    start_time, end_time = time_range

    # Assume that the time labels are stored in the last dimension, but can change this.
    time_points = np.array(labeled_array.labels[time_axs], dtype=float)  # convert to floats, ensure conversion to float numpy array

    # Find the indices of time points within the specified range
    time_indices = np.where((time_points >= start_time) & (time_points <= end_time))[0]

    # Use the take function to select the time indices along the time axis (axis=3)
    filtered_data = labeled_array.take(time_indices, axis=3)

    return filtered_data