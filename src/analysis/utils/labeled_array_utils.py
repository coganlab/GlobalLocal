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

def detect_data_type(subjects_data_objects):
    """
    Detect whether the input contains MNE Epochs or EpochsTFR objects.
    
    Returns:
        str: 'EpochsTFR' or 'Epochs' indicating the data type
    """
    
    # Try to find any data object to inspect
    for sub_data in subjects_data_objects.values():
        for cond_data in sub_data.values():
            # Handle the nested dictionary structure if present
            if isinstance(cond_data, dict) and 'HG_ev1_power_rescaled' in cond_data:
                test_obj = cond_data['HG_ev1_power_rescaled']
            else:
                test_obj = cond_data
            
            # Check using isinstance for the most reliable detection
            try:
                if isinstance(test_obj, mne.time_frequency.EpochsTFR):
                    return 'EpochsTFR'
                elif isinstance(test_obj, (mne.Epochs, mne.EpochsArray)):
                    return 'Epochs'
            except AttributeError:
                # In case mne.time_frequency doesn't have EpochsTFR in this version
                pass
            
            # Fallback method: check class name as string
            class_name = str(type(test_obj))
            if 'EpochsTFR' in class_name:
                return 'EpochsTFR'
            elif 'Epochs' in class_name and 'TFR' not in class_name:
                return 'Epochs'
            
            # Last resort: check data dimensions
            if hasattr(test_obj, 'get_data'):
                data = test_obj.get_data()
                if data.ndim == 4:  # TFR has 4D data
                    return 'EpochsTFR'
                elif data.ndim == 3:  # Epochs has 3D data
                    return 'Epochs'
    
    raise ValueError("Could not determine data type from input")

def get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects(
    subjects_mne_objects, condition_name, sub, electrodes
):
    """
    Get epochs data for a specific subject, condition name, and electrodes from subjects mne objects dict. Only grabs the electrodes that actually exist (i.e., when bad channels are dropped from the epochs object there will be a mismatch between the epochs object and the electrodes list)
    
    Args:
        subjects_mne_objects (dict): Dictionary of MNE epoch objects, structured as
                                     {subject_id: {condition_name: {'HG_ev1_power_rescaled': mne.Epochs}}}.
        condition_name (str): The condition name to process.
        sub (str): The subject identifier.
        electrodes (list): List of electrode names to include.

    Returns:
        mne.Epochs: The epochs data for the specified subject, condition, and ROI.
    """
    
    return subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(electrodes)

def get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects(
    subjects_tfr_objects, condition_name, sub, electrodes
):
    """
    Get epochs data for a specific subject, condition name, and electrodes from subjects tfr objects dict
    
    Args:
        subjects_tfr_objects (dict): Dictionary of MNE epochs tfr objects, structured as
                                     {subject_id: {condition_name: mne.EpochsTFR}}.
        condition_name (str): The condition name to process.
        sub (str): The subject identifier.
        electrodes (list): List of electrode names to include.

    Returns:
        mne.EpochsTFR: The epochs tfr data for the specified subject, condition, and ROI.
    """
    return subjects_tfr_objects[sub][condition_name].copy().pick(electrodes)

def get_max_trials_per_condition(
    subjects_data_objects, condition_names, subjects,
    electrodes_per_subject_roi, roi, obs_axs
):
    """
    Find the maximum number of trials per condition across all subjects for a given ROI,
    and identify which subject(s) have that maximum number of trials. Run this if input data is subjects_mne_objects

    Args:
        subjects_data_objects (dict): EITHER a dictionary of MNE epoch objects, structured as
                                     {subject_id: {condition_name: {'HG_ev1_power_rescaled': mne.Epochs}}} OR 
                                     a dictionary of MNE epochs tfr objects, structured as
                                     {subject_id: {condition_name: mne.EpochsTFR}}.
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
    
    data_type = detect_data_type(subjects_data_objects)
    for sub in subjects:
        electrodes = electrodes_per_subject_roi.get(roi, {}).get(sub, [])
        if not electrodes:
            continue
        for condition_name in condition_names:
            # Check if the subject has data for this condition
            if condition_name not in subjects_data_objects[sub]:
                continue
            if data_type == 'Epochs':
                epochs = get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects(
                    subjects_data_objects, condition_name, sub, electrodes
                )
            elif data_type == 'EpochsTFR':
                epochs = get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects(
                    subjects_data_objects, condition_name, sub, electrodes
                )
            else:
                raise ValueError("subjects_data_objects must be either Epochs or EpochsTFR from subjects_mne_objects or subjects_tfr_objects")

            epochs_data = epochs.get_data().copy()
            n_trials = epochs_data.shape[obs_axs]
            if n_trials > max_trials_per_condition[condition_name]:
                max_trials_per_condition[condition_name] = n_trials
                max_trials_subject_per_condition[condition_name] = [sub]
            elif n_trials == max_trials_per_condition[condition_name]:
                max_trials_subject_per_condition[condition_name].append(sub)
    return max_trials_per_condition, max_trials_subject_per_condition

def make_subject_labeled_array(
    sub, subjects_data_objects, condition_names, electrodes_per_subject_roi,
    roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, freq_axs=None, rng=None
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
        subjects_data_objects (dict): EITHER a dictionary of MNE epoch objects, structured as
                                     {subject_id: {condition_name: {'HG_ev1_power_rescaled': mne.Epochs}}} OR 
                                     a dictionary of MNE epochs tfr objects, structured as
                                     {subject_id: {condition_name: mne.EpochsTFR}}.
        condition_names (list): List of condition names.
        electrodes_per_subject_roi (dict): {roi_name: {subject_id: [electrode_list]}}.
        roi (str): The ROI name to process.
        max_trials_per_condition (dict): {condition_name: max_trial_count} for padding.
        obs_axs (int): Original trials axis index in epoch data (e.g., 0 for (trials, chans, time)).
        chans_axs (int): Original channels axis index in epoch data (e.g., 1 for (trials, chans, time)).
        time_axs (int): Original time axis index in epoch data (e.g., 2 for (trials, chans, time)).
        freq_axs (int): Original frequency axis index in epoch data (e.g., 3 for (trials, chans, time, frequency)). Use for tfr objects.
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

    data_type = detect_data_type(subjects_data_objects)
    
    # Loop through each condition
    for condition_name in condition_names:
        # Extract the epoch data for the current condition and subject
        if data_type == 'Epochs':
            epochs = get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects(
                subjects_data_objects, condition_name, sub, electrodes
            )
        elif data_type == 'EpochsTFR':
            epochs = get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects(
                subjects_data_objects, condition_name, sub, electrodes
            )
        else:
            raise ValueError("subjects_data_objects must be either Epochs or EpochsTFR from subjects_mne_objects or subjects_tfr_objects")
        epochs_data = epochs.get_data().copy()

        # Randomize the trial order - wait i don't think this is necessary, especially because i'm only making the roi labeled arrays once anyway.
        n_trials = epochs_data.shape[obs_axs]
        print(f'in roi {roi}, subject {sub} has {n_trials} trials for condition {condition_name}')
        # trial_indices = np.arange(n_trials)
        # rng.shuffle(trial_indices)
        # epochs_data = epochs_data.take(trial_indices, axis=obs_axs)

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

    # get frequencies
    if data_type == 'EpochsTFR':
        freqs = epochs.freqs
        str_freqs = [str(freq) for freq in freqs]
        np_array_str_freqs = np.array(str_freqs)
    else:
        # if not epochs tfr, then there are no frequencies
        freq_axs = None
        np_array_str_freqs = None
    
    # TODO: uh get channel names as labels..? i think the below would work, but test it
    # Get channel names for this subject's ROI
    sub_channel_names = [f"{sub}-{electrode}" for electrode in electrodes]

    np_array_sub_channel_names = np.array(sub_channel_names)
    
    # Create a LabeledArray for the subject
    # TODO: add freqs axs as optional input here
    subject_labeled_array = create_subject_labeled_array_from_dict(
        subject_nested_dict, np_array_sub_channel_names, np_array_str_times, np_array_str_freqs, chans_axs, time_axs, freq_axs
    )

    # Print the shape and time axis labels
    print(f"Subject {sub}, ROI {roi}, LabeledArray shape: {subject_labeled_array.shape}")
    time_axis_size = subject_labeled_array.shape[time_axs+1] # Adjusted time axis index - uhhh what is this for..?

    return subject_labeled_array

def create_subject_labeled_array_from_dict(
    subject_nested_dict, np_array_sub_channel_names, np_array_str_times, np_array_str_freqs, chans_axs, time_axs, freq_axs=None
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
        np_array_sub_channel_names (np.ndarray): Array of strings for channel labels.
        np_array_str_times (np.ndarray): Array of strings for time labels.
        np_array_str_freqs (np.ndarray): Array of strings for frequency labels.
        chans_axs (int): Original channels axis index in the per-condition data arrays
                         (e.g., 1 if data is trials, channels, time).
        time_axs (int): Original time axis index in the per-condition data arrays
                        (e.g., 2 if data is trials, channels, time).
        freq_axs (int): Original frequency axis index in the per-condition data arrays
                             (e.g., 3 if data is trials, channels, time, frequency).

    Returns:
        LabeledArray: A LabeledArray object with dimensions (Conditions, Trials, Channels, Timepoints) OR (Conditions, Trials, Channels, Timepoints, Frequencies)
                      and corresponding labels.
    """
    subject_labeled_array = LabeledArray.from_dict(subject_nested_dict)
    # Adjust axes indices due to the added conditions axis
    subject_labeled_array.labels[chans_axs + 1].values = np_array_sub_channel_names  # Channels axis
    subject_labeled_array.labels[time_axs + 1].values = np_array_str_times  # Time axis
    if freq_axs is not None:
        subject_labeled_array.labels[freq_axs + 1].values = np_array_str_freqs  # Frequency axis
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
    subjects_data_objects, condition_names, rois, subjects,
    electrodes_per_subject_roi, obs_axs=0, chans_axs=1, time_axs=2, freq_axs=None, concatenation_axis=1,
    random_state=None
):
    """
    Organize the MNE data into separate LabeledArrays for each ROI and subject,
    with randomized trial ordering within each subject before concatenation.
    Concatenates subject data along the channels axis.

    Parameters:
    - subjects_data_objects (dict): EITHER a dictionary of MNE epoch objects, structured as
                                    {subject_id: {condition_name: {'HG_ev1_power_rescaled': mne.Epochs}}} OR 
                                    a dictionary of MNE epochs tfr objects, structured as
                                    {subject_id: {condition_name: mne.EpochsTFR}}.
    - condition_names (list): List of condition names.
    - rois (list): List of region of interest (ROI) names.
    - subjects (list): List of subjects.
    - electrodes_per_subject_roi (dict): Dictionary mapping ROIs to subjects and their corresponding electrodes.
    - obs_axs (int): The trials dimension (ignoring the conditions dimension for now)
    - chans_axs (int): The channels dimension
    - time_axs (int): The time dimension
    - freq_axs (int): The frequency dimension
    - concatenation_axis (int): The axis along which to concatenate the subject's LabeledArray
                                to the ROI's LabeledArray. This will be adjusted to `concatenation_axis + 1`
                                for concatenation. By default, do channels.
    - random_state (int, RandomState, or None): Optional; an integer seed, NumPy RandomState, or None for random shuffling.

    Returns:
    - roi_labeled_arrays (dict): Dictionary of LabeledArrays for each ROI.
                                Each LabeledArray has dimensions: [Conditions, Trials, Channels, Timepoints]
    """
    # Set up the random state
    rng = np.random.RandomState(random_state)
    roi_labeled_arrays = {}
    data_type = detect_data_type(subjects_data_objects)
    
    # Loop through each ROI
    for roi in rois:
        # First pass: Find the max number of trials per condition across all subjects
        max_trials_per_condition, max_trials_subject_per_condition = get_max_trials_per_condition(
            subjects_data_objects, condition_names, subjects,
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
                sub, subjects_data_objects, condition_names, electrodes_per_subject_roi,
                roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, freq_axs, rng
            )
            if subject_labeled_array is None:
                continue  # Skip if subject has no data for this ROI

            # Concatenate subject's data into the ROI LabeledArray
            roi_labeled_array = concatenate_subject_labeled_arrays(
                roi_labeled_array, subject_labeled_array, concatenation_axis
            )

        # Add the concatenated LabeledArray to the ROI dictionary
        if roi_labeled_array is not None:
            roi_labeled_arrays[roi] = roi_labeled_array

    return roi_labeled_arrays

def remove_nans_from_labeled_array(labeled_array, obs_axs=0, chans_axs=1, time_axs=2, freq_axs=None):
    """
    Remove trials that have NaN values from a LabeledArray and identify conditions with no valid trials.

    Parameters:
    - labeled_array: A LabeledArray with conditions, trials, channels, and timepoints dimensions.
    - obs_axs: The trials dimension 
    - chans_axs: The channels dimension
    - time_axs: The time dimension
    - freq_axs: The frequency dimension

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
        if freq_axs is not None:
            valid_trial_indices = ~np.isnan(condition_data).any(axis=(chans_axs, time_axs, freq_axs))
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


def remove_nans_from_all_roi_labeled_arrays(roi_labeled_arrays, obs_axs=0, chans_axs=1, time_axs=2, freq_axs=None):
    """
    Loop through all ROIs and apply the NaN removal function to each LabeledArray.

    Parameters:
    - roi_labeled_arrays: Dictionary of LabeledArrays for each ROI.
    - obs_axs: The trials dimension.
    - chans_axs: The channels dimension.
    - time_axs: The time dimension.
    - freq_axs: The frequency dimension

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
            labeled_array, obs_axs=obs_axs, chans_axs=chans_axs, time_axs=time_axs, freq_axs=freq_axs
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
            data = roi_labeled_arrays[roi][cond]  # Shape: (trials, channels, timepoints) or (trials, channels, frequencies, timepoints)
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

def make_np_array_with_nan_trials_removed_for_each_channel(
    roi, subjects_data_objects, condition_names, subjects, 
    electrodes_per_subject_roi, chans_axs=1
):
    """
    Extracts epoch data for each channel in an ROI and removes NaN trials.

    This function iterates through each subject and their corresponding electrodes
    within a specified ROI. For each unique channel (subject-electrode pair),
    it extracts the data, removes any trial that contains one or more NaN
    values, and stores the resulting clean NumPy array in a nested dictionary.

    Args:
        roi (str): The name of the Region of Interest (ROI) to process.
        subjects_data_objects (dict): A dictionary of MNE Epochs or EpochsTFR objects.
        condition_names (list): A list of strings for the condition names.
        subjects (list): A list of subject identifiers.
        electrodes_per_subject_roi (dict): A dictionary mapping ROIs to subjects 
                                           and their electrodes.
        chans_axs (int, optional): The axis index for channels in the MNE data 
                                   array. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - nan_removed_data_dict (dict): A nested dictionary with the structure
              `{condition: {channel: np.ndarray}}` where each array has
              NaN-containing trials removed.
            - all_channels_in_roi (list): A list of all unique channel names
              (e.g., 'D57-LTM1') found in the ROI.
    """
    # Get epochs for each channel individually, and remove nan trials
    # Structure: nan_removed_data_dict[condition][channel_id] = nan_removed_numpy_array
    nan_removed_data_dict = {condition_name: {} for condition_name in condition_names}
    all_channels_in_roi = []
    
    data_type = detect_data_type(subjects_data_objects)
    
    for sub in subjects:
        electrodes = electrodes_per_subject_roi.get(roi, {}).get(sub, []) # get list of electrodes in this roi for this subject
        for electrode in electrodes:
            channel = f"{sub}-{electrode}" # add this subject's name to the electrode name to create the channel name
            if channel not in all_channels_in_roi:
                all_channels_in_roi.append(channel) 
            
            for condition_name in condition_names:
                
                print(f"ROI {roi}: Processing condition {condition_name} for subject {sub}, electrode {electrode}")

                if condition_name not in subjects_data_objects.get(sub, {}):
                    continue
                
                if data_type == 'Epochs':
                    epochs_for_this_sub_and_cond_and_elec = get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects(
                        subjects_data_objects, condition_name, sub, [electrode]
                    )
                elif data_type == 'EpochsTFR':
                    epochs_for_this_sub_and_cond_and_elec = get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects(
                        subjects_data_objects, condition_name, sub, [electrode]
                    )
                else:
                    raise ValueError('currently this only supports Epochs and EpochsTFR data')

                np_array_for_this_sub_and_cond_and_elec = epochs_for_this_sub_and_cond_and_elec.get_data(copy=True).squeeze(axis=chans_axs) # remove the unnecessary channels axis cuz it's just for one channel, so now it's (trials, time) or (trials, freqs, time)
                
                # remove trials with any NaN values for this sub, cond, and chan
                valid_trials_mask = ~np.isnan(np_array_for_this_sub_and_cond_and_elec).any(axis=tuple(range(1, np_array_for_this_sub_and_cond_and_elec.ndim)))
                nan_removed_data_dict[condition_name][channel] = np_array_for_this_sub_and_cond_and_elec[valid_trials_mask]
                
    print(f"ROI {roi}: Final conditions in nan_removed_data_dict: {list(nan_removed_data_dict.keys())}")

    if not all_channels_in_roi:
        print(f" No valid channels found for ROI: {roi}. Skipping.")
        return {}, []
    else:
        
        return nan_removed_data_dict, all_channels_in_roi

def subsample_to_min_trials_per_condition(roi, nan_removed_data_dict, condition_names):
    """
    Determines the minimum number of trials for subsampling.

    For each condition, this function finds the minimum number of valid (NaN-free)
    trials across all channels within a given ROI. This value is used to
    equalize trial counts across channels during bootstrapping.

    Args:
        roi (str): The name of the ROI being processed, for logging purposes.
        nan_removed_data_dict (dict): A nested dictionary from 
                                      `make_np_array_with_nan_trials_removed_for_each_channel`.
        condition_names (list): A list of condition names to process.

    Returns:
        dict: A dictionary `{condition_name: min_trial_count}` mapping each
              condition to the minimum number of trials found across all its
              channels.
    """
    min_trials_per_condition = {}
    for condition_name in condition_names:
        # get trial counts for all channels in this condition
        trial_counts = [len(data) for data in nan_removed_data_dict[condition_name].values()]
        
        if not trial_counts:
            # This can happen if a condition has no channels with valid data
            print(f"Warning: No valid trials found for condition '{condition_name}' in ROI {roi}. Setting min trials to 0.")
            min_trials_per_condition[condition_name] = 0
        else:
            min_trials_per_condition[condition_name] = min(trial_counts)
            print(f"condition '{condition_name}': subsampling to {min_trials_per_condition[condition_name]} trials")
            
    return min_trials_per_condition

def make_bootstrapped_labeled_arrays_for_roi(
    nan_removed_data_dict, min_trials_per_condition, all_channels_in_roi,
    condition_names, n_bootstraps, rng, chans_axs, time_axs, data_type,
    sample_times, freq_axs=None, sample_freqs=None
):
    """
    Generates bootstrapped LabeledArray samples for one ROI.

    For each bootstrap iteration, this function samples trials *without replacement*
    from each channel's cleaned data, ensuring each channel contributes an equal
    number of trials per condition. These resampled channels are then stacked
    to form a dense LabeledArray for that bootstrap instance.

    Args:
        nan_removed_data_dict (dict): Nested dictionary with NaN-free trial data.
        min_trials_per_condition (dict): Dictionary specifying the number of 
                                         trials to sample for each condition.
        all_channels_in_roi (list): List of all channel names in the ROI.
        condition_names (list): List of condition names.
        n_bootstraps (int): The number of bootstrap samples to generate.
        rng (np.random.RandomState): NumPy random number generator instance.
        chans_axs (int): The axis index for channels.
        time_axs (int): The axis index for time.
        data_type (str): The type of MNE data ('Epochs' or 'EpochsTFR').
        sample_times (np.ndarray): Array of time points for labeling the time axis.
        freq_axs (int, optional): The axis index for frequency. Defaults to None.
        sample_freqs (np.ndarray, optional): Array of frequencies for labeling. 
                                             Defaults to None.

    Returns:
        list: A list of `LabeledArray` objects, where each element is one 
              bootstrap sample.
    """
    bootstrapped_roi_arrays = []
    print(f"generating {n_bootstraps} bootstrap sample(s)")
    
    for i in range(n_bootstraps):
        bootstrapped_conditions_data = {}
        
        # loop through conditions to build one bootstrapped LabeledArray
        for condition_name in condition_names:
            n_samples = min_trials_per_condition[condition_name]
            if n_samples == 0:
                # Skip condition if it has no trials to sample
                raise ValueError(f"n_samples is 0")

            resampled_channels_for_condition = []
            for channel in all_channels_in_roi:
                channel_data = nan_removed_data_dict[condition_name][channel]
                
                if len(channel_data) < n_samples:
                     print(f"PROBLEM FOUND: Channel {channel} has only {len(channel_data)} trials for {condition_name}, needs {n_samples}")
                     # You might want to 'continue' or handle this case explicitly
                     # For now, just finding it is key.
                     import pdb; pdb.set_trace() # Force a stop if this happens
                     
                # randomly subsample trials *without* replacement down to the channel with the fewest good trials for this roi and condition
                sample_indices = rng.choice(len(channel_data), size=n_samples, replace=False) 
                resampled_channels_for_condition.append(channel_data[sample_indices])
                
            # Stack the resampled channels along the channel axis
            # this makes a dense array for this condition: (trials, channels, time) or (trials, channels, freqs, time)
            concatenated_chans = np.stack(resampled_channels_for_condition, axis=chans_axs)
            if condition_name == "Stimulus_i_in_25switchBlock": # <-- ADD THIS CHECK
                        # <<< SET BREAKPOINT HERE (3c) >>>
                        # Inspect specifically for the problematic condition:
                        #   - np.sum(np.isnan(concatenated_chans)) 
                        # Do NaNs appear right after stacking for *this* condition?
                        if np.isnan(concatenated_chans).any():
                            print(f"NANS APPEARED after stacking for {condition_name}!")
                            import pdb; pdb.set_trace() # Force a stop
                   
            bootstrapped_conditions_data[condition_name] = concatenated_chans
            
        print(f"Built bootstrapped_conditions_data with conditions: {list(bootstrapped_conditions_data.keys())}")
    
        if not bootstrapped_conditions_data:
            raise ValueError(f"Warning: unable to create bootstrapped conditions data on iteration {i}")
    
        # make the LabeledArray for this bootstrap
        bootstrapped_labeled_array = LabeledArray.from_dict(bootstrapped_conditions_data)
        print(f"Bootstrap {i}: LabeledArray has conditions: {list(bootstrapped_labeled_array.keys())}")

        # Add labels
        bootstrapped_labeled_array.labels[chans_axs+1].values = np.array(all_channels_in_roi) # channel axis
        bootstrapped_labeled_array.labels[time_axs+1].values = np.array([str(t) for t in sample_times]) # time axis
        if freq_axs is not None and data_type == 'EpochsTFR' and sample_freqs is not None:
            bootstrapped_labeled_array.labels[freq_axs+1].values = np.array([str(f) for f in sample_freqs]) # freq axis
        
        bootstrapped_roi_arrays.append(bootstrapped_labeled_array)
    
    return bootstrapped_roi_arrays

def make_bootstrapped_roi_labeled_array_with_nan_trials_removed_for_each_channel(
    roi, subjects_data_objects, condition_names,
    subjects, electrodes_per_subject_roi, n_bootstraps=1,
    obs_axs=0, chans_axs=1, time_axs=2, freq_axs=None,
    random_state=None                  
):
    """
    Orchestrates the creation of bootstrapped LabeledArrays for a single ROI.

    This function is a pipeline that:
    1. Removes NaN-containing trials on a per-channel basis.
    2. Determines the minimum trial count per condition to equalize trial numbers.
    3. Generates a list of bootstrapped `LabeledArray` samples by repeatedly
       subsampling trials from each channel.

    Args:
        roi (str): The single ROI name to be processed.
        subjects_data_objects (dict): A dictionary of MNE Epochs or EpochsTFR objects.
        condition_names (list): A list of condition names.
        subjects (list): A list of subjects.
        electrodes_per_subject_roi (dict): A dictionary mapping ROIs to subjects
                                           and their electrodes.
        n_bootstraps (int, optional): The number of bootstrap samples to create. 
                                      Defaults to 1.
        obs_axs (int, optional): The trials dimension index. Defaults to 0.
        chans_axs (int, optional): The channels dimension index. Defaults to 1.
        time_axs (int, optional): The time dimension index. Defaults to 2.
        freq_axs (int, optional): The frequency dimension index. Defaults to None.
        random_state (int, RandomState, or None): Seed for the random number
                                                   generator. Defaults to None.

    Returns:
        list: A list containing the generated bootstrapped `LabeledArray`
              objects for the specified ROI. Returns an empty list if no
              valid data is found for the ROI.
    """
    rng = np.random.RandomState(random_state)
    print(f"\nMaking LabeledArray for ROI: {roi} with NaN trials removed within each channel")
    
    # Step 1: Extract data and remove NaNs on a per-channel basis
    nan_removed_data_dict, all_channels_in_roi = make_np_array_with_nan_trials_removed_for_each_channel(
        roi, subjects_data_objects, condition_names, subjects, electrodes_per_subject_roi, chans_axs
    )

    if not all_channels_in_roi:
        return []

    # Step 2: Determine subsampling size
    min_trials_per_condition = subsample_to_min_trials_per_condition(
        roi, nan_removed_data_dict, condition_names
    )
    
    # Extract sample time/frequency info from the first available MNE object
    data_type = detect_data_type(subjects_data_objects)
    sample_epochs = None
    for sub in subjects:
        if sub in subjects_data_objects:
            for cond in condition_names:
                if cond in subjects_data_objects[sub]:
                    if data_type == 'Epochs':
                        sample_epochs = subjects_data_objects[sub][cond]['HG_ev1_power_rescaled']
                    elif data_type == 'EpochsTFR':
                        sample_epochs = subjects_data_objects[sub][cond]
                    break
        if sample_epochs is not None:
            break
    if sample_epochs is None:
        raise ValueError(f"Could not find any MNE data for ROI {roi} to extract time/freq labels.")

    sample_times = sample_epochs.times
    sample_freqs = sample_epochs.freqs if data_type == 'EpochsTFR' else None

    # Step 3: Generate bootstrapped arrays
    bootstrapped_roi_arrays = make_bootstrapped_labeled_arrays_for_roi(
        nan_removed_data_dict, min_trials_per_condition, all_channels_in_roi,
        condition_names, n_bootstraps, rng, chans_axs, time_axs, data_type,
        sample_times, freq_axs, sample_freqs
    )
    
    return bootstrapped_roi_arrays

def make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
    rois, subjects_data_objects, condition_names, subjects,
    electrodes_per_subject_roi, n_bootstraps=1, n_jobs=-1,
    obs_axs=0, chans_axs=1, time_axs=2, freq_axs=None, random_state=None
):
    """
    Parallelizes the creation of bootstrapped LabeledArrays across multiple ROIs.

    This function uses `joblib` to process each ROI in parallel, calling
    `make_bootstrapped_roi_labeled_array_with_nan_trials_removed_for_each_channel`
    for each one.

    Args:
        rois (list): A list of ROI names to process.
        subjects_data_objects (dict): A dictionary of MNE Epochs or EpochsTFR objects.
        condition_names (list): A list of condition names.
        subjects (list): A list of subjects.
        electrodes_per_subject_roi (dict): A dictionary mapping ROIs to subjects
                                           and their electrodes.
        n_bootstraps (int, optional): Number of bootstrap samples per ROI. Defaults to 1.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all CPUs).
        obs_axs (int, optional): The trials dimension index. Defaults to 0.
        chans_axs (int, optional): The channels dimension index. Defaults to 1.
        time_axs (int, optional): The time dimension index. Defaults to 2.
        freq_axs (int, optional): The frequency dimension index. Defaults to None.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        dict: A dictionary where keys are ROI names and values are lists of
              bootstrapped `LabeledArray` objects.
    """
    # Use joblib to parallelize the processing of each ROI
    results = Parallel(n_jobs=n_jobs)(
        delayed(make_bootstrapped_roi_labeled_array_with_nan_trials_removed_for_each_channel)(
            roi,
            subjects_data_objects,
            condition_names,
            subjects,
            electrodes_per_subject_roi,
            n_bootstraps,
            obs_axs,
            chans_axs,
            time_axs,
            freq_axs,
            random_state + i if random_state is not None else None  # Ensure different seeds for parallel jobs
        ) for i, roi in enumerate(rois)
    )

    # Combine the results from the parallel jobs into a dictionary
    roi_bootstrapped_arrays = {roi: result for roi, result in zip(rois, results) if result}

    return roi_bootstrapped_arrays
    

                
                
                
                
                
    
    
    
