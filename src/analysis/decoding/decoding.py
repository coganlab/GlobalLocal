def get_max_trials_per_condition(
    subjects_mne_objects, condition_names, subjects,
    sig_electrodes_per_subject_roi, roi, obs_axs
):
    """
    Find the maximum number of trials per condition across all subjects for a given ROI,
    and identify which subject(s) have that maximum number of trials.

    Args:
        subjects_mne_objects (dict): Dictionary of MNE epoch objects, structured as
                                     {subject_id: {condition_name: mne.Epochs}}.
        condition_names (list): List of strings representing the condition names.
        subjects (list): List of subject identifiers.
        sig_electrodes_per_subject_roi (dict): Dictionary mapping ROIs to subjects
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
        sig_electrodes = sig_electrodes_per_subject_roi.get(roi, {}).get(sub, [])
        if not sig_electrodes:
            continue
        for condition_name in condition_names:
            # Check if the subject has data for this condition
            if condition_name not in subjects_mne_objects[sub]:
                continue
            epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(sig_electrodes)
            epochs_data = epochs.get_data(copy=True)
            n_trials = epochs_data.shape[obs_axs]
            if n_trials > max_trials_per_condition[condition_name]:
                max_trials_per_condition[condition_name] = n_trials
                max_trials_subject_per_condition[condition_name] = [sub]
            elif n_trials == max_trials_per_condition[condition_name]:
                max_trials_subject_per_condition[condition_name].append(sub)
    return max_trials_per_condition, max_trials_subject_per_condition

def make_subject_labeled_array(
    sub, subjects_mne_objects, condition_names, sig_electrodes_per_subject_roi,
    roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, rng
):
    """
    Process data for a single subject in a given ROI to create a LabeledArray.

    This function performs the following steps:
    1. Retrieves significant electrodes in the specified ROI for the subject.
    2. For each condition:
        a. Extracts epoch data for significant electrodes.
        b. Randomizes trial order.
        c. Pads data with NaNs to match the `max_trials_per_condition` for that condition.
    3. Collects processed data for all conditions into a dictionary.
    4. Creates a LabeledArray from this dictionary, assigning channel and time labels.
       The LabeledArray will have a new leading dimension for conditions.

    Args:
        sub (str): Subject identifier.
        subjects_mne_objects (dict): {subject_id: {condition_name: mne.Epochs}}.
        condition_names (list): List of condition names.
        sig_electrodes_per_subject_roi (dict): {roi_name: {subject_id: [electrode_list]}}.
        roi (str): The ROI name to process.
        max_trials_per_condition (dict): {condition_name: max_trial_count} for padding.
        obs_axs (int): Original trials axis index in epoch data (e.g., 0 for (trials, chans, time)).
        chans_axs (int): Original channels axis index in epoch data (e.g., 1 for (trials, chans, time)).
        time_axs (int): Original time axis index in epoch data (e.g., 2 for (trials, chans, time)).
        rng (np.random.RandomState): NumPy random number generator instance for shuffling.

    Returns:
        LabeledArray or None: A LabeledArray for the subject with dimensions
                              (Conditions, Trials, Channels, Timepoints).
                              Returns None if the subject has no significant electrodes for the ROI.
    """
    sig_electrodes = sig_electrodes_per_subject_roi.get(roi, {}).get(sub, [])
    if not sig_electrodes:
        return None

    subject_nested_dict = {}

    # Get channel names for this subject's ROI
    sub_channel_names = [f"{sub}-{sig_electrode}" for sig_electrode in sig_electrodes]

    # Loop through each condition
    for condition_name in condition_names:
        # Extract the epoch data for the current condition and subject
        epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy().pick(sig_electrodes)
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
    subject_labeled_array = create_subject_labeled_array_from_dict(
        subject_nested_dict, sub_channel_names, np_array_str_times, chans_axs, time_axs
    )

    # Print the shape and time axis labels
    print(f"Subject {sub}, ROI {roi}, LabeledArray shape: {subject_labeled_array.shape}")
    time_axis_size = subject_labeled_array.shape[time_axs+1] # Adjusted time axis index

    return subject_labeled_array

def create_subject_labeled_array_from_dict(
    subject_nested_dict, sub_channel_names, np_array_str_times, chans_axs, time_axs
):
    """
    Create a LabeledArray for a subject from a dictionary of condition data.

    The input dictionary `subject_nested_dict` is expected to have condition names
    as keys and NumPy arrays (trials, channels, timepoints) as values.
    This function creates a LabeledArray where the first dimension is 'conditions'.
    Labels for channel and time axes are assigned by adjusting their original
    axis indices (`chans_axs`, `time_axs`) by +1 to account for the new 'conditions' axis.

    Args:
        subject_nested_dict (dict): {condition_name: np.ndarray(trials, channels, timepoints)}.
        sub_channel_names (list): List of strings for channel labels.
        np_array_str_times (np.ndarray): Array of strings for time labels.
        chans_axs (int): Original channels axis index in the per-condition data arrays
                         (e.g., 1 if data is trials, channels, time).
        time_axs (int): Original time axis index in the per-condition data arrays
                        (e.g., 2 if data is trials, channels, time).

    Returns:
        LabeledArray: A LabeledArray object with dimensions (Conditions, Trials, Channels, Timepoints)
                      and corresponding labels.
    """
    subject_labeled_array = LabeledArray.from_dict(subject_nested_dict)
    # Adjust axes indices due to the added conditions axis
    subject_labeled_array.labels[chans_axs + 1].values = sub_channel_names  # Channels axis
    subject_labeled_array.labels[time_axs + 1].values = np_array_str_times  # Time axis
    return subject_labeled_array

def concatenate_subject_labeled_arrays(
    roi_labeled_array, subject_labeled_array, chans_axs
):
    """
    Concatenate a subject's LabeledArray to an ROI's LabeledArray along the channels axis.

    The channels axis index (`chans_axs`) is adjusted by +1 to account for the
    leading 'conditions' dimension in the LabeledArrays. If `roi_labeled_array`
    is None (i.e., this is the first subject for the ROI), the `subject_labeled_array`
    is returned directly.

    Args:
        roi_labeled_array (LabeledArray or None): The LabeledArray for the ROI accumulated so far.
                                                  Can be None if this is the first subject.
        subject_labeled_array (LabeledArray): The LabeledArray for the current subject.
        chans_axs (int): Original channels axis index (e.g., 1 for (trials, chans, time)
                         data before becoming part of LabeledArray). This will be
                         adjusted to `chans_axs + 1` for concatenation.

    Returns:
        LabeledArray: The updated LabeledArray for the ROI, with the new subject's
                      data concatenated along the (adjusted) channels axis.
    """
    concatenation_axis = chans_axs + 1  # Adjusted channels axis index
    if roi_labeled_array is None:
        return subject_labeled_array
    else:
        return roi_labeled_array.concatenate(subject_labeled_array, axis=concatenation_axis)

def put_data_in_labeled_array_per_roi_subject(
    subjects_mne_objects, condition_names, rois, subjects,
    sig_electrodes_per_subject_roi, obs_axs=0, chans_axs=1, time_axs=2,
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
    - sig_electrodes_per_subject_roi: Dictionary mapping ROIs to subjects and their corresponding electrodes.
    - obs_axs: The trials dimension (ignoring the conditions dimension for now)
    - chans_axs: The channels dimension
    - time_axs: The time dimension
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
            sig_electrodes_per_subject_roi, roi, obs_axs
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
                sub, subjects_mne_objects, condition_names, sig_electrodes_per_subject_roi,
                roi, max_trials_per_condition, obs_axs, chans_axs, time_axs, rng
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