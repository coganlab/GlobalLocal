import mne
import json
import numpy as np
import os
import pandas as pd

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


def save_sig_chans(mask_name, mask, channels, subject, save_path):
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
    filename = os.path.join(save_path, f'sig_chans_{subject}_{mask_name}.json')
    
    # Save the dictionary as a JSON file
    with open(filename, 'w') as file:
        json.dump(data, file)
    
    print(f'Saved significant channels for subject {subject} and mask {mask_name} to {filename}')


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



def filter_and_average_epochs(epochs, start_idx, end_idx, accuracy_column='accuracy'):
    """
    Calculates trial averages for accurate trials and time averages with inaccurate trials marked as NaNs.

    Parameters:
    - epochs: MNE Epochs object with accuracy metadata.
    - start_idx: Start index for time averaging.
    - end_idx: End index for time averaging.
    - accuracy_column: Name of the column in the metadata that contains accuracy data.

    Returns:
    - trial_avg_data: Trial-averaged data across accurate trials.
    - time_avg_data: Time-averaged data with inaccurate trials marked as NaNs.
    """
    # Separate accurate and all trials data
    accurate_epochs_data = epochs[epochs.metadata[accuracy_column] == 1.0].get_data()
    all_epochs_data = epochs.get_data().copy()

    # Mark inaccurate trials as NaNs in the all_epochs_data
    inaccurate_indices = epochs.metadata[accuracy_column] != 1.0
    all_epochs_data[inaccurate_indices, :, :] = np.nan

    # Calculate trial average for accurate trials
    trial_avg_data = np.nanmean(accurate_epochs_data, axis=0)

    # Calculate time average within the specified window
    time_avg_data = np.nanmean(all_epochs_data[:, :, start_idx:end_idx], axis=2)

    return trial_avg_data, time_avg_data



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