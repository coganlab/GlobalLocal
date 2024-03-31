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

    # Calculate trial standard deviation for accurate trials
    trial_std_data = np.nanstd(accurate_epochs_data, axis=0)

    # Calculate time average within the specified window
    time_avg_data = np.nanmean(all_epochs_data[:, :, start_idx:end_idx], axis=2)

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


def load_mne_objects(sub, output_name, task, LAB_root=None):
    """
    Load MNE objects for a given subject and output name.

    Parameters:
    - sub (str): Subject identifier.
    - output_name (str): Output name used in the file naming.
    - task (str): Task identifier.
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

    # Define file paths
    HG_ev1_file = f'{save_dir}/{sub}_{output_name}_HG_ev1-epo.fif'
    HG_base_file = f'{save_dir}/{sub}_{output_name}_HG_base-epo.fif'
    HG_ev1_rescaled_file = f'{save_dir}/{sub}_{output_name}_HG_ev1_rescaled-epo.fif'

    # Load the objects
    HG_ev1 = mne.read_epochs(HG_ev1_file)
    HG_base = mne.read_epochs(HG_base_file)
    HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
    HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0))
    HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

    return {
        'HG_ev1': HG_ev1,
        'HG_base': HG_base,
        'HG_ev1_rescaled': HG_ev1_rescaled,
        'HG_ev1_evoke': HG_ev1_evoke,
        'HG_ev1_evoke_rescaled': HG_ev1_evoke_rescaled
    }

def create_subjects_mne_objects_dict(subjects, output_names_conditions, task, combined_data, acc_array, LAB_root=None):
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
        for output_name, conditions in output_names_conditions.items():
            print(f"  Loading output: {output_name} with conditions: {conditions}")
            
            # Build the filtering condition
            sub_without_zeroes = "D" + sub[1:].lstrip('0') 
            condition_filter = (combined_data['subject_ID'] == sub) # this previously indexed using sub_without_zeroes, but now just uses sub. 3/17.
                    
            for condition_column, condition_value in conditions.items():
                if isinstance(condition_value, list):
                    # If the condition needs to match any value in a list
                    condition_filter &= combined_data[condition_column].isin(condition_value)
                else:
                    # If the condition is a single value
                    condition_filter &= (combined_data[condition_column] == condition_value)
            
            # Filter combinedData for the specific subject and conditions
            subject_condition_data = combined_data[condition_filter]
            
            # Load MNE objects and update with accuracy data
            mne_objects = load_mne_objects(sub, output_name, task, LAB_root)
            
            if sub in acc_array:
                trial_counts = subject_condition_data['trialCount'].values.astype(int)
                accuracy_data = [acc_array[sub][i-1] for i in trial_counts if i-1 < len(acc_array[sub])] # Subtract 1 here for zero-based indexing in acc array.
                # Now pass trial_counts along with accuracy_data
                mne_objects['HG_ev1_rescaled'] = add_accuracy_to_epochs(mne_objects['HG_ev1_rescaled'], accuracy_data)

            sub_mne_objects[output_name] = mne_objects
        subjects_mne_objects[sub] = sub_mne_objects

    return subjects_mne_objects

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

def perform_modular_anova(df, time_window, save_dir, save_name):
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