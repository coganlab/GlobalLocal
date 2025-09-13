import mne
import sys
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
#from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle, find_outliers
from ieeg.viz.mri import gen_labels
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import OrderedDict, defaultdict
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def make_subjects_electrodes_to_ROIs_dict(subjects, task='GlobalLocal', LAB_root=None, save_dir=None, filename='subjects_electrodes_to_ROIs_dict.json'):
    """
    Creates mappings for each electrode to its corresponding Region of Interest (ROI)
    for a list of subjects and saves these mappings to a JSON file.

    The function processes electrophysiological data for each subject to determine
    electrode-ROI relationships. It generates three types of dictionaries for each subject:
    1.  `default_dict`: A direct mapping from each processed electrode name (str) to its ROI name (str).
    2.  `rawROI_dict`: A mapping from each ROI name (str) to a list of electrode names (list of str)
        belonging to that ROI. This includes all ROIs found.
    3.  `filtROI_dict`: Similar to `rawROI_dict`, but ROIs containing "White-Matter" in their name
        are excluded.

    Parameters:
    ----------
    subjects : list of str
        A list of subject identifiers (e.g., ['D0057', 'D0059']).
    task : str
        The task identifier string (e.g., 'GlobalLocal') used to locate data.
    LAB_root : str
        The absolute path to the root directory where lab data is stored.
    save_dir : str
        The absolute path to the directory where the output JSON file
        ('subjects_electrodes_to_ROIs_dict.json') will be saved.
    filename : str, optional
        The name of the JSON file (default is 'subjects_electrodes_to_ROIs_dict.json').

    Returns:
    -------
    dict
        A dictionary where keys are subject identifiers (str). Each subject key maps
        to another dictionary containing three keys:
        - 'default_dict': dict, maps electrode name (str) to ROI name (str).
        - 'rawROI_dict': dict, maps ROI name (str) to a list of electrode names (list of str).
        - 'filtROI_dict': dict, maps ROI name (str, excluding "White-Matter") to a list of
                          electrode names (list of str).
        Example:
        {
            'D0057': {
                'default_dict': {'LAH1': 'Left_Amygdala', ...},
                'rawROI_dict': {'Left_Amygdala': ['LAH1', 'LAH2'], ...},
                'filtROI_dict': {'Left_Amygdala': ['LAH1', 'LAH2'], ...}
            },
            ...
        }
    """
    
    # Initialize the outer dictionary.
    subjects_electrodes_to_ROIs_dict = {}

    if save_dir is None:
        raise ValueError("save_dir must be specified to save the dictionary.")
    
    # Determine LAB_root based on the operating system and environment
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)
        
        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            # Check if we're on the cluster by looking for /cwork directory
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                # Fallback for other Linux systems
                LAB_root = os.path.join(HOME, "CoganLab")

    for sub in subjects:
        print(sub)
        layout = get_data(task, root=LAB_root)
        filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                            extension='.edf', desc='clean', preload=False)

        good = crop_empty_data(filt)

        good.info['bads'] = channel_outlier_marker(good, 3, 2)

        # Drop the trigger channel if it exists 9/30
        if 'Trigger' in good.ch_names:
            good.drop_channels('Trigger')

        filt.drop_channels(good.info['bads'])  # this has to come first cuz if you drop from good first, then good.info['bads'] is just empty
        good.drop_channels(good.info['bads'])

        good.load_data()
        channels = good.ch_names

        ch_type = filt.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        if sub == 'D0107A':
            default_dict = gen_labels(good.info, sub='D107A')
        else:
            default_dict = gen_labels(good.info)

        # Create rawROI_dict for the subject
        rawROI_dict = defaultdict(list)
        for key, value in default_dict.items():
            rawROI_dict[value].append(key)
        rawROI_dict = dict(rawROI_dict)

        # Filter out keys containing "White-Matter"
        filtROI_dict = {key: value for key, value in rawROI_dict.items() if "White-Matter" not in key}

        # Store the dictionaries in the subjects dictionary
        subjects_electrodes_to_ROIs_dict[sub] = {
            'default_dict': dict(default_dict),
            'rawROI_dict': dict(rawROI_dict),
            'filtROI_dict': dict(filtROI_dict)
        }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")

    # # Save to a JSON file. Uncomment when actually running.
    save_filepath = os.path.join(save_dir, filename)

    with open(save_filepath, 'w') as file:
        json.dump(subjects_electrodes_to_ROIs_dict, file, indent=4)

    print(f"Saved subjects_dict to {save_filepath}")

    return subjects_electrodes_to_ROIs_dict

def load_subjects_electrodes_to_ROIs_dict(save_dir, filename='subjects_electrodes_to_ROIs_dict.json'):
    """
    Attempts to load the subjects' electrode-to-ROI dictionary from a specified
    JSON file located in the given output directory.

    Parameters:
    ----------
    save_dir : str
        The absolute path to the directory where the JSON file is expected to be.
    filename : str, optional
        The name of the JSON file (default is 'subjects_electrodes_to_ROIs_dict.json').

    Returns:
    -------
    dict or None
        A dictionary containing the loaded electrode-to-ROI mappings if the file
        is found and successfully parsed. The structure is expected to be:
        {
            'D0057': {
                'default_dict': {'RAI7': 'dlPFC', ...},
                'rawROI_dict': {'dlPFC': ['RAI7'], ...},
                'filtROI_dict': {'dlPFC': ['RAI7'], ...}
            },
            ...
        }
        Returns None if the file is not found or if a JSON decoding error occurs.
    """
    filepath = os.path.join(save_dir, filename)
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        print(f"Loaded data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {filepath}: {e}")
        return None
    except IOError as e:
        print(f"Failed to read file {filepath}: {e}")
        return None
    
def make_or_load_subjects_electrodes_to_ROIs_dict(subjects, task='GlobalLocal', LAB_root=None, save_dir=None, 
                                                filename='subjects_electrodes_to_ROIs_dict.json', 
                                                ):
    """
    Ensures the subjects' electrodes-to-ROIs dictionary is available.
    It first attempts to load the dictionary from a JSON file in the specified
    `output_dir`. If the file doesn't exist or cannot be loaded, this function
    calls `make_subjects_electrodes_to_ROIs_dict` to create it, save it, and then
    attempts to load it again.

    Parameters:
    ----------
    subjects : list of str
        A list of subject identifiers. Required if the dictionary needs to be created.
    task : str
        The task identifier string. Required if the dictionary needs to be created.
    LAB_root : str
        The absolute path to the root directory for lab data. Required if the
        dictionary needs to be created.
    save_dir : str
        The absolute path to the directory where the dictionary JSON file is
        (or will be) stored.
    filename : str, optional
        The name of the JSON file (default is 'subjects_electrodes_to_ROIs_dict.json').

    Returns:
    -------
    dict or None
        A dictionary mapping subject identifiers to their respective electrode and ROI
        information. The structure is as described in `make_subjects_electrodes_to_ROIs_dict`.
        Returns None if the dictionary cannot be loaded and also fails to be created
        or loaded after attempting creation.
    
    Raises:
    ------
    ValueError
        If essential parameters (`subjects`, `task`, `LAB_root`, `output_dir`)
        are missing when the dictionary needs to be created.
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified to save or load the dictionary.")
    
    print("Attempting to load the subjects' electrodes-to-ROIs dictionary...")
    subjects_electrodes_to_ROIs_dict = load_subjects_electrodes_to_ROIs_dict(save_dir, filename)

    if subjects_electrodes_to_ROIs_dict is None:
        print("No dictionary found. Looks like it's our lucky day to create one!")
        make_subjects_electrodes_to_ROIs_dict(subjects, task, LAB_root, save_dir, filename)
        subjects_electrodes_to_ROIs_dict = load_subjects_electrodes_to_ROIs_dict(save_dir, filename)
        print("Dictionary created and loaded successfully. Let's roll!")

    else:
        print("Dictionary loaded successfully. Ready to proceed!")

    return subjects_electrodes_to_ROIs_dict

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
    -------
    dict
        A dictionary where keys are descriptive strings of the MNE Epochs objects
        (e.g., 'HG_ev1', 'HG_ev1_rescaled', 'HG_base', 'HG_ev1_power_rescaled')
        and values are the corresponding loaded MNE Epochs objects. The specific
        keys present depend on the `just_HG_ev1_rescaled` flag.

    """

    # Determine LAB_root based on the operating system and environment
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)
        
        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            # Check if we're on the cluster by looking for /cwork directory
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                # Fallback for other Linux systems
                LAB_root = os.path.join(HOME, "CoganLab")
                
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
        HG_ev1_rescaled = mne.read_epochs(HG_ev1_rescaled_file)
        HG_ev1_power_rescaled = mne.read_epochs(HG_ev1_power_rescaled_file)

        mne_objects['HG_ev1'] = HG_ev1
        mne_objects['HG_base'] = HG_base
        mne_objects['HG_ev1_rescaled'] = HG_ev1_rescaled
        mne_objects['HG_ev1_power_rescaled'] = HG_ev1_power_rescaled

    return mne_objects

def create_subjects_mne_objects_dict(subjects, epochs_root_file, conditions, task, just_HG_ev1_rescaled=False, LAB_root=None, acc_trials_only=True, error_trials_only=False):
    """
    Create a nested dictionary of MNE Epochs objects for multiple subjects,
    organized by experimental conditions and MNE object types.

    This function iterates through a list of subjects, loads their relevant
    MNE Epochs objects using `load_mne_objects`, and then further processes
    these epochs based on specified experimental conditions. It can optionally
    filter epochs to include only accurate trials.

    Parameters:
    ----------
    subjects : list of str
        A list of subject identifiers (e.g., ['D0057', 'D0059']).
    epochs_root_file : str
        The base name of the original epochs file, passed to `load_mne_objects`.
        Example: 'Stimulus_1sec_preStimulusBase_decFactor_10'.
    conditions : dict
        A dictionary defining the experimental conditions to extract.
        - Keys (str): User-defined names for each condition (e.g., 'TargetAuditory', 'StandardVisual').
        - Values (dict): Parameters for each condition. Each condition's dictionary
          *must* contain a 'BIDS_events' key.
          The value for 'BIDS_events' can be:
            - A string: representing a single BIDS event type (e.g., 'auditory/target').
            - A list of strings: representing multiple BIDS event types to be
              concatenated for this condition (e.g., ['visual/target', 'visual/nontarget']).
        Example:
        ```python
        conditions = {
            'AuditoryTarget': {'BIDS_events': 'auditory/target', 'other_param': 'value'},
            'VisualCombined': {'BIDS_events': ['visual/target', 'visual/nontarget']}
        }
        ```
    task : str
        The name of the experimental task (i.e., GlobalLocal), passed to `load_mne_objects` 
    just_HG_ev1_rescaled : bool, optional
        Flag passed to `load_mne_objects`. If True, only a subset of MNE objects
        (baseline rescaled high-gamma for event 1) will be loaded and available for
        conditioning. Defaults to False.
    LAB_root : str, optional
        The root directory of the laboratory's data storage, passed to
        `load_mne_objects`. Defaults to None (auto-detection).
    acc_trials_only : bool, optional
        If True, filters the loaded MNE Epochs objects to include only trials
        marked as accurate. This is done by selecting epochs where the metadata
        field 'Accuracy1.0' (or a similar field indicating accuracy) is equal to 1.0.
        Defaults to True.
    error_trials_only : bool, optional
        If True, filters the loaded MNE Epochs objects to include only trials
        marked as inaccurate. This is done by selecting epochs where the metadata
        field 'Accuracy1.0' (or a similar field indicating accuracy) is equal to 0.0.
        Defaults to False.

    Returns:
    -------
    dict
        A nested dictionary with the following structure:
        `subjects_mne_objects[subject_id][condition_name][mne_object_type] = MNE Epochs or Evoked Object`
        Where:
        - `subject_id` (str): The subject identifier.
        - `condition_name` (str): The user-defined name of the experimental condition.
        - `mne_object_type` (str): The type of MNE data (e.g., 'HG_ev1_rescaled',
          'HG_ev1', 'HG_ev1_power_rescaled', 'HG_ev1_avg', 'HG_ev1_rescaled_avg', 'HG_ev1_power_rescaled_avg', 'HG_ev1_std_err', 'HG_ev1_rescaled_std_err', 'HG_ev1_power_rescaled_std_err'). The specific types available depend
          on `just_HG_ev1_rescaled` and the files loaded by `load_mne_objects`.
          Each value is an MNE Epochs or Evoked object corresponding to the specified
          subject, condition, and data type.
    """
    subjects_mne_objects = {}

    for sub in subjects:
        print(f"Loading data for subject: {sub}")
        sub_mne_objects = {}

        mne_objects = load_mne_objects(sub, epochs_root_file, task, just_HG_ev1_rescaled=just_HG_ev1_rescaled, LAB_root=LAB_root)
        for mne_object in mne_objects.keys():
            if acc_trials_only == True:
                mne_objects[mne_object] = mne_objects[mne_object]["Accuracy1.0"] # this needs to be done for all the epochs objects I think. So loop over them. Unless it's set to just_HG_ev1_rescaled.
            elif error_trials_only == True:
                mne_objects[mne_object] = mne_objects[mne_object]["Accuracy0.0"] # this needs to be done for all the epochs objects I think. So loop over them. Unless it's set to just_HG_ev1_rescaled.
            
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

                # create evoked objects for each condition (mean and standard error) using nanmean and nanstd
                # Get the epochs data
                epochs_data = event_epochs.get_data()
                print(f"    Original shape: {epochs_data.shape}")

                # DEBUGGING: Check NaN statistics
                nan_count_per_trial = np.sum(np.isnan(epochs_data), axis=(1, 2))
                fully_nan_trials = np.all(np.isnan(epochs_data), axis=(1, 2))

                print(f"    Trials with all NaN values: {np.sum(fully_nan_trials)}")
                print(f"    Average NaN count per trial: {np.mean(nan_count_per_trial):.1f}")
                print(f"    Max NaN count in a trial: {np.max(nan_count_per_trial)}")

                # DEBUGGING: Check if specific channels have many NaNs
                nan_per_channel = np.sum(np.isnan(epochs_data), axis=(0, 2))
                high_nan_channels = np.where(nan_per_channel > 0.5 * epochs_data.shape[0] * epochs_data.shape[2])[0]
                if len(high_nan_channels) > 0:
                    print(f"    Channels with >50% NaN: {[event_epochs.ch_names[i] for i in high_nan_channels]}")

                # Check how many trials have valid data (not all NaN)
                # A trial is valid if at least one channel has non-NaN data
                valid_trials_mask = ~np.all(np.isnan(epochs_data), axis=(1, 2))
                n_valid_trials = np.sum(valid_trials_mask)

                print(f"    {condition_name}: {n_valid_trials} valid trials out of {len(event_epochs)}")

                # Compute the nanmean across trials
                data_nanmean = np.nanmean(epochs_data, axis=0)
                evoked_avg = event_epochs.average()  # Create template
                evoked_avg.data = data_nanmean
                evoked_avg.nave = n_valid_trials
                sub_mne_objects[condition_name][mne_object + '_avg'] = evoked_avg

                # Compute the standard error across trials using nanstd
                # Calculate valid trials per channel-timepoint
                n_valid_per_channel_time = np.sum(~np.isnan(epochs_data), axis=0)

                # Avoid division by zero or sqrt of negative numbers
                n_valid_per_channel_time = np.maximum(n_valid_per_channel_time, 1)

                # Calculate standard error
                std_err_data = np.nanstd(epochs_data, axis=0, ddof=1) / np.sqrt(n_valid_per_channel_time)

                # Handle cases where all values are NaN - TODO: hmm is this line necessary? 8/21/25
                std_err_data = np.nan_to_num(std_err_data, nan=0.0)

                evoked_std_err = event_epochs.average()  # Create template
                evoked_std_err.data = std_err_data
                evoked_std_err.nave = int(np.mean(n_valid_per_channel_time))
                sub_mne_objects[condition_name][mne_object + '_std_err'] = evoked_std_err
                 
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


def save_sig_chans_with_reject(condition_name, reject, channels, subject, save_dir):
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
    filename = os.path.join(save_dir, f'sig_chans_{subject}_{condition_name}.json')
    
    # Save the dictionary as a JSON file
    with open(filename, 'w') as file:
        json.dump(data, file)
    
    print(f'Saved significant channels for subject {subject} and {condition_name} to {filename}')

def initialize_output_data(rois, condition_names):
    """
    Initialize dictionaries for storing data across different conditions and ROIs.
    """
    return {condition_name: {roi: [] for roi in rois} for condition_name in condition_names}

def process_data_for_roi(subjects_mne_objects, condition_names, rois, subjects, sig_electrodes_per_subject_roi, time_indices):
    """
    Processes and aggregates electrophysiological data for specified regions of interest (ROIs)
    across multiple subjects and conditions.

    This function iterates through subjects, ROIs, and conditions to extract and
    process MNE Epochs data. For each subject and ROI, it selects significant
    electrodes and computes trial averages, trial standard deviations, and
    time-windowed averages of power data (specifically 'HG_ev1_power_rescaled').
    It also generates mappings of electrodes to subjects and ROIs.

    Parameters:
    ----------
    subjects_mne_objects : dict
        A nested dictionary containing MNE Epochs objects.
        Structure: `subjects_mne_objects[subject_id][condition_name][mne_object_type]`
        where `mne_object_type` is typically 'HG_ev1_power_rescaled'.
    condition_names : list of str
        A list of condition names (e.g., ['ConditionA', 'ConditionB']) to process.
    rois : list of str
        A list of ROI names (e.g., ['dlPFC', 'ACC']) to analyze.
    subjects : list of str
        A list of subject identifiers (e.g., ['D0057', 'D0059']).
    sig_electrodes_per_subject_roi : dict
        A nested dictionary mapping ROIs to subjects and their significant electrodes.
        Structure: `sig_electrodes_per_subject_roi[roi_name][subject_id] = list_of_significant_electrode_names`.
    time_indices : dict
        A dictionary defining time windows for averaging.
        Keys are descriptive names (e.g., 'firstHalfSecond', 'fullSecond').
        Values are tuples of (start_index, end_index) for slicing the time axis of epochs data.
        Example: `{'firstHalfSecond': (0, 50), 'secondHalfSecond': (50, 100)}`.

    Returns:
    -------
    tuple
        A tuple containing five elements:
        1. data_trialAvg_lists (dict):
           Nested dictionary storing trial-averaged data.
           Structure: `data_trialAvg_lists[condition_name][roi_name] = list_of_trial_avg_arrays`.
           Each array in the list corresponds to a subject's trial-averaged data for the
           significant electrodes in that ROI and condition.
        2. data_trialStd_lists (dict):
           Nested dictionary storing trial standard deviation data.
           Structure: `data_trialStd_lists[condition_name][roi_name] = list_of_trial_std_arrays`.
        3. data_timeAvg_lists (dict):
           Nested dictionary storing time-windowed averaged data.
           Structure: `data_timeAvg_lists[time_window_suffix][condition_name][roi_name] = list_of_time_avg_arrays`.
        4. overall_electrode_mapping (list of tuples):
           A list where each tuple contains `(subject_id, roi_name, electrode_name, overall_index)`.
           This provides a flat mapping of all processed significant electrodes.
        5. electrode_mapping_per_roi (dict):
           A dictionary where keys are ROI names. Each value is a list of tuples:
           `(subject_id, electrode_name, roi_specific_index)`.
           This maps significant electrodes within each ROI.

    Notes:
    -----
    - The function expects 'HG_ev1_power_rescaled' data within `subjects_mne_objects`.
    - If a subject has no significant electrodes for a given ROI, they are skipped for that ROI.
    - `filter_and_average_epochs` is used internally for calculations.
    - Trial averages and standard deviations are computed across all time points for the selected epochs.
    - Time averages are computed for the specified `time_indices`.
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
    Concatenates lists of NumPy arrays (typically representing data from multiple
    subjects) for each Region of Interest (ROI) and experimental condition.

    This function is often used after `process_data_for_roi` to combine
    subject-specific data arrays (e.g., trial averages, time-windowed averages)
    into single arrays per ROI and condition for group-level analysis or plotting.

    Parameters:
    ----------
    data_lists : dict
        A nested dictionary containing lists of NumPy arrays to be concatenated.
        Expected structure: `data_lists[condition_name][roi_name] = list_of_numpy_arrays`.
        Each array in the list typically corresponds to data from one subject.
        The concatenation is performed along `axis=0`.
    rois : list of str
        A list of ROI names (e.g., ['dlPFC', 'ACC']) for which data should be concatenated.
        These ROI names must be keys in the inner dictionaries of `data_lists`.
    condition_names : list of str
        A list of condition names (e.g., ['ConditionA', 'ConditionB']) for which data
        should be concatenated. These condition names must be keys in the outer
        dictionary of `data_lists`.

    Returns:
    -------
    dict
        A nested dictionary with the same structure as `data_lists`, but where
        each list of NumPy arrays has been concatenated into a single NumPy array.
        Structure: `concatenated_data[condition_name][roi_name] = concatenated_numpy_array`.
        The shape of the `concatenated_numpy_array` will be (total_entities_across_subjects, ...),
        where `total_entities_across_subjects` is the sum of the 0-th dimension of
        the input arrays.

    Example:
    -------
    >>> data_lists = {
    ...     'ConditionA': {
    ...         'dlPFC': [np.random.rand(10, 5), np.random.rand(12, 5)], # Data for 2 subjects
    ...         'ACC': [np.random.rand(8, 5), np.random.rand(11, 5)]
    ...     }
    ... }
    >>> rois = ['dlPFC', 'ACC']
    >>> condition_names = ['ConditionA']
    >>> concatenated = concatenate_data(data_lists, rois, condition_names)
    >>> concatenated['ConditionA']['dlPFC'].shape
    (22, 5)
    >>> concatenated['ConditionA']['ACC'].shape
    (19, 5)
    """
    concatenated_data = {condition_name: {roi: np.concatenate(data_lists[condition_name][roi], axis=0) for roi in rois} for condition_name in condition_names}
    return concatenated_data

def calculate_mean_and_sem(concatenated_data, rois, condition_names):
    """
    Calculates the mean and Standard Error of the Mean (SEM) across the first
    axis (typically electrodes or trials from concatenated subject data) for
    each Region of Interest (ROI) and experimental condition.

    This function is typically used after `concatenate_data`, where subject-level
    data has been combined into a single array per ROI and condition. The mean and
    SEM are then computed over the aggregated data (e.g., across all significant
    electrodes from all subjects within an ROI).

    Parameters:
    ----------
    concatenated_data : dict
        A nested dictionary containing concatenated NumPy arrays.
        Expected structure: `concatenated_data[condition_name][roi_name] = concatenated_numpy_array`.
        The `concatenated_numpy_array` usually has dimensions like (n_electrodes_or_trials, n_timepoints).
    rois : list of str
        A list of ROI names (e.g., ['dlPFC', 'ACC']) for which mean and SEM should be calculated.
        These ROI names must be keys in the inner dictionaries of `concatenated_data`.
    condition_names : list of str
        A list of condition names (e.g., ['ConditionA', 'ConditionB']) for which
        mean and SEM should be calculated. These condition names must be keys in the
        outer dictionary of `concatenated_data`.

    Returns:
    -------
    dict
        A nested dictionary storing the calculated mean and SEM.
        Structure: `mean_and_sem[roi_name][condition_name] = {'mean': mean_array, 'sem': sem_array}`.
        - `mean_array`: NumPy array representing the mean across the 0-th axis of the input data.
        - `sem_array`: NumPy array representing the SEM across the 0-th axis of the input data.
        If the input data for a specific ROI/condition is empty or unsuitable for calculation (e.g., too few samples),
        the 'mean' and 'sem' might be NaNs or empty arrays, depending on `np.nanmean` and `np.std` behavior.

    Example:
    -------
    >>> concatenated_data = {
    ...     'ConditionA': {
    ...         'dlPFC': np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), # 3 electrodes, 3 timepoints
    ...         'ACC': np.array([[5, 6], [6, 7]])
    ...     }
    ... }
    >>> rois = ['dlPFC', 'ACC']
    >>> condition_names = ['ConditionA']
    >>> results = calculate_mean_and_sem(concatenated_data, rois, condition_names)
    >>> results['dlPFC']['ConditionA']['mean']
    array([2., 3., 4.])
    >>> results['dlPFC']['ConditionA']['sem'] # Example SEM values
    array([0.57735027, 0.57735027, 0.57735027])
    """
    mean_and_sem = {roi: {condition_name: {} for condition_name in condition_names} for roi in rois}
    for roi in rois:
        for condition_name in condition_names:
            trial_data = concatenated_data[condition_name][roi]
            mean = np.nanmean(trial_data, axis=0)
            sem = np.std(trial_data, axis=0, ddof=1) / np.sqrt(trial_data.shape[0])
            mean_and_sem[roi][condition_name] = {'mean': mean, 'sem': sem}
    return mean_and_sem

def calculate_time_perm_cluster_for_each_roi(concatenated_data, rois, condition_names, alpha=0.05, n_jobs=6):
    """
    Perform time permutation cluster tests between the first two outputs for each ROI.
    Assumes that there are at least two output conditions to compare.
    """
    time_perm_cluster_results = {}
    for roi in rois:
        time_perm_cluster_results[roi] = time_perm_cluster(
            concatenated_data[condition_names[0]][roi],
            concatenated_data[condition_names[1]][roi], alpha, n_jobs=n_jobs
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


def perform_modular_anova(df, time_window, conditions, save_dir, save_name):
    """
    Performs an ANOVA test on a filtered subset of the provided DataFrame
    for a specific time window and saves the results to a text file.

    The ANOVA model formula is dynamically constructed based on the keys
    found in the conditions dictionary. It includes main effects
    and interaction terms for these conditions. The dependent variable is
    assumed to be 'MeanActivity'.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data for ANOVA. Must include
        a 'TimeWindow' column to filter by `time_window` and a 'MeanActivity'
        column as the dependent variable. It should also contain columns
        corresponding to the condition keys derived from `condition_names_conditions`.
    time_window : str
        The specific value in the 'TimeWindow' column of `df` to filter the
        DataFrame for this ANOVA (e.g., 'FirstHalfSecond').
    conditions : dict
        A dictionary defining the experimental conditions. The keys of
        the inner dictionary (for any given condition name) are used to
        construct the formula terms related to experimental conditions.
        Example: `{'Stimulus_cr': {'congruency': 'c', 'switchType': 'r'}}`
    save_dir : str
        The absolute path to the directory where the ANOVA results file
        will be saved.
    save_name : str
        The name of the text file (e.g., 'anova_results_time_window_X.txt')
        to save the ANOVA results.

    Returns:
    -------
    statsmodels.iolib.table.SimpleTable or anova_lm object
        The ANOVA results table object from `statsmodels.stats.anova.anova_lm`.

    Notes:
    -----
    - The function uses a Type II ANOVA (`typ=2`).
    - The formula is constructed as:
      `MeanActivity ~ C(key1) + C(key2) + ... + C(key1)*C(key2)*...`
    - Ensure that `df` contains all necessary columns specified by the
      dynamically generated formula and the 'MeanActivity' and 'TimeWindow' columns.
    """
    # Filter for a specific time window (I should probably make this not have a time_window input and just loop over all time windows like the within electrode code does)
    df_filtered = df[df['TimeWindow'] == time_window]

    # Dynamically construct the model formula based on condition keys
    condition_keys = [key for key in conditions[next(iter(conditions))].keys()]
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
    # Determine LAB_root based on the operating system and environment
    if LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)
        
        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            # Check if we're on the cluster by looking for /cwork directory
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                # Fallback for other Linux systems
                LAB_root = os.path.join(HOME, "CoganLab")
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


def make_sig_electrodes_per_subject_and_roi_dict(rois_dict, subjects_electrodes_to_ROIs_dict, sig_chans_per_subject):
    """
    Processes electrodes by ROI and filters significant electrodes.

    Parameters:
    - rois_dict: A dictionary mapping each region of interest (ROI) to a list of brain regions.
    - subjects_electrodes_to_ROIs_dict: A dictionary mapping subjects to their electrode-to-ROI assignments.
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
            subjects_electrodes_to_ROIs_dict, sig_chans_per_subject, roi_regions)
        
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
    print(len(good.info['bads']))
    good.drop_channels(good.info['bads'])
    good.load_data()

    # Set EEG reference
    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    return good

def count_electrodes_across_subjects(data, subjects):
    total_electrodes = 0
    for subject, details in data.items():
        if subject in subjects:
            total_electrodes += len(details['default_dict'])
    return total_electrodes

def get_trials(data: mne.io.Raw, events: list[str], times: tuple[float, float], mark_outliers_as_nan=True) -> mne.Epochs:
    """
    Extract and concatenate non-outlier trials for specified events.

    This function extracts epochs for each event specified in `events` from the raw EEG data
    over a time window defined by `times` (with an extra 0.5-second padding on both sides).
    The resulting epochs are concatenated into a single Epochs object, and outlier data points
    are marked as NaN.

    Parameters
    ----------
    data : mne.io.Raw
        The preprocessed raw EEG data.
    events : list of str
        A list of event names to extract trials for.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    mark_outliers_as_nan : bool
        Whether to set outlier timepoints to NaNs or not
    Returns
    -------
    all_trials : mne.Epochs
        The concatenated epochs for all specified events with outliers marked as NaN.

    Examples
    --------
    >>> # Assume 'raw_data' is a preprocessed mne.io.Raw object containing event annotations.
    >>> events = ['Stimulus/c25', 'Stimulus/c75']
    >>> times = (-0.5, 1.5)
    >>> epochs = get_trials(raw_data, events, times)
    >>> isinstance(epochs, mne.Epochs)
    True
    """
    all_trials_list = []

    for event in events:
        # Adjust times for 0.5s padding before and after the epoch
        times_adj = [times[0] - 0.5, times[1] + 0.5]
        trials = trial_ieeg(data, event, times_adj, preload=True,
                            reject_by_annotation=False)
        all_trials_list.append(trials)

    # Concatenate all trials
    all_trials = mne.concatenate_epochs(all_trials_list)
    print(len(all_trials))
    
    if mark_outliers_as_nan:
        # Mark outliers as NaN
        outliers_to_nan(all_trials, outliers=10)
    
    return all_trials

def get_trials_with_outlier_analysis(data: mne.io.Raw, events: list[str], times: tuple[float, float], 
                                     outlier_threshold: float = 10, create_outlier_plots=False, mark_outliers_as_nan=True, save_dir=None) -> mne.Epochs:
    """
    Extract and concatenate non-outlier trials for specified events with detailed outlier analysis.
    
    This enhanced version provides detailed statistics about outliers across trials and channels,
    understanding that outliers_to_nan marks individual timepoints, not entire trials.
    
    Parameters:
    -----------
    data : mne.io.Raw
        The preprocessed raw EEG data.
    events : list of str
        A list of event names to extract trials for.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    outlier_threshold : float
        Number of standard deviations for outlier detection (default: 10)
    mark_outliers_as_nan : bool
        Whether to set outlier timepoints to NaN
    save_dir : str
        Save directory for plots and prints
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import label
    
    all_trials_list = []

    for event in events:
        # Adjust times for 0.5s padding before and after the epoch
        times_adj = [times[0] - 0.5, times[1] + 0.5]
        trials = trial_ieeg(data, event, times_adj, preload=True,
                            reject_by_annotation=False)
        all_trials_list.append(trials)

    # Concatenate all trials
    all_trials = mne.concatenate_epochs(all_trials_list)
    print(f"\nTotal trials before outlier marking: {len(all_trials)}")


    print(f"time range: {all_trials.tmin} to {all_trials.tmax}") # check this
    # Get data before marking outliers
    data_before = all_trials.get_data().copy()
    
    if mark_outliers_as_nan:
        # Mark outliers as NaN (using threshold of 10 SD by default)
        outliers_to_nan(all_trials, outliers=outlier_threshold)
        
        # Get data after marking outliers
        data_after = all_trials.get_data()
        
        # === OUTLIER ANALYSIS ===
        print("\n" + "="*60)
        print("OUTLIER ANALYSIS")
        print("="*60)
        
        # Identify where NaNs were introduced (these are the outliers)
        outlier_mask = np.isnan(data_after) & ~np.isnan(data_before)
        outlier_trial_mask = np.all(outlier_mask, axis=2)

        events_str = '-'.join(events).replace('/', '_')
        plot_filename = f'{events_str}_outlier_analysis.png'
        full_save_path = os.path.join(save_dir, plot_filename)

        sfreq = all_trials.info['sfreq'] # sampling frequency
        n_trials, n_channels, n_timepoints = outlier_mask.shape

        # Per-channel analysis
        outlier_trials_per_channel = np.sum(outlier_trial_mask, axis=0)
        channels_with_outliers = outlier_trials_per_channel > 0
        n_channels = len(outlier_trials_per_channel)
        n_channels_with_outliers = np.sum(channels_with_outliers)
        print(f"\nPer-Channel Statistics:")
        print(f"  Number of channels: {n_channels}")
        print(f"  Number of channels with outliers: {n_channels_with_outliers}")
        
        if np.any(channels_with_outliers):
            print(f"  For channels WITH outliers:")
            print(f"    Mean outlier trials: {np.mean(outlier_trials_per_channel):.1f}")
            print(f"    Median outlier trials: {np.median(outlier_trials_per_channel):.1f}")
            print(f"    Max outlier trials: {np.max(outlier_trials_per_channel):.1f}")
        
        # Identify problematic channels
        worst_channels = np.argsort(outlier_trials_per_channel)[-5:][::-1]
        channel_names = [all_trials.ch_names[i] for i in worst_channels]
        worst_channel_times_ms = outlier_trials_per_channel[worst_channels]
        print(f"\n  Top 5 worst channels: {channel_names}")
        print(f"  Their outlier times (ms): {[f'{t:.3f}' for t in worst_channel_times_ms]}")
    
        # 4. Impact assessment
        print(f"\n--- Impact Assessment ---")
        
        # How many trials would be lost if we drop any trial with outliers?
        print(f"If dropping outlier trials: {np.sum(outlier_trials_per_channel)}/{n_trials*n_channels} channel-trials lost ({100*np.sum(outlier_trials_per_channel)/(n_trials*n_channels):.1f}%)")
        
        # How many channels have >1% outlier trials?
        outlier_trials_per_channel_percentage = outlier_trials_per_channel / n_trials * 100
        print(f"Channels with >1% outlier trials: {np.sum(outlier_trials_per_channel_percentage > 1.0)}/{n_channels} channel-trials lost ({100*np.sum(outlier_trials_per_channel_percentage > 1.0)/n_channels:.1f}%)")

        # How many channels have >5% outlier trials?
        very_high_outlier_trials = outlier_trials_per_channel_percentage > 5.0
        n_very_high_outlier_trials = np.sum(very_high_outlier_trials)
        print(f"Channels with >5% outliers: {n_very_high_outlier_trials}/{n_channels} channel-trials lost ({100*n_very_high_outlier_trials/n_channels:.1f}%)")
        
        # 5. Create visualization
        if create_outlier_plots:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'Outlier Distribution Analysis (Threshold: {outlier_threshold} SD)', fontsize=14, fontweight='bold')
            
            # Plot 1: Histogram of outlier duration per channel
            axes[0].hist(outlier_trials_per_channel)
            axes[0].set_xlabel('Number of Outlier Trials')
            axes[0].set_ylabel('Number of Channels')
            axes[0].set_title('Number of Outlier Trials per Channel')
            
            # Plot 2: Heatmap of outliers (trials x channels)
            custom_cmap = ListedColormap(['black', 'yellow'])
            im = axes[1].imshow(outlier_trial_mask, aspect='auto', cmap=custom_cmap)
            axes[1].set_xlabel('Channel Index')
            axes[1].set_ylabel('Trial Index')
            axes[1].set_title('Outlier Trials Across Channels ')
            
            # Plot 3: Outlier trial percentage distribution
            if len(outlier_trials_per_channel_percentage) > 0:
                axes[2].hist(outlier_trials_per_channel_percentage)
                axes[2].set_xlabel('Percentage of Outlier Trials')
                axes[2].set_ylabel('Number of Channels')
                axes[2].set_title('Outlier Trial Percentage per Channel')
            
            plt.tight_layout()
            print(f"Attempting to save plot to: {full_save_path}")
            plt.savefig(full_save_path)
            print("Plot save command executed successfully.")
            plt.close(fig)
        
        print("="*60 + "\n")
    
    return all_trials

def handle_outliers(trials: mne.epochs.BaseEpochs, 
                    outliers: float,
                    outlier_policy: str = 'nan',
                    copy: bool = False, 
                    picks: list = 'data',
                    tmin: int | float = None,
                    tmax: int | float = None, 
                    threshold_percent = 5,
                    verbose=None) -> mne.epochs.BaseEpochs:
    """
    Identifies and handles outliers in epoched data.

    Parameters
    ----------
    trials : mne.epochs.BaseEpochs
        The trials to process for outliers.
    outliers : float
        The number of deviations above the mean to be considered an outlier.
    outlier_policy : str, optional
        How to handle identified outliers. Options are:
        - 'nan': Replace outliers with np.nan (default).
        - 'drop_and_impute': Drop channels with more outlier trials than a percentage threshold, and impute the remaining outlier trials with the channel mean across trials.
        - 'ignore': Do not modify the data.
    copy : bool, optional
        Whether to copy the data, by default False.
    picks : list, optional
        The channels to remove outliers from, by default 'data'.
    tmin : int | float, optional
        Start time of the window to check for outliers.
    tmax : int | float, optional
        End time of the window to check for outliers.
    threshold_percent : int | float, optional
        Channels with a greater percent of outlier trials than this threshold will be removed from further analyses.
    %(verbose)s

    Returns
    -------
    mne.epochs.BaseEpochs
        The trials with outliers handled according to the specified policy.
    """
    if outlier_policy not in ['nan', 'drop_and_impute', 'ignore']:
        raise ValueError("outlier_policy must be 'nan', 'drop_and_impute', or 'ignore'")

    if copy:
        trials = trials.copy()
        
    # If the policy is 'ignore', we're done.
    if outlier_policy == 'ignore':
        mne.utils.logger.info("Outlier policy set to 'ignore'. No changes made to data.")
        return trials, []
    
    # 1. Mark outlier timepoints as NaN directly on the Epochs object
    print(f"Marking outliers > {outliers} SD as NaN...")
    outliers_to_nan(trials, outliers=outliers)
    
    # 2. If the policy is 'drop and impute', then drop bad channels, and impute remaining outlier trials
    dropped_channels = []
    if outlier_policy == 'drop_and_impute':
        mne.utils.logger.info(f"Dropping channels with more than {threshold_percent}% outlier trials, and imputing the remaining outlier trials with the channel mean.")

        # identify and drop bad channels
        print(f"identifying channels with >{threshold_percent}% trial outliers")
        bad_channels = identify_bad_channels_by_trial_nan_rate(trials, threshold_percent)
        if bad_channels:
            print(f"bad channels: {bad_channels}")
            trials.drop_channels(bad_channels, on_missing='ignore')
            print(f" dropped {len(bad_channels)} bad channels")
        
        # impute remaining nan trials within the good channels
        print("imputing remaining nan trials")
        impute_trial_nans_by_channel_mean(trials)

    return trials, dropped_channels

def identify_bad_channels_by_trial_nan_rate(epochs: mne.Epochs, threshold_percent: float = 5.0) -> list:
    """
    Identifies channels where the percentage of trials with NaNs exceeds a threshold.
    A trial is counted if it has one or more NaN values.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]

    # Check for any NaNs along the time axis for each trial and channel
    trial_has_nan = np.isnan(data).any(axis=2)

    # Count NaN trials per channel and calculate the percentage
    nan_percentages = np.sum(trial_has_nan, axis=0) * 100 / n_epochs

    # Find channels exceeding the threshold
    bad_channel_indices = np.where(nan_percentages > threshold_percent)[0]
    
    all_channel_names = np.array(epochs.ch_names)
    bad_channels = all_channel_names[bad_channel_indices].tolist()

    if bad_channels:
        print(f"Found {len(bad_channels)} channels with > {threshold_percent}% outlier trials: {bad_channels}")
    else:
        print(f"No channels found with > {threshold_percent}% outlier trials.")
        
    return bad_channels

def impute_trial_nans_by_channel_mean(epochs: mne.Epochs):
    """
    Imputes NaN values in Epochs data on a per-channel, per-timepoint basis.

    This function modifies the MNE Epochs object in-place.
    """
    data = epochs.get_data() # Shape: (n_epochs, n_channels, n_times)

    # Iterate through each channel to process it independently
    for ch_idx in range(data.shape[1]):
        channel_data = data[:, ch_idx, :] # Shape: (n_epochs, n_times)
        
        # Find if this channel has any NaNs at all
        if not np.isnan(channel_data).any():
            continue # Skip if there's nothing to impute

        # Calculate the mean for each time point across all trials, ignoring NaNs
        # Result shape: (n_times,)
        mean_across_trials = np.nanmean(channel_data, axis=0)

        # Find the indices (epoch, time) of NaNs within this channel's data
        nan_indices = np.where(np.isnan(channel_data))

        # Replace each NaN with the mean of its corresponding time point
        channel_data[nan_indices] = np.take(mean_across_trials, nan_indices[1])
        
        # Put the imputed data back into the main data array
        data[:, ch_idx, :] = channel_data
    
    # Update the epochs object with the cleaned data
    epochs._data = data
    print("NaN values have been imputed using the per-channel, per-timepoint mean.")