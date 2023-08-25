import mne
import json
import numpy as np
import os

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
