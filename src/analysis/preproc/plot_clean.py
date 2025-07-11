# %% [markdown]
# this is an attempt at turning plot_clean.ipynb into a script
# # Example line noise filtering script
# 
# Filters the 60Hz line noise from the data, as well as the harmonics. Includes
# environment checks for SLURM jobs for convenience
# 

# %%
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to access modules in project root
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

import mne
import matplotlib.pyplot as plt
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative, update, bidspath_from_layout
from ieeg import viz
from bids import BIDSLayout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.viz.ensemble import figure_compare
from mne_bids import read_raw_bids

def fix_events_file(events_file):
    """Fix events file by removing events with negative durations
    
    Parameters
    ----------
    events_file : str
        Path to events.tsv file
    
    Returns
    -------
    bool
        True if file was modified, False otherwise
    """
    if not os.path.exists(events_file):
        return False
    
    # Load the events file
    events_df = pd.read_csv(events_file, sep='\t')
    
    # Check if 'duration' column exists and contains negative values
    if 'duration' in events_df.columns and (events_df['duration'] < 0).any():
        print(f"Found {(events_df['duration'] < 0).sum()} events with negative durations in {events_file}")
        
        # Create a backup of the original file
        backup_file = str(events_file) + '.bak'
        if not os.path.exists(backup_file):
            os.rename(events_file, backup_file)
            print(f"Created backup of original events file at {backup_file}")
            
            # Remove events with negative durations
            events_df = events_df[events_df['duration'] >= 0]
            
            # Write the modified events file
            events_df.to_csv(events_file, sep='\t', index=False)
            print(f"Removed events with negative durations and saved to {events_file}")
            return True
    
    return False

def main(subjects_list):
    """Main function to process and plot data for a list of subjects
    
    Parameters
    ----------
    subjects_list : list
        List of subject IDs to process
    """
    # Set up paths
    HOME = os.path.expanduser("~")
    task = 'GlobalLocal'
    
    # get box directory depending on OS
    if os.name == 'nt': # windows
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    else: # mac
        LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
    
    # Load Data
    layout = get_data(task, root=LAB_root)
    
    for subj in subjects_list:
        print(f"Processing subject: {subj}")
        
        # Find all events files for this subject and fix them if needed
        events_files = []
        for run in layout.get(subject=subj, return_type='id', target='run'):
            try:
                # Get the path to the BIDS file
                bids_path = bidspath_from_layout(layout, subject=subj, run=run, extension=".edf")
                # Get the corresponding events file
                events_file = bids_path.update(suffix='events', extension='.tsv').fpath
                events_files.append(events_file)
            except Exception as e:
                print(f"Error finding events file for subject {subj}, run {run}: {e}")
        
        # Fix all events files for this subject
        modified = False
        for events_file in events_files:
            if fix_events_file(events_file):
                modified = True
        
        if modified:
            print(f"Fixed events files for subject {subj}, now loading data...")
        
        # Load the raw data without excluding any channels
        actual_raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)
        
        # Make a copy for processing and comparison
        raw = actual_raw.copy()
        
        # Filter out negative durations from annotations (aka negative RTs)
        if raw.annotations is not None and len(raw.annotations) > 0:
            # Find indices of annotations with non-negative durations
            valid_indices = np.where(raw.annotations.duration >= 0)[0]
            
            # Create a new annotations object with only valid durations
            raw.set_annotations(mne.Annotations(
                onset=raw.annotations.onset[valid_indices],
                duration=raw.annotations.duration[valid_indices],
                description=raw.annotations.description[valid_indices],
                orig_time=raw.annotations.orig_time
            ))

        # this is to exclude the eeg channels if needed
        # List of channels you want to exclude. Uncomment if needed.
        # channels_to_exclude = ['T5', 'T6', 'FZ', 'CZ', 'PZ', 'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', '02', 'F7', 'F8', 'T3', 'T4']
        # raw.drop_channels(channels_to_exclude)
        
        # Plot raw data and save it
        fig_raw = figure_compare([actual_raw], 
                              labels=["Raw"], 
                              avg=True, 
                              n_jobs=6, 
                              verbose=10, 
                              proj=True, 
                              fmax=250)
        
        # Save the raw plot if the figure was created successfully
        if fig_raw is not None:
            raw_filename = f"{subj}_raw.png"
            save_path = os.path.join(layout.root, 'derivatives', 'clean', raw_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
            fig_raw.savefig(save_path)
            plt.close(fig_raw)
            print(f"Saved raw plot to {save_path}")
        else:
            print(f"Warning: Could not generate raw plot for subject {subj}")
        
        # Filter Data
        # A filter length of 700 ms does a good job of removing 60Hz line noise
        line_filter(raw,
                    mt_bandwidth=10.,
                    n_jobs=6,
                    filter_length='700ms',
                    verbose=10,
                    freqs=[60, 120, 180],
                    notch_widths=20,
                    copy=False)

        # filter again to get rid of harmonics (120Hz, 180Hz, 240Hz)
        line_filter(raw,
                    mt_bandwidth=10.,
                    n_jobs=6,
                    filter_length='20000ms',
                    verbose=10,
                    freqs=[60],
                    notch_widths=20,
                    copy=False)
        
        # Plot the comparison between raw and filtered data
        fig_compare = figure_compare([actual_raw, raw],
                             labels=["Raw", "Filtered"],
                             avg=True,
                             n_jobs=6,
                             verbose=10,
                             proj=True,
                             fmax=250)
        
        # Save the filtered comparison plot if the figure was created successfully
        if fig_compare is not None:
            filtered_filename = f"{subj}_700msAt60_120_180_20000msAt60.png"
            save_path = os.path.join(layout.root, 'derivatives', 'clean', filtered_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
            fig_compare.savefig(save_path)
            plt.close(fig_compare)
            print(f"Saved filtered comparison plot to {save_path}")
        else:
            print(f"Warning: Could not generate comparison plot for subject {subj}")
        
        # Save the filtered data
        save_derivative(raw, layout, "clean", True)
        
        # Mark channel outliers
        channel_outlier_marker(raw, 3, 2, save=True)

# For command line execution
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Process GlobalLocal data for specified subjects")
    parser.add_argument('--subjects', nargs='+', help='List of subject IDs to process', default=['D0121'])
    args = parser.parse_args()
    
    # Run the main function with provided subjects
    main(args.subjects)
    
    print("Processing complete!")