## count trials dcc
#comment to commit

import os
import sys
import argparse

print(sys.path)

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

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale

from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from src.analysis.utils.general_utils import get_good_data
from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets

def main(subject_id: str):
    """
    Loads data for a single subject, parses hierarchical event names,
    counts trial types, and saves the result.
    """
    try:
        print(f"--- Processing subject: {subject_id} ---")
        
        # --- 1. Set up paths and load data ---
        task = 'GlobalLocal'
        current_user = os.path.basename(os.path.expanduser("~"))
        LAB_root = os.path.join("/cwork", current_user, "BIDS-1.1_GlobalLocal")
        layout = get_data(task, root=LAB_root)
        raw_data = raw_from_layout(layout, subject=subject_id, preload=True)
        print("Data loaded successfully.")

        # --- 2. Loop through annotations, parse them, and count ---
        print("Parsing annotations and counting trials...")
        correct_count = 0
        error_count = 0

        for description in raw_data.annotations.description:
            entities = get_entities_from_path(description)
            
            if 'Accuracy' in entities:
                if entities['Accuracy'] == '1.0':
                    correct_count += 1
                elif entities['Accuracy'] == '0.0':
                    error_count += 1
        
        # --- 3. Store and print the final counts ---
        trial_counts = {
            'CorrectTrials': correct_count,
            'ErrorTrials': error_count
        }
        print("\n--- Final Counts ---")
        print(trial_counts)

        # --- 4. Save the results to a CSV file ---
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trial_counts')
        os.makedirs(save_dir, exist_ok=True)
        counts_filename = os.path.join(save_dir, f'{subject_id}_trial_counts.csv')
        
        with open(counts_filename, 'w') as f:
            f.write("condition,trial_count\n")
            for condition, count in trial_counts.items():
                f.write(f"{condition},{count}\n")
        print(f"\nSuccessfully saved final counts to {counts_filename}")

    except Exception as e:
        print(f"A critical error occurred for subject {subject_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count trials by parsing BIDS event descriptions.")
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()
    main(args.subject)