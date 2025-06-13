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

def main(subject_id):
    try:
        # Set up paths
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)
        task = 'GlobalLocal'

        # get box directory depending on OS
        LAB_root = os.path.join("/cwork", USER)

        # Load Data using the first 'get_data' function
        layout = get_data(task, root=LAB_root)

        # Use the second 'get_good_data' function, with the passed-in subject_id
        print(f"Loading good data for subject: {subject_id}")
        good_data = get_good_data(subject_id, layout)

        events_to_count = {}
        events_to_count['ErrorTrials_Stimulus_Locked'] = ["Stimulus/Accuracy0.0"] 
        events_to_count['CorrectTrials_Stimulus_Locked'] = ["Stimulus/Accuracy1.0"]

        trial_counts = {}

        # count trials
        for condition_name, event_list in events_to_count.items():
            to_find = event_list[0]

            count = np.sum([to_find in desc for desc in good_data.annotations.description])
            
            trial_counts[condition_name] = count
            print(f"Found {count} trials for {condition_name} ('{event_list}')")

        #edit to save to right pathway
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trial_counts')
        os.makedirs(output_dir, exist_ok=True)
        counts_filename = os.path.join(output_dir, f'{subject_id}_counts.csv')

        with open(counts_filename, 'w') as f:
                f.write("condition,trial_count\n")
                for condition, count in trial_counts.items():
                    f.write(f"{condition},{count}\n")
        print(f"\nSuccessfully saved final counts to {counts_filename}")

    except Exception as e:
        print(f"A critical error occurred for subject {subject_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count error vs correct for a given subject.")
    parser.add_argument('--subject', type=str, required=True, 
                        help='The subject ID to processs')
    args = parser.parse_args()
    main(args.subject)
