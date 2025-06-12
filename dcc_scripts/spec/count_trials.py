import os
import argparse
import pandas as pd
import numpy as np
import mne
from ieeg.io import get_data, raw_from_layout
from mne_bids import get_entities_from_path

def count_trials_for_subject(subject_id, layout):
    """Loads data for one subject and returns the trial counts."""
    print(f"---> Processing subject: {subject_id}")
    try:
        # Load all runs for the subject at once
        raw_data = raw_from_layout(layout, subject=subject_id, preload=True)
        print(f"     Data loaded for {subject_id}.")

        correct_count = 0
        error_count = 0

        # Loop through all event descriptions and parse them
        for description in raw_data.annotations.description:
            entities = get_entities_from_path(description)
            
            if 'Accuracy' in entities:
                if entities['Accuracy'] == '1.0':
                    correct_count += 1
                elif entities['Accuracy'] == '0.0':
                    error_count += 1
        
        print(f"     Found {correct_count} correct, {error_count} error trials.")
        return {'subject': subject_id, 'correct_trials': correct_count, 'error_trials': error_count}

    except Exception as e:
        print(f"     Could not process subject {subject_id}. Error: {e}")
        return {'subject': subject_id, 'correct_trials': 'ERROR', 'error_trials': 'ERROR'}


if __name__ == "__main__":
    # --- 1. Set up argument parser to get the data path ---
    parser = argparse.ArgumentParser(description="Count correct and error trials for all subjects in a BIDS dataset.")
    parser.add_argument('--bids-root', type=str, required=True,
                        help='The full path to the root of your BIDS dataset on your local machine.')
    args = parser.parse_args()

    # --- 2. Load the BIDS layout and find all subjects ---
    try:
        bids_layout = get_data('GlobalLocal', root=args.bids_root)
        subjects = bids_layout.get(return_type='id', target='subject')
        print(f"Found {len(subjects)} subjects: {subjects}")
    except Exception as e:
        print(f"Error loading BIDS layout at {args.bids_root}. Please check the path. Error: {e}")
        sys.exit(1) # Exit the script if the data can't be found
    
    # --- 3. Loop through subjects and collect results ---
    all_results = []
    for sub_id in subjects:
        result = count_trials_for_subject(sub_id, bids_layout)
        all_results.append(result)

    # --- 4. Save all results to a single CSV file ---
    results_df = pd.DataFrame(all_results)
    output_filename = "all_subjects_trial_counts.csv"
    results_df.to_csv(output_filename, index=False)

    print("\n--- All Done! ---")
    print(f"Saved summary of all subjects to: {output_filename}")