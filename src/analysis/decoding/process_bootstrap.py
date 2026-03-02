"""
process_bootstrap.py

Contains the process_bootstrap() function and its helper run_pooled_shuffle_for_roi(),
which together handle the generation and processing of a single bootstrap sample for
time-window decoding analysis.

process_bootstrap() is designed to be called in parallel via joblib (see decoding_dcc.py).
Each call generates an independent bootstrapped dataset, runs the sliding-window decoding
pipeline across all ROIs and condition comparisons, computes true and shuffle accuracies,
and returns a results dictionary that is later aggregated across bootstraps by the caller.
"""

import sys
import os

print(sys.path)

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
# Get the absolute path to the directory containing the current script
try:
    # This will work if running as a .py script
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at the beginning to prioritize it

import numpy as np
from src.analysis.config import experiment_conditions
from src.analysis.config.condition_registry import get_pooled_shuffle_settings

from src.analysis.utils.labeled_array_utils import (
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel,
    concatenate_conditions_by_string,
)
from src.analysis.decoding.decoding import (
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    get_time_averaged_confusion_matrix,
    make_pooled_shuffle_distribution,
)


def run_pooled_shuffle_for_roi(roi, roi_labeled_arrays, strings_to_find, args, random_state):
    """
    Helper function to run make_pooled_shuffle_distribution with standard args.
    """
    return make_pooled_shuffle_distribution(
        roi=roi,
        roi_labeled_arrays=roi_labeled_arrays,
        strings_to_find_pooled=strings_to_find,
        explained_variance=args.explained_variance,
        n_splits=args.n_splits,
        n_perm=args.n_shuffle_perms,
        random_state=random_state,
        balance_method='subsample', # Subsampling is recommended for pooling
        obs_axs=args.obs_axs,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
def process_bootstrap(bootstrap_idx, subjects_mne_objects, args, rois, condition_names, electrodes, condition_comparisons, save_dir):
    """
    Generates and processes a single bootstrap sample.
    This function is designed to be called in parallel by joblib.
    This is meant to be called in decoding_dcc.py.
    """
    # use a unique random state for each bootstrap to ensure independent and reproducible results
    bootstrap_random_state = args.random_state + bootstrap_idx if args.random_state is not None else None
    
    # this dictionary will store all results for this single bootstrap
    results_for_this_bootstrap = {
        'time_window_results': {},
        'time_averaged_cms': {},
        'cats_by_roi': {}
    }
    
    print(f"\n{'='*20}) PROCESSING BOOTSTRAP {bootstrap_idx+1}/{args.bootstraps} {'='*20}\n")
    
    # 1. Generate data for THIS bootstrap sample inside the worker
    # We set n_bootstraps=1 because this function handles one bootstrap.
    # We set n_jobs=1 to avoid nested parallelism, which is inefficient and can cause issues.
    print(f"Bootstrap {bootstrap_idx + 1}: Generating data sample...")

    roi_labeled_arrays_this_bootstrap_list = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
        rois=rois,
        subjects_data_objects=subjects_mne_objects,
        condition_names=condition_names,
        subjects=args.subjects,
        electrodes_per_subject_roi=electrodes,
        n_bootstraps=1,  # Generate only one sample
        chans_axs=args.chans_axs,
        time_axs=args.time_axs,
        freq_axs=None,
        random_state=bootstrap_random_state, # Unique seed
        n_jobs=1  # Run ROI generation serially within this worker
    )

    # Extract the single LabeledArray from the list returned for each ROI
    roi_labeled_arrays_this_bootstrap = {
        roi: arrays[0] for roi, arrays in roi_labeled_arrays_this_bootstrap_list.items() if arrays
    }

    # Add debugging
    print(f"\nBootstrap {bootstrap_idx + 1}: Extracted labeled arrays for ROIs:")
    for roi, labeled_array in roi_labeled_arrays_this_bootstrap.items():
        if labeled_array is not None:
            print(f"  ROI {roi}: conditions = {list(labeled_array.keys())}")
        else:
            print(f"  ROI {roi}: labeled_array is None")
        
    if not roi_labeled_arrays_this_bootstrap:
        print(f"Warning: No data generated for bootstrap {bootstrap_idx + 1}. Skipping.")
        return None
        
    # Main code
    # Directory to save confusion matrices
    cm_save_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(cm_save_dir, exist_ok=True)
    
    # --- 1. Calculate Time-Averaged CMs (using raw counts) ---
    for condition_comparison, strings_to_find in condition_comparisons.items():
        results_for_this_bootstrap['time_averaged_cms'][condition_comparison] = {}
        for roi in rois:
            # Get the 'cats' dictionary for this ROI
            _, _, cats = concatenate_conditions_by_string(
                roi_labeled_arrays_this_bootstrap, roi, strings_to_find, args.obs_axs
            )
            
            ## FIX: Store the 'cats' dictionary for this ROI so it can be used for plotting later.
            if roi not in results_for_this_bootstrap['cats_by_roi']:
                results_for_this_bootstrap['cats_by_roi'][roi] = cats
                
            # Get the raw-count confusion matrix
            cm = get_time_averaged_confusion_matrix(
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                roi=roi,
                strings_to_find=strings_to_find,
                cats=cats,
                clf=args.clf_model,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                obs_axs=args.obs_axs,
                balance_method=args.balance_method,
                explained_variance=args.explained_variance,
                random_state=args.random_state + bootstrap_idx,
            )
            if cm is not None:
                results_for_this_bootstrap['time_averaged_cms'][condition_comparison][roi] = cm

    for condition_comparison, strings_to_find in condition_comparisons.items():
        
        results_for_this_bootstrap['time_window_results'][condition_comparison] = {}
        
        # Get confusion matrices for each ROI
        cm_true_per_roi, cm_shuffle_per_roi = get_confusion_matrices_for_rois_time_window_decoding_jim(
            roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
            rois=rois,
            condition_comparison=condition_comparison,
            strings_to_find=strings_to_find,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            obs_axs=args.obs_axs,
            time_axs=-1,
            clf=args.clf_model,
            balance_method=args.balance_method,
            explained_variance=args.explained_variance,
            random_state=bootstrap_random_state,
            window_size=args.window_size,
            step_size=args.step_size,
            n_perm=args.n_shuffle_perms,
            sampling_rate=args.sampling_rate,
            first_time_point=-1,
            folds_as_samples=args.folds_as_samples
        )
        
        # Now compute accuracies and perform time permutation cluster test
        condition_save_dir = os.path.join(save_dir, f"{condition_comparison}")
        os.makedirs(condition_save_dir, exist_ok=True)

        for roi in rois:
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi] = {}
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['strings_to_find'] = strings_to_find

            cm_true = cm_true_per_roi[roi]['cm_true']
            cm_shuffle = cm_shuffle_per_roi[roi]['cm_shuffle']
            time_window_centers = cm_true_per_roi[roi]['time_window_centers']
            window_size = cm_true_per_roi[roi]['window_size']
            step_size = cm_true_per_roi[roi]['step_size']

            # store cm outputs and windowing parameters
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['cm_true'] = cm_true
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['cm_shuffle'] = cm_shuffle
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['time_window_centers'] = time_window_centers
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['window_size'] = window_size
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['step_size'] = step_size
            
            # Compute accuracies
            accuracies_true, accuracies_shuffle = compute_accuracies(cm_true, cm_shuffle)

            # find average accuracies, either across folds or repeats/perms, depending on whether folds_as_samples is set to true or not
            mean_accuracies_true = np.mean(accuracies_true, axis=1, keepdims=True) # Shape: (n_windows, 1), keeping folds/repeats dimension for compatibility with the percentile function
            mean_accuracies_shuffle = np.mean(accuracies_shuffle, axis=1, keepdims=True) 
            
            # store accuracies - TODO: average across bootstraps somehow for these
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['accuracies_true'] = accuracies_true
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['accuracies_shuffle'] = accuracies_shuffle
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['mean_accuracies_true'] = mean_accuracies_true
            results_for_this_bootstrap['time_window_results'][condition_comparison][roi]['mean_accuracies_shuffle'] = mean_accuracies_shuffle
            
    # 1. Determine configuration based on experiment conditions - make sure to update this as you add more conditions
    # pooled_settings is now a LIST of (key, strings_to_find) tuples
    pooled_settings_list = get_pooled_shuffle_settings(args.conditions)

    # 2. Execute logic for ALL pooled settings
    for pooled_settings in pooled_settings_list:
        result_key, strings_to_find_pooled = pooled_settings
        results_for_this_bootstrap['time_window_results'][result_key] = {}

        for roi in rois:
            accuracies_shuffle_pooled = run_pooled_shuffle_for_roi(
                roi=roi,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                strings_to_find=strings_to_find_pooled,
                args=args,
                random_state=bootstrap_random_state
            )

            results_for_this_bootstrap['time_window_results'][result_key][roi] = accuracies_shuffle_pooled

    return results_for_this_bootstrap