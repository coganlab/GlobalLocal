import sys
import os
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t

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
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at the beginning to prioritize it

from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne

import numpy as np
import pandas as pd
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.fast import mean_diff

# TODO: hmm fix these utils imports, import the funcs individually. 6/1/25.
from src.analysis.utils.general_utils import *
from src.analysis.utils.general_utils import (
    make_or_load_subjects_electrodes_to_ROIs_dict, 
    identify_bad_channels_by_trial_nan_rate, 
    impute_trial_nans_by_channel_mean,
    create_subjects_mne_objects_dict,
    filter_electrode_lists_against_subjects_mne_objects,
    find_difference_between_two_electrode_lists
)

# import matplotlib
# matplotlib.use('Agg') # <-- ADD THIS AND THE ABOVE LINE FOR DEBUGGING
import matplotlib.pyplot as plt

from pandas import read_csv
import scipy.stats as stats
from joblib import Parallel, delayed

from scipy.ndimage import label
from scipy.stats import norm, ttest_ind, ttest_rel

from functools import partial
import json
import pickle

from os.path import join, expanduser, basename
import glob, json
import numpy, tqdm, mne, pandas
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ieeg.decoding.models import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup

from src.analysis.config import experiment_conditions

from src.analysis.utils.labeled_array_utils import (
    put_data_in_labeled_array_per_roi_subject,
    remove_nans_from_labeled_array,
    remove_nans_from_all_roi_labeled_arrays,
    concatenate_conditions_by_string,
    get_data_in_time_range,
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel
)

from src.analysis.decoding.decoding import (
    concatenate_and_balance_data_for_decoding, 
    get_and_plot_confusion_matrix_for_rois_jim,
    Decoder, 
    windower,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    plot_true_vs_shuffle_accuracies,
    plot_accuracies_with_multiple_sig_clusters,
    plot_accuracies_nature_style,
    make_pooled_shuffle_distribution,
    find_significant_clusters_of_series_vs_distribution_based_on_percentile,
    compute_pooled_bootstrap_statistics,
    do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions,
    do_mne_paired_cluster_test,
    get_pooled_accuracy_distributions_for_comparison,
    get_time_averaged_confusion_matrix,
    cluster_perm_paired_ttest_by_duration,
    run_two_one_tailed_tests_with_time_perm_cluster
)
def process_bootstrap(bootstrap_idx, subjects_mne_objects, args, rois, condition_names, electrodes, condition_comparisons, save_dir):
    """
    Generates and processes a single bootstrap sample.
    This function is designed to be called in parallel by joblib.
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
            
    # lwpc
    if args.conditions == experiment_conditions.stimulus_lwpc_conditions:   
        results_for_this_bootstrap['time_window_results']['lwpc_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['c25_vs_i25'][roi]['time_window_centers']
            
            # get i vs c pooled shuffle distribution
            strings_to_find_pooled = [['c25', 'c75'], ['i25', 'i75']]
            
            accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                roi=roi,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                strings_to_find_pooled=strings_to_find_pooled,
                explained_variance=args.explained_variance,
                n_splits=args.n_splits,
                n_perm=args.n_shuffle_perms,
                random_state=bootstrap_random_state,
                balance_method='subsample', # Subsampling is recommended for pooling
                obs_axs=args.obs_axs,
                window_size=args.window_size,
                step_size=args.step_size
            )
            
            results_for_this_bootstrap['time_window_results']['lwpc_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled

    # lwps
    if args.conditions == experiment_conditions.stimulus_lwps_conditions:   
        results_for_this_bootstrap['time_window_results']['lwps_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['s25_vs_r25'][roi]['time_window_centers']
            
            # get i vs c pooled shuffle distribution
            strings_to_find_pooled = [['s25', 's75'], ['r25', 'r75']]
            
            accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                roi=roi,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                strings_to_find_pooled=strings_to_find_pooled,
                explained_variance=args.explained_variance,
                n_splits=args.n_splits,
                n_perm=args.n_shuffle_perms,
                random_state=bootstrap_random_state,
                balance_method='subsample', # Subsampling is recommended for pooling
                obs_axs=args.obs_axs,
                window_size=args.window_size,
                step_size=args.step_size
            )
            
            results_for_this_bootstrap['time_window_results']['lwps_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled

    # congruency by switch proportion
    if args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:   
        results_for_this_bootstrap['time_window_results']['congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['c_in_25switchBlock_vs_i_in_25switchBlock'][roi]['time_window_centers']
            
            # get i vs c pooled shuffle distribution
            strings_to_find_pooled = [['c_in'], ['i_in']]
            
            accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                roi=roi,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                strings_to_find_pooled=strings_to_find_pooled,
                explained_variance=args.explained_variance,
                n_splits=args.n_splits,
                n_perm=args.n_shuffle_perms,
                random_state=bootstrap_random_state,
                balance_method='subsample', # Subsampling is recommended for pooling
                obs_axs=args.obs_axs,
                window_size=args.window_size,
                step_size=args.step_size
            )
            
            results_for_this_bootstrap['time_window_results']['congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled
            
    # switch type by congruency proportion
    if args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:   
        results_for_this_bootstrap['time_window_results']['switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'][roi]['time_window_centers']
            
            # get s vs r pooled shuffle distribution
            strings_to_find_pooled = [['s_in'], ['r_in']]
            
            accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                roi=roi,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                strings_to_find_pooled=strings_to_find_pooled,
                explained_variance=args.explained_variance,
                n_splits=args.n_splits,
                n_perm=args.n_shuffle_perms,
                random_state=bootstrap_random_state,
                balance_method='subsample', # Subsampling is recommended for pooling
                obs_axs=args.obs_axs,
                window_size=args.window_size,
                step_size=args.step_size
            )
            
            results_for_this_bootstrap['time_window_results']['switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled
            
    return results_for_this_bootstrap

def main(args):
    # Determine LAB_root based on the operating system and environment
    if args.LAB_root is None:
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
    else:
        LAB_root = args.LAB_root

    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = load_subjects_electrodes_to_ROIs_dict(save_dir=config_dir, filename='subjects_electrodestoROIs_dict.json')
    
    condition_names = list(args.conditions.keys()) # get the condition names as a list

    # filename is too long to save so let's just drop the epochs root file from the conditions save name for now.
    if args.conditions == experiment_conditions.stimulus_conditions:
        conditions_save_name = 'stimulus_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_experiment_conditions:
        conditions_save_name = 'stimulus_experiment_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_main_effect_conditions:
        conditions_save_name = 'stimulus_main_effect_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        conditions_save_name = 'stimulus_lwpc_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        conditions_save_name = 'stimulus_lwps_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_big_letter_conditions:
        conditions_save_name = 'stimulus_big_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_small_letter_conditions:
        conditions_save_name = 'stimulus_small_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_task_conditions:
        conditions_save_name = 'stimulus_task_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_conditions:
        conditions_save_name = 'stimulus_congruency_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        conditions_save_name = 'stimulus_switch_type_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_err_corr_conditions:
        conditions_save_name = 'stimulus_err_corr_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        conditions_save_name = 'stimulus_congruency_by_switch_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        conditions_save_name = 'stimulus_switch_type_by_congruency_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_iR_cS_err_conditions:
        conditions_save_name = 'stimulus_iR_cS_err_conditions' +  '_' + str(len(args.subjects)) + '_' + 'subjects'
    
    elif args.conditions == experiment_conditions.response_conditions:
        conditions_save_name = 'response_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_experiment_conditions:
        conditions_save_name = 'response_experiment_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_big_letter_conditions:
        conditions_save_name = 'response_big_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_small_letter_conditions:
        conditions_save_name = 'response_small_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_task_conditions:
        conditions_save_name = 'response_task_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_congruency_conditions:
        conditions_save_name = 'response_congruency_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_switch_type_conditions:
        conditions_save_name = 'response_switch_type_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_err_corr_conditions:
        conditions_save_name = 'response_err_corr_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_congruency_by_switch_proportion_conditions:
        conditions_save_name = 'response_congruency_by_switch_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_switch_type_by_congruency_proportion_conditions:
        conditions_save_name = 'response_switch_type_by_congruency_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_iR_cS_err_conditions:
        conditions_save_name = 'response_iR_cS_err_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    
    save_dir = os.path.join(LAB_root, 'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'decoding', 'figs', f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")
    
    sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)
      
    subjects_mne_objects = create_subjects_mne_objects_dict(subjects=args.subjects, epochs_root_file=args.epochs_root_file, conditions=args.conditions, task=args.task, just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only)
    
    # determine which electrodes to use (all electrodes or just the task-significant ones)
    if args.electrodes == 'all':
        raw_electrodes = all_electrodes_per_subject_roi 
        elec_string_to_add_to_filename = 'all_elecs'
    elif args.electrodes == 'sig':
        raw_electrodes = sig_electrodes_per_subject_roi
        elec_string_to_add_to_filename = 'sig_elecs'

    else:
        raise ValueError("electrodes input must be set to all or sig")
    
    # ADD THIS BLOCK to create a string for the sampling method
    folds_info_str = 'folds_as_samples' if args.folds_as_samples else 'repeats_as_samples'

    # filter electrodes to only the ones that exist in the epochs objects. This mismatch can arise due to dropping channels when making the epochs objects, because the subjects_electrodestoROIs_dict is made based on all the electrodes, with no dropping.
    electrodes = filter_electrode_lists_against_subjects_mne_objects(rois, raw_electrodes, subjects_mne_objects)
    
    dropped_electrodes, _ = find_difference_between_two_electrode_lists(raw_electrodes, electrodes)
    print("\n--- Summary of Dropped Electrodes ---")
    total_dropped = 0
    for roi, sub_dict in dropped_electrodes.items():
        if not sub_dict: continue # Skip ROIs with no dropped electrodes
        print(f"ROI: {roi}")
        for sub, elec_list in sub_dict.items():
            if elec_list:
                print(f"  - Subject {sub}: Dropped {len(elec_list)} electrode(s)")
                total_dropped += len(elec_list)
    print(f"Total electrodes dropped across all subjects/ROIs: {total_dropped}")
    print("-------------------------------------\n")
            
    condition_comparisons = {}

    # update these as needed!
    if args.conditions == experiment_conditions.stimulus_experiment_conditions:
        condition_comparisons['congruency'] = [['c25', 'c75'], ['i25', 'i75']]
        condition_comparisons['switchType'] = [['r25', 'r75'], ['s25', 's75']]
    elif args.conditions == experiment_conditions.stimulus_conditions:
        condition_comparisons['BigLetter'] = ['bigS', 'bigH']
        condition_comparisons['SmallLetter'] = ['smallS', 'smallH']
        condition_comparisons['Task'] = ['taskG', 'taskL']
    elif args.conditions == experiment_conditions.stimulus_big_letter_conditions:
        condition_comparisons['BigLetter'] = ['bigS', 'bigH']
    elif args.conditions == experiment_conditions.stimulus_small_letter_conditions:
        condition_comparisons['SmallLetter'] = ['smallS', 'smallH']
    elif args.conditions == experiment_conditions.stimulus_task_conditions:
        condition_comparisons['task'] = ['taskG', 'taskL']
    elif args.conditions == experiment_conditions.stimulus_congruency_conditions:
        condition_comparisons['congruency'] = [['Stimulus_c'], ['Stimulus_i']]
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        condition_comparisons['switchType'] = [['Stimulus_r'], ['Stimulus_s']]
    elif args.conditions == experiment_conditions.stimulus_err_corr_conditions:
        condition_comparisons['err_vs_corr'] = [['Stimulus_err'], ['Stimulus_corr']]
    elif args.conditions == experiment_conditions.stimulus_iR_cS_err_conditions:
        condition_comparisons['iR_err_vs_cS_err'] = [['Stimulus_err_iR'], ['Stimulus_err_cS']]

        
    # 8/26/25 changes that should probably first involve an ANOVA between all four comparisons - note that this will be underpowered since I'm subsampling to the 25% condition
    # hm i think i should probably trial match too, so the c75 vs i75 would have to be subsampled to the c25 vs i25 trial counts. Ugh that loses sooooo many trials (50%). Rerun make epoched data with more stringent nan criteria so that i don't lose so many trials.
    elif args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        condition_comparisons['c25_vs_i25'] = ['c25', 'i25'] # lwpc
        condition_comparisons['c75_vs_i75'] = ['c75', 'i75']
        
        condition_comparisons['c25_vs_i75'] = ['c25', 'i75'] # control comparisons for lwpc, think more about how to interpret these
        condition_comparisons['c75_vs_i25'] = ['c75', 'i25']
        
        condition_comparisons['c25_vs_c75'] = ['c25', 'c75'] # these cross-block comparisons let me decode if there's pre-trial information about the congruency proportion
        condition_comparisons['i25_vs_i75'] = ['i25', 'i75']
        
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        condition_comparisons['s25_vs_r25'] = ['s25', 'r25'] # lwps
        condition_comparisons['s75_vs_r75'] = ['s75', 'r75']
        
        condition_comparisons['s25_vs_r75'] = ['s25', 'r75'] # control comparisons for lwps, think more about how to interpret these
        condition_comparisons['s75_vs_r25'] = ['s75', 'r25']

    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        condition_comparisons['c_in_25switchBlock_vs_i_in_25switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_25switchBlock']
        condition_comparisons['c_in_75switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_75switchBlock']
        condition_comparisons['c_in_25switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_75switchBlock']
        condition_comparisons['c_in_75switchBlock_vs_i_in_25switchBlock'] = ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_25switchBlock']
        condition_comparisons['c_in_25switchBlock_vs_c_in_75switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_c_in_75switchBlock']
        condition_comparisons['i_in_25switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_i_in_25switchBlock', 'Stimulus_i_in_75switchBlock']
        
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        condition_comparisons['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_25incongruentBlock']
        condition_comparisons['s_in_75incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_75incongruentBlock']
        condition_comparisons['s_in_25incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock']
        condition_comparisons['s_in_75incongruentBlock_vs_r_in_25incongruentBlock'] = ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_25incongruentBlock'] 
        condition_comparisons['s_in_25incongruentBlock_vs_s_in_75incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_s_in_75incongruentBlock'] 
        condition_comparisons['r_in_25incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_r_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock']

    # get the confusion matrix using the downsampled version
    # add elec and subject info to filename 6/11/25
    other_string_to_add = (
        f"{elec_string_to_add_to_filename}_{str(len(args.subjects))}_subjects_{folds_info_str}_ev_{args.explained_variance}"
    )
            
    time_window_decoding_results = {}     
     
    print(f"\n{'='*20} STARTING PARALLEL BOOTSTRAPPING ({args.bootstraps} samples across {args.n_jobs} jobs) {'='*20}\n")

    # use joblib to run the bootstrap processing in parallel
    bootstrap_results_list = Parallel(n_jobs=args.n_jobs, verbose=10, backend='loky')(
        delayed(process_bootstrap)(
            bootstrap_idx,
            subjects_mne_objects,
            args,
            rois,
            condition_names,
            electrodes,
            condition_comparisons,
            save_dir
        ) for bootstrap_idx in range(args.bootstraps)
    )

    # reconstruct the main results dictionary from the list returned by the parallel jobs
    time_window_decoding_results = {i: result['time_window_results'] for i, result in enumerate(bootstrap_results_list) if result is not None}
    time_averaged_cms_list = [result['time_averaged_cms'] for result in bootstrap_results_list if result]

    ## Extract the 'cats_by_roi' dictionary from the first valid bootstrap result.
    ## This is necessary to get the correct labels for plotting the confusion matrices.
    cats_by_roi = {}
    first_valid_result = next((res for res in bootstrap_results_list if res), None)
    if first_valid_result:
        cats_by_roi = first_valid_result.get('cats_by_roi', {})

    # --- Step 1: Aggregate and Plot Time-Averaged CMs ---
    print("\n📊 Aggregating and plotting time-averaged confusion matrices...")
    for condition_comparison in condition_comparisons.keys():
        for roi in rois:
            # Collect all raw CMs for this specific condition/ROI
            raw_cms = [
                boot_result[condition_comparison][roi] 
                for boot_result in time_averaged_cms_list 
                if condition_comparison in boot_result and roi in boot_result[condition_comparison]
            ]

            if not raw_cms:
                continue

            # Sum, normalize, and plot (same logic as before)
            total_cm_counts = np.sum(np.array(raw_cms), axis=0)
            row_sums = total_cm_counts.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1 
            normalized_cm = total_cm_counts.astype('float') / row_sums
            
            ## FIX: This check now correctly uses the `cats_by_roi` dictionary retrieved from the bootstrap results.
            if roi in cats_by_roi:
                display_labels = [str(key) for key in cats_by_roi[roi].keys()]
            else:
                print(f"Warning: 'cats' dictionary not found for ROI {roi}. Skipping CM plot.")
                continue

            # Plotting logic
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=display_labels)
            disp.plot(ax=ax, im_kw={"vmin": 0, "vmax": 1}, colorbar=True)
            ax.set_title(f'{roi} - {condition_comparison}\n(Counts summed across {args.bootstraps} bootstraps)')

            filename = (
                f'{args.timestamp}_{roi}_{condition_comparison}_SUMMED_{args.bootstraps}boots_ev_{args.explained_variance}'
                f'time_averaged_confusion_matrix.png'
            )
            plot_save_path = os.path.join(save_dir, condition_comparison, roi)
            os.makedirs(plot_save_path, exist_ok=True)
            plt.savefig(os.path.join(plot_save_path, filename))
            plt.close()
            print(f"✅ Saved summed & normalized CM for {roi} to {plot_save_path}")

            
    if not time_window_decoding_results:
        print("\n✗ Analysis failed: No bootstrap samples were successfully processed.")
        return
    
    print(f"\n{'-'*20} PARALLEL BOOTSTRAPPING COMPLETE {'='*20}\n")
    
    # after all bootstraps complete, run pooled statistics
    all_bootstrap_stats = compute_pooled_bootstrap_statistics(
        time_window_decoding_results,
        args.bootstraps,
        condition_comparisons,
        rois,
        percentile=args.percentile,
        cluster_percentile=args.cluster_percentile,
        n_cluster_perms=args.n_cluster_perms,
        random_state=args.random_state,
        unit_of_analysis=args.unit_of_analysis
    )
    
    sub_str = str(len(args.subjects))
    analysis_params_str = (
            f"{sub_str}_subs_{elec_string_to_add_to_filename}_{args.clf_model_str}_" 
            f"{args.bootstraps}boots_{args.n_splits}splits_{args.n_repeats}reps_"
            f"{args.unit_of_analysis}_unit_ev_{args.explained_variance}"
        )               
    master_results = {
        'stats': all_bootstrap_stats,
        'metadata': {
            'args': vars(args), # Save all arguments from the run
            'analysis_params_str': analysis_params_str
        },
        'comparison_clusters': {} # We will populate this in the loops below
    }
       
    # define color and linestyle for plotting true vs shuffle
    colors = {
        'true': '#0173B2',  # Blue
        'shuffle': '#949494'  # Gray
    }
    
    linestyles = {
        'true': '-',
        'shuffle': '--'
    }  
    
    # then plot using the pooled statistics
    for condition_comparison in condition_comparisons.keys():
        for roi in rois:
            if roi in all_bootstrap_stats[condition_comparison]:
                stats = all_bootstrap_stats[condition_comparison][roi] 
                time_window_centers = time_window_decoding_results[0][condition_comparison][roi]['time_window_centers']
                
                # extract the correct keys based on unit_of_analysis
                unit = stats['unit_of_analysis']
                
                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict={
                        'true': stats[f'{unit}_true_accs'], # use the full distribution
                        'shuffle': stats[f'{unit}_shuffle_accs']
                    },
                    significant_clusters=stats['significant_clusters'],
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'bootstrap_true_vs_shuffle_{condition_comparison}',
                    roi=roi,
                    save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                    timestamp=args.timestamp,
                    p_thresh=args.percentile,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=args.single_column,
                    show_legend=args.show_legend,
                    ylim=(0.3, 0.8),
                    show_chance_level=False, # The pooled shuffle line is the new chance level 
                    filename_suffix=analysis_params_str  
                )    
              
    print("\n📊 Extracting and plotting pooled CM traces for debugging...")
    
    # 1. Extract the pooled traces
    pooled_cm_traces = extract_pooled_cm_traces(
        time_window_decoding_results=time_window_decoding_results,
        n_bootstraps=args.bootstraps,
        condition_comparisons=condition_comparisons,
        rois=rois,
        unit_of_analysis=args.unit_of_analysis,
        cats_by_roi=cats_by_roi
    )
    
    # 2. Plot the traces for each comparison and ROI
    for condition_comparison, roi_data in pooled_cm_traces.items():
        for roi, traces_dict in roi_data.items():
            if not traces_dict:
                continue # Skip if no data
                
            # Get time points from one of the results
            time_window_centers = time_window_decoding_results[0][condition_comparison][roi]['time_window_centers']
            
            # Define custom colors and linestyles for your 2x2 case
            # This is an example for c25 vs i25
            trace_colors = {
                'True: c25, Pred: c25': '#0173B2', # Dark Blue (TP c25)
                'True: c25, Pred: i25': '#56B4E9', # Light Blue (FN c25)
                'True: i25, Pred: i25': '#DE8F05', # Dark Orange (TP i25)
                'True: i25, Pred: c25': '#CC78BC', # Pink/Purple (FN i25)
            }
            trace_linestyles = {
                'True: c25, Pred: c25': '-', # Solid for TP
                'True: c25, Pred: i25': '--',# Dashed for FN
                'True: i25, Pred: i25': '-', # Solid for TP
                'True: i25, Pred: c25': '--',# Dashed for FN
            }
            
            # You might need to adjust the labels for other comparisons, but this will work for 2-class
            
            plot_cm_traces_nature_style(
                time_points=time_window_centers,
                cm_traces_dict=traces_dict,
                comparison_name=f'DEBUG_CM_Traces_{condition_comparison}',
                roi=roi,
                save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                timestamp=args.timestamp,
                colors=trace_colors,
                linestyles=trace_linestyles,
                single_column=args.single_column,
                show_legend=True,
                ylabel="Mean Trial Count",
                filename_suffix=analysis_params_str
            )
               
    if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        print(f"\n--- Running LWPC Comparison Statistics (c25_vs_i25 vs c75_vs_i75) using '{args.unit_of_analysis}' as unit of analysis ---")
        
        lwpc_colors = {
            'c25_vs_i25': '#0173B2',  # Blue
            'c75_vs_i75': '#DE8F05',  # Orange
          'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'  # Gray
        }
        
        lwpc_linestyles = {
            'c25_vs_i25': '-',  # Solid
            'c75_vs_i75': '-',  # Solid
            'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'  # Dashed
        }
        
        # Perform the statistical comparison between the two true accuracy distributions

        for roi in rois:
            if roi not in all_bootstrap_stats.get('c25_vs_i25', {}):
                print(f"Skipping plot for ROI {roi} due to missing data.")
                continue

            # --- Pool the pooled shuffle distributions from each bootstrap ---
            lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps = []
            for b_idx in range(args.bootstraps):
                if b_idx in time_window_decoding_results:
                    shuffle_data = time_window_decoding_results[b_idx]['lwpc_shuffle_accs_across_pooled_conditions'][roi]
                    # Depending on folds_as_samples, shuffle_data might be (n_windows, n_perms*n_splits)
                    # We transpose to (n_samples, n_windows) to be consistent
                    lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps.append(shuffle_data.T)

            # Stack all samples from all bootstraps
            stacked_lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps = np.vstack(
                lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps
            )
            
            # 1. Get the pooled data using your existing helper function
            pooled_c25_vs_i25_accs, pooled_c75_vs_i75_accs = get_pooled_accuracy_distributions_for_comparison(
                time_window_decoding_results=time_window_decoding_results,
                n_bootstraps=args.bootstraps,
                condition_comparison_1='c25_vs_i25',
                condition_comparison_2='c75_vs_i75',
                roi=roi,
                unit_of_analysis=args.unit_of_analysis
            )

            # 2. Run the new paired cluster test
            sig_clusters_lwpc_25_over_75, sig_clusters_lwpc_75_over_25, _, _ = run_two_one_tailed_tests_with_time_perm_cluster(
                accuracies1=pooled_c25_vs_i25_accs,
                accuracies2=pooled_c75_vs_i75_accs,
                p_thresh=args.p_thresh_for_time_perm_cluster_stats,
                p_cluster=args.p_cluster,
                stat_func=args.stat_func,
                permutation_type=args.permutation_type,
                n_perm=args.n_cluster_perms,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            
            significance_clusters_lwpc_comparison = {
                '25_over_75': {
                    'clusters': sig_clusters_lwpc_25_over_75,
                    'label': '25% > 75% I',
                    'color': lwpc_colors['c25_vs_i25'], # Blue
                    'marker': '*' 
                },
                '75_over_25': {
                    'clusters': sig_clusters_lwpc_75_over_25,
                    'label': '75% > 25% I',
                    'color': lwpc_colors['c75_vs_i75'], # Orange
                    'marker': '*'
                }
            }
            
            if roi not in master_results['comparison_clusters']:
                    master_results['comparison_clusters'][roi] = {}
            master_results['comparison_clusters'][roi]['lwpc'] = significance_clusters_lwpc_comparison
        
            # --- Get data for plotting from the main stats dictionary ---
            c25_vs_i25_stats = all_bootstrap_stats['c25_vs_i25'][roi]
            c75_vs_i75_stats = all_bootstrap_stats['c75_vs_i75'][roi]
            
            # This ensures the plot's lines/error bars match the unit of analysis
            unit = c25_vs_i25_stats['unit_of_analysis']
            
            time_window_centers = time_window_decoding_results[0]['c25_vs_i25'][roi]['time_window_centers']
            
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    'c25_vs_i25': c25_vs_i25_stats[f'{unit}_true_accs'],
                    'c75_vs_i75': c75_vs_i75_stats[f'{unit}_true_accs'],
                    'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': stacked_lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps
                },
                # Pass the new dictionary here
                significance_clusters_dict=significance_clusters_lwpc_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_LWPC_comparison',
                roi=roi,
                save_dir=os.path.join(save_dir, "LWPC_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors=lwpc_colors,
                linestyles=lwpc_linestyles,
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=(0.3, 0.8),
                ylabel="Congruency Decoding Accuracy",
                show_chance_level=False, # The pooled shuffle line is our chance level
                filename_suffix=analysis_params_str,
                
                # Add new parameters for multi-cluster plotting
                show_sig_legend=True,
                sig_bar_base_position=0.72, # Set base y-position for the bars
                sig_bar_spacing=0.015,       # Vertical spacing between bars
                sig_bar_height=0.01          # Height of the bars
            )

            print(f"Plotting accuracy DIFFERENCE for LWPC in {roi}...")
            
            lwpc_differences = pooled_c25_vs_i25_accs - pooled_c75_vs_i75_accs
            
            mean_diff = np.mean(lwpc_differences, axis=0)
            std_diff = np.std(lwpc_differences, axis=0)
            max_abs_val = np.max(np.abs(mean_diff) + std_diff)  
            diff_ylim = (-max_abs_val * 1.2, max_abs_val * 1.2)
            if diff_ylim[0] == 0: diff_ylim = (-0.1, 0.1)  

            # 5. REPLACE the second plot call as well
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    'c25_vs_i25_minus_c75_vs_i75': lwpc_differences
                },
                # Pass the same dictionary here
                significance_clusters_dict=significance_clusters_lwpc_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_LWPC_ACC_DIFFERENCE_plot',
                roi=roi,
                save_dir=os.path.join(save_dir, "LWPC_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors={'c25_vs_i25_minus_c75_vs_i75': '#404040'},
                linestyles={'c25_vs_i25_minus_c75_vs_i75': '-'},
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=diff_ylim,
                ylabel="Accuracy Difference (c25 vs i25 - c75 vs i75)",
                show_chance_level=True,
                chance_level=0,
                filename_suffix=analysis_params_str + "_ACC_DIFFERENCE",
                
                # Add new parameters for multi-cluster plotting
                show_sig_legend=False, # No legend needed for sig bars on diff plot
                sig_bar_base_position=diff_ylim[1] * 0.8, # Base y-position
                sig_bar_spacing=0.015,
                sig_bar_height=0.01
            )
                
    if args.conditions == experiment_conditions.stimulus_lwps_conditions:
        print(f"\n--- Running LWPS Comparison Statistics (s25_vs_r25 vs s75_vs_r75) using '{args.unit_of_analysis}' as unit of analysis ---")
        
        lwps_colors = {
            's25_vs_r25': '#0173B2',  # Blue
            's75_vs_r75': '#DE8F05',  # Orange
          'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'  # Gray
        }
        
        lwps_linestyles = {
            's25_vs_r25': '-',  # Solid
            's75_vs_r75': '-',  # Solid
            'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'  # Dashed
        }

        for roi in rois:
            if roi not in all_bootstrap_stats.get('s25_vs_r25', {}):
                print(f"Skipping plot for ROI {roi} due to missing data.")
                continue

            # --- Pool the pooled shuffle distributions from each bootstrap ---
            lwps_shuffle_accs_across_pooled_conditions_across_bootstraps = []
            for b_idx in range(args.bootstraps):
                if b_idx in time_window_decoding_results:
                    shuffle_data = time_window_decoding_results[b_idx]['lwps_shuffle_accs_across_pooled_conditions'][roi]
                    # Depending on folds_as_samples, shuffle_data might be (n_windows, n_perms*n_splits)
                    # We transpose to (n_samples, n_windows) to be consistent
                    lwps_shuffle_accs_across_pooled_conditions_across_bootstraps.append(shuffle_data.T)
        
            # Stack all samples from all bootstraps
            stacked_lwps_shuffle_accs_across_pooled_conditions_across_bootstraps = np.vstack(
                lwps_shuffle_accs_across_pooled_conditions_across_bootstraps
            )
            
            # 1. Get the pooled data using your existing helper function
            pooled_s25_vs_r25_accs, pooled_s75_vs_r75_accs = get_pooled_accuracy_distributions_for_comparison(
                time_window_decoding_results=time_window_decoding_results,
                n_bootstraps=args.bootstraps,
                condition_comparison_1='s25_vs_r25',
                condition_comparison_2='s75_vs_r75',
                roi=roi,
                unit_of_analysis=args.unit_of_analysis
            )

            # 2. Run the new paired cluster test
            sig_clusters_lwps_25_over_75, sig_clusters_lwps_75_over_25, _, _ = run_two_one_tailed_tests_with_time_perm_cluster(
                accuracies1=pooled_s25_vs_r25_accs,
                accuracies2=pooled_s75_vs_r75_accs,
                p_thresh=args.p_thresh_for_time_perm_cluster_stats,
                p_cluster=args.p_cluster,
                stat_func=args.stat_func,
                permutation_type=args.permutation_type,
                n_perm=args.n_cluster_perms,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            
            significance_clusters_lwps_comparison = {
                '25_over_75': {
                    'clusters': sig_clusters_lwps_25_over_75,
                    'label': '25% > 75% S',
                    'color': lwps_colors['s25_vs_r25'], # Blue
                    'marker': '*' 
                },
                '75_over_25': {
                    'clusters': sig_clusters_lwps_75_over_25,
                    'label': '75% > 25% S',
                    'color': lwps_colors['s75_vs_r75'], # Orange
                    'marker': '*'
                }
            }
            
            if roi not in master_results['comparison_clusters']:
                    master_results['comparison_clusters'][roi] = {}
            master_results['comparison_clusters'][roi]['lwps'] = significance_clusters_lwps_comparison
        
            # --- Get data for plotting from the main stats dictionary ---
            s25_vs_r25_stats = all_bootstrap_stats['s25_vs_r25'][roi]
            s75_vs_r75_stats = all_bootstrap_stats['s75_vs_r75'][roi]
            
            # This ensures the plot's lines/error bars match the unit of analysis
            unit = s25_vs_r25_stats['unit_of_analysis']
            
            time_window_centers = time_window_decoding_results[0]['s25_vs_r25'][roi]['time_window_centers']
            
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    's25_vs_r25': s25_vs_r25_stats[f'{unit}_true_accs'],
                    's75_vs_r75': s75_vs_r75_stats[f'{unit}_true_accs'],
                    'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': stacked_lwps_shuffle_accs_across_pooled_conditions_across_bootstraps
                },
                # Pass the new dictionary here
                significance_clusters_dict=significance_clusters_lwps_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_LWPS_comparison',
                roi=roi,
                save_dir=os.path.join(save_dir, "LWPS_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors=lwps_colors,
                linestyles=lwps_linestyles,
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=(0.3, 0.8),
                ylabel="Switch Type Decoding Accuracy",
                show_chance_level=False, # The pooled shuffle line is our chance level
                filename_suffix=analysis_params_str,
                
                # Add new parameters for multi-cluster plotting
                show_sig_legend=True,
                sig_bar_base_position=0.72, # Set base y-position for the bars
                sig_bar_spacing=0.015,       # Vertical spacing between bars
                sig_bar_height=0.01          # Height of the bars
            )

            print(f"Plotting accuracy DIFFERENCE for LWPS in {roi}...")
            
            lwps_differences = pooled_s25_vs_r25_accs - pooled_s75_vs_r75_accs
            
            mean_diff = np.mean(lwps_differences, axis=0)
            std_diff = np.std(lwps_differences, axis=0)
            max_abs_val = np.max(np.abs(mean_diff) + std_diff)  
            diff_ylim = (-max_abs_val * 1.2, max_abs_val * 1.2)
            if diff_ylim[0] == 0: diff_ylim = (-0.1, 0.1)  

            # 5. REPLACE the second plot call as well
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    's25_vs_r25_minus_s75_vs_r75': lwps_differences
                },
                # Pass the same dictionary here
                significance_clusters_dict=significance_clusters_lwps_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_LWPS_ACC_DIFFERENCE_plot',
                roi=roi,
                save_dir=os.path.join(save_dir, "LWPS_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors={'s25_vs_r25_minus_s75_vs_r75': '#404040'},
                linestyles={'s25_vs_r25_minus_s75_vs_r75': '-'},
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=diff_ylim,
                ylabel="Accuracy Difference (s25 vs r25 - s75 vs r75)",
                show_chance_level=True,
                chance_level=0,
                filename_suffix=analysis_params_str + "_ACC_DIFFERENCE",
                # Add new parameters for multi-cluster plotting
                show_sig_legend=False, # No legend needed for sig bars on diff plot
                sig_bar_base_position=diff_ylim[1] * 0.8, # Base y-position
                sig_bar_spacing=0.015,
                sig_bar_height=0.01
            )
            
    if args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        print(f"\n--- Running congruency by switch proportion Comparison Statistics (c_in_25switchBlock_vs_i_in_25switchBlock vs c_in_75switchBlock_vs_i_in_75switchBlock) using '{args.unit_of_analysis}' as unit of analysis ---")
        
        congruency_by_switch_proportion_colors = {
            'c_in_25switchBlock_vs_i_in_25switchBlock': '#0173B2',  # Blue
            'c_in_75switchBlock_vs_i_in_75switchBlock': '#DE8F05',  # Orange
          'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'  # Gray
        }
        
        congruency_by_switch_proportion_linestyles = {
            'c_in_25switchBlock_vs_i_in_25switchBlock': '-',  # Solid
            'c_in_75switchBlock_vs_i_in_75switchBlock': '-',  # Solid
            'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'  # Dashed
        }
        
        for roi in rois:
            if roi not in all_bootstrap_stats.get('c_in_25switchBlock_vs_i_in_25switchBlock', {}):
                print(f"Skipping plot for ROI {roi} due to missing data.")
                continue

            # --- Pool the pooled shuffle distributions from each bootstrap ---
            congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps = []
            for b_idx in range(args.bootstraps):
                if b_idx in time_window_decoding_results:
                    shuffle_data = time_window_decoding_results[b_idx]['congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions'][roi]
                    # Depending on folds_as_samples, shuffle_data might be (n_windows, n_perms*n_splits)
                    # We transpose to (n_samples, n_windows) to be consistent
                    congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps.append(shuffle_data.T)
        
            # Stack all samples from all bootstraps
            stacked_congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps = np.vstack(
                congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps
            )
            
            # 1. Get the pooled data using your existing helper function
            pooled_c_in_25switchBlock_vs_i_in_25switchBlock_accs, pooled_c_in_75switchBlock_vs_i_in_75switchBlock_accs = get_pooled_accuracy_distributions_for_comparison(
                time_window_decoding_results=time_window_decoding_results,
                n_bootstraps=args.bootstraps,
                condition_comparison_1='c_in_25switchBlock_vs_i_in_25switchBlock',
                condition_comparison_2='c_in_75switchBlock_vs_i_in_75switchBlock',
                roi=roi,
                unit_of_analysis=args.unit_of_analysis
            )

            # 2. Run the new paired cluster test
            sig_clusters_congruency_by_switch_proportion_25_over_75, sig_clusters_congruency_by_switch_proportion_75_over_25, _, _ = run_two_one_tailed_tests_with_time_perm_cluster(
                accuracies1=pooled_c_in_25switchBlock_vs_i_in_25switchBlock_accs,
                accuracies2=pooled_c_in_75switchBlock_vs_i_in_75switchBlock_accs,
                p_thresh=args.p_thresh_for_time_perm_cluster_stats,
                p_cluster=args.p_cluster,
                stat_func=args.stat_func,
                permutation_type=args.permutation_type,
                n_perm=args.n_cluster_perms,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            
            significance_clusters_congruency_by_switch_proportion_comparison = {
                '25_over_75': {
                    'clusters': sig_clusters_congruency_by_switch_proportion_25_over_75,
                    'label': 'C/I (25% S) > C/I (75% S)',
                    'color': congruency_by_switch_proportion_colors['c_in_25switchBlock_vs_i_in_25switchBlock'], # Blue
                    'marker': '*' 
                },
                '75_over_25': {
                    'clusters': sig_clusters_congruency_by_switch_proportion_75_over_25,
                    'label': 'C/I (75% S) > C/I (25% S)',
                    'color': congruency_by_switch_proportion_colors['c_in_75switchBlock_vs_i_in_75switchBlock'], # Orange
                    'marker': '*'
                }
            }
            
            if roi not in master_results['comparison_clusters']:
                    master_results['comparison_clusters'][roi] = {}
            master_results['comparison_clusters'][roi]['congruency_by_switch_proportion'] = significance_clusters_congruency_by_switch_proportion_comparison
        
            # --- Get data for plotting from the main stats dictionary ---
            c_in_25switchBlock_vs_i_in_25switchBlock_stats = all_bootstrap_stats['c_in_25switchBlock_vs_i_in_25switchBlock'][roi]
            c_in_75switchBlock_vs_i_in_75switchBlock_stats = all_bootstrap_stats['c_in_75switchBlock_vs_i_in_75switchBlock'][roi]
            
            # This ensures the plot's lines/error bars match the unit of analysis
            unit = c_in_25switchBlock_vs_i_in_25switchBlock_stats['unit_of_analysis']
            
            time_window_centers = time_window_decoding_results[0]['c_in_25switchBlock_vs_i_in_25switchBlock'][roi]['time_window_centers']
            
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    'c_in_25switchBlock_vs_i_in_25switchBlock': c_in_25switchBlock_vs_i_in_25switchBlock_stats[f'{unit}_true_accs'],
                    'c_in_75switchBlock_vs_i_in_75switchBlock': c_in_75switchBlock_vs_i_in_75switchBlock_stats[f'{unit}_true_accs'],
                    'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': stacked_congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps
                },
                # Pass the new dictionary here
                significance_clusters_dict=significance_clusters_congruency_by_switch_proportion_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_congruency_by_switch_proportion_comparison',
                roi=roi,
                save_dir=os.path.join(save_dir, "congruency_by_switch_proportion_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors=congruency_by_switch_proportion_colors,
                linestyles=congruency_by_switch_proportion_linestyles,
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=(0.3, 0.8),
                ylabel="Congruency Decoding Accuracy",
                show_chance_level=False, # The pooled shuffle line is our chance level
                filename_suffix=analysis_params_str,
                
                # Add new parameters for multi-cluster plotting
                show_sig_legend=True,
                sig_bar_base_position=0.72, # Set base y-position for the bars
                sig_bar_spacing=0.015,       # Vertical spacing between bars
                sig_bar_height=0.01          # Height of the bars
            )

            print(f"Plotting accuracy DIFFERENCE for congruency by switch proportion in {roi}...")
            
            congruency_by_switch_proportion_differences = pooled_c_in_25switchBlock_vs_i_in_25switchBlock_accs - pooled_c_in_75switchBlock_vs_i_in_75switchBlock_accs
            
            mean_diff = np.mean(congruency_by_switch_proportion_differences, axis=0)
            std_diff = np.std(congruency_by_switch_proportion_differences, axis=0)
            max_abs_val = np.max(np.abs(mean_diff) + std_diff)  
            diff_ylim = (-max_abs_val * 1.2, max_abs_val * 1.2)
            if diff_ylim[0] == 0: diff_ylim = (-0.1, 0.1)  

            # 5. REPLACE the second plot call as well
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    'c_in_25switchBlock_vs_i_in_25switchBlock_minus_c_in_75switchBlock_vs_i_in_75switchBlock': congruency_by_switch_proportion_differences
                },
                # Pass the same dictionary here
                significance_clusters_dict=significance_clusters_congruency_by_switch_proportion_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_congruency_by_switch_proportion_ACC_DIFFERENCE_plot',
                roi=roi,
                save_dir=os.path.join(save_dir, "congruency_by_switch_proportion_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors={'c_in_25switchBlock_vs_i_in_25switchBlock_minus_c_in_75switchBlock_vs_i_in_75switchBlock': '#404040'},
                linestyles={'c_in_25switchBlock_vs_i_in_25switchBlock_minus_c_in_75switchBlock_vs_i_in_75switchBlock': '-'},
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=diff_ylim,
                ylabel="Accuracy Difference (c_in_25switchBlock_vs_i_in_25switchBlock - c_in_75switchBlock_vs_i_in_75switchBlock)",
                show_chance_level=True,
                chance_level=0,
                filename_suffix=analysis_params_str + "_ACC_DIFFERENCE",
                # Add new parameters for multi-cluster plotting
                show_sig_legend=False, # No legend needed for sig bars on diff plot
                sig_bar_base_position=diff_ylim[1] * 0.8, # Base y-position
                sig_bar_spacing=0.015,
                sig_bar_height=0.01
            )

    if args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        print(f"\n--- Running switch_type_by_congruency_proportion Comparison Statistics (s_in_25switchBlock_vs_i_in_25switchBlock vs c_in_75switchBlock_vs_i_in_75switchBlock) using '{args.unit_of_analysis}' as unit of analysis ---")
        
        switch_type_by_congruency_proportion_colors = {
            's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '#0173B2',  # Blue
            's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#DE8F05',  # Orange
          'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'  # Gray
        }
        
        switch_type_by_congruency_proportion_linestyles = {
            's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '-',  # Solid
            's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '-',  # Solid
            'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'  # Dashed
        }

        for roi in rois:
            if roi not in all_bootstrap_stats.get('s_in_25incongruentBlock_vs_r_in_25incongruentBlock', {}):
                print(f"Skipping plot for ROI {roi} due to missing data.")
                continue

            # --- Pool the pooled shuffle distributions from each bootstrap ---
            switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps = []
            for b_idx in range(args.bootstraps):
                if b_idx in time_window_decoding_results:
                    shuffle_data = time_window_decoding_results[b_idx]['switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions'][roi]
                    # Depending on folds_as_samples, shuffle_data might be (n_windows, n_perms*n_splits)
                    # We transpose to (n_samples, n_windows) to be consistent
                    switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps.append(shuffle_data.T)
        
            # Stack all samples from all bootstraps
            stacked_switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps = np.vstack(
                switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps
            )
            
            # 1. Get the pooled data using your existing helper function
            pooled_s_in_25incongruentBlock_vs_r_in_25incongruentBlock_accs, pooled_s_in_75incongruentBlock_vs_r_in_75incongruentBlock_accs = get_pooled_accuracy_distributions_for_comparison(
                time_window_decoding_results=time_window_decoding_results,
                n_bootstraps=args.bootstraps,
                condition_comparison_1='s_in_25incongruentBlock_vs_r_in_25incongruentBlock',
                condition_comparison_2='s_in_75incongruentBlock_vs_r_in_75incongruentBlock',
                roi=roi,
                unit_of_analysis=args.unit_of_analysis
            )
            
            # 2. Run the new paired cluster test
            sig_clusters_switch_type_by_congruency_proportion_25_over_75, sig_clusters_switch_type_by_congruency_proportion_75_over_25, _, _ = run_two_one_tailed_tests_with_time_perm_cluster(
                accuracies1=pooled_s_in_25incongruentBlock_vs_r_in_25incongruentBlock_accs,
                accuracies2=pooled_s_in_75incongruentBlock_vs_r_in_75incongruentBlock_accs,
                p_thresh=args.p_thresh_for_time_perm_cluster_stats,
                p_cluster=args.p_cluster,
                stat_func=args.stat_func,
                permutation_type=args.permutation_type,
                n_perm=args.n_cluster_perms,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            
            significance_clusters_switch_type_by_congruency_proportion_comparison = {
                '25_over_75': {
                    'clusters': sig_clusters_switch_type_by_congruency_proportion_25_over_75,
                    'label': 'S/R (25% I) > S/R (75% I)',
                    'color': switch_type_by_congruency_proportion_colors['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'], # Blue
                    'marker': '*' 
                },
                '75_over_25': {
                    'clusters': sig_clusters_switch_type_by_congruency_proportion_75_over_25,
                    'label': 'S/R (75% I) > S/R (25% I)',
                    'color': switch_type_by_congruency_proportion_colors['s_in_75incongruentBlock_vs_r_in_75incongruentBlock'], # Orange
                    'marker': '*'
                }
            }
            if roi not in master_results['comparison_clusters']:
                master_results['comparison_clusters'][roi] = {}
            master_results['comparison_clusters'][roi]['switch_type_by_congruency_proportion'] = significance_clusters_switch_type_by_congruency_proportion_comparison
                    
            # --- Get data for plotting from the main stats dictionary ---
            s_in_25incongruentBlock_vs_r_in_25incongruentBlock_stats = all_bootstrap_stats['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'][roi]
            s_in_75incongruentBlock_vs_r_in_75incongruentBlock_stats = all_bootstrap_stats['s_in_75incongruentBlock_vs_r_in_75incongruentBlock'][roi]
            
            # This ensures the plot's lines/error bars match the unit of analysis
            unit = s_in_25incongruentBlock_vs_r_in_25incongruentBlock_stats['unit_of_analysis']
            
            time_window_centers = time_window_decoding_results[0]['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'][roi]['time_window_centers']
            
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    's_in_25incongruentBlock_vs_r_in_25incongruentBlock': s_in_25incongruentBlock_vs_r_in_25incongruentBlock_stats[f'{unit}_true_accs'],
                    's_in_75incongruentBlock_vs_r_in_75incongruentBlock': s_in_75incongruentBlock_vs_r_in_75incongruentBlock_stats[f'{unit}_true_accs'],
                    'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': stacked_switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps
                },
                # Pass the new dictionary here
                significance_clusters_dict=significance_clusters_switch_type_by_congruency_proportion_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_switch_type_by_congruency_proportion_comparison',
                roi=roi,
                save_dir=os.path.join(save_dir, "switch_type_by_congruency_proportion_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors=switch_type_by_congruency_proportion_colors,
                linestyles=switch_type_by_congruency_proportion_linestyles,
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=(0.3, 0.8),
                ylabel="Switch Type Decoding Accuracy",
                show_chance_level=False, # The pooled shuffle line is our chance level
                filename_suffix=analysis_params_str,
                
                # Add new parameters for multi-cluster plotting
                show_sig_legend=True,
                sig_bar_base_position=0.72, # Set base y-position for the bars
                sig_bar_spacing=0.015,       # Vertical spacing between bars
                sig_bar_height=0.01          # Height of the bars
            )

            print(f"Plotting accuracy DIFFERENCE for switch type by congruency proportion in {roi}...")
            
            switch_type_by_congruency_proportion_differences = pooled_s_in_25incongruentBlock_vs_r_in_25incongruentBlock_accs - pooled_s_in_75incongruentBlock_vs_r_in_75incongruentBlock_accs
            
            mean_diff = np.mean(switch_type_by_congruency_proportion_differences, axis=0)
            std_diff = np.std(switch_type_by_congruency_proportion_differences, axis=0)
            max_abs_val = np.max(np.abs(mean_diff) + std_diff)  
            diff_ylim = (-max_abs_val * 1.2, max_abs_val * 1.2)
            if diff_ylim[0] == 0: diff_ylim = (-0.1, 0.1)  

            # 5. REPLACE the second plot call as well
            plot_accuracies_with_multiple_sig_clusters(
                time_points=time_window_centers,
                accuracies_dict={
                    's_in_25incongruentBlock_vs_r_in_25incongruentBlock_minus_s_in_75incongruentBlock_vs_r_in_75incongruentBlock': switch_type_by_congruency_proportion_differences
                },
                # Pass the same dictionary here
                significance_clusters_dict=significance_clusters_switch_type_by_congruency_proportion_comparison,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                comparison_name=f'bootstrap_switch_type_by_congruency_proportion_ACC_DIFFERENCE_plot',
                roi=roi,
                save_dir=os.path.join(save_dir, "switch_type_by_congruency_proportion_comparison", f"{roi}"),
                timestamp=args.timestamp,
                p_thresh=args.percentile,
                colors={'s_in_25incongruentBlock_vs_r_in_25incongruentBlock_minus_s_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#404040'},
                linestyles={'s_in_25incongruentBlock_vs_r_in_25incongruentBlock_minus_s_in_75incongruentBlock_vs_r_in_75incongruentBlock': '-'},
                single_column=args.single_column,
                show_legend=args.show_legend,
                ylim=diff_ylim,
                ylabel="Accuracy Difference (s_in_25incongruentBlock_vs_r_in_25incongruentBlock - s_in_75incongruentBlock_vs_r_in_75incongruentBlock)",
                show_chance_level=True,
                chance_level=0,
                filename_suffix=analysis_params_str + "_ACC_DIFFERENCE",
                # Add new parameters for multi-cluster plotting
                show_sig_legend=False, # No legend needed for sig bars on diff plot
                sig_bar_base_position=diff_ylim[1] * 0.8, # Base y-position
                sig_bar_spacing=0.015,
                sig_bar_height=0.01
            )

            

# --- Save all results to a single file ---
    results_filename = f"{args.timestamp}_MASTER_RESULTS_{analysis_params_str}.pkl"
    results_save_path = os.path.join(save_dir, results_filename)
    
    # Try to grab time_window_centers and add to metadata
    try:
        first_comp = list(time_window_decoding_results[0].keys())[0]
        first_roi = list(time_window_decoding_results[0][first_comp].keys())[0]
        twc = time_window_decoding_results[0][first_comp][first_roi]['time_window_centers']
        master_results['metadata']['time_window_centers'] = twc
    except Exception as e:
        print(f"Warning: Could not save time_window_centers to metadata. {e}")

    print(f"\n💾 Saving all statistical results to: {results_save_path}")
    with open(results_save_path, 'wb') as f:
        pickle.dump(master_results, f)

    print("\n✅ Analysis and saving complete.")
                  
if __name__ == "__main__":
    # This block is only executed when someone runs this script directly
    # Since your run script calls main() directly, this block won't be executed
    # But we'll keep it minimal for compatibility
    
    # Check if being called with SimpleNamespace (from run script)
    import sys
    if len(sys.argv) == 1:
        # No command line arguments, must be imported and called from run script
        pass
    else:
        # Someone is trying to run this directly with command line args
        print("This script should be called via run_decoding.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)
