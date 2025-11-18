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

import matplotlib
matplotlib.use('Agg') # <-- ADD THIS AND THE ABOVE LINE FOR DEBUGGING
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
from sklearn.pipeline import Pipeline, make_pipeline

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
    run_two_one_tailed_tests_with_time_perm_cluster,
    extract_pooled_cm_traces,
    plot_cm_traces_nature_style,
    plot_high_dim_decision_slice,
    run_context_comparison_analysis
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
            
            results_for_this_bootstrap['time_window_results']['switch_type_by_congruency_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled
            
    # task by congruency
    if args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:   
        results_for_this_bootstrap['time_window_results']['task_by_congruency_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['i_taskG_vs_i_taskL'][roi]['time_window_centers']
            
            # get task G vs task L pooled shuffle distribution across c and i - i think this is right..
            strings_to_find_pooled = [['taskG'], ['taskL']]
            
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
            
            results_for_this_bootstrap['time_window_results']['task_by_congruency_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled
    
    # task by switch type
    if args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:   
        results_for_this_bootstrap['time_window_results']['task_by_switch_type_shuffle_accs_across_pooled_conditions'] = {}
          
        for roi in rois:
            time_window_centers = results_for_this_bootstrap['time_window_results']['s_taskG_vs_s_taskL'][roi]['time_window_centers']
            
            # get task G vs task L pooled shuffle distribution across s and r - i think this is right..
            strings_to_find_pooled = [['taskG'], ['taskL']]
            
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
            
            results_for_this_bootstrap['time_window_results']['task_by_switch_type_shuffle_accs_across_pooled_conditions'][roi] = accuracies_shuffle_pooled
                   
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
    print('LAB_root: ', LAB_root)
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
    elif args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:
        conditions_save_name = 'stimulus_task_by_congruency_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:
        conditions_save_name = 'stimulus_task_by_switch_type_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
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
    elif args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:
        condition_comparisons['c_taskG_vs_c_taskL'] = ['Stimulus_c_taskG', 'Stimulus_c_taskL']
        condition_comparisons['i_taskG_vs_i_taskL'] = ['Stimulus_i_taskG', 'Stimulus_i_taskL']
        
        condition_comparisons['c_taskG_vs_i_taskG'] = ['Stimulus_c_taskG', 'Stimulus_i_taskG']
        condition_comparisons['c_taskL_vs_i_taskL'] = ['Stimulus_c_taskL', 'Stimulus_i_taskL']
    elif args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:
        condition_comparisons['r_taskG_vs_r_taskL'] = ['Stimulus_r_taskG', 'Stimulus_r_taskL']
        condition_comparisons['s_taskG_vs_s_taskL'] = ['Stimulus_s_taskG', 'Stimulus_s_taskL']
        
        condition_comparisons['r_taskG_vs_s_taskG'] = ['Stimulus_r_taskG', 'Stimulus_s_taskG']
        condition_comparisons['r_taskL_vs_s_taskL'] = ['Stimulus_r_taskL', 'Stimulus_s_taskL']
        
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
        
        condition_comparisons['s25_vs_s75'] = ['s25', 's75'] # these cross-block comparisons let me decode if there's pre-trial information about the congruency proportion
        condition_comparisons['r25_vs_r75'] = ['r25', 'r75']
        
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

    if args.run_visualization_debug:
        print(f"\n{'='*20} 🔬 RUNNING 2D VISUALIZATION DEBUG (first two PCs and decision boundary) {'='*20}\n")
        
        # 1. Define the visualization pairs for each condition set
        viz_pairs = []
        if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
            print("Setting up LWPC visualization pairs...")
            viz_pairs = [(['c25'], ['i25']), (['c75'], ['i75'])]
            condition_comparison = 'LWPC_comparison'
        elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
            print("Setting up LWPS visualization pairs...")
            viz_pairs = [(['s25'], ['r25']), (['s75'], ['r75'])]
            condition_comparison = 'LWPS_comparison'
        elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
            print("Setting up Congruency x Switch Prop. visualization pairs...")
            viz_pairs = [
                (['Stimulus_c_in_25switchBlock'], ['Stimulus_i_in_25switchBlock']),
                (['Stimulus_c_in_75switchBlock'], ['Stimulus_i_in_75switchBlock'])
            ]
            condition_comparison = 'congruency_by_switch_proportion_comparison'
        elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
            print("Setting up Switch Type x Congruency Prop. visualization pairs...")
            viz_pairs = [
                (['Stimulus_s_in_25incongruentBlock'], ['Stimulus_r_in_25incongruentBlock']),
                (['Stimulus_s_in_75incongruentBlock'], ['Stimulus_r_in_75incongruentBlock'])
            ]
            condition_comparison = 'switch_type_by_congruency_proportion_comparison'
        elif args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:
            print("Setting up Task by Congruency visualization pairs...")
            viz_pairs = [
                (['Stimulus_i_taskG'], ['Stimulus_i_taskL']),
                (['Stimulus_c_taskG'], ['Stimulus_c_taskL'])
            ]
            condition_comparison = 'task_by_congruency_comparison'
        elif args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:
            print("Setting up Task by Switch Type visualization pairs...")
            viz_pairs = [
                (['Stimulus_s_taskG'], ['Stimulus_s_taskL']),
                (['Stimulus_r_taskG'], ['Stimulus_r_taskL'])
            ]
            condition_comparison = 'task_by_switch_type_comparison'
        if not viz_pairs:
            print("Warning: No visualization pairs defined for the current condition set. Skipping debug plots.")
        else:
            # 2. Get the single data sample for visualization
            print("Generating LabeledArray data for visualization (n_bootstraps=1)...")
            roi_labeled_arrays_viz = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
                rois=rois,
                subjects_data_objects=subjects_mne_objects,
                condition_names=condition_names, # This is already defined in main()
                subjects=args.subjects,
                electrodes_per_subject_roi=electrodes, # This is already defined in main()
                n_bootstraps=1,
                chans_axs=args.chans_axs,
                time_axs=args.time_axs,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
            roi_labeled_arrays_viz = {roi: arrs[0] for roi, arrs in roi_labeled_arrays_viz.items() if arrs}

            # 3. Loop through ROIs and Pairs and plot
            for roi in rois:
                if roi not in roi_labeled_arrays_viz:
                    print(f"Skipping visualization for {roi}: No data found.")
                    continue
                    
                for pair in viz_pairs:
                    viz_strings = pair
                    pair_name = f"{viz_strings[0][0]}_vs_{viz_strings[1][0]}"
                    print(f"\n--- Plotting for ROI: {roi}, Pair: {pair_name} ---")

                    try:
                        # 4. Get balanced data and 'cats'
                        data, labels, cats = concatenate_and_balance_data_for_decoding(
                            roi_labeled_arrays_viz, roi, viz_strings, args.obs_axs,
                            balance_method='subsample', # Must use subsample for this viz
                            random_state=args.random_state
                        )
                        if data.size == 0:
                            print("No data after balancing. Skipping plot.")
                            continue
                            
                        data_flat = data.reshape(data.shape[0], -1)

                        # 5. Create and FIT the FULL pipeline
                        # This uses the *exact* classifier and PCA settings from your args
                        full_pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('pca', PCA(n_components=args.explained_variance)), 
                            ('clf', args.clf_model) 
                        ])
                        
                        print(f"Fitting pipeline for {pair_name}...")
                        full_pipeline.fit(data_flat, labels)
                        print("Fit complete.")

                        # 6. Call the plotting function
                        plot_high_dim_decision_slice(
                            fitted_pipeline=full_pipeline,
                            X_data=data_flat,
                            y_labels=labels,
                            cats=cats,
                            roi=f"{roi} ({pair_name})", # Add pair info to title,
                            save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}")
                        )
                    except Exception as e:
                        print(f"!! FAILED to generate plot for {roi} - {pair_name}: {e}")
                        
        print(f"\n{'='*20} ✅ VISUALIZATION DEBUG COMPLETE {'='*20}\n")
        
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
            
            # --- Dynamically define colors and linestyles based on user's request ---
            
            # 1. Define the desired colors
            color_correct = 'green' # Green for correct prediction
            color_incorrect = 'red' # Red for incorrect prediction
            
            # 2. Infer the two class labels from the trace dictionary keys
            labels = set()
            for key in traces_dict.keys():
                try:
                    parts = key.split(',')
                    true_part = parts[0].replace('True: ', '').strip()
                    pred_part = parts[1].replace('Pred: ', '').strip()
                    labels.add(true_part)
                    labels.add(pred_part)
                except Exception:
                    continue # Skip malformed keys if any
            
            if len(labels) == 2:
                # Sort labels to get a consistent order (e.g., label1, label2)
                label1, label2 = sorted(list(labels))
                
                # 3. Build the dictionaries based on your logic:
                #    - Color: Green for correct (True == Pred), Red for incorrect (True != Pred)
                #    - Style: Solid for Predicted Class 1, Dotted for Predicted Class 2
                
                trace_colors = {
                    f'True: {label1}, Pred: {label1}': color_correct,   # TP1 (Correct)
                    f'True: {label1}, Pred: {label2}': color_incorrect, # FN1 (Incorrect)
                    f'True: {label2}, Pred: {label2}': color_correct,   # TP2 (Correct)
                    f'True: {label2}, Pred: {label1}': color_incorrect, # FN2 (Incorrect)
                }
                
                trace_linestyles = {
                    f'True: {label1}, Pred: {label1}': '-',   # Predicted Class 1 (Solid)
                    f'True: {label1}, Pred: {label2}': '--',   # Predicted Class 2 (dashed)
                    f'True: {label2}, Pred: {label2}': '--',   # Predicted Class 2 (dashed)
                    f'True: {label2}, Pred: {label1}': '-',   # Predicted Class 1 (Solid)
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
            
               
    # LWPC analysis
    if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        run_context_comparison_analysis(
            condition_name='LWPC',
            condition_comparison_1='c25_vs_i25',
            condition_comparison_2='c75_vs_i75',
            pooled_shuffle_key='lwpc_shuffle_accs_across_pooled_conditions',
            colors={
                'c25_vs_i25': '#FF7E79',
                'c75_vs_i75': '#FF7E79',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                'c25_vs_i25': '-',
                'c75_vs_i75': '--',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Congruency Decoding Accuracy",
            significance_label_1='25% > 75% I',
            significance_label_2='75% > 25% I',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
        )
                
    # LWPS analysis
    if args.conditions == experiment_conditions.stimulus_lwps_conditions:
        run_context_comparison_analysis(
            condition_name='LWPS',
            condition_comparison_1='s25_vs_r25',
            condition_comparison_2='s75_vs_r75',
            pooled_shuffle_key='lwps_shuffle_accs_across_pooled_conditions',
            colors={
                's25_vs_r25': '#05B0F0',
                's75_vs_r75': '#05B0F0',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                's25_vs_r25': '-',
                's75_vs_r75': '--',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Switch Type Decoding Accuracy",
            significance_label_1='25% > 75% S',
            significance_label_2='75% > 25% S',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
        )
            
    # Congruency by switch proportion
    if args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        run_context_comparison_analysis(
            condition_name='congruency_by_switch_proportion',
            condition_comparison_1='c_in_25switchBlock_vs_i_in_25switchBlock',
            condition_comparison_2='c_in_75switchBlock_vs_i_in_75switchBlock',
            pooled_shuffle_key='congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                'c_in_25switchBlock_vs_i_in_25switchBlock': '#05B0F0',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '#05B0F0',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                'c_in_25switchBlock_vs_i_in_25switchBlock': '-',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '--',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Congruency Decoding Accuracy",
            significance_label_1='C/I (25% S) > C/I (75% S)',
            significance_label_2='C/I (75% S) > C/I (25% S)',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
        )

    # Switch type by congruency proportion
    if args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        run_context_comparison_analysis(
            condition_name='switch_type_by_congruency_proportion',
            condition_comparison_1='s_in_25incongruentBlock_vs_r_in_25incongruentBlock',
            condition_comparison_2='s_in_75incongruentBlock_vs_r_in_75incongruentBlock',
            pooled_shuffle_key='switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '#FF7E79',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#FF7E79',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '-',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '--',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Switch Type Decoding Accuracy",
            significance_label_1='S/R (25% I) > S/R (75% I)',
            significance_label_2='S/R (75% I) > S/R (25% I)',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
        )

    # Task by congruency
    if args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:
        run_context_comparison_analysis(
            condition_name='task_by_congruency',
            condition_comparison_1='c_taskG_vs_c_taskL',
            condition_comparison_2='i_taskG_vs_i_taskL',
            pooled_shuffle_key='task_by_congruency_shuffle_accs_across_pooled_conditions',
            colors={
                'c_taskG_vs_c_taskL': '#05B0F0',
                'i_taskG_vs_i_taskL': '#05B0F0',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                'c_taskG_vs_c_taskL': '-',
                'i_taskG_vs_i_taskL': '--',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Task Decoding Accuracy",
            significance_label_1='Task (C) > Task (I)',
            significance_label_2='Task (I) > Task (C)',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
        )

    # Task by switch type
    if args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:
        run_context_comparison_analysis(
            condition_name='task_by_switch_type',
            condition_comparison_1='s_taskG_vs_s_taskL',
            condition_comparison_2='r_taskG_vs_r_taskL',
            pooled_shuffle_key='task_by_switch_type_shuffle_accs_across_pooled_conditions',
            colors={
                's_taskG_vs_s_taskL': '#05B0F0',
                'r_taskG_vs_r_taskL': '#05B0F0',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494'
            },
            linestyles={
                's_taskG_vs_s_taskL': '-',
                'r_taskG_vs_r_taskL': '--',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '--'
            },
            ylabel="Task Decoding Accuracy",
            significance_label_1='Task (S) > Task (R)',
            significance_label_2='Task (R) > Task (S)',
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str
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
