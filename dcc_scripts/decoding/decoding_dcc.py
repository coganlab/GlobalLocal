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
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.mat import LabeledArray, combine
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

import matplotlib.pyplot as plt

from pandas import read_csv
import scipy.stats as stats
import joblib

from scipy.ndimage import label
from scipy.stats import norm, ttest_ind

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
from ieeg.decoding.decoders import PcaLdaClassification
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
    plot_accuracies_nature_style,
    make_pooled_shuffle_distribution,
    find_significant_clusters_of_series_vs_distribution_based_on_percentile
)

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
    elif args.conditions == experiment_conditions.stimulus_err_corr_conditions:
        conditions_save_name = 'stimulus_err_corr_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        conditions_save_name = 'stimulus_congruency_by_switch_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        conditions_save_name = 'stimulus_switch_type_by_congruency_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    
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
    
    save_dir = os.path.join(LAB_root, 'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'decoding', 'figs', f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")
    
    sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)
      
    subjects_mne_objects = create_subjects_mne_objects_dict(subjects=args.subjects, epochs_root_file=args.epochs_root_file, conditions=args.conditions, task="GlobalLocal", just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only)
    
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
    
    roi_bootstrapped_labeled_arrays = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
        rois=rois,
        subjects_data_objects=subjects_mne_objects,
        condition_names=condition_names,
        subjects=args.subjects,
        electrodes_per_subject_roi=electrodes,
        n_bootstraps=args.bootstraps,
        chans_axs=1,
        time_axs=2,
        freq_axs=None, # Set to 3 if using TFR data
        random_state=args.random_state,
        n_jobs=-1 
    )
    
    # old method of getting subject labeled arrays and concatenating them to make roi labeled array
    # roi_labeled_arrays = put_data_in_labeled_array_per_roi_subject(
    #     subjects_mne_objects,
    #     condition_names,
    #     rois,
    #     args.subjects,
    #     electrodes, 
    #     obs_axs=args.obs_axs,  # Trials dimension (ignoring the conditions dimension for now cuz we iterate over it)
    #     chans_axs=args.chans_axs,  # Channels dimension
    #     time_axs=args.time_axs,   # Time dimension
    #     random_state=args.random_state  # For reproducibility
    # )
            
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
        condition_comparisons['responseType'] = [['Stimulus_err'], ['Stimulus_corr']]
        
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
        
        condition_comparisons['s25_vs_r25'] = ['s25', 'r25'] # these cross-block comparisons let me decode if there's pre-trial information about the switch proportion
        condition_comparisons['s75_vs_r75'] = ['s75', 'r75'] # these cross-block comparisons let me decode if there's pre-trial information about the switch proportion

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
        f"{elec_string_to_add_to_filename}_{str(len(args.subjects))}_subjects_{folds_info_str}"
    )
    
    # make a dict to store the final statistical results (e.g., significance masks) for each comparison and ROI, aggregated across all bootstraps
    aggregated_bootstrap_stats_results = {}
    for roi in rois:
        # Initialize keys for all possible comparisons
        for condition_comparison in condition_comparisons.keys():
            aggregated_bootstrap_stats_results[(condition_comparison, roi)] = []
        # Add special comparison keys
        if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
            aggregated_bootstrap_stats_results[('lwpc', roi)] = []    # loop over each bootstrap sample
        if args.conditions == experiment_conditions.stimulus_lwps_conditions:
            aggregated_bootstrap_stats_results[('lwps', roi)] = []
        if args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
            aggregated_bootstrap_stats_results[('congruency_by_switch_proportion', roi)] = []
        if args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
            aggregated_bootstrap_stats_results[('switch_type_by_congruency_proportion', roi)] = []
            
    time_window_decoding_results = {}     
     
    for bootstrap_idx in range(args.bootstraps):
        time_window_decoding_results[bootstrap_idx] = {}
        print(f"\n{'='*20} PROCESSING BOOTSTRAP {bootstrap_idx+1}/{args.bootstraps} {'='*20}\n")
        
        # get the specific LabeledArray dict for this bootstrap iteration
        roi_labeled_arrays_this_bootstrap = {
            roi: bootstraps[bootstrap_idx]
            for roi, bootstraps in roi_bootstrapped_labeled_arrays.items()
            if len(bootstraps) > bootstrap_idx
        }
        for condition_comparison, strings_to_find in condition_comparisons.items():
            
            condition_save_dir = os.path.join(save_dir, f"{condition_comparison}")
            os.makedirs(condition_save_dir, exist_ok=True)
            
            confusion_matrices = get_and_plot_confusion_matrix_for_rois_jim(
                timestamp=args.timestamp,
                roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                rois=rois,
                condition_comparison=condition_comparison,
                strings_to_find=strings_to_find,
                save_dir=condition_save_dir,
                time_interval_name=None,
                other_string_to_add=elec_string_to_add_to_filename,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                obs_axs=args.obs_axs,
                balance_method=args.balance_method,  # Use 'subsample' to balance by subsampling
                explained_variance=args.explained_variance,
                random_state=args.random_state  # For reproducibility
            )
            
        # Main code
        # Directory to save confusion matrices
        cm_save_dir = os.path.join(save_dir, "confusion_matrices")
        os.makedirs(cm_save_dir, exist_ok=True)

        for condition_comparison, strings_to_find in condition_comparisons.items():
            
            time_window_decoding_results[bootstrap_idx][condition_comparison] = {}
            
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
                balance_method=args.balance_method,
                explained_variance=args.explained_variance,
                random_state=args.random_state,
                window_size=args.window_size,
                step_size=args.step_size,
                n_perm=args.n_perm,
                sampling_rate=args.sampling_rate,
                first_time_point=-1,
                folds_as_samples=args.folds_as_samples
            )

            filename_base = (
                f'{condition_comparison}_{args.n_splits}_splits_{args.n_repeats}_repeats_'
                f'{args.balance_method}_balance_method_{args.random_state}_random_state_'
                f'{args.window_size}_window_size_{args.step_size}_step_size_{args.n_perm}_permutations_'
                f'{folds_info_str}_{args.sampling_rate}_sampling_rate'
            )

            # Save the numpy arrays with the new filename
            np.save(os.path.join(cm_save_dir, f'{filename_base}_cm_true_per_roi.npy'), cm_true_per_roi)
            np.save(os.path.join(cm_save_dir, f'{filename_base}_cm_shuffle_per_roi.npy'), cm_shuffle_per_roi)

            # Now compute accuracies and perform time permutation cluster test
            condition_save_dir = os.path.join(save_dir, f"{condition_comparison}")
            os.makedirs(condition_save_dir, exist_ok=True)

            for roi in rois:
                condition_roi_stat_func_save_dir = os.path.join(condition_save_dir, f"{roi}", f"{args.stat_func_str}")
                os.makedirs(condition_roi_stat_func_save_dir, exist_ok=True)
                print(f"accuracies save dir directory created or already exists at: {condition_roi_stat_func_save_dir}")
                
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi] = {}
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['strings_to_find'] = strings_to_find

                cm_true = cm_true_per_roi[roi]['cm_true']
                cm_shuffle = cm_shuffle_per_roi[roi]['cm_shuffle']
                time_window_centers = cm_true_per_roi[roi]['time_window_centers']
                window_size = cm_true_per_roi[roi]['window_size']
                step_size = cm_true_per_roi[roi]['step_size']

                # store cm outputs nd windowing parameters
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['cm_true'] = cm_true
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['cm_shuffle'] = cm_shuffle
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['time_window_centers'] = time_window_centers
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['window_size'] = window_size
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['step_size'] = step_size
                
                # Compute accuracies
                accuracies_true, accuracies_shuffle = compute_accuracies(cm_true, cm_shuffle)

                # find average accuracies, either across folds or repeats/perms, depending on whether folds_as_samples is set to true or not
                mean_accuracies_true = np.mean(accuracies_true, axis=1, keepdims=True) # Shape: (n_windows, 1), keeping folds/repeats dimension for compatibility with the percentile function
                mean_accuracies_shuffle = np.mean(accuracies_shuffle, axis=1, keepdims=True) 
                
                # store accuracies - TODO: average across bootstraps somehow for these
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['accuracies_true'] = accuracies_true
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['accuracies_shuffle'] = accuracies_shuffle
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['mean_accuracies_true'] = mean_accuracies_true
                time_window_decoding_results[bootstrap_idx][condition_comparison][roi]['mean_accuracies_shuffle'] = mean_accuracies_shuffle
                
        # do lwpc comparison 
        if args.conditions == experiment_conditions.stimulus_lwpc_conditions:       
            for roi in rois:
                time_window_centers = time_window_decoding_results[bootstrap_idx]['c25_vs_i25'][roi]['time_window_centers']
                c25_vs_i25_acc = time_window_decoding_results[bootstrap_idx]['c25_vs_i25'][roi]['accuracies_true']
                c75_vs_i75_acc = time_window_decoding_results[bootstrap_idx]['c75_vs_i75'][roi]['accuracies_true']
                
                # doing a one-sided test first, but could do a two-sided test because either could be higher than the other, just find when they're different
                lwpc_significant_clusters, lwpc_p_values = time_perm_cluster(
                    c25_vs_i25_acc.T,
                    c75_vs_i75_acc.T,
                    p_thresh=args.p_thresh,
                    n_perm=args.n_perm,
                    tails=2,
                    axis=0, 
                    stat_func=args.stat_func,
                    n_jobs=args.n_jobs,
                    seed=args.random_state
                )

                # store boolean mask of significant lwpc clusters in bootstrap stats dict
                aggregated_bootstrap_stats_results[('lwpc', roi)].append(lwpc_significant_clusters)
                
                # get i vs c pooled shuffle distribution
                strings_to_find_pooled = [['c25', 'c75'], ['i25', 'i75']]
                
                accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                    roi=roi,
                    roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                    strings_to_find_pooled=strings_to_find_pooled,
                    explained_variance=args.explained_variance,
                    n_splits=args.n_splits,
                    n_perm=args.n_perm,
                    random_state=args.random_state,
                    balance_method='subsample', # Subsampling is recommended for pooling
                    obs_axs=args.obs_axs,
                    window_size=args.window_size,
                    step_size=args.step_size
                )
                
                # Plot accuracies comparing c25_vs_i25 and c75_vs_i75 for this condition comparison and roi
                # For LWPC comparisons
                accuracies_dict = {
                    'c25_vs_i25': c25_vs_i25_acc,
                    'c75_vs_i75': c75_vs_i75_acc,
                    'pooled_shuffle': accuracies_shuffle_pooled
                }

                colors = {
                    'c25_vs_i25': '#0173B2',  # Blue
                    'c75_vs_i75': '#DE8F05' ,   # Orange
                    'pooled_shuffle': '#949494'  # Gray
                }
                
                linestyles = {
                    'c25_vs_i25': '-',  # Solid
                    'c75_vs_i75': '-',  # Solid
                    'pooled_shuffle': '--'                       # Dashed
                }

                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict=accuracies_dict,
                    significant_clusters=lwpc_significant_clusters,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'lwpc_comparison_{roi}',
                    roi=roi,
                    save_dir=os.path.join(condition_save_dir, 'lwpc_plots'), # Save in a sub-directory
                    timestamp=args.timestamp,
                    p_thresh=args.p_thresh,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=False,
                    ylim=(0.4, 0.75),
                    show_chance_level=False # The pooled shuffle line is the new chance level
                )

        # do lwps comparison 
        if args.conditions == experiment_conditions.stimulus_lwps_conditions:       
            for roi in rois:
                time_window_centers = time_window_decoding_results[bootstrap_idx]['s25_vs_r25'][roi]['time_window_centers']
                s25_vs_r25_acc = time_window_decoding_results[bootstrap_idx]['s25_vs_r25'][roi]['accuracies_true']
                s75_vs_r75_acc = time_window_decoding_results[bootstrap_idx]['s75_vs_r75'][roi]['accuracies_true']
                
                # doing a one-sided test first, but could do a two-sided test because either could be higher than the other, just find when they're different
                lwps_significant_clusters, lwps_p_values = time_perm_cluster(
                    s25_vs_r25_acc.T,
                    s75_vs_r75_acc.T,
                    p_thresh=args.p_thresh,
                    n_perm=args.n_perm,
                    tails=2,
                    axis=0, 
                    stat_func=args.stat_func,
                    n_jobs=args.n_jobs,
                    seed=args.random_state
                )
                
                # store boolean mask of significant lwps clusters in bootstrap stats dict
                aggregated_bootstrap_stats_results[('lwps', roi)].append(lwps_significant_clusters)
                
                # get s vs r pooled shuffle distribution
                strings_to_find_pooled = [['s25', 's75'], ['r25', 'r75']]
                
                accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                    roi=roi,
                    roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                    strings_to_find_pooled=strings_to_find_pooled,
                    explained_variance=args.explained_variance,
                    n_splits=args.n_splits,
                    n_perm=args.n_perm,
                    random_state=args.random_state,
                    balance_method='subsample', # Subsampling is recommended for pooling
                    obs_axs=args.obs_axs,
                    window_size=args.window_size,
                    step_size=args.step_size
                )
                # Plot accuracies comparing s25_vs_r25 and s75_vs_r75 for this condition comparison and roi
                # For LWPS comparisons
                accuracies_dict = {
                    's25_vs_r25': s25_vs_r25_acc,
                    's75_vs_r75': s75_vs_r75_acc,
                    'pooled_shuffle': accuracies_shuffle_pooled
                }

                colors = {
                    's25_vs_r25': '#0173B2',  # Blue
                    's75_vs_r75': '#DE8F05' ,   # Orange
                    'pooled_shuffle': '#949494'  # Gray
                }
                
                linestyles = {
                    's25_vs_r25': '-',  # Solid
                    's75_vs_r75': '-',  # Solid
                    'pooled_shuffle': '--'                       # Dashed
                }

                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict=accuracies_dict,
                    significant_clusters=lwps_significant_clusters,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'lwps_comparison_{roi}',
                    roi=roi,
                    save_dir=os.path.join(condition_save_dir, 'lwps_plots'), # Save in a sub-directory
                    timestamp=args.timestamp,
                    p_thresh=args.p_thresh,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=False,
                    ylim=(0.4, 0.75),
                    show_chance_level=False # The pooled shuffle line is the new chance level
                )
                
        # do congruency by switch proportion comparison 
        if args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:       
            for roi in rois:
                time_window_centers = time_window_decoding_results[bootstrap_idx]['c_in_25switchBlock_vs_i_in_25switchBlock'][roi]['time_window_centers']
                c_in_25switchBlock_vs_i_in_25switchBlock_acc = time_window_decoding_results[bootstrap_idx]['c_in_25switchBlock_vs_i_in_25switchBlock'][roi]['accuracies_true']
                c_in_75switchBlock_vs_i_in_75switchBlock_acc = time_window_decoding_results[bootstrap_idx]['c_in_75switchBlock_vs_i_in_75switchBlock'][roi]['accuracies_true']
                
                # doing a one-sided test first, but could do a two-sided test because either could be higher than the other, just find when they're different
                significant_clusters, p_values = time_perm_cluster(
                    c_in_25switchBlock_vs_i_in_25switchBlock_acc.T,
                    c_in_75switchBlock_vs_i_in_75switchBlock_acc.T,
                    p_thresh=args.p_thresh,
                    n_perm=args.n_perm,
                    tails=2,
                    axis=0, 
                    stat_func=args.stat_func,
                    n_jobs=args.n_jobs,
                    seed=args.random_state
                )
                
                aggregated_bootstrap_stats_results[('congruency_by_switch_proportion', roi)] = []
                aggregated_bootstrap_stats_results[('congruency_by_switch_proportion', roi)].append(significant_clusters)
                
                # get i vs c pooled shuffle distribution - i think this can just be the same as before, it just needs to grab all i trials nad all c trials
                strings_to_find_pooled = [['c_in'], ['i_in']]
                
                accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                    roi=roi,
                    roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                    strings_to_find_pooled=strings_to_find_pooled,
                    explained_variance=args.explained_variance,
                    n_splits=args.n_splits,
                    n_perm=args.n_perm,
                    random_state=args.random_state,
                    balance_method='subsample', # Subsampling is recommended for pooling
                    obs_axs=args.obs_axs,
                    window_size=args.window_size,
                    step_size=args.step_size
                )
                # Plot accuracies for this condition comparison and roi
                accuracies_dict = {
                    'c_in_25switchBlock_vs_i_in_25switchBlock': c_in_25switchBlock_vs_i_in_25switchBlock_acc,
                    'c_in_75switchBlock_vs_i_in_75switchBlock': c_in_75switchBlock_vs_i_in_75switchBlock_acc,
                    'pooled_shuffle': accuracies_shuffle_pooled
                }

                colors = {
                    'c_in_25switchBlock_vs_i_in_25switchBlock': '#0173B2',  # Blue
                    'c_in_75switchBlock_vs_i_in_75switchBlock': '#DE8F05' ,   # Orange
                    'pooled_shuffle': '#949494'  # Gray
                }
                
                linestyles = {
                    'c_in_25switchBlock_vs_i_in_25switchBlock': '-',  # Solid
                    'c_in_75switchBlock_vs_i_in_75switchBlock': '-',  # Solid
                    'pooled_shuffle': '--'                       # Dashed
                }

                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict=accuracies_dict,
                    significant_clusters=significant_clusters,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'congruency_by_switch_proportion_comparison_{roi}',
                    roi=roi,
                    save_dir=os.path.join(condition_save_dir, 'congruency_by_switch_proportion_plots'), # Save in a sub-directory
                    timestamp=args.timestamp,
                    p_thresh=args.p_thresh,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=False,
                    ylim=(0.4, 0.75),
                    show_chance_level=False # The pooled shuffle line is the new chance level
                )

        # do switch type by congruency proportion comparison 
        if args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:       
            for roi in rois:
                time_window_centers = time_window_decoding_results[bootstrap_idx]['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'][roi]['time_window_centers']
                s_in_25incongruentBlock_vs_r_in_25incongruentBlock_acc = time_window_decoding_results[bootstrap_idx]['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'][roi]['accuracies_true']
                s_in_75incongruentBlock_vs_r_in_75incongruentBlock_acc = time_window_decoding_results[bootstrap_idx]['s_in_75incongruentBlock_vs_r_in_75incongruentBlock'][roi]['accuracies_true']

                # doing a one-sided test first, but could do a two-sided test because either could be higher than the other, just find when they're different
                significant_clusters, p_values = time_perm_cluster(
                    s_in_25incongruentBlock_vs_r_in_25incongruentBlock_acc.T,
                    s_in_75incongruentBlock_vs_r_in_75incongruentBlock_acc.T,
                    p_thresh=args.p_thresh,
                    n_perm=args.n_perm,
                    tails=2,
                    axis=0, 
                    stat_func=args.stat_func,
                    n_jobs=args.n_jobs,
                    seed=args.random_state
                )
                
                aggregated_bootstrap_stats_results[('switch_type_by_congruency_proportion', roi)] = []
                aggregated_bootstrap_stats_results[('switch_type_by_congruency_proportion', roi)].append(significant_clusters)
                
                # get s vs r pooled shuffle distribution - i think this can just be the same as before, it just needs to grab all s trials nad all r trials
                strings_to_find_pooled = [['s_in'], ['r_in']]
                
                accuracies_shuffle_pooled = make_pooled_shuffle_distribution(
                    roi=roi,
                    roi_labeled_arrays=roi_labeled_arrays_this_bootstrap,
                    strings_to_find_pooled=strings_to_find_pooled,
                    explained_variance=args.explained_variance,
                    n_splits=args.n_splits,
                    n_perm=args.n_perm,
                    random_state=args.random_state,
                    balance_method='subsample', # Subsampling is recommended for pooling
                    obs_axs=args.obs_axs,
                    window_size=args.window_size,
                    step_size=args.step_size
                )
                # Plot accuracies comparing c25_vs_i25 and c75_vs_i75 for this condition comparison and roi
                # For LWPC comparisons
                accuracies_dict = {
                    's_in_25incongruentBlock_vs_r_in_25incongruentBlock': s_in_25incongruentBlock_vs_r_in_25incongruentBlock_acc,
                    's_in_75incongruentBlock_vs_r_in_75incongruentBlock': s_in_75incongruentBlock_vs_r_in_75incongruentBlock_acc,
                    'pooled_shuffle': accuracies_shuffle_pooled
                }

                colors = {
                    's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '#0173B2',  # Blue
                    's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#DE8F05' ,   # Orange
                    'pooled_shuffle': '#949494'  # Gray
                }
                
                linestyles = {
                    's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '-',  # Solid
                    's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '-',  # Solid
                    'pooled_shuffle': '--'                       # Dashed
                }

                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict=accuracies_dict,
                    significant_clusters=significant_clusters,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'switch_type_by_congruency_proportion_comparison_{roi}',
                    roi=roi,
                    save_dir=os.path.join(condition_save_dir, 'switch_type_by_congruency_proportion_plots'), # Save in a sub-directory
                    timestamp=args.timestamp,
                    p_thresh=args.p_thresh,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=False,
                    ylim=(0.4, 0.75),
                    show_chance_level=False # The pooled shuffle line is the new chance level
                )
    
    # after all bootstraps complete, run pooled statistics
    pooled_bootstrap_stats = compute_pooled_bootstrap_statistics(
        time_window_decoding_results,
        args.bootstraps,
        condition_comparisons,
        rois,
        percentile=args.percentile,
        cluster_percentile=args.cluster_percentile,
        n_cluster_perms=args.n_cluster_perms,
        random_state=args.random_state
    )
    
    
                    

                    
    # then plot using the pooled statistics
    for condition_comparison in condition_comparisons.keys():
        for roi in rois:
            if roi in pooled_stats[condition_comparison]:
                stats = pooled_stats[condition_comparison][roi] 
                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict={
                        'true': np.array([stats['mean_true']]), # reshape for plotting
                        'shuffle': np.array([stats['mean_shuffle']])
                    },
                    significant_clusters=stats['significant_clusters'],
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'bootstrap_true_vs_shuffle_{condition_comparison}_{roi}',
                    roi=roi,
                    save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                    timestamp=args.timestamp,
                    p_thresh=args.percentile,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=False,
                    ylim=(0.4, 0.75),
                    show_chance_level=False # The pooled shuffle line is the new chance level
                    
                )     
            
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
