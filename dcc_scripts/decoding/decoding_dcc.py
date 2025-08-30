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

# TODO: hmm fix these utils imports, import the funcs individually. 6/1/25.
from src.analysis.utils.general_utils import *
from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict
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
    get_data_in_time_range
)

from src.analysis.decoding.decoding import (
    concatenate_and_balance_data_for_decoding, 
    get_and_plot_confusion_matrix_for_rois_jim,
    Decoder, 
    windower,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    plot_accuracies
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
        
    save_dir = os.path.join(LAB_root, 'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'decoding', 'figs', f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")
    
    sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)
      
    subjects_mne_objects = create_subjects_mne_objects_dict(subjects=args.subjects, epochs_root_file=args.epochs_root_file, conditions=args.conditions, task="GlobalLocal", just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only)

    # TODO: set electrodes as an input parameter (which electrodes to use)
    if args.electrodes == 'all':
        electrodes = all_electrodes_per_subject_roi # toggle this to sig_electrodes_per_subject_roi if just using sig elecs, or electrodes_per_subject_roi if using all elecs
    elif args.electrodes == 'sig':
        electrodes = sig_electrodes_per_subject_roi
    else:
        raise ValueError("electrodes input must be set to all or sig")
    
    if electrodes == all_electrodes_per_subject_roi:
        elec_string_to_add_to_filename = 'all_elecs'
    elif electrodes == sig_electrodes_per_subject_roi:
        elec_string_to_add_to_filename = 'sig_elecs'
    else:
        elec_string_to_add_to_filename = None
    
    roi_labeled_arrays = put_data_in_labeled_array_per_roi_subject(
        subjects_mne_objects,
        condition_names,
        rois,
        args.subjects,
        electrodes, 
        obs_axs=args.obs_axs,  # Trials dimension (ignoring the conditions dimension for now cuz we iterate over it)
        chans_axs=args.chans_axs,  # Channels dimension
        time_axs=args.time_axs,   # Time dimension
        random_state=args.random_state  # For reproducibility
    )
    
    roi_labeled_arrays_no_nans, conditions_with_no_valid_trials_per_roi = remove_nans_from_all_roi_labeled_arrays(roi_labeled_arrays, obs_axs=args.obs_axs, chans_axs=args.chans_axs, time_axs=args.time_axs)
        
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

    # get the confusion matrix using the downsampled version
    # add elec and subject info to filename 6/11/25
    other_string_to_add = elec_string_to_add_to_filename + '_' + str(len(args.subjects)) + '_subjects'

    for condition_comparison, strings_to_find in condition_comparisons.items():
        
        condition_save_dir = os.path.join(save_dir, f"{condition_comparison}")
        os.makedirs(condition_save_dir, exist_ok=True)
        
        confusion_matrices = get_and_plot_confusion_matrix_for_rois_jim(
            timestamp=args.timestamp,
            roi_labeled_arrays=roi_labeled_arrays_no_nans,
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

    time_window_decoding_results = {}

    for condition_comparison, strings_to_find in condition_comparisons.items():
        
        time_window_decoding_results[condition_comparison] = {}
        
        # Get confusion matrices for each ROI
        cm_true_per_roi, cm_shuffle_per_roi = get_confusion_matrices_for_rois_time_window_decoding_jim(
            roi_labeled_arrays=roi_labeled_arrays,
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
            first_time_point=-1
        )

        np.save(os.path.join(cm_save_dir, f'{condition_comparison}_{args.n_splits}_splits_{args.n_repeats}_repeats_{args.balance_method}_balance_method_{args.random_state}_random_state_{args.window_size}_window_size_{args.step_size}_step_size_{args.n_perm}_permutations_{args.sampling_rate}_sampling_rate_cm_true_per_roi.npy'), cm_true_per_roi)
        np.save(os.path.join(cm_save_dir, f'{condition_comparison}_{args.n_splits}_splits_{args.n_repeats}_repeats_{args.balance_method}_balance_method_{args.random_state}_random_state_{args.window_size}_window_size_{args.step_size}_step_size_{args.n_perm}_permutations_{args.sampling_rate}_sampling_rate_cm_shuffle_per_roi.npy'), cm_shuffle_per_roi)

        # Store the results in a dictionary
        # time_window_decoding_results[condition_comparison] = {
        #     'strings_to_find': strings_to_find,
        #     'cm_true_per_roi': cm_true_per_roi,
        #     'cm_shuffle_per_roi': cm_shuffle_per_roi
        # }

        # Now compute accuracies and perform time permutation cluster test
        for roi in rois:
            
            condition_roi_save_dir = os.path.join(save_dir, f"{condition_comparison}", f"{roi}")
            os.makedirs(condition_roi_save_dir, exist_ok=True)
            print(f"accuracies save dir directory created or already exists at: {condition_roi_save_dir}")
            
            time_window_decoding_results[condition_comparison][roi] = {}
            time_window_decoding_results[condition_comparison][roi]['strings_to_find'] = strings_to_find


            cm_true = cm_true_per_roi[roi]['cm_true']
            cm_shuffle = cm_shuffle_per_roi[roi]['cm_shuffle']
            time_window_centers = cm_true_per_roi[roi]['time_window_centers']
            window_size = cm_true_per_roi[roi]['window_size']
            step_size = cm_true_per_roi[roi]['step_size']

            # store cm outputs nd windowing parameters
            time_window_decoding_results[condition_comparison][roi]['cm_true'] = cm_true
            time_window_decoding_results[condition_comparison][roi]['cm_shuffle'] = cm_shuffle
            time_window_decoding_results[condition_comparison][roi]['time_window_centers'] = time_window_centers
            time_window_decoding_results[condition_comparison][roi]['window_size'] = window_size
            time_window_decoding_results[condition_comparison][roi]['step_size'] = step_size
            
            # Compute accuracies
            accuracies_true, accuracies_shuffle = compute_accuracies(cm_true, cm_shuffle)

            # store accuracies
            time_window_decoding_results[condition_comparison][roi]['accuracies_true'] = accuracies_true
            time_window_decoding_results[condition_comparison][roi]['accuracies_shuffle'] = accuracies_shuffle
            
            # Perform time permutation cluster test
            significant_clusters, p_values = time_perm_cluster(
                accuracies_true.T, # shape is (n_windows, n_repeats), we want to shuffle along n_repeats
                accuracies_shuffle.T,
                p_thresh=args.p_thresh,
                n_perm=args.n_perm,
                tails=1,
                axis=0, 
                stat_func=args.stat_func,
                n_jobs=args.n_jobs,
                seed=args.random_state
            )

            # Store significant clusters and p-values
            time_window_decoding_results[condition_comparison][roi]['significant_clusters'] = significant_clusters
            time_window_decoding_results[condition_comparison][roi]['p_values'] = p_values
    
            # Plot accuracies comparing true and shuffle for this condition comparison and roi
            plot_accuracies(
                time_points=time_window_centers,
                accuracies_true=accuracies_true,
                accuracies_shuffle=accuracies_shuffle,
                significant_clusters=significant_clusters,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                condition_comparison=condition_comparison,
                roi=roi,
                save_dir=condition_roi_save_dir,
                timestamp=args.timestamp,
                p_thresh=args.p_thresh
            )
            
            # # convert to a dataframe for further comparisons if necessary (lwpc, lwps) - untested and unfinished
            # time_window_decoding_results_df = pd.DataFrame.from_dict(time_window_decoding_results, orient='index')
    
    # do comparisons for lwpc decoding accuracies - UNTESTED AND UNFINISHED
    # if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
    #     c25_vs_i25_df = time_window_decoding_results_df[condition_comparison == 'c25_vs_i25']
    #     c75_vs_i75_df = time_window_decoding_results_df[condition_comparison == 'c75_vs_i75']
    #     c25_vs_i75_df = time_window_decoding_results_df[condition_comparison == 'c25_vs_i75']
    #     c75_vs_i25_df = time_window_decoding_results_df[condition_comparison == 'c75_vs_i25']

    #     for roi in rois:
    #         # Perform time permutation cluster test between c25 vs i25 and c75 vs i75
    #         significant_clusters, p_values = time_perm_cluster(
    #             c25_vs_i25_df[roi][accuracies_true.T], # shape is (n_windows, n_repeats), we want to shuffle along n_repeats
    #             c75_vs_i75_df[roi][accuracies_true.T],
    #             p_thresh=args.p_thresh,
    #             n_perm=args.n_perm,
    #             tails=1,
    #             axis=0, 
    #             stat_func=args.stat_func,
    #             n_jobs=args.n_jobs,
    #             seed=args.random_state
    #         )
            
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
