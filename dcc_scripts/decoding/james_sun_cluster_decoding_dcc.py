import sys
import os
import argparse
import gc
import glob
import json
import pickle
import copy
from functools import partial
from typing import Union, List, Sequence, Dict
from os.path import join, expanduser, basename

print(sys.path)
# sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")  # need to do this cuz otherwise ieeg isn't added to path...

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

# External libraries
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.stats as stats
from scipy.ndimage import label
from scipy.stats import norm, ttest_ind
import joblib
import tqdm

# MNE imports
import mne
import mne.time_frequency

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# IEEG imports
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.mat import LabeledArray, combine
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from ieeg.viz.parula import parula_map
from ieeg.decoding.decoders import PcaLdaClassification

# Project-specific imports
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
    perform_time_perm_cluster_test_for_accuracies,
    plot_accuracies,
    mixup2,
    plot_and_save_confusion_matrix,
    get_confusion_matrix_for_rois_tfr_cluster
)
from src.analysis.spec.wavelet_functions import (
    get_uncorrected_wavelets,
    get_uncorrected_multitaper,
    get_sig_tfr_differences,
    plot_mask_pages
)
from src.analysis.spec.subjects_tfr_objects_functions import (
    make_subjects_tfr_objects,
    get_sig_tfr_differences_per_subject,
    get_sig_tfr_differences_per_roi
)
from src.analysis.utils.general_utils import (
    load_subjects_electrodes_to_ROIs_dict,
    get_good_data,
    get_sig_chans_per_subject,
    make_sig_electrodes_per_subject_and_roi_dict,
    calculate_total_electrodes,
    check_sampling_rates
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

    layout = get_data(args.task, root=LAB_root)

    condition_names = list(args.conditions.keys()) # get the condition names as a list

    if args.conditions == experiment_conditions.stimulus_conditions:
        conditions_save_name = 'stimulus_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_experiment_conditions:
        conditions_save_name = 'stimulus_experiment_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_main_effect_conditions:
        conditions_save_name = 'stimulus_main_effect_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        conditions_save_name = 'stimulus_lwpc_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        conditions_save_name = 'stimulus_lwps_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_big_letter_conditions:
        conditions_save_name = 'stimulus_big_letter_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_small_letter_conditions:
        conditions_save_name = 'stimulus_small_letter_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_task_conditions:
        conditions_save_name = 'stimulus_task_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_conditions:
        conditions_save_name = 'stimulus_congruency_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        conditions_save_name = 'stimulus_switch_type_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'

    elif args.conditions == experiment_conditions.response_conditions:
        conditions_save_name = 'response_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_experiment_conditions:
        conditions_save_name = 'response_experiment_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_big_letter_conditions:
        conditions_save_name = 'response_big_letter_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_small_letter_conditions:
        conditions_save_name = 'response_small_letter_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_task_conditions:
        conditions_save_name = 'response_task_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_congruency_conditions:
        conditions_save_name = 'response_congruency_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_switch_type_conditions:
        conditions_save_name = 'response_switch_type_conditions' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'

    sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)

    subjects_tfr_objects = make_subjects_tfr_objects(
        layout=layout,
        spec_method=args.spec_method,
        conditions_save_name=conditions_save_name,
        subjects=args.subjects,
        conditions=args.conditions,
        signal_times=args.signal_times,
        freqs=args.freqs,
        n_cycles=args.n_cycles,
        time_bandwidth=args.time_bandwidth,
        return_itc=args.return_itc,
        n_jobs=args.n_jobs,
        average=args.average,
        acc_trials_only=args.acc_trials_only,
        error_trials_only=args.error_trials_only 
    )
    
    # This part is fine for getting the time and frequency axes
    first_sub = args.subjects[0]
    first_condition = list(subjects_tfr_objects[first_sub].keys())[0]
    times = subjects_tfr_objects[first_sub][first_condition].times
    freqs = subjects_tfr_objects[first_sub][first_condition].freqs
    
    # TODO: set electrodes as an input parameter (which electrodes to use)
    electrodes = all_electrodes_per_subject_roi # toggle this to sig_electrodes_per_subject_roi if just using sig elecs, or electrodes_per_subject_roi if using all elecs

    if electrodes == all_electrodes_per_subject_roi:
        elec_string_to_add_to_filename = 'all_elecs'
    elif electrodes == sig_electrodes_per_subject_roi:
        elec_string_to_add_to_filename = 'sig_elecs'
    else:
        elec_string_to_add_to_filename = None

    roi_labeled_arrays = put_data_in_labeled_array_per_roi_subject(
        subjects_tfr_objects,
        condition_names,
        rois,
        args.subjects,
        electrodes, 
        obs_axs=args.obs_axs,  # Trials dimension (ignoring the conditions dimension for now cuz we iterate over it)
        chans_axs=args.chans_axs,  # Channels dimension
        freq_axs=args.freq_axs,
        time_axs=args.time_axs,   # Time dimension
        random_state=args.random_state  # For reproducibility
    )

    del subjects_tfr_objects # clear this from memory now that we're done with it
    gc.collect()
    
    condition_comparisons = {}

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
        condition_comparisons['congruency'] = [['c25', 'c75'], ['i25', 'i75']]
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        condition_comparisons['switchType'] = [['r25', 'r75'], ['s25', 's75']]

    for condition_comparisons, strings_to_find in condition_comparisons.items():
        confusion_matrices, cats, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
            roi_labeled_arrays, rois, strings_to_find, args.stat_func, 
            Decoder, explained_variance=args.explained_variance,
            p_thresh=args.p_thresh, n_perm=args.n_perm, 
            n_splits=args.n_splits, n_repeats=args.n_repeats, obs_axs=args.obs_axs, chans_axs=args.chans_axs,
            balance_method=args.balance_method, oversample=args.oversample,
            random_state=args.random_state, alpha=args.alpha, ignore_adjacency=args.ignore_adjacency, seed=args.seed, tails=args.tails,
            clear_memory=args.clear_memory, normalize=args.normalize
        )

    for roi in rois:
        file_name = f'{roi}_{conditions_save_name}_time_averaged_confusion_matrix_{args.n_splits}splits_{args.n_repeats}repeats_{args.balance_method}.png'
        plot_and_save_confusion_matrix(confusion_matrices[roi], cats[roi], file_name, save_dir)
    
    # plotting

    save_dir = os.path.join(LAB_root, 'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'decoding', 'james_sun_cluster_decoding', f"{conditions_save_name}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")

    for roi in rois:
        # This assumes 'channel_masks' has the structure {roi: {repeat: {split: {ch_idx: mask}}}}
        if roi not in channel_masks:
            continue

        for repeat in range(args.n_repeats):
            if repeat not in channel_masks[roi]:
                continue
            for split in range(args.n_splits):
                if split not in channel_masks[roi][repeat]:
                    continue
                
                # This is the dictionary of 2D masks for the current split
                mask_dict = channel_masks[roi][repeat][split]

                if not mask_dict:
                    print(f"Skipping plot for {roi} Repeat {repeat+1} Split {split+1}: No masks found.")
                    continue

                # --- FIX #1 & #2: Get the correct channel names and stack the masks ---
                # Get the channel indices from the dictionary keys and sort them for consistent order
                roi_ch_indices = sorted(mask_dict.keys())
                
                # Use the indices to get the corresponding channel names from the LabeledArray for this ROI
                roi_ch_names = [roi_labeled_arrays[roi].labels[args.chans_axs][i] for i in roi_ch_indices]
                
                # Stack the 2D mask arrays into a single 3D NumPy array
                stacked_masks = np.array([mask_dict[ch_idx] for ch_idx in roi_ch_indices])

                # --- FIX #3: Correct the title prefix ---
                title = f"{roi} Repeat {repeat+1} Split {split+1}: "

                # Now, call the plotting function with the corrected data
                roi_repeat_split_pages = plot_mask_pages(
                    stacked_masks,
                    roi_ch_names, # Use the ROI-specific names
                    times=times,
                    freqs=freqs,
                    channels_per_page=60,
                    grid_shape=(6, 10),
                    cmap=parula_map,
                    title_prefix=title, # Use the corrected title
                    log_freq=True,
                    show=False)

                # Save each page as a separate figure file
                for i, fig in enumerate(roi_repeat_split_pages):
                    # Use the corrected filename that doesn't rely on `sub`
                    fig_name = f"{roi}_sig_clusters_repeat{repeat+1}_split{split+1}_{conditions_save_name}_page_{i+1}.png"
                    fig_pathname = os.path.join(save_dir, fig_name)
                    fig.savefig(fig_pathname, bbox_inches='tight')
                    print("Saved figure:", fig_pathname)
                    plt.close(fig) # Close the figure to free up memory
    
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
        print("This script should be called via run_james_sun_cluster_decoding.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)
