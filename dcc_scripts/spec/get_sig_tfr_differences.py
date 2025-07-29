# this is the first part of james_sun_cluster_decoding, just getting and plotting the sig tfr clusters. In the submit script, can choose to run this for each subject individually or for each roi individually. In that case, wrap the single subject or single roi in a list or dict, respectively.

import sys
import os

print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

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

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it
    
from functools import partial
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
from ieeg.viz.parula import parula_map

# TODO: hmm fix these utils imports, import the funcs individually. 6/1/25.

import matplotlib.pyplot as plt

from pandas import read_csv
import scipy.stats as stats
import joblib

from scipy.ndimage import label
from scipy.stats import norm
from scipy.stats import ttest_ind

import json
import pickle

# rsa toolbox imports
from rsatoolbox.io.mne import read_epochs
from rsatoolbox.data.ops import merge_datasets
from rsatoolbox.rdm import calc_rdm_movie
from rsatoolbox.rdm.calc import _parse_input
from rsatoolbox.util.build_rdm import _build_rdms
from rsatoolbox.rdm import compare
from rsatoolbox.vis import show_rdm
from rsatoolbox.vis.timecourse import plot_timecourse

from os.path import join, expanduser, basename
import glob, json
import numpy, tqdm, mne, pandas
import rsatoolbox
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import copy
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
    process_and_balance_data_for_decoding, 
    get_and_plot_confusion_matrix_for_rois_jim,
    Decoder, 
    windower,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    perform_time_perm_cluster_test_for_accuracies,
    plot_accuracies
)

from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets, get_uncorrected_multitaper, get_sig_tfr_differences, plot_mask_pages
from src.analysis.spec.subjects_tfr_objects_functions import load_or_make_subjects_tfr_objects, get_sig_tfr_differences_per_subject, get_sig_tfr_differences_per_roi

from src.analysis.utils.general_utils import (
    make_or_load_subjects_electrodes_to_ROIs_dict,
    get_good_data,
    get_sig_chans_per_subject,
    make_sig_electrodes_per_subject_and_roi_dict,
    calculate_total_electrodes,
    check_sampling_rates
)

import mne.time_frequency
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Union, List, Sequence, Dict
import doctest

# turn these all into input args from the submit script
subjects = ['D0103']
signal_times = [-1.0, 1.5]
acc_trials_only = False
error_trials_only = False
stat_func = partial(ttest_ind, equal_var=False, nan_policy='omit')
p_thresh = 0.2
ignore_adjacency = 1 # ignore the channels dimension for clusters, just find clusters over frequency and time
n_perm = 10
n_jobs = 1
freqs = np.arange(2, 200., 4.)
n_cycles = freqs / 2
return_itc = False
time_bandwidth=10 
spec_method = 'multitaper'
average=False
seed=None
tails=2
n_splits=2
n_repeats=1
random_state=42
task='GlobalLocal'
conditions = experiment_conditions.stimulus_big_letter_conditions # set this to whichever conditions you're running

stimulus_locked = True  #toggle
response_locked = not stimulus_locked

# set this in the submit script, pass in epochs root file as an input to get_sig_tfr_differences.py
if stimulus_locked:
    # epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    # epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_0.0-30.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"

elif response_locked:
    # epochs_root_file = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind"
    epochs_root_file = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"

# load in subjects electrodes to rois dict. If it doesn't already exist, make it and then load it.
config_dir = os.path.join(project_root, 'src', 'analysis', 'config')

# this needs to be passed in from the submit script somehow
rois_dict = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
}

def main():
    subjects_electrodestoROIs_dict = make_or_load_subjects_electrodes_to_ROIs_dict(subjects, task='GlobalLocal', LAB_root=None, save_dir=config_dir, 
                                                    filename='subjects_electrodestoROIs_dict.json')

    HOME = os.path.expanduser("~")
    USER = os.path.basename(HOME)

    # get box directory depending on OS
    if os.name == 'nt': # windows
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    else: # mac
        LAB_root = os.path.join("/cwork", USER)

    layout = get_data('GlobalLocal', root=LAB_root)


    condition_names = list(conditions.keys()) # get the condition names as a list

    if conditions == experiment_conditions.stimulus_conditions:
        conditions_save_name = 'stimulus_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_experiment_conditions:
        conditions_save_name = 'stimulus_experiment_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_main_effect_conditions:
        conditions_save_name = 'stimulus_main_effect_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_lwpc_conditions:
        conditions_save_name = 'stimulus_lwpc_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_lwps_conditions:
        conditions_save_name = 'stimulus_lwps_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_big_letter_conditions:
        conditions_save_name = 'stimulus_big_letter_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_small_letter_conditions:
        conditions_save_name = 'stimulus_small_letter_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_task_conditions:
        conditions_save_name = 'stimulus_task_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_congruency_conditions:
        conditions_save_name = 'stimulus_congruency_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.stimulus_switch_type_conditions:
        conditions_save_name = 'stimulus_switch_type_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'

    elif conditions == experiment_conditions.response_conditions:
        conditions_save_name = 'response_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_experiment_conditions:
        conditions_save_name = 'response_experiment_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_big_letter_conditions:
        conditions_save_name = 'response_big_letter_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_small_letter_conditions:
        conditions_save_name = 'response_small_letter_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_task_conditions:
        conditions_save_name = 'response_task_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_congruency_conditions:
        conditions_save_name = 'response_congruency_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'
    elif conditions == experiment_conditions.response_switch_type_conditions:
        conditions_save_name = 'response_switch_type_conditions' + '_' + epochs_root_file + '_' + str(len(subjects)) + '_' + 'subjects'

    sig_chans_per_subject = get_sig_chans_per_subject(subjects, epochs_root_file, task='GlobalLocal', LAB_root=None)

    rois = list(rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)

    subjects_tfr_objects = load_or_make_subjects_tfr_objects(
    layout=layout,
    spec_method=spec_method,
    conditions_save_name=conditions_save_name,
    subjects=subjects,
    conditions=conditions,
    signal_times=signal_times,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    return_itc=return_itc,
    n_jobs=n_jobs,
    average=average,
    acc_trials_only=acc_trials_only,
    error_trials_only=error_trials_only 
    )

    # For per-subject analysis (no electrode filtering needed)
    sig_elec_masks_per_subject, sig_elec_pvals_per_subject = get_sig_tfr_differences_per_subject(subjects_tfr_objects=subjects_tfr_objects, condition_names=condition_names, stat_func=stat_func, p_thresh=p_thresh, n_perm=n_perm, ignore_adjacency=ignore_adjacency, n_jobs=n_jobs, seed=seed, tails=tails)

    all_elec_masks_per_subject, all_elec_pvals_per_subject = get_sig_tfr_differences_per_subject(subjects_tfr_objects=subjects_tfr_objects, condition_names=condition_names, stat_func=stat_func, p_thresh=p_thresh, n_perm=n_perm, ignore_adjacency=ignore_adjacency, n_jobs=n_jobs, seed=seed, tails=tails)

    # For per-ROI analysis (with electrode filtering)
    sig_elec_masks_per_roi, sig_elec_pvals_per_roi = get_sig_tfr_differences_per_roi(subjects_tfr_objects=subjects_tfr_objects, electrodes_per_subject_roi=sig_electrodes_per_subject_roi, condition_names=condition_names, stat_func=stat_func, p_thresh=p_thresh, n_perm=n_perm, ignore_adjacency=ignore_adjacency, n_jobs=n_jobs, seed=seed, tails=tails)

    all_elec_masks_per_roi, all_elec_pvals_per_roi = get_sig_tfr_differences_per_roi(subjects_tfr_objects=subjects_tfr_objects, electrodes_per_subject_roi=all_electrodes_per_subject_roi, condition_names=condition_names, stat_func=stat_func, p_thresh=p_thresh, n_perm=n_perm, ignore_adjacency=ignore_adjacency, n_jobs=n_jobs, seed=seed, tails=tails)

    first_sub = subjects[0]
    first_condition = list(subjects_tfr_objects[first_sub].keys())[0]
    ch_names = subjects_tfr_objects[first_sub][first_condition].ch_names
    times = subjects_tfr_objects[first_sub][first_condition].times
    freqs = subjects_tfr_objects[first_sub][first_condition].freqs

    subjects_tfr_objects_save_dir = os.path.join(layout.root, 'derivatives', 'spec', spec_method, 'subjects_tfr_objects')
    if not os.path.exists(subjects_tfr_objects_save_dir):
        os.makedirs(subjects_tfr_objects_save_dir)

    # Now plot the mask pages:
    for sub in subjects:
        sig_elecs_mask = sig_elec_masks_per_subject[sub]
        sig_elecs_mask_pages = plot_mask_pages(sig_elecs_mask,
                        ch_names,
                        times=times,
                        freqs=freqs,
                        channels_per_page=60,
                        grid_shape=(6, 10),
                        cmap=parula_map,
                        title_prefix=f"{sub} ",
                        log_freq=True,
                        show=False)

        # Save each page as a separate figure file:
        for i, fig in enumerate(sig_elecs_mask_pages):
            fig_name = f"{sub}_sig_elecs_sig_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(subjects_tfr_objects_save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

        all_elecs_mask = all_elec_masks_per_subject[sub]
        all_elecs_mask_pages = plot_mask_pages(all_elecs_mask,
                            ch_names,
                            times=times,
                            freqs=freqs,
                            channels_per_page=60,
                            grid_shape=(6, 10),
                            cmap=parula_map,
                            title_prefix=f"{sub} ",
                            log_freq=True,
                            show=False)

        # Save each page as a separate figure file:
        for i, fig in enumerate(all_elecs_mask_pages):
            fig_name = f"{sub}_all_elecs_sig_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(subjects_tfr_objects_save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

    for roi in rois:
        sig_elecs_roi_mask = sig_elec_masks_per_roi[roi]
        sig_elecs_roi_mask_pages = plot_mask_pages(sig_elecs_roi_mask,
                        ch_names,
                        times=times,
                        freqs=freqs,
                        channels_per_page=60,
                        grid_shape=(6, 10),
                        cmap=parula_map,
                        title_prefix=f"{roi} ",
                        log_freq=True,
                        show=False)

        # Save each page as a separate figure file:
        for i, fig in enumerate(sig_elecs_roi_mask_pages):
            fig_name = f"{roi}_sig_elecs_sig_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(subjects_tfr_objects_save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

        all_elecs_roi_mask = all_elec_masks_per_roi[roi]
        all_elecs_roi_mask_pages = plot_mask_pages(all_elecs_roi_mask,
                        ch_names,
                        times=times,
                        freqs=freqs,
                        channels_per_page=60,
                        grid_shape=(6, 10),
                        cmap=parula_map,
                        title_prefix=f"{sub} {roi} ",
                        log_freq=True,
                        show=False)

        # Save each page as a separate figure file:
        for i, fig in enumerate(all_elecs_roi_mask_pages):
            fig_name = f"{sub}_{roi}_all_elecs_sig_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(subjects_tfr_objects_save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make subject and ROI level tfr difference masks.")
    parser.add_argument('--subject', type=str, required=True, 
                        help="The subject ID to process")
    parser.add_argument('--type', type=str, required=True, 
                        help='The type of analysis to run - wavelet or multitaper')
    args = parser.parse_args()
    main(args.type)

# turn these all into input args from the submit script
subjects = ['D0103']
signal_times = [-1.0, 1.5]
acc_trials_only = False
error_trials_only = False
stat_func = partial(ttest_ind, equal_var=False, nan_policy='omit')
p_thresh = 0.2
ignore_adjacency = 1 # ignore the channels dimension for clusters, just find clusters over frequency and time
n_perm = 10
n_jobs = 1
freqs = np.arange(2, 200., 4.)
n_cycles = freqs / 2
return_itc = False
time_bandwidth=10 
spec_method = 'multitaper'
average=False
seed=None
tails=2
n_splits=2
n_repeats=1
random_state=42
task='GlobalLocal'
conditions = experiment_conditions.stimulus_big_letter_conditions # set this to whichever conditions you're running

stimulus_locked = True  #toggle
response_locked = not stimulus_locked

# set this in the submit script, pass in epochs root file as an input to get_sig_tfr_differences.py
if stimulus_locked:
    # epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    # epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_0.0-30.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"

elif response_locked:
    # epochs_root_file = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind"
    epochs_root_file = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"

# load in subjects electrodes to rois dict. If it doesn't already exist, make it and then load it.
config_dir = os.path.join(project_root, 'src', 'analysis', 'config')

# this needs to be passed in from the submit script somehow
rois_dict = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
}
