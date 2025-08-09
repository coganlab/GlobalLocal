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
    concatenate_and_balance_data_for_decoding, 
    get_and_plot_confusion_matrix_for_rois_jim,
    Decoder, 
    windower,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    perform_time_perm_cluster_test_for_accuracies,
    plot_accuracies
)

from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets, get_uncorrected_multitaper, get_sig_tfr_differences, plot_mask_pages
from src.analysis.spec.subjects_tfr_objects_functions import make_subjects_tfr_objects, get_sig_tfr_differences_per_subject, get_sig_tfr_differences_per_roi

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
import argparse


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

    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = make_or_load_subjects_electrodes_to_ROIs_dict(args.subjects, task=args.task, LAB_root=LAB_root, save_dir=config_dir, 
                                                    filename='subjects_electrodestoROIs_dict.json')

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

    # For per-subject analysis (no electrode filtering needed)
    sig_elec_masks_per_subject, sig_elec_pvals_per_subject = get_sig_tfr_differences_per_subject(subjects_tfr_objects=subjects_tfr_objects, condition_names=condition_names, stat_func=args.stat_func, p_thresh=args.p_thresh, n_perm=args.n_perm, ignore_adjacency=args.ignore_adjacency, n_jobs=args.n_jobs, seed=args.seed, tails=args.tails)

    all_elec_masks_per_subject, all_elec_pvals_per_subject = get_sig_tfr_differences_per_subject(subjects_tfr_objects=subjects_tfr_objects, condition_names=condition_names, stat_func=args.stat_func, p_thresh=args.p_thresh, n_perm=args.n_perm, ignore_adjacency=args.ignore_adjacency, n_jobs=args.n_jobs, seed=args.seed, tails=args.tails)

    # For per-ROI analysis (with electrode filtering)
    sig_elec_masks_per_roi, sig_elec_pvals_per_roi = get_sig_tfr_differences_per_roi(subjects_tfr_objects=subjects_tfr_objects, electrodes_per_subject_roi=sig_electrodes_per_subject_roi, condition_names=condition_names, stat_func=args.stat_func, p_thresh=args.p_thresh, n_perm=args.n_perm, ignore_adjacency=args.ignore_adjacency, n_jobs=args.n_jobs, seed=args.seed, tails=args.tails)

    all_elec_masks_per_roi, all_elec_pvals_per_roi = get_sig_tfr_differences_per_roi(subjects_tfr_objects=subjects_tfr_objects, electrodes_per_subject_roi=all_electrodes_per_subject_roi, condition_names=condition_names, stat_func=args.stat_func, p_thresh=args.p_thresh, n_perm=args.n_perm, ignore_adjacency=args.ignore_adjacency, n_jobs=args.n_jobs, seed=args.seed, tails=args.tails)

    first_sub = args.subjects[0]
    first_condition = list(subjects_tfr_objects[first_sub].keys())[0]
    ch_names = subjects_tfr_objects[first_sub][first_condition].ch_names
    times = subjects_tfr_objects[first_sub][first_condition].times
    freqs = subjects_tfr_objects[first_sub][first_condition].freqs

    subjects_tfr_objects_save_dir = os.path.join(layout.root, 'derivatives', 'spec', args.spec_method, 'subjects_tfr_objects')
    if not os.path.exists(subjects_tfr_objects_save_dir):
        os.makedirs(subjects_tfr_objects_save_dir)

    # Now plot the mask pages:
    for sub in args.subjects:
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
            fig_name = f"{sub}_sig_elecs_sig_{args.spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
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
            fig_name = f"{sub}_all_elecs_sig_{args.spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
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
            fig_name = f"{roi}_sig_elecs_sig_{args.spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
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
            fig_name = f"{sub}_{roi}_all_elecs_sig_{args.spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(subjects_tfr_objects_save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make subject and ROI level tfr difference masks.")
    parser.add_argument('--LAB_root', type=str, required=True, 
                        help="The cogan lab root directory")
    parser.add_argument('--subjects', type=list, required=True, 
                        help="List of subject ID to process")
    parser.add_argument('--signal_times', type=list, required=True, default=[-1.0, 1.5], 
                        help='The signal times to use')
    parser.add_argument('--acc_trials_only', type=bool, required=True, default=True, 
                        help='Whether to only use accuracy trials')
    parser.add_argument('--error_trials_only', type=bool, required=True, default=False, 
                        help='Whether to only use error trials')
    parser.add_argument('--stat_func', type=func, required=True, 
                        help='The statistical function to use')
    parser.add_argument('--p_thresh', type=float, required=True, default=0.05,
                        help='the p threshold for your statistical test')
    parser.add_argument('--ignore_adjacency', type=int, required=True, default=1,
                        help='dimension to ignore when finding significant clusters. By default, ignore the channels dimension for clusters, just find clusters over frequency and time')
    parser.add_argument('--n_perm', type=int, required=True, default=100,
                        help='number of permutations for the statistical test')
    parser.add_argument('--n_jobs', type=int, required=True, default=1,
                        help='number of jobs to use for the statistical test')
    parser.add_argument('--freqs', type=numpy.ndarray, required=True, default=np.arange(2, 200., 2.),
                        help='frequency range to use for the statistical test')
    parser.add_argument('--n_cycles', type=numpy.ndarray, required=True, default=freqs / 2,
                        help='number of cycles to use for the statistical test')
    parser.add_argument('--return_itc', type=bool, required=True, default=False,
                        help='whether to return the itc')
    parser.add_argument('--time_bandwidth', type=int, required=True, default=10,
                        help='time bandwidth for the statistical test')
    parser.add_argument('--spec_method', type=str, required=True, default='multitaper',
                        help='spectral method to use for the statistical test')
    parser.add_argument('--average', type=bool, required=True, default=False,
                        help='whether to trial average the tfrs')
    parser.add_argument('--seed', type=int, required=True, default=None,
                        help='seed for the statistical test')
    parser.add_argument('--tails', type=int, required=True, default=2,
                        help='tails for the statistical test')
    parser.add_argument('--n_splits', type=int, required=True, default=2,
                        help='number of splits for decoding (not used for finding sig clusters but leave it in for now because we will use it later when adding in the decoding code)')
    parser.add_argument('--n_repeats', type=int, required=True, default=1,
                        help='number of repeats for decoding (not used for finding sig clusters but leave it in for now because we will use it later when adding in the decoding code)')
    parser.add_argument('--random_state', type=int, required=True, default=42,
                        help='random state for the statistical test')
    parser.add_argument('--task', type=str, required=True, default='GlobalLocal',
                        help='experiment name, should be GlobalLocal')
    parser.add_argument('--conditions', type=dict, required=True,
                        help='conditions to be compared')
    parser.add_argument('--epochs_root_file', type=str, required=True,
                        help='epochs root file name, this is used to find significant electrodes and is stored in the output name. TBH because this finds significance based on bandpass-filtered data, it is a little circular, so i should just use all electrodes in an roi, instead of the significant ones. But just leave it for now for testing purposes. Remove later though. No need for sig electrodes in this analysis.')
    parser.add_argument('--rois_dict', type=dict, required=True,
                        help='roi dictionary mapping destrieux atlas rois to big rois')

    args = parser.parse_args()
    main(args)