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
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

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
from utils import *
from utils import make_or_load_subjects_electrodes_to_ROIs_dict
import matplotlib.pyplot as plt

from pandas import read_csv
import scipy.stats as stats
import joblib

from scipy.ndimage import label
from scipy.stats import norm

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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from jim_decoding_functions import *

from src.analysis.config import experiment_conditions
from src.analysis.utils.labeled_array_utils import put_data_in_labeled_array_per_roi_subject, remove_nans_from_labeled_array, remove_nans_from_all_roi_labeled_arrays, concatenate_conditions_by_string

def process_and_balance_data_for_decoding(
    roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
):
    """
    Processes and balances the data for a given ROI.

    Parameters:
    - roi_labeled_arrays: Dictionary containing reshaped data for each ROI.
    - roi: The ROI to process.
    - strings_to_find: List of strings or string groups to identify condition labels.
    - obs_axs: The trials axis.
    - balance_method: 'pad_with_nans' or 'subsample' to balance trial counts between conditions.
    - random_state: Random seed for reproducibility.

    Returns:
    - concatenated_data: The processed and balanced numpy array for decoding. This gets the data out of the roi labeled arrays format.
    - labels: The processed labels array.
    - cats: Dictionary of condition categories.
    """
    rng = np.random.RandomState(random_state)

    # Concatenate the trials and get labels
    concatenated_data, labels, cats = concatenate_conditions_by_string(
        roi_labeled_arrays, roi, strings_to_find, obs_axs
    )

    print(f"Concatenated data shape for {roi}: {concatenated_data.shape}")

    if balance_method == 'subsample':
        # Remove NaN trials from concatenated_data and labels
        nan_trials = np.isnan(concatenated_data).any(axis=tuple(range(1, concatenated_data.ndim)))
        valid_trials = ~nan_trials

        # Keep only valid trials
        concatenated_data = concatenated_data[valid_trials]
        labels = labels[valid_trials]

    # Calculate trial counts per condition
    trial_counts = {}
    condition_indices = {}

    for string_group in strings_to_find:
        condition_label = cats[tuple(string_group) if isinstance(string_group, list) else (string_group,)]
        condition_trials = labels == condition_label
        data_for_condition = concatenated_data[condition_trials]

        # Store indices and counts
        condition_indices[condition_label] = np.where(condition_trials)[0]
        trial_counts[condition_label] = data_for_condition.shape[0]

        print(f'Condition {string_group} has {trial_counts[condition_label]} trials')

    if balance_method == 'pad_with_nans':
        max_trial_count = max(trial_counts.values())
        for condition_label, count in trial_counts.items():
            trials_to_add = max_trial_count - count
            if trials_to_add > 0:
                nan_trial_shape = (trials_to_add,) + concatenated_data.shape[1:]
                nan_trials = np.full(nan_trial_shape, np.nan)
                concatenated_data = np.concatenate([concatenated_data, nan_trials], axis=obs_axs)
                labels = np.concatenate([labels, [condition_label] * trials_to_add])
    elif balance_method == 'subsample':
        min_trial_count = min(trial_counts.values())
        subsampled_indices = []
        for condition_label in trial_counts.keys():
            indices = condition_indices[condition_label]
            if trial_counts[condition_label] > min_trial_count:
                selected_indices = rng.choice(indices, size=min_trial_count, replace=False)
            else:
                selected_indices = indices
            subsampled_indices.extend(selected_indices)

        subsampled_indices = np.array(subsampled_indices)
        concatenated_data = concatenated_data[subsampled_indices]
        labels = labels[subsampled_indices]
    else:
        raise ValueError(f"Invalid balance_method: {balance_method}. Choose 'pad_with_nans' or 'subsample'.")

    return concatenated_data, labels, cats