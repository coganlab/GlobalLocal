import sys
import os
import glob
import json
import pickle
import itertools # Added, was implicitly imported by tqdm? Or just good to have explicitly.

# ---- sys.path modifications ----
# This section should generally be at the top as it affects module resolution
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") # User-specific path

try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- Third-party library imports ----
import mne
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.ndimage import label # was imported separately, now grouped with scipy
from scipy.stats import norm # also from scipy.stats
import joblib
from joblib import Parallel, delayed # Add this line in your decoding.py
import matplotlib.pyplot as plt
from matplotlib import pyplot # Note: matplotlib.pyplot is typically imported as plt. This is redundant if plt is already imported.

# scikit-learn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Other third-party
from tqdm import tqdm 
from numpy.lib.stride_tricks import as_strided, sliding_window_view # sliding_window_view was imported separately
from typing import Union, List, Sequence
# ---- Local/Project-specific imports ----
# ieeg imports
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.mat import LabeledArray, combine
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from ieeg.viz.parula import parula_map

# src imports
from src.analysis.config import experiment_conditions
from src.analysis.utils.labeled_array_utils import (
    put_data_in_labeled_array_per_roi_subject,
    remove_nans_from_labeled_array,
    remove_nans_from_all_roi_labeled_arrays,
    concatenate_conditions_by_string,
    get_data_in_time_range
)

# utils imports
# TODO: hmm fix these utils imports, import the funcs individually. 6/1/25.
from src.analysis.utils.general_utils import * # This is generally discouraged.
from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict # Explicit import is good
import gc
# ---- Potentially Unused or Redundant ----
# from pandas import read_csv # pandas is already imported as pd, use pd.read_csv
# from os.path import join, expanduser, basename # os.path is available via 'import os'
# from joblib import Parallel, delayed # joblib is already imported

def concatenate_and_balance_data_for_decoding(
    roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
):
    """
    Processes and balances the data for a given ROI with improved debugging.
    Now correctly distinguishes between outlier-induced NaNs and missing channel NaNs.
    
    Parameters:
    - roi_labeled_arrays: Dictionary containing reshaped data for each ROI.
    - roi: The ROI to process.
    - strings_to_find: List of strings or string groups to identify condition labels.
    - obs_axs: The trials axis.
    - balance_method: 'pad_with_nans' or 'subsample' to balance trial counts between conditions.
    - random_state: Random seed for reproducibility.

    Returns:
    - concatenated_data: The processed and balanced numpy array for decoding. This gets the data out of the roi labeled arrays format and into a numpy array that is trials x channels x (freqs?) x timepoints.
    - labels: The processed labels array.
    - cats: Dictionary of condition categories.
    """
    rng = np.random.RandomState(random_state)

    # Concatenate the trials and get labels
    concatenated_data, labels, cats = concatenate_conditions_by_string(
        roi_labeled_arrays, roi, strings_to_find, obs_axs
    )

    print(f"\n{'='*60}")
    print(f"ROI: {roi}")
    print(f"{'='*60}")
    print(f"Initial concatenated data shape: {concatenated_data.shape}")
    print(f"  Trials: {concatenated_data.shape[0]}")
    print(f"  Channels: {concatenated_data.shape[1]}")
    if concatenated_data.ndim > 2:
        print(f"  Time points: {concatenated_data.shape[-1]}")
        if concatenated_data.ndim == 4:  # Has frequency dimension
            print(f"  Frequencies: {concatenated_data.shape[2]}")
    
    if balance_method == 'subsample':
        print(f"\n--- Detailed NaN Analysis ---")
        
        n_trials = concatenated_data.shape[0]
        n_channels = concatenated_data.shape[1] if concatenated_data.ndim >= 2 else 1
        
        # Reshape to (trials, channels, features) for easier analysis
        # where features = time*freq or just time
        reshaped_data = concatenated_data.reshape(n_trials, n_channels, -1)
        n_features = reshaped_data.shape[2]
        
        # 1. Identify channels that are completely NaN across all trials (missing channels)
        missing_channels = []
        for ch_idx in range(n_channels):
            ch_data = reshaped_data[:, ch_idx, :]
            if np.all(np.isnan(ch_data)):
                missing_channels.append(ch_idx)
        
        print(f"\nMissing Channels (all NaN across all trials): {len(missing_channels)}/{n_channels}")
        
        # 2. Analyze NaN patterns per trial
        trials_with_any_nan = 0
        trials_with_outlier_nans = 0
        trials_with_missing_channel_nans = 0
        trials_with_both = 0
        
        outlier_nan_counts = []
        missing_channel_counts_per_trial = []
        
        for trial_idx in range(n_trials):
            trial_data = reshaped_data[trial_idx]  # Shape: (channels, features)
            
            has_any_nan = False
            has_outlier_nan = False
            has_missing_channel = False
            outlier_nans_in_trial = 0
            missing_channels_in_trial = 0
            
            for ch_idx in range(n_channels):
                ch_data = trial_data[ch_idx]
                
                if np.any(np.isnan(ch_data)):
                    has_any_nan = True
                    
                    if np.all(np.isnan(ch_data)):
                        # Entire channel is NaN for this trial
                        has_missing_channel = True
                        missing_channels_in_trial += 1
                    else:
                        # Partial NaNs - likely outliers
                        has_outlier_nan = True
                        outlier_nans_in_trial += np.sum(np.isnan(ch_data))
            
            if has_any_nan:
                trials_with_any_nan += 1
            if has_outlier_nan:
                trials_with_outlier_nans += 1
                outlier_nan_counts.append(outlier_nans_in_trial)
            if has_missing_channel:
                trials_with_missing_channel_nans += 1
                missing_channel_counts_per_trial.append(missing_channels_in_trial)
            if has_outlier_nan and has_missing_channel:
                trials_with_both += 1
        
        print(f"\nTrial-level NaN Statistics:")
        print(f"  Trials with ANY NaN: {trials_with_any_nan}/{n_trials} ({100*trials_with_any_nan/n_trials:.1f}%)")
        print(f"  Trials with outlier NaNs (sparse): {trials_with_outlier_nans}/{n_trials} ({100*trials_with_outlier_nans/n_trials:.1f}%)")
        print(f"  Trials with missing channel NaNs (dense): {trials_with_missing_channel_nans}/{n_trials} ({100*trials_with_missing_channel_nans/n_trials:.1f}%)")
        print(f"  Trials with both types: {trials_with_both}/{n_trials} ({100*trials_with_both/n_trials:.1f}%)")
        
        if outlier_nan_counts:
            print(f"\nOutlier NaN Statistics (for affected trials):")
            print(f"  Mean outlier NaNs per affected trial: {np.mean(outlier_nan_counts):.1f}")
            print(f"  Max outlier NaNs in a trial: {np.max(outlier_nan_counts)}")
            print(f"  Total outlier NaNs across all trials: {np.sum(outlier_nan_counts)}")
        
        if missing_channel_counts_per_trial:
            print(f"\nMissing Channel Statistics (for affected trials):")
            print(f"  Mean missing channels per affected trial: {np.mean(missing_channel_counts_per_trial):.1f}")
            print(f"  Max missing channels in a trial: {np.max(missing_channel_counts_per_trial)}")
        
        # 3. Calculate percentage of data that is NaN
        total_nan_count = np.sum(np.isnan(concatenated_data))
        total_elements = concatenated_data.size
        print(f"\nOverall NaN Percentage: {100*total_nan_count/total_elements:.2f}%")
        
        # Remove trials with any NaNs
        nan_trials_mask = np.isnan(concatenated_data).any(axis=tuple(range(1, concatenated_data.ndim)))
        valid_trials = ~nan_trials_mask
        
        # Before removing, check which conditions are most affected
        print(f"\n--- NaN Impact by Condition ---")
        for string_group in strings_to_find:
            condition_label = cats[tuple(string_group) if isinstance(string_group, list) else (string_group,)]
            condition_mask = labels == condition_label
            condition_nan_mask = nan_trials_mask & condition_mask
            n_condition_trials = np.sum(condition_mask)
            n_nan_trials = np.sum(condition_nan_mask)
            print(f"  {string_group}: {n_nan_trials}/{n_condition_trials} trials with NaNs ({100*n_nan_trials/n_condition_trials:.1f}%)")
        
        concatenated_data = concatenated_data[valid_trials]
        labels = labels[valid_trials]
        
        print(f"\nAfter removing NaN trials:")
        print(f"  Trials kept: {np.sum(valid_trials)}/{len(valid_trials)} ({100*np.sum(valid_trials)/len(valid_trials):.1f}%)")
        print(f"  New data shape: {concatenated_data.shape}")

    # Calculate trial counts per condition
    trial_counts = {}
    condition_indices = {}

    print(f"\n--- Final Trial Counts by Condition ---")
    for string_group in strings_to_find:
        condition_label = cats[tuple(string_group) if isinstance(string_group, list) else (string_group,)]
        condition_trials = labels == condition_label
        data_for_condition = concatenated_data[condition_trials]

        # Store indices and counts
        condition_indices[condition_label] = np.where(condition_trials)[0]
        trial_counts[condition_label] = data_for_condition.shape[0]

        print(f'  Condition {string_group}: {trial_counts[condition_label]} trials')

    # Check if we have enough trials
    min_trials = min(trial_counts.values()) if trial_counts else 0
    if min_trials < 10:
        print(f"\n⚠️ WARNING: Very few trials remaining! Min trials per condition: {min_trials}")
        print("  This will likely result in poor decoding accuracy.")

    if balance_method == 'pad_with_nans':
        max_trial_count = max(trial_counts.values())
        print(f"\nPadding to max trial count: {max_trial_count}")
        for condition_label, count in trial_counts.items():
            trials_to_add = max_trial_count - count
            if trials_to_add > 0:
                print(f"  Adding {trials_to_add} NaN trials to condition {condition_label}")
                nan_trial_shape = (trials_to_add,) + concatenated_data.shape[1:]
                nan_trials = np.full(nan_trial_shape, np.nan)
                concatenated_data = np.concatenate([concatenated_data, nan_trials], axis=obs_axs)
                labels = np.concatenate([labels, [condition_label] * trials_to_add])

    elif balance_method == 'subsample':
        min_trial_count = min(trial_counts.values())
        print(f"\nSubsampling to min trial count: {min_trial_count}")
        subsampled_indices = []
        for condition_label in trial_counts.keys():
            indices = condition_indices[condition_label]
            if trial_counts[condition_label] > min_trial_count:
                selected_indices = rng.choice(indices, size=min_trial_count, replace=False)
                print(f"  Subsampled condition {condition_label}: {len(indices)} -> {min_trial_count}")
            else:
                selected_indices = indices
            subsampled_indices.extend(selected_indices)

        subsampled_indices = np.array(subsampled_indices, dtype=int)
        concatenated_data = concatenated_data[subsampled_indices]
        labels = labels[subsampled_indices]
        
        print(f"\nFinal balanced data shape: {concatenated_data.shape}")

    print(f"{'='*60}\n")
    return concatenated_data, labels, cats

# largely stolen from aaron's ieeg plot_decoding.py

def mixup2(arr: np.ndarray, labels: np.ndarray, obs_axs: int, alpha: float = 1.,
          seed: int = None) -> None:
    """Mixup the data using the labels

    Parameters
    ----------
    arr : array
        The data to mixup.
    labels : array
        The labels to use for mixing.
    obs_axs : int
        The axis along which to apply func.
    alpha : float
        The alpha value for the beta distribution.
    seed : int
        The seed for the random number generator.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> mixup2(arr, labels, 0)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [6.03943491, 7.03943491]])
           """
    if arr.ndim > 2:
        arr = arr.swapaxes(obs_axs, -2)
        for i in range(arr.shape[0]):
            mixup2(arr=arr[i], labels=labels, obs_axs=obs_axs, alpha=alpha, seed=seed)
    else:
        if seed is not None:
            np.random.seed(seed)
        if obs_axs == 1:
            arr = arr.T

        n_nan = np.where(np.isnan(arr).any(axis=1))[0]
        n_non_nan = np.where(~np.isnan(arr).any(axis=1))[0]

        for i in n_nan:
            l_class = labels[i]
            possible_choices = np.nonzero(np.logical_and(~np.isnan(arr).any(axis=1), labels == l_class))[0]
            choice1 = np.random.choice(possible_choices)
            choice2 = np.random.choice(n_non_nan)
            l = np.random.beta(alpha, alpha)
            if l < .5:
                l = 1 - l
            arr[i] = l * arr[choice1] + (1 - l) * arr[choice2]

class Decoder(PcaLdaClassification, MinimumNaNSplit):
    def __init__(self, categories: dict, *args, n_splits: int = 5, n_repeats: int = 10,
                 oversample: bool = True, max_features: int = float("inf"), **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats)
        if not oversample:
            self.oversample = lambda x, func, axis: x
        self.categories = categories
        self.max_features = max_features

    def cv_cm_jim(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            
            y_train = labels[train_idx]
            mixup2(arr=x_train, labels=y_train, obs_axs=obs_axs, alpha=1., seed=None)
            y_test = labels[test_idx]
            # for i in set(labels):
            #     # fill in train data nans with random combinations of existing train data trials (mixup)
            #     idx[obs_axs] = y_train == i
            #     x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)], axis=obs_axs, func=mixup)

            # fill in test data nans with noise from distribution
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

            # feature selection
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features, replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]

            # fit model and score results
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return matk / divisor
    
    # untested 11/30
    # def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray,
    #             normalize: str = None, obs_axs: int = -2, time_axs: int = -1, n_jobs: int = 1,
    #             window: int = None, step_size: int = 1,
    #                 shuffle: bool = False, oversample: bool = True) -> np.ndarray:
    #     """Cross-validated confusion matrix with windowing and optional shuffling. REPLACING THIS, DEPRECATED"""
    #     n_cats = len(set(labels))
    #     time_axs_positive = time_axs % x_data.ndim

    #     out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)

    #     if window is not None:
    #         # Include the step size in the windowed output shape
    #         steps = (x_data.shape[time_axs_positive] - window) // step_size + 1
    #         out_shape = (steps,) + out_shape
                
    #     mats = np.zeros(out_shape, dtype=np.uint8)
    #     data = x_data.swapaxes(0, obs_axs)

    #     if shuffle:
    #         # shuffled label pool
    #         label_stack = []
    #         for i in range(self.n_repeats):
    #             label_stack.append(labels.copy())
    #             self.shuffle_labels(data, label_stack[-1], 0)

    #         # build the test/train indices from the shuffled labels for each
    #         # repetition, then chain together the repetitions
    #         # splits = (train, test)

    #         print("Shuffle validation:")
    #         for i, labels in enumerate(label_stack):
    #             # Compare with the first repetition to ensure variety in shuffles
    #             if i > 0:
    #                 diff = np.sum(label_stack[0] != labels)

    #         idxs = ((self.split(data, l), l) for l in label_stack)
    #         idxs = ((itertools.islice(s, self.n_splits),
    #                  itertools.repeat(l, self.n_splits))
    #                 for s, l in idxs)
    #         splits, label = zip(*idxs)
    #         splits = itertools.chain.from_iterable(splits)
    #         label = itertools.chain.from_iterable(label)
    #         idxs = zip(splits, label)

    #     else:
    #         idxs = ((splits, labels) for splits in self.split(data, labels))
    
    #     # 11/1 below is aaron's code for windowing. 
    #     def proc(train_idx, test_idx, l):
    #         x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
    #         print(f"x_stacked shape: {x_stacked.shape}")

    #         # Use the updated windower function with step_size
    #         windowed = windower(x_stacked, window, axis=time_axs, step_size=step_size)
    #         print(f"windowed shape: {windowed.shape}")

    #         out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
    #         for i, x_window in enumerate(windowed):
    #             x_flat = x_window.reshape(x_window.shape[0], -1)
    #             x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
    #             out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
    #         return out

    #     # # loop over folds and repetitions
    #     if n_jobs == 1:
    #         idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
    #         results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
    #     else:
    #         results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
    #             delayed(proc)(train_idx, test_idx, l)
    #             for (train_idx, test_idx), l in idxs)

    #     # # Collect the results
    #     for i, result in enumerate(results):
    #         rep, fold = divmod(i, self.n_splits)
    #         mats[:, rep, fold] = result

    #     # normalize, sum the folds
    #     mats = np.sum(mats, axis=-3)
    #     if normalize == 'true':
    #         divisor = np.sum(mats, axis=-1, keepdims=True)
    #     elif normalize == 'pred':
    #         divisor = np.sum(mats, axis=-2, keepdims=True)
    #     elif normalize == 'all':
    #         divisor = self.n_repeats
    #     else:
    #         divisor = 1
    #     return mats / divisor

    def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray, normalize: str = None, 
        obs_axs : int = -2, time_axs: int = -1, window: int = None, step_size: int = 1, 
        shuffle: bool = False, oversample: bool = True) -> np.ndarray:
        """
        Cross-validated confusion matrix with windowing and optional shuffling.
        
        This function performs cross-validated decoding with optional sliding windows over time.
        It can shuffle labels (for permutation testing) and handles missing data via mixup.
        """
        
        # Step 1: Setup basic parameters
        # Count unique classes in the labels (e.g., 2 for binary classification)
        n_cats = len(set(labels))
        
        # Convert negative time axis to positive (e.g., -1 becomes 3 for 4D array)
        time_axs_positive = time_axs % x_data.ndim
        
        # Step 2: Determine output shape based on windowing
        # Base shape without windows: (repeats, splits, classes, classes)
        base_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)
        
        if window is not None:
            # Calculate how many windows fit with the given step size
            # E.g., 256 samples, window=64, step=32 → (256-64)/32 + 1 = 7 windows
            steps = (x_data.shape[time_axs_positive] - window) // step_size + 1
            
            # Add windows dimension: (repeats, splits, windows, classes, classes)
            out_shape = (self.n_repeats, self.n_splits, steps, n_cats, n_cats)
        else:
            # No windowing - use base shape
            out_shape = base_shape
        
        # Step 3: Initialize output array and prepare data
        # Create array to store all confusion matrices
        mats = np.zeros(out_shape, dtype=np.float32)
        
        # Move observations/trials to first axis for easier indexing
        # E.g., (trials, channels, freqs, time) stays same if obs_axs=0
        data = x_data.swapaxes(0, obs_axs)
        
        # Initialize random state for reproducibility
        rng = np.random.RandomState(seed=self.random_state if hasattr(self, 'random_state') else None)
        
        # Step 4: Main cross-validation loop
        for i in range(self.n_repeats):
            # Each repeat gets a different random split of the data
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=rng)
            
            # Iterate through each fold
            for f, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
                # Extract train/test data for this fold
                x_train = data[train_idx]
                y_train = labels[train_idx].copy()  # Copy to avoid modifying original
                x_test = data[test_idx]
                y_test = labels[test_idx]
                
                # Step 5: Optional label shuffling (for permutation testing)
                if shuffle:
                    # Randomly permute training labels to break label-data relationship
                    rng.shuffle(y_train)
                
                # Step 6: Window and predict
                # This returns confusion matrix(es) for this fold
                cm_windowed = self._window_and_predict_minimal(
                    x_train, y_train, x_test, y_test, 
                    window, step_size, time_axs_positive, oversample
                )
                
                # Step 7: Store results
                if window is not None:
                    # cm_windowed shape: (n_windows, n_cats, n_cats)
                    mats[i, f, :] = cm_windowed
                else:
                    # cm_windowed shape: (n_cats, n_cats) 
                    mats[i, f] = cm_windowed
        
        # Step 8: Reorganize dimensions for output
        if window is not None:
            # Current: (n_repeats, n_splits, n_windows, n_cats, n_cats)
            # Sum over splits (they're from the same repeat, should be combined)
            mats = np.sum(mats, axis=1)  # → (n_repeats, n_windows, n_cats, n_cats)
            
            # Move windows to first dimension (expected by compute_accuracies)
            mats = np.transpose(mats, (1, 0, 2, 3))  # → (n_windows, n_repeats, n_cats, n_cats)
        else:
            # No windows: just sum over splits
            mats = np.sum(mats, axis=1)  # → (n_repeats, n_cats, n_cats)
        
        # Step 9: Apply normalization
        if normalize == 'true':
            # Normalize by row sums (true class totals)
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            # Normalize by column sums (predicted class totals)
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            # Normalize by total sum
            divisor = np.sum(mats, axis=(-2,-1), keepdims=True)
        else:
            # No normalization
            divisor = 1
        
        # Step 10: Safe division and return
        with np.errstate(divide='ignore', invalid='ignore'):
            result = mats / divisor
            # Replace any inf/nan from division by zero with 0
            result[~np.isfinite(result)] = 0
        
        return result

    def _window_and_predict_minimal(self, x_train, y_train, x_test, y_test, 
                                window, step_size, time_axs, oversample):
        """
        helper function that handles windowing and prediction for a single CV fold

        EXAMPLE FLOW:

        Initial data:
        - x_train: (70, 10, 256) - 70 training trials, some with NaNs
        - x_test: (30, 10, 256) - 30 test trials, some with NaNs
        - window=64, step_size=32

        1. _window_and_predict_minimal combines data:
        x_stacked = (100, 10, 256)

        2. sample_fold is called:
        - Reorders data to put train first, test second
        - Applies mixup to fill training NaNs with smart combinations
        - Fills test NaNs with random noise
        - Returns processed (100, 10, 256) with no NaNs

        3. Windowing applied:
        windowed = (7, 100, 10, 64) - 7 time windows

        4. For each window:
        - Flatten: (100, 10, 64) → (100, 640)
        - Split: train (70, 640), test (30, 640)
        - Decode and get confusion matrix

        5. Return: (7, 2, 2) for binary classification with 7 windows

        """
        # step 1: get number of classes from decoder configuration
        n_cats = len(self.categories)
        
        # step 2: combine combine train and test data for consistent windowing
        # this ensures windows align properly across train/test boundary
        x_stacked = np.concatenate((x_train, x_test), axis=0)

        # step 3: create index arrays for sample_fold
        # these tell sample_fold which samples are train vs test
        train_idx = np.arange(len(y_train)) # [0,1,2,...,n_train-1]
        test_idx = np.arange(len(y_train), len(y_train) + len(y_test)) # [n_train, ..., n_total-1]
        
        # Step 4: Use sample_fold for preprocessing. 
        # This handles:
        # - Mixup augmentation for training NaNs
        # - Random noise filling for test NaNs
        # - Proper data splitting
        x_processed, y_train_proc, y_test_proc = sample_fold(
            train_idx, test_idx, x_stacked, 
            np.concatenate([y_train, y_test]), # combine labels for sample_fold
            axis=0, # trials are on axis 0
            oversample=oversample
        )
        
        # Step 5: Apply sliding window if specified
        if window is not None:
            # windower creates overlapping windows
            # E.g., (100, 10, 256) -> (7, 100, 10, 64)
            # where 7 windows of size 64 with step size 32
            windowed = windower(x_processed, window, axis=time_axs, step_size=step_size)
        else:
            # no windowing - add fake window dimension for consistency
            # (100, 10, 256) -> (1, 100, 10, 256)
            windowed = x_processed[np.newaxis, ...]
        
        # Step 6: Process each time window independently 
        out_cm = [] # list to collect confusion matrices

        for x_window in windowed:
            # Step 6a: Flatten all features except trials dimension
            # E.g., (100, 10, 64) -> (100, 640)
            # This creates feature vector for each trial
            x_flat = x_window.reshape(x_window.shape[0], -1)

            # Step 6b: Split back into train and test sets
            # We know first len(y_train_proc) samples are training
            x_train_w, x_test_w = np.split(x_flat, [len(y_train_proc)], axis=0)
            
            # Step 6c: Train model and predict
            self.fit(x_train_w, y_train_proc) # train on this window's features
            preds = self.predict(x_test_w) # predict test labels

            # Step 6d: Create confusion matrix for this window
            # Compares true test labels with predictions
            out_cm.append(confusion_matrix(y_test_proc, preds))
        
        # Step 7: Format output
        # If only one window, remove the window dimension
        # otherwise, return array of confusion matrices
        if len(out_cm) == 1:
            return np.squeeze(np.array(out_cm)) # remove window dimension
        else:
            return np.array(out_cm) # Shape: (n_windows, n_cats, n_cats)
    
    def fit_predict(self, x_train, x_test, y_train, y_test):
        # fit model and score results
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)
        return confusion_matrix(y_test, pred)
    
    def cv_cm_return_scores(self, x_data: np.ndarray, labels: np.ndarray,
                            normalize: str = None, obs_axs: int = -2):
        '''
        trying to get the scores manually from cv cm but i realize that in decoders.py, PcaLdaClassification already has a get_scores function. Try get_scores with shuffle=True to get fake, permuted scores.
        '''
        # Get the confusion matrix by calling `cv_cm`
        cm = self.cv_cm_jim(x_data, labels, normalize, obs_axs)

        # Average the confusion matrices across the repetitions
        cm_avg = np.mean(cm, axis=0)  # Now cm_avg will be of shape (2, 2)

        # Calculate the individual decoding scores (Accuracy, Precision, etc.)
        scores = self.calculate_scores(cm_avg)

        return cm_avg, scores

    def calculate_scores(self, cm):
        """
        Calculate the individual decoding scores from the confusion matrix. 10/27 Ugh Aaron already does this directly in the PcaLdaClassification class... 

        Parameters:
        - cm: The confusion matrix (averaged over folds).

        Returns:
        - scores: A dictionary containing the scores (accuracy, precision, recall, f1, d-prime) for each class.
        """
        scores = {}
        tp = np.diag(cm)  # True Positives
        fp = np.sum(cm, axis=0) - tp  # False Positives
        fn = np.sum(cm, axis=1) - tp  # False Negatives
        tn = np.sum(cm) - (fp + fn + tp)  # True Negatives

        # Calculate accuracy, precision, recall, and f1 score
        accuracy = np.sum(tp) / np.sum(cm)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Store the basic scores
        scores['accuracy'] = accuracy
        scores['precision'] = precision
        scores['recall'] = recall
        scores['f1'] = f1

        # Calculate hit rate and false alarm rate
        hit_rate = recall  # Hit rate is the same as recall (TP / (TP + FN))
        false_alarm_rate = fp / (fp + tn + 1e-8)  # False alarm rate (FP / (FP + TN))

        # Ensure hit_rate and false_alarm_rate are in valid range [0, 1] for Z-transform
        hit_rate = np.clip(hit_rate, 1e-8, 1 - 1e-8)
        false_alarm_rate = np.clip(false_alarm_rate, 1e-8, 1 - 1e-8)

        # Z-transform to calculate d-prime
        z_hit_rate = norm.ppf(hit_rate)  # Z-transform for hit rate
        z_false_alarm_rate = norm.ppf(false_alarm_rate)  # Z-transform for false alarm rate

        # Calculate d-prime
        d_prime = z_hit_rate - z_false_alarm_rate

        # Store d-prime in the scores dictionary
        scores['d_prime'] = d_prime

        return scores

def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)

def windower(x_data: np.ndarray, window_size: int = None, axis: int = -1,
             step_size: int = 1, insert_at: int = 0):
    if window_size is None:
        return x_data[np.newaxis, ...]  # Add a new axis for compatibility

    axis = axis % x_data.ndim
    data_length = x_data.shape[axis]
    
    # Compute the number of full steps (exclude remainder)
    full_steps = (data_length - window_size) // step_size + 1

    # Create the sliding window view for full windows
    windowed = sliding_window_view(x_data, window_shape=window_size, axis=axis)
    
    # Apply step_size by slicing
    if step_size > 1:
        slicing = [slice(None)] * windowed.ndim
        slicing[axis] = slice(0, None, step_size) # try 0, None, step_size for now, I think this should exclude the remainder..? Keep debugging with the bottom cell.
        windowed = windowed[tuple(slicing)]
    
    # Move the window dimension to the desired location
    if insert_at != axis:
        windowed = np.moveaxis(windowed, axis, insert_at)
    
    return windowed

# this is aaron's windower function. Replace with my windower that accounts for step size.
# def windower(x_data: np.ndarray, window_size: int, axis: int = -1, insert_at: int = 0):
#     """Create a sliding window view of the array with a given window size."""
#     # Compute the shape and strides for the sliding window view
#     shape = list(x_data.shape)
#     shape[axis] = x_data.shape[axis] - window_size + 1
#     shape.insert(axis, window_size)
#     strides = list(x_data.strides)
#     strides.insert(axis, x_data.strides[axis])

#     # Create the sliding window view
#     out = as_strided(x_data, shape=shape, strides=strides)

#     # Move the window size dimension to the front
#     out = np.moveaxis(out, axis, insert_at)

#     return out

# modified by jim 11/23, check aaron_decoding_init.py for original.
def sample_fold(train_idx: np.ndarray, test_idx: np.ndarray,
                x_data: np.ndarray, labels: np.ndarray,
                axis: int, oversample: bool = True):
    """
    This function prepares a single fold of cross-validation data by:
    1. Extracting train/test samples
    2. Applying mixup augmentation to handle NaNs in training data
    3. Filling test NaNs with random noise
    4. Returning the processed data
    
    Parameters:
    -----------
    train_idx : np.ndarray
        Indices of training samples (e.g., [0, 2, 3, 5, 7, 8])
    test_idx : np.ndarray
        Indices of test samples (e.g., [1, 4, 6, 9])
    x_data : np.ndarray
        Full data array (e.g., shape: (100, 10, 256) for 100 trials, 10 channels, 256 timepoints)
    labels : np.ndarray
        Labels for all samples (e.g., [0, 1, 0, 1, ...] for 100 trials)
    axis : int
        Axis along which to select samples (typically 0 for trials)
    oversample : bool
        Whether to apply mixup augmentation for NaN handling
    
    Returns:
    --------
    x_stacked : np.ndarray
        Combined train+test data with NaNs handled
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    """
    
    # Step 1: Combine train and test indices
    # This creates a single array of all indices we'll use
    # E.g., train_idx=[0,2,4], test_idx=[1,3] → idx_stacked=[0,2,4,1,3]
    idx_stacked = np.concatenate((train_idx, test_idx))
    
    # Step 2: Extract the data for these indices
    # np.take is like fancy indexing but handles axis parameter cleanly
    # If x_data is (100, 10, 256) and axis=0, this selects specific trials
    x_stacked = np.take(x_data, idx_stacked, axis)
    
    # Step 3: Extract corresponding labels
    # Labels are 1D, so we just index directly
    y_stacked = labels[idx_stacked]
    
    # Step 4: Determine where to split train/test
    # We know first 'sep' samples are training
    sep = train_idx.shape[0]  # Number of training samples
    
    # Step 5: Split labels into train and test
    # E.g., if sep=3, y_train gets first 3, y_test gets rest
    y_train, y_test = np.split(y_stacked, [sep])
    
    # Step 6: Split data into train and test
    # Same split but along the specified axis
    x_train, x_test = np.split(x_stacked, [sep], axis=axis)
    
    # Step 7: Apply mixup augmentation to training data if requested
    if oversample:
        # mixup2 modifies x_train IN PLACE
        # It finds NaN trials and fills them with weighted combinations
        # of other trials from the same class
        mixup2(arr=x_train, labels=y_train, obs_axs=axis, alpha=1., seed=None)
        
        # How mixup2 works internally:
        # 1. Finds trials with NaNs
        # 2. For each NaN trial:
        #    - Finds two random trials (one from same class, one from any class)
        #    - Creates weighted average: l * same_class + (1-l) * other_class
        #    - Where l is drawn from Beta(alpha, alpha) distribution
    
    # Step 8: Fill test data NaNs with random noise
    # This is simpler than mixup - just replace NaNs with Gaussian noise
    is_nan = np.isnan(x_test)  # Boolean mask of NaN locations
    x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))
    # Draws from standard normal (mean=0, std=1) for each NaN
    
    # Step 9: Recombine processed train and test data
    # Now both have NaNs handled appropriately
    x_stacked = np.concatenate((x_train, x_test), axis=axis)
    
    # Return processed data and split labels
    return x_stacked, y_train, y_test

def get_and_plot_confusion_matrix_for_rois_jim(
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, save_dir,
    time_interval_name=None, other_string_to_add=None, n_splits=5, n_repeats=5, obs_axs=0, balance_method='pad_with_nans', explained_variance=0.8, random_state=42, timestamp=None
):
    """
    Compute the confusion matrix for each ROI and return it. This function allows for balancing trial counts
    either by padding with NaNs or by subsampling trials to match the condition with the fewest valid (non-NaN) trials.
    
    Parameters:
    - roi_labeled_arrays: Dictionary containing the reshaped data for each ROI.
    - rois: List of regions of interest (ROIs) to process.
    - condition_comparison: The condition that we're comparing labels for (e.g., 'BigLetter').
    - strings_to_find: List of strings or string groups to identify condition labels.
    - save_dir: Directory to save the confusion matrix plots.
    - time_interval_name: Optional string to add to the filename for the time interval.
    - other_string_to_add: Optional string to add to the filename for other purposes.
    - n_splits: Number of splits for cross-validation.
    - n_repeats: Number of repeats for cross-validation.
    - obs_axs: The trials axis.
    - explained_variance: The amount of variance to explain in the PCA.
    - balance_method: 'pad_with_nans' or 'subsample' to balance trial counts between conditions.
    - random_state: Random seed for reproducibility.
    - timestamp: timestamp of when this script was run for filenaming purposes
    
    Returns:
    - confusion_matrices: Dictionary containing confusion matrices for each ROI.
    """
    confusion_matrices = {}
    rng = np.random.RandomState(random_state)

    for roi in rois:
        roi_save_dir = os.path.join(save_dir, f"{roi}")
        os.makedirs(roi_save_dir, exist_ok=True)
        print(f"Processing ROI: {roi}")
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
        )

        # Create a Decoder and run cross-validation
        decoder = Decoder(cats, explained_variance, oversample=True, n_splits=n_splits, n_repeats=n_repeats)

        # Use the concatenated data for the decoder
        cm = decoder.cv_cm_jim(concatenated_data, labels, normalize='true', obs_axs=obs_axs)
        cm_avg = np.mean(cm, axis=0)

        # Store the confusion matrix in the dictionary
        confusion_matrices[roi] = cm_avg

        # Convert tuple labels to simple strings for display
        display_labels = [
            key[0] if isinstance(key, tuple) and len(key) == 1 else str(key)
        for key in cats.keys()
        ]
        
        # Plot the Confusion Matrix
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_avg, display_labels=display_labels)
        disp.plot(ax=ax, im_kw={"vmin": 0, "vmax": 1})

        # Save the figure with the time interval in the filename
        time_str = f"_{time_interval_name}" if time_interval_name else ""
        other_str = f"_{other_string_to_add}" if other_string_to_add else ""
        timestamp_str = f"{timestamp}_" if timestamp else ""
        file_name = (
            f'{timestamp_str}{roi}_{condition_comparison}{time_str}{other_str}_time_averaged_confusion_matrix_'
            f'{n_splits}splits_{n_repeats}repeats_{balance_method}.png'
        )
        plt.savefig(os.path.join(roi_save_dir, file_name))
        plt.close()

    return confusion_matrices

# TODO: Clean this up.   
# Make subfunctions to break this down. Everything before defining the Decoder objects can be a function, that can be shared between this and the whole time window version.   
# The decoder true and decoder shuffle can be done with a function maybe.   
# And maybe return just accuracies, which I can then call this entire function separately for true and shuffled.
# ALSO STORE THE SHUFFLED OUTPUT IN A NUMPY ARRAY SO I DON'T HAVE TO MAKE IT EVERY TIME
def get_confusion_matrices_for_rois_time_window_decoding_jim(
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, n_splits=5, n_repeats=5, obs_axs=0, time_axs=-1,
    balance_method='pad_with_nans', explained_variance=0.8, random_state=42, window_size=None,
    step_size=1, n_perm=100, sampling_rate=256, first_time_point=-1
):
    """
    Performs time-windowed decoding analysis for specified regions of interest (ROIs) and conditions.

    This function iterates through each ROI, prepares the data, and then runs a decoding
    analysis using a sliding time window. It calculates confusion matrices for both true
    and shuffled labels. The results, including the confusion matrices and window parameters,
    are stored and returned.

    Parameters
    ----------
    roi_labeled_arrays : dict
        A dictionary where keys are ROI names and values are LabeledArray objects
        containing the epoched data for that ROI. The LabeledArray should have
        dimensions for conditions, trials, channels, and time samples.
    rois : list of str
        A list of ROI names (keys in `roi_labeled_arrays`) to process.
    condition_comparison : str
        A descriptive name for the comparison being made (e.g., 'BigLetter_vs_SmallLetter').
        Used for storing results.
    strings_to_find : list of list of str or list of str
        A list defining the groups of conditions to compare. Each inner list (or string
        if only one condition per group) contains condition names (or parts of names)
        that will be used to select and label data for each class in the decoding.
    n_splits : int, optional
        Number of splits for the cross-validation. Default is 5.
    n_repeats : int, optional
        Number of repetitions for the cross-validation for true labels. Default is 5.
    obs_axs : int, optional
        The axis in the data array that corresponds to observations (trials).
        Default is 0.
    time_axs : int, optional
        The axis in the data array that corresponds to time samples. Default is -1.
    balance_method : str, optional
        Method to balance trial counts across conditions:
        'pad_with_nans': Pads conditions with fewer trials with NaNs.
        'subsample': Subsamples trials from conditions with more trials.
        Default is 'pad_with_nans'.
    explained_variance : float, optional
        The amount of variance to explain in the PCA. Default is 0.8.
    random_state : int, optional
        Seed for the random number generator for reproducibility. Default is 42.
    window_size : int, optional
        The number of time samples in each sliding window. If None, the entire
        time axis length is used (i.e., no sliding window). Default is None.
    step_size : int, optional
        The number of time samples to slide the window by. Default is 1.
    n_perm : int, optional
        Number of permutations for the shuffled label decoding (effectively the
        n_repeats for the shuffle decoder). Default is 100.
    sampling_rate : int or float, optional
        The sampling rate of the data in Hz. Used to convert sample-based
        window parameters to time. Default is 256.
    first_time_point : int or float, optional
        The time (in seconds) corresponding to the first sample in the epoch.
        Used to adjust the `start_times` of the windows if they are not
        aligned to the beginning of the concatenated data. Default is -1.

    Returns
    -------
    tuple of (dict, dict)
        - cm_true_per_roi : dict
            A dictionary where keys are ROI names. Each value is another dictionary
            containing:
                - 'cm_true' (numpy.ndarray): Confusion matrices for true labels.
                  Shape: (n_windows, n_repeats, n_classes, n_classes).
                - 'time_window_centers' (list of float): Center times of each window.
                - 'window_size' (int): Effective window size used.
                - 'step_size' (int): Effective step size used.
                - 'condition_comparison' (str): The `condition_comparison` input.
        - cm_shuffle_per_roi : dict
            Similar to `cm_true_per_roi`, but for shuffled labels:
                - 'cm_shuffle' (numpy.ndarray): Confusion matrices for shuffled labels.
                  Shape: (n_windows, n_perm, n_classes, n_classes).
                - (other keys are the same as in `cm_true_per_roi[roi]`)
    """
    # Initialize dictionaries to store confusion matrices for each ROI
    cm_true_per_roi = {}
    cm_shuffle_per_roi = {}
    rng = np.random.RandomState(random_state)
    first_sample = first_time_point * sampling_rate

    for roi in rois:
        print(f"Processing ROI: {roi}")

        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
        )

        # Get the number of timepoints
        time_axis_length = concatenated_data.shape[time_axs]

        # Determine effective window size and step size
        if window_size is None:
            effective_window_size = time_axis_length
            effective_step_size = time_axis_length  # No overlap
            n_windows = 1
            start_times = [0] # only one window
        else:
            effective_window_size = window_size
            effective_step_size = step_size
            n_windows = (time_axis_length - effective_window_size) // effective_step_size + 1
            # Apply first_time_point offset
            start_times = [first_sample + effective_step_size * i for i in range(n_windows)]
            
        print(f"start times are: {start_times}")
        print(f"Effective window size: {effective_window_size}")
        print(f"Effective step size: {effective_step_size}")
        print(f"Number of windows: {n_windows}")

        # Calculate time centers based on window size and step size
        time_window_centers = [
            (start + effective_window_size / 2) / sampling_rate
            for start in start_times
        ]
        print(f"time_window_centers are: {time_window_centers}")
        
        # Create Decoder instances
        decoder_true = Decoder(cats, explained_variance, oversample=True, n_splits=n_splits, n_repeats=n_repeats)
        decoder_shuffle = Decoder(cats, explained_variance, oversample=True, n_splits=n_splits, n_repeats=n_perm)

        # Run decoding with true labels
        cm_true = decoder_true.cv_cm_jim_window_shuffle(
            concatenated_data, labels, normalize='true', obs_axs=obs_axs, time_axs=time_axs,
            window=effective_window_size, step_size=effective_step_size, shuffle=False
        )

        # Run decoding with shuffled labels
        cm_shuffle = decoder_shuffle.cv_cm_jim_window_shuffle(
            concatenated_data, labels, normalize='true', obs_axs=obs_axs, time_axs=time_axs,
            window=effective_window_size, step_size=effective_step_size, shuffle=True
        )

        # Store the confusion matrices and time info
        cm_true_per_roi[roi] = {
            'cm_true': cm_true,  # Shape: (n_windows, n_repeats, n_classes, n_classes)
            'time_window_centers': time_window_centers,
            'window_size': effective_window_size,
            'step_size': effective_step_size,
            'condition_comparison': condition_comparison
        }

        cm_shuffle_per_roi[roi] = {
            'cm_shuffle': cm_shuffle,  # Shape: (n_windows, n_perm, n_classes, n_classes)
            'time_window_centers': time_window_centers,
            'window_size': effective_window_size,
            'step_size': effective_step_size,
            'condition_comparison': condition_comparison
        }

    return cm_true_per_roi, cm_shuffle_per_roi

def compute_accuracies(cm_true, cm_shuffle):
    """
    Compute accuracies from true and shuffled confusion matrices.

    This function calculates the accuracy for each window and repetition/permutation
    by taking the trace of the confusion matrix (sum of true positives) and
    dividing by the total sum of the matrix (total number of instances).

    Parameters
    ----------
    cm_true : numpy.ndarray
        Confusion matrices for true labels.
        Expected shape: (n_windows, n_repeats, n_classes, n_classes).
    cm_shuffle : numpy.ndarray
        Confusion matrices for shuffled labels.
        Expected shape: (n_windows, n_perm, n_classes, n_classes).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - accuracies_true : numpy.ndarray
            Accuracies for true labels. Shape: (n_windows, n_repeats).
        - accuracies_shuffle : numpy.ndarray
            Accuracies for shuffled labels. Shape: (n_windows, n_perm).
    """
    n_windows = cm_true.shape[0]
    n_repeats = cm_true.shape[1]
    n_perm = cm_shuffle.shape[1]

    accuracies_true = np.zeros((n_windows, n_repeats))
    accuracies_shuffle = np.zeros((n_windows, n_perm))

    for win_idx in range(n_windows):
        # True accuracies
        for rep_idx in range(n_repeats):
            cm = cm_true[win_idx, rep_idx]
            accuracies_true[win_idx, rep_idx] = np.trace(cm) / np.sum(cm)
        # Shuffled accuracies
        for perm_idx in range(n_perm):
            cm = cm_shuffle[win_idx, perm_idx]
            accuracies_shuffle[win_idx, perm_idx] = np.trace(cm) / np.sum(cm)

    return accuracies_true, accuracies_shuffle

def perform_time_perm_cluster_test_for_accuracies(accuracies_true, accuracies_shuffle, p_thresh=0.05, n_perm=50, seed=42):
    """
    Perform a time permutation cluster test on true vs. shuffled accuracies.

    This function transposes the accuracy arrays to have time as the last dimension
    (as typically expected by `time_perm_cluster`) and then runs the cluster-based
    permutation test to find significant time clusters where true accuracy
    is greater than shuffled accuracy.

    Parameters
    ----------
    accuracies_true : numpy.ndarray
        Accuracies for true labels. Expected shape: (n_windows, n_repeats).
    accuracies_shuffle : numpy.ndarray
        Accuracies for shuffled labels. Expected shape: (n_windows, n_perm).
    p_thresh : float, optional
        P-value threshold for cluster formation. Default is 0.05.
    n_perm : int, optional
        Number of permutations for the cluster test. Default is 50.
    seed : int, optional
        Random seed for reproducibility of the permutation test. Default is 42.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - significant_clusters : numpy.ndarray
            A boolean array indicating significant time windows (clusters).
            Shape: (n_windows,).
        - p_values : numpy.ndarray
            P-values for each identified cluster.
    """
    accuracies_true_T = accuracies_true.T
    accuracies_shuffle_T = accuracies_shuffle.T

    significant_clusters, p_values = time_perm_cluster(
        accuracies_true_T,
        accuracies_shuffle_T,
        p_thresh=p_thresh,
        n_perm=n_perm,
        tails=1,
        axis=0,
        stat_func=stat_func,
        n_jobs=1,
        seed=seed
    )
    return significant_clusters, p_values

def plot_accuracies(time_points, accuracies_true, accuracies_shuffle, significant_clusters,
                    window_size, step_size, sampling_rate, condition_comparison, roi, save_dir, timestamp=None, p_thresh=0.05):
    """
    Plot mean true and shuffled accuracies over time with significance.

    This function visualizes the average decoding accuracy from true labels and
    shuffled labels across different time windows. It highlights significant
    time clusters (where true accuracy is significantly higher than shuffled)
    with horizontal bars and asterisks. The plot is saved to a file.

    Parameters
    ----------
    time_points : array-like
        The center time points (in seconds) for each window.
    accuracies_true : numpy.ndarray
        Accuracies for true labels. Shape: (n_windows, n_repeats).
    accuracies_shuffle : numpy.ndarray
        Accuracies for shuffled labels. Shape: (n_windows, n_perm).
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    window_size_samples : int
        The size of the decoding window in samples.
    step_size_samples : int
        The step size of the decoding window in samples. (Not directly used in plot rendering logic beyond filename).
    sampling_rate : float
        The sampling rate of the data in Hz.
    condition_comparison : str
        A string describing the condition comparison (e.g., "TaskA_vs_TaskB").
        Used in the plot title and filename.
    roi : str
        The Region of Interest (ROI) being plotted. Used in the plot title
        and filename.
    save_dir : str
        The directory where the plot image will be saved.
    first_time_point_s : float, optional
        The time in seconds of the first sample of the epoch, used for x-axis limits
        if needed, though current xlim are fixed. Default is 0.
    timestamp : str
        Timestamp string for filenaming purposes
    p_thresh : float
        p-value threshold for determining significant clusters
    """
    n_repeats = accuracies_true.shape[1]
    n_perm = accuracies_shuffle.shape[1]

    # Compute mean and standard error
    mean_true_accuracy = np.mean(accuracies_true, axis=1)
    std_true_accuracy = np.std(accuracies_true, axis=1)
    se_true_accuracy = std_true_accuracy / np.sqrt(n_repeats)

    mean_shuffle_accuracy = np.mean(accuracies_shuffle, axis=1)
    std_shuffle_accuracy = np.std(accuracies_shuffle, axis=1)
    se_shuffle_accuracy = std_shuffle_accuracy / np.sqrt(n_perm)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, mean_true_accuracy, label='True Accuracy', color='blue')
    plt.fill_between(
        time_points,
        mean_true_accuracy - se_true_accuracy,
        mean_true_accuracy + se_true_accuracy,
        alpha=0.2,
        color='blue'
    )

    plt.plot(time_points, mean_shuffle_accuracy, label='Shuffled Accuracy', color='red')
    plt.fill_between(
        time_points,
        mean_shuffle_accuracy - se_shuffle_accuracy,
        mean_shuffle_accuracy + se_shuffle_accuracy,
        alpha=0.2,
        color='red'
    )

    # Compute window duration
    window_duration = window_size / sampling_rate

    # Find contiguous significant clusters
    def find_clusters(significant_clusters: Union[np.ndarray, List[bool], Sequence[bool]]):
        """Helper to find start and end indices of contiguous True blocks."""
        clusters = []
        in_cluster = False
        for idx, val in enumerate(list(significant_clusters)):
            if val and not in_cluster:
                # Start of a new cluster
                start_idx = idx
                in_cluster = True
            elif not val and in_cluster:
                # End of the cluster
                end_idx = idx - 1
                clusters.append((start_idx, end_idx))
                in_cluster = False
        # Handle the case where the last value is in a cluster
        if in_cluster:
            end_idx = len(list(significant_clusters)) - 1
            clusters.append((start_idx, end_idx))
        return clusters

    clusters = find_clusters(significant_clusters)

    # # Determine y position for the bars
    # max_y = np.max(mean_true_accuracy + se_true_accuracy)
    # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
    # y_bar = max_y + 0.02  # Adjust as needed
    # plt.ylim([min_y, y_bar + 0.05])  # Adjust ylim to accommodate the bars

    # Set y_bar to a fixed value within the y-axis limits
    y_bar = 0.95  # Fixed value near the top of the y-axis

    # Plot horizontal bars and asterisks for significant clusters
    for cluster in clusters:
        start_idx, end_idx = cluster
        start_time = time_points[start_idx] - (window_duration / 2)
        end_time = time_points[end_idx] + (window_duration / 2)
        plt.hlines(y=y_bar, xmin=start_time, xmax=end_time, color='black', linewidth=2)
        # Place an asterisk at the center of the bar
        center_time = (start_time + end_time) / 2
        plt.text(center_time, y_bar + 0.01, '*', ha='center', va='bottom', fontsize=14)

    # Set axis limits
    plt.ylim(0, 1)  # Y-axis limits
    plt.xlim(-1, 1.5)  # X-axis limits

    plt.xlabel('Time from Stim Onset (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoding Accuracy over Time for {condition_comparison} in ROI {roi}')
    plt.legend()
    
    # CREATE TIMESTAMP PREFIX
    timestamp_str = f"{timestamp}_" if timestamp else ""

    # CREATE P THRESH PREFIX
    p_thresh_str = str(p_thresh)
    
    # Construct the filename
    filename = f"{timestamp_str}{condition_comparison}_ROI_{roi}_window{window_size}_step{step_size}_{n_repeats}_repeats_{n_perm}_perm_{p_thresh_str}_p_thresh.png"
    filepath = os.path.join(save_dir, filename)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Save and close the plot
    plt.savefig(filepath)
    plt.close()

# james sun cluster decoding functions 8/4/25, update as needed

def decode_on_sig_tfr_clusters(
    X_train_raw, y_train, X_test_raw,
    train_indices, test_indices,
    concatenated_data, labels, cats, 
    obs_axs, chans_axs,
    stat_func, p_thresh, n_perm,
    Decoder, explained_variance, oversample,
    ignore_adjacency=1, seed=42, tails=2, alpha=1.
):
    """
    Balance data and decode with TFR cluster masking. Returns the sig tfr cluster masks for later plotting.
    
    Parameters
    ----------
    X_train_raw : np.ndarray
        Raw training data (trials, channels, freqs, times)
    y_train : np.ndarray
        Training labels
    X_test_raw : np.ndarray
        Raw test data (trials, channels, freqs, times)
    train_indices : np.ndarray
        Indices of training trials in the original concatenated data
    test_indices : np.ndarray
        Indices of test trials in the original concatenated data
    concatenated_data : np.ndarray
        Full concatenated data array (all_trials, channels, freqs, times)
    labels : np.ndarray
        All labels corresponding to concatenated_data
    cats : dict
        Dictionary mapping condition names to label integers
    obs_axs : int
        Axis of the data that contains trial labels
    chans_axs : int
        Axis of the data that contains channel labels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold for significance
    n_perm : int
        Number of permutations for cluster test
    Decoder : class
        Decoder class to use for decoding
    explained_variance : float
        Proportion of variance to explain with PCA
    oversample : bool
        Whether to oversample the training data
    ignore_adjacency : int
        Whether to ignore adjacency in clustering (1=ignore, 0=use adjacency)
    seed : int
        Random seed for reproducibility
    tails : int
        Number of tails for statistical test (1 or 2)
    alpha : float
        Alpha parameter for mixup augmentation
        
    Returns
    -------
    preds : np.ndarray
        Predicted labels for test data
    channel_masks : dict
        Dictionary of channel masks for significant clusters
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    # Get condition names from cats dictionary
    condition_names = [k[0] if isinstance(k, tuple) else k for k in cats.keys()]
    
    # Step 1: Create training-only TFR masks
    channel_masks, channel_t_values = compute_sig_tfr_masks_from_concatenated_data(
        concatenated_data, labels, train_indices, condition_names, cats,
        obs_axs, chans_axs,
        stat_func, p_thresh, n_perm, 
        ignore_adjacency, seed, tails
    )
    
    # Step 2: Apply masks and flatten
    X_train_masked = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
        X_train_raw, obs_axs, chans_axs, channel_masks
    )
    X_test_masked = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
        X_test_raw, obs_axs, chans_axs, channel_masks
    )
    
    # Step 3: Decode
    decoder = Decoder(cats, explained_variance=explained_variance, n_splits=1, n_repeats=1, oversample=oversample)
    
    # Handle NaN filling using existing mixup2 function
    mixup2(arr=X_train_masked, labels=y_train, obs_axs=obs_axs, alpha=alpha, seed=seed)
    
    # Fill test NaNs with noise (as done in sample_fold)
    is_nan = np.isnan(X_test_masked)
    X_test_masked[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

    # Fit and predict
    decoder.fit(X_train_masked, y_train)
    preds = decoder.predict(X_test_masked)
    
    # debugging
    print(f"Number of significant clusters found: {sum(mask.any() for mask in channel_masks.values())}")
    print(f"Total significant features: {sum(mask.sum() for mask in channel_masks.values())}")

    return preds, channel_masks, channel_t_values

def compute_sig_tfr_masks_from_roi_labeled_array(
    roi_labeled_array, train_indices, condition_names,
    obs_axs, chans_axs, stat_func, p_thresh, n_perm, 
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR clusters using only training trials from roi_labeled_array.
    
    Returns:
    --------
    channel_masks : dict
        {channel_label: mask_array} where mask is (n_freqs, n_times)
    """
    # Get channel labels from the labeled array
    channel_labels = roi_labeled_array.labels[chans_axs+1]

    # Validate we have exactly 2 conditions for now
    if len(condition_names) != 2:
        raise ValueError(
            f"For now, just doing perm test instead of ANOVA, "
            f"so this will only work for two conditions. Got {len(condition_names)} conditions."
        )

    # Split training data by condition
    train_data_by_condition = {}
    for cond in condition_names:  # hmm the stats only work for two conditions, so just do two conditions for now. Can expand to >2 conditions in the future, would just need to do ANOVA i think instead of time perm cluster.
        # Extract training trials for this condition
        cond_data = roi_labeled_array[cond]  # Shape: (trials, channels, freqs, times). Test this!
        cond_train_data = np.take(cond_data, train_indices, axis=obs_axs) # TODO: keep going through this code 4:45 on 8/1 - huh? this is a 4d array, check what train_indices is. Should grab along the trials axis. Maybe do np.take(train_indices, axis=obs_axs) to be safe.
        train_data_by_condition[cond] = cond_train_data
    
    # Compute significant clusters for each channel
    channel_masks = compute_sig_tfr_masks_for_specified_channels(
        channel_labels, train_data_by_condition, condition_names, obs_axs, chans_axs
    )
    
    return channel_masks

def compute_sig_tfr_masks_for_specified_channels(
    n_channels, train_data_by_condition, condition_names, 
    obs_axs, chans_axs, stat_func, p_thresh, n_perm,
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR masks for each channel.
    
    Parameters
    ----------
    n_channels : int
        Number of channels to process
    train_data_by_condition : dict
        Dictionary with condition names as keys and data arrays as values
    condition_names : list
        List of condition names (must be exactly 2 for now)
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold
    n_perm : int
        Number of permutations
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed
    tails : int
        Number of tails for test
        
    Returns
    -------
    channel_masks : dict
        Dictionary where keys are channel indices (int) and values are 
        boolean masks of shape (n_freqs, n_times)
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """    
    channel_masks = {}
    channel_t_values = {}
    
    # For each channel, compute significant clusters
    for ch_idx in range(n_channels):
        # Get data for this channel across conditions
        cond0_data = train_data_by_condition[condition_names[0]]
        cond1_data = train_data_by_condition[condition_names[1]]
        
        # Extract channel data
        cond0_chan_data = np.take(cond0_data, ch_idx, axis=chans_axs)
        cond1_chan_data = np.take(cond1_data, ch_idx, axis=chans_axs)
        
        # Run time perm cluster test
        if len(cond0_chan_data) > 0 and len(cond1_chan_data) > 0:
            # let's grab the t values too for debugging and plotting - this will only work if using scipy stats ttest ind, otherwise it will crash!
            t_values = stat_func(cond0_chan_data, cond1_chan_data, axis=0).statistic #.statistic only exists for scipy stats
            channel_t_values[ch_idx] = t_values
            
            # get sig tfr mask for this channel
            mask, _ = time_perm_cluster(
                cond0_chan_data, cond1_chan_data,
                stat_func=stat_func,
                p_thresh=p_thresh,
                n_perm=n_perm,
                axis=0,  # trials are now first axis after taking channel
                ignore_adjacency=ignore_adjacency,
                seed=seed,
                tails=tails
            )
            channel_masks[ch_idx] = mask
        else:
            # No data for comparison - create zero mask
            if len(cond1_chan_data) > 0:
                mask_shape = (cond1_chan_data.shape[1], cond1_chan_data.shape[2])
            else:
                mask_shape = (cond0_chan_data.shape[1], cond0_chan_data.shape[2])
            channel_masks[ch_idx] = np.zeros(mask_shape, dtype=bool)
            print(f"Warning: Channel {ch_idx} has insufficient data for comparison")
    
    return channel_masks, channel_t_values

def compute_sig_tfr_masks_from_concatenated_data(
    concatenated_data, labels, train_indices, condition_names, cats,
    obs_axs, chans_axs, stat_func, p_thresh, n_perm, 
    ignore_adjacency=1, seed=42, tails=2
):
    """
    Compute significant TFR clusters using only training trials from concatenated data.
    
    Parameters
    ----------
    concatenated_data : np.ndarray
        Full data array (all_trials, channels, freqs, times)
    labels : np.ndarray
        Labels for all trials in concatenated_data
    train_indices : np.ndarray
        Indices of training trials to use for computing masks
    condition_names : list
        List of condition names to compare
    cats : dict
        Dictionary mapping condition names to label integers
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    stat_func : callable
        Statistical function for cluster computation
    p_thresh : float
        P-value threshold for significance
    n_perm : int
        Number of permutations
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed
    tails : int
        Number of tails for statistical test
        
    Returns
    -------
    channel_masks : dict
        Dictionary where keys are channel indices and values are boolean masks
        of shape (n_freqs, n_times) indicating significant clusters
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). 
        THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    # Validate we have exactly 2 conditions for now
    if len(condition_names) != 2:
        raise ValueError(
            f"For now, just doing perm test instead of ANOVA, "
            f"so this will only work for two conditions. Got {len(condition_names)} conditions."
        )
    
    # Get training data and labels
    train_data = concatenated_data[train_indices]
    train_labels = labels[train_indices]
    
    # Split training data by condition
    train_data_by_condition = {}
    for cond_name in condition_names:
        # Get the label value for this condition
        cond_label = cats[tuple([cond_name]) if isinstance(cond_name, str) else tuple(cond_name)]
        
        # Get indices for this condition
        cond_mask = train_labels == cond_label
        cond_data = train_data[cond_mask]
        train_data_by_condition[cond_name] = cond_data
    
    # Compute significant clusters for each channel
    n_channels = concatenated_data.shape[chans_axs]
    channel_masks, channel_t_values = compute_sig_tfr_masks_for_specified_channels(
        n_channels, train_data_by_condition, condition_names, 
        obs_axs, chans_axs, stat_func, p_thresh, n_perm,
        ignore_adjacency, seed, tails
    )
    
    return channel_masks, channel_t_values

def apply_tfr_masks_and_flatten_to_make_decoding_matrix(data, obs_axs, chans_axs, channel_masks):
    """
    Apply channel-specific TFR masks and flatten feature matrices.
    
    Parameters
    ----------
    data : np.ndarray
        Shape: (n_trials, n_channels, n_freqs, n_times)
    obs_axs : int
        Axis containing trials
    chans_axs : int
        Axis containing channels
    channel_masks : dict
        Dictionary where keys are channel indices and values are boolean masks
        
    Returns
    -------
    decoding_matrix : np.ndarray
        Shape: (n_trials, n_features) where n_features depends on the masks
    """
    n_trials = data.shape[obs_axs]
    n_channels = data.shape[chans_axs]
    feature_vectors = []
    
    # Move trials to first axis if needed
    if obs_axs != 0:
        data = np.moveaxis(data, obs_axs, 0)
        if chans_axs > obs_axs:
            chans_axs = chans_axs - 1
        else:
            chans_axs = chans_axs + 1

    # Iterate through each channel
    for ch_idx in range(n_channels):
        # Extract this channel's data for all trials
        channel_data = np.take(data, ch_idx, axis=chans_axs)
        
        # Check if we have a mask for this channel
        if ch_idx in channel_masks:
            # Get the boolean mask (n_freqs, n_times)
            mask = channel_masks[ch_idx]
            
            # flatten all dimensions except trials (axis 0)
            n_trials_ch = channel_data.shape[0]
            remaining_shape = channel_data.shape[1:]
            channel_data = channel_data.reshape(n_trials_ch, -1)
            
            # flatten mask
            mask_flat = mask.flatten()

            # make sure the mask size matches the flattened features
            if mask_flat.shape[0] != channel_data.shape[1]:
                raise ValueError("Mask size does not match flattened features size")
            else:
                # apply the mask
                masked_features = channel_data[:, mask_flat]
        
            # Add this channel's features to our list
            feature_vectors.append(masked_features)
    
    # Concatenate all channels' features horizontally
    if feature_vectors:
        decoding_matrix = np.concatenate(feature_vectors, axis=1)
    else:
        # Return empty matrix if no features
        raise ValueError("No features found for any channels.")

    return decoding_matrix

def get_confusion_matrix_for_rois_tfr_cluster(
    roi_labeled_arrays, rois, strings_to_find, stat_func, 
    Decoder, explained_variance=0.95,
    p_thresh=0.05, n_perm=100, 
    n_splits=5, n_repeats=5, obs_axs=0, chans_axs=1,
    balance_method='subsample', oversample=False,
    random_state=42, alpha=0.2, ignore_adjacency=1, seed=42, tails=2, normalize: str = None, clear_memory=True
):
    """
    Compute confusion matrices using TFR cluster masking for multiple ROIs. Also returns the sig tfr cluster masks for later plotting.
    
    Parameters
    ----------
    roi_labeled_arrays : dict
        Dictionary of labeled arrays by ROI
    rois : list
        List of ROIs to process
    strings_to_find : list
        List of condition strings to find
    stat_func : callable
        Statistical function for cluster computation
    Decoder : class
        Decoder class to use
    explained_variance : float
        Variance to explain in PCA
    p_thresh : float
        P-value threshold for clusters
    n_perm : int
        Number of permutations
    n_splits : int
        Number of CV splits
    n_repeats : int
        Number of CV repeats
    obs_axs : int
        Observation axis
    chans_axs : int
        Channel axis
    balance_method : str
        Method for balancing ('subsample' or 'pad_with_nans')
    oversample : bool
        Whether to oversample in decoder
    random_state : int
        Random seed
    alpha : float
        Mixup alpha parameter
    ignore_adjacency : int
        Whether to ignore adjacency in clustering
    seed : int
        Random seed for permutation test
    tails : int
        Number of tails for permutation test
    normalize : str
        Whether to normalize the confusion matrix
    Returns
    -------
    confusion_matrices : dict
        Dictionary of confusion matrices by ROI
    cats_dict : dict
        Dictionary of condition labels by ROI
    channel_masks : dict
        Dictionary of channel masks for significant clusters. Nested dictionary: {roi: {repeat: {fold: channel_masks}}}
    channel_t_values : dict
        Dictionary where keys are channel indices (int) and values are t values of shape (n_freqs, n_times). 
        THIS ONLY WORKS IF USING SCIPY STATS TTEST IND.
    """
    confusion_matrices = {}
    cats_dict = {}
    channel_masks = {}
    channel_t_values = {}
    
    for roi in rois:
        channel_masks[roi] = {}
        channel_t_values[roi] = {}
        print(f"Processing ROI: {roi}")
        
        # Get data and labels
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
        )
        cats_dict[roi] = cats
        
        # Set up cross-validation
        all_cms = []
        
        for repeat in range(n_repeats):
            channel_masks[roi][repeat] = {}
            channel_t_values[roi][repeat] = {}
            repeat_seed = random_state + repeat * 1000
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            
            fold_cms = []
            
            for fold_idx, (train_indices, test_indices) in enumerate(skf.split(concatenated_data, labels)):
                print(f"  Repeat {repeat+1}/{n_repeats}, Fold {fold_idx+1}/{n_splits}")
                
                # Get train/test data
                X_train_raw = concatenated_data[train_indices]
                X_test_raw = concatenated_data[test_indices]
                y_train = labels[train_indices]
                y_test = labels[test_indices]
                
                # Balance and decode with TFR masking
                preds, fold_channel_masks, fold_channel_t_values = decode_on_sig_tfr_clusters(
                    X_train_raw, y_train, X_test_raw,
                    train_indices, test_indices,
                    concatenated_data, labels, cats,
                    obs_axs, chans_axs,
                    stat_func, p_thresh, n_perm,
                    Decoder, explained_variance, oversample,
                    ignore_adjacency=ignore_adjacency, 
                    seed=repeat_seed + fold_idx, 
                    tails=tails, 
                    alpha=alpha
                )
                
                channel_masks[roi][repeat][fold_idx] = fold_channel_masks
                channel_t_values[roi][repeat][fold_idx] = fold_channel_t_values
                cm = confusion_matrix(y_test, preds)
                fold_cms.append(cm)

                if clear_memory:
                    del X_train_raw, X_test_raw, y_train, y_test, preds, fold_channel_masks, fold_channel_t_values
                    gc.collect()
            
            # Sum across folds
            repeat_cm = np.sum(fold_cms, axis=0)

            # Normalize the confusion matrix for this repeat if requested
            if normalize:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if normalize == 'true':
                        divisor = np.sum(repeat_cm, axis=-1, keepdims=True)
                    elif normalize == 'pred':
                        divisor = np.sum(repeat_cm, axis=-2, keepdims=True)
                    elif normalize == 'all':
                        divisor = np.sum(repeat_cm)
                    else:
                        divisor = 1
                    
                    # Avoid division by zero by setting the divisor to 1 where it is 0
                    if isinstance(divisor, np.ndarray):
                        divisor[divisor == 0] = 1
                    elif divisor == 0:
                        divisor = 1
                    
                    normalized_cm = repeat_cm.astype('float') / divisor
                    all_cms.append(normalized_cm)
            else:
                all_cms.append(repeat_cm) # Append the raw counts if no normalization
        
        # Average across repeats
        final_cm = np.mean(all_cms, axis=0)
        confusion_matrices[roi] = final_cm

        if clear_memory and roi in roi_labeled_arrays:
            del roi_labeled_arrays[roi]
            gc.collect()

        if clear_memory:
            del concatenated_data, labels, cats, all_cms
            gc.collect()
    
    return confusion_matrices, cats_dict, channel_masks, channel_t_values

def get_display_labels_from_cats(cats):
    """Extracts clean labels for plotting from the 'cats' dictionary."""
    return [key[0] if isinstance(key, tuple) and len(key) == 1 else str(key) for key in cats.keys()]

def plot_and_save_confusion_matrix(cm, display_labels, file_name, save_dir):
    """
    Plots and saves a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name))
    print(f"Saved figure to: {save_dir}")
    plt.close(fig)

def plot_and_save_tfr_masks(masks_dict, mask_type, subjects_or_rois, ch_names, times, freqs, 
                            spec_method, conditions_save_name, save_dir, 
                            channels_per_page=60, grid_shape=(6, 10)):
    """
    Plot and save TFR masks for subjects or ROIs.
    
    Parameters
    ----------
    masks_dict : dict
        Dictionary of masks (subjects or ROIs as keys)
    mask_type : str
        Type of mask ('sig_elecs' or 'all_elecs')
    subjects_or_rois : list
        List of subjects or ROIs to plot
    ch_names : list
        Channel names
    times : array
        Time points
    freqs : array
        Frequencies
    spec_method : str
        Spectral method used
    conditions_save_name : str
        Name for saving
    save_dir : str
        Directory to save figures
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for key in subjects_or_rois:
        if key not in masks_dict:
            continue
            
        mask = masks_dict[key]
        mask_pages = plot_mask_pages(
            mask,
            ch_names,
            times=times,
            freqs=freqs,
            channels_per_page=channels_per_page,
            grid_shape=grid_shape,
            cmap=parula_map,
            title_prefix=f"{key} ",
            log_freq=True,
            show=False
        )
        
        # Save each page
        for i, fig in enumerate(mask_pages):
            fig_name = f"{key}_{mask_type}_{spec_method}_clusters_{conditions_save_name}_page_{i+1}.png"
            fig_pathname = os.path.join(save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            print(f"Saved figure: {fig_name}")