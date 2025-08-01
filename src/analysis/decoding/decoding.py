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

# rsatoolbox imports
import rsatoolbox # Import the base module
from rsatoolbox.io.mne import read_epochs
from rsatoolbox.data.ops import merge_datasets
from rsatoolbox.rdm import calc_rdm_movie, compare # compare was imported separately
from rsatoolbox.rdm.calc import _parse_input # Consider if this private member access is necessary
from rsatoolbox.util.build_rdm import _build_rdms # Consider if this private member access is necessary
from rsatoolbox.vis import show_rdm
from rsatoolbox.vis.timecourse import plot_timecourse

# Other third-party
from tqdm import tqdm # Was imported as numpy, tqdm, mne, pandas - corrected
from numpy.lib.stride_tricks import as_strided, sliding_window_view # sliding_window_view was imported separately

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

# ---- Potentially Unused or Redundant ----
# from pandas import read_csv # pandas is already imported as pd, use pd.read_csv
# from os.path import join, expanduser, basename # os.path is available via 'import os'
# from joblib import Parallel, delayed # joblib is already imported

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
    def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray,
                normalize: str = None, obs_axs: int = -2, time_axs: int = -1, n_jobs: int = 1,
                window: int = None, step_size: int = 1,
                    shuffle: bool = False, oversample: bool = True) -> np.ndarray:
        """Cross-validated confusion matrix with windowing and optional shuffling."""
        n_cats = len(set(labels))
        time_axs_positive = time_axs % x_data.ndim
        out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)

        if window is not None:
            # Include the step size in the windowed output shape
            steps = (x_data.shape[time_axs_positive] - window) // step_size + 1
            out_shape = (steps,) + out_shape
                
        mats = np.zeros(out_shape, dtype=np.uint8)
        data = x_data.swapaxes(0, obs_axs)

        if shuffle:
            # shuffled label pool
            label_stack = []
            for i in range(self.n_repeats):
                label_stack.append(labels.copy())
                self.shuffle_labels(data, label_stack[-1], 0)

            # build the test/train indices from the shuffled labels for each
            # repetition, then chain together the repetitions
            # splits = (train, test)

            print("Shuffle validation:")
            for i, labels in enumerate(label_stack):
                # Compare with the first repetition to ensure variety in shuffles
                if i > 0:
                    diff = np.sum(label_stack[0] != labels)

            idxs = ((self.split(data, l), l) for l in label_stack)
            idxs = ((itertools.islice(s, self.n_splits),
                     itertools.repeat(l, self.n_splits))
                    for s, l in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)

        else:
            idxs = ((splits, labels) for splits in self.split(data, labels))

        # 11/1 below is aaron's code for windowing. 
        def proc(train_idx, test_idx, l):
            x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
            print(f"x_stacked shape: {x_stacked.shape}")

            # Use the updated windower function with step_size
            windowed = windower(x_stacked, window, axis=time_axs, step_size=step_size)
            print(f"windowed shape: {windowed.shape}")

            out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
            for i, x_window in enumerate(windowed):
                x_flat = x_window.reshape(x_window.shape[0], -1)
                x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
                out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
            return out

        # # loop over folds and repetitions
        if n_jobs == 1:
            idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
            results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
        else:
            results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
                delayed(proc)(train_idx, test_idx, l)
                for (train_idx, test_idx), l in idxs)

        # # Collect the results
        for i, result in enumerate(results):
            rep, fold = divmod(i, self.n_splits)
            mats[:, rep, fold] = result

        # normalize, sum the folds
        mats = np.sum(mats, axis=-3)
        if normalize == 'true':
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return mats / divisor
        
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

    # Combine train and test indices
    idx_stacked = np.concatenate((train_idx, test_idx))
    x_stacked = np.take(x_data, idx_stacked, axis)
    y_stacked = labels[idx_stacked]

    # Split into training and testing sets
    sep = train_idx.shape[0]
    y_train, y_test = np.split(y_stacked, [sep])
    x_train, x_test = np.split(x_stacked, [sep], axis=axis)

    if oversample:
        # Apply mixup2 to x_train
        mixup2(arr=x_train, labels=y_train, obs_axs=axis, alpha=1., seed=None)

    # Fill in test data nans with noise from distribution
    is_nan = np.isnan(x_test)
    x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

    # Recombine x_train and x_test
    x_stacked = np.concatenate((x_train, x_test), axis=axis)

    return x_stacked, y_train, y_test

def get_and_plot_confusion_matrix_for_rois_jim(
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, save_dir,
    time_interval_name=None, other_string_to_add=None, n_splits=5, n_repeats=5, obs_axs=0, balance_method='pad_with_nans', random_state=42,
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
    - balance_method: 'pad_with_nans' or 'subsample' to balance trial counts between conditions.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - confusion_matrices: Dictionary containing confusion matrices for each ROI.
    """
    confusion_matrices = {}
    rng = np.random.RandomState(random_state)

    for roi in rois:
        print(f"Processing ROI: {roi}")
        concatenated_data, labels, cats = process_and_balance_data_for_decoding(
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, random_state
        )

        # Create a Decoder and run cross-validation
        decoder = Decoder(cats, 0.80, oversample=True, n_splits=n_splits, n_repeats=n_repeats)

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
        file_name = (
            f'{roi}_{condition_comparison}{time_str}{other_str}_time_averaged_confusion_matrix_'
            f'{n_splits}splits_{n_repeats}repeats_{balance_method}.png'
        )
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    return confusion_matrices

# TODO: Clean this up.   
# Make subfunctions to break this down. Everything before defining the Decoder objects can be a function, that can be shared between this and the whole time window version.   
# The decoder true and decoder shuffle can be done with a function maybe.   
# And maybe return just accuracies, which I can then call this entire function separately for true and shuffled.
# ALSO STORE THE SHUFFLED OUTPUT IN A NUMPY ARRAY SO I DON'T HAVE TO MAKE IT EVERY TIME
def get_confusion_matrices_for_rois_time_window_decoding_jim(
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, n_splits=5, n_repeats=5, obs_axs=0, time_axs=-1,
    balance_method='pad_with_nans', random_state=42, window_size=None,
    step_size=1, n_permutations=100, sampling_rate=256, first_time_point=-1
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
    random_state : int, optional
        Seed for the random number generator for reproducibility. Default is 42.
    window_size : int, optional
        The number of time samples in each sliding window. If None, the entire
        time axis length is used (i.e., no sliding window). Default is None.
    step_size : int, optional
        The number of time samples to slide the window by. Default is 1.
    n_permutations : int, optional
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
                  Shape: (n_windows, n_permutations, n_classes, n_classes).
                - (other keys are the same as in `cm_true_per_roi[roi]`)
    """
    # Initialize dictionaries to store confusion matrices for each ROI
    cm_true_per_roi = {}
    cm_shuffle_per_roi = {}
    rng = np.random.RandomState(random_state)
    first_sample = first_time_point * sampling_rate

    for roi in rois:
        print(f"Processing ROI: {roi}")

        concatenated_data, labels, cats = process_and_balance_data_for_decoding(
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
        decoder_true = Decoder(cats, 0.80, oversample=True, n_splits=n_splits, n_repeats=n_repeats)
        decoder_shuffle = Decoder(cats, 0.80, oversample=True, n_splits=n_splits, n_repeats=n_permutations)

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
            'cm_shuffle': cm_shuffle,  # Shape: (n_windows, n_permutations, n_classes, n_classes)
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
        Expected shape: (n_windows, n_permutations, n_classes, n_classes).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - accuracies_true : numpy.ndarray
            Accuracies for true labels. Shape: (n_windows, n_repeats).
        - accuracies_shuffle : numpy.ndarray
            Accuracies for shuffled labels. Shape: (n_windows, n_permutations).
    """
    n_windows = cm_true.shape[0]
    n_repeats = cm_true.shape[1]
    n_permutations = cm_shuffle.shape[1]

    accuracies_true = np.zeros((n_windows, n_repeats))
    accuracies_shuffle = np.zeros((n_windows, n_permutations))

    for win_idx in range(n_windows):
        # True accuracies
        for rep_idx in range(n_repeats):
            cm = cm_true[win_idx, rep_idx]
            accuracies_true[win_idx, rep_idx] = np.trace(cm) / np.sum(cm)
        # Shuffled accuracies
        for perm_idx in range(n_permutations):
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
        Accuracies for shuffled labels. Expected shape: (n_windows, n_permutations).
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
        stat_func=lambda x, y, axis: np.mean(x, axis=axis),
        n_jobs=1,
        seed=seed
    )
    return significant_clusters, p_values

def plot_accuracies(time_points, accuracies_true, accuracies_shuffle, significant_clusters,
                    window_size, step_size, sampling_rate, condition_comparison, roi, save_dir):
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
        Accuracies for shuffled labels. Shape: (n_windows, n_permutations).
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
    """
    n_repeats = accuracies_true.shape[1]
    n_permutations = accuracies_shuffle.shape[1]

    # Compute mean and standard error
    mean_true_accuracy = np.mean(accuracies_true, axis=1)
    std_true_accuracy = np.std(accuracies_true, axis=1)
    se_true_accuracy = std_true_accuracy / np.sqrt(n_repeats)

    mean_shuffle_accuracy = np.mean(accuracies_shuffle, axis=1)
    std_shuffle_accuracy = np.std(accuracies_shuffle, axis=1)
    se_shuffle_accuracy = std_shuffle_accuracy / np.sqrt(n_permutations)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, mean_true_accuracy, label='True Accuracy', color='blue')
    plt.fill_between(
        time_points,
        mean_true_accuracy - std_true_accuracy,
        mean_true_accuracy + std_true_accuracy,
        alpha=0.2,
        color='blue'
    )

    plt.plot(time_points, mean_shuffle_accuracy, label='Shuffled Accuracy', color='red')
    plt.fill_between(
        time_points,
        mean_shuffle_accuracy - std_shuffle_accuracy,
        mean_shuffle_accuracy + std_shuffle_accuracy,
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

    # Construct the filename
    filename = f"{condition_comparison}_ROI_{roi}_window{window_size}_step{step_size}.png"
    filepath = os.path.join(save_dir, filename)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Save and close the plot
    plt.savefig(filepath)
    plt.close()