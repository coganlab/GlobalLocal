"""Per-ROI confusion-matrix orchestration (time-window decoding)."""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .data_prep import concatenate_and_balance_data_for_decoding
from .decoder import Decoder

def get_and_plot_confusion_matrix_for_rois_jim(
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, save_dir,
    time_interval_name=None, other_string_to_add=None, n_splits=5, n_repeats=5, obs_axs=0, balance_method='pad_with_nans', explained_variance=0.8, balance_strata=False, random_state=42, timestamp=None
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
    - balance_strata (bool) : 
        If True, when a class (string_group) matches multiple sub-conditions (e.g., the same congruency drawn from different blocks), 
        subsample each matched sub-condition down to the minimum count before concatenating. This prevents block imbalance from biasing the class.
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
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, balance_strata=balance_strata, random_state=random_state
        )

        # Create a Decoder and run cross-validation
        decoder = Decoder(cats,                 # classes mapped to their class numbers. E.g., {('c25',): 0, ('i25',): 1}
                          explained_variance,   # PCA: keep enough components for this much variance (usually 90%)
                          oversample=True,      # apply mixup to train NaNs
                          n_splits=n_splits,    # k in k-fold
                          n_repeats=n_repeats,  # how many times to redo k-fold with different splits
                          random_state=random_state) 

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
    roi_labeled_arrays, rois, condition_comparison, strings_to_find, clf=None, 
    n_splits=5, n_repeats=5, obs_axs=0, time_axs=-1,
    balance_method='pad_with_nans', explained_variance=0.8, random_state=42, window_size=None,
    step_size=1, n_perm=100, sampling_rate=256, first_time_point=-1, folds_as_samples: bool = False,
    balance_strata=True, random_seed=42, feature_mode='channels_x_time'
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
    folds_as_samples : bool, optional
        Whether to use the folds (splits) as the unit to be shuffled across for time perm cluster. Default is false and to sum across splits within repeats, and use repeats as the unit to be shuffled across instead.
    balance_strata : bool, optional
        If True, when a class (string_group) matches multiple sub-conditions (e.g., the same congruency drawn from different blocks), 
        subsample each matched sub-condition down to the minimum count before concatenating. This prevents block imbalance from biasing the class.
    random_state : int or RandomState, optional
        Used only when balance_strata=True
        
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
            roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, balance_strata=balance_strata, random_state=random_state
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
        decoder_true = Decoder(cats, explained_variance, oversample=True, clf=clf, n_splits=n_splits, n_repeats=n_repeats, clf_params={}, random_state=random_state, feature_mode=feature_mode)
        decoder_shuffle = Decoder(cats, explained_variance, oversample=True, clf=clf, n_splits=n_splits, n_repeats=n_perm, clf_params={}, random_state=random_state, feature_mode=feature_mode)

        # Run decoding with true labels
        cm_true = decoder_true.cv_cm_jim_window_shuffle(
            concatenated_data, labels, normalize=None, obs_axs=obs_axs, time_axs=time_axs,
            window=effective_window_size, step_size=effective_step_size, shuffle=False, folds_as_samples=folds_as_samples
        )

        # Run decoding with shuffled labels
        cm_shuffle = decoder_shuffle.cv_cm_jim_window_shuffle(
            concatenated_data, labels, normalize=None, obs_axs=obs_axs, time_axs=time_axs,
            window=effective_window_size, step_size=effective_step_size, shuffle=True, folds_as_samples=folds_as_samples
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
