"""Data preparation for decoding: balancing, mixup augmentation, feature
flattening and fold sampling."""

import numpy as np
from src.analysis.utils.labeled_array_utils import (
    put_data_in_labeled_array_per_roi_subject,
    remove_nans_from_labeled_array,
    remove_nans_from_all_roi_labeled_arrays,
    concatenate_conditions_by_string,
    get_data_in_time_range,
    gather_class_data_by_stratum
)

def concatenate_and_balance_data_for_decoding(
    roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method,
    balance_strata=True, random_state=42
):
    """
    Build a decoding-ready (X, labels, cats) from multi-stratum class definitions.

    Order of operations:
      1. Gather raw data per class per sub-condition (stratum).
      2. Drop NaN-containing trials per stratum.
         (This reveals the true, unpadded trial counts.)
      3. If balance_strata=True, subsample each stratum within a class down to
         the min count across that class's strata.
      4. Subsample each class down to the min count across classes
         (balance_method='subsample') or pad the shorter class with NaNs
         (balance_method='pad_with_nans').
      5. Concatenate into final (X, labels) arrays.
    """
    rng = (random_state if isinstance(random_state, np.random.RandomState)
           else np.random.RandomState(random_state))

    # --- Step 1: gather raw data grouped by class -> stratum ---
    class_data, cats = gather_class_data_by_stratum(
        roi_labeled_arrays, roi, strings_to_find, obs_axs
    )

    print(f"\n{'='*60}\nROI: {roi}\n{'='*60}")

    # --- Step 2: drop NaN trials per stratum ---
    clean_class_data = []   # list of dicts, same structure as class_data
    for class_idx, strata_dict in enumerate(class_data):
        clean_strata = {}
        for cond, data in strata_dict.items():
            # "trial has NaN" = any NaN anywhere in that trial's data
            nan_trial_mask = np.isnan(data).any(
                axis=tuple(range(1, data.ndim)) if obs_axs == 0
                else tuple(i for i in range(data.ndim) if i != obs_axs)
            )
            valid_mask = ~nan_trial_mask
            n_before = data.shape[obs_axs]
            n_after = int(valid_mask.sum())
            clean_strata[cond] = np.take(data, np.where(valid_mask)[0], axis=obs_axs)
            print(f"  [NaN filter] class={class_idx} stratum={cond}: "
                  f"{n_before} → {n_after} trials "
                  f"({100*(n_before-n_after)/max(n_before,1):.1f}% dropped)")
        clean_class_data.append(clean_strata)

    # --- Step 3: balance strata within each class ---
    if balance_strata:
        for class_idx, strata_dict in enumerate(clean_class_data):
            if len(strata_dict) <= 1:
                continue
            sizes = {cond: d.shape[obs_axs] for cond, d in strata_dict.items()}
            n_per_stratum = min(sizes.values())
            if n_per_stratum == 0:
                raise ValueError(
                    f"Class {class_idx} has a stratum with 0 valid trials after NaN removal: {sizes}"
                )
            print(f"  [balance_strata post-NaN] class={class_idx}: {sizes} → {n_per_stratum} per stratum")
            for cond, data in strata_dict.items():
                if data.shape[obs_axs] > n_per_stratum:
                    idx = rng.choice(data.shape[obs_axs], size=n_per_stratum, replace=False)
                    strata_dict[cond] = np.take(data, idx, axis=obs_axs)

    # --- Step 4 prep: get per-class totals now that strata are balanced ---
    class_totals = [
        sum(d.shape[obs_axs] for d in strata_dict.values())
        for strata_dict in clean_class_data
    ]
    print(f"  [class totals after stratum balancing] {dict(enumerate(class_totals))}")

    # --- Step 4: balance across classes ---
    if balance_method == 'subsample':
        target_n = min(class_totals)
        print(f"  [balance_method=subsample] subsampling each class to {target_n}")

        final_per_class = []
        for class_idx, strata_dict in enumerate(clean_class_data):
            # Concatenate strata within this class, then subsample if needed
            class_arr = np.concatenate(list(strata_dict.values()), axis=obs_axs)
            if class_arr.shape[obs_axs] > target_n:
                # IMPORTANT: if balance_strata is True and we subsample here,
                # we may re-unbalance the strata. But at this point we've already
                # equalized strata, so a uniform random draw preserves the
                # expected proportions. If you want strict preservation,
                # subsample per-stratum proportionally instead.
                idx = rng.choice(class_arr.shape[obs_axs], size=target_n, replace=False)
                class_arr = np.take(class_arr, idx, axis=obs_axs)
            final_per_class.append(class_arr)

    elif balance_method == 'pad_with_nans':
        target_n = max(class_totals)
        print(f"  [balance_method=pad_with_nans] padding each class to {target_n}")
        final_per_class = []
        for class_idx, strata_dict in enumerate(clean_class_data):
            class_arr = np.concatenate(list(strata_dict.values()), axis=obs_axs)
            n_have = class_arr.shape[obs_axs]
            if n_have < target_n:
                pad_shape = list(class_arr.shape)
                pad_shape[obs_axs] = target_n - n_have
                pad = np.full(pad_shape, np.nan)
                class_arr = np.concatenate([class_arr, pad], axis=obs_axs)
            final_per_class.append(class_arr)
    else:
        raise ValueError(f"unknown balance_method: {balance_method}")

    # --- Step 5: final concatenation ---
    final_data = np.concatenate(final_per_class, axis=obs_axs)
    labels = np.concatenate([
        np.full(arr.shape[obs_axs], class_idx, dtype=int)
        for class_idx, arr in enumerate(final_per_class)
    ])

    print(f"  [final] data shape: {final_data.shape}, labels: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"{'='*60}\n")

    return final_data, labels, cats


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


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)


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
    
    # Step 7: Handle incomplete training pseudo-trials.
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
    else:
        # Cross-decoding chooses subsampling rather than synthesizing training
        # data. First remove every incomplete pseudo-trial, then keep an equal
        # number from each class so missingness cannot change the class prior.
        other = tuple(i for i in range(x_train.ndim) if i != axis)
        complete = ~np.isnan(x_train).any(axis=other)
        x_train = np.compress(complete, x_train, axis=axis)
        y_train = y_train[complete]
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) < 2 or counts.min() == 0:
            raise ValueError("training fold has fewer than two complete classes after "
                             "subsampling incomplete pseudo-trials")
        n_per_class = counts.min()
        # Deterministic selection preserves the guarantee that within-level and
        # transfer calls made with the same folds train identical classifiers.
        keep = np.sort(np.concatenate([
            np.flatnonzero(y_train == label)[:n_per_class] for label in classes
        ]))
        x_train = np.take(x_train, keep, axis=axis)
        y_train = y_train[keep]
    
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
