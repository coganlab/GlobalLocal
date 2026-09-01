"""Statistics on decoding accuracies: permutation / bootstrap cluster tests,
pooled shuffle distributions and paired comparisons."""

import numpy as np
from mne.stats import permutation_cluster_1samp_test
import scipy.stats as stats
from scipy.ndimage import label
from scipy.stats import norm, t
from joblib import Parallel, delayed
from typing import Union, List, Sequence
from ieeg.calc.stats import time_perm_cluster
from typing import Dict, Optional, Union, List, Tuple

from .data_prep import concatenate_and_balance_data_for_decoding
from .decoder import Decoder

def compute_accuracies(cm_true, cm_shuffle):
    """
    Compute accuracies from true and shuffled confusion matrices.

    This function calculates the accuracy for each window and fold or repetition/permutation
    by taking the trace of the confusion matrix (sum of true positives) and
    dividing by the total sum of the matrix (total number of instances).

    Parameters
    ----------
    cm_true : numpy.ndarray
        Confusion matrices for true labels.
        Expected shape: (n_windows, n_repeats or n_folds, n_classes, n_classes).
    cm_shuffle : numpy.ndarray
        Confusion matrices for shuffled labels.
        Expected shape: (n_windows, n_perm or n_folds, n_classes, n_classes).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - accuracies_true : numpy.ndarray
            Accuracies for true labels. Shape: (n_windows, n_repeats or n_folds).
        - accuracies_shuffle : numpy.ndarray
            Accuracies for shuffled labels. Shape: (n_windows, n_perm or n_folds).
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


def perform_time_perm_cluster_test_for_accuracies(accuracies_true, accuracies_shuffle, p_thresh=0.05, n_perm=50, seed=42, stat_func=None):
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
    stat_func : callable, optional
        Per-window statistic passed to `time_perm_cluster`. None (default) uses
        the library's own default (`ieeg.calc.stats.ttest`).

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

    # `stat_func` used to be referenced here without ever being defined or passed,
    # so every call raised NameError. It is now an optional argument; omitting it
    # lets `time_perm_cluster` apply its own default rather than inventing one.
    optional = {} if stat_func is None else {'stat_func': stat_func}

    significant_clusters, p_values = time_perm_cluster(
        accuracies_true_T,
        accuracies_shuffle_T,
        p_thresh=p_thresh,
        n_perm=n_perm,
        tails=1,
        axis=0,
        n_jobs=1,
        seed=seed,
        **optional
    )
    return significant_clusters, p_values


def find_contiguous_clusters(significant_clusters: Union[np.ndarray, List[bool]]) -> List[Tuple[int, int]]:
    """
    Find start and end indices of contiguous True blocks.
    
    Parameters
    ----------
    significant_clusters : array-like of bool
        Boolean array indicating significant time points.
        
    Returns
    -------
    clusters : List[Tuple[int, int]]
        List of (start_idx, end_idx) tuples.
    """
    clusters = []
    in_cluster = False
    
    for idx, val in enumerate(list(significant_clusters)):
        if val and not in_cluster:
            start_idx = idx
            in_cluster = True
        elif not val and in_cluster:
            end_idx = idx - 1
            clusters.append((start_idx, end_idx))
            in_cluster = False
    
    if in_cluster:
        end_idx = len(list(significant_clusters)) - 1
        clusters.append((start_idx, end_idx))
    
    return clusters


def make_pooled_shuffle_distribution(
    roi: str,
    roi_labeled_arrays: dict,
    strings_to_find_pooled: list,
    explained_variance: float,
    n_splits: int,
    n_perm: int,
    random_state: int,
    balance_method: str,
    obs_axs: int,
    window_size: int,
    step_size: int
) -> np.ndarray:
    """
    Generates a pooled shuffle distribution for a given ROI.

    This function pools trials from multiple conditions, balances the dataset,
    and then performs time-windowed decoding with shuffled labels to create
    a null distribution of decoding accuracies.

    Parameters
    ----------
    roi : str
        The Region of Interest (ROI) to process.
    roi_labeled_arrays : dict
        A dictionary of LabeledArray objects containing the epoched data.
    strings_to_find_pooled : list
        A list of lists defining how to pool conditions into classes.
        (e.g., [['c25', 'c75'], ['i25', 'i75']]).
    explained_variance : float
        The proportion of variance for the PCA to explain.
    n_splits : int
        The number of splits for cross-validation.
    n_perm : int
        The number of permutations (shuffles) to run.
    random_state : int
        The random seed for reproducibility.
    balance_method : str
        The method to balance trial counts ('subsample' or 'pad_with_nans').
    obs_axs : int
        The axis in the data array corresponding to observations (trials).
    window_size : int
        The number of time samples in each sliding window.
    step_size : int
        The number of time samples to slide the window by.

    Returns
    -------
    np.ndarray
        An array of shuffled accuracies with shape (n_windows, n_permutations).
    """
    print(f"Generating pooled shuffle distribution for ROI: {roi}...")

    # 1. Use existing function to create a balanced, pooled dataset
    x_pooled, y_pooled, cats_pooled = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays,
        roi,
        strings_to_find_pooled,
        obs_axs=obs_axs,
        balance_method=balance_method,
        random_state=random_state
    )

    # 2. Instantiate a decoder for the shuffle permutation
    decoder_shuffle_pooled = Decoder(
        cats_pooled,
        explained_variance=explained_variance,
        oversample=True,
        n_splits=n_splits,
        n_repeats=n_perm,  # Use n_perm for repetitions
        random_state=random_state
    )

    # 3. Run the time-windowed decoding with shuffle=True
    cm_shuffle_pooled = decoder_shuffle_pooled.cv_cm_jim_window_shuffle(
        x_pooled,
        y_pooled,
        normalize=None, # FIXED: normalize should be None for shuffle 2/26/26.
        obs_axs=obs_axs,
        time_axs=-1,
        window=window_size,
        step_size=step_size,
        shuffle=True
    )

    # 4. Compute accuracies from the shuffled confusion matrices
    _, accuracies_shuffle_pooled = compute_accuracies(cm_shuffle_pooled, cm_shuffle_pooled)

    return accuracies_shuffle_pooled


def find_significant_clusters_of_series_vs_distribution_based_on_percentile(
    series, distribution, time_points,
    percentile=95,
    cluster_percentile=95, n_cluster_perms=1000,
    random_state=None
):
    """
    Find significant clusters of a single time-series (i.e., mean true decoding accuracies over time) vs. a distribution of time-series (i.e., distribution of shuffled label decoding accuracies over time).
    Check if min_cluster_size is necessary! I think any two consecutive significant time points should be counted as a cluster.
    
    Parameters
    ----------
    series : np.ndarray
        Shape: (1, indices) - A series of values, of length indices (make this 2D though so can compare with the distribution). Example is a time series of the mean true decoding accuracies.
    distribution : np.ndarray
        Shape: (n, indices) - 2D distribution of series. Example is a distribution of shuffled decoding accuracies. TIMES MUST MATCH BETWEEN THE SERIES AND THE DISTRIBUTION.
    significant_percentile : float
        Percentile of the distribution that a value from the series must be to be classified as significant (before cluster correction). For example, a true decoding accuracy timepoint must be in the 95th percentile of the shuffled decoding accuracy distribution.
    cluster_percentile : float
        Percentile of the cluster size distribution that a significant cluster must be to survive cluster correction. For example, if a cluster's size is in the 95th percentile of all cluster sizes.
    n_cluster_perms : int
        Number of permutations to be done to form a null distribution of "significant" cluster sizes.
         
    Returns
    -------
    significant_clusters : list of tuples
        List of (start_idx, end_idx) for significant clusters of consecutive values in the series that are greater than the significant percentile of the distribution.
    """
    
    rng = np.random.RandomState(random_state)
    times = len(series)
    
    # Step 1: Find the percentile threshold values for the distribution
    pointwise_distribution_threshold = np.percentile(distribution, percentile, axis=0)
    
    # Step 2: Find true significant clusters in which consecutive points in the series are above the percentile threshold values of the distribution
    true_sig_mask = (series > pointwise_distribution_threshold).squeeze()
    true_clusters = find_contiguous_clusters(true_sig_mask)
    
    # Step 3: Find true cluster lengths (in time, this is what we'll use as our measure of cluster size)
    true_cluster_lengths = find_cluster_lengths(true_clusters)
    
    # Step 4: Make null distribution of maximum cluster lengths
    max_perm_cluster_lengths = get_max_perm_cluster_lengths_based_on_percentile(n_cluster_perms, distribution, percentile, random_state)

    # Step 5: Apply cluster correction such that true cluster lengths must be greater than cluster_percentile of the maximum cluster sizes across permutations to survive
    max_perm_cluster_lengths_threshold = np.percentile(max_perm_cluster_lengths, cluster_percentile)
    significant_clusters = []
    
    for true_cluster, true_cluster_length in zip(true_clusters, true_cluster_lengths):
        if true_cluster_length > max_perm_cluster_lengths_threshold:
            significant_clusters.append(true_cluster)
    
    return significant_clusters


def find_cluster_lengths(clusters: list[tuple[int, int]]) -> list[int]:
    """Calculates the length of each cluster.
    
    This function takes a list of clusters, where each cluster is represented by a tuple of its start and end indices, and computes the number of points contained within each cluster (inclusive).
    
    Parameters
    ---------
    clusters : list[tuple[int, int]]
        A list of tuples, where each tuple contains the start and end
        index of a contiguous cluster
        
    Returns:
    cluster_lengths : list[int]
        A list of integers representing the length of each corresponding cluster.
        
    Example
    -------
    >>> clusters = [(10, 15), (25, 30)]
    >>> find_cluster_lengths(clusters)
    [6,6]
    """
    cluster_lengths = []
    for start_idx, end_idx in clusters:
        cluster_lengths.append(end_idx - start_idx + 1)                     
    return cluster_lengths


def get_max_perm_cluster_lengths_based_on_percentile(
    n_cluster_perms: int, 
    distribution: np.ndarray, 
    percentile: float, 
    random_state: int = None
    ) -> list[int]:
    '''
    Builds a null distribution of maximum cluster lengths.
    
    This function performs permutations to create a distribution of the max cluster size expected by chance. In each permutation, it:
    1. Randomly selects one time-series form the null 'distribution'.
    2. Uses the remaining series to calculate a pointwise percentile threshold.
    3. Finds clusters where the selected series exceeds this threshold.
    4. Records the length of the largest cluster found.
    
    The resulting list of maximum lengths can be used to determine a
    significance threshold for cluster-based permutation testing.
    
    Parameters
    ----------
    n_cluster_perms : int
        The number of permutations to run to build the null distribution.
    distribution : np.ndarray
        A 2D array of shape (n, time) representing the null distribution (e.g., from shuffled-label decoding)
    percentile : float
        The percentile (0-100) used to define pointwise significance when forming clusters within each permutation.
    random_state : int, optional
        Seed for the random number generator for reproducibility.
    
    Returns
    -------
    list[int]
        A list containing the maximum cluster length found in each permutation.
        The length of this list is equal to 'n_cluster_perms'.
    '''
    rng = np.random.RandomState(random_state)
    max_perm_cluster_lengths = []
    for _ in range(n_cluster_perms):
        # randomly select one row of the distribution to act as a series for this permutation
        perm_row_idx = rng.randint(0, distribution.shape[0])
        perm_series = distribution[perm_row_idx, :]
        
        # use remaining rows as the null distribution
        remaining_rows = np.delete(distribution, perm_row_idx, axis=0)
        perm_threshold = np.percentile(remaining_rows, percentile, axis=0)
        
        # find clusters in this permutation
        perm_sig_mask = (perm_series > perm_threshold).squeeze()
        perm_clusters = find_contiguous_clusters(perm_sig_mask)
        
        # find maximum cluster length for this permutation and add it to the list of maximum cluster lengths for all permutations
        perm_cluster_lengths = find_cluster_lengths(perm_clusters)
        
        if perm_cluster_lengths:
            max_perm_cluster_length = max(perm_cluster_lengths)
        else:
            max_perm_cluster_length = 0 # no clusters found, so max length is 0
            
        max_perm_cluster_lengths.append(max_perm_cluster_length)
        
    return max_perm_cluster_lengths


def compute_pooled_bootstrap_statistics(
    time_window_decoding_results, 
    n_bootstraps,
    condition_comparisons, 
    rois,
    percentile=95, 
    cluster_percentile=95,
    n_cluster_perms=1000, 
    random_state=42, 
    unit_of_analysis='bootstrap'
):
    """
    Pool samples and run statistics based on the specified unit of analysis.
    
    Parameters
    ----------
    time_window_decoding_results : dict
        Dictionary with bootstrap indices as keys, containing decoding results
    n_bootstraps : int
        Number of bootstrap samples
    condition_comparisons : dict
        Dictionary of condition comparisons to analyze
    rois : list
        List of ROIs to process
    percentile : float
        Percentile threshold for pointwise significance
    cluster_percentile : float
        Percentile threshold for cluster-level correction
    n_cluster_perms : int
        Number of permutations for cluster correction
    random_state : int
        Random seed for reproducibility
    unit_of_analysis : str
        'bootstrap': Average within each bootstrap (across repeats/folds)
        'repeat': Each repeat is a sample (average across folds only)
        'fold': Each fold is a sample (no averaging)
        
    Returns
    -------
    pooled_stats : dict
        Nested dictionary with statistics for each condition comparison and ROI
    """
    
    pooled_stats = {}
    
    for condition_comparison in condition_comparisons.keys():
        pooled_stats[condition_comparison] = {}
        
        for roi in rois:
            # Collect all accuracies across bootstraps
            all_true_accuracies = []  # Will collect based on unit_of_analysis
            all_shuffle_accuracies = []
            
            # First, gather raw data from all bootstraps
            for b_idx in range(n_bootstraps):
                if (condition_comparison in time_window_decoding_results[b_idx] and 
                    roi in time_window_decoding_results[b_idx][condition_comparison]):
                    
                    # Get raw accuracies - shape: (n_windows, n_repeats_or_folds)
                    true_acc = time_window_decoding_results[b_idx][condition_comparison][roi]['accuracies_true']
                    shuffle_acc = time_window_decoding_results[b_idx][condition_comparison][roi]['accuracies_shuffle']
                    
                    if unit_of_analysis == 'bootstrap':
                        # Average across repeats/folds for this bootstrap
                        # Result: one value per bootstrap
                        bootstrap_mean_true = np.mean(true_acc, axis=1)  # (n_windows,)
                        bootstrap_mean_shuffle = np.mean(shuffle_acc, axis=1)
                        
                        all_true_accuracies.append(bootstrap_mean_true)
                        all_shuffle_accuracies.append(bootstrap_mean_shuffle)
                        
                    elif unit_of_analysis == 'repeat':
                        # Keep repeats separate (assuming folds were already summed/averaged)
                        # Each repeat becomes a separate sample
                        # true_acc shape should be (n_windows, n_repeats) if folds_as_samples=False
                        for rep_idx in range(true_acc.shape[1]):
                            all_true_accuracies.append(true_acc[:, rep_idx])
                        for perm_idx in range(shuffle_acc.shape[1]):
                            all_shuffle_accuracies.append(shuffle_acc[:, perm_idx])
                            
                    elif unit_of_analysis == 'fold':
                        # Keep all folds as separate samples
                        # true_acc shape should be (n_windows, n_repeats*n_folds) if folds_as_samples=True
                        for sample_idx in range(true_acc.shape[1]):
                            all_true_accuracies.append(true_acc[:, sample_idx])
                        for perm_idx in range(shuffle_acc.shape[1]):
                            all_shuffle_accuracies.append(shuffle_acc[:, perm_idx])
                    
                    else:
                        raise ValueError(f"Invalid unit_of_analysis: {unit_of_analysis}")
            
            if not all_true_accuracies:
                print(f"Warning: No data for {condition_comparison} - {roi}")
                continue
            
            # Stack all samples - shape: (n_samples, n_windows)
            true_accs_stacked = np.vstack(all_true_accuracies)
            shuffle_accs_stacked = np.vstack(all_shuffle_accuracies)
            
            # Compute mean across samples for the "series" to test
            mean_true_accs = np.mean(true_accs_stacked, axis=0, keepdims=True)  # (1, n_windows)
            mean_shuffle_accs = np.mean(shuffle_accs_stacked, axis=0, keepdims=True)
            
            # Run significance test using percentile-based cluster correction
            time_window_centers = time_window_decoding_results[0][condition_comparison][roi]['time_window_centers']
            
            significant_cluster_indices = find_significant_clusters_of_series_vs_distribution_based_on_percentile(
                series=mean_true_accs,
                distribution=shuffle_accs_stacked,  # Use full distribution for comparison
                time_points=time_window_centers,
                percentile=percentile,
                cluster_percentile=cluster_percentile,
                n_cluster_perms=n_cluster_perms,
                random_state=random_state
            )
            
            # Convert cluster indices to boolean mask
            significant_clusters = np.zeros(len(time_window_centers), dtype=bool)
            for start_idx, end_idx in significant_cluster_indices:
                significant_clusters[start_idx:end_idx+1] = True
            
            # Store results with clear naming
            pooled_stats[condition_comparison][roi] = {
                f'{unit_of_analysis}_true_accs': true_accs_stacked,  # All samples
                f'mean_true_across_{unit_of_analysis}s': mean_true_accs,
                f'std_true_across_{unit_of_analysis}s': np.std(true_accs_stacked, axis=0),
                f'{unit_of_analysis}_shuffle_accs': shuffle_accs_stacked,  # All samples
                f'mean_shuffle_across_{unit_of_analysis}s': mean_shuffle_accs,
                f'std_shuffle_across_{unit_of_analysis}s': np.std(shuffle_accs_stacked, axis=0),
                'significant_clusters': significant_clusters,
                f'n_{unit_of_analysis}_samples': true_accs_stacked.shape[0],
                'unit_of_analysis': unit_of_analysis
            }
    
    return pooled_stats        


def do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions_for_one_roi(
    time_window_decoding_results, n_bootstraps, 
    condition_comparison_1, condition_comparison_2, roi, 
    stat_func, unit_of_analysis,
    p_thresh=0.05, p_cluster=0.05, n_perm=500, tails=2, axis=0, random_state=42, n_jobs=-1
):
    
    # Use the new, flexible data extraction function
    pooled_cond1_accs, pooled_cond2_accs = get_pooled_accuracy_distributions_for_comparison(
        time_window_decoding_results,
        n_bootstraps,
        condition_comparison_1,
        condition_comparison_2,
        roi,
        unit_of_analysis
    )
    
    # Handle cases with no data
    if pooled_cond1_accs.size == 0 or pooled_cond2_accs.size == 0:
        print(f"Warning: Insufficient data for LWPC comparison in ROI {roi}. Skipping stats.")
        # Return a correctly shaped empty result
        time_points_len = len(time_window_decoding_results[0][condition_comparison_1][roi]['time_window_centers'])
        return np.zeros(time_points_len, dtype=bool), np.array([])
        
    significant_clusters, p_values = time_perm_cluster(
        pooled_cond1_accs,
        pooled_cond2_accs,
        p_thresh=p_thresh,
        p_cluster=p_cluster,
        n_perm=n_perm,
        tails=tails,
        axis=axis, # Samples are on axis 0, time on axis 1
        stat_func=stat_func,
        n_jobs=n_jobs,
        seed=random_state
    )
    
    return significant_clusters, p_values


def get_pooled_accuracy_distributions_for_comparison(
    time_window_decoding_results: dict,
    n_bootstraps: int,
    condition_comparison_1: str,
    condition_comparison_2: str,
    roi: str,
    unit_of_analysis: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pools accuracy samples for two conditions based on the specified unit of analysis.

    This function iterates through all bootstrap results and collects accuracy time-series.
    The level of granularity (pooling bootstrap means, repeats, or folds) is
    determined by the `unit_of_analysis` parameter.

    Parameters
    ----------
    time_window_decoding_results : dict
        The main results dictionary from the parallel bootstrap processing.
    n_bootstraps : int
        The total number of bootstraps run.
    condition_comparison_1 : str
        The name of the first condition to pool (e.g., 'c25_vs_i25').
    condition_comparison_2 : str
        The name of the second condition to pool (e.g., 'c75_vs_i75').
    roi : str
        The Region of Interest to process.
    unit_of_analysis : str
        The sampling unit ('bootstrap', 'repeat', or 'fold').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The first array has shape (n_samples, n_windows) for condition 1.
        - The second array has shape (n_samples, n_windows) for condition 2.
        Returns empty arrays if data is not found.
    """
    all_cond1_accuracies = []
    all_cond2_accuracies = []

    for b_idx in range(n_bootstraps):
        # Ensure data exists for this bootstrap, ROI, and both conditions
        if (b_idx in time_window_decoding_results and
            condition_comparison_1 in time_window_decoding_results[b_idx] and
            roi in time_window_decoding_results[b_idx][condition_comparison_1] and
            condition_comparison_2 in time_window_decoding_results[b_idx] and
            roi in time_window_decoding_results[b_idx][condition_comparison_2]):

            # Get the raw accuracies for this bootstrap. Shape: (n_windows, n_repeats_or_folds)
            acc1 = time_window_decoding_results[b_idx][condition_comparison_1][roi]['accuracies_true']
            acc2 = time_window_decoding_results[b_idx][condition_comparison_2][roi]['accuracies_true']

            if unit_of_analysis == 'bootstrap':
                # Average across repeats/folds to get a single time-series per bootstrap
                all_cond1_accuracies.append(np.mean(acc1, axis=1)) # Shape: (n_windows,)
                all_cond2_accuracies.append(np.mean(acc2, axis=1))
            elif unit_of_analysis in ['repeat', 'fold']:
                # Treat each repeat or fold as an independent sample
                # Loop through the samples dimension (axis=1)
                for sample_idx in range(acc1.shape[1]):
                    all_cond1_accuracies.append(acc1[:, sample_idx])
                    all_cond2_accuracies.append(acc2[:, sample_idx])
            else:
                raise ValueError(f"Invalid unit_of_analysis: '{unit_of_analysis}'")

    if not all_cond1_accuracies or not all_cond2_accuracies:
        return np.array([]), np.array([])

    # Stack all collected samples into a 2D array (n_samples, n_windows)
    pooled_cond1_accs = np.vstack(all_cond1_accuracies)
    pooled_cond2_accs = np.vstack(all_cond2_accuracies)

    return pooled_cond1_accs, pooled_cond2_accs


def do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions(
    time_window_decoding_results, n_bootstraps, 
    condition_comparison_1, condition_comparison_2, rois, 
    stat_func, unit_of_analysis,
    p_thresh=0.05, p_cluster=0.05, n_perm=500, tails=2, axis=0, random_state=42, n_jobs=-1
):
    stats = {}
    
    for roi in rois:
        significant_clusters, p_values = do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions_for_one_roi(
            time_window_decoding_results=time_window_decoding_results, 
            n_bootstraps=n_bootstraps, 
            condition_comparison_1=condition_comparison_1, 
            condition_comparison_2=condition_comparison_2, 
            roi=roi, 
            stat_func=stat_func,
            unit_of_analysis=unit_of_analysis,
            p_thresh=p_thresh,
            p_cluster=p_cluster, 
            n_perm=n_perm, 
            tails=tails, 
            axis=axis, 
            random_state=random_state, 
            n_jobs=n_jobs
        )
        
        stats[roi] = significant_clusters, p_values
        
    return stats


def do_mne_paired_cluster_test(
    accuracies1: np.ndarray,
    accuracies2: np.ndarray,
    p_thresh: float = 0.05,
    n_perm: int = 1000,
    tails: int = 0, # MNE: 0 for two-tail, 1 for right-tail, -1 for left-tail
    random_state: int = 42,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Performs a paired-sample cluster permutation test using MNE.

    It tests the difference (accuracies1 - accuracies2) against zero.

    Args:
        accuracies1: Data for condition 1, shape (n_samples, n_times).
        accuracies2: Data for condition 2, shape (n_samples, n_times).
        p_thresh: The p-value threshold for forming clusters.
        n_perm: The number of permutations.
        tails: 0 for two-tailed, 1 for one-tailed (1 > 2), -1 for one-tailed (1 < 2).
        random_state: Seed for the random number generator.
        n_jobs: Number of jobs for parallel processing.

    Returns:
        A boolean mask of shape (n_times,) indicating significant time points.
    """
    if accuracies1.shape != accuracies2.shape:
        raise ValueError("Accuracy arrays must have the same shape for a paired test.")
    
    print("🔬 Running MNE paired (one-sample on differences) cluster permutation test...")

    # Step 1: Calculate the difference for each paired sample
    differences = accuracies1 - accuracies2
    n_samples = differences.shape[0]

    if n_samples < 2:
        print("⚠️ Warning: Not enough samples to run stats. Returning no significant clusters.")
        return np.zeros(differences.shape[1], dtype=bool)

    # Step 2: Calculate the t-statistic threshold from the p-value
    # For a two-tailed test, we divide the p-value by 2
    p_val_for_t = p_thresh / 2 if tails == 0 else p_thresh
    degrees_of_freedom = n_samples - 1
    t_threshold = t.ppf(1 - p_val_for_t, df=degrees_of_freedom)

    # Step 3: Run the MNE one-sample cluster test
    t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X=differences,
        threshold=t_threshold,
        n_permutations=n_perm,
        tail=tails,
        n_jobs=n_jobs,
        seed=random_state,
        verbose=False
    )

    # Step 4: Create a boolean mask from the significant clusters
    significant_clusters_mask = np.zeros(differences.shape[1], dtype=bool)
    for i_cluster, p_val in enumerate(cluster_p_values):
        if p_val < p_thresh:
            cluster_indices = clusters[i_cluster][0]
            significant_clusters_mask[cluster_indices] = True
            
    print(f"Found {len(cluster_p_values[cluster_p_values < p_thresh])} significant cluster(s).")
    
    return significant_clusters_mask


def get_time_averaged_confusion_matrix(
    roi_labeled_arrays, roi, strings_to_find, clf, n_splits, n_repeats,
    obs_axs, balance_method, balance_strata, explained_variance, random_state, cats
):
    """
    Computes a single time-averaged confusion matrix for one bootstrap sample.
    Returns the RAW COUNTS instead of a normalized matrix.
    """
    concatenated_data, labels, _ = concatenate_and_balance_data_for_decoding(
        roi_labeled_arrays, roi, strings_to_find, obs_axs, balance_method, balance_strata, random_state
    )

    if concatenated_data.size == 0:
        return None

    decoder = Decoder(cats, explained_variance, oversample=True, n_splits=n_splits, n_repeats=n_repeats, clf=clf, random_state=random_state)
    
    # Key Change: Set normalize=None to get raw counts
    # The result will be shape (n_repeats, n_classes, n_classes)
    cm_repeats = decoder.cv_cm_jim(concatenated_data, labels, normalize=None, obs_axs=obs_axs)
    
    # Sum across all repeats to get a single count matrix for this bootstrap
    return np.sum(cm_repeats, axis=0)


def _run_single_permutation(differences: np.ndarray, se_diff: np.ndarray, t_thresh: float, tails: int, seed: int) -> int:
    """
    Execute a single null permutation for the cluster paired perm t-test and return the max cluster duration.

    This is the core "worker" function that will be parallelized by joblib. It takes the
    pre-calculated differences, randomly flips their signs, computes a t-statistic,
    finds clusters, and returns the duration of the largest cluster found in this single
    permutation. 10/21/25 - Uses the same standard error as the true accuracy difference distribution to kep the noise identical.

    Parameters
    ----------
    differences : np.ndarray
        The paired differences between two conditions, shape (n_samples, n_times).
    se_diff : np.ndarray
        precomputed standard error from the true accuracy difference distribution
    t_thresh : float
        The t-value threshold for forming clusters.
    tails : int
        Specifies the type of test (1 for one-tailed, 2 for two-tailed).
    seed : int
        A unique seed for the random number generator to ensure independent permutations.

    Returns
    -------
    int
        The maximum cluster duration (number of consecutive time points) found in this permutation.
        Returns 0 if no clusters are found.
    """
    rng = np.random.default_rng(seed)
    n_samples = differences.shape[0]

    # Permute by randomly flipping the sign of the difference for each sample
    sign_flips = rng.choice([-1, 1], size=(n_samples, 1))
    perm_diffs = differences * sign_flips
    
    # # Calculate t-statistic for the permuted data, handling potential division by zero
    # std_perm = np.std(perm_diffs, axis=0, ddof=1)
    # # Prevent division by zero if std is 0 for a time point
    # std_perm[std_perm == 0] = 1 
    
    # calculate t-statistic using the same standard error as the true differencce
    t_perm = np.mean(perm_diffs, axis=0) / se_diff
    
    if tails == 2:
        perm_sig_points = np.abs(t_perm) > t_thresh
    else:
        perm_sig_points = t_perm > t_thresh
    
    perm_labeled, n_perm_clusters = label(perm_sig_points)
    
    if n_perm_clusters > 0:
        perm_cluster_durations = np.array([np.sum(perm_labeled == i) for i in range(1, n_perm_clusters + 1)])
        return np.max(perm_cluster_durations)
    else:
        return 0


def cluster_perm_paired_ttest_by_duration(
    accuracies1: np.ndarray,
    accuracies2: np.ndarray,
    p_thresh: float = 0.05,
    p_cluster: float = 0.05,
    n_perm: int = 1000,
    tails: int = 2, # 2 for two-tailed, 1 for one-tailed (1 > 2)
    random_state: int = 42,
    n_jobs: int = -1 # Number of jobs for parallel processing
) -> np.ndarray:
    """
    Perform a parallelized paired-sample cluster permutation test using cluster duration.

    This function tests for significant differences between two paired conditions over time.
    It uses a non-parametric approach based on cluster mass, where the "mass" is defined as
    the duration (number of consecutive time points) of a cluster.

    The process is as follows:
    1.  Calculate observed t-statistics for the difference between `accuracies1` and `accuracies2`.
    2.  Identify contiguous clusters where the t-statistic exceeds a threshold (`p_thresh`).
    3.  Calculate the duration of these observed clusters.
    4.  Build a null distribution of the *maximum* cluster duration by running permutations in parallel.
        In each permutation, the signs of the differences are randomly flipped.
    5.  Compare the observed cluster durations to this null distribution. A cluster is considered
        significant if its duration exceeds the threshold defined by `p_cluster` (e.g., the 95th
        percentile) of the null distribution.

    Parameters
    ----------
    accuracies1 : np.ndarray
        Data for the first condition, with shape (n_samples, n_times).
    accuracies2 : np.ndarray
        Data for the second condition, with shape (n_samples, n_times). Must be paired with `accuracies1`.
    p_thresh : float, optional
        The p-value used to set the initial t-statistic threshold for forming clusters. Default is 0.05.
    p_cluster : float, optional
        The p-value for determining the final significance of a cluster. An observed cluster is
        significant if its duration is greater than the `(1 - p_cluster)` percentile of the
        null distribution of maximum cluster durations. Default is 0.05.
    n_perm : int, optional
        The number of permutations to run to build the null distribution. Default is 1000.
    tails : int, optional
        The type of test to perform:
        - 2: two-tailed test (accuracies1 != accuracies2)
        - 1: one-tailed test (accuracies1 > accuracies2)
        Default is 2.
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.
    n_jobs : int, optional
        The number of CPU cores to use for parallelizing the permutation loop.
        -1 means using all available cores. Default is -1.

    Returns
    -------
    np.ndarray
        A boolean mask of shape (n_times,) where `True` indicates that a time point
        belongs to a statistically significant cluster.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_times = accuracies1.shape
    
    # --- Step 1: Calculate observed clusters ---
    differences = accuracies1 - accuracies2
    std_diff = np.std(differences, axis=0, ddof=1)
    std_diff[std_diff == 0] = 1 # Prevent division by zero
    se_diff = std_diff / np.sqrt(n_samples)
    
    t_obs = np.mean(differences, axis=0) / se_diff
    
    df = n_samples - 1
    p_for_t = p_thresh / 2 if tails == 2 else p_thresh
    t_thresh = t.ppf(1 - p_for_t, df=df)
    
    if tails == 2:
        significant_points = np.abs(t_obs) > t_thresh
    else:
        significant_points = t_obs > t_thresh
        
    labeled_clusters, n_clusters = label(significant_points)
    
    if n_clusters == 0:
        print("No clusters found in observed data.")
        return np.zeros(n_times, dtype=bool)
        
    observed_cluster_durations = np.array([np.sum(labeled_clusters == i) for i in range(1, n_clusters + 1)])

    # --- Step 2: Build null distribution in PARALLEL ---
    print(f" Building null distribution with {n_perm} permutations across {n_jobs} jobs...")
    
    # Generate independent seeds for each permutation job for reproducibility
    seeds = rng.integers(low=0, high=2**32-1, size=n_perm)

    max_perm_durations = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_permutation)(differences, se_diff, t_thresh, tails, seed) for seed in seeds
    )

    # --- Step 3: Determine significance ---
    critical_duration = np.percentile(max_perm_durations, 100 * (1 - p_cluster))
    
    final_sig_mask = np.zeros(n_times, dtype=bool)
    for i in range(1, n_clusters + 1):
        if observed_cluster_durations[i-1] > critical_duration:
            final_sig_mask[labeled_clusters == i] = True
            
    print(f" Found {np.sum(observed_cluster_durations > critical_duration)} significant cluster(s) using duration statistic.")
    return final_sig_mask


def run_two_one_tailed_tests_with_time_perm_cluster(
    accuracies1, accuracies2, 
    p_thresh=0.025, p_cluster=0.025, stat_func=None,
    permutation_type='independent',
    n_perm=100, random_state=42, n_jobs=-1
):
    """
    Run two one-tailed tests using time_perm_cluster function.
    
    Parameters:
    -----------
    accuracies1, accuracies2 : arrays of shape (n_samples, n_times)
        The accuracy distributions to compare
        
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html#permutation-test
    """
    # Two one-tailed tests
    
    # Test 1: accuracies1 > accuracies2 (tails=1 for greater)
    sig_mask_positive, p_obs_positive = time_perm_cluster(
        sig1=accuracies1,
        sig2=accuracies2,
        p_thresh=p_thresh,
        p_cluster=p_cluster,
        n_perm=n_perm,
        tails=1,  # One-tailed test for sig1 > sig2
        axis=0,   # Observations/samples axis
        stat_func=stat_func,
        permutation_type=permutation_type,  # Do 'samples' for paired samples, and 'independent' for independent t-tests
        n_jobs=n_jobs,
        seed=random_state,
        verbose=40
    )
    
    # Test 2: accuracies2 > accuracies1 (still tails=1, but swap inputs)
    sig_mask_negative, p_obs_negative = time_perm_cluster(
        sig1=accuracies2,  # Swapped
        sig2=accuracies1,  # Swapped
        p_thresh=p_thresh,
        p_cluster=p_cluster,
        n_perm=n_perm,
        tails=1,  # One-tailed test for sig1 > sig2 (now acc2 > acc1)
        axis=0,
        stat_func=stat_func,
        permutation_type=permutation_type,  # Do 'samples' for paired samples, and 'independent' for independent t-tests
        n_jobs=n_jobs,
        seed=random_state + 1 if random_state else None,
        verbose=40
    )
    
    return sig_mask_positive, sig_mask_negative, p_obs_positive, p_obs_negative
