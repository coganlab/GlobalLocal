import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.calc.fast import mean_diff
from ieeg.calc.stats import time_perm_cluster
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from utils import calculate_RTs, get_good_data

def get_wavelet_baseline(inst: mne.io.BaseRaw, base_times: tuple[float, float]):
    inst = inst.copy()
    inst.load_data()
    ch_type = inst.get_channel_types(only_data_chs=True)[0]
    inst.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    adjusted_base_times = [base_times[0] - 0.5, base_times[1] + 0.5]
    trials = trial_ieeg(inst, "Stimulus", adjusted_base_times, preload=True)
    outliers_to_nan(trials, outliers=10)
    base = wavelet_scaleogram(trials, n_jobs=-2, decim=int(inst.info['sfreq'] / 100))
    crop_pad(base, "0.5s")
    del inst
    return base

def get_trials_for_wavelets(data, events, times):
    '''
    Extract and concatenate non-outlier trials for wavelet analysis.

    Parameters:
    -----------
    data : mne.io.Raw
        The preprocessed raw EEG data.
    events : list of str
        List of event names to extract trials for.
    times : list of float
        Time window relative to the events to extract data from.

    Returns:
    --------
    all_trials : mne.Epochs
        The concatenated epochs for all specified events.

    Examples:
    ---------
    >>> events = ['Stimulus/c25', 'Stimulus/c75']
    >>> times = [-0.5, 1.5]
    >>> all_trials = get_trials_for_wavelets(good, events, times)
    >>> isinstance(all_trials, mne.Epochs)
    True
    '''
    all_trials_list = []

    for event in events:
        # Adjust times for padding
        times_adj = [times[0] - 0.5, times[1] + 0.5]
        trials = trial_ieeg(data, event, times_adj, preload=True,
                            reject_by_annotation=False)
        all_trials_list.append(trials)

    # Concatenate all trials
    all_trials = mne.concatenate_epochs(all_trials_list)

    # Mark outliers as NaN
    outliers_to_nan(all_trials, outliers=10)

    return all_trials

def get_uncorrected_wavelets(sub, layout, events, times):
    '''
    Get non-baseline-corrected wavelets for trials corresponding to those in events.

    Parameters:
    -----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events : list of str
        List of event names to extract trials for.
    times : list of float
        Time window relative to the events to extract data from.

    Returns:
    --------
    spec : mne.time_frequency.EpochsTFR
        The time-frequency representation of the wavelet-transformed data.

    Examples:
    ---------
    >>> sub = 'sub-01'
    >>> events = ['Stimulus/c25', 'Stimulus/c75']
    >>> times = [-0.5, 1.5]
    >>> spec = get_uncorrected_wavelets(sub, layout, events, times)
    >>> isinstance(spec, mne.time_frequency.EpochsTFR)
    True
    '''
    # Preprocess data and extract trials
    good = get_good_data(sub, layout)
    all_trials = get_trials_for_wavelets(good, events, times)

    # Compute wavelets
    spec = wavelet_scaleogram(all_trials, n_jobs=1, decim=int(good.info['sfreq'] / 100))
    crop_pad(spec, "0.5s")

    return spec

def get_wavelet_differences(sub, layout, events_condition_1, events_condition_2, times, stat_func: callable = mean_diff, p_thresh=0.05, ignore_adjacency=1, n_perm=100, n_jobs=1, make_wavelets=False):
    '''
    Compares two signals, loading their wavelets and computing the significantly different clusters in their wavelets. 

    Parameters:
    -----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events_condition_1 : list of str
        List of event names for the first condition.
    events_condition_2 : list of str
        List of event names for the second condition.
    times : list of float
        Time window relative to the events to extract data from.
    make_wavelets : boolean
        Whether to make or load the wavelets. If True, it will make the wavelets. If False, it will attempt to load the wavelets using the output name.
    stat_func : callable, optional
            The statistical function to use to compare populations. Requires an
            axis keyword input to denote observations (trials, for example).
            Default function is `mean_diff`, but may be substituted with other test
            functions found here:
            https://scipy.github.io/devdocs/reference/stats.html#independent
            -sample-tests    

    p_thresh : float, optional
        The p-value threshold for significance (default is 0.05).
    ignore_adjacency : int, optional
        The number of adjacent time points to ignore when forming clusters (default is 1).
    n_perm : int, optional
        The number of permutations to perform (default is 100).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 1).
    Returns:
    --------
    mask : numpy.ndarray
        A boolean mask indicating significant clusters.
    pvals : numpy.ndarray
        The p-values for each cluster.

    Examples:
    ---------
    >>> sub = 'sub-01'
    >>> events_condition_1 = ['Stimulus/c25']
    >>> events_condition_2 = ['Stimulus/c75']
    >>> times = [-0.5, 1.5]
    >>> mask, pvals = get_wavelet_differences(sub, layout, events_condition_1, events_condition_2, times)
    >>> isinstance(mask, np.ndarray)
    True
    >>> isinstance(pvals, np.ndarray)
    True
    '''
    if make_wavelets:
        spec_condition_1 = get_uncorrected_wavelets(sub, layout, events_condition_1, times)
        spec_condition_2 = get_uncorrected_wavelets(sub, layout, events_condition_2, times)
    else:
        # TODO: load wavelets based on output_name (make a load_wavelets function that loads in wavelets based on their output_name)
        pass

    mask, pvals = time_perm_cluster(spec_condition_1._data, spec_condition_2._data, stat_func=stat_func, p_thresh=p_thresh, ignore_adjacency=ignore_adjacency, n_perm=n_perm, n_jobs=n_jobs)
    return mask, pvals