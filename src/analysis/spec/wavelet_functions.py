import os
import sys

# Conditionally append a path for a specific user on a specific OS
# This is useful for developers who have local dependencies on their machine.
if sys.platform == 'win32' and os.getlogin() == 'jz421':
    # sys.platform returns 'win32' for both 32-bit and 64-bit Windows
    # os.getlogin() returns the name of the user logged in to the terminal
    custom_path = "C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/"
    
    # Check if the path is not already in sys.path to avoid duplicates
    if custom_path not in sys.path:
        sys.path.append(custom_path)
        print(f"Appended custom path for user 'jz421' on Windows: {custom_path}")

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

import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.calc.fast import mean_diff
from ieeg.calc.stats import time_perm_cluster
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from src.analysis.utils.general_utils import calculate_RTs, get_good_data, get_trials

import matplotlib.pyplot as plt

from typing import Union, List, Tuple, Callable, Optional

def get_wavelet_baseline(inst: mne.io.BaseRaw, base_times: tuple[float, float]):
    """
    Compute a wavelet scaleogram baseline from an EEG recording.

    This function creates a copy of the raw EEG data, loads it, and sets an average reference.
    It then extracts a baseline segment (with an extra 0.5-second padding before and after the
    specified baseline times) around the "Stimulus" event, marks outlier data as NaN, computes the
    wavelet scaleogram, and applies cropping/padding to the resulting time-frequency representation.

    Parameters
    ----------
    inst : mne.io.BaseRaw
        The raw EEG data instance.
    base_times : tuple of float
        A tuple (start, end) in seconds defining the baseline window.

    Returns
    -------
    base : mne.time_frequency.EpochsTFR
        The wavelet scaleogram of the baseline segment.

    Example
    -------
    >>> # Assume 'raw' is a preloaded mne.io.Raw instance with EEG data.
    >>> baseline = get_wavelet_baseline(raw, (0, 1))
    >>> hasattr(baseline, 'data')
    True
    """
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

def get_uncorrected_wavelets(sub: str, layout, events: list[str], times: tuple[float, float], n_jobs: int=1) -> mne.time_frequency.EpochsTFR:
    """
    Compute non-baseline-corrected wavelets for specified trials.

    This function retrieves preprocessed EEG data for a subject using `get_good_data`, extracts epochs
    for the specified events (with additional padding), computes the wavelet scaleogram for the epochs,
    and then applies cropping/padding to the resulting time-frequency representation.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events : list of str
        A list of event names to extract trials for.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    n_jobs : int
        The number of parallel jobs to use for the wavelet scaleogram.
    Returns
    -------
    spec : mne.time_frequency.EpochsTFR
        The time-frequency representation (wavelet scaleogram) of the extracted epochs.

    Examples
    --------
    >>> # Assume 'layout' is a valid BIDSLayout and subject 'sub-01' has corresponding data.
    >>> events = ['Stimulus/c25', 'Stimulus/c75']
    >>> times = (-0.5, 1.5)
    >>> tfr = get_uncorrected_wavelets('sub-01', layout, events, times)
    >>> isinstance(tfr, mne.time_frequency.EpochsTFR)
    True
    """
    # Retrieve preprocessed data for the subject
    good = get_good_data(sub, layout)
    padded_times = (times[0] - 0.5, times[1] + 0.5)
    all_trials = get_trials(good, events, padded_times)

    # Compute wavelets for the extracted trials
    spec = wavelet_scaleogram(all_trials, n_jobs=n_jobs, decim=int(good.info['sfreq'] / 100))
    
    # add fnames to spec.info
    fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
    spec.info['subject_info']['files'] = tuple(fnames)
    
    crop_pad(spec, "0.5s")

    return spec

def get_uncorrected_multitaper(sub: str, layout, events: list[str], times: tuple[float, float], freqs: np.ndarray, n_cycles: int | np.ndarray, time_bandwidth: int, return_itc: bool, average: bool, n_jobs: int = 1) -> mne.time_frequency.EpochsTFR:
    """
    Compute non-baseline-corrected multitaper spectrogram for specified trials.

    This function retrieves preprocessed EEG data for a subject using `get_good_data`, extracts epochs
    for the specified events (with additional padding), computes the multitaper spectrogram for the epochs,
    and then applies cropping/padding to the resulting time-frequency representation.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events : list of str
        A list of event names to extract trials for.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    freqs : np.ndarray
        The frequencies to compute the multitaper spectrogram for.
    n_cycles : int | array of int, shape (n_freqs)
        The number of cycles to use for each frequency.
    time_bandwidth : int
        The time-bandwidth product to use for the multitaper spectrogram.
    return_itc : bool
        Whether to return the intertrial coherence (ITC) as well as the multitaper spectrogram.
    average : bool
        Whether to average the multitaper spectrogram across trials.
    n_jobs : int
        The number of parallel jobs to use for the multitaper spectrogram.
    Returns
    -------
    spec : mne.time_frequency.EpochsTFR
        The time-frequency representation (multitaper spectrogram) of the extracted epochs.

    Examples
    --------
    >>> # Assume 'layout' is a valid BIDSLayout and subject 'sub-01' has corresponding data.
    >>> events = ['Stimulus/c25', 'Stimulus/c75']
    >>> times = (-0.5, 1.5)
    >>> freqs = np.arange(10, 200, 2)
    >>> n_cycles = freqs / 2    
    >>> time_bandwidth = 10
    >>> return_itc = False
    >>> tfr = get_uncorrected_multitaper('sub-01', layout, events, times, freqs, n_cycles, time_bandwidth, return_itc)
    >>> isinstance(tfr, mne.time_frequency.EpochsTFR)
    True
    """
    # Retrieve preprocessed data for the subject
    good = get_good_data(sub, layout)
    padded_times = (times[0] - 0.5, times[1] + 0.5)
    all_trials = get_trials(good, events, padded_times)

    # Compute multitaper for the extracted trials (this can be replaced by epochs.compute_tfr): https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#sphx-glr-auto-examples-time-frequency-time-frequency-simulated-py
    spec = mne.time_frequency.tfr_multitaper(inst=all_trials, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=return_itc, average=average, n_jobs=n_jobs, decim=int(good.info['sfreq'] / 100))
    
    # add fnames to spec.info
    fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
    spec.info['subject_info']['files'] = tuple(fnames)

    crop_pad(spec, "0.5s")

    return spec

def make_and_get_sig_wavelet_differences(sub: str, layout, events_condition_1: List[str],
                                     events_condition_2: List[str], times: Tuple[float, float],
                                     stat_func: callable = mean_diff, p_thresh: float = 0.05,
                                     ignore_adjacency: int = 1, n_perm: int = 100, n_jobs: int = 1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelet TFRs from raw EEG data for two conditions and then compute their differences.

    This function extracts epochs for two sets of events from the raw data for a given subject,
    computes their wavelet representations using `get_uncorrected_wavelets`, and then applies a
    permutation cluster test to identify significant differences between the two conditions.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events_condition_1 : list of str
        List of event names corresponding to condition 1.
    events_condition_2 : list of str
        List of event names corresponding to condition 2.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    stat_func : callable, optional
        The statistical function for comparing the two datasets (default: mean_diff).
    p_thresh : float, optional
        The p-value threshold for significance (default: 0.05).
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_perm : int, optional
        The number of permutations to perform (default: 100).
    n_jobs : int, optional
        The number of parallel jobs (default: 1).

    Returns
    -------
    mask : np.ndarray
        A boolean array indicating significant clusters.
    pvals : np.ndarray
        An array of p-values corresponding to each identified cluster.

    Example
    -------
    >>> # Assuming valid values for sub, layout, events, and times:
    >>> mask, pvals = make_and_get_wavelet_differences('sub-01', layout, ['Stimulus/c25'], ['Stimulus/c75'], (-0.5, 1.5))
    >>> isinstance(mask, np.ndarray)
    True
    """
    spec_condition_1 = get_uncorrected_wavelets(sub, layout, events_condition_1, times, n_jobs=n_jobs)
    spec_condition_2 = get_uncorrected_wavelets(sub, layout, events_condition_2, times, n_jobs=n_jobs)

    mask, pvals = get_sig_tfr_differences(spec_condition_1, spec_condition_2,
                                        stat_func=stat_func, p_thresh=p_thresh,
                                        ignore_adjacency=ignore_adjacency, n_perm=n_perm, n_jobs=n_jobs)
    
    return mask, pvals

def make_and_get_sig_multitaper_differences(sub: str, layout, events_condition_1: List[str],
                                     events_condition_2: List[str], times: Tuple[float, float],
                                     stat_func: callable = mean_diff, p_thresh: float = 0.05,
                                     freqs: np.ndarray = np.arange(10, 200, 2), n_cycles: int = 50, 
                                     time_bandwidth: int = 10, return_itc: bool = False,
                                     ignore_adjacency: int = 1, average: bool = True, n_perm: int = 100, n_jobs: int = 1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multitaper TFRs from raw EEG data for two conditions and then compute their differences.

    This function extracts epochs for two sets of events from the raw data for a given subject,
    computes their multitaper representations using `get_uncorrected_multitaper`, and then applies a
    permutation cluster test to identify significant differences between the two conditions.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    events_condition_1 : list of str
        List of event names corresponding to condition 1.
    events_condition_2 : list of str
        List of event names corresponding to condition 2.
    times : tuple of float
        A tuple (start, end) in seconds relative to each event defining the extraction window.
    freqs : np.ndarray
        The frequencies to compute the multitaper spectrogram
    n_cycles : int | array of int, shape (n_freqs)
        The number of cycles to use for each frequency
    time_bandwidth : int
        The time-bandwidth product to use for the multitaper spectrogram
    return_itc : bool
        Whether to return the intertrial coherence (ITC) as well as the multitaper spectrogram
    stat_func : callable, optional
        The statistical function for comparing the two datasets (default: mean_diff).
    p_thresh : float, optional
        The p-value threshold for significance (default: 0.05).
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    average : bool
        Whether to average the multitaper spectrogram across trials.
    n_perm : int, optional
        The number of permutations to perform (default: 100).
    n_jobs : int, optional
        The number of parallel jobs (default: 1).

    Returns
    -------
    mask : np.ndarray
        A boolean array indicating significant clusters.
    pvals : np.ndarray
        An array of p-values corresponding to each identified cluster.

    Example
    -------
    >>> # Assuming valid values for sub, layout, events, and times:
    >>> mask, pvals = make_and_get_wavelet_differences('sub-01', layout, ['Stimulus/c25'], ['Stimulus/c75'], (-0.5, 1.5))
    >>> isinstance(mask, np.ndarray)
    True
    """
    spec_condition_1 = get_uncorrected_multitaper(sub, layout, events_condition_1, times, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=return_itc, average=average, n_jobs=n_jobs)
    spec_condition_2 = get_uncorrected_multitaper(sub, layout, events_condition_2, times, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=return_itc, average=average, n_jobs=n_jobs)

    mask, pvals = get_sig_tfr_differences(spec_condition_1, spec_condition_2,
                                        stat_func=stat_func, p_thresh=p_thresh,
                                        ignore_adjacency=ignore_adjacency, n_perm=n_perm, n_jobs=n_jobs)
    
    return mask, pvals

def get_sig_tfr_differences(
    tfr_data_cond1,
    tfr_data_cond2,
    stat_func: callable,
    elecs_to_pick: list = None,
    p_thresh: float = 0.05,
    p_cluster: float = None,
    n_perm: int = 1000,
    tails: int = 1,
    axis: int = 0,
    ignore_adjacency: int = None,
    n_jobs: int = 1,
    seed: int = None):
    """
    Compute permutation cluster differences between two TFR objects.

    This function takes two time-frequency representations (TFRs) and applies a
    permutation cluster test to determine statistically significant differences between them.

    Parameters
    ----------
    tfr_data_cond1 : mne.time_frequency.EpochsTFR
        The TFR data for the first condition.
    tfr_data_cond2 : mne.time_frequency.EpochsTFR
        The TFR data for the second condition.
    stat_func : callable, optional
        The statistical function to use to compare populations. Requires an
        axis keyword input to denote observations (trials, for example).
        Default function is `mean_diff`, but may be substituted with other test
        functions found here:
        https://scipy.github.io/devdocs/reference/stats.html#independent
        -sample-tests
    elecs_to_pick : list, optional
        A list of electrode names to include in the analysis. Defaults to None.
    p_thresh : float, optional
        P-value threshold for forming clusters. Default is 0.05.
    p_cluster : float, optional
        P-value threshold for significant clusters. If None, uses `p_thresh`. Default is None.
    n_perm : int, optional
        Number of permutations. Default is 1000.
    tails : int, optional
        Number of tails for the test (1 for one-tailed, 2 for two-tailed). Default is 1.
    axis : int, optional
        The axis for permutation (e.g., trials axis). Default is 0.
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_jobs : int, optional
        Number of parallel jobs. Default is 1.
    seed : int, optional
        Random seed for permutations. Default is None.

    Returns
    -------
    mask : np.ndarray
        A boolean array indicating significant clusters.
    pvals : np.ndarray
        An array of p-values corresponding to each identified cluster.

    Example
    -------
    >>> # Assuming spec1 and spec2 are valid mne.time_frequency.EpochsTFR objects with a _data attribute:
    >>> mask, pvals = get_sig_tfr_differences(spec1, spec2)
    >>> isinstance(mask, np.ndarray)
    True
    """
    tfr1 = tfr_data_cond1.copy()
    tfr2 = tfr_data_cond2.copy()

    if elecs_to_pick:
        tfr1.pick(elecs_to_pick)
        tfr2.pick(elecs_to_pick)

    # Pass all statistical parameters to the underlying test function
    mask, pvals = time_perm_cluster(
        tfr1._data, tfr2._data,
        stat_func=stat_func,
        p_thresh=p_thresh,
        p_cluster=p_cluster,
        n_perm=n_perm,
        tails=tails,
        axis=axis,
        ignore_adjacency=ignore_adjacency,
        n_jobs=n_jobs,
        seed=seed
    )

    return mask, pvals

def load_and_get_sig_wavelet_differences(sub: str, layout, output_name_condition_1: str,
                                     output_name_condition_2: str, rescaled: bool,
                                     stat_func: callable = mean_diff, p_thresh: float = 0.05,
                                     ignore_adjacency: int = 1, n_perm: int = 100, n_jobs: int = 1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed wavelet TFRs for two conditions and compute their sig differences.

    This function loads precomputed wavelet time-frequency representations for two conditions using the provided
    output names and a rescaling flag. It then performs a permutation cluster test to determine statistically
    significant differences between the conditions.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    output_name_cond1 : str
        Base name for the TFR file corresponding to condition 1 (without suffix).
    output_name_cond2 : str
        Base name for the TFR file corresponding to condition 2 (without suffix).
    rescaled : bool
        If True, load the rescaled (baseline-corrected) TFR; if False, load the uncorrected TFR.
    stat_func : callable, optional
        The statistical function for comparing the two datasets (default: mean_diff).
    p_thresh : float, optional
        The p-value threshold for significance (default: 0.05).
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_perm : int, optional
        The number of permutations to perform (default: 100).
    n_jobs : int, optional
        The number of parallel jobs (default: 1).

    Returns
    -------
    mask : np.ndarray
        A boolean array indicating significant clusters.
    pvals : np.ndarray
        An array of p-values corresponding to each identified cluster.

    Example
    -------
    >>> # Assuming layout is set and the appropriate precomputed TFR files exist:
    >>> mask, pvals = load_and_get_wavelet_differences('sub-01', layout,
    ...                      'Stimulus_c25and75_fixationCrossBase_0.5sec',
    ...                      'Stimulus_c25and75_fixationCrossBase_0.5sec', False)
    >>> isinstance(mask, np.ndarray)
    True
    """
    spec_condition_1 = load_wavelets(sub, layout, output_name_condition_1, rescaled)
    spec_condition_2 = load_wavelets(sub, layout, output_name_condition_2, rescaled)

    mask, pvals = get_sig_tfr_differences(spec_condition_1, spec_condition_2,
                    stat_func=stat_func, p_thresh=p_thresh,
                    ignore_adjacency=ignore_adjacency, n_perm=n_perm, n_jobs=n_jobs)
    
    return mask, pvals

def load_and_get_sig_multitaper_differences(sub: str, layout, output_name_condition_1: str,
                                     output_name_condition_2: str, rescaled: bool,
                                     stat_func: callable = mean_diff, p_thresh: float = 0.05,
                                     ignore_adjacency: int = 1, n_perm: int = 100, n_jobs: int = 1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed multitaper TFRs for two conditions and compute their sig differences.

    This function loads precomputed multitaper time-frequency representations for two conditions using the provided
    output names and a rescaling flag. It then performs a permutation cluster test to determine statistically
    significant differences between the conditions.

    Parameters
    ----------
    sub : str
        The subject identifier.
    layout : BIDSLayout
        The BIDS layout object containing the data.
    output_name_cond1 : str
        Base name for the TFR file corresponding to condition 1 (without suffix).
    output_name_cond2 : str
        Base name for the TFR file corresponding to condition 2 (without suffix).
    rescaled : bool
        If True, load the rescaled (baseline-corrected) TFR; if False, load the uncorrected TFR.
    stat_func : callable, optional
        The statistical function for comparing the two datasets (default: mean_diff).
    p_thresh : float, optional
        The p-value threshold for significance (default: 0.05).
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_perm : int, optional
        The number of permutations to perform (default: 100).
    n_jobs : int, optional
        The number of parallel jobs (default: 1).

    Returns
    -------
    mask : np.ndarray
        A boolean array indicating significant clusters.
    pvals : np.ndarray
        An array of p-values corresponding to each identified cluster.

    Example
    -------
    >>> # Assuming layout is set and the appropriate precomputed TFR files exist:
    >>> mask, pvals = load_and_get_multitaper_differences('sub-01', layout,
    ...                      'Stimulus_c25and75_fixationCrossBase_0.5sec',
    ...                      'Stimulus_c25and75_fixationCrossBase_0.5sec', False)
    >>> isinstance(mask, np.ndarray)
    True
    """
    spec_condition_1 = load_multitaper(sub, layout, output_name_condition_1, rescaled)
    spec_condition_2 = load_multitaper(sub, layout, output_name_condition_2, rescaled)

    mask, pvals = get_sig_tfr_differences(spec_condition_1, spec_condition_2,
                    stat_func=stat_func, p_thresh=p_thresh,
                    ignore_adjacency=ignore_adjacency, n_perm=n_perm, n_jobs=n_jobs)
    
    return mask, pvals

def load_tfrs(filename: str):
    """
    Load a time-frequency representation (TFR) object from an HDF5 file.

    This function uses mne.time_frequency.read_tfrs to read a TFR instance from the
    specified file. Since the reader may return a list of TFR objects, this function
    checks the output and returns the first TFR object if a list is provided, or the
    TFR object directly if not.

    Parameters
    ----------
    filename : str
        The full path to the TFR HDF5 file.

    Returns
    -------
    tfr : instance of mne.time_frequency.EpochsTFR or mne.average.Evoked
        The loaded time-frequency representation object.

    Examples
    --------
    >>> # For testing purposes, suppose we have a file 'dummy-tfr.h5' that contains a TFR.
    >>> # Since we cannot actually read a file in this doctest, we'll simulate the behavior:
    >>> class DummyTFR:
    ...     pass
    >>> def dummy_read_tfrs(fname):
    ...     if fname == "dummy-tfr.h5":
    ...         return [DummyTFR()]  # simulate list return
    ...     return DummyTFR()
    >>> mne.time_frequency.read_tfrs = dummy_read_tfrs  # monkey patch for testing
    >>> tfr_obj = load_tfrs("dummy-tfr.h5")
    >>> isinstance(tfr_obj, DummyTFR)
    True
    """
    # read in the tfr instance
    tfr_result = mne.time_frequency.read_tfrs(filename)
    if isinstance(tfr_result, list):
        # If it's a list, pick the first TFR object
        tfr = tfr_result[0]
    else:
        # Otherwise it's already just a single TFR object
        tfr = tfr_result

    return tfr


def load_wavelets(sub: str, layout, epochs_root_file: str, rescaled: bool = False):
    """
    Load precomputed wavelet time-frequency representations for a subject.

    This function constructs the file path for a wavelet TFR stored in a BIDS-style derivatives
    directory and loads it using the `load_tfrs` function. If `rescaled` is True, the function loads
    the rescaled, baseline corrected TFR file (with suffix "_rescaled-tfr.h5"). Otherwise, it loads the
    uncorrected TFR file (with suffix "_uncorrected-tfr.h5").

    Parameters
    ----------
    sub : str
        The subject identifier (e.g., 'sub-01').
    layout : object
        An object with a 'root' attribute pointing to the BIDS dataset root directory.
    epochs_root_file : str
        The base name for the epochs root file (without the suffix).
    rescaled : bool, optional
        If True, load the rescaled, baseline corrected TFR object.
        If False, load the uncorrected TFR object. Default is False.

    Returns
    -------
    spec : instance of mne.time_frequency.EpochsTFR or similar
        The loaded wavelet time-frequency representation.

    Examples
    --------
    >>> # Create a dummy layout object with a 'root' attribute
    >>> class DummyLayout:
    ...     def __init__(self, root):
    ...         self.root = root
    >>> layout = DummyLayout('/tmp/bids')
    >>> import os
    >>> # For an uncorrected object, the expected filename:
    >>> expected_uncorrected = os.path.join('/tmp/bids', 'derivatives', 'spec', 'wavelet', 'D0057', 'example_uncorrected-tfr.h5')
    >>> # For a rescaled object, the expected filename:
    >>> expected_rescaled = os.path.join('/tmp/bids', 'derivatives', 'spec', 'wavelet', 'D0057', 'example_rescaled-tfr.h5')
    >>> # Dummy TFR class for testing
    >>> class DummyTFR:
    ...     pass
    >>> # Dummy function to simulate mne.time_frequency.read_tfrs
    >>> def dummy_read_tfrs(fname):
    ...     if fname == expected_uncorrected or fname == expected_rescaled:
    ...         return DummyTFR()
    ...     return None
    >>> import mne.time_frequency
    >>> mne.time_frequency.read_tfrs = dummy_read_tfrs  # Monkey-patch for testing
    >>> # Test uncorrected loading
    >>> spec_uncorrected = load_wavelets('sub-01', layout, 'example', rescaled=False)
    >>> isinstance(spec_uncorrected, DummyTFR)
    True
    >>> # Test rescaled loading
    >>> spec_rescaled = load_wavelets('sub-01', layout, 'example', rescaled=True)
    >>> isinstance(spec_rescaled, DummyTFR)
    True
    """
    if rescaled:
        filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub, f'{epochs_root_file}_rescaled-tfr.h5')
    else:
        filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub, f'{epochs_root_file}_uncorrected-tfr.h5')

    spec = load_tfrs(filename)
    return spec

def load_multitaper(sub: str, layout, epochs_root_file: str, rescaled: bool = False):
    """
    Load precomputed multitaper time-frequency representations for a subject.

    This function constructs the file path for a multitaper TFR stored in a BIDS-style derivatives
    directory and loads it using the `load_tfrs` function. If `rescaled` is True, the function loads
    the rescaled, baseline corrected TFR file (with suffix "_rescaled-tfr.h5"). Otherwise, it loads the
    uncorrected TFR file (with suffix "_uncorrected-tfr.h5").

    Parameters
    ----------
    sub : str
        The subject identifier (e.g., 'sub-01').
    layout : object
        An object with a 'root' attribute pointing to the BIDS dataset root directory.
    epochs_root_file : str
        The base name for the epochs root file (without the suffix).
    rescaled : bool, optional
        If True, load the rescaled, baseline corrected TFR object.
        If False, load the uncorrected TFR object. Default is False.

    Returns
    -------
    spec : instance of mne.time_frequency.EpochsTFR or similar
        The loaded multitaper time-frequency representation.

    Examples
    --------
    >>> # Create a dummy layout object with a 'root' attribute
    >>> class DummyLayout:
    ...     def __init__(self, root):
    ...         self.root = root
    >>> layout = DummyLayout('/tmp/bids')
    >>> import os
    >>> # For an uncorrected object, the expected filename:
    >>> expected_uncorrected = os.path.join('/tmp/bids', 'derivatives', 'spec', 'wavelet', 'D0057', 'example_uncorrected-tfr.h5')
    >>> # For a rescaled object, the expected filename:
    >>> expected_rescaled = os.path.join('/tmp/bids', 'derivatives', 'spec', 'wavelet', 'D0057', 'example_rescaled-tfr.h5')
    >>> # Dummy TFR class for testing
    >>> class DummyTFR:
    ...     pass
    >>> # Dummy function to simulate mne.time_frequency.read_tfrs
    >>> def dummy_read_tfrs(fname):
    ...     if fname == expected_uncorrected or fname == expected_rescaled:
    ...         return DummyTFR()
    ...     return None
    >>> import mne.time_frequency
    >>> mne.time_frequency.read_tfrs = dummy_read_tfrs  # Monkey-patch for testing
    >>> # Test uncorrected loading
    >>> spec_uncorrected = load_multitaper('sub-01', layout, 'example', rescaled=False)
    >>> isinstance(spec_uncorrected, DummyTFR)
    True
    >>> # Test rescaled loading
    >>> spec_rescaled = load_multitaper('sub-01', layout, 'example', rescaled=True)
    >>> isinstance(spec_rescaled, DummyTFR)
    True
    """
    if rescaled:
        filename = os.path.join(layout.root, 'derivatives', 'spec', 'multitaper', sub, f'{epochs_root_file}_rescaled-tfr.h5')
    else:
        filename = os.path.join(layout.root, 'derivatives', 'spec', 'multitaper', sub, f'{epochs_root_file}_uncorrected-tfr.h5')

    spec = load_tfrs(filename)
    return spec

def plot_mask_pages(mask: np.ndarray, ch_names: List[str],
                    times: Optional[np.ndarray] = None,
                    freqs: Optional[np.ndarray] = None,
                    channels_per_page: int = 60,
                    grid_shape: Optional[Tuple[int, int]] = None,
                    cmap: str = 'gray', title_prefix: str = "",
                    log_freq: bool = False,
                    show: bool = False,
                    colorbar_range: Optional[Tuple[float, float]] = None) -> List[plt.Figure]:
    """
    Create multiple figures to visualize a 3D mask array (channels × frequencies × times) divided into pages.
    
    This function is useful for visualizing channel-specific time-frequency data or masks from
    statistical tests, especially when there are many channels that cannot fit on a single figure.
    
    Parameters
    ----------
    mask : np.ndarray
        3D array of shape (n_channels, n_freqs, n_times) containing the data to plot.
    ch_names : List[str]
        List of channel names corresponding to the first dimension of the mask.
    times : Optional[np.ndarray], default=None
        Array of time points in seconds. If None, uses sample indices.
    freqs : Optional[np.ndarray], default=None
        Array of frequency values in Hz. If None, uses frequency bin indices.
    channels_per_page : int, default=60
        Maximum number of channels to display on each page.
    grid_shape : Optional[Tuple[int, int]], default=None
        Tuple specifying the (rows, columns) grid layout for each page.
        If None, an approximately square grid is computed automatically.
    cmap : str, default='gray'
        Colormap to use for the plots.
    title_prefix : str, default=""
        Optional prefix to add before each channel name in subplot titles.
    log_freq : bool, default=False
        Whether to use logarithmic scaling for the frequency axis.
    show : bool, default=False
        Whether to display each figure immediately after creation.
    colorbar_range : Optional[Tuple[float, float]], default=None
        Tuple specifying (vmin, vmax) range for the colorbar. If None, 
        the range is determined automatically based on the data.
    
    Returns
    -------
    List[plt.Figure]
        List of matplotlib Figure objects, one for each page.
    
    Notes
    -----
    - Each page contains a grid of subplots, with each subplot showing the 
      time-frequency data for a single channel.
    - If both times and freqs are provided, the x and y axes will be labeled with
      the appropriate units (seconds and Hz).
    - For log-scaled frequency plots, custom tick positions are generated on a 
      logarithmic scale.
    - Each subplot includes its own colorbar for intensity reference.
    """
    n_channels = mask.shape[0]
    pages = []
    
    if grid_shape is not None:
        n_rows, n_cols = grid_shape
        channels_per_page = n_rows * n_cols

    if times is not None:
        x_min, x_max = times[0], times[-1]
        xlabel = "Time (s)"
    else:
        x_min, x_max = 0, mask.shape[-1]
        xlabel = "Time (samples)"
    if freqs is not None:
        y_min, y_max = freqs[0], freqs[-1]
        ylabel = "Frequency (Hz)"
    else:
        y_min, y_max = 0, mask.shape[1]
        ylabel = "Frequency (bins)"

    for start in range(0, n_channels, channels_per_page):
        end = min(start + channels_per_page, n_channels)
        page_mask = mask[start:end]
        page_ch_names = ch_names[start:end]
        n_page_ch = page_mask.shape[0]
        
        if grid_shape is None:
            n_rows = int(np.floor(np.sqrt(n_page_ch)))
            n_rows = max(n_rows, 1)
            n_cols = int(np.ceil(n_page_ch / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (chan_mask, chan_name) in enumerate(zip(page_mask, page_ch_names)):
            ax = axes[i]
            extent = [x_min, x_max, y_min, y_max]
            # If a colorbar_range is provided, unpack vmin and vmax
            if colorbar_range is not None:
                vmin, vmax = colorbar_range
            else:
                vmin, vmax = None, None
            im = ax.imshow(chan_mask, aspect='auto', origin='lower', cmap=cmap,
                           extent=extent, vmin=vmin, vmax=vmax)
            # Add a colorbar next to each subplot
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{title_prefix}{chan_name}", fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both', which='major', labelsize=7, direction="in")
            if log_freq and (freqs is not None):
                ax.set_yscale('log')
                n_ticks = 12
                tick_positions = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_ticks)
                ax.set_yticks(tick_positions)
                ax.set_yticklabels([f"{tick:.1f}" for tick in tick_positions])
        
        # Turn off any unused subplots.
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        pages.append(fig)
        if show:
            plt.show()
    return pages

def load_and_get_sig_wavelet_ratio_differences(sub: str, layout, 
                                                 output_name_condition_1: str,
                                                 output_name_condition_2: str,
                                                 rescaled: bool,
                                                 stat_func: callable = mean_diff, 
                                                 p_thresh: float = 0.05,
                                                 ignore_adjacency: int = 1, 
                                                 n_perm: int = 100, 
                                                 n_jobs: int = 1
                                                 ) -> Tuple[np.ndarray, np.ndarray, mne.time_frequency.EpochsTFR]:
    """
    Load precomputed wavelet TFRs for two conditions from file, compute their ratio,
    convert that ratio to decibels, and perform a one-sample permutation cluster test 
    comparing the ratio (in dB) against 0.

    The ratio is computed as:
        ratio_dB = 20 * log10( condition1 / condition2 )
    so that a value of 0 dB indicates no difference between conditions.
    """
    # Load precomputed TFRs for each condition using the load_wavelets function.
    spec_condition_1 = load_wavelets(sub, layout, output_name_condition_1, rescaled)
    spec_condition_2 = load_wavelets(sub, layout, output_name_condition_2, rescaled)
    
    # Ensure both TFRs have the same number of trials by trimming the one with more trials.
    n_trials = min(spec_condition_1._data.shape[0], spec_condition_2._data.shape[0])
    data1 = spec_condition_1._data[:n_trials]
    data2 = spec_condition_2._data[:n_trials]
    
    # Compute the element-wise ratio and convert to decibels:
    #   ratio_dB = 20 * log10( condition1 / condition2 )
    ratio_data = 20 * np.log10(data1 / data2)
    
    # Create a new TFR object to hold the ratio, preserving metadata from one of the conditions.
    ratio_spec = spec_condition_1.copy()
    ratio_spec._data = ratio_data
    
    # To test if the ratio is significantly different from 0, create a zero array.
    zeros = np.zeros_like(ratio_data)
    
    # Run a permutation cluster test comparing the ratio (in dB) to 0.
    mask, pvals = time_perm_cluster(ratio_data, zeros,
                                    stat_func=stat_func,
                                    p_thresh=p_thresh,
                                    ignore_adjacency=ignore_adjacency,
                                    n_perm=n_perm,
                                    n_jobs=n_jobs)
    return mask, pvals, ratio_spec
