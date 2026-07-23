"""Shared helpers for the ``make_epoched_data*`` preprocessing scripts.

These functions were previously duplicated verbatim across
``make_epoched_data.py``, ``make_epoched_data_saved.py``, and
``make_epoched_data_with_phase.py``. They now live here so a single
definition is imported by all three.
"""
import random

import mne
import numpy as np
from ieeg.navigate import trial_ieeg

try:
    # Only needed by extract_amplitude_and_phase_and_freqs. Imported lazily so
    # callers that don't need it (e.g. make_epoched_data.py) still work if
    # naplib isn't installed.
    from naplib.preprocessing import filterbank_hilbert as fb_hilb
except ImportError:  # pragma: no cover
    fb_hilb = None


def shuffle_array(arr):
    """Shuffle ``arr`` in place along its first axis and return it.

    ``np.random.shuffle`` mutates its argument and returns ``None``, so the
    array itself is returned here for convenience.
    """
    np.random.shuffle(arr)
    return arr


def extract_amplitude_and_phase_and_freqs(data, fs=None,
            passband: tuple[int, int] = (70, 150), copy: bool = True,
            n_jobs=-1, verbose: bool = True):
    """
    Extract gamma band envelope, phase, and center frequencies from data.
    Supports both numpy arrays and MNE Epochs/Raw.
    """
    if fb_hilb is None:
        raise ImportError(
            "naplib is required for extract_amplitude_and_phase_and_freqs "
            "(pip install naplib)."
        )

    if hasattr(data, 'get_data'):
        if fs is None:
            fs = data.info['sfreq']
        in_data = data.get_data()
    else:
        if fs is None:
            raise ValueError("fs must be provided if data is not a Signal")
        in_data = data.copy() if copy else data

    passband = list(passband)
    env = np.zeros(in_data.shape)
    phase = np.zeros(in_data.shape)

    if in_data.ndim == 3:
        for idx in range(in_data.shape[0]):
            trial_data = in_data[idx]  # shape: (channels, times)
            x_phase, x_envelope, freqs = fb_hilb(trial_data.T, fs, passband, n_jobs)
            phase[idx] = np.sum(x_phase, axis=-1).T
            env[idx]   = np.sum(x_envelope, axis=-1).T
    elif in_data.ndim == 2:
        x_phase, x_envelope, freqs = fb_hilb(in_data.T, fs, passband, n_jobs)
        phase = np.sum(x_phase, axis=-1).T
        env   = np.sum(x_envelope, axis=-1).T
    else:
        raise ValueError(f"Unsupported data dimensions: {in_data.ndim}")

    return env, phase, freqs


def trial_ieeg_rand_offset(raw: mne.io.Raw, event: str | list[str, ...], within_times: tuple[float,float], times_length: float, pad_length: float,
               verbose=None, **kwargs) -> mne.Epochs:
    """Epochs data from a mne Raw iEEG instance.

    Takes a mne Raw instance and randomly epochs the data around a specified event, for each instance of the event,
    for a duration of times_length, within a range of within_times.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to epoch.
    event : str
        The event to epoch around.
    within_times : tuple[float, float]
        The time window within which to randomly select intervals for each event.
    times_length : float,
        The length of the time intervals to randomly select within `within_times`.
    pad_length : float,
        The length to pad each time interval. Will be removed later.
    %(picks_all)s
    %(reject_epochs)s
    %(flat)s
    %(decim)s
    %(epochs_reject_tmin_tmax)s
    %(detrend_epochs)s
    %(proj_epochs)s
    %(on_missing_epochs)s
    %(verbose)s

    Returns
    -------
    mne.Epochs
        The epoched data.
    """

    sfreq = raw.info['sfreq'] #raw.info in function

    # get padded within times and times_length
    within_times_padded = [within_times[0] - pad_length, within_times[1] + pad_length]
    times_length_padded = times_length + 2 * pad_length

    # Convert times to samples
    within_times_samples = [int(t * sfreq) for t in within_times_padded]
    times_length_samples = int((times_length_padded) * sfreq)

    # Shift the indices to be positive
    shift = abs(within_times_samples[0])
    within_times_samples_pos = [s + shift for s in within_times_samples]

    trials = trial_ieeg(raw, event, within_times_padded, preload=True, reject_by_annotation=False)

    rand_offset_data = []

    # Randomly select subsets for each trial
    for trial in trials.get_data():
        start_sample = random.randint(within_times_samples_pos[0], within_times_samples_pos[1] - times_length_samples)
        end_sample = start_sample + times_length_samples
        rand_offset_data.append(trial[:, start_sample:end_sample+1]) #across all channels, grab this time subset

    # Reassign data to rand_offset_trials and adjust the times in rand_offset_trials
    new_tmin = within_times_padded[0]
    new_tmax = new_tmin + times_length_padded
    rand_offset_trials = trial_ieeg(raw, event, [new_tmin, new_tmax], preload=True, reject_by_annotation=False)
    rand_offset_trials._data = np.array(rand_offset_data)

    return rand_offset_trials
