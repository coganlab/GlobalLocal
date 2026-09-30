"""Overall-activity control for pseudopopulation decoding.

The question
------------
Incongruent and switch trials are both harder, and a harder trial can simply
raise high gamma on most electrodes at once. A decoder trained on congruency
would then learn "overall activity is up", and that axis would also separate
switch from repeat: a transfer driven by a shared GAIN, not a shared
representational pattern. This module asks how much of a decode (or a
transfer) that uniform component carries.

Two transforms, applied to the decoded arrays before decoding:

- `remove_mean`: for every subject, subtract the mean across that subject's
  decoded electrodes, per pseudo-trial and per time point. The part of the
  signal that is the same on all of a subject's electrodes is gone; what is
  left is the pattern across electrodes. If a transfer survives this (and keeps
  a similar share of its ceiling), it is not carried by a uniform gain.
- `mean_only`: replace each subject's electrodes by their mean, one feature per
  subject. This is the uniform component alone. If it transfers about as well as
  the full pattern, overall activity is enough to explain the transfer.

Per subject, because a pseudo-trial concatenates DIFFERENT physical trials from
each subject: the mean across all channels of a row would mix trials. Channel
names follow the pseudopopulation convention `<subject>-<electrode>`
(`put_data_in_labeled_array_per_roi_subject`); names without a '-' are treated
as one subject (the synthetic data).

Limits worth stating when reporting it:
- `remove_mean` removes only a shift shared by ALL of a subject's decoded
  electrodes. A gain on a subset of them survives it and still reads as
  "pattern".
- A subject with a single decoded electrode has nothing left after
  `remove_mean` (its channel becomes 0); the info dict counts them.
- Read every result against its own ceiling from the same transformed data (the
  ceilings change too), i.e. compare the `retained` shares, not raw accuracy.

Entry points: `apply_activity_control(roi_arrays, roi, channel_names, mode)` for
the `{roi: {condition: (trials, channels, time)}}` dicts the decoders use, and
`remove_subject_mean` / `subject_mean` for a single array.
"""

from __future__ import annotations

import warnings

import numpy as np

ACTIVITY_CONTROLS = ('none', 'remove_mean', 'mean_only')


def channel_subjects(channel_names):
    """'D0057-LFMI8' -> 'D0057', per channel. Names without a '-' map to ''."""
    return [str(name).split('-', 1)[0] if '-' in str(name) else ''
            for name in channel_names]


def _subject_index(subjects):
    """Subjects in first-seen order, and the channel indices of each."""
    order = list(dict.fromkeys(subjects))
    subjects = np.asarray(subjects, dtype=object)
    return order, {s: np.flatnonzero(subjects == s) for s in order}


def _nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN rows stay NaN
        return np.nanmean(x, axis=axis, keepdims=True)


def remove_subject_mean(x, subjects, chans_axis=1):
    """Each subject's channels minus their mean across those channels.

    `x` is any array with channels on `chans_axis` (usually trials x channels x
    time); the mean is taken per trial and time point, NaN-aware. A row in which
    all of a subject's channels are NaN stays NaN; a channel that is NaN stays
    NaN. Returns a new float array of the same shape.
    """
    x = np.asarray(x, dtype=float)
    if len(subjects) != x.shape[chans_axis]:
        raise ValueError(f"{len(subjects)} subject labels for {x.shape[chans_axis]} channels")
    out = x.copy()
    _, index = _subject_index(subjects)
    for idx in index.values():
        block = np.take(x, idx, axis=chans_axis)
        shifted = block - _nanmean(block, chans_axis)
        slicer = [slice(None)] * x.ndim
        slicer[chans_axis] = idx
        out[tuple(slicer)] = shifted
    return out


def subject_mean(x, subjects, chans_axis=1):
    """Each subject's mean across its channels, as one channel per subject.

    Returns `(array, subject_order)`; the array has `len(subject_order)` channels
    in first-seen order.
    """
    x = np.asarray(x, dtype=float)
    if len(subjects) != x.shape[chans_axis]:
        raise ValueError(f"{len(subjects)} subject labels for {x.shape[chans_axis]} channels")
    order, index = _subject_index(subjects)
    means = [_nanmean(np.take(x, index[s], axis=chans_axis), chans_axis) for s in order]
    return np.concatenate(means, axis=chans_axis), order


def apply_activity_control(roi_arrays, roi, channel_names, mode, chans_axis=1):
    """Apply `mode` to every condition array of one ROI.

    Parameters
    ----------
    roi_arrays : {roi: {condition: array}} (arrays or LabeledArrays, channels on
        `chans_axis`), e.g. the pseudopopulation dict, or an electrode group
        restricted from it.
    channel_names : the arrays' channels, in order (`<subject>-<electrode>`).
    mode : 'none' | 'remove_mean' | 'mean_only'.

    Returns
    -------
    (arrays, channel_names, info). For 'none' the input dict is returned as is.
    For 'mean_only' the channels become one per subject, named '<subject>-mean'.
    `info` records the mode, electrode and subject counts, the number of features
    decoded, and (for 'remove_mean') the subjects with a single electrode, whose
    channel carries nothing after the mean is removed.
    """
    if mode not in ACTIVITY_CONTROLS:
        raise ValueError(f"activity control must be one of {ACTIVITY_CONTROLS}; got {mode!r}")
    channel_names = list(channel_names)
    subjects = channel_subjects(channel_names)
    order, index = _subject_index(subjects)
    single = sorted(s for s in order if len(index[s]) == 1)
    info = dict(mode=mode, n_electrodes=len(channel_names), n_subjects=len(order),
                n_features=len(channel_names), single_electrode_subjects=[])
    if mode == 'none':
        return roi_arrays, channel_names, info

    out = {}
    for condition, arr in roi_arrays[roi].items():
        arr = np.asarray(arr)
        if arr.shape[chans_axis] != len(channel_names):
            raise ValueError(f"condition {condition!r} has {arr.shape[chans_axis]} channels "
                             f"but {len(channel_names)} channel names were given")
        if mode == 'remove_mean':
            out[condition] = remove_subject_mean(arr, subjects, chans_axis)
        else:
            out[condition], _ = subject_mean(arr, subjects, chans_axis)
    if mode == 'remove_mean':
        info['single_electrode_subjects'] = single
        new_names = channel_names
    else:
        new_names = [f'{s}-mean' if s else 'mean' for s in order]
        info['n_features'] = len(new_names)
    return {roi: out}, new_names, info


def describe(info):
    """One line for a summary: what the control did to one electrode group."""
    mode = info['mode']
    if mode == 'none':
        return 'none'
    if mode == 'mean_only':
        return (f"mean_only: {info['n_electrodes']} electrodes -> {info['n_features']} "
                f"per-subject means")
    single = info['single_electrode_subjects']
    return (f"remove_mean over {info['n_subjects']} subjects"
            + (f" ({len(single)} with one electrode, which carry nothing after it: "
               f"{', '.join(single)})" if single else ""))
