"""Cross-tabulate per-electrode power against block type.

In a blocked design, condition and block are confounded: every trial of a given
condition comes from a particular block type, so anything that is wrong with one
block's recording arrives pre-labelled as a condition effect. It shows up before
the stimulus, where no real effect can be, and it lands on the proportion
interaction terms.

The two questions this answers, per electrode:

* **Is there a block-locked baseline offset?** One block sitting at a different
  pre-stimulus level than the other three is the mechanism behind a cluster that
  spans the baseline. ``block_offset`` measures it in z units.
* **Is there a block-locked excursion?** A single enormous sample in one block's
  trials moves the across-electrode mean by ``excursion / n_electrodes``, which
  over a large ROI is enough to draw a visible spike.

Deliberately free of MNE and ieeg imports: the entrypoint loads the epochs and
hands over plain arrays, so the logic here is testable without the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_mask(times, window):
    """Boolean mask over ``times`` for ``window=(tmin, tmax)``.

    Either bound may be ``None`` for "the edge of the epoch". Returns an
    all-True mask when the window selects nothing, so a caller that passes a
    window outside the epoch scores the whole epoch instead of an empty slice.
    """
    times = np.asarray(times, dtype=float)
    if window is None:
        return np.ones(times.shape, dtype=bool)
    tmin, tmax = window
    mask = np.ones(times.shape, dtype=bool)
    if tmin is not None:
        mask &= times >= tmin
    if tmax is not None:
        mask &= times <= tmax
    return mask if mask.any() else np.ones(times.shape, dtype=bool)


def max_abs_z_per_trial(data, times=None, window=None):
    """Largest ``|z|`` each (trial, channel) reaches anywhere in the window.

    The statistic the ``max_abs_z`` pass in ``make_epoched_data.py`` rejects on,
    factored out so the diagnostics that choose the threshold score the same
    thing the pipeline acts on. A value read off
    ``src/analysis/vis/trial_z_distribution_vis.py`` therefore means exactly
    what it will mean when passed as ``--max_abs_z``.

    Parameters
    ----------
    data : (n_trials, n_channels, n_times) array
        Baseline-rescaled epoch data, NaNs allowed. Plain array rather than an
        Epochs object, per this module's no-MNE rule.
    times : (n_times,) array, optional
        Required only when ``window`` is given.
    window : (tmin, tmax), optional
        Restrict to a window, in seconds. Defaults to the whole epoch, which is
        what the rejection in ``make_epoched_data.py`` uses. A window selecting
        nothing falls back to the whole epoch, per :func:`window_mask`.

    Returns
    -------
    (n_trials, n_channels) array
        NaN for a (trial, channel) pair that is already entirely NaN, i.e. one
        the raw-voltage ``outliers_to_nan`` pass has already removed. NaN
        compares False against any threshold, so such a pair is never counted
        as newly rejected.
    """
    values = np.abs(np.asarray(data, dtype=float))
    if values.ndim != 3:
        raise ValueError("data must have shape (n_trials, n_channels, n_times)")
    if window is not None:
        if times is None:
            raise ValueError("times is required to apply a window")
        values = values[:, :, window_mask(times, window)]

    # np.nanmax over an all-NaN pair warns and the caller has to special-case
    # the result anyway, so fill with -inf, take a plain max, and restore NaN
    # where there was nothing to score.
    all_nan = np.all(np.isnan(values), axis=2)
    scored = np.where(np.isnan(values), -np.inf, values).max(axis=2)
    return np.where(all_nan, np.nan, scored)


def block_deviation_table(data, block_labels, times, ch_names=None,
                          baseline_window=(None, 0.0)):
    """One row per (channel, block) describing that block's contribution.

    Parameters
    ----------
    data : (n_trials, n_channels, n_times) array
        Single-subject epoch data, NaNs allowed.
    block_labels : sequence of length n_trials
        Block identity per trial, e.g. ``(75.0, 75.0)`` for 75% incongruent /
        75% switch. Any hashable label works.
    times : (n_times,) array
    ch_names : sequence of str, optional
    baseline_window : (tmin, tmax)
        Window the statistics are computed over. Defaults to epoch start
        through stimulus onset -- the period where a condition difference
        cannot be a real effect.

    Returns
    -------
    pandas.DataFrame with columns:
        channel, block, n_trials, n_valid_first_sample, baseline_mean,
        peak_abs_trial, peak_abs_trial_time
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must have shape (n_trials, n_channels, n_times)")
    n_trials, n_channels, n_times = data.shape
    if len(block_labels) != n_trials:
        raise ValueError(
            f"block_labels has {len(block_labels)} entries for {n_trials} trials"
        )
    times = np.asarray(times, dtype=float)
    if times.shape[0] != n_times:
        raise ValueError(f"times has {times.shape[0]} entries for {n_times} samples")
    if ch_names is None:
        ch_names = [f"channel_{i}" for i in range(n_channels)]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names has {len(ch_names)} entries for {n_channels} channels"
        )

    mask = window_mask(times, baseline_window)
    windowed = data[:, :, mask]
    windowed_times = times[mask]

    labels = pd.Series(list(block_labels))
    rows = []
    with np.errstate(invalid='ignore'):
        for block in sorted(labels.dropna().unique(), key=repr):
            trial_idx = np.flatnonzero((labels == block).to_numpy())
            if trial_idx.size == 0:
                continue
            block_data = windowed[trial_idx]                 # (trials, ch, t)

            # Per-channel mean over trials and time: the block's tonic level in
            # the window. This is the quantity a block-locked offset moves.
            baseline_mean = np.nanmean(block_data, axis=(0, 2))

            # Largest single-trial excursion, and when it happened. The mean can
            # look fine while one trial carries a value that dominates a small
            # condition cell.
            abs_data = np.abs(block_data)
            all_nan = np.all(np.isnan(abs_data), axis=(0, 2))
            flat = abs_data.reshape(abs_data.shape[0], n_channels, -1)
            peak_abs = np.where(all_nan, np.nan,
                                np.nanmax(np.nan_to_num(abs_data, nan=-np.inf),
                                          axis=(0, 2)))
            peak_time = np.full(n_channels, np.nan)
            for ch in range(n_channels):
                if all_nan[ch]:
                    continue
                collapsed = np.nanmax(np.nan_to_num(flat[:, ch, :], nan=-np.inf),
                                      axis=0)
                peak_time[ch] = windowed_times[int(np.argmax(collapsed))]

            # How many trials actually contribute at the epoch's first sample.
            # A "mean" backed by one trial is not an average.
            n_valid_first = np.sum(~np.isnan(data[trial_idx, :, 0]), axis=0)

            for ch in range(n_channels):
                rows.append({
                    'channel': ch_names[ch],
                    'block': block,
                    'n_trials': int(trial_idx.size),
                    'n_valid_first_sample': int(n_valid_first[ch]),
                    'baseline_mean': float(baseline_mean[ch]),
                    'peak_abs_trial': float(peak_abs[ch]),
                    'peak_abs_trial_time': float(peak_time[ch]),
                })

    return pd.DataFrame(rows)


def rank_block_confined_channels(table):
    """Rank channels by how much one block departs from the others.

    ``block_offset`` is the largest absolute departure of a block's
    ``baseline_mean`` from the median across that channel's blocks, in z units.
    The median is the reference because with four blocks a single bad one cannot
    drag it, so the score measures the outlier against the well-behaved rest.

    ``peak_ratio`` does the same for the largest single-trial excursion, as a
    ratio rather than a difference: a channel whose worst block carries a trial
    twenty times bigger than any other block's is a block-locked artifact even
    when its mean looks unremarkable.

    Returns one row per channel, most block-confined first.
    """
    if table.empty:
        return pd.DataFrame(columns=['channel', 'block_offset', 'worst_block',
                                     'peak_ratio', 'peak_block', 'n_blocks'])

    rows = []
    for channel, group in table.groupby('channel', sort=False):
        means = group['baseline_mean'].to_numpy(dtype=float)
        peaks = group['peak_abs_trial'].to_numpy(dtype=float)
        blocks = list(group['block'])

        with np.errstate(invalid='ignore'):
            median = np.nanmedian(means)
            departures = np.abs(means - median)
            worst = int(np.nanargmax(departures)) if not np.all(np.isnan(departures)) else 0

            peak_median = np.nanmedian(peaks)
            if peak_median and np.isfinite(peak_median) and peak_median > 0:
                ratios = peaks / peak_median
            else:
                ratios = np.full_like(peaks, np.nan)
            peak_worst = int(np.nanargmax(ratios)) if not np.all(np.isnan(ratios)) else 0

        rows.append({
            'channel': channel,
            'block_offset': float(departures[worst]),
            'worst_block': blocks[worst],
            'peak_ratio': float(ratios[peak_worst]),
            'peak_block': blocks[peak_worst],
            'n_blocks': len(blocks),
        })

    out = pd.DataFrame(rows)
    return out.sort_values('peak_ratio', ascending=False,
                           na_position='last').reset_index(drop=True)


def block_labels_from_metadata(metadata,
                               columns=('incongruent_proportion', 'switch_proportion')):
    """Build one hashable block label per trial from epochs metadata.

    Returns ``(labels, missing_columns)``. When a column is absent the label
    falls back to whatever is available, so a caller can still group by the
    other factor rather than failing outright.
    """
    present = [c for c in columns if c in metadata.columns]
    missing = [c for c in columns if c not in metadata.columns]
    if not present:
        raise KeyError(
            f"metadata has none of {list(columns)}; columns are "
            f"{list(metadata.columns)}"
        )
    labels = [tuple(row) for row in metadata[present].to_numpy()]
    return labels, missing
