"""Per-trial z-score diagnostics for baseline-rescaled high gamma.

``make_epoched_data.py`` runs two outlier passes: ``outliers_to_nan`` on raw
voltage before ``gamma.extract``, then an optional absolute ``max_abs_z``
rejection on the baseline-rescaled power. This module is for choosing that
second number by looking at the data rather than guessing.

Two views, both per electrode:

* :func:`plot_trial_traces_over_mean_grid` -- every trial's trace in light gray
  under the across-trial mean, one panel per electrode. Same idiom as the
  electrode traces under the ROI mean in ``power/plots.py``, one level down:
  there the gray lines are electrodes and the colored line is the ROI mean,
  here the gray lines are trials and the colored line is the electrode mean.
* :func:`plot_z_distribution_grid` -- the distribution of each trial's largest
  excursion, which is the statistic ``max_abs_z`` actually thresholds on, so
  the histogram and the cutoff are in the same units.

:func:`summarize_z_threshold_tradeoff` turns that statistic into the table you
need to pick a value: what each candidate threshold would cost in trials.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_Z_STYLE = {
    'trial_trace_color': '#9e9e9e',
    # A few hundred trials at 0.25 saturates into a solid block, so the density
    # of the gray mat stops carrying information. 0.12 keeps it readable.
    'trial_trace_alpha': 0.12,
    'trial_trace_linewidth': 0.5,
    'mean_color': '#D62728',
    'mean_linewidth': 2.0,
    'threshold_color': '#1F77B4',
    'panel_size': (3.4, 2.8),
    'title_font_size': 9,
    'tick_font_size': 7,
    # Trials to name in each panel, ranked by their largest |z|. Naming them is
    # the point: a threshold is only defensible if you can go back and look at
    # the trials it would remove.
    'n_outlier_labels': 3,
    'outlier_label_font_size': 6,
}


def _grid_shape(n, grid_shape=None):
    """Rows/cols for ``n`` panels, roughly square unless told otherwise."""
    if grid_shape is not None:
        return grid_shape
    n_rows = max(1, int(np.floor(np.sqrt(n))))
    n_cols = int(np.ceil(n / n_rows))
    return n_rows, n_cols


def _resolve_picks(epochs, electrodes):
    """Split requested electrode names into present and missing.

    Missing is normal rather than an error: ``drop_and_impute`` and the
    ``--electrodes-to-drop`` list both remove channels upstream, and an
    electrode that is gone is itself an answer about that electrode.
    """
    if electrodes is None:
        return list(epochs.ch_names), []
    present = [ch for ch in electrodes if ch in epochs.ch_names]
    missing = [ch for ch in electrodes if ch not in epochs.ch_names]
    return present, missing


def _window_mask(times, tmin=None, tmax=None):
    """Boolean mask over ``times``; either bound may be None for "epoch edge"."""
    times = np.asarray(times, dtype=float)
    mask = np.ones(times.shape, dtype=bool)
    if tmin is not None:
        mask &= times >= tmin
    if tmax is not None:
        mask &= times <= tmax
    if not mask.any():
        raise ValueError(f'window ({tmin}, {tmax}) selects no samples')
    return mask


def per_trial_max_abs_z(epochs, electrodes=None, tmin=None, tmax=None):
    """Largest ``|z|`` each (trial, channel) reaches anywhere in the window.

    This is exactly the statistic ``make_epoched_data.py`` compares against
    ``max_abs_z``, so a cutoff read off its distribution means the same thing
    there as it does here.

    Parameters
    ----------
    epochs : mne.Epochs
        Baseline-rescaled epochs -- ``HG_ev1_power_rescaled`` for the pass that
        ``max_abs_z`` governs.
    electrodes : list of str or None
        Channels to score. ``None`` scores every channel, which is what you
        want for the reference distribution the threshold has to survive.
    tmin, tmax : float or None
        Restrict to a window, in seconds. ``None`` uses the whole epoch, which
        is what the rejection in ``make_epoched_data.py`` does.

    Returns
    -------
    stat : ndarray, shape (n_trials, n_picks)
        NaN where the (trial, channel) pair is entirely NaN already, i.e. where
        the voltage-level ``outliers_to_nan`` pass (or a previous ``max_abs_z``
        run) has already removed it.
    ch_names : list of str
        Channel names, in ``stat`` column order.
    missing : list of str
        Requested names that are not in ``epochs``.
    """
    picks, missing = _resolve_picks(epochs, electrodes)
    if not picks:
        return np.empty((len(epochs), 0)), [], missing

    data = epochs.get_data(picks=picks)          # (n_trials, n_picks, n_times)
    mask = _window_mask(epochs.times, tmin, tmax)
    data = np.abs(data[:, :, mask])

    # An all-NaN (trial, channel) pair is already-rejected data, not a small
    # value: np.nanmax would warn and return -inf, so score it as NaN and let
    # the callers count it separately.
    all_nan = np.all(np.isnan(data), axis=2)
    stat = np.full(data.shape[:2], np.nan)
    if not all_nan.all():
        with np.errstate(invalid='ignore'):
            filled = np.where(np.isnan(data), -np.inf, data)
            stat_full = filled.max(axis=2)
        stat = np.where(all_nan, np.nan, stat_full)
    return stat, picks, missing


def summarize_z_threshold_tradeoff(stat, ch_names, thresholds):
    """What each candidate ``max_abs_z`` would cost, per threshold.

    Parameters
    ----------
    stat : ndarray, shape (n_trials, n_channels)
        Output of :func:`per_trial_max_abs_z`. NaN entries are already-rejected
        pairs; they are excluded from the denominators so the percentages
        describe the data a new threshold would actually act on.
    ch_names : list of str
    thresholds : sequence of float

    Returns
    -------
    pandas.DataFrame
        One row per threshold: pairs rejected, percent of scorable pairs,
        trials touched on at least one channel, and channels touched.
    """
    import pandas as pd

    scorable = ~np.isnan(stat)
    n_scorable = int(scorable.sum())
    n_trials = stat.shape[0]

    rows = []
    for thr in thresholds:
        over = np.zeros_like(stat, dtype=bool)
        np.greater(stat, thr, out=over, where=scorable)
        n_over = int(over.sum())
        trials_hit = int(over.any(axis=1).sum())
        chans_hit = int(over.any(axis=0).sum())
        rows.append({
            'max_abs_z': thr,
            'pairs_rejected': n_over,
            'pct_of_scorable_pairs': 100 * n_over / n_scorable if n_scorable else np.nan,
            'trials_touched': trials_hit,
            'pct_trials_touched': 100 * trials_hit / n_trials if n_trials else np.nan,
            'channels_touched': chans_hit,
            'channels_touched_names': ', '.join(
                np.asarray(ch_names)[over.any(axis=0)][:8]),
        })
    return pd.DataFrame(rows)


def describe_z_per_channel(stat, ch_names,
                           quantiles=(50, 90, 95, 99, 99.9, 100)):
    """Per-channel quantiles of the per-trial max ``|z|``, plus NaN counts.

    The gap between the 99th percentile and the max is the whole question: a
    channel whose max sits a few z above its 99th has a heavy-tailed but
    continuous distribution, one whose max is two orders of magnitude above it
    has a handful of artifacts sitting on top of an otherwise sane channel.
    """
    import pandas as pd

    rows = []
    for j, ch in enumerate(ch_names):
        col = stat[:, j]
        good = col[~np.isnan(col)]
        row = {
            'channel': ch,
            'n_scorable_trials': int(good.size),
            'n_already_nan': int(np.isnan(col).sum()),
        }
        for q in quantiles:
            row[f'p{q:g}'] = np.percentile(good, q) if good.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_trial_traces_over_mean_grid(epochs, electrodes=None, tmin=None, tmax=None,
                                     thresholds=(), ylim=None, robust_ylim_pct=None,
                                     yscale='linear', title=None, grid_shape=None,
                                     style=None, save_path=None):
    """Grid of single-electrode panels: each trial in gray under the mean.

    Parameters
    ----------
    epochs : mne.Epochs
        Baseline-rescaled epochs; the y axis is in z.
    electrodes : list of str or None
        One panel per electrode. ``None`` plots every channel.
    tmin, tmax : float or None
        Crop the x axis to this window, in seconds.
    thresholds : sequence of float
        Draw dashed horizontal lines at ``+/- thr``, so a candidate cutoff can
        be read against the traces it would remove.
    ylim : tuple or None
        Explicit y limits, applied to every panel.
    robust_ylim_pct : float or None
        Instead of ``ylim``, clip each panel to +/- this percentile of ``|z|``
        over its own trials. Use ~99.5 to see the bulk of the trials; leave
        both None to see the full range, where one artifact flattens
        everything else into a line at zero. Both views are worth a look.
    yscale : {'linear', 'symlog'}
        ``'symlog'`` shows the bulk and the artifacts in one panel.
    grid_shape : tuple or None
        ``(n_rows, n_cols)``. Default is roughly square.
    style : dict or None
        Overrides for :data:`DEFAULT_Z_STYLE`.
    save_path : str or Path or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = {**DEFAULT_Z_STYLE, **(style or {})}
    picks, missing = _resolve_picks(epochs, electrodes)
    if missing:
        print(f'not in epochs (dropped upstream?): {missing}')
    if not picks:
        raise ValueError('none of the requested electrodes are in these epochs')

    times = epochs.times
    mask = _window_mask(times, tmin, tmax)
    times = times[mask]
    data = epochs.get_data(picks=picks)[:, :, mask]   # (n_trials, n_picks, n_times)

    n_rows, n_cols = _grid_shape(len(picks), grid_shape)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(s['panel_size'][0] * n_cols,
                                      s['panel_size'][1] * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for i, ch in enumerate(picks):
        ax = axes[i]
        trials = data[:, i, :]                        # (n_trials, n_times)
        n_valid = int((~np.all(np.isnan(trials), axis=1)).sum())

        ax.plot(times, trials.T, color=s['trial_trace_color'],
                alpha=s['trial_trace_alpha'],
                linewidth=s['trial_trace_linewidth'], zorder=1)
        with warnings.catch_warnings():
            # An electrode whose trials are all NaN is a legitimate outcome
            # here, not something to warn about -- the empty panel says it.
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_trace = np.nanmean(trials, axis=0)
        ax.plot(times, mean_trace, color=s['mean_color'],
                linewidth=s['mean_linewidth'], zorder=3)

        for thr in thresholds:
            for sign in (1, -1):
                ax.axhline(sign * thr, color=s['threshold_color'],
                           linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)

        # Name the biggest trials. Chasing an artifact means going back to a
        # trial number, so the panel has to hand you one. Take the absolute
        # value first and only then fill the NaNs, so a NaN sample can never
        # come back as +inf and win the ranking.
        abs_trials = np.abs(trials)
        filled = np.where(np.isnan(abs_trials), -np.inf, abs_trials)
        per_trial = np.where(np.all(np.isnan(trials), axis=1), np.nan,
                             filled.max(axis=1))
        finite = np.flatnonzero(np.isfinite(per_trial))
        worst = finite[np.argsort(per_trial[finite])[::-1]][:max(0, int(s['n_outlier_labels']))]

        peak = np.nanmax(per_trial) if np.isfinite(per_trial).any() else np.nan
        ax.set_title(f'{ch}  (n={n_valid}, max|z|={peak:.0f})',
                     fontsize=s['title_font_size'])
        ax.axhline(0, color='black', linewidth=0.4, linestyle=':')
        ax.axvline(0, color='black', linewidth=0.4, linestyle=':')
        ax.tick_params(labelsize=s['tick_font_size'])
        if yscale == 'symlog':
            ax.set_yscale('symlog', linthresh=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif robust_ylim_pct is not None and np.isfinite(trials).any():
            lim = np.nanpercentile(np.abs(trials), robust_ylim_pct)
            if np.isfinite(lim) and lim > 0:
                ax.set_ylim(-lim, lim)

        # Label after the limits are final. The worst trial sits at the top of
        # the panel by construction, so a label offset upward from it lands
        # outside the axes and is clipped away -- exactly the one you most want
        # to read. Drop the label below its point when it is near the top, and
        # stagger by rank, since several contacts often peak on the same sample.
        y_lo, y_hi = ax.get_ylim()
        for rank, trial_idx in enumerate(worst):
            t_arg = int(filled[trial_idx].argmax())
            y = trials[trial_idx, t_arg]
            near_top = np.isfinite(y) and y > y_lo + 0.85 * (y_hi - y_lo)
            dy = -(10 + 9 * rank) if near_top else (4 + 9 * rank)
            ax.annotate(f'trial {trial_idx}', (times[t_arg], y),
                        xytext=(4, dy), textcoords='offset points',
                        fontsize=s['outlier_label_font_size'],
                        color='#333333', clip_on=True, zorder=4)

    for j in range(len(picks), len(axes)):
        axes[j].axis('off')

    fig.supxlabel('Time (s)', fontsize=10)
    fig.supylabel('Baseline-rescaled power (z)', fontsize=10)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
    return fig


def plot_z_distribution_grid(stat, ch_names, thresholds=(), bins=40,
                             log_x=True, share_x=True, grid_shape=None,
                             title=None, style=None, save_path=None):
    """Grid of per-electrode histograms of the per-trial max ``|z|``.

    Parameters
    ----------
    stat : ndarray, shape (n_trials, n_channels)
        Output of :func:`per_trial_max_abs_z`.
    ch_names : list of str
    thresholds : sequence of float
        Vertical lines at each candidate cutoff, with the count above it in the
        panel legend.
    log_x : bool
        Bin on log10(z). With excursions spanning 1 to 10,000 z a linear axis
        is one bar at the left edge and nothing else.
    share_x : bool
        One set of bin edges across all panels, so a channel with a detached
        tail is visibly different from one without rather than just rescaled.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = {**DEFAULT_Z_STYLE, **(style or {})}
    n_rows, n_cols = _grid_shape(len(ch_names), grid_shape)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(s['panel_size'][0] * n_cols,
                                      s['panel_size'][1] * n_rows),
                             squeeze=False, sharex=share_x)
    axes = axes.flatten()

    shared_bins = bins
    if share_x:
        pooled = stat[np.isfinite(stat) & (stat > 0)]
        if pooled.size:
            lo, hi = pooled.min(), pooled.max()
            if log_x:
                lo, hi = np.log10(lo), np.log10(hi)
            if hi > lo:
                shared_bins = np.linspace(lo, hi, int(bins) + 1)

    for i, ch in enumerate(ch_names):
        ax = axes[i]
        col = stat[:, i]
        good = col[np.isfinite(col) & (col > 0)]
        if good.size == 0:
            ax.text(0.5, 0.5, 'no scorable trials', ha='center', va='center',
                    transform=ax.transAxes, fontsize=s['tick_font_size'])
            ax.set_title(ch, fontsize=s['title_font_size'])
            continue

        values = np.log10(good) if log_x else good
        ax.hist(values, bins=shared_bins, color=s['trial_trace_color'],
                edgecolor='#666666', linewidth=0.4)
        for thr in thresholds:
            x = np.log10(thr) if log_x else thr
            n_over = int((good > thr).sum())
            ax.axvline(x, color=s['threshold_color'], linestyle='--',
                       linewidth=1.0,
                       label=f'z={thr:g} ({n_over} trials over)')
        ax.set_title(f'{ch}  (n={good.size}, {int(np.isnan(col).sum())} already NaN)',
                     fontsize=s['title_font_size'])
        ax.tick_params(labelsize=s['tick_font_size'])
        if thresholds:
            ax.legend(fontsize=max(5, s['tick_font_size'] - 1), loc='upper right')

    for j in range(len(ch_names), len(axes)):
        axes[j].axis('off')

    fig.supxlabel('log10(max |z| per trial)' if log_x else 'max |z| per trial',
                  fontsize=10)
    fig.supylabel('Trials', fontsize=10)
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
    return fig


def plot_z_survival(stat_by_label, thresholds=(), title=None, save_path=None,
                    figsize=(8, 5)):
    """Survival curve: fraction of (trial, channel) pairs above each ``|z|``.

    One curve per entry in ``stat_by_label`` (e.g. the problem electrodes
    against every channel in the subject), on log-log axes. The threshold you
    want is where the curve stops falling smoothly and the last few points sit
    detached from the body of the distribution -- that separation is the
    difference between a heavy tail and an artifact.

    Parameters
    ----------
    stat_by_label : dict of {str: ndarray}
        Label -> per-trial max ``|z|`` array (any shape; it is flattened).
    thresholds : sequence of float
        Vertical reference lines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, stat in stat_by_label.items():
        values = np.asarray(stat).ravel()
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            continue
        ordered = np.sort(values)
        # Fraction strictly above each observed value.
        survival = 1.0 - np.arange(1, ordered.size + 1) / ordered.size
        ax.loglog(ordered, np.maximum(survival, 1.0 / ordered.size),
                  lw=1.6, label=f'{label} (n={ordered.size})')

    for thr in thresholds:
        ax.axvline(thr, color='#1F77B4', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.annotate(f'{thr:g}', (thr, ax.get_ylim()[1]), fontsize=7,
                    ha='center', va='top', color='#1F77B4')

    ax.set_xlabel('max |z| per trial')
    ax.set_ylabel('fraction of (trial, channel) pairs above')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
    return fig
