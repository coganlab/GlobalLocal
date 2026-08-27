"""Re-draw decoding figures from saved MASTER_RESULTS pickles.

Every decoding job writes a ``*_MASTER_RESULTS_*.pkl`` next to its figures,
holding the pooled statistics, the significance masks and the run's arguments.
That is everything a figure needs, so the figures can be regenerated -- with
new labels, colours or layout -- without re-running any decoding.

Typical use, from a notebook::

    from src.analysis.decoding.plots.replot import find_master_results, replot_all

    runs = find_master_results(FIGS_ROOT)          # what is on disk
    runs[runs.condition_label.str.contains('lwpc')]

    replot_all(runs, save_dir=CLEAN_FIGS)          # re-draw the lot
"""

import os
import pickle
import re
import traceback
from glob import glob

import numpy as np
import pandas as pd

from src.analysis.config.condition_registry import (
    get_context_comparison_kwargs,
    get_display_name,
    get_trace_labels,
)
from src.analysis.decoding.anova_electrode_selection import (
    short_decoding_figure_title,
)
from .accuracies import plot_accuracies_with_multiple_sig_clusters

__all__ = [
    'find_master_results',
    'load_master_results',
    'replot_master_results',
    'replot_all',
]

# 20260814_000636_MASTER_RESULTS_job52094028_<condition>_24_subs_<elecs>_LDA_...
_FILENAME_RE = re.compile(
    r'^(?P<timestamp>\d{8}_\d{6})_MASTER_RESULTS_(?P<params>.*)\.pkl$'
)

# Keys of master_results['stats'] that are not comparisons.
_NON_COMPARISON_STATS_KEYS = {'pooled_shuffles'}


def find_master_results(root, pattern='**/*MASTER_RESULTS*.pkl', load_metadata=True):
    """Index every MASTER_RESULTS pickle under ``root``.

    Parameters
    ----------
    root : str
        Directory to search, e.g. the decoding ``figs`` root.
    load_metadata : bool
        Read each pickle's metadata for the authoritative condition label,
        electrode set and ROI list. Slower, but the filename is only a
        convention. With False, fields are parsed from the filename alone.

    Returns
    -------
    pandas.DataFrame
        One row per pickle: ``path``, ``timestamp``, ``condition_label``,
        ``electrode_set_label``, ``rois``, ``n_electrodes``, ``n_subjects``,
        ``params``. Sort or filter it, then hand rows to :func:`replot_all`.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f'no such directory: {root}\n'
            'FIGS_ROOT should be the tree the decoding jobs wrote into -- '
            "run_decoding_dcc.py uses <dcc_scripts/decoding>/figs/<EPOCHS_ROOT_FILE>."
        )

    rows = []
    for path in sorted(glob(os.path.join(root, pattern), recursive=True)):
        match = _FILENAME_RE.match(os.path.basename(path))
        row = {
            'path': path,
            'timestamp': match.group('timestamp') if match else '',
            'params': match.group('params') if match else '',
            'condition_label': '',
            'electrode_set_label': '',
            'rois': [],
            'n_electrodes': np.nan,
            'n_subjects': np.nan,
            'error': '',
        }
        if load_metadata:
            try:
                master = load_master_results(path)
                meta = master.get('metadata', {})
                args = meta.get('args', {})
                row['condition_label'] = args.get('condition_label', '')
                row['electrode_set_label'] = (
                    meta.get('electrode_set_label')
                    or meta.get('electrode_set_name')
                    or args.get('electrodes', '')
                )
                row['rois'] = sorted(_rois_in(master))
                row['n_electrodes'] = meta.get('n_electrodes', np.nan)
                row['n_subjects'] = len(args.get('subjects', []) or [])
            except Exception as exc:  # a truncated or half-written pickle
                row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)

    return pd.DataFrame(rows, columns=[
        'path', 'timestamp', 'condition_label', 'electrode_set_label',
        'rois', 'n_electrodes', 'n_subjects', 'params', 'error',
    ])


def load_master_results(path):
    """Unpickle one MASTER_RESULTS file."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _rois_in(master):
    """ROIs present in a results file's stats."""
    rois = set()
    for key, per_roi in master.get('stats', {}).items():
        if key in _NON_COMPARISON_STATS_KEYS or not isinstance(per_roi, dict):
            continue
        rois.update(per_roi)
    return rois


def _comparison_keys(master):
    return [k for k in master.get('stats', {})
            if k not in _NON_COMPARISON_STATS_KEYS]


def _contrast_entries(master, roi, condition_name, colors, comp1, comp2,
                      significance_label_1, significance_label_2):
    """The two directions of the between-condition test, from stored clusters.

    Older results files key these ``'25_over_75'`` / ``'75_over_25'`` and
    newer ones ``'1_over_2'`` / ``'2_over_1'``, so take whatever two keys are
    there, in order, rather than looking for a fixed pair.
    """
    stored = (master.get('comparison_clusters', {})
              .get(roi, {})
              .get(condition_name.lower(), {}))
    if not stored:
        return {}

    labels = [significance_label_1, significance_label_2]
    bar_colors = [colors.get(comp1), colors.get(comp2)]
    # A solid and a dashed bar: the two directions cannot overlap in time, so
    # they share the top row and the linestyle tells them apart in print.
    linestyles = ['-', '--']

    entries = {}
    for i, (name, info) in enumerate(list(stored.items())[:2]):
        info = info if isinstance(info, dict) else {'clusters': info}
        entries[name] = {
            'clusters': info.get('clusters'),
            'label': info.get('label') or labels[i],
            'color': bar_colors[i] or info.get('color') or 'black',
            'linestyle': linestyles[i],
            'kind': 'contrast',
        }
    return entries


def replot_master_results(
    master,
    save_dir,
    rois=None,
    electrode_set_label=None,
    ylim=(0.3, 0.8),
    include_true_vs_shuffle=True,
    include_difference=False,
    filename_suffix='clean',
    **plot_kwargs,
):
    """Re-draw every figure a results file can support.

    Produces, per ROI, the context-comparison panel (two conditions, the
    pooled shuffle, the contrast bars and each condition's own bar against
    chance) when the analysis has one, plus a true-vs-shuffle panel per
    comparison.

    Parameters
    ----------
    master : dict or str
        A loaded master_results dict, or a path to one.
    save_dir : str
        Root for the new figures; a ``<condition>/<roi>`` tree is made inside.
    rois : list, optional
        Defaults to every ROI in the file.
    electrode_set_label : str, optional
        Overrides the electrode-set name used in the title.
    include_difference : bool
        Also draw the (condition 1 - condition 2) difference panel. Off by
        default: it needs the per-sample accuracies of both conditions, which
        the pooled statistics only carry when the units line up.
    **plot_kwargs
        Passed through to the plotting function, e.g. ``base_fontsize=14``,
        ``figsize=(6, 5)``, ``chance_bar_color='black'``.

    Returns
    -------
    list of str
        Paths (without extension) of the figures written.
    """
    if isinstance(master, str):
        master = load_master_results(master)

    meta = master.get('metadata', {})
    args = meta.get('args', {})
    condition_label = args.get('condition_label', '')
    unit = args.get('unit_of_analysis', 'repeat')
    # np.asarray(None) is a size-1 array, not an empty one, so check the key
    # itself or a results file with no time axis fails much later with a
    # confusing message about mismatched significance masks.
    stored_time_points = meta.get('time_window_centers')
    time_points = (np.asarray(stored_time_points, dtype=float)
                   if stored_time_points is not None else np.array([]))
    elec_label = electrode_set_label or meta.get('electrode_set_label') or \
        meta.get('electrode_set_name') or ''
    timestamp = args.get('timestamp', '')

    if time_points.size == 0:
        raise ValueError(
            'results file has no metadata["time_window_centers"]; it was '
            'written by a run too old to re-plot'
        )

    rois = list(rois) if rois is not None else sorted(_rois_in(master))
    trace_labels = get_trace_labels(condition_label)
    title_stem = short_decoding_figure_title(
        get_display_name(condition_label), elec_label)
    context = get_context_comparison_kwargs(condition_label) or {}
    # The registry already names the y-axis per analysis; only analyses with
    # no context comparison fall back to the generic label.
    ylabel = context.get('ylabel', 'Decoding accuracy')

    written = []
    for roi in rois:
        if context:
            written += _replot_context_comparison(
                master, context, roi, unit, time_points, trace_labels,
                title_stem, save_dir, timestamp, filename_suffix, ylim,
                include_difference, plot_kwargs,
            )
        if include_true_vs_shuffle:
            written += _replot_true_vs_shuffle(
                master, roi, unit, time_points, trace_labels, title_stem,
                ylabel, save_dir, timestamp, filename_suffix, plot_kwargs,
            )
    return written


def _replot_context_comparison(master, context, roi, unit, time_points,
                               trace_labels, title_stem, save_dir, timestamp,
                               filename_suffix, ylim, include_difference,
                               plot_kwargs):
    comp1 = context['condition_comparison_1']
    comp2 = context['condition_comparison_2']
    stats = master.get('stats', {})
    if roi not in stats.get(comp1, {}) or roi not in stats.get(comp2, {}):
        return []

    colors = context['colors']
    linestyles = context['linestyles']
    condition_name = context['condition_name']
    shuffle_key = f"{context['pooled_shuffle_key']}_across_bootstraps"
    label_1 = trace_labels.get(comp1, comp1.replace('_', ' '))
    label_2 = trace_labels.get(comp2, comp2.replace('_', ' '))
    shuffle_label = 'Shuffle'

    stats_1, stats_2 = stats[comp1][roi], stats[comp2][roi]
    accuracies = {
        label_1: stats_1[f'{unit}_true_accs'],
        label_2: stats_2[f'{unit}_true_accs'],
    }
    pooled_shuffle = (stats.get('pooled_shuffles', {})
                      .get(roi, {})
                      .get(condition_name.lower()))
    if pooled_shuffle is not None:
        accuracies[shuffle_label] = pooled_shuffle

    significance = _contrast_entries(
        master, roi, condition_name, colors, comp1, comp2,
        context.get('significance_label_1', ''),
        context.get('significance_label_2', ''),
    )
    # Each condition's own test against chance, one row below the contrast.
    significance.update({
        f'{comp1}_vs_chance': {
            'clusters': stats_1.get('significant_clusters'),
            'color': colors.get(comp1),
            'kind': 'chance',
        },
        f'{comp2}_vs_chance': {
            'clusters': stats_2.get('significant_clusters'),
            'color': colors.get(comp2),
            'kind': 'chance',
        },
    })

    out_dir = os.path.join(save_dir, f'{condition_name}_comparison', roi)
    kwargs = dict(
        time_points=time_points,
        significance_clusters_dict=significance,
        colors={label_1: colors.get(comp1), label_2: colors.get(comp2),
                shuffle_label: colors.get(shuffle_key, '#949494')},
        linestyles={label_1: linestyles.get(comp1, '-'),
                    label_2: linestyles.get(comp2, '-'),
                    shuffle_label: linestyles.get(shuffle_key, '--')},
        roi=roi,
        save_dir=out_dir,
        timestamp=timestamp,
        ylabel=context.get('ylabel', 'Decoding accuracy'),
        ylim=ylim,
        show_chance_level=False,
        show_sig_legend=True,
        filename_suffix=filename_suffix,
    )
    kwargs.update(plot_kwargs)

    plot_accuracies_with_multiple_sig_clusters(
        accuracies_dict=accuracies,
        comparison_name=f'{condition_name}_comparison',
        title=title_stem,
        **kwargs,
    )
    written = [os.path.join(out_dir, '_'.join(filter(None, [
        timestamp, f'{condition_name}_comparison', roi, filename_suffix])))]

    if include_difference:
        accs_1 = np.asarray(stats_1[f'{unit}_true_accs'], dtype=float)
        accs_2 = np.asarray(stats_2[f'{unit}_true_accs'], dtype=float)
        if accs_1.shape == accs_2.shape:
            differences = accs_1 - accs_2
            spread = np.max(np.abs(np.mean(differences, axis=0))
                            + np.std(differences, axis=0))
            diff_ylim = (-spread * 1.2, spread * 1.2) if spread else (-0.1, 0.1)
            diff_label = f'{label_1} − {label_2}'
            diff_kwargs = dict(kwargs)
            diff_kwargs.update(
                colors={diff_label: '#404040'},
                linestyles={diff_label: '-'},
                ylim=diff_ylim,
                ylabel='Accuracy difference',
                show_chance_level=True,
                chance_level=0,
                filename_suffix=f'{filename_suffix}_ACC_DIFFERENCE',
            )
            # Against-chance bars belong to the individual traces, not to
            # their difference.
            diff_kwargs['significance_clusters_dict'] = {
                k: v for k, v in significance.items()
                if v.get('kind') != 'chance'
            }
            diff_kwargs.update(plot_kwargs)
            plot_accuracies_with_multiple_sig_clusters(
                accuracies_dict={diff_label: differences},
                comparison_name=f'{condition_name}_ACC_DIFFERENCE',
                title=f'{title_stem}: {label_1} − {label_2}',
                **diff_kwargs,
            )
            written.append(os.path.join(out_dir, '_'.join(filter(None, [
                timestamp, f'{condition_name}_ACC_DIFFERENCE', roi,
                f'{filename_suffix}_ACC_DIFFERENCE']))))
        else:
            print(f'  · {roi}: skipping difference panel, '
                  f'{accs_1.shape} vs {accs_2.shape} samples do not pair')

    return written


def _replot_true_vs_shuffle(master, roi, unit, time_points, trace_labels,
                            title_stem, ylabel, save_dir, timestamp,
                            filename_suffix, plot_kwargs):
    written = []
    for comparison in _comparison_keys(master):
        per_roi = master['stats'][comparison]
        if not isinstance(per_roi, dict) or roi not in per_roi:
            continue
        stats = per_roi[roi]
        if f'{unit}_true_accs' not in stats:
            continue

        true_label = trace_labels.get(comparison, comparison.replace('_', ' '))
        shuffle_label = 'Shuffle'
        out_dir = os.path.join(save_dir, comparison, roi)
        kwargs = dict(
            time_points=time_points,
            accuracies_dict={
                true_label: stats[f'{unit}_true_accs'],
                shuffle_label: stats[f'{unit}_shuffle_accs'],
            },
            significance_clusters_dict={
                'vs_shuffle': {
                    'clusters': stats.get('significant_clusters'),
                    'color': '#0173B2',
                    'kind': 'chance',
                },
            },
            colors={true_label: '#0173B2', shuffle_label: '#949494'},
            linestyles={true_label: '-', shuffle_label: '--'},
            comparison_name=f'true_vs_shuffle_{comparison}',
            roi=roi,
            save_dir=out_dir,
            timestamp=timestamp,
            ylabel=ylabel,
            ylim=(0.3, 1.0),
            show_chance_level=False,
            show_sig_legend=True,
            chance_bar_label='> shuffle',
            title=f'{title_stem}: {true_label}',
            filename_suffix=filename_suffix,
        )
        kwargs.update(plot_kwargs)
        plot_accuracies_with_multiple_sig_clusters(**kwargs)
        written.append(os.path.join(out_dir, '_'.join(filter(None, [
            timestamp, f'true_vs_shuffle_{comparison}', roi, filename_suffix]))))
    return written


def replot_all(runs, save_dir, group_by=('condition_label', 'electrode_set_label'),
               **replot_kwargs):
    """Re-plot every run in ``runs``, keeping going past individual failures.

    Parameters
    ----------
    runs : pandas.DataFrame or iterable of str
        Output of :func:`find_master_results` (or a filtered subset), or just
        a list of paths.
    save_dir : str
        Root for the new figures. Each run gets its own subdirectory named
        from ``group_by``, so two electrode sets never overwrite each other.
    **replot_kwargs
        Passed to :func:`replot_master_results`.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``path``, ``n_figures``, ``error``. Check that
        ``error`` is empty everywhere before believing the output is complete.
    """
    if isinstance(runs, pd.DataFrame):
        records = runs.to_dict('records')
    else:
        records = [{'path': p} for p in runs]

    results = []
    for i, record in enumerate(records, 1):
        path = record['path']
        print(f'[{i}/{len(records)}] {os.path.basename(path)}')
        try:
            master = load_master_results(path)
            meta = master.get('metadata', {})
            args = meta.get('args', {})
            parts = []
            for field in group_by:
                value = record.get(field) or args.get(field) or meta.get(field)
                if value:
                    parts.append(re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)))
            run_dir = os.path.join(save_dir, *parts) if parts else save_dir

            written = replot_master_results(master, run_dir, **replot_kwargs)
            results.append({'path': path, 'out_dir': run_dir,
                            'n_figures': len(written), 'error': ''})
            print(f'    → {len(written)} figures in {run_dir}')
        except Exception as exc:
            traceback.print_exc()
            results.append({'path': path, 'out_dir': '', 'n_figures': 0,
                            'error': f'{type(exc).__name__}: {exc}'})

    # Always give the caller the same columns, so indexing the result of an
    # empty sweep says "no rows" rather than raising KeyError on a bare frame.
    frame = pd.DataFrame(results, columns=['path', 'out_dir', 'n_figures', 'error'])
    if not len(frame):
        print('\nNothing to re-plot: no runs were passed in.')
        return frame
    failed = int((frame['error'] != '').sum())
    print(f'\n{len(frame) - failed}/{len(frame)} runs re-plotted'
          + (f', {failed} failed (see the error column)' if failed else ''))
    return frame
