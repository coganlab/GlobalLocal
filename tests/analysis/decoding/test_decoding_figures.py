"""Tests for the cleaned-up decoding accuracy figures and the re-plot path.

These assert layout *behaviour* -- what the axes, ticks, legends and
significance rows end up being -- rather than comparing pixels, so they stay
useful when the styling is tweaked.
"""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.config.condition_registry import (
    CONDITION_REGISTRY,
    get_display_name,
    get_trace_labels,
)
from src.analysis.decoding.anova_electrode_selection import (
    short_decoding_figure_title,
    short_electrode_set_label,
)
from src.analysis.decoding.plots.accuracies import (
    _cluster_spans,
    _nice_ticks,
    _tick_label,
    plot_accuracies_with_multiple_sig_clusters,
)


N_TIME = 32
TIME_POINTS = np.linspace(-0.5, 1.4375, N_TIME)
DATA_YLIM = (0.3, 0.8)


@pytest.fixture
def accuracies():
    rng = np.random.default_rng(0)
    ramp = 0.25 / (1 + np.exp(-(TIME_POINTS - 0.2) / 0.25))
    return {
        'strong': 0.47 + ramp + rng.normal(0, 0.05, (40, N_TIME)),
        'weak': 0.47 + ramp / 3 + rng.normal(0, 0.05, (40, N_TIME)),
        'shuffle': 0.5 + rng.normal(0, 0.05, (40, N_TIME)),
    }


def _sig_bars(ax):
    """The significance bars on ``ax``.

    They are drawn in axes-fraction y like the stimulus-onset line, so match
    on the zorder the strip uses rather than on the transform alone.
    """
    return [line for line in ax.lines
            if line.get_transform() == ax.get_xaxis_transform()
            and line.get_zorder() == 5]


def _figure(accuracies, significance, **kwargs):
    options = dict(
        time_points=TIME_POINTS,
        accuracies_dict=accuracies,
        significance_clusters_dict=significance,
        ylim=DATA_YLIM,
        show_chance_level=False,
        show_sig_legend=True,
        return_fig=True,
    )
    options.update(kwargs)
    return plot_accuracies_with_multiple_sig_clusters(**options)


# --- tick helpers -------------------------------------------------------

@pytest.mark.parametrize('lo,hi,expected', [
    (0.3, 0.8, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    (-0.1, 0.1, [-0.1, -0.05, 0.0, 0.05, 0.1]),
])
def test_nice_ticks_are_round_numbers(lo, hi, expected):
    """linspace(0.3, 0.8, 5) would give 0.42 and 0.68; these are readable."""
    assert _nice_ticks(lo, hi, 5) == pytest.approx(expected)


def test_nice_ticks_stay_inside_the_range():
    ticks = _nice_ticks(0.42, 0.79, 5)
    assert ticks.min() >= 0.42 and ticks.max() <= 0.79


def test_tick_label_drops_trailing_zeros():
    assert _tick_label(0.3) == '0.3'
    assert _tick_label(-0.0) == '0'
    assert _tick_label(0.05) == '0.05'


# --- significance cluster geometry --------------------------------------

def test_single_window_cluster_is_visible():
    """Un-padded, a one-window cluster draws a zero-length line and vanishes."""
    mask = np.zeros(N_TIME, dtype=bool)
    mask[10] = True
    spans = _cluster_spans(mask, TIME_POINTS)
    assert len(spans) == 1
    start, end = spans[0]
    assert end > start
    step = np.median(np.diff(TIME_POINTS))
    assert (end - start) == pytest.approx(step)


def test_cluster_spans_are_clipped_to_the_axes():
    """Half-step padding must not push an edge cluster past the panel."""
    mask = np.zeros(N_TIME, dtype=bool)
    mask[[0, -1]] = True
    xlim = (TIME_POINTS[0], TIME_POINTS[-1])
    for start, end in _cluster_spans(mask, TIME_POINTS, xlim):
        assert start >= xlim[0] - 1e-9
        assert end <= xlim[1] + 1e-9


def test_empty_mask_yields_no_spans():
    assert _cluster_spans(np.zeros(N_TIME, dtype=bool), TIME_POINTS) == []
    assert _cluster_spans(None, TIME_POINTS) == []


def test_mask_of_wrong_length_is_rejected(accuracies):
    """A mask from a different run must fail loudly, not draw silently wrong."""
    with pytest.raises(ValueError, match='time points'):
        _figure(accuracies, {'bad': {'clusters': np.ones(N_TIME + 5, dtype=bool)}})


# --- layout -------------------------------------------------------------

def test_significance_strip_sits_above_the_data(accuracies):
    """The data keeps its ylim; the axes grow upward to make room for bars."""
    mask = TIME_POINTS > 0.3
    fig = _figure(accuracies, {'c': {'clusters': mask, 'label': 'a > b'}})
    ax = fig.axes[0]
    assert ax.get_ylim()[0] == pytest.approx(DATA_YLIM[0])
    assert ax.get_ylim()[1] > DATA_YLIM[1]
    plt.close(fig)


def test_no_strip_is_reserved_when_there_is_nothing_significant(accuracies):
    fig = _figure(accuracies, {'c': {'clusters': np.zeros(N_TIME, dtype=bool)}})
    assert fig.axes[0].get_ylim()[1] == pytest.approx(DATA_YLIM[1])
    plt.close(fig)


def test_yticks_never_enter_the_significance_strip(accuracies):
    fig = _figure(accuracies, {'c': {'clusters': TIME_POINTS > 0.3, 'label': 'a > b'}})
    ax = fig.axes[0]
    assert max(ax.get_yticks()) <= DATA_YLIM[1] + 1e-9
    assert ax.spines['left'].get_bounds() == pytest.approx(DATA_YLIM)
    plt.close(fig)


def test_chance_bars_are_drawn_below_contrast_bars(accuracies):
    """The against-chance rows have to sit under the contrast row, not over it."""
    fig = _figure(accuracies, {
        'contrast': {'clusters': TIME_POINTS > 0.3, 'label': 'a > b',
                     'color': 'red', 'kind': 'contrast'},
        'a_chance': {'clusters': TIME_POINTS > 0.1, 'color': 'red', 'kind': 'chance'},
        'b_chance': {'clusters': TIME_POINTS > 0.6, 'color': 'blue', 'kind': 'chance'},
    })
    bar_ys = [line.get_ydata()[0] for line in _sig_bars(fig.axes[0])]
    assert len(bar_ys) == 3
    contrast_y, chance_a_y, chance_b_y = bar_ys
    assert contrast_y > chance_a_y > chance_b_y
    plt.close(fig)


def test_legend_shows_the_labels_it_was_given(accuracies):
    """Legend text is whatever the caller passed, never an internal key."""
    fig = _figure(accuracies, {
        'contrast': {'clusters': TIME_POINTS > 0.3, 'label': '25% > 75%',
                     'kind': 'contrast'},
        'chance': {'clusters': TIME_POINTS > 0.1, 'kind': 'chance'},
    })
    ax = fig.axes[0]
    all_labels = [t.get_text() for legend in ([ax.get_legend()] + ax.artists)
                  if hasattr(legend, 'get_texts') for t in legend.get_texts()]
    assert 'strong' in all_labels and 'weak' in all_labels
    assert '25% > 75%' in all_labels
    assert '> chance' in all_labels
    plt.close(fig)


def test_no_asterisks_are_drawn_by_default(accuracies):
    """The old default littered the panel with '*' next to every bar."""
    fig = _figure(accuracies, {'c': {'clusters': TIME_POINTS > 0.3, 'label': 'a > b'}})
    assert [t for t in fig.axes[0].texts if '*' in t.get_text()] == []
    plt.close(fig)


def test_superseded_bar_placement_arguments_are_still_accepted(accuracies):
    """Existing callers pass these; they must not raise."""
    fig = _figure(
        accuracies, {'c': {'clusters': TIME_POINTS > 0.3, 'label': 'a > b'}},
        sig_bar_base_position=0.72, sig_bar_spacing=0.015, sig_bar_height=0.01,
    )
    assert fig.axes[0].get_ylim()[1] > DATA_YLIM[1]
    plt.close(fig)


def test_chance_bar_color_overrides_the_trace_colors(accuracies):
    fig = _figure(
        accuracies,
        {'a': {'clusters': TIME_POINTS > 0.1, 'color': 'red', 'kind': 'chance'},
         'b': {'clusters': TIME_POINTS > 0.6, 'color': 'blue', 'kind': 'chance'}},
        chance_bar_color='black',
    )
    bar_colors = {line.get_color() for line in _sig_bars(fig.axes[0])}
    assert bar_colors == {'black'}
    plt.close(fig)


def test_figure_is_written_in_every_requested_format(accuracies, tmp_path):
    plot_accuracies_with_multiple_sig_clusters(
        time_points=TIME_POINTS,
        accuracies_dict=accuracies,
        significance_clusters_dict={'c': {'clusters': TIME_POINTS > 0.3}},
        ylim=DATA_YLIM,
        show_chance_level=False,
        comparison_name='LWPC_comparison',
        roi='lpfc',
        timestamp='20260814_000636',
        save_dir=str(tmp_path),
        formats=('png', 'pdf'),
    )
    stem = '20260814_000636_LWPC_comparison_lpfc'
    assert (tmp_path / f'{stem}.png').exists()
    assert (tmp_path / f'{stem}.pdf').exists()


# --- titles and labels --------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    ('anova_csv_elecs_anova_congruency_only', 'congruency-only'),
    ('anova_csv_elecs_anova_switchType_only', 'switchType-only'),
    ('all_elecs', 'all'),
    ('elecset_overlap', 'overlap'),
    ('', ''),
])
def test_short_electrode_set_label(raw, expected):
    assert short_electrode_set_label(raw) == expected


def test_short_figure_title_is_one_line():
    title = short_decoding_figure_title(
        'LWPC', 'anova_csv_elecs_anova_congruency_only')
    assert title == 'LWPC in congruency-only electrodes'
    assert '\n' not in title


def test_short_figure_title_without_an_electrode_set():
    assert short_decoding_figure_title('LWPC', None) == 'LWPC'


# --- registry display config -------------------------------------------

CONTEXT_CONDITIONS = [
    name for name, entry in CONDITION_REGISTRY.items()
    if entry.get('context_comparison')
]


@pytest.mark.parametrize('condition', CONTEXT_CONDITIONS)
def test_every_context_comparison_has_legend_labels(condition):
    """Otherwise that analysis' legend falls back to raw comparison keys."""
    context = CONDITION_REGISTRY[condition]['context_comparison']
    labels = get_trace_labels(condition)
    for key in ('condition_comparison_1', 'condition_comparison_2'):
        assert context[key] in labels, f'{condition}: no label for {context[key]}'
        assert labels[context[key]]


@pytest.mark.parametrize('condition', CONTEXT_CONDITIONS)
def test_the_two_traces_have_distinguishable_colors(condition):
    """One colour for both conditions leaves them separable only by linestyle."""
    context = CONDITION_REGISTRY[condition]['context_comparison']
    colors = context['colors']
    assert (colors[context['condition_comparison_1']]
            != colors[context['condition_comparison_2']])


@pytest.mark.parametrize('condition', CONTEXT_CONDITIONS)
def test_every_context_comparison_has_a_short_title(condition):
    name = get_display_name(condition)
    assert name and '\n' not in name


def test_display_names_are_unique():
    """Two analyses sharing a title makes their figures unattributable."""
    names = [get_display_name(c) for c in CONTEXT_CONDITIONS]
    assert len(names) == len(set(names)), sorted(names)


# --- re-plotting from a saved results file ------------------------------

CONDITION = 'stimulus_lwpc_block_balanced_conditions'
COMPARISONS = ['i_vs_c_at_inc25', 'i_vs_c_at_inc75']
ROIS = ['lpfc', 'acc']


def _master_results():
    """A results dict shaped exactly like the one decoding_dcc.py pickles."""
    rng = np.random.default_rng(1)
    condition_name = (CONDITION_REGISTRY[CONDITION]['context_comparison']
                      ['condition_name'])
    stats = {}
    for i, comparison in enumerate(COMPARISONS):
        stats[comparison] = {
            roi: {
                'repeat_true_accs': 0.55 + rng.normal(0, 0.05, (20, N_TIME)),
                'repeat_shuffle_accs': 0.5 + rng.normal(0, 0.05, (20, N_TIME)),
                'significant_clusters': TIME_POINTS > (0.15 if i == 0 else 0.55),
                'unit_of_analysis': 'repeat',
            } for roi in ROIS
        }
    stats['pooled_shuffles'] = {
        roi: {condition_name.lower(): 0.5 + rng.normal(0, 0.05, (80, N_TIME))}
        for roi in ROIS
    }
    return {
        'stats': stats,
        'metadata': {
            'args': {
                'condition_label': CONDITION,
                'unit_of_analysis': 'repeat',
                'timestamp': '20260814_000636',
                'subjects': ['D0057', 'D0059'],
            },
            'electrode_set_label': 'anova_csv_elecs_anova_congruency_only',
            'n_electrodes': 32,
            'time_window_centers': list(TIME_POINTS),
        },
        'comparison_clusters': {
            roi: {condition_name.lower(): {
                '1_over_2': {'clusters': (TIME_POINTS > 0.45) & (TIME_POINTS < 1.2)},
                '2_over_1': {'clusters': np.zeros(N_TIME, dtype=bool)},
            }} for roi in ROIS
        },
    }


@pytest.fixture
def results_tree(tmp_path):
    """A figs tree holding two runs that differ only by electrode set."""
    from src.analysis.decoding.plots import replot  # noqa: F401  (import check)

    for i, elecs in enumerate(['anova_csv_elecs_anova_congruency_only', 'all_elecs']):
        master = _master_results()
        master['metadata']['electrode_set_label'] = elecs
        run_dir = tmp_path / f'run{i}'
        run_dir.mkdir()
        name = f'2026081{i}_000636_MASTER_RESULTS_job5209402{i}_{CONDITION}.pkl'
        with open(run_dir / name, 'wb') as handle:
            pickle.dump(master, handle)
    return tmp_path


def test_find_master_results_reads_metadata_not_just_filenames(results_tree):
    from src.analysis.decoding.plots.replot import find_master_results

    runs = find_master_results(str(results_tree))
    assert len(runs) == 2
    assert runs['error'].eq('').all()
    assert set(runs['condition_label']) == {CONDITION}
    assert set(runs['electrode_set_label']) == {
        'anova_csv_elecs_anova_congruency_only', 'all_elecs'}
    assert runs.loc[0, 'rois'] == sorted(ROIS)
    assert runs.loc[0, 'n_subjects'] == 2


def test_replot_master_results_covers_every_roi_and_comparison(tmp_path):
    from src.analysis.decoding.plots.replot import replot_master_results

    written = replot_master_results(_master_results(), str(tmp_path))
    # per ROI: one comparison panel + one true-vs-shuffle panel per comparison
    assert len(written) == len(ROIS) * (1 + len(COMPARISONS))
    for stem in written:
        assert os.path.exists(f'{stem}.png'), stem


def test_replot_uses_the_registry_labels_not_raw_keys(tmp_path):
    from src.analysis.decoding.plots.replot import replot_master_results

    written = replot_master_results(
        _master_results(), str(tmp_path), rois=['lpfc'],
        include_true_vs_shuffle=False)
    assert len(written) == 1
    # The title comes from the registry, via the electrode-set label.
    fig = plt.gcf()
    plt.close(fig)
    assert 'LWPC_block_balanced_comparison' in written[0]


def test_replot_all_keeps_electrode_sets_apart(results_tree, tmp_path):
    """Two runs differing only by electrode set must not overwrite each other."""
    from src.analysis.decoding.plots.replot import find_master_results, replot_all

    out = tmp_path / 'clean'
    runs = find_master_results(str(results_tree))
    report = replot_all(runs, str(out))

    assert report['error'].eq('').all(), report['error'].tolist()
    assert report['n_figures'].sum() == 2 * len(ROIS) * (1 + len(COMPARISONS))
    electrode_dirs = sorted(os.listdir(out / CONDITION))
    assert electrode_dirs == ['all_elecs', 'anova_csv_elecs_anova_congruency_only']


def test_replot_survives_a_file_it_cannot_read(results_tree, tmp_path):
    """One bad pickle must not abort a sweep over hundreds of runs."""
    from src.analysis.decoding.plots.replot import replot_all

    broken = results_tree / 'broken_MASTER_RESULTS.pkl'
    broken.write_bytes(b'not a pickle')

    report = replot_all([str(broken)], str(tmp_path / 'out'))
    assert len(report) == 1
    assert report.loc[0, 'error']
    assert report.loc[0, 'n_figures'] == 0


def test_replot_rejects_a_results_file_with_no_time_axis(tmp_path):
    from src.analysis.decoding.plots.replot import replot_master_results

    master = _master_results()
    del master['metadata']['time_window_centers']
    with pytest.raises(ValueError, match='time_window_centers'):
        replot_master_results(master, str(tmp_path))


# --- backwards compatibility with pre-strip callers ---------------------

def test_entries_without_kind_each_keep_their_own_row(accuracies):
    """customize_decoding_plots.ipynb passes three entries and no 'kind'.

    Grouping those onto one row -- the default for entries that declare
    'kind' -- would draw all three over each other.
    """
    fig = _figure(accuracies, {
        'sig_25_vs_chance': {'clusters': TIME_POINTS > 0.1,
                             'label': '25% I > Chance', 'color': 'salmon', 'marker': ''},
        'sig_75_vs_chance': {'clusters': TIME_POINTS > 0.4,
                             'label': '75% I > Chance', 'color': 'orange', 'marker': ''},
        'sig_comparison': {'clusters': TIME_POINTS > 0.6,
                           'label': '25% I > 75% I', 'color': 'black',
                           'marker': '', 'level': 2},
    })
    bar_ys = {line.get_color(): line.get_ydata()[0]
              for line in _sig_bars(fig.axes[0])}
    assert len(set(bar_ys.values())) == 3, 'entries were collapsed onto one row'
    # Dict order stacks bottom-up, so the comparison bar listed last is on top.
    assert bar_ys['black'] > bar_ys['orange'] > bar_ys['salmon']
    plt.close(fig)


def test_declaring_kind_switches_to_the_grouped_layout(accuracies):
    fig = _figure(accuracies, {
        'up': {'clusters': TIME_POINTS < 0.5, 'label': 'a > b',
               'color': 'red', 'kind': 'contrast'},
        'down': {'clusters': TIME_POINTS > 0.8, 'label': 'b > a',
                 'color': 'blue', 'kind': 'contrast'},
        'chance': {'clusters': TIME_POINTS > 0.1, 'color': 'green', 'kind': 'chance'},
    })
    bar_ys = {line.get_color(): line.get_ydata()[0]
              for line in _sig_bars(fig.axes[0])}
    assert bar_ys['red'] == bar_ys['blue'], 'contrast directions should share a row'
    assert bar_ys['green'] < bar_ys['red']
    plt.close(fig)


def test_eps_is_still_written_by_default(accuracies, tmp_path):
    """The pre-existing callers wrote pdf, png and eps; keep all three."""
    plot_accuracies_with_multiple_sig_clusters(
        time_points=TIME_POINTS,
        accuracies_dict=accuracies,
        significance_clusters_dict={'c': {'clusters': TIME_POINTS > 0.3}},
        ylim=DATA_YLIM,
        show_chance_level=False,
        comparison_name='LWPC_comparison',
        roi='lpfc',
        timestamp='20260814_000636',
        save_dir=str(tmp_path),
    )
    stem = '20260814_000636_LWPC_comparison_lpfc'
    for extension in ('png', 'pdf', 'eps'):
        assert (tmp_path / f'{stem}.{extension}').exists(), extension


# --- pointing the sweep at the wrong place ------------------------------

def test_find_master_results_names_the_missing_directory(tmp_path):
    """The first thing anyone gets wrong is FIGS_ROOT; say so plainly."""
    from src.analysis.decoding.plots.replot import find_master_results

    with pytest.raises(FileNotFoundError, match='EPOCHS_ROOT_FILE'):
        find_master_results(str(tmp_path / 'does_not_exist'))


def test_find_master_results_on_an_empty_tree_returns_usable_columns(tmp_path):
    from src.analysis.decoding.plots.replot import find_master_results

    runs = find_master_results(str(tmp_path))
    assert len(runs) == 0
    assert 'condition_label' in runs.columns and 'error' in runs.columns


def test_replot_all_on_no_runs_returns_usable_columns(tmp_path):
    """An empty sweep must not blow up the next cell with a KeyError."""
    from src.analysis.decoding.plots.replot import replot_all

    report = replot_all([], str(tmp_path / 'out'))
    assert len(report) == 0
    # The notebook indexes these straight after the call.
    assert list(report[['n_figures', 'error', 'out_dir']].columns) == [
        'n_figures', 'error', 'out_dir']
