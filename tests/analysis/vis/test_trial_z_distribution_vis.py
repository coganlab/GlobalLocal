"""Tests for the per-trial z diagnostics in src/analysis/vis/trial_z_distribution_vis.py."""
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.vis.trial_z_distribution_vis import (
    per_trial_max_abs_z,
    describe_z_per_channel,
    summarize_z_threshold_tradeoff,
    plot_trial_traces_over_mean_grid,
    plot_z_distribution_grid,
    plot_z_survival,
)


class FakeEpochs:
    """Minimal duck-type: the plotters only need times, ch_names, get_data, len."""

    def __init__(self, data, times, ch_names):
        self._data = data
        self.times = times
        self.ch_names = list(ch_names)

    def __len__(self):
        return self._data.shape[0]

    def get_data(self, picks=None):
        if picks is None:
            return self._data
        idx = [self.ch_names.index(ch) for ch in picks]
        return self._data[:, idx, :]


@pytest.fixture
def epochs():
    """20 trials x 3 channels of unit-ish noise, with planted pathologies."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (20, 3, 50))
    data[3, 0, 10] = 500.0        # one huge excursion
    data[4, 1, :] = np.nan        # a (trial, channel) pair already rejected
    data[5, :, 7] = np.nan        # a single NaN'd sample on every channel
    return FakeEpochs(data, np.linspace(-1.0, 1.5, 50), ['A', 'B', 'C'])


def test_per_trial_max_abs_z_matches_the_rejection_rule(epochs):
    stat, names, missing = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    assert names == ['A', 'B', 'C'] and missing == []
    assert stat.shape == (20, 3)
    assert stat[3, 0] == 500.0


def test_already_nan_pair_scores_nan_not_neg_inf(epochs):
    """An all-NaN pair is already-rejected data, so it must not read as a value."""
    stat, _, _ = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    assert np.isnan(stat[4, 1])
    assert np.isnan(stat).sum() == 1


def test_partial_nan_trial_still_scored(epochs):
    """A trial with some NaN samples is scored from the samples that survive."""
    stat, _, _ = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    assert np.isfinite(stat[5]).all()


def test_missing_electrodes_reported_not_raised(epochs):
    stat, names, missing = per_trial_max_abs_z(epochs, ['A', 'LFMI8'])
    assert names == ['A'] and missing == ['LFMI8']
    assert stat.shape == (20, 1)


def test_window_restricts_the_score(epochs):
    """Scoring after the excursion's timepoint must not see it."""
    times = epochs.times
    stat_all, _, _ = per_trial_max_abs_z(epochs, ['A'])
    stat_late, _, _ = per_trial_max_abs_z(epochs, ['A'], tmin=times[20])
    assert stat_all[3, 0] == 500.0
    assert stat_late[3, 0] < 500.0


def test_empty_window_falls_back_to_whole_epoch(epochs):
    """Matches block_diagnostics.window_mask, which this now delegates to."""
    stat_all, _, _ = per_trial_max_abs_z(epochs, ['A'])
    stat_empty, _, _ = per_trial_max_abs_z(epochs, ['A'], tmin=10.0, tmax=20.0)
    assert np.allclose(stat_all, stat_empty, equal_nan=True)


def test_matches_the_pipeline_rejection_rule(epochs):
    """The diagnostics and make_epoched_data.py must score the same thing."""
    from src.analysis.power.block_diagnostics import max_abs_z_per_trial

    stat, names, _ = per_trial_max_abs_z(epochs, epochs.ch_names)
    pipeline = max_abs_z_per_trial(epochs.get_data())
    assert np.allclose(stat, pipeline, equal_nan=True)
    # And against the expression it replaced in make_epoched_data.py. That one
    # warns on the all-NaN pair and returns NaN there; both compare False, so
    # the rejection masks agree -- which is what had to be preserved.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        legacy = np.nanmax(np.abs(epochs.get_data()), axis=2)
    assert np.array_equal(pipeline > 100, legacy > 100)


def test_threshold_tradeoff_excludes_already_nan_from_denominator(epochs):
    stat, names, _ = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    table = summarize_z_threshold_tradeoff(stat, names, [10, 1000])

    tight = table.iloc[0]
    assert tight['pairs_rejected'] == 1
    assert tight['trials_touched'] == 1
    assert tight['channels_touched'] == 1
    # 60 pairs, one of them already NaN, so 59 are scorable.
    assert tight['pct_of_scorable_pairs'] == pytest.approx(100 / 59)

    assert table.iloc[1]['pairs_rejected'] == 0


def test_describe_z_per_channel_counts_nans(epochs):
    stat, names, _ = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    desc = describe_z_per_channel(stat, names).set_index('channel')
    assert desc.loc['B', 'n_already_nan'] == 1
    assert desc.loc['B', 'n_scorable_trials'] == 19
    assert desc.loc['A', 'p100'] == 500.0


def test_plotters_run(epochs):
    stat, names, _ = per_trial_max_abs_z(epochs, ['A', 'B', 'C'])
    figs = [
        plot_trial_traces_over_mean_grid(epochs, ['A', 'B', 'C'], thresholds=[30]),
        plot_trial_traces_over_mean_grid(epochs, ['A', 'B', 'C'], yscale='symlog',
                                         robust_ylim_pct=99.0),
        plot_z_distribution_grid(stat, names, thresholds=[10, 100]),
        plot_z_survival({'ABC': stat}, thresholds=[10]),
    ]
    for fig in figs:
        assert fig is not None
        plt.close(fig)


def test_all_nan_channel_does_not_blow_up():
    """Every plotter has to survive a channel whose trials are all NaN."""
    allnan = FakeEpochs(np.full((4, 2, 10), np.nan), np.linspace(0, 1, 10), ['A', 'B'])
    stat, names, _ = per_trial_max_abs_z(allnan, ['A', 'B'])
    assert np.isnan(stat).all()

    table = summarize_z_threshold_tradeoff(stat, names, [30])
    assert table.iloc[0]['pairs_rejected'] == 0

    for fig in (plot_trial_traces_over_mean_grid(allnan, ['A', 'B']),
                plot_z_distribution_grid(stat, names, thresholds=[30])):
        plt.close(fig)
