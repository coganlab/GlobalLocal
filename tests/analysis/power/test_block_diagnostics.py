import numpy as np
import pandas as pd
import pytest

from src.analysis.power.block_diagnostics import (
    block_deviation_table,
    block_labels_from_metadata,
    rank_block_confined_channels,
    window_mask,
)


TIMES = np.linspace(-1.0, 1.5, 26)          # -1.0 .. 1.5, 0.1 s steps
BLOCKS = [(75.0, 75.0), (75.0, 25.0), (25.0, 75.0), (25.0, 25.0)]
CH_NAMES = ['CLEAN1', 'CLEAN2', 'BAD_IN_B']


def _data(rng=None, spike=120.0, offset=0.0):
    """4 blocks x 10 trials x 3 channels. BAD_IN_B misbehaves in (75, 75) only."""
    rng = rng or np.random.default_rng(0)
    labels, trials = [], []
    for block in BLOCKS:
        for _ in range(10):
            trial = rng.normal(0, 0.05, (3, TIMES.size))
            if block == (75.0, 75.0):
                trial[2, 0] += spike          # first sample, epoch edge
                trial[2, :] += offset         # tonic shift across the epoch
            labels.append(block)
            trials.append(trial)
    return np.stack(trials), labels


def test_window_mask_bounds_may_be_open():
    mask = window_mask(TIMES, (None, 0.0))
    assert TIMES[mask].max() <= 0.0
    assert TIMES[mask].min() == pytest.approx(-1.0)


def test_window_mask_falls_back_when_window_selects_nothing():
    assert window_mask(TIMES, (50.0, 60.0)).all()


def test_table_has_one_row_per_channel_and_block():
    data, labels = _data()
    table = block_deviation_table(data, labels, TIMES, ch_names=CH_NAMES)

    assert len(table) == len(BLOCKS) * len(CH_NAMES)
    assert set(table['channel']) == set(CH_NAMES)
    assert (table['n_trials'] == 10).all()


def test_peak_excursion_is_found_in_the_offending_block_at_the_epoch_edge():
    data, labels = _data()
    table = block_deviation_table(data, labels, TIMES, ch_names=CH_NAMES)
    bad = table[table['channel'] == 'BAD_IN_B'].set_index('block')

    assert bad.loc[[(75.0, 75.0)], 'peak_abs_trial'].item() > 100
    assert bad.loc[[(75.0, 75.0)], 'peak_abs_trial_time'].item() == pytest.approx(-1.0)
    for block in BLOCKS[1:]:
        assert bad.loc[[block], 'peak_abs_trial'].item() < 1


def test_ranking_puts_the_block_confined_channel_first():
    data, labels = _data()
    table = block_deviation_table(data, labels, TIMES, ch_names=CH_NAMES)
    ranked = rank_block_confined_channels(table)

    assert ranked.iloc[0]['channel'] == 'BAD_IN_B'
    assert ranked.iloc[0]['peak_block'] == (75.0, 75.0)
    # The clean channels have no block standing out from the others.
    assert ranked.iloc[0]['peak_ratio'] > 20 * ranked.iloc[1]['peak_ratio']


def test_block_offset_measures_a_tonic_shift_not_a_spike():
    # No spike, only a sustained 0.4 z offset in one block.
    data, labels = _data(spike=0.0, offset=0.4)
    ranked = rank_block_confined_channels(
        block_deviation_table(data, labels, TIMES, ch_names=CH_NAMES)
    ).set_index('channel')

    assert ranked.loc['BAD_IN_B', 'block_offset'] == pytest.approx(0.4, abs=0.05)
    assert ranked.loc['BAD_IN_B', 'worst_block'] == (75.0, 75.0)
    for clean in ('CLEAN1', 'CLEAN2'):
        assert ranked.loc[clean, 'block_offset'] < 0.05


def test_nan_trials_are_counted_at_the_first_sample():
    data, labels = _data()
    data[:, 0, 0] = np.nan               # CLEAN1 is NaN at t=-1 in every trial
    table = block_deviation_table(data, labels, TIMES, ch_names=CH_NAMES)

    assert (table[table['channel'] == 'CLEAN1']['n_valid_first_sample'] == 0).all()
    assert (table[table['channel'] == 'CLEAN2']['n_valid_first_sample'] == 10).all()


def test_shape_mismatches_are_rejected():
    data, labels = _data()
    with pytest.raises(ValueError, match="block_labels"):
        block_deviation_table(data, labels[:-1], TIMES, ch_names=CH_NAMES)
    with pytest.raises(ValueError, match="ch_names"):
        block_deviation_table(data, labels, TIMES, ch_names=['only_one'])
    with pytest.raises(ValueError, match="times"):
        block_deviation_table(data, labels, TIMES[:-1], ch_names=CH_NAMES)


def test_block_labels_from_metadata_reports_missing_columns():
    md = pd.DataFrame({'incongruent_proportion': [75.0, 25.0],
                       'something_else': ['a', 'b']})
    labels, missing = block_labels_from_metadata(md)

    assert labels == [(75.0,), (25.0,)]
    assert missing == ['switch_proportion']

    with pytest.raises(KeyError):
        block_labels_from_metadata(md, columns=('nope', 'also_nope'))
