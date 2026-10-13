"""Window-mean decoding and the matched, membership-permutation group comparison."""
import os

import numpy as np
import mne

from src.analysis.decoding.window_mean_decoding import (
    compare_electrode_groups, window_mean_dataset)
from src.analysis.decoding.plots.group_bars import plot_group_accuracy_bars
from src.analysis.utils.labeled_array_utils import (
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel)

CATS = {('c',): 0, ('i',): 1}


def _datasets(n_boot=3, n_trials=60, n_sig=6, n_non=8, effect=0.8, seed=0):
    """Window means where only the first n_sig channels ('sig') carry the class."""
    rng = np.random.default_rng(seed)
    channels = np.array([f"S-e{i}" for i in range(n_sig + n_non)])
    out = []
    for _ in range(n_boot):
        y = np.repeat([0, 1], n_trials // 2)
        X = rng.normal(size=(n_trials, n_sig + n_non))
        X[y == 1, :n_sig] += effect
        out.append((X, y, CATS))
    groups = {'sig': channels[:n_sig], 'nonsig': channels[n_sig:], 'all': channels}
    return out, channels, groups


def test_window_mean_dataset_averages_the_window():
    mne.set_log_level('ERROR')
    rng = np.random.default_rng(0)

    def ep(n, shift):
        info = mne.create_info(['CH0', 'CH1'], 100., 'eeg')
        d = rng.normal(scale=0.01, size=(n, 2, 50))
        d[:, :, 20:] += shift                     # t >= 0
        return mne.EpochsArray(d, info, tmin=-0.2)

    subs = {'sub1': {'Stimulus_c': {'HG_ev1_power_rescaled': ep(10, 0.0)},
                     'Stimulus_i': {'HG_ev1_power_rescaled': ep(10, 1.0)}}}
    la = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
        ['lpfc'], subs, ['Stimulus_c', 'Stimulus_i'], ['sub1'],
        {'lpfc': {'sub1': ['CH0', 'CH1']}}, n_bootstraps=1, chans_axs=1, time_axs=-1,
        n_jobs=1, random_state=0)['lpfc'][0]
    X, y, cats, channels = window_mean_dataset(la, [['Stimulus_c'], ['Stimulus_i']],
                                               time_range=(0.0, 0.29))
    assert X.shape == (20, 2) and list(channels) == ['sub1-CH0', 'sub1-CH1']
    assert np.allclose(X[y == 1].mean(), 1.0, atol=0.02)
    assert np.allclose(X[y == 0].mean(), 0.0, atol=0.02)


def test_groups_are_matched_in_size_and_a_real_difference_is_found(tmp_path):
    data, channels, groups = _datasets()
    res = compare_electrode_groups(data, channels, groups, n_perm=40, n_splits=3, n_repeats=2)
    s = res['summary'].set_index('group')
    assert res['n_match'] == 6 and (s['n_match'] == 6).all()
    assert s.loc['sig', 'mean'] > 0.75 and s.loc['nonsig', 'mean'] < 0.62
    assert abs(s['chance'].mean() - 0.5) < 0.08
    t = res['tests'].set_index(['group_1', 'group_2'])
    assert t.loc[('sig', 'nonsig'), 'p'] < 0.05
    assert t.loc[('sig', 'nonsig'), 'difference'] > 0.15
    assert (res['tests']['p_holm'] >= res['tests']['p']).all()
    paths = plot_group_accuracy_bars(res, str(tmp_path / 'bars'), title='congruency')
    assert all(os.path.exists(p) for p in paths)


def test_no_difference_when_membership_does_not_matter():
    data, channels, _ = _datasets(n_sig=12, n_non=0, effect=0.5, seed=3)
    groups = {'a': channels[:6], 'b': channels[6:]}
    res = compare_electrode_groups(data, channels, groups, n_perm=40, n_splits=3, n_repeats=2)
    assert res['tests'].loc[0, 'p'] > 0.05


def test_nested_null_keeps_the_superset_whole():
    # 'all' contains 'sig': each permutation deals pseudo-sig from all, and the
    # back slice of the dealt pool is the pool itself
    data, channels, groups = _datasets(n_boot=2)
    res = compare_electrode_groups(data, channels, {'sig': groups['sig'], 'all': groups['all']},
                                   n_perm=20, n_splits=3, n_repeats=2)
    assert len(res['null'][('sig', 'all')]) == 20
    assert res['group_sizes'] == {'sig': 6, 'all': 14}
