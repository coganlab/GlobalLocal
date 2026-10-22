"""Window-mean decoding and the matched, membership-permutation group comparison."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import mne

from dcc_scripts.decoding.window_mean_groups_dcc import run_window_mean_group_comparison
from src.analysis.decoding.window_mean_decoding import (
    accuracy_by_electrode_count, compare_electrode_groups, electrode_counts, window_mean_dataset)
from src.analysis.decoding.plots.electrode_curve import plot_accuracy_by_electrode_count
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


def test_unmatched_decodes_each_group_whole(tmp_path):
    data, channels, groups = _datasets()
    res = compare_electrode_groups(data, channels, groups, match_electrodes=False, n_perm=20,
                                   n_splits=3, n_repeats=2)
    s = res['summary'].set_index('group')
    assert res['n_match'] is None
    assert s['n_decoded'].to_dict() == {'sig': 6, 'nonsig': 8, 'all': 14}
    assert s.loc['sig', 'mean'] > 0.75 and s.loc['nonsig', 'mean'] < 0.62
    t = res['tests'].set_index(['group_1', 'group_2'])
    # no permutation reaches sig's lead: the smallest two-sided p
    assert t.loc[('sig', 'nonsig'), 'p'] == pytest.approx(2 / 21)
    paths = plot_group_accuracy_bars(res, str(tmp_path / 'bars'))
    assert all(os.path.exists(p) for p in paths)


def test_unmatched_null_carries_the_size_difference():
    # every channel carries the class, so the 9-electrode group wins on size
    # alone; the permutation deals groups of 3 and 9 too, so its null is centred
    # below zero and the difference is not significant
    data, channels, _ = _datasets(n_sig=12, n_non=0, effect=0.3, seed=5)
    groups = {'small': channels[:3], 'large': channels[3:]}
    res = compare_electrode_groups(data, channels, groups, match_electrodes=False, n_perm=40,
                                   n_splits=3, n_repeats=2)
    t = res['tests'].iloc[0]
    assert res['null'][('small', 'large')].mean() < -0.02
    assert t['difference'] < -0.02 and t['p'] > 0.05


def test_n_match_needs_matching():
    data, channels, groups = _datasets(n_boot=1)
    with pytest.raises(ValueError, match='match_electrodes'):
        compare_electrode_groups(data, channels, groups, n_match=4, match_electrodes=False)


def test_dcc_run_unmatched_writes_its_own_tag(tmp_path):
    mne.set_log_level('ERROR')
    rng = np.random.default_rng(0)
    chans = [f'CH{i}' for i in range(5)]

    def ep(shift):
        d = rng.normal(size=(20, 5, 30))
        d[:, :3, 10:] += shift                    # CH0-2 carry the condition
        return mne.EpochsArray(d, mne.create_info(chans, 100., 'eeg'), tmin=-0.1)

    subs = {'sub1': {'Stimulus_c': {'HG_ev1_power_rescaled': ep(0.0)},
                     'Stimulus_i': {'HG_ev1_power_rescaled': ep(1.0)}}}
    args = SimpleNamespace(
        condition_label='congruency', window_mean_tmin=0.0, window_mean_tmax=0.19,
        window_mean_bootstraps=2, window_mean_n_perm=4, window_mean_n_match=None,
        window_mean_match_electrodes=False, window_mean_curve_subsets=2, random_state=0,
        subjects=['sub1'], chans_axs=1,
        time_axs=-1, balance_method='subsample', balance_strata=True, n_jobs=1,
        clf_model=None, clf_model_str='lda', explained_variance=0.9, n_splits=3, n_repeats=1,
        epochs_root_file='synthetic')
    run_window_mean_group_comparison(
        args, ['lpfc'], ['Stimulus_c', 'Stimulus_i'],
        {'c_vs_i': [['Stimulus_c'], ['Stimulus_i']]}, {'lpfc': {'sub1': chans}},
        {'lpfc': {'sub1': chans[:3]}}, subs, str(tmp_path))
    out = tmp_path / 'window_mean_groups' / 'congruency_0to0.19s_2bts_4perm_seed0_unmatched'
    out = out / 'c_vs_i' / 'lpfc'
    params = json.loads((out / 'params.json').read_text())
    assert params['match_electrodes'] is False and params['n_match'] is None
    assert params['group_sizes'] == {'sig': 3, 'nonsig': 2, 'all': 5}
    assert (out / 'group_bars.png').exists()
    assert (out / 'electrode_curve.png').exists() and (out / 'curve_summary.csv').exists()


def test_electrode_counts_end_at_the_group_size():
    assert electrode_counts(171, 'log') == [1, 2, 4, 8, 16, 32, 64, 128, 171]
    assert electrode_counts(128, 'log') == [1, 2, 4, 8, 16, 32, 64, 128]
    assert electrode_counts(65, 'linear', step=20) == [20, 40, 60, 65]
    assert electrode_counts(1, 'log') == electrode_counts(1, 'linear') == [1]
    with pytest.raises(ValueError, match='spacing'):
        electrode_counts(10, 'sqrt')


def test_curve_rises_and_ends_at_the_unmatched_bars(tmp_path):
    data, channels, groups = _datasets()
    kw = dict(n_splits=3, n_repeats=2, seed=7)
    curve = accuracy_by_electrode_count(data, channels, groups, step=4, n_subsets=3, **kw)
    acc, s = curve['accuracies'], curve['summary'].set_index(['group', 'n_electrodes'])
    # sig: log grid 1, 2, 4, 6 and linear grid 4, 6
    assert sorted(acc.loc[acc['group'] == 'sig', 'n_electrodes'].unique()) == [1, 2, 4, 6]
    assert s.loc[('sig', 2), 'log_grid'] and not s.loc[('sig', 2), 'linear_grid']
    # 3 subsets per bootstrap below the full size, 1 at it
    per = acc.groupby(['group', 'n_electrodes']).size()
    assert per[('sig', 4)] == 3 * len(data) and per[('sig', 6)] == len(data)
    assert s.loc[('sig', 6), 'mean'] > s.loc[('sig', 1), 'mean'] + 0.05
    bars = compare_electrode_groups(data, channels, groups, match_electrodes=False, n_perm=0,
                                    **kw)
    assert bars['tests'].empty
    for g, size in bars['group_sizes'].items():
        assert s.loc[(g, size), 'mean'] == pytest.approx(
            bars['summary'].set_index('group').loc[g, 'mean'])
    paths = plot_accuracy_by_electrode_count(curve, str(tmp_path / 'curve'), title='congruency')
    assert all(os.path.exists(p) for p in paths)
