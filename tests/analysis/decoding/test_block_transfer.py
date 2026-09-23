"""Tests for N3b block-transfer cross-decoding (docs/n3b_block_transfer.md).

1. `Decoder.cv_cm_jim_window_shuffle(test_only=...)` — train on one set of trials,
   score on another, with folds cut only from the training trials.
2. `block_transfer.prepare / balanced_subsample / center_levels` — the balance
   groups and the balance-then-center order (no `ieeg` needed).
3. `block_transfer.run_block_transfer` on synthetic data with a planted answer:
   a block-invariant code transfers, a block-specific code fails X1 but passes
   X3, and a tonic block offset breaks only UNcentered transfer.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.analysis.decoding import block_transfer as bt  # noqa: E402
from src.analysis.decoding import cross_decoding as cd  # noqa: E402

CELLS = cd.synthetic_condition_cells()
X1 = ('congruency', 'incongruent_proportion')
X3 = ('congruency', 'switch_proportion')

ieeg_required = pytest.mark.skipif(
    __import__('importlib').util.find_spec('ieeg') is None,
    reason="requires the `ieeg` package (cluster environment)")


def _decoder(**kw):
    """Import Decoder lazily so the pure-array tests run without `ieeg`."""
    from src.analysis.decoding.decoder import Decoder
    kw.setdefault('n_splits', 3)
    kw.setdefault('n_repeats', 2)
    kw.setdefault('oversample', False)
    kw.setdefault('random_state', 7)
    return Decoder({('a',): 0, ('b',): 1}, 0.8, **kw)


def _record_folds(monkeypatch):
    """Spy on every fold: record the trial ids (stored in feature [0, 0]) that
    were trained on and scored."""
    from src.analysis.decoding.decoder import Decoder
    seen = []
    original = Decoder._window_and_predict_minimal

    def spy(self, x_train, y_train, x_test, y_test, *args, **kwargs):
        seen.append((x_train[:, 0, 0].copy(), x_test[:, 0, 0].copy()))
        return original(self, x_train, y_train, x_test, y_test, *args, **kwargs)

    monkeypatch.setattr(Decoder, '_window_and_predict_minimal', spy)
    return seen


def _trials_with_ids(n=60, seed=0):
    X = np.random.default_rng(seed).normal(0, 1, (n, 4, 16))
    X[:, 0, 0] = np.arange(n)                  # trial id, readable inside every fold
    return X, np.array([0, 1] * (n // 2))


# ---------------------------------------------------------------------------
# 1. test_only in the Decoder
# ---------------------------------------------------------------------------
@ieeg_required
def test_test_only_trials_are_never_trained_on_and_always_scored(monkeypatch):
    X, y = _trials_with_ids()
    test_only = np.arange(60) >= 40
    seen = _record_folds(monkeypatch)

    _decoder().cv_cm_jim_window_shuffle(X, y, obs_axs=0, window=8, step_size=8,
                                        test_only=test_only)

    assert len(seen) == 2 * 3                          # n_repeats x n_splits folds
    for train_ids, test_ids in seen:
        assert not np.isin(train_ids, np.arange(40, 60)).any()
        assert np.array_equal(np.sort(test_ids), np.arange(40, 60))


@ieeg_required
def test_transfer_trains_on_the_same_folds_as_the_within_block_decode(monkeypatch):
    """Same seed -> the transfer's classifiers are the within-block decode's
    classifiers, which is what makes the two accuracies comparable."""
    X, y = _trials_with_ids()
    test_only = np.arange(60) >= 40
    strata = np.arange(60) % 4
    kw = dict(obs_axs=0, window=8, step_size=8)
    seen = _record_folds(monkeypatch)

    _decoder().cv_cm_jim_window_shuffle(X[~test_only], y[~test_only],
                                        stratify_labels=strata[~test_only], **kw)
    within = [train for train, _ in seen]
    seen.clear()
    _decoder().cv_cm_jim_window_shuffle(X, y, stratify_labels=strata,
                                        test_only=test_only, **kw)
    transfer = [train for train, _ in seen]

    assert len(within) == len(transfer)
    for a, b in zip(within, transfer):
        assert np.array_equal(a, b)


@ieeg_required
def test_test_only_is_validated():
    X, y = _trials_with_ids()
    with pytest.raises(ValueError, match="test_only has"):
        _decoder().cv_cm_jim_window_shuffle(X, y, obs_axs=0, window=8,
                                            test_only=np.zeros(10, bool))
    with pytest.raises(ValueError, match="at least one trial on each side"):
        _decoder().cv_cm_jim_window_shuffle(X, y, obs_axs=0, window=8,
                                            test_only=np.ones(60, bool))


# ---------------------------------------------------------------------------
# 2. balance groups and the balance-then-center order
# ---------------------------------------------------------------------------
def _task_like_arrays(**kw):
    """Synthetic trials with the task's 3:1 cell proportions."""
    return cd.synthetic_roi_labeled_arrays(code='orthogonal', design_proportions=True, **kw)


def test_prepare_groups_are_one_class_in_one_physical_block():
    arrs = _task_like_arrays(seed=0)
    name = 'Stimulus_i_s_25inc_25sw'
    arrs['synthetic'][name] = np.concatenate(
        [arrs['synthetic'][name], np.full((50, 40, 32), np.nan)])   # padding rows

    p = bt.prepare(arrs, 'synthetic', CELLS, *X1)

    sizes = dict(zip(*np.unique(p['groups'], return_counts=True)))
    assert len(sizes) == 8                        # 2 classes x 4 physical blocks
    assert sizes['i|inc25|sw25'] == 40            # 30 + 10 real trials, padding gone
    assert sizes['c|inc25|sw25'] == 120           # the 3:1 majority
    assert set(p['block']) == {25, 75}
    assert not np.isnan(p['X']).all(axis=(1, 2)).any()


def test_prepare_needs_both_block_factors():
    pooled = {name: dict(cell, incongruent_proportion=None) for name, cell in CELLS.items()}
    with pytest.raises(ValueError, match="stimulus_experiment_conditions"):
        bt.prepare(_task_like_arrays(), 'synthetic', pooled, *X1)


def test_balanced_subsample_keeps_the_smallest_group_size_from_every_group():
    groups = np.array(['a'] * 5 + ['b'] * 9 + ['c'] * 7)
    k = bt.balanced_subsample(groups, np.random.default_rng(0))
    assert np.array_equal(k, np.sort(k)) and len(set(k)) == len(k)
    assert dict(zip(*np.unique(groups[k], return_counts=True))) == {'a': 5, 'b': 5, 'c': 5}


def _midpoint_along_class_axis(X, y, level_mask):
    """Where a level's class midpoint sits along its class difference, in units
    of that difference (0 = centered on the midpoint)."""
    m0, m1 = X[level_mask & (y == 0)].mean(0), X[level_mask & (y == 1)].mean(0)
    d = m1 - m0
    return float(((m0 + m1) / 2 * d).sum() / (d * d).sum())


def test_centering_balanced_trials_puts_each_levels_class_midpoint_at_zero():
    """And centering the raw 3:1 trials would not: each level's midpoint lands a
    quarter of the class difference off zero, in opposite directions for the two
    levels, i.e. a spurious shift along the decoded axis."""
    p = bt.prepare(_task_like_arrays(seed=0), 'synthetic', CELLS, *X1)
    X, y, block = p['X'], p['y'], p['block']

    k = bt.balanced_subsample(p['groups'], np.random.default_rng(0))
    balanced = bt.center_levels(X[k], block[k])
    raw = bt.center_levels(X, block)

    for level in (25, 75):
        assert abs(_midpoint_along_class_axis(balanced, y[k], block[k] == level)) < 1e-9
        assert np.isclose(abs(_midpoint_along_class_axis(raw, y, block == level)), 0.25)


# ---------------------------------------------------------------------------
# 3. run_block_transfer on a planted answer
# ---------------------------------------------------------------------------
def _accuracies(arrs, design, center=False):
    """{(train_level, test_level): mean accuracy} for one design."""
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    out = bt.run_block_transfer(arrs, 'synthetic', CELLS, *design, center=center,
                                n_resamples=2, n_splits=3, window=16, step_size=16)
    accs = {}
    for pair, cms in out['cells'].items():
        assert cms['cm_true'].shape == (2, 2, 2, 2)    # (windows, resamples, cats, cats)
        acc, acc_shuffle = compute_accuracies(cms['cm_true'], cms['cm_shuffle'])
        assert 0.35 < acc_shuffle.mean() < 0.65          # the shuffle null is at chance
        accs[pair] = acc.mean()
    return accs


@ieeg_required
def test_a_block_invariant_code_transfers_as_well_as_it_decodes():
    acc = _accuracies(_task_like_arrays(seed=0), X1)
    assert min(acc.values()) > 0.7
    # each transfer vs the ceiling of the level it is SCORED on
    assert abs(acc[(25, 75)] - acc[(75, 75)]) < 0.07
    assert abs(acc[(75, 25)] - acc[(25, 25)]) < 0.07


@ieeg_required
def test_a_block_specific_code_fails_x1_but_still_passes_x3():
    """Congruency on a different axis in each incongruent-proportion level: X1
    must fail in both directions, while X3 (across switch proportion, same
    trials) still transfers."""
    arrs = _task_like_arrays(seed=0, block_code='specific')
    x1, x3 = _accuracies(arrs, X1), _accuracies(arrs, X3)
    assert x1[(25, 25)] > 0.7 and x1[(75, 75)] > 0.7
    assert x1[(25, 75)] < 0.6 and x1[(75, 25)] < 0.6
    assert x3[(25, 75)] > 0.6 and x3[(75, 25)] > 0.6
    assert x3[(75, 75)] - x3[(25, 75)] < 0.07 and x3[(25, 25)] - x3[(75, 25)] < 0.07


@ieeg_required
def test_a_tonic_block_offset_breaks_only_uncentered_transfer():
    arrs = _task_like_arrays(seed=0, block_offset=2.0)
    raw, centered = _accuracies(arrs, X1), _accuracies(arrs, X1, center=True)
    assert raw[(25, 75)] < 0.6 and raw[(75, 25)] < 0.6
    assert centered[(25, 75)] > 0.75 and centered[(75, 25)] > 0.75
    # centering subtracts the same vector from a level's train and test trials,
    # so the within-level ceilings cannot move
    assert np.isclose(raw[(25, 25)], centered[(25, 25)], atol=0.02)
    assert np.isclose(raw[(75, 75)], centered[(75, 75)], atol=0.02)
