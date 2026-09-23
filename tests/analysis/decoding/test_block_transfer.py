"""Tests for N3b block-transfer cross-decoding (docs/n3b_block_transfer.md).

1. `Decoder.cv_cm_jim_window_shuffle(test_only=...)` — train on one set of trials,
   score on another, with folds cut only from the training trials.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

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
