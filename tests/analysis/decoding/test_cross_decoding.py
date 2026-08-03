"""Tests for A4 cross-decoding: the label-pair glue and the Decoder features it needs.

The circularity/double-dipping table is tested separately in
``test_cross_decoding_circularity.py`` (pure predicates, no numerics).

These tests cover the three things that are genuinely new:
  1. `build_cross_decoding_arrays` — turning an ROI LabeledArray into two label
     vectors over the SAME trials, plus the condition-cell strata.
  2. `Decoder.cv_cm_jim_window_shuffle(labels_test=...)` — train on one labelling,
     score against another, with the CV split as the circularity guard.
  3. `frac_train` (train/test proportion) and `temporal_generalization`.

Everything else the cross-decode needs — the pseudopopulation, the shuffle null,
the cluster correction — is the ordinary decoding pipeline and is tested there.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.analysis.decoding.cross_decoding import (  # noqa: E402
    build_cross_decoding_arrays,
    resolve_contrast,
    synthetic_roi_labeled_arrays,
)

# the project's substring convention, matching `synthetic_roi_labeled_arrays`
STAB = [["_i_"], ["_c_"]]        # congruency:  i vs c
FLEX = [["_s_"], ["_r_"]]        # switch type: s vs r


def _decoder(**kw):
    """Import Decoder lazily so the array-building tests run without `ieeg`."""
    from src.analysis.decoding.decoder import Decoder
    kw.setdefault('n_splits', 3)
    kw.setdefault('n_repeats', 2)
    kw.setdefault('oversample', False)
    kw.setdefault('random_state', 7)
    return Decoder({('a',): 0, ('b',): 1}, 0.8, **kw)


ieeg_required = pytest.mark.skipif(
    __import__('importlib').util.find_spec('ieeg') is None,
    reason="requires the `ieeg` package (cluster environment)")


# ---------------------------------------------------------------------------
# 1. build_cross_decoding_arrays
# ---------------------------------------------------------------------------
def test_builds_two_label_vectors_over_the_same_trials():
    arrays = build_cross_decoding_arrays(
        synthetic_roi_labeled_arrays(seed=0), "synthetic", STAB, FLEX)

    n = len(arrays['data'])
    assert len(arrays['labels_train']) == n
    assert len(arrays['labels_test']) == n
    assert len(arrays['strata']) == n
    # both labellings are binary and both classes are present
    assert set(arrays['labels_train']) == {0, 1}
    assert set(arrays['labels_test']) == {0, 1}


def test_all_four_joint_cells_are_populated_and_balanced():
    """The two contrasts must CROSS, otherwise a transfer is not identifiable."""
    arrays = build_cross_decoding_arrays(
        synthetic_roi_labeled_arrays(seed=0), "synthetic", STAB, FLEX)
    pairs, counts = np.unique(
        np.stack([arrays['labels_train'], arrays['labels_test']], axis=1),
        axis=0, return_counts=True)
    assert len(pairs) == 4, "expected all four (train, test) label combinations"
    assert len(set(counts)) == 1, "the 2x2 of labels should be balanced"


def test_strata_index_source_conditions():
    arrays = build_cross_decoding_arrays(
        synthetic_roi_labeled_arrays(seed=0), "synthetic", STAB, FLEX)
    # one stratum per surviving condition, and strata are contiguous 0..k-1
    assert set(arrays['strata']) == set(range(len(arrays['conditions'])))
    # a stratum never spans two different label pairs — that is what makes
    # stratifying on it balance BOTH labellings at once
    for s in set(arrays['strata']):
        m = arrays['strata'] == s
        assert len(set(arrays['labels_train'][m])) == 1
        assert len(set(arrays['labels_test'][m])) == 1


def test_conditions_labelled_by_only_one_contrast_are_dropped():
    arrs = synthetic_roi_labeled_arrays(seed=0)
    # a condition the FLEX contrast cannot label (no _s_ / _r_ token)
    arrs['synthetic']['Stimulus_i_XX_25inc_25sw'] = np.zeros((5, 40, 32))
    out = build_cross_decoding_arrays(arrs, "synthetic", STAB, FLEX)
    assert 'Stimulus_i_XX_25inc_25sw' not in out['conditions']
    assert len(out['conditions']) == 16


def test_raises_when_no_condition_crosses_both_contrasts():
    arrs = {"roi": {"Stimulus_i_s_25inc_25sw": np.zeros((4, 3, 8))}}
    # only one condition -> the train contrast has a single class
    with pytest.raises(ValueError, match="every surviving condition the same class"):
        build_cross_decoding_arrays(arrs, "roi", STAB, FLEX)


def test_raises_on_unlabellable_conditions():
    arrs = {"roi": {"nothing_matches_here": np.zeros((4, 3, 8))}}
    with pytest.raises(ValueError, match="labelled by BOTH contrasts"):
        build_cross_decoding_arrays(arrs, "roi", STAB, FLEX)


def test_raises_on_unknown_roi():
    with pytest.raises(KeyError, match="not in roi_labeled_arrays"):
        build_cross_decoding_arrays(synthetic_roi_labeled_arrays(), "nope", STAB, FLEX)


def test_flat_string_list_is_accepted():
    """['_i_', '_c_'] should behave like [['_i_'], ['_c_']]."""
    arrs = synthetic_roi_labeled_arrays(seed=0)
    flat = build_cross_decoding_arrays(arrs, "synthetic", ["_i_", "_c_"], FLEX)
    nested = build_cross_decoding_arrays(arrs, "synthetic", STAB, FLEX)
    assert np.array_equal(flat['labels_train'], nested['labels_train'])


def test_resolve_contrast_names_and_callables():
    cell = {"congruency": "i", "switchType": "r"}
    assert resolve_contrast("congruency")(cell) == 1
    assert resolve_contrast("stability")(cell) == 1      # alias
    assert resolve_contrast("switchType")(cell) == 0
    assert resolve_contrast(lambda c: 42)(cell) == 42
    with pytest.raises(KeyError):
        resolve_contrast("not_a_contrast")


# ---------------------------------------------------------------------------
# 2. cross-label decoding through the ordinary Decoder
# ---------------------------------------------------------------------------
@ieeg_required
def test_labels_test_must_match_labels_length():
    X = np.random.default_rng(0).normal(0, 1, (20, 4, 16))
    with pytest.raises(ValueError, match="must label the SAME trials"):
        _decoder().cv_cm_jim_window_shuffle(
            X, np.array([0, 1] * 10), obs_axs=0, window=8,
            labels_test=np.array([0, 1] * 5))


@ieeg_required
def test_stratify_labels_must_match_labels_length():
    X = np.random.default_rng(0).normal(0, 1, (20, 4, 16))
    with pytest.raises(ValueError, match="stratify_labels"):
        _decoder().cv_cm_jim_window_shuffle(
            X, np.array([0, 1] * 10), obs_axs=0, window=8,
            stratify_labels=np.array([0, 1] * 3))


@ieeg_required
def test_cross_decode_scores_against_the_other_labelling():
    """With `labels_test`, accuracy must reflect the TEST labels, not the train ones.

    Data carries a signal aligned to `y_train`. Scored against `y_train` it decodes;
    scored against an independent `y_test` it must fall to chance — proof the
    confusion matrix is built from the second labelling.
    """
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    rng = np.random.default_rng(0)
    n = 160
    X = rng.normal(0, 1, (n, 6, 32))
    y_train = np.array([0, 1] * (n // 2))
    X[y_train == 1] += 1.0                       # signal follows y_train only
    y_indep = rng.permutation(y_train)           # unrelated labelling

    kw = dict(normalize='true', obs_axs=0, window=16, step_size=16)
    same = _decoder().cv_cm_jim_window_shuffle(X, y_train, **kw)
    cross = _decoder().cv_cm_jim_window_shuffle(X, y_train, labels_test=y_indep, **kw)

    acc_same, acc_cross = compute_accuracies(same, cross)
    assert acc_same.mean() > 0.85, "self-decode should be easy on a planted signal"
    assert 0.35 < acc_cross.mean() < 0.65, "an unrelated test labelling must be chance"


@ieeg_required
def test_shared_code_transfers_and_orthogonal_code_does_not():
    """The ground-truth check the whole analysis rests on."""
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    from src.analysis.decoding.cross_decoding import run_cross_decoding

    accs = {}
    for code in ("shared", "orthogonal"):
        arrs = synthetic_roi_labeled_arrays(code=code, seed=1)
        for name, (tr, te) in (("within", (STAB, STAB)), ("cross", (STAB, FLEX))):
            out = run_cross_decoding(arrs, "synthetic", tr, te, n_splits=3,
                                     n_repeats=2, window=16, step_size=16)
            acc, _ = compute_accuracies(out['cm_true'], out['cm_shuffle'])
            accs[(code, name)] = acc.mean()

    # both worlds are individually decodable ...
    assert accs[("shared", "within")] > 0.65
    assert accs[("orthogonal", "within")] > 0.65
    # ... but only the shared code transfers across contrasts
    assert accs[("shared", "cross")] > 0.65
    assert 0.35 < accs[("orthogonal", "cross")] < 0.65


@ieeg_required
def test_shuffle_null_is_at_chance_for_a_real_cross_decode():
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    from src.analysis.decoding.cross_decoding import run_cross_decoding
    out = run_cross_decoding(synthetic_roi_labeled_arrays(code="shared", seed=1),
                             "synthetic", STAB, FLEX, n_splits=3, n_repeats=2,
                             window=16, step_size=16)
    _, acc_shuffle = compute_accuracies(out['cm_true'], out['cm_shuffle'])
    assert 0.35 < acc_shuffle.mean() < 0.65, "permuting train labels must give chance"


# ---------------------------------------------------------------------------
# 3. frac_train and temporal generalization
# ---------------------------------------------------------------------------
@ieeg_required
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_frac_train_is_validated(bad):
    X = np.random.default_rng(0).normal(0, 1, (20, 4, 16))
    with pytest.raises(ValueError, match="frac_train"):
        _decoder().cv_cm_jim_window_shuffle(
            X, np.array([0, 1] * 10), obs_axs=0, window=8, frac_train=bad)


@ieeg_required
@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_frac_train_runs_and_keeps_the_output_shape(frac):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 4, 32))
    y = np.array([0, 1] * 40)
    out = _decoder().cv_cm_jim_window_shuffle(
        X, y, normalize='true', obs_axs=0, window=16, step_size=16, frac_train=frac)
    assert out.shape == (2, 2, 2, 2)   # (windows, repeats, cats, cats)


@ieeg_required
def test_frac_train_changes_the_training_set_size():
    """A smaller training fraction should cost accuracy on a fixed weak signal."""
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 6, 32))
    y = np.array([0, 1] * 100)
    X[y == 1] += 0.45                       # weak enough that training size matters

    kw = dict(normalize='true', obs_axs=0, window=32, step_size=32)
    small = _decoder(n_repeats=6).cv_cm_jim_window_shuffle(X, y, frac_train=0.05, **kw)
    large = _decoder(n_repeats=6).cv_cm_jim_window_shuffle(X, y, frac_train=0.9, **kw)
    acc_small, acc_large = compute_accuracies(small, large)
    assert acc_large.mean() > acc_small.mean()


@ieeg_required
def test_temporal_generalization_requires_a_window():
    X = np.random.default_rng(0).normal(0, 1, (20, 4, 16))
    with pytest.raises(ValueError, match="requires `window`"):
        _decoder().cv_cm_jim_window_shuffle(
            X, np.array([0, 1] * 10), obs_axs=0, temporal_generalization=True)


@ieeg_required
def test_temporal_generalization_shape_and_diagonal():
    """The tempgen diagonal (train t, test t) must equal the ordinary windowed run."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (160, 8, 64))
    y = np.array([0, 1] * 80)
    X[y == 1, :, 16:48] += 0.8
    kw = dict(normalize='true', obs_axs=0, window=16, step_size=16)

    np.random.seed(3)
    plain = _decoder().cv_cm_jim_window_shuffle(X, y, **kw)
    np.random.seed(3)
    tg = _decoder().cv_cm_jim_window_shuffle(X, y, temporal_generalization=True, **kw)

    n_win = plain.shape[0]
    assert tg.shape == (n_win, n_win) + plain.shape[1:]
    diag = np.stack([tg[i, i] for i in range(n_win)])
    assert np.allclose(diag, plain, equal_nan=True)


@ieeg_required
def test_temporal_generalization_with_folds_as_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 4, 64))
    y = np.array([0, 1] * 40)
    out = _decoder().cv_cm_jim_window_shuffle(
        X, y, normalize='true', obs_axs=0, window=16, step_size=16,
        temporal_generalization=True, folds_as_samples=True)
    # (train_windows, test_windows, n_repeats * n_splits, cats, cats)
    assert out.shape == (4, 4, 2 * 3, 2, 2)


@ieeg_required
def test_missing_class_in_a_fold_still_returns_a_full_confusion_matrix():
    """`cm_labels` pins the CM axes so a lopsided fold cannot reshape the output."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 4, 32))
    y_train = np.array([0, 1] * 30)
    # a test labelling that is almost entirely one class
    y_test = np.zeros(60, dtype=int)
    y_test[:2] = 1
    out = _decoder().cv_cm_jim_window_shuffle(
        X, y_train, labels_test=y_test, normalize='true', obs_axs=0,
        window=16, step_size=16, stratify_labels=y_test)
    assert out.shape[-2:] == (2, 2)
