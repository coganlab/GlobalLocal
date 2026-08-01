"""Unit tests for the disjoint trial-splitting primitives (circularity fix).

These cover the pure functions in
``src/analysis/decoding/trial_splitting.py``. The pipeline orchestration
(`apply_electrode_definition_split`) needs real MNE Epochs and is smoke-tested on
the cluster, not here.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.decoding.trial_splitting import (
    stratified_trial_split,
    strata_key_from_metadata,
    select_responsive_channels,
)


# --------------------------- stratified_trial_split -------------------------
def test_split_is_disjoint_and_covers_all():
    strata = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    idx_def, idx_dec = stratified_trial_split(strata, frac_def=0.5, seed=0)
    # disjoint
    assert set(idx_def).isdisjoint(set(idx_dec))
    # covers every trial exactly once
    assert sorted(np.concatenate([idx_def, idx_dec])) == list(range(len(strata)))


def test_split_preserves_stratum_proportions():
    # 100 trials of stratum A, 40 of stratum B
    strata = np.array(["A"] * 100 + ["B"] * 40)
    idx_def, idx_dec = stratified_trial_split(strata, frac_def=0.5, seed=1)
    def_set = set(idx_def)
    a_idx = set(range(100))
    b_idx = set(range(100, 140))
    # ~half of each stratum on the definition side
    assert abs(len(def_set & a_idx) - 50) <= 1
    assert abs(len(def_set & b_idx) - 20) <= 1


def test_split_is_deterministic_with_seed():
    strata = np.array([0, 1] * 25)
    a = stratified_trial_split(strata, frac_def=0.5, seed=7)
    b = stratified_trial_split(strata, frac_def=0.5, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_split_independent_of_input_order():
    # Same multiset of strata, shuffled order → same partition of stratum members.
    base = np.array(["A"] * 6 + ["B"] * 6)
    perm = base.copy()
    rng = np.random.RandomState(3)
    order = rng.permutation(len(base))
    # Map: a fixed seed should split each stratum's *members* reproducibly by
    # sorted-unique stratum iteration, so the count on each side is stable.
    d1, _ = stratified_trial_split(base, frac_def=0.5, seed=0)
    d2, _ = stratified_trial_split(perm[order], frac_def=0.5, seed=0)
    assert len(d1) == len(d2)


def test_frac_def_controls_sizes():
    strata = np.zeros(100, dtype=int)
    idx_def, idx_dec = stratified_trial_split(strata, frac_def=0.8, seed=0)
    assert abs(len(idx_def) - 80) <= 1
    assert len(idx_def) + len(idx_dec) == 100


def test_singleton_stratum_goes_to_definition():
    strata = np.array([0, 1, 1, 1, 1])  # stratum 0 has a single trial
    idx_def, idx_dec = stratified_trial_split(strata, frac_def=0.5, seed=0)
    assert 0 in set(idx_def)  # never silently dropped
    assert set(idx_def).isdisjoint(set(idx_dec))


def test_empty_input():
    idx_def, idx_dec = stratified_trial_split(np.array([]), frac_def=0.5)
    assert len(idx_def) == 0 and len(idx_dec) == 0


def test_bad_frac_raises():
    with pytest.raises(ValueError):
        stratified_trial_split(np.array([0, 1]), frac_def=0.0)
    with pytest.raises(ValueError):
        stratified_trial_split(np.array([0, 1]), frac_def=1.0)


# --------------------------- strata_key_from_metadata -----------------------
def test_strata_key_combines_present_columns():
    md = pd.DataFrame({
        "congruency": ["c", "i", "c", "i"],
        "switchType": ["s", "s", "r", "r"],
    })
    key = strata_key_from_metadata(md, ["congruency", "switchType"])
    assert list(key) == ["c|s", "i|s", "c|r", "i|r"]


def test_strata_key_skips_missing_columns_with_warning():
    md = pd.DataFrame({"congruency": ["c", "i"]})
    with pytest.warns(UserWarning):
        key = strata_key_from_metadata(md, ["congruency", "not_a_col"])
    assert list(key) == ["c", "i"]


def test_strata_key_all_missing_gives_constant():
    md = pd.DataFrame({"congruency": ["c", "i"]})
    with pytest.warns(UserWarning):
        key = strata_key_from_metadata(md, ["nope"])
    assert len(set(key)) == 1  # constant → unstratified random split


def test_strata_key_none_gives_constant():
    md = pd.DataFrame({"congruency": ["c", "i", "c"]})
    key = strata_key_from_metadata(md, None)
    assert len(set(key)) == 1


# --------------------------- select_responsive_channels ---------------------
def test_selects_only_the_responsive_channel_one_sample():
    rng = np.random.RandomState(0)
    n_trials = 200
    # channel 0: strong positive window activity; channel 1: centered at 0
    ch0 = rng.normal(1.0, 1.0, n_trials)
    ch1 = rng.normal(0.0, 1.0, n_trials)
    window_means = np.column_stack([ch0, ch1])
    mask = select_responsive_channels(window_means, baseline_means=None, alpha=0.05)
    assert mask[0] and not mask[1]


def test_paired_test_detects_window_vs_baseline():
    rng = np.random.RandomState(0)
    n_trials = 200
    baseline = rng.normal(0.0, 1.0, (n_trials, 2))
    window = baseline.copy()
    window[:, 0] += 0.8  # channel 0 rises above its own baseline
    mask = select_responsive_channels(window, baseline_means=baseline, alpha=0.05)
    assert mask[0] and not mask[1]


def test_too_few_trials_returns_all_false():
    mask = select_responsive_channels(np.array([[1.0, 2.0]]), alpha=0.05)
    assert mask.shape == (2,) and not mask.any()


def test_zero_variance_channel_not_selected():
    window = np.column_stack([np.ones(50) * 5.0, np.random.RandomState(0).normal(0, 1, 50)])
    # channel 0 has zero variance → t is undefined (nan) → not selected
    mask = select_responsive_channels(window, alpha=0.05)
    assert not mask[0]
