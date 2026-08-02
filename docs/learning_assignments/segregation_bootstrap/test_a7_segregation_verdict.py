"""Self-check for A7. Run from the repo root:

    python -m pytest docs/learning_assignments/segregation_bootstrap/test_a7_segregation_verdict.py -q

RED now (the stub raises NotImplementedError); GREEN when your implementation is
correct. Each test says, in its name and asserts, exactly which idea it checks.
The synthetic-data tests lean on `_synthetic_df`, whose bx/by are drawn
INDEPENDENTLY, so the ground truth is "no real overlap" -> OR ~ 1, verdict
'inconclusive'. If your code manufactures a signal there, a test will catch it.
"""

import os
import sys

import numpy as np
import pytest

# Make both the repo root (for `src...`) and this folder (for the stub) importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.analysis.stats import stability_flexibility_segregation as seg  # noqa: E402
from a7_segregation_verdict import (  # noqa: E402
    bootstrap_conjunction_or,
    classify_segregation,
    segregation_verdict,
)


# --------------------------------------------------------------------------- #
# Fixtures — build synthetic labels ONCE (permutation labelling is the slow bit)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_df():
    return seg._synthetic_df(effect_measure="cohens_d", seed=0)


@pytest.fixture(scope="module")
def labels(synthetic_df):
    # small n_perm keeps the self-check quick; the labels are still valid S/F
    return seg.per_electrode_labels(
        synthetic_df, n_perm=50, alpha=0.05, contrast_mode="condition"
    )


# --------------------------------------------------------------------------- #
# Task 2 — the decision rule. Pure, fast, no data: check every branch.
# --------------------------------------------------------------------------- #
def test_classify_segregated_when_both_layers_agree_low():
    cont = {"corr": -0.4, "p": 0.001}
    cat = {"mh_odds_ratio": 0.35, "p": 0.001}
    assert classify_segregation(cont, cat, alpha=0.05) == "segregated"


def test_classify_shared_when_both_layers_agree_high():
    cont = {"corr": 0.4, "p": 0.001}
    cat = {"mh_odds_ratio": 2.5, "p": 0.001}
    assert classify_segregation(cont, cat, alpha=0.05) == "shared"


def test_classify_inconclusive_when_nothing_significant():
    cont = {"corr": -0.4, "p": 0.40}   # direction is segregated but NOT significant
    cat = {"mh_odds_ratio": 0.35, "p": 0.60}
    assert classify_segregation(cont, cat, alpha=0.05) == "inconclusive"


def test_classify_conflicting_when_layers_disagree():
    cont = {"corr": -0.4, "p": 0.001}  # continuous says segregated
    cat = {"mh_odds_ratio": 2.5, "p": 0.001}  # categorical says shared
    assert classify_segregation(cont, cat, alpha=0.05) == "conflicting"


def test_classify_one_significant_layer_carries():
    # only the categorical layer is significant -> its vote decides
    cont = {"corr": 0.1, "p": 0.9}
    cat = {"mh_odds_ratio": 0.3, "p": 0.01}
    assert classify_segregation(cont, cat, alpha=0.05) == "segregated"


# --------------------------------------------------------------------------- #
# Task 1 — subject-clustered bootstrap OR CI.
# --------------------------------------------------------------------------- #
def test_bootstrap_point_matches_cmh(labels):
    out = bootstrap_conjunction_or(labels, n_boot=200, seed=1)
    expected = seg.cmh_conjunction(labels)["mh_odds_ratio"]
    assert out["point"] == pytest.approx(expected, rel=1e-9)


def test_bootstrap_ci_brackets_point(labels):
    out = bootstrap_conjunction_or(labels, n_boot=400, seed=1)
    assert out["lo"] <= out["point"] <= out["hi"]


def test_bootstrap_ci_covers_one_on_independent_synthetic(labels):
    # ground truth: bx, by independent -> no real overlap -> OR ~ 1
    out = bootstrap_conjunction_or(labels, n_boot=500, seed=1)
    assert out["lo"] <= 1.0 <= out["hi"]


def test_bootstrap_is_reproducible(labels):
    a = bootstrap_conjunction_or(labels, n_boot=200, seed=7)
    b = bootstrap_conjunction_or(labels, n_boot=200, seed=7)
    assert np.allclose(np.asarray(a["boots"]), np.asarray(b["boots"]))


def test_bootstrap_resamples_whole_subjects_not_electrodes(labels):
    # A subject drawn twice must become two separate strata, so the CI must
    # reflect between-subject variability: it should be WIDER than a naive CI
    # that treats every electrode as its own stratum. We approximate the naive
    # version by relabelling each electrode as its own "subject" and bootstrapping
    # THAT; the correct subject-clustered CI should not be narrower than it.
    correct = bootstrap_conjunction_or(labels, n_boot=400, seed=3)
    per_elec = labels.copy()
    per_elec["subject"] = per_elec["electrode"].astype(str)
    naive = bootstrap_conjunction_or(per_elec, n_boot=400, seed=3)
    correct_width = correct["hi"] - correct["lo"]
    naive_width = naive["hi"] - naive["lo"]
    assert correct_width >= naive_width * 0.9  # not artificially narrow


# --------------------------------------------------------------------------- #
# Task 3 — end-to-end verdict on synthetic ground truth.
# --------------------------------------------------------------------------- #
def test_verdict_structure_and_inconclusive(synthetic_df):
    out = segregation_verdict(
        synthetic_df, n_splits=20, n_perm=800, n_boot=300,
        alpha=0.05, contrast_mode="condition", seed=0,
    )
    assert set(out) >= {"verdict", "continuous", "categorical", "contrast_mode"}
    assert set(out["categorical"]) >= {
        "mh_odds_ratio", "p", "or_lo", "or_hi", "observed_both"
    }
    # independent bx/by -> the honest answer is "no evidence either way"
    assert out["verdict"] == "inconclusive"
    # and the OR CI must bracket 1
    assert out["categorical"]["or_lo"] <= 1.0 <= out["categorical"]["or_hi"]
