"""Tests for the four-interaction electrode definition (`per_electrode_anova_labels`).

Covers the change that promotes the two CROSS interactions (congruency x
switch_proportion = CS, switchType x incongruent_proportion = SI) from report-only
p-values into full FDR'd, flagged electrode-definition groups alongside the two
constructs of interest (S = LWPC, F = LWPS). Runs on the module's synthetic
generator, whose ground truth has NO true cross-effect, so CS/SI should stay
~null while S/F recover the planted interactions.
"""

import numpy as np
import pytest

from src.analysis.stats import stability_flexibility_segregation as sfs


@pytest.fixture(scope="module")
def labels():
    df = sfs._synthetic_df(seed=1)
    return sfs.per_electrode_anova_labels(df, alpha=0.05, contrast_mode="proportion")


def test_all_four_interaction_flags_present(labels):
    for flag in ("S", "F", "CS", "SI"):
        assert flag in labels.columns, f"missing electrode-definition flag {flag!r}"
        assert set(labels[flag].unique()) <= {0, 1}


def test_cross_group_qvalues_present(labels):
    # cross interactions now get their own FDR'd q-values, not just raw p-values
    for col in ("p_cross_cs", "q_cross_cs", "cs_sign",
                "p_cross_si", "q_cross_si", "si_sign"):
        assert col in labels.columns, f"missing cross-interaction column {col!r}"


def test_constructs_recover_planted_effects(labels):
    # the generator plants congruency x inc_prop (S) and switchType x switch_prop (F)
    assert labels["S"].sum() > 0
    assert labels["F"].sum() > 0


def test_cross_groups_near_null(labels):
    # no true cross-effect in the generator -> far fewer CS/SI than S/F survivors
    n_construct = labels["S"].sum() + labels["F"].sum()
    n_cross = labels["CS"].sum() + labels["SI"].sum()
    assert n_cross < 0.5 * n_construct, (
        f"cross groups not near-null: S+F={n_construct}, CS+SI={n_cross} "
        "(Type III sum-coding orthogonalization may have failed)"
    )


def test_backward_compatible_without_cross_controls():
    df = sfs._synthetic_df(seed=2)
    lab = sfs.per_electrode_anova_labels(df, contrast_mode="proportion",
                                         include_cross_controls=False)
    # S/F path unchanged; no cross columns emitted
    assert {"S", "F", "q_cong", "q_switch"} <= set(lab.columns)
    assert "CS" not in lab.columns and "SI" not in lab.columns


def test_still_drops_into_cmh_conjunction(labels):
    # output stays a valid `labels` input to the conjunction (needs subject, S, F)
    res = sfs.cmh_conjunction(labels)
    assert "mh_odds_ratio" in res
