"""Tests for the four-interaction electrode definition (`per_electrode_anova_labels`).

The four interaction-defined electrode groups are named `{condition}P{modulator}`:

- `CPC` = congruency x proportion-congruent (incongruent_proportion) -- LWPC / stability
- `SPS` = switchType x switch-proportion                             -- LWPS / flexibility
- `CPS` = congruency x switch-proportion                             -- cross
- `SPC` = switchType x proportion-congruent (incongruent_proportion) -- cross

This covers the change that promotes the two CROSS interactions (CPS, SPC) from
report-only p-values into full FDR'd, flagged groups alongside the two constructs
(CPC, SPS), plus the backward-compatible S=CPC / F=SPS aliases. Runs on the
module's synthetic generator, whose ground truth has NO cross-effect, so CPS/SPC
should stay ~null while CPC/SPS recover the planted interactions.
"""

import pytest

from src.analysis.stats import stability_flexibility_segregation as sfs


@pytest.fixture(scope="module")
def labels():
    df = sfs._synthetic_df(seed=1)
    return sfs.per_electrode_anova_labels(df, alpha=0.05, contrast_mode="proportion")


def test_all_four_interaction_flags_present(labels):
    for flag in ("CPC", "SPS", "CPS", "SPC"):
        assert flag in labels.columns, f"missing electrode-definition flag {flag!r}"
        assert set(labels[flag].unique()) <= {0, 1}


def test_cross_group_qvalues_present(labels):
    # cross interactions now get their own FDR'd q-values and signs
    for col in ("p_cps", "q_cps", "cps_sign", "p_spc", "q_spc", "spc_sign"):
        assert col in labels.columns, f"missing cross-interaction column {col!r}"


def test_constructs_recover_planted_effects(labels):
    # the generator plants congruency x inc_prop (CPC) and switchType x switch_prop (SPS)
    assert labels["CPC"].sum() > 0
    assert labels["SPS"].sum() > 0


def test_cross_groups_near_null(labels):
    # no true cross-effect in the generator -> far fewer CPS/SPC than CPC/SPS survivors
    n_construct = labels["CPC"].sum() + labels["SPS"].sum()
    n_cross = labels["CPS"].sum() + labels["SPC"].sum()
    assert n_cross < 0.5 * n_construct, (
        f"cross groups not near-null: CPC+SPS={n_construct}, CPS+SPC={n_cross} "
        "(Type III sum-coding orthogonalization may have failed)"
    )


def test_backward_compatible_aliases(labels):
    # S/F and the old effect columns are kept so the conjunction/anatomy stack works
    assert (labels["S"] == labels["CPC"]).all()
    assert (labels["F"] == labels["SPS"]).all()
    assert (labels["q_cong"] == labels["q_cpc"]).all()
    assert (labels["q_switch"] == labels["q_sps"]).all()


def test_backward_compatible_without_cross_controls():
    df = sfs._synthetic_df(seed=2)
    lab = sfs.per_electrode_anova_labels(df, contrast_mode="proportion",
                                         include_cross_controls=False)
    # constructs + aliases present; no cross columns emitted
    assert {"CPC", "SPS", "S", "F", "q_cong", "q_switch"} <= set(lab.columns)
    assert "CPS" not in lab.columns and "SPC" not in lab.columns


def test_still_drops_into_cmh_conjunction(labels):
    # output stays a valid `labels` input to the conjunction (needs subject, S, F)
    res = sfs.cmh_conjunction(labels)
    assert "mh_odds_ratio" in res
