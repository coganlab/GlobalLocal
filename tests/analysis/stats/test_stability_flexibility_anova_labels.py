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

import numpy as np
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


# ---------------------------------------------------------------------------
# direction-agnostic selection
# ---------------------------------------------------------------------------
# In behavior the congruency effect SHRINKS in high-incongruent-proportion blocks
# (and the switch cost shrinks in high-switch-proportion blocks), but for a neural
# population the direction of the block-proportion modulation is not known a
# priori. Selection must therefore key on the two-sided interaction only.
def test_selection_keeps_both_modulation_directions(labels):
    """Electrodes whose condition effect SHRINKS with the modulator are kept."""
    for flag, sign_col in (("CPC", "cpc_sign"), ("SPS", "sps_sign")):
        selected = labels[labels[flag] == 1]
        assert (selected[sign_col] > 0).any(), f"{flag}: no growing-direction electrodes"
        assert (selected[sign_col] < 0).any(), (
            f"{flag}: no shrinking-direction electrodes survived selection — the "
            "definition appears to assume a modulation direction"
        )


def test_flag_is_exactly_the_two_sided_q_threshold(labels):
    """The flag is the FDR'd two-sided q-value alone — no sign gate anywhere."""
    alpha = 0.05
    for flag, qcol in (("CPC", "q_cpc"), ("SPS", "q_sps"),
                       ("CPS", "q_cps"), ("SPC", "q_spc")):
        expected = (labels[qcol] < alpha).astype(int)
        assert (labels[flag] == expected).all(), (
            f"{flag} is not exactly (q < alpha); a directional filter may have "
            "been reintroduced"
        )


def test_can_flag_on_uncorrected_p_values_for_sensitivity_runs():
    """`fdr_correction='none'` keeps the schema but makes q columns raw p-values."""
    df = sfs._synthetic_df(seed=4)
    lab = sfs.per_electrode_anova_labels(
        df, alpha=0.05, contrast_mode="proportion", fdr_correction="none")
    for flag, pcol, qcol in (("CPC", "p_cpc", "q_cpc"),
                             ("SPS", "p_sps", "q_sps"),
                             ("CPS", "p_cps", "q_cps"),
                             ("SPC", "p_spc", "q_spc")):
        np.testing.assert_allclose(lab[qcol], lab[pcol], equal_nan=True)
        assert (lab[flag] == (lab[pcol] < 0.05).astype(int)).all()


def test_no_sign_gating_option_is_exposed():
    """No `require_sign`-style knob — a directional filter shouldn't be offerable."""
    import inspect

    params = inspect.signature(sfs.per_electrode_anova_labels).parameters
    assert "require_sign" not in params


# ---------------------------------------------------------------------------
# time-course (cluster-mode) input
# ---------------------------------------------------------------------------
# `assemble_long_df(..., effect_measure='cluster' | 'peak_t')` stores each trial's
# windowed time COURSE in `hg`, and the cross-decoding battery builds its table
# that way before calling this (window-mean) definition. The ANOVA fit already
# reduced those cells to window means; the `<g>_sign` direction did not, so it
# reached `_interaction_effect(..., 'cohens_d')` with a (n, T) array and raised
# "effect_measure='cohens_d' requires scalar (window-mean) hg".
@pytest.fixture(scope="module")
def cluster_labels():
    df = sfs._synthetic_df(effect_measure="cluster", n_time=12, seed=1)
    assert df["hg"].dtype == object, "fixture is not a time-course table"
    return sfs.per_electrode_anova_labels(df, alpha=0.05, contrast_mode="proportion")


def test_accepts_time_course_table(cluster_labels):
    """A cluster-mode table runs through instead of raising on the sign step."""
    assert len(cluster_labels) > 0
    for flag in ("CPC", "SPS", "CPS", "SPC"):
        assert set(cluster_labels[flag].unique()) <= {0, 1}
    assert cluster_labels["CPC"].sum() > 0 and cluster_labels["SPS"].sum() > 0


def test_time_course_signs_are_scalar_and_signed(cluster_labels):
    """Signs stay per-electrode scalars in {-1, +1} — not length-T vectors."""
    for col in ("cpc_sign", "sps_sign", "cps_sign", "spc_sign"):
        vals = cluster_labels[col].to_numpy()
        assert vals.ndim == 1, f"{col} is not scalar per electrode"
        assert set(np.unique(vals)) <= {-1.0, 0.0, 1.0}
    # both modulation directions still represented (as in the scalar path)
    assert (cluster_labels["cpc_sign"] > 0).any()
    assert (cluster_labels["cpc_sign"] < 0).any()


def test_time_course_matches_window_mean_table():
    """Reduction is exactly the window mean: pre-averaging the cells gives the
    same labels, so cluster-mode input is not a different analysis."""
    df = sfs._synthetic_df(effect_measure="cluster", n_time=12, seed=3)
    reduced = df.assign(hg=df["hg"].apply(np.nanmean))

    from_courses = sfs.per_electrode_anova_labels(df, contrast_mode="proportion")
    from_means = sfs.per_electrode_anova_labels(reduced, contrast_mode="proportion")

    for col in ("q_cpc", "q_sps", "cpc_sign", "sps_sign"):
        np.testing.assert_allclose(from_courses[col], from_means[col], rtol=1e-9)
    for flag in ("CPC", "SPS", "CPS", "SPC"):
        assert (from_courses[flag] == from_means[flag]).all()
