import pandas as pd
import pytest

from src.analysis.utils.anova_label_selection import (
    anova_label_run_slug, load_anova_labels, load_anova_label_electrodes,
    selected_pairs)


def _csv(tmp_path):
    path = tmp_path / "anova_labels.csv"
    pd.DataFrame({
        "subject": ["D1", "D1", "D2"],
        "electrode": ["D1-A1", "A2", "B1"],
        "roi": ["lpfc", "lpfc", "acc"],
        "S": [1, 0, 1], "F": [0, 1, 1],
        "p_cong": [.01, .03, .2], "q_cong": [.03, .06, .2],
        "p_switch": [.2, .01, .02], "q_switch": [.2, .03, .06],
    }).to_csv(path, index=False)
    return path


def test_anova_label_run_slug_records_source_and_selection_parameters(tmp_path):
    path = tmp_path / "window_0.0to1.5_condition_none" / "anova_labels.csv"
    slug = anova_label_run_slug(path, "both", "none", 0.05, "lpfc")

    assert slug.startswith("window_0.0to1.5_condition_none__effect-both")
    assert "__correction-none__alpha-0.05__roi-lpfc__" in slug
    assert len(slug.rsplit("__", 1)[1]) == 8


def test_anova_label_run_slug_changes_with_path_and_parameters(tmp_path):
    first = tmp_path / "first" / "same" / "anova_labels.csv"
    second = tmp_path / "second" / "same" / "anova_labels.csv"

    assert anova_label_run_slug(first) != anova_label_run_slug(second)
    assert anova_label_run_slug(first, effect="lwpc") != anova_label_run_slug(
        first, effect="lwps")


def test_saved_flags_and_prefixed_electrode_names(tmp_path):
    got = load_anova_label_electrodes(_csv(tmp_path), "lwpc", roi="lpfc")
    assert got == {"lpfc": {"D1": ["A1"]}}


def test_raw_and_fdr_thresholds_are_explicit(tmp_path):
    path = _csv(tmp_path)
    raw = selected_pairs(load_anova_label_electrodes(
        path, "lwpc", correction="none", roi="lpfc"))
    fdr = selected_pairs(load_anova_label_electrodes(
        path, "lwpc", correction="fdr_bh", roi="lpfc"))
    assert raw == {("D1", "A1"), ("D1", "A2")}
    assert fdr == {("D1", "A1")}


def test_both_selects_stability_flexibility_intersection(tmp_path):
    path = _csv(tmp_path)
    saved = selected_pairs(load_anova_label_electrodes(
        path, "both", correction="flags"))
    raw = selected_pairs(load_anova_label_electrodes(
        path, "both", correction="none"))
    assert saved == {("D2", "B1")}
    assert raw == {("D1", "A2")}


def test_both_requires_stability_and_flexibility_columns(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({
        "subject": ["D1"], "electrode": ["A1"], "S": [1],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"missing \['F'\]"):
        load_anova_label_electrodes(path, "both")


def test_roi_request_requires_roi_column(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"subject": ["D1"], "electrode": ["A1"], "S": [1]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="no 'roi' column"):
        load_anova_label_electrodes(path, roi="lpfc")


def test_cross_decode_flags_are_recomputed_together(tmp_path):
    labels = load_anova_labels(_csv(tmp_path), correction="none")
    # Raw p selects D1-A2 for both LWPC and LWPS even though the saved S flag is
    # zero. This guards the cross-decoder against mixing raw and saved-FDR flags.
    row = labels.loc[labels.electrode == "A2"].iloc[0]
    assert row.S == 1
    assert row.F == 1


def test_result_directory_is_accepted_as_the_path(tmp_path):
    _csv(tmp_path)
    got = load_anova_label_electrodes(tmp_path, "lwpc", roi="lpfc")
    assert got == {"lpfc": {"D1": ["A1"]}}
