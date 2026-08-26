from dcc_scripts.vis import condition_plot_specs as specs


def test_lpfc_plot_registry_filters_sets_and_preserves_draw_order(monkeypatch):
    monkeypatch.setenv("PLOT_SETS", "both,all_lpfc,congruency_only")

    resolved = specs.resolve_plot_set("lpfc_power_trace_sets")

    assert list(resolved["conditions"]) == [
        "all_lpfc", "congruency_only", "both"]
    assert resolved["conditions"]["all_lpfc"]["color"] == (0.0, 0.7, 0.0)
    assert resolved["conditions"]["all_lpfc"]["all_roi"] is True
    assert resolved["conditions"]["congruency_only"]["color"] == (1.0, 0.0, 0.0)
    assert resolved["conditions"]["both"]["color"] == (0.0, 0.0, 0.0)
    assert resolved["mutually_exclusive"] is False
    assert list(resolved["rois_dict"]) == ["lpfc"]


def test_a1_label_sets_are_a_distinct_selectable_method(monkeypatch):
    monkeypatch.setenv("PLOT_SETS", "both_labels")

    resolved = specs.resolve_plot_set("lpfc_power_trace_sets")

    config = resolved["conditions"]["both_labels"]
    assert config["anova_label_effect"] == "both"
    assert config["anova_label_correction"] == "flags"
    assert config["color"] == (0.0, 0.0, 0.0)


def test_a1_main_effect_set_includes_the_shared_population(monkeypatch):
    monkeypatch.setenv("PLOT_SETS", "congruency_labels")

    config = specs.resolve_plot_set(
        "lpfc_power_trace_sets")["conditions"]["congruency_labels"]

    assert config["anova_label_effect"] == "congruency"
    assert config["color"] == (1.0, 0.0, 0.0)
