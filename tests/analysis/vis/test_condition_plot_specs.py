from dcc_scripts.vis import condition_plot_specs as specs


def test_lpfc_plot_registry_filters_sets_and_preserves_draw_order(monkeypatch):
    monkeypatch.setenv("PLOT_SETS", "both,all_lpfc,congruency_only")

    resolved = specs.resolve_plot_set("lpfc_power_trace_sets")

    assert list(resolved["conditions"]) == [
        "all_lpfc", "congruency_only", "both"]
    assert resolved["conditions"]["all_lpfc"]["color"] == (0.0, 0.7, 0.0)
    assert resolved["conditions"]["congruency_only"]["color"] == (1.0, 0.0, 0.0)
    assert resolved["conditions"]["both"]["color"] == (0.0, 0.0, 0.0)
    assert resolved["mutually_exclusive"] is False

