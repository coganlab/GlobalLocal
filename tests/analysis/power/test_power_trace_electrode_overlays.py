from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from src.analysis.power.plots import (  # noqa: E402
    plot_power_trace_for_roi,
    rank_electrode_deviations,
)


def _evoked():
    return SimpleNamespace(
        times=np.array([0.0, 0.1, 0.2]),
        data=np.array([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [8.0, 8.0, 8.0]]),
        ch_names=["E1", "E2", "E_OUTLIER"],
    )


def test_rank_electrode_deviations_puts_largest_departure_first():
    ranking = rank_electrode_deviations(_evoked())

    assert ranking[0][0] == "E_OUTLIER"
    assert ranking[0][1] > ranking[1][1]


def test_power_plot_adds_gray_traces_labels_and_report(tmp_path):
    evoked = _evoked()
    fig = plot_power_trace_for_roi(
        {"condition": {"roi": evoked}},
        "roi",
        ["condition"],
        "example",
        {"condition": {"color": "red", "condition_parameter": "Mean"}},
        save_dir=tmp_path,
        show_std=False,
        plot_style={"n_outlier_labels": 1},
        save_name_suffix="all",
    )

    # Three gray channel traces, then the red across-channel mean.
    assert len(fig.axes[0].lines) >= 6  # includes horizontal/vertical zero lines
    assert any(text.get_text() == "E_OUTLIER" for text in fig.axes[0].texts)
    report = tmp_path / "roi_example_all_no_error_shading_electrode_deviations.txt"
    contents = report.read_text()
    assert "diagnostic ranking, not a statistical outlier test" in contents
    assert contents.index("E_OUTLIER") < contents.index("E1")
