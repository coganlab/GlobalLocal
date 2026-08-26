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


def _evoked_with_prestimulus_outlier():
    """One channel deviates only before t=0, another only after.

    Ranking over the whole epoch is dominated by the post-stimulus channel;
    ranking over the pre-stimulus window must surface the other one.
    """
    times = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    return SimpleNamespace(
        times=times,
        data=np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],       # E_FLAT
            [5.0, 5.0, 0.0, 0.0, 0.0],       # E_PRE   -- baseline-period only
            [0.0, 0.0, 0.0, 20.0, 20.0],     # E_POST  -- response-period only
        ]),
        ch_names=["E_FLAT", "E_PRE", "E_POST"],
    )


def test_rank_electrode_deviations_puts_largest_departure_first():
    ranking = rank_electrode_deviations(_evoked())

    assert ranking[0][0] == "E_OUTLIER"
    assert ranking[0][1] > ranking[1][1]


def test_rank_electrode_deviations_respects_time_window():
    evoked = _evoked_with_prestimulus_outlier()

    assert rank_electrode_deviations(evoked)[0][0] == "E_POST"
    # (None, 0.0) == "epoch start through stimulus onset"
    pre = rank_electrode_deviations(evoked, time_window=(None, 0.0))
    assert pre[0][0] == "E_PRE"


def test_rank_electrode_deviations_falls_back_when_window_is_empty():
    evoked = _evoked_with_prestimulus_outlier()

    # A window entirely outside the epoch scores on the full epoch instead of
    # raising on an empty slice.
    ranking = rank_electrode_deviations(evoked, time_window=(50.0, 60.0))
    assert ranking[0][0] == "E_POST"


def test_power_plot_draws_only_the_top_n_electrode_traces(tmp_path):
    evoked = _evoked_with_prestimulus_outlier()
    fig = plot_power_trace_for_roi(
        {"condition": {"roi": evoked}},
        "roi",
        ["condition"],
        "example",
        {"condition": {"color": "red", "condition_parameter": "Mean"}},
        save_dir=tmp_path,
        show_std=False,
        plot_style={
            "n_electrode_traces": 1,
            "electrode_trace_window": (None, 0.0),
            "n_outlier_labels": 1,
        },
        save_name_suffix="all",
    )

    ax = fig.axes[0]
    # 1 electrode trace + the across-electrode mean + the two zero lines.
    assert len(ax.lines) == 4
    # Ranked on the pre-stimulus window, so the pre-stimulus deviant is drawn.
    assert any(text.get_text() == "E_PRE" for text in ax.texts)
    assert not any(text.get_text() == "E_POST" for text in ax.texts)

    report = tmp_path / "roi_example_all_no_error_shading_electrode_deviations.txt"
    contents = report.read_text()
    assert "Scoring window: epoch start to 0s" in contents
    assert contents.index("E_PRE") < contents.index("E_POST")


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


def test_ci_shading_splits_the_excluded_mass_between_both_tails():
    """ci=0.95 must be the 2.5th-97.5th percentile, not the 5th-95th."""
    rng = np.random.default_rng(0)
    times = np.linspace(-1.0, 1.5, 8)
    data = rng.normal(0, 1, (400, times.size))
    evoked = SimpleNamespace(times=times, data=data,
                             ch_names=[f"E{i}" for i in range(data.shape[0])])

    fig = plot_power_trace_for_roi(
        {"condition": {"roi": evoked}}, "roi", ["condition"], "example",
        {"condition": {"color": "red", "condition_parameter": "Mean"}},
        show_std=False, show_ci=True, ci=0.95,
        plot_style={"show_electrode_traces": False},
        save_name_suffix="all",
    )

    band = fig.axes[0].collections[0].get_paths()[0].vertices[:, 1]
    expected = np.nanpercentile(data, [2.5, 97.5], axis=0)
    assert np.isclose(band.min(), expected[0].min(), atol=1e-9)
    assert np.isclose(band.max(), expected[1].max(), atol=1e-9)
