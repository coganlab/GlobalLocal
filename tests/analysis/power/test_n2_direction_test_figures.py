"""N2 direction-test figures (docs/n2_direction_tests.md).

The three direction tests used to leave exactly one figure behind -- the
subtraction plot, carrying the interaction bar -- so the two simple effects had
no figure of their own. These cover the per-test figures that replace that: that
each test gets its own file, that the files are named so the test and the
contrast can be read off them, and that a difference-wave bar lands somewhere
visible instead of at the raw-trace height.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.analysis.power.plots import (  # noqa: E402
    DIRECTION_TEST_SUBDIR,
    direction_test_save_name,
    plot_direction_test_traces,
    plot_power_trace_for_roi,
)

TIMES = np.linspace(-0.5, 1.0, 16)


def _evoked(level):
    return SimpleNamespace(
        times=TIMES,
        data=np.full((3, TIMES.size), float(level)),
        ch_names=["E1", "E2", "E3"],
    )


def _panels():
    """The shape power_traces_dcc.py hands over, on an LWPC-style naming."""
    evks = {
        "Stimulus_i25": {"lpfc": _evoked(0.6)},
        "Stimulus_c25": {"lpfc": _evoked(0.2)},
        "Stimulus_i75": {"lpfc": _evoked(0.5)},
        "Stimulus_c75": {"lpfc": _evoked(0.4)},
    }
    diffs = {
        "Stimulus_i25-Stimulus_c25": {"lpfc": _evoked(0.4)},
        "Stimulus_i75-Stimulus_c75": {"lpfc": _evoked(0.1)},
    }
    return {
        "simple_low": {
            "stem": "Stimulus_i25-Stimulus_c25", "title": "low",
            "evks": evks, "traces": ["Stimulus_i25", "Stimulus_c25"],
        },
        "simple_high": {
            "stem": "Stimulus_i75-Stimulus_c75", "title": "high",
            "evks": evks, "traces": ["Stimulus_i75", "Stimulus_c75"],
        },
        "interaction": {
            "stem": "low_minus_high", "title": "interaction",
            "evks": diffs,
            "traces": ["Stimulus_i25-Stimulus_c25",
                       "Stimulus_i75-Stimulus_c75"],
        },
    }


def _plotting_parameters():
    names = ["Stimulus_i25", "Stimulus_c25", "Stimulus_i75", "Stimulus_c75",
             "Stimulus_i25-Stimulus_c25", "Stimulus_i75-Stimulus_c75"]
    return {n: {"color": "black", "line_style": "-",
                "condition_parameter": n} for n in names}


def _style(**overrides):
    style = {"show_electrode_traces": False, "label_outliers": False}
    style.update(overrides)
    return style


def test_each_direction_test_gets_its_own_figure(tmp_path):
    masks = {
        "simple_low": {"lpfc": np.zeros(TIMES.size, bool)},
        "simple_high": {"lpfc": np.zeros(TIMES.size, bool)},
        "interaction": {"lpfc": np.zeros(TIMES.size, bool)},
    }

    save_names = plot_direction_test_traces(
        _panels(), masks, ["lpfc"], "stimulus_lwpc_conditions", 24,
        _plotting_parameters(), save_dir=str(tmp_path),
        plot_style=_style(), save_name_suffix="sig_elecs",
    )

    roi_dir = tmp_path / DIRECTION_TEST_SUBDIR / "lpfc"
    pngs = sorted(p.name for p in roi_dir.glob("*.png"))
    assert len(pngs) == 3, pngs
    # Three figures in one folder that differ only by the test -- the test name
    # and the contrast both have to be on the filename.
    assert set(save_names) == {"simple_low", "simple_high", "interaction"}
    for test_key, stem in save_names.items():
        assert any(name.startswith(f"lpfc_{stem}_sig_elecs") for name in pngs)
        assert test_key in stem
        assert (roi_dir / f"lpfc_{stem}_sig_elecs_sem_shading.pdf").exists()


def test_save_name_carries_label_test_contrast_and_n():
    name = direction_test_save_name(
        "stimulus_lwps_conditions", "simple_low", "Stimulus_s25-Stimulus_r25", 24)

    assert name == ("stimulus_lwps_conditions_n2_direction_simple_low_"
                    "Stimulus_s25-Stimulus_r25_24_subjects")


def test_a_missing_roi_mask_draws_the_traces_without_a_bar(tmp_path):
    # An ROI skipped for missing evoked data has no mask; that must not take the
    # other ROIs' figures down with it.
    masks = {"simple_low": {}, "simple_high": {}, "interaction": {}}

    plot_direction_test_traces(
        _panels(), masks, ["lpfc"], "stimulus_lwpc_conditions", 24,
        _plotting_parameters(), save_dir=str(tmp_path),
        plot_style=_style(), save_name_suffix="sig_elecs",
    )

    assert len(list((tmp_path / DIRECTION_TEST_SUBDIR / "lpfc").glob("*.png"))) == 3


def _bar_heights(fig):
    """y of every horizontal significance bar drawn on the figure's axis."""
    ax = fig.axes[0]
    return [coll.get_segments()[0][0][1] for coll in ax.collections
            if coll.get_segments()]


def test_explicit_bar_height_is_still_honoured(tmp_path):
    mask = np.zeros(TIMES.size, bool)
    mask[4:9] = True

    fig = plot_power_trace_for_roi(
        {"Stimulus_i25": {"lpfc": _evoked(0.6)}}, "lpfc", ["Stimulus_i25"],
        "run", _plotting_parameters(), significant_clusters=mask,
        save_dir=None, show_std=False,
        plot_style=_style(sig_cluster_height=0.8, ylim=(-0.35, 0.9)),
    )

    assert _bar_heights(fig) == pytest.approx([0.8])


def test_auto_bar_height_lands_inside_a_difference_wave_axis(tmp_path):
    # The whole reason for `None`: on a difference wave the raw-trace height
    # (0.8, against data around 0.03) is off the top of the axis.
    mask = np.zeros(TIMES.size, bool)
    mask[4:9] = True
    diff = SimpleNamespace(times=TIMES,
                           data=np.full((3, TIMES.size), 0.03),
                           ch_names=["E1", "E2", "E3"])

    fig = plot_power_trace_for_roi(
        {"d": {"lpfc": diff}}, "lpfc", ["d"], "run",
        {"d": {"color": "black", "line_style": "-", "condition_parameter": "d"}},
        significant_clusters=mask, save_dir=None, show_std=False,
        plot_style=_style(sig_cluster_height=None, ylim=None, yticks=None),
    )

    (bar_y,) = _bar_heights(fig)
    low, high = fig.axes[0].get_ylim()
    assert low < bar_y < high
    # Above the trace, not through it.
    assert bar_y > 0.03
