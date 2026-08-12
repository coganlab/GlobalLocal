import mne
import numpy as np
import pytest

from src.analysis.power.evoked_builders import (
    create_subtracted_evokeds_dict,
    subtract_evoked_conditions,
)


def _evoked(value):
    info = mne.create_info(["electrode"], sfreq=100, ch_types="seeg")
    return mne.EvokedArray(np.full((1, 4), value), info, tmin=0, nave=5)


def test_subtract_evoked_conditions_combines_available_data():
    evokeds = {"first": {"roi": _evoked(3)}, "second": {"roi": _evoked(1)}}

    result = subtract_evoked_conditions(evokeds, "first", "second", "roi")

    np.testing.assert_array_equal(result.data, np.full((1, 4), 2))


@pytest.mark.parametrize(
    "evokeds",
    [
        {"first": {"roi": None}, "second": {"roi": _evoked(1)}},
        {"first": {"roi": _evoked(1)}, "second": {"roi": None}},
        {"first": {}, "second": {"roi": _evoked(1)}},
    ],
)
def test_subtract_evoked_conditions_preserves_missing_roi_data(evokeds):
    assert subtract_evoked_conditions(evokeds, "first", "second", "roi") is None


def test_create_subtracted_evokeds_dict_handles_each_roi_independently():
    evokeds = {
        "first": {"populated": _evoked(3), "empty": None},
        "second": {"populated": _evoked(1), "empty": _evoked(1)},
    }

    result = create_subtracted_evokeds_dict(
        evokeds, [("first", "second")], ["populated", "empty"]
    )

    np.testing.assert_array_equal(
        result["first-second"]["populated"].data, np.full((1, 4), 2)
    )
    assert result["first-second"]["empty"] is None
