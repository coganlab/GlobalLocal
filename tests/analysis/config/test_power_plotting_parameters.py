import re

import pytest

from src.analysis.config import experiment_conditions
from src.analysis.config.plotting_parameters import plotting_parameters


# Keys like Stimulus_c25 / Stimulus_i75 / Stimulus_s25 / Stimulus_r75, whose
# numeric suffix is a proportion that must agree with the BIDS event naming.
_PROPORTION_KEYS = sorted(
    k for k in plotting_parameters
    if re.fullmatch(r"Stimulus_[cisr](25|75)", k)
)


@pytest.mark.parametrize("key", _PROPORTION_KEYS)
def test_proportion_condition_labels_match_their_key(key):
    """The legend label must name the same cell as the key.

    These four LWPC labels once carried the old congruency-proportion
    convention (Stimulus_c25 labeled "c75") while the rest of the codebase had
    moved to incongruent proportion, so every legend built from
    `condition_parameter` was inverted. Pin it so they cannot drift apart
    again.
    """
    assert plotting_parameters[key]["condition_parameter"] == \
        key[len("Stimulus_"):]


@pytest.mark.parametrize("key", _PROPORTION_KEYS)
def test_proportion_condition_labels_match_the_bids_events(key):
    """...and that the key itself matches the BIDS events it selects."""
    conditions = {
        **experiment_conditions.stimulus_lwpc_conditions,
        **experiment_conditions.stimulus_lwps_conditions,
    }
    if key not in conditions:
        pytest.skip(f"{key} is not an LWPC/LWPS condition")
    suffix = key[len("Stimulus_"):]            # e.g. "c25"
    expected_event = f"Stimulus/{suffix[0]}{suffix[1:]}.0"   # "Stimulus/c25.0"
    assert conditions[key]["BIDS_events"] == [expected_event]


def test_congruency_by_switch_proportion_conditions_have_unique_labels():
    condition_names = [
        "Stimulus_c_in_25switchBlock",
        "Stimulus_c_in_75switchBlock",
        "Stimulus_i_in_25switchBlock",
        "Stimulus_i_in_75switchBlock",
    ]

    labels = [plotting_parameters[name]["condition_parameter"]
              for name in condition_names]

    assert len(set(labels)) == 4
    assert set(labels) == {
        "congruent, 25% switch",
        "congruent, 75% switch",
        "incongruent, 25% switch",
        "incongruent, 75% switch",
    }


def test_switch_type_by_incongruent_proportion_conditions_have_unique_labels():
    condition_names = [
        "Stimulus_s_in_25incongruentBlock",
        "Stimulus_r_in_25incongruentBlock",
        "Stimulus_s_in_75incongruentBlock",
        "Stimulus_r_in_75incongruentBlock",
    ]

    labels = [plotting_parameters[name]["condition_parameter"]
              for name in condition_names]

    assert len(set(labels)) == 4
    assert set(labels) == {
        "switch, 25% incongruent",
        "repeat, 25% incongruent",
        "switch, 75% incongruent",
        "repeat, 75% incongruent",
    }
