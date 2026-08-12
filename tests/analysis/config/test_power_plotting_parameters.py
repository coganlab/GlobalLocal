from src.analysis.config.plotting_parameters import plotting_parameters


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
