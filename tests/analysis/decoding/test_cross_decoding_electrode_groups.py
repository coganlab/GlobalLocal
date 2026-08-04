"""Electrode-group derivation for the A4 cross-decoding job.

Covers the two things the job has to get right before any decoding happens:

1. **Channel keys.** The decoded ROI LabeledArray labels channels
   `f'{subject}-{electrode}'`. `per_electrode_anova_labels` already emits that
   prefixed form, but the power-traces route
   (`power_traces_conjunction.electrode_labels`) keeps `subject` and a BARE
   `electrode` in separate columns. If the two are not normalised to one
   convention, `_restrict_to_electrodes` matches nothing and the run silently
   reports "group has 0 electrodes in ROI" instead of a key mismatch.

2. **The unselected reference group.** both / S_only / F_only were all chosen
   for carrying an interaction, so none of them is a baseline for "does this ROI
   cross-decode at all". The reference group is every channel in the decoded
   array, defined by nothing the decode is about.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dcc_scripts.decoding import stability_flexibility_cross_decoding_dcc as xd


def _anova_style_labels():
    """`per_electrode_anova_labels` convention: `electrode` already prefixed."""
    return pd.DataFrame(dict(
        subject=['D1', 'D1', 'D2'],
        electrode=['D1-A1', 'D1-A2', 'D2-B1'],
        S=[1, 1, 0], F=[1, 0, 1],
        CPC=[1, 1, 0], SPS=[1, 0, 1], CPS=[0, 0, 1], SPC=[0, 1, 0]))


def _power_traces_style_labels():
    """`ptc.electrode_labels` convention: bare `electrode`, same electrodes."""
    return pd.DataFrame(dict(
        subject=['D1', 'D1', 'D2'],
        electrode=['A1', 'A2', 'B1'],
        S=[1, 1, 0], F=[1, 0, 1],
        CPC=[1, 1, 0], SPS=[1, 0, 1], CPS=[0, 0, 1], SPC=[0, 1, 0]))


# ---------------------------------------------------------------------------
# channel keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('labels', [_anova_style_labels(), _power_traces_style_labels()])
def test_channel_keys_match_the_roi_array_convention(labels):
    assert list(xd._channel_keys(labels)) == ['D1-A1', 'D1-A2', 'D2-B1']


def test_both_definition_routes_give_identical_groups():
    """The route must not change WHICH channels a group names."""
    assert (xd._electrode_groups(_anova_style_labels())
            == xd._electrode_groups(_power_traces_style_labels()))
    assert (xd._interaction_groups(_anova_style_labels())
            == xd._interaction_groups(_power_traces_style_labels()))


def test_groups_are_the_disjoint_s_f_partition():
    groups = xd._electrode_groups(_anova_style_labels())
    assert groups == {'both': ['D1-A1'], 'S_only': ['D1-A2'], 'F_only': ['D2-B1']}


def test_interaction_groups_keyed_by_definition_flag():
    inter = xd._interaction_groups(_anova_style_labels())
    assert set(inter) == {'CPC', 'SPS', 'CPS', 'SPC'}
    assert inter['CPC'] == ['D1-A1', 'D1-A2']


def test_power_traces_keys_actually_slice_the_array():
    """End-to-end on the restriction itself: bare-electrode labels must not
    silently select zero channels."""
    channel_names = ['D1-A1', 'D1-A2', 'D2-B1', 'D2-B2']
    arrays = {'roi': {'cond': np.zeros((5, 4, 3))}}
    groups = xd._electrode_groups(_power_traces_style_labels())
    _, n_kept = xd._restrict_to_electrodes(arrays, 'roi', channel_names, groups['both'])
    assert n_kept == 1


# ---------------------------------------------------------------------------
# the unselected reference group
# ---------------------------------------------------------------------------
def test_reference_group_is_every_channel_in_the_array():
    channel_names = ['D1-A1', 'D1-A2', 'D2-B1', 'D2-B2']
    groups = xd._add_reference_group(
        xd._electrode_groups(_anova_style_labels()), channel_names,
        SimpleNamespace(reference_group='all'))
    assert groups['all'] == channel_names
    # the selected groups are untouched
    assert groups['both'] == ['D1-A1']


def test_reference_group_can_be_opted_out():
    groups = xd._add_reference_group(
        {'both': ['D1-A1']}, ['D1-A1', 'D1-A2'],
        SimpleNamespace(reference_group=''))
    assert groups == {'both': ['D1-A1']}


def test_reference_group_not_duplicated_when_a_group_already_covers_everything():
    """The synthetic path's 'both' IS every channel — don't decode it twice."""
    channel_names = ['ch0', 'ch1']
    groups = xd._add_reference_group(
        {'both': list(channel_names)}, channel_names,
        SimpleNamespace(reference_group='all'))
    assert 'all' not in groups


def test_reference_group_name_collision_is_an_error():
    with pytest.raises(ValueError, match='collides'):
        xd._add_reference_group({'both': ['a']}, ['a', 'b'],
                                SimpleNamespace(reference_group='both'))


# ---------------------------------------------------------------------------
# definition-route selection
# ---------------------------------------------------------------------------
def test_unknown_definition_route_is_rejected():
    with pytest.raises(ValueError, match='electrode_definition must be'):
        xd._resolve_labels(SimpleNamespace(electrode_definition='bogus', alpha=0.05))


def test_power_traces_route_requires_run_directories():
    with pytest.raises(ValueError, match='power_traces_runs'):
        xd._resolve_labels(SimpleNamespace(electrode_definition='power_traces',
                                           alpha=0.05, power_traces_runs=None))
