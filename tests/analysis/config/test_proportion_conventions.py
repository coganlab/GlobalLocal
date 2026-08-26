"""The number in a BIDS event is the INCONGRUENT proportion, not the congruent one.

``Stimulus/c25.0`` is a congruent trial in a block that is 25% incongruent (a
mostly-congruent block); ``Stimulus/c75.0`` is a congruent trial in a block that
is 75% incongruent. Likewise ``s25.0``/``r25.0`` mark a block that is 25% switch.

The codebase used to name the first number ``congruencyProportion`` (75% for
c25), and the conversion to incongruent proportion left several values behind on
the old convention. Rather than spot-check them, these tests re-derive every
factor value from the BIDS event strings the condition actually selects, so a
value can never silently disagree with the trials it labels.
"""

import re

import pytest

from src.analysis.config import experiment_conditions as ec
from src.analysis.config.plotting_parameters import plotting_parameters


_CONG_EVENT = re.compile(r'/([ci])(25|75)\.0')
_SWITCH_EVENT = re.compile(r'/([rs])(25|75)\.0')

# Numbers embedded in condition keys: Stimulus_c25, Stimulus_c25r75,
# Stimulus_taskG_in_25switchBlock, Stimulus_s_in_75incongruentBlock.
_KEY_PATTERNS = [
    (re.compile(r'_([ci])(25|75)'), 'inc', 2),
    (re.compile(r'([rs])(25|75)$'), 'switch', 2),
    (re.compile(r'_in_(25|75)incongruentBlock'), 'inc', 1),
    (re.compile(r'_in_(25|75)switchBlock'), 'switch', 1),
]


def _condition_objects():
    for name in dir(ec):
        obj = getattr(ec, name)
        if isinstance(obj, dict) and name.endswith('conditions'):
            yield name, obj


def _events(cdict):
    evs = cdict.get('BIDS_events')
    if isinstance(evs, str):
        return [evs]
    return list(evs or [])


def _proportions_in_events(evs):
    """Return (inc_proportions, switch_proportions) as sets of '25'/'75'."""
    inc = {m[1] for e in evs for m in _CONG_EVENT.findall(e)}
    switch = {m[1] for e in evs for m in _SWITCH_EVENT.findall(e)}
    return inc, switch


def _all_conditions():
    for obj_name, obj in _condition_objects():
        for cname, cdict in obj.items():
            if isinstance(cdict, dict):
                yield obj_name, cname, cdict


_CASES = [pytest.param(o, c, d, id=f"{o}::{c}") for o, c, d in _all_conditions()]


@pytest.mark.parametrize("obj_name,cname,cdict", _CASES)
def test_stated_proportions_match_the_bids_events(obj_name, cname, cdict):
    """`incongruentProportion`/`switchProportion` must equal the event number."""
    inc, switch = _proportions_in_events(_events(cdict))

    for key, found in (('incongruentProportion', inc), ('switchProportion', switch)):
        stated = cdict.get(key)
        if stated is None or not found:
            continue
        assert found == {str(stated).rstrip('%')}, (
            f"{obj_name}[{cname}] declares {key}={stated} but its BIDS events "
            f"select proportion(s) {sorted(found)}. The number in the event is "
            f"the incongruent/switch proportion; do not flip it."
        )


@pytest.mark.parametrize("obj_name,cname,cdict", _CASES)
def test_metadata_queries_match_the_bids_events(obj_name, cname, cdict):
    """A cell's metadata_query must select the same block as its BIDS events."""
    query = cdict.get('metadata_query')
    evs = _events(cdict)
    if not query or not evs:
        return
    inc, switch = _proportions_in_events(evs)

    for column, found in (('incongruent_proportion', inc),
                          ('switch_proportion', switch)):
        m = re.search(rf'{column}\s*==\s*(\d+)', query)
        if not m or not found:
            continue
        assert found == {m.group(1)}, (
            f"{obj_name}[{cname}] queries {column} == {m.group(1)} but its BIDS "
            f"events select {sorted(found)}."
        )


@pytest.mark.parametrize("obj_name,cname,cdict", _CASES)
def test_numbers_in_condition_names_match_the_bids_events(obj_name, cname, cdict):
    """Stimulus_c25 must select c25.0 events, not c75.0 ones."""
    inc, switch = _proportions_in_events(_events(cdict))
    by_kind = {'inc': inc, 'switch': switch}

    for pattern, kind, group in _KEY_PATTERNS:
        m = pattern.search(cname)
        found = by_kind[kind]
        if not m or not found:
            continue
        assert found == {m.group(group)}, (
            f"{obj_name}[{cname}] is named for proportion {m.group(group)} but "
            f"selects {sorted(found)}."
        )


def test_no_condition_declares_a_congruency_proportion():
    """The old `congruencyProportion` key means the complement; it must not return."""
    offenders = [
        f"{obj_name}[{cname}]"
        for obj_name, cname, cdict in _all_conditions()
        if 'congruencyProportion' in cdict or 'congruency_proportion' in cdict
    ]
    assert not offenders, (
        "These conditions use the old congruent-proportion key, whose number is "
        f"the complement of the BIDS event's: {offenders}"
    )


def test_plotting_parameters_do_not_declare_a_congruency_proportion():
    offenders = [
        key for key, params in plotting_parameters.items()
        if isinstance(params, dict)
        and ('congruencyProportion' in params or 'congruency_proportion' in params)
    ]
    assert not offenders, offenders
