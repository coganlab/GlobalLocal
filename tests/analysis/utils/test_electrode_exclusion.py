import pytest

from src.analysis.utils.electrode_exclusion import (
    filter_out_excluded_electrodes,
    parse_exclusions,
)


ELECTRODES = {
    'lpfc': {
        'D0121': ['LFMI8', 'LFMI9', 'LFA10'],
        'D0116': ['LFMI8', 'LOF13'],
    },
    'occ': {
        'D0121': ['LFMI8', 'LOC1'],
    },
}


def test_parse_exclusions_splits_global_and_per_subject():
    global_names, per_subject = parse_exclusions(
        ['LFMI8', ' D0121:LFA10 ', '', 'D0116:LOF13'])

    assert global_names == {'LFMI8'}
    assert per_subject == {('D0121', 'LFA10'), ('D0116', 'LOF13')}


def test_bare_name_drops_the_channel_in_every_subject_and_roi():
    filtered, n_dropped = filter_out_excluded_electrodes(ELECTRODES, ['LFMI8'])

    assert n_dropped == 3
    assert filtered['lpfc']['D0121'] == ['LFMI9', 'LFA10']
    assert filtered['lpfc']['D0116'] == ['LOF13']
    assert filtered['occ']['D0121'] == ['LOC1']


def test_qualified_name_drops_the_channel_in_one_subject_only():
    filtered, n_dropped = filter_out_excluded_electrodes(
        ELECTRODES, ['D0121:LFMI8'])

    # Dropped once per ROI the subject contributes to: lpfc and occ.
    assert n_dropped == 2
    assert filtered['lpfc']['D0121'] == ['LFMI9', 'LFA10']
    assert filtered['lpfc']['D0116'] == ['LFMI8', 'LOF13']
    # 'occ' is a different ROI but the same subject, so it drops there too.
    assert filtered['occ']['D0121'] == ['LOC1']


def test_input_is_not_mutated():
    filter_out_excluded_electrodes(ELECTRODES, ['LFMI8'])

    assert ELECTRODES['lpfc']['D0121'] == ['LFMI8', 'LFMI9', 'LFA10']


def test_empty_exclusions_are_a_no_op():
    filtered, n_dropped = filter_out_excluded_electrodes(ELECTRODES, [])

    assert n_dropped == 0
    assert filtered == ELECTRODES


def test_excluding_everything_raises_rather_than_running_on_nothing():
    with pytest.raises(ValueError, match="removed every electrode"):
        filter_out_excluded_electrodes(
            {'lpfc': {'D0121': ['LFMI8']}}, ['LFMI8'])
