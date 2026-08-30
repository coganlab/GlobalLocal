"""Unit tests for the coupling-defined electrode-set path.

Covers the pure pieces of ``src/analysis/decoding/coupling_electrode_selection.py``
and the aggregation half of ``coupling_comparison.py``: filename parsing, the
bipolar→monopolar expansion, ROI resolution that does NOT trust column order,
the coupled/control split, and the matched draws. The decoding battery itself
needs real epochs and ``ieeg``, so it is smoke-tested on the cluster — the same
division as ``test_anova_electrode_selection.py``.
"""

import ast
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.decoding.coupling_electrode_selection import (
    COUPLED_SET_NAME,
    build_channel_records,
    build_coupling_electrode_sets,
    contact_pools,
    contacts_by_roi,
    coupled_electrode_set,
    coupling_run_slug,
    draw_matched_control_sets,
    find_coupling_csvs,
    format_pool_summary,
    is_control_set_name,
    load_coupling_table,
    normalize_pair_type,
    pair_type_rois,
    parse_coupling_filename,
    parse_draw_range,
    split_bipolar,
    summarize_pools,
)
from src.analysis.decoding.coupling_comparison import (
    draw_mean_traces,
    exceedance_pvalues,
    set_accuracy_samples,
)


# ---------------------------------------------------------------------------
# fixtures: a miniature two-subject dataset with a known answer
# ---------------------------------------------------------------------------
ROIS_DICT = {
    'lpfc': ['G_front_middle', 'S_front_inf'],
    'occ': ['G_cuneus', 'S_calcarine'],
}


@pytest.fixture
def roi_dict_json():
    """Two subjects; contact -> ROI membership is unambiguous by shank name."""
    return {
        'D0001': {'filtROI_dict': {
            'ctx_lh_G_front_middle': ['LF1', 'LF2', 'LF3', 'LF4'],
            'ctx_lh_G_cuneus': ['LO1', 'LO2', 'LO3'],
        }},
        'D0002': {'filtROI_dict': {
            'ctx_rh_S_front_inf': ['RF1', 'RF2', 'RF3'],
            'ctx_rh_S_calcarine': ['RO1', 'RO2'],
        }},
    }


@pytest.fixture
def roi_contacts(roi_dict_json):
    return contacts_by_roi(roi_dict_json, ROIS_DICT)


@pytest.fixture
def decoding_pool():
    """The pool the decoder would offer — every contact above."""
    return {
        'lpfc': {'D0001': ['LF1', 'LF2', 'LF3', 'LF4'],
                 'D0002': ['RF1', 'RF2', 'RF3']},
        'occ': {'D0001': ['LO1', 'LO2', 'LO3'],
                'D0002': ['RO1', 'RO2']},
    }


@pytest.fixture
def lpfc_occ_table():
    """A complete bipartite lpfc x occ design, with ch1/ch2 order DELIBERATELY
    inconsistent — real files put the occ channel in ch1 on a minority of rows,
    and membership must be resolved from the ROI dict, not the column."""
    rows = [
        # D0001: 2 lpfc bipolars x 2 occ bipolars
        ('D0001', 'LF1-LF2', 'LO1-LO2', True),
        ('D0001', 'LF1-LF2', 'LO2-LO3', False),
        ('D0001', 'LO1-LO2', 'LF3-LF4', False),   # reversed column order
        ('D0001', 'LF3-LF4', 'LO2-LO3', False),
        # D0002: 1 lpfc bipolar x 1 occ bipolar, significant
        ('D0002', 'RO1-RO2', 'RF1-RF2', True),    # reversed column order
        ('D0002', 'RF2-RF3', 'RO1-RO2', False),
    ]
    return pd.DataFrame(rows, columns=['subject', 'ch1', 'ch2',
                                       'is_significantly_high'])


# ---------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('text,expected', [
    ('lpfc-occ', 'lpfcocc'),
    ('lpfcocc', 'lpfcocc'),
    ('lpfc_occ', 'lpfcocc'),
    ('LPFC-OCC', 'lpfcocc'),
    ('occ-occ', 'occocc'),
])
def test_normalize_pair_type_collapses_spellings(text, expected):
    assert normalize_pair_type(text) == expected


def test_pair_type_rois():
    assert pair_type_rois('lpfc-occ') == ('lpfc', 'occ')
    assert pair_type_rois('occocc') == ('occ', 'occ')
    with pytest.raises(ValueError, match='unknown pair type'):
        pair_type_rois('lpfc-parietal')


@pytest.mark.parametrize('name,pair,condition,subject', [
    ('high_corr_lpfcocc_Stimulus_D0103.csv', 'lpfcocc', 'Stimulus', 'D0103'),
    ('high_corr_lpfc-occ_Stimulus_D0107A.csv', 'lpfcocc', 'Stimulus', 'D0107A'),
    ('high_corr_occocc_Stimulus_D0065.csv', 'occocc', 'Stimulus', 'D0065'),
    # a multi-token condition must survive the split
    ('high_corr_lpfc-occ_Stimulus_BigLetters_D0065.csv',
     'lpfcocc', 'Stimulus_BigLetters', 'D0065'),
])
def test_parse_coupling_filename(name, pair, condition, subject):
    parsed = parse_coupling_filename(name)
    assert parsed == {'pair_type': pair, 'condition': condition,
                      'subject': subject}


@pytest.mark.parametrize('name', [
    'envcorr_summary_D0103_lpfc_Stimulus.csv',
    'high_corr_lpfcocc.csv',
    'notes.txt',
])
def test_parse_coupling_filename_rejects_other_files(name):
    assert parse_coupling_filename(name) is None


def test_find_and_load_coupling_csvs(tmp_path, lpfc_occ_table):
    for sub, frame in lpfc_occ_table.groupby('subject'):
        # write with the extra columns a real file carries, to prove they are
        # ignored rather than choked on
        out = frame.copy()
        out['mean_corr_across_time'] = 0.01
        out['w0'] = 0.02
        out.to_csv(tmp_path / f'high_corr_lpfc-occ_Stimulus_{sub}.csv', index=False)
    (tmp_path / 'high_corr_occ-occ_Stimulus_D0001.csv').write_text(
        'subject,ch1,ch2,is_significantly_high\nD0001,LO1-LO2,LO2-LO3,True\n')

    found = find_coupling_csvs(str(tmp_path), 'lpfc-occ', 'Stimulus')
    assert [p[1]['subject'] for p in found] == ['D0001', 'D0002']

    table, used = load_coupling_table(str(tmp_path), 'lpfcocc', 'Stimulus')
    assert len(used) == 2
    assert list(table.columns) == ['subject', 'ch1', 'ch2', 'is_significantly_high']
    assert table['is_significantly_high'].dtype == bool
    assert table['is_significantly_high'].sum() == 2

    # subject filtering happens at both the filename and the row level
    table_one, used_one = load_coupling_table(
        str(tmp_path), 'lpfc-occ', 'Stimulus', subjects=['D0001'])
    assert len(used_one) == 1
    assert set(table_one['subject']) == {'D0001'}


def test_load_coupling_table_raises_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match='no high_corr CSVs'):
        load_coupling_table(str(tmp_path), 'lpfc-occ', 'Stimulus')


# ---------------------------------------------------------------------------
# channel names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name,expected', [
    ('LFAM9-LFAM10', ('LFAM9', 'LFAM10')),      # the coupling CSVs' spelling
    ('LFAM9-10', ('LFAM9', 'LFAM10')),          # save_bipolar_derivatives' spelling
    ('D0103_LFAM9-LFAM10', ('LFAM9', 'LFAM10')),  # optional subject prefix
    ('LFAM9', ('LFAM9',)),                      # already monopolar
])
def test_split_bipolar(name, expected):
    assert split_bipolar(name) == expected


def test_contacts_by_roi_matches_the_decoding_rule(roi_dict_json):
    contacts = contacts_by_roi(roi_dict_json, ROIS_DICT)
    assert contacts['lpfc']['D0001'] == {'LF1', 'LF2', 'LF3', 'LF4'}
    assert contacts['occ']['D0002'] == {'RO1', 'RO2'}
    assert contacts['lpfc']['D0002'] == {'RF1', 'RF2', 'RF3'}


def test_contacts_by_roi_can_restrict_subjects(roi_dict_json):
    contacts = contacts_by_roi(roi_dict_json, ROIS_DICT, subjects={'D0001'})
    assert set(contacts['lpfc']) == {'D0001'}


# ---------------------------------------------------------------------------
# pair rows -> per-ROI records
# ---------------------------------------------------------------------------
def test_records_ignore_column_order(lpfc_occ_table, roi_contacts):
    records = build_channel_records(lpfc_occ_table, 'lpfc-occ', roi_contacts)
    assert set(records) == {'lpfc', 'occ'}

    # 'LO1-LO2' appears in ch1 on one row and ch2 on another; either way it is
    # an occ channel and must never land in the lpfc records.
    assert 'LO1-LO2' in records['occ']['D0001']
    assert 'LO1-LO2' not in records['lpfc']['D0001']
    # 'RF1-RF2' appears only in ch2, and must still be found as lpfc.
    assert 'RF1-RF2' in records['lpfc']['D0002']


def test_records_track_coupling_and_degree(lpfc_occ_table, roi_contacts):
    records = build_channel_records(lpfc_occ_table, 'lpfc-occ', roi_contacts)
    lf12 = records['lpfc']['D0001']['LF1-LF2']
    assert lf12['coupled'] is True
    assert lf12['partners'] == {'LO1-LO2', 'LO2-LO3'}   # degree 2

    lf34 = records['lpfc']['D0001']['LF3-LF4']
    assert lf34['coupled'] is False
    assert len(lf34['partners']) == 2

    # occ side: LO1-LO2 is in the significant pair, LO2-LO3 is not
    assert records['occ']['D0001']['LO1-LO2']['coupled'] is True
    assert records['occ']['D0001']['LO2-LO3']['coupled'] is False


def test_records_require_the_complementary_roi(roi_contacts):
    """A within-LPFC row must not populate the lpfc pool of an lpfc-occ run."""
    table = pd.DataFrame(
        [('D0001', 'LF1-LF2', 'LF3-LF4', True)],
        columns=['subject', 'ch1', 'ch2', 'is_significantly_high'])
    records = build_channel_records(table, 'lpfc-occ', roi_contacts)
    assert records['lpfc'].get('D0001', {}) == {}
    assert records['occ'].get('D0001', {}) == {}


def test_homotypic_pair_type_uses_both_endpoints(roi_contacts):
    table = pd.DataFrame(
        [('D0001', 'LO1-LO2', 'LO2-LO3', True)],
        columns=['subject', 'ch1', 'ch2', 'is_significantly_high'])
    records = build_channel_records(table, 'occ-occ', roi_contacts)
    assert set(records) == {'occ'}
    assert set(records['occ']['D0001']) == {'LO1-LO2', 'LO2-LO3'}
    assert all(r['coupled'] for r in records['occ']['D0001'].values())


# ---------------------------------------------------------------------------
# records -> monopolar contact pools
# ---------------------------------------------------------------------------
def test_contact_pools_expand_both_poles(lpfc_occ_table, roi_contacts, decoding_pool):
    records = build_channel_records(lpfc_occ_table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, decoding_pool)
    # LF1-LF2 was the coupled bipolar -> BOTH poles are coupled contacts
    assert pools['lpfc']['D0001']['coupled'] == ['LF1', 'LF2']
    assert pools['lpfc']['D0001']['control'] == ['LF3', 'LF4']


def test_contact_pools_respect_the_decoding_pool(lpfc_occ_table, roi_contacts):
    """A pole the decoder does not offer cannot enter either set."""
    narrowed = {'lpfc': {'D0001': ['LF1'], 'D0002': []},
                'occ': {'D0001': ['LO1'], 'D0002': []}}
    records = build_channel_records(lpfc_occ_table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, narrowed)
    assert pools['lpfc']['D0001']['coupled'] == ['LF1']   # LF2 dropped
    assert pools['lpfc']['D0001']['control'] == []        # LF3/LF4 dropped
    assert pools['lpfc']['D0002']['coupled'] == []


def test_coupled_and_control_pools_are_disjoint(roi_contacts, decoding_pool):
    """Overlapping bipolars share a contact; coupled must win, not both."""
    table = pd.DataFrame(
        [('D0001', 'LF1-LF2', 'LO1-LO2', True),
         ('D0001', 'LF2-LF3', 'LO1-LO2', False)],
        columns=['subject', 'ch1', 'ch2', 'is_significantly_high'])
    records = build_channel_records(table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, decoding_pool)
    coupled = set(pools['lpfc']['D0001']['coupled'])
    control = set(pools['lpfc']['D0001']['control'])
    assert 'LF2' in coupled           # shared contact goes to coupled
    assert not (coupled & control)    # and to coupled ONLY
    assert control == {'LF3'}


def test_degree_is_the_max_over_contributing_bipolars(roi_contacts, decoding_pool):
    table = pd.DataFrame(
        [('D0001', 'LF1-LF2', 'LO1-LO2', False),
         ('D0001', 'LF1-LF2', 'LO2-LO3', False),
         ('D0001', 'LF2-LF3', 'LO1-LO2', False)],
        columns=['subject', 'ch1', 'ch2', 'is_significantly_high'])
    records = build_channel_records(table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, decoding_pool)
    degree = pools['lpfc']['D0001']['degree']
    assert degree['LF1'] == 2      # only from LF1-LF2 (2 partners)
    assert degree['LF2'] == 2      # max(LF1-LF2 = 2, LF2-LF3 = 1)
    assert degree['LF3'] == 1


# ---------------------------------------------------------------------------
# matched draws
# ---------------------------------------------------------------------------
@pytest.fixture
def wide_pools():
    """One ROI, two subjects, plenty of controls to draw from."""
    return {'lpfc': {
        'D0001': {'coupled': ['LF1', 'LF2'],
                  'control': [f'LC{i}' for i in range(10)],
                  'degree': {}},
        'D0002': {'coupled': ['RF1'],
                  'control': [f'RC{i}' for i in range(10)],
                  'degree': {}},
    }}


def test_draws_match_the_coupled_count(wide_pools):
    sets, shortfalls = draw_matched_control_sets(
        wide_pools, ['lpfc'], n_draws=5, seed=0)
    assert len(sets) == 5
    for name, elecset in sets.items():
        n = sum(len(v) for v in elecset['lpfc'].values())
        assert n == 3, f"{name} drew {n}, expected 3 to match the coupled set"
        assert shortfalls[name]['lpfc'] == 0


def test_draws_are_disjoint_from_the_coupled_set(wide_pools):
    sets, _ = draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=5, seed=1)
    coupled = {c for entry in wide_pools['lpfc'].values() for c in entry['coupled']}
    for elecset in sets.values():
        drawn = {c for chans in elecset['lpfc'].values() for c in chans}
        assert not (drawn & coupled)


def test_draws_differ_from_each_other_and_are_reproducible(wide_pools):
    first, _ = draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=8, seed=7)
    again, _ = draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=8, seed=7)
    assert first == again                       # same seed -> same draws
    flat = {name: tuple(sorted(c for chans in elecset['lpfc'].values()
                               for c in chans))
            for name, elecset in first.items()}
    assert len(set(flat.values())) > 1          # draws are not all identical


def test_global_match_can_move_electrodes_between_subjects(wide_pools):
    """The default global match constrains only the TOTAL n."""
    sets, _ = draw_matched_control_sets(
        wide_pools, ['lpfc'], n_draws=30, seed=3, match_level='global')
    per_subject = {tuple(sorted((sub, len(chans))
                                for sub, chans in elecset['lpfc'].items()))
                   for elecset in sets.values()}
    assert len(per_subject) > 1, "global matching should vary subject composition"


def test_within_subject_match_fixes_the_subject_composition(wide_pools):
    sets, _ = draw_matched_control_sets(
        wide_pools, ['lpfc'], n_draws=10, seed=3, match_level='within_subject')
    for elecset in sets.values():
        assert len(elecset['lpfc']['D0001']) == 2
        assert len(elecset['lpfc']['D0002']) == 1


def test_degree_matching_tracks_the_coupled_degrees():
    pools = {'lpfc': {'D0001': {
        'coupled': ['C1', 'C2'],
        'control': ['U_low1', 'U_low2', 'U_high1', 'U_high2'],
        'degree': {'C1': 10, 'C2': 10,
                   'U_low1': 1, 'U_low2': 1, 'U_high1': 9, 'U_high2': 11},
    }}}
    sets, _ = draw_matched_control_sets(
        pools, ['lpfc'], n_draws=4, seed=0, match_degree=True)
    for elecset in sets.values():
        drawn = set(elecset['lpfc']['D0001'])
        assert drawn == {'U_high1', 'U_high2'}, (
            "degree matching should pick the high-degree controls for "
            f"high-degree coupled electrodes, got {drawn}")


def test_short_control_pool_is_reported_not_silently_padded():
    pools = {'lpfc': {'D0001': {
        'coupled': ['C1', 'C2', 'C3'],
        'control': ['U1'],
        'degree': {},
    }}}
    sets, shortfalls = draw_matched_control_sets(pools, ['lpfc'], n_draws=2, seed=0)
    for name, elecset in sets.items():
        assert elecset['lpfc']['D0001'] == ['U1']
        assert shortfalls[name]['lpfc'] == 2


def test_draw_count_must_be_positive(wide_pools):
    with pytest.raises(ValueError, match='n_draws'):
        draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=0)


def test_bad_match_level_is_rejected(wide_pools):
    with pytest.raises(ValueError, match='match_level'):
        draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=1,
                                  match_level='per_shank')


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_build_coupling_electrode_sets_end_to_end(
        lpfc_occ_table, roi_contacts, decoding_pool):
    sets, report = build_coupling_electrode_sets(
        table=lpfc_occ_table,
        pair_type='lpfc-occ',
        roi_contacts=roi_contacts,
        decoding_pool=decoding_pool,
        n_draws=3,
        seed=0,
    )
    assert COUPLED_SET_NAME in sets
    assert sum(1 for name in sets if is_control_set_name(name)) == 3

    coupled = sets[COUPLED_SET_NAME]
    assert set(coupled) == {'lpfc', 'occ'}
    # D0001 LF1-LF2 and D0002 RF1-RF2 were the significant lpfc channels
    assert coupled['lpfc']['D0001'] == ['LF1', 'LF2']
    assert coupled['lpfc']['D0002'] == ['RF1', 'RF2']
    # occ side of the same pairs
    assert coupled['occ']['D0001'] == ['LO1', 'LO2']
    assert coupled['occ']['D0002'] == ['RO1', 'RO2']

    n_coupled = sum(len(v) for v in coupled['lpfc'].values())
    for name in sets:
        if is_control_set_name(name):
            n = sum(len(v) for v in sets[name]['lpfc'].values())
            assert n <= n_coupled

    assert report['pair_type'] == 'lpfcocc'
    assert report['n_significant_pairs'] == 2
    assert report['per_roi']['lpfc']['n_coupled_total'] == 4


def test_report_is_json_serializable(lpfc_occ_table, roi_contacts, decoding_pool):
    _, report = build_coupling_electrode_sets(
        table=lpfc_occ_table, pair_type='lpfc-occ', roi_contacts=roi_contacts,
        decoding_pool=decoding_pool, n_draws=2, seed=0)
    from src.analysis.decoding.coupling_electrode_selection import _json_default
    json.dumps(report, default=_json_default)   # must not raise


def test_format_pool_summary_flags_an_empty_roi(roi_contacts, decoding_pool):
    table = pd.DataFrame(
        [('D0001', 'LF1-LF2', 'LO1-LO2', False)],
        columns=['subject', 'ch1', 'ch2', 'is_significantly_high'])
    records = build_channel_records(table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, decoding_pool)
    report = summarize_pools(pools, ['lpfc', 'occ'])
    report['pair_type'] = 'lpfcocc'
    text = format_pool_summary(report)
    assert 'no coupled electrodes' in text


def test_coupled_electrode_set_reads_straight_off_the_pools(
        lpfc_occ_table, roi_contacts, decoding_pool):
    records = build_channel_records(lpfc_occ_table, 'lpfc-occ', roi_contacts)
    pools = contact_pools(records, decoding_pool)
    assert coupled_electrode_set(pools, ['lpfc'])['lpfc']['D0001'] == ['LF1', 'LF2']


def test_coupling_run_slug_is_filename_safe():
    slug = coupling_run_slug('lpfc-occ', 'Stimulus', n_draws=20, seed=0)
    assert slug == 'lpfcocc_Stimulus_20draws_seed0'
    assert coupling_run_slug('occ-occ', 'Stimulus', match_level='within_subject',
                             match_degree=True, n_draws=5, seed=1).endswith(
        'within_subject_degmatch')


# ---------------------------------------------------------------------------
# comparison-side aggregation
# ---------------------------------------------------------------------------
def test_set_accuracy_samples_reads_the_unit_specific_key():
    master = {'stats': {'task': {'lpfc': {
        'unit_of_analysis': 'repeat',
        'repeat_true_accs': np.zeros((6, 4)),
    }}}}
    accs, unit = set_accuracy_samples(master, 'task', 'lpfc')
    assert unit == 'repeat'
    assert accs.shape == (6, 4)


def test_set_accuracy_samples_missing_roi_returns_none():
    accs, _ = set_accuracy_samples({'stats': {}}, 'task', 'lpfc')
    assert accs is None


def test_draw_mean_traces_averages_within_each_draw():
    draws = [np.array([[0.6, 0.8], [0.4, 0.6]]),   # means 0.5, 0.7
             np.array([[0.5, 0.5], [0.5, 0.5]])]   # means 0.5, 0.5
    means = draw_mean_traces(draws)
    assert means.shape == (2, 2)
    np.testing.assert_allclose(means[0], [0.5, 0.7])
    np.testing.assert_allclose(means[1], [0.5, 0.5])


def test_draw_mean_traces_rejects_mismatched_widths():
    with pytest.raises(ValueError, match='time windows'):
        draw_mean_traces([np.zeros((2, 3)), np.zeros((2, 4))])


def test_exceedance_pvalues_are_one_sided_with_a_floor():
    series = np.array([0.9, 0.5])
    distribution = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    p = exceedance_pvalues(series, distribution)
    # window 0: no draw reaches 0.9 -> the floor, 1/(3+1)
    assert p[0] == pytest.approx(0.25)
    # window 1: all three draws beat 0.5 -> (1+3)/4
    assert p[1] == pytest.approx(1.0)


def test_exceedance_pvalue_floor_matches_the_draw_count():
    series = np.array([10.0])
    for n_draws in (9, 19, 99):
        distribution = np.zeros((n_draws, 1))
        assert exceedance_pvalues(series, distribution)[0] == pytest.approx(
            1.0 / (n_draws + 1))


def test_is_control_set_name():
    assert is_control_set_name('uncoupled_draw07')
    assert not is_control_set_name('coupled')


# ---------------------------------------------------------------------------
# fanning the draws out across jobs
# ---------------------------------------------------------------------------
# One decoding battery takes hours and a whole run is 1 + n_draws of them, which
# overruns the SLURM time limit in a single job. The draws are therefore split
# across jobs. That is a SCHEDULING change only, and these tests are what pins
# it: a slice must reproduce exactly the draws the whole run would have made,
# or the fan-out has quietly changed the null distribution.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('text,expected', [
    ('0-4', [0, 1, 2, 3, 4]),
    ('3', [3]),
    ('0-2,7', [0, 1, 2, 7]),
    (' 5 - 6 ', [5, 6]),
    ('4-4', [4]),
    ('2,1,0', [0, 1, 2]),        # sorted and de-duplicated
    ('1-3,2-4', [1, 2, 3, 4]),
    (None, None),
    ('', None),
    ('all', None),
])
def test_parse_draw_range(text, expected):
    assert parse_draw_range(text, 20) == expected


@pytest.mark.parametrize('text,match', [
    ('0-25', 'outside'),
    ('20', 'outside'),           # n_draws=20 means valid indices stop at 19
    # a leading minus reads as an empty 'lo', so it is rejected by the parser
    # rather than the bounds check — either way it never reaches a draw index
    ('-1', "not an index"),
    ('5-3', 'backwards'),
    ('a-b', 'not an index'),
])
def test_parse_draw_range_rejects_bad_input(text, match):
    with pytest.raises(ValueError, match=match):
        parse_draw_range(text, 20)


def test_a_slice_reproduces_the_whole_runs_draws_exactly(wide_pools):
    """THE property the fan-out rests on.

    Draw k is seeded `seed + k` and named from k and n_draws alone, so decoding
    draws 4-7 in their own job must give byte-identical electrode sets to draws
    4-7 of a single 20-draw job. If this ever fails, a fanned-out run and a
    single-job run are different experiments.
    """
    whole, whole_short = draw_matched_control_sets(
        wide_pools, ['lpfc'], n_draws=20, seed=0)

    for lo, hi in [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]:
        chunk, chunk_short = draw_matched_control_sets(
            wide_pools, ['lpfc'], n_draws=20, seed=0,
            draw_indices=range(lo, hi + 1))
        expected = {name: whole[name] for name in whole
                    if lo <= int(name[len('uncoupled_draw'):]) <= hi}
        assert chunk == expected, f"chunk {lo}-{hi} differs from the whole run"
        assert {k: chunk_short[k] for k in chunk_short} == {
            k: whole_short[k] for k in expected}


def test_slices_reassemble_into_the_whole_run(wide_pools):
    whole, _ = draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=20, seed=0)
    rebuilt = {}
    for lo in range(0, 20, 6):
        chunk, _ = draw_matched_control_sets(
            wide_pools, ['lpfc'], n_draws=20, seed=0,
            draw_indices=range(lo, min(lo + 6, 20)))
        rebuilt.update(chunk)
    assert rebuilt == whole


def test_slice_names_do_not_depend_on_the_slice(wide_pools):
    """Zero padding comes from n_draws, not from how many draws this job builds.

    Otherwise a 4-draw chunk would write `uncoupled_draw00..03` for draws 16-19
    and collide with the first chunk's directories.
    """
    chunk, _ = draw_matched_control_sets(
        wide_pools, ['lpfc'], n_draws=20, seed=0, draw_indices=[16, 17, 18, 19])
    assert sorted(chunk) == ['uncoupled_draw16', 'uncoupled_draw17',
                             'uncoupled_draw18', 'uncoupled_draw19']


def test_draw_indices_are_bounds_checked(wide_pools):
    with pytest.raises(ValueError, match='outside'):
        draw_matched_control_sets(wide_pools, ['lpfc'], n_draws=5,
                                  draw_indices=[0, 5])


def test_build_can_emit_a_slice_without_the_coupled_set(
        lpfc_occ_table, roi_contacts, decoding_pool):
    """Only one job of a fan-out decodes the coupled set; the rest skip it."""
    sets, report = build_coupling_electrode_sets(
        table=lpfc_occ_table, pair_type='lpfc-occ', roi_contacts=roi_contacts,
        decoding_pool=decoding_pool, n_draws=10, seed=0,
        draw_indices=[4, 5], include_coupled=False)
    assert sorted(sets) == ['uncoupled_draw04', 'uncoupled_draw05']
    assert COUPLED_SET_NAME not in sets
    assert report['is_complete_run'] is False
    assert report['draw_indices'] == [4, 5]
    # the pool summary still describes the WHOLE selection, so an underpowered
    # run is visible in every slice's log
    assert report['per_roi']['lpfc']['n_coupled_total'] == 4


def test_a_slice_does_not_change_the_sets_it_builds(
        lpfc_occ_table, roi_contacts, decoding_pool):
    kwargs = dict(table=lpfc_occ_table, pair_type='lpfc-occ',
                  roi_contacts=roi_contacts, decoding_pool=decoding_pool,
                  n_draws=10, seed=0)
    whole, _ = build_coupling_electrode_sets(**kwargs)
    slice_, _ = build_coupling_electrode_sets(
        **kwargs, draw_indices=[4, 5], include_coupled=False)
    for name in slice_:
        assert slice_[name] == whole[name]


@pytest.mark.parametrize('draw_indices,include_coupled,complete', [
    (None, True, True),
    (list(range(10)), True, True),    # a one-chunk fan-out states the range
    (None, False, False),             # draws only, coupled decoded elsewhere
    ([0, 1], True, False),
    ([0, 1], False, False),
])
def test_is_complete_run_tracks_coverage_not_spelling(
        lpfc_occ_table, roi_contacts, decoding_pool,
        draw_indices, include_coupled, complete):
    """decoding_dcc runs the comparison in-job iff this is True.

    A one-chunk fan-out passes an explicit '0-9' rather than None and DOES hold
    the whole run, so completeness has to be about coverage.
    """
    _, report = build_coupling_electrode_sets(
        table=lpfc_occ_table, pair_type='lpfc-occ', roi_contacts=roi_contacts,
        decoding_pool=decoding_pool, n_draws=10, seed=0,
        draw_indices=draw_indices, include_coupled=include_coupled)
    assert report['is_complete_run'] is complete


def test_default_call_is_unchanged_by_the_fan_out_parameters(
        lpfc_occ_table, roi_contacts, decoding_pool):
    """The single-job path must behave exactly as it did before."""
    sets, report = build_coupling_electrode_sets(
        table=lpfc_occ_table, pair_type='lpfc-occ', roi_contacts=roi_contacts,
        decoding_pool=decoding_pool, n_draws=3, seed=0)
    assert COUPLED_SET_NAME in sets
    assert sum(1 for n in sets if is_control_set_name(n)) == 3
    assert report['is_complete_run'] is True
    assert report['draw_indices'] is None


# ---------------------------------------------------------------------------
# wiring: the coupling path has to be REACHABLE from decoding_dcc.main
# ---------------------------------------------------------------------------
# The coupling block was originally indented one level too deep, inside
# `if args.anova_electrode_selection:`. The coupling launcher never sets
# ANOVA_ELECTRODE_SELECTION, so the branch was never entered: no `coupled` or
# `uncoupled_drawNN` sets were built, `run_coupling_comparison` never ran, and
# the job fell through to the ordinary single-set decode and exited 0 after
# writing a full battery of "in sig electrodes" figures. Nothing in the log said
# so. These tests parse the source rather than import it, because importing
# decoding_dcc pulls in `ieeg` and `mne`.
DECODING_DCC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'dcc_scripts', 'decoding', 'decoding_dcc.py'))


def _decoding_dcc_main():
    with open(DECODING_DCC) as handle:
        tree = ast.parse(handle.read())
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')


def _calls(node, func_name):
    return any(isinstance(sub, ast.Call)
               and getattr(sub.func, 'id', getattr(sub.func, 'attr', None)) == func_name
               for sub in ast.walk(node))


def _selection_block(main, builder):
    """The unique ``if`` that calls ``builder``, wherever it sits in ``main``."""
    blocks = [node for node in ast.walk(main)
              if isinstance(node, ast.If) and _calls(node, builder)]
    # An enclosing `if` also "contains" the call, so the innermost one — the
    # smallest — is the guard that actually gates the builder.
    assert blocks, f"nothing in main() calls {builder}"
    return min(blocks, key=lambda node: len(list(ast.walk(node))))


def _index_in(body, node):
    """Position of ``node`` in ``body``, or ``None`` if it is nested deeper."""
    for i, stmt in enumerate(body):
        if stmt is node:
            return i
    return None


def test_coupling_selection_is_not_nested_inside_anova_selection():
    main = _decoding_dcc_main()
    block = _selection_block(main, 'build_coupling_selected_electrode_sets')
    assert _index_in(main.body, block) is not None, (
        "the block calling build_coupling_selected_electrode_sets is not a "
        "direct statement of main(). Nested inside the anova_electrode_selection "
        "branch (as it originally was) it is unreachable from the coupling "
        "launcher, which silently decodes all sig electrodes instead of the "
        "coupled and uncoupled_drawNN sets")


def test_anova_and_coupling_selection_are_siblings():
    main = _decoding_dcc_main()
    anova = _index_in(main.body,
                      _selection_block(main, 'build_anova_selected_electrode_sets'))
    coupling = _index_in(main.body,
                         _selection_block(main, 'build_coupling_selected_electrode_sets'))
    assert anova is not None and coupling is not None, (
        "both selection blocks must live in main()'s body")
    # coupling reads `electrode_sets` to refuse a double definition, so it has to
    # come after the anova block that may have populated it.
    assert coupling > anova


def test_a_requested_electrode_set_option_cannot_fall_through_silently():
    """The single-set path must refuse to run when a set option was requested.

    Without this, any future failure to build the sets repeats the original
    symptom: a job that runs for hours, exits 0, and writes an ordinary
    all-electrode decode that looks like a successful coupling run.
    """
    main = _decoding_dcc_main()
    fallthrough = next(node for node in main.body
                       if isinstance(node, ast.If)
                       and isinstance(node.test, ast.UnaryOp)
                       and isinstance(node.test.op, ast.Not)
                       and getattr(node.test.operand, 'id', None) == 'electrode_sets')
    raised = [sub for sub in ast.walk(fallthrough) if isinstance(sub, ast.Raise)]
    assert raised, ("the `if not electrode_sets:` fall-through must raise when "
                    "anova_electrode_selection or coupling_electrode_selection "
                    "is on, not quietly decode every electrode")
    flags = {sub.value for sub in ast.walk(fallthrough)
             if isinstance(sub, ast.Constant) and isinstance(sub.value, str)}
    assert {'anova_electrode_selection', 'coupling_electrode_selection'} <= flags


# ---------------------------------------------------------------------------
# aggregating a fanned-out run
# ---------------------------------------------------------------------------
def _write_fake_set(base, set_name, comparison, roi, n_windows=4, seed=0):
    """A minimal MASTER_RESULTS pickle, enough for the comparison to load."""
    import pickle
    rng = np.random.RandomState(seed)
    out = os.path.join(base, f'elecset_{set_name}')
    os.makedirs(out, exist_ok=True)
    master = {
        'stats': {comparison: {roi: {
            'unit_of_analysis': 'repeat',
            'repeat_true_accs': rng.uniform(0.4, 0.6, size=(5, n_windows)),
        }}},
        'metadata': {'time_window_centers': list(range(n_windows))},
    }
    with open(os.path.join(out, f'{set_name}_MASTER_RESULTS_x.pkl'), 'wb') as fh:
        pickle.dump(master, fh)


def test_expect_n_draws_catches_a_chunk_that_never_landed(tmp_path):
    """A dead chunk must be an error, not a quietly smaller null distribution.

    Without the check the comparison happily runs against whatever draws are on
    disk, so a 20-draw run that lost a chunk reports a p-value floor of 1/17
    while the figure and the filenames still say it was a 20-draw run.
    """
    from src.analysis.decoding.coupling_comparison import run_coupling_comparison
    base = str(tmp_path)
    _write_fake_set(base, COUPLED_SET_NAME, 'task', 'lpfc', seed=99)
    for k in range(16):                      # 16 of the 20 draws arrived
        _write_fake_set(base, f'uncoupled_draw{k:02d}', 'task', 'lpfc', seed=k)

    with pytest.raises(ValueError, match='expected 20 control draws'):
        run_coupling_comparison(
            base_save_dir=base,
            args=SimpleNamespace(window_size=64, step_size=16, sampling_rate=256),
            condition_comparisons=['task'], rois=['lpfc'],
            make_plots=False, expect_n_draws=20)


def test_expect_n_draws_passes_when_every_chunk_landed(tmp_path):
    # The count check itself needs nothing, but getting past it runs the real
    # comparison, which needs ieeg — present on the cluster, not in a bare env.
    pytest.importorskip('ieeg', reason="the comparison itself needs ieeg")
    from src.analysis.decoding.coupling_comparison import run_coupling_comparison
    base = str(tmp_path)
    _write_fake_set(base, COUPLED_SET_NAME, 'task', 'lpfc', seed=99)
    for k in range(20):
        _write_fake_set(base, f'uncoupled_draw{k:02d}', 'task', 'lpfc', seed=k)

    results, summary = run_coupling_comparison(
        base_save_dir=base,
        args=SimpleNamespace(window_size=64, step_size=16, sampling_rate=256),
        condition_comparisons=['task'], rois=['lpfc'],
        make_plots=False, expect_n_draws=20)
    assert summary['task']['lpfc']['n_draws'] == 20
    # the floor the run was designed for, not one inflated by a missing chunk
    assert summary['task']['lpfc']['exceedance_p_floor'] == pytest.approx(1 / 21, abs=1e-4)
