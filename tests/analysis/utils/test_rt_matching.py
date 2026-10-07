"""RT matching (`src/analysis/utils/rt_matching.py`).

The synthetic trials plant the GlobalLocal pattern: incongruent and switch trials
are slower. Matching must remove that difference per subject, keep only trials
it could use, be reproducible, and hand back an epochs structure of the same
shape. The count-matched random control must keep the same counts while leaving
the RT difference in place.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.utils import rt_matching as rm

GROUPS = ['congruency', 'task_sequence']


def _trials(subjects=('S1', 'S2', 'S3'), n_per_cell=120, i_cost=150., s_cost=200.,
            seed=0, cell_sizes=None):
    """One row per trial: 4 congruency x switch cells per subject, with the
    incongruent and switch costs added to a lognormal RT."""
    rng = np.random.default_rng(seed)
    rows, tid = [], 0
    for sub in subjects:
        base = rng.uniform(900, 1200)
        for cong in ('c', 'i'):
            for sw in ('r', 's'):
                n = (cell_sizes or {}).get((cong, sw), n_per_cell)
                rt = (base * rng.lognormal(0, 0.25, n)
                      + i_cost * (cong == 'i') + s_cost * (sw == 's'))
                for r in rt:
                    tid += 1
                    rows.append(dict(subject=sub, congruency=cong, task_sequence=sw,
                                     reaction_time=r, trial_count=float(tid)))
    return pd.DataFrame(rows)


def _mean_diffs(df, keep, factor):
    t = rm.rt_contrast_table(df, keep, [factor])
    return t['diff_before'].to_numpy(), t['diff_after'].to_numpy()


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('balance', rm.BALANCE_MODES)
def test_matching_removes_the_planted_rt_differences(balance):
    df = _trials()
    keep = rm.rt_match(df, GROUPS, balance=balance, seed=1)
    for factor, cost in (('congruency', 150.), ('task_sequence', 200.)):
        before, after = _mean_diffs(df, keep, factor)
        assert np.all(before > 0.6 * cost)
        assert np.all(np.abs(after) < 0.2 * cost)
    assert 0.2 < keep.mean() < 0.9


def test_equal_balance_keeps_identical_counts_per_cell_and_bin():
    df = _trials()
    keep = rm.rt_match(df, GROUPS, n_bins=6, balance='equal', seed=0)
    for sub, s in df[keep.to_numpy()].groupby('subject'):
        full = df[df.subject == sub]
        bins = rm._rt_bins(full.reaction_time.to_numpy(), 6)
        full_bins = pd.Series(bins, index=full.index)
        counts = (s.assign(bin=full_bins.loc[s.index])
                  .groupby(GROUPS + ['bin']).size().unstack(fill_value=0))
        assert len(counts) == 4
        assert (counts.nunique(axis=0) == 1).all()      # every cell, same count per bin


def test_proportional_balance_keeps_group_size_ratios():
    sizes = {('c', 'r'): 300, ('c', 's'): 100, ('i', 'r'): 100, ('i', 's'): 100}
    df = _trials(n_per_cell=100, cell_sizes=sizes, i_cost=0., s_cost=0.)
    keep = rm.rt_match(df, GROUPS, balance='proportional', seed=0)
    kept = df[keep.to_numpy()].groupby(['subject'] + GROUPS).size().unstack([1, 2])
    ratio = kept[('c', 'r')] / kept[('i', 's')]
    assert np.all((ratio > 2.3) & (ratio < 3.7))
    # equal balance, by contrast, forces the cells to the same size
    keep_eq = rm.rt_match(df, GROUPS, balance='equal', seed=0)
    eq = df[keep_eq.to_numpy()].groupby(['subject'] + GROUPS).size().unstack([1, 2])
    assert (eq.nunique(axis=1) == 1).all()


def test_keep_counts_never_exceed_what_is_available():
    rng = np.random.default_rng(3)
    for _ in range(200):
        counts = rng.integers(0, 15, size=(rng.integers(2, 5), rng.integers(1, 12)))
        for balance in rm.BALANCE_MODES:
            take = rm._keep_counts(counts, balance)
            assert take.shape == counts.shape
            assert np.all(take <= counts) and np.all(take >= 0)


def test_unusable_rows_are_never_kept():
    df = _trials(subjects=('S1',))
    df.loc[df.index[:10], 'reaction_time'] = np.nan
    df.loc[df.index[10:15], 'reaction_time'] = np.inf
    df.loc[df.index[15:20], 'congruency'] = None
    keep = rm.rt_match(df, GROUPS, seed=0)
    assert not keep.iloc[:20].any()
    assert keep.iloc[20:].any()


def test_a_stratum_with_one_group_is_dropped():
    df = _trials(subjects=('S1', 'S2'))
    df = df[~((df.subject == 'S2') & (df.congruency == 'i'))]
    df = df[~((df.subject == 'S2') & (df.task_sequence == 's'))].reset_index(drop=True)
    keep = rm.rt_match(df, GROUPS, seed=0)
    assert not keep[df.subject == 'S2'].any()
    assert keep[df.subject == 'S1'].any()


def test_matching_is_reproducible_and_per_stratum():
    df = _trials()
    a = rm.rt_match(df, GROUPS, seed=5)
    b = rm.rt_match(df, GROUPS, seed=5)
    c = rm.rt_match(df, GROUPS, seed=6)
    assert a.equals(b) and not a.equals(c)
    # adding a subject leaves the others' draws untouched
    more = pd.concat([df, _trials(subjects=('S4',), seed=9)], ignore_index=True)
    d = rm.rt_match(more, GROUPS, seed=5)
    assert d.iloc[:len(df)].reset_index(drop=True).equals(a.reset_index(drop=True))


def test_within_adds_strata():
    df = _trials()
    df['block'] = np.where(df.index % 2, 'A', 'B')
    keep = rm.rt_match(df, GROUPS, within=['subject', 'block'], seed=0)
    counts = df[keep.to_numpy()].groupby(['subject', 'block'] + GROUPS).size()
    assert (counts.groupby(['subject', 'block']).nunique() == 1).all()


def test_bad_arguments_raise():
    df = _trials(subjects=('S1',))
    with pytest.raises(ValueError, match='balance'):
        rm.rt_match(df, GROUPS, balance='nope')
    with pytest.raises(KeyError, match='no column'):
        rm.rt_match(df, ['switchType'])
    with pytest.raises(ValueError, match='n_bins'):
        rm.rt_match(df, GROUPS, n_bins=0)


def test_factor_names_map_to_metadata_columns():
    assert rm.metadata_columns(('congruency', 'switchType')) == ('congruency', 'task_sequence')
    assert rm.metadata_columns('switch_type') == ('task_sequence',)
    assert rm.metadata_columns(('incongruentProportion', 'prev_congruency')) == \
        ('incongruent_proportion', 'prev_congruency')


# ---------------------------------------------------------------------------
# the count-matched random control
# ---------------------------------------------------------------------------
def test_random_control_matches_counts_but_not_rt():
    df = _trials()
    keep = rm.rt_match(df, GROUPS, seed=0)
    ctrl = rm.count_matched_random(df, keep, GROUPS, seed=0)
    by = ['subject'] + GROUPS
    assert df[keep.to_numpy()].groupby(by).size().equals(df[ctrl.to_numpy()].groupby(by).size())
    assert not keep.equals(ctrl)
    _, after_rt = _mean_diffs(df, keep, 'task_sequence')
    before, after_random = _mean_diffs(df, ctrl, 'task_sequence')
    assert np.all(after_random > 0.6 * before)
    assert np.all(np.abs(after_rt) < np.abs(after_random))


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def test_reports_describe_before_and_after():
    df = _trials()
    keep = rm.rt_match(df, GROUPS, seed=0)
    bal = rm.rt_balance_table(df, keep, GROUPS)
    assert len(bal) == 3 * 4
    assert (bal.n_after <= bal.n_before).all()
    contrasts = rm.rt_contrast_table(df, keep, GROUPS)
    assert set(contrasts.contrast) == {'i - c', 's - r'}
    summary = rm.summarize_rt_contrasts(contrasts)
    row = summary.set_index('factor').loc['task_sequence']
    assert row.mean_before > 150 and abs(row.mean_after) < 40
    assert row.n_strata == 3


# ---------------------------------------------------------------------------
# the adapter on real mne Epochs
# ---------------------------------------------------------------------------
def _structure(df, n_ch=3, n_times=5):
    mne = pytest.importorskip('mne')
    info = mne.create_info([f'ch{i}' for i in range(n_ch)], 100., 'seeg')
    rng = np.random.default_rng(0)
    out = {}
    for sub, s in df.groupby('subject'):
        out[sub] = {}
        for (cong, sw), cell in s.groupby(GROUPS):
            md = cell.drop(columns='subject').reset_index(drop=True)
            md['subject'] = f'D{sub}'          # the metadata's own spelling differs
            epochs = mne.EpochsArray(rng.normal(size=(len(cell), n_ch, n_times)), info,
                                     metadata=md, verbose=False)
            out[sub][f'Stimulus_{cong}{sw}'] = {'HG_ev1_rescaled': epochs}
    return out


@pytest.mark.parametrize('mode', rm.MATCH_MODES)
def test_adapter_slices_every_epochs_object(mode):
    df = _trials(subjects=('S1', 'S2'), n_per_cell=60)
    structure = _structure(df)
    matched, report = rm.rt_match_subjects_mne_objects(
        structure, groups=('congruency', 'switchType'), seed=0, mode=mode, verbose=False)
    keep = rm.rt_match(df, GROUPS, seed=0)
    if mode == 'random':
        keep = rm.count_matched_random(df, keep, GROUPS, seed=0)
    assert set(matched) == set(structure)
    for sub, conds in matched.items():
        assert set(conds) == set(structure[sub])
        for cond, obj in conds.items():
            ep = obj['HG_ev1_rescaled']
            ids = set(ep.metadata['trial_count'])
            want = set(df.loc[keep.to_numpy() & (df.subject == sub).to_numpy()
                              & (('Stimulus_' + df.congruency + df.task_sequence) == cond)
                              .to_numpy(), 'trial_count'])
            assert ids == want
            assert ep.get_data().shape[0] == len(want)
    assert {'balance', 'contrasts', 'summary'} == set(report)
    after = report['summary'].set_index('factor').loc['task_sequence', 'mean_after']
    assert (abs(after) < 40) if mode == 'rt' else (after > 100)


def test_adapter_rebuilds_the_loaders_evoked_averages():
    """`create_subjects_mne_objects_dict` stores <key>_avg / <key>_std_err next to
    each Epochs and the power-trace plots read them; after matching they must
    average the kept trials, not the full set."""
    df = _trials(subjects=('S1',), n_per_cell=60)
    structure = _structure(df)
    for obj in structure['S1'].values():
        ep = obj['HG_ev1_rescaled']
        ep._data[0, 0, 0] = np.nan                    # a NaN the mean must skip
        obj['HG_ev1_rescaled_avg'] = ep.average()
        obj['HG_ev1_rescaled_std_err'] = ep.average()
    matched, _ = rm.rt_match_subjects_mne_objects(structure, seed=0, verbose=False)
    for cond, obj in matched['S1'].items():
        data = obj['HG_ev1_rescaled'].get_data()
        assert data.shape[0] < len(structure['S1'][cond]['HG_ev1_rescaled'])
        np.testing.assert_allclose(obj['HG_ev1_rescaled_avg'].data, np.nanmean(data, axis=0))
        assert obj['HG_ev1_rescaled_avg'].nave == data.shape[0]
        n = np.maximum(np.sum(~np.isnan(data), axis=0), 1)
        np.testing.assert_allclose(obj['HG_ev1_rescaled_std_err'].data,
                                   np.nan_to_num(np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)))
        # the input structure is left as it was
        assert structure['S1'][cond]['HG_ev1_rescaled_avg'].nave == 60


def test_adapter_counts_a_trial_shared_by_two_conditions_once():
    """Some condition sets are separate epoch sets over the same trials; a trial
    must enter the matching once, and be kept or dropped everywhere."""
    df = _trials(subjects=('S1',), n_per_cell=60)
    structure = _structure(df)
    dup = structure['S1']['Stimulus_ir']['HG_ev1_rescaled'].copy()
    structure['S1']['Stimulus_ir_copy'] = {'HG_ev1_rescaled': dup}
    table = rm.trials_table(structure)
    assert len(table) == len(df)
    matched, _ = rm.rt_match_subjects_mne_objects(structure, seed=0, verbose=False)
    a = set(matched['S1']['Stimulus_ir']['HG_ev1_rescaled'].metadata['trial_count'])
    b = set(matched['S1']['Stimulus_ir_copy']['HG_ev1_rescaled'].metadata['trial_count'])
    assert a == b


def test_adapter_needs_trial_ids():
    mne = pytest.importorskip('mne')
    info = mne.create_info(['ch0'], 100., 'seeg')
    ep = mne.EpochsArray(np.zeros((3, 1, 4)), info, verbose=False)
    with pytest.raises(ValueError, match='trial_count'):
        rm.rt_match_subjects_mne_objects({'S1': {'c': {'HG': ep}}}, verbose=False)
    with pytest.raises(ValueError, match='mode'):
        rm.rt_match_subjects_mne_objects({'S1': {'c': {'HG': ep}}}, mode='x',
                                         verbose=False)
