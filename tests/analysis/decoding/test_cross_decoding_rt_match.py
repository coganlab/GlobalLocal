"""RT matching inside the A4 / transfer job (`_apply_rt_match`, `_build_roi_arrays`).

The matching itself is tested in tests/analysis/utils/test_rt_matching.py; these
pin the job's side: matching runs on the decode trials (after the selection
split), writes its report, lands in summary.txt, refuses synthetic data, and
never lets a subject drop out of the pseudopopulation.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip('mne')

from src.analysis.config import experiment_conditions as ec
from src.analysis.decoding import cross_decoding as cd
from dcc_scripts.decoding import stability_flexibility_cross_decoding_dcc as xd


def _structure(subjects=('D1', 'D2'), n_per_cell=60, n_ch=3, n_times=8, seed=0,
               cells=(('c', 'r'), ('c', 's'), ('i', 'r'), ('i', 's'))):
    """{subject: {Stimulus_<cong><sw>: {'HG_ev1_rescaled': EpochsArray}}}, with
    incongruent and switch trials slower, as in the task."""
    rng = np.random.default_rng(seed)
    info = mne.create_info([f'E{i}' for i in range(n_ch)], 256., 'seeg')
    out, tid = {}, 0
    for sub in subjects:
        out[sub] = {}
        for cong, sw in cells:
            rt = (1000 * rng.lognormal(0, 0.25, n_per_cell)
                  + 150 * (cong == 'i') + 200 * (sw == 's'))
            ids = np.arange(tid, tid + n_per_cell, dtype=float)
            tid += n_per_cell
            md = pd.DataFrame(dict(congruency=cong, task_sequence=sw, reaction_time=rt,
                                   trial_count=ids, accuracy=1.0))
            ep = mne.EpochsArray(rng.normal(size=(n_per_cell, n_ch, n_times)), info,
                                 metadata=md, verbose=False)
            out[sub][f'Stimulus_{cong}{sw}'] = {'HG_ev1_rescaled': ep}
    return out


def _args(tmp_path, **kw):
    base = dict(rt_match='rt', rt_match_bins=5, rt_match_balance='equal',
                rt_match_within=(), rt_match_groups=None, rt_match_seed=0, seed=0,
                data_source='real', save_dir=str(tmp_path), rt_match_log=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _n(structure, sub, cond):
    return len(structure[sub][cond]['HG_ev1_rescaled'])


def test_no_matching_leaves_the_trials_alone(tmp_path):
    s = _structure()
    assert xd._apply_rt_match(_args(tmp_path, rt_match='none'), s,
                              cd.CROSS_DECODE_FIELDS) is s
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', ['rt', 'random'])
def test_matching_subsamples_writes_its_report_and_logs(tmp_path, mode):
    s = _structure()
    args = _args(tmp_path, rt_match=mode)
    matched = xd._apply_rt_match(args, s, cd.CROSS_DECODE_FIELDS)
    for sub in s:
        counts = {c: _n(matched, sub, c) for c in s[sub]}
        assert len(set(counts.values())) == 1                 # equal balance
        assert all(0 < n < 60 for n in counts.values())
    for name in ('balance', 'contrasts', 'summary'):
        assert (tmp_path / f'rt_match_congruency_task_sequence_{name}.csv').exists()
    summary = pd.read_csv(tmp_path / 'rt_match_congruency_task_sequence_summary.csv')
    after = summary.set_index('factor').loc['task_sequence', 'mean_after']
    assert abs(after) < 60 if mode == 'rt' else after > 100
    assert len(args.rt_match_log) == 1
    assert ('RT-matched' if mode == 'rt' else 'RANDOM control') in args.rt_match_log[0]


def test_rt_and_random_keep_the_same_counts(tmp_path):
    s = _structure()
    rt = xd._apply_rt_match(_args(tmp_path / 'a'), s, cd.CROSS_DECODE_FIELDS)
    rnd = xd._apply_rt_match(_args(tmp_path / 'b', rt_match='random'), s,
                             cd.CROSS_DECODE_FIELDS)
    for sub in s:
        for cond in s[sub]:
            assert _n(rt, sub, cond) == _n(rnd, sub, cond)


def test_a_subject_left_with_nothing_is_an_error(tmp_path):
    """Every subject's channels sit in every pseudo-trial, so one that loses all
    its trials would turn every row incomplete; refuse instead."""
    s = _structure(subjects=('D1',))
    s.update(_structure(subjects=('D2',), cells=(('c', 'r'),), seed=1))
    with pytest.raises(ValueError, match="no trials for subjects \\['D2'\\]"):
        xd._apply_rt_match(_args(tmp_path), s, cd.CROSS_DECODE_FIELDS)


def test_synthetic_data_refuse_rt_matching(tmp_path):
    args = _args(tmp_path, data_source='synthetic')
    with pytest.raises(ValueError, match='needs real epochs'):
        xd._check_rt_match_args(args)
    with pytest.raises(ValueError, match='rt_match must be one of'):
        xd._check_rt_match_args(_args(tmp_path, rt_match='median'))


def test_build_roi_arrays_matches_the_decode_trials(tmp_path, monkeypatch):
    """The real `_build_roi_arrays`, with only the cluster loaders stubbed: the
    pseudopopulation is built from the RT-matched trials, after the split."""
    from src.analysis.utils import general_utils as gu
    from src.analysis.utils import labeled_array_utils as lau

    s = _structure()
    seen = {}

    class _Stop(Exception):
        pass

    def fake_put(subjects_mne_objects, condition_names, rois, subjects, electrodes, **kw):
        seen['structure'] = subjects_mne_objects
        raise _Stop

    monkeypatch.setattr(gu, 'load_subjects_electrodes_to_ROIs_dict', lambda **kw: {})
    monkeypatch.setattr(gu, 'get_sig_chans_per_subject', lambda *a, **kw: {})
    monkeypatch.setattr(gu, 'make_sig_electrodes_per_subject_and_roi_dict',
                        lambda *a, **kw: ({'lpfc': {}}, {'lpfc': {}}))
    monkeypatch.setattr(gu, 'create_subjects_mne_objects_dict', lambda **kw: s)
    monkeypatch.setattr(gu, 'filter_electrode_lists_against_subjects_mne_objects',
                        lambda rois, raw, objs: raw)
    monkeypatch.setattr(gu, 'print_summary_of_dropped_electrodes', lambda *a: None)
    monkeypatch.setattr(lau, 'put_data_in_labeled_array_per_roi_subject', fake_put)

    args = _args(tmp_path, roi='lpfc', rois_dict={'lpfc': []}, subjects=list(s),
                 epochs_root_file='epochs', task='GlobalLocal', electrodes='sig',
                 acc_trials_only=True, conditions=ec.stimulus_main_effect_conditions,
                 LAB_root=None)
    # a split that keeps every trial on the decode side, to show matching runs after it
    partitions = {sub: {'select': set(),
                        'decode': {t for c in s[sub].values()
                                   for t in c['HG_ev1_rescaled'].metadata['trial_count']}}
                  for sub in s}
    with pytest.raises(_Stop):
        xd._build_roi_arrays(args, '/nonexistent', trial_partitions=partitions)
    built = seen['structure']
    for sub in s:
        for cond in s[sub]:
            assert 0 < _n(built, sub, cond) < _n(s, sub, cond)
    assert args.rt_match_log


def test_groups_pair_up_trials_that_are_clean_on_their_own_electrodes():
    """An outlier on an electrode the group does not decode must not cost the
    group a complete pseudo-trial."""
    rng = np.random.default_rng(0)
    channels = ['D0-E0', 'D0-E1', 'D1-E0']
    x = rng.normal(size=(10, 3, 4))
    x[:7, 1] = np.nan                     # D0's outliers sit on E1 only
    arrays = {'lpfc': {'a': x}}
    args = SimpleNamespace(activity_control='none')

    every, n_every, _ = xd._decoded_group(args, arrays, 'lpfc', channels, channels)
    group, n_group, _ = xd._decoded_group(args, arrays, 'lpfc', channels,
                                          ['D0-E0', 'D1-E0'])

    def complete(a):
        return int((~np.isnan(a['lpfc']['a']).any(axis=(1, 2))).sum())

    assert (n_every, complete(every)) == (3, 3)
    assert (n_group, complete(group)) == (2, 10)


def test_rt_matched_trials_with_outliers_still_decode(tmp_path, monkeypatch):
    """The reported failure: RT matching halves each subject's trials, outlier
    electrodes are NaN, and with two dozen subjects randomly paired almost no
    pseudo-trial is complete, so LDA got fewer training rows than classes. The
    job pairs each subject's clean trials up before decoding."""
    cells = cd.condition_cells(ec.stimulus_main_effect_conditions)
    rng = np.random.default_rng(0)
    n_sub, n_ch = 16, 2
    channels = [f'D{i // n_ch}-E{i % n_ch}' for i in range(n_sub * n_ch)]
    arrays = {'lpfc': {}}
    for name in cells:
        x = rng.normal(size=(20, n_sub * n_ch, 16))
        x[rng.random((20, n_sub * n_ch)) < 0.1] = np.nan
        arrays['lpfc'][name] = x
    random_pairing = min(int((~np.isnan(x).any(axis=(1, 2))).sum())
                         for x in arrays['lpfc'].values())
    assert random_pairing <= 3

    monkeypatch.setattr(xd, '_build_roi_arrays',
                        lambda args, lab_root, trial_partitions=None:
                        ('lpfc', arrays, channels, cells))
    monkeypatch.setattr('src.analysis.utils.general_utils.resolve_lab_root',
                        lambda explicit=None: '/nonexistent')
    args = SimpleNamespace(
        data_source='real', synthetic_code=None, LAB_root=None, subjects=[],
        task='GlobalLocal', acc_trials_only=True, epochs_root_file='epochs',
        window_tmin=0.0, window_tmax=0.5, conditions=ec.stimulus_main_effect_conditions,
        electrodes='sig', rois_dict={'lpfc': []}, alpha=0.05, roi='lpfc',
        electrode_definition='none', reference_group='all', tempgen_groups=(),
        window_size=8, step_size=8, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=10, min_group_size=2,
        seed=0, save_dir=str(tmp_path), rt_match='none')

    results = xd.main(args)

    assert set(results['label_transfer']['all']) == {
        'stab_to_stab', 'flex_to_flex', 'stab_to_flex', 'flex_to_stab'}


def test_summary_records_the_split_and_rt_matching(tmp_path, monkeypatch):
    """summary.txt says whether the decode trials were split off and RT-matched,
    and carries the matching's own before/after line."""
    cells = cd.condition_cells(ec.stimulus_main_effect_conditions)
    rng = np.random.default_rng(0)
    arrays = {'lpfc': {name: rng.normal(size=(24, 6, 16)) for name in cells}}
    channels = [f'D{i // 3}-E{i % 3}' for i in range(6)]

    def fake_build(args, lab_root, trial_partitions=None):
        args.rt_match_log.append('[rt-match] RT-matched, kept 1 of 2 trials (50%)')
        return 'lpfc', arrays, channels, cells

    monkeypatch.setattr(xd, '_build_roi_arrays', fake_build)
    monkeypatch.setattr('src.analysis.utils.general_utils.resolve_lab_root',
                        lambda explicit=None: '/nonexistent')
    args = SimpleNamespace(
        data_source='real', synthetic_code=None, LAB_root=None, subjects=[],
        task='GlobalLocal', acc_trials_only=True, epochs_root_file='epochs',
        window_tmin=0.0, window_tmax=0.5, conditions=ec.stimulus_main_effect_conditions,
        electrodes='sig', rois_dict={'lpfc': []}, alpha=0.05, roi='lpfc',
        electrode_definition='none', reference_group='all', tempgen_groups=(),
        window_size=8, step_size=8, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=10, min_group_size=2,
        seed=0, save_dir=str(tmp_path), rt_match='rt', rt_match_bins=10,
        rt_match_balance='equal', rt_match_within=(), rt_match_groups=None,
        rt_match_seed=0)

    results = xd.main(args)

    text = (tmp_path / 'summary.txt').read_text()
    assert 'rt_match: RT-matched; groups=the decoded factors' in text
    assert 'electrode_selection_split: off' in text
    assert 'RT matching of the decode trials:' in text
    assert 'kept 1 of 2 trials' in text
    assert results['rt_match'] == ['[rt-match] RT-matched, kept 1 of 2 trials (50%)']
