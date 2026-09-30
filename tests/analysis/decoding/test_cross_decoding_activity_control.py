"""ACTIVITY_CONTROL inside the A4 / transfer job.

The transforms themselves (and the planted gain-vs-pattern answers) are tested
in test_activity_control.py. These pin the job's side: every decode receives the
transformed arrays, per electrode group, and the summary says what was done.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('ieeg')

from src.analysis.config import experiment_conditions as ec
from src.analysis.decoding import activity_control as ac
from src.analysis.decoding import block_transfer as bt
from src.analysis.decoding import cross_decoding as cd
from dcc_scripts.decoding import stability_flexibility_cross_decoding_dcc as xd

CHANNELS = [f'D{i // 4}-E{i % 4}' for i in range(8)]      # two subjects, four each


def _args(tmp_path, conditions, **kw):
    base = dict(
        data_source='real', synthetic_code=None, LAB_root=None, subjects=[],
        task='GlobalLocal', acc_trials_only=True, epochs_root_file='epochs',
        window_tmin=0.0, window_tmax=0.5, conditions=conditions,
        electrodes='sig', rois_dict={'lpfc': []}, alpha=0.05, roi='lpfc',
        electrode_definition='none', reference_group='all', tempgen_groups=(),
        window_size=8, step_size=8, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=10, min_group_size=2,
        seed=0, save_dir=str(tmp_path))
    base.update(kw)
    return SimpleNamespace(**base)


def _stub(monkeypatch, conditions):
    """Fake pseudopopulation data, and a spy recording what every decode is fed."""
    cells = cd.condition_cells(conditions)
    rng = np.random.default_rng(0)
    arrays = {'lpfc': {name: rng.normal(size=(24, len(CHANNELS), 16)) + 5.0
                       for name in cells}}
    monkeypatch.setattr(xd, '_build_roi_arrays',
                        lambda args, lab_root, trial_partitions=None:
                        ('lpfc', arrays, CHANNELS, cells))
    monkeypatch.setattr('src.analysis.utils.general_utils.resolve_lab_root',
                        lambda explicit=None: '/nonexistent')
    fed = []
    real = cd.run_cross_decoding

    def spy(roi_arrays, roi, *a, **kw):
        fed.append(np.concatenate([np.asarray(x) for x in roi_arrays[roi].values()]))
        return real(roi_arrays, roi, *a, **kw)
    monkeypatch.setattr(cd, 'run_cross_decoding', spy)
    return fed


def test_mean_only_feeds_one_feature_per_subject(tmp_path, monkeypatch):
    fed = _stub(monkeypatch, ec.stimulus_main_effect_conditions)
    results = xd.main(_args(tmp_path, ec.stimulus_main_effect_conditions,
                            activity_control='mean_only'))
    assert fed and all(x.shape[1] == 2 for x in fed)
    info = results['label_transfer']['all']['stab_to_flex']['activity_control']
    assert info['n_features'] == 2 and info['n_electrodes'] == 8
    assert results['label_transfer']['all']['stab_to_flex']['n_channels'] == 8
    text = (tmp_path / 'summary.txt').read_text()
    assert 'activity_control: mean_only (one feature per subject' in text
    assert '[all] activity control: mean_only: 8 electrodes -> 2 per-subject means' in text


def test_remove_mean_reaches_every_design(tmp_path, monkeypatch):
    """The 16-cell set runs the within-block designs too; all of them, and the
    label transfer, see the per-subject means removed."""
    fed = _stub(monkeypatch, ec.stimulus_experiment_conditions)
    results = xd.main(_args(tmp_path, ec.stimulus_experiment_conditions,
                            activity_control='remove_mean'))
    assert results['within_block'] and results['label_transfer']
    assert len(fed) >= 8                             # 4 within-block + 4 label transfer
    for x in fed:
        assert x.shape[1] == 8
        np.testing.assert_allclose(x[:, :4].mean(axis=1), 0, atol=1e-9)
        np.testing.assert_allclose(x[:, 4:].mean(axis=1), 0, atol=1e-9)
    text = (tmp_path / 'summary.txt').read_text()
    assert '[all] activity control: remove_mean over 2 subjects' in text


def test_no_control_leaves_the_data_alone(tmp_path, monkeypatch):
    fed = _stub(monkeypatch, ec.stimulus_main_effect_conditions)
    results = xd.main(_args(tmp_path, ec.stimulus_main_effect_conditions))
    assert all(x.shape[1] == 8 and x.mean() > 4 for x in fed)
    assert results['label_transfer']['all']['stab_to_flex']['activity_control']['mode'] == 'none'
    text = (tmp_path / 'summary.txt').read_text()
    assert 'activity_control: none' in text and '] activity control:' not in text


def test_groups_are_controlled_over_their_own_electrodes(tmp_path, monkeypatch):
    """A subject's mean is taken over the electrodes the GROUP decodes, not over
    every loaded electrode."""
    arrays = {'lpfc': {'a': np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3)}}
    args = SimpleNamespace(activity_control='remove_mean')
    group = CHANNELS[:3] + CHANNELS[4:6]
    out, n_kept, info = xd._decoded_group(args, arrays, 'lpfc', CHANNELS, group)
    x = out['lpfc']['a']
    assert n_kept == 5 and x.shape[1] == 5
    np.testing.assert_allclose(x[:, :3].mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(x[:, 3:].mean(axis=1), 0, atol=1e-12)
    assert info['n_subjects'] == 2
    assert xd._decoded_group(args, arrays, 'lpfc', CHANNELS, ['nope']) == (None, 0, None)


def test_an_unknown_control_is_refused(tmp_path):
    args = _args(tmp_path, ec.stimulus_main_effect_conditions, activity_control='zscore')
    with pytest.raises(ValueError, match='activity_control must be one of'):
        xd.main(args)


def test_the_transfer_job_applies_it_too(tmp_path, monkeypatch):
    fed = []
    real = bt.run_block_transfer

    def spy(arrays, roi, *a, **kw):
        fed.append(np.concatenate([np.asarray(x) for x in arrays[roi].values()]))
        return real(arrays, roi, *a, **kw)
    monkeypatch.setattr(bt, 'run_block_transfer', spy)
    args = SimpleNamespace(
        analysis='task_transfer', data_source='synthetic', synthetic_code='shared',
        electrodes='sig', window_size=16, step_size=16, sampling_rate=256,
        n_splits=3, n_repeats=2, explained_variance=0.8, frac_train=None, n_perm=10,
        seed=0, save_dir=str(tmp_path), activity_control='remove_mean')
    xd.main(args)
    # synthetic channels carry no subject prefix, so they form one subject
    assert fed and all(np.allclose(x.mean(axis=1), 0, atol=1e-9) for x in fed)
    assert 'activity_control: remove_mean over 1 subjects' in \
        (tmp_path / 'summary.txt').read_text()
