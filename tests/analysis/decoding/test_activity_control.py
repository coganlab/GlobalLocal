"""The overall-activity control (`src/analysis/decoding/activity_control.py`).

The decoding tests plant the two answers the control has to tell apart, in a
two-subject pseudopopulation where congruency and switch type each have their
own zero-mean pattern plus a SHARED component:

- gain world: the shared component is a uniform rise on every electrode.
  Removing the subject mean must kill the transfer (the ceilings survive on the
  specific patterns); the mean alone must still transfer.
- pattern world: the shared component is a zero-mean pattern. Removing the mean
  must leave the transfer; the mean alone must carry nothing.
"""

import numpy as np
import pytest

from src.analysis.decoding import activity_control as ac
from src.analysis.decoding import cross_decoding as cd

CHANNELS = [f'S{s}-E{e}' for s in (1, 2) for e in range(8)]


# ---------------------------------------------------------------------------
# the transforms
# ---------------------------------------------------------------------------
def test_channel_subjects():
    assert ac.channel_subjects(['D0057-LFMI8', 'D0107A-RAM3', 'ch0']) == ['D0057', 'D0107A', '']
    # an electrode name with its own '-' still belongs to the subject before the first
    assert ac.channel_subjects(['D0057-A1-A2']) == ['D0057']


def test_remove_subject_mean_zeroes_each_subjects_mean_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 16, 4)) + np.r_[np.full(8, 3.), np.full(8, -2.)][None, :, None]
    subjects = ac.channel_subjects(CHANNELS)
    y = ac.remove_subject_mean(x, subjects)
    np.testing.assert_allclose(y[:, :8].mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(y[:, 8:].mean(axis=1), 0, atol=1e-12)
    # differences between a subject's own electrodes are untouched
    np.testing.assert_allclose(y[:, 1] - y[:, 0], x[:, 1] - x[:, 0])
    # but nothing is mixed across subjects
    assert not np.allclose(y[:, :8], x[:, :8] - x.mean(axis=1, keepdims=True))


def test_remove_subject_mean_keeps_nans_where_they_are():
    x = np.ones((3, 4, 2))
    x[0, :2] = np.nan            # subject A absent from pseudo-trial 0 (padding)
    x[1, 0, 0] = np.nan          # one rejected trace
    y = ac.remove_subject_mean(x, ['A', 'A', 'B', 'B'])
    assert np.isnan(y[0, :2]).all() and not np.isnan(y[0, 2:]).any()
    assert np.isnan(y[1, 0, 0]) and y[1, 1, 0] == 0.0


def test_a_single_electrode_subject_is_zeroed():
    x = np.random.default_rng(1).normal(size=(4, 3, 2))
    y = ac.remove_subject_mean(x, ['A', 'A', 'B'])
    np.testing.assert_array_equal(y[:, 2], 0.0)


def test_subject_mean_is_one_feature_per_subject():
    x = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    m, order = ac.subject_mean(x, ['A', 'A', 'B', 'B', 'B'])
    assert order == ['A', 'B'] and m.shape == (2, 2, 3)
    np.testing.assert_allclose(m[:, 0], x[:, :2].mean(axis=1))
    np.testing.assert_allclose(m[:, 1], x[:, 2:].mean(axis=1))


def test_apply_activity_control_on_roi_dicts():
    rng = np.random.default_rng(2)
    arrays = {'lpfc': {c: rng.normal(size=(6, 16, 4)) for c in ('a', 'b')}}
    same, names, info = ac.apply_activity_control(arrays, 'lpfc', CHANNELS, 'none')
    assert same is arrays and names == CHANNELS and info['n_features'] == 16
    out, names, info = ac.apply_activity_control(arrays, 'lpfc', CHANNELS, 'mean_only')
    assert names == ['S1-mean', 'S2-mean'] and out['lpfc']['a'].shape == (6, 2, 4)
    assert info == dict(mode='mean_only', n_electrodes=16, n_subjects=2, n_features=2,
                        single_electrode_subjects=[])
    nine = {'lpfc': {c: a[:, :9] for c, a in arrays['lpfc'].items()}}   # S2 keeps one
    out, names, info = ac.apply_activity_control(nine, 'lpfc', CHANNELS[:9], 'remove_mean')
    assert names == CHANNELS[:9] and info['single_electrode_subjects'] == ['S2']
    np.testing.assert_array_equal(out['lpfc']['a'][:, 8], 0.0)
    assert 'one electrode' in ac.describe(info)
    with pytest.raises(ValueError, match='must be one of'):
        ac.apply_activity_control(arrays, 'lpfc', CHANNELS, 'zscore')
    with pytest.raises(ValueError, match='channel names'):
        ac.apply_activity_control(arrays, 'lpfc', CHANNELS[:3], 'remove_mean')


# ---------------------------------------------------------------------------
# planted answers, decoded
# ---------------------------------------------------------------------------
def _world(shared, n_trials=80, n_time=8, a_shared=0.25, a_specific=0.12, seed=0):
    """Congruency and switch type: own zero-mean pattern each (`a_specific`), plus
    a component they share (`a_shared`): 'gain' = uniform on every electrode,
    'pattern' = a third zero-mean pattern.

    Effects are kept weak, in the regime of the real data (ceilings ~0.8). With
    large effects LDA treats the other factor's effect as within-class variance
    and projects the shared direction out, so nothing transfers in any world."""
    rng = np.random.default_rng(seed)
    n = 8
    basis = np.linalg.qr(np.c_[np.ones(n), rng.normal(size=(n, 3))])[0]
    gain, p_cong, p_sw, p_shared = basis.T          # p_* are orthogonal to the gain
    common = gain if shared == 'gain' else p_shared
    arrays = {}
    for cong in ('c', 'i'):
        for sw in ('r', 's'):
            x = rng.normal(size=(n_trials, 2 * n, n_time))
            pattern = ((cong == 'i') * (a_specific * p_cong + a_shared * common)
                       + (sw == 's') * (a_specific * p_sw + a_shared * common))
            x += np.tile(pattern, 2)[None, :, None] * np.sqrt(n)
            arrays[f'Stimulus_{cong}_{sw}'] = x
    cells = {name: dict(congruency=name.split('_')[1], switchType=name.split('_')[2])
             for name in arrays}
    return {'roi': arrays}, cells


def _acc(arrays, cells, train, test):
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    stab, flex = cd.stability_flexibility_strings(cells)
    strings = {'stab': stab, 'flex': flex}
    out = cd.run_cross_decoding(arrays, 'roi', strings[train], strings[test],
                                n_splits=3, n_repeats=2, window=8, step_size=8)
    acc, _ = compute_accuracies(out['cm_true'], out['cm_shuffle'])
    return float(acc.mean())


@pytest.mark.parametrize('shared', ['gain', 'pattern'])
def test_the_controls_separate_a_shared_gain_from_a_shared_pattern(shared):
    arrays, cells = _world(shared)
    full = {k: _acc(arrays, cells, *k) for k in (('stab', 'stab'), ('stab', 'flex'))}
    assert full['stab', 'stab'] > 0.7 and full['stab', 'flex'] > 0.6   # both worlds transfer

    removed, _, _ = ac.apply_activity_control(arrays, 'roi', CHANNELS, 'remove_mean')
    means, _, _ = ac.apply_activity_control(arrays, 'roi', CHANNELS, 'mean_only')
    rm = {k: _acc(removed, cells, *k) for k in (('stab', 'stab'), ('stab', 'flex'))}
    mo = {k: _acc(means, cells, *k) for k in (('stab', 'stab'), ('stab', 'flex'))}

    # the specific pattern keeps the ceiling above chance once the mean is gone
    assert rm['stab', 'stab'] > 0.56
    if shared == 'gain':
        assert abs(rm['stab', 'flex'] - 0.5) < 0.06     # the transfer was the gain
        assert mo['stab', 'flex'] > 0.65                # and the mean alone carries it
    else:
        assert rm['stab', 'flex'] > 0.6                 # the transfer is a pattern
        assert abs(mo['stab', 'stab'] - 0.5) < 0.06     # the mean carries nothing
        assert abs(mo['stab', 'flex'] - 0.5) < 0.06
