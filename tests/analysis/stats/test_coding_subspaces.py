"""The subspace test (docs/analysis_plans.md, subspace plan §3.4).

What is pinned:

  * the primitives: the subspace is orthonormal and spans a rank-1 matrix's
    pattern; the principal-angle cosines equal SciPy's; the alignment index is
    1 for a matrix's own subspace, 0 for an orthogonal one, cos^2 at k = 1;
  * the plan's two worked examples (§3.6): the uncentred SVD keeps a sustained
    pattern that PCA drops, and principal angles do not care about PC order or
    sign where the Spearman r of the PC1s does;
  * k = 1 on one window reduces to the §3.2 coding-axis cosine;
  * the ceiling normalization: near 1 for a planted shared subspace, near 0 for
    orthogonal ones, through noise;
  * on trial-level data: the dimension test finds the planted ranks, a shared
    direction is marked shared and not one code, orthogonal codes are not
    marked shared;
  * the job runs end to end on synthetic data and writes every output.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import subspace_angles

from src.analysis.stats import coding_axes as ca
from src.analysis.stats import coding_subspaces as cs
from test_coding_axes import _scored, _trial_long          # same directory (pytest rootdir import)


def _per_split_matrices(C, S, n_splits=4, noise=0.0, n_subj=4, seed=0):
    """A per-split table from electrode x window matrices C (congruency) and S
    (switch): every half is the matrix plus independent noise."""
    rng = np.random.default_rng(seed)
    n_e, n_w = C.shape
    e, w = np.meshgrid(np.arange(n_e), np.arange(n_w), indexing='ij')
    frames = []
    for k in range(n_splits):
        cols = {col: (M + noise * rng.normal(size=M.shape)).ravel()
                for col, M in (('mxA', C), ('mxB', C), ('myA', S), ('myB', S))}
        frames.append(pd.DataFrame(dict(subject=[f'P{i % n_subj}' for i in e.ravel()],
                                        electrode=[f'e{i}' for i in e.ravel()],
                                        split=k, window=w.ravel(), **cols)))
    return pd.concat(frames, ignore_index=True)


def _orthonormal(n, k, seed=0):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(n, k)))[0]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def test_primitives_match_scipy_and_planted_geometry():
    rng = np.random.default_rng(0)
    u = rng.normal(size=30)
    rank1 = np.outer(u, [1.0, 2.0, 0.5, 3.0])           # one pattern, growing and shrinking
    U = cs.coding_subspace(rank1, 1)
    assert abs(U[:, 0] @ u / np.linalg.norm(u)) == pytest.approx(1.0)

    A, B = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    UA, UB = cs.coding_subspace(A, 3), cs.coding_subspace(B, 3)
    assert np.allclose(UA.T @ UA, np.eye(3))
    ours = cs.principal_angle_cosines(UA, UB)
    assert np.allclose(ours, np.sort(np.cos(subspace_angles(UA, UB)))[::-1])
    assert np.all(np.diff(ours) <= 0)                   # cos θ1 >= cos θ2 >= ...

    assert cs.alignment_index(A, UA) == pytest.approx(1.0)        # its own top 3
    Q = _orthonormal(30, 4, seed=1)
    only_first = np.outer(Q[:, 0], rng.normal(size=6))
    assert cs.alignment_index(only_first, Q[:, 1:3]) == pytest.approx(0.0, abs=1e-12)
    c, s = rng.normal(size=30), rng.normal(size=30)
    cos = c @ s / np.linalg.norm(c) / np.linalg.norm(s)
    assert cs.alignment_index(c[:, None], (s / np.linalg.norm(s))[:, None]) == pytest.approx(cos ** 2)
    with pytest.raises(ValueError, match='k='):
        cs.coding_subspace(A, 7)


def test_uncentred_svd_keeps_the_sustained_pattern_that_pca_drops():
    """Plan §3.6: a sustained pattern u in every window, a second v only late."""
    u, v = _orthonormal(40, 2, seed=2).T
    C = np.column_stack([u] * 5 + [u + v] * 5)
    uncentred = cs.coding_subspace(C, 2)
    assert cs.principal_angle_cosines(uncentred, np.column_stack([u, v])) == pytest.approx([1, 1])

    centred = C - C.mean(axis=1, keepdims=True)         # PCA: each electrode centred over windows
    pc1 = cs.coding_subspace(centred, 1)[:, 0]
    assert abs(pc1 @ v) == pytest.approx(1.0)           # PCA finds only the change...
    assert abs(pc1 @ u) == pytest.approx(0.0, abs=1e-12)  # ...and loses the sustained code


def test_principal_angles_ignore_pc_order_and_sign():
    """Plan §3.6: both variables use u and v, congruency weights u more, switch
    weights v more. PC1s differ, so Spearman r of the PC1s reads 'orthogonal';
    the 2-D subspaces are identical."""
    from scipy.stats import spearmanr
    u, v = _orthonormal(40, 2, seed=3).T
    t = np.linspace(0, 1, 8)
    C = np.outer(u, 2 + t) + np.outer(v, 1 - t)
    S = np.outer(v, 2 + t) + np.outer(u, 1 - t)
    UC, US = cs.coding_subspace(C, 2), cs.coding_subspace(S, 2)
    assert cs.principal_angle_cosines(UC, US) == pytest.approx([1, 1])
    assert cs.principal_angle_cosines(UC[:, :1], US[:, :1])[0] < 0.6
    rho = spearmanr(UC[:, 0], US[:, 0])[0]
    flipped = spearmanr(UC[:, 0], -US[:, 0])[0]
    assert rho == pytest.approx(-flipped)               # the sign is arbitrary
    assert cs.principal_angle_cosines(UC, -US) == pytest.approx([1, 1])


# ---------------------------------------------------------------------------
# per-split statistics
# ---------------------------------------------------------------------------
def test_k1_on_one_window_is_the_coding_axis_cosine():
    rng = np.random.default_rng(4)
    c = rng.normal(size=24)
    s = 0.6 * c + rng.normal(size=24)
    per = _per_split_matrices(c[:, None], s[:, None], n_splits=3)
    table, _ = cs.subspace_stats(per, k_max=1)
    cross = ca.coding_axis_cosines(per)['cross'].iloc[0]
    angle = table.query("measure == 'angle'").iloc[0]
    assert angle['cross'] == pytest.approx(abs(cross))
    assert angle['cross_deg'] == pytest.approx(np.degrees(np.arccos(abs(cross))))
    for direction in cs.ALIGNMENT_DIRECTIONS:
        ai = table.query("measure == 'alignment' and component == @direction").iloc[0]
        assert ai['cross'] == pytest.approx(cross ** 2)
    with pytest.raises(ValueError, match='k_max'):
        cs.subspace_stats(per, k_max=2)                 # one window: rank 1 at most


def test_normalized_values_through_noise():
    """Noise pulls the raw values down; dividing by the ceiling puts a shared
    2-D subspace back near 1 and leaves orthogonal ones near 0. At k > 1 the
    geometric-mean ceiling is a heuristic: right on average, not per draw (a
    single draw's θ2 can land at 1.15), so this averages planted geometries."""
    shared, orth = [], []
    for seed in range(8):
        U = _orthonormal(80, 4, seed=seed) * 4
        rng = np.random.default_rng(100 + seed)
        coef = lambda: rng.normal(size=(2, 10))
        for out, (C, S) in ((shared, (U[:, :2] @ coef(), U[:, :2] @ coef())),
                            (orth, (U[:, :2] @ coef(), U[:, 2:] @ coef()))):
            t, _ = cs.subspace_stats(_per_split_matrices(C, S, noise=1.0, seed=seed), k_max=2)
            out.append(t.loc[t['k'] == 2, ['cross', 'normalized']].to_numpy())
    shared, orth = np.mean(shared, axis=0), np.mean(orth, axis=0)    # rows: θ1, θ2, C→S, S→C

    assert (shared[:, 0] < 0.75).all()                  # noise really did pull them down
    assert np.allclose(shared[:, 1], 1.0, atol=0.1)
    assert (orth[:, 1] < 0.3).all()


# ---------------------------------------------------------------------------
# trial-level data: dimension test, the marks
# ---------------------------------------------------------------------------
def _planted(switch_uses_u, seed=6):
    """Congruency: a sustained pattern u and a late one v (rank 2). Switch: w,
    plus u if `switch_uses_u` (rank 1 either way). One shared direction or none."""
    n_elec, n_time = 12, 24
    u, v, w = _orthonormal(n_elec, 3, seed=seed).T * 3
    late = (np.arange(n_time) >= 12).astype(float)
    C = np.outer(u, np.ones(n_time)) + np.outer(v, late)
    S = np.outer(w + (0.8 * u if switch_uses_u else 0), np.ones(n_time))
    df, times = _trial_long(C, S, n_subj=6, n_elec=n_elec, n_time=n_time, noise=1.0, seed=seed)
    per, labels, means = _scored(df, times, size=4, step=4, n_splits=10)
    null = {f: cs.subspace_null(labels, means, f, k_max=3, n_perm=30, n_splits=4, seed=i)
            for i, f in enumerate(ca.NULL_FACTORS)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cs.compare_subspaces(per, null=null, k_max=3, n_boot=100)


@pytest.fixture(scope='module')
def partly_shared():
    return _planted(switch_uses_u=True)


def test_dimension_test_finds_the_planted_ranks(partly_shared):
    _, dims, summary = partly_shared
    assert summary['k'] == {'congruency': 2, 'switchType': 1}
    assert summary['primary_k'] == 1
    kept = dims.set_index(['variable', 'dimension'])['kept']
    assert kept[('congruency', 2)] and not kept[('congruency', 3)]
    assert not kept[('switchType', 2)]


def test_a_shared_direction_is_shared_and_not_one_code(partly_shared):
    table, _, summary = partly_shared
    t1 = summary['theta_1']
    assert t1['shared'] and t1['reliable'] and t1['not_one_code']
    assert 0.2 < t1['normalized'] < 0.8
    for key in ('alignment_congruency_to_switch', 'alignment_switch_to_congruency'):
        assert summary[key]['shared'] and summary[key]['not_one_code']
    assert table['primary'].sum() == 3                  # θ1 and both directions at k = 1


def test_orthogonal_codes_are_not_shared():
    table, _, summary = _planted(switch_uses_u=False)
    assert not summary['theta_1']['shared']
    primary = table[table['primary']]
    assert not primary['shared'].any()
    assert (primary['normalized'] < 0.3).all()


def test_subspace_null_is_reproducible():
    df, times = _trial_long(np.ones(4), np.ones(4), n_subj=2, n_elec=4, n_time=8, seed=7)
    _, labels, means = _scored(df, times, size=2, step=2, n_splits=2)
    a = cs.subspace_null(labels, means, 'congruency', k_max=2, n_perm=3, n_splits=2, seed=1)
    b = cs.subspace_null(labels, means, 'congruency', k_max=2, n_perm=3, n_splits=2, seed=1)
    assert set(a) == {'cross', 'ceiling', 'dims'}
    assert a['cross'].shape == (3, len(cs._rows(2))) and a['dims'].shape == (3, 2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match='factor'):
        cs.subspace_null_stats(_scored(df, times, size=2, step=2, n_splits=2)[0], 'block')


# ---------------------------------------------------------------------------
# the job, end to end
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_job_writes_every_output_on_synthetic_data(tmp_path):
    from types import SimpleNamespace
    from dcc_scripts.subspace.coding_axes_dcc import main

    args = SimpleNamespace(
        timestamp='test', LAB_root=None, subjects=None, task='GlobalLocal',
        acc_trials_only=True, data_source='synthetic', synthetic_rho=0.4,
        synthetic_sfreq=256.0, epochs_root_file=None, windows_tmin=0.0, windows_tmax=1.5,
        window_size=64, step_size=16, remove_mean=False, rt_control='none',
        rt_match_bins=10, electrodes='all', rois_dict=None, n_splits=4, n_perm=5,
        n_null_splits=2, n_boot=20, alpha=0.05, seed=0, k_max=3,
        traj_axes=[(0.0, 1.5), (0.1, 0.5)], traj_first='congruency', save_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main(args)
    for name in ('subspace_test.csv', 'subspace_dimensions.csv', 'subspace_summary.json',
                 'null_subspace.npz', 'subspace_test.png', 'trajectories.csv',
                 'trajectory_fractions.csv', 'trajectory_summary.json',
                 'trajectories_0.0-1.5s.png', 'trajectories_0.1-0.5s_uncentred.png',
                 'coding_axis_cosines.csv', 'null_cross.npz'):
        assert (tmp_path / name).stat().st_size > 0, name
    config = json.loads((tmp_path / 'run_config.json').read_text())
    assert config['k_max'] == 3 and config['traj_axes'] == [[0.0, 1.5], [0.1, 0.5]]
    null = np.load(tmp_path / 'null_subspace.npz')
    assert null['congruency_cross'].shape == (5, len(cs._rows(3)))
