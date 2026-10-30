"""The condition trajectories (docs/analysis_plans.md, subspace plan §4) and
the block-split separations (§6).

What is pinned:

  * the module matches a direct computation on one split: axes from one half,
    block-balanced cell means from the other, Gram-Schmidt, projection, both
    directions averaged;
  * noise-free, the centred cells sit at half the planted effects, the shared
    part of switch shows up along the congruency axis, `first='switch'` swaps
    the roles, and centring removes an evoked response the uncentred cells keep;
  * the four proportion blocks are weighted equally, so a block offset with
    unequal trial counts does not move the cells;
  * the plane fractions: an interaction outside the plane lowers both, and
    with noise the noise-corrected one recovers the planted share while the
    raw one falls below it;
  * the block separations match a direct one-split computation, and a planted
    adaptation of each code lands on its own proportion and not the other;
  * the proportion trajectories put each planted change on its own axis in
    oblique coordinates, with partly shared axes;
  * raw-axis trajectories (`first=None`) are the projections on the unit c and
    s axes, with unchanged plane fractions; unique-axis block separations
    drop an adaptation along the shared direction.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import coding_trajectories as ct
from src.analysis.stats import stability_flexibility_segregation as sfs
from test_coding_axes import _trial_long          # same directory (pytest rootdir import)

N_SUBJ, N_ELEC = 3, 8


def _with_terms(df, terms):
    """Add `terms(trials)` -> (trials, electrodes) per participant to every
    sample of each row's hg; `trials` is the participant's trial table."""
    df = df.copy()
    out = []
    for _, sub in df.groupby('subject', sort=False):
        trials = sub.drop_duplicates('trial').set_index('trial')
        add = terms(trials)                                       # (n_trials, n_elec)
        e = sub['electrode'].str.split('-e').str[-1].astype(int).to_numpy()
        t = trials.index.get_indexer(sub['trial'])
        out.append(pd.Series([h + a for h, a in zip(sub['hg'], add[t, e])], index=sub.index))
    df['hg'] = pd.concat(out)
    return df


def _run(df, times, n_splits=4, **kw):
    hg, _ = sfs.window_means(df['hg'].to_numpy(), times, 1, 1)
    axis = ct.axis_window_means(df['hg'].to_numpy(), times, [(times[0], times[-1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ct.condition_trajectories(df.drop(columns='hg'), hg, axis, ['all'],
                                         n_splits=n_splits, n_boot=kw.pop('n_boot', 50), **kw)


def _centred(traj, cells='centred'):
    return traj[traj['cells'] == cells].groupby('cell')[['x', 'y']].mean()


def _norm(v):
    return np.linalg.norm(np.tile(v, N_SUBJ))                    # pooled over participants


# ---------------------------------------------------------------------------
def test_matches_a_direct_computation_on_one_split():
    rng = np.random.default_rng(0)
    df, times = _trial_long(rng.normal(size=N_ELEC), rng.normal(size=N_ELEC),
                            n_subj=N_SUBJ, n_elec=N_ELEC, n_time=3, seed=0)
    traj, _, _ = _run(df, times, n_splits=1, seed=5)

    labels = df.drop(columns='hg')
    hg = np.vstack(df['hg'].to_numpy())                           # (rows, time)
    halves = sfs.shared_halves(labels, n_splits=1, seed=5)
    axes = sfs.main_effect_mean_diffs(labels, hg.mean(1, keepdims=True), halves=halves)
    axes = axes.sort_values(['subject', 'electrode'])
    d = labels.assign(half=[halves[s].loc[t, 0] for s, t in zip(labels['subject'], labels['trial'])])
    cell_of = {'i': 'incongruent', 'c': 'congruent'}, {'s': 'switch', 'r': 'repeat'}
    d['cell'] = [f'{cell_of[0][a]}-{cell_of[1][b]}' for a, b in zip(d['congruency'], d['switchType'])]
    order = list(ct.CELLS)
    blocks = ['incongruent_proportion', 'switch_proportion']
    cells = {}                                                    # half -> (elec, time, cell)
    for h in (0, 1):
        sub = d[d['half'] == h]
        rows = {c: sub.index[sub['cell'] == c] for c in order}
        block_means = {c: pd.DataFrame(hg[r], index=r).groupby(
            [sub.loc[r, 'subject'], sub.loc[r, 'electrode']] + [sub.loc[r, b] for b in blocks]
        ).mean().groupby(level=[0, 1]).mean() for c, r in rows.items()}
        cells[h] = np.stack([block_means[c].to_numpy() for c in order], -1)
        cells[h] = cells[h] - cells[h].mean(-1, keepdims=True)

    x_sum, y_sum = 0, 0
    for axes_half, cells_half in ((0, 1), (1, 0)):
        c = axes[f'mx{"AB"[axes_half]}'].to_numpy()
        s = axes[f'my{"AB"[axes_half]}'].to_numpy()
        x_hat = c / np.linalg.norm(c)
        s_perp = s - (s @ x_hat) * x_hat
        y_hat = s_perp / np.linalg.norm(s_perp)
        x_sum = x_sum + np.einsum('etc,e->tc', cells[cells_half], x_hat)
        y_sum = y_sum + np.einsum('etc,e->tc', cells[cells_half], y_hat)
    got = traj[traj['cells'] == 'centred'].pivot(index='window', columns='cell')
    assert np.allclose(got['x'][order].to_numpy(), x_sum / 2)
    assert np.allclose(got['y'][order].to_numpy(), y_sum / 2)


def test_noise_free_cells_sit_at_half_the_effects():
    rng = np.random.default_rng(1)
    c = rng.normal(size=N_ELEC)
    s = 0.6 * c + rng.normal(size=N_ELEC)                         # partly shared with c
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)
    evoked = _with_terms(df, lambda t: np.full((len(t), N_ELEC), 5.0))
    traj, frac, summary = _run(evoked, times)
    m = _centred(traj)

    x_hat = np.tile(c, N_SUBJ) / _norm(c)
    s_all = np.tile(s, N_SUBJ)
    s_x = s_all @ x_hat
    s_perp = np.linalg.norm(s_all - s_x * x_hat)
    assert m.loc['incongruent-switch', 'x'] == pytest.approx(_norm(c) / 2 + s_x / 2)
    assert m.loc['incongruent-repeat', 'x'] == pytest.approx(_norm(c) / 2 - s_x / 2)
    assert m.loc['congruent-switch', 'y'] == pytest.approx(s_perp / 2)
    assert m.loc['incongruent-repeat', 'y'] == pytest.approx(-s_perp / 2)
    assert summary['axes']['all']['axis_cosine'] == pytest.approx(s_x / np.linalg.norm(s_all))
    assert frac['plane_fraction'].to_numpy() == pytest.approx(1.0)

    # the evoked response is in every cell: centring removes it, uncentred keeps it
    u = _centred(traj, 'uncentred')
    assert (u['x'] - m['x']).to_numpy() == pytest.approx(5.0 * x_hat.sum())

    raw_traj, raw_frac, _ = _run(evoked, times, first=None)
    raw = _centred(raw_traj)
    c_s = np.tile(c, N_SUBJ) @ (s_all / np.linalg.norm(s_all))
    assert np.allclose(raw['x'], m['x'])
    assert raw.loc['incongruent-switch', 'y'] == pytest.approx((np.linalg.norm(s_all) + c_s) / 2)
    assert raw.loc['congruent-switch', 'y'] == pytest.approx((np.linalg.norm(s_all) - c_s) / 2)
    assert np.allclose(raw_frac['plane_fraction'], frac['plane_fraction'])

    swapped = _centred(_run(evoked, times, first='switch')[0])
    assert swapped.loc['incongruent-switch', 'x'] == pytest.approx(np.linalg.norm(s_all) / 2
                                                                   + (s_all @ np.tile(c, N_SUBJ))
                                                                   / np.linalg.norm(s_all) / 2)


def test_blocks_are_weighted_equally():
    """Incongruent trials 3x as common in the 75% blocks, plus a block offset:
    pooled means would put the offset on the incongruent side."""
    rng = np.random.default_rng(2)
    c, s, offset = rng.normal(size=(3, N_ELEC))
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)
    trial = df.drop_duplicates(['subject', 'trial']).copy()
    rare = (((trial['incongruent_proportion'] == 25) & (trial['congruency'] == 'i'))
            | ((trial['incongruent_proportion'] == 75) & (trial['congruency'] == 'c')))
    cell = trial.groupby(['subject', 'congruency', 'switchType', 'incongruent_proportion',
                          'switch_proportion'])
    drop = trial[rare & (cell.cumcount() < 2 * cell['trial'].transform('size') // 3)]
    keep = ~df.set_index(['subject', 'trial']).index.isin(
        pd.MultiIndex.from_frame(drop[['subject', 'trial']]))
    unbalanced = df[keep].reset_index(drop=True)
    shifted = _with_terms(unbalanced, lambda t: np.outer(t['incongruent_proportion'] == 75, offset) * 3.0)

    plain, shift = _centred(_run(unbalanced, times)[0]), _centred(_run(shifted, times)[0])
    assert np.allclose(plain.to_numpy(), shift.to_numpy())
    assert plain.loc['incongruent-repeat', 'x'] > 0


def test_plane_fractions_with_an_interaction_outside_the_plane():
    q = np.linalg.qr(np.random.default_rng(3).normal(size=(N_ELEC, 3)))[0] * 3
    c, s, inter = q.T                                              # mutually orthogonal
    expected = (c @ c + s @ s) / (c @ c + s @ s + inter @ inter / 4)

    def with_interaction(noise, seed):
        df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=noise,
                                seed=seed)
        xc = lambda t: np.where(t['congruency'] == 'i', 0.5, -0.5)
        xs = lambda t: np.where(t['switchType'] == 's', 0.5, -0.5)
        return _with_terms(df, lambda t: np.outer(xc(t) * xs(t), inter)), times

    clean = _run(*with_interaction(0.0, 0))[2]['axes']['all']
    assert clean['plane_fraction'] == pytest.approx(expected)
    assert clean['plane_fraction_signal'] == pytest.approx(expected)

    noisy = _run(*with_interaction(2.0, 1), n_splits=10)[2]['axes']['all']
    assert noisy['plane_fraction'] < expected - 0.1                # noise in every direction
    assert noisy['plane_fraction_signal'] == pytest.approx(expected, abs=0.1)
    lo, hi = noisy['plane_fraction_signal_ci']
    assert lo <= noisy['plane_fraction_signal'] <= hi


def _run_blocks(df, times, n_splits=4, **kw):
    hg, _ = sfs.window_means(df['hg'].to_numpy(), times, 1, 1)
    axis = ct.axis_window_means(df['hg'].to_numpy(), times, [(times[0], times[-1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ct.block_separations(df.drop(columns='hg'), hg, axis, ['all'],
                                    n_splits=n_splits, n_boot=kw.pop('n_boot', 50), **kw)


def _level(prop):
    return np.where(prop == 25, 'low', 'high')


def _own(table):
    """The rows of a block table on each variable's own axis."""
    return table[table['axis'] == table['variable'].map({v: a[0] for v, a in ct.AXES_OF.items()})]


def test_block_separations_match_a_direct_computation_on_one_split():
    rng = np.random.default_rng(4)
    df, times = _trial_long(rng.normal(size=N_ELEC), rng.normal(size=N_ELEC),
                            n_subj=N_SUBJ, n_elec=N_ELEC, n_time=3, seed=4)
    seps, _, _ = _run_blocks(df, times, n_splits=1, seed=5)
    seps = _own(seps)

    labels = df.drop(columns='hg')
    hg = pd.DataFrame(np.vstack(df['hg'].to_numpy()))            # (rows, time)
    halves = sfs.shared_halves(labels, n_splits=1, seed=5)
    axes = sfs.main_effect_mean_diffs(labels, hg.to_numpy().mean(1, keepdims=True), halves=halves)
    axes = axes.sort_values(['subject', 'electrode'])
    keys = [pd.Series([halves[s].loc[t, 0] for s, t in zip(labels['subject'], labels['trial'])],
                      name='half'),
            labels['subject'], labels['electrode'],
            pd.Series(_level(labels['incongruent_proportion']), name='inc'),
            pd.Series(_level(labels['switch_proportion']), name='sw'),
            labels['congruency'], labels['switchType']]
    cells = hg.groupby(keys).mean()
    outer = ['half', 'subject', 'electrode', 'inc', 'sw']
    sep = {'congruency': (cells.xs('i', level='congruency').groupby(level=outer).mean()
                          - cells.xs('c', level='congruency').groupby(level=outer).mean()),
           'switchType': (cells.xs('s', level='switchType').groupby(level=outer).mean()
                          - cells.xs('r', level='switchType').groupby(level=outer).mean())}
    for var, col in (('congruency', 'mx'), ('switchType', 'my')):
        for inc, sw in ct.BLOCKS:
            want = 0
            for axes_half, cells_half in ((0, 1), (1, 0)):
                a = axes[f'{col}{"AB"[axes_half]}'].to_numpy()
                block = sep[var].loc[cells_half].xs((inc, sw), level=['inc', 'sw'])
                want = want + (a / np.linalg.norm(a)) @ block.to_numpy()
            got = seps[(seps['variable'] == var) & (seps['inc_prop'] == inc)
                       & (seps['switch_prop'] == sw) & (seps['window'] >= 0)]
            assert np.allclose(got.sort_values('window')['separation'], want / 2)


def test_planted_adaptation_lands_on_its_own_proportion():
    """Congruency code scaled by 1 +- g with incProp, switch code by 1 +- h with
    switchProp (low = +): noise-free, each variable's separation changes with
    its own proportion by 2g|c| (2h|s|) and not at all with the other."""
    rng = np.random.default_rng(5)
    c, s = rng.normal(size=(2, N_ELEC))
    g, h = 0.3, 0.2
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)

    def adaptation(t):
        xc = np.where(t['congruency'] == 'i', 0.5, -0.5)
        xs = np.where(t['switchType'] == 's', 0.5, -0.5)
        low_inc = np.where(t['incongruent_proportion'] == 25, 1.0, -1.0)
        low_sw = np.where(t['switch_proportion'] == 25, 1.0, -1.0)
        return np.outer(xc * low_inc, g * c) + np.outer(xs * low_sw, h * s)

    seps, cons, summary = _run_blocks(_with_terms(df, adaptation), times)
    seps, cons = _own(seps), _own(cons)
    overall = cons[cons['window'] == -1].set_index(['variable', 'contrast'])['estimate']
    assert overall['congruency', 'incProp'] == pytest.approx(2 * g * _norm(c))
    assert overall['congruency', 'switchProp'] == pytest.approx(0, abs=1e-9)
    assert overall['congruency', 'own_minus_other'] == pytest.approx(2 * g * _norm(c))
    assert overall['switchType', 'switchProp'] == pytest.approx(2 * h * _norm(s))
    assert overall['switchType', 'incProp'] == pytest.approx(0, abs=1e-9)
    assert overall['switchType', 'own_minus_other'] == pytest.approx(2 * h * _norm(s))

    low = seps[(seps['variable'] == 'congruency') & (seps['inc_prop'] == 'low')]
    assert low['separation'].to_numpy() == pytest.approx(_norm(c) * (1 + g))
    assert summary['axes']['all']['congruency']['incProp']['estimate'] == pytest.approx(
        2 * g * _norm(c))
    assert summary['n_electrodes'] == N_SUBJ * N_ELEC

    cons_u = _own(_run_blocks(_with_terms(df, adaptation), times, unique=True)[1])
    u = cons_u[cons_u['window'] == -1].set_index(['variable', 'contrast'])['estimate']
    c_all, s_all = np.tile(c, N_SUBJ), np.tile(s, N_SUBJ)
    c_perp = c_all - (c_all @ s_all) / (s_all @ s_all) * s_all
    s_perp = s_all - (c_all @ s_all) / (c_all @ c_all) * c_all
    assert u['congruency', 'incProp'] == pytest.approx(2 * g * np.linalg.norm(c_perp))
    assert u['congruency', 'switchProp'] == pytest.approx(0, abs=1e-9)
    assert u['switchType', 'switchProp'] == pytest.approx(2 * h * np.linalg.norm(s_perp))


def test_unique_axes_ignore_an_adaptation_along_the_shared_direction():
    """A switch-separation change with switchProp planted along c only (the
    switch axis stays s, since blocks are weighted equally): on the raw switch
    axis it shows as 2h (c . s_hat); on the unique axis, orthogonal to c, as 0."""
    rng = np.random.default_rng(7)
    c, s = rng.normal(size=(2, N_ELEC))
    h = 0.3
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)

    def along_c(t):
        xs = np.where(t['switchType'] == 's', 0.5, -0.5)
        low_sw = np.where(t['switch_proportion'] == 25, 1.0, -1.0)
        return np.outer(xs * low_sw, h * c)

    planted = _with_terms(df, along_c)
    c_all, s_all = np.tile(c, N_SUBJ), np.tile(s, N_SUBJ)
    for unique, want in ((False, 2 * h * c_all @ s_all / np.linalg.norm(s_all)), (True, 0.0)):
        cons = _own(_run_blocks(planted, times, unique=unique)[1])
        got = cons[(cons['window'] == -1) & (cons['variable'] == 'switchType')
                   & (cons['contrast'] == 'switchProp')]['estimate'].item()
        assert got == pytest.approx(want, abs=1e-9), unique


def _run_props(df, times, n_splits=4, **kw):
    hg, _ = sfs.window_means(df['hg'].to_numpy(), times, 1, 1)
    axis = ct.axis_window_means(df['hg'].to_numpy(), times, [(times[0], times[-1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ct.proportion_trajectories(df.drop(columns='hg'), hg, axis, ['all'],
                                          n_splits=n_splits, n_boot=kw.pop('n_boot', 50), **kw)


def test_proportion_trajectories_put_each_planted_change_on_its_axis():
    """Noise-free, with partly shared axes: congruency changes with incProp
    along c and with switchProp along s, and switch type changes with
    switchProp along s. In oblique coordinates each change lands on its axis
    only (a projection would leak cos(c, s) of it onto the other), and
    switch type x incProp stays at 0."""
    rng = np.random.default_rng(6)
    c = rng.normal(size=N_ELEC)
    s = 0.6 * c + rng.normal(size=N_ELEC)
    g, k, h = 0.3, 0.25, 0.2
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)

    def adaptation(t):
        xc = np.where(t['congruency'] == 'i', 0.5, -0.5)
        xs = np.where(t['switchType'] == 's', 0.5, -0.5)
        low_inc = np.where(t['incongruent_proportion'] == 25, 1.0, -1.0)
        low_sw = np.where(t['switch_proportion'] == 25, 1.0, -1.0)
        return (np.outer(xc * low_inc, g * c) + np.outer(xc * low_sw, k * s)
                + np.outer(xs * low_sw, h * s))

    traj, changes, summary = _run_props(_with_terms(df, adaptation), times)
    overall = changes[changes['window'] == -1].set_index(['set', 'axis'])['estimate']
    want = {('congruency x incProp', 'congruency'): 2 * g * _norm(c),
            ('congruency x incProp', 'switch'): 0.0,
            ('congruency x switchProp', 'congruency'): 0.0,
            ('congruency x switchProp', 'switch'): 2 * k * _norm(s),
            ('switchType x switchProp', 'congruency'): 0.0,
            ('switchType x switchProp', 'switch'): 2 * h * _norm(s),
            ('switchType x incProp', 'congruency'): 0.0,
            ('switchType x incProp', 'switch'): 0.0}
    for key, value in want.items():
        assert overall[key] == pytest.approx(value, abs=1e-9), key
    i25 = traj[(traj['set'] == 'congruency x incProp') & (traj['condition'] == 'i25')]
    assert i25['congruency'].to_numpy() == pytest.approx(_norm(c) * (1 + g) / 2)
    assert i25['switch'].to_numpy() == pytest.approx(0, abs=1e-9)
    sets = summary['axes']['all']['sets']
    assert sets['congruency x incProp']['own'] and not sets['switchType x incProp']['own']


def test_a_change_along_the_other_axis_shows_on_its_unique_part():
    """Noise-free, partly shared axes, congruency's separation changing with
    incProp along s only: it is 0 on congruency's unique axis and 2k|s_perp| on
    switch's, which is the oblique switch coordinate of the proportion
    trajectories times sin(c, s); on the raw switch axis it is 2k|s|."""
    rng = np.random.default_rng(8)
    c = rng.normal(size=N_ELEC)
    s = 0.6 * c + rng.normal(size=N_ELEC)
    k = 0.25
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.0)

    def along_s(t):
        xc = np.where(t['congruency'] == 'i', 0.5, -0.5)
        low_inc = np.where(t['incongruent_proportion'] == 25, 1.0, -1.0)
        return np.outer(xc * low_inc, k * s)

    planted = _with_terms(df, along_s)
    c_all, s_all = np.tile(c, N_SUBJ), np.tile(s, N_SUBJ)
    s_perp = s_all - (c_all @ s_all) / (c_all @ c_all) * c_all
    sin = np.sqrt(1 - (c_all @ s_all) ** 2 / (c_all @ c_all * (s_all @ s_all)))

    def inc_change(cons, axis):
        return cons[(cons['window'] == -1) & (cons['variable'] == 'congruency')
                    & (cons['axis'] == axis) & (cons['contrast'] == 'incProp')]['estimate'].item()

    raw = _run_blocks(planted, times)[1]
    unique = _run_blocks(planted, times, unique=True)[1]
    assert inc_change(raw, 'switch') == pytest.approx(2 * k * np.linalg.norm(s_all))
    assert inc_change(unique, 'congruency') == pytest.approx(0, abs=1e-9)
    assert inc_change(unique, 'switch') == pytest.approx(2 * k * np.linalg.norm(s_perp))

    changes = _run_props(planted, times)[1]
    oblique = changes[(changes['window'] == -1) & (changes['set'] == 'congruency x incProp')
                      & (changes['axis'] == 'switch')]['estimate'].item()
    assert inc_change(unique, 'switch') == pytest.approx(oblique * sin)


def test_block_null_is_reproducible_and_below_a_planted_adaptation():
    """With a planted switch adaptation (switch code scaled with switchProp)
    and noise: the shuffled-blocks draws are reproducible, have the tables'
    shapes, centre on 0, and give the planted change the smallest p the draws
    allow, while the absent switch x incProp change is not significant."""
    rng = np.random.default_rng(9)
    c, s = rng.normal(size=(2, N_ELEC))
    h = 0.5
    df, times = _trial_long(c, s, n_subj=N_SUBJ, n_elec=N_ELEC, n_time=2, noise=0.5, seed=3)

    def adaptation(t):
        xs = np.where(t['switchType'] == 's', 0.5, -0.5)
        low_sw = np.where(t['switch_proportion'] == 25, 1.0, -1.0)
        return np.outer(xs * low_sw, h * s)

    planted = _with_terms(df, adaptation)
    hg, _ = sfs.window_means(planted['hg'].to_numpy(), times, 1, 1)
    axis = ct.axis_window_means(planted['hg'].to_numpy(), times, [(times[0], times[-1])])
    labels = planted.drop(columns='hg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        null = ct.block_null(labels, hg, axis, n_perm=19, n_splits=2, seed=4)
        again = ct.block_null(labels, hg, axis, n_perm=19, n_splits=2, seed=4)
    assert all(np.array_equal(null[key], again[key]) for key in null)
    assert np.abs(null['contrasts'].mean(0)).max() < 0.5 * 2 * h * _norm(s)

    _, cons, summary = _run_blocks(planted, times, null=null)
    _, changes, prop_summary = _run_props(planted, times, null=null)
    assert null['contrasts'].shape[1:] == (2, 2, 1, len(times) + 1, len(ct.CONTRASTS))
    sw = summary['axes']['all']['switchType']
    assert sw['switchProp']['p'] == pytest.approx(1 / 20)
    assert sw['incProp']['p'] > 0.05
    assert summary['n_null'] == 19 and 'p' in cons
    assert prop_summary['axes']['all']['sets']['switchType x switchProp']['switch']['p'] == \
        pytest.approx(1 / 20)
    assert changes['p'].between(1 / 20, 1).all()


def test_axis_window_means():
    hg = [np.arange(5.0), np.arange(5.0) * 2]
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = ct.axis_window_means(hg, times, [(0.0, 1.0), (0.25, 0.5)])
    assert out.tolist() == [[2.0, 1.5], [4.0, 3.0]]
    with pytest.raises(ValueError, match='no samples'):
        ct.axis_window_means(hg, times, [(2.0, 3.0)])
