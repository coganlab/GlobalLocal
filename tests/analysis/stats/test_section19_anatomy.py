"""§19 of docs/n4_continuous_anatomy.md (advisor meeting 2026-10-02).

The two pre-specified anatomy tests with participants as the unit, local
similarity, the combined Figure 5, and the RT-adjusted check on Fig. 3. Each
test plants a known world and checks the analysis finds it, and that the
participant-level versions decompose the pooled numbers exactly.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_anatomy as sfa
from src.analysis.stats import stability_flexibility_segregation as sfs


# ---------------------------------------------------------------------------
# planted worlds
# ---------------------------------------------------------------------------
def _gradient_world(n_subj=18, slope=0.012, seed=0, n_splits=6, shared_sd=0.5):
    """LWPC weakens dorsally relative to LWPS by ``slope`` SD/mm; both share a
    component (SD ``shared_sd``), so they correlate across electrodes."""
    rng = np.random.default_rng(seed)
    rows, coords = [], {}
    for s in range(n_subj):
        z0 = rng.uniform(-5, 30)
        for e in range(int(rng.integers(10, 25))):
            z = z0 + rng.normal(0, 15)
            shared = shared_sd * rng.normal()
            el = f'S{s}-e{e}'
            coords[el] = (rng.choice([-1, 1]) * rng.uniform(25, 55), rng.uniform(0, 60), z)
            rows.append(dict(subject=f'S{s}', electrode=el,
                             x=shared + rng.normal() + slope * (20 - z),
                             y=shared + rng.normal() + 0.2,
                             resp=rng.uniform(0.5, 1.5)))
    scores = pd.DataFrame(rows)
    tab = sfa.attach_scores(scores, {}, electrodes_to_coords=coords)
    return tab, sfa._synthetic_per_split(scores, n_splits=n_splits, seed=seed)


def _local_world(kind, n_subj=14, seed=0, n_splits=10, noise=0.7, bipolar=False):
    """Four shafts of six contacts (3.5 mm apart) per participant and a smooth
    field over them. 'intermixed': LWPC and LWPS share the field, so their
    balance has no local structure. 'patchy': independent fields, so it does."""
    rng = np.random.default_rng(seed)
    rows, ps, coords = [], [], {}
    for s in range(n_subj):
        P = []
        for c in rng.uniform(-30, 30, size=(4, 3)) + [40, 30, 20]:
            u = rng.normal(size=3)
            P += [c + k * 3.5 * u / np.linalg.norm(u) for k in range(6)]
        P = np.asarray(P)
        K = np.exp(-(np.linalg.norm(P[:, None] - P[None], axis=-1) / 6.0) ** 2)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(P)))
        f1, f2 = L @ rng.normal(size=len(P)), L @ rng.normal(size=len(P))
        lwpc = f1 + 0.5 * rng.normal(size=len(P))
        lwps = (f1 if kind == 'intermixed' else f2) + 0.5 * rng.normal(size=len(P))
        for e in range(len(P)):
            shaft, k = divmod(e, 6)
            ch = f'L{shaft}{k + 1}-L{shaft}{k + 2}' if bipolar else f'L{shaft}{k + 1}'
            el = f'S{s}-{ch}'
            coords[el] = tuple(P[e])
            rows.append(dict(subject=f'S{s}', electrode=el, x=lwpc[e], y=lwps[e],
                             resp=1.0 + 0.1 * rng.normal()))
            for sp in range(n_splits):
                ps.append(dict(subject=f'S{s}', electrode=el, split=sp,
                               xA=lwpc[e] + noise * rng.normal(), xB=lwpc[e] + noise * rng.normal(),
                               yA=lwps[e] + noise * rng.normal(), yB=lwps[e] + noise * rng.normal()))
    tab = sfa.attach_scores(pd.DataFrame(rows), {}, electrodes_to_coords=coords)
    # each half carries its own independent noise, as disjoint trials shared by
    # all of a participant's electrodes would: a shared split
    return tab, pd.DataFrame(ps).assign(split_scheme='participant')


def _shared_noise_world(seed=0, n_subj=10, n_trials=160, n_splits=30, noise_len=5.0,
                        tau=0.25, sigma=4.0, field=0.0):
    """Like real shared-split data: contacts on shafts 3.5 mm apart, trial noise
    shared by nearby contacts, and halves that are disjoint trial sets of ONE
    dataset (so the full-data noise is fixed across splits and only its split
    into halves varies). True scores have no local structure unless ``field``
    adds a smooth one to LWPC. Reliability ~0.25 with the defaults."""
    rng = np.random.default_rng(seed)
    rows, ps, coords = [], [], {}
    for s in range(n_subj):
        P = []
        for c in rng.uniform(-25, 25, size=(int(rng.integers(2, 5)), 3)) + [40, 30, 20]:
            u = rng.normal(size=3)
            P += [c + k * 3.5 * u / np.linalg.norm(u) for k in range(int(rng.integers(4, 8)))]
        P = np.asarray(P)
        n = len(P)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        L = np.linalg.cholesky(np.exp(-(D / noise_len) ** 2) + 1e-6 * np.eye(n))
        sx, sy = tau * rng.normal(size=n), tau * rng.normal(size=n)
        if field:
            Lf = np.linalg.cholesky(np.exp(-(D / 6.0) ** 2) + 1e-6 * np.eye(n))
            sx = sx + field * (Lf @ rng.normal(size=n))
        ex = sigma * (L @ rng.normal(size=(n, n_trials)))
        ey = sigma * (L @ rng.normal(size=(n, n_trials)))
        for e in range(n):
            coords[f'S{s}-L{e}'] = tuple(P[e])
            rows.append(dict(subject=f'S{s}', electrode=f'S{s}-L{e}', x=sx[e] + ex[e].mean(),
                             y=sy[e] + ey[e].mean(), resp=1.0 + 0.1 * rng.normal()))
        for k in range(n_splits):
            m = np.zeros(n_trials, bool)
            m[rng.permutation(n_trials)[: n_trials // 2]] = True
            xA, xB = sx + ex[:, m].mean(1), sx + ex[:, ~m].mean(1)
            yA, yB = sy + ey[:, m].mean(1), sy + ey[:, ~m].mean(1)
            ps += [(f'S{s}', f'S{s}-L{e}', k, xA[e], xB[e], yA[e], yB[e]) for e in range(n)]
    tab = sfa.attach_scores(pd.DataFrame(rows), {}, electrodes_to_coords=coords)
    ps = pd.DataFrame(ps, columns=['subject', 'electrode', 'split', 'xA', 'xB', 'yA', 'yB'])
    return tab, ps.assign(split_scheme='participant')


# ---------------------------------------------------------------------------
# electrodes as the unit (2026-10-06)
# ---------------------------------------------------------------------------
def test_electrode_slope_is_the_coordinate_test_with_an_interval():
    tab, _ = _gradient_world(slope=0.03)
    coord = sfa.relative_score_coordinate_test(tab, n_perm=500)
    el = sfa.coordinate_slope_by_electrode(tab, n_boot=500, coord_res=coord)
    row = coord['all']['slopes'].set_index('axis').loc['mni_z']

    assert el['slope_per_mm'] == pytest.approx(row['slope_per_mm'])
    assert el['pooled_slope'] == pytest.approx(row['slope_per_mm'], rel=1e-9)
    assert el['p'] == row['p'] and el['p'] < 0.01
    assert el['ci'][0] < el['slope_per_mm'] < el['ci'][1] < 0
    lines = sfa.electrode_slope_lines(el)
    assert 'ELECTRODES AS THE UNIT' in lines[0]


def test_electrode_split_corr_is_the_pooled_test_with_intervals():
    tab, ps = _gradient_world(slope=0.0, shared_sd=1.0)
    resp = tab.set_index('electrode')['resp']
    pooled = sfs.split_resolved_corr(ps, resp, n_perm=200)
    ec = sfs.electrode_split_corr(ps, resp, n_perm=200, n_boot=300)

    assert ec['corr'] == pooled['corr'] and ec['p'] == pooled['p']
    assert ec['ci'][0] < ec['corr'] < ec['ci'][1] and ec['ci'][0] > 0
    for k in ('x', 'y'):
        lo, hi = ec[f'reliability_{k}_ci']
        assert lo < ec[f'reliability_{k}'] < hi
    lo, hi = ec['corr_noise_corrected_ci']
    assert lo < ec['corr_noise_corrected'] < hi and not ec['noise_corrected_note']
    # halves of pure noise: no reliability, so no noise-corrected interval
    noisy = sfa._synthetic_per_split(tab.rename(columns={'lwpc_score': 'x',
                                                         'lwps_score': 'y'}),
                                     n_splits=6, noise=50.0)
    ec = sfs.electrode_split_corr(noisy, resp, n_perm=50, n_boot=200)
    assert ec['noise_corrected_note'].startswith('not estimable')


def test_local_similarity_electrode_se_matches_the_spread_without_local_structure():
    """The electrode-level SE must carry the trial noise neighbours share: over
    null datasets the excess's spread matches it, where a position shuffle's
    spread is ~1.5 times too narrow (§19.3 of the N4 doc). The reliability's SE
    must allow for the per-split standardisation."""
    ex, se, rel, rse = [], [], [], []
    for seed in range(40):
        tab, ps = _shared_noise_world(seed=seed)
        res = sfa.local_similarity(ps, tab, n_perm=50, n_boot=20, seed=seed)
        t = res['table'].set_index(['score', 'bin'])
        ex.append(t.loc[('LWPC', res['bins'][0]), 'excess'])
        se.append(t.loc[('LWPC', res['bins'][0]), 'se_electrode'])
        rel.append(t.loc[('LWPC', 'same electrode'), 'similarity'])
        rse.append(t.loc[('LWPC', 'same electrode'), 'se_electrode'])
    assert 0.7 < np.std(ex, ddof=1) / np.mean(se) < 1.35
    assert 0.6 < np.std(rel, ddof=1) / np.mean(rse) < 1.4
    assert abs(np.mean(ex)) < 2 * np.mean(se) / np.sqrt(len(ex))


def test_local_similarity_electrode_level_finds_a_planted_patch():
    tab, ps = _shared_noise_world(seed=3, n_subj=14, field=0.6)
    res = sfa.local_similarity(ps, tab, n_perm=200, n_boot=200)
    near = _near(res, 'LWPC')
    assert near['p_greater_electrode'] < 0.01
    assert near['excess_lo_electrode'] > 0
    assert _near(res, 'LWPS')['p_greater_electrode'] > 0.01
    c = res['contrasts'].set_index('score').loc['LWPC']
    assert c['p_greater_electrode'] < 0.01
    lines = sfa.local_similarity_lines(res, unit='electrode')
    assert 'ELECTRODES AS THE UNIT' in lines[0]
    assert 'PARTICIPANTS AS THE UNIT' in sfa.local_similarity_lines(res, unit='participant')[0]


def test_local_similarity_electrode_comparison_when_both_maps_are_reliable():
    mixed = sfa.local_similarity(*reversed(_local_world('intermixed')), n_perm=200, n_boot=200)
    cmp = mixed['comparison'].set_index('score').loc['LWPC']
    assert cmp['note_electrode'] == ''
    assert cmp['difference_hi_electrode'] < 0        # the balance shares less than LWPC
    assert cmp['difference_lo_electrode'] < cmp['difference_electrode'] \
        < cmp['difference_hi_electrode']


# ---------------------------------------------------------------------------
# participants as the unit
# ---------------------------------------------------------------------------
def test_participant_slopes_decompose_the_coordinate_test_slope():
    tab, _ = _gradient_world()
    pooled = sfa.relative_score_coordinate_test(tab, n_perm=50)['all']['slopes'] \
        .set_index('axis').loc['mni_z', 'slope_per_mm']
    res = sfa.coordinate_slope_by_participant(tab, n_perm=500, n_boot=200, mixed_model=False)
    per = res['per_participant'].dropna(subset=['slope_per_mm'])

    assert res['pooled_slope'] == pytest.approx(pooled, rel=1e-9)
    assert (per['weight'] * per['slope_per_mm']).sum() / per['weight'].sum() == \
        pytest.approx(pooled, rel=1e-9)
    assert res['weighted']['slope_per_mm'] == pytest.approx(pooled, rel=1e-9)


def test_participant_slope_finds_a_planted_gradient_and_not_a_null_one():
    planted = sfa.coordinate_slope_by_participant(_gradient_world(slope=0.03)[0],
                                                  n_perm=2000, n_boot=200)
    null = sfa.coordinate_slope_by_participant(_gradient_world(slope=0.0, seed=1)[0],
                                               n_perm=2000, n_boot=200, mixed_model=False)

    assert planted['weighted']['slope_per_mm'] < 0
    assert planted['weighted']['p_signflip'] < 0.01
    assert planted['unweighted']['p_t'] < 0.01
    assert planted['mixed']['random_slope']['slope_per_mm'] < 0
    # with predictors centred within participant, the random-intercept model's
    # fixed slope is the within-participant slope, i.e. the pooled one
    assert planted['mixed']['random_intercept']['slope_per_mm'] == \
        pytest.approx(planted['pooled_slope'], rel=1e-3)
    assert null['weighted']['p_signflip'] > 0.01


def test_slope_loso_has_one_fold_per_participant_and_the_full_fit_first():
    tab, _ = _gradient_world(n_subj=8)
    loso = sfa.coordinate_slope_loso(tab, n_perm=50)
    full = sfa.relative_score_coordinate_test(tab, n_perm=50)['all']['slopes'] \
        .set_index('axis').loc['mni_z', 'slope_per_mm']

    assert list(loso['dropped'][:1]) == ['(none)']
    assert len(loso) == tab['subject'].nunique() + 1
    assert loso['slope_per_mm'].iloc[0] == pytest.approx(full)


def test_participant_corr_with_one_participant_is_the_pooled_test():
    tab, ps = _gradient_world(n_subj=1, slope=0.0)
    resp = tab.set_index('electrode')['resp']
    pooled = sfs.split_resolved_corr(ps, resp, n_perm=50)
    part = sfs.participant_split_corr(ps, resp, n_perm=50, n_boot=20)

    assert part['per_participant']['corr'].iloc[0] == pytest.approx(pooled['corr'])
    assert part['per_participant']['reliability_x'].iloc[0] == \
        pytest.approx(pooled['reliability_x'])


def test_participant_corr_finds_shared_electrodes():
    tab, ps = _gradient_world(slope=0.0, shared_sd=1.5)
    res = sfs.participant_split_corr(ps, tab.set_index('electrode')['resp'], n_perm=2000,
                                     n_boot=200)
    assert res['corr_weighted'] > 0
    assert res['p_signflip'] < 0.01
    assert res['ci_weighted'][0] < res['corr_weighted'] < res['ci_weighted'][1]
    assert res['n_positive'] > res['n_participants'] / 2


# ---------------------------------------------------------------------------
# local similarity
# ---------------------------------------------------------------------------
def _near(res, score):
    t = res['table']
    return t[(t['score'] == score) & (t['bin'] == res['bins'][0])].iloc[0]


def test_local_similarity_tells_intermixed_from_patchy():
    mixed = sfa.local_similarity(*reversed(_local_world('intermixed')), n_perm=500, n_boot=200)
    patchy = sfa.local_similarity(*reversed(_local_world('patchy')), n_perm=500, n_boot=200)

    # the positive control: single maps share signal with their neighbours in both
    for res in (mixed, patchy):
        assert _near(res, 'LWPC')['p_greater'] < 0.01
        assert _near(res, 'LWPC')['excess'] > 0
    # the balance does only when the two fields differ. (In the intermixed world
    # it keeps a sliver, ~10 % of LWPC's: the two pooled scale factors differ by
    # sampling, so the shared field does not cancel exactly.)
    assert _near(mixed, 'LWPC − LWPS')['excess'] < 0.25 * _near(mixed, 'LWPC')['excess']
    assert _near(patchy, 'LWPC − LWPS')['excess'] > 0.5 * _near(patchy, 'LWPC')['excess']
    assert _near(patchy, 'LWPC − LWPS')['p_greater'] < 0.01
    cmp = mixed['comparison'].set_index('score').loc['LWPC']
    assert cmp['difference_hi'] < 0          # balance shares less than LWPC does


def test_local_similarity_same_electrode_row_is_the_reliability():
    res = sfa.local_similarity(*reversed(_local_world('intermixed', n_subj=6)), n_perm=50,
                               n_boot=20, remove_gradient=False)
    t = res['table']
    self_rows = t[t['bin'] == 'same electrode']
    assert set(self_rows['score']) == {'LWPC − LWPS', 'LWPC', 'LWPS'}
    assert (self_rows['similarity'] > 0.3).all()          # noise 0.7 against signal ~1
    pairs = t[(t['bin'] != 'same electrode') & (t['score'] == 'LWPC')]['n_pairs']
    assert pairs.sum() == 6 * (24 * 23 // 2)          # every within-participant pair, once


def test_bipolar_pairs_sharing_a_contact_are_dropped():
    assert sfa._channel_poles('D57-LA1-LA2') == {'LA1', 'LA2'}
    assert sfa._channel_poles('D57-LA1') == {'LA1'}
    tab, ps = _local_world('intermixed', n_subj=3, bipolar=True)
    res = sfa.local_similarity(ps, tab, n_perm=20, n_boot=10)
    # neighbours on a shaft share a pole: 5 per shaft, 4 shafts, 3 participants
    assert res['n_pairs_excluded'] == 5 * 4 * 3


# ---------------------------------------------------------------------------
# the combined figure
# ---------------------------------------------------------------------------
def test_figure5_height_shows_the_planted_lean_and_matches_the_test(tmp_path):
    tab, ps = _gradient_world(slope=0.02)
    fig = sfa.figure5_height(tab, str(tmp_path), per_split=ps, n_perm=200, n_boot=200)
    c = fig['centroids'].set_index('band')

    assert list(c.index) == list(sfa.HEIGHT_BANDS)
    assert c.loc['dorsal', 'balance'] < c.loc['ventral', 'balance']    # dorsal leans LWPS
    assert c.loc['dorsal', 'balance_hi'] < c.loc['ventral', 'balance_lo'] or \
        c.loc['dorsal', 'balance'] < 0
    coord = sfa.relative_score_coordinate_test(tab, n_perm=50)['all']['slopes'] \
        .set_index('axis').loc['mni_z', 'slope_per_mm']
    assert fig['check_slope'] == pytest.approx(coord, rel=0.25)
    for name in ('fig5_height.png', 'fig5_height.pdf', 'fig5_height_centroids.csv',
                 'fig5_height_balance.csv', 'fig5_height_points.csv'):
        assert (tmp_path / name).exists()


def test_figure5_height_at_both_levels(tmp_path):
    tab, ps = _gradient_world(slope=0.02)
    el = sfa.figure5_height(tab, str(tmp_path), per_split=ps, n_perm=200, n_boot=200)
    pa = sfa.figure5_height(tab, str(tmp_path), per_split=ps, n_perm=200, n_boot=200,
                            unit='participant')

    for name in ('fig5_height.png', 'fig5_height_participants.png',
                 'fig5_height_participants_centroids.csv', 'fig5_height_participants_balance.csv',
                 'fig5_height_balance_by_participant.csv'):
        assert (tmp_path / name).exists()
    assert 'n_electrodes' in el['balance'] and 'n_participants' in pa['balance']
    assert el['balance']['n_electrodes'].sum() == len(tab)
    # same centroids, different intervals
    assert np.allclose(el['centroids']['balance'], pa['centroids']['balance'])
    assert 'ELECTRODES AS THE UNIT' in el['lines'][0]
    with pytest.raises(ValueError, match='unit'):
        sfa.figure5_height(tab, str(tmp_path), per_split=ps, unit='shaft')


def test_height_bands_are_tertiles():
    labels, edges = sfa.height_bands(np.arange(30.0))
    assert labels.value_counts().to_dict() == {'ventral': 10, 'middle': 10, 'dorsal': 10}
    assert len(edges) == 4


def test_section19_runs_every_part_and_skips_without_a_per_split_table(tmp_path):
    tab, ps = _local_world('intermixed', n_subj=8)
    lines, out = sfa.section19(tab, ps, str(tmp_path), n_perm=100, n_boot=50)
    assert {'slope_by_electrode', 'slope_by_participant', 'electrode_corr', 'participant_corr',
            'local_similarity', 'figure5_height', 'figure5_height_participants'} <= set(out)
    assert not any('failed' in line for line in lines)
    # electrode level first, participant level second, in every part
    text = '\n'.join(lines)
    for part in ('SLOPE', 'SEPARATE-HALF r', 'LOCAL SIMILARITY', 'FIGURE 5 (height)'):
        first = [ln for ln in lines if part in ln and 'AS THE UNIT' in ln and '§19' not in ln]
        assert 'ELECTRODES' in first[0] and 'PARTICIPANTS' in first[1], part
    assert text.index('mni_z SLOPE, ELECTRODES') < text.index('mni_z SLOPE, PARTICIPANTS')
    for name in ('local_similarity.png', 'local_similarity_by_participant.png'):
        assert (tmp_path / name).exists()

    lines, out = sfa.section19(tab, None, str(tmp_path / 'no_split'), n_perm=50, n_boot=20,
                               sections=(1, 2, 3))
    assert 'participant_corr' not in out and 'local_similarity' not in out
    assert any('no per-split table' in line for line in lines)


# ---------------------------------------------------------------------------
# Fig. 3 with the RT-linked part of HG removed
# ---------------------------------------------------------------------------
def test_rt_check_removes_an_adaptation_that_is_only_rt_coupling():
    from src.analysis.stats import stability_flexibility_brain_behavior as sbb
    df, _ = sbb._synthetic_long_df(n_subj=16, seed=0, neural_mean=0.0, neural_sd=0.0,
                                   link=0.0, rt_coupling=1.5)
    chk = sbb.group_adaptation_rt_check(sbb.participant_scores(df, n_splits=2)['electrodes'],
                                        n_perm=500).set_index(['effect', 'variant'])

    # RT coupling alone makes a "neural LWPC" with behavior's sign ...
    assert chk.loc[('LWPC', 'raw'), 'participant_mean'] > 0.2
    assert chk.loc[('LWPC', 'raw'), 'p_t'] < 0.01
    # ... and the adjustment takes it away (seeds 0-3 retain -5 % to 25 %)
    assert abs(chk.loc[('LWPC', 'rtadj'), 'participant_mean']) < 0.1
    assert chk.loc[('LWPC', 'rtadj'), 'retained'] < 0.4


def test_rt_check_keeps_an_adaptation_without_rt_coupling():
    from src.analysis.stats import stability_flexibility_brain_behavior as sbb
    df, _ = sbb._synthetic_long_df(n_subj=12, seed=1, neural_mean=0.3, rt_coupling=0.0)
    chk = sbb.group_adaptation_rt_check(sbb.participant_scores(df, n_splits=2)['electrodes'],
                                        n_perm=500).set_index(['effect', 'variant'])
    for eff in ('LWPC', 'LWPS'):
        assert chk.loc[(eff, 'rtadj'), 'participant_mean'] > 0.15
        assert chk.loc[(eff, 'rtadj'), 'retained'] == pytest.approx(1.0, abs=0.2)
    lines = sbb.group_adaptation_rt_lines(chk.reset_index())
    assert any('retained after RT adjustment' in line for line in lines)


# ---------------------------------------------------------------------------
# split schemes and the overlap controls (2026-10-05)
# ---------------------------------------------------------------------------
def _trial_world(n_subj=4, n_shafts=2, per_shaft=6, tpb=64, noise_len=5.0, seed=0):
    """Trial-level long table: contacts on shafts, trial noise correlated between
    neighbouring contacts, each electrode's LWPC and LWPS drawn independently.
    No score has any local structure."""
    from src.analysis.stats.stability_flexibility_brain_behavior import _BLOCK_PROPORTION_MAP
    rng = np.random.default_rng(seed)
    frames, coords = [], {}
    for s in range(n_subj):
        P = []
        for c in rng.uniform(-25, 25, size=(n_shafts, 3)) + [40, 30, 20]:
            u = rng.normal(size=3)
            P += [c + k * 3.5 * u / np.linalg.norm(u) for k in range(per_shaft)]
        P = np.asarray(P)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        L = np.linalg.cholesky(np.exp(-(D / noise_len) ** 2) + 1e-6 * np.eye(len(P)))
        cols = dict(congruency=[], switchType=[], incongruent_proportion=[],
                    switch_proportion=[])
        for props in _BLOCK_PROPORTION_MAP.values():
            n_i = round(tpb * props['incongruent_proportion'] / 100)
            n_s = round(tpb * props['switch_proportion'] / 100)
            cols['congruency'] += list(rng.permutation(['i'] * n_i + ['c'] * (tpb - n_i)))
            cols['switchType'] += list(rng.permutation(['s'] * n_s + ['r'] * (tpb - n_s)))
            cols['incongruent_proportion'] += [props['incongruent_proportion']] * tpb
            cols['switch_proportion'] += [props['switch_proportion']] * tpb
        n_t = len(cols['congruency'])
        inc = np.array(cols['congruency']) == 'i'
        sw = np.array(cols['switchType']) == 's'
        hc = np.where(np.array(cols['incongruent_proportion']) == 25.0, 0.5, -0.5)
        hs = np.where(np.array(cols['switch_proportion']) == 25.0, 0.5, -0.5)
        noise = L @ rng.normal(size=(len(P), n_t))
        lw = rng.normal(0.1, 0.15, size=(len(P), 2))
        for e in range(len(P)):
            el = f'S{s}-L{e}'
            coords[el] = tuple(P[e])
            hg = 0.5 + inc * (0.3 + hc * lw[e, 0]) + sw * (0.3 + hs * lw[e, 1]) + noise[e]
            frames.append(pd.DataFrame(dict(subject=f'S{s}', electrode=el,
                                            trial=np.arange(n_t), hg=hg, **cols)))
    return pd.concat(frames, ignore_index=True), coords


def test_shared_split_gives_a_participants_electrodes_the_same_balanced_halves():
    df, _ = _trial_world(n_subj=2)
    contrasts = sfs.finalize_contrasts(df, sfs.resolve_contrasts('proportion'))
    work = sfs._canonical_labels(df, contrasts)
    strata = sfs._strata_columns(contrasts)
    halves = sfs._shared_trial_halves(work, strata, 4, np.random.default_rng(0))

    assert set(halves) == {'S0', 'S1'}
    trials = work[work['subject'] == 'S0'].drop_duplicates('trial').set_index('trial')
    for _, cell in trials.groupby(strata):
        assert ((halves['S0'].loc[cell.index] == 0).sum(0) == len(cell) // 2).all()
    shared = sfs.compute_sensitivities_per_split(df, n_splits=2, contrast_mode='proportion',
                                                 shared_split=True)
    own = sfs.compute_sensitivities_per_split(df, n_splits=2, contrast_mode='proportion')
    assert sfa.is_shared_split(shared) and not sfa.is_shared_split(own)


def test_per_electrode_splits_turn_shared_trial_noise_into_local_similarity():
    """The 2026-10-05 finding: with no local structure planted, a split drawn per
    electrode makes neighbours look similar; a shared split does not."""
    df, coords = _trial_world(n_subj=5, seed=3)
    near = {}
    for shared in (False, True):
        ps = sfs.compute_sensitivities_per_split(df, n_splits=8, seed=1,
                                                 contrast_mode='proportion',
                                                 shared_split=shared)
        tab = sfa.attach_scores(sfs.add_responsiveness(sfs.average_over_splits(ps), df), {},
                                electrodes_to_coords=coords)
        res = sfa.local_similarity(ps, tab, n_perm=200, n_boot=50, require_shared_split=False)
        near[shared] = _near(res, 'LWPC')['excess']
    assert near[False] > near[True] + 0.03


def test_local_similarity_refuses_a_per_electrode_split():
    tab, ps = _local_world('intermixed', n_subj=3)
    with pytest.raises(ValueError, match='one trial split per participant'):
        sfa.local_similarity(ps.drop(columns='split_scheme'), tab, n_perm=10, n_boot=5)


def test_section19_skips_local_similarity_on_a_per_electrode_split(tmp_path):
    tab, ps = _local_world('intermixed', n_subj=6)
    lines, out = sfa.section19(tab, ps.drop(columns='split_scheme'), str(tmp_path),
                               n_perm=50, n_boot=20, sections=(3,))
    assert 'local_similarity' not in out
    assert any('local similarity: skipped' in line for line in lines)
    lines, out = sfa.section19(tab, ps.drop(columns='split_scheme'), str(tmp_path / 's'),
                               n_perm=50, n_boot=20, sections=(3,), per_split_shared=ps)
    assert 'local_similarity' in out and 'reliability_by_split_scheme' in out
    assert 'overlap_shared' in out and 'reliability_x_ci' in out['overlap_shared']
    assert any('the rescored table' in line for line in lines)


def test_local_similarity_does_not_divide_by_an_unreliable_map():
    tab, ps = _local_world('intermixed', n_subj=6, noise=25.0)      # halves ~pure noise
    res = sfa.local_similarity(ps, tab, n_perm=50, n_boot=100)
    assert res['notes']
    assert res['comparison'][['balance_relative', 'score_relative']].isna().all().all()
    near = res['table'][res['table']['bin'] == res['bins'][0]]
    assert 'relative' not in near or near['relative'].isna().all()


def _confound_world(kind, n_subj=14, seed=0, n_splits=10, noise=0.5):
    """LWPC and LWPS correlate across electrodes ONLY through a confound:
    'base', both scale with an electrode's base effects, which share a factor;
    'rt', both carry the electrode's RT coupling times the behavioral effect."""
    rng = np.random.default_rng(seed)
    rows, ps, coords, rt = [], [], {}, []
    for s in range(n_subj):
        for e in range(int(rng.integers(12, 25))):
            el = f'S{s}-e{e}'
            f, c = rng.normal(), rng.normal()
            cong, switch = f + 0.5 * rng.normal(), f + 0.5 * rng.normal()
            if kind == 'base':
                lwpc, lwps = cong + rng.normal(), switch + rng.normal()
            else:
                lwpc, lwps = 0.8 * c + rng.normal(), 0.8 * c + rng.normal()
            coords[el] = tuple(rng.normal(0, 20, 3) + [40, 30, 20])
            rows.append(dict(subject=f'S{s}', electrode=el, x=lwpc, y=lwps, mx=cong,
                             my=switch, resp=1.0 + 0.1 * rng.normal()))
            rt.append(dict(electrode=el, rt_r=c))
            for k in range(n_splits):
                ps.append(dict(subject=f'S{s}', electrode=el, split=k,
                               xA=lwpc + noise * rng.normal(), xB=lwpc + noise * rng.normal(),
                               yA=lwps + noise * rng.normal(), yB=lwps + noise * rng.normal(),
                               mxA=cong + 0.1 * rng.normal(), mxB=cong + 0.1 * rng.normal(),
                               myA=switch + 0.1 * rng.normal(),
                               myB=switch + 0.1 * rng.normal()))
    tab = sfa.attach_scores(pd.DataFrame(rows), {}, electrodes_to_coords=coords)
    return tab, pd.DataFrame(ps), pd.DataFrame(rt)


def test_overlap_controls_remove_the_confound_that_made_the_overlap():
    for kind, row in (('base', '+ base effects, same half'), ('rt', '+ RT coupling')):
        tab, ps, rt = _confound_world(kind)
        table, loso = sfa.overlap_controls(tab, ps, rt_coupling=rt, n_perm=300)
        t = table.set_index('control')
        assert t.loc['pre-specified', 'corr'] > 0.15
        assert t.loc['pre-specified', 'p'] < 0.01
        assert abs(t.loc[row, 'corr']) < 0.08                 # the right control removes it
        assert t.loc['+ MNI coordinates', 'corr'] > 0.15       # a wrong one does not
        assert len(loso) == tab['subject'].nunique()
    lines = sfa.overlap_control_lines(table, loso)
    assert any('leave one participant out' in line for line in lines)


def test_same_half_covariates_keep_the_halves_apart():
    """Partialling the same half's base effects must not touch an overlap the
    base effects do not carry."""
    tab, ps, _ = _confound_world('rt')
    resp = tab.set_index('electrode')['resp']
    plain = sfs.split_resolved_corr(ps, resp, n_perm=100)['corr']
    part = sfs.split_resolved_corr(ps, resp, n_perm=100,
                                   half_covariates=sfa._SAME_HALF_BASE)['corr']
    assert part == pytest.approx(plain, abs=0.05)


def test_section19_script_skips_only_local_similarity_without_trial_ids(tmp_path, capsys):
    """A long table assembled before `trial` existed: section 3 is skipped with
    the ways to get trial ids, and section 5 still runs."""
    from dcc_scripts.stats import n4_section19_followups as script
    df, coords = _trial_world(n_subj=4, seed=1)
    ps = sfs.compute_sensitivities_per_split(df, n_splits=4, contrast_mode='proportion',
                                             main_effects=True)
    tab = sfa.attach_scores(sfs.add_responsiveness(sfs.average_over_splits(ps), df), {},
                            electrodes_to_coords=coords)
    tab.to_csv(tmp_path / 'scores_with_anatomy.csv', index=False)
    ps.to_csv(tmp_path / 'per_split.csv', index=False)
    df.drop(columns='trial').to_csv(tmp_path / 'long_df.csv', index=False)

    script.main(['--anatomy-dir', str(tmp_path), '--long-df', str(tmp_path / 'long_df.csv'),
                 '--sections', '3,5', '--n-perm', '50', '--n-boot', '20'])
    out = capsys.readouterr().out
    assert 'has no `trial` column' in out and 'SCATTER_ONLY=1' in out
    assert 'local similarity: skipped' in out
    assert 'CANDIDATE CONFOUND REMOVED' in out


def test_section19_script_labels_an_rt_adjusted_long_table(tmp_path, capsys):
    from dcc_scripts.stats import n4_section19_followups as script
    df, coords = _trial_world(n_subj=4, seed=2)
    ps = sfs.compute_sensitivities_per_split(df, n_splits=4, contrast_mode='proportion',
                                             main_effects=True)
    tab = sfa.attach_scores(sfs.add_responsiveness(sfs.average_over_splits(ps), df), {},
                            electrodes_to_coords=coords)
    tab.to_csv(tmp_path / 'scores_with_anatomy.csv', index=False)
    ps.to_csv(tmp_path / 'per_split.csv', index=False)
    seg = tmp_path / 'seg_rt_adjusted'
    seg.mkdir()
    df.to_csv(seg / 'long_df.csv', index=False)
    pd.DataFrame({'electrode': tab['electrode'], 'rt_slope': 0.0, 'rt_r': 0.1}).to_csv(
        seg / 'rt_adjustment_slopes.csv', index=False)

    script.main(['--anatomy-dir', str(tmp_path), '--long-df', str(seg / 'long_df.csv'),
                 '--shared-n-splits', '4', '--sections', '3', '--n-perm', '50',
                 '--n-boot', '20'])
    out = capsys.readouterr().out
    assert 'RT_ADJUST_HG=1 run' in out and 'RT-adjusted HG' in out
    # the rescored table's reliabilities with electrode-level intervals, for
    # the adaptations and for the base effects
    assert 'LWPC–LWPS SEPARATE-HALF r, ELECTRODES AS THE UNIT (the rescored table' in out
    assert 'congruency–switch SEPARATE-HALF r, ELECTRODES AS THE UNIT' in out
    assert 'congruency − switch' in out
    assert (tmp_path / 'section19_rt_adjusted' / 'summary_section19.txt').exists()
