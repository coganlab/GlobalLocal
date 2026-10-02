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
    return tab, pd.DataFrame(ps)


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
    # the balance does only when the two fields differ
    assert _near(mixed, 'LWPC − LWPS')['p_greater'] > 0.05
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


def test_height_bands_are_tertiles():
    labels, edges = sfa.height_bands(np.arange(30.0))
    assert labels.value_counts().to_dict() == {'ventral': 10, 'middle': 10, 'dorsal': 10}
    assert len(edges) == 4


def test_section19_runs_every_part_and_skips_without_a_per_split_table(tmp_path):
    tab, ps = _local_world('intermixed', n_subj=8)
    lines, out = sfa.section19(tab, ps, str(tmp_path), n_perm=100, n_boot=50)
    assert {'slope_by_participant', 'participant_corr', 'local_similarity',
            'figure5_height'} <= set(out)
    assert not any('failed' in line for line in lines)

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
