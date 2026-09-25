"""Main effects as the reference for the delta tilt (docs/closing_figure_plan.md).

Scoring: congruency/switch main effects come from the proportion run's own
cells, equal weight over the proportion levels, on the same halves as
LWPC/LWPS. Linking: the two planted worlds the plan asks for -- a tilt
inherited from the main effects must shrink with dm as a covariate and be
tracked by dm; a tilt of its own must do neither.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_anatomy as sfa
from src.analysis.stats import stability_flexibility_segregation as sfs


def _block_offset_df(n_elec=40, seed=0):
    """A true congruency effect (d = 0.5) plus an HG offset in 75%-incongruent
    blocks, at the plan's per-participant cell counts. The offset is no
    congruency effect, but a trial-count-weighted congruency score picks up
    about half of it."""
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_elec):
        for cong, prop, n in (('i', 25.0, 42), ('c', 25.0, 152),
                              ('i', 75.0, 137), ('c', 75.0, 50)):
            frames.append(pd.DataFrame(dict(
                subject='S1', electrode=f'S1-e{e}', congruency=cong,
                switchType=rng.choice(['s', 'r'], n), incongruent_proportion=prop,
                switch_proportion=rng.choice([25.0, 75.0], n),
                hg=1.0 * (prop == 75.0) + 0.5 * (cong == 'i') + rng.normal(0, 1, n))))
    return pd.concat(frames, ignore_index=True)


def test_main_effects_share_the_halves_and_leave_lwpc_lwps_unchanged():
    df = _block_offset_df(n_elec=4)
    with_main = sfs.compute_sensitivities_per_split(df, n_splits=5, contrast_mode='proportion',
                                                    main_effects=True)
    without = sfs.compute_sensitivities_per_split(df, n_splits=5, contrast_mode='proportion')

    pd.testing.assert_frame_equal(with_main[list(without.columns)], without)
    assert with_main[list(sfs.MAIN_EFFECT_COLS)].notna().all().all()
    assert {'mx', 'my'} <= set(sfs.average_over_splits(with_main).columns)
    with pytest.raises(ValueError, match='proportion'):
        sfs.compute_sensitivities_per_split(df, n_splits=1, contrast_mode='condition',
                                            main_effects=True)


def test_main_effect_is_balanced_over_proportion_so_block_offsets_stay_out():
    df = _block_offset_df()
    avg = sfs.average_over_splits(sfs.compute_sensitivities_per_split(
        df, n_splits=2, contrast_mode='proportion', main_effects=True))

    assert abs(avg['mx'].mean() - 0.5) < 0.1      # the planted congruency effect
    assert abs(avg['x'].mean()) < 0.1             # LWPC untouched by the offset
    # a condition-mode score is trial-count weighted over proportion and absorbs
    # about half the offset -- the trap the proportion-run scores avoid
    assert sfs.naive_sensitivities(df, contrast_mode='condition')['x'].mean() > 0.8


def test_attach_scores_adds_dm_only_with_main_effects():
    scores = pd.DataFrame({'subject': ['A', 'A', 'B'], 'electrode': ['A-1', 'A-2', 'B-1'],
                           'x': [1.0, 0.0, -1.0], 'y': [0.5, 0.5, 0.0],
                           'mx': [2.0, 1.0, 0.0], 'my': [0.0, 3.0, 1.0]})
    tab = sfa.attach_scores(scores, {})

    assert np.allclose(tab['cong_s'], scores['mx'] / scores['mx'].std(ddof=1))
    assert np.allclose(tab['dm'], tab['cong_s'] - tab['switch_s'])
    assert np.allclose(tab['delta'], tab['lwpc_s'] - tab['lwps_s'])
    assert 'dm' not in sfa.attach_scores(scores.drop(columns=['mx', 'my']), {})


def test_split_resolved_corr_covariates_remove_a_shared_gradient():
    """Two maps that share only a gradient correlate; partialling it removes that."""
    rng = np.random.default_rng(0)
    elec = pd.DataFrame({'subject': np.repeat([f'S{i}' for i in range(12)], 30),
                         'electrode': [f'e{i}' for i in range(360)],
                         'z': rng.uniform(-20, 70, 360)})
    rows = []
    for k in range(10):
        noise = rng.normal(0, 1, (4, 360))
        rows.append(elec.assign(split=k, xA=0.03 * elec['z'] + noise[0],
                                xB=0.03 * elec['z'] + noise[1],
                                yA=0.03 * elec['z'] + noise[2],
                                yB=0.03 * elec['z'] + noise[3]))
    per_split = pd.concat(rows).drop(columns='z')
    resp = pd.Series(1.0 + rng.uniform(0, 1, 360), index=elec['electrode'])

    shared = sfs.split_resolved_corr(per_split, resp, n_perm=200)
    partial = sfs.split_resolved_corr(per_split, resp, n_perm=200,
                                      covariates=elec.set_index('electrode')[['z']])
    assert shared['corr'] > 0.3 and shared['p'] < 0.05
    assert abs(partial['corr']) < 0.05 and partial['p'] > 0.05


@pytest.fixture(scope='module')
def worlds():
    out = {}
    for world in ('inherited', 'independent'):
        scores, e2r, e2a, e2c = sfa._synthetic_scores(n_subj=14, seed=0, main_effects=world)
        tab = sfa.attach_scores(scores, e2r, electrodes_to_anat=e2a, electrodes_to_coords=e2c)
        out[world] = (tab, sfa._synthetic_per_split(scores, n_splits=20, seed=0))
    return out


def test_test2_shrinks_an_inherited_tilt_and_leaves_an_independent_one(worlds):
    # the planted layout is anterior/posterior, so the tilt is on y here
    inh = sfa.tilt_with_main_effect_covariate(*worlds['inherited'], axis='mni_y', n_perm=2000)
    ind = sfa.tilt_with_main_effect_covariate(*worlds['independent'], axis='mni_y', n_perm=2000)
    ti, tn = inh['table'].set_index('fit'), ind['table'].set_index('fit')

    assert ti.loc['delta', 'p'] < 0.05 and tn.loc['delta', 'p'] < 0.05  # a tilt to explain
    assert ti.loc['dm', 'p'] < 0.05 and tn.loc['dm', 'p'] > 0.05        # main effects tilt?
    assert min(inh['shrinkage'].values()) > 0.5                        # carried by dm
    assert max(abs(v) for v in ind['shrinkage'].values()) < 0.2        # survives dm
    assert tn.loc['delta + dm', 'p'] < 0.05


def test_test1_finds_tracking_only_when_inherited(worlds):
    inh = sfa.delta_tracking_test(*worlds['inherited'], n_perm=500).set_index('comparison')
    ind = sfa.delta_tracking_test(*worlds['independent'], n_perm=500).set_index('comparison')

    assert inh.loc['dm vs delta', 'corr'] > 0.2 and inh.loc['dm vs delta', 'p'] < 0.05
    assert inh.loc['dm vs delta, + MNI covariates', 'corr'] > 0.1
    assert abs(ind.loc['dm vs delta', 'corr']) < 0.1   # p is ~uniform here; r is small
    matched = inh.loc[['congruency vs LWPC', 'switch vs LWPS'], 'corr']
    crossed = inh.loc[['congruency vs LWPS (crossed)', 'switch vs LWPC (crossed)'], 'corr']
    assert matched.min() > crossed.max()
