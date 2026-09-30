"""Tests for A6 level (1): one neural and one behavioral LWPC / LWPS per participant.

`participant_scores` averages each participant's per-electrode d and scores
behavior on the same trials; `participant_brain_behavior` correlates them. The
tests pin the four properties the across-participant correlation rests on:

  * the scores ARE the segregation module's per-electrode d and `_dod_rt`'s
    behavioral d-o-d, just vectorized and averaged;
  * `rt_adjust_hg` removes a brain-behavior correlation manufactured by HG that
    tracks RT, which the specificity checks (matched vs cross, the joint
    regression) do not catch, and leaves effects unrelated to RT alone;
  * the reliability, taken with ONE trial split per participant, is ~0 when
    participants do not differ, where splitting each electrode separately
    reports a sizeable one from common-mode noise alone;
  * a planted link, and its reliability ceiling, are recovered.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..', '..', '..')))

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats import stability_flexibility_brain_behavior as sbb
from dcc_scripts.stats import stability_flexibility_brain_behavior_dcc as dcc


@pytest.fixture(scope='module')
def planted():
    """A planted link with RT coupling on top: every quantity is non-trivial."""
    df, truth = sbb._synthetic_long_df(n_subj=12, seed=0, link=0.6, rt_coupling=0.3)
    return df, truth, sbb.participant_scores(df, n_splits=5)


# ---------------------------------------------------------------------------
# the scores are the segregation / behavioral scores, averaged
# ---------------------------------------------------------------------------
def test_electrode_scores_match_the_segregation_scores(planted):
    df, _, ps = planted
    e = ps['electrodes']
    ref = sfs.naive_sensitivities(df, contrast_mode='proportion')
    m = e.merge(ref, on=['subject', 'electrode'])
    np.testing.assert_allclose(m['lwpc_neural'], m['x'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(m['lwps_neural'], m['y'], rtol=0, atol=1e-10)

    adj, _ = sbb.rt_adjust_hg(df)
    ref_adj = sfs.naive_sensitivities(adj, contrast_mode='proportion')
    m = e.merge(ref_adj, on=['subject', 'electrode'])
    np.testing.assert_allclose(m['lwpc_neural_rtadj'], m['x'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(m['lwps_neural_rtadj'], m['y'], rtol=0, atol=1e-10)


def test_participant_score_is_the_mean_of_its_electrodes(planted):
    _, _, ps = planted
    e = ps['electrodes']
    expected = e[e['usable']].groupby('subject')['lwpc_neural'].mean()
    got = ps['scores'].set_index('subject')['lwpc_neural']
    np.testing.assert_allclose(got.loc[expected.index], expected, rtol=0, atol=1e-12)


def test_abs_and_positive_only_summaries(planted):
    """`_abs` is the mean |d| of the usable electrodes, `_pos` the mean d over the
    usable ones with d > 0 (NaN below `min_elec` of them), `_pos_n` their count."""
    _, _, ps = planted
    e = ps['electrodes'][ps['electrodes']['usable']]
    s = ps['scores'].set_index('subject')
    for col in ('lwpc_neural', 'lwps_neural_rtadj'):
        np.testing.assert_allclose(
            s[f'{col}_abs'], e[col].abs().groupby(e['subject']).mean().loc[s.index],
            rtol=0, atol=1e-12)
        pos = e[e[col] > 0].groupby('subject')[col]
        n_pos = pos.size().reindex(s.index, fill_value=0)
        expected = pos.mean().reindex(s.index).where(n_pos >= ps['min_elec'])
        np.testing.assert_allclose(s[f'{col}_pos'], expected, rtol=0, atol=1e-12)
        assert (s[f'{col}_pos_n'] == n_pos).all()
    for variant in sbb.NEURAL_VARIANTS:
        res = sbb.participant_brain_behavior(ps, variant=variant)
        col = res['neural_columns']['lwpc']
        assert col == f"lwpc_neural{sbb.NEURAL_VARIANTS[variant]}"
        assert np.isfinite(res['reliability_neural_lwpc'])
    with pytest.raises(ValueError, match='variant'):
        sbb.participant_brain_behavior(ps, variant='nonsense')


def test_participants_below_min_elec_keep_behavior_but_lose_the_neural_score():
    df, _ = sbb._synthetic_long_df(n_subj=6, n_elec=(2, 6), seed=3)
    ps = sbb.participant_scores(df, n_splits=2, min_elec=4)
    s = ps['scores']
    few = s['n_elec'] < 4
    assert few.any() and (~few).any()
    assert s.loc[few, 'lwpc_neural'].isna().all()
    assert s.loc[~few, 'lwpc_neural'].notna().all()
    assert s['lwpc_behav'].notna().all()


def test_behavior_matches_dod_rt_and_the_dcc_helper(planted):
    df, _, ps = planted
    trials = df.drop_duplicates(['subject', 'trial'])
    ref = sbb.behavioral_lwpc_lwps_magnitudes(trials, rt_col='rt').set_index('subject')
    got = ps['scores'].set_index('subject')
    for eff in ('lwpc', 'lwps'):
        np.testing.assert_allclose(got.loc[ref.index, f'{eff}_behav'], ref[eff],
                                   rtol=0, atol=1e-9)
    via_dcc = dcc.behavior_from_long_df(df).set_index('subject')
    np.testing.assert_allclose(via_dcc.loc[ref.index, 'lwpc'], ref['lwpc'])


def test_behavior_scores_track_the_planted_effects(planted):
    _, truth, ps = planted
    m = ps['scores'].merge(truth, on='subject')
    for eff in ('lwpc', 'lwps'):
        assert np.corrcoef(m[f'{eff}_behav'], m[f'{eff}_behav_true'])[0, 1] > 0.6


# ---------------------------------------------------------------------------
# the RT confound
# ---------------------------------------------------------------------------
def test_rt_adjustment_removes_an_rt_linked_component_exactly():
    """The same data with and without HG that tracks RT (the generator draws the
    same numbers either way). The coupling pushes the raw electrode scores towards
    behavior's own (positive) sign; the adjusted scores are identical in both,
    because the added term is linear in RT and the within-cell slope absorbs it."""
    kw = dict(n_subj=16, seed=1, neural_mean=0.0, neural_sd=0.0)
    coupled, _ = sbb._synthetic_long_df(rt_coupling=0.4, **kw)
    clean, _ = sbb._synthetic_long_df(rt_coupling=0.0, **kw)
    e1 = sbb.participant_scores(coupled, n_splits=2)['electrodes']
    e0 = sbb.participant_scores(clean, n_splits=2)['electrodes']
    for eff in ('lwpc', 'lwps'):
        assert (e1[f'{eff}_neural'] - e0[f'{eff}_neural']).mean() > 0.08
        np.testing.assert_allclose(e1[f'{eff}_neural_rtadj'], e0[f'{eff}_neural_rtadj'],
                                   rtol=0, atol=1e-8)
    assert e1['rt_r'].median() > 0.2 > abs(e0['rt_r'].median())


def test_rt_adjustment_leaves_effects_unrelated_to_rt_alone():
    df, _ = sbb._synthetic_long_df(n_subj=16, seed=2, neural_mean=0.3,
                                   rt_coupling=0.0)
    e = sbb.participant_scores(df, n_splits=2)['electrodes']
    for eff in ('lwpc', 'lwps'):
        assert e[f'{eff}_neural'].mean() > 0.2
        assert abs(e[f'{eff}_neural_rtadj'].mean() - e[f'{eff}_neural'].mean()) < 0.02


def test_rt_confound_passes_the_specificity_checks_but_not_the_adjustment():
    """With no brain-behavior link, RT coupling alone produces a matched
    correlation that looks SPECIFIC (joint beta: matched large, cross ~0). Only
    the RT adjustment removes it; the disjoint-half r flags part of it."""
    df, _ = sbb._synthetic_long_df(n_subj=48, seed=5, neural_mean=0.0,
                                   neural_sd=0.0, rt_coupling=0.4)
    ps = sbb.participant_scores(df, n_splits=10)
    raw = sbb.participant_brain_behavior(ps, variant='raw')
    adj = sbb.participant_brain_behavior(ps, variant='rtadj')
    for eff in ('lwpc', 'lwps'):
        assert raw[f'corr_{eff}'] > 0.4 and raw[f'p_{eff}'] < 0.05
        assert raw[f'joint_{eff}']['beta_matched'] > raw[f'joint_{eff}']['beta_cross'] + 0.3
        assert raw[f'corr_{eff}_same_half'] > raw[f'corr_{eff}_disjoint_half'] + 0.15
        assert abs(adj[f'corr_{eff}']) < 0.3


def test_rt_adjust_hg_is_the_ancova_correction():
    """Adjusted cell-mean contrast = raw contrast - slope x the RT contrast."""
    df, _ = sbb._synthetic_long_df(n_subj=2, n_elec=(1, 1), seed=4, rt_coupling=0.5)
    adj, slopes = sbb.rt_adjust_hg(df)
    for elec, g in df.groupby('electrode'):
        a = adj.loc[g.index]
        b = float(slopes.set_index('electrode').loc[elec, 'rt_slope'])
        low = g['incongruent_proportion'] == 25.0
        inc = g['congruency'] == 'i'

        def dod(v):
            return ((v[inc & low].mean() - v[~inc & low].mean())
                    - (v[inc & ~low].mean() - v[~inc & ~low].mean()))

        assert dod(a['hg']) == pytest.approx(dod(g['hg']) - b * dod(g['rt']), abs=1e-10)


def test_rt_adjust_hg_rejects_time_courses():
    df = pd.DataFrame(dict(electrode=['e'], hg=[np.zeros(3)], rt=[900.0],
                           congruency=['i'], switchType=['s'],
                           incongruent_proportion=[25.0], switch_proportion=[25.0]))
    with pytest.raises(ValueError, match='scalar'):
        sbb.rt_adjust_hg(df)


# ---------------------------------------------------------------------------
# reliability: one split per participant
# ---------------------------------------------------------------------------
def _per_electrode_split_reliability(df, n_splits=10, seed=0):
    """What the reliability would be if each ELECTRODE were split on its own."""
    d = df.reset_index(drop=True)
    codes = sbb._cell_codes(d)
    ec, _ = sbb._codes(d['electrode'].to_numpy())
    sc, sn = sbb._codes(d['subject'].to_numpy())
    n_e, n_p = int(ec.max()) + 1, len(sn)
    subj_of = np.zeros(n_e, dtype=int)
    subj_of[ec] = sc
    strata = d.groupby(['electrode', *sbb._DESIGN_CELLS], sort=False).ngroup().to_numpy()
    sizes = np.bincount(strata)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        half = sbb._shared_half(strata, sizes, rng)
        ab = sbb._dod_scores(d['hg'].to_numpy(float), ec * 2 + half, codes['lwpc'],
                             n_e * 2).reshape(n_e, 2)
        ok = np.isfinite(ab).all(1)
        cnt = np.bincount(subj_of[ok], minlength=n_p)
        m = [np.bincount(subj_of[ok], weights=ab[ok, j], minlength=n_p) / cnt
             for j in (0, 1)]
        rs.append(np.corrcoef(m[0], m[1])[0, 1])
    return float(np.mean(rs))


def test_shared_split_reliability_is_unbiased_where_per_electrode_splits_are_not():
    """No true between-participant differences, noise partly common to a
    participant's electrodes: one split per participant gives ~0, one split per
    electrode reports a sizeable 'reliability' from the common noise alone."""
    df, _ = sbb._synthetic_long_df(n_subj=30, n_elec=(10, 10), seed=2,
                                   neural_sd=0.0, rt_coupling=0.0, common_noise=0.4)
    rel = sbb.participant_scores(df, n_splits=10)['reliability'].set_index('score')
    shared = rel.loc['lwpc_neural', 'r_half']
    assert abs(shared) < 0.3
    assert _per_electrode_split_reliability(df) > shared + 0.3


def test_planted_link_and_its_ceiling_are_recovered():
    df, _ = sbb._synthetic_long_df(n_subj=40, seed=3, neural_sd=0.5, link=0.8,
                                   rt_coupling=0.0)
    ps = sbb.participant_scores(df, n_splits=10)
    rel = ps['reliability'].set_index('score')['reliability']
    assert rel['lwpc_neural'] > 0.8 and rel['lwps_neural'] > 0.8
    res = sbb.participant_brain_behavior(ps, variant='rtadj')
    for eff in ('lwpc', 'lwps'):
        assert res[f'corr_{eff}'] > 0.4 and res[f'p_{eff}'] < 0.05
        assert res[f'joint_{eff}']['beta_matched'] > res[f'joint_{eff}']['beta_cross']
        assert 0 < res[f'ceiling_{eff}'] <= 1
        lo, hi = res[f'ci_{eff}']
        assert lo < res[f'corr_{eff}'] < hi
    assert res['r_crit'] == pytest.approx(0.312, abs=0.005)      # n = 40


# ---------------------------------------------------------------------------
# inputs that lack columns
# ---------------------------------------------------------------------------
def test_without_rt_the_neural_scores_survive_and_behavior_is_skipped():
    df, truth = sbb._synthetic_long_df(n_subj=6, seed=0)
    ps = sbb.participant_scores(df.drop(columns='rt'), n_splits=2)
    assert 'lwpc_neural' in ps['scores'] and 'lwpc_behav' not in ps['scores']
    assert 'lwpc_neural_rtadj' not in ps['scores']
    assert any('reaction times' in n for n in ps['notes'])
    with pytest.raises(KeyError):
        sbb.participant_brain_behavior(ps, variant='raw')
    # a behavior table stands in for the missing trial behavior; the ceiling cannot
    table = truth.rename(columns={'lwpc_behav_true': 'lwpc', 'lwps_behav_true': 'lwps'})
    res = sbb.participant_brain_behavior(ps, variant='raw', behavior=table)
    assert res['n_participants'] == 6 and np.isnan(res['ceiling_lwpc'])
    with pytest.raises(KeyError):
        sbb.participant_brain_behavior(ps, variant='rtadj', behavior=table)


def test_without_trial_ids_aligned_tables_work_and_ragged_ones_refuse():
    df, _ = sbb._synthetic_long_df(n_subj=4, seed=0)
    ps = sbb.participant_scores(df.drop(columns='trial'), n_splits=2)
    ref = sbb.participant_scores(df, n_splits=2)
    np.testing.assert_allclose(ps['scores']['lwpc_neural'], ref['scores']['lwpc_neural'])
    ragged = df.drop(columns='trial').drop(index=df.index[0])
    with pytest.raises(ValueError, match='trial'):
        sbb.participant_scores(ragged, n_splits=2)


def test_error_trials_are_dropped_for_brain_and_behavior_alike():
    df, _ = sbb._synthetic_long_df(n_subj=4, seed=0)
    df.loc[df['trial'] % 10 == 0, 'acc'] = 0.0
    ps = sbb.participant_scores(df, n_splits=2)
    n_correct = df[df['acc'] == 1].groupby('subject')['trial'].nunique()
    got = ps['scores'].set_index('subject')['n_trials']
    assert (got.loc[n_correct.index] == n_correct).all()
    assert any('acc != 1' in n for n in ps['notes'])


# ---------------------------------------------------------------------------
# behavior from a per-subject table (the subject-level effects CSV)
# ---------------------------------------------------------------------------
def test_subject_stem_matches_epoch_and_csv_ids():
    assert sbb.subject_stem('D0107A') == sbb.subject_stem('D0107') == 'D0107'


def test_match_behavior_to_subjects_uses_the_stem():
    table = pd.DataFrame(dict(subject=['D0107', 'D0057', 'D0071'],
                              lwpc=[1.0, 2.0, 3.0], lwps=[4.0, 5.0, 6.0]))
    matched, missing = sbb.match_behavior_to_subjects(table, ['D0057', 'D0107A', 'D0144'])
    assert list(matched['subject']) == ['D0057', 'D0107A']
    assert list(matched['lwpc']) == [2.0, 1.0]
    assert missing == ['D0144']


def test_given_behavior_replaces_the_trial_scored_one(planted, tmp_path):
    """With `behavior`, every correlation uses the table. It has no trials, so the
    half-split checks are NaN, and the ceiling borrows the trial-scored behavioral
    reliability only when allowed to."""
    _, truth, ps = planted
    table = truth.rename(columns={'lwpc_behav_true': 'lwpc',
                                  'lwps_behav_true': 'lwps'}).iloc[1:]
    res = sbb.participant_brain_behavior(ps, variant='rtadj', behavior=table)
    trial = sbb.participant_brain_behavior(ps, variant='rtadj')
    assert res['behavior_from'] == 'table' and trial['behavior_from'] == 'trials'

    t = res['table'].merge(table, on='subject')
    assert len(t) == len(res['table']) == len(table)       # the missing one dropped
    np.testing.assert_allclose(t['lwpc_behav'], t['lwpc'])
    assert res['corr_lwpc'] == pytest.approx(
        np.corrcoef(t['lwpc_neural_rtadj'], t['lwpc'])[0, 1])
    assert res['corr_lwpc'] != pytest.approx(trial['corr_lwpc'])
    for eff in ('lwpc', 'lwps'):
        assert np.isnan(res[f'corr_{eff}_same_half'])
        assert np.isnan(res[f'corr_{eff}_disjoint_half'])
        np.testing.assert_equal(res[f'reliability_behav_{eff}'],
                                trial[f'reliability_behav_{eff}'])
    no_rel = sbb.participant_brain_behavior(ps, variant='rtadj', behavior=table,
                                            reliability_from_trials=False)
    assert np.isnan(no_rel['reliability_behav_lwpc']) and np.isnan(no_rel['ceiling_lwpc'])

    s = sbb.attach_behavior(ps['scores'], table)
    np.testing.assert_allclose(s['lwpc_behav_trials'], ps['scores']['lwpc_behav'])
    assert s['lwpc_behav'].isna().sum() == 1

    # the job's summary and figure take the table's NaN half-split values
    lines = "\n".join(dcc._participant_lines(ps, {'rtadj': res}))
    assert 'no trial halves' in lines and '(iEEG trials)' in lines
    dcc.make_participant_plots({'rtadj': res}, str(tmp_path))
    assert (tmp_path / 'participant_brain_behavior.png').exists()


def test_scatter_reports_the_correlation_it_plots(planted, tmp_path):
    """The R^2 / p printed on the scatter are the r^2 / p of
    `participant_brain_behavior`, from the result's table or participant_scores.csv."""
    _, _, ps = planted
    for variant in ('rtadj', 'raw'):
        res = sbb.participant_brain_behavior(ps, variant=variant)
        for table in (res['table'], ps['scores']):
            stats = dcc.make_participant_scatter(table, str(tmp_path), variant=variant,
                                                 formats=('png',))
            for eff in ('lwpc', 'lwps'):
                assert stats[eff]['r2'] == pytest.approx(res[f'corr_{eff}'] ** 2)
                assert stats[eff]['p'] == pytest.approx(res[f'p_{eff}'])
                assert stats[eff]['n'] == res['n_participants']
        assert (tmp_path / f'participant_brain_behavior_scatter_{variant}.png').exists()


# ---------------------------------------------------------------------------
# the DCC job end to end (synthetic)
# ---------------------------------------------------------------------------


def test_dcc_synthetic_run_writes_the_participant_outputs(tmp_path):
    args = SimpleNamespace(
        data_source='synthetic', task='GlobalLocal', synthetic_n_subj=12,
        synthetic_across_beta=1.2, synthetic_within_beta=0.6,
        synthetic_cross_frac=0.25, synthetic_link=0.6, synthetic_rt_coupling=0.3,
        alpha=0.05, neural_summary='count', run_trialwise=True, min_elec=3,
        participant_n_splits=5, seed=0, save_dir=str(tmp_path), rois_dict=None,
        electrodes='all', window_tmin=0.0, window_tmax=1.5, epochs_root_file=None,
        behavior_csv=None, fdr_correction='fdr_bh')
    out = dcc.main(args)
    for name in ('participant_scores.csv', 'participant_electrode_scores.csv',
                 'participant_reliability.csv', 'participant_brain_behavior.json',
                 'participant_brain_behavior.png', 'summary.txt',
                 'participant_brain_behavior_scatter_rtadj.png',
                 'participant_brain_behavior_scatter_raw.pdf',
                 'participant_brain_behavior_scatter_abs_rtadj.png',
                 'participant_brain_behavior_scatter_pos_rtadj.png'):
        assert (tmp_path / name).exists(), name
    assert set(out['participant']) == set(sbb.NEURAL_VARIANTS)
    summary = (tmp_path / 'summary.txt').read_text()
    assert '(1) ACROSS PARTICIPANTS, CONTINUOUS SCORES' in summary
    assert 'RT-ADJUSTED' in summary and 'ceiling' in summary
