"""The coding-axes analysis (docs/analysis_plans.md, subspace plan §3.1-3.3).

What is pinned, in the order the analysis uses it:

  * `main_effect_mean_diffs` gives exactly the loop's numbers
    (`compute_sensitivities_per_window`, mean_diff, shared split), including
    electrodes that lost trials, and its `remove_mean` removes a shift shared by
    a participant's electrodes;
  * the cosines: identical axes -> alignment 1, orthogonal -> 0, the §3.1
    worked example (cosine 0.8, Pearson -1), the ceiling guard, and agreement
    with `map_reliability(method='cosine')`;
  * the shuffle keeps the design and gives every electrode of a trial the same
    label; the null is reproducible from its seed and sits below a planted
    shared code;
  * the bootstrap flags windows without a code as unreliable, and the cluster
    marks land where the planted sharing is.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import coding_axes as ca
from src.analysis.stats import stability_flexibility_anatomy as sfa
from src.analysis.stats import stability_flexibility_segregation as sfs


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _trial_long(c_w, s_w, n_subj=6, n_elec=8, n_trials=192, noise=1.0, seed=0,
                n_time=1):
    """Long table whose HG carries planted congruency and switch codes.

    Electrode e of every participant has congruency weight c_w[e] and switch
    weight s_w[e] (per sample, `c_w`/`s_w` of shape (n_elec,) or
    (n_elec, n_time)); HG = c * (+-1/2) + s * (+-1/2) + noise. Returns the
    table (hg = time courses of `n_time` samples) and `times`."""
    rng = np.random.default_rng(seed)
    c_w = np.broadcast_to(np.asarray(c_w, float).reshape(n_elec, -1), (n_elec, n_time))
    s_w = np.broadcast_to(np.asarray(s_w, float).reshape(n_elec, -1), (n_elec, n_time))
    frames = []
    for p in range(n_subj):
        t = pd.DataFrame(dict(
            trial=np.arange(n_trials),
            congruency=np.tile(['i', 'c'], n_trials // 2),
            switchType=np.tile(['s', 's', 'r', 'r'], n_trials // 4),
            incongruent_proportion=np.tile([25.0] * 4 + [75.0] * 4, n_trials // 8),
            switch_proportion=np.tile([25.0] * 8 + [75.0] * 8, n_trials // 16)))
        xc = np.where(t.congruency == 'i', 0.5, -0.5)[:, None]
        xs = np.where(t.switchType == 's', 0.5, -0.5)[:, None]
        for e in range(n_elec):
            hg = xc * c_w[e] + xs * s_w[e] + rng.normal(0, noise, (n_trials, n_time))
            frames.append(t.assign(subject=f'P{p}', electrode=f'P{p}-e{e}', hg=list(hg)))
    return pd.concat(frames, ignore_index=True), np.arange(n_time) / 256.0


def _scored(df, times, size=1, step=1, n_splits=10, seed=0, remove_mean=False):
    means, grid = sfs.window_means(df['hg'].to_numpy(), times, size, step)
    labels = df.drop(columns='hg')
    per = sfs.main_effect_mean_diffs(labels, means, n_splits=n_splits, seed=seed,
                                     remove_mean=remove_mean).merge(grid, on='window')
    return per, labels, means


def _per_split(c, s, n_splits=3, subjects=None, noise=0.0, seed=0):
    """A per-split table straight from axis vectors c, s (one window): halves
    are the vectors plus independent noise."""
    rng = np.random.default_rng(seed)
    c, s = np.asarray(c, float), np.asarray(s, float)
    subjects = subjects or ['P0'] * len(c)
    rows = []
    for k in range(n_splits):
        draw = {col: v + rng.normal(0, noise, len(v)) if noise else v
                for col, v in (('mxA', c), ('mxB', c), ('myA', s), ('myB', s))}
        for e in range(len(c)):
            rows.append(dict(subject=subjects[e], electrode=f'e{e}', split=k,
                             **{col: draw[col][e] for col in draw}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_fast_main_effects_match_the_loop():
    rng = np.random.default_rng(1)
    df, times = _trial_long(rng.normal(size=5), rng.normal(size=5), n_subj=3,
                            n_elec=5, n_time=40, seed=1)
    df = df.sample(frac=0.95, random_state=0).reset_index(drop=True)  # lost trials
    ref = sfs.compute_sensitivities_per_window(
        df, times, 16, 8, n_splits=4, seed=7, contrast_mode='proportion',
        main_effects=True, shared_split=True, effect_measure='mean_diff')
    fast, _, _ = _scored(df, times, 16, 8, n_splits=4, seed=7)

    m = ref.merge(fast, on=['electrode', 'split', 'window'], suffixes=('_ref', ''))
    assert len(m) == len(ref) == len(fast)
    for col in ca.AXIS_COLS:
        assert np.allclose(m[f'{col}_ref'], m[col], equal_nan=True)


def test_shared_halves_reproduce_the_scorer_and_its_cell_means():
    """`halves=` scores on exactly the trials the seed draws, and the cell means
    behind the effects are plain means over those trials."""
    rng = np.random.default_rng(3)
    df, times = _trial_long(rng.normal(size=4), rng.normal(size=4), n_subj=2, n_elec=4,
                            n_time=8, seed=3)
    means, _ = sfs.window_means(df['hg'].to_numpy(), times, 4, 4)
    labels = df.drop(columns='hg')
    halves = sfs.shared_halves(labels, n_splits=3, seed=11)
    a = sfs.main_effect_mean_diffs(labels, means, n_splits=3, seed=11)
    b = sfs.main_effect_mean_diffs(labels, means, halves=halves)
    assert a.equals(b)

    system = {'cong': (('_scond',), ((1.0,), (0.0,)))}           # incongruent, congruent
    subj, elecs, out = next(sfs.half_cell_means(labels, means, system, halves))
    cell_means, counts = out['cong']                               # (e, windows, splits, 2, 2)
    rows = (labels['subject'] == subj) & (labels['electrode'] == elecs[1])
    half = halves[subj][2].reindex(labels.loc[rows, 'trial']).to_numpy()   # split 2
    in_b = rows.to_numpy().copy()
    in_b[rows.to_numpy()] = half == 1
    inc = in_b & (labels['congruency'] == 'i').to_numpy()
    assert cell_means[1, 0, 2, 1, 0] == pytest.approx(means[inc, 0].mean())
    assert counts[1, 0, 2, 1, 0] == inc.sum()


def test_remove_mean_ignores_a_shift_shared_by_a_participants_electrodes():
    rng = np.random.default_rng(2)
    df, times = _trial_long(rng.normal(size=6), rng.normal(size=6), n_subj=2, n_elec=6)
    # a per-trial shift on all of a participant's electrodes, larger on
    # incongruent trials: a uniform "everything up on incongruent trials"
    trials = df.drop_duplicates(['subject', 'trial'])[['subject', 'trial', 'congruency']]
    trials['shift'] = (trials['congruency'] == 'i') * 3.0 + rng.normal(0, 1, len(trials))
    shift = df.merge(trials, on=['subject', 'trial', 'congruency'])['shift']
    shifted = df.assign(hg=[h + d for h, d in zip(df['hg'], shift)])
    a, _, _ = _scored(df, times, remove_mean=True)
    b, _, _ = _scored(shifted, times, remove_mean=True)
    plain, _, _ = _scored(shifted, times, remove_mean=False)

    assert np.allclose(a[list(ca.AXIS_COLS)], b[list(ca.AXIS_COLS)])
    assert not np.allclose(a['mxA'], plain['mxA'])        # it was there before


# ---------------------------------------------------------------------------
# cosines
# ---------------------------------------------------------------------------
def test_identical_axes_align_fully_and_orthogonal_ones_not_at_all():
    c = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    s_orth = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])      # c . s_orth = 2 - 2 = 0
    same = ca.coding_axis_cosines(_per_split(c, 2 * c)).iloc[0]
    orth = ca.coding_axis_cosines(_per_split(c, s_orth)).iloc[0]

    assert same['cross'] == pytest.approx(1.0)              # direction only, not size
    assert same['alignment'] == pytest.approx(1.0)
    assert orth['cross'] == pytest.approx(0.0, abs=1e-12)
    assert orth['alignment'] == pytest.approx(0.0, abs=1e-12)


def test_cosine_keeps_the_uniform_part_that_pearson_removes():
    """analysis_plans.md §3.1: shared uniform shift, opposite patterns."""
    row = ca.coding_axis_cosines(_per_split([2, 1, 2, 1], [1, 2, 1, 2])).iloc[0]
    assert row['cross'] == pytest.approx(0.8)
    assert row['cross_pearson'] == pytest.approx(-1.0)


def test_ceiling_is_undefined_unless_both_reliabilities_are_positive():
    rng = np.random.default_rng(3)
    per = _per_split(np.zeros(30), rng.normal(size=30), noise=1.0, n_splits=5)
    # half A and half B of congruency are independent noise: its "ceiling" can be <= 0
    per['mxB'] = -per['mxA']
    row = ca.coding_axis_cosines(per).iloc[0]
    assert row['congruency_ceiling'] < 0
    assert np.isnan(row['ceiling']) and np.isnan(row['alignment'])


def test_cosines_agree_with_map_reliability():
    rng = np.random.default_rng(4)
    c = rng.normal(size=40)
    per = _per_split(c, 0.5 * c + rng.normal(size=40), noise=0.7, n_splits=6,
                     subjects=[f'P{i % 4}' for i in range(40)])
    row = ca.coding_axis_cosines(per).iloc[0]
    ref = sfa.map_reliability(sfs.main_effect_view(per), method='cosine')

    assert row['cross'] == pytest.approx(ref['between'])
    assert row['congruency_ceiling'] == pytest.approx(ref['reliability_lwpc'])
    assert row['switch_ceiling'] == pytest.approx(ref['reliability_lwps'])
    assert row['alignment'] == pytest.approx(ref['between_noise_corrected'])


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------
def test_shuffle_keeps_the_design_and_is_shared_across_electrodes():
    df, _ = _trial_long(np.ones(4), np.ones(4), n_subj=2, n_elec=4)
    out = ca.permute_within_cells(df, 'congruency', np.random.default_rng(0))

    others = ['subject', 'switchType', 'incongruent_proportion', 'switch_proportion']
    count = lambda d: d.groupby(others + ['congruency']).size()
    assert count(out).equals(count(df))                              # design unchanged
    assert (out.groupby(['subject', 'trial'])['congruency'].nunique() == 1).all()
    assert (out['congruency'] != df['congruency']).any()             # it did shuffle
    assert out[others + ['trial', 'electrode']].equals(df[others + ['trial', 'electrode']])


def test_null_is_reproducible_and_below_a_planted_shared_code():
    w = np.linspace(-1.5, 2.0, 8)
    df, times = _trial_long(w, w, noise=1.0, seed=5)                 # c = s: shared
    per, labels, means = _scored(df, times, n_splits=10)
    observed = ca.coding_axis_cosines(per)['cross'].iloc[0]

    null = ca.cross_cosine_null(labels, means, 'congruency', n_perm=20, n_splits=4, seed=9)
    again = ca.cross_cosine_null(labels, means, 'congruency', n_perm=20, n_splits=4, seed=9)
    assert null.shape == (20, 1)
    assert np.array_equal(null, again)
    assert observed > np.percentile(null, 95)
    with pytest.raises(ValueError, match='factor'):
        ca.cross_cosine_null(labels, means, 'incongruent_proportion', n_perm=1)


# ---------------------------------------------------------------------------
# the full comparison
# ---------------------------------------------------------------------------
def test_compare_marks_shared_windows_and_greys_out_windows_without_a_code():
    n_time = 12
    w = np.linspace(-1.5, 2.0, 8)
    code = np.zeros((8, n_time))
    code[:, 4:10] = w[:, None]                    # a shared code in samples 4-9 only
    df, times = _trial_long(code, code, n_time=n_time, noise=1.0, seed=6)
    per, labels, means = _scored(df, times, size=2, step=2, n_splits=10)
    null = {f: ca.cross_cosine_null(labels, means, f, n_perm=40, n_splits=4, seed=i)
            for i, f in enumerate(ca.NULL_FACTORS)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table, summary = ca.compare_coding_axes(per, null=null, n_boot=300, seed=0)

    assert list(table['window']) == list(range(6))
    assert table['reliable'].tolist() == [False, False, True, True, True, False]
    assert table.loc[table['reliable'], 'shared'].all()
    assert not table.loc[~table['reliable'], 'shared'].any()
    # c = s, so the code is shared as far as noise allows: alignment near 1 and
    # the ceiling - cross gap's CI includes 0 (not "not one code")
    assert summary['alignment_reliable_mean'] == pytest.approx(1.0, abs=0.15)
    assert not table['not_one_code'].any()
    lo, hi = summary['alignment_reliable_ci']
    assert lo <= summary['alignment_reliable_mean'] <= hi


# ---------------------------------------------------------------------------
# RT control
# ---------------------------------------------------------------------------
def _rt_driven(seed=8):
    """HG that tracks RT and nothing else, with incongruent trials 100 ms slower:
    a congruency effect that is all RT."""
    df, times = _trial_long(np.zeros(6), np.zeros(6), n_subj=4, n_elec=6, seed=seed)
    rng = np.random.default_rng(seed)
    trials = df.drop_duplicates(['subject', 'trial'])[['subject', 'trial', 'congruency']]
    trials['rt'] = 600 + 100 * (trials['congruency'] == 'i') + rng.normal(0, 80, len(trials))
    df = df.merge(trials, on=['subject', 'trial', 'congruency'])
    df['hg'] = [h + 0.01 * rt for h, rt in zip(df['hg'], df['rt'])]
    means, _ = sfs.window_means(df['hg'].to_numpy(), times, 1, 1)
    return df.drop(columns='hg'), means


def test_rt_adjust_removes_an_effect_carried_by_rt():
    labels, means = _rt_driven()
    raw = sfs.main_effect_mean_diffs(labels, means, n_splits=5)
    lab2, adj_means, report = ca.rt_control(labels, means, 'adjust')
    adjusted = sfs.main_effect_mean_diffs(lab2, adj_means, n_splits=5)

    assert raw['mxA'].mean() == pytest.approx(1.0, abs=0.2)        # 0.01 x 100 ms
    assert abs(adjusted['mxA'].mean()) < 0.15
    assert set(report) == {'rt_adjust_slopes'}
    assert report['rt_adjust_slopes']['rt_slope'].mean() == pytest.approx(0.01, abs=0.003)


def test_rt_match_equalises_rt_and_random_keeps_its_trial_counts():
    labels, means = _rt_driven()
    matched, m_means, report = ca.rt_control(labels, means, 'match')
    random, r_means, _ = ca.rt_control(labels, means, 'random')

    after = report['rt_match_contrasts'].query("factor == 'congruency'")['diff_after']
    assert after.abs().max() < 20                                   # from ~100 ms
    assert len(m_means) == len(matched) and len(r_means) == len(random)
    # every electrode of a participant keeps the same trials
    per_elec = matched.groupby(['subject', 'electrode'])['trial'].apply(frozenset)
    assert (per_elec.groupby('subject').nunique() == 1).all()
    cells = ['subject', 'congruency', 'switchType']
    count = lambda d: d.drop_duplicates(['subject', 'trial']).groupby(cells).size()
    assert count(random).equals(count(matched))
    same = ca.rt_control(labels, means, 'none')
    assert same[0] is labels and np.array_equal(same[1], means) and same[2] == {}
    with pytest.raises(ValueError, match='mode'):
        ca.rt_control(labels, means, 'regress')


# ---------------------------------------------------------------------------
# cluster correction
# ---------------------------------------------------------------------------
def test_cluster_mass_finds_what_a_correlated_null_hides_from_run_length():
    """Each null draw is one shuffle scored in every window, so a high draw is
    high everywhere. Then the null's longest runs span all windows and A4's
    run-length test can never pass; cluster mass still separates a real effect."""
    from src.analysis.decoding.accuracy_stats import \
        find_significant_clusters_of_series_vs_distribution_based_on_percentile as by_length
    rng = np.random.default_rng(0)
    null = 0.1 * rng.normal(size=(400, 1)) * np.ones((1, 12))     # perfectly correlated
    cross = np.where((np.arange(12) >= 3) & (np.arange(12) <= 8), 0.5, 0.0)

    assert np.flatnonzero(ca.shared_windows(cross, null)).tolist() == list(range(3, 9))
    assert by_length(cross, null, None, n_cluster_perms=400, random_state=0) == []
    assert not ca.shared_windows(np.zeros(12), null).any()


# ---------------------------------------------------------------------------
# the controls figure
# ---------------------------------------------------------------------------
def test_controls_figure_and_table(tmp_path):
    import json
    from dcc_scripts.subspace import plot_coding_axes_controls as pcc

    t = np.arange(6) * 0.0625 + 0.125
    for name, rt in (('main', 'none'), ('matched', 'match')):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame(dict(window=range(6), window_center=t, cross=0.3, ceiling=0.6,
                          alignment=0.5, alignment_lo=0.4, alignment_hi=0.6,
                          reliable=[False, True, True, True, True, False],
                          shared=[False, False, True, True, False, False],
                          not_one_code=False)).to_csv(d / 'coding_axis_cosines.csv', index=False)
        (d / 'run_config.json').write_text(json.dumps(dict(rois=['lpfc'], remove_mean=False,
                                                          rt_control=rt)))
        (d / 'summary.json').write_text(json.dumps(dict(
            n_electrodes=10, n_subjects=3, n_reliable_windows=4,
            alignment_reliable_mean=0.5, alignment_reliable_ci=[0.4, 0.6])))
    pcc.main([str(tmp_path / 'main'), str(tmp_path / 'matched'), '--out', str(tmp_path)])

    assert (tmp_path / 'coding_axes_controls.png').stat().st_size > 0
    table = pd.read_csv(tmp_path / 'coding_axes_controls.csv')
    assert table['run'].tolist() == ['lpfc', 'lpfc, RT matched']
    assert table['shared_from_s'].tolist() == [t[2], t[2]]


def test_time_axis_label_follows_the_epochs_lock():
    from dcc_scripts.subspace.subspace_figures import time_axis_label

    assert time_axis_label('Response_-1.0to1.5sec_zmax_20').endswith('(s from response)')
    assert time_axis_label('Stimulus_-1.0to1.5sec_zmax_20').endswith('(s from stimulus onset)')
    assert time_axis_label(None) == time_axis_label('Stimulus_x')       # synthetic runs
