"""Tests for the power_traces -> conjunction bridge.

Covers the label plumbing (effect-name resolution, electrode alignment, the
three correction modes), the cluster-p recomputation from `.npz`, and the
count test -- including the identity between "shared vs distinct" and the joint
count null, which is the claim the module's docstring makes.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import power_traces_conjunction as ptc


# ----------------------------------------------------------------------------
# fixtures: a fake within-electrode ANOVA run on disk
# ----------------------------------------------------------------------------
def _summary_rows(subject, electrode, roi, effect, p, extent=0, sign=0,
                  best_p=None):
    """A summary.csv row. `best_p=None` omits the graded columns entirely, which
    is how legacy runs (written before `best_cluster_p` existed) look on disk."""
    row = dict(subject=subject, electrode=electrode, roi=roi, effect=effect,
               cluster_idx=0 if extent else -1, sign=sign,
               extent_windows=extent, cluster_onset=0.1, cluster_offset=0.3,
               cluster_p_value=p, peak_F=5.0, peak_time=0.2)
    if best_p is not None:
        row.update(best_cluster_p=best_p, best_cluster_extent=extent,
                   best_cluster_sign=sign)
    return row


def _write_run(tmp_path, name, effect, entries, roi='lpfc'):
    """entries: list of (subject, electrode, p, extent, sign)."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_rows(s, e, roi, effect, p, ext, sg)
            for s, e, p, ext, sg in entries]
    pd.DataFrame(rows).to_csv(run_dir / 'summary.csv', index=False)
    return run_dir


CPC_EFFECT = 'C(congruency):C(incongruentProportion)'
SPS_EFFECT = 'C(switchType):C(switchProportion)'
CPS_EFFECT = 'C(congruency):C(switchProportion)'
SPC_EFFECT = 'C(switchType):C(incongruentProportion)'


@pytest.fixture
def four_runs(tmp_path):
    """Four runs over the same 8 electrodes across 2 subjects.

    Designed so CPC and SPS overlap heavily (a shared core) while the two cross
    interactions recruit almost nobody.
    """
    elecs = [('S1', f'e{i}') for i in range(4)] + [('S2', f'e{i}') for i in range(4)]

    def entries(sig_idx):
        out = []
        for i, (s, e) in enumerate(elecs):
            if i in sig_idx:
                out.append((s, e, 0.002, 5, 1))
            else:
                out.append((s, e, 1.0, 0, 0))
        return out

    return {
        'CPC': _write_run(tmp_path, 'lwpc', CPC_EFFECT, entries({0, 1, 2, 4, 5})),
        'SPS': _write_run(tmp_path, 'lwps', SPS_EFFECT, entries({0, 1, 2, 4, 6})),
        'CPS': _write_run(tmp_path, 'cps', CPS_EFFECT, entries(set())),
        'SPC': _write_run(tmp_path, 'spc', SPC_EFFECT, entries({7})),
    }


# ----------------------------------------------------------------------------
# effect names
# ----------------------------------------------------------------------------
def test_effect_name_matches_statsmodels_convention():
    assert ptc.anova_effect_name('congruency', 'incongruentProportion') == CPC_EFFECT


def test_reversed_factor_order_still_resolves(tmp_path):
    """windowed_anova pins effect names off `anova_factors` order, so a run
    declared the other way round must still be found."""
    reversed_name = 'C(incongruentProportion):C(congruency)'
    run = _write_run(tmp_path, 'rev', reversed_name,
                     [('S1', 'e0', 0.01, 3, 1), ('S1', 'e1', 1.0, 0, 0)])
    tbl = ptc._best_cluster_per_electrode(
        ptc.load_run_summary(run),
        ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert len(tbl) == 2


def test_unknown_effect_raises_with_available_listed(tmp_path):
    run = _write_run(tmp_path, 'x', SPS_EFFECT, [('S1', 'e0', 1.0, 0, 0)])
    with pytest.raises(KeyError, match='available effects'):
        ptc._best_cluster_per_electrode(
            ptc.load_run_summary(run),
            ptc._effect_name_variants('congruency', 'incongruentProportion'),
            roi='lpfc')


def test_missing_summary_raises(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError, match='summary.csv'):
        ptc.load_run_summary(tmp_path / 'empty')


# ----------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------
def test_labels_have_the_conjunction_contract(four_runs):
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                               use_npz=False)
    for col in ('subject', 'electrode', 'S', 'F', 'CPC', 'SPS', 'CPS', 'SPC'):
        assert col in lab.columns
    assert (lab['S'] == lab['CPC']).all()
    assert (lab['F'] == lab['SPS']).all()
    assert len(lab) == 8


def test_cluster_mode_reproduces_raw_thresholding(four_runs):
    lab = ptc.electrode_labels(four_runs, roi='lpfc', alpha=0.05,
                               correction='cluster', use_npz=False)
    assert lab['CPC'].sum() == 5
    assert lab['SPS'].sum() == 5
    assert int(((lab.S == 1) & (lab.F == 1)).sum()) == 4       # e0,e1,e2 (S1), e0 (S2)
    assert lab['CPS'].sum() == 0                                # cross control null
    assert lab['SPC'].sum() == 1


def test_none_mode_flags_any_surviving_cluster(four_runs):
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='none',
                               use_npz=False)
    assert lab['CPC'].sum() == 5


def test_fdr_is_never_more_liberal_than_raw(four_runs):
    raw = ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                               use_npz=False)
    fdr = ptc.electrode_labels(four_runs, roi='lpfc', correction='fdr_bh',
                               use_npz=False)
    assert fdr['CPC'].sum() <= raw['CPC'].sum()
    assert fdr['q_cpc'].notna().all()


def test_require_all_intersects_electrode_sets(tmp_path):
    """A 2x2 table over inconsistent denominators is not interpretable, so
    electrodes missing from any run are dropped and counted."""
    cpc = _write_run(tmp_path, 'a', CPC_EFFECT,
                     [('S1', 'e0', 0.01, 3, 1), ('S1', 'e1', 1.0, 0, 0)])
    sps = _write_run(tmp_path, 'b', SPS_EFFECT,
                     [('S1', 'e0', 0.01, 3, 1)])                 # e1 missing
    lab = ptc.electrode_labels({'CPC': cpc, 'SPS': sps}, roi='lpfc',
                               correction='cluster', use_npz=False)
    assert len(lab) == 1
    assert lab.attrs['n_dropped'] == 1


def test_require_all_false_keeps_union(tmp_path):
    cpc = _write_run(tmp_path, 'a', CPC_EFFECT,
                     [('S1', 'e0', 0.01, 3, 1), ('S1', 'e1', 1.0, 0, 0)])
    sps = _write_run(tmp_path, 'b', SPS_EFFECT, [('S1', 'e0', 0.01, 3, 1)])
    lab = ptc.electrode_labels({'CPC': cpc, 'SPS': sps}, roi='lpfc',
                               correction='cluster', use_npz=False,
                               require_all=False)
    assert len(lab) == 2
    assert lab.loc[lab.electrode == 'e1', 'SPS'].iloc[0] == 0   # missing -> not sig


def test_roi_filter_excludes_other_rois(tmp_path):
    rows = [_summary_rows('S1', 'e0', 'lpfc', CPC_EFFECT, 0.01, 3, 1),
            _summary_rows('S1', 'e9', 'acc', CPC_EFFECT, 0.01, 3, 1)]
    run = tmp_path / 'r'
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / 'summary.csv', index=False)
    tbl = ptc._best_cluster_per_electrode(
        ptc.load_run_summary(run),
        ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert list(tbl['electrode']) == ['e0']


def test_missing_construct_group_raises(tmp_path):
    cpc = _write_run(tmp_path, 'a', CPC_EFFECT, [('S1', 'e0', 0.01, 3, 1)])
    with pytest.raises(ValueError, match='SPS'):
        ptc.electrode_labels({'CPC': cpc}, roi='lpfc', use_npz=False)


def test_best_cluster_is_the_smallest_p(tmp_path):
    """An electrode with several clusters is represented by its best one."""
    run = tmp_path / 'multi'
    run.mkdir()
    rows = [_summary_rows('S1', 'e0', 'lpfc', CPC_EFFECT, 0.30, 2, -1),
            _summary_rows('S1', 'e0', 'lpfc', CPC_EFFECT, 0.01, 7, 1)]
    pd.DataFrame(rows).to_csv(run / 'summary.csv', index=False)
    tbl = ptc._best_cluster_per_electrode(
        ptc.load_run_summary(run),
        ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert len(tbl) == 1
    assert tbl['p_cluster'].iloc[0] == pytest.approx(0.01)
    assert tbl['sign'].iloc[0] == 1
    assert tbl['extent'].iloc[0] == 7


# ----------------------------------------------------------------------------
# cluster-p recomputation from npz
# ----------------------------------------------------------------------------
def _write_npz(run_dir, roi, sub, elec, observed_F, null_F, effect_names):
    out = run_dir / roi / sub
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{elec}.npz',
             observed_F=observed_F, null_F=null_F,
             signed_contrast=np.full(observed_F.shape, np.nan),
             effect_names=np.array(effect_names),
             window_centers=np.arange(observed_F.shape[1], dtype=float))


def test_npz_recovers_graded_p_where_summary_floors_at_one(tmp_path):
    """The point of the npz path: summary.csv writes 1.0 for every electrode that
    missed the extent threshold, which BH cannot rank. Recomputing gives a real p."""
    rng = np.random.default_rng(0)
    n_win, n_perm = 20, 200
    null = rng.chisquare(2, size=(n_perm, 1, n_win)).astype(np.float32)
    # A short, strong bump: clears the pointwise threshold on 3 windows.
    obs = np.full((1, n_win), 0.1)
    obs[0, 5:8] = 50.0

    run = tmp_path / 'run'
    _write_npz(run, 'lpfc', 'S1', 'e0', obs, null, [CPC_EFFECT])

    tbl = ptc.refine_cluster_pvalues_from_npz(
        run, ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert len(tbl) == 1
    assert tbl['extent'].iloc[0] == 3
    assert 0.0 <= tbl['p_cluster'].iloc[0] < 1.0


def test_npz_flat_trace_gives_p_one(tmp_path):
    rng = np.random.default_rng(1)
    null = rng.chisquare(2, size=(100, 1, 15)).astype(np.float32)
    obs = np.zeros((1, 15))
    run = tmp_path / 'run'
    _write_npz(run, 'lpfc', 'S1', 'e0', obs, null, [CPC_EFFECT])
    tbl = ptc.refine_cluster_pvalues_from_npz(
        run, ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert tbl['p_cluster'].iloc[0] == 1.0
    assert tbl['extent'].iloc[0] == 0


def test_npz_absent_falls_back_to_summary(four_runs):
    """No .npz written by the fixture, so use_npz=True must not crash."""
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                               use_npz=True)
    assert len(lab) == 8


# ----------------------------------------------------------------------------
# the graded cluster p recorded at run time (preferred over both fallbacks)
# ----------------------------------------------------------------------------
def test_best_cluster_p_is_preferred_over_cluster_p_value(tmp_path):
    """A near-miss electrode: no surviving cluster (cluster_p_value floored at
    1.0) but a real graded p of 0.08. The graded value must win, otherwise it is
    indistinguishable from a true null and BH cannot rank it."""
    run = tmp_path / 'r'
    run.mkdir()
    rows = [_summary_rows('S1', 'e0', 'lpfc', CPC_EFFECT, 1.0, 0, 0, best_p=0.08),
            _summary_rows('S1', 'e1', 'lpfc', CPC_EFFECT, 1.0, 0, 0, best_p=0.97)]
    pd.DataFrame(rows).to_csv(run / 'summary.csv', index=False)
    tbl = ptc._best_cluster_per_electrode(
        ptc.load_run_summary(run),
        ptc._effect_name_variants('congruency', 'incongruentProportion'),
        roi='lpfc')
    assert sorted(tbl['p_cluster']) == pytest.approx([0.08, 0.97])


def test_graded_p_beats_npz_recompute_when_present(tmp_path):
    """When summary.csv carries the graded column, the npz is not consulted --
    the run-time value is the more correct one (symmetric sign-split)."""
    run = tmp_path / 'r'
    run.mkdir()
    pd.DataFrame([_summary_rows('S1', 'e0', 'lpfc', CPC_EFFECT, 1.0, 0, 0,
                                best_p=0.11)]).to_csv(run / 'summary.csv',
                                                      index=False)
    rng = np.random.default_rng(0)
    _write_npz(run, 'lpfc', 'S1', 'e0', np.zeros((1, 12)),
               rng.chisquare(2, size=(50, 1, 12)).astype(np.float32), [CPC_EFFECT])
    sps = _write_run(tmp_path, 'sps', SPS_EFFECT, [('S1', 'e0', 1.0, 0, 0)])
    lab = ptc.electrode_labels({'CPC': run, 'SPS': sps}, roi='lpfc',
                               correction='fdr_bh', use_npz=True)
    assert lab['p_cpc'].iloc[0] == pytest.approx(0.11)   # not the npz's 1.0


def test_legacy_summary_without_graded_column_still_works(four_runs):
    """The fixture writes no best_cluster_p, i.e. a pre-existing run."""
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                               use_npz=False)
    assert len(lab) == 8
    assert lab['CPC'].sum() == 5


# ----------------------------------------------------------------------------
# the count test
# ----------------------------------------------------------------------------
def _labels(sf_pairs):
    """sf_pairs: list of (subject, S, F)."""
    return pd.DataFrame([dict(subject=s, electrode=f'e{i}', S=a, F=b)
                         for i, (s, a, b) in enumerate(sf_pairs)])


def test_shared_vs_distinct_is_the_same_test_as_the_joint_count():
    """With marginals fixed, D = 3*both - n_S - n_F is monotone in `both`, so the
    two permutation tests are identical. The module claims this; verify it."""
    rng = np.random.default_rng(3)
    pairs = []
    for subj in ('S1', 'S2', 'S3'):
        for _ in range(12):
            pairs.append((subj, int(rng.integers(2)), int(rng.integers(2))))
    lab = _labels(pairs)

    bvd = ptc.both_vs_distinct_test(lab, n_perm=2000, seed=7)
    from src.analysis.stats.stability_flexibility_segregation import (
        conjunction_permutation_null)
    joint = conjunction_permutation_null(lab, n_perm=2000, seed=7)

    assert bvd['n_both'] == joint['observed']
    assert bvd['p_two_sided'] == pytest.approx(joint['p_two_sided'])
    # and the identity itself
    n_S, n_F = int(lab.S.sum()), int(lab.F.sum())
    assert bvd['observed'] == 3 * bvd['n_both'] - n_S - n_F


def test_perfect_overlap_gives_significant_shared_excess():
    pairs = ([('S1', 1, 1)] * 6 + [('S1', 0, 0)] * 6
             + [('S2', 1, 1)] * 6 + [('S2', 0, 0)] * 6)
    res = ptc.both_vs_distinct_test(_labels(pairs), n_perm=2000, seed=0)
    assert res['n_both'] == 12
    assert res['n_stab_only'] == 0 and res['n_flex_only'] == 0
    assert res['p_shared_greater'] < 0.01
    assert res['z'] > 0


def test_perfect_segregation_gives_shared_deficit():
    pairs = ([('S1', 1, 0)] * 6 + [('S1', 0, 1)] * 6
             + [('S2', 1, 0)] * 6 + [('S2', 0, 1)] * 6)
    res = ptc.both_vs_distinct_test(_labels(pairs), n_perm=2000, seed=0)
    assert res['n_both'] == 0
    assert res['observed'] < res['null_mean']
    assert res['p_shared_greater'] > 0.9          # not evidence for sharing
    assert res['p_two_sided'] < 0.05              # but a real deficit


def test_independent_labels_are_not_significant():
    """The built-in check that the test does not manufacture a result."""
    rng = np.random.default_rng(11)
    pairs = [(f'S{1 + i // 20}', int(rng.integers(2)), int(rng.integers(2)))
             for i in range(80)]
    res = ptc.both_vs_distinct_test(_labels(pairs), n_perm=3000, seed=1)
    assert res['p_two_sided'] > 0.05


def test_counts_partition_the_electrodes():
    pairs = [('S1', 1, 1), ('S1', 1, 0), ('S1', 0, 1), ('S1', 0, 0)]
    res = ptc.both_vs_distinct_test(_labels(pairs), n_perm=200, seed=0)
    total = (res['n_both'] + res['n_stab_only']
             + res['n_flex_only'] + res['n_neither'])
    assert total == res['n_electrodes'] == 4


def test_null_respects_subject_nesting():
    """S and F perfectly anti-aligned across subjects but constant within them:
    a within-subject shuffle cannot change the joint count."""
    pairs = [('S1', 1, 1)] * 5 + [('S2', 0, 0)] * 5
    res = ptc.both_vs_distinct_test(_labels(pairs), n_perm=500, seed=0)
    assert res['null_sd'] == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def test_full_battery_runs_and_summarizes(four_runs):
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='fdr_bh')
    res = ptc.run_power_traces_conjunction(lab, n_perm=500)
    assert set(res) >= {'cmh', 'joint_count_null', 'both_vs_distinct',
                        'cross_controls', 'attrs'}
    assert len(res['cross_controls']) == 4
    text = ptc.summarize(res)
    assert 'MH odds ratio' in text
    assert 'cross controls' in text


def test_cross_controls_report_all_four_interactions(four_runs):
    lab = ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                               use_npz=False)
    cc = ptc.cross_control_counts(lab).set_index('group')
    assert cc.loc['CPS', 'n_significant'] == 0
    assert cc.loc['SPC', 'n_significant'] == 1
    assert cc.loc['CPC', 'n_significant'] == 5


def test_sweep_present_only_with_fdr(four_runs):
    fdr = ptc.run_power_traces_conjunction(
        ptc.electrode_labels(four_runs, roi='lpfc', correction='fdr_bh'),
        n_perm=200)
    raw = ptc.run_power_traces_conjunction(
        ptc.electrode_labels(four_runs, roi='lpfc', correction='cluster',
                             use_npz=False),
        n_perm=200)
    assert 'sweep' in fdr
    assert 'sweep' not in raw


def test_bad_correction_mode_raises(four_runs):
    with pytest.raises(ValueError, match='correction must be'):
        ptc.electrode_labels(four_runs, roi='lpfc', correction='bonferroni')
