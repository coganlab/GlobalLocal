"""End-to-end tests for the graded cluster p-values written by the
within-electrode windowed ANOVA.

`run_within_electrode_windowed_anova_cluster_correction` records
`cluster_p_value` as the minimum over clusters that SURVIVED the extent
threshold, falling back to exactly 1.0 when none did. That floors every
non-surviving electrode at the same value, so a downstream across-electrode
correction cannot tell a near-miss from a true null. `best_cluster_p` records the
p of each electrode's largest cluster unconditionally.

These tests drive the real pipeline (no mocks) on a small synthetic 2x2 design.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.power.windowed_anova import (
    run_within_electrode_windowed_anova_cluster_correction,
)


ROI = 'lpfc'
SUBJECTS = ['S1', 'S2']
ELECTRODES = {'lpfc': {'S1': ['e0', 'e1'], 'S2': ['e2', 'e3']}}
ANOVA_FACTORS = ['congruency', 'incongruentProportion']
INTERACTION = 'C(congruency):C(incongruentProportion)'

# The 2x2: congruency x incongruent-proportion.
CONDITIONS = {
    'i25': {'congruency': 'i', 'incongruentProportion': 25},
    'i75': {'congruency': 'i', 'incongruentProportion': 75},
    'c25': {'congruency': 'c', 'incongruentProportion': 25},
    'c75': {'congruency': 'c', 'incongruentProportion': 75},
}

N_TRIALS = 12
N_WINDOWS = 10
WINDOW_SIZE = 4
STEP_SIZE = 2
N_TIMES = WINDOW_SIZE + STEP_SIZE * (N_WINDOWS - 1)


def _windowed_data(seed=0, interaction_windows=(), interaction_size=0.0):
    """Synthetic (n_trials, n_windows, n_channels) arrays per condition/subject.

    `interaction_size` is added to the two "+1" diagonal cells and subtracted
    from the two "-1" cells over `interaction_windows`, so it plants a genuine
    difference-of-differences with no main effect.
    """
    rng = np.random.default_rng(seed)
    sign = {'i75': +1, 'c25': +1, 'i25': -1, 'c75': -1}
    out = {}
    for cond in CONDITIONS:
        out[cond] = {ROI: []}
        for sub in SUBJECTS:
            n_chan = len(ELECTRODES[ROI][sub])
            arr = rng.normal(0, 1, size=(N_TRIALS, N_WINDOWS, n_chan))
            for w in interaction_windows:
                arr[:, w, :] += sign[cond] * interaction_size
            out[cond][ROI].append(arr)
    return out


def _run(windowed_data, tmp_path, n_perm=40, seed=1):
    return run_within_electrode_windowed_anova_cluster_correction(
        windowed_data=windowed_data,
        conditions_obj=CONDITIONS,
        anova_factors=ANOVA_FACTORS,
        rois=[ROI],
        subjects=SUBJECTS,
        electrodes_per_subject_roi=ELECTRODES,
        times=np.linspace(0.0, 0.5, N_TIMES),
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        sampling_rate=256,
        n_perm=n_perm,
        seed=seed,
        n_jobs=1,
        verbose=False,
        save_dir=str(tmp_path),
        run_label='run',
    )


@pytest.fixture(scope='module')
def null_run(tmp_path_factory):
    """Pure noise: nothing should survive, but graded p's must still be defined."""
    tmp = tmp_path_factory.mktemp('null')
    results, centers, summary, skipped = _run(_windowed_data(seed=0), tmp)
    return dict(results=results, summary=summary, run_dir=tmp / 'run')


@pytest.fixture(scope='module')
def effect_run(tmp_path_factory):
    """A planted interaction over a contiguous run of windows."""
    tmp = tmp_path_factory.mktemp('effect')
    data = _windowed_data(seed=2, interaction_windows=range(2, 8),
                          interaction_size=1.6)
    results, centers, summary, skipped = _run(data, tmp)
    return dict(results=results, summary=summary, run_dir=tmp / 'run')


# ----------------------------------------------------------------------------
# the columns exist and are populated for every electrode
# ----------------------------------------------------------------------------
def test_graded_columns_written_for_every_electrode(null_run):
    summary = null_run['summary']
    for col in ('best_cluster_p', 'best_cluster_extent', 'best_cluster_sign'):
        assert col in summary.columns
        assert summary[col].notna().all()


def test_all_four_electrodes_present(null_run):
    rows = null_run['summary']
    rows = rows[rows['effect'] == INTERACTION]
    assert set(rows['electrode']) == {'e0', 'e1', 'e2', 'e3'}


def test_graded_p_is_a_probability(null_run, effect_run):
    for run in (null_run, effect_run):
        p = run['summary']['best_cluster_p']
        assert (p >= 0).all() and (p <= 1).all()


# ----------------------------------------------------------------------------
# the invariant that motivates the column
# ----------------------------------------------------------------------------
def test_best_cluster_p_never_worse_than_surviving_cluster_p(null_run, effect_run):
    """The best cluster is taken over ALL clusters, so its p can only be smaller
    than or equal to the min over the surviving subset (which is 1.0 when the
    subset is empty)."""
    for run in (null_run, effect_run):
        s = run['summary']
        assert (s['best_cluster_p'] <= s['cluster_p_value'] + 1e-12).all()


def test_electrodes_with_no_surviving_cluster_can_still_be_ranked(null_run):
    """The whole point: cluster_p_value collapses to 1.0, best_cluster_p does not
    have to, so BH downstream has something to order."""
    s = null_run['summary']
    missed = s[s['cluster_idx'] == -1]
    assert len(missed) > 0
    assert (missed['cluster_p_value'] == 1.0).all()
    assert missed['best_cluster_p'].nunique() > 1 or (
        missed['best_cluster_extent'] == 0).all()


def test_no_supra_threshold_window_gives_p_one_and_extent_zero(null_run):
    s = null_run['summary']
    flat = s[s['best_cluster_extent'] == 0]
    if len(flat):
        assert (flat['best_cluster_p'] == 1.0).all()
        assert (flat['best_cluster_sign'] == 0).all()


def test_planted_interaction_lowers_the_graded_p(null_run, effect_run):
    """A real, sustained interaction should produce smaller graded p's on the
    interaction term than pure noise does."""
    def med(run):
        s = run['summary']
        s = s[s['effect'] == INTERACTION]
        return s.groupby('electrode')['best_cluster_p'].min().median()
    assert med(effect_run) < med(null_run)


def test_planted_interaction_produces_longer_clusters(null_run, effect_run):
    def max_extent(run):
        s = run['summary']
        return s[s['effect'] == INTERACTION]['best_cluster_extent'].max()
    assert max_extent(effect_run) > max_extent(null_run)


# ----------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------
def test_graded_values_round_trip_through_summary_csv(effect_run):
    on_disk = pd.read_csv(effect_run['run_dir'] / 'summary.csv')
    assert 'best_cluster_p' in on_disk.columns
    pd.testing.assert_series_equal(
        on_disk['best_cluster_p'].reset_index(drop=True),
        effect_run['summary']['best_cluster_p'].reset_index(drop=True),
        check_dtype=False)


def test_graded_values_also_saved_in_npz(effect_run):
    npz_path = effect_run['run_dir'] / ROI / 'S1' / 'e0.npz'
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as z:
        names = [str(n) for n in z['effect_names']]
        assert INTERACTION in names
        ei = names.index(INTERACTION)
        assert 0.0 <= float(z['best_cluster_p'][ei]) <= 1.0
        assert int(z['best_cluster_extent'][ei]) >= 0


def test_results_dict_carries_the_extent_threshold(effect_run):
    per_effect = effect_run['results'][ROI][('S1', 'e0')]
    info = per_effect[INTERACTION]
    assert 'extent_threshold' in info
    assert info['extent_threshold'] >= 0
    # any surviving cluster must have exceeded it
    for c in info['sig_clusters_with_sign']:
        assert c['extent'] > info['extent_threshold']


# ----------------------------------------------------------------------------
# the bridge consumes the run directly
# ----------------------------------------------------------------------------
def test_conjunction_bridge_reads_a_real_run(effect_run):
    from src.analysis.stats import power_traces_conjunction as ptc
    run_dir = effect_run['run_dir']
    lab = ptc.electrode_labels(
        {'CPC': run_dir, 'SPS': run_dir},
        interactions={'CPC': ('congruency', 'incongruentProportion'),
                      'SPS': ('congruency', 'incongruentProportion')},
        roi=ROI, correction='fdr_bh')
    assert len(lab) == 4
    assert lab['p_cpc'].notna().all()
    # same effect passed twice, so the labels must agree exactly
    assert (lab['CPC'] == lab['SPS']).all()
