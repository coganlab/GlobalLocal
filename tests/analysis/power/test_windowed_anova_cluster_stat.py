"""`cluster_stat` selects how a candidate cluster is scored.

'extent' counts suprathreshold windows and is magnitude-blind: with
window_size=64 / step_size=16 the windows overlap 75%, so one 250 ms effect
already spans four of them and a long weak cluster outscores a short strong one.
'mass' sums each window's excess of F over the threshold.

Default stays 'extent' so existing runs reproduce. Note that neither choice is
what produces a cluster spanning the pre-stimulus baseline -- a sustained offset
wins under both (see test_block_balanced_anova_conditions.py for the cause).
"""

import numpy as np
import pytest

from src.analysis.power.windowed_anova import run_windowed_anova_cluster_correction


ROI = 'lpfc'
WINDOW_SIZE, STEP_SIZE, N_WINDOWS = 4, 2, 8
N_TIMES = WINDOW_SIZE + STEP_SIZE * (N_WINDOWS - 1)
TIMES = np.linspace(-1.0, 1.5, N_TIMES)
SUBJECTS = ['S1', 'S2']
N_ELEC = 3

CONDITIONS = {
    'c25': {'congruency': 'c', 'incongruentProportion': '25%'},
    'c75': {'congruency': 'c', 'incongruentProportion': '75%'},
    'i25': {'congruency': 'i', 'incongruentProportion': '25%'},
    'i75': {'congruency': 'i', 'incongruentProportion': '75%'},
}
FACTORS = ['congruency', 'incongruentProportion']
INTERACTION = 'C(congruency):C(incongruentProportion)'


def _windowed(seed=0):
    rng = np.random.default_rng(seed)
    return {c: {ROI: [rng.normal(0, 0.1, size=(8, N_WINDOWS, N_ELEC))
                      for _ in SUBJECTS]}
            for c in CONDITIONS}


def _run(**kw):
    elecs = {ROI: {s: [f'{s}_e{j}' for j in range(N_ELEC)] for s in SUBJECTS}}
    results, _ = run_windowed_anova_cluster_correction(
        _windowed(), CONDITIONS, FACTORS, [ROI], SUBJECTS,
        electrodes_per_subject_roi=elecs, times=TIMES,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE, sampling_rate=256,
        n_perm=60, percentile=95, cluster_percentile=95,
        seed=7, n_jobs=1, verbose=False, **kw,
    )
    return results[ROI]


def test_mass_is_accepted_and_records_the_value_it_thresholded():
    res = _run(cluster_stat='mass')
    assert INTERACTION in res
    for c in res[INTERACTION]['sig_clusters_with_sign']:
        assert 'cluster_stat' in c and 'extent' in c
        # mass is the summed excess over threshold, not a window count
        assert c['cluster_stat'] != c['extent'] or c['extent'] == 0


def test_extent_is_the_default():
    default = _run()
    explicit = _run(cluster_stat='extent')
    np.testing.assert_array_equal(default[INTERACTION]['window_mask'],
                                  explicit[INTERACTION]['window_mask'])


def test_unknown_cluster_stat_raises():
    with pytest.raises(ValueError, match="cluster_stat must be"):
        _run(cluster_stat='bogus')
