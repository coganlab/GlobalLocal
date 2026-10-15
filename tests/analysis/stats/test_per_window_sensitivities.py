"""`effect_measure='mean_diff'` and `compute_sensitivities_per_window`.

The coding axes of the subspace plan (docs/analysis_plans.md, subspace plan
§3.1) are per-electrode differences of condition means, with the proportion
blocks weighted equally, scored in every A4 window on the same trial halves.
These tests pin the three things that definition needs:

  * `mean_diff` is the equal-weight average of the within-block differences,
    in HG units, not the trial-count-weighted pooled difference;
  * every window is scored on the same halves, so windows differ only by HG;
  * the window grid (size, step, dropped remainder, A4 centre convention).
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_segregation as sfs

KW = dict(contrast_mode='proportion', main_effects=True, shared_split=True,
          effect_measure='mean_diff')


def _trials():
    """One participant's trials: every congruency x incProp x switchType x
    switchProp cell, with incongruent trials 3x as frequent in the 75% block and
    3x as rare in the 25% block, as the proportion design makes them."""
    rows = []
    for inc_prop, n_i, n_c in ((25.0, 4, 12), (75.0, 12, 4)):
        for cong, n in (('i', n_i), ('c', n_c)):
            for sw in ('s', 'r'):
                for sw_prop in (25.0, 75.0):
                    rows += [dict(congruency=cong, switchType=sw,
                                  incongruent_proportion=inc_prop,
                                  switch_proportion=sw_prop)] * n
    t = pd.DataFrame(rows)
    t['trial'] = np.arange(len(t))
    return t


def _long(hg_of, n_elec=3):
    """The long table: the same trials on `n_elec` electrodes of one participant,
    with `hg_of(trials, electrode_index)` giving each row's hg."""
    t = _trials()
    return pd.concat([t.assign(subject='S1', electrode=f'S1-e{e}', hg=hg_of(t, e))
                      for e in range(n_elec)], ignore_index=True)


def _cell_hg(t, e=0):
    """Noise-free HG with cell means c25 = 0, i25 = 1, c75 = 2, i75 = 6."""
    i, hi = (t.congruency == 'i').to_numpy(), (t.incongruent_proportion == 75).to_numpy()
    return 1.0 * i + 2.0 * hi + 3.0 * (i & hi)


def test_mean_diff_weights_the_blocks_equally():
    per = sfs.compute_sensitivities_per_split(_long(_cell_hg), n_splits=3, seed=0, **KW)

    # 0.5 (i25 - c25) + 0.5 (i75 - c75) = 0.5 * 1 + 0.5 * 4. The pooled i - c
    # difference would be 4.25: the frequent i75 cell dominates the i side.
    assert np.allclose(per[['mxA', 'mxB']], 2.5)
    assert np.allclose(per[['myA', 'myB']], 0.0)      # HG ignores switch type


def test_mean_diff_needs_window_means():
    df = _long(lambda t, e: list(np.tile(_cell_hg(t)[:, None], (1, 5))))
    with pytest.raises(ValueError, match='window-mean'):
        sfs.compute_sensitivities_per_split(df, n_splits=1, seed=0, **KW)


def test_every_window_uses_the_same_halves():
    rng = np.random.default_rng(0)
    df = _long(lambda t, e: _cell_hg(t) + rng.normal(0, 1, len(t)))
    # Each trial's time course is its scalar HG in every sample, so any
    # difference between windows could only come from a different split.
    courses = df.assign(hg=list(np.tile(df['hg'].to_numpy()[:, None], (1, 40))))
    per_w = sfs.compute_sensitivities_per_window(
        courses, np.arange(40) / 256, window_size=16, step_size=8,
        n_splits=4, seed=3, **KW)
    single = sfs.compute_sensitivities_per_split(df, n_splits=4, seed=3, **KW)

    cols = ['mxA', 'mxB', 'myA', 'myB', 'xA', 'xB', 'yA', 'yB']
    assert per_w['window'].nunique() == 4
    for _, w in per_w.groupby('window'):
        assert np.allclose(w[cols].to_numpy(), single[cols].to_numpy())


def test_window_grid_and_where_the_effect_lands():
    n_time, sfreq = 70, 256.0
    times = 0.5 + np.arange(n_time) / sfreq

    def courses(t, e):
        tc = np.zeros((len(t), n_time))
        tc[:, 32:48] = _cell_hg(t)[:, None]            # the effect, in window 2 only
        return list(tc)

    per = sfs.compute_sensitivities_per_window(
        _long(courses), times, window_size=16, step_size=16, n_splits=2, seed=0, **KW)
    grid = per.drop_duplicates('window').set_index('window')

    # starts 0, 16, 32, 48; the last 6 samples do not fill a window
    assert list(grid.index) == [0, 1, 2, 3]
    assert np.allclose(grid['window_tmin'], times[[0, 16, 32, 48]])
    assert np.allclose(grid['window_tmax'], times[[15, 31, 47, 63]])
    assert np.allclose(grid['window_center'], times[[0, 16, 32, 48]] + 8 / sfreq)
    m = per.groupby('window')['mxA'].mean()
    assert np.isclose(m[2], 2.5) and np.allclose(m.drop(2), 0.0)


def test_per_window_refuses_time_resolved_measures():
    df = _long(lambda t, e: list(np.zeros((len(t), 20))))
    with pytest.raises(ValueError, match='SCALAR|mean_diff'):
        sfs.compute_sensitivities_per_window(
            df, np.arange(20) / 256, 16, 16, n_splits=1, seed=0,
            **dict(KW, effect_measure='cluster'))
