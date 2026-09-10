"""Tests for the joint scatter (docs/analysis_simplification_plan.md 2.5).

The figure's job is to make the plan's five readings visible, and the fifth --
"all the structure in one colour or a few points -> artifact" -- is the one that
has to be a number rather than an impression. So most of these tests build a
cloud whose structure is known to sit in one subject, or in three electrodes, or
between subjects rather than within them, and check that the corresponding
diagnostic catches it.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.stats import segregation_scatter as scat


# ----------------------------------------------------------------------------
# fixtures: electrode tables with known joint structure
# ----------------------------------------------------------------------------
def _elec(x, y, subjects, prefix='e'):
    return pd.DataFrame(dict(
        subject=list(subjects), x=np.asarray(x, float), y=np.asarray(y, float),
        electrode=[f"{s}-{prefix}{i}" for i, s in enumerate(subjects)]))


def _independent(n_subj=6, n_elec=25, seed=0):
    """No relationship on either axis: the 'independent mechanisms' reading."""
    rng = np.random.default_rng(seed)
    subs = np.repeat([f"S{i:02d}" for i in range(n_subj)], n_elec)
    n = len(subs)
    return _elec(rng.normal(0, 1, n), rng.normal(0, 1, n), subs)


def _shared(n_subj=6, n_elec=25, rho=0.8, seed=0):
    """Positive diagonal: the 'shared mechanism' reading."""
    rng = np.random.default_rng(seed)
    subs = np.repeat([f"S{i:02d}" for i in range(n_subj)], n_elec)
    n = len(subs)
    x = rng.normal(0, 1, n)
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    return _elec(x, y, subs)


# ----------------------------------------------------------------------------
# the diagnostics
# ----------------------------------------------------------------------------
class TestJointScatterDiagnostics:

    def test_independent_cloud_is_flat_and_unflagged(self):
        d = scat.joint_scatter_diagnostics(_independent())
        assert abs(d['corr']) < 0.15
        assert d['n_electrodes'] == 150 and d['n_subjects'] == 6
        # nothing to flag: no subject dominates, no point carries the (absent)
        # structure, and there is no between-subject offset
        assert d['flags'] == []

    def test_shared_cloud_recovers_the_diagonal(self):
        d = scat.joint_scatter_diagnostics(_shared(rho=0.8))
        assert d['corr'] > 0.6
        # real, distributed structure survives dropping the extreme points and
        # survives within-subject centring
        assert d['corr_drop_top'] > 0.5
        assert d['corr_within_subject'] > 0.5
        assert d['flags'] == []

    def test_opponent_cloud_is_negative(self):
        d = scat.joint_scatter_diagnostics(_shared(rho=-0.8))
        assert d['corr'] < -0.6

    def test_structure_in_one_subject_is_flagged(self):
        """One subject carries a strong diagonal; everyone else is noise. The
        pooled correlation looks real; leave-one-subject-out must expose it."""
        rng = np.random.default_rng(3)
        base = _independent(n_subj=4, n_elec=15, seed=5)
        n = 60
        x = rng.normal(0, 3, n)
        hot = pd.DataFrame(dict(subject='HOT', x=x, y=x + rng.normal(0, .2, n),
                                electrode=[f"HOT-e{i}" for i in range(n)]))
        d = scat.joint_scatter_diagnostics(pd.concat([base, hot], ignore_index=True))

        assert d['corr'] > 0.3
        assert d['most_influential_subject'] == 'HOT'
        # removing HOT collapses it
        assert abs(d['most_influential_subject_delta']) > 0.5 * abs(d['corr'])
        assert any('HOT' in f for f in d['flags'])
        per_sub = d['per_subject'].set_index('subject')
        assert per_sub.loc['HOT', 'corr'] > 0.9

    def test_structure_in_a_few_points_is_flagged(self):
        """Three far-out electrodes on the diagonal, everything else round.

        Read with Pearson, which is where this pathology actually bites: three
        points at (8,8)-(10,10) drag r from ~0 to ~0.7 on their own."""
        base = _independent(n_subj=5, n_elec=20, seed=2)
        out = _elec([8., 9., 10.], [8., 9., 10.], ['S00', 'S01', 'S02'], prefix='out')
        e = pd.concat([base, out], ignore_index=True)
        d = scat.joint_scatter_diagnostics(e, method='pearson', n_top=3)

        assert d['corr'] > 0.4
        assert d['n_top_dropped'] == 3
        assert abs(d['corr_drop_top']) < 0.5 * abs(d['corr'])
        assert any('most influential electrodes' in f for f in d['flags'])

    def test_spearman_resists_the_same_few_points(self):
        """The default is Spearman precisely because ranks blunt this: the same
        three outliers move it far less, so the flag does not fire. Worth
        knowing when comparing the two -- a big Pearson/Spearman gap is itself
        the outlier signal."""
        base = _independent(n_subj=5, n_elec=20, seed=2)
        out = _elec([8., 9., 10.], [8., 9., 10.], ['S00', 'S01', 'S02'], prefix='out')
        e = pd.concat([base, out], ignore_index=True)
        r_spear = scat.joint_scatter_diagnostics(e, method='spearman')['corr']
        r_pears = scat.joint_scatter_diagnostics(e, method='pearson')['corr']
        assert r_pears > r_spear + 0.3

    def test_between_subject_structure_is_flagged(self):
        """Each subject is internally flat, but subjects sit on a diagonal: a
        subject-level offset, not electrode-level co-localization. The pipeline
        centres this out; the scatter has to say so rather than hide it."""
        rng = np.random.default_rng(7)
        rows = []
        for i in range(8):
            offset = (i - 3.5) * 2.0
            for e in range(20):
                rows.append(dict(subject=f"S{i:02d}", electrode=f"S{i:02d}-e{e}",
                                 x=offset + rng.normal(0, .5),
                                 y=offset + rng.normal(0, .5)))
        d = scat.joint_scatter_diagnostics(pd.DataFrame(rows))

        assert d['corr'] > 0.8
        assert abs(d['corr_within_subject']) < 0.2
        assert any('BETWEEN subjects' in f for f in d['flags'])

    def test_one_subject_share_is_flagged(self):
        big = _independent(n_subj=1, n_elec=100, seed=1)
        big['subject'] = 'BIG'
        small = _independent(n_subj=3, n_elec=5, seed=2)
        d = scat.joint_scatter_diagnostics(pd.concat([big, small], ignore_index=True))
        assert d['max_subject_share'] > 0.8
        assert any('contributes' in f for f in d['flags'])

    def test_nans_are_dropped_not_propagated(self):
        e = _shared(rho=0.8)
        e.loc[e.index[:10], 'x'] = np.nan
        e.loc[e.index[10:20], 'y'] = np.nan
        d = scat.joint_scatter_diagnostics(e)
        assert d['n_electrodes'] == len(e) - 20
        assert np.isfinite(d['corr'])

    def test_too_few_electrodes_degrades_gracefully(self):
        d = scat.joint_scatter_diagnostics(_elec([1., 2.], [1., 2.], ['A', 'A']))
        assert not np.isfinite(d['corr'])
        assert d['flags'] == ["correlation undefined (too few electrodes)"]

    def test_pearson_and_spearman_both_available(self):
        e = _shared(rho=0.7)
        rs = scat.joint_scatter_diagnostics(e, method='spearman')['corr']
        rp = scat.joint_scatter_diagnostics(e, method='pearson')['corr']
        assert rs > 0.4 and rp > 0.4
        assert rs != rp

    def test_diagnostics_are_json_serialisable(self):
        import json
        d = scat.joint_scatter_diagnostics(_shared())
        blob = json.dumps(scat.diagnostics_to_json(d))
        back = json.loads(blob)
        assert isinstance(back['per_subject'], list)
        assert back['per_subject'][0]['subject'] == 'S00'
        assert back['n_electrodes'] == d['n_electrodes']


# ----------------------------------------------------------------------------
# the figure itself
# ----------------------------------------------------------------------------
class TestPlotJointScatter:

    def teardown_method(self):
        plt.close('all')

    def test_figure_has_scatter_marginals_and_subject_colours(self):
        e = _shared(n_subj=4, n_elec=20)
        fig, d = scat.plot_joint_scatter(e)

        # main axes + two marginals + the per-subject panel
        assert len(fig.axes) == 4
        main = fig.axes[0]
        # one PathCollection per subject, and every subject a different colour
        cols = main.collections
        assert len(cols) == 4
        rgba = [tuple(np.ravel(c.get_facecolor())[:3]) for c in cols]
        assert len(set(rgba)) == 4
        # every electrode is drawn exactly once
        assert sum(c.get_offsets().shape[0] for c in cols) == len(e)
        # marginals are histograms (patches), one stack per subject
        assert len(fig.axes[1].patches) > 0
        assert len(fig.axes[2].patches) > 0
        assert d['n_subjects'] == 4

    def test_axis_labels_follow_contrast_mode(self):
        e = _shared(n_subj=3, n_elec=10)
        fig, _ = scat.plot_joint_scatter(e, contrast_mode='proportion')
        assert 'LWPC' in fig.axes[0].get_xlabel()
        assert 'LWPS' in fig.axes[0].get_ylabel()
        plt.close(fig)

        fig, _ = scat.plot_joint_scatter(e, contrast_mode='condition')
        assert 'stability' in fig.axes[0].get_xlabel()
        assert 'flexibility' in fig.axes[0].get_ylabel()

    def test_effect_measure_is_shown_on_the_axes(self):
        fig, _ = scat.plot_joint_scatter(_shared(n_subj=3, n_elec=10),
                                         effect_measure='cluster')
        assert 'cluster' in fig.axes[0].get_xlabel()

    def test_saves_a_png(self, tmp_path):
        p = tmp_path / 'joint_scatter.png'
        fig, _ = scat.plot_joint_scatter(_shared(n_subj=3, n_elec=10),
                                         save_path=str(p))
        assert p.exists() and p.stat().st_size > 0

    def test_precomputed_diagnostics_are_reused(self):
        e = _shared(n_subj=3, n_elec=10)
        d0 = scat.joint_scatter_diagnostics(e)
        fig, d1 = scat.plot_joint_scatter(e, diagnostics=d0)
        assert d1 is d0

    def test_empty_table_raises(self):
        e = _shared(n_subj=3, n_elec=10)
        e['x'] = np.nan
        with pytest.raises(ValueError, match="no electrode"):
            scat.plot_joint_scatter(e)

    def test_many_subjects_get_distinct_colours(self):
        """This dataset has 24 subjects, past the 20 of a single tab20."""
        pal = scat.subject_palette([f"S{i:02d}" for i in range(24)])
        assert len(pal) == 24
        assert len({tuple(c) for c in pal.values()}) == 24


# ----------------------------------------------------------------------------
# the long-table -> sensitivities route
# ----------------------------------------------------------------------------
def _long_df(n_subj=4, n_elec=6, n_trials=160, rho=0.9, seed=0):
    """Trial table with a congruency x proportion and switch x proportion
    interaction whose per-electrode strengths are correlated by `rho`."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_subj):
        cong = rng.choice(['c', 'i'], n_trials)
        sw = rng.choice(['s', 'r'], n_trials)
        ip = rng.choice([25.0, 75.0], n_trials)
        sp = rng.choice([25.0, 75.0], n_trials)
        for e in range(n_elec):
            bx = rng.normal(0, 1)
            by = rho * bx + np.sqrt(1 - rho ** 2) * rng.normal(0, 1)
            base = (bx * (cong == 'i') * (ip == 75.0)
                    + by * (sw == 's') * (sp == 75.0))
            frames.append(pd.DataFrame(dict(
                subject=f"S{s:02d}", electrode=f"S{s:02d}-e{e}",
                congruency=cong, switchType=sw,
                incongruent_proportion=ip, switch_proportion=sp,
                hg=2.0 * base + rng.normal(0, 1, n_trials))))
    return pd.concat(frames, ignore_index=True)


class TestSensitivitiesForScatter:

    def test_naive_route_returns_one_row_per_electrode(self):
        df = _long_df()
        e = scat.sensitivities_for_scatter(df, contrast_mode='proportion')
        assert set(e.columns) >= {'subject', 'electrode', 'x', 'y'}
        assert len(e) == df.electrode.nunique()

    def test_naive_route_recovers_an_injected_positive_diagonal(self):
        e = scat.sensitivities_for_scatter(_long_df(rho=0.9, n_elec=10),
                                           contrast_mode='proportion')
        assert scat.joint_scatter_diagnostics(e)['corr'] > 0.3

    def test_split_route_also_runs_and_agrees_in_sign(self):
        df = _long_df(rho=0.9, n_elec=8, n_trials=240, seed=4)
        naive = scat.sensitivities_for_scatter(df, contrast_mode='proportion')
        split = scat.sensitivities_for_scatter(df, contrast_mode='proportion',
                                               n_splits=4, seed=1)
        assert len(split) == len(naive)
        r_naive = scat.joint_scatter_diagnostics(naive)['corr']
        r_split = scat.joint_scatter_diagnostics(split)['corr']
        assert r_naive > 0 and r_split > 0

    def test_condition_mode_runs(self):
        e = scat.sensitivities_for_scatter(_long_df(), contrast_mode='condition')
        assert len(e) and e[['x', 'y']].notna().any().any()
