"""Tests for the A3 anatomy layer.

Covers the two anatomical levels (coarse ROI groups vs raw Destrieux labels),
the ROI restriction, the electrode-id reconciliation between the A1 and
power_traces label spellings, and the group lists the brain figure is built
from. The enrichment test's behaviour (planted association detected, null not
manufacturing significance) is checked at both levels.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_anatomy as sfa


ROIS_DICT = {
    'lpfc': ['G_front_middle', 'S_front_inf', 'G_front_inf-Triangul'],
    'parietal': ['G_parietal_sup', 'S_intrapariet_and_P_trans'],
}


@pytest.fixture
def subjects_rois_dict():
    """Two subjects, the shape `make_or_load_subjects_electrodes_to_ROIs_dict` returns."""
    return {
        'D0057': {'default_dict': {
            'LTP1': 'G_front_middle',
            'LTP2': 'S_front_inf',
            'LTP3': 'G_parietal_sup',
            'LTP4': 'Left-Cerebral-White-Matter',
            'LTP5': 'G_temp_sup-Lateral',      # in no ROI group
        }},
        'D0059': {'default_dict': {
            'RTA1': 'G_front_inf-Triangul',
            'RTA2': 'S_intrapariet_and_P_trans',
            'RTA3': 'Unknown',
        }},
    }


# ---------------------------------------------------------------------------
# the two maps
# ---------------------------------------------------------------------------
def test_roi_map_groups_and_anat_map_keeps_raw_labels(subjects_rois_dict):
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, ROIS_DICT)
    e2a = sfa.build_electrode_anat_map(subjects_rois_dict)

    assert e2r['D0057-LTP1'] == 'lpfc'
    assert e2r['D0057-LTP3'] == 'parietal'
    # only the channels that fall in a known ROI group survive the grouping
    assert 'D0057-LTP5' not in e2r and 'D0057-LTP4' not in e2r

    # the anat map keeps the raw label, including labels outside any ROI group
    assert e2a['D0057-LTP1'] == 'G_front_middle'
    assert e2a['D0057-LTP5'] == 'G_temp_sup-Lateral'
    # ...but drops the non-cortical ones
    assert 'D0057-LTP4' not in e2a and 'D0059-RTA3' not in e2a


def test_roi_map_accepts_freesurfer_hemisphere_qualified_labels():
    """Newer recon exports prefix cortical parcels with ctx_lh_/ctx_rh_."""
    atlas = {
        'D0145': {'default_dict': {
            'LFM1': 'ctx_lh_G_front_middle',
            'RFI1': 'ctx_rh_G_front_inf-Triangul',
            'RAM1': 'Right-Amygdala',
        }},
    }

    e2r = sfa.build_electrode_roi_map(atlas, ROIS_DICT)
    e2a = sfa.build_electrode_anat_map(atlas)

    assert e2r == {'D0145-LFM1': 'lpfc', 'D0145-RFI1': 'lpfc'}
    # Fine-grained output remains verbatim, including hemisphere information.
    assert e2a['D0145-LFM1'] == 'ctx_lh_G_front_middle'
    assert e2a['D0145-RFI1'] == 'ctx_rh_G_front_inf-Triangul'


def test_overlapping_groups_need_the_dict_subset_first(subjects_rois_dict):
    """dlpfc is listed before lpfc and shares labels with it — first group wins.

    This is the trap the ROI restriction has to avoid: filtering `roi == 'lpfc'`
    on a map built from the FULL dict silently keeps only lpfc's exclusive
    labels. Subsetting the dict first gives lpfc its whole label list.
    """
    overlapping = {                        # same shape as config/rois.py
        'dlpfc': ['G_front_middle', 'S_front_inf'],
        'lpfc': ['G_front_middle', 'S_front_inf', 'G_front_inf-Triangul'],
    }
    full = sfa.build_electrode_roi_map(subjects_rois_dict, overlapping)
    assert full['D0057-LTP1'] == 'dlpfc'           # shared label -> first group
    assert sum(v == 'lpfc' for v in full.values()) == 1   # only the exclusive one

    lpfc_only = sfa.build_electrode_roi_map(
        subjects_rois_dict, sfa.subset_rois_dict(overlapping, 'lpfc'))
    assert lpfc_only['D0057-LTP1'] == 'lpfc'
    assert sum(v == 'lpfc' for v in lpfc_only.values()) == 3


def test_subset_rois_dict_rejects_unknown_names():
    with pytest.raises(KeyError):
        sfa.subset_rois_dict(ROIS_DICT, 'not_an_roi')


def test_anat_map_can_keep_white_matter(subjects_rois_dict):
    e2a = sfa.build_electrode_anat_map(subjects_rois_dict, drop_labels=())
    assert e2a['D0057-LTP4'] == 'Left-Cerebral-White-Matter'


# ---------------------------------------------------------------------------
# electrode-id reconciliation: A1 vs power_traces spellings
# ---------------------------------------------------------------------------
def _labels(electrode_col):
    return pd.DataFrame({
        'subject': ['D0057', 'D0057', 'D0059'],
        'electrode': electrode_col,
        'S': [1, 0, 1],
        'F': [1, 1, 0],
    })


@pytest.mark.parametrize('electrode_col', [
    ['D0057-LTP1', 'D0057-LTP3', 'D0059-RTA1'],   # A1 / assemble_long_df spelling
    ['LTP1', 'LTP3', 'RTA1'],                     # power_traces summary.csv spelling
])
def test_attach_roi_handles_both_electrode_spellings(subjects_rois_dict,
                                                     electrode_col):
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, ROIS_DICT)
    e2a = sfa.build_electrode_anat_map(subjects_rois_dict)

    out = sfa.attach_roi(_labels(electrode_col), e2r, electrodes_to_anat=e2a)

    assert list(out['roi']) == ['lpfc', 'parietal', 'lpfc']
    assert list(out['anat']) == ['G_front_middle', 'G_parietal_sup',
                                 'G_front_inf-Triangul']
    assert list(out['group']) == ['both', 'F_only', 'S_only']


def test_attach_roi_preserves_an_incoming_anova_roi_column(subjects_rois_dict):
    """power_traces labels carry the ANOVA's ROI; it must not be clobbered."""
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, ROIS_DICT)
    labels = _labels(['LTP1', 'LTP3', 'RTA1'])
    labels['roi'] = 'lpfc'                       # what electrode_labels() writes

    out = sfa.attach_roi(labels, e2r)

    assert list(out['anova_roi']) == ['lpfc', 'lpfc', 'lpfc']
    assert list(out['roi']) == ['lpfc', 'parietal', 'lpfc']   # the anatomical one


# ---------------------------------------------------------------------------
# ROI restriction
# ---------------------------------------------------------------------------
def test_restrict_to_roi_subsets_electrodes_and_subjects(subjects_rois_dict):
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, ROIS_DICT)
    e2a = sfa.build_electrode_anat_map(subjects_rois_dict)
    out = sfa.attach_roi(_labels(['LTP1', 'LTP3', 'RTA1']), e2r,
                         electrodes_to_anat=e2a)

    lpfc = sfa.restrict_to_roi(out, 'lpfc', verbose=False)

    assert set(lpfc['electrode']) == {'LTP1', 'RTA1'}
    assert lpfc.attrs['restricted_to'] == ['lpfc']
    # inside lpfc the coarse column is constant — the Destrieux one is not
    assert lpfc['roi'].nunique() == 1
    assert lpfc['anat'].nunique() == 2


def test_restrict_to_roi_accepts_a_list_and_passes_none_through(subjects_rois_dict):
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, ROIS_DICT)
    out = sfa.attach_roi(_labels(['LTP1', 'LTP3', 'RTA1']), e2r)

    both = sfa.restrict_to_roi(out, ['lpfc', 'parietal'], verbose=False)
    assert len(both) == 3
    assert sfa.restrict_to_roi(out, None) is out


# ---------------------------------------------------------------------------
# histograms at both levels
# ---------------------------------------------------------------------------
def test_histogram_counts_destrieux_labels_when_asked():
    labels, e2r, e2a = sfa._synthetic_anatomy(seed=3, return_anat=True)
    lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)
    lpfc = sfa.restrict_to_roi(lab, 'lpfc', verbose=False)

    coarse = sfa.roi_group_histogram(lpfc)
    fine = sfa.roi_group_histogram(lpfc, roi_col='anat')

    # the whole point: at the group level an lpfc-only table is ONE column
    assert list(coarse.columns) == ['lpfc']
    assert len(fine.columns) > 1
    assert all(c.startswith('lpfc_lab') for c in fine.columns)
    # no electrode is lost or double-counted by the finer split
    assert fine.to_numpy().sum() == coarse.to_numpy().sum()


def test_histogram_top_n_keeps_the_most_populated_labels():
    labels, e2r, e2a = sfa._synthetic_anatomy(seed=3, return_anat=True)
    lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)

    full = sfa.roi_group_histogram(lab, roi_col='anat')
    top2 = sfa.roi_group_histogram(lab, roi_col='anat', top_n=2)

    assert len(top2.columns) == 2
    assert list(top2.columns) == list(full.sum(axis=0)
                                      .sort_values(ascending=False).index[:2])


# ---------------------------------------------------------------------------
# coverage + the enrichment test, at both levels
# ---------------------------------------------------------------------------
def test_coverage_matrix_is_built_at_the_requested_level():
    labels, e2r, e2a = sfa._synthetic_anatomy(seed=5, return_anat=True)
    lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)

    cov_group = sfa.build_coverage_matrix(lab)
    cov_anat = sfa.build_coverage_matrix(lab, roi_col='anat')

    assert set(cov_group.columns) <= {'dlpfc', 'lpfc', 'acc', 'parietal', 'occ', 'v1'}
    assert cov_anat.shape[1] > cov_group.shape[1]
    assert cov_group.dtypes.unique().tolist() == [np.dtype(bool)]


@pytest.mark.parametrize('roi_col,restrict', [('roi', None), ('anat', 'lpfc')])
def test_enrichment_detects_planted_association_and_not_the_null(roi_col, restrict):
    """Planted association found, null not manufactured — at BOTH levels.

    The two levels are driven by their own knobs: `enrichment` plants the
    group x ROI-group association the whole-brain test should see, and
    `sublabel_enrichment` plants the group x Destrieux association the
    ROI-restricted test should see. An lpfc-only subset is a fraction of the
    electrodes, so the within-ROI signal is planted more strongly to have
    comparable power.
    """
    for planted in (True, False):
        labels, e2r, e2a = sfa._synthetic_anatomy(
            n_subj=24, seed=7,
            enrichment=0.6 if planted else 0.0,
            sublabel_enrichment=0.9 if planted else 0.0,
            return_anat=True)
        lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)
        if restrict:
            lab = sfa.restrict_to_roi(lab, restrict, verbose=False)
        cov = sfa.build_coverage_matrix(lab, roi_col=roi_col)
        res = sfa.roi_group_enrichment_test(lab, cov, min_subjects=3,
                                            n_perm=2000, roi_col=roi_col, seed=0)

        assert res['roi_col'] == roi_col
        if planted:
            assert res['p'] < 0.05, f"{roi_col}: planted association missed"
        else:
            assert res['p'] > 0.05, f"{roi_col}: null manufactured significance"


def test_enrichment_returns_a_well_formed_null_result_when_degenerate():
    """A single ROI column can't support a group x ROI test — say so, don't crash."""
    labels, e2r, e2a = sfa._synthetic_anatomy(seed=1, return_anat=True)
    lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)
    lpfc = sfa.restrict_to_roi(lab, 'lpfc', verbose=False)
    cov = sfa.build_coverage_matrix(lpfc)          # coarse: one column

    res = sfa.roi_group_enrichment_test(lpfc, cov, min_subjects=3, n_perm=100)

    assert res['p'] == 1.0
    assert 'note' in res
    assert list(res['contingency'].index) == ['both', 'S_only', 'F_only']


# ---------------------------------------------------------------------------
# the group lists the brain figure is drawn from
# ---------------------------------------------------------------------------
def test_group_electrodes_by_subject_strips_the_subject_prefix():
    labels = pd.DataFrame({
        'subject': ['D0057', 'D0057', 'D0059', 'D0059'],
        'electrode': ['D0057-LTP1', 'D0057-LTP2', 'RTA1', 'RTA2'],
        'S': [1, 1, 0, 0],
        'F': [1, 0, 1, 0],
    })
    lab = sfa.attach_roi(labels, {})            # no ROI map needed for this

    by_subject = sfa.group_electrodes_by_subject(lab)

    assert by_subject['both'] == {'D0057': ['LTP1']}
    assert by_subject['S_only'] == {'D0057': ['LTP2']}
    assert by_subject['F_only'] == {'D0059': ['RTA1']}
    # groups are disjoint: every selective electrode appears exactly once
    seen = [ch for grp in by_subject.values()
            for chans in grp.values() for ch in chans]
    assert len(seen) == len(set(seen)) == 3


def test_brain_plot_falls_back_to_a_histogram_without_the_surface_stack(tmp_path):
    """Off-cluster (no mne/pyvista/recons) the job must still produce a figure."""
    labels, e2r, e2a = sfa._synthetic_anatomy(seed=2, return_anat=True)
    lab = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)

    out = sfa.plot_selectivity_groups_on_brain(
        lab, str(tmp_path / 'selectivity_groups_on_brain.png'))

    if out['fallback']:
        assert out['combined'].endswith('_roi_hist.png')
        assert (tmp_path / 'selectivity_groups_on_brain_roi_hist.png').exists()
    else:                                        # a full stack is installed
        assert (tmp_path / 'selectivity_groups_on_brain.png').exists()


# ---------------------------------------------------------------------------
# the CONTINUOUS arm (plan §5–§7): scores -> anatomy
# ---------------------------------------------------------------------------
@pytest.fixture
def planted_scores():
    """Scores with a planted anatomy x effect-type interaction (+ the null)."""
    def _make(gradient, seed=3):
        scores, e2r, e2a, e2c = sfa._synthetic_scores(n_subj=14, gradient=gradient,
                                                      seed=seed)
        tab = sfa.attach_scores(scores, e2r, electrodes_to_anat=e2a,
                                electrodes_to_coords=e2c)
        return tab, sfa.build_coverage_matrix(tab)
    return _make


def test_attach_scores_pools_the_scaling_and_keeps_tiny_subjects():
    """ONE scale factor per effect, pooled — not a within-subject z-score.

    The failure mode the plan names: a within-subject z forces a 2-electrode
    subject to exactly +/-0.707 and drops a 1-electrode subject (SD is NaN).
    Pooled scaling has to survive both, and has to preserve the RATIOS between
    electrodes, which is what the maps and the anatomy model read.
    """
    scores = pd.DataFrame({
        'subject': ['A', 'A', 'B', 'B', 'C'],          # C has one electrode
        'electrode': ['A-1', 'A-2', 'B-1', 'B-2', 'C-1'],
        'x': [2.0, 1.0, -1.0, 0.5, 4.0],
        'y': [0.0, 1.0, 1.0, -0.5, 1.0],
    })
    tab = sfa.attach_scores(scores, {}, electrodes_to_coords=None)

    assert len(tab) == 5                                # nobody is dropped
    sd = scores['x'].std(ddof=1)
    assert np.allclose(tab['lwpc_s'], scores['x'] / sd)
    # one factor for the whole column => ratios between electrodes are untouched
    assert np.isclose(tab['lwpc_s'].iloc[0] / tab['lwpc_s'].iloc[1],
                      scores['x'].iloc[0] / scores['x'].iloc[1])
    assert np.allclose(tab['delta'], tab['lwpc_s'] - tab['lwps_s'])
    assert np.allclose(tab['abs_lwpc'], tab['lwpc_s'].abs())


def test_attach_scores_joins_anatomy_and_both_electrode_spellings():
    scores = pd.DataFrame({
        'subject': ['D0057', 'D0059'],
        'electrode': ['D0057-LTP1', 'RTA1'],            # prefixed and bare
        'x': [1.0, -1.0], 'y': [0.5, 0.25],
    })
    tab = sfa.attach_scores(
        scores, {'D0057-LTP1': 'lpfc', 'D0059-RTA1': 'parietal'},
        electrodes_to_anat={'D0057-LTP1': 'G_front_middle'},
        electrodes_to_coords={'D0057-LTP1': (-40., 30., 20.),
                              'D0059-RTA1': (45., -50., 10.)})

    assert tab['roi'].tolist() == ['lpfc', 'parietal']
    assert tab['anat'].tolist()[0] == 'G_front_middle'
    assert pd.isna(tab['anat'].tolist()[1])
    assert tab['hemi'].tolist() == ['lh', 'rh']         # sign of x
    assert tab['mni_y'].tolist() == [30., -50.]


@pytest.mark.parametrize('gradient,planted', [(0.8, True), (0.0, False)])
def test_relative_score_roi_test_finds_planted_anatomy_and_not_the_null(
        planted_scores, gradient, planted):
    tab, cover = planted_scores(gradient)

    res = sfa.relative_score_roi_test(tab, cover, min_subjects=3, n_perm=1000,
                                      seed=0)

    assert res['n_electrodes'] == len(tab.dropna(subset=['delta']))
    if planted:
        assert res['p'] < 0.05
        per_roi = res['per_roi'].set_index('roi')['mean_delta_adj']
        # LWPC was planted anteriorly, LWPS posteriorly, both negative:
        # delta = lwpc - lwps is therefore NEGATIVE in front, POSITIVE behind
        assert per_roi[['dlpfc', 'lpfc', 'acc']].max() < 0
        assert per_roi[['occ', 'parietal', 'v1']].min() > 0
    else:
        assert res['p'] > 0.05


def test_roi_test_is_invariant_to_the_sign_of_delta(planted_scores):
    """The swap null IS a sign flip of delta, so the test cannot prefer a direction.

    Swapping an electrode's LWPC and LWPS scores negates delta. That makes the
    null exactly symmetric, and it means relabelling which effect is 'first'
    cannot change the answer — worth pinning, because a statistic that failed
    this would be reading effect-type order rather than anatomy.
    """
    tab, cover = planted_scores(0.8)
    flipped = tab.copy()
    flipped['delta'] = -flipped['delta']

    a = sfa.relative_score_roi_test(tab, cover, n_perm=300, seed=0)
    b = sfa.relative_score_roi_test(flipped, cover, n_perm=300, seed=0)

    assert np.isclose(a['observed_stat'], b['observed_stat'])
    assert a['p'] == b['p']
    assert np.allclose(a['per_roi']['mean_delta_adj'],
                       -b['per_roi']['mean_delta_adj'])


def test_roi_test_drops_rois_below_the_coverage_threshold(planted_scores):
    tab, cover = planted_scores(0.8)
    per_roi_cov = cover.sum(axis=0)
    threshold = int(per_roi_cov.max())               # keeps at most a few ROIs

    res = sfa.relative_score_roi_test(tab, cover, min_subjects=threshold,
                                      n_perm=200, seed=0)

    assert all(per_roi_cov[r] >= threshold for r in res['rois_tested'])
    assert len(res['rois_tested']) < cover.shape[1]


def test_roi_test_returns_a_well_formed_result_when_degenerate(planted_scores):
    tab, cover = planted_scores(0.8)
    res = sfa.relative_score_roi_test(tab, cover, min_subjects=999, n_perm=100)

    assert res['p'] == 1.0 and 'note' in res
    assert res['per_roi'].empty


@pytest.mark.parametrize('gradient,planted', [(0.8, True), (0.0, False)])
def test_coordinate_test_recovers_the_planted_axis(planted_scores, gradient,
                                                   planted):
    tab, _ = planted_scores(gradient)

    res = sfa.relative_score_coordinate_test(tab, n_perm=1000, seed=0)

    assert set(res) == {'all', 'lh', 'rh'}
    y = res['all']['slopes'].set_index('axis').loc['mni_y']
    if planted:
        assert res['all']['p'] < 0.05
        # delta falls as y rises (LWPC dominance is anterior and delta is
        # negative there), so the anterior slope must be negative and reliable
        assert y['slope_per_mm'] < 0 and y['p'] < 0.05
    else:
        assert res['all']['p'] > 0.05


def test_map_reliability_ceiling_falls_as_noise_rises():
    """The ceiling has to behave like a ceiling, or it cannot license a null."""
    scores, e2r, _, _ = sfa._synthetic_scores(n_subj=10, gradient=0.8, seed=6)
    clean = sfa.map_reliability(sfa._synthetic_per_split(scores, noise=0.3, seed=1))
    noisy = sfa.map_reliability(sfa._synthetic_per_split(scores, noise=2.0, seed=1))

    assert clean['reliability_lwpc'] > noisy['reliability_lwpc']
    assert clean['reliability_lwps'] > noisy['reliability_lwps']
    assert clean['reliability_lwpc'] <= 1.0
    # ...and the parcel-level version aggregates to one value per ROI
    parcel = sfa.map_reliability(sfa._synthetic_per_split(scores, noise=0.3, seed=1),
                                 parcels=e2r)
    assert parcel['unit'] == 'parcel'
    assert parcel['n_units'] == len(set(e2r.values()))


def test_score_centers_are_real_electrodes_and_use_the_swap_null(planted_scores):
    tab, _ = planted_scores(0.8)

    res = sfa.score_centers_per_subject(tab, n_perm=500, seed=0, min_elec=3)

    assert res['center'] == 'medoid'
    assert res['n_groups'] > 0
    # planted: |LWPC| is larger anteriorly, so its centre sits anterior to LWPS's
    assert res['mean_displacement']['dy'] > 0 and res['p']['dy'] < 0.05
    # every group is one subject x one hemisphere, as the plan requires
    assert set(res['per_group']['hemi']) <= {'lh', 'rh'}
    assert (res['per_group']['n_electrodes'] >= 3).all()

    null = sfa.score_centers_per_subject(planted_scores(0.0)[0], n_perm=500,
                                         seed=0, min_elec=3)
    assert null['p']['dy'] > 0.05


def test_leave_one_subject_out_covers_every_subject(planted_scores):
    tab, cover = planted_scores(0.8)

    sweep = sfa.leave_one_subject_out(
        lambda t: sfa.relative_score_roi_test(t, cover, n_perm=100, seed=0), tab)

    assert sweep['dropped'].iloc[0] == '(none)'
    assert len(sweep) == tab['subject'].nunique() + 1
    assert {'observed_stat', 'p', 'n_electrodes'} <= set(sweep.columns)
    # each fold really drops a subject's electrodes
    assert (sweep['n_electrodes'].iloc[1:] < sweep['n_electrodes'].iloc[0]).all()


def test_score_map_falls_back_to_the_by_roi_figure_without_the_surface_stack(
        planted_scores, tmp_path):
    tab, cover = planted_scores(0.8)

    out = sfa.plot_scores_on_brain(tab, str(tmp_path / 'delta_map.png'),
                                   value_col='delta', coverage=cover)

    assert os.path.exists(out['colorbar'])          # written either way
    if out['fallback']:
        assert out['combined'].endswith('_by_roi.png')
        assert os.path.exists(out['combined'])
    else:
        assert os.path.exists(str(tmp_path / 'delta_map.png'))
