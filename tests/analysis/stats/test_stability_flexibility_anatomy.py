"""Tests for the A3 anatomy layer.

Covers the two anatomical levels (coarse ROI groups vs raw Destrieux labels),
the ROI restriction, the electrode-id reconciliation between the A1 and
power_traces label spellings, and the group lists the brain figure is built
from. The enrichment test's behaviour (planted association detected, null not
manufacturing significance) is checked at both levels.
"""

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
