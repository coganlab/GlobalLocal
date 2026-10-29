"""Electrode-set logic behind the significant-electrode brain figures.

Only the pure set logic is tested here; nothing is rendered. Skipped wherever
the surface stack (mne + ieeg + pyvista) isn't installed.
"""
from collections import OrderedDict

import pytest

plot_sig = pytest.importorskip("dcc_scripts.vis.plot_sig_electrodes_dcc",
                               reason="needs the mne/ieeg surface stack")

LPFC = {"lpfc": ["G_front_middle"]}

# One kept subject and one dropped subject. 'A4' has an LPFC label but was
# dropped during epoching, so it must never show up.
ROI_MAP = {
    "D0001": {"filtROI_dict": {"ctx_lh_G_front_middle": ["A1", "A4"],
                               "ctx_lh_G_temporal_middle": ["A2"]}},
    "D0099": {"filtROI_dict": {"ctx_lh_G_front_middle": ["Z1"]}},
}
ANALYZED = {"D0001": ["A1", "A2", "A3"]}   # A3 is in white matter
SIGNIFICANT = {"D0001": ["A1", "A3"]}


@pytest.fixture
def fake_loaders(monkeypatch):
    def fake(chans):
        return lambda subjects, root, task=None, LAB_root=None: {
            s: chans.get(s, []) for s in subjects}
    monkeypatch.setattr(plot_sig, "get_analyzed_chans_per_subject", fake(ANALYZED))
    monkeypatch.setattr(plot_sig, "get_sig_chans_per_subject", fake(SIGNIFICANT))


def electrodes(cfg, rois_dict=None):
    out = plot_sig.get_condition_electrodes(
        ["D0001"], cfg, task="GlobalLocal", LAB_root=None, rois_dict=rois_dict,
        subjects_electrodes_to_ROIs_dict=ROI_MAP)
    return {subject: elecs for subject, elecs in out.items() if elecs}


def test_all_chans_whole_brain_includes_white_matter(fake_loaders):
    cfg = {"epochs_root_file": "root", "all_chans": True, "rois_dict": None}
    assert electrodes(cfg) == {"D0001": ["A1", "A2", "A3"]}


def test_all_chans_in_roi_drops_unanalyzed_channels_and_subjects(fake_loaders):
    cfg = {"epochs_root_file": "root", "all_chans": True, "rois_dict": LPFC}
    assert electrodes(cfg) == {"D0001": ["A1"]}


def test_condition_roi_overrides_figure_roi(fake_loaders):
    cfg = {"epochs_root_file": "root", "rois_dict": None}
    assert electrodes(cfg, rois_dict=LPFC) == {"D0001": ["A1", "A3"]}


def test_condition_without_roi_falls_back_to_figure_roi(fake_loaders):
    cfg = {"epochs_root_file": "root"}
    assert electrodes(cfg, rois_dict=LPFC) == {"D0001": ["A1"]}


def test_assign_to_last_group_draws_each_electrode_once():
    nested = OrderedDict([("all", {1, 2, 3, 4}), ("lpfc", {3, 4}),
                          ("sig", {2, 4}), ("sig_lpfc", {4})])

    groups = plot_sig.assign_to_last_group(nested)

    assert groups == OrderedDict([("all", [1]), ("lpfc", [3]),
                                  ("sig", [2]), ("sig_lpfc", [4])])


def test_all_recon_returns_every_localized_contact(monkeypatch):
    recon = {"D1": ["A1", "A2", "BAD1"]}
    monkeypatch.setattr(plot_sig, "subject_to_info",
                        lambda subject, subjects_dir=None: {"ch_names": recon[subject]})

    out = plot_sig.get_condition_electrodes(
        ["D0001"], {"all_recon": True}, task="GlobalLocal", LAB_root=None)

    assert out == {"D0001": ["A1", "A2", "BAD1"]}
