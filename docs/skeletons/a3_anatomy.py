"""A3 skeleton — anatomy: brain maps, ROI histograms, coverage-conditioned test
(plan §3).

DROP-IN TARGET
--------------
The stats functions (`build_coverage_matrix`, `roi_group_enrichment_test`) can
live in a new `src/analysis/stats/stability_flexibility_anatomy.py`. The brain
plot should REUSE the existing renderer in `src/analysis/vis/`
(`brain_figure_glasser_separate_svgs_lateral_medial_view_less_bold.py`) — pass
your per-group electrode lists as its highlight sets rather than writing new
surface code.

Electrode -> ROI mapping comes from the shared utils:
    from src.analysis.utils.general_utils import (
        make_or_load_subjects_electrodes_to_ROIs_dict,
    )
    from src.analysis.config.rois import rois_dict

WHY THIS EXISTS
---------------
"Are the distinct subpopulations in different PLACES?" This is descriptive
anatomy — and the layer most exposed to COVERAGE BIAS: iEEG coverage is
clinically determined, so a raw ROI difference can just reflect where electrodes
happen to be. Every claim must be conditioned on coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_roi(labels, electrodes_to_rois):
    """Add an 'roi' column to the labels table.

    labels : DataFrame with subject, electrode, S, F (from A1).
    electrodes_to_rois : mapping electrode -> ROI name (built by
        make_or_load_subjects_electrodes_to_ROIs_dict + rois_dict).

    Returns labels + 'roi' and a derived 'group' column in
    {'both','S_only','F_only','neither'}.

    Steps: map electrode->roi; derive group from (S, F).
    """
    raise NotImplementedError("A3: join labels to ROI + derive the 4-way group")


def build_coverage_matrix(labels):
    """Subject × ROI boolean coverage: does subject s have ANY electrode in ROI r?

    Returns a DataFrame indexed by subject, columns = ROIs, values bool.
    Used to (a) report per-ROI coverage and (b) restrict the enrichment test to
    ROIs sampled in >= k subjects.

    Steps: pivot_table over (subject, roi) with an existence aggregate.
    """
    raise NotImplementedError("A3: subject x ROI coverage matrix")


def roi_group_enrichment_test(
    labels_with_roi,
    coverage,
    min_subjects: int = 3,
    n_perm: int = 10000,
    seed: int = 0,
):
    """Is selectivity-group membership associated with ROI, conditioned on coverage?

    Parameters
    ----------
    labels_with_roi : output of `attach_roi`.
    coverage : output of `build_coverage_matrix`.
    min_subjects : keep only ROIs covered in >= this many subjects.

    Returns
    -------
    dict:
        rois_tested : list[str]
        observed_stat : float          # chi-square (or your chosen statistic)
        p : float                       # permutation p-value
        contingency : DataFrame         # group × ROI counts (restricted set)
        per_roi_coverage : Series       # n subjects covering each tested ROI

    Implementation steps
    ---------------------
    1. Restrict to ROIs with coverage.sum(axis=0) >= min_subjects; drop electrodes
       outside them.
    2. Build the group × ROI contingency table; compute a chi-square statistic.
    3. NULL: permute the group label WITHIN SUBJECT (so the null respects both
       nesting and coverage), recompute the statistic n_perm times.
       p = (sum(null >= observed) + 1) / (n_perm + 1).
    4. Report per-ROI coverage alongside, so a reader sees the difference isn't
       pure placement.
    """
    raise NotImplementedError("A3: coverage-conditioned ROI enrichment test")


def plot_selectivity_groups_on_brain(labels_with_roi, out_path, **vis_kwargs):
    """Render S-only / F-only / both electrodes on the cortical surface.

    Thin wrapper over the existing vis renderer — build the per-group electrode
    lists and hand them to the Glasser SVG figure function as color-coded
    highlight sets. Return/save the figure path(s).
    """
    raise NotImplementedError("A3: color-coded brain figure via the vis pipeline")
