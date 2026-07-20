#!/usr/bin/env python
"""
Plot significant electrodes for one or more conditions on the fsaverage brain,
each condition in its own color, and make histograms of the anatomical ROIs
that those electrodes fall in.

This is the cleaned-up, submittable version of ``src/analysis/vis/plot_subjects.ipynb``.
It is driven by an ``args`` namespace (see ``run_plot_sig_electrodes_dcc.py``) so it
can be launched on the cluster via sbatch, or imported and called interactively.

For each condition the caller supplies:
  - an ``epochs_root_file`` that identifies the ``sig_chans_{subject}_{epochs_root_file}.json``
    files written by the stats pipeline, and
  - a ``color`` (matplotlib RGB tuple) to draw that condition's electrodes in.

Outputs (written to ``args.save_dir``):
  - ``brain_<combined_name>.png``            : all conditions overlaid on one brain.
  - ``brain_<condition>.png``                : one brain per condition.
  - ``roi_hist_<condition>.png``             : ROI histogram for each condition.
  - ``sig_electrodes_<condition>.json``      : the electrodes plotted, for reproducibility.

The heavy lifting (loading sig chans, mapping electrodes to ROIs) reuses the
shared helpers in ``src.analysis.utils.general_utils`` so this stays in sync with
the rest of the pipeline.
"""

import os
import re
import json
from collections import Counter, OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless: never open a window on the cluster
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Off-screen 3D rendering. On the cluster there is no display, so PyVista must
# render into an off-screen buffer. This has to be set before the brain backend
# is imported. The sbatch script additionally wraps python in ``xvfb-run`` as a
# belt-and-suspenders fallback.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.allow_empty_mesh = True
except Exception as e:  # pragma: no cover - only matters at runtime on cluster
    print(f"Warning: could not configure pyvista for off-screen rendering: {e}")

from ieeg.viz.mri import subject_to_info

from src.analysis.vis.jim_mri import plot_on_average
from src.analysis.utils.general_utils import (
    get_sig_chans_per_subject,
    make_or_load_subjects_electrodes_to_ROIs_dict,
    make_sig_electrodes_per_subject_and_roi_dict,
)

# Regex that strips leading zeros from a subject id: 'D0057' -> 'D57', 'D0107A' -> 'D107A'.
_SUBJECT_PATTERN = re.compile(r"^D(0*)(\d+)([A-Za-z]*)$")


# ===========================================================================
# Helpers
# ===========================================================================
def strip_leading_zeros(subject_id):
    """'D0057' -> 'D57', 'D0107A' -> 'D107A'. Unrecognized ids are returned as-is."""
    match = _SUBJECT_PATTERN.match(subject_id)
    if not match:
        return subject_id
    _, numbers, letters = match.groups()
    return f"D{int(numbers)}{letters}"


def collapse_rois_to_subject_dict(sig_electrodes_per_subject_roi):
    """Collapse a {roi: {subject: [electrodes]}} dict into {subject: [electrodes]}.

    Duplicate electrodes (an electrode that lands in more than one ROI) are kept
    only once.
    """
    collapsed = {}
    for _roi, subject_dict in sig_electrodes_per_subject_roi.items():
        for subject, electrodes in subject_dict.items():
            bucket = collapsed.setdefault(subject, [])
            bucket.extend(e for e in electrodes if e not in bucket)
    return collapsed


def get_condition_electrodes(subjects, epochs_root_file, task, LAB_root,
                             rois_dict=None, subjects_electrodes_to_ROIs_dict=None,
                             config_dir=None):
    """Significant electrodes for one condition, as {subject_with_zeros: [names]}.

    Parameters
    ----------
    subjects : list of str
        Subject ids with leading zeros, e.g. ``['D0057', 'D0059']``.
    epochs_root_file : str
        Identifies the ``sig_chans_{subject}_{epochs_root_file}.json`` files.
    task, LAB_root :
        Passed through to :func:`get_sig_chans_per_subject`.
    rois_dict : dict, optional
        If given (``{roi_name: [anatomical_labels]}``), restrict the significant
        electrodes to those falling in the listed ROIs. If ``None``, every
        significant electrode is kept (whole-brain).
    subjects_electrodes_to_ROIs_dict : dict, optional
        Prebuilt electrode->ROI mapping. If ``None`` and ROI filtering is
        requested, it is loaded/built from ``config_dir``.
    config_dir : str, optional
        Where the ``subjects_electrodestoROIs_dict.json`` lives.

    Returns
    -------
    dict
        ``{subject_with_zeros: [electrode_names]}``.
    """
    sig_chans_per_subject = get_sig_chans_per_subject(
        subjects, epochs_root_file, task=task, LAB_root=LAB_root)

    if not rois_dict:
        # Whole brain: keep every significant channel.
        return {subj: list(chans) for subj, chans in sig_chans_per_subject.items()}

    if subjects_electrodes_to_ROIs_dict is None:
        subjects_electrodes_to_ROIs_dict = make_or_load_subjects_electrodes_to_ROIs_dict(
            subjects, task=task, LAB_root=LAB_root, save_dir=config_dir,
            filename="subjects_electrodestoROIs_dict.json")

    _all, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(
        rois_dict, subjects_electrodes_to_ROIs_dict, sig_chans_per_subject)
    return collapse_rois_to_subject_dict(sig_electrodes_per_subject_roi)


def build_global_index_map(subjects_no_zeros):
    """Map each subject to its offset in the concatenated fsaverage channel space.

    ``plot_on_average`` addresses electrodes by a single flat index that runs
    across all subjects (in the order they are passed). Using one shared map for
    every condition keeps the indices comparable, which is what lets us detect
    electrodes shared by multiple conditions.

    Returns
    -------
    offsets : OrderedDict
        ``{subject_no_zeros: (offset, ch_names)}``.
    """
    offsets = OrderedDict()
    running = 0
    for subject in subjects_no_zeros:
        info = subject_to_info(subject)
        offsets[subject] = (running, info["ch_names"])
        running += len(info["ch_names"])
    return offsets


def electrodes_to_global_indices(condition_electrodes, offsets):
    """Convert {subject_with_zeros: [names]} into a sorted list of global indices.

    Electrodes that are not present in a subject's fsaverage info (e.g. dropped
    during preprocessing) are skipped with a warning.
    """
    indices = []
    for subject_with_zeros, electrodes in condition_electrodes.items():
        subject = strip_leading_zeros(subject_with_zeros)
        if subject not in offsets:
            print(f"  Warning: {subject_with_zeros} not in the plotting subject list; skipping.")
            continue
        offset, ch_names = offsets[subject]
        for electrode in electrodes:
            if electrode in ch_names:
                indices.append(offset + ch_names.index(electrode))
            else:
                print(f"  Warning: electrode '{electrode}' not found in {subject}; skipping.")
    return sorted(set(indices))


def split_unique_and_overlap(condition_indices):
    """Split condition index sets into per-condition-unique and shared-overlap.

    Parameters
    ----------
    condition_indices : OrderedDict
        ``{condition_name: set_of_global_indices}``.

    Returns
    -------
    unique : OrderedDict
        ``{condition_name: [indices in this condition and no other]}``.
    overlap : list
        Indices that appear in two or more conditions.
    """
    counts = Counter()
    for idx_set in condition_indices.values():
        counts.update(idx_set)
    overlap = sorted(idx for idx, n in counts.items() if n > 1)
    overlap_set = set(overlap)
    unique = OrderedDict(
        (name, sorted(i for i in idx_set if i not in overlap_set))
        for name, idx_set in condition_indices.items()
    )
    return unique, overlap


# ===========================================================================
# ROI histogram
# ===========================================================================
def electrode_roi_counts(condition_electrodes, subjects_electrodes_to_ROIs_dict,
                         drop_white_matter=True):
    """Count the anatomical ROI of every electrode in a condition group.

    Uses the per-subject ``default_dict`` (electrode -> anatomical label).
    """
    counts = Counter()
    for subject_with_zeros, electrodes in condition_electrodes.items():
        subj_entry = subjects_electrodes_to_ROIs_dict.get(subject_with_zeros, {})
        default_dict = subj_entry.get("default_dict", {})
        for electrode in electrodes:
            roi = default_dict.get(electrode, "Unknown")
            if drop_white_matter and ("White-Matter" in roi or roi in ("Unknown", "unknown")):
                continue
            counts[roi] += 1
    return counts


def plot_roi_histogram(counts, title, save_path, color=(0.2, 0.4, 0.8), top_n=None):
    """Horizontal bar chart of ROI counts, sorted most-common first."""
    if not counts:
        print(f"  No ROIs to plot for '{title}'; skipping histogram.")
        return
    items = counts.most_common(top_n)
    labels = [label for label, _ in items][::-1]   # reverse so largest is on top
    values = [value for _, value in items][::-1]

    height = max(3.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, values, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of significant electrodes")
    ax.set_title(title)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value}", va="center", fontsize=9)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROI histogram -> {save_path}")


# ===========================================================================
# Main
# ===========================================================================
def main(args):
    """Run the plotting pipeline.

    Expected attributes on ``args`` (see ``run_plot_sig_electrodes_dcc.py``):
        subjects                : list[str]        subject ids with leading zeros
        conditions              : OrderedDict      {name: {'epochs_root_file': str, 'color': (r,g,b)}}
        task                    : str
        LAB_root                : str | None
        rois_dict               : dict | None      restrict to these ROIs, or None for whole brain
        config_dir              : str              where subjects_electrodestoROIs_dict.json lives
        save_dir                : str              output directory (created if missing)
        hemi                    : str              'both' | 'lh' | 'rh' | 'split'
        size                    : float            electrode marker size
        transparency            : float            brain transparency
        rm_wm                   : bool             drop white-matter electrodes when plotting
        mutually_exclusive      : bool             plot shared electrodes in overlap_color
        overlap_color           : (r,g,b)          color for electrodes shared by >1 condition
        make_histograms         : bool
        combined_name           : str              filename stem for the combined brain figure
    """
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving figures to: {args.save_dir}")

    subjects = list(args.subjects)
    subjects_no_zeros = [strip_leading_zeros(s) for s in subjects]

    rois_dict = getattr(args, "rois_dict", None)
    make_histograms = getattr(args, "make_histograms", True)

    # Electrode -> ROI mapping is needed for ROI filtering and for histograms.
    subjects_electrodes_to_ROIs_dict = None
    if rois_dict or make_histograms:
        subjects_electrodes_to_ROIs_dict = make_or_load_subjects_electrodes_to_ROIs_dict(
            subjects, task=args.task, LAB_root=args.LAB_root, save_dir=args.config_dir,
            filename="subjects_electrodestoROIs_dict.json")

    # ---- 1. Load each condition's significant electrodes -------------------
    condition_electrodes = OrderedDict()   # {name: {subj_with_zeros: [names]}}
    for name, cfg in args.conditions.items():
        print(f"\n=== Loading significant electrodes for condition: {name} ===")
        electrodes = get_condition_electrodes(
            subjects, cfg["epochs_root_file"], task=args.task, LAB_root=args.LAB_root,
            rois_dict=rois_dict,
            subjects_electrodes_to_ROIs_dict=subjects_electrodes_to_ROIs_dict,
            config_dir=args.config_dir)
        condition_electrodes[name] = electrodes
        n = sum(len(v) for v in electrodes.values())
        print(f"  {n} significant electrodes across {len(electrodes)} subjects.")

        # Save the plotted electrodes for reproducibility.
        out_json = os.path.join(args.save_dir, f"sig_electrodes_{name}.json")
        with open(out_json, "w") as fh:
            json.dump(electrodes, fh, indent=2)

    # ---- 2. Convert to a shared global index space -------------------------
    offsets = build_global_index_map(subjects_no_zeros)
    condition_indices = OrderedDict(
        (name, set(electrodes_to_global_indices(elecs, offsets)))
        for name, elecs in condition_electrodes.items()
    )

    mutually_exclusive = getattr(args, "mutually_exclusive", True)
    overlap_color = getattr(args, "overlap_color", (0, 0, 0))

    if mutually_exclusive:
        plot_groups, overlap = split_unique_and_overlap(condition_indices)
    else:
        plot_groups, overlap = condition_indices, []

    # ---- 3. Combined brain: every condition overlaid on one figure ---------
    print("\n=== Plotting combined brain figure ===")
    fig = None
    for name, indices in plot_groups.items():
        if not indices:
            print(f"  {name}: no electrodes to plot.")
            continue
        color = args.conditions[name]["color"]
        fig = plot_on_average(
            subjects_no_zeros, picks=sorted(indices), rm_wm=args.rm_wm,
            hemi=args.hemi, color=color, size=args.size,
            transparency=args.transparency, fig=fig, show=False)
    if mutually_exclusive and overlap:
        print(f"  overlap: {len(overlap)} electrodes shared by >1 condition.")
        fig = plot_on_average(
            subjects_no_zeros, picks=overlap, rm_wm=args.rm_wm, hemi=args.hemi,
            color=overlap_color, size=args.size, transparency=args.transparency,
            fig=fig, show=False)

    if fig is not None:
        combined_name = getattr(args, "combined_name", "combined")
        combined_path = os.path.join(args.save_dir, f"brain_{combined_name}.png")
        fig.save_image(combined_path)
        fig.close()
        print(f"  Saved combined brain -> {combined_path}")

    # ---- 4. One brain per condition (in that condition's color) ------------
    for name, indices in condition_indices.items():
        if not indices:
            continue
        color = args.conditions[name]["color"]
        cfig = plot_on_average(
            subjects_no_zeros, picks=sorted(indices), rm_wm=args.rm_wm,
            hemi=args.hemi, color=color, size=args.size,
            transparency=args.transparency, show=False)
        cpath = os.path.join(args.save_dir, f"brain_{name}.png")
        cfig.save_image(cpath)
        cfig.close()
        print(f"  Saved brain for {name} -> {cpath}")

    # ---- 5. ROI histograms -------------------------------------------------
    if make_histograms:
        print("\n=== Plotting ROI histograms ===")
        for name, electrodes in condition_electrodes.items():
            counts = electrode_roi_counts(
                electrodes, subjects_electrodes_to_ROIs_dict,
                drop_white_matter=getattr(args, "hist_drop_white_matter", True))
            color = args.conditions[name]["color"]
            hist_path = os.path.join(args.save_dir, f"roi_hist_{name}.png")
            plot_roi_histogram(
                counts, title=f"ROIs of significant electrodes: {name}",
                save_path=hist_path, color=color,
                top_n=getattr(args, "hist_top_n", None))

    print("\nDone.")
