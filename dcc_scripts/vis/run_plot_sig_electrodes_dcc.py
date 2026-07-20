#!/usr/bin/env python
"""
Submit script for plotting significant electrodes per condition on the brain.
This sets up the input args for and calls ``plot_sig_electrodes_dcc.main``.
Should be wrapped in an sbatch script for cluster submission
(see ``sbatch_plot_sig_electrodes_dcc.sh`` / ``submit_plot_sig_electrodes_dcc.sh``).

=============================================================================
HOW TO USE
=============================================================================
*What* to compare is defined once, by label, in ``condition_plot_specs.py``
(the ``PLOT_CONDITION_SETS`` registry). You pick which comparison to render by
setting the ``PLOT_SET_LABEL`` environment variable in the submit script --
exactly like ``CONDITION_LABEL`` in the power / decoding pipelines. So swapping
in a different set of conditions is a one-line change in the submit script (or a
new registry entry); nothing in this file is specific to any comparison.

This file only holds things that are the same across comparisons: subjects,
brain style, output location, and histogram options.

Figures are written to SAVE_DIR, which defaults to the ``figs`` folder inside
``src/analysis/vis``.
"""
import os
import sys
from datetime import datetime
from types import SimpleNamespace

# ============================================================================
# PATH SETUP  (mirrors the other run_*_dcc.py scripts)
# ============================================================================
if os.path.exists("/hpc/home"):
    # On the cluster
    USER = os.environ.get("USER")
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Local machine (Windows)
    sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")
    try:
        current_file_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_file_path)
    except NameError:
        current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dcc_scripts.vis.plot_sig_electrodes_dcc import main
from dcc_scripts.vis.condition_plot_specs import resolve_plot_set, PLOT_CONDITION_SETS

# ============================================================================
# WHICH COMPARISON  (choose the named set to plot)
# ============================================================================
# Set via the submit script: --export=ALL,PLOT_SET_LABEL=<label>.
# Falls back to a default so the script is still runnable by hand.
PLOT_SET_LABEL = os.environ.get("PLOT_SET_LABEL", "lwpc_vs_lwps")
plot_set = resolve_plot_set(PLOT_SET_LABEL)

# ============================================================================
# CONFIG  (shared across comparisons)
# ============================================================================
LAB_ROOT = None      # auto-detected in the helpers
TASK = "GlobalLocal"

# For ANOVA-sourced comparisons, this should be a superset of the subjects the
# ANOVA ran on (the "_N_subjects" run dir), so every significant electrode can be
# located on the average brain. This is the 24-subject list used by the power
# ANOVA pipeline.
SUBJECTS = [
    "D0057", "D0059", "D0063", "D0065", "D0069", "D0077", "D0090", "D0094",
    "D0100", "D0102", "D0103", "D0107A", "D0110", "D0116", "D0117", "D0121",
    "D0133", "D0134", "D0137", "D0138", "D0139A", "D0144", "D0145", "D0146",
]

# Where subjects_electrodestoROIs_dict.json lives (built if absent).
CONFIG_DIR = os.path.join(project_root, "src", "analysis", "config")

# Save figures to the figs folder inside src/analysis/vis.
SAVE_DIR = os.path.join(project_root, "src", "analysis", "vis", "figs")

# ---- Brain plotting style --------------------------------------------------
HEMI = "both"
MARKER_SIZE = 0.45
TRANSPARENCY = 0.4
RM_WM = False               # drop white-matter electrodes when plotting

# ---- Histograms ------------------------------------------------------------
MAKE_HISTOGRAMS = True
HIST_DROP_WHITE_MATTER = True
HIST_TOP_N = None           # None = all ROIs, or an int to keep the top-N


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conditions = plot_set["conditions"]
    combined_name = PLOT_SET_LABEL  # figures are grouped under the set label

    args = SimpleNamespace(
        timestamp=timestamp,
        subjects=SUBJECTS,
        conditions=conditions,
        task=TASK,
        LAB_root=LAB_ROOT,
        rois_dict=plot_set["rois_dict"],
        config_dir=CONFIG_DIR,
        save_dir=SAVE_DIR,
        hemi=HEMI,
        size=MARKER_SIZE,
        transparency=TRANSPARENCY,
        rm_wm=RM_WM,
        mutually_exclusive=plot_set["mutually_exclusive"],
        overlap_color=plot_set["overlap_color"],
        make_histograms=MAKE_HISTOGRAMS,
        hist_drop_white_matter=HIST_DROP_WHITE_MATTER,
        hist_top_n=HIST_TOP_N,
        combined_name=combined_name,
    )

    print("=" * 70)
    print("PLOT SIGNIFICANT ELECTRODES PER CONDITION")
    print("=" * 70)
    print(f"Plot set:     {PLOT_SET_LABEL}")
    print(f"Available:    {sorted(PLOT_CONDITION_SETS)}")
    print(f"Subjects:     {SUBJECTS}")
    print(f"Conditions:   {[(n, c['color']) for n, c in conditions.items()]}")
    print(f"ROIs:         {list(args.rois_dict.keys()) if args.rois_dict else 'whole brain'}")
    print(f"Mutually exclusive colors: {args.mutually_exclusive}")
    print(f"Save dir:     {SAVE_DIR}")
    print("=" * 70)

    try:
        main(args)
        print("\n✓ Plotting completed successfully!")
    except Exception as e:
        print(f"\n✗ Plotting failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_analysis()
