#!/usr/bin/env python
"""
Submit script for plotting significant electrodes per condition on the brain.
This sets up the input args for and calls ``plot_sig_electrodes_dcc.main``.
Should be wrapped in an sbatch script for cluster submission
(see ``sbatch_plot_sig_electrodes_dcc.sh`` / ``submit_plot_sig_electrodes_dcc.sh``).

=============================================================================
HOW TO USE
=============================================================================
Everything you would normally toggle in the notebook lives in the CONFIG block
below:

  * SUBJECTS          - which subjects to include.
  * CONDITIONS        - which conditions to plot and what color each gets. This
                        is the main knob: add/remove entries to choose which
                        conditions appear on the brain, each in its own color.
                        Each condition points at the ``epochs_root_file`` used to
                        find its ``sig_chans_{subject}_{epochs_root_file}.json``.
  * ROIS_DICT         - set to a dict to restrict to specific ROIs, or None to
                        plot every significant electrode (whole brain).
  * MUTUALLY_EXCLUSIVE - if True, electrodes significant in more than one
                        condition are drawn once in OVERLAP_COLOR instead of
                        being over-plotted; each condition then shows only its
                        unique electrodes.

Figures are written to SAVE_DIR, which defaults to the ``figs`` folder inside
``src/analysis/vis``.
"""
import os
import sys
from collections import OrderedDict
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

# ============================================================================
# CONFIG
# ============================================================================
LAB_ROOT = None      # auto-detected in the helpers
TASK = "GlobalLocal"

SUBJECTS = [
    "D0057", "D0059", "D0063", "D0065", "D0069", "D0071", "D0077", "D0090",
    "D0094", "D0100", "D0102", "D0103", "D0110", "D0116", "D0117", "D0121",
]

# ----------------------------------------------------------------------------
# Conditions to plot. Add/remove entries to choose what gets drawn; each is
# given its own color (matplotlib RGB tuple, values in 0-1). With
# MUTUALLY_EXCLUSIVE=True (below), electrodes significant in >1 condition are
# drawn once in OVERLAP_COLOR and each condition shows only its unique electrodes.
#
# Each condition picks ONE electrode source:
#   A) 'epochs_root_file' : sig_chans_{subject}_{epochs_root_file}.json files
#                           (a single contrast, e.g. congruency or response).
#   B) 'anova_run_dir' + 'effect' : electrodes flagged significant for a specific
#                           effect in a within-electrode ANOVA run. This is how
#                           interaction effects like LWPC / LWPS are defined.
#                           Optional: 'use_fdr' (default True), 'p_thresh'
#                           (default 0.05), 'anova_roi' (restrict to one ROI).
# ----------------------------------------------------------------------------
_STIM_ROOT = ("Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_"
              "drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_"
              "ttest_ind_equal_var_False_nan_policy_omit")

# --- Example 1: LWPC vs LWPS interaction electrodes (the multi-condition,
#     unique-in-color / overlap-in-black use case). Point anova_run_dir at your
#     within-electrode ANOVA run directories (the ones containing
#     significant_effects_structure.json / summary.csv). ---
_LWPC_ANOVA_RUN_DIR = "/path/to/your/lwpc_within_elec_anova_run"   # <- EDIT ME
_LWPS_ANOVA_RUN_DIR = "/path/to/your/lwps_within_elec_anova_run"   # <- EDIT ME

CONDITIONS = OrderedDict([
    ("lwpc", {"anova_run_dir": _LWPC_ANOVA_RUN_DIR,
              "effect": "C(congruency):C(incongruentProportion)",
              "use_fdr": True, "color": (1.0, 0.0, 0.0)}),   # unique LWPC -> red
    ("lwps", {"anova_run_dir": _LWPS_ANOVA_RUN_DIR,
              "effect": "C(switchType):C(switchProportion)",
              "use_fdr": True, "color": (0.0, 0.0, 1.0)}),   # unique LWPS -> blue
])
# (overlap of lwpc & lwps -> OVERLAP_COLOR, black, via MUTUALLY_EXCLUSIVE below.)

# --- Example 2: plain sig_chans contrasts (uncomment to use instead) ---
# CONDITIONS = OrderedDict([
#     ("congruency", {"epochs_root_file": _STIM_ROOT, "color": (1.0, 0.0, 0.0)}),
#     ("switchType", {"epochs_root_file": _STIM_ROOT, "color": (0.0, 0.0, 1.0)}),
# ])

# Restrict to these anatomical ROIs, or set to None to plot every significant
# electrode. NOTE: this only applies to 'epochs_root_file'-sourced conditions;
# 'anova_run_dir'-sourced conditions (like the LWPC/LWPS example above) are
# already ROI-scoped by the ANOVA run, so this is ignored for them.
ROIS_DICT = None  # whole brain
# ROIS_DICT = {
#     "lpfc": [
#         "G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul",
#         "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont",
#         "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup",
#         "S_front_inf", "S_front_middle", "S_front_sup",
#     ],
# }

# Where subjects_electrodestoROIs_dict.json lives (built if absent).
CONFIG_DIR = os.path.join(project_root, "src", "analysis", "config")

# Save figures to the figs folder inside src/analysis/vis.
SAVE_DIR = os.path.join(project_root, "src", "analysis", "vis", "figs")

# ---- Brain plotting style --------------------------------------------------
HEMI = "both"
MARKER_SIZE = 0.45
TRANSPARENCY = 0.4
RM_WM = False               # drop white-matter electrodes when plotting

# ---- Overlap handling ------------------------------------------------------
MUTUALLY_EXCLUSIVE = True   # electrodes shared by >1 condition drawn in OVERLAP_COLOR
OVERLAP_COLOR = (0.0, 0.0, 0.0)  # black

# ---- Histograms ------------------------------------------------------------
MAKE_HISTOGRAMS = True
HIST_DROP_WHITE_MATTER = True
HIST_TOP_N = None           # None = all ROIs, or an int to keep the top-N


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_name = "_".join(CONDITIONS.keys())
    if ROIS_DICT:
        combined_name = f"{combined_name}_{'_'.join(ROIS_DICT.keys())}"

    args = SimpleNamespace(
        timestamp=timestamp,
        subjects=SUBJECTS,
        conditions=CONDITIONS,
        task=TASK,
        LAB_root=LAB_ROOT,
        rois_dict=ROIS_DICT,
        config_dir=CONFIG_DIR,
        save_dir=SAVE_DIR,
        hemi=HEMI,
        size=MARKER_SIZE,
        transparency=TRANSPARENCY,
        rm_wm=RM_WM,
        mutually_exclusive=MUTUALLY_EXCLUSIVE,
        overlap_color=OVERLAP_COLOR,
        make_histograms=MAKE_HISTOGRAMS,
        hist_drop_white_matter=HIST_DROP_WHITE_MATTER,
        hist_top_n=HIST_TOP_N,
        combined_name=combined_name,
    )

    print("=" * 70)
    print("PLOT SIGNIFICANT ELECTRODES PER CONDITION")
    print("=" * 70)
    print(f"Subjects:     {SUBJECTS}")
    print(f"Conditions:   {list(CONDITIONS.keys())}")
    print(f"ROIs:         {list(ROIS_DICT.keys()) if ROIS_DICT else 'whole brain'}")
    print(f"Save dir:     {SAVE_DIR}")
    print(f"Mutually exclusive colors: {MUTUALLY_EXCLUSIVE}")
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
