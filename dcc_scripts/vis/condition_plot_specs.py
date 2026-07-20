#!/usr/bin/env python
"""
Registry of named condition-sets to plot on the brain.

This is the single place you edit to define *what to compare*. Each entry in
``PLOT_CONDITION_SETS`` is a self-contained comparison (which conditions, where
their significant electrodes come from, and — optionally — colors, ROI
restriction, and overlap behavior). Select which one to render by setting the
``PLOT_SET_LABEL`` environment variable in the submit script, exactly like
``CONDITION_LABEL`` in the power / decoding pipelines. Nothing here is specific
to any one comparison; LWPC-vs-LWPS is just one example entry.

To add a comparison:
    1. (If needed) point ANOVA_RUNS_BASE / add an epochs_root_file below.
    2. Add an entry to PLOT_CONDITION_SETS using the ``sig_chans(...)`` and
       ``anova_effect(...)`` builders.
    3. Submit with PLOT_SET_LABEL=<your_label>.

Two electrode sources are supported per condition:
    * sig_chans(epochs_root_file)      -> sig_chans_{subject}_{root}.json (one contrast)
    * anova_effect(run_dir, effect)    -> electrodes significant for a specific effect
                                          in a within-electrode ANOVA run
                                          (e.g. interaction terms like LWPC / LWPS).
"""
import os
from collections import OrderedDict

# ===========================================================================
# Colors
# ===========================================================================
# Visually distinct RGB tuples (0-1). Conditions with no explicit 'color' are
# assigned from this palette in order. Add/reorder as you like.
DEFAULT_PALETTE = [
    (1.0, 0.0, 0.0),   # red
    (0.0, 0.0, 1.0),   # blue
    (0.0, 0.7, 0.0),   # green
    (1.0, 0.6, 0.0),   # orange
    (0.6, 0.0, 0.8),   # purple
    (0.0, 0.75, 0.75),  # teal
    (0.9, 0.4, 0.7),   # pink
]

# Electrodes significant in >1 condition are drawn once in this color.
OVERLAP_COLOR = (0.0, 0.0, 0.0)  # black


def assign_colors(conditions, palette=None):
    """Fill in a 'color' for any condition that didn't specify one."""
    palette = palette or DEFAULT_PALETTE
    out = OrderedDict()
    i = 0
    for name, cfg in conditions.items():
        cfg = dict(cfg)
        if "color" not in cfg or cfg["color"] is None:
            cfg["color"] = palette[i % len(palette)]
            i += 1
        out[name] = cfg
    return out


# ===========================================================================
# Condition-source builders (keep each condition a one-liner)
# ===========================================================================
def sig_chans(epochs_root_file, color=None, **extra):
    """A condition whose electrodes come from sig_chans_{subject}_{root}.json."""
    cfg = {"epochs_root_file": epochs_root_file}
    if color is not None:
        cfg["color"] = color
    cfg.update(extra)
    return cfg


def anova_effect(anova_run_dir, effect, color=None, use_fdr=True, p_thresh=0.05,
                 anova_roi=None, **extra):
    """A condition whose electrodes are significant for ``effect`` in an ANOVA run.

    ``effect`` is the statsmodels term name, e.g. ``'C(congruency)'`` (main effect)
    or ``'C(congruency):C(incongruentProportion)'`` (interaction).
    """
    cfg = {"anova_run_dir": anova_run_dir, "effect": effect,
           "use_fdr": use_fdr, "p_thresh": p_thresh}
    if anova_roi is not None:
        cfg["anova_roi"] = anova_roi
    if color is not None:
        cfg["color"] = color
    cfg.update(extra)
    return cfg


# ===========================================================================
# Data locations (edit these once for your machine / cluster)
# ===========================================================================
# Within-electrode ANOVA runs are written by run_power_traces_dcc.py into:
#
#   <POWER_FIGS_BASE>/<ANOVA_EPOCHS_ROOT>/anova_within_<ANOVA_UNIT>/
#       <condition_label>_<n_subjects>_subjects/
#           significant_effects_structure.json   (and summary.csv)
#
# so anova_run(condition_label, n_subjects) rebuilds that path. Any of the three
# pieces can be overridden with an env var (exported by the submit script)
# without editing this file.
_USER = os.environ.get("USER", "")
if os.path.exists("/hpc/home") and _USER:
    _DEFAULT_POWER_FIGS_BASE = (
        f"/hpc/home/{_USER}/coganlab/{_USER}/GlobalLocal/dcc_scripts/power/figs")
else:
    _DEFAULT_POWER_FIGS_BASE = "/path/to/GlobalLocal/dcc_scripts/power/figs"  # <- EDIT ME

POWER_FIGS_BASE = os.environ.get("POWER_FIGS_BASE", _DEFAULT_POWER_FIGS_BASE)

# The epochs_root_file the ANOVA was run on (the directory name under figs/).
ANOVA_EPOCHS_ROOT = os.environ.get(
    "ANOVA_EPOCHS_ROOT",
    "Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_"
    "drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_"
    "ttest_ind_equal_var_False_nan_policy_omit")

# 'electrode' (within-electrode ANOVA) -> anova_within_electrode. Must be
# 'electrode' to get per-electrode significance for brain plotting.
ANOVA_UNIT = os.environ.get("ANOVA_UNIT", "electrode")


def anova_run(condition_label, n_subjects):
    """Path to a within-electrode ANOVA run dir produced by run_power_traces_dcc.

    ``condition_label`` is the CONDITION_LABEL the ANOVA was submitted with (e.g.
    'stimulus_lwpc_conditions'); ``n_subjects`` is the subject count baked into the
    run-directory name (e.g. 24) -- this must match the existing directory, which
    is independent of the SUBJECTS list you plot with.
    """
    return os.path.join(POWER_FIGS_BASE, ANOVA_EPOCHS_ROOT,
                        f"anova_within_{ANOVA_UNIT}",
                        f"{condition_label}_{n_subjects}_subjects")


# A couple of commonly-used sig_chans epochs_root_files, for reference.
STIM_HG_ROOT = ("Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_"
                "drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_"
                "ttest_ind_equal_var_False_nan_policy_omit")


# ===========================================================================
# THE REGISTRY  --  add your comparisons here
# ===========================================================================
# Each entry: {'conditions': OrderedDict(name -> source cfg), and optionally
#              'rois_dict', 'mutually_exclusive', 'overlap_color'}.
# Colors are optional; unspecified ones are auto-assigned from DEFAULT_PALETTE.
# Subject count baked into the ANOVA run-directory names (the "_N_subjects"
# suffix). Set to whatever the ANOVA was actually run with.
N_SUBJECTS_IN_ANOVA = 24

PLOT_CONDITION_SETS = {

    # LWPC vs LWPS interaction electrodes.
    # unique LWPC -> palette[0] (red), unique LWPS -> palette[1] (blue),
    # overlap -> black.
    "lwpc_vs_lwps": {
        "conditions": OrderedDict([
            ("lwpc", anova_effect(
                anova_run("stimulus_lwpc_conditions", N_SUBJECTS_IN_ANOVA),
                "C(congruency):C(incongruentProportion)")),
            ("lwps", anova_effect(
                anova_run("stimulus_lwps_conditions", N_SUBJECTS_IN_ANOVA),
                "C(switchType):C(switchProportion)")),
        ]),
    },

    # Congruency vs switch-type MAIN effects. Both live in the full 16-cell
    # experiment ANOVA run (factors congruency, incongruentProportion,
    # switchType, switchProportion), so pull both from that one run.
    "congruency_vs_switchType": {
        "conditions": OrderedDict([
            ("congruency", anova_effect(
                anova_run("stimulus_experiment_conditions", N_SUBJECTS_IN_ANOVA),
                "C(congruency)")),
            ("switchType", anova_effect(
                anova_run("stimulus_experiment_conditions", N_SUBJECTS_IN_ANOVA),
                "C(switchType)")),
        ]),
    },

    # Example of the OTHER electrode source: two different sig_chans contrasts
    # (not ANOVA-based). Replace the roots with your actual epochs_root_files.
    "stim_vs_response": {
        "conditions": OrderedDict([
            ("stimulus", sig_chans(STIM_HG_ROOT)),
            ("response", sig_chans("Response_..._REPLACE_ME")),            # <- EDIT root
        ]),
    },
}


def resolve_plot_set(label, palette=None):
    """Return a normalized spec for ``label``: colors filled in, defaults set.

    Returns
    -------
    dict with keys: 'conditions' (OrderedDict), 'rois_dict', 'mutually_exclusive',
    'overlap_color'.
    """
    if label not in PLOT_CONDITION_SETS:
        raise KeyError(
            f"Unknown PLOT_SET_LABEL '{label}'. "
            f"Available: {sorted(PLOT_CONDITION_SETS)}")
    spec = PLOT_CONDITION_SETS[label]
    return {
        "conditions": assign_colors(spec["conditions"], palette),
        "rois_dict": spec.get("rois_dict"),
        "mutually_exclusive": spec.get("mutually_exclusive", True),
        "overlap_color": spec.get("overlap_color", OVERLAP_COLOR),
    }
