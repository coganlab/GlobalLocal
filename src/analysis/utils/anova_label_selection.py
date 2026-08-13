"""Load electrode sets from an A1 ``anova_labels.csv`` result.

This is deliberately independent of MNE so power-trace, decoding, and
cross-decoding entrypoints use exactly the same selection semantics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pandas as pd


_EFFECTS = {
    "lwpc": ("S", "p_cong", "q_cong"),
    "s": ("S", "p_cong", "q_cong"),
    "cpc": ("CPC", "p_cpc", "q_cpc"),
    "lwps": ("F", "p_switch", "q_switch"),
    "f": ("F", "p_switch", "q_switch"),
    "sps": ("SPS", "p_sps", "q_sps"),
    "cps": ("CPS", "p_cps", "q_cps"),
    "spc": ("SPC", "p_spc", "q_spc"),
    # A condition-mode A1 table uses the same backward-compatible S/F columns
    # for the congruency and switch-type main effects.  Giving those effects
    # explicit names lets callers describe the population they are selecting
    # rather than having to call a congruency main effect "lwpc".
    "congruency": ("S", "p_cong", "q_cong"),
    "switch_type": ("F", "p_switch", "q_switch"),
}

_SET_EFFECTS = {
    "both": (("S", "F"), ()),
    "lwpc_only": (("S",), ("F",)),
    "lwps_only": (("F",), ("S",)),
    "congruency_only": (("S",), ("F",)),
    "switch_type_only": (("F",), ("S",)),
}


def anova_label_run_slug(path, effect="lwpc", correction="flags", alpha=0.05,
                         roi=None):
    """Return a unique, readable output-folder name for an A1 selection.

    The source result-directory name records the statistical window/config.
    The remaining selection parameters are appended explicitly, while a short
    hash of the resolved CSV path prevents collisions between equally named
    result directories in different trees.
    """
    csv_path = Path(path).expanduser()
    if csv_path.name == "anova_labels.csv" or csv_path.suffix.lower() == ".csv":
        source_name = csv_path.parent.name
    else:
        source_name = csv_path.name
        csv_path = csv_path / "anova_labels.csv"

    def safe(value):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_") or "all"

    alpha_text = format(float(alpha), ".12g")
    path_hash = hashlib.sha1(str(csv_path.resolve()).encode()).hexdigest()[:8]
    return (f"{safe(source_name)}__effect-{safe(effect)}"
            f"__correction-{safe(correction)}__alpha-{safe(alpha_text)}"
            f"__roi-{safe(roi or 'all')}__{path_hash}")


def load_anova_labels(path, correction="flags", alpha=0.05, roi=None):
    """Read an A1 table and consistently (re)compute all available flags.

    ``correction='flags'`` consumes the saved binary flag (and therefore honors
    whichever correction created the CSV). Use ``'none'`` to threshold raw
    p-values or ``'fdr_bh'`` to threshold saved q-values explicitly.
    """
    path = Path(path).expanduser()
    # Launchers commonly point at the A1 result directory (whose name records
    # the window/correction) rather than spelling out its one CSV file.
    if path.is_dir():
        path = path / "anova_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"ANOVA labels CSV does not exist: {path}")
    labels = pd.read_csv(path)
    required = {"subject", "electrode"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    correction = correction.strip().lower()
    if correction not in {"flags", "none", "raw", "fdr", "fdr_bh", "q"}:
        raise ValueError("correction must be 'flags', 'none', or 'fdr_bh'")
    if roi is not None:
        if "roi" not in labels:
            raise ValueError("ROI filtering was requested but the CSV has no 'roi' column")
        labels = labels[labels["roi"].astype(str) == str(roi)]
    labels = labels.copy()

    # Recompute every group, not only the requested one. Cross-decoding consumes
    # S and F together, and must not accidentally use saved FDR flags when the
    # caller requested raw p-values (or vice versa).
    for flag_col, p_col, q_col in dict.fromkeys(_EFFECTS.values()):
        flag_col = {"CPC": "S", "SPS": "F"}.get(flag_col, flag_col)
        if correction == "flags":
            if flag_col not in labels:
                continue
            labels[flag_col] = pd.to_numeric(
                labels[flag_col], errors="coerce").eq(1).astype(int)
            continue
        value_col = p_col if correction in {"none", "raw"} else q_col
        if value_col in labels:
            labels[flag_col] = pd.to_numeric(
                labels[value_col], errors="coerce").lt(alpha).astype(int)
    if "S" in labels:
        labels["CPC"] = labels["S"]
    if "F" in labels:
        labels["SPS"] = labels["F"]
    return labels


def load_anova_label_electrodes(path, effect="lwpc", correction="flags",
                                alpha=0.05, roi=None):
    """Return ``{roi: {subject: [bare electrode names]}}`` from A1 labels.

    ``lwpc``/``lwps`` and ``congruency``/``switch_type`` select the complete
    corresponding set, including overlap.  Their ``*_only`` forms exclude
    electrodes significant for the other effect. ``both`` selects the
    intersection.  Use the LWPC/LWPS names with a proportion-mode A1 table and
    the main-effect names with a condition-mode A1 table; both table modes use
    the backward-compatible ``S`` and ``F`` flags.
    """
    key = effect.strip().lower()
    if key not in _EFFECTS and key not in _SET_EFFECTS:
        raise ValueError(f"Unknown ANOVA effect {effect!r}; choose one of "
                         f"{sorted(set(_EFFECTS) | set(_SET_EFFECTS))}")
    labels = load_anova_labels(path, correction=correction, alpha=alpha, roi=roi)
    if key in _SET_EFFECTS:
        include_cols, exclude_cols = _SET_EFFECTS[key]
    else:
        flag_col = {"CPC": "S", "SPS": "F"}.get(
            _EFFECTS[key][0], _EFFECTS[key][0])
        include_cols, exclude_cols = (flag_col,), ()
    needed_cols = include_cols + exclude_cols
    missing = [column for column in needed_cols if column not in labels]
    if missing:
        raise ValueError(f"{path} has no columns needed to define effect={effect!r} "
                         f"with correction={correction!r}; missing {missing}")
    mask = labels[list(include_cols)].eq(1).all(axis=1)
    if exclude_cols:
        mask &= labels[list(exclude_cols)].eq(0).all(axis=1)
    labels = labels[mask]

    selected = {}
    for row in labels.itertuples(index=False):
        subject = str(row.subject)
        electrode = str(row.electrode)
        prefix = f"{subject}-"
        if electrode.startswith(prefix):
            electrode = electrode[len(prefix):]
        row_roi = str(getattr(row, "roi", roi if roi is not None else "all"))
        selected.setdefault(row_roi, {}).setdefault(subject, []).append(electrode)
    for subjects in selected.values():
        for subject, electrodes in subjects.items():
            subjects[subject] = list(dict.fromkeys(electrodes))
    return selected


def selected_pairs(selection):
    """Flatten :func:`load_anova_label_electrodes` output to ``(sub, elec)``."""
    return {(subject, electrode)
            for subjects in selection.values()
            for subject, electrodes in subjects.items()
            for electrode in electrodes}
