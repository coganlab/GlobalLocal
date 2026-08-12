"""Load electrode sets from an A1 ``anova_labels.csv`` result.

This is deliberately independent of MNE so power-trace, decoding, and
cross-decoding entrypoints use exactly the same selection semantics.
"""

from __future__ import annotations

from pathlib import Path

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
}


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
    """Return ``{roi: {subject: [bare electrode names]}}`` from A1 labels."""
    key = effect.strip().lower()
    if key not in _EFFECTS:
        raise ValueError(f"Unknown ANOVA effect {effect!r}; choose one of "
                         f"{sorted(_EFFECTS)}")
    labels = load_anova_labels(path, correction=correction, alpha=alpha, roi=roi)
    flag_col = {"CPC": "S", "SPS": "F"}.get(_EFFECTS[key][0], _EFFECTS[key][0])
    if flag_col not in labels:
        raise ValueError(f"{path} has no columns needed to define effect={effect!r} "
                         f"with correction={correction!r}")
    labels = labels[labels[flag_col] == 1]

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
