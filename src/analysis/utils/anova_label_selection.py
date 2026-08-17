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
    """Load electrodes associated with a selected effect from an A1 label table.

    Read an A1 ``anova_labels.csv`` table, select electrodes associated with
    the requested ANOVA effect, and group the resulting bare electrode names
    by ROI and subject.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an A1 ``anova_labels.csv`` file or to the result directory
        containing that file. The table must contain ``subject`` and
        ``electrode`` columns, along with the flag, p-value, or q-value columns
        required by ``effect`` and ``correction``.

    effect : str, default="lwpc"
        Effect or electrode-set definition to select. The value is
        case-insensitive and surrounding whitespace is ignored. Supported
        values are:

        - ``"lwpc"``, ``"s"``, or ``"congruency"``: select every electrode
          with the ``S`` flag, including electrodes that also have the ``F``
          flag.
        - ``"lwps"``, ``"f"``, or ``"switch_type"``: select every electrode
          with the ``F`` flag, including electrodes that also have the ``S``
          flag.
        - ``"lwpc_only"`` or ``"congruency_only"``: select electrodes with
          ``S == 1`` and ``F == 0``.
        - ``"lwps_only"`` or ``"switch_type_only"``: select electrodes with
          ``F == 1`` and ``S == 0``.
        - ``"both"``: select the intersection, where both ``S == 1`` and
          ``F == 1``.
        - ``"cpc"``, ``"sps"``, ``"cps"``, or ``"spc"``: select the
          corresponding effect. For backward compatibility, ``"cpc"`` uses
          the ``S`` selection and ``"sps"`` uses the ``F`` selection.

        Use the ``lwpc`` and ``lwps`` names for proportion-mode A1 results,
        and the ``congruency`` and ``switch_type`` names for condition-mode A1
        results. Both modes use the backward-compatible ``S`` and ``F`` flags.

    correction : {"flags", "none", "raw", "fdr", "fdr_bh", "q"}, \
default="flags"
        Method used to determine whether each effect is significant:

        - ``"flags"`` uses the binary effect flags already stored in the CSV
          and therefore preserves the correction method used when the table
          was generated.
        - ``"none"`` or ``"raw"`` recomputes the flags by testing the
          applicable raw p-value column against ``alpha``.
        - ``"fdr"``, ``"fdr_bh"``, or ``"q"`` recomputes the flags by testing
          the applicable saved q-value column against ``alpha``.

        Significance comparisons are strict: a value is selected when it is
        less than ``alpha``.

    alpha : float, default=0.05
        Significance threshold used when ``correction`` selects raw p-values
        or saved q-values. This argument does not change saved flags when
        ``correction="flags"``.

    roi : str or None, default=None
        If provided, restrict rows to those whose ``roi`` value matches this
        value after conversion to strings. The input table must contain an
        ``roi`` column when this argument is not ``None``. If omitted, rows
        from all ROIs are eligible for selection.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Nested mapping of the form
        ``{roi: {subject: [electrode, ...]}}``.

        ROI and subject keys are strings. Electrode names are returned in
        their original table order with duplicates removed. If an electrode
        name begins with ``"<subject>-"``, that prefix is removed; for
        example, ``"D1-A1"`` for subject ``"D1"`` is returned as ``"A1"``.
        If the table has no ``roi`` column and no ROI filter was requested,
        selected electrodes are placed under the ``"all"`` key. An empty
        dictionary is returned when no electrodes satisfy the selection.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not identify an existing CSV file or a directory
        containing ``anova_labels.csv``.

    ValueError
        If the table lacks the required ``subject`` or ``electrode`` columns;
        if ``effect`` or ``correction`` is unsupported; if ROI filtering is
        requested but the table has no ``roi`` column; or if the table lacks
        the flag, p-value, or q-value columns needed to define the requested
        selection.

    Notes
    -----
    Electrode ordering follows the order of matching rows in the input table.
    Duplicate electrode names within the same ROI and subject are discarded
    while preserving their first occurrence.
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
