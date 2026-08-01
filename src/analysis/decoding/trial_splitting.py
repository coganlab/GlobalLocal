"""Disjoint trial splitting to avoid electrode-definition ↔ decoding circularity.

Motivation
----------
When decoding is restricted to a *selected* electrode set, and that selection is
computed on the **same trials** the decoder then scores, the selection biases the
decoding accuracy upward (double-dipping; see plan §0.1/§0.2 and
`docs/decoding_and_electrode_definition_notes.md` §C). The bulletproof fix is a
disjoint trial partition:

    1. split each subject's trials into a definition set (``P_def``) and a
       decode set (``P_dec``), stratified so both stay balanced,
    2. define / select electrodes on ``P_def`` only,
    3. decode on ``P_dec`` only, restricted to those electrodes.

This module provides the **pure, unit-tested primitives** for steps 1–2
(`stratified_trial_split`, `select_responsive_channels`, plus metadata helpers),
and a thin orchestration helper (`apply_electrode_definition_split`) that wires
them onto the decoding pipeline's ``subjects_mne_objects`` structure. The
primitives carry the correctness guarantees; the orchestration is I/O glue and
should be smoke-tested on one subject before a full cluster run (it cannot be
validated outside the cluster).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Pure primitives (unit-tested in tests/analysis/decoding/test_trial_splitting.py)
# ---------------------------------------------------------------------------
def stratified_trial_split(strata, frac_def=0.5, seed=0):
    """Split trial indices into two disjoint sets, stratified by ``strata``.

    Within every stratum, a fixed fraction of trials is assigned to the
    definition set and the rest to the decode set, so the two sets stay balanced
    on ``strata`` (condition cell, block, run, …) and neither is temporally
    lopsided if ``strata`` encodes run/block.

    Parameters
    ----------
    strata : array-like, shape (n_trials,)
        A stratum key per trial (any hashable dtype). Pass a constant array for
        an unstratified random split.
    frac_def : float, default 0.5
        Fraction of each stratum assigned to the definition set (0 < frac < 1).
    seed : int
        Seed for the RNG, so a run is reproducible.

    Returns
    -------
    (idx_def, idx_dec) : tuple of np.ndarray
        Sorted int index arrays. Guaranteed **disjoint**, and together they
        cover every trial exactly once.

    Notes
    -----
    A stratum of size 1 (or any size where ``round(frac_def * n)`` hits 0 or n)
    is placed wholly on one side; at least one trial per stratum always goes to
    the definition set when the stratum is non-empty and ``frac_def > 0``.
    """
    if not 0.0 < frac_def < 1.0:
        raise ValueError(f"frac_def must be in (0, 1), got {frac_def}")
    strata = np.asarray(strata)
    n = strata.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.RandomState(seed)
    def_idx = []
    dec_idx = []
    # Deterministic stratum order (sorted unique) so the seed fully determines
    # the split regardless of input trial order.
    for key in _sorted_unique(strata):
        members = np.flatnonzero(strata == key)
        rng.shuffle(members)
        n_k = members.shape[0]
        n_def = int(round(frac_def * n_k))
        # Never let a non-empty stratum contribute zero definition trials (that
        # would silently drop it from the selection); cap so dec keeps ≥1 too
        # only when the stratum is big enough.
        n_def = min(max(n_def, 1), n_k)
        def_idx.append(members[:n_def])
        dec_idx.append(members[n_def:])

    idx_def = np.sort(np.concatenate(def_idx)) if def_idx else np.array([], dtype=int)
    idx_dec = (np.sort(np.concatenate(dec_idx)) if dec_idx
               else np.array([], dtype=int))
    return idx_def.astype(int), idx_dec.astype(int)


def _sorted_unique(arr):
    """Unique values in a stable, sortable order (falls back to str keys)."""
    try:
        return np.sort(np.unique(arr))
    except TypeError:
        return sorted(set(arr.tolist()), key=str)


def strata_key_from_metadata(metadata, strata_cols):
    """Build a single stratum key per trial from selected metadata columns.

    Missing columns are skipped with a warning (so a partial stratification still
    runs rather than crashing a long job). If no requested column is present, a
    constant key is returned (→ an unstratified random split).

    Parameters
    ----------
    metadata : pandas.DataFrame, shape (n_trials, ...)
        Per-trial metadata (e.g. ``epochs.metadata``).
    strata_cols : sequence of str or None
        Columns to stratify on (e.g. ``['congruency', 'switchType', 'blockType']``).

    Returns
    -------
    np.ndarray of str, shape (n_trials,)
    """
    n = len(metadata)
    if not strata_cols:
        return np.zeros(n, dtype=object)

    present = [c for c in strata_cols if c in metadata.columns]
    missing = [c for c in strata_cols if c not in metadata.columns]
    if missing:
        warnings.warn(
            f"strata_key_from_metadata: columns not in metadata, skipping: {missing}",
            stacklevel=2,
        )
    if not present:
        return np.zeros(n, dtype=object)

    # Join the present columns into one string key per row.
    key = metadata[present[0]].astype(str).to_numpy(dtype=object)
    for c in present[1:]:
        key = np.char.add(np.char.add(key.astype(str), "|"),
                           metadata[c].astype(str).to_numpy(dtype=str))
    return key


def select_responsive_channels(window_means, baseline_means=None, alpha=0.05):
    """Select responsive channels by a per-channel t-test with FDR across channels.

    This is the **held-out selector**: run it on the *definition* partition only.
    It deliberately uses a task-responsiveness contrast (window vs. baseline, or
    window vs. 0 for baseline-rescaled data), which is approximately orthogonal
    to any single decode contrast — the disjoint trials then make the selection
    strictly independent of the decode accuracy.

    Parameters
    ----------
    window_means : np.ndarray, shape (n_trials, n_channels)
        Per-trial mean activity in the analysis window.
    baseline_means : np.ndarray or None
        Per-trial mean activity in the baseline window. If None, a one-sample
        test of ``window_means`` against 0 is used (appropriate for
        baseline-rescaled high-gamma, which is centered at 0 pre-stimulus).
    alpha : float
        FDR (Benjamini–Hochberg) q-value threshold across channels.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_channels,)
        True for channels judged responsive. Channels with too few trials or a
        degenerate (zero-variance) distribution are False.
    """
    window_means = np.asarray(window_means, dtype=float)
    if window_means.ndim != 2:
        raise ValueError("window_means must be 2-D (n_trials, n_channels)")
    n_trials, n_channels = window_means.shape
    if n_trials < 2:
        return np.zeros(n_channels, dtype=bool)

    with np.errstate(invalid="ignore"):
        if baseline_means is None:
            sample = window_means
            t, p = stats.ttest_1samp(window_means, popmean=0.0, axis=0,
                                     nan_policy="omit")
        else:
            baseline_means = np.asarray(baseline_means, dtype=float)
            sample = window_means - baseline_means
            t, p = stats.ttest_rel(window_means, baseline_means, axis=0,
                                   nan_policy="omit")

    p = np.asarray(p, dtype=float)
    # Degenerate channels (≈zero across-trial variance in the tested sample) give
    # t = ±inf, p ≈ 0 — scipy would flag them "significant". These are flat /
    # dead channels; never select them.
    with np.errstate(invalid="ignore"):
        sample_std = np.nanstd(sample, axis=0)
    degenerate = ~np.isfinite(sample_std) | (sample_std <= 0)
    finite = np.isfinite(p) & ~degenerate
    mask = np.zeros(n_channels, dtype=bool)
    if not finite.any():
        return mask
    reject = np.zeros(n_channels, dtype=bool)
    reject_finite, _, _, _ = multipletests(p[finite], alpha=alpha, method="fdr_bh")
    reject[finite] = reject_finite
    return reject


# ---------------------------------------------------------------------------
# Pipeline orchestration (thin glue; smoke-test on one subject before a full run)
# ---------------------------------------------------------------------------
def _is_epochs_like(obj):
    """Duck-type check for an MNE Epochs-like object (indexable, has times)."""
    return (obj is not None
            and hasattr(obj, "get_data")
            and hasattr(obj, "times")
            and hasattr(obj, "__getitem__")
            and hasattr(obj, "__len__"))


def _time_mask(times, tmin=None, tmax=None):
    mask = np.ones_like(times, dtype=bool)
    if tmin is not None:
        mask &= times >= tmin
    if tmax is not None:
        mask &= times < tmax
    return mask


def apply_electrode_definition_split(
    subjects_mne_objects,
    electrodes,
    rois,
    *,
    frac_def=0.5,
    strata_cols=("congruency", "switchType", "blockType"),
    seed=0,
    alpha=0.05,
    window_tmin=0.0,
    baseline_tmax=None,
    epochs_key="HG_ev1_power_rescaled",
):
    """Recompute electrode selection on a held-out split; return decode-only data.

    Splits every ``(subject, condition)`` epochs object into a definition and a
    decode partition (stratified within the object by ``strata_cols``), selects
    responsive channels on the pooled definition trials per subject, restricts
    the passed-in ``electrodes`` to those channels, and returns the decode
    partition so the downstream bootstrap/decoder never sees the definition
    trials.

    Structures (matching ``decoding_dcc.py``):
      - ``subjects_mne_objects[sub][condition][epochs_key]`` → MNE Epochs
      - ``electrodes[roi][sub]`` → list of channel names

    Returns
    -------
    (decode_subjects_mne_objects, electrodes_restricted)

    Caveats (see docs §C)
    ---------------------
    - The split is applied per condition object. A single decode comparison uses
      *disjoint* event sets, so its def/dec split is clean; if a physical trial
      appears in two overlapping condition keys, it can land in def for one and
      dec for another, a minor residual leak. Prefer stratifying on a stable
      trial id if your metadata carries one.
    - Selection uses a task-responsiveness contrast on the definition trials
      (orthogonal to the decode), not the exact upstream sig-chan pipeline.
    - This helper is I/O glue: validate it on one subject before a full run.
    """
    decode_objs = {}
    def_partition = {}

    for sub, cond_dict in subjects_mne_objects.items():
        decode_objs[sub] = {}
        def_partition[sub] = {}
        for cond, obj_dict in cond_dict.items():
            epochs = obj_dict.get(epochs_key)
            if epochs is None:
                decode_objs[sub][cond] = obj_dict
                continue
            metadata = getattr(epochs, "metadata", None)
            n = len(epochs)
            if metadata is not None:
                strata = strata_key_from_metadata(metadata, strata_cols)
            else:
                strata = np.zeros(n, dtype=object)
            # Seed per (subject, condition) so the split is deterministic but
            # not identical across conditions.
            cond_seed = seed + (abs(hash((sub, cond))) % 100_000)
            idx_def, idx_dec = stratified_trial_split(strata, frac_def=frac_def,
                                                      seed=cond_seed)
            # Apply the SAME indices to every Epochs object in the dict (they
            # share trials in the same order), so any downstream signal stays
            # trial-aligned; carry non-Epochs values through untouched.
            dec_obj_dict = {}
            for key, val in obj_dict.items():
                if _is_epochs_like(val) and len(val) == n:
                    dec_obj_dict[key] = val[idx_dec]
                else:
                    dec_obj_dict[key] = val
            decode_objs[sub][cond] = dec_obj_dict
            def_partition[sub][cond] = epochs[idx_def]

    electrodes_restricted = _select_on_definition_partition(
        def_partition, electrodes, rois,
        alpha=alpha, window_tmin=window_tmin, baseline_tmax=baseline_tmax,
    )
    return decode_objs, electrodes_restricted


def _select_on_definition_partition(def_partition, electrodes, rois, *,
                                    alpha, window_tmin, baseline_tmax):
    """Restrict ``electrodes[roi][sub]`` to channels responsive on def trials."""
    restricted = {roi: {} for roi in rois}
    for roi in rois:
        for sub, candidate_chans in electrodes.get(roi, {}).items():
            candidate_chans = list(candidate_chans)
            if not candidate_chans:
                restricted[roi][sub] = []
                continue
            win_list, base_list = [], []
            for epochs in def_partition.get(sub, {}).values():
                present = [c for c in candidate_chans if c in epochs.ch_names]
                if not present or len(epochs) == 0:
                    continue
                data = epochs.copy().pick(present).get_data(copy=False)  # (trials, chan, time)
                times = epochs.times
                wmask = _time_mask(times, tmin=window_tmin)
                win = data[:, :, wmask].mean(axis=2)  # (trials, chan)
                # Reorder to full candidate ordering, NaN for absent channels.
                win_full = _to_full(win, present, candidate_chans)
                win_list.append(win_full)
                if baseline_tmax is not None:
                    bmask = _time_mask(times, tmax=baseline_tmax)
                    base = data[:, :, bmask].mean(axis=2)
                    base_list.append(_to_full(base, present, candidate_chans))
            if not win_list:
                restricted[roi][sub] = []
                continue
            window_means = np.concatenate(win_list, axis=0)
            baseline_means = (np.concatenate(base_list, axis=0)
                              if baseline_tmax is not None and base_list else None)
            mask = select_responsive_channels(window_means, baseline_means, alpha=alpha)
            restricted[roi][sub] = [c for c, keep in zip(candidate_chans, mask) if keep]
    return restricted


def _to_full(arr, present_chans, all_chans):
    """Expand (trials, len(present)) → (trials, len(all)) with NaN for missing."""
    idx = {c: j for j, c in enumerate(present_chans)}
    out = np.full((arr.shape[0], len(all_chans)), np.nan)
    for k, c in enumerate(all_chans):
        if c in idx:
            out[:, k] = arr[:, idx[c]]
    return out
