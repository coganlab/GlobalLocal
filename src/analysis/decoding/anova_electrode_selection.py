"""Electrode selection on a held-out trial partition, via the power_traces ANOVA.

What this adds on top of ``trial_splitting.py``
-----------------------------------------------
``trial_splitting.apply_electrode_definition_split`` splits trials and then picks
electrodes with a **task-responsiveness t-test** — an orthogonal, cheap selector.
This module keeps the same disjoint-trial logic but swaps in the selector the
power-traces figures already use: the **within-electrode windowed ANOVA with
permutation cluster correction**
(:func:`src.analysis.power.windowed_anova.run_within_electrode_windowed_anova_cluster_correction`).
That makes the electrode sets fed to the decoder the *same* sets the power
traces call significant, so "decode from LWPC electrodes" means exactly what the
LWPC power-trace figure means.

Three pieces, in the order the pipeline uses them:

1. **A trial partition keyed on a stable trial id** (``metadata['trial_count']``),
   not on each condition object's positional index. This matters here in a way it
   did not before: selection now runs over a *different condition set* than the
   decode (LWPC/LWPS conditions to select, whatever you are decoding to decode),
   and a positional split cannot keep those disjoint — the same physical trial
   would land in the selection half of one condition object and the decode half
   of another. Keying on the trial id makes "selection trials" and "decode
   trials" disjoint **sets of physical trials**, whatever conditions each analysis
   slices them into. (`collect_subject_trial_strata`, `assign_trial_partitions`,
   `apply_trial_partition`.)
2. **Selection** — run the windowed ANOVA on the selection partition only and
   keep the electrodes with a surviving cluster for the requested effect
   (by default the highest-order interaction, i.e. the LWPC / LWPS term).
   (`select_electrodes_by_windowed_anova`.)
3. **Set algebra** over the resulting groups so the decoder can be pointed at
   LWPC-only, LWPS-only, the overlap, or the union.
   (`combine_electrode_sets`.)

The decoder then never sees a selection trial, so a set's decoding accuracy is
not inflated by the selection — the trial-level guard of guide §8, with the
selector upgraded to the ANOVA. The contrast-level guard (§3.2's
"ignore the diagonal") still applies on top: decoding the *same* contrast the
electrodes were selected on is only interpretable **because** of this split, and
comparing across sets (LWPC-only vs LWPS-only) is the intended reading.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from src.analysis.decoding.trial_splitting import (
    stratified_trial_split,
    strata_key_from_metadata,
)

# Metadata column holding a per-subject-unique physical trial id. Written by
# `make_metadata_from_event_names` from the event name's `TrialCount<n>` field.
DEFAULT_TRIAL_ID_COL = 'trial_count'

# Stratify the split on real metadata columns (see `parse_event_name`). NOTE the
# older `electrode_definition_split` default was
# ('congruency', 'switchType', 'blockType'); only `congruency` is an actual
# metadata column, so the other two were silently dropped with a warning. The
# parsed names are `task_sequence` (s/r) and `block_type` ((inc_prop, sw_prop)).
DEFAULT_STRATA_COLS = ('congruency', 'task_sequence', 'block_type')

DEFAULT_EPOCHS_KEY = 'HG_ev1_power_rescaled'


# ---------------------------------------------------------------------------
# 1. Trial partition keyed on a stable trial id
# ---------------------------------------------------------------------------
def _is_epochs_like(obj):
    """Duck-type check for an MNE Epochs-like object (indexable, has times)."""
    return (obj is not None
            and hasattr(obj, "get_data")
            and hasattr(obj, "times")
            and hasattr(obj, "__getitem__")
            and hasattr(obj, "__len__"))


def _trial_ids(epochs, id_col):
    """Per-trial ids for an Epochs object, or None if the column is absent."""
    metadata = getattr(epochs, "metadata", None)
    if metadata is None or id_col not in metadata.columns:
        return None
    return metadata[id_col].to_numpy()


def collect_subject_trial_strata(structures, strata_cols=DEFAULT_STRATA_COLS,
                                 id_col=DEFAULT_TRIAL_ID_COL,
                                 epochs_key=DEFAULT_EPOCHS_KEY):
    """Pool every structure's trials per subject into one (ids, strata) table.

    Parameters
    ----------
    structures : sequence of dict
        One or more ``subjects_mne_objects``-shaped dicts
        (``[sub][condition][epochs_key] -> Epochs``). Pass *every* structure the
        partition must cover — the decode conditions and each selection
        condition set — so one partition governs them all.
    strata_cols : sequence of str
        Metadata columns to stratify the split on.
    id_col : str
        Metadata column holding the stable per-subject trial id.

    Returns
    -------
    dict
        ``{subject: (ids ndarray, strata ndarray)}``, one row per *unique* trial
        id seen for that subject (first occurrence wins for the stratum key).

    Raises
    ------
    ValueError
        If no epochs object carries ``id_col``. Without a stable id the split
        cannot be made disjoint across condition sets; use the positional
        ``electrode_definition_split`` path instead, or attach trial ids.
    """
    per_subject = {}
    saw_id_col = False
    for structure in structures:
        for sub, cond_dict in structure.items():
            for obj_dict in cond_dict.values():
                epochs = obj_dict.get(epochs_key) if isinstance(obj_dict, dict) else None
                if not _is_epochs_like(epochs) or len(epochs) == 0:
                    continue
                ids = _trial_ids(epochs, id_col)
                if ids is None:
                    continue
                saw_id_col = True
                metadata = epochs.metadata
                strata = strata_key_from_metadata(metadata, strata_cols)
                store = per_subject.setdefault(sub, {})
                for tid, skey in zip(ids, strata):
                    store.setdefault(tid, skey)

    if not saw_id_col:
        raise ValueError(
            f"No epochs object carries a '{id_col}' metadata column, so trials "
            f"cannot be split disjointly across condition sets. Attach trial ids "
            f"(make_metadata_from_event_names writes 'trial_count') or use the "
            f"positional electrode_definition_split path."
        )

    out = {}
    for sub, store in per_subject.items():
        ids = np.array(list(store.keys()))
        strata = np.array([store[t] for t in ids], dtype=object)
        order = np.argsort(ids)   # deterministic order regardless of dict order
        out[sub] = (ids[order], strata[order])
    return out


def assign_trial_partitions(sub_trials, frac_select=0.3, seed=0):
    """Split each subject's trial ids into a selection set and a decode set.

    Parameters
    ----------
    sub_trials : dict
        Output of :func:`collect_subject_trial_strata`.
    frac_select : float
        Fraction of each stratum assigned to **electrode selection**; the rest
        goes to decoding (e.g. 0.3 → 30% select / 70% decode).
    seed : int
        Base RNG seed. Offset per subject so subjects do not share a shuffle.

    Returns
    -------
    dict
        ``{subject: {'select': set_of_ids, 'decode': set_of_ids}}`` — disjoint,
        and together covering every id.
    """
    partitions = {}
    for sub in sorted(sub_trials):
        ids, strata = sub_trials[sub]
        sub_seed = seed + (abs(hash(sub)) % 100_000)
        idx_sel, idx_dec = stratified_trial_split(strata, frac_def=frac_select,
                                                  seed=sub_seed)
        partitions[sub] = {'select': set(ids[idx_sel].tolist()),
                           'decode': set(ids[idx_dec].tolist())}
    return partitions


def apply_trial_partition(structure, partitions, which='decode',
                          id_col=DEFAULT_TRIAL_ID_COL, verbose=True):
    """Restrict every Epochs object in ``structure`` to one side of the split.

    Objects in the same ``[sub][condition]`` dict that are trial-aligned with the
    reference epochs (same length) are sliced with the same indices, so any
    downstream signal stays aligned; everything else (Evoked averages, scalars)
    is carried through untouched. Conditions left with zero trials are dropped,
    as are subjects left with no conditions — the same cleanup
    ``create_subjects_mne_objects_dict`` does, so downstream code sees a
    structure of the shape it already expects.

    Parameters
    ----------
    structure : dict
        ``[sub][condition][key] -> Epochs``.
    partitions : dict
        Output of :func:`assign_trial_partitions`.
    which : {'select', 'decode'}
    """
    if which not in ('select', 'decode'):
        raise ValueError(f"which must be 'select' or 'decode', got {which!r}")

    out = {}
    for sub, cond_dict in structure.items():
        keep_ids = partitions.get(sub, {}).get(which)
        if keep_ids is None:
            if verbose:
                print(f"  [trial-split] {sub}: no partition assigned; dropping.")
            continue
        sub_out = {}
        for cond, obj_dict in cond_dict.items():
            ref = None
            for val in obj_dict.values():
                if _is_epochs_like(val) and _trial_ids(val, id_col) is not None:
                    ref = val
                    break
            if ref is None:
                # Nothing sliceable / identifiable — carry the condition through
                # untouched rather than silently dropping data.
                sub_out[cond] = obj_dict
                continue
            n = len(ref)
            ids = _trial_ids(ref, id_col)
            idx = np.flatnonzero(np.isin(ids, list(keep_ids)))
            if idx.size == 0:
                if verbose:
                    print(f"  [trial-split] {sub}/{cond}: 0 trials on the "
                          f"'{which}' side; dropping this condition.")
                continue
            new_obj = {}
            for key, val in obj_dict.items():
                if _is_epochs_like(val) and len(val) == n:
                    new_obj[key] = val[idx]
                else:
                    new_obj[key] = val
            sub_out[cond] = new_obj
        if sub_out:
            out[sub] = sub_out
        elif verbose:
            print(f"  [trial-split] {sub}: no conditions survived the "
                  f"'{which}' partition; dropping subject.")
    return out


def partition_trial_counts(structure, epochs_key=DEFAULT_EPOCHS_KEY):
    """``{subject: total trials}`` across conditions — for split sanity prints."""
    counts = {}
    for sub, cond_dict in structure.items():
        total = 0
        for obj_dict in cond_dict.values():
            epochs = obj_dict.get(epochs_key) if isinstance(obj_dict, dict) else None
            if _is_epochs_like(epochs):
                total += len(epochs)
        counts[sub] = total
    return counts


# ---------------------------------------------------------------------------
# 2. Selection: the power_traces cluster-corrected windowed ANOVA
# ---------------------------------------------------------------------------
def interaction_effect_name(anova_factors):
    """statsmodels name of the highest-order interaction over ``anova_factors``.

    ``['congruency', 'incongruentProportion']`` →
    ``'C(congruency):C(incongruentProportion)'`` — the LWPC term. This is the
    default selector because the construct under test *is* the interaction (guide
    §3.1): a congruency main effect means "responds to conflict", not "implements
    the list-wide adjustment".
    """
    return ':'.join(f'C({f})' for f in anova_factors)


def resolve_effect(effect, anova_factors):
    """Map an ``effect`` request onto a concrete effect name (or None for 'any').

    ``'interaction'`` → the highest-order interaction; ``'any'`` → None (an
    electrode counts if *any* effect survives); a bare factor name (e.g.
    ``'congruency'``) → ``'C(congruency)'``; anything else is passed through
    verbatim so an explicit statsmodels name always works.
    """
    if effect in (None, 'any'):
        return None
    if effect == 'interaction':
        if not anova_factors or len(anova_factors) < 2:
            raise ValueError(
                f"effect='interaction' needs >=2 anova_factors, got {anova_factors}")
        return interaction_effect_name(anova_factors)
    if anova_factors and effect in anova_factors:
        return f'C({effect})'
    return effect


def safe_effect_slug(effect_name):
    """Filename-safe form of a statsmodels effect name (matches windowed_anova)."""
    if effect_name is None:
        return 'any_effect'
    return effect_name.replace(':', '_x_').replace('C(', '').replace(')', '')


def _usable_subjects(structure, condition_names, subjects, epochs_key,
                     min_trials=1, verbose=True):
    """Subjects that have every ANOVA condition with >= ``min_trials`` trials.

    ``process_windowed_data_for_anova`` indexes ``[sub][cond][epochs_key]``
    unguarded, so a subject missing one cell of the design would raise. Dropping
    those subjects here (loudly) is also statistically right: a subject missing a
    cell cannot contribute to a factorial ANOVA anyway.
    """
    usable = []
    for sub in subjects:
        cond_dict = structure.get(sub)
        if not cond_dict:
            if verbose:
                print(f"  [anova-select] {sub}: no data; excluded.")
            continue
        missing = []
        for cond in condition_names:
            epochs = (cond_dict.get(cond) or {}).get(epochs_key)
            if not _is_epochs_like(epochs) or len(epochs) < min_trials:
                missing.append(cond)
        if missing:
            if verbose:
                print(f"  [anova-select] {sub}: missing/empty conditions "
                      f"{missing}; excluded from the ANOVA.")
            continue
        usable.append(sub)
    return usable


def _first_times(structure, epochs_key=DEFAULT_EPOCHS_KEY):
    for cond_dict in structure.values():
        for obj_dict in cond_dict.values():
            epochs = obj_dict.get(epochs_key) if isinstance(obj_dict, dict) else None
            if _is_epochs_like(epochs) and len(epochs) > 0:
                return np.asarray(epochs.times)
    return None


def significant_electrodes_from_summary(summary_df, rois, candidate_electrodes,
                                        effect_name=None, use_fdr=True,
                                        alpha=0.05):
    """Filter a within-electrode ANOVA ``summary.csv`` frame to an electrode set.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Tidy summary from
        ``run_within_electrode_windowed_anova_cluster_correction`` — one row per
        (subject, electrode, roi, effect, cluster).
    effect_name : str or None
        Keep only this effect; None keeps an electrode if *any* effect survives.
    use_fdr : bool
        Use the BH-FDR-across-electrodes decision (``sig_after_fdr``) rather than
        the raw cluster p. FDR is the right family here: one test per electrode
        (guide §2.4).
    alpha : float
        Threshold for the raw-p route when ``use_fdr`` is False.

    Returns
    -------
    dict
        ``{roi: {subject: [electrode, ...]}}``, intersected with
        ``candidate_electrodes`` so the ROI membership the caller established is
        never widened by the selection.
    """
    selected = {roi: {} for roi in rois}
    if summary_df is None or summary_df.empty:
        return selected

    df = summary_df
    if effect_name is not None:
        df = df[df['effect'] == effect_name]
    if use_fdr and 'sig_after_fdr' in df.columns:
        df = df[df['sig_after_fdr'].astype(bool)]
    else:
        df = df[df['cluster_p_value'].fillna(1.0) < alpha]

    sig_pairs = set(map(tuple, df[['subject', 'electrode']].drop_duplicates().to_numpy()))
    for roi in rois:
        for sub, chans in candidate_electrodes.get(roi, {}).items():
            selected[roi][sub] = [c for c in chans if (sub, c) in sig_pairs]
    return selected


def select_electrodes_by_windowed_anova(
    selection_mne_objects,
    electrodes,
    rois,
    subjects,
    conditions_obj,
    anova_factors,
    *,
    effect='interaction',
    window_size=64,
    step_size=16,
    sampling_rate=256,
    n_perm=200,
    percentile=95,
    cluster_percentile=95,
    min_trials_per_cell=2,
    split_clusters_by_sign=True,
    use_fdr=True,
    alpha=0.05,
    seed=42,
    n_jobs=-1,
    verbose=True,
    save_dir=None,
    run_label=None,
    epochs_key=DEFAULT_EPOCHS_KEY,
):
    """Run the power-traces within-electrode ANOVA and return the significant set.

    This is the *same* call the power-traces job makes
    (``process_windowed_data_for_anova`` →
    ``run_within_electrode_windowed_anova_cluster_correction``), pointed at the
    selection partition instead of all trials. Passing ``save_dir``/``run_label``
    makes it write the ordinary run directory (``summary.csv``, per-electrode
    ``.npz``, ``sig_electrodes_*.json``, ``run_config.json``), so a selection run
    is inspectable with the same tooling as a power-traces run and can be
    re-loaded later with ``load_significant_electrodes``.

    Returns
    -------
    dict
        ``{'selected': {roi: {sub: [chan, ...]}}, 'summary_df': DataFrame,
        'effect': str or None, 'skipped': list, 'window_centers': ndarray,
        'subjects_used': list}``
    """
    # Imported lazily: `windowed_anova` pulls in `ieeg` through `general_utils`,
    # which is not importable off-cluster. Keeping it here means this module's
    # pure helpers (and their tests) stay importable anywhere.
    from src.analysis.power.windowed_anova import (
        process_windowed_data_for_anova,
        run_within_electrode_windowed_anova_cluster_correction,
    )

    condition_names = list(conditions_obj.keys())
    subjects_used = _usable_subjects(selection_mne_objects, condition_names,
                                     subjects, epochs_key,
                                     min_trials=min_trials_per_cell,
                                     verbose=verbose)
    effect_name = resolve_effect(effect, anova_factors)

    empty = {roi: {sub: [] for sub in electrodes.get(roi, {})} for roi in rois}
    if not subjects_used:
        warnings.warn("select_electrodes_by_windowed_anova: no subject has every "
                      "ANOVA condition on the selection partition; selecting no "
                      "electrodes.", stacklevel=2)
        return {'selected': empty, 'summary_df': pd.DataFrame(),
                'effect': effect_name, 'skipped': [], 'window_centers': None,
                'subjects_used': []}

    # Restrict the candidate electrodes to the usable subjects so the ANOVA's
    # electrode list and its data list stay index-aligned (the windowed-data
    # builder and the dataframe builder skip subjects in lockstep).
    candidates = {roi: {sub: list(chans)
                        for sub, chans in electrodes.get(roi, {}).items()
                        if sub in subjects_used and chans}
                  for roi in rois}

    times = _first_times(selection_mne_objects, epochs_key)
    if times is None:
        raise RuntimeError("No usable epochs found on the selection partition.")

    if verbose:
        n_cand = sum(len(v) for d in candidates.values() for v in d.values())
        print(f"[anova-select] {len(subjects_used)} subjects, {n_cand} candidate "
              f"electrodes, effect={effect_name or 'any'}, n_perm={n_perm}")

    windowed_data = process_windowed_data_for_anova(
        selection_mne_objects, condition_names, rois, subjects_used,
        candidates, window_size=window_size, step_size=step_size,
        sampling_rate=sampling_rate,
    )

    (_results, window_centers, summary_df, skipped) = \
        run_within_electrode_windowed_anova_cluster_correction(
            windowed_data, conditions_obj, anova_factors, rois, subjects_used,
            electrodes_per_subject_roi=candidates,
            times=times,
            window_size=window_size, step_size=step_size,
            sampling_rate=sampling_rate,
            n_perm=n_perm,
            percentile=percentile,
            cluster_percentile=cluster_percentile,
            min_trials_per_cell=min_trials_per_cell,
            split_clusters_by_sign=split_clusters_by_sign,
            seed=seed, n_jobs=n_jobs, verbose=verbose,
            save_dir=save_dir, run_label=run_label,
        )

    selected = significant_electrodes_from_summary(
        summary_df, rois, candidates, effect_name=effect_name,
        use_fdr=use_fdr, alpha=alpha,
    )
    # Re-add the excluded subjects with empty lists so every downstream consumer
    # sees the same subject keys it started with.
    for roi in rois:
        for sub in electrodes.get(roi, {}):
            selected[roi].setdefault(sub, [])

    if verbose:
        n_sel, n_subs = electrode_set_counts(selected)
        print(f"[anova-select] {n_sel} electrodes across {n_subs} subjects "
              f"survived ({'FDR q' if use_fdr else 'cluster p'} < {alpha})")
        if skipped:
            print(f"[anova-select] {len(skipped)} electrodes skipped "
                  f"(min_trials_per_cell={min_trials_per_cell}) — expected to "
                  f"rise as the selection fraction shrinks.")

    return {'selected': selected, 'summary_df': summary_df,
            'effect': effect_name, 'skipped': skipped,
            'window_centers': window_centers, 'subjects_used': subjects_used}


# ---------------------------------------------------------------------------
# 3. Set algebra over the selected groups
# ---------------------------------------------------------------------------
def _as_pairs(elecset, rois):
    """``{roi: {(sub, chan), ...}}`` view of a ``{roi: {sub: [chan]}}`` set."""
    return {roi: {(sub, c) for sub, chans in elecset.get(roi, {}).items()
                  for c in chans}
            for roi in rois}


def _all_subjects(named_sets, rois):
    subs = {roi: [] for roi in rois}
    for elecset in named_sets.values():
        for roi in rois:
            for sub in elecset.get(roi, {}):
                if sub not in subs[roi]:
                    subs[roi].append(sub)
    return subs


def available_combo_names(labels):
    """Combo names produced by :func:`combine_electrode_sets` for ``labels``."""
    names = list(labels)
    if len(labels) >= 2:
        names += [f'{lab}_only' for lab in labels] + ['overlap', 'union']
    return names


def combine_electrode_sets(named_sets, rois, combos=None):
    """Build unique / overlapping / union electrode sets from named sets.

    Parameters
    ----------
    named_sets : dict
        ``{label: {roi: {subject: [chan, ...]}}}`` — e.g.
        ``{'lwpc': ..., 'lwps': ...}``.
    rois : sequence of str
    combos : sequence of str or None
        Which combinations to build. ``None`` builds all of
        :func:`available_combo_names`. Recognized names:

        =============  ==========================================================
        ``<label>``    every electrode significant for that label
        ``<label>_only``  significant for that label and **no** other label
        ``overlap``    significant for **every** label (the shared population)
        ``union``      significant for **at least one** label
        =============  ==========================================================

    Returns
    -------
    dict
        ``{combo_name: {roi: {subject: [chan, ...]}}}``, channel order preserved
        from the first label that contributed the electrode so a set is
        reproducible run to run.

    Notes
    -----
    ``<label>_only`` is the electrode-level analogue of the conjunction's
    ``S_only`` / ``F_only`` cells (guide §5), and the *point* of the comparison:
    if LWPC-only and LWPS-only electrodes each decode their own contrast but not
    the other's, that is positive evidence for segregated subpopulations in a way
    a single pooled decode cannot give. ``overlap`` is the mixed-selectivity
    control — a shared code and orthogonal codes on the same electrodes look the
    same in a count, but not in a decode.
    """
    labels = list(named_sets.keys())
    if combos is None:
        combos = available_combo_names(labels)

    pairs = {lab: _as_pairs(named_sets[lab], rois) for lab in labels}
    subs_by_roi = _all_subjects(named_sets, rois)

    def _build(pred):
        out = {}
        for roi in rois:
            all_pairs = set().union(*[pairs[lab][roi] for lab in labels]) if labels else set()
            keep = {p for p in all_pairs if pred(p, roi)}
            by_sub = {sub: [] for sub in subs_by_roi[roi]}
            # Preserve each label's channel ordering, first label wins.
            for lab in labels:
                for sub, chans in named_sets[lab].get(roi, {}).items():
                    for c in chans:
                        if (sub, c) in keep and c not in by_sub.setdefault(sub, []):
                            by_sub[sub].append(c)
            out[roi] = by_sub
        return out

    result = {}
    for name in combos:
        if name in labels:
            result[name] = _build(lambda p, roi, lab=name: p in pairs[lab][roi])
        elif name.endswith('_only') and name[:-len('_only')] in labels:
            lab = name[:-len('_only')]
            others = [o for o in labels if o != lab]
            result[name] = _build(
                lambda p, roi, lab=lab, others=others:
                p in pairs[lab][roi] and not any(p in pairs[o][roi] for o in others))
        elif name == 'overlap':
            result[name] = _build(
                lambda p, roi: all(p in pairs[lab][roi] for lab in labels))
        elif name == 'union':
            result[name] = _build(
                lambda p, roi: any(p in pairs[lab][roi] for lab in labels))
        else:
            raise ValueError(
                f"Unknown electrode-set combination {name!r}. "
                f"Available: {available_combo_names(labels)}")
    return result


# ---------------------------------------------------------------------------
# 4. Naming — so every figure says which electrodes it was decoded from
# ---------------------------------------------------------------------------
def electrode_set_counts(elecset):
    """``(n_electrodes, n_subjects_contributing)`` for a ``{roi: {sub: [chan]}}``."""
    n_elec, subs = 0, set()
    for by_sub in elecset.values():
        for sub, chans in by_sub.items():
            n_elec += len(chans)
            if chans:
                subs.add(sub)
    return n_elec, len(subs)


def electrode_set_counts_for_roi(elecset, roi):
    """``(n_electrodes, n_subjects)`` restricted to one ROI."""
    by_sub = elecset.get(roi, {})
    n_elec = sum(len(c) for c in by_sub.values())
    n_subs = sum(1 for c in by_sub.values() if c)
    return n_elec, n_subs


def electrode_set_slug(name):
    """Filename-safe slug for an electrode-set name."""
    return re.sub(r'[^0-9A-Za-z]+', '_', str(name)).strip('_')


_PRETTY = {
    'overlap': 'overlap',
    'union': 'union',
}


def pretty_electrode_set(name):
    """Human-readable electrode-set name for a figure title."""
    if name in _PRETTY:
        return _PRETTY[name]
    if name.endswith('_only'):
        return f"{name[:-len('_only')].upper()}-only"
    return name.replace('_', ' ').upper() if len(name) <= 8 else name.replace('_', ' ')


# Prefixes the pipeline glues onto an electrode-set label on its way into a
# filename. They carry no information a reader of the figure needs.
_ELECSET_PREFIXES = ('anova_csv_elecs_', 'anova_csv_', 'csv_elecs_', 'elecset_')


def short_electrode_set_label(raw_label):
    """Electrode-set name with the pipeline's filename prefixes stripped.

    ``'anova_csv_elecs_anova_congruency_only'`` -> ``'congruency-only'``.
    The full label is what ``describe_electrode_set`` puts on diagnostic
    figures; this is the version that fits in a one-line title.
    """
    if not raw_label:
        return ''
    text = str(raw_label)
    for prefix in _ELECSET_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = re.sub(r'^anova_', '', text)
    text = re.sub(r'_elecs$', '', text)
    if text in ('', 'all'):
        return 'all'
    return text.replace('_only', '-only').replace('_', ' ')


def short_decoding_figure_title(analysis_name, electrode_set_label=None):
    """One-line title: ``'LWPC in congruency-only electrodes'``.

    ``decoding_figure_title`` stacks the electrode count, the subject count
    and the raw comparison key onto a figure that also has to carry two
    legends, which is how these panels ended up with more text than data. The
    ROI, the counts and the analysis parameters are all still in the filename
    and the directory tree, where they are searchable and where they are not
    competing with the traces for space.
    """
    title = str(analysis_name).replace('_', ' ')
    electrodes = short_electrode_set_label(electrode_set_label)
    if electrodes:
        title = f"{title} in {electrodes} electrodes"
    return title


def describe_electrode_set(name, elecset, roi=None):
    """``'LWPC-only electrodes (n = 42, 11 subjects)'`` for titles/prints."""
    if roi is None:
        n_elec, n_subs = electrode_set_counts(elecset)
    else:
        n_elec, n_subs = electrode_set_counts_for_roi(elecset, roi)
    return (f"{pretty_electrode_set(name)} electrodes "
            f"(n = {n_elec}, {n_subs} subjects)")


def decoding_figure_title(condition_comparison, roi, electrode_set_desc=None,
                          extra=None):
    """Two-line title: what is being decoded, then from which electrodes.

    The second line is the whole point of this module — a decoding figure that
    does not say which electrode set it came from is unreadable once several
    sets are in play.
    """
    line1 = f"{str(condition_comparison).replace('_', ' ')} — {roi}"
    parts = [line1]
    if electrode_set_desc:
        parts.append(str(electrode_set_desc))
    if extra:
        parts.append(str(extra))
    return "\n".join(parts)
