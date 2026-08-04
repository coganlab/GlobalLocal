"""Orchestration: held-out trial split → ANOVA electrode sets → decode partition.

This is the glue `decoding_dcc.py` calls when
``args.anova_electrode_selection`` is on. It is deliberately thin — every piece
of logic it strings together lives in
:mod:`src.analysis.decoding.anova_electrode_selection` (split, selection, set
algebra) or in :mod:`src.analysis.power.windowed_anova` (the ANOVA itself), so
this file can be read as the recipe:

1. Load a *separate* epochs structure per selection condition set (e.g. the four
   LWPC cells, the four LWPS cells). Selection needs the 2×2 the ANOVA's
   interaction term is defined over, which is generally **not** the condition
   set being decoded.
2. Build one trial partition per subject over the union of every structure's
   trials, keyed on the stable trial id, and split it ``frac_select`` /
   ``1 - frac_select``.
3. Run the power-traces within-electrode windowed ANOVA on the *selection* side
   of that partition, once per condition set, and keep the electrodes with a
   surviving cluster on the requested effect.
4. Hand back the *decode* side of the partition plus the electrode sets
   (each label, each label's unique electrodes, the overlap, the union).

Why a separate load per selection label rather than reusing the decode epochs:
the ANOVA is fit on condition cells, and the decode conditions may be a
different (often coarser or block-balanced) partition of the same trials. Going
back to the epochs file for each selection condition set is the only way to get
the exact factorial cells the registry declares — and because step 2 keys the
split on trial ids rather than positions, the reload costs nothing in
disjointness.
"""

from __future__ import annotations

import json
import os

from src.analysis.config.condition_registry import (
    get_anova_factors,
    get_conditions_obj,
)
from src.analysis.utils.general_utils import (
    create_subjects_mne_objects_dict,
    filter_electrode_lists_against_subjects_mne_objects,
)

from src.analysis.decoding.anova_electrode_selection import (
    DEFAULT_STRATA_COLS,
    DEFAULT_TRIAL_ID_COL,
    apply_trial_partition,
    assign_trial_partitions,
    available_combo_names,
    collect_subject_trial_strata,
    combine_electrode_sets,
    describe_electrode_set,
    electrode_set_counts,
    partition_trial_counts,
    resolve_effect,
    safe_effect_slug,
    select_electrodes_by_windowed_anova,
)


def short_label(condition_label):
    """``'stimulus_lwpc_conditions'`` → ``'lwpc'`` — a name fit for a filename."""
    name = str(condition_label)
    for prefix in ('stimulus_', 'response_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith('_conditions'):
        name = name[:-len('_conditions')]
    return name or str(condition_label)


def build_anova_selected_electrode_sets(
    args,
    rois,
    electrodes,
    decode_subjects_mne_objects,
    save_dir,
    LAB_root=None,
):
    """Split trials, select electrodes by ANOVA, and return the decode partition.

    Parameters
    ----------
    args : namespace
        Reads: ``subjects``, ``epochs_root_file``, ``task``, ``acc_trials_only``,
        ``window_size``, ``step_size``, ``sampling_rate``, ``n_jobs``, and the
        ``electrode_selection_*`` knobs (see ``run_decoding_dcc.py``).
    electrodes : dict
        ``{roi: {sub: [chan]}}`` — the candidate pool the ANOVA is run over
        (whatever ``args.electrodes`` already narrowed it to).
    decode_subjects_mne_objects : dict
        The structure built from the *decode* conditions. Returned split down to
        the decode partition.

    Returns
    -------
    (decode_objects, electrode_sets, report)
        ``electrode_sets`` maps set name → ``{roi: {sub: [chan]}}``.
    """
    labels = list(args.electrode_selection_condition_labels)
    if not labels:
        raise ValueError(
            "anova_electrode_selection is on but "
            "electrode_selection_condition_labels is empty — nothing to select on.")

    frac_select = float(args.electrode_selection_frac)
    seed = int(getattr(args, 'electrode_selection_seed', 0))
    strata_cols = tuple(getattr(args, 'electrode_selection_strata', None)
                        or DEFAULT_STRATA_COLS)
    id_col = getattr(args, 'electrode_selection_trial_id_col', DEFAULT_TRIAL_ID_COL)

    sel_dir = os.path.join(save_dir, 'electrode_selection')
    os.makedirs(sel_dir, exist_ok=True)

    print(f"\n{'='*20} ANOVA ELECTRODE SELECTION ON A HELD-OUT TRIAL SPLIT "
          f"({frac_select:.0%} select / {1 - frac_select:.0%} decode) {'='*20}\n")
    print(f"  selection condition sets: {labels}")
    print(f"  stratified on: {list(strata_cols)}   trial id: {id_col}   seed: {seed}")

    # --- 1. Load one epochs structure per selection condition set -------------
    selection_structures = {}
    for label in labels:
        conditions_obj = get_conditions_obj(label)
        print(f"\n[anova-select] loading epochs for selection set '{label}' "
              f"({len(conditions_obj)} conditions)")
        selection_structures[label] = create_subjects_mne_objects_dict(
            subjects=args.subjects,
            epochs_root_file=args.epochs_root_file,
            conditions=conditions_obj,
            task=args.task,
            just_HG_ev1_rescaled=True,
            acc_trials_only=args.acc_trials_only,
            LAB_root=LAB_root,
        )

    # --- 2. One trial partition per subject, over every structure ------------
    sub_trials = collect_subject_trial_strata(
        [decode_subjects_mne_objects] + list(selection_structures.values()),
        strata_cols=strata_cols, id_col=id_col,
    )
    partitions = assign_trial_partitions(sub_trials, frac_select=frac_select,
                                         seed=seed)
    n_sel_ids = sum(len(p['select']) for p in partitions.values())
    n_dec_ids = sum(len(p['decode']) for p in partitions.values())
    print(f"\n[anova-select] partitioned {n_sel_ids + n_dec_ids} trials across "
          f"{len(partitions)} subjects: {n_sel_ids} selection / {n_dec_ids} decode")

    # --- 3. Select electrodes on the selection partition, per condition set ---
    named_sets = {}
    per_label_report = {}
    for label in labels:
        name = short_label(label)
        anova_factors = get_anova_factors(label)
        if not anova_factors:
            raise ValueError(
                f"Selection condition set '{label}' has no 'anova_factors' in "
                f"condition_registry.py — the ANOVA selector needs them.")

        sel_structure = apply_trial_partition(
            selection_structures[label], partitions, which='select', id_col=id_col)
        # Candidate electrodes must exist in the *selection* epochs too.
        candidates = filter_electrode_lists_against_subjects_mne_objects(
            rois, electrodes, sel_structure)

        print(f"\n[anova-select] --- '{name}' ({label}) --- factors={anova_factors}")
        counts = partition_trial_counts(sel_structure)
        if counts:
            print(f"  selection trials per subject (min/median/max): "
                  f"{min(counts.values())}/"
                  f"{sorted(counts.values())[len(counts) // 2]}/"
                  f"{max(counts.values())}")

        effect_name = resolve_effect(args.electrode_selection_effect, anova_factors)
        result = select_electrodes_by_windowed_anova(
            sel_structure,
            candidates,
            rois,
            args.subjects,
            get_conditions_obj(label),
            anova_factors,
            effect=args.electrode_selection_effect,
            window_size=args.window_size,
            step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            n_perm=int(args.electrode_selection_n_perm),
            percentile=int(getattr(args, 'electrode_selection_percentile', 95)),
            cluster_percentile=int(
                getattr(args, 'electrode_selection_cluster_percentile', 95)),
            min_trials_per_cell=int(
                getattr(args, 'electrode_selection_min_trials_per_cell', 2)),
            split_clusters_by_sign=bool(
                getattr(args, 'electrode_selection_split_clusters_by_sign', True)),
            use_fdr=bool(getattr(args, 'electrode_selection_use_fdr', True)),
            alpha=float(getattr(args, 'electrode_selection_alpha', 0.05)),
            seed=seed,
            n_jobs=args.n_jobs,
            verbose=True,
            save_dir=sel_dir,
            run_label=f"{name}_{safe_effect_slug(effect_name)}",
        )
        named_sets[name] = result['selected']
        n_elec, n_subs = electrode_set_counts(result['selected'])
        per_label_report[name] = {
            'condition_label': label,
            'anova_factors': list(anova_factors),
            'effect': result['effect'],
            'n_electrodes': n_elec,
            'n_subjects': n_subs,
            'n_skipped_electrodes': len(result['skipped']),
            'subjects_used': result['subjects_used'],
        }

    # --- 4. Set algebra ------------------------------------------------------
    requested = getattr(args, 'electrode_selection_sets', None)
    if requested:
        requested = list(requested)
        allowed = available_combo_names(list(named_sets))
        unknown = [r for r in requested if r not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown electrode_selection_sets {unknown}; available: {allowed}")
    electrode_sets = combine_electrode_sets(named_sets, rois, combos=requested)

    print(f"\n[anova-select] electrode sets built:")
    for name, elecset in electrode_sets.items():
        print(f"  - {describe_electrode_set(name, elecset)}")

    # --- 5. Decode partition -------------------------------------------------
    decode_objects = apply_trial_partition(
        decode_subjects_mne_objects, partitions, which='decode', id_col=id_col)
    dec_counts = partition_trial_counts(decode_objects)
    print(f"\n[anova-select] decode partition: {len(decode_objects)} subjects, "
          f"{sum(dec_counts.values())} trials in the decode conditions")

    report = {
        'frac_select': frac_select,
        'seed': seed,
        'strata_cols': list(strata_cols),
        'trial_id_col': id_col,
        'effect': args.electrode_selection_effect,
        'use_fdr': bool(getattr(args, 'electrode_selection_use_fdr', True)),
        'alpha': float(getattr(args, 'electrode_selection_alpha', 0.05)),
        'n_perm': int(args.electrode_selection_n_perm),
        'labels': per_label_report,
        'sets': {name: {'n_electrodes': electrode_set_counts(es)[0],
                        'n_subjects': electrode_set_counts(es)[1],
                        'electrodes': {roi: {s: list(c) for s, c in by_sub.items() if c}
                                       for roi, by_sub in es.items()}}
                 for name, es in electrode_sets.items()},
        'n_selection_trials': n_sel_ids,
        'n_decode_trials': n_dec_ids,
    }
    report_path = os.path.join(sel_dir, 'electrode_selection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[anova-select] wrote {report_path}")

    return decode_objects, electrode_sets, report
