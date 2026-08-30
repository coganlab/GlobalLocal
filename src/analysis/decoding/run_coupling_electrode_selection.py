"""Orchestration: coupling CSVs → coupled / matched-uncoupled electrode sets.

This is the glue ``decoding_dcc.py`` calls when ``args.coupling_electrode_selection``
is on. Every piece of logic lives in
:mod:`src.analysis.decoding.coupling_electrode_selection`; this file is the
recipe:

1. Read the PAC path's ``high_corr_<pairtype>_<condition>_<subject>.csv`` files
   for one pair type and one coupling-defining condition.
2. Resolve each bipolar channel's ROI from the electrode→ROI dict (never from
   which CSV column it landed in) and expand it to the monopolar contacts the
   decoder uses, intersected with the candidate pool the caller already built.
3. Emit one ``coupled`` set and ``n_draws`` n-matched ``uncoupled_drawNN`` sets.
4. Print and save the pool summary, so an underpowered run is visible in the job
   log before the bootstraps start rather than after.

The decoding battery then runs once per set, exactly as it already does for the
ANOVA electrode sets (`run_anova_electrode_selection`), and
:mod:`src.analysis.decoding.coupling_comparison` compares the saved results
across sets afterwards.

Circularity: READ THIS BEFORE INTERPRETING A POSITIVE RESULT
------------------------------------------------------------
Using the all-trial ``Stimulus`` file avoids the *obvious* circularity — defining
coupling on the big-letter split and then decoding the big letter would select
electrodes using the labels being decoded. It does **not** make the definition
orthogonal to the decodes, and an earlier version of this docstring wrongly
claimed it did.

The correlation in the CSVs is computed **across trials at each timepoint**, with
all stimulus trials pooled. Pooling does not remove the condition structure — it
contains it. Two electrodes that both respond more on task-g than task-l rise
together on g trials and fall together on l trials, so they correlate across
pooled trials from shared task tuning alone, with no coupling of any kind. The
across-trial correlation decomposes as::

    r_pooled  =  signal correlation (shared condition tuning)
               + noise correlation  (shared trial-to-trial fluctuation)

and only the second term is "coupling" in the sense the analysis intends.
Selecting on ``r_pooled`` therefore preferentially picks electrodes with strong
condition tuning — which is precisely what makes a condition decodable. In
simulation with the coupling set to exactly zero, selecting this way returned
electrodes with ~16x the condition variance of the unselected ones.

Whether that bias is actually present depends on the permutation null behind
``is_significantly_high``, which lives in the PAC script, not here:

- null shuffles trials **across all trials** → breaks signal and noise
  correlation alike, null sits at r ≈ 0, shared tuning is flagged significant.
  **The bias is live and points in the direction of the hypothesis.**
- null shuffles **within condition cell** → preserves condition alignment in the
  null, so only correlation above what shared tuning explains is flagged. The
  bias is controlled.

As of writing, the files in use come from the across-all-trials null, so a
positive coupled-vs-uncoupled decoding result **cannot be attributed to coupling**
without one of these fixes on the PAC side:

1. Demean each channel within the finest condition cell (bigLetter x smallLetter
   x task, 8 cells) before correlating, so the correlation is a noise
   correlation and one definition stays clean for all three decodes; or
2. Switch the permutation null to within-cell shuffling.

Signal correlation is not an uninteresting quantity — "do these electrodes share
task-related variance" is a real question. It just cannot be tested by decoding
task from the electrodes it selected, because that is the same measurement twice.
"""

from __future__ import annotations

import os

from src.analysis.decoding.coupling_electrode_selection import (
    build_coupling_electrode_sets,
    contacts_by_roi,
    coupling_run_slug,
    format_pool_summary,
    load_coupling_table,
    pair_type_rois,
    parse_draw_range,
    save_report,
)


# Where the PAC path writes its per-subject coupling CSVs, relative to LAB_root.
DEFAULT_COUPLING_CSV_SUBPATH = os.path.join(
    'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'freqFilt', 'figs',
    'high_corr_stats')


def resolve_coupling_csv_dir(args, LAB_root=None):
    """``args.coupling_csv_dir`` if set, else the standard derivatives path."""
    explicit = getattr(args, 'coupling_csv_dir', None)
    if explicit:
        return explicit
    if not LAB_root:
        raise ValueError(
            "coupling_csv_dir is unset and no LAB_root was supplied, so the "
            "high_corr CSVs cannot be located. Set COUPLING_CSV_DIR.")
    return os.path.join(LAB_root, DEFAULT_COUPLING_CSV_SUBPATH)


def resolve_coupling_rois(args):
    """The ROIs a pair type spans, checked against the run's ``rois_dict``."""
    roi_a, roi_b = pair_type_rois(args.coupling_pair_type)
    wanted = (roi_a,) if roi_a == roi_b else (roi_a, roi_b)
    known = set(args.rois_dict)
    missing = [roi for roi in wanted if roi not in known]
    if missing:
        raise ValueError(
            f"coupling pair type {args.coupling_pair_type!r} spans ROI(s) {missing} "
            f"that are not in this run's rois_dict ({sorted(known)}). Add them to "
            "ROIS before running, or the decode has nothing to run on.")
    return list(wanted)


def build_coupling_selected_electrode_sets(
    args,
    rois,
    electrodes,
    subjects_electrodes_to_rois_dict,
    save_dir,
    LAB_root=None,
):
    """Build the coupled and matched-uncoupled electrode sets for one pair type.

    Parameters
    ----------
    args : namespace
        Reads ``subjects``, ``rois_dict`` and the ``coupling_*`` knobs (see
        ``run_decoding_dcc.py``).
    rois : sequence of str
        The run's ROIs. Only those the pair type spans are populated; the rest
        are emitted empty so the decode loop skips them cleanly.
    electrodes : dict
        ``{roi: {sub: [chan]}}`` — the candidate pool (whatever ``args.electrodes``
        and the loaded epochs already narrowed it to). Coupled and control
        contacts are both intersected with this, so neither set can contain an
        electrode the decoder could not use anyway.
    subjects_electrodes_to_rois_dict : dict
        Already loaded by ``decoding_dcc.main``; passed in rather than re-read so
        the ROI assignment here is byte-identical to the one the decode used.

    Returns
    -------
    (electrode_sets, report)
        ``electrode_sets`` maps set name → ``{roi: {sub: [chan]}}``.
    """
    pair_rois = resolve_coupling_rois(args)
    csv_dir = resolve_coupling_csv_dir(args, LAB_root)
    subjects = list(args.subjects)

    # Which slice of the run this job builds. Defaults reproduce the old
    # behaviour exactly: every draw, plus the coupled set, in one job.
    draw_indices = parse_draw_range(
        getattr(args, 'coupling_draw_range', None), args.coupling_n_draws)
    include_coupled = bool(getattr(args, 'coupling_decode_coupled', True))
    if draw_indices is not None and not include_coupled and not draw_indices:
        raise ValueError("this job was asked to decode nothing: no coupled set "
                         "and no draws")

    slug = coupling_run_slug(
        args.coupling_pair_type, args.coupling_condition,
        match_level=args.coupling_match_level,
        match_degree=args.coupling_match_degree,
        n_draws=args.coupling_n_draws, seed=args.coupling_seed,
    )
    sel_dir = os.path.join(save_dir, 'coupling_selection', slug)
    os.makedirs(sel_dir, exist_ok=True)

    print(f"\n{'='*20} COUPLING-DEFINED ELECTRODE SETS "
          f"(pair type '{args.coupling_pair_type}', condition "
          f"'{args.coupling_condition}') {'='*20}\n")
    print(f"  csv_dir:      {csv_dir}")
    print(f"  rois:         {pair_rois}")
    print(f"  draws:        {args.coupling_n_draws} (seed {args.coupling_seed})")
    print(f"  match level:  {args.coupling_match_level}"
          f"{' + degree' if args.coupling_match_degree else ''}")
    covers_all = (draw_indices is None
                  or draw_indices == list(range(args.coupling_n_draws)))
    if covers_all and include_coupled:
        print(f"  this job:     the whole run (coupled + all "
              f"{args.coupling_n_draws} draws)")
    else:
        print(f"  this job:     {'coupled + ' if include_coupled else ''}"
              f"draws {_summarize_indices(draw_indices)} of "
              f"{args.coupling_n_draws} — a slice of a fanned-out run, so the "
              f"comparison must be run separately once every job has finished")

    table, used_files = load_coupling_table(
        csv_dir, args.coupling_pair_type,
        args.coupling_condition, subjects=subjects)
    print(f"  loaded {len(table)} pair rows from {len(used_files)} files "
          f"({table['subject'].nunique()} subjects)")

    roi_contacts = contacts_by_roi(
        subjects_electrodes_to_rois_dict, args.rois_dict, subjects=set(subjects))

    electrode_sets, report = build_coupling_electrode_sets(
        table=table,
        pair_type=args.coupling_pair_type,
        roi_contacts=roi_contacts,
        decoding_pool=electrodes,
        n_draws=args.coupling_n_draws,
        seed=args.coupling_seed,
        match_level=args.coupling_match_level,
        match_degree=args.coupling_match_degree,
        draw_indices=draw_indices,
        include_coupled=include_coupled,
    )

    report['condition'] = args.coupling_condition
    report['csv_dir'] = csv_dir
    report['csv_files'] = used_files
    report['subjects'] = subjects
    report['slug'] = slug

    print()
    print(format_pool_summary(report))

    empty = [roi for roi in pair_rois
             if report['per_roi'].get(roi, {}).get('n_coupled_total', 0) == 0]
    if len(empty) == len(pair_rois):
        raise ValueError(
            f"no coupled electrodes survived for any of {pair_rois}. Check that "
            f"the coupling CSVs' channel names match the electrode→ROI dict, that "
            f"the pair type is right, and that args.electrodes is not filtering "
            f"the coupled contacts away.")

    # Emit every ROI the run asked for. ROIs the pair type does not span come
    # back empty, and `run_decoding_for_one_electrode_set` skips empty sets.
    for name, elecset in electrode_sets.items():
        for roi in rois:
            elecset.setdefault(roi, {})

    # Fanned-out jobs run concurrently against one save_dir, so a partial job
    # must not clobber the whole-run report (or another slice's).
    filename = ('coupling_selection_report.json' if report['is_complete_run']
                else f"coupling_selection_report_draws-"
                     f"{_slug_indices(draw_indices)}"
                     f"{'_with_coupled' if include_coupled else ''}.json")
    path = save_report(report, sel_dir, filename=filename)
    print(f"\n  selection report saved to {path}")

    return electrode_sets, report


def _summarize_indices(indices):
    """``[0,1,2,5]`` → ``'0-2,5'`` — compact enough for a log line."""
    if indices is None:
        return 'all'
    runs, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    if start is not None:
        runs.append((start, prev))
    return ','.join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs)


def _slug_indices(indices):
    """Filename-safe form of :func:`_summarize_indices`."""
    return _summarize_indices(indices).replace(',', '_')
