"""Drop named electrodes from a {roi: {subject: [channels]}} selection.

One electrode with a large excursion shifts the across-electrode mean by
``excursion / n_electrodes``. For a ~170-electrode ROI that is enough to put a
visible feature on the power trace and a cluster on the ANOVA, and no amount of
condition balancing or cluster-statistic tuning removes it -- the electrode has
to come out. ``dcc_scripts/power/diagnose_electrode_deviations.py`` names the
candidates; this applies the decision.

Deliberately free of MNE and ieeg imports so every entrypoint can share it.
"""

from __future__ import annotations


def parse_exclusions(entries):
    """Split exclusion entries into (global_names, per_subject_pairs).

    An entry is either ``"CHAN"`` (drop that channel name in every subject) or
    ``"SUB:CHAN"`` (drop it in one subject only). Whitespace is stripped and
    empty entries are ignored.
    """
    global_names = set()
    per_subject = set()
    for entry in entries or []:
        entry = str(entry).strip()
        if not entry:
            continue
        if ':' in entry:
            sub, chan = entry.split(':', 1)
            sub, chan = sub.strip(), chan.strip()
            if sub and chan:
                per_subject.add((sub, chan))
        else:
            global_names.add(entry)
    return global_names, per_subject


def filter_out_excluded_electrodes(electrodes, entries):
    """Return (filtered_electrodes, n_dropped).

    Parameters
    ----------
    electrodes : dict
        ``{roi: {subject: [channel_name, ...]}}``. Not modified in place.
    entries : iterable of str
        Exclusion entries as accepted by :func:`parse_exclusions`.

    Raises
    ------
    ValueError
        If the exclusions would leave no electrodes at all. That is nearly
        always a name mismatch rather than an intended selection, and it fails
        far less confusingly here than downstream on an empty ROI.
    """
    global_names, per_subject = parse_exclusions(entries)
    if not global_names and not per_subject:
        return {roi: dict(per_sub) for roi, per_sub in electrodes.items()}, 0

    filtered = {}
    n_dropped = 0
    for roi, per_sub in electrodes.items():
        filtered[roi] = {}
        for sub, elec_list in per_sub.items():
            kept = [e for e in elec_list
                    if e not in global_names and (sub, e) not in per_subject]
            n_dropped += len(elec_list) - len(kept)
            filtered[roi][sub] = kept

    remaining = sum(len(v) for per_sub in filtered.values() for v in per_sub.values())
    if not remaining:
        raise ValueError(
            f"exclude_electrodes {sorted(global_names | {f'{s}:{c}' for s, c in per_subject})} "
            f"removed every electrode; check the names against the deviation report."
        )
    return filtered, n_dropped
