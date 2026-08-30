"""Electrode sets defined by high-gamma envelope coupling (the PAC path's pairs).

What this is for
----------------
The PAC path (`src/analysis/pac/`) tests every within-ROI-pair channel pair for
significantly high gamma-envelope correlation and writes one CSV per subject per
pair type::

    high_corr_<pairtype>_<condition>_<subject>.csv
    subject,ch1,ch2,is_significantly_high,mean_corr_across_time,w0,...,w249

A channel is **coupled** if it takes part in at least one pair flagged
``is_significantly_high``. This module turns that flag into the
``{roi: {subject: [channel]}}`` electrode sets the decoding battery consumes, so
the question "do occ electrodes coupled to LPFC decode the small letter better
than occ electrodes that are not?" becomes an ordinary run of the existing
pipeline over two named electrode sets.

The three things that make this non-trivial
-------------------------------------------
1. **Two namespaces.** PAC runs on bipolar derivatives (``LFAM9-LFAM10``);
   decoding runs on monopolar contacts (``LFAM9``) drawn from ``filtROI_dict``
   (`general_utils.filter_electrodes_by_roi`). Every coupled bipolar channel is
   therefore expanded to *both* its poles and intersected with the decoding pool
   that the caller already narrowed (task-significant ∩ ROI ∩ actually loaded).
   The **same** expansion runs when building the control pool, so the two sets
   are constructed by an identical rule and any bias the rule introduces is
   shared.

2. **Column order does not encode ROI.** In a real ``lpfc-occ`` file a sizeable
   minority of rows have ``ch1`` in occ and ``ch2`` in LPFC. ROI membership is
   therefore resolved per channel from the ROI dict
   (:func:`contacts_by_roi`), never from which column a name landed in. A
   bipolar channel counts as being in an ROI if *either* pole is
   (`env_correlation.find_roi_names`), which is also why a channel can in
   principle satisfy both ROIs of a heterotypic pair type.

3. **Set size drives decoding accuracy.** The decoding bootstrap resamples
   *trials*, not electrodes (`labeled_array_utils.make_bootstrapped_labeled_arrays_for_roi`);
   every electrode in a set is in every bootstrap. So an unmatched comparison
   between 15 coupled and 85 non-coupled electrodes measures set size, not
   coupling. :func:`draw_matched_control_sets` therefore draws control sets of
   the *same n*, repeatedly, and the comparison is against the distribution over
   draws rather than against one arbitrary draw.

Confounds this module can control (all off by default except the n-match)
------------------------------------------------------------------------
``match_level='within_subject'``
    Match n per subject rather than only in total, so the two sets have
    identical subject composition. Without it a coupled set concentrated in one
    subject can be compared against controls drawn mostly from another, and
    coverage/trial-count differences ride along with the coupling label.
``match_degree=True``
    "Coupled" means *significant in at least one pair*, so a channel tested
    against 13 partners is likelier to qualify than one tested against 5 with no
    real coupling difference. Partner counts do vary within a real file. With
    this on, controls are matched to coupled electrodes by greedy
    nearest-neighbour on partner count.

Neither is on by default: the default is the plain global n-match.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Pair types, and the ROI each side of the pair must fill. Keys are the
# *normalized* spellings (see `normalize_pair_type`) so `lpfc-occ`, `lpfcocc`
# and `lpfc_occ` in a filename all resolve to the same entry.
PAIR_TYPE_ROIS = {
    'lpfcocc': ('lpfc', 'occ'),
    'occlpfc': ('occ', 'lpfc'),
    'lpfclpfc': ('lpfc', 'lpfc'),
    'occocc': ('occ', 'occ'),
}

COUPLED_SET_NAME = 'coupled'
CONTROL_SET_PREFIX = 'uncoupled_draw'

SIG_COLUMN = 'is_significantly_high'
REQUIRED_COLUMNS = ('ch1', 'ch2', SIG_COLUMN)


# ---------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------
def normalize_pair_type(text):
    """``'lpfc-occ'``, ``'lpfc_occ'``, ``'LPFCocc'`` → ``'lpfcocc'``."""
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def pair_type_rois(pair_type):
    """The ``(roi_a, roi_b)`` a pair type's two endpoints must fill."""
    key = normalize_pair_type(pair_type)
    if key not in PAIR_TYPE_ROIS:
        raise ValueError(
            f"unknown pair type {pair_type!r} (normalized {key!r}); "
            f"known: {sorted(PAIR_TYPE_ROIS)}")
    return PAIR_TYPE_ROIS[key]


def parse_coupling_filename(name):
    """``'high_corr_lpfc-occ_Stimulus_D0107A.csv'`` → dict, or ``None``.

    Returns ``{'pair_type', 'condition', 'subject'}`` with ``pair_type``
    normalized. The pair type is the first ``_``-separated token (no spelling of
    it contains an underscore) and the subject is the last, so a condition made
    of several tokens survives the split.
    """
    stem = os.path.basename(str(name))
    if stem.endswith('.csv'):
        stem = stem[:-len('.csv')]
    if not stem.startswith('high_corr_'):
        return None
    rest = stem[len('high_corr_'):]
    pair_token, sep, remainder = rest.partition('_')
    if not sep or not remainder:
        return None
    condition, sep2, subject = remainder.rpartition('_')
    if not sep2 or not condition or not subject:
        return None
    return {'pair_type': normalize_pair_type(pair_token),
            'condition': condition,
            'subject': subject}


def find_coupling_csvs(csv_dir, pair_type, condition, subjects=None):
    """Every ``high_corr_*`` CSV in ``csv_dir`` matching pair type + condition.

    ``subjects`` (optional) restricts to those subject ids. Returns a list of
    ``(path, parsed)`` sorted by subject so a run is reproducible.
    """
    want_pair = normalize_pair_type(pair_type)
    keep_subs = set(subjects) if subjects is not None else None
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"coupling CSV directory not found: {csv_dir}")

    found = []
    for fname in sorted(os.listdir(csv_dir)):
        parsed = parse_coupling_filename(fname)
        if parsed is None:
            continue
        if parsed['pair_type'] != want_pair:
            continue
        if str(parsed['condition']) != str(condition):
            continue
        if keep_subs is not None and parsed['subject'] not in keep_subs:
            continue
        found.append((os.path.join(csv_dir, fname), parsed))
    return sorted(found, key=lambda pair: pair[1]['subject'])


def _coerce_bool(value):
    """CSV significance flags arrive as ``True``/``'True'``/``'TRUE'``/``1``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) and not (isinstance(value, float) and np.isnan(value))
    text = str(value).strip().lower()
    return text in ('true', 't', 'yes', 'y', '1')


def load_coupling_table(csv_dir, pair_type, condition, subjects=None):
    """Concatenate the matching per-subject CSVs into one tidy frame.

    Only the identity columns are kept — the ``w0..wN`` per-window correlations
    are not needed to decide membership and are large. Returns a frame with
    ``subject, ch1, ch2, is_significantly_high`` (bool), plus the list of files
    it came from.
    """
    paths = find_coupling_csvs(csv_dir, pair_type, condition, subjects=subjects)
    if not paths:
        raise FileNotFoundError(
            f"no high_corr CSVs for pair_type={pair_type!r} condition={condition!r} "
            f"in {csv_dir}"
            + (f" (subjects={sorted(subjects)})" if subjects is not None else ""))

    frames, used = [], []
    for path, parsed in paths:
        frame = pd.read_csv(path, usecols=lambda c: c in ('subject',) + REQUIRED_COLUMNS)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing required column(s) {missing}")
        if 'subject' not in frame.columns:
            frame['subject'] = parsed['subject']
        frame['subject'] = frame['subject'].astype(str)
        frame[SIG_COLUMN] = frame[SIG_COLUMN].map(_coerce_bool)
        frames.append(frame[['subject', 'ch1', 'ch2', SIG_COLUMN]])
        used.append(path)

    table = pd.concat(frames, ignore_index=True)
    if subjects is not None:
        table = table[table['subject'].isin(set(subjects))].reset_index(drop=True)
    return table, used


# ---------------------------------------------------------------------------
# channel names and ROI membership
# ---------------------------------------------------------------------------
_SUBJECT_PREFIX = re.compile(r'^D\d+[A-Za-z]?_')


def split_bipolar(channel):
    """``'LFAM9-LFAM10'`` → ``('LFAM9', 'LFAM10')``; a monopolar name → itself.

    Tolerates the two spellings in circulation: the fully written
    ``LFAM9-LFAM10`` the coupling CSVs use, and the abbreviated ``LFAM9-10``
    that ``save_bipolar_derivatives`` emits, whose right-hand side carries no
    letter prefix. An optional leading subject prefix (``D0103_``) is stripped.
    """
    name = _SUBJECT_PREFIX.sub('', str(channel).strip())
    left, sep, right = name.partition('-')
    if not sep or not right:
        return (name,) if name else ()
    if right[0].isdigit():
        prefix = re.match(r'^(\D+)\d+$', left)
        if prefix:
            right = prefix.group(1) + right
    return (left, right)


def contacts_by_roi(subjects_electrodes_to_rois_dict, rois_dict, subjects=None):
    """``{roi: {subject: set(contact)}}`` using the same rule as decoding.

    Mirrors :func:`general_utils.filter_electrodes_by_roi`: a ``filtROI_dict``
    key belongs to an ROI when any of that ROI's Destrieux selectors is a
    substring of the key.
    """
    out = {roi: {} for roi in rois_dict}
    for sub, entry in subjects_electrodes_to_rois_dict.items():
        if subjects is not None and sub not in subjects:
            continue
        filt = (entry or {}).get('filtROI_dict', {}) or {}
        for roi, selectors in rois_dict.items():
            contacts = set()
            for key, values in filt.items():
                if any(sel in key for sel in selectors):
                    contacts.update(values)
            out[roi][sub] = contacts
    return out


def bipolar_rois(channel, subject, roi_contacts, rois=None):
    """ROIs a bipolar channel belongs to — either pole suffices."""
    poles = split_bipolar(channel)
    names = rois if rois is not None else list(roi_contacts)
    hits = set()
    for roi in names:
        pool = roi_contacts.get(roi, {}).get(subject, set())
        if any(p in pool for p in poles):
            hits.add(roi)
    return hits


# ---------------------------------------------------------------------------
# from pair rows to per-ROI channel records
# ---------------------------------------------------------------------------
def build_channel_records(table, pair_type, roi_contacts):
    """``{roi: {subject: {bipolar_channel: {'partners': set, 'coupled': bool}}}}``.

    A row contributes an endpoint to ``roi`` only when the *other* endpoint
    fills the complementary ROI of the pair type — otherwise a row that merely
    happens to contain an LPFC channel would inflate the LPFC pool with pairs
    that were never LPFC↔occ tests. ``partners`` is what
    :func:`draw_matched_control_sets` uses as the electrode's degree.
    """
    roi_a, roi_b = pair_type_rois(pair_type)
    pair_rois = (roi_a,) if roi_a == roi_b else (roi_a, roi_b)
    complement = {roi_a: roi_b, roi_b: roi_a}

    records = {roi: defaultdict(dict) for roi in pair_rois}
    for row in table.itertuples(index=False):
        sub = str(row.subject)
        c1, c2 = str(row.ch1), str(row.ch2)
        sig = bool(getattr(row, SIG_COLUMN))
        rois1 = bipolar_rois(c1, sub, roi_contacts, pair_rois)
        rois2 = bipolar_rois(c2, sub, roi_contacts, pair_rois)

        for this_ch, this_rois, other_ch, other_rois in (
                (c1, rois1, c2, rois2), (c2, rois2, c1, rois1)):
            for roi in this_rois:
                if complement[roi] not in other_rois:
                    continue
                rec = records[roi][sub].setdefault(
                    this_ch, {'partners': set(), 'coupled': False})
                rec['partners'].add(other_ch)
                if sig:
                    rec['coupled'] = True

    return {roi: dict(by_sub) for roi, by_sub in records.items()}


def contact_pools(records, decoding_pool):
    """Expand bipolar records to monopolar contacts inside the decoding pool.

    Returns ``{roi: {subject: {'coupled': [...], 'control': [...],
    'degree': {contact: int}}}}``.

    A contact reached by both a coupled and an uncoupled bipolar channel is
    assigned to ``coupled`` and excluded from ``control`` — the sets must be
    disjoint or the "non-coupled" comparison silently contains coupled sites.
    ``degree`` is the largest partner count among the bipolar channels that
    contributed the contact.
    """
    pools = {}
    for roi, by_sub in records.items():
        allowed_by_sub = decoding_pool.get(roi, {})
        pools[roi] = {}
        for sub, by_chan in by_sub.items():
            allowed = set(allowed_by_sub.get(sub, []) or [])
            coupled, control, degree = set(), set(), {}
            for chan, rec in by_chan.items():
                contacts = [p for p in split_bipolar(chan) if p in allowed]
                if not contacts:
                    continue
                n_partners = len(rec['partners'])
                for contact in contacts:
                    degree[contact] = max(degree.get(contact, 0), n_partners)
                    (coupled if rec['coupled'] else control).add(contact)
            control -= coupled
            order = list(allowed_by_sub.get(sub, []) or [])
            pools[roi][sub] = {
                'coupled': [c for c in order if c in coupled],
                'control': [c for c in order if c in control],
                'degree': degree,
            }
    return pools


# ---------------------------------------------------------------------------
# matched draws
# ---------------------------------------------------------------------------
def _greedy_degree_match(targets, candidates, degree, rng):
    """Pick one candidate per target, nearest on degree, without replacement.

    ``degree`` is keyed by the full ``(subject, contact)`` key — contact names
    repeat across subjects, so keying on the name alone would cross subjects.
    Ties are broken at random so the match is not an artefact of list order.
    """
    remaining = list(candidates)
    chosen = []
    for target in targets:
        if not remaining:
            break
        want = degree.get(target, 0)
        dists = np.array([abs(degree.get(c, 0) - want) for c in remaining])
        best = np.flatnonzero(dists == dists.min())
        pick = int(best[rng.randint(len(best))])
        chosen.append(remaining.pop(pick))
    return chosen


def _draw_once(pools, roi, rng, match_level, match_degree):
    """One matched control draw for one ROI → ``({subject: [contact]}, shortfall)``."""
    by_sub = pools.get(roi, {})
    degree = {(sub, contact): deg
              for sub, entry in by_sub.items()
              for contact, deg in entry['degree'].items()}

    def flat(kind, subs):
        return [(sub, c) for sub in subs for c in by_sub[sub][kind]]

    if match_level == 'within_subject':
        groups = [[sub] for sub in sorted(by_sub)]
    elif match_level == 'global':
        groups = [sorted(by_sub)]
    else:
        raise ValueError(f"match_level must be 'global' or 'within_subject', "
                         f"got {match_level!r}")

    picked = []
    shortfall = 0
    for subs in groups:
        targets = flat('coupled', subs)
        candidates = flat('control', subs)
        n_want = len(targets)
        if n_want == 0:
            continue
        if match_degree:
            got = _greedy_degree_match(targets, candidates, degree, rng)
        elif len(candidates) <= n_want:
            got = list(candidates)
        else:
            idx = rng.choice(len(candidates), size=n_want, replace=False)
            got = [candidates[i] for i in sorted(idx)]
        shortfall += n_want - len(got)
        picked.extend(got)

    drawn = defaultdict(list)
    for sub, contact in picked:
        drawn[sub].append(contact)
    # keep each subject's channel order stable for reproducibility
    for sub in drawn:
        order = by_sub[sub]['control']
        drawn[sub] = [c for c in order if c in set(drawn[sub])]
    return dict(drawn), shortfall


def coupled_electrode_set(pools, rois):
    """The fixed ``{roi: {subject: [contact]}}`` set of coupled electrodes."""
    return {roi: {sub: list(entry['coupled'])
                  for sub, entry in pools.get(roi, {}).items()}
            for roi in rois}


def parse_draw_range(text, n_draws):
    """``'0-4'``, ``'3'``, ``'0-2,7'`` → ``[0, 1, 2, 3, 4]`` etc.; blank → ``None``.

    ``None`` means "every draw", which is what the single-job path wants. A
    subset is how the draws are fanned out across SLURM jobs: each job decodes
    only its slice, and every slice reproduces exactly the draw the whole-run
    job would have made at that index (see :func:`draw_matched_control_sets`).
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() == 'all':
        return None

    indices = set()
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        start, sep, end = token.partition('-')
        try:
            lo = int(start)
            hi = int(end) if sep else lo
        except ValueError:
            raise ValueError(
                f"draw range {text!r}: {token!r} is not an index or 'lo-hi' range")
        if sep and hi < lo:
            raise ValueError(f"draw range {text!r}: {token!r} runs backwards")
        indices.update(range(lo, hi + 1))

    if not indices:
        raise ValueError(f"draw range {text!r} selects no draws")
    out_of_range = sorted(i for i in indices if not 0 <= i < n_draws)
    if out_of_range:
        raise ValueError(
            f"draw range {text!r} selects draw(s) {out_of_range}, outside "
            f"0..{n_draws - 1}. The range indexes into the n_draws the "
            "comparison expects, so widening it means raising n_draws.")
    return sorted(indices)


def draw_matched_control_sets(pools, rois, n_draws=20, seed=0,
                              match_level='global', match_degree=False,
                              draw_indices=None):
    """``n_draws`` independent n-matched control sets, plus a shortfall report.

    Each draw is an ordinary ``{roi: {subject: [contact]}}`` electrode set, so
    the decoding battery runs on it unchanged. The distribution over draws is
    what the coupled set is ultimately tested against — that is the null
    "an arbitrary set of n matched non-coupled electrodes", which no single
    draw can supply.

    ``draw_indices`` builds only those draws, for splitting one logical run
    across several jobs. Draw ``k`` is seeded ``seed + k`` and named from ``k``
    and ``n_draws`` alone, so a subset is byte-identical to the same draws in a
    whole run — a fanned-out run and a single-job run produce the same sets, and
    the comparison cannot tell them apart. ``n_draws`` therefore stays the size
    of the WHOLE run even when only a slice is built.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if draw_indices is None:
        draw_indices = range(n_draws)
    else:
        draw_indices = sorted(set(int(i) for i in draw_indices))
        bad = [i for i in draw_indices if not 0 <= i < n_draws]
        if bad:
            raise ValueError(f"draw_indices {bad} are outside 0..{n_draws - 1}")
        if not draw_indices:
            raise ValueError("draw_indices is empty; pass None for every draw")

    sets, shortfalls = {}, {}
    width = max(2, len(str(n_draws - 1)))
    for draw in draw_indices:
        rng = np.random.RandomState(seed + draw)
        name = f"{CONTROL_SET_PREFIX}{draw:0{width}d}"
        by_roi, per_roi_short = {}, {}
        for roi in rois:
            drawn, shortfall = _draw_once(pools, roi, rng, match_level, match_degree)
            by_roi[roi] = drawn
            per_roi_short[roi] = shortfall
        sets[name] = by_roi
        shortfalls[name] = per_roi_short
    return sets, shortfalls


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def build_coupling_electrode_sets(
    table,
    pair_type,
    roi_contacts,
    decoding_pool,
    n_draws=20,
    seed=0,
    match_level='global',
    match_degree=False,
    draw_indices=None,
    include_coupled=True,
):
    """Coupling CSV rows → ``({set_name: electrode_set}, report)``.

    The returned dict is exactly what ``decoding_dcc.main`` feeds to
    ``run_decoding_for_one_electrode_set``: one ``'coupled'`` set and
    ``n_draws`` ``'uncoupled_drawNN'`` sets, each restricted to the ROIs the
    pair type spans.

    ``draw_indices`` and ``include_coupled`` select which of those sets this
    call emits, for fanning one run out across jobs. They change only *which*
    sets are returned, never their contents: the pools, the coupled set and
    draw ``k`` are all computed the same way regardless. The report always
    describes the full pools, so an underpowered run is visible in every job's
    log rather than only the one that happened to build the coupled set.
    """
    roi_a, roi_b = pair_type_rois(pair_type)
    rois = (roi_a,) if roi_a == roi_b else (roi_a, roi_b)

    records = build_channel_records(table, pair_type, roi_contacts)
    pools = contact_pools(records, decoding_pool)

    coupled = coupled_electrode_set(pools, rois)
    controls, shortfalls = draw_matched_control_sets(
        pools, rois, n_draws=n_draws, seed=seed,
        match_level=match_level, match_degree=match_degree,
        draw_indices=draw_indices)

    sets = {COUPLED_SET_NAME: coupled} if include_coupled else {}
    sets.update(controls)

    report = summarize_pools(pools, rois)
    report.update({
        'pair_type': normalize_pair_type(pair_type),
        'rois': list(rois),
        'n_draws': int(n_draws),
        'seed': int(seed),
        'match_level': match_level,
        'match_degree': bool(match_degree),
        'control_shortfall_per_draw': shortfalls,
        'n_pairs_in_table': int(len(table)),
        'n_significant_pairs': int(table[SIG_COLUMN].sum()) if len(table) else 0,
        # Which slice of the whole run this call built. `is_complete_run` is what
        # decoding_dcc checks before running the comparison in-job: a partial job
        # would compare the coupled trace against only the draws it happened to
        # build, silently shrinking the null distribution and the p-value floor.
        #
        # Completeness is about COVERAGE, not about how the caller spelled it: a
        # one-chunk fan-out passes an explicit '0-<n_draws-1>' rather than None,
        # and that job does hold the whole run.
        'draw_indices': list(draw_indices) if draw_indices is not None else None,
        'include_coupled': bool(include_coupled),
        'decoded_sets': list(sets),
        'is_complete_run': bool(
            include_coupled
            and (draw_indices is None
                 or sorted(set(draw_indices)) == list(range(n_draws)))),
    })
    return sets, report


def summarize_pools(pools, rois):
    """Per-ROI / per-subject counts — read this BEFORE spending cluster time.

    Coupled pairs are sparse (single digits per subject in real files), so the
    honest first question is whether enough coupled electrodes survive the
    bipolar→monopolar expansion and the decoding-pool intersection to decode at
    all. This is what answers it.
    """
    summary = {'per_roi': {}}
    for roi in rois:
        by_sub = pools.get(roi, {})
        rows = {}
        for sub in sorted(by_sub):
            entry = by_sub[sub]
            rows[sub] = {
                'n_coupled': len(entry['coupled']),
                'n_control_available': len(entry['control']),
                'coupled': list(entry['coupled']),
                'coupled_degrees': [entry['degree'].get(c, 0)
                                    for c in entry['coupled']],
                'control_degrees': [entry['degree'].get(c, 0)
                                    for c in entry['control']],
            }
        n_coupled = sum(r['n_coupled'] for r in rows.values())
        n_control = sum(r['n_control_available'] for r in rows.values())
        summary['per_roi'][roi] = {
            'n_coupled_total': n_coupled,
            'n_control_available_total': n_control,
            'n_subjects_with_coupled': sum(1 for r in rows.values() if r['n_coupled']),
            'n_subjects_tested': len(rows),
            'control_to_coupled_ratio': (n_control / n_coupled) if n_coupled else None,
            'per_subject': rows,
        }
    return summary


def format_pool_summary(report):
    """Human-readable table of :func:`summarize_pools` output, for job logs."""
    lines = []
    pair = report.get('pair_type', '?')
    lines.append(f"coupling electrode pools — pair type '{pair}' "
                 f"({report.get('n_significant_pairs', '?')} significant of "
                 f"{report.get('n_pairs_in_table', '?')} tested pairs)")
    for roi, stats in report.get('per_roi', {}).items():
        ratio = stats['control_to_coupled_ratio']
        ratio_str = f"{ratio:.1f}x" if ratio else "n/a"
        lines.append(
            f"  {roi}: {stats['n_coupled_total']} coupled electrodes in "
            f"{stats['n_subjects_with_coupled']}/{stats['n_subjects_tested']} subjects; "
            f"{stats['n_control_available_total']} eligible controls ({ratio_str})")
        for sub, row in stats['per_subject'].items():
            if row['n_coupled'] or row['n_control_available']:
                lines.append(f"    {sub}: coupled={row['n_coupled']:3d}  "
                             f"control_available={row['n_control_available']:3d}")
        if stats['n_coupled_total'] == 0:
            lines.append(f"    ✗ {roi} has no coupled electrodes — nothing to decode.")
        elif (stats['n_control_available_total'] < stats['n_coupled_total']):
            lines.append(f"    ⚠ {roi} has fewer eligible controls than coupled "
                         f"electrodes; draws will be short and unmatched.")
    return "\n".join(lines)


def save_report(report, save_dir, filename='coupling_selection_report.json'):
    """Write the selection report next to the decoding output."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def coupling_run_slug(pair_type, condition, match_level='global',
                      match_degree=False, n_draws=20, seed=0):
    """Filename-safe slug identifying one coupling-selection configuration."""
    parts = [normalize_pair_type(pair_type), re.sub(r'[^0-9A-Za-z]+', '', str(condition)),
             f"{n_draws}draws", f"seed{seed}"]
    if match_level != 'global':
        parts.append(match_level)
    if match_degree:
        parts.append('degmatch')
    return "_".join(parts)


def is_control_set_name(name):
    """True for the ``uncoupled_drawNN`` sets produced by this module."""
    return str(name).startswith(CONTROL_SET_PREFIX)
