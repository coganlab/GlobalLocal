"""Electrode sets from the top fraction of a per-electrode effect score,
defined on held-out trials.

The selection job (``dcc_scripts/decoding/score_electrode_selection_dcc.py``)
scores every candidate electrode on a ``frac_select`` share of each subject's
trials and writes the sets and the trial split to one JSON. The decoding and
power-traces jobs read that file and use only the other trials, so every job
works on the same split and no electrode is selected on the trials it is later
measured on.

The scores are those of ``naive_sensitivities(contrast_mode='proportion',
main_effects=True)`` on the selection trials: LWPC and LWPS (low- minus
high-proportion block, positive = the behavioural adaptation direction) and
the congruency and switch-type main effects (incongruent minus congruent,
switch minus repeat), each a Cohen's d on window-mean HG.

A selection made on every trial (``frac_select`` = 1) leaves the jobs every
trial too: selection and measurement then share their trial noise, so each
set's effects are inflated by having been picked for them (circular). Use it
only to describe the sets, never to test them.
"""

from __future__ import annotations

import json

import numpy as np

from src.analysis.utils.ranking import top_fraction

from .anova_electrode_selection import combine_electrode_sets

# naive_sensitivities' column -> set name
SCORE_SETS = {'x': 'lwpc', 'y': 'lwps', 'mx': 'congruency', 'my': 'switch_type'}
# pairs whose unique and shared electrodes are compared: <a>_only, <b>_only, <pair>_overlap
SCORE_PAIRS = {'adaptation': ('lwpc', 'lwps'), 'main_effects': ('congruency', 'switch_type')}


def score_electrode_sets(scores, frac=0.2, sign='positive'):
    """The electrode sets, from one row per electrode.

    ``scores`` has ``roi``, ``subject``, ``channel`` and any of the
    :data:`SCORE_SETS` columns. Within each ROI, with electrodes pooled over
    subjects, each score's top ``frac`` (by :func:`top_fraction`) is a set; each
    :data:`SCORE_PAIRS` pair with both scores present adds ``<a>_only``,
    ``<b>_only`` and ``<pair>_overlap``.

    Returns ``(sets, scores)``: ``{name: {roi: {subject: [channel]}}}``, and
    ``scores`` with an ``in_<name>`` column per score set.
    """
    scores = scores.copy()
    rois = sorted(scores['roi'].unique())
    named = {}
    for col, name in SCORE_SETS.items():
        if col not in scores:
            continue
        scores[f'in_{name}'] = False
        for _, g in scores.groupby('roi'):
            scores.loc[g.index, f'in_{name}'] = top_fraction(g[col], frac, sign)
        sel = scores[scores[f'in_{name}']]
        named[name] = {roi: {s: list(c['channel']) for s, c in
                             sel[sel['roi'] == roi].groupby('subject')}
                       for roi in rois}
    sets = dict(named)
    for pair, (a, b) in SCORE_PAIRS.items():
        if a in named and b in named:
            combo = combine_electrode_sets({a: named[a], b: named[b]}, rois,
                                           combos=[f'{a}_only', f'{b}_only', 'overlap'])
            sets[f'{a}_only'], sets[f'{b}_only'] = combo[f'{a}_only'], combo[f'{b}_only']
            sets[f'{pair}_overlap'] = combo['overlap']
    return sets, scores


def save_score_selection(path, sets, partitions, params):
    """Write the sets, the trial split (``{subject: {'select': ids, 'decode':
    ids}}``, from ``assign_trial_partitions``) and the parameters to one JSON."""
    doc = dict(params=params, sets=sets,
               partitions={s: {k: sorted(float(i) for i in ids) for k, ids in p.items()}
                           for s, p in partitions.items()})
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)
    return path


def load_score_selection(path):
    """``(sets, partitions, params)`` from :func:`save_score_selection`'s file;
    each side of the split is a set of trial ids, as ``apply_trial_partition``
    takes it."""
    with open(path) as f:
        doc = json.load(f)
    partitions = {s: {k: set(ids) for k, ids in p.items()}
                  for s, p in doc['partitions'].items()}
    return doc['sets'], partitions, doc['params']


def held_out_electrode_sets(path, rois, structure, names=None):
    """What a decoding or power-traces job needs from a selection file: the
    requested sets restricted to the electrodes ``structure`` holds, and
    ``structure`` cut to the trials selection did not use (every trial, for a
    selection made on all of them).

    ``structure`` is ``[subject][condition][key] -> Epochs`` with
    ``trial_count`` metadata (``create_subjects_mne_objects_dict``). ``names``
    picks sets (default: all). Raises when a ROI or a subject of the job is not
    in the file, rather than measuring on a split it does not cover.

    Returns ``(sets, decode_structure, params)``.
    """
    from src.analysis.utils.general_utils import (
        filter_electrode_lists_against_subjects_mne_objects)
    from .anova_electrode_selection import apply_trial_partition

    sets, partitions, params = load_score_selection(path)
    names = list(names) if names else list(sets)
    unknown = [n for n in names if n not in sets]
    if unknown:
        raise ValueError(f"{path} has no sets {unknown}; it has {list(sets)}")
    file_rois = set().union(*(sets[n].keys() for n in names))
    missing_rois = [r for r in rois if r not in file_rois]
    if missing_rois:
        raise ValueError(f"ROIs {missing_rois} were not selected in {path} "
                         f"(it covers {sorted(file_rois)})")
    missing_subs = [s for s in structure if s not in partitions]
    if missing_subs:
        raise ValueError(f"subjects {missing_subs} have no trial split in {path}")
    decode = apply_trial_partition(structure, partitions, which='decode')
    kept = {n: filter_electrode_lists_against_subjects_mne_objects(
                rois, {r: sets[n].get(r, {}) for r in rois}, decode)
            for n in names}
    return kept, decode, params
