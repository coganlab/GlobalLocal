"""A4 — cross-decoding: train on one contrast, score against another (plan §4).

Co-localization != shared CODE. The counting analyses (A1/A2) can show that the
*same electrodes* are selective for both stability (LWPC) and flexibility (LWPS),
but they cannot say whether those "both" electrodes carry ONE shared
representational geometry or a mix of two orthogonal codes. Cross-decoding is the
representation-level test that disambiguates them: fit a classifier on one
contrast and ask whether its decision axis transfers to the other.

This module is deliberately thin. The decoding pipeline it needs already exists:

- **Pseudopopulation** — `put_data_in_labeled_array_per_roi_subject` pads each
  subject's trials to the per-condition max with NaN and concatenates subjects
  along the CHANNEL axis, so an ROI's LabeledArray is already a cross-subject
  pseudopopulation; `mixup2` fills the padding.
- **Disjoint train/test** — `Decoder.cv_cm_jim_window_shuffle` cross-validates,
  so the trials a transferred accuracy is scored on were never trained on. That
  is the circularity guard; nothing extra is required.
- **The null** — `shuffle=True` permutes the TRAIN labels and refits, so the null
  carries the variance of the entire estimation pipeline. For a cross-decode it
  is exactly the right null: "does an axis trained on real congruency labels
  predict switchType better than one trained on scrambled labels?"
- **Multiple comparisons** — the ordinary bootstrap + `time_perm_cluster` stack
  corrects the time-resolved accuracy trace across windows.

So all this module supplies is (1) the contrast definitions, (2) the
electrode-definition ↔ decode double-dipping table, and (3) the glue that turns
an ROI LabeledArray into the two label vectors `cv_cm_jim_window_shuffle` needs.

Everything else — pseudo-trial synthesis, reservoir splitting, a private
classifier factory, a bespoke permutation null — used to live here and has been
removed in favour of the pipeline above.

Designs
-------
- **Label transfer** (Design a): `run_cross_decoding(..., train_strings=STABILITY,
  test_strings=FLEXIBILITY)`. Run it per electrode group (LWPC-only, LWPS-only,
  `both`); the prediction is that only `both` transfers above chance. This is
  the ALL-congruency vs ALL-switchType decode: the classes span every condition
  cell, so it is already pooled over both block proportions. Only the
  within-block designs below split by proportion.
- **Set comparison** (Design b): the same contrast decoded within each electrode
  set — an ordinary decode, just with `electrodes` restricted. Use the normal
  decoding path.
- **Temporal generalization** (Design c, Fig 10): pass
  `temporal_generalization=True` to get a train-window × test-window matrix.
- **Within-block 2×2** (Design 0, Fig 9): decode a contrast separately within each
  block level — ordinary condition comparisons in `condition_registry`. Use
  `is_circular_decode` to skip the cell that would double-dip (below).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# contrasts: a contrast maps a condition cell to a binary class {0, 1} (or None
# to drop). The full cell also carries the block proportions, so a "within-block"
# restriction is just a contrast that returns None outside the block.
# ---------------------------------------------------------------------------
DEFAULT_CELL_COLS = ("congruency", "switchType",
                     "incongruent_proportion", "switch_proportion")


def _congruency_label(cell):
    v = cell["congruency"]
    return {"i": 1, "c": 0}.get(v, None)


def _switch_label(cell):
    v = cell["switchType"]
    return {"s": 1, "r": 0}.get(v, None)


# named contrasts. 'stability' == congruency (i vs c), 'flexibility' == switchType
# (s vs r); the process aliases make cross-decoding calls read like the plan.
CONTRASTS = {
    "congruency": _congruency_label,
    "switchType": _switch_label,
    "stability": _congruency_label,
    "flexibility": _switch_label,
}


def resolve_contrast(contrast):
    """Accept a name (str, looked up in CONTRASTS) or a cell->label callable."""
    if callable(contrast):
        return contrast
    if contrast in CONTRASTS:
        return CONTRASTS[contrast]
    raise KeyError(f"unknown contrast {contrast!r}; known: {list(CONTRASTS)} "
                   "or pass a cell->{0,1,None} callable")


# ---------------------------------------------------------------------------
# condition cells -> the class groups `build_cross_decoding_arrays` wants
# ---------------------------------------------------------------------------
# The Decoder identifies classes by SUBSTRINGS of the condition name, which makes
# the class definitions a hostage to the naming convention. The two conventions
# in play here really do collide:
#
#   real (`experiment_conditions.stimulus_experiment_conditions`)
#       Stimulus_{c|i}{25|75}{s|r}{25|75}      e.g. Stimulus_i25s75
#   synthetic (`synthetic_roi_labeled_arrays`)
#       Stimulus_{c|i}_{r|s}_{25|75}inc_{25|75}sw   e.g. Stimulus_i_s_25inc_75sw
#
# '75s' means "switch trial in the 75%-incongruent block" in the first and
# "75%-switch block" in the second, so ONE hand-written substring set cannot
# serve both -- and getting it wrong does not raise, it silently decodes the
# wrong contrast.
#
# So don't parse names at all. Each condition already declares its factor levels
# (the real conditions in their config entry, the synthetic ones via
# `synthetic_condition_cells`); read those, and emit the class groups as the FULL
# condition names belonging to each level. A full name is trivially a substring
# of itself, so the existing substring machinery works unchanged and no collision
# is possible.
CELL_FIELDS = ("congruency", "switchType",
               "incongruent_proportion", "switch_proportion")

# The two factors a cross-decode transfers BETWEEN. They are the only ones a
# condition must declare: the block proportions are needed by the within-block
# designs, not by the transfer itself, so a condition set that pools over them
# (`stimulus_main_effect_conditions` — Stimulus_i{r,s} / Stimulus_c{r,s}, each
# collapsing all four proportion cells) is a legitimate — and larger-per-cell —
# input for "ALL congruency vs ALL switch type". See `has_block_factor`.
CROSS_DECODE_FIELDS = ("congruency", "switchType")

# The config spells the proportions in camelCase; the long-table / stats side of
# the project uses snake_case. Accept either so a condition dict from either
# convention drops in.
_FIELD_ALIASES = {
    "congruency": ("congruency",),
    "switchType": ("switchType", "switch_type", "task_sequence"),
    "incongruent_proportion": ("incongruentProportion", "incongruent_proportion"),
    "switch_proportion": ("switchProportion", "switch_proportion"),
}


def _as_proportion(value):
    """'25%' / '25' / 25.0 / 25 -> 25 (int). None if it isn't a proportion."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip('%')
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def condition_cells(conditions, required=CROSS_DECODE_FIELDS):
    """`{condition_name: {field: level}}` from an `experiment_conditions`-style dict.

    `conditions` maps condition name -> its metadata entry (the same dict the
    decoding pipeline is driven by, e.g.
    ``experiment_conditions.stimulus_experiment_conditions``). A condition is kept
    only if it declares every factor in `required`; undeclared factors are
    recorded as None.

    `required` defaults to the two factors the transfer runs BETWEEN
    (`CROSS_DECODE_FIELDS`), so both of these are valid inputs:

    - the full 2x2x2x2 (`stimulus_experiment_conditions`, 16 cells) — needed by
      the within-block designs, and stratifying folds on all four factors;
    - the pooled 2x2 (`stimulus_main_effect_conditions`, 4 cells) — congruency x
      switchType collapsed over BOTH proportions, i.e. "all congruency vs all
      switch type" with ~4x the trials per cell. `has_block_factor` reports
      False for both proportions there, and the block-split designs are skipped.

    Pass `required=CELL_FIELDS` to demand the full four-factor cell.
    """
    unknown = set(required) - set(CELL_FIELDS)
    if unknown:
        raise ValueError(f"unknown required field(s) {sorted(unknown)}; "
                         f"condition cells carry {CELL_FIELDS}")
    cells = {}
    for name, meta in conditions.items():
        if not isinstance(meta, dict):
            continue
        cell = {}
        for field, aliases in _FIELD_ALIASES.items():
            value = next((meta[a] for a in aliases if a in meta), None)
            if field.endswith('proportion'):
                value = _as_proportion(value)
            cell[field] = value
        if any(cell[f] is None for f in required):
            continue                       # missing a factor the decode needs
        cells[name] = cell
    if not cells:
        raise ValueError(
            f"no condition declares all of {tuple(required)}; a cross-decode "
            "transfers BETWEEN congruency and switchType, so both must be "
            "declared on the SAME condition (e.g. "
            "experiment_conditions.stimulus_experiment_conditions for the full "
            "2x2x2x2, or stimulus_main_effect_conditions for the 2x2 pooled over "
            "the proportions). Condition sets that carry only one of the two "
            "factors (stimulus_congruency_conditions, stimulus_switch_type_conditions) "
            "cannot be crossed: a transfer needs both labellings of the same trial.")
    return cells


def class_strings(cells, field, pos, neg):
    """Class groups (`[[pos names], [neg names]]`) for one factor of the cells.

    Emitted as full condition names rather than shorthand substrings, so the
    grouping is exactly the factor level the condition declared.
    """
    groups = []
    for level in (pos, neg):
        names = sorted(n for n, c in cells.items() if c[field] == level)
        if not names:
            raise ValueError(
                f"no condition has {field}={level!r}; "
                f"levels present: {sorted({c[field] for c in cells.values()})}")
        groups.append(names)
    return groups


def stability_flexibility_strings(cells):
    """The two contrasts A4 transfers between: (congruency i-vs-c, switch s-vs-r)."""
    return (class_strings(cells, 'congruency', 'i', 'c'),
            class_strings(cells, 'switchType', 's', 'r'))


def has_block_factor(cells, field):
    """True when `cells` can be split on `field` — every cell declares it and at
    least two levels are present.

    False for a condition set that POOLS over a proportion (e.g.
    `stimulus_main_effect_conditions`, which collapses both): the label-transfer
    and temporal-generalization designs still run there, the block-split designs
    cannot and are skipped rather than silently splitting on a constant.
    """
    levels = {c.get(field) for c in cells.values()}
    return None not in levels and len(levels) > 1


def block_condition_sets(cells, field):
    """`{level: [condition names]}` for a block factor, low level first.

    Feeds the within-block designs: "decode a contrast inside one block level" is
    `filter_conditions(..., block_condition_sets(cells, field)[level])`.
    """
    if not has_block_factor(cells, field):
        raise ValueError(
            f"the condition set cannot be split on {field!r} (levels present: "
            f"{sorted({str(c.get(field)) for c in cells.values()})}). It either "
            "pools over that factor or declares only one level, so there is no "
            "block contrast to make; check `has_block_factor` before calling.")
    levels = sorted({c[field] for c in cells.values()})
    return {level: sorted(n for n, c in cells.items() if c[field] == level)
            for level in levels}


def synthetic_condition_cells(**kwargs):
    """The `condition_cells` table for `synthetic_roi_labeled_arrays`' conditions.

    Kept next to the generator so the two can never drift: both build the same
    names from the same nested loop.
    """
    cells = {}
    for cong in ("c", "i"):
        for sw in ("r", "s"):
            for inc_prop in (25, 75):
                for sw_prop in (25, 75):
                    cells[f"Stimulus_{cong}_{sw}_{inc_prop}inc_{sw_prop}sw"] = dict(
                        congruency=cong, switchType=sw,
                        incongruent_proportion=inc_prop, switch_proportion=sw_prop)
    return cells


# ---------------------------------------------------------------------------
# Double-dipping guard for the electrode-definition <-> decode diagonal
# ---------------------------------------------------------------------------
# Electrodes are defined by one of the FOUR two-way interactions
# (`per_electrode_anova_labels`: CPC, SPS, CPS, SPC -- named {condition}P{modulator}).
# The within-block decoding 2x2 decodes {contrast} split by {block modulator}, and
# each of those four decode cells is the multivariate readout analog of exactly one
# interaction. Decoding a cell on the electrode set that same interaction *defined*
# is circular (double-dipping / selection bias, plan §0.1): the electrodes were
# chosen for having that very difference-of-differences, so its decodability is
# guaranteed to be inflated. The clean cells are the OFF-diagonal ones (define on
# one interaction, decode a different cell) -- the "define on LWPC, test LWPS"
# workhorse generalized to all four groups.
#
# Map: definition-group flag -> the (decode contrast, block modulator) it must
# NOT be scored on. Use `circular_decode_for_group` / `is_circular_decode`
# below to skip (or flag) the diagonal cell when a decode is restricted to a
# defined electrode group.
#
# The other way to keep the diagonal cell is `trial_splitting.py`: define the
# electrodes on one set of trials and decode on a disjoint set. Cross-validation
# alone does NOT fix this, because the selection happened before the CV split,
# using every trial.
DEFINITION_DECODE_DIAGONAL = {
    "CPC": ("congruency", "incongruent_proportion"),   # congruency x proportion-congruent (LWPC)
    "SPS": ("switchType", "switch_proportion"),         # switchType x switch-proportion (LWPS)
    "CPS": ("congruency", "switch_proportion"),         # congruency x switch-proportion (cross)
    "SPC": ("switchType", "incongruent_proportion"),    # switchType x proportion-congruent (cross)
}


def circular_decode_for_group(definition_group):
    """The (contrast, block_col) within-block decode cell that would double-dip on
    electrodes selected by `definition_group` ('CPC'|'SPS'|'CPS'|'SPC'), or None if
    the group name is unknown (e.g. 'both', 'all' -- not a single interaction)."""
    return DEFINITION_DECODE_DIAGONAL.get(definition_group)


def is_circular_decode(definition_group, contrast, block_col):
    """True when decoding `contrast` split by `block_col` on the `definition_group`
    electrode set is on the *defining* interaction (double-dipping) and its result
    must be ignored. Off-diagonal (cross) cells return False -- those are the
    non-circular tests to keep."""
    diag = circular_decode_for_group(definition_group)
    return diag is not None and diag == (contrast, block_col)


# ---------------------------------------------------------------------------
# LabeledArray -> the two label vectors the Decoder needs
# ---------------------------------------------------------------------------
def _class_of(condition_name, string_groups):
    """Index of the first string group that matches `condition_name`, else None.

    Same substring convention as `gather_class_data_by_stratum` /
    `concatenate_conditions_by_string`: `string_groups` is a list of classes, each
    a list of substrings, and a condition joins a class if ANY of that class's
    substrings occur in the condition name.
    """
    for class_idx, group in enumerate(string_groups):
        if any(s in condition_name for s in group):
            return class_idx
    return None


def _normalize_groups(strings_to_find):
    """Accept ['a', 'b'] or [['a1','a2'], ['b']] and return the nested form."""
    if all(isinstance(s, str) for s in strings_to_find):
        return [[s] for s in strings_to_find]
    return [list(g) for g in strings_to_find]


def build_cross_decoding_arrays(roi_labeled_arrays, roi, train_strings,
                                test_strings, obs_axs=0):
    """Turn one ROI's LabeledArray into the arrays `cv_cm_jim_window_shuffle` needs.

    A cross-decode needs TWO labellings of the SAME trials, so a condition only
    enters if both contrasts assign it a class. Conditions labelled by only one
    contrast are dropped (they could train but never be scored, or vice versa).

    Parameters
    ----------
    roi_labeled_arrays : {roi: LabeledArray} keyed [condition][trial][channel][time]
    roi : which ROI to pull.
    train_strings, test_strings : class definitions in the project's substring
        convention, e.g. ``[['_i_'], ['_c_']]`` for congruency and
        ``[['_s_'], ['_r_']]`` for switch type. A flat list of strings is accepted
        and treated as one substring per class.
    obs_axs : trials axis within a condition's array (0 in the standard layout).

    Returns
    -------
    dict with
        data          : (n_trials, n_channels, n_time) — conditions concatenated
        labels_train  : (n_trials,) class from `train_strings`
        labels_test   : (n_trials,) class from `test_strings`
        strata        : (n_trials,) the source condition index — pass this as
                        `stratify_labels` so every fold stays balanced on the
                        condition CELL, hence on both labellings at once
        cats_train, cats_test : {tuple(group): class_idx}, for the Decoder and plots
        conditions    : the condition names that survived, in concatenation order

    Notes
    -----
    Stratifying on `strata` rather than on `labels_train` is the point: a split
    balanced only on the training contrast can hand you a test fold that is
    lopsided on — or entirely missing — a class of the contrast you SCORE.
    """
    train_groups = _normalize_groups(train_strings)
    test_groups = _normalize_groups(test_strings)

    if roi not in roi_labeled_arrays:
        raise KeyError(f"ROI {roi!r} not in roi_labeled_arrays "
                       f"(have {list(roi_labeled_arrays)})")

    chunks, ltrain, ltest, strata, kept = [], [], [], [], []
    for condition_name in roi_labeled_arrays[roi].keys():
        c_train = _class_of(condition_name, train_groups)
        c_test = _class_of(condition_name, test_groups)
        if c_train is None or c_test is None:
            continue                      # not labelled by BOTH contrasts -> unusable
        arr = np.asarray(roi_labeled_arrays[roi][condition_name])
        n = arr.shape[obs_axs]
        chunks.append(arr)
        ltrain.append(np.full(n, c_train))
        ltest.append(np.full(n, c_test))
        strata.append(np.full(n, len(kept)))
        kept.append(condition_name)

    if not chunks:
        raise ValueError(
            f"no condition in ROI {roi!r} is labelled by BOTH contrasts "
            f"({train_strings} / {test_strings}); a cross-decode needs conditions "
            f"that cross the two factors, e.g. a block-balanced 2x2 condition set")

    labels_train = np.concatenate(ltrain)
    labels_test = np.concatenate(ltest)
    for name, lab in (("train", labels_train), ("test", labels_test)):
        if len(set(lab)) < 2:
            raise ValueError(f"the {name} contrast labelled every surviving "
                             f"condition the same class; nothing to decode")

    return dict(
        data=np.concatenate(chunks, axis=obs_axs),
        labels_train=labels_train,
        labels_test=labels_test,
        strata=np.concatenate(strata),
        cats_train={tuple(g): i for i, g in enumerate(train_groups)},
        cats_test={tuple(g): i for i, g in enumerate(test_groups)},
        conditions=kept,
    )


def filter_conditions(roi_labeled_arrays, roi, keep, new_roi=None):
    """Keep only the conditions matching `keep`.

    `keep` is one substring, or a collection of them — a condition is kept if ANY
    matches. The collection form is what `block_condition_sets` returns (full
    condition names), and it is what a real block level NEEDS: with the real
    naming, "the 25%-incongruent block" is the conditions matching `i25` OR `c25`,
    which no single substring picks out.

    How the within-block 2x2 (Design 0 / Fig 9) is expressed: decoding a contrast
    *within one block level* is just an ordinary decode over that block's
    conditions, so restrict here and run `run_cross_decoding` with the same
    contrast for train and test. Comparing the two block levels' accuracies is the
    decoding analogue of the univariate LWPC / LWPS effect.

    Returns a `{roi: {condition: array}}` dict ready to pass back in. Raises if
    nothing matches, since a silently empty ROI would look like a failed decode
    rather than a bad filter.
    """
    wanted = [keep] if isinstance(keep, str) else list(keep)
    kept = {name: arr for name, arr in roi_labeled_arrays[roi].items()
            if any(w in name for w in wanted)}
    if not kept:
        raise ValueError(
            f"no condition in ROI {roi!r} matches {wanted!r}; "
            f"have {sorted(roi_labeled_arrays[roi].keys())[:6]}...")
    return {new_roi or roi: kept}


def run_cross_decoding(roi_labeled_arrays, roi, train_strings, test_strings, *,
                       n_splits=5, n_repeats=10, n_shuffle_repeats=None,
                       explained_variance=0.8, obs_axs=0, time_axs=-1,
                       window=None, step_size=1, frac_train=None,
                       temporal_generalization=False, folds_as_samples=False,
                       oversample=True, random_state=42):
    """Train on `train_strings`, score against `test_strings`, plus a shuffle null.

    A thin wrapper over `Decoder.cv_cm_jim_window_shuffle` — it exists so the
    train/test label pair and the condition-cell stratification are constructed
    the same way everywhere, not to add any statistics of its own. The returned
    confusion matrices have exactly the shape the ordinary decoding path produces,
    so `compute_accuracies`, the bootstrap aggregation and
    `perform_time_perm_cluster_test_for_accuracies` all apply unchanged.

    Parameters
    ----------
    frac_train : proportion of trials to train on. None (default) keeps
        `StratifiedKFold`, i.e. (n_splits-1)/n_splits. Set it to sweep the
        train/test proportion directly (`StratifiedShuffleSplit`); `n_splits`
        then counts random resamples per repeat rather than folds.
    n_shuffle_repeats : repeats for the shuffle run. Defaults to `n_repeats`;
        raise it for a smoother null without paying for it on the true run.
    temporal_generalization : return a train-window × test-window matrix (Fig 10).
    others : passed through to the Decoder / `cv_cm_jim_window_shuffle`.

    Returns
    -------
    dict with `cm_true`, `cm_shuffle` (confusion matrices), plus `cats`,
    `conditions`, `labels_train`, `labels_test`, `strata` for bookkeeping.
    """
    arrays = build_cross_decoding_arrays(roi_labeled_arrays, roi, train_strings,
                                         test_strings, obs_axs=obs_axs)
    from .decoder import Decoder      # local import: keeps `ieeg` off the import path
                                      # for callers that only want the tables above

    common = dict(normalize='true', obs_axs=obs_axs, time_axs=time_axs,
                  window=window, step_size=step_size, oversample=oversample,
                  folds_as_samples=folds_as_samples,
                  labels_test=arrays['labels_test'],
                  stratify_labels=arrays['strata'],
                  frac_train=frac_train,
                  temporal_generalization=temporal_generalization)

    def _decoder(n_rep):
        return Decoder(arrays['cats_test'], explained_variance, oversample=oversample,
                       n_splits=n_splits, n_repeats=n_rep, random_state=random_state)

    cm_true = _decoder(n_repeats).cv_cm_jim_window_shuffle(
        arrays['data'], arrays['labels_train'], shuffle=False, **common)
    cm_shuffle = _decoder(n_shuffle_repeats or n_repeats).cv_cm_jim_window_shuffle(
        arrays['data'], arrays['labels_train'], shuffle=True, **common)

    return dict(cm_true=cm_true, cm_shuffle=cm_shuffle,
                cats=arrays['cats_test'], conditions=arrays['conditions'],
                labels_train=arrays['labels_train'],
                labels_test=arrays['labels_test'], strata=arrays['strata'])


# ---------------------------------------------------------------------------
# synthetic ground truth — a LabeledArray-shaped pseudopopulation with a KNOWN
# answer, so every launcher keeps its no-data dry run
# ---------------------------------------------------------------------------
def synthetic_roi_labeled_arrays(code="shared", n_channels=40, n_trials_per_cell=40,
                                 n_time=32, seed=0, amp=1.2, noise=1.0, roi="synthetic"):
    """`{roi: {condition_name: ndarray(trials, channels, time)}}` with a known truth.

    code='shared'      : congruency and switchType load on the SAME population axis
                         -> a decoder trained on one should transfer to the other.
    code='orthogonal'  : congruency on axis w1, switchType on w2 ⟂ w1 -> training on
                         one should NOT transfer, even though BOTH are individually
                         decodable.

    Condition names follow the project's substring convention
    (``Stimulus_<c|i>_<r|s>_<25|75>inc_<25|75>sw``) so the same
    `train_strings` / `test_strings` work on synthetic and real data. The effect
    occupies the middle half of the window, so temporal generalization shows an
    on-effect block rather than a full square.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n_channels); w1 /= np.linalg.norm(w1)
    w2 = rng.normal(size=n_channels); w2 -= (w2 @ w1) * w1; w2 /= np.linalg.norm(w2)
    cong_axis = w1
    switch_axis = w1 if code == "shared" else w2
    win = slice(n_time // 4, 3 * n_time // 4)

    conditions = {}
    for cong in ("c", "i"):
        for sw in ("r", "s"):
            for inc_prop in (25, 75):
                for sw_prop in (25, 75):
                    name = f"Stimulus_{cong}_{sw}_{inc_prop}inc_{sw_prop}sw"
                    x = rng.normal(0, noise, (n_trials_per_cell, n_channels, n_time))
                    pattern = (amp * (cong == "i") * cong_axis
                               + amp * (sw == "s") * switch_axis)
                    x[:, :, win] += pattern[None, :, None]
                    conditions[name] = x
    return {roi: conditions}


if __name__ == "__main__":
    # smoke test: a SHARED code cross-decodes; an ORTHOGONAL one does not, even
    # though each contrast is individually decodable in both worlds.
    from .accuracy_stats import compute_accuracies

    STAB = [["_i_"], ["_c_"]]        # congruency
    FLEX = [["_s_"], ["_r_"]]        # switch type
    for code in ("shared", "orthogonal"):
        arrs = synthetic_roi_labeled_arrays(code=code, seed=1)
        for label, (tr, te) in (("within congruency", (STAB, STAB)),
                                ("cross stab->flex", (STAB, FLEX))):
            out = run_cross_decoding(arrs, "synthetic", tr, te,
                                     n_splits=5, n_repeats=3, window=16, step_size=8)
            acc_t, acc_s = compute_accuracies(out['cm_true'], out['cm_shuffle'])
            print(f"[{code:>10}] {label:<18} true={acc_t.mean():.3f} "
                  f"shuffle={acc_s.mean():.3f}")
