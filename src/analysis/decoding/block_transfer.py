"""N3b: block-transfer cross-decoding (docs/n3b_block_transfer.md).

Train a contrast in one level of a block factor and score it in the other. For
example, X1 learns congruency in 25%-incongruent blocks and tests it in
75%-incongruent blocks. The only new idea is WHICH trials train and which
test; everything else is the ordinary cross-decoding pipeline:

- `build_cross_decoding_arrays` for the trials and labels;
- `make_decoder` for the classifier (PCA -> LDA, equal priors);
- `Decoder.cv_cm_jim_window_shuffle(test_only=...)` for the folds;
- `compute_accuracies` and the usual cluster test downstream, which take the
  confusion matrices returned here unchanged.

Each resample does three things, in this order:

1. **Balance.** Keep the same number of trials of each class from each physical
   block (class x incongruent proportion x switch proportion = 8 groups). Classes
   run about 3:1 inside a block and the majority flips between levels; this
   evens them out, and keeps the classes from differing in which physical
   blocks they come from.
2. **Center** (optional). Subtract each level's mean trial. The trials are
   already balanced, so that mean is the midpoint between the classes: the
   level moves as a whole. Centering on the raw 3:1 mean would instead shift
   the two levels apart along the very axis being decoded.
3. **Decode the 2x2.** For each training level, cut folds inside it and score
   every fold's classifier on its own held-out trials (within: the ceiling) and
   on every trial of the other level (transfer). The same seed gives both the
   same folds, so both come from the same classifiers.
"""

from __future__ import annotations

import numpy as np

from .cross_decoding import build_cross_decoding_arrays, class_strings, make_decoder

BLOCK_FACTORS = ("incongruent_proportion", "switch_proportion")
# decoded contrast -> (class 0, class 1) levels, the order `class_strings` uses
CONTRAST_LEVELS = {"congruency": ("i", "c"), "switchType": ("s", "r")}


def prepare(roi_labeled_arrays, roi, cells, contrast, block_col):
    """One ROI's trials with the labels block transfer needs.

    `cells` is the condition -> factor-level table (`cd.condition_cells` or
    `cd.synthetic_condition_cells`), `contrast` is 'congruency' or 'switchType',
    `block_col` is the factor transferred across.

    Returns dict with
        X      : (n_trials, n_channels, n_time); padding rows already dropped
        y      : class of each trial (0 = incongruent / switch)
        strata : source condition of each trial; the folds are stratified on it
        block  : the trial's level of `block_col` (25 or 75)
        groups : the balance group, e.g. 'i|inc25|sw75' = one class in one
                 physical block
        cats   : class definitions, for the Decoder
    """
    if contrast not in CONTRAST_LEVELS:
        raise ValueError(f"contrast must be one of {list(CONTRAST_LEVELS)}; got {contrast!r}")
    if block_col not in BLOCK_FACTORS:
        raise ValueError(f"block_col must be one of {BLOCK_FACTORS}; got {block_col!r}")
    missing = [f for f in BLOCK_FACTORS if any(c.get(f) is None for c in cells.values())]
    if missing:
        raise ValueError(f"every condition must declare {missing}; block transfer needs "
                         "the full 2x2x2x2 set (stimulus_experiment_conditions)")

    strings = class_strings(cells, contrast, *CONTRAST_LEVELS[contrast])
    a = build_cross_decoding_arrays(roi_labeled_arrays, roi, strings, strings)
    source = [cells[c] for c in np.asarray(a['conditions'])[a['strata']]]
    groups = np.array([f"{c[contrast]}|inc{c['incongruent_proportion']}"
                       f"|sw{c['switch_proportion']}" for c in source])
    block = np.array([c[block_col] for c in source])
    return dict(X=a['data'], y=a['labels_train'], strata=a['strata'], block=block,
                groups=groups, cats=a['cats_train'])


def balanced_subsample(groups, rng):
    """Indices keeping the same number of trials from every group (the smallest
    group's size), drawn without replacement and returned in their original order."""
    levels, counts = np.unique(groups, return_counts=True)
    keep = [rng.choice(np.flatnonzero(groups == g), counts.min(), replace=False)
            for g in levels]
    return np.sort(np.concatenate(keep))


def center_levels(X, block):
    """Subtract each block level's mean trial, per channel and time point.

    Use on BALANCED trials: each level's mean is then the midpoint between its
    classes, so the whole level moves and neither class moves relative to the
    other. NaN gaps are ignored in the mean and stay NaN.
    """
    X = X.copy()
    for level in np.unique(block):
        m = block == level
        X[m] -= np.nanmean(X[m], axis=0)
    return X


def run_block_transfer(roi_labeled_arrays, roi, cells, contrast, block_col, *,
                       center=False, n_resamples=10, n_splits=5, explained_variance=0.8,
                       window=None, step_size=1, frac_train=None, seed=0):
    """Decode the train-level x test-level 2x2 of `contrast` across `block_col`.

    Each resample is a new balanced draw (so every trial is used across
    resamples) with its own folds, and supplies one sample to every cell.

    Returns dict with
        cells       : {(train_level, test_level): {'cm_true', 'cm_shuffle'}}, each
                      (n_windows, n_resamples, 2, 2) like the ordinary decoder's
                      output. Diagonal = within-level CV (the ceiling);
                      off-diagonal = transfer.
        levels      : (low, high) levels of `block_col`
        group_sizes : {group: trials available before balancing}
        n_per_group : trials kept from every group in each resample
    """
    p = prepare(roi_labeled_arrays, roi, cells, contrast, block_col)
    X, y, strata, block, groups = p['X'], p['y'], p['strata'], p['block'], p['groups']
    lo, hi = sorted(set(block))
    pairs = ((lo, lo), (lo, hi), (hi, hi), (hi, lo))
    out = {pair: {'cm_true': [], 'cm_shuffle': []} for pair in pairs}

    rng = np.random.default_rng(seed)
    for r in range(n_resamples):
        k = balanced_subsample(groups, rng)
        Xk, yk, sk, bk = X[k], y[k], strata[k], block[k]
        if center:
            Xk = center_levels(Xk, bk)
        dec = make_decoder(p['cats'], explained_variance=explained_variance,
                           n_splits=n_splits, n_repeats=1, random_state=seed + r)
        for train, test in pairs:
            use = np.isin(bk, (train, test))
            test_only = None if train == test else bk[use] == test
            for shuffle in (False, True):
                cm = dec.cv_cm_jim_window_shuffle(
                    Xk[use], yk[use], normalize='true', obs_axs=0, window=window,
                    step_size=step_size, frac_train=frac_train,
                    stratify_labels=sk[use], test_only=test_only, shuffle=shuffle)
                out[(train, test)]['cm_shuffle' if shuffle else 'cm_true'].append(cm)

    names, counts = np.unique(groups, return_counts=True)
    return dict(
        cells={pair: {key: np.concatenate(cms, axis=-3) for key, cms in d.items()}
               for pair, d in out.items()},
        levels=(lo, hi),
        group_sizes=dict(zip(names.tolist(), counts.tolist())),
        n_per_group=int(counts.min()))
