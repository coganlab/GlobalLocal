"""A4 — cross-decoding: pseudo-trials + label / set / temporal transfer (plan §4).

Co-localization != shared CODE. The counting analyses (A1/A2) can show that the
*same electrodes* are selective for both stability (LWPC) and flexibility (LWPS),
but they cannot say whether those "both" electrodes carry ONE shared
representational geometry or a mix of two orthogonal codes. This module is the
representation-level test that disambiguates them.

Data model
----------
Everything runs on the SAME long-format single-trial table the A1-A3 jobs use,
built with ``effect_measure='cluster'`` so each row's ``hg`` is that trial's HG
**time course** over the window (a 1-D array), not the window mean:

    subject | electrode (= subject-channel) | congruency (c/i) | switchType (s/r)
            | incongruent_proportion | switch_proportion | hg = ndarray(n_time)

Subjects don't share trials, so we pool electrodes into a **pseudopopulation** and
synthesize **pseudo-trials** by matching on the full condition cell
(congruency × inc_prop × switchType × switch_prop). Train and test pseudo-trials
are drawn from DISJOINT reservoirs of the underlying single trials, so the
electrodes/trials used to define/train are never the ones a transferred accuracy
is computed on (the plan's circularity guard, §0.8).

The non-negotiable confound controls (plan §0.8) are baked in:
- ``remove_condition_means`` — subtract each condition's per-feature mean; a
  cross-decoding effect that doesn't survive this is a univariate offset, not a
  code. Every ``cross_decode`` call reports accuracy both raw and mean-removed.
- trial-count matching — pseudo-trials are built with a fixed ``n_per_cell`` per
  electrode per cell, and cells enter symmetrically, so class/block trial counts
  are equal by construction.
- label-permutation null — chance level is estimated by permuting the transferred
  (test) labels, not assumed to be 0.5.

Classifier
----------
The classifier is the same scaler -> PCA -> LDA pipeline the project ``Decoder``
wraps (``ieeg.decoding.models.PcaEstimateDecoder``). We reuse ``Decoder`` when it
imports (on the cluster); otherwise we fall back to an equivalent scikit-learn
``Pipeline`` so the pseudo-trial / transfer logic — the genuinely new code — runs
anywhere (see ``make_classifier``). Pass your own via ``classifier_factory``.

Designs implemented
-------------------
- ``within_block_decoding_baseline`` (Design 0, Fig 9): decode a contrast within
  each block type and compare — the decoding analog of the univariate LWPC/LWPS
  effects, and where neural cross-effects surface.
- ``cross_decode`` (Designs a/b): label transfer (train on one contrast, test on
  another, same electrodes) and set comparison (same label, each electrode set).
- ``temporal_generalization`` (Design c, Fig 10): train at time t, test at t' ->
  a train×test accuracy matrix, within and across contrasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
# (s vs r); the process aliases make cross_decode calls read like the plan.
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
# classifier
# ---------------------------------------------------------------------------
def _sklearn_pipeline(explained_variance=0.8, random_state=None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=explained_variance, random_state=random_state)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


# Resolve the classifier backend ONCE (importing ieeg's Decoder is cluster-only
# and, on failure, must not be retried on every fold). None => not yet resolved.
_DECODER_AVAILABLE = None


def make_classifier(explained_variance: float = 0.8, random_state=None):
    """Return a fresh scaler -> PCA -> LDA classifier.

    Uses the project ``Decoder``'s underlying sklearn pipeline when ``ieeg`` is
    importable (so cluster runs share the exact estimator the rest of the decoding
    pipeline uses); otherwise an equivalent scikit-learn ``Pipeline``. The choice
    is resolved once and cached. The returned object exposes ``.fit(X, y)`` /
    ``.predict(X)`` on 2-D feature matrices."""
    global _DECODER_AVAILABLE
    if _DECODER_AVAILABLE is None:
        try:                                # pragma: no cover - cluster-only path
            from src.analysis.decoding.decoder import Decoder  # leaf module, no plotting
            Decoder(categories={}, explained_variance=explained_variance,
                    n_splits=2, n_repeats=1, oversample=False)
            _DECODER_AVAILABLE = True
        except Exception:
            _DECODER_AVAILABLE = False
    if _DECODER_AVAILABLE:
        from src.analysis.decoding.decoder import Decoder
        return Decoder(categories={}, explained_variance=explained_variance,
                       n_splits=2, n_repeats=1, oversample=False,
                       random_state=random_state).model
    return _sklearn_pipeline(explained_variance, random_state)


# ---------------------------------------------------------------------------
# pseudopopulation assembly
# ---------------------------------------------------------------------------
def _cell_key(row, cell_cols):
    return tuple(row[c] for c in cell_cols)


def assemble_pseudopopulation(df, electrodes=None, cell_cols=DEFAULT_CELL_COLS):
    """Index the long df into ``{electrode: {cell_key: ndarray(n_trials, n_time)}}``.

    ``electrodes`` restricts the pseudopopulation (e.g. to the A1 'both' group);
    None uses every electrode in ``df``. Returns ``(pop, channels, n_time,
    cell_cols)`` where ``channels`` is the sorted electrode list = feature axis.
    """
    if electrodes is not None:
        electrodes = list(electrodes)
        df = df[df["electrode"].isin(electrodes)]
    channels = sorted(df["electrode"].unique())
    if not channels:
        raise ValueError("no electrodes in the (restricted) pseudopopulation")

    # infer n_time from the first hg cell
    first = df["hg"].iloc[0]
    n_time = int(np.asarray(first).shape[0]) if np.ndim(first) else 1

    pop = {}
    for elec, g in df.groupby("electrode"):
        cells = {}
        for key, gg in g.groupby(list(cell_cols)):
            key = key if isinstance(key, tuple) else (key,)
            arr = np.stack([np.asarray(a, dtype=float).reshape(-1)[:n_time]
                            for a in gg["hg"].to_numpy()])
            cells[key] = arr
        pop[elec] = cells
    return pop, channels, n_time, tuple(cell_cols)


def _split_reservoirs(pop, rng, frac_train=0.5):
    """Per (electrode, cell), split trial rows into disjoint train/test index sets.

    Splitting the underlying single trials BEFORE any pseudo-trial is drawn is the
    circularity guard: a trial that helps build a train pseudo-trial can never
    reappear in a test pseudo-trial."""
    reservoir = {}
    for elec, cells in pop.items():
        reservoir[elec] = {}
        for key, arr in cells.items():
            n = len(arr)
            idx = rng.permutation(n)
            cut = max(1, int(round(frac_train * n))) if n > 1 else 1
            reservoir[elec][key] = (idx[:cut], idx[cut:] if n > 1 else idx[:cut])
    return reservoir


def build_pseudo_trials(pop, channels, n_time, cell_cols, contrast,
                        reservoir, side, n_per_cell=5, n_pseudo=60, rng=None):
    """Construct pseudo-trials for one side (train/test) of a transfer.

    For each condition cell that the ``contrast`` labels {0,1}, and for each of
    ``n_pseudo`` draws, every channel contributes the average of ``n_per_cell``
    trials sampled from that channel's ``side`` reservoir for that cell; the
    channels are stacked into a ``(n_channel, n_time)`` pseudo-trial. A channel
    with no trials in a cell contributes zeros (mean-centred), so a missing cell
    neither helps nor hurts the decode.

    Parameters
    ----------
    pop, channels, n_time, cell_cols : from ``assemble_pseudopopulation``.
    contrast : cell->{0,1,None} callable (see ``resolve_contrast``).
    reservoir : from ``_split_reservoirs``.
    side : 0 for the train reservoir, 1 for the test reservoir.
    n_per_cell : trials averaged per channel to form one pseudo-trial.
    n_pseudo : pseudo-trials per class-labelled cell.

    Returns
    -------
    X : ndarray (n_pseudo_total, n_channel, n_time)
    y : ndarray (n_pseudo_total,)          — binary class from the contrast
    cell_id : ndarray (n_pseudo_total,)    — index of the source cell (for
              per-condition mean removal)
    """
    rng = rng if rng is not None else np.random.default_rng()
    contrast = resolve_contrast(contrast)

    # collect the labelled cells (any cell present on any channel)
    all_cells = {}
    for elec in channels:
        for key in pop.get(elec, {}):
            all_cells.setdefault(key, dict(zip(cell_cols, key)))
    labelled = [(key, contrast(cd)) for key, cd in all_cells.items()]
    labelled = [(key, lab) for key, lab in labelled if lab is not None]
    if not labelled:
        raise ValueError("contrast labelled no cells in this pseudopopulation")

    X, y, cid = [], [], []
    for ci, (key, lab) in enumerate(sorted(labelled, key=lambda t: str(t[0]))):
        for _ in range(n_pseudo):
            feat = np.zeros((len(channels), n_time))
            for k, elec in enumerate(channels):
                cell = pop.get(elec, {}).get(key)
                if cell is None or len(cell) == 0:
                    continue
                pool = reservoir[elec][key][side]
                if len(pool) == 0:
                    pool = reservoir[elec][key][1 - side]   # degenerate: reuse
                take = rng.choice(pool, size=n_per_cell,
                                  replace=len(pool) < n_per_cell)
                feat[k] = cell[take].mean(axis=0)
            X.append(feat)
            y.append(lab)
            cid.append(ci)
    return np.asarray(X), np.asarray(y), np.asarray(cid)


# ---------------------------------------------------------------------------
# confound control: per-condition mean removal
# ---------------------------------------------------------------------------
def remove_condition_means(X, condition_labels):
    """Subtract each condition's per-feature mean (the univariate-offset control).

    ``X`` is (n_trials, ...); ``condition_labels`` is length n_trials. For every
    feature and every condition, the condition's mean over its trials is removed.
    If cross-decoding survives this, it is multivariate structure, not a mean
    shift. Returns a copy of ``X``."""
    X = np.asarray(X, dtype=float).copy()
    labels = np.asarray(condition_labels)
    for c in np.unique(labels):
        m = labels == c
        X[m] -= X[m].mean(axis=0, keepdims=True)
    return X


# ---------------------------------------------------------------------------
# small scoring helpers
# ---------------------------------------------------------------------------
def _flatten(X):
    return X.reshape(X.shape[0], -1)


def _fit_transfer_accuracy(Xtr, ytr, Xte, yte, classifier_factory):
    clf = classifier_factory()
    clf.fit(_flatten(Xtr), ytr)
    pred = clf.predict(_flatten(Xte))
    return float(np.mean(pred == yte))


def _perm_null(yte, observed_acc, n_perm, rng, Xtr, ytr, Xte, classifier_factory):
    """Chance distribution by permuting the (test) labels. Statistic = accuracy;
    we report the two-sided deviation from 0.5 so it is robust to which class the
    transferred decoder happens to align with."""
    clf = classifier_factory()
    clf.fit(_flatten(Xtr), ytr)
    pred = clf.predict(_flatten(Xte))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean(pred == rng.permutation(yte))
    p = (np.sum(np.abs(null - 0.5) >= abs(observed_acc - 0.5)) + 1) / (n_perm + 1)
    return null, float(p)


# ---------------------------------------------------------------------------
# Design 0 / Fig 9 — within-block decoding baseline
# ---------------------------------------------------------------------------
def within_block_decoding_baseline(df, contrast="congruency", block_col=None,
                                   electrodes=None, n_per_cell=5, n_pseudo=80,
                                   n_folds=5, n_perm=200, rng=None,
                                   classifier_factory=None,
                                   cell_cols=DEFAULT_CELL_COLS):
    """Decode a contrast within each block type and compare accuracies (Fig 9).

    E.g. decode congruency (inc/con) within the mostly-congruent vs
    mostly-incongruent blocks (``contrast='congruency'``, block =
    ``incongruent_proportion``), or switch/repeat within mostly-repeat vs
    mostly-switch blocks (``contrast='switchType'``, block =
    ``switch_proportion``). This is the decoding analog of the univariate
    LWPC/LWPS effects; a block difference in decodability is a neural cross-effect.

    Trial counts are matched across blocks by construction (fixed ``n_per_cell`` /
    ``n_pseudo`` per cell). Returns per-block accuracy (mean over CV folds) with a
    label-permutation null, plus the block difference and its null.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    classifier_factory = classifier_factory or make_classifier
    if block_col is None:
        block_col = ("incongruent_proportion" if resolve_contrast(contrast) is _congruency_label
                     else "switch_proportion")

    blocks = sorted(pd.unique(df[block_col].dropna()))
    per_block = {}
    for b in blocks:
        sub = df[df[block_col] == b]
        pop, channels, n_time, cc = assemble_pseudopopulation(sub, electrodes, cell_cols)
        accs = []
        for _ in range(n_folds):
            res = _split_reservoirs(pop, rng)
            Xtr, ytr, ctr = build_pseudo_trials(pop, channels, n_time, cc, contrast,
                                                res, side=0, n_per_cell=n_per_cell,
                                                n_pseudo=n_pseudo, rng=rng)
            Xte, yte, cte = build_pseudo_trials(pop, channels, n_time, cc, contrast,
                                                res, side=1, n_per_cell=n_per_cell,
                                                n_pseudo=n_pseudo, rng=rng)
            accs.append(_fit_transfer_accuracy(Xtr, ytr, Xte, yte, classifier_factory))
        acc = float(np.mean(accs))
        null, p = _perm_null(yte, acc, n_perm, rng, Xtr, ytr, Xte, classifier_factory)
        per_block[b] = dict(accuracy=acc, acc_folds=np.array(accs),
                            null_mean=float(null.mean()), p=p, n_channels=len(channels))

    result = dict(contrast=contrast if isinstance(contrast, str) else "custom",
                  block_col=block_col, blocks=list(blocks), per_block=per_block)
    if len(blocks) == 2:
        b0, b1 = blocks
        result["block_difference"] = (per_block[b1]["accuracy"]
                                      - per_block[b0]["accuracy"])
    return result


# ---------------------------------------------------------------------------
# Designs a/b — cross-decode
# ---------------------------------------------------------------------------
def cross_decode(df, train_contrast, test_contrast=None, electrodes=None,
                 mode="label_transfer", strip_condition_means=False,
                 electrode_sets=None, n_per_cell=5, n_pseudo=80, n_folds=5,
                 n_perm=200, rng=None, classifier_factory=None,
                 cell_cols=DEFAULT_CELL_COLS):
    """Train a decoder on one contrast/set and test it on another.

    mode='label_transfer' (Design a)
        Same electrodes; train on ``train_contrast`` (e.g. stability), test on
        ``test_contrast`` (e.g. flexibility). Run separately on the LWPC-only,
        LWPS-only, and 'both' electrode groups via ``electrodes``. Prediction:
        only the 'both' group cross-decodes above chance. Train and test
        pseudo-trials come from DISJOINT reservoirs (circularity guard).

    mode='set_transfer' (Design b)
        Same label (``train_contrast``), decoded WITHIN each electrode set in
        ``electrode_sets`` ({name: [electrodes]}), so you can compare where the
        code lives. (True cross-feature transfer across *non-corresponding*
        electrodes is not identifiable — different channels are different feature
        axes — so this reports each set's same-label decodability with its own
        null rather than pretending to port an LDA axis across disjoint features.)

    Confound controls: with ``strip_condition_means=True`` each condition's
    per-feature mean is removed before fitting (report both, per §0.8). Chance is
    estimated by permuting the test labels (``n_perm``).

    Returns a dict with observed accuracy, CV-fold accuracies, null mean and p
    (per group/set as applicable).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    classifier_factory = classifier_factory or make_classifier
    test_contrast = test_contrast if test_contrast is not None else train_contrast

    if mode == "set_transfer":
        if not electrode_sets:
            raise ValueError("mode='set_transfer' needs electrode_sets={name:[...]}")
        out = {}
        for name, elset in electrode_sets.items():
            out[name] = cross_decode(
                df, train_contrast, train_contrast, electrodes=elset,
                mode="label_transfer", strip_condition_means=strip_condition_means,
                n_per_cell=n_per_cell, n_pseudo=n_pseudo, n_folds=n_folds,
                n_perm=n_perm, rng=rng, classifier_factory=classifier_factory,
                cell_cols=cell_cols)
        return dict(mode="set_transfer", contrast=train_contrast, per_set=out)

    # ---- label_transfer ----
    pop, channels, n_time, cc = assemble_pseudopopulation(df, electrodes, cell_cols)
    accs = []
    Xtr = ytr = Xte = yte = None
    for _ in range(n_folds):
        res = _split_reservoirs(pop, rng)
        Xtr, ytr, ctr = build_pseudo_trials(pop, channels, n_time, cc, train_contrast,
                                            res, side=0, n_per_cell=n_per_cell,
                                            n_pseudo=n_pseudo, rng=rng)
        Xte, yte, cte = build_pseudo_trials(pop, channels, n_time, cc, test_contrast,
                                            res, side=1, n_per_cell=n_per_cell,
                                            n_pseudo=n_pseudo, rng=rng)
        if strip_condition_means:
            Xtr = remove_condition_means(Xtr, ctr)
            Xte = remove_condition_means(Xte, cte)
        accs.append(_fit_transfer_accuracy(Xtr, ytr, Xte, yte, classifier_factory))
    acc = float(np.mean(accs))
    null, p = _perm_null(yte, acc, n_perm, rng, Xtr, ytr, Xte, classifier_factory)
    return dict(mode="label_transfer",
                train_contrast=train_contrast if isinstance(train_contrast, str) else "custom",
                test_contrast=test_contrast if isinstance(test_contrast, str) else "custom",
                strip_condition_means=strip_condition_means,
                accuracy=acc, acc_folds=np.array(accs),
                null_mean=float(null.mean()), null=null, p=p,
                n_channels=len(channels), n_pseudo_per_class=n_pseudo)


# ---------------------------------------------------------------------------
# Design c / Fig 10 — temporal generalization
# ---------------------------------------------------------------------------
def temporal_generalization(df, train_contrast, test_contrast=None, electrodes=None,
                            n_per_cell=5, n_pseudo=60, n_folds=3, rng=None,
                            classifier_factory=None, strip_condition_means=False,
                            cell_cols=DEFAULT_CELL_COLS):
    """Train at time t, test at t' -> a (n_train_time × n_test_time) accuracy matrix.

    ``test_contrast=None`` runs WITHIN a contrast (is the code stable or moving
    across the trial?); passing a different ``test_contrast`` gives a
    cross-temporal LABEL transfer. Off-diagonal generalization -> a
    sustained/stable code; a narrow diagonal -> a moving/phasic code (Fig 10).
    Reuses the disjoint pseudo-trial discipline. Features at each time bin are the
    channel vector (no time flattening), so the classifier is time-bin-local.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    classifier_factory = classifier_factory or make_classifier
    test_contrast = test_contrast if test_contrast is not None else train_contrast

    pop, channels, n_time, cc = assemble_pseudopopulation(df, electrodes, cell_cols)
    acc_sum = np.zeros((n_time, n_time))
    for _ in range(n_folds):
        res = _split_reservoirs(pop, rng)
        Xtr, ytr, ctr = build_pseudo_trials(pop, channels, n_time, cc, train_contrast,
                                            res, side=0, n_per_cell=n_per_cell,
                                            n_pseudo=n_pseudo, rng=rng)
        Xte, yte, cte = build_pseudo_trials(pop, channels, n_time, cc, test_contrast,
                                            res, side=1, n_per_cell=n_per_cell,
                                            n_pseudo=n_pseudo, rng=rng)
        if strip_condition_means:
            Xtr = remove_condition_means(Xtr, ctr)
            Xte = remove_condition_means(Xte, cte)
        # fit one classifier per train time bin, predict at every test time bin
        clfs = []
        for tt in range(n_time):
            clf = classifier_factory()
            clf.fit(Xtr[:, :, tt], ytr)
            clfs.append(clf)
        for tt in range(n_time):
            clf = clfs[tt]
            for te in range(n_time):
                acc_sum[tt, te] += np.mean(clf.predict(Xte[:, :, te]) == yte)
    matrix = acc_sum / n_folds
    return dict(matrix=matrix, n_time=n_time, n_channels=len(channels),
                train_contrast=train_contrast if isinstance(train_contrast, str) else "custom",
                test_contrast=test_contrast if isinstance(test_contrast, str) else "custom")


# ---------------------------------------------------------------------------
# synthetic ground truth — SHARED vs ORTHOGONAL code, no data on disk
# ---------------------------------------------------------------------------
def _synthetic_cross_df(code="shared", n_subj=4, n_elec_per_subj=10, n_time=16,
                        seed=0, amp=1.2, noise=1.0):
    """Pseudopopulation with a KNOWN cross-decoding truth, for validation/tutorial.

    code='shared'      : congruency and switchType load on the SAME population
                         axis -> a decoder trained on one contrast should transfer
                         to the other (cross-decoding above chance).
    code='orthogonal'  : congruency loads on axis w1, switchType on w2 ⟂ w1 ->
                         training on one contrast should NOT transfer to the other
                         (cross-decoding ≈ chance) even though BOTH are individually
                         decodable.

    The effect is injected into the middle of the window (bins n_time/4..3n_time/4),
    so ``temporal_generalization`` shows an on-effect block. Returns a cluster-mode
    long df (hg column = per-trial time course).
    """
    rng = np.random.default_rng(seed)
    n_elec = n_subj * n_elec_per_subj
    # two population axes over electrodes
    w1 = rng.normal(size=n_elec); w1 /= np.linalg.norm(w1)
    w2 = rng.normal(size=n_elec)
    w2 -= (w2 @ w1) * w1                     # orthogonalize w2 ⟂ w1
    w2 /= np.linalg.norm(w2)
    cong_axis = w1
    switch_axis = w1 if code == "shared" else w2

    win = slice(n_time // 4, 3 * n_time // 4)
    rows = []
    e_global = 0
    for s in range(n_subj):
        subject = f"S{s:02d}"
        n_tr = int(rng.integers(160, 240))
        cong = rng.choice(["c", "i"], n_tr)
        sw = rng.choice(["s", "r"], n_tr)
        inc_prop = rng.choice([25.0, 75.0], n_tr)
        sw_prop = rng.choice([25.0, 75.0], n_tr)
        # latent per-trial coordinate along each axis
        cscore = amp * (cong == "i").astype(float)      # congruency signal
        sscore = amp * (sw == "s").astype(float)        # switch signal
        for _ in range(n_elec_per_subj):
            electrode = f"{subject}-e{e_global}"
            a1 = cong_axis[e_global]
            a2 = switch_axis[e_global]
            base = np.outer(cscore, [a1]).ravel() + np.outer(sscore, [a2]).ravel()
            tc = rng.normal(0, noise, (n_tr, n_time))
            tc[:, win] += base[:, None]
            col = np.empty(n_tr, dtype=object)
            for i in range(n_tr):
                col[i] = tc[i]
            rows.append(pd.DataFrame(dict(
                subject=subject, electrode=electrode,
                congruency=cong, switchType=sw,
                incongruent_proportion=inc_prop, switch_proportion=sw_prop,
                hg=col)))
            e_global += 1
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    # smoke test: SHARED code cross-decodes; ORTHOGONAL code does not — even though
    # each contrast is individually decodable in both.
    for code in ("shared", "orthogonal"):
        df = _synthetic_cross_df(code=code, seed=1)
        rng = np.random.default_rng(0)
        within = cross_decode(df, "congruency", "congruency", n_folds=3, n_perm=200, rng=rng)
        cross = cross_decode(df, "stability", "flexibility", n_folds=3, n_perm=200, rng=rng)
        cross_mr = cross_decode(df, "stability", "flexibility", strip_condition_means=True,
                                n_folds=3, n_perm=200, rng=rng)
        print(f"[{code:>10}] within congruency acc={within['accuracy']:.3f} "
              f"(p={within['p']:.3f}) | cross stab->flex acc={cross['accuracy']:.3f} "
              f"(p={cross['p']:.3f}) | mean-removed acc={cross_mr['accuracy']:.3f} "
              f"(p={cross_mr['p']:.3f})")
