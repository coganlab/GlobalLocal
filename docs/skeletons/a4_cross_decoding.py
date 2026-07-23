"""A4 skeleton — cross-decoding: pseudo-trials + label/set/temporal transfer
(plan §4).

DROP-IN TARGET
--------------
A new `src/analysis/decoding/cross_decoding.py` that BUILDS ON the existing
`Decoder` class and CV machinery in `src/analysis/decoding/decoding.py` — do not
reimplement the classifier. Reuse the balancing/downsampling helpers in
`src/analysis/utils/labeled_array_utils.py`.

    from src.analysis.decoding.decoding import Decoder
    from src.analysis.utils.labeled_array_utils import (
        make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel,
        put_data_in_labeled_array_per_roi_subject,
    )

WHY THIS EXISTS
---------------
Co-localization != shared CODE. This is the representation-level test that
disambiguates the "both" group (shared geometry vs. mixed selectivity with
orthogonal codes). It is the piece the counting analyses (§2) cannot do.

THE NON-NEGOTIABLE CONFOUND CONTROLS (plan §0.8) are baked into the signatures
below: trial-count matching, RT control, and per-condition mean removal. A
cross-decoding effect that doesn't survive `remove_condition_means` is a
univariate offset, not a code.
"""

from __future__ import annotations

import numpy as np


def build_pseudo_trials(labeled_array, cell_keys, n_per_cell, n_pseudo, rng):
    """Construct disjoint pseudo-trials aligned within condition cells.

    Subjects don't share trials, so we pool electrodes into a pseudopopulation
    and synthesize pseudo-trials by matching on the full condition cell
    (congruency × inc_prop × switchType × switch_prop).

    Parameters
    ----------
    labeled_array : per-electrode trial data (obs × channel × time), e.g. from
        put_data_in_labeled_array_per_roi_subject.
    cell_keys : the condition-cell coordinates to align on.
    n_per_cell : trials averaged/sampled per electrode to form one pseudo-trial.
    n_pseudo : number of pseudo-trials to build per cell.

    Returns
    -------
    (X, y, cell_id) where X is (n_pseudo_total × channel × time) and the pool is
    later split into DISJOINT train/test folds.

    Steps: group trials by cell; within a cell, for each pseudo-trial sample
    n_per_cell trials per electrode (without replacement within a fold) and
    average; stack across electrodes into the pseudopopulation feature vector.
    Keep a disjoint reservoir for the test side.
    """
    raise NotImplementedError("A4: pseudo-trial construction (disjoint train/test)")


def remove_condition_means(X, condition_labels):
    """Subtract each condition's per-feature mean (the univariate-offset control).

    Returns X with, for every feature (channel×time), the mean of each condition
    removed. If cross-decoding survives this, it's multivariate structure, not a
    mean shift. Run the analysis both with and without this and report both.
    """
    raise NotImplementedError("A4: per-condition mean removal")


def within_block_decoding_baseline(labeled_array, rt=None, **kw):
    """(Design 0, Fig 9) Decode a contrast within each block type and compare.

    Decode inc/con within mostly-congruent vs mostly-incongruent blocks (and
    switch/repeat within mostly-repeat vs mostly-switch blocks); compare
    accuracies. This is the decoding analog of the univariate LWPC/LWPS effects
    and where the NEURAL CROSS-EFFECTS surface (congruency decoding differing by
    switch-proportion block, and vice versa).

    Before comparing: match trial counts across blocks; regress out / match RT
    (`rt`). Interpret only after those controls.

    Returns per-block accuracy traces + the block difference with a null.
    """
    raise NotImplementedError("A4: within-block decoding baseline (Fig 9)")


def cross_decode(
    labeled_array,
    train_contrast: str,
    test_contrast: str,
    electrode_set=None,
    mode: str = "label_transfer",
    strip_condition_means: bool = False,
    rng=None,
    **decoder_kw,
):
    """Train a decoder on one contrast/set, test on another. Designs (a) and (b).

    mode='label_transfer'  (design a): same electrodes, train on `train_contrast`
        (e.g. stability), test on `test_contrast` (flexibility). Run separately on
        the LWPC-only, LWPS-only, and 'both' electrode groups. Prediction: only
        'both' cross-decodes.
    mode='set_transfer'    (design b): SAME label, train on one electrode set,
        test on the other (`electrode_set` selects train side).

    CIRCULARITY GUARD: the electrodes/trials used to DEFINE a group must not be
    the ones the transferred (test) accuracy is computed on. Enforce with a
    disjoint train/test split from `build_pseudo_trials`.

    Steps
    -----
    1. Build pseudo-trials; optionally `remove_condition_means`.
    2. Instantiate `Decoder(...)`; fit on train-contrast pseudo-trials.
    3. Predict test-contrast pseudo-trials; accumulate accuracy over CV folds.
    4. NULL: permute the transferred (test) labels; repeat for a chance
       distribution. Return observed accuracy, null, p.
    """
    raise NotImplementedError("A4: cross-condition / cross-set label transfer")


def temporal_generalization(
    labeled_array,
    train_contrast: str,
    test_contrast: str = None,
    rng=None,
    **decoder_kw,
):
    """(Design c, Fig 10) Train at time t, test at t' -> train×test accuracy matrix.

    Run WITHIN a contrast (test_contrast=None -> same as train: is the code
    stable or moving across the trial?) and ACROSS contrasts (cross-temporal
    label transfer). Off-diagonal generalization -> sustained/stable code; a
    narrow diagonal -> moving/phasic code.

    Reuse the disjoint pseudo-trial discipline from `build_pseudo_trials`.
    Returns a (n_train_times × n_test_times) accuracy matrix (+ optional null).
    """
    raise NotImplementedError("A4: temporal generalization matrix")
