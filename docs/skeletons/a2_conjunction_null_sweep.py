"""A2 skeleton — conjunction permutation null + threshold sweep (plan §2).

DROP-IN TARGET
--------------
Add to `src/analysis/stats/stability_flexibility_segregation.py`, next to
`cmh_conjunction`. Both functions are thin wrappers that REUSE existing
machinery — do not reimplement the CMH.

    from src.analysis.stats.stability_flexibility_segregation import (
        cmh_conjunction, per_electrode_labels,   # or per_electrode_anova_labels from A1
    )

WHY THIS EXISTS
---------------
`cmh_conjunction` gives the MH odds ratio and its parametric tests. The plan
(§2) wants two robustness pieces on top:
  * a within-subject PERMUTATION null on the overlap count (respects nesting
    better than the analytic hypergeometric), and
  * a THRESHOLD SWEEP so a segregation claim is shown stable across alpha, not an
    artifact of one cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conjunction_permutation_null(labels, n_perm: int = 10000, seed: int = 0):
    """Empirical null for the number of double-selective ('both') electrodes.

    Parameters
    ----------
    labels : DataFrame
        As returned by `per_electrode_labels` / `per_electrode_anova_labels`:
        columns subject, S (0/1), F (0/1).

    Returns
    -------
    dict:
        observed  : int    # observed count of S==1 & F==1
        null      : ndarray(n_perm)
        p_two_sided : float
        z         : float  # (observed - null.mean()) / null.std(), descriptive

    Implementation steps
    ---------------------
    1. observed = int(((labels.S == 1) & (labels.F == 1)).sum()).
    2. For each permutation, shuffle F *within each subject* (groupby('subject')),
       leaving S untouched, and recount the overlap. Shuffling within subject
       holds each subject's S-count and F-count fixed and only randomizes the
       PAIRING — exactly the CMH null.
       Vectorize with a per-subject index permutation (rng.permutation on each
       subject's row indices), like `subject_clustered_corr` does for the
       continuous null.
    3. p_two_sided = (sum(|null - mean| >= |observed - mean|) + 1) / (n_perm + 1).

    INVARIANT TO ASSERT (acceptance criterion): for every permutation, each
    subject's S.sum() and F.sum() are unchanged. Add an assert in a test.
    """
    raise NotImplementedError("A2: within-subject permutation null on overlap count")


def conjunction_threshold_sweep(
    labels_by_threshold,
    thresholds,
):
    """Recompute overlap OR + counts across a range of selection thresholds.

    Parameters
    ----------
    labels_by_threshold : callable
        threshold -> labels DataFrame (S/F recomputed at that q-cutoff or
        effect-size percentile). Passing a callable keeps this function agnostic
        to whether you threshold on q-values (parametric ANOVA / permutation) or
        on effect-size percentiles.
    thresholds : iterable of float

    Returns
    -------
    DataFrame, one row per threshold:
        threshold, n_S, n_F, n_both, mh_odds_ratio, cmh_p

    Implementation steps
    ---------------------
    1. For each t in thresholds: labels = labels_by_threshold(t);
       res = cmh_conjunction(labels);
       record n_S=labels.S.sum(), n_F=labels.F.sum(),
              n_both=((labels.S==1)&(labels.F==1)).sum(),
              mh_odds_ratio=res['mh_odds_ratio'],
              cmh_p=res['cmh'].pvalue.
    2. Return as a tidy DataFrame ready to plot OR-vs-threshold.

    Acceptance criterion: on synthetic data with INDEPENDENT bx/by, the OR should
    hover near 1 across the whole sweep and the conclusion should not flip sign
    at any reasonable threshold.
    """
    raise NotImplementedError("A2: threshold sweep over the conjunction OR")
