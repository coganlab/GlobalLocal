"""A7 (self-check) skeleton — bootstrap OR CI + reconciled segregation verdict.

This is a LEARNING assignment, not part of the official A1–A6 plan. It exists to
check that you understand how the segregation battery's two analysis layers
(continuous correlation + categorical conjunction) fit together, and how the
subject-nesting principle that shows up everywhere in this module (within-subject
permutation, CMH stratification) also dictates how you resample.

DROP-IN TARGET
--------------
Everything you write here composes the PUBLIC functions already in
`src/analysis/stats/stability_flexibility_segregation.py`. You reimplement none
of them — you wire them together and add one new resampling estimator.

    from src.analysis.stats.stability_flexibility_segregation import (
        compute_sensitivities, add_responsiveness, prepare_continuous,
        subject_clustered_corr,                      # continuous layer
        per_electrode_labels, cmh_conjunction,       # categorical layer
        conjunction_permutation_null,                # A2 (already implemented)
    )

WHY THIS EXISTS
---------------
`cmh_conjunction` returns a *parametric* odds-ratio CI. The plan's cross-cutting
checklist keeps insisting that inference respect SUBJECT NESTING — that's why the
continuous test permutes within subject and the categorical null shuffles F
within subject. A parametric CI doesn't honor that structure. A subject-CLUSTER
bootstrap does: resample whole subjects (not electrodes) with replacement and
recompute the pooled OR. Then a single verdict function has to reconcile what the
continuous and categorical layers say — including when they disagree.

There is no solution file. The reference implementations you're leaning on live
in the module above; the NEW logic is yours. `test_a7_segregation_verdict.py`
next to this file is the objective check — it is RED now (these raise
NotImplementedError) and turns GREEN when you understand the pieces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.stats.stability_flexibility_segregation import (
    compute_sensitivities,
    add_responsiveness,
    prepare_continuous,
    subject_clustered_corr,
    per_electrode_labels,
    cmh_conjunction,
    conjunction_permutation_null,
)


# ----------------------------------------------------------------------------
# Task 1 — subject-clustered bootstrap CI for the conjunction odds ratio
# ----------------------------------------------------------------------------
def bootstrap_conjunction_or(labels, n_boot: int = 2000, seed: int = 0,
                             ci=(2.5, 97.5)):
    """Percentile CI for the MH pooled odds ratio, resampling SUBJECTS.

    Parameters
    ----------
    labels : DataFrame
        As returned by `per_electrode_labels` / `per_electrode_anova_labels`:
        columns include subject, S (0/1), F (0/1).
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        Seed for a `np.random.default_rng`.
    ci : (float, float)
        Lower/upper percentiles, default the central 95%.

    Returns
    -------
    dict:
        point : float          # OR on the observed labels (no resampling)
        lo, hi : float         # ci percentiles of the bootstrap OR distribution
        boots : ndarray(n_boot) # the bootstrap ORs (drop any that came back NaN)

    Implementation steps
    ---------------------
    1. Drop rows with NaN S/F. Get the array of unique subject ids.
    2. `point` = cmh_conjunction(labels)['mh_odds_ratio'] on the observed labels.
    3. For each of n_boot replicates:
       a. Draw len(subjects) subjects WITH REPLACEMENT (rng.choice).
       b. Build a labels frame from the drawn subjects. CRUCIAL: a subject drawn
          twice must become TWO separate CMH strata, not one doubled stratum —
          otherwise you've merged their 2x2 tables and destroyed the nesting the
          whole test is built on. Give each draw a fresh unique `subject` id
          (e.g. f"{orig}__{j}") before concatenating.
       c. OR_b = cmh_conjunction(that frame)['mh_odds_ratio']; collect it.
    4. lo, hi = np.percentile(boots, ci). Return the dict above.

    WHY SUBJECTS, NOT ELECTRODES: electrodes within a subject are not independent
    (shared coverage, shared state); resampling electrodes would understate the
    CI. Resampling subjects is the categorical analogue of the within-subject
    permutation used in `subject_clustered_corr` / `conjunction_permutation_null`.

    Acceptance criterion: `point` must equal cmh_conjunction(labels)['mh_odds_ratio']
    and satisfy lo <= point <= hi. On synthetic data (independent bx/by) the CI
    must COVER 1.
    """
    raise NotImplementedError("A7 Task 1: subject-clustered bootstrap OR CI")


# ----------------------------------------------------------------------------
# Task 2 — the decision rule (pure function, no data)
# ----------------------------------------------------------------------------
def classify_segregation(continuous: dict, categorical: dict,
                         alpha: float = 0.05) -> str:
    """Reconcile the continuous and categorical evidence into one verdict.

    Parameters
    ----------
    continuous : dict
        Must have 'corr' (residualised within-subject correlation) and 'p'
        (its within-subject permutation p), as returned by
        `subject_clustered_corr`.
    categorical : dict
        Must have 'mh_odds_ratio' and 'p' (a two-sided p for the overlap, e.g.
        the permutation-null p_two_sided from `conjunction_permutation_null`).
    alpha : float

    Returns
    -------
    One of: 'segregated', 'shared', 'conflicting', 'inconclusive'.

    Decision rule (implement EXACTLY this — the point is the reasoning, not a
    clever shortcut):
      * A layer "votes segregated" if it is significant (p < alpha) AND points the
        segregation way: continuous corr < 0, or categorical OR < 1.
      * A layer "votes shared"     if it is significant (p < alpha) AND points the
        shared way:      continuous corr > 0, or categorical OR > 1.
      * Combine the two layers' votes:
          - any 'segregated' vote and no 'shared' vote  -> 'segregated'
          - any 'shared' vote and no 'segregated' vote  -> 'shared'
          - at least one of each                        -> 'conflicting'
          - neither layer significant                   -> 'inconclusive'

    Note WHY 'conflicting' is its own outcome and not silently resolved: the two
    layers ask genuinely different questions (continuous co-selectivity gradient
    vs. categorical double-selective overlap). A real disagreement is a result to
    surface, not average away — same spirit as the plan's "if the sweep flips,
    that's a finding to report, not to hide."
    """
    raise NotImplementedError("A7 Task 2: the reconciled decision rule")


# ----------------------------------------------------------------------------
# Task 3 — end-to-end orchestrator that runs both layers and calls the rule
# ----------------------------------------------------------------------------
def segregation_verdict(df, n_splits: int = 100, n_perm: int = 5000,
                        n_boot: int = 2000, alpha: float = 0.05,
                        contrast_mode: str = 'condition', seed: int = 0):
    """Run the continuous and categorical layers on `df` and return one verdict.

    Parameters
    ----------
    df : DataFrame
        Long format as documented at the top of the segregation module
        (one row per electrode-trial). `_synthetic_df` produces a valid one.
    n_splits, n_perm, n_boot, alpha, contrast_mode, seed : passthrough knobs.

    Returns
    -------
    dict:
        verdict     : str  (from classify_segregation)
        continuous  : dict (corr, p, ... straight from subject_clustered_corr)
        categorical : dict (mh_odds_ratio, p, or_lo, or_hi, observed_both)
        contrast_mode : str

    Implementation steps (mind the ORDER — each line consumes the previous)
    ----------------------------------------------------------------------
    CONTINUOUS layer:
      1. elec = compute_sensitivities(df, n_splits=n_splits,
                                      contrast_mode=contrast_mode, alpha=alpha)
         -> per-electrode x (stability) and y (flexibility) on DISJOINT halves.
      2. elec = add_responsiveness(elec, df)          # attach the gain confound
      3. cont = prepare_continuous(elec)              # residualise + within-subj centre
      4. continuous = subject_clustered_corr(cont, n_perm=n_perm)

    CATEGORICAL layer:
      5. labels = per_electrode_labels(df, alpha=alpha,
                                       contrast_mode=contrast_mode)
      6. conj = cmh_conjunction(labels)               # the pooled OR
      7. perm = conjunction_permutation_null(labels, n_perm=n_perm, seed=seed)
      8. boot = bootstrap_conjunction_or(labels, n_boot=n_boot, seed=seed)  # Task 1
      9. categorical = dict(mh_odds_ratio=conj['mh_odds_ratio'],
                            p=perm['p_two_sided'],
                            or_lo=boot['lo'], or_hi=boot['hi'],
                            observed_both=perm['observed'])

    VERDICT:
      10. verdict = classify_segregation(continuous, categorical, alpha=alpha)  # Task 2
          return dict(verdict=verdict, continuous=continuous,
                      categorical=categorical, contrast_mode=contrast_mode)

    Acceptance criterion: on `_synthetic_df` (bx, by drawn independently -> the
    true overlap is exactly chance) the verdict must be 'inconclusive' and the
    bootstrap CI (or_lo, or_hi) must bracket 1.
    """
    raise NotImplementedError("A7 Task 3: compose both layers into one verdict")
