"""A1 skeleton — per-electrode two-way ANOVA electrode definition (plan §1).

DROP-IN TARGET
--------------
These functions are meant to live in
`src/analysis/stats/stability_flexibility_segregation.py`, alongside
`per_electrode_labels` (the nonparametric sibling). Paste them there once
implemented; the output is designed to be a drop-in `labels` argument for the
existing `cmh_conjunction(labels)`.

They rely on helpers already in that module:
    _canonical_labels, _interaction_effect, _is_interaction, resolve_contrasts,
    finalize_contrasts
so implement/test this by importing from the real module, e.g.

    from src.analysis.stats.stability_flexibility_segregation import (
        _canonical_labels, _interaction_effect, resolve_contrasts,
        finalize_contrasts, cmh_conjunction,
    )

WHY THIS EXISTS
---------------
The ANOVA interaction is the plan's *primary* electrode definition (§1) and is
"not yet wired into the pipeline". `per_electrode_labels` is the distribution-
free cross-check; this is the parametric headline. The subtle part is Type III
SS with SUM/EFFECT coding, so the interaction stays orthogonal to the main
effects under the deliberately unequal proportion cells (~75/25). With default
treatment coding a pure congruency/switch main effect leaks into the
"interaction".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def _anova_interaction_stats(elec_df, cond_col, mod_col, hg_col="hg"):
    """Fit ONE electrode's two-way Type III ANOVA and return the interaction stats.

    Parameters
    ----------
    elec_df : DataFrame
        One electrode's trials. Must contain `hg_col`, `cond_col` (e.g.
        'congruency'), `mod_col` (e.g. 'incongruent_proportion').
    cond_col, mod_col : str
        The two factors whose interaction defines selectivity (LWPC =
        congruency × incongruent_proportion; LWPS = switchType × switch_proportion).

    Returns
    -------
    dict with keys:
        'F'  : interaction F statistic (unsigned)     -> np.nan on a singular fit
        'p'  : interaction p-value                     -> np.nan on a singular fit

    Implementation steps
    ---------------------
    1. Build the formula with SUM coding on both factors so Type III interaction
       SS is orthogonal to the main effects:
           f"{hg_col} ~ C({cond_col}, Sum) * C({mod_col}, Sum)"
    2. Fit with statsmodels OLS: `smf.ols(formula, data=elec_df).fit()`.
    3. `anova_lm(model, typ=3)`; pull the row whose index is the interaction term
       (the "C(a, Sum):C(b, Sum)" row) and read its 'F' and 'PR(>F)'.
    4. Wrap the whole thing in try/except -> return {'F': nan, 'p': nan} for
       electrodes missing a cell / singular design (mirror how the module's
       effect helpers return NaN for too-few trials).
    """
    raise NotImplementedError("A1: implement the per-electrode Type III ANOVA fit")


def per_electrode_anova_labels(
    df,
    alpha: float = 0.05,
    contrast_mode: str = "proportion",
    contrasts=None,
    include_cross_controls: bool = True,
):
    """Parametric per-electrode S/F labels from the two-way interaction ANOVA.

    Mirrors `per_electrode_labels` so it is interchangeable as the `labels`
    input to `cmh_conjunction`.

    Returns
    -------
    DataFrame, one row per electrode, with (at least) columns:
        subject, electrode,
        p_cong, q_cong, S,          # stability  (LWPC interaction, FDR'd)
        p_switch, q_switch, F,      # flexibility (LWPS interaction, FDR'd)
        s_sign, f_sign,             # sign of the difference-of-differences
        (if include_cross_controls) p_cross_cs, p_cross_si   # specificity controls
    Column names intentionally match `per_electrode_labels` where they overlap.

    Implementation steps
    ---------------------
    1. `contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))`
       then `work = _canonical_labels(df, contrasts)` — this attaches the
       `_scond/_smod/_fcond/_fmod` cell labels the sign step reuses.
    2. For each (subject, electrode) group:
         a. stability  = _anova_interaction_stats(g, 'congruency', 'incongruent_proportion')
         b. flexibility= _anova_interaction_stats(g, 'switchType', 'switch_proportion')
         c. SIGN: the F-test is unsigned. Get the signed effect from the module's
            own estimator so your sign matches the §2 correlation exactly:
                s_sign = np.sign(_interaction_effect(g['hg'].to_numpy(),
                                                     g['_scond'].to_numpy(),
                                                     g['_smod'].to_numpy(),
                                                     'cohens_d', alpha))
            (and likewise f_sign with '_fcond'/'_fmod').
         d. cross controls (report only): interaction of congruency ×
            switch_proportion and switchType × incongruent_proportion.
    3. FDR across electrodes (Benjamini-Hochberg) on p_cong and p_switch:
           q = multipletests(p.fillna(1), method='fdr_bh')[1]
    4. S = (q_cong < alpha).astype(int); F = (q_switch < alpha).astype(int).
       Optionally require the sign to be in the predicted (positive) direction
       before setting the flag — decide and document which you do.

    Acceptance check to write yourself
    -----------------------------------
    On `_synthetic_df(contrast_mode='proportion')` the S/F flags should agree
    closely with `per_electrode_labels(df, contrast_mode='proportion')` (same
    balanced interaction, different estimator). The cross-control p-values should
    be ~uniform (no true cross-effect in the generator).
    """
    raise NotImplementedError("A1: assemble per-electrode ANOVA labels + FDR")
