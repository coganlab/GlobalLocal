"""
Joint-distribution analysis: do distinct iEEG subpopulations uniquely support
stability (congruency / LWPC) vs flexibility (switch / LWPS), while others do both?

Two complementary, subject-aware tests:

  (1) CONTINUOUS  — partial correlation between per-electrode stability
      sensitivity x and flexibility sensitivity y, across electrodes.
        corr <= 0 -> segregated (distinct subpopulations)
        corr  > 0 -> shared / domain-general core
  (2) CATEGORICAL — 2x2 conjunction via Cochran-Mantel-Haenszel (the
      subject-stratified analogue of Fisher's exact test).
        MH odds ratio < 1 -> segregation ; > 1 -> overlap

Four things can fake an x-y correlation, and each needs its own correction:
  - shared TRIAL noise -> x and y estimated on DISJOINT trial halves, and the
      correlation taken WITHIN a split before averaging over splits. Averaging
      the sensitivities first (as this module originally did) forfeits the
      split entirely: see `compute_sensitivities_per_split`.
  - shared GAIN / SNR  -> x and y residualized on overall responsiveness. The
      proxy must measure gain and NOT the effects: mean|HG|, never |mean HG|
      (see `add_responsiveness`).
  - DESIGN non-orthogonality -> contrasts scored with equal CELL weights, so
      stability and flexibility stay orthogonal however lopsided the 2x2
      cross-tab is. This is a signal confound; the disjoint split does nothing
      about it (see `BALANCE_MAIN_EFFECTS`).
  - SUBJECT nesting -> within-subject centering + within-subject permutation
      (continuous), CMH stratification (categorical).

A null correlation is only interpretable against the split-half NOISE CEILING
(`reliability_x`, `reliability_y` from `split_resolved_corr`): without it,
"the two effects live on different electrodes" cannot be told apart from
"neither effect is measured well enough to correlate with anything".

INPUT (`df`): long format, one row per (electrode, trial):
    subject     : subject id (hashable)
    electrode   : electrode id (unique across subjects)
    hg          : single-trial high-gamma. By default a scalar summarised over
                  your analysis window (e.g. mean HG in the task window,
                  baseline-normalised). When `effect_measure='cluster'` it is
                  instead the per-trial *time course* over the window (a 1-D
                  array), so effects can be measured as an aggregate cluster
                  statistic rather than a difference of window means.
    congruency  : 'c' or 'i'                    (condition contrast)
    switchType  : 's' or 'r'                     (condition contrast)
    incongruent_proportion : block % incongruent (proportion contrast, optional)
    switch_proportion      : block % switch       (proportion contrast, optional)
Optional `responsiveness`: dict/Series {electrode: value}. PREFER your
baseline-vs-signal time_perm_cluster statistic here. Falls back to mean|HG|.

TWO OPTIONS (independent, combinable; both default to the original behaviour):

  * `contrast_mode`  : 'condition'  -> stability = congruency (i vs c),
                                       flexibility = switchType (s vs r)   [default]
                       'proportion' -> stability = LWPC = congruency x
                                       incongruent_proportion interaction,
                                       flexibility = LWPS = switchType x
                                       switch_proportion interaction (each a 2x2
                                       difference-of-differences, LOW minus HIGH
                                       block, so positive = the condition effect
                                       shrinks in the high-proportion block, the
                                       direction behaviour shows -- see
                                       SIGN CONVENTION below)
                       Or pass an explicit `contrasts` spec (see `resolve_contrasts`).
  * `effect_measure` : 'cohens_d'   -> standardized mean difference on window-mean HG [default]
                       'cluster'    -> aggregate time-permutation cluster statistic
                                       (signed cluster mass) on time-resolved HG.
                       'peak_t'     -> signed per-bin t at the moment of maximal
                                       |t| (amplitude only, timing/duration
                                       invariant). A robustness complement to
                                       'cluster': cluster mass conflates effect
                                       amplitude with its duration and is mildly
                                       trial-count sensitive, whereas peak_t reads
                                       the strongest-tuning instant regardless of
                                       how long it lasts. For scalar (single-bin)
                                       HG it reduces to the two-sample t.
"""

import copy
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats import t as _t_dist
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# ----------------------------------------------------------------------------
# contrast specification (condition vs proportion)
# ----------------------------------------------------------------------------
# A `contrasts` spec is a dict with 'stability' and 'flexibility' entries. Each
# entry is one of two kinds:
#
#   SIMPLE      {'col': column, 'pos': positive group, 'neg': negative group}
#       A two-group contrast; the effect is pos - neg, so its sign encodes
#       selectivity direction (the original congruency i-vs-c, switch s-vs-r).
#
#   INTERACTION {'kind': 'interaction',
#                'cond': {simple spec},   # e.g. congruency i vs c
#                'mod':  {simple spec}}   # e.g. incongruent_proportion low vs high
#       A 2x2 difference-of-differences: how much the `cond` effect changes
#       between the two `mod` levels -- i.e. the LWPC (congruency x incongruent-
#       proportion) and LWPS (switchType x switch-proportion) interactions.
#       The sign is set by which `mod` level is `pos`: the preset makes the LOW-
#       proportion block `pos`, so the effect is (cond | low) - (cond | high) and
#       a POSITIVE score is the behavioural adaptation direction (see
#       SIGN CONVENTION below).
#       It is scored as the difference-of-differences of the four CELL means,
#       weighted EQUALLY across cells (see `_interaction_effect`). This matters
#       because the proportion design makes the four cells deliberately unequal
#       (mostly-incongruent ~ 75/25); a naive pooled "+1 diagonal vs -1 diagonal"
#       mean difference is trial-count weighted, so the "+1" super-group is
#       dominated by the frequent cells and a pure main effect leaks into the
#       "interaction". Equal cell weighting makes the estimate orthogonal to both
#       main effects. The +/-1 super-group labels (`_slab`/`_flab`) are still
#       attached for the simple path, but the interaction's effect and its
#       (main-effect-preserving) permutation use the per-cell factor labels
#       `_scond`/`_smod` and `_fcond`/`_fmod` instead.
#
# `pos`/`neg` in a simple spec may be a category label ('i'), an explicit value
# (75.0), a collection, or the sentinels 'high'/'low' (resolved to the column's
# extreme values by `finalize_contrasts`).
# ============================ SIGN CONVENTION ==============================
# The proportion interactions are scored LOW-proportion MINUS HIGH-proportion:
#
#     LWPC = (i - c | 25% incongruent) - (i - c | 75% incongruent)
#     LWPS = (s - r | 25% switch)      - (s - r | 75% switch)
#
# so that a POSITIVE score means the condition effect SHRINKS in the
# high-proportion block -- the direction behaviour shows (the classic
# list-wide proactive-control adjustment), for both LWPC and LWPS.
#
# This is a convention, not a hypothesis: nothing downstream assumes the neural
# effect runs that way. Every test on these scores is two-sided, the electrode
# labels come from an unsigned F, and `stability_flexibility_timing` orients each
# waveform by its own dominant deflection. The convention exists so that "+" means
# the same thing in the neural scores, in the behavioural d-o-d
# (`stability_flexibility_brain_behavior`) and in the manuscript's prose.
#
# It is implemented in ONE place -- which `mod` level is `pos` below. `W_INTERACTION`
# is keyed on pos/neg, not on high/low, so it needs no change; neither does any
# consumer. `tests/analysis/stats/test_effect_sign_conventions.py` pins it.
#
# NOT on this convention (deliberately, and stated where they are defined):
#   * `windowed_anova._signed_contrast_per_window` orders factor levels
#     ALPHABETICALLY, which for congruency x incongruentProportion works out to
#     high - low, i.e. the negative of LWPC here. It is used to split clusters at
#     sign flips and to colour pos/neg cluster bars, neither of which depends on
#     the absolute orientation.
#   * the cross-decoding `block_difference` is high - low on decoding ACCURACY,
#     which is not a condition effect and has no adaptation direction.
# ===========================================================================
_CONTRAST_PRESETS = {
    'condition': {
        'stability':   dict(col='congruency', pos='i', neg='c'),
        'flexibility': dict(col='switchType', pos='s', neg='r'),
    },
    # LWPC / LWPS: the congruency x proportion and switch x proportion interactions,
    # oriented LOW minus HIGH so positive = the behavioural adaptation direction.
    'proportion': {
        'stability':   dict(kind='interaction',
                            cond=dict(col='congruency', pos='i', neg='c'),
                            mod=dict(col='incongruent_proportion', pos='low', neg='high')),
        'flexibility': dict(kind='interaction',
                            cond=dict(col='switchType', pos='s', neg='r'),
                            mod=dict(col='switch_proportion', pos='low', neg='high')),
    },
}


def _copy_contrasts(contrasts):
    return {k: copy.deepcopy(v) for k, v in contrasts.items()}


def _is_interaction(spec):
    return spec.get('kind') == 'interaction'


def resolve_contrasts(contrast_mode='condition', contrasts=None):
    """Return a {'stability':..., 'flexibility':...} contrast spec.

    Pass an explicit `contrasts` to override, else pick a preset by
    `contrast_mode` ('condition' or 'proportion')."""
    if contrasts is not None:
        return _copy_contrasts(contrasts)
    if contrast_mode not in _CONTRAST_PRESETS:
        raise ValueError(f"contrast_mode must be one of {list(_CONTRAST_PRESETS)} "
                         f"(or pass an explicit `contrasts`); got {contrast_mode!r}")
    return _copy_contrasts(_CONTRAST_PRESETS[contrast_mode])


def _finalize_simple(df, spec, key):
    """Fill numeric 'high'/'low' thresholds from the WHOLE df so every electrode
    uses the same split (an electrode may lack a level)."""
    if 'high' in (spec.get('pos'), spec.get('neg')) or \
       'low' in (spec.get('pos'), spec.get('neg')):
        col = spec['col']
        if col not in df.columns:
            raise KeyError(f"contrast column '{col}' for {key} not in "
                           f"df columns {list(df.columns)}")
        num = pd.to_numeric(df[col], errors='coerce').to_numpy()
        finite = num[np.isfinite(num)]
        if finite.size == 0:
            raise ValueError(f"proportion column '{col}' has no numeric values")
        spec.setdefault('_hi', float(np.nanmax(finite)))
        spec.setdefault('_lo', float(np.nanmin(finite)))


def finalize_contrasts(df, contrasts):
    """Resolve any 'high'/'low' sentinels to concrete thresholds, recursing into
    the `cond`/`mod` sub-specs of interaction contrasts."""
    for key in ('stability', 'flexibility'):
        spec = contrasts[key]
        if _is_interaction(spec):
            _finalize_simple(df, spec['cond'], f"{key}.cond")
            _finalize_simple(df, spec['mod'], f"{key}.mod")
        else:
            _finalize_simple(df, spec, key)
    return contrasts


def _group_masks(values, spec):
    """Boolean (pos_mask, neg_mask) for a 1-D array of a simple contrast column.

    Each of `spec['pos']`/`spec['neg']` may be 'high'/'low' (thresholded at the
    df-wide extremes filled in by `finalize_contrasts`), a collection of labels,
    or a single label / numeric value."""
    v = np.asarray(values)
    num = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy()

    def resolve(target):
        if target == 'high':
            return num >= spec['_hi']
        if target == 'low':
            return num <= spec['_lo']
        if isinstance(target, (list, tuple, set, np.ndarray)):
            return np.isin(v, list(target))
        # scalar: numeric closeness where the column parses as numbers, else exact
        try:
            if np.isfinite(num).any():
                return np.isclose(num, float(target))
        except (ValueError, TypeError):
            pass
        return v == target

    return resolve(spec.get('pos')), resolve(spec.get('neg'))


def _contrast_membership(df, spec, key):
    """Boolean (pos_mask, neg_mask) for a stability/flexibility contrast, simple
    or interaction. For an interaction these are the +1/-1 super-groups."""
    if _is_interaction(spec):
        cond, mod = spec['cond'], spec['mod']
        for sub, name in ((cond, 'cond'), (mod, 'mod')):
            if sub['col'] not in df.columns:
                raise KeyError(f"contrast column '{sub['col']}' for {key}.{name} "
                               f"not in df columns {list(df.columns)}")
        cp, cn = _group_masks(df[cond['col']].to_numpy(), cond)
        mp, mn = _group_masks(df[mod['col']].to_numpy(), mod)
        pos = (cp & mp) | (cn & mn)      # interaction contrast weights +1
        neg = (cp & mn) | (cn & mp)      #                             -1
        return pos, neg
    if spec['col'] not in df.columns:
        raise KeyError(f"contrast column '{spec['col']}' for {key} not in df "
                       f"columns {list(df.columns)}")
    return _group_masks(df[spec['col']].to_numpy(), spec)


def _strata_columns(contrasts):
    """The raw factor columns to stratify the disjoint-half split on (the finest
    partition across both contrasts), so every 2x2 cell stays balanced."""
    cols = []
    for key in ('stability', 'flexibility'):
        spec = contrasts[key]
        subs = [spec['cond'], spec['mod']] if _is_interaction(spec) else [spec]
        for s in subs:
            if s['col'] not in cols:
                cols.append(s['col'])
    return cols


def _sub_memberships(df, spec):
    """Per-trial {1.0, 0.0, NaN} membership of an interaction's two sub-factors
    (cond, mod); (None, None) for a simple contrast. Lets the interaction be
    scored as a balanced (equal-cell-weight) difference-of-differences instead of
    a trial-count-weighted pooled-super-group difference (see `_interaction_effect`)."""
    if not _is_interaction(spec):
        return None, None

    def lab(sub):
        pos, neg = _group_masks(df[sub['col']].to_numpy(), sub)
        v = np.full(len(df), np.nan)
        v[np.asarray(neg, bool)] = 0.0
        v[np.asarray(pos, bool)] = 1.0
        return v

    return lab(spec['cond']), lab(spec['mod'])


def _canonical_labels(df, contrasts):
    """Attach '_slab'/'_flab' in {1 (pos), 0 (neg), NaN (excluded)} for the
    stability and flexibility contrasts, so downstream code is agnostic to
    whether the contrast is a simple condition or a 2x2 interaction. For an
    interaction, also attach the two sub-factor labels ('_scond'/'_smod',
    '_fcond'/'_fmod') so the effect can be scored as a balanced difference-of-
    differences (equal cell weights) rather than a pooled-super-group contrast."""
    out = df.copy()
    for lab, key, condcol, modcol in (('_slab', 'stability', '_scond', '_smod'),
                                      ('_flab', 'flexibility', '_fcond', '_fmod')):
        pmask, nmask = _contrast_membership(out, contrasts[key], key)
        v = np.full(len(out), np.nan)
        v[np.asarray(nmask, bool)] = 0.0
        v[np.asarray(pmask, bool)] = 1.0   # pos wins any (unexpected) overlap
        out[lab] = v
        cond, mod = _sub_memberships(out, contrasts[key])
        if cond is not None:
            out[condcol] = cond
            out[modcol] = mod
    return out


# ----------------------------------------------------------------------------
# effect-size helpers
# ----------------------------------------------------------------------------
def _cohens_d(a, b):
    """Standardised mean difference (a - b), pooled SD. NaN if too few trials."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return np.nan if sp == 0 else (a.mean() - b.mean()) / sp


# The cluster-mass effect can be computed two ways, selected by this toggle:
#   USE_TIME_PERM_CLUSTER = False (default) -> deterministic parametric mass
#       (per-bin t thresholded at the alpha t-critical). Fast, dependency-free,
#       and safe to call inside the disjoint-half resampling and the
#       per-electrode label-permutation null.
#   USE_TIME_PERM_CLUSTER = True -> use the project's real
#       `ieeg.calc.stats.time_perm_cluster` to get the cluster-corrected
#       significance mask, then sum the observed statistic over it. This is the
#       genuine pipeline cluster mass, but each call runs `CLUSTER_N_PERM`
#       permutations; only enable it where that cost is acceptable (it is NOT
#       recommended inside the n_splits x label-permutation loops).
USE_TIME_PERM_CLUSTER = False
CLUSTER_N_PERM = 1000
CLUSTER_TAILS = 2                 # two-sided: the sensitivity may go either way


def _time_perm_cluster():
    try:
        from ieeg.calc.stats import time_perm_cluster
        return time_perm_cluster
    except Exception:
        return None


def _cluster_effect(a, b, alpha=0.05):
    """Aggregate cluster-mass statistic between two trial groups' time courses.

    `a`, `b` are (n_trials, n_time) HG time courses; the effect is a - b. We form
    the per-time-bin two-sample t and return the SIGNED cluster mass = summed t
    over the significant bins -- exactly the cluster mass `time_perm_cluster`
    accumulates for the contrast, with the sign retained so it plays the role
    Cohen's d does in the correlation / conjunction.

    Which bins are "significant" depends on `USE_TIME_PERM_CLUSTER`:
      * False (default): bins whose |t| clears the alpha t-critical (a fast,
        deterministic parametric threshold -- no nested permutation, so it is
        safe and cheap inside the disjoint-half resampling and the label
        permutation null).
      * True: the cluster-corrected mask from the real `time_perm_cluster`
        (`CLUSTER_N_PERM` permutations, two-sided by default).

    Returns 0.0 if no bin survives; NaN if too few trials.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = np.where(se > 0, (a.mean(0) - b.mean(0)) / se, 0.0)

    if USE_TIME_PERM_CLUSTER:
        tpc = _time_perm_cluster()
        if tpc is not None:
            try:
                mask, _ = tpc(a, b, alpha, n_perm=CLUSTER_N_PERM,
                              tails=CLUSTER_TAILS, axis=0)
                mask = np.asarray(mask, bool).reshape(-1)
                if mask.shape == tvals.shape:
                    return float(tvals[mask].sum()) if mask.any() else 0.0
            except Exception:
                pass  # fall back to the parametric mass below

    thr = _t_dist.ppf(1 - alpha / 2, na + nb - 2)
    supra = np.isfinite(tvals) & (np.abs(tvals) > thr)
    if not supra.any():
        return 0.0
    return float(tvals[supra].sum())      # signed cluster mass


def _peak_t_effect(a, b):
    """Signed per-bin two-sample t at the moment of maximal |t| (a - b).

    An amplitude-only complement to `_cluster_effect`: cluster MASS sums t over
    every supra-threshold bin, so it grows with the effect's DURATION (and, via
    the number of bins that clear threshold, with trial count). peak_t instead
    returns the t at the single strongest-tuning bin -- timing- and duration-
    invariant -- so a segregation/overlap verdict that holds under BOTH measures
    isn't an artifact of one contrast simply lasting longer. No threshold and no
    permutation, so it is cheap inside the disjoint-half and label-permutation
    loops. For scalar (single-bin) HG it reduces to the ordinary two-sample t.

    Returns 0.0 if no finite bin; NaN if too few trials.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = np.where(se > 0, (a.mean(0) - b.mean(0)) / se, 0.0)
    tvals = tvals[np.isfinite(tvals)]
    if tvals.size == 0:
        return 0.0
    return float(tvals[np.argmax(np.abs(tvals))])   # signed peak t


def _stack(vals):
    """List/array of per-trial hg -> (n,) for scalars or (n, T) for time courses."""
    vals = list(vals)
    if not vals:
        return np.empty(0)
    if np.ndim(vals[0]) == 0:
        return np.asarray(vals, float)
    return np.vstack([np.asarray(v, float) for v in vals])


def _require_scalar_hg(arr, effect_measure):
    """Guard: 'cohens_d' is only defined on scalar (window-mean) HG.

    `_cohens_d` / `_interaction_cohens_d` standardise a per-time-bin mean by a
    SCALAR pooled SD, so handing them time courses returns a length-T vector
    rather than one number. That propagates silently into `compute_sensitivities`
    as an array-valued sensitivity and corrupts the correlation instead of
    failing. Assemble the table with window means for 'cohens_d', or use
    'cluster' / 'peak_t' for time courses.
    """
    if effect_measure == 'cohens_d' and np.ndim(arr) > 1 and arr.shape[1] > 1:
        raise ValueError(
            "effect_measure='cohens_d' requires scalar (window-mean) hg, but the "
            f"table holds time courses of length {arr.shape[1]}. Either build the "
            "long table with window means, or use effect_measure='cluster' / "
            "'peak_t', which are defined on time courses.")


def _effect_from_arrays(hg, lab, effect_measure, alpha):
    """Effect between the label==1 and label==0 trials of one electrode."""
    pos = _stack(hg[lab == 1])
    neg = _stack(hg[lab == 0])
    _require_scalar_hg(pos, effect_measure)
    if effect_measure == 'cluster':
        return _cluster_effect(pos, neg, alpha=alpha)
    if effect_measure == 'peak_t':
        return _peak_t_effect(pos, neg)
    if effect_measure == 'cohens_d':
        return _cohens_d(pos, neg)
    raise ValueError("effect_measure must be 'cohens_d', 'cluster' or 'peak_t'; "
                     f"got {effect_measure!r}")


def _contrast_effect(frame, labcol, effect_measure, alpha):
    lab = frame[labcol].to_numpy()
    hg = frame['hg'].to_numpy()
    return _effect_from_arrays(hg, lab, effect_measure, alpha)


# --- balanced (equal-cell-weight) interaction effect -------------------------
# The proportion contrasts (LWPC/LWPS) are 2x2 interactions. Scoring them by
# pooling the two "+1" diagonal cells against the two "-1" cells and taking a
# trial-count-weighted mean difference confounds the interaction with cell
# imbalance: in a proportion design the "+1" super-group is dominated by the
# frequent (majority) cells and "-1" by the rare (minority) cells, so a pure
# congruency/switch MAIN effect (and any oddball/frequency response) leaks in.
# The difference-of-differences of the four CELL means (equal weights) is
# orthogonal to both main effects, so it isolates the interaction.
def _dod_cells(hg, cond, mod):
    """The four (cond, mod) cells as stacked arrays; None if any cell is empty."""
    cells = {}
    for cv in (1.0, 0.0):
        for mv in (1.0, 0.0):
            sel = (cond == cv) & (mod == mv)
            if not np.any(sel):
                return None
            cells[(cv, mv)] = _stack(hg[sel])
    return cells


# Cell weights over the four (cond, mod) cells. Both are contrasts in cell-mean
# space; they are mutually ORTHOGONAL, which is the point (see BALANCE_MAIN_EFFECTS).
W_INTERACTION = {(1.0, 1.0): +1.0, (0.0, 1.0): -1.0,      # difference-of-differences
                 (1.0, 0.0): -1.0, (0.0, 0.0): +1.0}
W_MAIN = {(1.0, 1.0): +0.5, (0.0, 1.0): -0.5,             # main effect of `cond`,
          (1.0, 0.0): +0.5, (0.0, 0.0): -0.5}             # equal weight over `mod`


def _cell_stats(cells):
    """Per-cell mean, variance and n; None if any cell has < 2 trials."""
    means, varis, ns = {}, {}, {}
    for k, v in cells.items():
        if len(v) < 2:
            return None
        means[k] = v.mean(0); varis[k] = v.var(0, ddof=1); ns[k] = len(v)
    return means, varis, ns


def _combine(means, w):
    return sum(w[k] * means[k] for k in means)


def _interaction_cohens_d(cells, w=None):
    """Standardised cell-weighted contrast of the four cell means (equal cell
    weight), pooled within-cell SD. NaN if any cell has < 2 trials."""
    w = W_INTERACTION if w is None else w
    st = _cell_stats(cells)
    if st is None:
        return np.nan
    means, _, ns = st
    ssq = sum((ns[k] - 1) * cells[k].var(ddof=1) for k in cells)
    sp = np.sqrt(ssq / sum(ns[k] - 1 for k in ns))
    return np.nan if sp == 0 else _combine(means, w) / sp


def _cell_weighted_t(cells, w):
    """Per-bin t of the cell-weighted contrast; (tvals, total_n) or (None, None)."""
    st = _cell_stats(cells)
    if st is None:
        return None, None
    means, varis, ns = st
    num = _combine(means, w)
    se = np.sqrt(sum(w[k] ** 2 * varis[k] / ns[k] for k in cells))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(se > 0, num / se, 0.0), sum(ns.values())


def _interaction_cluster(cells, alpha, w=None):
    """Signed cluster mass of the per-bin cell-weighted t (equal cell weight).
    Parametric threshold only -- the two-group time_perm_cluster path does not
    apply to a 4-cell contrast."""
    tvals, n_tot = _cell_weighted_t(cells, W_INTERACTION if w is None else w)
    if tvals is None:
        return np.nan
    thr = _t_dist.ppf(1 - alpha / 2, n_tot - 4)
    supra = np.isfinite(tvals) & (np.abs(tvals) > thr)
    return 0.0 if not supra.any() else float(np.sum(tvals[supra]))


def _interaction_peak_t(cells, w=None):
    """Signed per-bin cell-weighted t at the moment of maximal |t| (equal cell
    weight). Amplitude-only complement to `_interaction_cluster`, in the same
    spirit as `_peak_t_effect` for the two-group case."""
    tvals, _ = _cell_weighted_t(cells, W_INTERACTION if w is None else w)
    if tvals is None:
        return np.nan
    tvals = np.atleast_1d(tvals)
    tvals = tvals[np.isfinite(tvals)]
    if tvals.size == 0:
        return 0.0
    return float(tvals[np.argmax(np.abs(tvals))])   # signed peak cell-weighted t


def _interaction_effect(hg, cond, mod, effect_measure, alpha, w=None):
    """Balanced 2x2 cell-weighted effect between the (cond, mod) cells of one
    electrode; mirrors `_effect_from_arrays` but equal-cell-weighted. `w` selects
    the contrast: `W_INTERACTION` (default) for the difference-of-differences,
    `W_MAIN` for the main effect of `cond` balanced over `mod`."""
    valid = ~(np.isnan(cond) | np.isnan(mod))
    cells = _dod_cells(hg[valid], cond[valid], mod[valid])
    if cells is None:
        return np.nan
    _require_scalar_hg(next(iter(cells.values())), effect_measure)
    if effect_measure == 'cluster':
        return _interaction_cluster(cells, alpha, w)
    if effect_measure == 'peak_t':
        return _interaction_peak_t(cells, w)
    if effect_measure == 'cohens_d':
        return _interaction_cohens_d(cells, w)
    raise ValueError("effect_measure must be 'cohens_d', 'cluster' or 'peak_t'; "
                     f"got {effect_measure!r}")


# A MAIN-effect contrast (contrast_mode='condition') scored by pooling all
# trials on one side against all on the other is trial-count-weighted, and so is
# NOT orthogonal to the other contrast unless the 2x2 cross-tab happens to be
# proportional. When it is not -- e.g. incongruent trials are disproportionately
# switch trials -- a purely congruency-driven electrode also scores a switch
# effect, and every electrode inherits that leakage, which is exactly what a
# spurious across-electrode correlation is made of. It is a SIGNAL confound, so
# the disjoint-half split does not touch it: both halves carry the same design.
#
# Scoring the main effect as the equal-weight mean of the within-cell
# differences (W_MAIN) makes the two contrasts orthogonal in cell-mean space by
# construction, whatever the cell counts. The interaction path has always done
# this (W_INTERACTION); this extends it to main effects.
#
# Simulated under a true null (independent per-electrode sensitivities), 12
# subjects, ~330 electrodes, congruency correlated with switchType:
#
#     scoring                          naive     within-split
#     pooled two-group (old)          +0.58         +0.44
#     cell-weighted    (new)          -0.21         -0.07
#
# Set False to recover the old trial-count-weighted behaviour for comparison.
BALANCE_MAIN_EFFECTS = True


def _effect_for(frame, key, labcol, contrasts, effect_measure, alpha, w=None):
    """Per-electrode effect for a stability/flexibility contrast: an equal-cell-
    weight difference-of-differences for an interaction, and (when
    BALANCE_MAIN_EFFECTS) an equal-cell-weight main effect balanced over the
    OTHER contrast's factor for a simple contrast -- so the two contrasts are
    orthogonal regardless of cell imbalance. For an interaction, `w=W_MAIN`
    scores the `cond` main effect on the same four cells instead."""
    spec = contrasts[key]
    if _is_interaction(spec):
        condcol, modcol = (('_scond', '_smod') if key == 'stability'
                           else ('_fcond', '_fmod'))
        return _interaction_effect(frame['hg'].to_numpy(),
                                   frame[condcol].to_numpy(),
                                   frame[modcol].to_numpy(),
                                   effect_measure, alpha, w=w)
    othercol = '_flab' if labcol == '_slab' else '_slab'
    if BALANCE_MAIN_EFFECTS and othercol in frame.columns:
        other = frame[othercol].to_numpy()
        if np.unique(other[~np.isnan(other)]).size == 2:
            return _interaction_effect(frame['hg'].to_numpy(),
                                       frame[labcol].to_numpy(), other,
                                       effect_measure, alpha, w=W_MAIN)
    return _contrast_effect(frame, labcol, effect_measure, alpha)


# Main effects (congruency, switch type) scored in the proportion run on the
# same cells and halves as LWPC/LWPS (`main_effects=True`). See
# docs/closing_figure_plan.md for why not a separate condition-mode run: its
# halves do not line up, and block effects leak into its congruency score.
MAIN_EFFECT_COLS = ('mxA', 'mxB', 'myA', 'myB')


def main_effect_view(per_split):
    """The per-split table with the main effects in the xA/xB/yA/yB slots, so
    `split_resolved_corr`, `map_reliability` etc. run on them unchanged."""
    return per_split[['subject', 'electrode', 'split', *MAIN_EFFECT_COLS]].rename(
        columns=dict(zip(MAIN_EFFECT_COLS, ('xA', 'xB', 'yA', 'yB'))))


def _stratified_half_split(sub, rng, strata_cols=('congruency', 'switchType')):
    """Split one electrode's trials into two disjoint halves, balanced on the
    contrast cells so neither half is confounded."""
    h1, h2 = [], []
    for _, cell in sub.groupby(list(strata_cols), dropna=False):
        idx = cell.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = len(idx) // 2
        h1.append(idx[:cut]); h2.append(idx[cut:])
    return np.concatenate(h1), np.concatenate(h2)


# ----------------------------------------------------------------------------
# (A) per-electrode sensitivities x (stability) and y (flexibility)
# ----------------------------------------------------------------------------
def compute_sensitivities(df, n_splits=200, seed=0, contrast_mode='condition',
                          contrasts=None, effect_measure='cohens_d', alpha=0.05):
    """x from the stability contrast on one trial-half, y from the flexibility
    contrast on the DISJOINT half, averaged over many random disjoint splits.

    DIAGNOSTIC ONLY -- do not feed this to the correlation. Averaging over
    splits before correlating forfeits the disjoint-half design (see
    `compute_sensitivities_per_split`); the analysis uses the per-split
    estimator and correlates within a split.

    `contrast_mode`/`contrasts` pick which manipulations define stability vs
    flexibility; `effect_measure` picks how each contrast is quantified
    ('cohens_d' on window-mean HG, 'cluster' aggregate cluster-mass statistic on
    time courses, or 'peak_t' amplitude-only peak of the per-bin t)."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)
    strata = _strata_columns(contrasts)
    rng = np.random.default_rng(seed)
    rows = []
    for (subj, elec), sub in work.groupby(['subject', 'electrode']):
        sub = sub.reset_index(drop=True)
        xs, ys = [], []
        for _ in range(n_splits):
            h1, h2 = _stratified_half_split(sub, rng, strata_cols=strata)
            hx, hy = (h1, h2) if rng.random() < 0.5 else (h2, h1)  # use data symmetrically
            gx, gy = sub.loc[hx], sub.loc[hy]
            xs.append(_effect_for(gx, 'stability', '_slab', contrasts, effect_measure, alpha))
            ys.append(_effect_for(gy, 'flexibility', '_flab', contrasts, effect_measure, alpha))
        rows.append(dict(subject=subj, electrode=elec,
                         x=np.nanmean(xs), y=np.nanmean(ys)))
    return pd.DataFrame(rows)


def compute_sensitivities_per_split(df, n_splits=200, seed=0,
                                    contrast_mode='condition', contrasts=None,
                                    effect_measure='cohens_d', alpha=0.05,
                                    main_effects=False):
    """Per-split, per-electrode effects on BOTH halves: xA, xB, yA, yB.

    `compute_sensitivities` averages x and y over splits and hands one (x, y)
    per electrode to the correlation. That forfeits the disjoint-half design:
    the average is dominated by the K(K-1) cross terms cov(x_j, y_k), j != k,
    whose halves overlap ~50%, so the across-electrode correlation reinstates
    essentially all of the same-trial bias (see `split_resolved_corr`).

    Keeping the splits separate lets the correlation be taken WITHIN a split,
    where x and y really are estimated on disjoint trials, and averaged
    afterwards. Computing the contrast on both halves rather than one also
    buys the split-half reliabilities for free, which is what makes a null
    correlation interpretable -- and it uses the data symmetrically by
    construction, so the coin flip in `compute_sensitivities` is unnecessary
    here.

    Costs 2x the effect evaluations of `compute_sensitivities` (four per split
    instead of two); with `effect_measure='cluster'`, where each evaluation runs
    a permutation test, drop `n_splits` accordingly.

    `main_effects=True` (proportion mode only) also scores each process's main
    effect, W_MAIN over the same four cells, on the same halves: `mxA`/`mxB`
    (congruency) and `myA`/`myB` (switch type). `xA`..`yB` are unchanged.
    """
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    if main_effects and not _is_interaction(contrasts['stability']):
        raise ValueError("main_effects=True needs contrast_mode='proportion'; in "
                         "condition mode x and y already are the main effects")
    work = _canonical_labels(df, contrasts)
    strata = _strata_columns(contrasts)
    rng = np.random.default_rng(seed)
    rows = []
    for (subj, elec), sub in work.groupby(['subject', 'electrode']):
        sub = sub.reset_index(drop=True)
        for k in range(n_splits):
            h1, h2 = _stratified_half_split(sub, rng, strata_cols=strata)
            g1, g2 = sub.loc[h1], sub.loc[h2]
            row = dict(
                subject=subj, electrode=elec, split=k,
                xA=_effect_for(g1, 'stability', '_slab', contrasts, effect_measure, alpha),
                xB=_effect_for(g2, 'stability', '_slab', contrasts, effect_measure, alpha),
                yA=_effect_for(g1, 'flexibility', '_flab', contrasts, effect_measure, alpha),
                yB=_effect_for(g2, 'flexibility', '_flab', contrasts, effect_measure, alpha),
            )
            if main_effects:
                for col, g, key, lab in (('mxA', g1, 'stability', '_slab'),
                                         ('mxB', g2, 'stability', '_slab'),
                                         ('myA', g1, 'flexibility', '_flab'),
                                         ('myB', g2, 'flexibility', '_flab')):
                    row[col] = _effect_for(g, key, lab, contrasts, effect_measure,
                                           alpha, w=W_MAIN)
            rows.append(row)
    return pd.DataFrame(rows)


def average_over_splits(per_split):
    """Collapse the per-split table to one (x, y) per electrode, for the scatter
    plot and for the split-averaged diagnostic correlation. Both halves are
    averaged, which is the symmetric (lower-variance) version of the coin flip
    in `compute_sensitivities`. This is NOT the estimator the inference uses.
    Main-effect columns, when present, become `mx` and `my` the same way."""
    g = per_split.groupby(['subject', 'electrode'], as_index=False).mean(numeric_only=True)
    g['x'] = g[['xA', 'xB']].mean(axis=1)
    g['y'] = g[['yA', 'yB']].mean(axis=1)
    if set(MAIN_EFFECT_COLS) <= set(g.columns):
        g['mx'] = g[['mxA', 'mxB']].mean(axis=1)
        g['my'] = g[['myA', 'myB']].mean(axis=1)
    return g[[c for c in ('subject', 'electrode', 'x', 'y', 'mx', 'my') if c in g]]


def _unit_vector(v, method):
    """Centre (and rank, for spearman) then scale to unit norm, so that the
    correlation between two such vectors is exactly their dot product."""
    a = np.asarray(v, float)
    if method == 'spearman':
        a = pd.Series(a).rank().to_numpy()
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def split_resolved_corr(per_split, resp, min_elec=3, method='spearman',
                        n_perm=10000, seed=1, covariates=None):
    """Correlate within each split, then average -- with a noise ceiling.

    Three quantities, all averaged over splits, all on residualised and
    within-subject-centred values (same treatment as `prepare_continuous`):

      S      = mean_k  1/2 [ corr(xA_k, yB_k) + corr(xB_k, yA_k) ]
      rel_x  = mean_k  corr(xA_k, xB_k)
      rel_y  = mean_k  corr(yA_k, yB_k)

    `S` is the co-localization estimate: each term correlates a stability
    effect against a flexibility effect measured on the OTHER half of that
    electrode's trials, so same-trial noise cannot contribute. Both cross
    directions are used, so the halves enter symmetrically.

    `rel_x` and `rel_y` are the split-half reliabilities -- how well each effect
    correlates with *itself* across halves. They are the noise ceiling, and
    without them `S ~ 0` is uninterpretable: it cannot distinguish "the two
    effects load on different electrodes" from "neither effect is measured well
    enough to correlate with anything". `S_corrected = S / sqrt(rel_x * rel_y)`
    is reported when both reliabilities are positive.

    Note these are half-length reliabilities, i.e. they describe an effect
    estimated from half the trials, not from all of them -- so read them as a
    ceiling on `S`, not as "the reliability of the reported sensitivity" (for
    that, Spearman-Brown them upward). Deliberately un-corrected here: `S` is
    itself built from half-length estimates, so numerator and denominator are at
    the same trial count and the ratio is the attenuation correction it should
    be.

    The null permutes the electrode labels of y within subject, using the SAME
    permutation for every split, so it breaks the x-y electrode correspondence
    while leaving each split's internal structure intact. `p` applies to
    `corr_noise_corrected` as well: the two differ only by a fixed positive
    denominator, so they order the null identically.

    `covariates` (optional DataFrame indexed by electrode, e.g. MNI
    coordinates) are regressed out together with responsiveness. They are
    centred within subject first, which makes the within-subject residuals
    exactly orthogonal to them.
    """
    d = per_split.merge(pd.Series(resp, name='resp').rename_axis('electrode').reset_index(),
                        on='electrode', how='left')
    cov_cols = [] if covariates is None else list(covariates.columns)
    if cov_cols:
        d = d.merge(covariates.rename_axis('electrode').reset_index(),
                    on='electrode', how='left')
    n_elec_in = d['electrode'].nunique()
    d = d.dropna(subset=['xA', 'xB', 'yA', 'yB', 'resp', *cov_cols])

    # Keep only electrodes present in EVERY split, so the per-split vectors are
    # comparable and the permutation applies to a fixed electrode set. An
    # electrode whose effect is undefined on even one split (a contrast cell
    # emptied by that split) is dropped outright; `n_electrodes_dropped` reports
    # how many, since with few trials per cell this can bite.
    n_sp = per_split['split'].nunique()
    complete = d.groupby('electrode')['split'].transform('size') == n_sp
    d = d[complete]
    d = d[d.groupby('subject')['electrode'].transform('nunique') >= min_elec]
    if d.empty:
        raise ValueError(
            "no electrode survived the completeness / min_elec filter. With "
            "BALANCE_MAIN_EFFECTS, an effect needs >= 2 trials in EACH of the "
            "four 2x2 cells of each half -- i.e. >= 4 trials per cell before "
            "splitting -- where the old two-group scoring needed only 2 per "
            "group. Check your smallest cell count per electrode.")
    n_dropped = int(n_elec_in - d['electrode'].nunique())
    if n_dropped > 0.1 * n_elec_in:
        warnings.warn(
            f"{n_dropped}/{n_elec_in} electrodes dropped from the continuous "
            "test because their effect was undefined on at least one split. "
            "That usually means some 2x2 cell is too small to survive halving "
            "(>= 4 trials per cell needed). Treat `corr` as computed on a "
            "biased subset until you have checked the cell counts.",
            RuntimeWarning, stacklevel=2)

    elecs = np.sort(d['electrode'].unique())
    e_index = {e: i for i, e in enumerate(elecs)}
    subj_of = d.drop_duplicates('electrode').set_index('electrode')['subject']
    subj = subj_of.loc[elecs].to_numpy()
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]

    splits = np.sort(d['split'].unique())
    mats = {k: np.full((len(splits), len(elecs)), np.nan) for k in ('xA', 'xB', 'yA', 'yB')}
    for si, s in enumerate(splits):
        block = d[d['split'] == s]
        cols = block['electrode'].map(e_index).to_numpy()
        for k in mats:
            mats[k][si, cols] = block[k].to_numpy()

    # Residualise on responsiveness, then within-subject centre -- per split,
    # per quantity, mirroring prepare_continuous.
    resp_vec = d.drop_duplicates('electrode').set_index('electrode')['resp'].loc[elecs].to_numpy()
    if cov_cols:
        C = d.drop_duplicates('electrode').set_index('electrode')[cov_cols].loc[elecs].to_numpy(
            float, copy=True)
        for g in groups:
            C[g] -= C[g].mean(axis=0)
        design = np.column_stack([np.ones(len(elecs)), resp_vec, C])
    for k in mats:
        for si in range(len(splits)):
            r = (mats[k][si] - design @ np.linalg.lstsq(design, mats[k][si], rcond=None)[0]
                 if cov_cols else _ols_resid(mats[k][si], resp_vec))
            for g in groups:
                r[g] -= r[g].mean()
            mats[k][si] = r

    U = {k: np.vstack([_unit_vector(mats[k][si], method) for si in range(len(splits))])
         for k in mats}

    # Because each row of U is centred and unit-norm, a correlation is a dot
    # product, so the whole split-averaged cross-correlation under an electrode
    # permutation `perm` of y is just a lookup in one precomputed matrix:
    #   M[i, j] = mean_k 1/2 [ U_xA[k,i] U_yB[k,j] + U_xB[k,i] U_yA[k,j] ]
    #   S(perm) = sum_i M[i, perm[i]],   S(identity) = trace(M)
    # That turns each permutation from O(n_splits * n_elec) into O(n_elec).
    M = 0.5 * (U['xA'].T @ U['yB'] + U['xB'].T @ U['yA']) / len(splits)
    identity = np.arange(len(elecs))
    obs = float(np.trace(M))
    rel_x = float((U['xA'] * U['xB']).sum(1).mean())
    rel_y = float((U['yA'] * U['yB']).sum(1).mean())

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = identity.copy()
        for g in groups:
            perm[g] = g[rng.permutation(len(g))]
        null[i] = M[identity, perm].sum()
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))

    denom = np.sqrt(rel_x * rel_y) if (rel_x > 0 and rel_y > 0) else np.nan
    # A reliability <= 0 is not a small effect, it is no measurement: the two
    # halves of the same effect do not agree at all. It happens here mainly
    # because these values are within-subject centred, which at small
    # per-subject electrode counts removes most of the between-electrode
    # variance the reliability is computed over. Say so rather than returning a
    # bare NaN, and never report `corr` as a null result on that basis.
    note = None
    if not (rel_x > 0 and rel_y > 0):
        small = int((pd.Series(subj).value_counts() <= 3).sum())
        note = (f"reliability_x={rel_x:+.3f}, reliability_y={rel_y:+.3f}: a "
                "split-half reliability <= 0 means that map carries no "
                "recoverable signal after within-subject centring "
                f"({small} subject(s) contribute <= 3 electrodes). `corr` is "
                "uninterpretable here -- it cannot distinguish 'the effects "
                "load on different electrodes' from 'neither effect is "
                "measured well enough to correlate with anything'. Compare "
                "against map_reliability, which does not centre.")
    return dict(corr=obs, p=p, method=method,
                n_electrodes=len(elecs), n_subjects=len(groups), n_splits=len(splits),
                n_electrodes_dropped=int(n_elec_in - len(elecs)),
                reliability_x=rel_x, reliability_y=rel_y, reliability_note=note,
                corr_noise_corrected=(float(obs / denom) if np.isfinite(denom) else np.nan))


def naive_sensitivities(df, contrast_mode='condition', contrasts=None,
                        effect_measure='cohens_d', alpha=0.05):
    """Per-electrode (x, y) from ALL trials (shares trial noise). Only for the
    diagnostic naive-vs-disjoint comparison; the analysis itself always uses the
    disjoint-half estimator in `compute_sensitivities`."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)
    rows = []
    for (subj, elec), g in work.groupby(['subject', 'electrode']):
        rows.append(dict(subject=subj, electrode=elec,
                         x=_effect_for(g, 'stability', '_slab', contrasts, effect_measure, alpha),
                         y=_effect_for(g, 'flexibility', '_flab', contrasts, effect_measure, alpha)))
    return pd.DataFrame(rows)


def add_responsiveness(elec_df, df, responsiveness=None):
    """Overall task-drive per electrode (the gain confound). Prefer passing your
    baseline-vs-signal cluster statistic; else use mean|HG| as a proxy.

    The proxy must measure the electrode's GAIN, not its effects. The scalar
    branch previously returned |mean HG| -- the absolute value of the electrode's
    mean -- which is a function of the very contrasts being residualised: with
    both effects pushing the mean the same way, |mean HG| behaves like their sum,
    and regressing x and y on their own sum drives the residuals apart. Simulated
    under a true null with a population-level effect in both contrasts, that
    alone produced correlations of -0.12 to -0.21, i.e. spurious *segregation*;
    mean|HG| lands at -0.07 to +0.03, level with the true-gain oracle. Both
    branches now compute mean|HG|, as the docstring always said.
    """
    elec_df = elec_df.copy()
    if responsiveness is not None:
        r = pd.Series(responsiveness)
    else:
        def _mean_abs(v):
            # mean |HG| over trials (and time bins, for time-resolved HG)
            return float(np.nanmean(np.abs(_stack(v.to_numpy()))))
        r = df.groupby('electrode')['hg'].apply(_mean_abs)
    elec_df['resp'] = elec_df['electrode'].map(r)
    return elec_df


# ----------------------------------------------------------------------------
# (B) continuous test: subject-aware partial correlation, gain-controlled
# ----------------------------------------------------------------------------
def _ols_resid(y, x):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1, b0 = np.polyfit(x, y, 1)
    return y - (b0 + b1 * x)


def prepare_continuous(elec_df, min_elec=3):
    """1) regress out overall responsiveness (kills gain-driven co-inflation);
       2) within-subject centre (isolates WITHIN-subject co-selectivity, so the
          estimate matches the within-subject permutation null)."""
    d = elec_df.dropna(subset=['x', 'y', 'resp']).copy()
    d['x1'] = _ols_resid(d['x'], d['resp'])
    d['y1'] = _ols_resid(d['y'], d['resp'])
    d = d[d.groupby('subject')['electrode'].transform('size') >= min_elec].copy()
    d['x_resid'] = d['x1'] - d.groupby('subject')['x1'].transform('mean')
    d['y_resid'] = d['y1'] - d.groupby('subject')['y1'].transform('mean')
    return d


def subject_clustered_corr(d, method='spearman', n_perm=10000, seed=1):
    """Correlation between residualised x and y, with a null built by permuting
    y WITHIN each subject. Because between-subject structure is preserved under
    that permutation, the test isolates the within-subject association."""
    corr_fn = spearmanr if method == 'spearman' else pearsonr
    x = d['x_resid'].to_numpy(); y = d['y_resid'].to_numpy()
    obs = corr_fn(x, y)[0]
    subj = d['subject'].to_numpy()
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = y.copy()
        for idx in groups:
            yp[idx] = y[rng.permutation(idx)]
        null[i] = corr_fn(x, yp)[0]
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
    return dict(corr=obs, p=p, method=method, n_electrodes=len(x),
                n_subjects=len(groups))


def mixedlm_check(d):
    """Optional cross-check: within-subject slope with a subject random intercept."""
    import statsmodels.formula.api as smf
    m = smf.mixedlm('y1 ~ x1', d, groups=d['subject']).fit(reml=False)
    return dict(slope=m.params['x1'], p=m.pvalues['x1'])


# ----------------------------------------------------------------------------
# (C) categorical labels + 2x2 conjunction (CMH, subject-stratified)
# ----------------------------------------------------------------------------
def per_electrode_labels(df, n_perm=2000, alpha=0.05, seed=2,
                         contrast_mode='condition', contrasts=None,
                         effect_measure='cohens_d', fdr_correction='fdr_bh'):
    """Binary S (stability-selective) and F (flexibility-selective) per electrode,
    from within-electrode permutation p-values, optionally FDR-corrected across electrodes.

    `contrast_mode`/`contrasts`/`effect_measure` behave as in
    `compute_sensitivities`. Output columns keep the names `p_cong`/`q_cong`
    (stability) and `p_switch`/`q_switch` (flexibility) whatever the contrast.
    `fdr_correction='none'` leaves q-values equal to raw p-values and flags at
    raw p < alpha for exploratory threshold-sensitivity runs."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)
    rng = np.random.default_rng(seed)
    recs = []
    for (subj, elec), sub in work.groupby(['subject', 'electrode']):
        hg = sub['hg'].to_numpy()

        def perm_p_simple(lab):
            valid = ~np.isnan(lab)
            l = lab[valid].astype(int)
            h = hg[valid]
            obs = _effect_from_arrays(h, l, effect_measure, alpha)
            if np.isnan(obs):
                return np.nan
            cnt = sum(abs(_effect_from_arrays(h, rng.permutation(l), effect_measure, alpha)) >= abs(obs)
                      for _ in range(n_perm))
            return (cnt + 1) / (n_perm + 1)

        def perm_p_interaction(condcol, modcol):
            # Balanced difference-of-differences, nulled by permuting the
            # modulator WITHIN each condition level. Cell counts are held fixed
            # and the interaction is nulled -- unlike a free label shuffle, which
            # lets a main effect masquerade as an interaction under cell
            # imbalance.
            #
            # Two honest caveats about this null:
            #
            # (1) It does NOT hold both main effects fixed. Permuting `mod`
            #     within each `cond` level preserves the CONDITION main effect
            #     exactly but destroys the MODULATOR main effect, which reappears
            #     as extra variance in the permuted cell means and widens the
            #     null -- i.e. conservative. The mirror scheme (permute `cond`
            #     within each `mod` level) is equally valid and conservative in
            #     the other direction; this one is preferred only because
            #     congruency/switch main effects in HG are typically much larger
            #     than block-proportion main effects, so less variance leaks in.
            #
            # (2) The modulator is a BLOCK-level variable (constant within a
            #     block), so permuting it trial-by-trial builds label
            #     configurations the design cannot produce and ignores
            #     within-block dependence (drift, fatigue, tonic block state).
            #     Trials within a block are not exchangeable in `mod`, so this
            #     null is anticonservative on that axis. The mirror scheme does
            #     not have this problem -- congruency and switchType ARE
            #     randomized trial-to-trial within a block, so they are genuinely
            #     exchangeable. Permuting the condition within block is the
            #     stricter choice; see `docs/analysis_guide.md` §14.2.
            cond = sub[condcol].to_numpy(); mod = sub[modcol].to_numpy()
            valid = ~(np.isnan(cond) | np.isnan(mod))
            cond = cond[valid]; mod = mod[valid]; h = hg[valid]
            obs = _interaction_effect(h, cond, mod, effect_measure, alpha)
            if np.isnan(obs):
                return np.nan
            blocks = [np.where(cond == cv)[0] for cv in (1.0, 0.0)]
            cnt = 0
            for _ in range(n_perm):
                mp = mod.copy()
                for idx in blocks:
                    mp[idx] = mod[rng.permutation(idx)]
                if abs(_interaction_effect(h, cond, mp, effect_measure, alpha)) >= abs(obs):
                    cnt += 1
            return (cnt + 1) / (n_perm + 1)

        def perm_p(key, labcol, condcol, modcol):
            if _is_interaction(contrasts[key]):
                return perm_p_interaction(condcol, modcol)
            return perm_p_simple(sub[labcol].to_numpy())

        recs.append(dict(subject=subj, electrode=elec,
                         p_cong=perm_p('stability', '_slab', '_scond', '_smod'),
                         p_switch=perm_p('flexibility', '_flab', '_fcond', '_fmod')))
    out = pd.DataFrame(recs)
    if fdr_correction not in ('fdr_bh', 'none'):
        raise ValueError("fdr_correction must be 'fdr_bh' or 'none'")
    if fdr_correction == 'fdr_bh':
        out['q_cong'] = multipletests(out['p_cong'].fillna(1), method='fdr_bh')[1]
        out['q_switch'] = multipletests(out['p_switch'].fillna(1), method='fdr_bh')[1]
    else:
        out['q_cong'] = out['p_cong']
        out['q_switch'] = out['p_switch']
    out['S'] = (out['q_cong'] < alpha).astype(int)
    out['F'] = (out['q_switch'] < alpha).astype(int)
    return out


# ----------------------------------------------------------------------------
# A1: parametric per-electrode electrode definition (two-way interaction ANOVA)
# ----------------------------------------------------------------------------
# The plan's *primary* (parametric) electrode definition. Complements the
# nonparametric `per_electrode_labels`: it tests the SAME balanced 2x2
# interaction but with a Type III ANOVA F rather than a within-electrode
# permutation, so the two should agree closely and cross-check each other's
# assumptions. Both feed `cmh_conjunction` unchanged (same output columns).
def _scalar_hg(hg):
    """Per-trial hg as plain scalars, reducing time-course cells to window means.

    `assemble_long_df(..., effect_measure='cluster' | 'peak_t')` stores each
    trial's windowed time COURSE in `hg` (one object cell per trial) so the
    time-resolved measures can use it. The parametric ANOVA route in this
    section is defined on the window MEAN, so every quantity it reports has to
    be computed on that same reduction -- the fit in `_anova_interaction_stats`
    and the `<g>_sign` direction that accompanies each F/p alike. Reducing in
    only one of the two places is what made `per_electrode_anova_labels` raise
    from `_require_scalar_hg` on a cluster-mode table.
    """
    hg = pd.Series(hg)
    if hg.dtype == object:                   # cluster-mode time courses
        return hg.apply(np.nanmean).to_numpy(dtype=float)
    return hg.to_numpy(dtype=float)


def _anova_interaction_stats(elec_df, cond_col, mod_col, hg_col='hg'):
    """One electrode's two-way Type III ANOVA; return {'F', 'p'} for the
    interaction term (NaN, NaN on a singular fit / missing cell).

    Sum-code BOTH factors so the fit is a well-posed, equal-cell-weighted model
    over the deliberately unequal proportion cells (~75/25). Read Type III (the
    margin-respecting SS): the interaction row is orthogonal to both main
    effects, so a pure congruency/switch main effect does not inflate it. `hg`
    is the window-mean scalar; a time-course cell is reduced to its mean."""
    d = elec_df
    if d[hg_col].dtype == object:            # cluster-mode time courses -> window mean
        d = d.assign(**{hg_col: _scalar_hg(d[hg_col])})
    try:
        formula = f"{hg_col} ~ C({cond_col}, Sum) * C({mod_col}, Sum)"
        model = smf.ols(formula, data=d).fit()
        aov = anova_lm(model, typ=3)                       # Type III
        inter = f"C({cond_col}, Sum):C({mod_col}, Sum)"    # the interaction row
        return {'F': float(aov.loc[inter, 'F']),
                'p': float(aov.loc[inter, 'PR(>F)'])}
    except Exception:
        return {'F': np.nan, 'p': np.nan}                  # singular / missing cell


def _anova_simple_stats(elec_df, lab_col, hg_col='hg'):
    """One electrode's one-factor ANOVA for a simple two-level contrast."""
    d = elec_df
    if d[hg_col].dtype == object:
        d = d.assign(**{hg_col: _scalar_hg(d[hg_col])})
    try:
        sub = d.dropna(subset=[lab_col, hg_col])
        if sub[lab_col].nunique() < 2:
            return {'F': np.nan, 'p': np.nan}
        formula = f"{hg_col} ~ C({lab_col}, Sum)"
        model = smf.ols(formula, data=sub).fit()
        aov = anova_lm(model, typ=3)
        row = f"C({lab_col}, Sum)"
        return {'F': float(aov.loc[row, 'F']),
                'p': float(aov.loc[row, 'PR(>F)'])}
    except Exception:
        return {'F': np.nan, 'p': np.nan}


def per_electrode_anova_labels(df, alpha=0.05, contrast_mode='proportion',
                               contrasts=None, include_cross_controls=True,
                               fdr_correction='fdr_bh'):
    """Parametric per-electrode selectivity labels from the two-way interaction
    ANOVA -- ALL FOUR interactions, not just the two constructs of interest.
    Drop-in `labels` for `cmh_conjunction`.

    The four interaction groups are named `{condition}P{modulator}` -- each a
    balanced 2x2 difference-of-differences, FDR'd across electrodes, flagged at
    `alpha`:

    - `CPC` = congruency x proportion-congruent (incongruent_proportion) -- LWPC / stability
    - `SPS` = switchType x switch-proportion                             -- LWPS / flexibility
    - `CPS` = congruency x switch-proportion                             -- cross
    - `SPC` = switchType x proportion-congruent (incongruent_proportion) -- cross

    Why the two cross interactions (CPS, SPC) are *defined* groups, not just
    report-only p-values: the decoding battery (§4) decodes a 2x2 of {contrast} x
    {block modulator}, and every one of those four decode cells is the readout
    analog of one of these four interactions. To keep the decoding non-circular we
    must be able to name the electrode set each cell would double-dip on -- i.e.
    we need the CPS/SPC electrode groups, not only their p-values. In *univariate
    HG* the two cross interactions are still expected to be ~null, so their
    surviving counts double as the specificity control they always were
    (`include_cross_controls=False` drops them for the pure conjunction path).

    Columns per group `G` in {CPC, SPS, CPS, SPC}: `p_<g>`, `F_<g>`, `q_<g>`,
    `<g>_sign`, and the binary flag `G` (lowercase `<g>` in the effect columns).
    **Backward-compatible aliases** are also emitted so the conjunction / anatomy
    / brain-behavior stack keeps working unchanged: `S` = `CPC`, `F` = `SPS`, and
    the older effect columns `p_cong`/`q_cong`/`F_cong`/`s_sign` (= the CPC
    columns) and `p_switch`/`q_switch`/`F_switch`/`f_sign` (= the SPS columns).

    The ANOVA F is UNSIGNED, and the flag is deliberately DIRECTION-AGNOSTIC: an
    electrode is selected for an interaction whenever the (two-sided) interaction
    is significant, whether the condition effect GROWS or SHRINKS across the
    modulator's levels. This is on purpose -- in behavior the congruency effect
    shrinks in high-incongruent-proportion blocks (and the switch cost shrinks in
    high-switch-proportion blocks), but in a given neural population the sign of
    the block-proportion modulation is not known a priori, so we do not assume it.
    An electrode counts as an LWPC/LWPS electrode as long as it carries the
    interaction, regardless of direction.

    A signed direction is still recorded per electrode in `<g>_sign` (from the
    module's own equal-cell-weight estimator `_interaction_effect(..., 'cohens_d')`
    -- the same quantity the continuous §2 correlation uses) so the direction can
    be REPORTED downstream, but it is not used to include or exclude electrodes.

    This is the window-MEAN route throughout. A table built for a time-resolved
    measure (`assemble_long_df(..., effect_measure='cluster' | 'peak_t')`) holds
    per-trial time courses in `hg`; they are reduced to window means here, so
    passing one is fine and the F/p and `<g>_sign` still describe the same
    statistic. For a cluster-corrected, time-resolved definition use
    `power_traces_conjunction` instead.
    FDR (Benjamini-Hochberg) is applied across electrodes per interaction by
    default; each flag is set at `alpha`. Set `fdr_correction='none'` to copy
    raw p-values into the q-value columns and flag on raw p < alpha for
    exploratory threshold-sensitivity runs."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)               # attaches _scond/_smod/_fcond/_fmod
    if _is_interaction(contrasts['stability']) or _is_interaction(contrasts['flexibility']):
        # The FOUR interactions as (flag, condition_col, modulator_col, cond_sublabel,
        # mod_sublabel). The sub-label columns were attached by _canonical_labels:
        # _scond=congruency, _smod=incongruent_proportion, _fcond=switchType,
        # _fmod=switch_proportion -- so the two cross specs are just those recombined.
        specs = [('cpc', 'interaction', 'congruency', 'incongruent_proportion', '_scond', '_smod'),
                 ('sps', 'interaction', 'switchType', 'switch_proportion',       '_fcond', '_fmod')]
        if include_cross_controls:
            specs += [('cps', 'interaction', 'congruency', 'switch_proportion',      '_scond', '_fmod'),
                      ('spc', 'interaction', 'switchType', 'incongruent_proportion', '_fcond', '_smod')]
    else:
        # Main-effect reanalysis: keep the downstream CPC/SPS/S/F schema, but the
        # two flags now mean the simple stability/flexibility contrasts selected
        # by `contrast_mode='condition'` (congruency and switchType by default).
        specs = [('cpc', 'simple', None, None, '_slab', None),
                 ('sps', 'simple', None, None, '_flab', None)]
    recs = []
    for (subj, elec), g in work.groupby(['subject', 'electrode']):
        # Window means, matching the ANOVA fit below: a cluster-mode table holds
        # per-trial time courses, and the signed direction has to describe the
        # same statistic its F/p does (see `_scalar_hg`).
        hg = _scalar_hg(g['hg'])
        rec = dict(subject=subj, electrode=elec)
        for name, kind, cond_col, mod_col, cond_sub, mod_sub in specs:
            stats = (_anova_interaction_stats(g, cond_col, mod_col)   # Type III, sum-coded
                     if kind == 'interaction' else _anova_simple_stats(g, cond_sub))
            rec[f'p_{name}'] = stats['p']
            rec[f'F_{name}'] = stats['F']
            if kind == 'interaction':
                effect = _interaction_effect(     # signed d-o-d direction
                    hg, g[cond_sub].to_numpy(), g[mod_sub].to_numpy(), 'cohens_d', alpha)
            else:
                effect = _contrast_effect(g.assign(hg=hg), cond_sub, 'cohens_d', alpha)
            rec[f'{name}_sign'] = np.sign(effect)
        recs.append(rec)
    out = pd.DataFrame(recs)
    if fdr_correction not in ('fdr_bh', 'none'):
        raise ValueError("fdr_correction must be 'fdr_bh' or 'none'")
    for name, *_ in specs:
        if fdr_correction == 'fdr_bh':
            out[f'q_{name}'] = multipletests(out[f'p_{name}'].fillna(1), method='fdr_bh')[1]
        else:
            out[f'q_{name}'] = out[f'p_{name}']
        # Direction-agnostic: flag on the two-sided interaction significance only.
        # The `<g>_sign` column still records which way each electrode's effect
        # goes for downstream reporting, but the sign is NOT used to gate selection
        # -- the modulation direction is not assumed for any neural population.
        flag = out[f'q_{name}'] < alpha
        out[name.upper()] = flag.astype(int)
    # backward-compatible aliases (the stability/flexibility pair + old effect names)
    out['S'] = out['CPC']; out['F'] = out['SPS']
    out['p_cong'] = out['p_cpc']; out['q_cong'] = out['q_cpc']
    out['F_cong'] = out['F_cpc']; out['s_sign'] = out['cpc_sign']
    out['p_switch'] = out['p_sps']; out['q_switch'] = out['q_sps']
    out['F_switch'] = out['F_sps']; out['f_sign'] = out['sps_sign']
    return out


# A stratum with a zero MARGINAL (no S electrodes, no F electrodes, or all of
# them S / all of them F) carries no information about the S-F association: with
# a+b == 0 both `a*d` and `b*c` are zero, so it contributes nothing to either
# side of the Mantel-Haenszel ratio, and its CMH variance term is zero too.
# Dropping it is therefore a no-op on an UNSHIFTED analysis -- but it is not a
# no-op once `shift_zeros=True` is in play, which is why this exists.
_NullTest = namedtuple('_NullTest', 'statistic pvalue')


def _informative_stratum(a, b, c, e):
    """True when a subject's 2x2 can speak to the S-F association at all."""
    return (a + b) > 0 and (c + e) > 0 and (a + c) > 0 and (b + e) > 0


def cmh_conjunction(labels, drop_uninformative_strata=True):
    """Cochran-Mantel-Haenszel: is S-selectivity associated with F-selectivity,
    pooling over subject strata (each subject its own 2x2)?

    Uninformative strata are dropped before pooling, and this is load-bearing
    rather than cosmetic. `StratifiedTable(..., shift_zeros=True)` adds 0.5 to
    ALL FOUR cells of any stratum containing a zero -- the standard fix for a
    single empty cell in an otherwise informative table, but a fabricator of
    evidence when the whole stratum is empty. A subject with no S electrodes has
    the table `[[0, 0], [c, e]]`, which says nothing about whether S predicts F;
    shifted, it becomes `[[.5, .5], [c + .5, e + .5]]` and starts contributing a
    positive association to the pool. Two ways that bites:

      * A selection threshold at which NOTHING is selected gives every subject
        `[[0, 0], [0, n]]` -> `[[.5, .5], [.5, n + .5]]`, whose pooled OR is
        ~2n with a vanishing p. A threshold sweep run through the unguarded
        version reports its STRONGEST shared-core evidence exactly where it has
        no evidence at all (8 subjects x 25 electrodes: OR = 51, p = 4e-12).
      * Real runs are affected too, not just sweep endpoints. Adding four
        subjects with no S electrodes to four genuinely informative strata moves
        the pooled OR from 4.00 to 4.10 and the CMH p from 6.9e-4 to 1.6e-4 --
        entirely from subjects that carry no information.

    When NO stratum is informative the odds ratio is genuinely undefined, and
    that is reported as NaN rather than as a number: `mh_odds_ratio` is NaN and
    `cmh`/`homogeneity` carry NaN statistics. Callers should test
    `n_informative_strata` before reading the OR.

    Pass `drop_uninformative_strata=False` to reproduce the old (biased)
    behaviour for comparison against previously-recorded numbers.

    Adds to the returned dict: `n_strata`, `n_informative_strata`,
    `n_dropped_strata`, and an `informative` column on `per_subject`.
    """
    tables, per_subj, keep = [], [], []
    for subj, g in labels.dropna(subset=['S', 'F']).groupby('subject'):
        a = int(((g.S == 1) & (g.F == 1)).sum())   # both
        b = int(((g.S == 1) & (g.F == 0)).sum())   # stability only
        c = int(((g.S == 0) & (g.F == 1)).sum())   # flexibility only
        e = int(((g.S == 0) & (g.F == 0)).sum())   # neither
        ok = _informative_stratum(a, b, c, e)
        tables.append([[a, b], [c, e]])
        keep.append(ok)
        per_subj.append(dict(subject=subj, both=a, stab_only=b, flex_only=c,
                             neither=e, informative=ok))

    used = ([t for t, ok in zip(tables, keep) if ok]
            if drop_uninformative_strata else list(tables))
    res = dict(per_subject=pd.DataFrame(per_subj),
               n_strata=len(tables), n_informative_strata=int(sum(keep)),
               n_dropped_strata=int(len(tables) - sum(keep))
               if drop_uninformative_strata else 0)

    if not used:
        # No stratum can speak to the association. The OR is undefined; say so.
        res.update(mh_odds_ratio=np.nan,
                   cmh=_NullTest(np.nan, np.nan),
                   homogeneity=_NullTest(np.nan, np.nan),
                   summary=None)
    else:
        st = StratifiedTable(used, shift_zeros=True)
        res.update(mh_odds_ratio=st.oddsratio_pooled,
                   cmh=st.test_null_odds(),            # H0: common OR = 1
                   homogeneity=st.test_equal_odds(),   # H0: OR equal across subjects
                   summary=st.summary())
        try:
            res['or_95ci'] = st.oddsratio_pooled_confint()
        except Exception:
            pass

    # Descriptive only, and deliberately over ALL strata: this is the count
    # readout the reader checks against `n_both` / `n_stab_only` / ..., not an
    # input to the stratified inference above (it ignores the nesting entirely).
    pooled = np.sum(tables, axis=0) if tables else np.zeros((2, 2), int)
    res['pooled_table'] = pooled
    res['pooled_fisher_or'], res['pooled_fisher_p'] = fisher_exact(pooled)
    return res


# ----------------------------------------------------------------------------
# A2: overlap / conjunction inference -- permutation null + threshold sweep
# ----------------------------------------------------------------------------
def conjunction_permutation_null(labels, n_perm=10000, seed=0):
    """Empirical null for the count of 'both' (S==1 & F==1) electrodes.

    Shuffle F WITHIN each subject only, so every subject's S-count and F-count
    stay fixed and only the S/F *pairing* is randomized -- the exact null the
    CMH assumes, and the categorical analogue of the within-subject permutation
    used by `subject_clustered_corr`. A global shuffle would break the subject
    nesting and manufacture significance.

    Returns dict(observed:int, null:ndarray(n_perm), p_two_sided:float, z:float),
    the p two-sided against the null mean."""
    lab = labels.dropna(subset=['S', 'F']).copy()
    S = lab['S'].to_numpy().astype(int)
    F = lab['F'].to_numpy().astype(int)
    subj = lab['subject'].to_numpy()
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]   # per-subject rows
    observed = int(((S == 1) & (F == 1)).sum())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Fp = F.copy()
        for idx in groups:                 # shuffle F WITHIN each subject only
            Fp[idx] = F[rng.permutation(idx)]
        null[i] = int(((S == 1) & (Fp == 1)).sum())
    mean = null.mean()
    p = (np.sum(np.abs(null - mean) >= abs(observed - mean)) + 1) / (n_perm + 1)
    z = (observed - mean) / null.std() if null.std() > 0 else np.nan
    return dict(observed=observed, null=null, p_two_sided=float(p), z=float(z))


def conjunction_threshold_sweep(labels_by_threshold, thresholds):
    """Recompute the overlap odds ratio + counts across selection thresholds.

    A segregation (or shared-core) claim should be stable across thresholds, not
    an artifact of one alpha. `labels_by_threshold` is a callable
    threshold -> labels DataFrame (S/F recomputed at that cutoff); keeping it a
    callable makes the sweep agnostic to whether you threshold ANOVA q-values,
    permutation q-values, or effect-size percentiles. Returns a tidy DataFrame:
    threshold, n_S, n_F, n_both, mh_odds_ratio, cmh_p, n_informative_strata.

    The strict end of a sweep is where selection runs out, so `mh_odds_ratio` is
    NaN there by design (see `cmh_conjunction`): with no informative stratum the
    odds ratio is undefined, and a sweep is a claim about STABILITY across
    thresholds, which an undefined endpoint cannot support either way. Read
    `n_informative_strata` alongside every row -- a row resting on one or two
    subjects is not evidence that the verdict is threshold-stable, even when its
    OR is finite.
    """
    rows = []
    for t in thresholds:
        lab = labels_by_threshold(t)                 # fresh S/F at this threshold
        res = cmh_conjunction(lab)                   # reuse the existing CMH machinery
        rows.append(dict(threshold=t,
                         n_S=int(lab.S.sum()), n_F=int(lab.F.sum()),
                         n_both=int(((lab.S == 1) & (lab.F == 1)).sum()),
                         mh_odds_ratio=res['mh_odds_ratio'],
                         cmh_p=res['cmh'].pvalue,
                         n_informative_strata=res['n_informative_strata']))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------
def run_joint_distribution_analysis(df, responsiveness=None,
                                    n_splits=200, n_perm_corr=10000,
                                    n_perm_label=2000, alpha=0.05, min_elec=3,
                                    contrast_mode='condition', contrasts=None,
                                    effect_measure='cohens_d',
                                    fdr_correction='fdr_bh',
                                    corr_method='spearman',
                                    main_effects=False):
    """Continuous (A2) + categorical (A3) segregation tests on one trial table.

    Returned keys:
      correlation      PRIMARY. Split-resolved co-localization: the correlation
                       is taken WITHIN each disjoint half-split and averaged, so
                       x and y never share trials. Carries the split-half
                       reliabilities (`reliability_x`, `reliability_y`) and the
                       noise-corrected estimate, so a null result is
                       interpretable rather than ambiguous.
      correlation_split_averaged
                       DIAGNOSTIC. The old estimator: average x and y over
                       splits, then correlate once. Retained for comparison
                       only -- averaging first reinstates the same-trial bias
                       the split was meant to remove, so it typically sits
                       between `correlation` and `naive_sensitivities`.
      electrodes       per-electrode x, y (split-averaged), responsiveness, and
                       the S/F labels -- for the scatter plot.
      continuous       residualised/centred table behind the diagnostic.
      labels, conjunction  the categorical (A3) arm, unchanged.
      main_effect_correlation
                       with `main_effects=True`: `correlation` for congruency
                       vs switch type on the same halves. The main-effect maps
                       are far more reliable, so compare the two levels only
                       next to their reliabilities.
    """
    per_split = compute_sensitivities_per_split(
        df, n_splits, contrast_mode=contrast_mode, contrasts=contrasts,
        effect_measure=effect_measure, alpha=alpha, main_effects=main_effects)
    elec = add_responsiveness(average_over_splits(per_split), df, responsiveness)
    resp = elec.drop_duplicates('electrode').set_index('electrode')['resp']

    corr = split_resolved_corr(per_split, resp, min_elec=min_elec,
                               method=corr_method, n_perm=n_perm_corr)
    cont = prepare_continuous(elec, min_elec=min_elec)
    corr_avg = subject_clustered_corr(cont, method=corr_method, n_perm=n_perm_corr)

    labels = per_electrode_labels(df, n_perm=n_perm_label, alpha=alpha,
                                  contrast_mode=contrast_mode, contrasts=contrasts,
                                  effect_measure=effect_measure,
                                  fdr_correction=fdr_correction)
    conj = cmh_conjunction(labels)
    extra = {} if not main_effects else dict(main_effect_correlation=split_resolved_corr(
        main_effect_view(per_split), resp, min_elec=min_elec, method=corr_method,
        n_perm=n_perm_corr))
    return dict(electrodes=elec.merge(labels[['electrode', 'S', 'F']], on='electrode'),
                continuous=cont, correlation=corr,
                correlation_split_averaged=corr_avg,
                per_split=per_split, labels=labels, conjunction=conj,
                contrast_mode=contrast_mode, effect_measure=effect_measure, **extra)


# ----------------------------------------------------------------------------
# runnable smoke test with synthetic data
# ----------------------------------------------------------------------------
def _synthetic_df(effect_measure='cohens_d', n_time=20, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(12):                       # 12 subjects
        n_tr = rng.integers(300, 600)         # unequal trial counts
        cong = rng.choice(['c', 'i'], n_tr); sw = rng.choice(['s', 'r'], n_tr)
        inc_prop = rng.choice([25.0, 75.0], n_tr)
        sw_prop = rng.choice([25.0, 75.0], n_tr)
        for e in range(rng.integers(15, 40)):  # unequal electrode counts
            gain = rng.lognormal(0, 0.5)      # per-electrode SNR (gain confound)
            bx, by = rng.normal(0, .4), rng.normal(0, .4)   # true sensitivities
            # bx drives the congruency effect AND makes it depend on incongruent
            # proportion (a congruency x proportion INTERACTION = LWPC); likewise
            # by for switch x switch-proportion (LWPS). So condition mode recovers
            # bx via the main effect and proportion mode via the interaction.
            # bx/by are drawn from a ZERO-MEAN normal, so across electrodes the
            # interaction runs in BOTH directions (the condition effect is larger
            # in the high-proportion block for some electrodes, smaller for
            # others). That is deliberate: the electrode definition must recover
            # an interaction regardless of its sign, since the direction of the
            # block-proportion modulation is not assumed for any population.
            base = (bx * (cong == 'i') * (1.0 + (inc_prop == 75.0))
                    + by * (sw == 's') * (1.0 + (sw_prop == 75.0)))
            fr = dict(subject=s, electrode=f'{s}_{e}',
                      congruency=cong, switchType=sw,
                      incongruent_proportion=inc_prop, switch_proportion=sw_prop)
            frame = pd.DataFrame(fr)
            if effect_measure in ('cluster', 'peak_t'):   # both need time courses
                tc = rng.normal(0, 1, (n_tr, n_time)) * gain
                w = slice(n_time // 4, 3 * n_time // 4)
                tc[:, w] += (gain * base)[:, None]
                col = np.empty(n_tr, dtype=object)
                for i in range(n_tr):
                    col[i] = tc[i]
                frame['hg'] = col
            else:
                frame['hg'] = gain * base + rng.normal(0, 1, n_tr) * gain
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _report(tag, out, df=None, contrast_mode='condition', effect_measure='cohens_d'):
    c = out['correlation']
    print(f"[{tag}] split-resolved r={c['corr']:+.3f} p={c['p']:.4f} "
          f"(noise-corrected {c['corr_noise_corrected']:+.3f}; "
          f"rel_x={c['reliability_x']:.3f} rel_y={c['reliability_y']:.3f}; "
          f"n={c['n_electrodes']} elec, {c['n_electrodes_dropped']} dropped)")
    a = out['correlation_split_averaged']
    print(f"[{tag}] split-AVERAGED (biased diagnostic) r={a['corr']:+.3f} p={a['p']:.4f}")
    if df is not None:
        nv = add_responsiveness(
            naive_sensitivities(df, contrast_mode=contrast_mode,
                                effect_measure=effect_measure), df)
        nv = subject_clustered_corr(prepare_continuous(nv), n_perm=500)
        print(f"[{tag}] NAIVE (shared trials)          r={nv['corr']:+.3f} p={nv['p']:.4f}")
    print(f"[{tag}] MH OR:", out['conjunction']['mh_odds_ratio'],
          'CMH p:', out['conjunction']['cmh'].pvalue)


if __name__ == '__main__':
    # small/fast settings so the smoke test exercises every option path quickly;
    # bump n_splits / n_perm_* for a real run.
    for cm in ('condition', 'proportion'):
        df = _synthetic_df(effect_measure='cohens_d')
        out = run_joint_distribution_analysis(df, n_splits=20, n_perm_corr=800,
                                              n_perm_label=100, contrast_mode=cm)
        _report(f'{cm} / cohens_d', out, df, contrast_mode=cm)

    # cluster-mass effect measure on time-resolved HG (see USE_TIME_PERM_CLUSTER
    # to swap the deterministic mass for the real time_perm_cluster mask)
    dfc = _synthetic_df(effect_measure='cluster', n_time=16)
    outc = run_joint_distribution_analysis(dfc, n_splits=10, n_perm_corr=500,
                                           n_perm_label=50, effect_measure='cluster')
    _report('condition / cluster', outc)
    print(outc['conjunction']['pooled_table'])

    # peak_t: amplitude-only robustness complement to the cluster mass. Run it on
    # the SAME time-resolved data and compare the segregation verdict.
    outp = run_joint_distribution_analysis(dfc, n_splits=10, n_perm_corr=500,
                                           n_perm_label=50, effect_measure='peak_t')
    _report('condition / peak_t', outp)
