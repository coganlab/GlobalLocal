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

Shared-noise inflation is corrected at BOTH sources:
  - shared TRIAL noise -> x and y estimated on DISJOINT trial halves
  - shared GAIN / SNR  -> x and y residualized on overall responsiveness
Subject nesting is handled by within-subject centering + within-subject
permutation (continuous) and by CMH stratification (categorical).

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
                       'proportion' -> stability = incongruent_proportion (high vs low),
                                       flexibility = switch_proportion (high vs low)
                       Or pass an explicit `contrasts` spec (see `resolve_contrasts`).
  * `effect_measure` : 'cohens_d'   -> standardized mean difference on window-mean HG [default]
                       'cluster'    -> aggregate time-permutation cluster statistic
                                       (signed cluster mass) on time-resolved HG.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, fisher_exact
from scipy.stats import t as _t_dist
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable


# ----------------------------------------------------------------------------
# contrast specification (condition vs proportion)
# ----------------------------------------------------------------------------
# A `contrasts` spec is a dict with 'stability' and 'flexibility' entries, each
# {'col': column, 'pos': positive group, 'neg': negative group}. `pos`/`neg` may
# be a category label ('i'), an explicit value (75.0), a collection, or the
# sentinels 'high'/'low' (resolved to the column's extreme values by
# `finalize_contrasts`). The effect is measured as pos - neg, so its sign
# encodes selectivity direction just like the original congruency/switch d's.
_CONTRAST_PRESETS = {
    'condition': {
        'stability':   dict(col='congruency', pos='i', neg='c'),
        'flexibility': dict(col='switchType', pos='s', neg='r'),
    },
    'proportion': {
        'stability':   dict(col='incongruent_proportion', pos='high', neg='low'),
        'flexibility': dict(col='switch_proportion',      pos='high', neg='low'),
    },
}


def _copy_contrasts(contrasts):
    return {k: dict(v) for k, v in contrasts.items()}


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


def finalize_contrasts(df, contrasts):
    """Fill in numeric 'high'/'low' thresholds from the WHOLE df so every
    electrode uses the same proportion split (an electrode may lack a level)."""
    for key in ('stability', 'flexibility'):
        spec = contrasts[key]
        if 'high' in (spec.get('pos'), spec.get('neg')) or \
           'low' in (spec.get('pos'), spec.get('neg')):
            col = spec['col']
            if col not in df.columns:
                raise KeyError(f"proportion contrast column '{col}' for {key} not in "
                               f"df columns {list(df.columns)}")
            num = pd.to_numeric(df[col], errors='coerce').to_numpy()
            finite = num[np.isfinite(num)]
            if finite.size == 0:
                raise ValueError(f"proportion column '{col}' has no numeric values")
            spec.setdefault('_hi', float(np.nanmax(finite)))
            spec.setdefault('_lo', float(np.nanmin(finite)))
    return contrasts


def _group_masks(values, spec):
    """Boolean (pos_mask, neg_mask) for a 1-D array of a contrast column.

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


def _canonical_labels(df, contrasts):
    """Attach '_slab'/'_flab' in {1 (pos), 0 (neg), NaN (excluded)} for the
    stability and flexibility contrasts, so downstream code is agnostic to
    whether the contrast is a condition or a proportion."""
    out = df.copy()
    for lab, key in (('_slab', 'stability'), ('_flab', 'flexibility')):
        spec = contrasts[key]
        col = spec['col']
        if col not in out.columns:
            raise KeyError(f"contrast column '{col}' for {key} not in df columns "
                           f"{list(out.columns)}")
        pmask, nmask = _group_masks(out[col].to_numpy(), spec)
        v = np.full(len(out), np.nan)
        v[np.asarray(nmask, bool)] = 0.0
        v[np.asarray(pmask, bool)] = 1.0   # pos wins any (unexpected) overlap
        out[lab] = v
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


def _stack(vals):
    """List/array of per-trial hg -> (n,) for scalars or (n, T) for time courses."""
    vals = list(vals)
    if not vals:
        return np.empty(0)
    if np.ndim(vals[0]) == 0:
        return np.asarray(vals, float)
    return np.vstack([np.asarray(v, float) for v in vals])


def _effect_from_arrays(hg, lab, effect_measure, alpha):
    """Effect between the label==1 and label==0 trials of one electrode."""
    pos = _stack(hg[lab == 1])
    neg = _stack(hg[lab == 0])
    if effect_measure == 'cluster':
        return _cluster_effect(pos, neg, alpha=alpha)
    if effect_measure == 'cohens_d':
        return _cohens_d(pos, neg)
    raise ValueError(f"effect_measure must be 'cohens_d' or 'cluster'; got {effect_measure!r}")


def _contrast_effect(frame, labcol, effect_measure, alpha):
    lab = frame[labcol].to_numpy()
    hg = frame['hg'].to_numpy()
    return _effect_from_arrays(hg, lab, effect_measure, alpha)


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
    contrast on the DISJOINT half -> their sampling noise is independent.
    Averaged over many random disjoint splits for stability.

    `contrast_mode`/`contrasts` pick which manipulations define stability vs
    flexibility; `effect_measure` picks how each contrast is quantified
    ('cohens_d' on window-mean HG, or 'cluster' aggregate statistic on time
    courses)."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)
    rng = np.random.default_rng(seed)
    rows = []
    for (subj, elec), sub in work.groupby(['subject', 'electrode']):
        sub = sub.reset_index(drop=True)
        xs, ys = [], []
        for _ in range(n_splits):
            h1, h2 = _stratified_half_split(sub, rng, strata_cols=('_slab', '_flab'))
            hx, hy = (h1, h2) if rng.random() < 0.5 else (h2, h1)  # use data symmetrically
            gx, gy = sub.loc[hx], sub.loc[hy]
            xs.append(_contrast_effect(gx, '_slab', effect_measure, alpha))
            ys.append(_contrast_effect(gy, '_flab', effect_measure, alpha))
        rows.append(dict(subject=subj, electrode=elec,
                         x=np.nanmean(xs), y=np.nanmean(ys)))
    return pd.DataFrame(rows)


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
                         x=_contrast_effect(g, '_slab', effect_measure, alpha),
                         y=_contrast_effect(g, '_flab', effect_measure, alpha)))
    return pd.DataFrame(rows)


def add_responsiveness(elec_df, df, responsiveness=None):
    """Overall task-drive per electrode (the gain confound). Prefer passing your
    baseline-vs-signal cluster statistic; else use mean|HG| as a proxy."""
    elec_df = elec_df.copy()
    if responsiveness is not None:
        r = pd.Series(responsiveness)
    else:
        def _mean_abs(v):
            arr = _stack(v.to_numpy())
            if arr.ndim == 1:
                return np.abs(np.nanmean(arr))        # scalar HG: |mean HG|
            return np.nanmean(np.abs(arr))            # time-resolved HG: mean |HG|
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
                         effect_measure='cohens_d'):
    """Binary S (stability-selective) and F (flexibility-selective) per electrode,
    from within-electrode permutation p-values, FDR-corrected across electrodes.

    `contrast_mode`/`contrasts`/`effect_measure` behave as in
    `compute_sensitivities`. Output columns keep the names `p_cong`/`q_cong`
    (stability) and `p_switch`/`q_switch` (flexibility) whatever the contrast."""
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    work = _canonical_labels(df, contrasts)
    rng = np.random.default_rng(seed)
    recs = []
    for (subj, elec), sub in work.groupby(['subject', 'electrode']):
        hg = sub['hg'].to_numpy()
        slab = sub['_slab'].to_numpy()
        flab = sub['_flab'].to_numpy()

        def perm_p(lab):
            valid = ~np.isnan(lab)
            l = lab[valid].astype(int)
            h = hg[valid]
            obs = _effect_from_arrays(h, l, effect_measure, alpha)
            if np.isnan(obs):
                return np.nan
            cnt = sum(abs(_effect_from_arrays(h, rng.permutation(l), effect_measure, alpha)) >= abs(obs)
                      for _ in range(n_perm))
            return (cnt + 1) / (n_perm + 1)

        recs.append(dict(subject=subj, electrode=elec,
                         p_cong=perm_p(slab), p_switch=perm_p(flab)))
    out = pd.DataFrame(recs)
    out['q_cong'] = multipletests(out['p_cong'].fillna(1), method='fdr_bh')[1]
    out['q_switch'] = multipletests(out['p_switch'].fillna(1), method='fdr_bh')[1]
    out['S'] = (out['q_cong'] < alpha).astype(int)
    out['F'] = (out['q_switch'] < alpha).astype(int)
    return out


def cmh_conjunction(labels):
    """Cochran-Mantel-Haenszel: is S-selectivity associated with F-selectivity,
    pooling over subject strata (each subject its own 2x2)?"""
    tables, per_subj = [], []
    for subj, g in labels.dropna(subset=['S', 'F']).groupby('subject'):
        a = int(((g.S == 1) & (g.F == 1)).sum())   # both
        b = int(((g.S == 1) & (g.F == 0)).sum())   # stability only
        c = int(((g.S == 0) & (g.F == 1)).sum())   # flexibility only
        e = int(((g.S == 0) & (g.F == 0)).sum())   # neither
        tables.append([[a, b], [c, e]])
        per_subj.append(dict(subject=subj, both=a, stab_only=b, flex_only=c, neither=e))

    st = StratifiedTable(tables, shift_zeros=True)
    res = dict(mh_odds_ratio=st.oddsratio_pooled,
               cmh=st.test_null_odds(),            # H0: common OR = 1
               homogeneity=st.test_equal_odds(),   # H0: OR equal across subjects
               per_subject=pd.DataFrame(per_subj),
               summary=st.summary())
    try:
        res['or_95ci'] = st.oddsratio_pooled_confint()
    except Exception:
        pass
    pooled = np.sum(tables, axis=0)                 # descriptive, ignores nesting
    res['pooled_table'] = pooled
    res['pooled_fisher_or'], res['pooled_fisher_p'] = fisher_exact(pooled)
    return res


# ----------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------
def run_joint_distribution_analysis(df, responsiveness=None,
                                    n_splits=200, n_perm_corr=10000,
                                    n_perm_label=2000, alpha=0.05, min_elec=3,
                                    contrast_mode='condition', contrasts=None,
                                    effect_measure='cohens_d'):
    elec = add_responsiveness(
        compute_sensitivities(df, n_splits, contrast_mode=contrast_mode,
                              contrasts=contrasts, effect_measure=effect_measure,
                              alpha=alpha),
        df, responsiveness)
    cont = prepare_continuous(elec, min_elec=min_elec)
    corr = subject_clustered_corr(cont, n_perm=n_perm_corr)
    labels = per_electrode_labels(df, n_perm=n_perm_label, alpha=alpha,
                                  contrast_mode=contrast_mode, contrasts=contrasts,
                                  effect_measure=effect_measure)
    conj = cmh_conjunction(labels)
    return dict(electrodes=elec.merge(labels[['electrode', 'S', 'F']], on='electrode'),
                continuous=cont, correlation=corr, labels=labels, conjunction=conj,
                contrast_mode=contrast_mode, effect_measure=effect_measure)


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
            # effect from trial condition AND its block proportion, so both
            # contrast_modes have recoverable signal
            base = (bx * (cong == 'i') + by * (sw == 's')
                    + 0.5 * bx * (inc_prop == 75.0) + 0.5 * by * (sw_prop == 75.0))
            fr = dict(subject=s, electrode=f'{s}_{e}',
                      congruency=cong, switchType=sw,
                      incongruent_proportion=inc_prop, switch_proportion=sw_prop)
            frame = pd.DataFrame(fr)
            if effect_measure == 'cluster':
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


if __name__ == '__main__':
    # small/fast settings so the smoke test exercises every option path quickly;
    # bump n_splits / n_perm_* for a real run.
    for cm in ('condition', 'proportion'):
        df = _synthetic_df(effect_measure='cohens_d')
        out = run_joint_distribution_analysis(df, n_splits=30, n_perm_corr=1000,
                                              n_perm_label=200, contrast_mode=cm)
        print(f"[{cm} / cohens_d] partial corr:", out['correlation'])
        print(f"[{cm} / cohens_d] MH OR:", out['conjunction']['mh_odds_ratio'],
              'CMH p:', out['conjunction']['cmh'].pvalue)

    # cluster-mass effect measure on time-resolved HG (see USE_TIME_PERM_CLUSTER
    # to swap the deterministic mass for the real time_perm_cluster mask)
    dfc = _synthetic_df(effect_measure='cluster', n_time=20)
    outc = run_joint_distribution_analysis(dfc, n_splits=15, n_perm_corr=1000,
                                           n_perm_label=100, effect_measure='cluster')
    print("[condition / cluster] partial corr:", outc['correlation'])
    print("[condition / cluster] MH OR:", outc['conjunction']['mh_odds_ratio'],
          'CMH p:', outc['conjunction']['cmh'].pvalue)
    print(outc['conjunction']['pooled_table'])
