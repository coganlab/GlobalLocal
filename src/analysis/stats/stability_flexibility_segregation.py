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
    hg          : single-trial high-gamma, summarised over your analysis window
                  (e.g. mean HG in the task window, baseline-normalised)
    congruency  : 'c' or 'i'
    switchType  : 's' or 'r'
Optional `responsiveness`: dict/Series {electrode: value}. PREFER your
baseline-vs-signal time_perm_cluster statistic here. Falls back to mean|HG|.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable


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


def _stratified_half_split(sub, rng):
    """Split one electrode's trials into two disjoint halves, balanced on the
    congruency x switchType cells so neither half is confounded."""
    h1, h2 = [], []
    for _, cell in sub.groupby(['congruency', 'switchType']):
        idx = cell.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = len(idx) // 2
        h1.append(idx[:cut]); h2.append(idx[cut:])
    return np.concatenate(h1), np.concatenate(h2)


# ----------------------------------------------------------------------------
# (A) per-electrode sensitivities x (stability) and y (flexibility)
# ----------------------------------------------------------------------------
def compute_sensitivities(df, n_splits=200, seed=0):
    """x from the congruency contrast on one trial-half, y from the switch
    contrast on the DISJOINT half -> their sampling noise is independent.
    Averaged over many random disjoint splits for stability."""
    rng = np.random.default_rng(seed)
    rows = []
    for (subj, elec), sub in df.groupby(['subject', 'electrode']):
        sub = sub.reset_index(drop=True)
        xs, ys = [], []
        for _ in range(n_splits):
            h1, h2 = _stratified_half_split(sub, rng)
            hx, hy = (h1, h2) if rng.random() < 0.5 else (h2, h1)  # use data symmetrically
            gx, gy = sub.loc[hx], sub.loc[hy]
            xs.append(_cohens_d(gx.loc[gx.congruency == 'i', 'hg'],
                                gx.loc[gx.congruency == 'c', 'hg']))
            ys.append(_cohens_d(gy.loc[gy.switchType == 's', 'hg'],
                                gy.loc[gy.switchType == 'r', 'hg']))
        rows.append(dict(subject=subj, electrode=elec,
                         x=np.nanmean(xs), y=np.nanmean(ys)))
    return pd.DataFrame(rows)


def add_responsiveness(elec_df, df, responsiveness=None):
    """Overall task-drive per electrode (the gain confound). Prefer passing your
    baseline-vs-signal cluster statistic; else use mean|HG| as a proxy."""
    elec_df = elec_df.copy()
    if responsiveness is not None:
        r = pd.Series(responsiveness)
    else:
        r = df.groupby('electrode')['hg'].apply(lambda v: np.abs(v.mean()))
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
def per_electrode_labels(df, n_perm=2000, alpha=0.05, seed=2):
    """Binary S (stability-selective) and F (flexibility-selective) per electrode,
    from within-electrode permutation p-values, FDR-corrected across electrodes."""
    rng = np.random.default_rng(seed)
    recs = []
    for (subj, elec), sub in df.groupby(['subject', 'electrode']):
        hg = sub['hg'].to_numpy()
        cong = (sub['congruency'].to_numpy() == 'i').astype(int)
        sw = (sub['switchType'].to_numpy() == 's').astype(int)

        def perm_p(label):
            obs = _cohens_d(hg[label == 1], hg[label == 0])
            if np.isnan(obs):
                return np.nan
            cnt = sum(abs(_cohens_d(hg[lp == 1], hg[lp == 0])) >= abs(obs)
                      for lp in (rng.permutation(label) for _ in range(n_perm)))
            return (cnt + 1) / (n_perm + 1)

        recs.append(dict(subject=subj, electrode=elec,
                         p_cong=perm_p(cong), p_switch=perm_p(sw)))
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
                                    n_perm_label=2000, alpha=0.05, min_elec=3):
    elec = add_responsiveness(compute_sensitivities(df, n_splits), df, responsiveness)
    cont = prepare_continuous(elec, min_elec=min_elec)
    corr = subject_clustered_corr(cont, n_perm=n_perm_corr)
    labels = per_electrode_labels(df, n_perm=n_perm_label, alpha=alpha)
    conj = cmh_conjunction(labels)
    return dict(electrodes=elec.merge(labels[['electrode', 'S', 'F']], on='electrode'),
                continuous=cont, correlation=corr, labels=labels, conjunction=conj)


# ----------------------------------------------------------------------------
# runnable smoke test with synthetic data
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    rows = []
    for s in range(12):                       # 12 subjects
        n_tr = rng.integers(300, 600)         # unequal trial counts
        cong = rng.choice(['c', 'i'], n_tr); sw = rng.choice(['s', 'r'], n_tr)
        for e in range(rng.integers(15, 40)): # unequal electrode counts
            gain = rng.lognormal(0, 0.5)      # per-electrode SNR (gain confound)
            bx, by = rng.normal(0, .4), rng.normal(0, .4)   # true sensitivities
            hg = (gain * (bx * (cong == 'i') + by * (sw == 's'))
                  + rng.normal(0, 1, n_tr) * gain)
            rows.append(pd.DataFrame(dict(subject=s, electrode=f'{s}_{e}',
                                          hg=hg, congruency=cong, switchType=sw)))
    df = pd.concat(rows, ignore_index=True)
    out = run_joint_distribution_analysis(df, n_splits=50, n_perm_corr=2000,
                                          n_perm_label=500)
    print('partial corr:', out['correlation'])
    print('MH OR:', out['conjunction']['mh_odds_ratio'],
          'CMH p:', out['conjunction']['cmh'].pvalue)
    print(out['conjunction']['pooled_table'])
