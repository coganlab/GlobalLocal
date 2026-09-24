"""Reproduce every number in §15 of docs/n4_continuous_anatomy.md from two CSVs.

§15 mixes two kinds of result:

* pipeline results -- produced by functions in
  ``src/analysis/stats/stability_flexibility_anatomy.py`` exactly as the DCC job
  calls them (``relative_score_roi_test``, ``relative_score_coordinate_test``,
  ``map_reliability``, ``score_centers_per_subject``); and
* follow-up analyses that were run by hand on the job's outputs and are not
  part of the job: the cross-validated gradient (§15.5), every magnitude
  version of the gradient including the unbiased cross-validated mu^2 one
  (§15.7), the concordance subsets, the co-occurrence count (§15.4), the
  sign-defined centres (§15.8), and a simulation of what fitting the gradient
  on the positive-on-both electrodes alone does (section 9).

This script is the single place both live, so every §15 claim can be re-run
from the archived outputs without the cluster::

    python dcc_scripts/stats/n4_section15_followups.py \\
        --scores  <run>/scores_with_anatomy.csv \\
        --per-split <run>/per_split.csv \\
        [--n-perm 20000] [--sections 3,4,5,7,9]

It only reads the two CSVs; nothing touches epochs, atlases or recon files.

Scaling. ``per_split.csv`` holds raw half-trial scores. The half-trial maps are
put on the SAME scale as ``lwpc_s``/``lwps_s`` by dividing by the full-data
pooled SD read off ``scores_with_anatomy.csv`` (``lwpc_score / lwpc_s``), so a
half-trial slope is directly comparable with the full-data slope and their
average reproduces it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as stats_t

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402

COORDS = ('mni_y', 'mni_z', 'mni_x')
Z = COORDS.index('mni_z')


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(scores_csv, per_split_csv):
    """Scores table + per-split table rescaled to the ``*_s`` units, same electrode order."""
    s = pd.read_csv(scores_csv)
    ps = pd.read_csv(per_split_csv)
    sd_x = float(np.median(s['lwpc_score'] / s['lwpc_s']))
    sd_y = float(np.median(s['lwps_score'] / s['lwps_s']))
    for c in ('xA', 'xB'):
        ps[c] = ps[c] / sd_x
    for c in ('yA', 'yB'):
        ps[c] = ps[c] / sd_y
    ps = ps[ps['electrode'].isin(s['electrode'])]
    return s.reset_index(drop=True), ps


def half_matrices(s, ps):
    """``{'xA': (n_elec, n_splits), ...}`` aligned to the row order of ``s``."""
    out = {}
    for c in ('xA', 'xB', 'yA', 'yB'):
        w = ps.pivot(index='electrode', columns='split', values=c)
        out[c] = w.reindex(s['electrode']).to_numpy(float)
    return out


# ---------------------------------------------------------------------------
# the linear algebra shared by the follow-ups
# ---------------------------------------------------------------------------
def coord_projector(d, covariates=('resp',)):
    """``(P, R)``: slopes = ``P @ v``; ``R @ v`` residualises v on the nuisance design.

    The same Frisch-Waugh fit `sfa._coordinate_fit` uses -- nuisance = intercept
    + subject dummies + centred responsiveness -- so a slope from here equals the
    pipeline's slope for the same vector.
    """
    X, _ = sfa._nuisance_design(d, covariates=covariates)
    R = np.eye(len(d)) - X @ np.linalg.pinv(X)
    Zr = R @ d[list(COORDS)].to_numpy(float)
    return np.linalg.pinv(Zr) @ R, R


def within_subject_perms(subjects, n_perm, seed):
    """Index arrays that shuffle rows within subject (coordinates move, scores stay)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(subjects == g) for g in pd.unique(subjects)]
    for _ in range(n_perm):
        idx = np.arange(len(subjects))
        for g in groups:
            idx[g] = rng.permutation(g)
        yield idx


def coord_perm_p(d, values, n_perm, seed, stat=lambda sl: sl[Z]):
    """Two-sided p for a single score's z slope under a within-subject coordinate shuffle.

    The sign-flip swap null of `sfa._swap_null` is only valid for a PAIRED
    DIFFERENCE, where negating the value is the same as swapping the two effect
    labels. A single score has no partner to swap with, so its null has to move
    the coordinates instead.
    """
    P, _ = coord_projector(d)
    obs = stat(P @ values)
    subj = d['subject'].to_numpy()
    null = []
    for idx in within_subject_perms(subj, n_perm, seed):
        P_perm, _ = coord_projector(d.iloc[idx].assign(subject=d['subject'].to_numpy(),
                                                       resp=d['resp'].to_numpy()))
        null.append(stat(P_perm @ values))
    null = np.asarray(null)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


def swap_slope(d, value_col, n_perm, seed):
    """z slope and swap-null p of ``value_col`` via the pipeline's own coordinate test."""
    r = sfa.relative_score_coordinate_test(d, n_perm=n_perm, seed=seed,
                                           value_col=value_col, by_hemisphere=False)['all']
    row = r['slopes'].set_index('axis').loc['mni_z']
    return float(row['slope_per_mm']), float(row['p']), int(r['n_electrodes'])


def banner(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


# ---------------------------------------------------------------------------
# §15.3 reliability
# ---------------------------------------------------------------------------
def section_3(s, ps, H, args):
    banner('§15.3  split-half reliability and the noise ceiling  [pipeline: map_reliability]')
    rel = sfa.map_reliability(ps)
    print(f"Spearman: between {rel['between']:+.3f}  rel LWPC {rel['reliability_lwpc']:+.3f}"
          f"  rel LWPS {rel['reliability_lwps']:+.3f}"
          f"  -> Spearman ratio {rel['between'] / np.sqrt(rel['reliability_lwpc'] * rel['reliability_lwps']):+.3f}"
          "  (the old, out-of-range number)")
    print(f"Pearson : between {rel['between_pearson']:+.3f}  rel LWPC {rel['reliability_lwpc_pearson']:+.3f}"
          f"  rel LWPS {rel['reliability_lwps_pearson']:+.3f}"
          f"  -> corrected {rel['between_noise_corrected']:+.3f}"
          f"  split-bootstrap CI {tuple(round(v, 3) for v in rel['between_noise_corrected_ci'])}")
    rb = rel['reliability_lwpc_pearson'], rel['reliability_lwps_pearson']
    print("full-data reliability (Spearman-Brown of the half-data value, for reading the "
          f"averaged map only): LWPC {2 * rb[0] / (1 + rb[0]):.2f}  LWPS {2 * rb[1] / (1 + rb[1]):.2f}")

    # The split bootstrap resamples 200 re-partitions of the SAME trials and
    # electrodes, so it measures only which-half-went-where jitter. The
    # uncertainty that matters is over electrodes/subjects: resample subjects.
    rng = np.random.default_rng(0)
    xa, xb, ya, yb = (H[c] for c in ('xA', 'xB', 'yA', 'yB'))

    def corrected(rows):
        def c(a, b):
            a = a - a.mean(0); b = b - b.mean(0)
            return np.mean((a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0)))
        btw = 0.5 * (c(xa[rows], yb[rows]) + c(xb[rows], ya[rows]))
        rx, ry = c(xa[rows], xb[rows]), c(ya[rows], yb[rows])
        return btw / np.sqrt(rx * ry) if rx > 0 and ry > 0 else np.nan

    subj = s['subject'].to_numpy()
    groups = [np.flatnonzero(subj == g) for g in pd.unique(subj)]
    boot = [corrected(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(args.n_boot)]
    boot = np.asarray(boot)
    print(f"SUBJECT-bootstrap of the corrected value ({args.n_boot} resamples): "
          f"95% [{np.nanpercentile(boot, 2.5):+.2f}, {np.nanpercentile(boot, 97.5):+.2f}], "
          f"{np.mean(~np.isfinite(boot)):.1%} undefined")

    sign_x = np.mean(np.sign(xa) == np.sign(xb))
    sign_y = np.mean(np.sign(ya) == np.sign(yb))
    print(f"sign agreement between disjoint halves: LWPC {sign_x:.3f}  LWPS {sign_y:.3f}  (coin flip = 0.500)")


# ---------------------------------------------------------------------------
# §15.4 shared population
# ---------------------------------------------------------------------------
def section_4(s, ps, H, args):
    banner('§15.4  do the two effects share electrodes?  [follow-up]')
    for c in ('lwpc_s', 'lwps_s'):
        print(f"{c}: mean {s[c].mean():+.3f}  median {s[c].median():+.3f}  "
              f"fraction<0 {np.mean(s[c] < 0):.1%} ({int(np.sum(s[c] < 0))}/{len(s)})")

    both = int(np.sum((s['lwpc_s'] > 0) & (s['lwps_s'] > 0)))
    subj = s['subject'].to_numpy()
    pos_x = (s['lwpc_s'] > 0).to_numpy()
    pos_y = (s['lwps_s'] > 0).to_numpy()
    null = np.array([np.sum(pos_x & pos_y[idx])
                     for idx in within_subject_perms(subj, args.n_perm, args.seed)])
    print(f"positive on both: {both}; within-subject shuffle null {null.mean():.1f} ± {null.std():.1f}; "
          f"one-sided p = {(np.sum(null >= both) + 1) / (len(null) + 1):.4f}; "
          f"independence predicts {len(s) * pos_x.mean() * pos_y.mean():.0f}")

    def within(a, b):
        g = s['subject'].to_numpy()
        a = a - pd.Series(a).groupby(g).transform('mean').to_numpy()
        b = b - pd.Series(b).groupby(g).transform('mean').to_numpy()
        return pearsonr(a, b)[0]

    print('per-split correlation, averaged over splits   pooled r   within-subject r')
    for lab, a, b in (('xA vs yA (shares trials)', 'xA', 'yA'), ('xB vs yB (shares trials)', 'xB', 'yB'),
                      ('xA vs yB (disjoint)', 'xA', 'yB'), ('xB vs yA (disjoint)', 'xB', 'yA')):
        pooled = np.mean([pearsonr(H[a][:, k], H[b][:, k])[0] for k in range(H[a].shape[1])])
        wth = np.mean([within(H[a][:, k], H[b][:, k]) for k in range(H[a].shape[1])])
        print(f"  {lab:<28} {pooled:+.3f}     {wth:+.3f}")

    # Averaging each half over the splits rebuilds the full-data map (every trial
    # lands in half A about half the time), so mean(xA) vs mean(yB) is NOT
    # disjoint -- it is the full-data, shared-trial correlation. Kept only to
    # label that number correctly.
    xa_bar, yb_bar = H['xA'].mean(1), H['yB'].mean(1)
    xb_bar, ya_bar = H['xB'].mean(1), H['yA'].mean(1)
    r_dis = 0.5 * (within(xa_bar, yb_bar) + within(xb_bar, ya_bar))
    print(f"split-averaged 'disjoint' within-subject r = {r_dis:+.3f} -- SHARES TRIALS: "
          f"corr(mean xA, mean xB) = {pearsonr(H['xA'].mean(1), H['xB'].mean(1))[0]:.3f}, "
          "so each averaged half is the full-data map")
    print(f"thresholded maps: expected overlap under independence = "
          f"{pos_x.mean() * pos_y.mean():.0%} of electrodes")

    # Controlling for responsiveness as well as subject. `split_resolved_corr`
    # is the pipeline's version (summary.txt §5.1 sweep): disjoint halves,
    # residualised on resp, centred within subject.
    for c in ('lwpc_s', 'lwps_s'):
        print(f"corr({c}, resp) within subject {within(s[c].to_numpy(float), s['resp'].to_numpy(float)):+.3f}")
    from src.analysis.stats import stability_flexibility_segregation as sfs
    raw = pd.read_csv(args.per_split)
    resp = s.set_index('electrode')['resp']
    for m in ('spearman', 'pearson'):
        r = sfs.split_resolved_corr(raw, resp, min_elec=1, method=m, n_perm=args.n_perm_cv, seed=args.seed)
        print(f"[pipeline: split_resolved_corr] subject + resp, disjoint halves ({m}): r {r['corr']:+.3f}  "
              f"p {r['p']:.4f}  reliabilities LWPC {r['reliability_x']:+.3f} / LWPS {r['reliability_y']:+.3f}")
    print("  NB: a cross-map r above sqrt(rel_x * rel_y) is impossible for independent halves. "
          "compute_sensitivities_per_split draws a new split per ELECTRODE, so within a "
          "subject electrode i's half A overlaps electrode j's half B; trial noise shared "
          "across electrodes then biases within-subject reliabilities down. The cross-effect "
          "r is barely affected (different contrasts on the same trials).")


# ---------------------------------------------------------------------------
# §15.5 dorsoventral gradient + cross-validation
# ---------------------------------------------------------------------------
def section_5(s, ps, H, args):
    banner('§15.5  dorsoventral gradient in delta  [pipeline: relative_score_coordinate_test]')
    res = sfa.relative_score_coordinate_test(s, n_perm=args.n_perm, seed=args.seed)
    for k, r in res.items():
        sl = r['slopes'].set_index('axis')
        print(f"[{k}] block F {r['observed_stat']:.3f} p {r['p']:.4f} n {r['n_electrodes']}  "
              + '  '.join(f"{a} {sl.loc[a, 'slope_per_mm']:+.5f} (p {sl.loc[a, 'p']:.4f})" for a in COORDS))

    banner('§15.5  cross-validated slope: refit on each disjoint half of every split  [follow-up]')
    P, _ = coord_projector(s)
    dA = H['xA'] - H['yA']                      # delta on half A, every split
    dB = H['xB'] - H['yB']
    sA, sB = (P @ dA)[Z], (P @ dB)[Z]           # one z slope per split
    prod = float(np.mean(sA * sB))
    full = float((P @ s['delta'].to_numpy(float))[Z])
    print(f"mean z slope  A {sA.mean():+.5f}   B {sB.mean():+.5f}   full {full:+.5f}")
    print(f"sign agreement between halves {np.mean(np.sign(sA) == np.sign(sB)):.1%}  "
          f"(both negative {np.mean((sA < 0) & (sB < 0)):.1%})")
    print(f"E[slope_A * slope_B] = {prod:+.3e}  (unbiased for slope^2: A and B share no trials)")
    if prod > 0:
        print(f"sqrt = {np.sqrt(prod):.4f} vs |observed| {abs(full):.4f} -> "
              f"{np.sqrt(prod) / abs(full):.0%} of the observed slope is signal")

    # null: shuffle coordinates within subject, refit both halves of every split
    subj = s['subject'].to_numpy()
    null = []
    for idx in within_subject_perms(subj, args.n_perm_cv, args.seed):
        Pp, _ = coord_projector(s.iloc[idx].assign(subject=subj, resp=s['resp'].to_numpy()))
        null.append(np.mean((Pp @ dA)[Z] * (Pp @ dB)[Z]))
    null = np.asarray(null)
    print(f"within-subject coordinate-shuffle null on E[slope_A*slope_B]: "
          f"p = {(np.sum(null >= prod) + 1) / (len(null) + 1):.4f} ({args.n_perm_cv} shuffles)")

    banner('§15.5  robustness  [follow-up]')
    s2 = s.assign(is_rh=(s['hemi'] == 'rh').astype(float))
    r = sfa._coordinate_fit(s2, 'delta', COORDS, ('resp', 'is_rh'), args.n_perm, args.seed)
    z = r['slopes'].set_index('axis').loc['mni_z']
    print(f"+ hemisphere dummy: z {z['slope_per_mm']:+.5f} p {z['p']:.4f}")
    s3 = s2.assign(z_rh=s2['mni_z'] * s2['is_rh'])
    r = sfa._coordinate_fit(s3, 'delta', (*COORDS, 'z_rh'), ('resp', 'is_rh'), args.n_perm, args.seed)
    zi = r['slopes'].set_index('axis').loc['z_rh']
    print(f"z x hemisphere interaction: coef {zi['slope_per_mm']:+.5f} p {zi['p']:.4f}")
    r = sfa._coordinate_fit(s, 'delta', COORDS, (), args.n_perm, args.seed)
    z = r['slopes'].set_index('axis').loc['mni_z']
    print(f"drop resp covariate: z {z['slope_per_mm']:+.5f} p {z['p']:.4f}")
    loso = sfa.leave_one_subject_out(
        lambda t: {'p': float(sfa._coordinate_fit(t, 'delta', COORDS, ('resp',), max(1000, args.n_perm // 10),
                                                  args.seed)['slopes'].set_index('axis').loc['mni_z', 'p'])},
        s, keys=('p',))
    folds = loso[loso['dropped'] != '(none)']
    worst = folds.loc[folds['p'].idxmax()]
    print(f"leave-one-subject-out: z p < .05 in {int(np.sum(folds['p'] < .05))}/{len(folds)} folds "
          f"(worst: drop {worst['dropped']} -> p {worst['p']:.3f})")

    banner('§15.5  parcel-level omnibus  [pipeline: relative_score_roi_test]')
    cov = sfa.build_coverage_matrix(s, roi_col='anat')
    r = sfa.relative_score_roi_test(s, cov, n_perm=args.n_perm, seed=args.seed, roi_col='anat')
    print(f"F {r['observed_stat']:.3f}  p {r['p']:.4f}  ({r['n_electrodes']} electrodes)")
    print(r['per_roi'].sort_values('p').head(4).to_string(index=False))


# ---------------------------------------------------------------------------
# §15.6 anterior-posterior
# ---------------------------------------------------------------------------
def section_6(s, ps, H, args):
    banner('§15.6  anterior-posterior  [pipeline: relative_score_coordinate_test]')
    res = sfa.relative_score_coordinate_test(s, n_perm=args.n_perm, seed=args.seed)
    print('  '.join(f"{k}: y p {r['slopes'].set_index('axis').loc['mni_y', 'p']:.2f}"
                    for k, r in res.items()))


# ---------------------------------------------------------------------------
# §15.7 magnitude versions
# ---------------------------------------------------------------------------
def section_7(s, ps, H, args):
    banner('§15.7  every magnitude version of the gradient  [follow-up]')
    d = s.copy()
    d['abs_diff'] = d['abs_lwpc'] - d['abs_lwps']
    # unbiased magnitude: for two independent estimates of the same effect,
    # E[x1 * x2] = mu^2 -- no rectification floor, unlike E|x|.
    d['mu2_lwpc'] = np.mean(H['xA'] * H['xB'], axis=1)
    d['mu2_lwps'] = np.mean(H['yA'] * H['yB'], axis=1)
    d['mu2_diff'] = d['mu2_lwpc'] - d['mu2_lwps']

    print(f"{'value':<46}{'z slope':>10}{'z p':>9}  null")
    for lab, col in (('delta = lwpc_s - lwps_s', 'delta'),
                     ('abs_lwpc - abs_lwps', 'abs_diff'),
                     ('cross-validated mu^2 difference', 'mu2_diff')):
        b, p, _ = swap_slope(d, col, args.n_perm, args.seed)
        print(f"{lab:<46}{b:>+10.4f}{p:>9.4f}  sign-flip swap")
    for col in ('lwpc_s', 'lwps_s'):
        b, p = coord_perm_p(d, d[col].to_numpy(float), args.n_perm_cv, args.seed)
        print(f"{col + ' alone':<46}{b:>+10.4f}{p:>9.4f}  within-subject coordinate shuffle")
    pp = d[(d['lwpc_s'] > 0) & (d['lwps_s'] > 0)]
    b, p, n = swap_slope(pp, 'delta', args.n_perm, args.seed)
    print(f"{f'delta, both-positive electrodes only (n={n})':<46}{b:>+10.4f}{p:>9.4f}  sign-flip swap")

    print(f"\nmean mu^2: LWPC {d['mu2_lwpc'].mean():+.3f}  LWPS {d['mu2_lwps'].mean():+.3f}")
    r = sfa.relative_score_coordinate_test(d, n_perm=args.n_perm, seed=args.seed, value_col='mu2_diff',
                                           by_hemisphere=False)['all']
    print(f"mu^2 difference block F p {r['p']:.3f}")
    print(f"corr(mu2_lwpc, z) {pearsonr(d['mu2_lwpc'], d['mni_z'])[0]:+.3f}   "
          f"corr(mu2_lwps, z) {pearsonr(d['mu2_lwps'], d['mni_z'])[0]:+.3f}")
    null_abs = np.sqrt(2 / np.pi)
    print(f"noise floor of |score| for a null electrode: E|N(0, s)| = {null_abs:.2f} s")

    banner('§15.7  is delta just sign disagreement? concordance subsets  [follow-up]')
    same = np.sign(d['lwpc_s']) == np.sign(d['lwps_s'])
    subsets = {'all': d, 'concordant (++ or --)': d[same], 'discordant (+- or -+)': d[~same],
               '++ only': d[(d['lwpc_s'] > 0) & (d['lwps_s'] > 0)],
               '-- only': d[(d['lwpc_s'] < 0) & (d['lwps_s'] < 0)]}
    for lab, sub in subsets.items():
        b, p, n = swap_slope(sub, 'delta', args.n_perm, args.seed)
        print(f"  {lab:<24} n {n:>4}   delta z slope {b:+.4f}  p {p:.4f}")
    b, p, _ = swap_slope(d[same], 'abs_diff', args.n_perm, args.seed)
    print(f"  magnitude contrast within concordant: {b:+.4f}  p {p:.4f}")

    banner('§15.7  what moves with z: the sign of each score, or their order?  [follow-up]')
    # within-subject: both the indicator and z centred on their subject mean, so
    # a subject with dorsal coverage and many LWPS-dominant sites cannot drive it
    g = d['subject'].to_numpy()

    def centre(v):
        v = pd.Series(np.asarray(v, float))
        return (v - v.groupby(g).transform('mean')).to_numpy()

    zc = centre(d['mni_z'])
    for lab, v in (('P(lwpc_s > 0)', d['lwpc_s'] > 0), ('P(lwps_s > 0)', d['lwps_s'] > 0),
                   ('P(delta > 0)', d['delta'] > 0)):
        vc = centre(v)
        r = pearsonr(vc, zc)[0]
        null = np.array([pearsonr(vc, zc[idx])[0] for idx in within_subject_perms(g, args.n_perm_cv, args.seed)])
        p = (np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)
        print(f"  {lab:<16} vs z   within-subject r {r:+.3f}  p {p:.4f} (coordinate shuffle)")


# ---------------------------------------------------------------------------
# §15.8 centres
# ---------------------------------------------------------------------------
def section_8(s, ps, H, args):
    banner('§15.8  weighted centres  [pipeline: score_centers_per_subject]')
    for lab, kw, dd in (('abs, medoid=True', dict(medoid=True), s),
                        ('abs, medoid=False', dict(medoid=False), s),
                        ('positive-clipped, medoid=True', dict(medoid=True, weight_cols=('pos_lwpc', 'pos_lwps')),
                         s.assign(pos_lwpc=s['lwpc_s'].clip(lower=0), pos_lwps=s['lwps_s'].clip(lower=0)))):
        c = sfa.score_centers_per_subject(dd, n_perm=args.n_perm, seed=args.seed, **kw)
        tab = c['per_group']
        print(f"  {lab:<32}" + '  '.join(f"{a} {tab[a].mean():+.2f} (p {c['p'][a]:.2f})" for a in ('dx', 'dy', 'dz'))
              + f"   zero-displacement groups {int(np.sum(tab['distance'] == 0))}/{len(tab)}")

    banner('§15.8  sign-defined centres (descriptive only -- circular as a test)  [follow-up]')
    for lab, sub in (('pooled', s), ('lh', s[s['hemi'] == 'lh']), ('rh', s[s['hemi'] == 'rh'])):
        pc, ps_ = sub[sub['delta'] > 0], sub[sub['delta'] < 0]
        print(f"  {lab:<7} LWPC-dominant n {len(pc):>3} z {pc['mni_z'].mean():5.1f}   "
              f"LWPS-dominant n {len(ps_):>3} z {ps_['mni_z'].mean():5.1f}   dz {pc['mni_z'].mean() - ps_['mni_z'].mean():+.1f}")
    rows = []
    for _, g in s.groupby(['subject', 'hemi']):
        pc, ps_ = g[g['delta'] > 0], g[g['delta'] < 0]
        if len(pc) >= 2 and len(ps_) >= 2:
            rows.append(pc['mni_z'].mean() - ps_['mni_z'].mean())
    rows = np.asarray(rows)
    print(f"  within subject x hemisphere ({len(rows)} groups): mean dz {rows.mean():+.2f} mm, "
          f"{int(np.sum(rows < 0))}/{len(rows)} in the expected direction")


# ---------------------------------------------------------------------------
# why not select the positive electrodes first?
# ---------------------------------------------------------------------------
def section_9(s, ps, H, args):
    banner('why not fit the gradient on the ++ electrodes only?  [simulation on the real layout]')
    # Plant the observed single-score slopes into maps with the observed
    # full-data reliability (~0.30) and an overlapping component (so the maps
    # correlate), on the real electrodes, subjects and coordinates; then compare
    # fitting all electrodes with fitting only those observed positive on both.
    rel, gx, gy = 0.30, -0.0037, +0.0040
    zc = s['mni_z'].to_numpy(float) - s['mni_z'].mean()
    n = len(s)
    rng = np.random.default_rng(args.seed)
    cache = {}

    def z_fit(idx, v):                         # parametric t for speed; slope = pipeline slope
        key = idx.tobytes()
        if key not in cache:
            d = s.iloc[idx]
            X, _ = sfa._nuisance_design(d)
            R = np.eye(len(d)) - X @ np.linalg.pinv(X)
            Zr = R @ d[list(COORDS)].to_numpy(float)
            cache.clear()
            cache[key] = (R, Zr, np.linalg.pinv(Zr), np.linalg.inv(Zr.T @ Zr)[Z, Z], len(d) - X.shape[1] - 3)
        R, Zr, Zp, vzz, dfe = cache[key]
        vr = R @ v
        b = Zp @ vr
        se = np.sqrt(np.sum((vr - Zr @ b) ** 2) / dfe * vzz)
        return b[Z], 2 * stats_t.sf(abs(b[Z] / se), dfe)

    for label, sx, sy in (('planted gradient', gx, gy), ('no gradient', 0.0, 0.0)):
        res = {k: [] for k in ('all electrodes', '++ selected on the same data',
                               '++ selected on half A, fitted on half B')}
        purity = []
        for _ in range(args.n_sim):
            u = rng.normal(0, np.sqrt(rel * 0.6), n)
            tx = 0.03 + sx * zc + u + rng.normal(0, np.sqrt(rel * 0.4), n)
            ty = 0.18 + sy * zc + u + rng.normal(0, np.sqrt(rel * 0.4), n)
            hx = [tx + rng.normal(0, np.sqrt(2 * (1 - rel)), n) for _ in range(2)]
            hy = [ty + rng.normal(0, np.sqrt(2 * (1 - rel)), n) for _ in range(2)]
            ox, oy = (hx[0] + hx[1]) / 2, (hy[0] + hy[1]) / 2
            res['all electrodes'].append(z_fit(np.arange(n), ox - oy))
            sel = np.flatnonzero((ox > 0) & (oy > 0))
            purity.append(np.mean((tx[sel] > 0) & (ty[sel] > 0)))
            res['++ selected on the same data'].append(z_fit(sel, (ox - oy)[sel]))
            sel_a = np.flatnonzero((hx[0] > 0) & (hy[0] > 0))
            res['++ selected on half A, fitted on half B'].append(z_fit(sel_a, (hx[1] - hy[1])[sel_a]))
        print(f"{label} (true delta slope {sx - sy:+.4f}); "
              f"{np.mean(purity):.0%} of observed-++ electrodes are truly ++")
        for k, v in res.items():
            v = np.asarray(v)
            print(f"  {k:<42} mean slope {v[:, 0].mean():+.4f}   P(p < .05) {np.mean(v[:, 1] < .05):.2f}")


SECTIONS = {3: section_3, 4: section_4, 5: section_5, 6: section_6, 7: section_7, 8: section_8,
            9: section_9}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', required=True, help='scores_with_anatomy.csv from the N4 job')
    ap.add_argument('--per-split', required=True, help='per_split.csv from the segregation job')
    ap.add_argument('--n-perm', type=int, default=20000, help='sign-flip swap permutations')
    ap.add_argument('--n-perm-cv', type=int, default=2000,
                    help='coordinate shuffles (each refits every split, so slower)')
    ap.add_argument('--n-boot', type=int, default=2000, help='subject bootstrap resamples')
    ap.add_argument('--n-sim', type=int, default=500, help='simulations for the ++ selection check')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sections', default='3,4,5,6,7,8,9')
    args = ap.parse_args(argv)

    s, ps = load(args.scores, args.per_split)
    H = half_matrices(s, ps)
    print(f"{len(s)} electrodes, {s['subject'].nunique()} subjects, {ps['split'].nunique()} splits")
    for k in (int(x) for x in args.sections.split(',')):
        SECTIONS[k](s, ps, H, args)


if __name__ == '__main__':
    main()
