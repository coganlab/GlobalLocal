"""Reproduce the follow-up numbers in §16.6 of docs/n4_continuous_anatomy.md.

The anatomy job prints the main-effect results itself: the dm label test, its
leave-one-subject-out sweep, the dm coordinate test, and Tests 1 and 2
(``summary.txt``, ``dm_per_roi.csv``, ``dm_roi_loso.csv``, ``dm_coordinates.csv``,
``delta_tracking.csv``, ``tilt_with_dm.csv``). This script adds what the job does
not compute, from the same run's ``scores_with_anatomy.csv``:

1. height against distance from the midline: how tangled the two are, and
   delta and dm fitted on each (swap null);
2. each single score (congruency, switch, LWPC, LWPS) on the pipeline's
   coordinates and on y + z + |x| (within-participant coordinate shuffle);
3. adjusted Cohen's d per band of distance from the midline and of height
   (figure panel c);
4. the Destrieux-label means of dm against those of delta (figure panel b);
5. Test 2 on both axes, a participant bootstrap of its shrinkage, and the scale
   the shrinkage is read on: dm's split-half reliability, from the segregation
   run's ``correlation_main_effects.json``;
6. Figure 5 (``paper_draft.md`` §1.4): a, the overlap at both levels; b, each
   adaptation against its own and the other base effect. The anatomy job
   already draws it (``fig5.png``); this redraws it from the job's outputs,
   e.g. after a style change in ``sfa.plot_figure5``. Needs ``--seg-dir`` and
   ``--out-dir``.

    python dcc_scripts/stats/n4_section16_followups.py \\
        --scores     <anatomy run>/continuous/scores_with_anatomy.csv \\
        [--seg-dir   <segregation run>] \\
        [--main-json <segregation run>/correlation_main_effects.json] \\
        [--tilt      <anatomy run>/continuous/tilt_with_dm.csv] \\
        [--out-dir   <where to write the figure-panel tables>]

``--seg-dir`` is the ``_main_effects`` segregation run the anatomy job read.
Section 6 takes panel a's r, p and n from its ``correlation.json`` and
``correlation_main_effects.json`` (the latter is also section 5's default
``--main-json``), and ``delta_tracking.csv`` from beside ``--scores``
(``--tracking`` to override).

It only reads those files; nothing touches epochs, atlases or recon files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402
from dcc_scripts.stats import n4_section15_followups as fu  # noqa: E402

PIPELINE = ('mni_y', 'mni_z', 'mni_x')   # what the anatomy job fits
MIDLINE = ('mni_y', 'mni_z', 'abs_x')    # the same, with distance from the midline for signed x
SCORES = (('congruency', 'cong_s', 'cong_score'), ('switch', 'switch_s', 'switch_score'),
          ('LWPC', 'lwpc_s', 'lwpc_score'), ('LWPS', 'lwps_s', 'lwps_score'))


def load(path):
    s = pd.read_csv(path)
    missing = sorted({'dm', 'cong_s', 'switch_s'} - set(s.columns))
    if missing:
        raise SystemExit(f"{path} has no main-effect columns {missing}: run the anatomy job "
                         "on a MAIN_EFFECTS=1 segregation run (§16.2)")
    s = s.dropna(subset=list(PIPELINE)).reset_index(drop=True)
    s['abs_x'] = s['mni_x'].abs()
    return s


def fmt(v):
    return f'{v:.4g}'


def swap_fit(d, value, coords, covariates=('resp',), n_perm=10000, seed=0):
    """``relative_score_coordinate_test``'s fit on any coordinates: (slopes, block F, block p)."""
    r = sfa._coordinate_fit(d, value, coords, covariates, n_perm, seed)
    return r['slopes'].set_index('axis'), r['observed_stat'], r['p']


def within_participant_r(d, a, b):
    """Pearson r of two columns once participant and responsiveness are removed."""
    _, R = fu.coord_projector(d)
    return pearsonr(R @ d[a].to_numpy(float), R @ d[b].to_numpy(float))[0]


def adjusted(d, col):
    """Cohen's d with participant and responsiveness offsets removed, mean added back (§15.7)."""
    X, _ = sfa._nuisance_design(d)
    v = d[col].to_numpy(float)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0] + v.mean()


# ---------------------------------------------------------------------------
# 1. height vs distance from the midline
# ---------------------------------------------------------------------------
def section_1(s, args):
    fu.banner('1. height vs distance from the midline  [follow-up; swap null]')
    print(f"within-participant r(z, |x|) = {within_participant_r(s, 'mni_z', 'abs_x'):+.3f}"
          "   (dorsal lPFC sits near the midline)")
    rows = []
    for value in ('delta', 'dm'):
        for model, coords in (('y + z + x (pipeline)', PIPELINE), ('y + z + |x|', MIDLINE),
                              ('z alone', ('mni_z',)), ('|x| alone', ('abs_x',))):
            sl, F, p = swap_fit(s, value, coords, n_perm=args.n_perm, seed=args.seed)
            row = dict(value=value, model=model, block_F=F, block_p=p)
            for ax, name in (('mni_z', 'z'), ('abs_x', '|x|')):
                if ax in sl.index:
                    row[name], row[name + ' p'] = sl.loc[ax, 'slope_per_mm'], sl.loc[ax, 'p']
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt, na_rep=''))
    print("slopes in SD units per mm; a POSITIVE |x| slope = relatively more LWPC (for delta) "
          "or congruency (for dm) away from the midline")


# ---------------------------------------------------------------------------
# 2. the four single scores
# ---------------------------------------------------------------------------
def section_2(s, args):
    fu.banner('2. single scores on the coordinates  [follow-up; within-participant '
              'coordinate shuffle]')
    for model, coords in (('y + z + x (pipeline)', PIPELINE), ('y + z + |x|', MIDLINE)):
        P, _ = fu.coord_projector(s, coords=coords)
        rows = []
        for label, col, _ in SCORES:
            v = s[col].to_numpy(float)
            row = dict(score=label, y=(P @ v)[0])
            for ax in coords[1:]:
                i = coords.index(ax)
                row[ax], row[ax + ' p'] = fu.coord_perm_p(
                    s, v, args.n_perm_shuffle, args.seed, stat=lambda sl, i=i: sl[i], coords=coords)
            rows.append(row)
        print(f"\nscore ~ {model} + resp + participant")
        print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
    print("\na single score has no partner to swap with, so its null moves the coordinates "
          "among each participant's electrodes (§15.9)")


# ---------------------------------------------------------------------------
# 3. bands (figure panel c)
# ---------------------------------------------------------------------------
def band_tables(s, axis, labels):
    """Electrode-level adjusted d per tertile of ``axis``, and a participant-level table."""
    d = s.copy()
    for _, _, raw in SCORES:
        d[raw + '_adj'] = adjusted(d, raw)
    edges = np.percentile(d[axis], [0, 100 / 3, 200 / 3, 100])
    d['band'] = pd.cut(d[axis], edges, labels=labels, include_lowest=True)
    rows = []
    for band, g in d.groupby('band', observed=True):
        row = dict(band=band, n=len(g), mm=f"{g[axis].min():.0f} to {g[axis].max():.0f}")
        row.update({label: g[raw + '_adj'].mean() for label, _, raw in SCORES})
        row.update(mean_dm=g['dm'].mean(), mean_delta=g['delta'].mean())
        rows.append(row)
    long = d.melt(id_vars=['subject', 'band'], value_vars=[raw + '_adj' for _, _, raw in SCORES],
                  var_name='score', value_name='d')
    long['score'] = long['score'].map({raw + '_adj': label for label, _, raw in SCORES})
    per_subject = long.groupby(['band', 'score', 'subject'], observed=True)['d'].mean().reset_index()
    summary = (per_subject.groupby(['band', 'score'], observed=True)['d']
               .agg(mean='mean', sem='sem', n_participants='count').reset_index())
    return pd.DataFrame(rows), per_subject, summary


def section_3(s, args):
    fu.banner('3. adjusted d by band  [follow-up; descriptive; figure panel c]')
    for axis, labels, name in (('abs_x', ['medial', 'middle', 'lateral'], 'midline'),
                               ('mni_z', ['ventral', 'middle', 'dorsal'], 'height')):
        table, per_subject, summary = band_tables(s, axis, labels)
        what = 'distance from the midline, |x|' if name == 'midline' else 'height, MNI z'
        print(f"\ntertiles of {what} (electrode means; participant and responsiveness removed)")
        print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        if args.out_dir:
            per_subject.to_csv(os.path.join(args.out_dir, f'panel_c_{name}_by_participant.csv'),
                               index=False)
            summary.to_csv(os.path.join(args.out_dir, f'panel_c_{name}.csv'), index=False)
    print("\nplot participant means ± SEM across participants (panel_c_*.csv), not "
          "electrode means")


# ---------------------------------------------------------------------------
# 4. label means (figure panel b)
# ---------------------------------------------------------------------------
def section_4(s, args):
    fu.banner('4. label means of dm against delta  [follow-up; descriptive; figure panel b]')
    cov = sfa.build_coverage_matrix(s, roi_col=args.roi_col)
    per = {v: sfa.relative_score_roi_test(s, cov, min_subjects=args.min_subjects, n_perm=100,
                                          seed=args.seed, roi_col=args.roi_col,
                                          value_col=v)['per_roi'] for v in ('delta', 'dm')}
    m = (per['delta'][[args.roi_col, 'n_electrodes', 'n_subjects', 'mean_delta_adj']]
         .rename(columns={'mean_delta_adj': 'delta_adj'})
         .merge(per['dm'][[args.roi_col, 'mean_delta_adj']]
                .rename(columns={'mean_delta_adj': 'dm_adj'}), on=args.roi_col))
    m = m.join(s.groupby(args.roi_col)[['mni_z', 'abs_x']].mean(), on=args.roi_col)
    m = m.sort_values('abs_x').reset_index(drop=True)
    print(m.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    same = int((np.sign(m['delta_adj']) == np.sign(m['dm_adj'])).sum())
    print(f"label r(dm, delta) = {pearsonr(m['dm_adj'], m['delta_adj'])[0]:+.2f} "
          f"(Spearman {spearmanr(m['dm_adj'], m['delta_adj'])[0]:+.2f}); same sign in "
          f"{same} of {len(m)} labels")
    print("adjusted means as in dm_per_roi.csv / delta_per_roi.csv. Full-data dm and delta "
          "share trials, so this is descriptive; Test 1 is the separate-half test.")
    if args.out_dir:
        m.to_csv(os.path.join(args.out_dir, 'panel_b_label_means.csv'), index=False)


# ---------------------------------------------------------------------------
# 5. Test 2 on both axes, and its scale
# ---------------------------------------------------------------------------
def _bootstrap_shrinkage(s, coords, ax, n_boot, seed):
    """Participant bootstrap of 1 - slope(delta | dm) / slope(delta) on ``ax``."""
    rng = np.random.default_rng(seed)
    groups = {sub: g for sub, g in s.groupby('subject')}
    subjects = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        d = pd.concat([groups[sub].assign(subject=f'{sub}#{k}') for k, sub in enumerate(pick)],
                      ignore_index=True)
        without = sfa._coordinate_fit(d, 'delta', coords, ('resp',), 0, seed)['slopes']
        with_dm = sfa._coordinate_fit(d, 'delta', coords, ('resp', 'dm'), 0, seed)['slopes']
        if len(without) and len(with_dm):
            a = without.set_index('axis').loc[ax, 'slope_per_mm']
            b = with_dm.set_index('axis').loc[ax, 'slope_per_mm']
            out.append(1 - b / a)
    return np.asarray(out)


def section_5(s, args):
    fu.banner('5. Test 2 on both axes, and the scale for its shrinkage  [follow-up]')
    rows, boots = [], {}
    for name, coords, ax in (('height (pipeline model)', PIPELINE, 'mni_z'),
                             ('distance from midline', MIDLINE, 'abs_x')):
        fit = {k: swap_fit(s, value, coords, covs, args.n_perm, args.seed)[0].loc[ax]
               for k, value, covs in (('dm', 'dm', ('resp',)), ('delta', 'delta', ('resp',)),
                                      ('delta + dm', 'delta', ('resp', 'dm')))}
        boots[name] = _bootstrap_shrinkage(s, coords, ax, args.n_boot, args.seed)
        lo, hi = np.percentile(boots[name], [2.5, 97.5]) if len(boots[name]) else (np.nan,) * 2
        rows.append({'axis': name, 'dm': fit['dm']['slope_per_mm'], 'dm p': fit['dm']['p'],
                     'delta': fit['delta']['slope_per_mm'], 'delta p': fit['delta']['p'],
                     'delta + dm': fit['delta + dm']['slope_per_mm'],
                     'delta + dm p': fit['delta + dm']['p'],
                     'shrinkage': 1 - fit['delta + dm']['slope_per_mm'] / fit['delta']['slope_per_mm'],
                     'boot 2.5%': lo, 'boot 97.5%': hi})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=fmt))
    print(f"swap-null p; shrinkage on the same trials, with a {args.n_boot}-resample "
          "participant bootstrap interval")

    observed = {'same trials, height': table.loc[0, 'shrinkage'],
                'same trials, midline': table.loc[1, 'shrinkage']}
    if args.tilt:
        t = pd.read_csv(args.tilt).set_index('fit')['slope_per_mm']
        if {'delta, split halves', 'delta + dm from the opposite half'} <= set(t.index):
            observed['opposite half, height'] = (1 - t['delta + dm from the opposite half']
                                                 / t['delta, split halves'])
    if not args.main_json:
        print("\npass --main-json to put the shrinkage on dm's reliability scale")
        return
    with open(args.main_json) as f:
        m = json.load(f)
    rx, ry, r = m['reliability_x'], m['reliability_y'], m['corr']
    lam_half = (rx + ry - 2 * r) / (2 - 2 * r)
    lam_full = 2 * lam_half / (1 + lam_half)
    print(f"\ncongruency-switch r on separate halves = {r:+.3f} (reliabilities {rx:.3f} / "
          f"{ry:.3f}; {r / np.sqrt(rx * ry):.2f} of the ceiling)")
    print(f"dm split-half reliability = ({rx:.3f} + {ry:.3f} - 2*{r:.3f}) / (2 - 2*{r:.3f}) "
          f"= {lam_half:.3f}; full data (Spearman-Brown) {lam_full:.3f}")
    print("a tilt carried entirely by dm would shrink by about dm's reliability when dm is "
          "the covariate:")
    for k, v in observed.items():
        lam = lam_half if k.startswith('opposite') else lam_full
        print(f"   {k:24s} observed {v:+.3f}   expected if fully inherited ~{lam:.3f}   "
              f"implied share {v / lam:+.2f}")
    print("within-participant reliabilities are biased low (§15.4), so the implied shares lean "
          "high; the same-trials rows also carry shared-trial noise (~+0.05)")


# ---------------------------------------------------------------------------
# 6. Figure 5 (the anatomy job draws it too; this redraws it from the outputs)
# ---------------------------------------------------------------------------
def section_6(s, args):
    fu.banner('6. Figure 5: the overlap at both levels (a), matched vs crossed (b)')
    if not (args.seg_dir and args.out_dir):
        print("(section 6 skipped: pass --seg-dir <segregation run> and --out-dir)")
        return
    tracking = args.tracking or os.path.join(os.path.dirname(os.path.abspath(args.scores)),
                                             'delta_tracking.csv')
    centroid = fu.centroid_shuffle_test(s, ('subject',), n_perm=args.n_perm, seed=args.seed)
    fig = sfa.figure5(pd.read_csv(args.scores), pd.read_csv(tracking), args.out_dir,
                      seg_dir=args.seg_dir, centroid=centroid, n_perm=args.n_perm)
    print('\n'.join(fig['lines']))
    print(f"written to {args.out_dir}")


SECTIONS = {1: section_1, 2: section_2, 3: section_3, 4: section_4, 5: section_5, 6: section_6}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', required=True, help='scores_with_anatomy.csv from the N4 job')
    ap.add_argument('--seg-dir', default=None,
                    help="the _main_effects segregation run the anatomy job read (section 6)")
    ap.add_argument('--main-json', default=None,
                    help="the segregation run's correlation_main_effects.json (section 5; "
                         "default: the one in --seg-dir)")
    ap.add_argument('--tilt', default=None, help="the anatomy run's tilt_with_dm.csv (section 5)")
    ap.add_argument('--tracking', default=None,
                    help="the anatomy run's delta_tracking.csv (section 6; default: beside --scores)")
    ap.add_argument('--out-dir', default=None, help='write the figure-panel tables here')
    ap.add_argument('--roi-col', default='anat', help="label column the job tested ('anat')")
    ap.add_argument('--min-subjects', type=int, default=3)
    ap.add_argument('--n-perm', type=int, default=10000, help='sign-flip swap permutations')
    ap.add_argument('--n-perm-shuffle', type=int, default=2000, help='coordinate shuffles')
    ap.add_argument('--n-boot', type=int, default=2000, help='participant bootstrap resamples')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sections', default='1,2,3,4,5,6')
    args = ap.parse_args(argv)

    seg_json = os.path.join(args.seg_dir or '', 'correlation_main_effects.json')
    if args.seg_dir and not args.main_json and os.path.exists(seg_json):
        args.main_json = seg_json
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    s = load(args.scores)
    print(f"{len(s)} electrodes, {s['subject'].nunique()} participants")
    for k in (int(x) for x in args.sections.split(',')):
        SECTIONS[k](s, args)


if __name__ == '__main__':
    main()
