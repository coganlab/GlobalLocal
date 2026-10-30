#!/usr/bin/env python
"""
Why the anatomy's congruency-switch overlap and the coding axes' alignment
differ: one segregation run's main effects scored as mean differences (the
coding axes' measure) and as Cohen's d (the anatomy's), on the same split
halves, each compared three ways (uncentred cosine, Pearson r across the
pooled electrodes, cosine after centring within each participant), every
one with the coding axes' noise ceiling (`coding_axes.compare_coding_axes`).

Reads the run's `long_df.csv` (one row per electrode x trial, HG averaged over
the run's window). That table has no `trial` column, so each electrode's
trials are split on their own; every term of every cosine pairs one electrode's
half A with the same electrode's half B, so the halves are still disjoint
wherever it matters.

Writes to `<seg-dir>/overlap_measures/`:
    overlap_measures.csv   one row per measure x comparison: cross, the two
                           ceilings, ceiling, alignment = cross / ceiling with
                           its participant-bootstrap CI
    run_config.json        every parameter

    python dcc_scripts/subspace/overlap_measures.py --seg-dir <segregation run>
"""
import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats import coding_axes as ca

MEASURES = ('mean_diff', 'cohens_d')


def centre_within_participants(per_split):
    """`per_split` with each axis column centred on its participant's mean over
    electrodes, per split: what the anatomy's within-participant centring does
    to the scores before they are correlated."""
    out = per_split.copy()
    cols = list(ca.AXIS_COLS)
    out[cols] = out[cols] - out.groupby(['subject', 'split'])[cols].transform('mean')
    return out


def overlap_rows(per_split, n_boot, seed):
    """The three comparisons of one per-split table as rows: cosine and Pearson
    from one `compare_coding_axes` call, the within-participant one from a
    second on the centred table."""
    table, _ = ca.compare_coding_axes(per_split, n_boot=n_boot, seed=seed)
    centred, _ = ca.compare_coding_axes(centre_within_participants(per_split),
                                        n_boot=n_boot, seed=seed)
    t, c = table.iloc[0], centred.iloc[0]
    base = dict(n_electrodes=int(t['n_electrodes']), n_subjects=int(t['n_subjects']))
    return [
        dict(comparison='cosine', cross=t['cross'], congruency_ceiling=t['congruency_ceiling'],
             switch_ceiling=t['switch_ceiling'], ceiling=t['ceiling'], alignment=t['alignment'],
             alignment_lo=t['alignment_lo'], alignment_hi=t['alignment_hi'], **base),
        dict(comparison='pearson', cross=t['cross_pearson'],
             congruency_ceiling=t['congruency_ceiling_pearson'],
             switch_ceiling=t['switch_ceiling_pearson'], ceiling=t['ceiling_pearson'],
             alignment=t['alignment_pearson'], **base),
        dict(comparison='within_participant', cross=c['cross'],
             congruency_ceiling=c['congruency_ceiling'], switch_ceiling=c['switch_ceiling'],
             ceiling=c['ceiling'], alignment=c['alignment'],
             alignment_lo=c['alignment_lo'], alignment_hi=c['alignment_hi'], **base),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seg-dir', required=True, help='a segregation run with long_df.csv')
    p.add_argument('--n-splits', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-boot', type=int, default=2000)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.seg_dir, 'long_df.csv'))
    rows = []
    for measure in MEASURES:
        per = sfs.compute_sensitivities_per_split(
            df, n_splits=args.n_splits, seed=args.seed, contrast_mode='proportion',
            effect_measure=measure, main_effects=True)
        rows += [dict(measure=measure, **r) for r in overlap_rows(per, args.n_boot, args.seed)]
    out = pd.DataFrame(rows)

    out_dir = os.path.join(args.seg_dir, 'overlap_measures')
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, 'overlap_measures.csv'), index=False)
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump(dict(seg_dir=args.seg_dir, n_splits=args.n_splits, seed=args.seed,
                       n_boot=args.n_boot, measures=list(MEASURES),
                       split_scheme='per electrode'), f, indent=2)
    print(out.round(3).to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == '__main__':
    main()
