"""§19 of docs/n4_continuous_anatomy.md from a finished anatomy run.

The advisor meeting of 2026-10-02 asked for three things the anatomy job did
not compute. New runs of the job compute them (``sfa.section19``); this script
computes them from an existing run's outputs, with nothing touching epochs,
atlases or recon files:

1. the height (MNI z) slope of LWPC - LWPS with participants as the unit:
   its decomposition into per-participant slopes (weighted and unweighted
   tests across participants), mixed models with a participant random
   intercept and with a random slope, and the slope with each participant
   left out;
2. the pre-specified LWPC-LWPS separate-half correlation with participants
   as the unit (per-participant r, Fisher z, tested across participants);
3. local similarity: cross-half similarity of electrode pairs within
   participant by distance, for the balance and for each single score;
4. the combined Figure 5: LWPC against LWPS coloured by height tertile, the
   tertile centroids, and the balance by height.

    python dcc_scripts/stats/n4_section19_followups.py \\
        --anatomy-dir <anatomy run>/continuous \\
        [--seg-dir    <the _main_effects segregation run the anatomy job read>] \\
        [--per-split  <per_split.csv; default: --anatomy-dir, then --seg-dir>] \\
        [--out-dir    <default: <anatomy-dir>/section19>] [--sections 1,2,3,4]

``--seg-dir`` gives panel b of the figure its pre-specified r
(``correlation.json``); without it the r is recomputed from the per-split table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402


def _per_split_path(args):
    for d in (args.anatomy_dir, args.seg_dir):
        p = os.path.join(d, 'per_split.csv') if d else None
        if p and os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--anatomy-dir', required=True,
                    help="the anatomy run's continuous/ folder (scores_with_anatomy.csv)")
    ap.add_argument('--seg-dir', default=None, help='the segregation run the anatomy job read')
    ap.add_argument('--per-split', default=None, help='per_split.csv (default: found)')
    ap.add_argument('--out-dir', default=None, help='default: <anatomy-dir>/section19')
    ap.add_argument('--axis', default='mni_z')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sections', default='1,2,3,4')
    args = ap.parse_args(argv)

    scores = pd.read_csv(os.path.join(args.anatomy_dir, 'scores_with_anatomy.csv'))
    ps_path = args.per_split or _per_split_path(args)
    per_split = pd.read_csv(ps_path) if ps_path else None
    out_dir = args.out_dir or os.path.join(args.anatomy_dir, 'section19')
    print(f"{len(scores)} electrodes, {scores['subject'].nunique()} participants; per-split "
          f"table: {ps_path or 'none (sections 2 and 3 skipped)'}")

    lines, out = sfa.section19(scores, per_split, out_dir, seg_dir=args.seg_dir, axis=args.axis,
                               n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
                               sections=tuple(int(x) for x in args.sections.split(',')))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary_section19.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'section19.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"written to {out_dir}")


if __name__ == '__main__':
    main()
