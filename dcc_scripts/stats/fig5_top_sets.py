"""Figure 5 with the top LWPC and top LWPS electrodes picked out, from a
finished anatomy run (advisor request, 2026-10-22).

The same brain, scatter and balance-by-height panels as ``fig5_height``, with
the top ``--frac`` of electrodes by LWPC and by LWPS in colour and the rest in
grey, and panel c's height slope refit on those electrodes alone
(``sfa.figure5_top``). Reads the run's saved tables; nothing touches epochs.
``--brain`` renders the sets on the fsaverage brain, which needs the recon
files and a display (run it under ``xvfb-run``, as
``sbatch_fig5_top_sets_dcc.sh`` does).

    python dcc_scripts/stats/fig5_top_sets.py \\
        --anatomy-dir <anatomy run>/continuous \\
        --seg-dir     <the segregation run the anatomy job read> \\
        [--frac 0.2] [--brain [--brain-hemi both] [--brain-zoom 0.8]]

Writes ``fig5_height_top20.*`` and its tables into ``--anatomy-dir`` (or
``--out-dir``), with ``summary_fig5_height_top20.txt`` and
``fig5_height_top20.json`` (the numbers and the parameters that produced them).
``--seg-dir`` gives panel b its pre-specified r (``correlation.json``); without
it the r is recomputed from the run's ``per_split.csv``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--anatomy-dir', required=True,
                    help="the anatomy run's continuous/ folder (scores_with_anatomy.csv)")
    ap.add_argument('--seg-dir', default=None,
                    help="the segregation run the anatomy job read (panel b's r)")
    ap.add_argument('--out-dir', default=None, help='default: --anatomy-dir')
    ap.add_argument('--frac', type=float, default=0.2,
                    help='share of electrodes kept per score (default 0.2)')
    ap.add_argument('--axis', default='mni_z')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--brain', action='store_true',
                    help='also render the sets on the fsaverage brain (needs the recon files)')
    ap.add_argument('--brain-hemi', default='both', choices=('both', 'lh', 'rh', 'split'))
    ap.add_argument('--brain-zoom', type=float, default=None,
                    help='per-panel camera zoom for --brain (<1 zooms out)')
    args = ap.parse_args(argv)

    scores = pd.read_csv(os.path.join(args.anatomy_dir, 'scores_with_anatomy.csv'))
    ps_path = os.path.join(args.anatomy_dir, 'per_split.csv')
    per_split = pd.read_csv(ps_path) if os.path.exists(ps_path) else None
    out_dir = args.out_dir or args.anatomy_dir
    res = sfa.figure5_top(scores, out_dir, frac=args.frac, per_split=per_split,
                          seg_dir=args.seg_dir, axis=args.axis, n_perm=args.n_perm,
                          seed=args.seed, make_brain=args.brain,
                          brain_kwargs=dict(hemi=args.brain_hemi, zoom=args.brain_zoom))

    text = '\n'.join(res['lines'])
    print(text)
    with open(os.path.join(out_dir, f"summary_{res['prefix']}.txt"), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, f"{res['prefix']}.json"), 'w') as f:
        json.dump(dict(params=vars(args),
                       sets=res['points']['top'].value_counts().to_dict(),
                       stat=res['stat'],
                       fits={name: {k: v for k, v in fit.items() if k != 'partials'}
                             for name, fit in res['fits'].items()},
                       brain=res['brain']), f, indent=2, default=str)
    print(f"written to {out_dir}")


if __name__ == '__main__':
    main()
