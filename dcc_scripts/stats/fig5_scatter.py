"""The LWPC-LWPS overlap scatter alone, from a finished anatomy run.

Panel b of ``fig5_height`` without the height tertile colours, the band
centroids or the zoom (``sfa.plot_figure5_scatter``), for when the height
gradient is not emphasized and the scatter becomes a panel of another figure.
Reads the ``<prefix>_points.csv`` that ``figure5_height`` saved; nothing is
recomputed, and the full ``fig5_height.*`` is left as it is. New anatomy runs
write the same file themselves.

    python dcc_scripts/stats/fig5_scatter.py \\
        --dir <anatomy run>/continuous[/section19...] [--pair adaptation|main_effects]

Writes ``<prefix>_scatter.png``/``.pdf`` into ``--dir`` (``fig5_height_scatter``
for the adaptation pair), with ``<prefix>_scatter.json`` naming the points file
it was drawn from.
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
    ap.add_argument('--dir', required=True,
                    help="folder holding the run's fig5_height*_points.csv; output goes here")
    ap.add_argument('--pair', default='adaptation', choices=tuple(sfa.FIG5_PAIRS))
    args = ap.parse_args(argv)

    prefix = sfa.FIG5_PAIRS[args.pair]['prefix']
    src = os.path.join(args.dir, f'{prefix}_points.csv')
    points = pd.read_csv(src)
    paths = sfa.plot_figure5_scatter(points, os.path.join(args.dir, f'{prefix}_scatter'),
                                     pair=args.pair)
    n = int(points[['x_plot', 'y_plot']].notna().all(axis=1).sum())
    with open(os.path.join(args.dir, f'{prefix}_scatter.json'), 'w') as f:
        json.dump(dict(params=vars(args), points_csv=os.path.abspath(src), n_electrodes=n,
                       n_participants=int(points['subject'].nunique())), f, indent=2)
    print(f"{n} electrodes -> {', '.join(paths)}")


if __name__ == '__main__':
    main()
