"""The LWPC-LWPS (or congruency-switch) overlap scatter alone, from a finished
anatomy run.

Panel b of ``fig5_height`` without the height tertile colours, the band
centroids, the zoom or the identity line (``sfa.plot_figure5_scatter``), for
when the height gradient is not emphasized and the scatter becomes a panel of
another figure. Reads the ``<prefix>_points.csv`` that ``figure5_height``
saved; the full ``fig5_height.*`` is left as it is. New anatomy runs write the
same file themselves.

With ``--seg-dir`` (the segregation run the anatomy was scored from) the
pair's pre-specified separate-half test is printed in the panel: read from that
run's ``correlation.json`` / ``correlation_main_effects.json`` when it tested
the same electrodes, else recomputed from its ``per_split.csv``.

    python dcc_scripts/stats/fig5_scatter.py \\
        --dir <anatomy run>/continuous[/section19...] [--pair adaptation|main_effects] \\
        [--seg-dir <segregation_results/window_...>]

Writes ``<prefix>_scatter.png``/``.pdf`` into ``--dir`` (``fig5_height_scatter``
for the adaptation pair), with ``<prefix>_scatter.json`` naming the points file
it was drawn from and the test it prints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402
from src.analysis.stats.stability_flexibility_segregation import main_effect_view  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True,
                    help="folder holding the run's fig5_height*_points.csv; output goes here")
    ap.add_argument('--pair', default='adaptation', choices=tuple(sfa.FIG5_PAIRS))
    ap.add_argument('--seg-dir', default=None,
                    help='segregation run the scores came from; its test is printed in the panel')
    # used only if the test has to be recomputed (figure5_height's defaults)
    ap.add_argument('--min-elec', type=int, default=3)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args(argv)

    spec = sfa.FIG5_PAIRS[args.pair]
    prefix = spec['prefix']
    src = os.path.join(args.dir, f'{prefix}_points.csv')
    points = pd.read_csv(src)
    stat = None
    if args.seg_dir:
        # the points table carries each electrode's responsiveness, all the
        # recompute needs from the scores
        stat = sfa._figure5_test(points, points, os.path.join(args.seg_dir, spec['json']),
                                 main_effect_view if spec['main_effects'] else (lambda ps: ps),
                                 None, args.seg_dir, args.min_elec, args.n_perm, args.seed)
    paths = sfa.plot_figure5_scatter(points, os.path.join(args.dir, f'{prefix}_scatter'),
                                     pair=args.pair, stat=stat)
    n = int(points[['x_plot', 'y_plot']].notna().all(axis=1).sum())
    test = None if stat is None else {k: stat[k] for k in
                                      ('corr', 'p', 'method', 'n_electrodes', 'n_subjects',
                                       'source') if k in stat}
    with open(os.path.join(args.dir, f'{prefix}_scatter.json'), 'w') as f:
        json.dump(dict(params=vars(args), points_csv=os.path.abspath(src), n_electrodes=n,
                       n_participants=int(points['subject'].nunique()), test=test),
                  f, indent=2, default=str)
    print(f"{n} electrodes -> {', '.join(paths)}"
          + ('' if stat is None else f"; r = {stat['corr']:.3f}, p = {stat['p']:.2g}"))


if __name__ == '__main__':
    main()
