#!/usr/bin/env python
"""
Redraw every figure of a finished coding-axes run from its saved tables, with
the current plotting code (no data loading, no statistics): the cosine figure,
the subspace test, the condition trajectories (Gram-Schmidt and raw axes), the
block separations (raw and unique axes) and the proportion trajectories, for
each trajectory axis window the run used.

    python dcc_scripts/subspace/redraw_figures.py --run-dir <coding-axes run> [--out-dir DIR]

``--out-dir`` defaults to the run folder itself (overwriting its figures).
"""
import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from dcc_scripts.subspace.coding_axes_dcc import plot_coding_axes, VARIABLE_NAMES
from dcc_scripts.subspace.subspace_figures import (
    plot_subspace_test, plot_trajectories, plot_block_separations,
    plot_proportion_trajectories, time_axis_label)


def redraw(run_dir, out_dir=None):
    """Write the run's figures to ``out_dir`` (default ``run_dir``); returns the paths."""
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    path = lambda name: os.path.join(run_dir, name)
    with open(path('run_config.json')) as f:
        config = json.load(f)
    time_label = time_axis_label(config.get('epochs_root_file'))
    written = []

    def out(name):
        written.append(os.path.join(out_dir, name))
        return written[-1]

    plot_coding_axes(pd.read_csv(path('coding_axis_cosines.csv')), out('coding_axis_cosines.png'),
                     time_label=time_label)
    if os.path.exists(path('subspace_test.csv')):
        with open(path('subspace_summary.json')) as f:
            k = json.load(f).get('k')
        plot_subspace_test(pd.read_csv(path('subspace_test.csv')),
                           pd.read_csv(path('subspace_dimensions.csv')), out('subspace_test.png'),
                           k=k, names=VARIABLE_NAMES[config.get('effect', 'main')])
    if not os.path.exists(path('trajectories.csv')):        # interaction runs stop here
        return written

    labels = [f'{a}-{b}s' for a, b in config['traj_axes']]
    fractions = pd.read_csv(path('trajectory_fractions.csv'))
    with open(path('trajectory_summary.json')) as f:
        traj_summary = json.load(f)
    for table, first, tag in ((pd.read_csv(path('trajectories.csv')), config.get('traj_first'), ''),
                              (pd.read_csv(path('trajectories_raw.csv')), None, '_raw')):
        for label in labels:
            for cells, suffix in (('centred', ''), ('uncentred', '_uncentred')):
                plot_trajectories(table, fractions, traj_summary['axes'].get(label), label,
                                  out(f'trajectories_{label}{suffix}{tag}.png'), cells=cells,
                                  first=first, time_label=time_label)
    for tag, unique in (('', False), ('_unique', True)):
        seps = pd.read_csv(path(f'block_separations{tag}.csv'))
        cons = pd.read_csv(path(f'block_contrasts{tag}.csv'))
        for label in labels:
            plot_block_separations(seps, cons, label, out(f'block_separations_{label}{tag}.png'),
                                   time_label=time_label, unique=unique)
    ptraj, pchanges = (pd.read_csv(path('proportion_trajectories.csv')),
                       pd.read_csv(path('proportion_changes.csv')))
    for label in labels:
        plot_proportion_trajectories(ptraj, pchanges, label,
                                     out(f'proportion_trajectories_{label}.png'))
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    written = redraw(args.run_dir, args.out_dir)
    print(f"wrote {len(written)} figures to {args.out_dir or args.run_dir}")


if __name__ == '__main__':
    main()
