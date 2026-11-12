#!/usr/bin/env python
"""
Redraw every figure of a finished coding-axes run from its saved tables, with
the current plotting code (no data loading, no statistics): the cosine figure,
the subspace test, the condition trajectories (Gram-Schmidt and raw axes), the
block separations (raw and unique axes) and the proportion trajectories, for
each trajectory axis window the run used. Each figure is written as PNG and
PDF, with its supplementary companion where it has one (``*_alignment``,
``*_changes``).

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

import numpy as np
import pandas as pd

from src.analysis.stats.coding_trajectories import contrast_clusters
from dcc_scripts.subspace.coding_axes_dcc import VARIABLE_NAMES
from dcc_scripts.subspace.subspace_figures import (
    plot_coding_axes, plot_subspace_test, plot_trajectories, plot_block_separations,
    plot_proportion_trajectories, plot_proportion_differences,
    plot_proportion_change_vectors, time_axis_label)


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
    # runs before 2026-11-12 saved the block null but not its band and clusters: add them
    block_null = (dict(np.load(path('null_block.npz'))) if os.path.exists(path('null_block.npz'))
                  else None)
    for tag, unique in (('', False), ('_unique', True)):
        seps = pd.read_csv(path(f'block_separations{tag}.csv'))
        cons = pd.read_csv(path(f'block_contrasts{tag}.csv'))
        if block_null is not None and 'null_lo' not in cons:
            cons = contrast_clusters(cons, block_null['contrasts_unique' if unique
                                                      else 'contrasts'])
        for label in labels:
            plot_block_separations(seps, cons, label, out(f'block_separations_{label}{tag}.png'),
                                   time_label=time_label, unique=unique)
    ptraj, pchanges = (pd.read_csv(path('proportion_trajectories.csv')),
                       pd.read_csv(path('proportion_changes.csv')))
    prop_summary = None
    if os.path.exists(path('proportion_summary.json')):
        with open(path('proportion_summary.json')) as f:
            prop_summary = json.load(f)
    for label in labels:
        plot_proportion_trajectories(ptraj, pchanges, label,
                                     out(f'proportion_trajectories_{label}.png'))
        plot_proportion_differences(ptraj, label, out(f'proportion_differences_{label}.png'))
        if prop_summary is not None:
            plot_proportion_change_vectors(prop_summary, label,
                                           out(f'proportion_change_vectors_{label}.png'))
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
