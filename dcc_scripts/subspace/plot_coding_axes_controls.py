#!/usr/bin/env python
"""Compare coding-axes runs -- the main result and its §3.3 controls -- in one
figure and one table.

    python plot_coding_axes_controls.py RUN_DIR [RUN_DIR ...] [--out DIR]

Each RUN_DIR is one coding-axes output directory (submit_coding_axes_dcc.sh);
runs are named from their run_config.json and drawn in the order given, so put
the main run first. Writes to --out (default: the directory holding the first
RUN_DIR):

    coding_axes_controls.png   cross cosine, alignment, and where each run is
                               reliable / shared (`plot_coding_axes_controls`)
    coding_axes_controls.csv   one row per run: the alignment over its reliable
                               windows with its CI, and where it is shared /
                               not one code (window centres, s)

Reads the runs' outputs only; nothing is recomputed.
"""
import argparse
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from dcc_scripts.subspace.coding_axes_dcc import (
    load_run, run_label, plot_coding_axes_controls, time_axis_label)


def _centres(table, column):
    """First and last window centre (s) where `column` is True, or NaN."""
    t = table.loc[table[column].astype(bool), 'window_center'] if column in table else []
    return (float(t.min()), float(t.max())) if len(t) else (np.nan, np.nan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_dirs', nargs='+', help='coding-axes output directories')
    parser.add_argument('--out', default=None, help='where to write (default: parent of the first run)')
    args = parser.parse_args(argv)

    loaded = [load_run(d) for d in args.run_dirs]
    labels = [run_label(config) for config, _, _ in loaded]
    # Two runs with the same settings but e.g. another seed: tell them apart.
    labels = [f"{label} ({os.path.basename(os.path.normpath(d))})" if labels.count(label) > 1
              else label for label, d in zip(labels, args.run_dirs)]

    out = args.out or os.path.dirname(os.path.abspath(os.path.normpath(args.run_dirs[0])))
    os.makedirs(out, exist_ok=True)
    # Compare runs locked to the same event; the first run's sets the time axis.
    plot_coding_axes_controls([(label, table) for label, (_, table, _) in zip(labels, loaded)],
                              os.path.join(out, 'coding_axes_controls.png'),
                              time_label=time_axis_label(loaded[0][0].get('epochs_root_file')))

    rows = []
    for label, d, (_, table, summary) in zip(labels, args.run_dirs, loaded):
        ci = summary['alignment_reliable_ci'] or [np.nan, np.nan]
        shared = _centres(table, 'shared')
        not_one = _centres(table, 'not_one_code')
        rows.append(dict(run=label, run_dir=os.path.abspath(d),
                         n_electrodes=summary['n_electrodes'],
                         n_subjects=summary['n_subjects'],
                         n_reliable_windows=summary['n_reliable_windows'],
                         alignment_reliable_mean=summary['alignment_reliable_mean'],
                         alignment_ci_lo=ci[0], alignment_ci_hi=ci[1],
                         shared_from_s=shared[0], shared_to_s=shared[1],
                         not_one_code_from_s=not_one[0], not_one_code_to_s=not_one[1]))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'coding_axes_controls.csv'), index=False)
    print(f"wrote coding_axes_controls.png / .csv to {out}")


if __name__ == '__main__':
    main()
