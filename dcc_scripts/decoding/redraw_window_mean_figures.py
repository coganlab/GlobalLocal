#!/usr/bin/env python
"""
Redraw the window-mean group figures (group_bars and electrode_curve, PNG and
PDF) from their saved tables, with the current plotting code: no data loading,
no decoding. Every ``<comparison>/<roi>/`` folder under ``--root`` that has a
summary.csv is redrawn in place.

    python dcc_scripts/decoding/redraw_window_mean_figures.py --root <dir>

``--root`` can be the ``window_mean_groups`` folder (every condition, matched
and unmatched), one run tag inside it, or one ``<comparison>/<roi>`` folder.
"""
import argparse
import glob
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.analysis.decoding.plots.electrode_curve import plot_accuracy_by_electrode_count
from src.analysis.decoding.plots.group_bars import plot_group_accuracy_bars


def redraw_folder(folder):
    """Redraw one ``<comparison>/<roi>`` folder's bars (always) and curve (when
    it has curve_summary.csv); returns the paths written."""
    read = lambda name: pd.read_csv(os.path.join(folder, name))
    with open(os.path.join(folder, 'params.json')) as f:
        n_match = json.load(f)['n_match']
    tests = read('tests.csv') if os.path.getsize(os.path.join(folder, 'tests.csv')) > 2 \
        else pd.DataFrame()
    paths = plot_group_accuracy_bars(
        {'summary': read('summary.csv'), 'tests': tests, 'accuracies': read('accuracies.csv'),
         'n_match': n_match}, os.path.join(folder, 'group_bars'))
    if os.path.exists(os.path.join(folder, 'curve_summary.csv')):
        paths += plot_accuracy_by_electrode_count({'summary': read('curve_summary.csv')},
                                                  os.path.join(folder, 'electrode_curve'))
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    folders = sorted({os.path.dirname(f) for f in
                      glob.glob(os.path.join(args.root, '**', 'summary.csv'), recursive=True)})
    for folder in folders:
        redraw_folder(folder)
        print(os.path.relpath(folder, args.root))
    print(f"redrew {len(folders)} folders under {args.root}")


if __name__ == '__main__':
    main()
