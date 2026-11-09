#!/usr/bin/env python
"""
Add the cross and ceiling bootstrap CIs (``cross_lo/_hi``, ``ceiling_lo/_hi``)
to a finished coding-axes run's coding_axis_cosines.csv, for runs made before
2026-11-09, when ``compare_coding_axes`` started saving them.

Recomputes the table from the run's saved inputs (per_split_windows.csv,
null_cross.npz, run_config.json: the same bootstrap seed, n_boot and alpha), so
the draws are the run's own. Refuses to write if any column the run already has
comes out different, so nothing but the four new columns can change.

    python dcc_scripts/subspace/add_cosine_cis.py --run-dir <coding-axes run>

Then redraw with redraw_figures.py.
"""
import argparse
import json
import os
import sys
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from src.analysis.stats import coding_axes as ca

NEW = ['cross_lo', 'cross_hi', 'ceiling_lo', 'ceiling_hi']


def add_cosine_cis(run_dir):
    """Recompute the run's cosine table and return it with the four CI columns,
    after checking every existing column is reproduced. Raises if not."""
    path = lambda name: os.path.join(run_dir, name)
    with open(path('run_config.json')) as f:
        config = json.load(f)
    saved = pd.read_csv(path('coding_axis_cosines.csv'))
    per = pd.read_csv(path('per_split_windows.csv'))
    npz = np.load(path('null_cross.npz'))
    null = {f: npz[f] for f in npz.files if f != 'window'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        table, _ = ca.compare_coding_axes(per, null=null, n_boot=config['n_boot'],
                                          seed=config['seeds']['bootstrap'],
                                          alpha=config['alpha'])
    for col in saved.columns:
        a, b = saved[col], table[col]
        same = (np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
                if pd.api.types.is_numeric_dtype(a) else a.astype(str).equals(b.astype(str)))
        if not same:
            raise ValueError(f"{col} does not reproduce in {run_dir}; not writing")
    return saved.assign(**{c: table[c].to_numpy() for c in NEW})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    table = add_cosine_cis(args.run_dir)
    table.to_csv(os.path.join(args.run_dir, 'coding_axis_cosines.csv'), index=False)
    print(f"added {', '.join(NEW)} to {args.run_dir}/coding_axis_cosines.csv")


if __name__ == '__main__':
    main()
