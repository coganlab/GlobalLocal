#!/usr/bin/env python
"""Re-run the coupled-vs-uncoupled comparison over saved decoding results.

``decoding_dcc`` already runs this at the end of a coupling job. Use this script
when you want to redo the comparison without redoing the decoding: to change the
percentile or the cluster permutation count, to add draws from a second job, or
because the aggregation failed after the expensive part had finished.

It reads the ``elecset_coupled/`` and ``elecset_uncoupled_drawNN/`` directories
under one decoding output directory and writes
``coupled_vs_uncoupled/`` beside them::

    python dcc_scripts/decoding/run_coupling_comparison_dcc.py \
        --save_dir dcc_scripts/decoding/figs/<epochs_root_file>/<...> \
        --condition_comparisons task_g_vs_task_l \
        --rois lpfc occ
"""

import argparse
import os
import sys
from types import SimpleNamespace

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.config.condition_registry import get_comparisons
from src.analysis.decoding.coupling_comparison import run_coupling_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare coupled vs matched-uncoupled decoding accuracies.")
    parser.add_argument('--save_dir', required=True,
                        help="directory holding the elecset_* subdirectories")
    parser.add_argument('--condition_label', default=None,
                        help="condition_registry key; its comparisons are used "
                             "when --condition_comparisons is not given")
    parser.add_argument('--condition_comparisons', nargs='*', default=None)
    parser.add_argument('--rois', nargs='+', default=['lpfc', 'occ'])
    parser.add_argument('--percentile', type=float, default=95)
    parser.add_argument('--cluster_percentile', type=float, default=95)
    parser.add_argument('--n_cluster_perms', type=int, default=1000)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--step_size', type=int, default=16)
    parser.add_argument('--sampling_rate', type=float, default=256)
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    comparisons = args.condition_comparisons
    if not comparisons:
        if not args.condition_label:
            parser.error("pass --condition_comparisons or --condition_label")
        comparisons = list(get_comparisons(args.condition_label).keys())

    run_coupling_comparison(
        base_save_dir=args.save_dir,
        args=SimpleNamespace(
            percentile=args.percentile,
            cluster_percentile=args.cluster_percentile,
            n_cluster_perms=args.n_cluster_perms,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            window_size=args.window_size,
            step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            condition_label=args.condition_label or '',
            timestamp=None,
            single_column=True,
            show_legend=True,
        ),
        condition_comparisons=comparisons,
        rois=args.rois,
        make_plots=not args.no_plots,
    )


if __name__ == '__main__':
    main()
