"""Fig. 3's adaptation direction with the RT-linked part of high gamma removed.

Fig. 3's LWPC and LWPS clusters run past the median RT, and RT coupling alone
predicts a neural adaptation with behavior's sign (``rt_adjust_hg`` in
``stability_flexibility_brain_behavior``). The A6 job already scores every
task-significant lPFC electrode with and without that part
(``participant_electrode_scores.csv``). This reads that table and asks whether
the mean LWPC and LWPS, participants as the unit, are still positive after the
adjustment (``sbb.group_adaptation_rt_check``). New A6 runs write the same
table themselves (``group_adaptation_rt_check.csv``, block (0) of
``summary.txt``); this script is for runs made before that.

    python dcc_scripts/stats/f3_rt_adjusted_check.py \\
        --electrodes <A6 run>/participant_electrode_scores.csv \\
        [--out <where to write group_adaptation_rt_check.csv>] [--min-elec 1]

The window is the A6 run's (WINDOW_TMIN/WINDOW_TMAX; 0-1.5 s by default). For a
check that excludes most responses, rerun A6 with WINDOW_TMAX=0.5 and read
this table from that run too.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_brain_behavior as sbb  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--electrodes', required=True,
                    help="the A6 run's participant_electrode_scores.csv")
    ap.add_argument('--out', default=None,
                    help='write group_adaptation_rt_check.csv here (default: beside --electrodes)')
    ap.add_argument('--min-elec', type=int, default=1,
                    help='usable electrodes a participant needs to contribute (default 1)')
    ap.add_argument('--n-perm', type=int, default=10000, help='participant sign flips')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    table = sbb.group_adaptation_rt_check(pd.read_csv(args.electrodes), min_elec=args.min_elec,
                                          n_perm=args.n_perm, seed=args.seed)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.electrodes)),
                                   'group_adaptation_rt_check.csv')
    table.to_csv(out, index=False)
    print("GROUP-LEVEL ADAPTATION, RAW vs RT-ADJUSTED HG (the check on Fig. 3)")
    print('\n'.join(sbb.group_adaptation_rt_lines(table)))
    print(f"written to {out}")
    return table


if __name__ == '__main__':
    main()
