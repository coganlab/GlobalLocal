"""§19 of docs/n4_continuous_anatomy.md from a finished anatomy run.

The advisor meeting of 2026-10-02 asked for three things the anatomy job did
not compute. New runs of the job compute them (``sfa.section19``); this script
computes them from an existing run's outputs, with nothing touching epochs,
atlases or recon files:

1. the height (MNI z) slope of LWPC - LWPS with participants as the unit:
   its decomposition into per-participant slopes (weighted and unweighted
   tests across participants), mixed models with a participant random
   intercept and with a random slope, and the slope with each participant
   left out;
2. the pre-specified LWPC-LWPS separate-half correlation with participants
   as the unit (per-participant r, Fisher z, tested across participants);
3. local similarity: cross-half similarity of electrode pairs within
   participant by distance, for the balance and for each single score;
4. the combined Figure 5: LWPC against LWPS coloured by height tertile, the
   tertile centroids, and the balance by height;
5. the LWPC-LWPS overlap r with each candidate confound removed in turn
   (nonlinear responsiveness, coordinates, same-half base effects, RT
   coupling) and with each participant left out.

Section 3 needs halves shared by all of a participant's electrodes; the
segregation job's default per-split table splits each electrode on its own,
which makes neighbours' halves share trials (§19.3 of the N4 doc). Give it
``--long-df`` (the segregation run's ``long_df.csv``; this script then rescores
with shared splits, a few minutes) or ``--per-split-shared`` (the
``per_split.csv`` of a ``SHARED_SPLIT=1`` segregation run).

    python dcc_scripts/stats/n4_section19_followups.py \\
        --anatomy-dir <anatomy run>/continuous \\
        [--seg-dir    <the _main_effects segregation run the anatomy job read>] \\
        [--per-split  <per_split.csv; default: --anatomy-dir, then --seg-dir>] \\
        [--long-df    <segregation run>/long_df.csv]        (section 3) \\
        [--rt-coupling <A6 run>/participant_electrode_scores.csv]   (section 5) \\
        [--out-dir    <default: <anatomy-dir>/section19>] [--sections 1,2,3,4,5]

``--seg-dir`` gives panel b of the figure its pre-specified r
(``correlation.json``); without it the r is recomputed from the per-split table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.stats import stability_flexibility_anatomy as sfa  # noqa: E402


NO_TRIAL_IDS = '''
{path} has no `trial` column: it was assembled before 2026-09-27. A split shared
by a participant's electrodes needs to know which rows are the same trial, and
row order cannot stand in for it: rows with NaN high gamma were dropped electrode
by electrode, so electrodes can list different trials. Section 3 (local
similarity) is skipped; the other sections run. Two ways to get trial ids:
  - task-significant lPFC, no rerun: --long-df <A6 run>/long_df.csv (the A6 job
    wrote it after the column existed; same epochs file, correct trials only)
  - all lPFC: cd dcc_scripts/stats && SCATTER_ONLY=1 bash
    submit_stability_flexibility_segregation_dcc.sh  (minutes; writes
    <segregation_results>/window_..._scatter_only_splits0/long_df.csv), then
    --long-df that file
'''


def _per_split_path(args):
    for d in (args.anatomy_dir, args.seg_dir):
        p = os.path.join(d, 'per_split.csv') if d else None
        if p and os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--anatomy-dir', required=True,
                    help="the anatomy run's continuous/ folder (scores_with_anatomy.csv)")
    ap.add_argument('--seg-dir', default=None, help='the segregation run the anatomy job read')
    ap.add_argument('--per-split', default=None, help='per_split.csv (default: found)')
    ap.add_argument('--out-dir', default=None, help='default: <anatomy-dir>/section19')
    ap.add_argument('--axis', default='mni_z')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sections', default='1,2,3,4,5')
    ap.add_argument('--long-df', default=None,
                    help="the segregation run's long_df.csv: rescore with one trial split per "
                         "participant for section 3")
    ap.add_argument('--shared-n-splits', type=int, default=200,
                    help='splits for the --long-df rescoring (default 200)')
    ap.add_argument('--per-split-shared', default=None,
                    help='per_split.csv of a SHARED_SPLIT=1 segregation run (instead of --long-df)')
    ap.add_argument('--rt-coupling', default=None,
                    help='CSV with electrode and rt_r (A6 participant_electrode_scores.csv or '
                         'the RT-adjusted run\'s rt_adjustment_slopes.csv) for section 5')
    args = ap.parse_args(argv)

    scores = pd.read_csv(os.path.join(args.anatomy_dir, 'scores_with_anatomy.csv'))
    ps_path = args.per_split or _per_split_path(args)
    per_split = pd.read_csv(ps_path) if ps_path else None
    out_dir = args.out_dir or os.path.join(args.anatomy_dir, 'section19')
    print(f"{len(scores)} electrodes, {scores['subject'].nunique()} participants; per-split "
          f"table: {ps_path or 'none (sections 2 and 3 skipped)'}")

    os.makedirs(out_dir, exist_ok=True)
    shared = None
    if args.per_split_shared:
        shared = pd.read_csv(args.per_split_shared)
    elif args.long_df:
        long_df = pd.read_csv(args.long_df)
        if 'trial' not in long_df.columns:
            print(NO_TRIAL_IDS.format(path=args.long_df))
        else:
            from src.analysis.stats import stability_flexibility_segregation as sfs
            long_df = long_df[long_df['electrode'].isin(scores['electrode'])]
            n_e = long_df['electrode'].nunique()
            print(f"rescoring {n_e} electrodes with {args.shared_n_splits} participant-shared "
                  f"splits ..." + (f" (the long table covers {n_e} of the {len(scores)} "
                                   "electrodes; section 3 runs on those)"
                                   if n_e < scores['electrode'].nunique() else ''))
            shared = sfs.compute_sensitivities_per_split(
                long_df, n_splits=args.shared_n_splits, seed=args.seed,
                contrast_mode='proportion', effect_measure='cohens_d', main_effects=True,
                shared_split=True)
            shared.to_csv(os.path.join(out_dir, 'per_split_shared.csv'), index=False)
    rt = pd.read_csv(args.rt_coupling) if args.rt_coupling else None
    if rt is not None and 'rt_r' not in rt.columns:
        raise SystemExit(f"{args.rt_coupling} has no rt_r column")

    lines, out = sfa.section19(scores, per_split, out_dir, seg_dir=args.seg_dir, axis=args.axis,
                               n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
                               sections=tuple(int(x) for x in args.sections.split(',')),
                               per_split_shared=shared, rt_coupling=rt)
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary_section19.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'section19.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"written to {out_dir}")


if __name__ == '__main__':
    main()
