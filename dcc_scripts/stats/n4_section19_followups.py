"""§19 of docs/n4_continuous_anatomy.md from a finished anatomy run.

The advisor meeting of 2026-10-02 asked for three things the anatomy job did
not compute. New runs of the job compute them (``sfa.section19``); this script
computes them from an existing run's outputs, with nothing touching epochs,
atlases or recon files. Each part is reported with electrodes as the unit
first (the pre-specified tests' level, for the main text) and with
participants as the unit second (for the supplement):

1. the height (MNI z) slope of LWPC - LWPS: the coordinate test's slope and p
   with an electrode-bootstrap interval; then its decomposition into
   per-participant slopes (weighted and unweighted tests across
   participants), mixed models with a participant random intercept and with
   a random slope, and the slope with each participant left out;
2. the pre-specified LWPC-LWPS separate-half correlation with an
   electrode-bootstrap interval; then per participant (Fisher z, tested
   across participants);
3. local similarity: cross-half similarity of electrode pairs within
   participant by distance, for the balance and for each single score, with
   an electrode-level SE that includes the trial noise neighbours share, and
   with participant sign flips; plus the overlap r, reliabilities and
   noise-corrected r on the shared-split table, with electrode-level intervals;
4. the combined Figure 5: LWPC against LWPS coloured by height tertile, the
   tertile centroids, and the balance by height (fig5_height.* at the
   electrode level, fig5_height_participants.* at the participant level);
   with ``--brain``, also the electrodes on the fsaverage brain coloured by
   tertile, with each tertile's centroid per hemisphere drawn larger in a
   darker shade, and the tertile cuts as lines (the one step that reads recon files; run it under
   ``xvfb-run`` or on a node with a display);
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
        [--out-dir    <default: <anatomy-dir>/section19>] [--sections 1,2,3,4,5] \\
        [--brain [--brain-hemi both|lh|rh|split] [--brain-zoom 0.8]]   (section 4)

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
  - all lPFC: cd dcc_scripts/stats && RT_ADJUST_HG=0 SCATTER_N_SPLITS=0
    SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh
    (minutes; writes <segregation_results>/window_..._scatter_only_splits0/
    long_df.csv), then --long-df that file
'''


RT_LABEL = 'shared by participant, RT-adjusted HG'


def _per_split_path(args):
    for d in (args.anatomy_dir, args.seg_dir):
        p = os.path.join(d, 'per_split.csv') if d else None
        if p and os.path.exists(p):
            return p
    return None


def rt_adjusted(path):
    """Whether a segregation run's table comes from an RT_ADJUST_HG=1 run: its
    high gamma has the RT-linked part removed, and rt_adjustment_slopes.csv
    sits beside it."""
    return os.path.exists(os.path.join(os.path.dirname(os.path.abspath(path)),
                                       'rt_adjustment_slopes.csv'))


def rescore_shared(long_df_path, electrodes, n_splits=200, seed=0):
    """The per-split table section 3 needs: the long table rescored with one
    trial split per participant, on the electrodes in ``electrodes``. None, after
    saying how to get trial ids, when the table has no `trial` column."""
    long_df = pd.read_csv(long_df_path)
    if 'trial' not in long_df.columns:
        print(NO_TRIAL_IDS.format(path=long_df_path))
        return None
    from src.analysis.stats import stability_flexibility_segregation as sfs
    n_all = pd.Series(electrodes).nunique()
    long_df = long_df[long_df['electrode'].isin(set(electrodes))]
    n_e = long_df['electrode'].nunique()
    print(f"rescoring {n_e} electrodes with {n_splits} participant-shared "
          f"splits ..." + (f" (the long table covers {n_e} of the {n_all} "
                           "electrodes; section 3 runs on those)" if n_e < n_all else ''))
    return sfs.compute_sensitivities_per_split(
        long_df, n_splits=n_splits, seed=seed, contrast_mode='proportion',
        effect_measure='cohens_d', main_effects=True, shared_split=True)


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
    ap.add_argument('--brain', action='store_true',
                    help='section 4: also draw the height tertiles and their centroids on the '
                         'fsaverage brain (fig5_height_brain.png; needs the recon files)')
    ap.add_argument('--brain-hemi', default='both', choices=('both', 'lh', 'rh', 'split'))
    ap.add_argument('--brain-zoom', type=float, default=None,
                    help='per-panel camera zoom for --brain (<1 zooms out)')
    args = ap.parse_args(argv)

    scores = pd.read_csv(os.path.join(args.anatomy_dir, 'scores_with_anatomy.csv'))
    ps_path = args.per_split or _per_split_path(args)
    per_split = pd.read_csv(ps_path) if ps_path else None
    out_dir = args.out_dir or os.path.join(args.anatomy_dir, 'section19')
    print(f"{len(scores)} electrodes, {scores['subject'].nunique()} participants; per-split "
          f"table: {ps_path or 'none (sections 2 and 3 skipped)'}")

    shared, shared_label = None, 'shared by participant'
    # A long table from an RT_ADJUST_HG=1 run has the RT-linked part of high gamma
    # removed. Label it, and keep its results apart from the raw ones.
    src = args.long_df or args.per_split_shared
    if src and rt_adjusted(src):
        shared_label = RT_LABEL
        if not args.out_dir:
            out_dir = os.path.join(args.anatomy_dir, 'section19_rt_adjusted')
        print(f"NOTE: {src} comes from an RT_ADJUST_HG=1 run: its high gamma has the "
              "RT-linked part removed, so section 3 describes RT-adjusted scores. Output: "
              f"{out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    if args.per_split_shared:
        shared = pd.read_csv(args.per_split_shared)
    elif args.long_df:
        shared = rescore_shared(args.long_df, scores['electrode'],
                                n_splits=args.shared_n_splits, seed=args.seed)
        if shared is not None:
            shared.to_csv(os.path.join(out_dir, 'per_split_shared.csv'), index=False)
    rt = pd.read_csv(args.rt_coupling) if args.rt_coupling else None
    if rt is not None and 'rt_r' not in rt.columns:
        raise SystemExit(f"{args.rt_coupling} has no rt_r column")

    lines, out = sfa.section19(scores, per_split, out_dir, seg_dir=args.seg_dir, axis=args.axis,
                               n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
                               sections=tuple(int(x) for x in args.sections.split(',')),
                               per_split_shared=shared, rt_coupling=rt,
                               shared_label=shared_label, make_brain=args.brain,
                               brain_kwargs=dict(hemi=args.brain_hemi, zoom=args.brain_zoom))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary_section19.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'section19.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"written to {out_dir}")


if __name__ == '__main__':
    main()
