"""Top-fraction electrode sets on held-out trials, for decoding and power traces.

Splits each subject's correct trials ``--frac-select`` / rest (stratified on
congruency, task sequence and block type, keyed on ``trial_count``), scores
every candidate electrode on the selection share (LWPC, LWPS, congruency and
switch type: ``naive_sensitivities``, window-mean Cohen's d), and keeps each
score's top ``--top-frac`` per ROI (``--sign``: positive, negative or absolute).
Writes into ``--out-dir``:

* ``score_selection.json``: the sets, the trial split and the parameters. Pass
  it to the decoding job (``ELECTRODE_SETS_JSON``,
  ``submit_decoding_with_score_electrode_sets_dcc.sh``) and the power-traces
  job (``ELECTRODE_SETS_JSON`` + ``ELECTRODE_SET``); both then use only the
  other trials. ``--frac-select 1`` selects on every trial and leaves the jobs
  every trial too: circular, so the sets' effects are inflated by selection.
* ``selection_scores.csv``: every candidate's scores and set membership.

Docs: ``docs/decoding.md`` §3.7.

    python dcc_scripts/decoding/score_electrode_selection_dcc.py --rois lpfc
    python dcc_scripts/decoding/score_electrode_selection_dcc.py --rois lpfc --sign absolute
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.config.rois import select_rois  # noqa: E402
from src.analysis.decoding.anova_electrode_selection import (  # noqa: E402
    DEFAULT_STRATA_COLS, assign_trial_partitions, collect_subject_trial_strata)
from src.analysis.decoding.score_electrode_selection import (  # noqa: E402
    save_score_selection, score_electrode_sets)
from src.analysis.stats.stability_flexibility_segregation import naive_sensitivities  # noqa: E402
from src.analysis.utils.ranking import SIGNS  # noqa: E402
from src.analysis.utils.general_utils import (  # noqa: E402
    get_default_LAB_root, get_sig_chans_per_subject, load_HG_ev1_rescaled_per_subject,
    load_subjects_electrodes_to_ROIs_dict, make_sig_electrodes_per_subject_and_roi_dict)
from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df  # noqa: E402
from src.analysis.config.subjects import resolve_subject_set  # noqa: E402

# the decoding and power-traces runs' subjects: SUBJECT_SET in the environment
# (src/analysis/config/subjects.py, default all24); a set other than the default
# adds its suffix to the output folder
SUBJECTS, SUBJECT_SET_SUFFIX = resolve_subject_set()
EPOCHS_ROOT_FILE = ('Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_'
                    '70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rois', default='lpfc', help='comma-separated names from config/rois.py')
    ap.add_argument('--electrodes', default='all', choices=('all', 'sig'),
                    help='candidates: every ROI electrode, or the task-significant ones')
    ap.add_argument('--frac-select', type=float, default=0.3,
                    help='share of each subject\'s trials used to select; 1 = every trial, '
                         'which the later jobs then also measure on (circular)')
    ap.add_argument('--top-frac', type=float, default=0.2,
                    help='share of each ROI\'s electrodes kept per score')
    ap.add_argument('--sign', default='positive', choices=SIGNS)
    ap.add_argument('--tmin', type=float, default=0.0, help='score window start (s)')
    ap.add_argument('--tmax', type=float, default=1.5, help='score window end (s)')
    ap.add_argument('--strata', default=','.join(DEFAULT_STRATA_COLS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--subjects', default=','.join(SUBJECTS))
    ap.add_argument('--epochs-root-file', default=os.environ.get('EPOCHS_ROOT_FILE',
                                                                 EPOCHS_ROOT_FILE))
    ap.add_argument('--task', default='GlobalLocal')
    ap.add_argument('--lab-root', default=None)
    ap.add_argument('--out-dir', default=None,
                    help='default: dcc_scripts/decoding/figs/<epochs>/score_selections/<params>')
    return ap.parse_args(argv)


def main(args):
    lab_root = args.lab_root or get_default_LAB_root()
    subjects = args.subjects.split(',')
    rois_dict = select_rois(args.rois)
    tag = (f"{args.rois.replace(',', '-')}_{args.electrodes}_sel{args.frac_select:g}_"
           f"top{args.top_frac:g}_{args.sign}_{args.tmin:g}to{args.tmax:g}s_seed{args.seed}"
           f"{SUBJECT_SET_SUFFIX}")
    out_dir = args.out_dir or os.path.join(current_script_dir, 'figs', args.epochs_root_file,
                                           'score_selections', tag)
    os.makedirs(out_dir, exist_ok=True)

    # candidates: the ROI's electrodes (or its task-significant ones)
    roi_map = load_subjects_electrodes_to_ROIs_dict(
        save_dir=os.path.join(project_root, 'src', 'analysis', 'config'),
        filename='subjects_electrodestoROIs_dict.json')
    sig_chans = get_sig_chans_per_subject(subjects, args.epochs_root_file, task=args.task,
                                          LAB_root=lab_root)
    all_el, sig_el = make_sig_electrodes_per_subject_and_roi_dict(rois_dict, roi_map, sig_chans)
    candidates = all_el if args.electrodes == 'all' else sig_el

    # one split per subject over all its correct trials, keyed on trial_count
    epochs = load_HG_ev1_rescaled_per_subject(subjects, args.epochs_root_file, args.task,
                                              LAB_root=lab_root)
    trials = collect_subject_trial_strata(
        [{s: {'all': {'HG_ev1_power_rescaled': ep}} for s, ep in epochs.items()}],
        strata_cols=tuple(args.strata.split(',')))
    partitions = assign_trial_partitions(trials, frac_select=args.frac_select, seed=args.seed)
    if args.frac_select >= 1:
        print("NOTE: --frac-select 1: the sets are selected on every trial and the decoding and "
              "power jobs measure them on the same trials, so their effects are inflated by "
              "selection (circular).")
    selection = {s: ep[np.isin(ep.metadata['trial_count'].to_numpy(), list(partitions[s]['select']))]
                 for s, ep in epochs.items() if s in partitions}

    # scores on the selection trials only, then the top fraction per ROI
    rows = []
    for roi in rois_dict:
        keep = {s: set(ch) for s, ch in candidates[roi].items() if ch}
        df = assemble_long_df({s: selection[s] for s in keep if s in selection},
                              args.tmin, args.tmax, electrodes_to_keep=keep)
        sc = naive_sensitivities(df, contrast_mode='proportion', main_effects=True)
        sc['roi'] = roi
        sc['channel'] = [e.split('-', 1)[1] for e in sc['electrode']]
        rows.append(sc)
    scores = pd.concat(rows, ignore_index=True)
    sets, scores = score_electrode_sets(scores, frac=args.top_frac, sign=args.sign)

    params = dict(vars(args), subjects=subjects, out_dir=out_dir,
                  score_columns=dict(x='lwpc', y='lwps', mx='congruency', my='switch_type'),
                  n_select_trials={s: len(p['select']) for s, p in partitions.items()},
                  n_decode_trials={s: len(p['decode']) for s, p in partitions.items()})
    path = save_score_selection(os.path.join(out_dir, 'score_selection.json'), sets,
                                partitions, params)
    scores.rename(columns=params['score_columns']).to_csv(
        os.path.join(out_dir, 'selection_scores.csv'), index=False)

    print(f"wrote {path}")
    for name, es in sets.items():
        for roi, by_sub in es.items():
            n = sum(len(c) for c in by_sub.values())
            print(f"  {name:22s} {roi}: {n} electrodes, {sum(1 for c in by_sub.values() if c)} "
                  f"subjects")
    return path


if __name__ == '__main__':
    main(parse_args())
