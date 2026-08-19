#!/usr/bin/env python
"""Count coupled and eligible-control electrodes BEFORE running any decoding.

Coupled pairs are sparse — single digits per subject in the files seen so far —
and each one still has to survive the bipolar→monopolar expansion and the
intersection with the decoding pool. So the first question is not "what is the
accuracy difference" but "is there anything left to decode". This script answers
that in seconds, without loading a single epochs file.

Run it for each pair type you plan to decode and read the per-ROI totals::

    python dcc_scripts/decoding/report_coupling_counts.py \
        --csv_dir /path/to/derivatives/freqFilt/figs/high_corr_stats \
        --pair_types lpfc-occ lpfc-lpfc occ-occ \
        --condition Stimulus \
        --epochs_root_file Stimulus_1sec_preStimulusBase_decFactor_10

What the numbers mean
---------------------
``n_coupled``        electrodes in >=1 significant pair, after expansion and
                     intersection with the pool. This is the decoding set size.
``n_control``        tested-but-never-significant electrodes available to draw
                     matched controls from. Must be at least ``n_coupled``, and
                     comfortably more if the draws are to differ from each other.
``ratio``            control:coupled. Below ~2x, independent draws overlap so
                     heavily that the "distribution over draws" is nearly a
                     point mass and the exceedance test loses its meaning.

The counts here are a slight OVER-estimate of what the decode will see: this
script does not open the epochs, so it cannot drop channels that preprocessing
removed. ``decoding_dcc`` applies that filter
(`filter_electrode_lists_against_subjects_mne_objects`) and prints the final
numbers in the job log.
"""

import argparse
import json
import os
import sys

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.config.rois import rois_dict as DEFAULT_ROIS_DICT
from src.analysis.decoding.coupling_electrode_selection import (
    build_coupling_electrode_sets,
    contacts_by_roi,
    format_pool_summary,
    load_coupling_table,
    pair_type_rois,
)

DEFAULT_SUBJECTS = [
    'D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077', 'D0090', 'D0094',
    'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121',
    'D0133', 'D0134', 'D0137', 'D0138', 'D0139A', 'D0144', 'D0145', 'D0146',
]


def build_candidate_pool(args, roi_contacts):
    """``{roi: {sub: [chan]}}`` — the same pool ``decoding_dcc`` starts from.

    With ``--electrodes all`` this is just the ROI membership, so the script
    runs anywhere. ``--electrodes sig`` additionally intersects with the saved
    task-responsiveness channels, which needs the BIDS layout and therefore the
    cluster.
    """
    pool = {roi: {sub: sorted(contacts) for sub, contacts in by_sub.items()}
            for roi, by_sub in roi_contacts.items()}
    if args.electrodes == 'all':
        return pool

    from src.analysis.utils.general_utils import get_sig_chans_per_subject
    sig_chans = get_sig_chans_per_subject(
        args.subjects, args.epochs_root_file, task=args.task,
        LAB_root=args.LAB_root)
    return {roi: {sub: [c for c in contacts if c in sig_chans.get(sub, [])]
                  for sub, contacts in by_sub.items()}
            for roi, by_sub in pool.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Report coupled / eligible-control electrode counts.")
    parser.add_argument('--csv_dir', required=True,
                        help="directory holding the high_corr_*.csv files")
    parser.add_argument('--pair_types', nargs='+',
                        default=['lpfc-occ', 'lpfc-lpfc', 'occ-occ'])
    parser.add_argument('--condition', default='Stimulus',
                        help="condition token in the CSV filenames that defines coupling")
    parser.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    parser.add_argument('--electrodes', choices=['sig', 'all'], default='sig',
                        help="candidate pool: task-significant electrodes, or all in ROI")
    parser.add_argument('--epochs_root_file',
                        default='Stimulus_1sec_preStimulusBase_decFactor_10',
                        help="only needed when --electrodes sig")
    parser.add_argument('--task', default='GlobalLocal')
    parser.add_argument('--LAB_root', default=None)
    parser.add_argument('--roi_json', default=None,
                        help="subjects_electrodestoROIs_dict.json "
                             "(default: src/analysis/config/)")
    parser.add_argument('--n_draws', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--match_level', choices=['global', 'within_subject'],
                        default='global')
    parser.add_argument('--match_degree', action='store_true')
    parser.add_argument('--out', default=None,
                        help="write the full JSON report here")
    args = parser.parse_args()

    # decoding_dcc reads this from src/analysis/config; a copy also ships with
    # the PAC path. Take whichever exists so this runs off-cluster too.
    candidates = [args.roi_json] if args.roi_json else [
        os.path.join(project_root, 'src', 'analysis', 'config',
                     'subjects_electrodestoROIs_dict.json'),
        os.path.join(project_root, 'src', 'analysis', 'pac',
                     'subjects_electrodestoROIs_dict.json'),
    ]
    roi_json = next((p for p in candidates if p and os.path.isfile(p)), None)
    if roi_json is None:
        raise FileNotFoundError(
            f"subjects_electrodestoROIs_dict.json not found in {candidates}; "
            "pass --roi_json")
    print(f"electrode→ROI dict: {roi_json}")
    with open(roi_json) as handle:
        subjects_electrodes_to_rois_dict = json.load(handle)

    args.rois_dict = DEFAULT_ROIS_DICT
    roi_contacts = contacts_by_roi(
        subjects_electrodes_to_rois_dict, DEFAULT_ROIS_DICT, subjects=set(args.subjects))
    pool = build_candidate_pool(args, roi_contacts)

    reports = {}
    for pair_type in args.pair_types:
        print(f"\n{'='*70}\n{pair_type}\n{'='*70}")
        try:
            table, used = load_coupling_table(
                args.csv_dir, pair_type, args.condition, subjects=args.subjects)
        except FileNotFoundError as exc:
            print(f"  ✗ {exc}")
            continue

        _, report = build_coupling_electrode_sets(
            table=table,
            pair_type=pair_type,
            roi_contacts=roi_contacts,
            decoding_pool=pool,
            n_draws=args.n_draws,
            seed=args.seed,
            match_level=args.match_level,
            match_degree=args.match_degree,
        )
        report['csv_files'] = used
        report['condition'] = args.condition
        reports[pair_type] = report
        print(format_pool_summary(report))

        # dict.fromkeys, not set(): a homotypic pair type names the same ROI
        # twice, and the warning should be printed once.
        for roi in dict.fromkeys(pair_type_rois(pair_type)):
            stats = report['per_roi'].get(roi)
            if not stats or not stats['n_coupled_total']:
                continue
            ratio = stats['control_to_coupled_ratio'] or 0
            if ratio < 2:
                print(f"    ⚠ {roi}: only {ratio:.1f}x controls per coupled "
                      "electrode — independent draws will overlap heavily and "
                      "the draw distribution will be near-degenerate.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as handle:
            json.dump(reports, handle, indent=2, default=str)
        print(f"\nFull report written to {args.out}")


if __name__ == '__main__':
    main()
