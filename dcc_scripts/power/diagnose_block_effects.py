#!/usr/bin/env python
"""Cross-tabulate every electrode against block type, straight from the epochs.

In a blocked design condition and block are confounded, so anything wrong with
one block's recording arrives pre-labelled as a condition effect -- before the
stimulus, where no real effect can be, and on the proportion interaction terms.

Per (electrode, block) this prints the trial count, the pre-stimulus mean, the
largest single-trial excursion and when it happened, and how many trials are
actually non-NaN at the epoch's first sample. Then it ranks electrodes by how
far one block departs from that electrode's other blocks.

Examples
--------
One subject, the electrode whose block is suspect::

    python diagnose_block_effects.py --subjects D0121 \
        --epochs-root-file "$EPOCHS_ROOT_FILE" --roi lpfc

Every subject, to see whether block-locked offsets are a class of problem
rather than a one-off::

    python diagnose_block_effects.py --all-subjects \
        --epochs-root-file "$EPOCHS_ROOT_FILE" --roi lpfc --top 5 \
        --csv block_effects.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.power.block_diagnostics import (          # noqa: E402
    block_deviation_table, block_labels_from_metadata,
    rank_block_confined_channels,
)


def _parse_bound(text):
    if text is None or str(text).lower() in ('none', 'null', ''):
        return None
    return float(text)


def load_subject_epochs(sub, epochs_root_file, task, lab_root, acc_trials_only):
    """Return the rescaled-power epochs with metadata attached."""
    from src.analysis.utils.general_utils import load_mne_objects
    from src.analysis.utils.epoch_metadata_utils import make_metadata_from_event_names

    objs = load_mne_objects(sub, epochs_root_file, task,
                            just_HG_ev1_rescaled=True, LAB_root=lab_root)
    key = 'HG_ev1_power_rescaled'
    if key not in objs:
        raise KeyError(f"{sub}: no {key} in {sorted(objs)}")
    epochs = objs[key]

    # The .fif carries metadata, but re-parse if the columns we group on are
    # absent so an older file still works.
    if epochs.metadata is None or 'incongruent_proportion' not in epochs.metadata:
        epochs.metadata = make_metadata_from_event_names(epochs)
    if acc_trials_only:
        epochs = epochs["Accuracy1.0"]
    return epochs


def roi_channels(sub, roi, roi_labels):
    """Channels of `sub` that fall in `roi`, or None to use every channel.

    Uses the same electrodes-to-ROIs mapping the power-traces script builds, so
    the electrode set here matches the one in the figures being diagnosed.
    """
    from src.analysis.utils.general_utils import (
        make_or_load_subjects_electrodes_to_ROIs_dict)

    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    mapping = make_or_load_subjects_electrodes_to_ROIs_dict(
        subjects=[sub], save_dir=config_dir,
        filename='subjects_electrodestoROIs_dict.json')
    per_sub = mapping.get(sub, {}).get('default_dict', {})
    if not per_sub:
        print(f"[roi] no ROI mapping for {sub}; using all channels")
        return None

    keep = [ch for ch, label in per_sub.items()
            if any(lbl in str(label) for lbl in roi_labels)]
    if not keep:
        print(f"[roi] no {roi} channels for {sub}; using all channels")
        return None
    return keep


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+',
                        help="Subject IDs, e.g. D0121 D0116.")
    parser.add_argument('--all-subjects', action='store_true',
                        help="Use the SUBJECTS list from run_power_traces_dcc.py.")
    parser.add_argument('--epochs-root-file',
                        default=os.environ.get('EPOCHS_ROOT_FILE'),
                        help="Defaults to $EPOCHS_ROOT_FILE.")
    parser.add_argument('--task', default='GlobalLocal')
    parser.add_argument('--lab-root', default=None)
    parser.add_argument('--roi', default=None,
                        help="ROI key from config/rois.py. Omit for all channels.")
    parser.add_argument('--window', nargs=2, metavar=('TMIN', 'TMAX'),
                        default=('None', '0.0'),
                        help="Scoring window in seconds; either bound may be "
                             "'None'. Default: None 0.0 (epoch start through "
                             "stimulus onset).")
    parser.add_argument('--acc-trials-only', action='store_true', default=True)
    parser.add_argument('--all-trials', dest='acc_trials_only',
                        action='store_false')
    parser.add_argument('--top', type=int, default=8,
                        help="Electrodes to detail per subject (default 8).")
    parser.add_argument('--csv', default=None,
                        help="Write the full (electrode, block) table here.")
    args = parser.parse_args(argv)

    if not args.epochs_root_file:
        parser.error("--epochs-root-file (or $EPOCHS_ROOT_FILE) is required")

    subjects = list(args.subjects or [])
    if args.all_subjects:
        from dcc_scripts.power.run_power_traces_dcc import SUBJECTS
        subjects = list(SUBJECTS)
    if not subjects:
        parser.error("pass --subjects or --all-subjects")

    roi_labels = None
    if args.roi:
        from src.analysis.config.rois import rois_dict
        if args.roi not in rois_dict:
            parser.error(f"--roi {args.roi} not in rois.py ({sorted(rois_dict)})")
        roi_labels = rois_dict[args.roi]

    window = (_parse_bound(args.window[0]), _parse_bound(args.window[1]))
    lo = 'epoch start' if window[0] is None else f'{window[0]:g}s'
    hi = 'epoch end' if window[1] is None else f'{window[1]:g}s'
    print(f"\nScoring window: {lo} to {hi}")
    print(f"Epochs: {args.epochs_root_file}\n")

    all_tables = []
    for sub in subjects:
        try:
            epochs = load_subject_epochs(sub, args.epochs_root_file, args.task,
                                         args.lab_root, args.acc_trials_only)
        except Exception as exc:                      # noqa: BLE001
            print(f"[{sub}] skipped: {exc}")
            continue

        picks = roi_channels(sub, args.roi, roi_labels) if roi_labels else None
        if picks:
            picks = [ch for ch in picks if ch in epochs.ch_names]
            if not picks:
                print(f"[{sub}] no {args.roi} channels present in the epochs; skipped")
                continue
            epochs = epochs.copy().pick(picks)

        labels, missing = block_labels_from_metadata(epochs.metadata)
        if missing:
            print(f"[{sub}] metadata is missing {missing}; grouping on what is there")

        table = block_deviation_table(
            epochs.get_data(), labels, epochs.times,
            ch_names=epochs.ch_names, baseline_window=window)
        table.insert(0, 'subject', sub)
        all_tables.append(table)

        ranked = rank_block_confined_channels(table.drop(columns='subject'))
        print(f"[{sub}] {len(epochs.ch_names)} channels, {len(epochs)} trials, "
              f"{table['block'].nunique()} blocks")
        print(f"{'channel':<14}{'peak_ratio':>12}{'peak_block':>16}"
              f"{'block_offset':>14}{'worst_block':>16}")
        for _, row in ranked.head(args.top).iterrows():
            print(f"{row['channel']:<14}{row['peak_ratio']:>12.4g}"
                  f"{str(row['peak_block']):>16}{row['block_offset']:>14.4g}"
                  f"{str(row['worst_block']):>16}")

        worst = ranked.iloc[0]['channel'] if len(ranked) else None
        if worst is not None:
            print(f"\n  per-block detail for {worst}:")
            detail = table[table['channel'] == worst]
            print(f"  {'block':>16}{'n':>6}{'n@first':>9}{'baseline':>11}"
                  f"{'peak|trial|':>13}{'at (s)':>9}")
            for _, row in detail.iterrows():
                print(f"  {str(row['block']):>16}{row['n_trials']:>6}"
                      f"{row['n_valid_first_sample']:>9}{row['baseline_mean']:>11.4g}"
                      f"{row['peak_abs_trial']:>13.4g}"
                      f"{row['peak_abs_trial_time']:>9.3f}")
        print()

    if not all_tables:
        print("No subjects produced a table.")
        return 1

    combined = pd.concat(all_tables, ignore_index=True)
    if args.csv:
        combined.to_csv(args.csv, index=False)
        print(f"Wrote {len(combined)} rows to {args.csv}")

    if len(subjects) > 1:
        print("\nMost block-confined electrodes across all subjects:")
        per_sub = []
        for sub, group in combined.groupby('subject', sort=False):
            ranked = rank_block_confined_channels(group.drop(columns='subject'))
            ranked.insert(0, 'subject', sub)
            per_sub.append(ranked)
        overall = pd.concat(per_sub, ignore_index=True)
        overall = overall.sort_values('peak_ratio', ascending=False,
                                      na_position='last')
        print(f"{'subject':<10}{'channel':<14}{'peak_ratio':>12}"
              f"{'peak_block':>16}{'block_offset':>14}")
        for _, row in overall.head(20).iterrows():
            print(f"{row['subject']:<10}{row['channel']:<14}"
                  f"{row['peak_ratio']:>12.4g}{str(row['peak_block']):>16}"
                  f"{row['block_offset']:>14.4g}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
