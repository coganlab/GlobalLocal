"""Window-mean decoding in task-significant, non-significant and all electrodes,
at a matched electrode count (or each group whole, with
``WINDOW_MEAN_MATCH_ELECTRODES=false``), with membership-permutation tests between
them.

Called by ``decoding_dcc.main`` when ``args.window_mean_groups`` is on
(``WINDOW_MEAN_GROUPS=1`` through ``run_decoding_dcc.py``); the computation is
:mod:`src.analysis.decoding.window_mean_decoding`, this file loads, saves and
plots. Docs: ``docs/decoding.md`` §3.6.
"""
import json
import os

from src.analysis.decoding.plots.electrode_curve import plot_accuracy_by_electrode_count
from src.analysis.decoding.plots.group_bars import plot_group_accuracy_bars
from src.analysis.decoding.window_mean_decoding import (
    accuracy_by_electrode_count, compare_electrode_groups, window_mean_dataset)
from src.analysis.utils.labeled_array_utils import (
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel)

# Spacing of the curve's linear grid, in electrodes (its log grid is 1, 2, 4, ...).
CURVE_STEP = 20


def electrode_groups(all_electrodes, sig_electrodes, roi):
    """``{'sig', 'nonsig', 'all'}`` -> set of ``'<subject>-<electrode>'`` names for
    one ROI: sig is the task-significant electrodes among all, nonsig the rest."""
    every = {f"{s}-{e}" for s, es in all_electrodes.get(roi, {}).items() for e in es}
    sig = {f"{s}-{e}" for s, es in sig_electrodes.get(roi, {}).items() for e in es} & every
    return {'sig': sig, 'nonsig': every - sig, 'all': every}


def run_window_mean_group_comparison(args, rois, condition_names, condition_comparisons,
                                     all_electrodes, sig_electrodes, subjects_mne_objects,
                                     save_dir):
    """Decode every comparison of the condition set from window means in each
    ROI's sig / nonsig / all electrodes and write, per comparison and ROI, into
    ``<save_dir>/window_mean_groups/<tag>/<comparison>/<roi>/``:
    ``accuracies.csv`` (group x bootstrap x repeat), ``summary.csv``,
    ``tests.csv`` (pairwise permutation p and Holm p), ``params.json`` and
    ``group_bars.png``/``.pdf``; with ``window_mean_curve_subsets`` > 0 also the
    accuracy-by-electrode-count curve, ``curve_accuracies.csv``,
    ``curve_summary.csv`` and ``electrode_curve.png``/``.pdf``.
    ``all_electrodes`` must already be filtered to the loaded epochs. Without
    electrode-count matching the tag ends in ``_unmatched``."""
    tmin, tmax = args.window_mean_tmin, args.window_mean_tmax
    n_boot = int(args.window_mean_bootstraps)
    curve_subsets = int(getattr(args, 'window_mean_curve_subsets', 0))
    match = getattr(args, 'window_mean_match_electrodes', True)
    if not match and args.window_mean_n_match is not None:
        raise ValueError("window_mean_n_match needs window_mean_match_electrodes; "
                         "unmatched, every group is decoded on all its electrodes")
    tag = (f"{args.condition_label}_{tmin:g}to{tmax:g}s_{n_boot}bts_"
           f"{args.window_mean_n_perm}perm_seed{args.random_state}"
           f"{'' if match else '_unmatched'}")
    out_root = os.path.join(save_dir, 'window_mean_groups', tag)

    # one pseudo-trial draw per bootstrap over ALL electrodes; each group is a
    # subset of its columns, so every group sees the same trials
    datasets = {(c, roi): [] for c in condition_comparisons for roi in rois}
    channels = {}
    for b in range(n_boot):
        arrays = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
            rois=rois, subjects_data_objects=subjects_mne_objects,
            condition_names=condition_names, subjects=args.subjects,
            electrodes_per_subject_roi=all_electrodes, n_bootstraps=1,
            chans_axs=args.chans_axs, time_axs=args.time_axs,
            random_state=args.random_state + b, n_jobs=1)
        for roi, arrs in arrays.items():
            if not arrs:
                continue
            for comparison, strings in condition_comparisons.items():
                X, y, cats, chans = window_mean_dataset(
                    arrs[0], strings, time_range=(tmin, tmax),
                    balance_method=args.balance_method, balance_strata=args.balance_strata,
                    random_state=args.random_state + b)
                datasets[(comparison, roi)].append((X, y, cats))
                channels[(comparison, roi)] = chans
        print(f"[window-mean groups] bootstrap {b + 1}/{n_boot} built")

    for (comparison, roi), data in datasets.items():
        if not data:
            print(f"[window-mean groups] {comparison} / {roi}: no data, skipped")
            continue
        groups = electrode_groups(all_electrodes, sig_electrodes, roi)
        res = compare_electrode_groups(
            data, channels[(comparison, roi)], groups, n_match=args.window_mean_n_match,
            match_electrodes=match, n_perm=args.window_mean_n_perm, seed=args.random_state,
            n_jobs=args.n_jobs, clf=args.clf_model, explained_variance=args.explained_variance,
            n_splits=args.n_splits, n_repeats=args.n_repeats)
        out = os.path.join(out_root, comparison, roi)
        os.makedirs(out, exist_ok=True)
        res['accuracies'].to_csv(os.path.join(out, 'accuracies.csv'), index=False)
        res['summary'].to_csv(os.path.join(out, 'summary.csv'), index=False)
        res['tests'].to_csv(os.path.join(out, 'tests.csv'), index=False)
        with open(os.path.join(out, 'params.json'), 'w') as f:
            json.dump(dict(condition_label=args.condition_label, comparison=comparison, roi=roi,
                           time_range_s=[tmin, tmax], bootstraps=n_boot,
                           n_perm=args.window_mean_n_perm, match_electrodes=match,
                           n_match=res['n_match'],
                           group_sizes=res['group_sizes'], seed=args.random_state,
                           curve_subsets=curve_subsets, curve_step=CURVE_STEP,
                           classifier=args.clf_model_str,
                           explained_variance=args.explained_variance,
                           n_splits=args.n_splits, n_repeats=args.n_repeats,
                           epochs_root_file=args.epochs_root_file, subjects=args.subjects),
                      f, indent=2)
        plot_group_accuracy_bars(res, os.path.join(out, 'group_bars'),
                                 title=f"{roi}: {comparison}, {tmin:g}-{tmax:g} s means"
                                       f"{'' if match else ', unmatched'}")
        if curve_subsets:
            curve = accuracy_by_electrode_count(
                data, channels[(comparison, roi)], groups, step=CURVE_STEP,
                n_subsets=curve_subsets, seed=args.random_state, n_jobs=args.n_jobs,
                clf=args.clf_model, explained_variance=args.explained_variance,
                n_splits=args.n_splits, n_repeats=args.n_repeats)
            curve['accuracies'].to_csv(os.path.join(out, 'curve_accuracies.csv'), index=False)
            curve['summary'].to_csv(os.path.join(out, 'curve_summary.csv'), index=False)
            plot_accuracy_by_electrode_count(
                curve, os.path.join(out, 'electrode_curve'),
                title=f"{roi}: {comparison}, {tmin:g}-{tmax:g} s means, "
                      f"{curve_subsets} random subsets x {n_boot} bootstraps per count")
        print(f"[window-mean groups] {comparison} / {roi}: "
              f"n_match = {res['n_match'] if match else 'off (each group whole)'}\n"
              f"{res['summary'].to_string(index=False)}\n{res['tests'].to_string(index=False)}")
