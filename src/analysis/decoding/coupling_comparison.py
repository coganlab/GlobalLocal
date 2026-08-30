"""Compare decoding accuracy between the coupled set and its matched controls.

The decoding battery writes one ``MASTER_RESULTS`` pickle per electrode set,
into ``<save_dir>/elecset_<slug>/`` (`decoding_dcc.run_decoding_for_one_electrode_set`).
For a coupling run those sets are ``coupled`` plus ``uncoupled_draw00 …
uncoupled_drawNN``. This module reads them back and answers the actual question:
**is the coupled set's decoding accuracy higher than you would get from an
arbitrary matched set of non-coupled electrodes?**

Why a new comparison was needed
-------------------------------
``accuracy_stats.do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions``
compares two *condition comparisons within one electrode set*. Here the
condition comparison is held fixed and the *electrode set* varies, and one side
of the contrast is a distribution over random draws rather than over bootstraps.
Nothing in the existing stack does that.

Two tests, from one set of runs
-------------------------------
**Exceedance (primary).** The coupled set is fixed, so it has one accuracy
trace; the controls give ``R`` draw-level traces. Ask where the coupled trace
falls in that distribution, per time window, and cluster-correct across time.
This is exactly the ``(series, distribution)`` shape that
``find_significant_clusters_of_series_vs_distribution_based_on_percentile``
already implements for true-vs-shuffle, reused with the draw distribution in
place of the shuffle distribution. It is the honest null for this design: "an
arbitrary matched set of n non-coupled electrodes".

**Pooled two-sample (secondary).** Pool every control draw's samples and run the
ordinary ``time_perm_cluster`` against the coupled set's samples. This is more
familiar and more powerful, but it treats draws and bootstraps as exchangeable
samples and so understates the electrode-draw variance. Reported alongside, not
instead.

A one-sided exceedance p-value of ``(1 + #{draw >= coupled}) / (R + 1)`` has a
floor of ``1/(R+1)``: with 20 draws nothing can beat p = 0.048. Raise
``coupling_n_draws`` if you need headroom below that.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from collections import OrderedDict

import numpy as np

from src.analysis.decoding.coupling_electrode_selection import (
    COUPLED_SET_NAME,
    is_control_set_name,
)

MASTER_RESULTS_GLOB = '*MASTER_RESULTS*.pkl'
ELECSET_DIR_PREFIX = 'elecset_'


# ---------------------------------------------------------------------------
# finding and loading per-set results
# ---------------------------------------------------------------------------
def find_electrode_set_results(base_save_dir, set_names=None):
    """``{set_slug: path}`` for every ``elecset_*`` directory with results.

    When a directory holds several ``MASTER_RESULTS`` pickles (a re-run), the
    most recently modified one wins.
    """
    if not os.path.isdir(base_save_dir):
        raise FileNotFoundError(f"decoding output directory not found: {base_save_dir}")

    found = {}
    for entry in sorted(os.listdir(base_save_dir)):
        if not entry.startswith(ELECSET_DIR_PREFIX):
            continue
        slug = entry[len(ELECSET_DIR_PREFIX):]
        if set_names is not None and slug not in set_names:
            continue
        matches = glob.glob(os.path.join(base_save_dir, entry, MASTER_RESULTS_GLOB))
        if not matches:
            continue
        found[slug] = max(matches, key=os.path.getmtime)
    return found


def load_set_results(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def load_coupling_run(base_save_dir, set_names=None):
    """Load the coupled set and every control draw from one decoding run."""
    paths = find_electrode_set_results(base_save_dir, set_names=set_names)
    if COUPLED_SET_NAME not in paths:
        raise FileNotFoundError(
            f"no '{COUPLED_SET_NAME}' electrode-set results under {base_save_dir}; "
            f"found {sorted(paths)}. Did the coupling decoding run finish?")
    draws = sorted(name for name in paths if is_control_set_name(name))
    if not draws:
        raise FileNotFoundError(
            f"no '{COUPLED_SET_NAME}' control draws under {base_save_dir}; "
            "the comparison needs at least two draws to have a distribution.")

    results = OrderedDict()
    results[COUPLED_SET_NAME] = load_set_results(paths[COUPLED_SET_NAME])
    for name in draws:
        results[name] = load_set_results(paths[name])
    return results, paths


# ---------------------------------------------------------------------------
# pulling accuracy traces out of the per-set results
# ---------------------------------------------------------------------------
def set_accuracy_samples(master_results, condition_comparison, roi):
    """``(n_samples, n_windows)`` true accuracies for one set, or ``None``.

    The sample axis is whatever ``unit_of_analysis`` the run used (bootstrap,
    repeat or fold) — the key is written by
    ``accuracy_stats.compute_pooled_bootstrap_statistics``.
    """
    stats = (master_results or {}).get('stats', {})
    by_roi = stats.get(condition_comparison, {})
    entry = by_roi.get(roi)
    if not entry:
        return None, None
    unit = entry.get('unit_of_analysis')
    accs = entry.get(f'{unit}_true_accs')
    if accs is None:
        return None, unit
    return np.asarray(accs, dtype=float), unit


def time_window_centers(master_results):
    return (master_results or {}).get('metadata', {}).get('time_window_centers')


def collect_accuracies(results_by_set, condition_comparison, roi):
    """Coupled trace + per-draw traces for one comparison and ROI.

    Returns ``(coupled_samples, draw_samples, draw_names, unit)`` where
    ``draw_samples`` is a list of ``(n_samples, n_windows)`` arrays. Sets with no
    data for this comparison/ROI (an empty electrode set, say) are dropped and
    reported by their absence from ``draw_names``.
    """
    coupled, unit = set_accuracy_samples(
        results_by_set[COUPLED_SET_NAME], condition_comparison, roi)
    draw_samples, draw_names = [], []
    for name, results in results_by_set.items():
        if not is_control_set_name(name):
            continue
        accs, this_unit = set_accuracy_samples(results, condition_comparison, roi)
        if accs is None:
            continue
        if unit is not None and this_unit is not None and this_unit != unit:
            raise ValueError(
                f"electrode sets disagree on unit_of_analysis for "
                f"{condition_comparison}/{roi}: '{unit}' vs '{this_unit}'. They "
                "must come from the same run configuration to be comparable.")
        draw_samples.append(accs)
        draw_names.append(name)
    return coupled, draw_samples, draw_names, unit


def draw_mean_traces(draw_samples):
    """``(R, n_windows)`` — one mean accuracy trace per control draw."""
    if not draw_samples:
        return np.empty((0, 0))
    widths = {accs.shape[1] for accs in draw_samples}
    if len(widths) > 1:
        raise ValueError(f"control draws have differing numbers of time windows: {widths}")
    return np.vstack([accs.mean(axis=0) for accs in draw_samples])


def exceedance_pvalues(series, distribution):
    """One-sided p per time window: how often a draw matches or beats coupled.

    ``(1 + #{draw >= coupled}) / (R + 1)`` — the ``+1`` counts the observed
    value itself, which keeps the p-value from ever being exactly zero and makes
    it valid for finite ``R``.
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    distribution = np.asarray(distribution, dtype=float)
    if distribution.size == 0:
        return np.full(series.shape, np.nan)
    n_draws = distribution.shape[0]
    exceed = (distribution >= series[None, :]).sum(axis=0)
    return (1.0 + exceed) / (n_draws + 1.0)


# ---------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------
def compare_one(results_by_set, condition_comparison, roi,
                percentile=95, cluster_percentile=95, n_cluster_perms=1000,
                p_thresh=0.05, p_cluster=0.05, n_perm=500, random_state=42,
                n_jobs=-1, run_pooled_test=True):
    """Both tests for one condition comparison in one ROI.

    Returns ``None`` when either side is missing (an empty electrode set), so a
    caller can skip a ROI the pair type does not span without special-casing it.
    """
    # Imported here rather than at module import: these pull in `ieeg` and
    # `mne`, and the pure aggregation above is worth being able to import (and
    # unit-test) without them.
    from ieeg.calc.stats import time_perm_cluster
    from src.analysis.decoding.accuracy_stats import (
        find_significant_clusters_of_series_vs_distribution_based_on_percentile,
    )

    coupled, draw_samples, draw_names, unit = collect_accuracies(
        results_by_set, condition_comparison, roi)
    if coupled is None or len(draw_samples) < 2:
        return None

    centers = time_window_centers(results_by_set[COUPLED_SET_NAME])
    if centers is None:
        raise ValueError(
            "the coupled set's results carry no time_window_centers; the "
            "comparison cannot align traces without them.")
    centers = np.asarray(centers, dtype=float)

    coupled_mean = coupled.mean(axis=0, keepdims=True)     # (1, n_windows)
    draw_means = draw_mean_traces(draw_samples)            # (R, n_windows)
    if draw_means.shape[1] != coupled_mean.shape[1]:
        raise ValueError(
            f"coupled set has {coupled_mean.shape[1]} time windows but the "
            f"control draws have {draw_means.shape[1]}; the sets must come from "
            "the same run configuration.")

    # --- primary: coupled trace vs the distribution over draws ---------------
    cluster_indices = find_significant_clusters_of_series_vs_distribution_based_on_percentile(
        series=coupled_mean,
        distribution=draw_means,
        time_points=centers,
        percentile=percentile,
        cluster_percentile=cluster_percentile,
        n_cluster_perms=n_cluster_perms,
        random_state=random_state,
    )
    exceedance_mask = np.zeros(len(centers), dtype=bool)
    for start, end in cluster_indices:
        exceedance_mask[start:end + 1] = True

    result = {
        'condition_comparison': condition_comparison,
        'roi': roi,
        'unit_of_analysis': unit,
        'n_draws': int(draw_means.shape[0]),
        'draw_names': list(draw_names),
        'time_window_centers': centers,
        'coupled_samples': coupled,
        'coupled_mean': coupled_mean.reshape(-1),
        'draw_means': draw_means,
        'uncoupled_mean': draw_means.mean(axis=0),
        'accuracy_difference': coupled_mean.reshape(-1) - draw_means.mean(axis=0),
        'exceedance_p': exceedance_pvalues(coupled_mean, draw_means),
        'exceedance_clusters': cluster_indices,
        'exceedance_significant': exceedance_mask,
    }

    # --- secondary: pooled two-sample cluster test ---------------------------
    if run_pooled_test:
        pooled = np.vstack(draw_samples)
        sig, pvals = time_perm_cluster(
            coupled, pooled,
            p_thresh=p_thresh, p_cluster=p_cluster, n_perm=n_perm,
            tails=1, axis=0, n_jobs=n_jobs, seed=random_state,
        )
        result['pooled_uncoupled_samples'] = pooled
        result['pooled_significant'] = np.asarray(sig, dtype=bool)
        result['pooled_p'] = np.asarray(pvals, dtype=float)

    return result


def summarize_comparison(result, decimals=4):
    """Compact, JSON-safe view of :func:`compare_one` for a report file."""
    if result is None:
        return None
    centers = np.asarray(result['time_window_centers'], dtype=float)
    mask = np.asarray(result['exceedance_significant'], dtype=bool)
    windows = []
    if mask.any():
        edges = np.diff(mask.astype(int))
        starts = list(np.flatnonzero(edges == 1) + 1)
        ends = list(np.flatnonzero(edges == -1))
        if mask[0]:
            starts = [0] + starts
        if mask[-1]:
            ends = ends + [len(mask) - 1]
        windows = [[float(centers[s]), float(centers[e])] for s, e in zip(starts, ends)]
    return {
        'condition_comparison': result['condition_comparison'],
        'roi': result['roi'],
        'unit_of_analysis': result['unit_of_analysis'],
        'n_draws': result['n_draws'],
        'coupled_mean_accuracy': round(float(np.mean(result['coupled_mean'])), decimals),
        'uncoupled_mean_accuracy': round(float(np.mean(result['uncoupled_mean'])), decimals),
        'mean_accuracy_difference': round(float(np.mean(result['accuracy_difference'])), decimals),
        'peak_accuracy_difference': round(float(np.max(result['accuracy_difference'])), decimals),
        'min_exceedance_p': round(float(np.nanmin(result['exceedance_p'])), decimals),
        'exceedance_p_floor': round(1.0 / (result['n_draws'] + 1.0), decimals),
        'significant_windows_sec': windows,
        'any_significant': bool(mask.any()),
    }


def plot_comparison(result, args, save_dir, filename_suffix='',
                    electrode_set_desc=None):
    """Coupled trace over the control-draw distribution, cluster bar included."""
    from src.analysis.decoding.plots.accuracies import plot_accuracies_nature_style
    from src.analysis.decoding.anova_electrode_selection import decoding_figure_title

    colors = {'coupled': '#0173B2', 'uncoupled (matched draws)': '#DE8F05'}
    linestyles = {'coupled': '-', 'uncoupled (matched draws)': '--'}
    title = decoding_figure_title(
        result['condition_comparison'], result['roi'],
        electrode_set_desc or f"coupled vs {result['n_draws']} matched "
                              f"non-coupled draws")

    plot_accuracies_nature_style(
        time_points=np.asarray(result['time_window_centers'], dtype=float),
        accuracies_dict={
            'coupled': result['coupled_samples'],
            'uncoupled (matched draws)': result['draw_means'],
        },
        significant_clusters=result['exceedance_significant'],
        window_size=args.window_size,
        step_size=args.step_size,
        sampling_rate=args.sampling_rate,
        comparison_name=f"coupled_vs_uncoupled_{result['condition_comparison']}",
        roi=result['roi'],
        save_dir=save_dir,
        timestamp=getattr(args, 'timestamp', None),
        colors=colors,
        linestyles=linestyles,
        title=title,
        ylim=(0.3, 1.0),
        show_chance_level=False,
        single_column=getattr(args, 'single_column', True),
        show_legend=getattr(args, 'show_legend', True),
        filename_suffix=filename_suffix,
    )


def run_coupling_comparison(base_save_dir, args, condition_comparisons, rois,
                            out_dirname='coupled_vs_uncoupled', make_plots=True,
                            expect_n_draws=None):
    """Compare every condition comparison × ROI and write a report.

    ``base_save_dir`` is the directory holding the ``elecset_*`` subdirectories,
    i.e. the same ``save_dir`` the decoding run used.

    ``expect_n_draws`` is the draw count the run was designed around. Pass it
    when the draws were decoded by several jobs: this function is happy to
    compare against whatever draws it finds, and a chunk that died leaves a
    smaller null distribution and a higher p-value floor with nothing in the
    output to say a draw is missing.
    """
    results_by_set, paths = load_coupling_run(base_save_dir)
    out_dir = os.path.join(base_save_dir, out_dirname)
    os.makedirs(out_dir, exist_ok=True)

    n_draws = sum(1 for name in results_by_set if is_control_set_name(name))
    if expect_n_draws is not None and n_draws != expect_n_draws:
        found = sorted(name for name in results_by_set if is_control_set_name(name))
        raise ValueError(
            f"expected {expect_n_draws} control draws under {base_save_dir} but "
            f"found {n_draws}: {found}. A missing draw shrinks the null the "
            f"coupled trace is tested against and raises the p-value floor from "
            f"{1 / (expect_n_draws + 1):.3f} to {1 / (n_draws + 1):.3f}. Re-run "
            "the missing chunk, or pass the true draw count deliberately.")
    print(f"\n{'='*20} COUPLED vs UNCOUPLED ({n_draws} matched draws) {'='*20}\n")
    print(f"  reading electrode-set results from {base_save_dir}")

    summary, all_results = {}, {}
    for comparison in condition_comparisons:
        summary[comparison] = {}
        all_results[comparison] = {}
        for roi in rois:
            result = compare_one(
                results_by_set, comparison, roi,
                percentile=getattr(args, 'percentile', 95),
                cluster_percentile=getattr(args, 'cluster_percentile', 95),
                n_cluster_perms=getattr(args, 'n_cluster_perms', 1000),
                n_perm=getattr(args, 'n_cluster_perms', 500),
                random_state=getattr(args, 'random_state', 42),
                n_jobs=getattr(args, 'n_jobs', -1),
            )
            if result is None:
                print(f"  – {comparison} / {roi}: no data on both sides, skipped")
                continue
            all_results[comparison][roi] = result
            summary[comparison][roi] = summarize_comparison(result)
            row = summary[comparison][roi]
            verdict = ('SIGNIFICANT' if row['any_significant'] else 'n.s.')
            print(f"  · {comparison} / {roi}: coupled {row['coupled_mean_accuracy']:.3f} "
                  f"vs uncoupled {row['uncoupled_mean_accuracy']:.3f} "
                  f"(Δ {row['mean_accuracy_difference']:+.3f}, "
                  f"min p = {row['min_exceedance_p']:.3f}) — {verdict}")

            if make_plots:
                plot_comparison(
                    result, args,
                    save_dir=os.path.join(out_dir, comparison, roi),
                    filename_suffix=getattr(args, 'condition_label', ''))

    summary_path = os.path.join(out_dir, 'coupled_vs_uncoupled_summary.json')
    with open(summary_path, 'w') as handle:
        json.dump({'source_dirs': paths, 'comparisons': summary}, handle, indent=2)
    print(f"\n  summary written to {summary_path}")

    results_path = os.path.join(out_dir, 'coupled_vs_uncoupled_results.pkl')
    with open(results_path, 'wb') as handle:
        pickle.dump(all_results, handle)
    print(f"  full traces written to {results_path}")

    return all_results, summary
