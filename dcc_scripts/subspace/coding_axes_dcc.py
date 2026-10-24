#!/usr/bin/env python
"""
DCC core for the coding axes and the cosine between them (docs/analysis_plans.md,
subspace plan §3.1-3.3): each electrode's congruency and switch-type effect in
every A4 decoding window on shared split halves, compared across electrodes by
the cosine, against a label-shuffle null and a split-half ceiling. On the same
axes, halves and shuffles: the subspace test (§3.4), the condition
trajectories (§4) and adaptation as block-split separations (§6).

Writes to `args.save_dir`:
    per_split_windows.csv   the coding axes: congruency in `mxA`/`mxB`, switch type
                            in `myA`/`myB` (mean differences, HG units), one row
                            per electrode x split x window, with the window grid
    null_cross.npz          the raw cross cosine under each shuffle,
                            (n_perm, windows) per factor
    coding_axis_cosines.csv per window: cross, ceilings, alignment, Pearson
                            versions, bootstrap CIs, null band, p, the marks
                            (`coding_axes.compare_coding_axes`)
    summary.json            the alignment over the reliable windows with its CI,
                            the marked windows
    coding_axis_cosines.png the §3.2 figure
    run_config.json         every parameter and seed that produced the above
    rt_*.csv                with an RT control: the HG-on-RT slopes ('adjust'), or
                            the per-participant RT differences before and after
                            and their summary ('match', 'random')

  subspace test (§3.4, `coding_subspaces.compare_subspaces`):
    subspace_test.csv       per (measure, k, component): principal-angle cosines
                            and alignment indices, cross / ceilings / normalized,
                            bootstrap CIs, null thresholds, p, the marks
    subspace_dimensions.csv per variable and dimension: held-out variance against
                            the null, which dimensions are kept (sets k)
    subspace_summary.json   k per variable, the primary k, its θ₁ and alignment
                            indices with CIs
    null_subspace.npz       per factor: the cross values, the shuffled variable's
                            ceilings and per-dimension held-out variance per shuffle
    subspace_test.png       the §3.4 figure

  trajectories (§4, `coding_trajectories.condition_trajectories`):
    trajectories.csv        per axis window, centred / uncentred, cell and window:
                            x, y (and bootstrap CIs for centred)
    trajectory_fractions.csv  the share of the centred cells' variance in the plane,
                            raw and noise-corrected, per window and overall
    trajectory_summary.json electrodes used, per axis window the cosine of its axes
                            and the overall fractions with CIs
    trajectories_<axes>.png the §4 figure per axis window (centred cells), and
                            trajectories_<axes>_uncentred.png
    trajectories_raw.csv    the same cells on the raw (not orthogonalized) unit c
                            and s axes; figures trajectories_<axes>[_uncentred]_raw.png

  adaptation (§6, `coding_trajectories.block_separations`), on the same axes:
    block_separations.csv   per axis window, variable, proportion block and
                            window: the held-out separation on the variable's
                            own unit axis, with bootstrap CI (window -1 = mean)
    block_contrasts.csv     per axis window, variable and window: the change in
                            separation with incProp, with switchProp (low - high)
                            and own - other, with bootstrap CIs
    block_summary.json      the overall (window -1) contrasts with CIs
    block_separations_<axes>.png  the §6 figure per axis window
    block_*_unique.*, block_separations_<axes>_unique.png  the same on each
                            axis's part orthogonal to the other (robustness check)

  adaptation as trajectories (§6, `coding_trajectories.proportion_trajectories`):
    proportion_trajectories.csv  per axis window, set (variable x proportion),
                            condition (e.g. i25) and window: the held-out mean on
                            the congruency and switch axes (oblique), with CIs
    proportion_changes.csv  per axis window, set, axis and window: the change in
                            separation (25 % - 75 %) along that axis, with CIs
    proportion_summary.json the overall (window -1) changes with CIs
    proportion_trajectories_<axes>.png  the 2 x 2 figure per axis window

Driven by `run_coding_axes_dcc.py` (wrapped by `sbatch_coding_axes_dcc.sh`);
call `main(args)` with a populated argument namespace.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json

# ---------------------------------------------------------------------------
# PATH SETUP (mirrors the other dcc_scripts entrypoints)
# ---------------------------------------------------------------------------
try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if os.path.exists("/hpc/home"):
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats import coding_axes as ca
from src.analysis.stats import coding_subspaces as cs
from src.analysis.stats import coding_trajectories as ct
from src.analysis.utils.general_utils import resolve_lab_root, resolve_electrodes_to_keep
from dcc_scripts.stats.stability_flexibility_segregation_dcc import (
    assemble_long_df, make_synthetic_df, _window_indices)
from dcc_scripts.subspace.subspace_figures import (
    plot_subspace_test, plot_trajectories, plot_block_separations,
    plot_proportion_trajectories, time_axis_label)

# The two scores' names per `effect`, for the figures.
VARIABLE_NAMES = {'main': ('congruency', 'switch type'), 'interaction': ('LWPC', 'LWPS')}

# Reference-palette slots 1-2 (dataviz skill), and recessive neutrals.
CROSS, CEILING = '#2a78d6', '#eb6834'
INK, MUTED, NULL_FILL, UNRELIABLE = '#0b0b0b', '#52514e', '#d9d8d4', '#f0efec'


def load_time_courses(args, LAB_root):
    """The long table with each trial's HG time course over
    [windows_tmin, windows_tmax] s, and the time (s) of each of its samples."""
    if args.data_source == 'synthetic':
        n_time = int(round((args.windows_tmax - args.windows_tmin)
                           * args.synthetic_sfreq)) + 1
        df = make_synthetic_df(rho_true=args.synthetic_rho, effect_measure='cluster',
                               n_time=n_time)
        return df, args.windows_tmin + np.arange(n_time) / args.synthetic_sfreq

    from src.analysis.utils.general_utils import load_HG_ev1_rescaled_per_subject
    subjects_epochs = load_HG_ev1_rescaled_per_subject(
        subjects=args.subjects, epochs_root_file=args.epochs_root_file,
        task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
    # One time axis for everyone, or the same window index would mean different
    # times in different participants.
    if len({tuple(ep.times) for ep in subjects_epochs.values()}) > 1:
        raise ValueError("participants' epochs have different time axes")
    times = next(iter(subjects_epochs.values())).times
    s, e = _window_indices(times, args.windows_tmin, args.windows_tmax)
    keep = resolve_electrodes_to_keep(args, LAB_root)
    df = assemble_long_df(subjects_epochs, args.windows_tmin, args.windows_tmax,
                          electrodes_to_keep=keep, time_courses=True)
    return df, times[s:e]


def _runs(mask):
    """(first, last) index of each run of True in `mask`."""
    from src.analysis.decoding.accuracy_stats import find_contiguous_clusters
    return find_contiguous_clusters(np.asarray(mask, bool))


def plot_coding_axes(table, out_path, time_label=time_axis_label(), title=None):
    """The §3.2 figure. (a) Cosines against time, A4's cross-decoding layout:
    the raw cross cosine (and its Pearson version), the ceiling, the shuffle
    null band (its top edge is the test's threshold under the stricter of the two
    shuffles), and bars where something is shared (cluster-corrected) and where
    it is not one code. (b) Alignment = cross / ceiling with its participant-
    bootstrap CI. Windows whose ceilings are not reliable are greyed out in both
    panels, and (b) leaves them blank. `title`, if given, heads panel (a)."""
    t = table['window_center'].to_numpy()
    half = np.median(np.diff(t)) / 2 if len(t) > 1 else 0.05
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12))
    for ax in (a, b):
        for start, end in _runs(~table['reliable'].to_numpy()):
            ax.axvspan(t[start] - half, t[end] + half, color=UNRELIABLE, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)

    null_lo = [c for c in table if c.startswith('null_lo_')]
    if null_lo:
        lo = table[null_lo].min(axis=1)
        hi = table[[c.replace('_lo_', '_hi_') for c in null_lo]].max(axis=1)
        a.fill_between(t, lo, hi, color=NULL_FILL, lw=0, zorder=2,
                       label='shuffle null (top edge = threshold)')
    a.plot(t, table['ceiling'], color=CEILING, lw=2, zorder=3, label='ceiling')
    a.plot(t, table['cross'], color=CROSS, lw=2, zorder=4, label='cross (cosine)')
    a.plot(t, table['cross_pearson'], color=CROSS, lw=1.5, ls='--', zorder=5,
           label='cross (Pearson r)')
    for col, name in (('cross', 'cross'), ('ceiling', 'ceiling')):
        y = table[col].to_numpy()
        last = np.flatnonzero(np.isfinite(y))
        if last.size:
            a.annotate(name, (t[last[-1]], y[last[-1]]), xytext=(4, 0),
                       textcoords='offset points', va='center', fontsize=8, color=INK)

    top = np.nanmax(table[['cross', 'ceiling']].to_numpy())
    step = 0.06 * max(top, 0.1)
    for i, (col, color, label) in enumerate((('shared', CROSS, 'shared (cross > null)'),
                                             ('not_one_code', CEILING, 'not one code (cross < ceiling)'))):
        if col not in table:
            continue
        y = top + step * (i + 1.5)
        for j, (start, end) in enumerate(_runs(table[col].to_numpy())):
            a.hlines(y, t[start] - half, t[end] + half, color=color, lw=4,
                     label=label if j == 0 else None, zorder=5)
    a.set_ylabel('cosine across electrodes', fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1))

    align = table['alignment'].where(table['reliable'])
    b.fill_between(t, table['alignment_lo'].where(table['reliable']),
                   table['alignment_hi'].where(table['reliable']),
                   color=CROSS, alpha=0.2, lw=0, zorder=2, label='95% CI (participants)')
    b.plot(t, align, color=CROSS, lw=2, zorder=3, label='alignment')
    b.axhline(1, color=MUTED, lw=0.8, ls=':', zorder=1)
    b.set_ylabel('alignment\n(cross / ceiling)', fontsize=9, color=INK)
    b.set_xlabel(time_label, fontsize=9, color=INK)
    b.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1))
    if title:
        a.set_title(title, fontsize=10, color=INK, loc='left')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# Reference-palette categorical slots 1-6, in fixed order: one per run.
RUN_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
_RT_NAMES = {'adjust': 'RT adjusted', 'match': 'RT matched', 'random': 'random subset'}


def run_label(config):
    """A short name for a run from its run_config.json: the region, then any
    control, e.g. 'lpfc', 'lpfc, remove_mean', 'lpfc, RT matched', 'occ'."""
    rois = config.get('rois', 'all')
    parts = ['+'.join(rois) if isinstance(rois, list) else str(rois)]
    if config.get('remove_mean'):
        parts.append('remove_mean')
    if config.get('rt_control', 'none') != 'none':
        parts.append(_RT_NAMES[config['rt_control']])
    return ', '.join(parts)


def load_run(run_dir):
    """``(config, table, summary)`` of one coding-axes output directory: its
    run_config.json, coding_axis_cosines.csv and summary.json."""
    with open(os.path.join(run_dir, 'run_config.json')) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, 'summary.json')) as f:
        summary = json.load(f)
    table = pd.read_csv(os.path.join(run_dir, 'coding_axis_cosines.csv'))
    return config, table, summary


def plot_coding_axes_controls(runs, out_path, time_label=time_axis_label()):
    """The main result and its §3.3 controls in one figure; `runs` is a list of
    ``(label, table)`` with `table` a run's coding_axis_cosines.csv.

    (a) the raw cross cosine of each run; (b) its alignment (cross / ceiling)
    with the participant-bootstrap CI, on reliable windows only, so runs whose
    ceilings differ are compared as shares of their own ceiling (A4's
    "retained" logic); (c) one row per run: a thin line over its reliable
    windows and a thick one where something is shared (cross above both
    shuffle nulls, cluster-corrected). Each run's own figure has its ceilings
    and null band."""
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"at most {len(RUN_COLORS)} runs per figure; got {len(runs)}")
    fig, (a, b, c) = plt.subplots(
        3, 1, figsize=(9.0, 7.5), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 3, 0.35 * len(runs) + 0.5], hspace=0.15))
    for color, (label, table) in zip(RUN_COLORS, runs):
        t = table['window_center'].to_numpy()
        reliable = table['reliable'].to_numpy(bool)
        a.plot(t, table['cross'], color=color, lw=2, label=label)
        b.fill_between(t, table['alignment_lo'].where(reliable),
                       table['alignment_hi'].where(reliable), color=color, alpha=0.15, lw=0)
        b.plot(t, table['alignment'].where(reliable), color=color, lw=2, label=label)
    half = np.median(np.diff(runs[0][1]['window_center'])) / 2
    for i, (color, (label, table)) in enumerate(zip(RUN_COLORS, runs)):
        t = table['window_center'].to_numpy()
        y = len(runs) - 1 - i
        for start, end in _runs(table['reliable']):
            c.hlines(y, t[start] - half, t[end] + half, color=color, lw=1)
        if 'shared' in table:
            for start, end in _runs(table['shared']):
                c.hlines(y, t[start] - half, t[end] + half, color=color, lw=5)
    for ax in (a, b, c):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    for ax in (a, b):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    b.axhline(1, color=MUTED, lw=0.8, ls=':', zorder=0)
    a.set_ylabel('cross cosine', fontsize=9, color=INK)
    b.set_ylabel('alignment\n(cross / ceiling)', fontsize=9, color=INK)
    c.set_yticks(range(len(runs)))
    c.set_yticklabels([label for label, _ in runs][::-1], fontsize=8)
    c.set_ylim(-0.7, len(runs) - 0.3)
    c.spines['left'].set_visible(False)
    c.tick_params(axis='y', length=0)
    c.set_title('thin: reliable windows; thick: shared (cross > null)',
                fontsize=8, color=MUTED, loc='left')
    c.set_xlabel(time_label, fontsize=9, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main_effect_extras(args, labels, hg_windows, axis_hg, axis_labels, grid, seeds,
                       time_label):
    """Steps 5-7, defined on the congruency and switch main-effect axes only:
    the condition trajectories (§4, Gram-Schmidt and raw axes), the
    block-split separations (§6, raw and unique axes) and the proportion
    trajectories (§6). Writes their tables and figures to `args.save_dir` and
    prints their summaries."""
    # 5. the trajectories (§4): axes and cells on the coding axes' own halves
    traj, fractions, traj_summary = ct.condition_trajectories(
        labels, hg_windows, axis_hg, axis_labels=axis_labels, n_splits=args.n_splits,
        seed=seeds['halves'], remove_mean=args.remove_mean, first=args.traj_first,
        n_boot=args.n_boot, alpha=args.alpha, boot_seed=seeds['trajectory_bootstrap'])
    traj = traj.merge(grid, on='window')
    fractions = fractions.merge(grid, on='window', how='left')
    traj.to_csv(os.path.join(args.save_dir, 'trajectories.csv'), index=False)
    fractions.to_csv(os.path.join(args.save_dir, 'trajectory_fractions.csv'), index=False)
    with open(os.path.join(args.save_dir, 'trajectory_summary.json'), 'w') as f:
        json.dump(traj_summary, f, indent=2)
    traj_raw, _, _ = ct.condition_trajectories(
        labels, hg_windows, axis_hg, axis_labels=axis_labels, n_splits=args.n_splits,
        seed=seeds['halves'], remove_mean=args.remove_mean, first=None,
        n_boot=args.n_boot, alpha=args.alpha, boot_seed=seeds['trajectory_bootstrap'])
    traj_raw = traj_raw.merge(grid, on='window')
    traj_raw.to_csv(os.path.join(args.save_dir, 'trajectories_raw.csv'), index=False)
    for label in axis_labels:
        for tab, first, tag in ((traj, args.traj_first, ''), (traj_raw, None, '_raw')):
            for cells, suffix in (('centred', ''), ('uncentred', '_uncentred')):
                plot_trajectories(tab, fractions, traj_summary['axes'][label], label,
                                  os.path.join(args.save_dir,
                                               f'trajectories_{label}{suffix}{tag}.png'),
                                  cells=cells, first=first, time_label=time_label)

    # 6. adaptation as block-split separations (§6, supplement): the
    #    trajectories' axes, halves and electrodes
    block_summaries = {}
    for unique, tag in ((False, ''), (True, '_unique')):
        seps, cons, block_summaries[tag] = ct.block_separations(
            labels, hg_windows, axis_hg, axis_labels=axis_labels, n_splits=args.n_splits,
            seed=seeds['halves'], remove_mean=args.remove_mean, unique=unique,
            n_boot=args.n_boot, alpha=args.alpha, boot_seed=seeds['block_bootstrap'])
        seps = seps.merge(grid, on='window', how='left')
        cons = cons.merge(grid, on='window', how='left')
        seps.to_csv(os.path.join(args.save_dir, f'block_separations{tag}.csv'), index=False)
        cons.to_csv(os.path.join(args.save_dir, f'block_contrasts{tag}.csv'), index=False)
        with open(os.path.join(args.save_dir, f'block_summary{tag}.json'), 'w') as f:
            json.dump(block_summaries[tag], f, indent=2)
        for label in axis_labels:
            plot_block_separations(seps, cons, label,
                                   os.path.join(args.save_dir,
                                                f'block_separations_{label}{tag}.png'),
                                   time_label=time_label, unique=unique)

    # 7. adaptation as trajectories (§6): each variable's two levels in the 25 %
    #    and 75 % blocks of each proportion, on the trajectories' axes and halves
    ptraj, pchanges, prop_summary = ct.proportion_trajectories(
        labels, hg_windows, axis_hg, axis_labels=axis_labels, n_splits=args.n_splits,
        seed=seeds['halves'], remove_mean=args.remove_mean, n_boot=args.n_boot,
        alpha=args.alpha, boot_seed=seeds['proportion_bootstrap'])
    ptraj = ptraj.merge(grid, on='window')
    pchanges = pchanges.merge(grid, on='window', how='left')
    ptraj.to_csv(os.path.join(args.save_dir, 'proportion_trajectories.csv'), index=False)
    pchanges.to_csv(os.path.join(args.save_dir, 'proportion_changes.csv'), index=False)
    with open(os.path.join(args.save_dir, 'proportion_summary.json'), 'w') as f:
        json.dump(prop_summary, f, indent=2)
    for label in axis_labels:
        plot_proportion_trajectories(
            ptraj, pchanges, label,
            os.path.join(args.save_dir, f'proportion_trajectories_{label}.png'))

    for label, s in traj_summary['axes'].items():
        print(f"trajectories, axes {label}: plane holds {s['plane_fraction']:.3f} of the "
              f"centred cells' variance (chance {s['plane_fraction_chance']:.3f}), "
              f"{s['plane_fraction_signal']:.2f} of the signal")
    for tag, block_summary in block_summaries.items():
        for label, by_var in block_summary['axes'].items():
            for var, s in by_var.items():
                own = s['own_proportion']
                other = 'switchProp' if own == 'incProp' else 'incProp'
                print(f"block separations{' (unique axes)' if tag else ''}, axes {label}, {var}: "
                      + ' | '.join(f"{con} {s[con]['estimate']:+.3f} "
                                   f"[{s[con]['ci'][0]:+.3f}, {s[con]['ci'][1]:+.3f}]"
                                   for con in (own, other, 'own_minus_other')))
    for label, s in prop_summary['axes'].items():
        for name, v in s['sets'].items():
            print(f"proportion trajectories, axes {label}, {name} ({'own' if v['own'] else 'cross'}): "
                  + ' | '.join(f"along {a} {v[a]['estimate']:+.3f} "
                               f"[{v[a]['ci'][0]:+.3f}, {v[a]['ci'][1]:+.3f}]"
                               for a in ('congruency', 'switch')))


def main(args):
    effect = getattr(args, 'effect', 'main')
    LAB_root = resolve_lab_root(args.LAB_root)
    df, times = load_time_courses(args, LAB_root)
    print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
          f"{df.electrode.nunique()} electrodes | {len(times)} samples over "
          f"[{times[0]:.3f}, {times[-1]:.3f}] s")

    # Each trial's window means, and its mean over each trajectory axis window
    # (§4); the time courses are not needed after this.
    hg_windows, grid = sfs.window_means(df['hg'].to_numpy(), times,
                                        args.window_size, args.step_size)
    axis_hg = ct.axis_window_means(df['hg'].to_numpy(), times, args.traj_axes)
    axis_labels = [f'{tmin}-{tmax}s' for tmin, tmax in args.traj_axes]
    labels = df.drop(columns='hg')
    del df
    print(f"{len(grid)} windows of {args.window_size} samples every {args.step_size}")

    # Seeds, all derived from one so a run is reproducible from `args.seed`.
    seeds = dict(halves=args.seed, null_congruency=args.seed + 1,
                 null_switchType=args.seed + 2, bootstrap=args.seed + 3,
                 rt_match=args.seed + 4, subspace_bootstrap=args.seed + 5,
                 trajectory_bootstrap=args.seed + 6, block_bootstrap=args.seed + 7,
                 proportion_bootstrap=args.seed + 8)
    os.makedirs(args.save_dir, exist_ok=True)

    # 0. the RT control (§3.3), before anything is scored, so the real axes, every
    #    null draw and the trajectories see the same trials and HG.
    n_win = hg_windows.shape[1]
    labels, controlled, rt_report = ca.rt_control(
        labels, np.column_stack([hg_windows, axis_hg]), mode=args.rt_control,
        n_bins=args.rt_match_bins, seed=seeds['rt_match'])
    hg_windows, axis_hg = controlled[:, :n_win], controlled[:, n_win:]
    if 'rt_adjust_slopes' in rt_report:                  # the analysis windows only
        slopes = rt_report['rt_adjust_slopes']
        rt_report['rt_adjust_slopes'] = slopes[slopes['window'] < n_win]
    for name, frame in rt_report.items():
        frame.to_csv(os.path.join(args.save_dir, f'{name}.csv'), index=False)
    if args.rt_control != 'none':
        print(f"RT control '{args.rt_control}': {len(labels)} rows | "
              f"{labels.electrode.nunique()} electrodes")
        for name, frame in rt_report.items():
            if name.endswith('_summary'):
                print(frame.round(3).to_string(index=False))

    # 1. the coding axes (§3.1) -- proportion blocks weighted equally (W_MAIN), as
    #    in the main-effect anatomy; one split per participant shared by all of its
    #    electrodes, which the cross-electrode cosines need.
    per = sfs.main_effect_mean_diffs(labels, hg_windows, n_splits=args.n_splits,
                                     seed=seeds['halves'], remove_mean=args.remove_mean,
                                     effect=effect)
    per = per.merge(grid, on='window')
    per.to_csv(os.path.join(args.save_dir, 'per_split_windows.csv'), index=False)

    # 2. the shuffle nulls (§3.2), one factor at a time; each shuffle is scored
    #    once and feeds both the cosine null and the subspace null (§3.4)
    null, sub_null = {}, {}
    for factor in ca.NULL_FACTORS:
        print(f"null: shuffling {factor} ({args.n_perm} permutations, "
              f"{args.n_null_splits} splits each)")
        cross, records = [], []
        for per_null in ca.null_draws(
                labels, hg_windows, factor, n_perm=args.n_perm, n_splits=args.n_null_splits,
                seed=seeds[f'null_{factor}'], remove_mean=args.remove_mean, effect=effect):
            cross.append(ca.cross_cosine(per_null))
            records.append(cs.subspace_null_stats(per_null, factor, k_max=args.k_max))
        null[factor] = np.array(cross)
        sub_null[factor] = cs.stack_null(records)
    np.savez(os.path.join(args.save_dir, 'null_cross.npz'),
             window=grid['window'].to_numpy(), **null)
    np.savez(os.path.join(args.save_dir, 'null_subspace.npz'),
             **{f'{f}_{key}': v for f, d in sub_null.items() for key, v in d.items()})

    # 3. cosines, ceilings, alignment, bootstrap, marks
    table, summary = ca.compare_coding_axes(
        per, null=null, n_boot=args.n_boot, seed=seeds['bootstrap'],
        alpha=args.alpha)
    table.to_csv(os.path.join(args.save_dir, 'coding_axis_cosines.csv'), index=False)
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    time_label = time_axis_label(args.epochs_root_file)
    plot_coding_axes(table, os.path.join(args.save_dir, 'coding_axis_cosines.png'),
                     time_label=time_label,
                     title=None if effect == 'main' else 'LWPC vs LWPS coding axes')

    # 4. the subspace test (§3.4): the windows' axes stacked into effect matrices
    sub_table, sub_dims, sub_summary = cs.compare_subspaces(
        per, null=sub_null, k_max=args.k_max, n_boot=args.n_boot,
        seed=seeds['subspace_bootstrap'], alpha=args.alpha)
    sub_table.to_csv(os.path.join(args.save_dir, 'subspace_test.csv'), index=False)
    sub_dims.to_csv(os.path.join(args.save_dir, 'subspace_dimensions.csv'), index=False)
    with open(os.path.join(args.save_dir, 'subspace_summary.json'), 'w') as f:
        json.dump(sub_summary, f, indent=2)
    plot_subspace_test(sub_table, sub_dims, os.path.join(args.save_dir, 'subspace_test.png'),
                       k=sub_summary['k'],
                       names=VARIABLE_NAMES[effect])

    # 5-7. trajectories, block separations, proportion trajectories: built on the
    #      main-effect axes, so skipped when scoring the interactions
    if effect == 'main':
        main_effect_extras(args, labels, hg_windows, axis_hg, axis_labels, grid, seeds,
                           time_label)

    config = dict(
        timestamp=args.timestamp, data_source=args.data_source,
        epochs_root_file=args.epochs_root_file, task=args.task,
        acc_trials_only=args.acc_trials_only, subjects=args.subjects,
        electrodes=args.electrodes,
        rois=(list(args.rois_dict) if args.rois_dict else 'all'),
        windows_tmin=args.windows_tmin, windows_tmax=args.windows_tmax,
        window_size=args.window_size, step_size=args.step_size,
        effect_measure='mean_diff', contrast_mode='proportion', effect=effect,
        shared_split=True, remove_mean=args.remove_mean,
        rt_control=args.rt_control, rt_match_bins=args.rt_match_bins,
        n_splits=args.n_splits, n_perm=args.n_perm, n_null_splits=args.n_null_splits,
        n_boot=args.n_boot, alpha=args.alpha,
        k_max=args.k_max, traj_axes=[list(w) for w in args.traj_axes],
        traj_first=args.traj_first,
        seed=args.seed, seeds=seeds,
        n_subjects=int(labels.subject.nunique()),
        n_electrodes=int(labels.electrode.nunique()),
        windows=grid.to_dict('records'))
    with open(os.path.join(args.save_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=lambda o: o.item())

    print(f"alignment over {summary['n_reliable_windows']} reliable windows: "
          f"{summary['alignment_reliable_mean']} (CI {summary['alignment_reliable_ci']})")
    print(f"shared windows: {summary['shared_windows']} | not one code: "
          f"{summary['not_one_code_windows']}")
    print(f"subspace test: k = {sub_summary['k']}, primary k = {sub_summary['primary_k']}")
    if sub_summary['theta_1']:
        t1 = sub_summary['theta_1']
        print(f"  theta_1 at the primary k: {t1['cross_deg']:.1f} deg (cosine {t1['cross']:.3f}), "
              f"{t1['normalized']:.2f} of the ceiling | shared: {t1.get('shared')} | "
              f"not one code: {t1['not_one_code']}")
    print(f"wrote {args.save_dir}")
    return table, summary
