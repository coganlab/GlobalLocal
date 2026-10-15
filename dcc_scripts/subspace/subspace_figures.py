"""Figures for the subspace test (docs/analysis_plans.md, subspace plan §3.4)
and the condition trajectories (§4). Plotting only: each function takes the
tables `coding_subspaces.compare_subspaces` and
`coding_trajectories.condition_trajectories` return (or their CSVs) and writes
one PNG. Called by coding_axes_dcc.py; also usable on saved outputs.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

# Reference-palette slots 1-2 (dataviz skill), as in the coding-axes figure:
# blue = the congruency-switch (cross) value, orange = the ceiling.
CROSS, CEILING = '#2a78d6', '#eb6834'
INK, MUTED, NULL_FILL, PRIMARY_FILL = '#0b0b0b', '#52514e', '#d9d8d4', '#f0efec'
# Trajectories: colour = congruency (slots 1-2), line style = switch type.
CELL_STYLE = {'incongruent-switch': ('#2a78d6', '-'), 'incongruent-repeat': ('#2a78d6', '--'),
              'congruent-switch': ('#eb6834', '-'), 'congruent-repeat': ('#eb6834', '--')}
_ARROWS = {'congruency_to_switch': 'C→S', 'switch_to_congruency': 'S→C'}


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def _row_label(r):
    if r['measure'] == 'angle':
        return f"k={r['k']}\nθ{r['component']}"
    return f"k={r['k']}\n{_ARROWS[r['component']]}"


def _comparison_panel(ax, rows, ylabel):
    """One dot per row: the cross value (filled = shared, above both shuffle
    nulls) with its bootstrap CI, the ceiling as an orange tick, the stricter
    null threshold as a grey bar from 0. Rows at the primary k sit on a light
    band; rows whose ceilings are unreliable are faded."""
    x = np.arange(len(rows))
    null_cols = [c for c in rows if c.startswith('null_hi_')]
    for i, (_, r) in enumerate(rows.iterrows()):
        if r.get('primary', False):
            ax.axvspan(i - 0.5, i + 0.5, color=PRIMARY_FILL, lw=0, zorder=0)
        alpha = 1.0 if r['reliable'] else 0.35
        if null_cols:
            ax.bar(i, r[null_cols].max(), width=0.5, color=NULL_FILL, lw=0, zorder=1,
                   label='shuffle null (top = threshold)' if i == 0 else None)
        ax.hlines(r['ceiling'], i - 0.3, i + 0.3, color=CEILING, lw=2.5, alpha=alpha, zorder=2,
                  label='ceiling' if i == 0 else None)
        ax.vlines(i, r['cross_lo'], r['cross_hi'], color=CROSS, lw=1.5, alpha=alpha, zorder=3)
        shared = bool(r.get('shared', False))
        ax.plot(i, r['cross'], 'o', ms=8, mec=CROSS, mew=2, alpha=alpha, zorder=4,
                mfc=CROSS if shared else 'white',
                label=('cross (filled = shared)' if i == 0 else None))
    ax.set_xticks(x)
    ax.set_xticklabels([_row_label(r) for _, r in rows.iterrows()], fontsize=8)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    _style(ax)


def _dims_panel(ax, d, variable, k):
    """Held-out variance captured by each dimension of half A's subspace (log
    scale), filled where it is part of the kept (reproducible) run, with the
    shuffle null's threshold as a grey tick."""
    for _, r in d.iterrows():
        ax.plot(r['dimension'], r['heldout'], 'o', ms=8, mec=INK, mew=1.5,
                mfc=INK if r.get('kept', False) else 'white', zorder=3)
        if np.isfinite(r.get('null_hi', np.nan)):
            ax.hlines(r['null_hi'], r['dimension'] - 0.25, r['dimension'] + 0.25,
                      color=MUTED, lw=3, zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(d['dimension'])
    ax.set_xlim(0.4, d['dimension'].max() + 0.6)
    ax.set_xlabel('dimension', fontsize=9, color=INK)
    ax.set_title(f"{variable}: k = {k if k is not None else '–'}", fontsize=9, color=INK,
                 loc='left')
    _style(ax)


def plot_subspace_test(table, dims, out_path, k=None):
    """The §3.4 figure. (a) principal angles, (b) alignment index, each
    congruency vs switch per k, against ceiling and null; (c, d) how many
    dimensions of each variable are reproducible (filled), which sets k.
    `k` is `summary['k']` (variable -> k), for the panel titles."""
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.45, wspace=0.25)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _comparison_panel(a, table[table['measure'] == 'angle'], 'cosine of principal angle')
    _comparison_panel(b, table[table['measure'] == 'alignment'], 'alignment index')
    a.set_title('a  principal angles (congruency vs switch)', fontsize=9, color=INK, loc='left')
    b.set_title('b  alignment index', fontsize=9, color=INK, loc='left')
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=7.5, ncol=3,
               loc='lower center', bbox_to_anchor=(0.5, 0.94))
    k = k or {}
    for col, (var, name) in enumerate((('congruency', 'c  congruency'),
                                       ('switchType', 'd  switch type'))):
        ax = fig.add_subplot(gs[1, col])
        _dims_panel(ax, dims[dims['variable'] == var], name, k.get(var))
        if col == 0:
            ax.set_ylabel('held-out variance\n(filled = reproducible)', fontsize=9, color=INK)
    fig.text(0.01, 0.005, 'Grey band: rows at the primary k. Faded: a ceiling not above its '
             'own shuffle null. Grey ticks (c, d): 95th percentile of the shuffle null.',
             fontsize=7.5, color=MUTED)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_trajectories(traj, fractions, axes_summary, label, out_path, cells='centred',
                      first='congruency'):
    """The §4 figure for one axis window. (a) the four condition means in the
    plane, time as the path (open circle = first window, arrow = last);
    (b) x(t) and (c) y(t), with the participant-bootstrap CI for centred cells.

    `traj` is the trajectories table with `window_center` merged in; `fractions`
    the fractions table; `axes_summary` that axis window's summary entry."""
    t_all = traj[(traj['axes'] == label) & (traj['cells'] == cells)]
    x_name, y_name = (('congruency', 'switch beyond congruency') if first == 'congruency'
                      else ('switch', 'congruency beyond switch'))
    fig = plt.figure(figsize=(10.5, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.15, wspace=0.28)
    plane = fig.add_subplot(gs[:, 0])
    xt, yt = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1], sharex=None)
    xt.sharex(yt)
    for ax in (plane, xt, yt):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        _style(ax)
    plane.axvline(0, color=MUTED, lw=0.8, zorder=0)
    for cell, (color, ls) in CELL_STYLE.items():
        d = t_all[t_all['cell'] == cell].sort_values('window')
        x, y, t = d['x'].to_numpy(), d['y'].to_numpy(), d['window_center'].to_numpy()
        plane.plot(x, y, color=color, ls=ls, lw=2, label=cell)
        plane.plot(x[0], y[0], 'o', ms=7, mfc='white', mec=color, mew=1.5, zorder=3)
        if len(x) > 1:
            plane.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                           arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))
        for ax, col in ((xt, 'x'), (yt, 'y')):
            ax.plot(t, d[col], color=color, ls=ls, lw=2)
            if f'{col}_lo' in d and d[f'{col}_lo'].notna().any():
                ax.fill_between(t, d[f'{col}_lo'], d[f'{col}_hi'], color=color, alpha=0.12, lw=0)
    plane.set_xlabel(f'x: {x_name} axis (HG)', fontsize=9, color=INK)
    plane.set_ylabel(f'y: {y_name} axis (HG)', fontsize=9, color=INK)
    plane.set_aspect('equal', adjustable='datalim')
    plane.legend(frameon=False, fontsize=7.5, loc='best')
    plane.set_title(f'a  {cells} condition means, axes from {label}', fontsize=9,
                    color=INK, loc='left')
    xt.set_ylabel('x (HG)', fontsize=9, color=INK)
    yt.set_ylabel('y (HG)', fontsize=9, color=INK)
    plt.setp(xt.get_xticklabels(), visible=False)
    yt.set_xlabel('window centre (s from stimulus onset)', fontsize=9, color=INK)
    xt.set_title('b, c  coordinates over time', fontsize=9, color=INK, loc='left')
    overall = fractions[(fractions['axes'] == label) & (fractions['window'] == -1)]
    if not overall.empty and axes_summary:
        o = overall.iloc[0]
        fig.text(0.01, -0.02,
                 f"Held-out cells; axes and cells from different halves. Plane holds "
                 f"{o['plane_fraction']:.1%} of the centred cells' variance "
                 f"(chance {axes_summary['plane_fraction_chance']:.1%}) and "
                 f"{o['plane_fraction_signal']:.0%} of its noise-corrected signal. "
                 f"cos(c, s) of the plot's axes before Gram-Schmidt: "
                 f"{axes_summary['axis_cosine']:.2f}.",
                 fontsize=7.5, color=MUTED)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
