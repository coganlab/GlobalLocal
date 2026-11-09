"""Figures for the coding axes (docs/analysis_plans.md, subspace plan): the
cosine over time (§3.2), the subspace test (§3.4), the condition trajectories
(§4), the block-split separations and the proportion trajectories (§6).

Plotting only: each function takes the tables the analysis writes (or their
CSVs) and writes ``<stem>.png`` and ``<stem>.pdf`` at print size (``out_path``'s
extension is ignored). A figure with a supplementary companion writes it in
the same call, as ``<stem>_<suffix>``. Called by coding_axes_dcc.py and
redraw_figures.py; also usable on saved outputs.

To change the look of every figure here, edit the constants in the next
block; docs/figure_editing_guide.md §4 says what each one does.
"""
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.decoding.accuracy_stats import find_contiguous_clusters
from src.analysis.config.plotting_parameters import PROPORTION_COLORS
from src.analysis.decoding.plots.style import nature_style

# ---- the look of every figure in this file ---------------------------------
# Axis-label size (pt); ticks are 1 pt and legends 1.5 pt smaller (nature_style).
FONT_SIZE = 7
# Type-42 fonts keep the PDF's text editable in Illustrator.
RC = {**nature_style(FONT_SIZE), 'pdf.fonttype': 42, 'ps.fonttype': 42}
LW = 1.2                       # data lines (pt)
# Neutrals: ink for data without a colour of its own, grey, muted for zero
# lines, a pale fill for the shuffle null.
INK, GREY, MUTED = '#0b0b0b', '#8c8c8c', '#52514e'
NULL_FILL = '#ebebeb'
PRIMARY_FILL = '#f0efec'      # band behind the primary k in the subspace test
# Cross and ceiling (cosine figure, subspace test): Wong reddish purple and
# bluish green, the two Wong colours the proportion blocks do not use.
CROSS, CEILING = '#CC79A7', '#009E73'
# Line style per condition level, as the power traces
# (src/analysis/config/plotting_parameters.py): i and s dashed, c and r solid.
LEVEL_LS = {'i': '--', 'c': '-', 's': '--', 'r': '-'}
# Trajectory cells: shade = congruency (incongruent ink, congruent grey),
# line style = switch type (LEVEL_LS: switch dashed, repeat solid).
CELL_STYLE = {'incongruent-switch': (INK, LEVEL_LS['s']), 'incongruent-repeat': (INK, LEVEL_LS['r']),
              'congruent-switch': (GREY, LEVEL_LS['s']), 'congruent-repeat': (GREY, LEVEL_LS['r'])}
# Proportion blocks: the paper's shared palette (Wong 2011;
# plotting_parameters.PROPORTION_COLORS, also used by the behavioural, power-
# trace and decoding figures), light 25 %, dark 75 %. Change colours there.
PROP_COLOR = {prop: {'low': PROPORTION_COLORS[factor][25], 'high': PROPORTION_COLORS[factor][75]}
              for prop, factor in (('incProp', 'incongruentProportion'),
                                   ('switchProp', 'switchProportion'))}
# Figure sizes in inches, at print size (Nature: 89 mm = 3.5 in single
# column, 183 mm = 7.2 in double). Change these, not the font size, to fit a
# panel into a layout.
SIZES = dict(cosines=(3.4, 2.0), alignment=(3.4, 1.5), subspace_test=(7.2, 4.2),
             trajectories=(2.6, 4.0), separations=(7.2, 1.8), separation_changes=(5.0, 1.8),
             proportion_trajectories=(4.6, 4.8))
# Display names: change a name here and every subspace figure follows. The
# switch-type factor is called "task sequence"; its levels stay switch / repeat.
NAMES = dict(congruency='congruency', switch='task seq.', congruency_short='C',
             switch_short='TS', incProp='inc. prop.', switchProp='swi. prop.')
# Significance bars of the cosine figure, bottom to top: (table column,
# colour, label drawn beside the bar).
SIG_BARS = (('shared', CROSS, 'cross > shuffle'), ('not_one_code', CEILING, 'ceiling > cross'))
# -----------------------------------------------------------------------------

_ARROWS = {'congruency_to_switch': f"{NAMES['congruency_short']}→{NAMES['switch_short']}",
           'switch_to_congruency': f"{NAMES['switch_short']}→{NAMES['congruency_short']}"}
PCT = {'low': '25%', 'high': '75%'}
# Each variable's name, its own proportion and the other one.
_VAR_NAMES = {'congruency': (NAMES['congruency'], 'incProp', 'switchProp'),
              'switchType': (NAMES['switch'], 'switchProp', 'incProp')}
_PROP_COL = {'incProp': 'inc_prop', 'switchProp': 'switch_prop'}
_PROP_LABEL = {'incProp': NAMES['incProp'], 'switchProp': NAMES['switchProp']}
# Proportion shorthand, as the decoding figures write it ('25% I > 75% I').
_PROP_SHORT = {'incProp': 'I', 'switchProp': 'S'}
_PCT_LEVEL = {'25': 'low', '75': 'high'}


def true_runs(mask):
    """(first, last) index of each run of True in ``mask``."""
    return find_contiguous_clusters(np.asarray(mask, bool))


def _stem(out_path):
    """``out_path`` without a .png/.pdf extension. Not ``os.path.splitext``: the
    window labels have dots in them ('..._0.0-1.5s')."""
    return out_path[:-4] if out_path.lower().endswith(('.png', '.pdf')) else out_path


def _save(fig, stem):
    """Write ``<stem>.png`` and ``<stem>.pdf``, close the figure, return the paths.
    Call inside the figure's ``rc_context`` so its savefig settings apply."""
    paths = [f'{stem}.{ext}' for ext in ('png', 'pdf')]
    for p in paths:
        fig.savefig(p, dpi=300, facecolor='white')
    plt.close(fig)
    return paths


def _zero(ax, vertical=False):
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0)
    if vertical:
        ax.axvline(0, color=MUTED, lw=0.5, zorder=0)


def time_axis_label(epochs_root_file=None):
    """x-axis label for window centres. Epochs files are named for the event
    they are locked to (`Stimulus_...` or `Response_...`); anything else
    (synthetic runs: None) is stimulus-locked."""
    event = 'response' if str(epochs_root_file).startswith('Response') else 'stimulus onset'
    return f'Time from {event} (s)'


def plot_coding_axes(table, out_path, time_label=time_axis_label(), title=None):
    """The §3.2 figure (main text): the cross cosine (congruency axis from one
    half against switch axis from the other) over time, with the ceiling, the
    shuffle-null band (its top edge is the test's threshold under the stricter
    of the two shuffles) and a bar over the windows where the cosine is shared
    (cluster-corrected) and where the ceiling is above it (bootstrap CI of
    the gap, pointwise). Bands are the participant-bootstrap CIs of cross and
    ceiling, when the table has them. The ceiling is drawn only in windows
    where both codes are reliable (the caption says why it is missing early).

    Also writes the supplementary ``<stem>_alignment``: cross / ceiling with
    its participant-bootstrap CI, blank where unreliable.
    ``table`` is coding_axis_cosines.csv. ``title`` is accepted for callers and
    not drawn: titles and panel letters go in the caption and Illustrator."""
    t = table['window_center'].to_numpy()
    half = np.median(np.diff(t)) / 2 if len(t) > 1 else 0.05
    stem = _stem(out_path)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=SIZES['cosines'])
        reliable = table['reliable'].to_numpy(bool)
        handles = []
        null_lo = [c for c in table if c.startswith('null_lo_')]
        if null_lo:
            # The shuffle as a trace, as in the decoding figures: its band (the
            # stricter of the two shuffles' central 90 %) and a dotted centre
            # line. The centre is drawn at 0: the shuffles' medians are within
            # +-0.03 of 0 in every window, and only the band's edges are saved.
            lo = table[null_lo].min(axis=1)
            hi = table[[c.replace('_lo_', '_hi_') for c in null_lo]].max(axis=1)
            ax.fill_between(t, lo, hi, color=NULL_FILL, lw=0, zorder=2)
            ax.plot(t, np.zeros_like(t), color=GREY, lw=0.8, ls=':', zorder=2)
            handles.append(((Patch(color=NULL_FILL, lw=0), Line2D([], [], color=GREY, lw=0.8,
                                                                   ls=':')), 'shuffle'))
        else:
            _zero(ax)
        # error bands: the participant-bootstrap CIs (tables written before
        # 2026-11-09 have none; add_cosine_cis.py adds them)
        if 'cross_lo' in table:
            ax.fill_between(t, table['cross_lo'], table['cross_hi'], color=CROSS, alpha=0.15,
                            lw=0, zorder=3)
            ax.fill_between(t, table['ceiling_lo'].where(reliable),
                            table['ceiling_hi'].where(reliable), color=CEILING, alpha=0.2,
                            lw=0, zorder=3)
        handles += [(ax.plot(t, table['ceiling'].where(reliable), color=CEILING, lw=LW,
                             zorder=4)[0], 'ceiling'),
                    (ax.plot(t, table['cross'], color=CROSS, lw=LW, zorder=5)[0], 'cross')]
        # Significance bars above the curves, each in the colour of the curve
        # that is larger and named beside its first run: 'shared' is cross
        # above both shuffles (cluster-corrected), 'not_one_code' is ceiling
        # above cross (participant bootstrap of the gap, pointwise).
        top = np.nanmax(table[[c for c in ('cross_hi', 'ceiling_hi', 'cross', 'ceiling')
                               if c in table]].to_numpy())
        for i, (col, color, text) in enumerate(SIG_BARS):
            if col not in table or not table[col].any():
                continue
            y = top * (1.1 + 0.08 * i)
            runs = true_runs(table[col])
            for start, end in runs:
                ax.hlines(y, t[start] - half, t[end] + half, color=color, lw=2, zorder=5)
            ax.annotate(text, (t[runs[0][0]] - half, y), xytext=(-3, 0),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=FONT_SIZE - 1.5, color=color)
        ax.set(xlabel=time_label, ylabel=f"{NAMES['congruency']}–{NAMES['switch']} cosine")
        # below the bars' labels, in the early windows (no ceiling there)
        ax.legend(*zip(*handles), loc='center left', bbox_to_anchor=(0, 0.6))
        paths = _save(fig, stem)

        fig, ax = plt.subplots(figsize=SIZES['alignment'])
        _zero(ax)
        ax.fill_between(t, table['alignment_lo'].where(reliable),
                        table['alignment_hi'].where(reliable), color=INK, alpha=0.15, lw=0,
                        zorder=2)
        ax.plot(t, table['alignment'].where(reliable), color=INK, lw=LW, zorder=3)
        ax.axhline(1, color=MUTED, lw=0.5, ls=':', zorder=1)
        ax.set(xlabel=time_label, ylabel='cross / ceiling')
        paths += _save(fig, f'{stem}_alignment')
    return paths


def _row_label(r):
    if r['measure'] == 'angle':
        return f"k={r['k']}\nθ{r['component']}"
    return f"k={r['k']}\n{_ARROWS[r['component']]}"


def _comparison_panel(ax, rows, ylabel):
    """One dot per row: the cross value (filled = shared, above both shuffle
    nulls) with its bootstrap CI, the ceiling as a grey tick, the stricter
    null threshold as a pale bar from 0. Rows at the primary k sit on a light
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
        ax.hlines(r['ceiling'], i - 0.3, i + 0.3, color=CEILING, lw=2, alpha=alpha, zorder=2,
                  label='ceiling' if i == 0 else None)
        ax.vlines(i, r['cross_lo'], r['cross_hi'], color=CROSS, lw=LW, alpha=alpha, zorder=3)
        shared = bool(r.get('shared', False))
        ax.plot(i, r['cross'], 'o', ms=4, mec=CROSS, mew=1, alpha=alpha, zorder=4,
                mfc=CROSS if shared else 'white',
                label=('cross (filled = shared)' if i == 0 else None))
    ax.set_xticks(x)
    ax.set_xticklabels([_row_label(r) for _, r in rows.iterrows()])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)


def _dims_panel(ax, d, variable):
    """Held-out variance captured by each dimension of half A's subspace (log
    scale), filled where it is part of the kept (reproducible) run, with the
    shuffle null's threshold as a grey tick."""
    for _, r in d.iterrows():
        ax.plot(r['dimension'], r['heldout'], 'o', ms=4, mec=INK, mew=1,
                mfc=INK if r.get('kept', False) else 'white', zorder=3)
        if np.isfinite(r.get('null_hi', np.nan)):
            ax.hlines(r['null_hi'], r['dimension'] - 0.25, r['dimension'] + 0.25,
                      color=GREY, lw=2, zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(d['dimension'])
    ax.set_xlim(0.4, d['dimension'].max() + 0.6)
    ax.set_xlabel('dimension')
    ax.set_title(variable, loc='left')


def plot_subspace_test(table, dims, out_path, k=None, names=(NAMES['congruency'], NAMES['switch'])):
    """The §3.4 figure (supplement). (a) principal angles, (b) alignment index,
    each congruency vs switch per k, against ceiling and null; (c, d) how many
    dimensions of each variable are reproducible (filled), which sets k.
    ``k`` is accepted for callers and not drawn. ``names`` are the two scores'
    display names (e.g. ('LWPC', 'LWPS') when the job scores the interactions);
    the table's variables stay 'congruency', 'switchType'."""
    with plt.rc_context(RC):
        fig = plt.figure(figsize=SIZES['subspace_test'])
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.6, wspace=0.3)
        a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        _comparison_panel(a, table[table['measure'] == 'angle'], 'cosine of principal angle')
        _comparison_panel(b, table[table['measure'] == 'alignment'], 'alignment index')
        fig.legend(*a.get_legend_handles_labels(), ncol=3, loc='lower center',
                   bbox_to_anchor=(0.5, 0.93))
        for col, (var, name) in enumerate((('congruency', names[0]), ('switchType', names[1]))):
            ax = fig.add_subplot(gs[1, col])
            _dims_panel(ax, dims[dims['variable'] == var], name)
            if col == 0:
                ax.set_ylabel('held-out variance (z²)\n(filled = reproducible)')
        return _save(fig, _stem(out_path))


def plot_trajectories(traj, fractions, axes_summary, label, out_path, cells='centred',
                      first='congruency', time_label=time_axis_label()):
    """The §4 figure for one axis window (main text). Top: the four condition
    means in the plane, time as the path (open circle = first window, arrow =
    last). Beneath, as thin strips: x(t) and y(t), with the participant-
    bootstrap CI for centred cells.

    ``traj`` is the trajectories table with ``window_center`` merged in;
    ``fractions`` and ``axes_summary`` are accepted for callers and not drawn."""
    t_all = traj[(traj['axes'] == label) & (traj['cells'] == cells)]
    c, s = NAMES['congruency'], NAMES['switch']
    x_name, y_name = {'congruency': (c, f'{s} beyond\n{c}'),
                      'switch': (s, f'{c} beyond\n{s}'),
                      None: (c, s)}[first]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=SIZES['trajectories'])
        outer = fig.add_gridspec(2, 1, height_ratios=[2.6, 2], hspace=0.3)
        strips = outer[1].subgridspec(2, 1, hspace=0.15)
        plane, xt = fig.add_subplot(outer[0]), fig.add_subplot(strips[0])
        yt = fig.add_subplot(strips[1], sharex=xt)
        for ax in (xt, yt):
            _zero(ax)
        _zero(plane, vertical=True)
        for cell, (color, ls) in CELL_STYLE.items():
            d = t_all[t_all['cell'] == cell].sort_values('window')
            x, y, t = d['x'].to_numpy(), d['y'].to_numpy(), d['window_center'].to_numpy()
            plane.plot(x, y, color=color, ls=ls, lw=LW, label=cell.replace('-', ', '))
            plane.plot(x[0], y[0], 'o', ms=3, mfc='white', mec=color, mew=0.8, zorder=3)
            if len(x) > 1:
                plane.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                               arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8,
                                               mutation_scale=7))
            for ax, col in ((xt, 'x'), (yt, 'y')):
                ax.plot(t, d[col], color=color, ls=ls, lw=LW)
                if f'{col}_lo' in d and d[f'{col}_lo'].notna().any():
                    ax.fill_between(t, d[f'{col}_lo'], d[f'{col}_hi'], color=color,
                                    alpha=0.12, lw=0)
        plane.set(xlabel=f'x: {x_name} axis (z)', ylabel=f'y: {y_name} axis (z)')
        plane.set_aspect('equal', adjustable='datalim')
        plane.legend(loc='lower right')
        xt.set_ylabel('x (z)')
        yt.set_ylabel('y (z)')
        plt.setp(xt.get_xticklabels(), visible=False)
        yt.set_xlabel(time_label)
        return _save(fig, _stem(out_path))


def plot_block_separations(seps, cons, label, out_path, time_label=time_axis_label(),
                           unique=False):
    """The §6 figure for one axis window (main text): each variable's held-out
    separation on its own axis, one panel per level of the OTHER proportion,
    with the variable's OWN proportion in its paradigm colours (light 25 %, dark
    75 %). Adaptation shows as the light and dark lines apart in both panels.

    Also writes the supplementary ``<stem>_changes``: the change in separation
    (25 % − 75 %) with the own proportion (ink, solid) and the other one (grey,
    dashed), with the participant-bootstrap CI; its numbers are
    block_contrasts.csv.

    ``seps``, ``cons`` are the ``block_separations`` tables with
    ``window_center`` merged in. Only each variable's own-axis rows are drawn.
    ``unique`` is accepted for callers (the ``_unique`` tables are drawn the
    same way)."""
    seps = seps[(seps['axes'] == label) & (seps['window'] >= 0)]
    cons = cons[(cons['axes'] == label) & (cons['window'] >= 0)]
    if 'axis' in seps:                       # own-axis rows: the axis named after the variable
        own_axis = {v: n[0] for v, n in _VAR_NAMES.items()}
        seps = seps[seps['axis'] == seps['variable'].map(own_axis)]
        cons = cons[cons['axis'] == cons['variable'].map(own_axis)]
    stem = _stem(out_path)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=SIZES['separations'])
        outer = fig.add_gridspec(1, 2, wspace=0.25)
        for v, (var, (name, own, other)) in enumerate(_VAR_NAMES.items()):
            inner = outer[v].subgridspec(1, 2, wspace=0.08)
            left = fig.add_subplot(inner[0])
            pair = (left, fig.add_subplot(inner[1], sharex=left, sharey=left))
            s = seps[seps['variable'] == var]
            for ax, other_level in zip(pair, ('low', 'high')):
                for own_level in ('low', 'high'):
                    d = s[(s[_PROP_COL[own]] == own_level) & (s[_PROP_COL[other]] == other_level)]
                    d = d.sort_values('window_center')
                    ax.plot(d['window_center'], d['separation'], lw=LW,
                            color=PROP_COLOR[own][own_level],
                            label=f'{PCT[own_level]} {_PROP_SHORT[own]}')
                _zero(ax)
                ax.set_title(f'{PCT[other_level]} {_PROP_SHORT[other]} blocks', loc='left')
            plt.setp(pair[1].get_yticklabels(), visible=False)
            left.set_ylabel(f'{name} separation (z)')
            left.legend(loc='upper left')
        fig.supxlabel(time_label, fontsize=FONT_SIZE, y=-0.08)
        paths = _save(fig, stem)

        fig, axs = plt.subplots(1, 2, figsize=SIZES['separation_changes'], sharex=True,
                                gridspec_kw=dict(wspace=0.4))
        for ax, (var, (name, own, other)) in zip(axs, _VAR_NAMES.items()):
            c = cons[cons['variable'] == var]
            for con, color, ls in ((own, INK, '-'), (other, GREY, '--')):
                d = c[c['contrast'] == con].sort_values('window_center')
                ax.fill_between(d['window_center'], d['lo'], d['hi'], color=color, alpha=0.15,
                                lw=0)
                ax.plot(d['window_center'], d['estimate'], color=color, ls=ls, lw=LW,
                        label=_PROP_LABEL[con])
            _zero(ax)
            ax.set(xlabel=time_label, ylabel=f'change in {name} separation,\n25% − 75% (z)')
            ax.legend(loc='upper left')
        paths += _save(fig, f'{stem}_changes')
    return paths


def plot_proportion_trajectories(traj, changes, label, out_path):
    """The §6 trajectory figure for one axis window (main text): rows =
    variable, columns = proportion, own sets on the diagonal. Each panel draws
    the variable's two levels in the 25 % and 75 % blocks on the congruency (x)
    and switch (y) axes, time as the path (open circle = first window, arrow =
    last); colour = the block (the column's proportion, ``PROP_COLOR``), line
    style = the level (``LEVEL_LS``, as the power traces), one legend entry per
    path ('i, 25% I').
    Oblique coordinates: a change along the congruency axis moves only x, one
    along the switch axis only y. The prediction: on the diagonal the two
    levels' separation differs between 25 % and 75 % along the panel's own
    axis only; off the diagonal it does not differ.

    ``traj`` is the ``proportion_trajectories`` table; ``changes`` is accepted
    for callers and not drawn."""
    traj = traj[traj['axes'] == label]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 2, figsize=SIZES['proportion_trajectories'], sharex=True,
                                sharey=True, gridspec_kw=dict(hspace=0.3, wspace=0.1))
        for r, (var, var_name) in enumerate((('congruency', NAMES['congruency']),
                                             ('switchType', NAMES['switch']))):
            for col, prop in enumerate(('incProp', 'switchProp')):
                ax = axs[r, col]
                d = traj[traj['set'] == f'{var} x {prop}']
                for cond, dc in d.groupby('condition', sort=False):
                    dc = dc.sort_values('window')
                    level, pct = cond[0], cond[1:]
                    x, y = dc['congruency'].to_numpy(), dc['switch'].to_numpy()
                    color = PROP_COLOR[prop][_PCT_LEVEL[pct]]
                    ax.plot(x, y, color=color, ls=LEVEL_LS[level], lw=LW,
                            label=f'{level}, {pct}% {_PROP_SHORT[prop]}')
                    ax.plot(x[0], y[0], 'o', ms=3, mfc='white', mec=color, mew=0.8, zorder=3)
                    if len(x) > 1:
                        ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                                    arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8,
                                                    mutation_scale=7))
                _zero(ax, vertical=True)
                ax.set_title(f'{var_name} × {NAMES[prop]}', loc='left')
                ax.legend(loc='best')
        for ax in axs[1]:
            ax.set_xlabel(f"{NAMES['congruency']} axis (z)")
        for ax in axs[:, 0]:
            ax.set_ylabel(f"{NAMES['switch']} axis (z)")
        return _save(fig, _stem(out_path))
