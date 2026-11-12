"""Figures for the coding axes (docs/analysis_plans.md, subspace plan): the
cosine over time (§3.2), the subspace test (§3.4), the condition trajectories
(§4), the block-split separations and the proportion trajectories (§6).

Plotting only: each function takes the tables the analysis writes (or their
CSVs) and writes ``<stem>.png``, ``.pdf`` and ``.tif`` (450 dpi) at print size (``out_path``'s
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

from src.analysis.decoding.accuracy_stats import find_contiguous_clusters
from src.analysis.config.plotting_parameters import PROPORTION_COLORS
from src.analysis.decoding.plots.style import nature_style
from src.analysis.utils.panel_layout import save_figure

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
# The shuffle drawn as in the decoding figures (paper_panels.py,
# decoding/plots/replot.py): a grey dashed trace with a band of the same grey,
# and the bands' opacity; LABEL_INK is the colour of the significance-row
# labels there.
SHUFFLE, BAND_ALPHA, LABEL_INK = '#949494', 0.18, '#3d3d3d'
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
             trajectories=(5.2, 2.5), separations=(7.2, 1.8), separation_changes=(5.6, 1.8),
             separation_change_panel=(2.4, 1.8), separation_changes_stacked=(2.4, 4.6),
             proportion_trajectories=(4.6, 4.8), proportion_differences=(4.6, 4.8),
             proportion_change_vectors=(2.2, 2.2))
# Display names: change a name here and every subspace figure follows. The
# switch-type factor is called "task sequence"; its levels stay switch / repeat.
NAMES = dict(congruency='congruency', switch='task seq.', congruency_short='C',
             switch_short='TS', incProp='inc. prop.', switchProp='swi. prop.')
# Significance bars of the cosine figure, bottom to top: (table column,
# colour, label drawn beside the bar).
SIG_BARS = (('shared', CROSS, 'Cross > shuffle'), ('not_one_code', CEILING, 'Ceiling > cross'))
# -----------------------------------------------------------------------------

_ARROWS = {'congruency_to_switch': f"{NAMES['congruency_short']}→{NAMES['switch_short']}",
           'switch_to_congruency': f"{NAMES['switch_short']}→{NAMES['congruency_short']}"}
PCT = {'low': '25%', 'high': '75%'}
# Each variable's name, its own proportion and the other one.
_VAR_NAMES = {'congruency': (NAMES['congruency'], 'incProp', 'switchProp'),
              'switchType': (NAMES['switch'], 'switchProp', 'incProp')}
# Each variable's own axis as the result tables name it (the ``axis`` column).
# Data keys, not display names: renaming a factor in NAMES must not change
# which rows are drawn.
_OWN_AXIS = {'congruency': 'congruency', 'switchType': 'switch'}
_PROP_COL = {'incProp': 'inc_prop', 'switchProp': 'switch_prop'}
# Each proportion's colour where it is a contrast, not a block (the changes
# figure): the dark (75 %) shade of its block palette, so incongruent
# proportion is orange and switch proportion blue in every panel.
CHANGE_COLOR = {prop: PROP_COLOR[prop]['high'] for prop in ('incProp', 'switchProp')}
# Each variable's name in file names of single panels.
_FILE_NAME = {'congruency': 'congruency', 'switchType': 'task_seq'}
_PROP_FILE = {'incProp': 'inc_prop', 'switchProp': 'swi_prop'}
_PROP_LABEL = {'incProp': NAMES['incProp'], 'switchProp': NAMES['switchProp']}
# Proportion shorthand, as the decoding figures write it ('25% I > 75% I').
_PROP_SHORT = {'incProp': 'I', 'switchProp': 'S'}
_PCT_LEVEL = {'25': 'low', '75': 'high'}


def _sentence(text):
    """Sentence case for an axis label or legend entry, as the power-trace and
    decoding figures write them: first letter capitalised, the rest as given
    (so 'task seq.' and 'z' keep their case)."""
    return text[:1].upper() + text[1:]


def true_runs(mask):
    """(first, last) index of each run of True in ``mask``."""
    return find_contiguous_clusters(np.asarray(mask, bool))


def _stem(out_path):
    """``out_path`` without a .png/.pdf extension. Not ``os.path.splitext``: the
    window labels have dots in them ('..._0.0-1.5s')."""
    return out_path[:-4] if out_path.lower().endswith(('.png', '.pdf')) else out_path


def _save(fig, stem):
    """Write ``<stem>.png``, ``.pdf`` and ``.tif`` (450 dpi, LZW; the paper
    panels' `panel_layout.save_figure`), close the figure, return the paths.
    Call inside the figure's ``rc_context`` so its savefig settings apply."""
    paths = save_figure(fig, stem, formats=('png', 'pdf', 'tif'), dpi=450, facecolor='white')
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
        # error bands: the participant-bootstrap CIs (tables written before
        # 2026-11-09 have none; add_cosine_cis.py adds them)
        if 'cross_lo' in table:
            ax.fill_between(t, table['cross_lo'], table['cross_hi'], color=CROSS,
                            alpha=BAND_ALPHA, lw=0, zorder=1)
            ax.fill_between(t, table['ceiling_lo'].where(reliable),
                            table['ceiling_hi'].where(reliable), color=CEILING,
                            alpha=BAND_ALPHA, lw=0, zorder=1)
        handles += [(ax.plot(t, table['cross'], color=CROSS, lw=LW, zorder=5)[0], 'Cross'),
                    (ax.plot(t, table['ceiling'].where(reliable), color=CEILING, lw=LW,
                             zorder=4)[0], 'Ceiling')]
        null_lo = [c for c in table if c.startswith('null_lo_')]
        if null_lo:
            # The shuffle as in the decoding figures: a grey dashed trace with a
            # band of the same grey, listed last in the legend. The band is the
            # stricter of the two shuffles' central 90 %; the trace is drawn at
            # 0, since the shuffles' medians are within +-0.03 of 0 in every
            # window and only the band's edges are saved.
            lo = table[null_lo].min(axis=1)
            hi = table[[c.replace('_lo_', '_hi_') for c in null_lo]].max(axis=1)
            ax.fill_between(t, lo, hi, color=SHUFFLE, alpha=BAND_ALPHA, lw=0, zorder=1)
            handles.append((ax.plot(t, np.zeros_like(t), color=SHUFFLE, lw=LW, ls='--',
                                    zorder=2)[0], 'Shuffle'))
        else:
            _zero(ax)
        # Significance bars above the curves, each in the colour of the curve
        # that is larger: 'shared' is cross above both shuffles (cluster-
        # corrected), 'not_one_code' is ceiling above cross (participant
        # bootstrap of the gap, pointwise). As in the decoding figures, each
        # row is named to the right of the axes and the legend sits under the
        # names, outside the data.
        top = np.nanmax(table[[c for c in ('cross_hi', 'ceiling_hi', 'cross', 'ceiling')
                               if c in table]].to_numpy())
        names_at = ax.get_yaxis_transform()          # x in axes units, y in data
        for i, (col, color, text) in enumerate(SIG_BARS):
            if col not in table or not table[col].any():
                continue
            y = top * (1.1 + 0.08 * i)
            for start, end in true_runs(table[col]):
                ax.hlines(y, t[start] - half, t[end] + half, color=color, lw=2, zorder=5)
            ax.annotate(text, (1.0, y), xycoords=names_at, xytext=(4, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=FONT_SIZE - 1.5, color=LABEL_INK, annotation_clip=False)
        ax.set(xlabel=time_label,
               ylabel=_sentence(f"{NAMES['congruency']}–{NAMES['switch']} cosine"))
        ax.legend(*zip(*handles), loc='upper left', bbox_to_anchor=(1.02, 0.72),
                  frameon=False, borderaxespad=0, handlelength=1.6)
        paths = _save(fig, stem)

        fig, ax = plt.subplots(figsize=SIZES['alignment'])
        _zero(ax)
        ax.fill_between(t, table['alignment_lo'].where(reliable),
                        table['alignment_hi'].where(reliable), color=INK, alpha=0.15, lw=0,
                        zorder=2)
        ax.plot(t, table['alignment'].where(reliable), color=INK, lw=LW, zorder=3)
        ax.axhline(1, color=MUTED, lw=0.5, ls=':', zorder=1)
        ax.set(xlabel=time_label, ylabel='Cross / ceiling')
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
                   label='Shuffle' if i == 0 else None)
        ax.hlines(r['ceiling'], i - 0.3, i + 0.3, color=CEILING, lw=2, alpha=alpha, zorder=2,
                  label='Ceiling' if i == 0 else None)
        ax.vlines(i, r['cross_lo'], r['cross_hi'], color=CROSS, lw=LW, alpha=alpha, zorder=3)
        shared = bool(r.get('shared', False))
        ax.plot(i, r['cross'], 'o', ms=4, mec=CROSS, mew=1, alpha=alpha, zorder=4,
                mfc=CROSS if shared else 'white',
                label=('Cross (filled = shared)' if i == 0 else None))
    ax.set_xticks(x)
    ax.set_xticklabels([_row_label(r) for _, r in rows.iterrows()])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)


def _dims_panel(ax, d, variable):
    """Share (%) of the variable's held-out variance that each dimension of
    half A's subspace captures, on a linear scale, filled where it is part of
    the kept (reproducible) run, with the shuffle null's threshold as a grey
    tick on the same scale. The table's ``heldout`` is in z²; dividing by the
    variable's total held-out variance (``heldout / heldout_fraction``, one
    number per variable) turns it and the null into shares."""
    total = (d['heldout'] / d['heldout_fraction']).median()
    for _, r in d.iterrows():
        ax.plot(r['dimension'], 100 * r['heldout_fraction'], 'o', ms=4, mec=INK, mew=1,
                mfc=INK if r.get('kept', False) else 'white', zorder=3)
        if np.isfinite(r.get('null_hi', np.nan)):
            ax.hlines(100 * r['null_hi'] / total, r['dimension'] - 0.25, r['dimension'] + 0.25,
                      color=GREY, lw=2, zorder=2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(d['dimension'])
    ax.set_xlim(0.4, d['dimension'].max() + 0.6)
    ax.set_xlabel('Dimension')
    ax.set_title(_sentence(variable), loc='left')


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
        _comparison_panel(a, table[table['measure'] == 'angle'], 'Cosine of principal angle')
        _comparison_panel(b, table[table['measure'] == 'alignment'], 'Alignment index')
        fig.legend(*a.get_legend_handles_labels(), ncol=3, loc='lower center',
                   bbox_to_anchor=(0.5, 0.93))
        dim_axes = []
        for col, (var, name) in enumerate((('congruency', names[0]), ('switchType', names[1]))):
            ax = fig.add_subplot(gs[1, col])
            _dims_panel(ax, dims[dims['variable'] == var], name)
            dim_axes.append(ax)
            if col == 0:
                ax.set_ylabel('Held-out variance (%)\n(filled = reproducible)')
        _same_ylim(dim_axes)
        return _save(fig, _stem(out_path))


def plot_trajectories(traj, fractions, axes_summary, label, out_path, cells='centred',
                      first='congruency', time_label=time_axis_label()):
    """The §4 figure for one axis window (main text). Left: the four condition
    means in the plane, time as the path (open circle = first window, arrow =
    last). To its right, stacked: x(t) and y(t), with the participant-
    bootstrap CI for centred cells (descriptive; no test).

    ``traj`` is the trajectories table with ``window_center`` merged in;
    ``fractions`` and ``axes_summary`` are accepted for callers and not drawn."""
    t_all = traj[(traj['axes'] == label) & (traj['cells'] == cells)]
    c, s = NAMES['congruency'], NAMES['switch']
    x_name, y_name = {'congruency': (c, f'{s} beyond\n{c}'),
                      'switch': (s, f'{c} beyond\n{s}'),
                      None: (c, s)}[first]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=SIZES['trajectories'])
        # the plane on the left, the x(t) and y(t) strips stacked to its right
        outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.45)
        strips = outer[1].subgridspec(2, 1, hspace=0.15)
        plane, xt = fig.add_subplot(outer[0]), fig.add_subplot(strips[0])
        yt = fig.add_subplot(strips[1], sharex=xt)
        for ax in (xt, yt):
            _zero(ax)
        _zero(plane, vertical=True)
        for cell, (color, ls) in CELL_STYLE.items():
            d = t_all[t_all['cell'] == cell].sort_values('window')
            x, y, t = d['x'].to_numpy(), d['y'].to_numpy(), d['window_center'].to_numpy()
            # 'incongruent-switch' -> 'i-s', the power traces' level letters
            plane.plot(x, y, color=color, ls=ls, lw=LW,
                       label='-'.join(part[0] for part in cell.split('-')))
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
        plane.set(xlabel=f'x: {x_name} axis (z)', ylabel=f'y: {y_name} axis (z)')   # x, y: variables
        plane.set_aspect('equal', adjustable='datalim')
        plane.legend(loc='lower right')
        xt.set_ylabel('x (z)')
        yt.set_ylabel('y (z)')
        _same_ylim([xt, yt])            # x(t) and y(t) read on one scale
        fig.align_ylabels([xt, yt])
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
        seps = seps[seps['axis'] == seps['variable'].map(_OWN_AXIS)]
        cons = cons[cons['axis'] == cons['variable'].map(_OWN_AXIS)]
    stem = _stem(out_path)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=SIZES['separations'])
        outer = fig.add_gridspec(1, 2, wspace=0.25)
        panels = []
        for v, (var, (name, own, other)) in enumerate(_VAR_NAMES.items()):
            inner = outer[v].subgridspec(1, 2, wspace=0.08)
            left = fig.add_subplot(inner[0])
            pair = (left, fig.add_subplot(inner[1], sharex=left, sharey=left))
            panels += pair
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
            left.set_ylabel(_sentence(f'{name} separation (z)'))
            left.legend(loc='upper left')
        _same_ylim(panels)          # congruency and task seq. on one scale
        fig.supxlabel(time_label, fontsize=FONT_SIZE, y=-0.08)
        paths = _save(fig, stem)

        # combined, then each panel alone on the same scale (for Illustrator)
        fig, axs = plt.subplots(1, 2, figsize=SIZES['separation_changes'], sharex=True,
                                gridspec_kw=dict(wspace=0.6))
        for ax, (var, spec) in zip(axs, _VAR_NAMES.items()):
            _changes_panel(ax, cons[cons['variable'] == var], *spec, time_label)
        _same_ylim(axs)
        ylim = axs[0].get_ylim()
        for ax, (var, (name, own, other)) in zip(axs, _VAR_NAMES.items()):
            _change_bars(ax, cons[cons['variable'] == var], own, other)
        paths += _save(fig, f'{stem}_changes')
        # the same two panels stacked vertically, time axis shared; room
        # between them for the lower panel's significance rows
        fig, axs = plt.subplots(2, 1, figsize=SIZES['separation_changes_stacked'],
                                sharex=True, gridspec_kw=dict(hspace=0.45))
        for ax, (var, (name, own, other)) in zip(axs, _VAR_NAMES.items()):
            c = cons[cons['variable'] == var]
            _changes_panel(ax, c, name, own, other, time_label)
            ax.set_ylim(ylim)
            _change_bars(ax, c, own, other)
        axs[0].set_xlabel('')
        fig.align_ylabels(axs)
        paths += _save(fig, f'{stem}_changes_stacked')
        for var, (name, own, other) in _VAR_NAMES.items():
            fig, ax = plt.subplots(figsize=SIZES['separation_change_panel'])
            c = cons[cons['variable'] == var]
            _changes_panel(ax, c, name, own, other, time_label)
            ax.set_ylim(ylim)
            _change_bars(ax, c, own, other)
            paths += _save(fig, f'{stem}_changes_{_FILE_NAME[var]}')
    return paths


def _changes_panel(ax, cons, name, own, other, time_label):
    """One variable's 25 % − 75 % separation changes with each proportion
    (`cons`: its rows of the contrasts table). Colour = the proportion, the
    same in every panel (the paper's incongruent-proportion orange,
    switch-proportion blue), both solid, with their bootstrap bands
    (precision), legend in the figure's lower-case shorthand and in the same
    order in every panel. Behind them, as in the decoding and cosine figures,
    the block null as the grey 'shuffle' (chance): its central 1 − alpha band,
    one band for both contrasts (the union of their nearly identical nulls),
    and its median dashed. Tables without the null get a plain zero line."""
    if 'null_lo' in cons:
        null = (cons[cons['contrast'].isin(['incProp', 'switchProp'])]
                .groupby('window_center')[['null_lo', 'null_mid', 'null_hi']]
                .agg({'null_lo': 'min', 'null_mid': 'mean', 'null_hi': 'max'}))
        ax.fill_between(null.index, null['null_lo'], null['null_hi'], color=SHUFFLE,
                        alpha=BAND_ALPHA, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.5, zorder=0)
        shuffle = ax.plot(null.index, null['null_mid'], color=SHUFFLE, lw=LW, ls='--',
                          zorder=1)[0]
    else:
        _zero(ax)
        shuffle = None
    for con in ('incProp', 'switchProp'):
        d = cons[cons['contrast'] == con].sort_values('window_center')
        ax.fill_between(d['window_center'], d['lo'], d['hi'], color=CHANGE_COLOR[con],
                        alpha=BAND_ALPHA, lw=0)
        ax.plot(d['window_center'], d['estimate'], color=CHANGE_COLOR[con], lw=LW,
                label=_PROP_LABEL[con])
    if shuffle is not None:
        shuffle.set_label('shuffle')                 # listed last, as in decoding
    ax.set(xlabel=time_label, ylabel=f'Change in {name} separation,\n25% − 75% (z)')
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == 'shuffle')
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left')


def _change_bars(ax, cons, own, other):
    """Significance rows above a changes panel, as in the decoding figures:
    one bar per contrast with a cluster (`coding_trajectories.contrast_clusters`,
    cluster-corrected against the block null), in the contrast's colour, named
    to the right of the axes, as the cosine figure's rows ('25% S > 75% S').
    Rows: the own change, the other change, own − other (ink). Call after the
    y range is final."""
    if 'cluster' not in cons:
        return
    t = np.sort(cons['window_center'].dropna().unique())
    half = np.median(np.diff(t)) / 2 if len(t) > 1 else 0.05
    at = ax.get_xaxis_transform()               # x in data, y in axes units
    row = 0
    def texts(con):                       # (cluster sign, row label)
        if con == 'own_minus_other':
            return ((1, 'own > other'), (-1, 'own < other'))
        p = _PROP_SHORT[con]
        return ((1, f'25% {p} > 75% {p}'), (-1, f'25% {p} < 75% {p}'))

    for con, color in ((own, CHANGE_COLOR[own]), (other, CHANGE_COLOR[other]),
                       ('own_minus_other', INK)):
        d = cons[cons['contrast'] == con].sort_values('window_center')
        for sign, text in texts(con):
            mask = (d['cluster'] == sign).to_numpy()
            if not mask.any():
                continue
            y = 1.05 + 0.08 * row
            for start, end in true_runs(mask):
                ax.hlines(y, d['window_center'].iloc[start] - half,
                          d['window_center'].iloc[end] + half, color=color, lw=2,
                          transform=at, clip_on=False)
            ax.annotate(text, (1.0, y), xycoords='axes fraction', xytext=(4, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=FONT_SIZE - 1.5, color=LABEL_INK, annotation_clip=False)
            row += 1


def _same_ylim(axes):
    """Give ``axes`` one y range, the union of their own (autoscaled) ranges,
    so panels that compare two variables are read on the same scale."""
    lims = np.array([ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_ylim(lims[:, 0].min(), lims[:, 1].max())


def plot_proportion_differences(traj, label, out_path):
    """The proportion trajectories as differences, for one axis window: each
    variable's separation vector (i − c, or s − r) in the 25 % and in the 75 %
    block of a proportion, as a path in the congruency (x) / task-seq. (y)
    plane (oblique coordinates, as `plot_proportion_trajectories`; open circle
    = first window, arrow = last). Two paths per panel instead of four.
    Rows = variable, columns = proportion, own sets on the diagonal; all panels
    on one scale. Also writes each panel alone, on that scale,
    ``<stem>_<variable>_<proportion>``. The prediction: on the diagonal the
    25 % and 75 % paths part along the panel's own axis; off the diagonal
    they overlap.

    Their 25 % − 75 % differences, per axis and window, with CIs and the block
    null's p, are ``proportion_changes.csv`` (`proportion_trajectories`).
    ``traj`` is the ``proportion_trajectories`` table."""
    traj = traj[traj['axes'] == label]
    return _plane_panels(traj, _difference_panel, SIZES['proportion_differences'],
                         _stem(out_path))


def _difference_panel(ax, d, var, prop):
    """One `plot_proportion_differences` panel: `d` is the set's rows."""
    levels = {'congruency': ('i', 'c'), 'switchType': ('s', 'r')}[var]
    for pct in ('25', '75'):
        hi, lo = (d[d['condition'] == f'{lv}{pct}'].sort_values('window') for lv in levels)
        x = hi['congruency'].to_numpy() - lo['congruency'].to_numpy()
        y = hi['switch'].to_numpy() - lo['switch'].to_numpy()
        _path(ax, x, y, PROP_COLOR[prop][_PCT_LEVEL[pct]], '-', f'{pct}% {_PROP_SHORT[prop]}')
    ax.set_title(f'{levels[0]} − {levels[1]}, by {NAMES[prop]}', loc='left')


def _proportion_panel(ax, d, var, prop):
    """One `plot_proportion_trajectories` panel: `d` is the set's rows."""
    for cond, dc in d.groupby('condition', sort=False):
        dc = dc.sort_values('window')
        level, pct = cond[0], cond[1:]
        _path(ax, dc['congruency'].to_numpy(), dc['switch'].to_numpy(),
              PROP_COLOR[prop][_PCT_LEVEL[pct]], LEVEL_LS[level],
              f'{level}, {pct}% {_PROP_SHORT[prop]}')
    var_name = NAMES['congruency' if var == 'congruency' else 'switch']
    ax.set_title(f'{var_name} × {NAMES[prop]}', loc='left')


def _path(ax, x, y, color, ls, label):
    """A path in the plane: open circle at the first window, arrow at the last."""
    ax.plot(x, y, color=color, ls=ls, lw=LW, label=label)
    ax.plot(x[0], y[0], 'o', ms=3, mfc='white', mec=color, mew=0.8, zorder=3)
    if len(x) > 1:
        ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8, mutation_scale=7))


def _plane_panels(traj, draw, size, stem):
    """The 2 x 2 proportion figure (rows = variable, columns = proportion, one
    shared scale) drawn by ``draw(ax, rows, variable, proportion)``, written to
    `stem`; then each panel alone on the same scale, to
    ``<stem>_<variable>_<proportion>`` (single panels for Illustrator)."""
    sets = [(r, c, var, prop) for r, var in enumerate(('congruency', 'switchType'))
            for c, prop in enumerate(('incProp', 'switchProp'))]

    def finish(ax, xlabel=True, ylabel=True):
        _zero(ax, vertical=True)
        ax.legend(loc='best')
        if xlabel:
            ax.set_xlabel(_sentence(f"{NAMES['congruency']} axis (z)"))
        if ylabel:
            ax.set_ylabel(_sentence(f"{NAMES['switch']} axis (z)"))

    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 2, figsize=size, sharex=True, sharey=True,
                                gridspec_kw=dict(hspace=0.3, wspace=0.1))
        for r, c, var, prop in sets:
            draw(axs[r, c], traj[traj['set'] == f'{var} x {prop}'], var, prop)
            finish(axs[r, c], xlabel=r == 1, ylabel=c == 0)
        xlim, ylim = axs[0, 0].get_xlim(), axs[0, 0].get_ylim()
        paths = _save(fig, stem)
        for r, c, var, prop in sets:
            fig, ax = plt.subplots(figsize=(size[0] / 2, size[1] / 2))
            draw(ax, traj[traj['set'] == f'{var} x {prop}'], var, prop)
            finish(ax)
            ax.set(xlim=xlim, ylim=ylim)
            paths += _save(fig, f'{stem}_{_FILE_NAME[var]}_{_PROP_FILE[prop]}')
    return paths


def plot_proportion_change_vectors(prop_summary, label, out_path):
    """The proportion changes in one picture, for one axis window: each
    variable's time-averaged 25 % − 75 % change of its separation vector
    (i − c, s − r) as an arrow from the origin in the congruency (x) /
    task-seq. (y) plane, oblique coordinates as `plot_proportion_trajectories`.
    Own proportions (i − c by inc. prop., s − r by swi. prop.) solid, in the
    proportion's colour, with the bootstrap CI of each coordinate as a cross at
    the tip (marginal CIs, not a joint region); the other proportions thin and
    dashed, the controls. A change along the variable's own axis only means
    the adaptation is specific to that code; a component along the other axis
    means it runs partly through the shared direction.

    ``prop_summary`` is ``proportion_summary.json`` (`proportion_trajectories`):
    per set and axis, the overall (all windows) estimate, CI and block-null p."""
    sets = prop_summary['axes'][label]['sets']
    rows = (('congruency x incProp', 'incProp', 'i − c, by inc. prop.'),
            ('switchType x switchProp', 'switchProp', 's − r, by swi. prop.'),
            ('congruency x switchProp', 'switchProp', 'i − c, by swi. prop.'),
            ('switchType x incProp', 'incProp', 's − r, by inc. prop.'))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=SIZES['proportion_change_vectors'])
        _zero(ax, vertical=True)
        extent = [0.0]
        for name, prop, text in rows:
            s = sets[name]
            x, y = s['congruency']['estimate'], s['switch']['estimate']
            color, own = CHANGE_COLOR[prop], s['own']
            ax.annotate('', xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=LW if own else 0.7, ls='-' if own else '--',
                                        mutation_scale=8 if own else 5,
                                        alpha=1.0 if own else 0.6,
                                        shrinkA=0, shrinkB=0))     # tip on the estimate
            ax.plot([], [], color=color, lw=LW if own else 0.7, ls='-' if own else '--',
                    alpha=1.0 if own else 0.6, label=text)
            if own:
                (xl, xh), (yl, yh) = s['congruency']['ci'], s['switch']['ci']
                ax.plot([xl, xh], [y, y], color=color, lw=0.8)
                ax.plot([x, x], [yl, yh], color=color, lw=0.8)
                extent += [abs(v) for v in (xl, xh, yl, yh)]
            extent += [abs(x), abs(y)]
        lim = 1.15 * max(extent)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal',
               xlabel=_sentence(f"25% − 75%, {NAMES['congruency']} axis (z)"),
               ylabel=_sentence(f"25% − 75%, {NAMES['switch']} axis (z)"))
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
        return _save(fig, _stem(out_path))


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

    Also writes each panel alone, on the same scale,
    ``<stem>_<variable>_<proportion>``.

    ``traj`` is the ``proportion_trajectories`` table; ``changes`` is accepted
    for callers and not drawn."""
    traj = traj[traj['axes'] == label]
    return _plane_panels(traj, _proportion_panel, SIZES['proportion_trajectories'],
                         _stem(out_path))
