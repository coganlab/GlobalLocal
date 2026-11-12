#!/usr/bin/env python
"""
The paper's power-trace (Fig 3) and decoding (Fig 4) panels, redrawn from the
saved outputs at one size and style, for assembly in Illustrator.

Every panel uses `panel_layout.PANEL_MM`: same canvas, axes at the same place,
same x range (-1 to 1.5 s) and the same y range within each figure, so panels
line up when stacked. Each is saved full and without its x and/or y axis label
(`panel_layout.save_panel_variants`), as PNG and TIFF (450 dpi) and PDF.

Fig 3: the windowed-ANOVA runs' per-electrode cell means (`*_evoked.npz`). By
default (`PAPER_DESIGN`) each term comes from its own pooled 2 x 2 run, whose
four cells hold every trial of that condition at that proportion level.
`--power-design three_way` draws from the two 8-cell block-balanced runs, and
`four_way` from the full 16-cell run; there each trace is the equal-weight mean
of the term's cells over the factors outside it. Mean +- SEM across electrodes,
with the term's cluster-corrected interaction mask (`*_interaction_cluster.npz`).
Colour = proportion block (`PROPORTION_COLORS`, shared with the behavioural and
decoding figures), line style = condition (solid c / r, dashed i / s); one
legend entry per trace, e.g. 'r, 25% S'. Each term also gets a difference
panel: the per-block effect (i − c or s − r), one solid line per block, with
the same interaction bar.

Fig 4: the decoding results files (`replot.context_comparison_panel`): the two
blocks' accuracies, the pooled shuffle, the contrast and against-chance bars;
plus each comparison's accuracy difference.

Writes <out>/<panel><variant>.<png|pdf|tif> and <out>/paper_panels_config.json.

    python dcc_scripts/vis/paper_panels.py [--subject-set good] [--power-design four_way] [--out DIR]
"""
import argparse
import glob
import json
import os
import re
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.analysis.config.plotting_parameters import PROPORTION_COLORS
from src.analysis.config.condition_registry import get_conditions_obj
from src.analysis.decoding.plots.accuracies import plot_accuracies_with_multiple_sig_clusters
from src.analysis.decoding.plots.replot import context_comparison_panel, load_master_results
from src.analysis.utils.panel_layout import PANEL_MM, save_panel_variants

EPOCHS = ('Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_'
          '70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20')
DCC = os.path.join(project_root, 'dcc_scripts')

# --- choices for these figures -------------------------------------------------
XLIM = (-1.0, 1.5)                    # s, every panel
DECODING_YLIM = (0.3, 0.95)              # the highest mean + SD drawn is 0.94
BASE_FONTSIZE = 8.0                   # pt: axis labels 8, ticks 7, legend 6.5, row labels 7
SIG_ROW_PITCH = 0.065                 # share of the axes height between significance rows: 7-pt labels clear each other
BETWEEN_BLOCK_COLOR = '#0b0b0b'       # every between-block test (power interaction, decoding 25 % vs 75 %); colour marks vs chance
# Fig 3 panels: (name, the term's two factors). Rows of the figure: switch cost,
# congruency effect; columns: switch, incongruent proportion.
POWER_PANELS = (('fig3_switch_x_switchprop_LWPS', ('switchType', 'switchProportion')),
                ('fig3_switch_x_incprop_cross', ('switchType', 'incongruentProportion')),
                ('fig3_congruency_x_switchprop_cross', ('congruency', 'switchProportion')),
                ('fig3_congruency_x_incprop_LWPC', ('congruency', 'incongruentProportion')))
# The windowed-ANOVA condition set behind each Fig 3 term, by design. A term is
# '<condition>_x_<proportion>', as in the runs' interaction file names.
# two_way: one pooled 4-cell run per term; each cell holds every trial of that
#   condition at that proportion level, so everything else is trial-weighted.
# three_way: the 8-cell block-balanced runs; each cell is one block type,
#   pooling the other trial type at that block's ratio.
# four_way: one 16-cell run, every trial type its own cell.
TERMS = [f'{c}_x_{p}' for _, (c, p) in POWER_PANELS]
POWER_CONDITIONS = {
    'two_way': {'switchType_x_switchProportion': 'stimulus_lwps_conditions',
                'switchType_x_incongruentProportion': 'stimulus_switch_type_by_incongruent_proportion_conditions',
                'congruency_x_switchProportion': 'stimulus_congruency_by_switch_proportion_conditions',
                'congruency_x_incongruentProportion': 'stimulus_lwpc_conditions'},
    'three_way': {t: 'stimulus_lwpc_block_balanced_conditions' if t.startswith('congruency')
                  else 'stimulus_lwps_block_balanced_conditions' for t in TERMS},
    'four_way': {t: 'stimulus_experiment_conditions' for t in TERMS},
}
PAPER_DESIGN = 'two_way'              # the design the paper's Fig 3 uses; the others are drawn on request
# Fig 4 and its cross-effect supplement: decoding job per panel, by subject
# set. all24 has three runs per condition (10-06 twice, 10-07), so each panel
# names its job instead of globbing.
DECODING_JOBS = {
    'all24': {'fig4_LWPC': 57902167, 'fig4_LWPS': 57902168,
              'sfig4_congruency_by_switchprop_cross': 57902169,
              'sfig4_switch_by_incprop_cross': 57902170},
    'good': {'fig4_LWPC': 60508450, 'fig4_LWPS': 60508451,
             'sfig4_congruency_by_switchprop_cross': 60508452,
             'sfig4_switch_by_incprop_cross': 60508453},
}

# Legends use the abbreviations the paper defines earlier: c / i / r / s, and
# I / S for the block proportions.
ABBREV = {'incongruent': 'I', 'switch': 'S'}    # decoding legends: '25% switch' -> '25% S'
PROPORTION_LETTER = {'incongruentProportion': 'I', 'switchProportion': 'S'}
# Legend label -> colour, e.g. '25% I' -> the shared 25 % incongruent colour
COLORS = {f'{pct}% {letter}': PROPORTION_COLORS[factor][pct]
          for factor, letter in PROPORTION_LETTER.items() for pct in PROPORTION_COLORS[factor]}
EASY = ('c', 'r')                     # solid; the other condition dashed
# Difference-panel y label, by the term's condition factor
EFFECT_LABEL = {'congruency': 'i − c (z)', 'switchType': 's − r (z)'}


def abbreviate(text):
    """``text`` with the condition words of `ABBREV` shortened, e.g.
    '25% incongruent' -> '25% I'."""
    return re.sub(r'\b(' + '|'.join(ABBREV) + r')\b', lambda m: ABBREV[m.group(1)], text)


def abbreviated(panel):
    """A `context_comparison_panel` dict with its trace labels abbreviated
    (`abbreviate`) in every place they key: the accuracies, colours, line
    styles and ``labels``."""
    out = dict(panel)
    for key in ('accuracies_dict', 'colors', 'linestyles'):
        out[key] = {abbreviate(k): v for k, v in panel[key].items()}
    out['labels'] = tuple(abbreviate(lab) for lab in panel['labels'])
    return out


def power_panel(run_dir, condition_label, factors, n_subjects, roi='lpfc'):
    """One Fig 3 panel's data from a windowed-ANOVA run of ``condition_label``:
    ``(times, traces, mask)``. ``traces`` maps (condition, block), e.g.
    ``('c', '25% S')``, to (electrodes, times): the equal-weight mean of every
    cell at that condition and block, so the factors outside the term count
    equally. ``mask`` is the term's cluster mask over ``times``. Raises if the
    cells' electrodes differ, since cells are averaged and subtracted electrode
    by electrode."""
    condition, proportion = factors
    stem = os.path.join(run_dir, roi, f'{condition_label}_{n_subjects}_subjects')
    traces, times, channels = {}, None, None
    for cell, levels in get_conditions_obj(condition_label).items():
        z = np.load(f'{stem}_{cell}_{roi}_evoked.npz')
        if channels is None:
            channels = z['ch_names']
        elif not np.array_equal(z['ch_names'], channels):
            raise ValueError(f"{cell}: electrodes differ from the run's other cells")
        key = (levels[condition], f'{levels[proportion]} {PROPORTION_LETTER[proportion]}')
        traces.setdefault(key, []).append(z['data'])
        times = z['times']
    mask = np.load(f'{stem}_{roi}_{condition}_x_{proportion}_interaction_cluster.npz')['mask']
    return times, {k: np.mean(v, axis=0) for k, v in traces.items()}, mask.astype(bool)


def effect_traces(traces):
    """Per-block effect, per electrode, from `power_panel` traces:
    ``{(None, block): i − c or s − r}``. Keyed like the traces so `draw_power`
    styles them the same way; no condition -> solid line, block-only legend."""
    out = {}
    for (condition, block), data in traces.items():
        if condition not in EASY:
            easy = next(c for c, b in traces if b == block and c in EASY)
            out[(None, block)] = data - traces[(easy, block)]
    return out


def common_power_ylim(panels, symmetric=False):
    """One y range for a set of Fig 3 panels ``[(times, traces, mask)]``: the
    mean +- SEM extremes within XLIM, rounded out to 0.1. ``symmetric`` makes
    it +- the larger extreme, for difference panels centred on zero."""
    lo, hi = np.inf, -np.inf
    for times, traces, _ in panels:
        sel = (times >= XLIM[0]) & (times <= XLIM[1])
        for data in traces.values():
            m = data.mean(0)[sel]
            sem = data.std(0)[sel] / np.sqrt(len(data))
            lo, hi = min(lo, (m - sem).min()), max(hi, (m + sem).max())
    lo, hi = np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10
    if symmetric:
        hi = max(-lo, hi)
        lo = -hi
    return lo, hi


def sig_layout(n_rows):
    """Significance-strip keywords for a figure whose fullest panel has
    ``n_rows`` rows. Every panel of the figure is laid out for that many
    slots, so its rows sit at the same heights as in the others, and the strip
    grows with ``n_rows`` so rows are always `SIG_ROW_PITCH` apart. (0.035 is
    the strip's inset above and below its rows, set in
    `accuracies._draw_sig_strip`.)"""
    return dict(sig_band_frac=0.035 + SIG_ROW_PITCH * max(n_rows - 1, 1),
                sig_row_slots=n_rows, sig_split_contrasts=True)


def n_sig_rows(sig):
    """Rows a significance dict draws with each contrast direction on its own
    row: one per entry with a significant cluster."""
    return int(sum(np.asarray(e['clusters']).any() for e in sig.values()))


def draw_power(times, traces, mask, ylim, layout, ylabel='High gamma (z)'):
    """``draw(ax)`` for one Fig 3 panel. ``traces`` maps (condition, block) to
    (electrodes, times): colour = block, dashed = i / s, solid otherwise;
    legend 'c, 25% I', or just '25% I' for a difference wave (condition None,
    `effect_traces`), whose panel also gets a zero line. Plus the interaction bar."""
    order = sorted(traces, key=lambda k: (k[1], k[0] not in EASY))
    labels = {k: k[1] if k[0] is None else f'{k[0]}, {k[1]}' for k in order}
    difference = any(k[0] is None for k in order)    # an effect_traces panel

    def draw(ax):
        plot_accuracies_with_multiple_sig_clusters(
            time_points=times, accuracies_dict={labels[k]: traces[k] for k in order},
            significance_clusters_dict={'interaction': {
                'clusters': mask, 'label': None if difference else 'Interaction',
                'color': BETWEEN_BLOCK_COLOR, 'kind': 'contrast'}},
            colors={labels[k]: COLORS[k[1]] for k in order},
            linestyles={labels[k]: '-' if k[0] in (*EASY, None) else '--' for k in order},
            band='sem', ylim=ylim, xlim=XLIM, ylabel=ylabel,
            show_chance_level=False, show_legend=True, legend_loc='right',
            show_sig_legend=True, sig_legend_loc='rows', **layout,
            base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax)
        if difference:
            ax.axhline(0, color='#BFBFBF', linewidth=0.8, zorder=0)
    return draw


def paper_sig_entries(panel):
    """The panel's significance rows in one scheme for every figure: each
    contrast direction in black, labelled '25% > 75%' or '75% > 25%', and
    each block against chance in its block's colour, labelled
    '25% > chance'. The legend beside them names the variable."""
    a, b = (lab.split(' ')[0] for lab in panel['labels'])        # '25%', '75%'
    full_a = panel['labels'][0]
    out, n_contrast = {}, 0
    for key, entry in panel['significance_clusters_dict'].items():
        entry = dict(entry)
        if entry.get('kind') == 'chance':
            who = a if entry.get('label', '').startswith(full_a) else b
            entry['label'] = f'{who} > chance'
        else:
            # `replot._contrast_entries` stores the two directions in order:
            # first labels[0] > labels[1], then the reverse.
            entry.update(label=f'{a} > {b}' if n_contrast == 0 else f'{b} > {a}',
                         color=BETWEEN_BLOCK_COLOR, linestyle='-')
            n_contrast += 1
        out[key] = entry
    return out


def draw_decoding(panel, layout):
    """``draw(ax)`` for one Fig 4 comparison panel (`context_comparison_panel`)."""
    short = abbreviated(panel)
    kwargs = {k: short[k] for k in ('time_points', 'accuracies_dict', 'colors', 'linestyles')}
    kwargs['significance_clusters_dict'] = paper_sig_entries(panel)

    def draw(ax):
        plot_accuracies_with_multiple_sig_clusters(
            **kwargs, ylim=DECODING_YLIM, xlim=XLIM, ylabel='Decoding accuracy',
            show_chance_level=False, show_sig_legend=True, legend_loc='right',
            sig_legend_loc='rows', **layout,
            base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax)
    return draw


def difference_panel(panel):
    """The (25 % - 75 %) accuracy difference and the contrast bars of one
    comparison, as ``(times, {label: (samples, times)}, significance)``."""
    a, b = panel['labels']
    diff = np.asarray(panel['accuracies_dict'][a], float) - np.asarray(panel['accuracies_dict'][b], float)
    # Black bars, no row label: beside a difference wave the curve's sign gives the direction.
    sig = {k: dict(v, label='') for k, v in paper_sig_entries(panel).items()
           if v.get('kind') != 'chance'}
    return panel['time_points'], {f'{abbreviate(a)} − {abbreviate(b)}': diff}, sig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject-set', default='good', help="the power and decoding runs' set: good or all24")
    p.add_argument('--power-design', default=PAPER_DESIGN, choices=POWER_CONDITIONS,
                   help=f'Fig 3 source (default {PAPER_DESIGN}): two_way = one pooled 4-cell run per term, '
                        'three_way = the 8-cell block-balanced runs, four_way = the 16-cell run')
    p.add_argument('--out', default=None,
                   help='default dcc_scripts/vis/paper_panels for the paper design, paper_panels_<design> otherwise')
    args = p.parse_args()
    out = args.out or os.path.join(
        DCC, 'vis', 'paper_panels' if args.power_design == PAPER_DESIGN else f'paper_panels_{args.power_design}')
    conditions = POWER_CONDITIONS[args.power_design]

    suffix = '' if args.subject_set == 'all24' else f'_subjects-{args.subject_set}'
    n_subjects = 24 if args.subject_set == 'all24' else 23
    power_dir = os.path.join(DCC, 'power', 'figs', EPOCHS, f'anova_within_roi{suffix}')
    os.makedirs(out, exist_ok=True)

    dec_dir = os.path.join(DCC, 'decoding', 'figs', EPOCHS + suffix)
    decoding_files, decoding_panels = {}, {}
    for name, job in DECODING_JOBS[args.subject_set].items():
        decoding_files[name] = glob.glob(os.path.join(dec_dir, f'*_MASTER_RESULTS_job{job}_*.pkl'))[0]
        decoding_panels[name] = context_comparison_panel(load_master_results(decoding_files[name]), 'lpfc')
    # One strip layout per figure, sized for its fullest panel: the main figure
    # (power and fig4) and the cross-effect supplement (sfig4).
    rows = {name: n_sig_rows(paper_sig_entries(panel)) for name, panel in decoding_panels.items()}
    layouts = {fig: sig_layout(max([1] + [n for name, n in rows.items() if name.startswith(fig)]))
               for fig in ('fig4_', 'sfig4_')}

    power = [power_panel(power_dir, conditions[f'{c}_x_{p}'], (c, p), n_subjects) for _, (c, p) in POWER_PANELS]
    power_ylim = common_power_ylim(power)
    effects = [(times, effect_traces(traces), mask) for times, traces, mask in power]
    effect_ylim = common_power_ylim(effects, symmetric=True)
    written = []
    for (name, factors), raw, effect in zip(POWER_PANELS, power, effects):
        written += save_panel_variants(draw_power(*raw, power_ylim, layouts['fig4_']), os.path.join(out, name))
        written += save_panel_variants(
            draw_power(*effect, effect_ylim, layouts['fig4_'], ylabel=EFFECT_LABEL[factors[0]]),
            os.path.join(out, f'{name}_difference'))

    diffs = []
    for name, panel in decoding_panels.items():
        layout = layouts['sfig4_' if name.startswith('sfig4_') else 'fig4_']
        written += save_panel_variants(draw_decoding(panel, layout), os.path.join(out, name))
        if name.startswith('fig4_'):
            diffs.append((f'{name}_difference', difference_panel(panel)))
    # one symmetric y range for the difference panels
    spread = max(np.max(np.abs(d.mean(0)) + d.std(0)) for _, (_, acc, _) in diffs for d in acc.values())
    diff_ylim = (-np.ceil(spread * 10) / 10, np.ceil(spread * 10) / 10)
    for name, (times, acc, sig) in diffs:
        label = next(iter(acc))
        written += save_panel_variants(
            lambda ax, times=times, acc=acc, sig=sig, label=label: plot_accuracies_with_multiple_sig_clusters(
                time_points=times, accuracies_dict=acc, significance_clusters_dict=sig,
                colors={label: '#404040'}, ylim=diff_ylim, xlim=XLIM, ylabel='Accuracy difference',
                show_chance_level=True, chance_level=0.0, show_legend=True, legend_loc='right',
                show_sig_legend=True, sig_legend_loc='rows', **layouts['fig4_'],
                base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax),
            os.path.join(out, name))

    with open(os.path.join(out, 'paper_panels_config.json'), 'w') as f:
        json.dump(dict(subject_set=args.subject_set, power_design=args.power_design,
                       power_conditions=conditions, power_run=power_dir,
                       decoding_files=decoding_files, panel_mm=PANEL_MM, xlim=XLIM,
                       power_ylim=list(map(float, power_ylim)),
                       power_difference_ylim=list(map(float, effect_ylim)),
                       decoding_ylim=DECODING_YLIM,
                       difference_ylim=list(map(float, diff_ylim)), base_fontsize=BASE_FONTSIZE,
                       sig_layouts=layouts), f, indent=2)
    print(f"wrote {len(written)} files to {out} (power y {power_ylim}, power difference y {effect_ylim}, difference y {diff_ylim})")


if __name__ == '__main__':
    main()
