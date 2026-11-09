#!/usr/bin/env python
"""
The paper's power-trace (Fig 3) and decoding (Fig 4) panels, redrawn from the
saved outputs at one size and style, for assembly in Illustrator.

Every panel uses `panel_layout.PANEL_MM`: same canvas, axes at the same place,
same x range (-1 to 1.5 s) and the same y range within each figure, so panels
line up when stacked. Each is saved full and without its x and/or y axis label
(`panel_layout.save_panel_variants`), as PNG and TIFF (450 dpi) and PDF.

Fig 3: the windowed-ANOVA run's per-electrode cell means (`*_evoked.npz`),
averaged to the four sub-cells of each 2 x 2 term with the other proportion's
blocks weighted equally, mean +- SEM across electrodes, and the term's
cluster-corrected interaction mask (`*_interaction_cluster.npz`).
Colour = proportion block (`PROPORTION_COLORS`, shared with the behavioural and
decoding figures), line style = condition (solid c / r, dashed i / s); one
legend entry per trace, e.g. 'r, 25% S'. Each term also gets a difference
panel: the per-block effect (i − c or s − r), one solid line per block, with
the same interaction bar.

Fig 4: the decoding results files (`replot.context_comparison_panel`): the two
blocks' accuracies, the pooled shuffle, the contrast and against-chance bars;
plus each comparison's accuracy difference.

Writes <out>/<panel><variant>.<png|pdf|tif> and <out>/paper_panels_config.json.

    python dcc_scripts/vis/paper_panels.py [--subject-set good] [--out DIR]
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
SIG_BAND_FRAC = 0.10                  # share of the axes height for the significance rows
# Fig 3 panels: (name, condition set, the term's two factors). Rows of the
# figure: switch cost, congruency effect; columns: switch, incongruent proportion.
POWER_PANELS = (('fig3_switch_x_switchprop_LWPS', 'lwps', ('switchType', 'switchProportion')),
                ('fig3_switch_x_incprop_cross', 'lwps', ('switchType', 'incongruentProportion')),
                ('fig3_congruency_x_switchprop_cross', 'lwpc', ('congruency', 'switchProportion')),
                ('fig3_congruency_x_incprop_LWPC', 'lwpc', ('congruency', 'incongruentProportion')))
# Fig 4 and its cross-effect supplement: (name, decoding job id)
DECODING_PANELS = (('fig4_LWPC', 57902167), ('fig4_LWPS', 57902168),
                   ('sfig4_congruency_by_switchprop_cross', 57902169),
                   ('sfig4_switch_by_incprop_cross', 57902170))

# Cell names: Stimulus_<c|i|r|s>_<MC|MI>_<MR|MS>; MC / MI = 25 / 75 % incongruent
# blocks, MR / MS = 25 / 75 % switch blocks. Legends use the abbreviations the
# paper defines earlier: c / i / r / s, and I / S for the block proportions.
ABBREV = {'incongruent': 'I', 'switch': 'S'}    # decoding legends: '25% switch' -> '25% S'
# Proportion factor -> (its level's position in the cell name, legend letter,
# cell-name level -> % of the block)
PROPORTION = {'incongruentProportion': (1, 'I', {'MC': 25, 'MI': 75}),
              'switchProportion': (2, 'S', {'MR': 25, 'MS': 75})}
# Legend label -> colour, e.g. '25% I' -> the shared 25 % incongruent colour
COLORS = {f'{pct}% {letter}': PROPORTION_COLORS[factor][pct]
          for factor, (_, letter, levels) in PROPORTION.items() for pct in levels.values()}
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


def power_panel(run_dir, condition_set, factors, n_subjects):
    """One Fig 3 panel's data from a windowed-ANOVA run: ``(times, traces,
    mask)`` with traces ``{(condition, block): (electrodes, times)}`` (e.g.
    ``('c', '25% S')``), each the equal-weight mean of its two cells over the
    other proportion, and the term's cluster mask over ``times``. Raises if the
    cells' electrodes differ, since cells are averaged and subtracted
    electrode by electrode."""
    stem = f'stimulus_{condition_set}_block_balanced_conditions_{n_subjects}_subjects'
    condition, proportion = factors
    pos, letter, levels = PROPORTION[proportion]
    traces, times, channels = {}, None, None
    for path in sorted(glob.glob(os.path.join(run_dir, 'lpfc', f'{stem}_Stimulus_*_lpfc_evoked.npz'))):
        cell = os.path.basename(path).split('_Stimulus_')[1].split('_lpfc_')[0].split('_')
        key = (cell[0], f'{levels[cell[pos]]}% {letter}')
        z = np.load(path)
        if channels is None:
            channels = z['ch_names']
        elif not np.array_equal(z['ch_names'], channels):
            raise ValueError(f"{path}: electrodes differ from the run's other cells")
        traces.setdefault(key, []).append(z['data'])
        times = z['times']
    if len(traces) != 4 or any(len(v) != 2 for v in traces.values()):
        raise ValueError(f"expected 8 cells in {run_dir} for {stem}; got {traces.keys()}")
    term = f'{condition}_x_{proportion}'
    mask = np.load(os.path.join(run_dir, 'lpfc', f'{stem}_lpfc_{term}_interaction_cluster.npz'))['mask']
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


def draw_power(times, traces, mask, ylim, ylabel='High gamma (z)'):
    """``draw(ax)`` for one Fig 3 panel. ``traces`` maps (condition, block) to
    (electrodes, times): colour = block, dashed = i / s, solid otherwise;
    legend 'c, 25% I', or just '25% I' for a difference wave (condition None,
    `effect_traces`), whose panel also gets a zero line. Plus the interaction bar."""
    order = sorted(traces, key=lambda k: (k[1], k[0] not in EASY))
    labels = {k: k[1] if k[0] is None else f'{k[0]}, {k[1]}' for k in order}

    def draw(ax):
        plot_accuracies_with_multiple_sig_clusters(
            time_points=times, accuracies_dict={labels[k]: traces[k] for k in order},
            significance_clusters_dict={'interaction': {
                'clusters': mask, 'label': 'Interaction', 'color': '#0b0b0b', 'kind': 'contrast'}},
            colors={labels[k]: COLORS[k[1]] for k in order},
            linestyles={labels[k]: '-' if k[0] in (*EASY, None) else '--' for k in order},
            band='sem', ylim=ylim, xlim=XLIM, ylabel=ylabel,
            show_chance_level=False, show_legend=True, legend_loc='right',
            show_sig_legend=True, sig_legend_loc='rows', sig_band_frac=SIG_BAND_FRAC,
            base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax)
        if any(k[0] is None for k in order):
            ax.axhline(0, color='#BFBFBF', linewidth=0.8, zorder=0)
    return draw


def short_sig_labels(panel):
    """The panel's significance rows relabelled in one scheme for every
    figure: '25% > 75%' (or '75% > 25%') for the contrast, '25% > chance' for
    each block against chance. The legend beside them names the variable."""
    a, b = (lab.split(' ')[0] for lab in panel['labels'])        # '25%', '75%'
    full_a, full_b = panel['labels']
    out = {}
    for key, entry in panel['significance_clusters_dict'].items():
        entry = dict(entry)
        if entry.get('kind') == 'chance':
            who = a if entry.get('label', '').startswith(full_a) else b
            entry['label'] = f'{who} > chance'
        elif entry.get('label'):
            entry['label'] = f'{a} > {b}' if entry['label'].startswith(a) else f'{b} > {a}'
        out[key] = entry
    return out


def draw_decoding(panel):
    """``draw(ax)`` for one Fig 4 comparison panel (`context_comparison_panel`)."""
    short = abbreviated(panel)
    kwargs = {k: short[k] for k in ('time_points', 'accuracies_dict', 'colors', 'linestyles')}
    kwargs['significance_clusters_dict'] = short_sig_labels(panel)

    def draw(ax):
        plot_accuracies_with_multiple_sig_clusters(
            **kwargs, ylim=DECODING_YLIM, xlim=XLIM, ylabel='Decoding accuracy',
            show_chance_level=False, show_sig_legend=True, legend_loc='right',
            sig_legend_loc='rows', sig_band_frac=SIG_BAND_FRAC,
            base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax)
    return draw


def difference_panel(panel, master, roi='lpfc'):
    """The (25 % - 75 %) accuracy difference and the contrast bars of one
    comparison, as ``(times, {label: (samples, times)}, significance)``."""
    a, b = panel['labels']
    diff = np.asarray(panel['accuracies_dict'][a], float) - np.asarray(panel['accuracies_dict'][b], float)
    sig = {k: v for k, v in short_sig_labels(panel).items() if v.get('kind') != 'chance'}
    return panel['time_points'], {f'{abbreviate(a)} − {abbreviate(b)}': diff}, sig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject-set', default='good', help="power run's set: good or all24")
    p.add_argument('--out', default=os.path.join(DCC, 'vis', 'paper_panels'))
    args = p.parse_args()

    suffix = '' if args.subject_set == 'all24' else f'_subjects-{args.subject_set}'
    n_subjects = 24 if args.subject_set == 'all24' else 23
    power_dir = os.path.join(DCC, 'power', 'figs', EPOCHS, f'anova_within_roi{suffix}')
    os.makedirs(args.out, exist_ok=True)

    power = [power_panel(power_dir, cs, factors, n_subjects) for _, cs, factors in POWER_PANELS]
    power_ylim = common_power_ylim(power)
    effects = [(times, effect_traces(traces), mask) for times, traces, mask in power]
    effect_ylim = common_power_ylim(effects, symmetric=True)
    written = []
    for (name, _, factors), raw, effect in zip(POWER_PANELS, power, effects):
        written += save_panel_variants(draw_power(*raw, power_ylim), os.path.join(args.out, name))
        written += save_panel_variants(
            draw_power(*effect, effect_ylim, ylabel=EFFECT_LABEL[factors[0]]),
            os.path.join(args.out, f'{name}_difference'))

    dec_dir = os.path.join(DCC, 'decoding', 'figs', EPOCHS)
    decoding_files, diffs = {}, []
    for name, job in DECODING_PANELS:
        path = glob.glob(os.path.join(dec_dir, f'*_MASTER_RESULTS_job{job}_*.pkl'))[0]
        decoding_files[name] = path
        master = load_master_results(path)
        panel = context_comparison_panel(master, 'lpfc')
        written += save_panel_variants(draw_decoding(panel), os.path.join(args.out, name))
        if name.startswith('fig4_'):
            diffs.append((f'{name}_difference', difference_panel(panel, master)))
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
                show_sig_legend=True, sig_legend_loc='rows', sig_band_frac=SIG_BAND_FRAC,
                base_fontsize=BASE_FONTSIZE, linewidth=1.2, ax=ax),
            os.path.join(args.out, name))

    with open(os.path.join(args.out, 'paper_panels_config.json'), 'w') as f:
        json.dump(dict(subject_set=args.subject_set, power_run=power_dir,
                       decoding_files=decoding_files, panel_mm=PANEL_MM, xlim=XLIM,
                       power_ylim=list(map(float, power_ylim)),
                       power_difference_ylim=list(map(float, effect_ylim)),
                       decoding_ylim=DECODING_YLIM,
                       difference_ylim=list(map(float, diff_ylim)), base_fontsize=BASE_FONTSIZE,
                       sig_band_frac=SIG_BAND_FRAC), f, indent=2)
    print(f"wrote {len(written)} files to {args.out} (power y {power_ylim}, power difference y {effect_ylim}, difference y {diff_ylim})")


if __name__ == '__main__':
    main()
