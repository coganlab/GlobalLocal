"""Accuracy-by-electrode-count figure for
:func:`src.analysis.decoding.window_mean_decoding.accuracy_by_electrode_count`."""
import os

import matplotlib.pyplot as plt

from src.analysis.config.plotting_parameters import ELECTRODE_SET_COLORS
from .style import nature_style

# Groups named in ELECTRODE_SET_COLORS (sig, nonsig, all) take the paper's
# electrode-set colours; any other group takes the next of these slots.
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
_INK = '#262626'


def plot_accuracy_by_electrode_count(result, out_stem, title=None, base_fontsize=9.0):
    """Mean accuracy against the number of electrodes decoded from, on the log
    grid (1, 2, 4, ... electrodes and each group's full size) and a log x axis.
    Bands are the 2.5-97.5 percentiles over bootstraps x random subsets and the
    grey line is chance (0.5); both go in the caption. Each line ends at its
    group's full size. ``result`` is what ``accuracy_by_electrode_count``
    returns; its linear-grid counts are not drawn. ``title`` is accepted for
    callers and not drawn. Writes ``<out_stem>.png``/``.pdf``.
    """
    summary = result['summary']
    log_grid = summary[summary['log_grid'].astype(bool)]
    groups = list(dict.fromkeys(summary['group']))
    colors = {g: ELECTRODE_SET_COLORS.get(g, fallback) for g, fallback in zip(groups, _SERIES)}
    with plt.rc_context(nature_style(base_fontsize)):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for g in groups:
            s = log_grid[log_grid['group'] == g].sort_values('n_electrodes')
            ax.fill_between(s['n_electrodes'], s['lo'], s['hi'], color=colors[g],
                            alpha=0.18, linewidth=0)
            ax.plot(s['n_electrodes'], s['mean'], color=colors[g], lw=1.5, marker='o',
                    markersize=3, label=f"{g} (n = {s['n_electrodes'].max()})")
            # direct label at the line's end, in ink, so colour is not the only cue
            end = s.iloc[-1]
            ax.annotate(g, (end['n_electrodes'], end['mean']), xytext=(4, 0),
                        textcoords='offset points', va='center',
                        fontsize=base_fontsize - 2, color=_INK)
        ax.set_xscale('log')
        # label the grid's own counts (powers of two) rather than powers of ten
        ticks = [t for t in sorted(log_grid['n_electrodes'].unique()) if t & (t - 1) == 0]
        ax.set_xticks(ticks, labels=[str(t) for t in ticks])
        ax.minorticks_off()
        ax.axhline(0.5, color='#cfcfcf', lw=0.6, zorder=0)
        ax.set(xlabel='# of electrodes decoded from', ylabel='accuracy')
        ax.legend(loc='upper left')
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
        for p in paths:
            fig.savefig(p)
        plt.close(fig)
    return paths
