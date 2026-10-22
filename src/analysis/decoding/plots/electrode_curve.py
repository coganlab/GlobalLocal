"""Accuracy-by-electrode-count figure for
:func:`src.analysis.decoding.window_mean_decoding.accuracy_by_electrode_count`."""
import os

import matplotlib.pyplot as plt

from .style import nature_style

# categorical slots in fixed order, so a group keeps its color across figures
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
_INK, _MUTED = '#262626', '#737373'


def plot_accuracy_by_electrode_count(result, out_stem, title=None, base_fontsize=9.0):
    """Two panels of the same curves: the log grid (1, 2, 4, ... electrodes) on a
    log x axis, and the linear grid (every ``step`` electrodes) on a linear one.
    Lines are mean accuracy, bands the 2.5-97.5 percentiles over bootstraps x
    subsets; each line ends at its group's full size. ``result`` is what
    ``accuracy_by_electrode_count`` returns. Writes ``<out_stem>.png``/``.pdf``.
    """
    summary = result['summary']
    groups = list(dict.fromkeys(summary['group']))
    colors = dict(zip(groups, _SERIES))
    with plt.rc_context(nature_style(base_fontsize)):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
        for ax, grid, scale in zip(axes, ('log_grid', 'linear_grid'), ('log', 'linear')):
            for g in groups:
                s = summary[(summary['group'] == g) & summary[grid]].sort_values('n_electrodes')
                ax.fill_between(s['n_electrodes'], s['lo'], s['hi'], color=colors[g],
                                alpha=0.18, linewidth=0)
                ax.plot(s['n_electrodes'], s['mean'], color=colors[g], lw=1.5, marker='o',
                        markersize=3, label=f"{g} (n = {s['n_electrodes'].max()})")
                # direct label at the line's end, in ink, so color is not the only cue
                end = s.iloc[-1]
                ax.annotate(g, (end['n_electrodes'], end['mean']), xytext=(4, 0),
                            textcoords='offset points', va='center',
                            fontsize=base_fontsize - 2, color=_INK)
            ax.set_xscale(scale)
            if scale == 'log':
                # label the grid's own counts rather than powers of ten
                ticks = sorted(summary.loc[summary[grid], 'n_electrodes'].unique())
                ticks = [t for t in ticks if t & (t - 1) == 0]   # powers of two
                ax.set_xticks(ticks, labels=[str(t) for t in ticks])
                ax.minorticks_off()
            ax.axhline(0.5, color='#cfcfcf', lw=0.6, zorder=0)
            ax.set_xlabel('electrodes decoded' + (' (log scale)' if scale == 'log' else ''))
        axes[0].set_ylabel('accuracy')
        axes[0].legend(loc='upper left')
        if title:
            fig.suptitle(title, x=0.01, ha='left', fontsize=base_fontsize)
        fig.text(0.99, -0.04, 'bands: 2.5-97.5 % over bootstraps x random subsets; '
                 'grey line: 0.5', ha='right', va='top', fontsize=base_fontsize - 2.5,
                 color=_MUTED)
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
        for p in paths:
            fig.savefig(p)
        plt.close(fig)
    return paths
