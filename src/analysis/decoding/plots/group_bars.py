"""Bar figure for :func:`src.analysis.decoding.window_mean_decoding.compare_electrode_groups`."""
import os

import numpy as np
import matplotlib.pyplot as plt

from src.analysis.config.plotting_parameters import ELECTRODE_SET_COLORS
from .style import nature_style

# Groups named in ELECTRODE_SET_COLORS (sig, nonsig, all) take the paper's
# electrode-set colours, as in the electrode-count curves; any other, _BAR.
_BAR, _INK = '#0173B2', '#262626'


def _fmt_p(p):
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def plot_group_accuracy_bars(result, out_stem, title=None, base_fontsize=9.0):
    """One bar per electrode group: mean accuracy at the matched size (or on all
    the group's electrodes when ``result['n_match']`` is None), whiskers
    the 2.5-97.5 percentiles over bootstraps and repeats, a dot per bootstrap
    mean, chance (shuffled labels) as a dark dashed line across each bar, and,
    for matched runs only, a bracket per tested pair with its Holm-adjusted
    permutation p. ``result`` is what ``compare_electrode_groups`` returns.
    ``title`` is accepted for callers and not drawn; what the whiskers, dashes
    and p values are goes in the caption. Writes ``<out_stem>.png``/``.pdf``.
    """
    summary, tests, acc = result['summary'], result['tests'], result['accuracies']
    # unmatched, a pair's p compares each group with random groups of its own
    # size, not the two bars drawn, so it stays in tests.csv and gets no bracket
    if result['n_match'] is None:
        tests = tests.iloc[:0]
    groups = list(summary['group'])
    x = {g: i for i, g in enumerate(groups)}
    with plt.rc_context(nature_style(base_fontsize)):
        fig, ax = plt.subplots(figsize=(0.75 * len(groups) + 1.2, 2.6))
        for r in summary.itertuples():
            i = x[r.group]
            ax.bar(i, r.mean, width=0.6, color=ELECTRODE_SET_COLORS.get(r.group, _BAR),
                   alpha=0.85, zorder=2)
            ax.plot([i, i], [r.lo, r.hi], color=_INK, lw=0.8, zorder=3)
            # dark, so it shows on the yellow bar as well as the grey and olive
            ax.plot([i - 0.3, i + 0.3], [r.chance] * 2, color=_INK, lw=1.0, ls='--', zorder=4)
            boots = acc[acc['group'] == r.group].groupby('bootstrap')['accuracy'].mean()
            jitter = np.linspace(-0.12, 0.12, len(boots)) if len(boots) > 1 else [0.0]
            ax.scatter(i + np.asarray(jitter), boots, s=6, color=_INK, alpha=0.6,
                       linewidths=0, zorder=5)
        top = float(summary['hi'].max())
        step = 0.05
        # stack the brackets shortest first, so a long bracket sits above the
        # short ones and never runs through their p values (each p is centred
        # over its bracket, where a shorter bracket's end could be)
        pairs = sorted(tests.itertuples() if len(tests) else (),
                       key=lambda t: abs(x[t.group_1] - x[t.group_2]))
        for k, t in enumerate(pairs):
            y = top + step * (k + 1)
            i1, i2 = x[t.group_1], x[t.group_2]
            ax.plot([i1, i1, i2, i2], [y - 0.01, y, y, y - 0.01], color=_INK, lw=0.7)
            ax.text((i1 + i2) / 2, y + 0.008, _fmt_p(t.p_holm), ha='center', va='bottom',
                    fontsize=base_fontsize - 2, color=_INK)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n = {n})" for g, n in
                            zip(groups, summary['n_electrodes'])])
        ax.set_ylabel(f"accuracy, {result['n_match']} electrodes per group"
                      if result['n_match'] is not None else "accuracy, all electrodes per group")
        lo = min(0.4, float(summary['chance'].min()) - 0.05)
        ax.set_ylim(lo, top + step * (len(tests) + 1.5))
        ax.axhline(0.5, color='#cfcfcf', lw=0.6, zorder=0)
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
        for p in paths:
            fig.savefig(p)
        plt.close(fig)
    return paths
