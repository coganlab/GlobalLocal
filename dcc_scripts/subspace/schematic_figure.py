#!/usr/bin/env python
"""
Schematic of the coding-axes comparison (drafted 2026-11-09; not used in the
paper): left, how the cross cosine and its ceiling are computed from split
halves; right, the three geometries the cross can show (separate codes, partly
shared, one code). Illustrative numbers, not data. Styled with the constants
of subspace_figures.py.

    python dcc_scripts/subspace/schematic_figure.py --out <stem>    # writes <stem>.png/.pdf
"""
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from dcc_scripts.subspace.subspace_figures import RC, INK, GREY, MUTED, NULL_FILL, FONT_SIZE, NAMES

N_ELECTRODES = 9           # bars drawn per effect vector


def _effect_column(ax, x0, values, color, label):
    """One effect vector as a column of horizontal bars, one per electrode."""
    n = len(values)
    for i, v in enumerate(values):
        ax.plot([x0, x0 + 0.18 * v], [n - i] * 2, color=color, lw=3.2, solid_capstyle='butt')
    ax.plot([x0, x0], [0.4, n + 0.6], color=MUTED, lw=0.5)
    ax.text(x0, n + 1.4, label, ha='center', va='bottom')


def _arrow(ax, start, end, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7,
                                 shrinkA=0, shrinkB=0, **kw))


def plot_schematic(out_stem, seed=3):
    """Draw the schematic and write ``<out_stem>.png``/``.pdf``; returns the paths.
    ``seed`` only picks the illustrative bar lengths."""
    rng = np.random.default_rng(seed)
    n = N_ELECTRODES
    c = rng.normal(0, 1, n)
    s = 0.6 * c + 0.8 * rng.normal(0, 1, n)            # partly shared
    small = FONT_SIZE - 1
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.2, 2.3))

        # left: split halves -> per-electrode effects -> cross and ceiling
        left = fig.add_axes([0.0, 0.0, 0.52, 1.0])
        left.set_xlim(0, 10)
        left.set_ylim(-2.2, n + 3.2)
        left.set_axis_off()
        left.text(0.55, n + 2.4, "trials of each\nparticipant", ha='center', va='center')
        for y, label in ((n - 1.5, 'half A'), (2.5, 'half B')):
            left.add_patch(plt.Rectangle((0.05, y - 1.3), 1.0, 2.6, fc=NULL_FILL, ec=MUTED, lw=0.6))
            left.text(0.55, y, label, ha='center', va='center')
        noisy = lambda v: v + 0.3 * rng.normal(0, 1, n)   # each half's estimate is noisy
        cn, sn = NAMES['congruency'], NAMES['switch']
        _effect_column(left, 2.6, noisy(c), INK, f'{cn}\neffect, A')
        _effect_column(left, 3.9, noisy(s), GREY, f'{sn}\neffect, A')
        _effect_column(left, 5.4, noisy(c), INK, f'{cn}\neffect, B')
        _effect_column(left, 6.7, noisy(s), GREY, f'{sn}\neffect, B')
        left.text(2.0, n / 2 + 0.5, 'electrodes', rotation=90, ha='center', va='center',
                  color=MUTED, fontsize=small)
        _arrow(left, (1.1, n - 1.5), (2.15, n - 1.5), color=MUTED, lw=0.6)
        _arrow(left, (1.1, 2.5), (4.95, 2.5), color=MUTED, lw=0.6)
        left.text(8.55, n - 0.3, 'cross', ha='center', fontweight='bold')
        left.text(8.55, n - 1.4, f'cos({cn} A, {sn} B)\nand B with A', ha='center',
                  va='top', fontsize=small)
        left.text(8.55, n - 4.9, 'ceiling', ha='center', fontweight='bold', color=GREY)
        left.text(8.55, n - 6.0, f'cos({cn} A, {cn} B)\n× cos({sn} A, {sn} B),\n'
                  'square root', ha='center', va='top', fontsize=small, color=GREY)
        left.text(4.6, -1.6, 'one split; repeated over 200 splits and every time window',
                  ha='center', fontsize=small, color=MUTED)

        # right: the three geometries, each half's axis wobbling faintly
        for j, (title, angle, note) in enumerate((('separate codes', 90, 'cross ≈ 0'),
                                                 ('partly shared', 55, '0 < cross < ceiling'),
                                                 ('one code', 0, 'cross ≈ ceiling'))):
            ax = fig.add_axes([0.56 + j * 0.148, 0.12, 0.13, 0.7])
            ax.set_xlim(-0.25, 1.15)
            ax.set_ylim(-0.25, 1.15)
            ax.set_aspect('equal')
            ax.set_axis_off()
            th = np.deg2rad(angle)
            for wobble in np.deg2rad((-12, 12)):
                for base, color in ((0.0, INK), (th, GREY)):
                    ax.plot([0, np.cos(base + wobble)], [0, np.sin(base + wobble)], color=color,
                            lw=0.5, alpha=0.35)
            _arrow(ax, (0, 0), (1.0, 0.0), color=INK, lw=1.2)
            _arrow(ax, (0, 0), (np.cos(th), np.sin(th)), color=GREY, lw=1.2,
                   ls='--' if angle == 0 else '-')       # dashed so it shows on top of c
            ax.set_title(title, pad=2)
            ax.text(0.45, -0.22, note, ha='center', va='top', fontsize=small)
        fig.text(0.56, 0.93, f'{cn} axis', color=INK, fontsize=small)
        fig.text(0.70, 0.93, f'{sn} axis', color=GREY, fontsize=small)
        fig.text(0.80, 0.93, 'faint: one half of the trials', color=MUTED, fontsize=small)
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        for p in paths:
            fig.savefig(p, dpi=300, facecolor='white')
        plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output path without extension')
    args = p.parse_args()
    print('\n'.join(plot_schematic(args.out)))


if __name__ == '__main__':
    main()
