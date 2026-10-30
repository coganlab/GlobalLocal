#!/usr/bin/env python
"""
Draft panel for the overlap controls (N4 §19.7-19.8): the LWPC-LWPS overlap r
of an anatomy run under each control, with the base-effect control (each
half's congruency and switch effects partialled out) picked out. Reads the
run's `overlap_controls.csv`; writes `overlap_controls.png` / `.pdf` beside it.

    python dcc_scripts/stats/plot_overlap_controls.py --anatomy-dir <run>/continuous
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Rows drawn, top to bottom: (label in the csv, label on the axis). The
# descriptive responsiveness tertiles and the RT-coupling reference row (the
# same electrodes as the pre-specified row here) are left out.
ROWS = (('pre-specified', 'pre-specified'),
        ('+ responsiveness, nonlinear', '+ responsiveness (nonlinear)'),
        ('+ MNI coordinates', '+ MNI coordinates'),
        ('+ RT coupling', '+ RT coupling'),
        ('+ base effects, same half', '+ congruency and switch effects'),
        ('all of the above', 'all of the above'))
HIGHLIGHT = '+ base effects, same half'

# Reference-palette slot 1 (dataviz skill) and recessive neutrals, as in the
# coding-axes figures.
ACCENT, INK, MUTED = '#2a78d6', '#0b0b0b', '#52514e'


def plot_overlap_controls(table, out_path):
    """Dot per control at its r, labelled with r and p; the base-effect row in
    the accent colour. `table` is an `overlap_controls.csv`."""
    t = table.set_index('control')
    rows = [(key, label) for key, label in ROWS if key in t.index]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for y, (key, label) in enumerate(rows[::-1]):
        r, p = t.loc[key, 'corr'], t.loc[key, 'p']
        color = ACCENT if key == HIGHLIGHT else INK
        ax.plot([0, r], [y, y], color=color, lw=2, solid_capstyle='round')
        ax.plot(r, y, 'o', ms=8, color=color, mec='white', mew=2)
        ax.annotate(f'r = {r:.3f}, p = {p:.2g}', (max(r, 0), y), xytext=(8, 0),
                    textcoords='offset points', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label in rows[::-1]], fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.02, 0.17)
    ax.set_xlabel('LWPC–LWPS overlap across electrodes (separate-half r)', fontsize=8, color=INK)
    ax.tick_params(axis='x', colors=MUTED, labelcolor=INK, labelsize=8)
    ax.tick_params(axis='y', length=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.splitext(out_path)[0] + ext, dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--anatomy-dir', required=True, help="an anatomy run's continuous/ folder")
    args = p.parse_args()
    table = pd.read_csv(os.path.join(args.anatomy_dir, 'overlap_controls.csv'))
    out = os.path.join(args.anatomy_dir, 'overlap_controls.png')
    plot_overlap_controls(table, out)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
