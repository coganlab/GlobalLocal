"""Shared Matplotlib style constants for decoding figures."""

NATURE_STYLE = {
    'figure.figsize': (89/25.4, 89/25.4),  # 89mm (single column) converted to inches
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 1,
    'lines.markersize': 3,
    'legend.frameon': False,
    'legend.columnspacing': 0.5,
    'legend.handlelength': 1,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
}


def nature_style(base_fontsize=9.0):
    """NATURE_STYLE with every font size derived from one number.

    The plotting functions used to set 12 pt in the rc dict and then pass
    ``fontsize=7`` to ``set_ylabel`` and ``fontsize=12`` to ``set_xlabel``, so
    the two axis labels of one panel came out at different sizes. Deriving the
    sizes from a single argument keeps a panel internally consistent and makes
    "same figure, poster size" one parameter.
    """
    style = dict(NATURE_STYLE)
    style.update({
        'font.size': base_fontsize,
        'axes.labelsize': base_fontsize,
        'axes.titlesize': base_fontsize,
        'xtick.labelsize': base_fontsize - 1,
        'ytick.labelsize': base_fontsize - 1,
        'legend.fontsize': base_fontsize - 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.6,
        'legend.labelspacing': 0.35,
        'legend.borderpad': 0.2,
        'legend.borderaxespad': 0.3,
    })
    return style
