"""Figure panels with one fixed geometry, for assembling figures in Illustrator.

Every panel made here has the same canvas size and its axes at the same place
on it, in millimetres, whatever it draws and whichever labels it keeps: files
are saved whole (no tight bounding box), so a panel without its axis labels
drops exactly onto the full one, and panels of different analyses line up.

    save_panel_variants(draw, 'out/fig3_lwpc')

calls ``draw(ax)`` once on a fresh panel and writes, per format,

    fig3_lwpc.<fmt>            full
    fig3_lwpc_noxlabel.<fmt>   without the x-axis label
    fig3_lwpc_noylabel.<fmt>   without the y-axis label
    fig3_lwpc_nolabels.<fmt>   without either

Tick labels, legends and keys stay in every variant. PDFs keep text as text
(TrueType, editable in Illustrator).

save_figure(fig, stem) saves any figure as PDF, PNG and 450-dpi TIFF.
"""

import os

import matplotlib.pyplot as plt

# Canvas and axes box, mm. 89 mm is one Nature column. The right margin holds
# the key column (significance-row labels and legend, drawn at x > 1 in axes
# coordinates); the left and bottom margins hold the tick and axis labels.
PANEL_MM = dict(width=89.0, height=60.0, left=13.0, right=21.0, bottom=10.0, top=3.0)

VARIANTS = {'': (True, True), '_noxlabel': (False, True),
            '_noylabel': (True, False), '_nolabels': (False, False)}


def panel_axes(geometry=None):
    """``(fig, ax)``: a figure of ``geometry`` (default `PANEL_MM`) size with
    one axes placed at its fixed margins."""
    g = {**PANEL_MM, **(geometry or {})}
    fig = plt.figure(figsize=(g['width'] / 25.4, g['height'] / 25.4))
    ax = fig.add_axes([g['left'] / g['width'], g['bottom'] / g['height'],
                       (g['width'] - g['left'] - g['right']) / g['width'],
                       (g['height'] - g['bottom'] - g['top']) / g['height']])
    return fig, ax


def overflow_mm(fig, ax):
    """How far the axes' content (ticks, labels, legends, keys) reaches past
    each edge of the canvas, in mm: {'left', 'right', 'bottom', 'top'};
    positive = cut off when the whole canvas is saved."""
    renderer = fig.canvas.get_renderer()
    box = ax.get_tightbbox(renderer)
    fig_box = fig.bbox
    mm = 25.4 / fig.dpi
    return {'left': (fig_box.x0 - box.x0) * mm, 'right': (box.x1 - fig_box.x1) * mm,
            'bottom': (fig_box.y0 - box.y0) * mm, 'top': (box.y1 - fig_box.y1) * mm}


def save_figure(fig, stem, formats=('pdf', 'png', 'tif'), dpi=450, **savefig_kwargs):
    """Save ``fig`` once per format to ``<stem>.<fmt>``; returns the paths.
    ``dpi`` applies to the raster formats; TIFFs are LZW-compressed. Extra
    keywords go to ``savefig`` (e.g. ``bbox_inches='tight'``)."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    paths = []
    for fmt in formats:
        path = f'{stem}.{fmt}'
        fig.savefig(path, dpi=dpi, **savefig_kwargs,
                    **({'pil_kwargs': {'compression': 'tiff_lzw'}} if fmt == 'tif' else {}))
        paths.append(path)
    return paths


def save_panel_variants(draw, stem, geometry=None, formats=('png', 'pdf', 'tif'), dpi=450):
    """Draw one panel with ``draw(ax)`` and save it in every `VARIANTS`
    variant and format to ``<stem><variant>.<fmt>``; returns the paths.

    The canvas is saved whole, so all variants (and all panels drawn with the
    same ``geometry``) share their axes position exactly. ``dpi`` applies to
    the raster formats; TIFFs are LZW-compressed."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    # 'standard' = the whole canvas, whatever a style sheet set ('tight' would
    # crop each variant differently and break the alignment)
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.bbox': 'standard'}):
        fig, ax = panel_axes(geometry)
        draw(ax)
        over = overflow_mm(fig, ax)
        if any(v > 0 for v in over.values()):
            print(f"WARNING {os.path.basename(stem)}: content runs off the canvas by "
                  + ', '.join(f'{k} {v:.1f} mm' for k, v in over.items() if v > 0)
                  + ' (cut off in every file); widen the margin or shorten the text')
        xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
        paths = []
        for tag, (keep_x, keep_y) in VARIANTS.items():
            ax.set_xlabel(xlabel if keep_x else '')
            ax.set_ylabel(ylabel if keep_y else '')
            paths += save_figure(fig, f'{stem}{tag}', formats, dpi, facecolor='white')
        plt.close(fig)
    return paths
