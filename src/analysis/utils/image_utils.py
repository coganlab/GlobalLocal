"""Helpers for raster images placed inside matplotlib figures."""

import numpy as np


def _ink(img, tol):
    """Boolean (H x W) mask of the non-white pixels: some RGB channel below
    ``tol`` of full scale, and not fully transparent."""
    full = 255 if np.issubdtype(img.dtype, np.integer) else 1.0
    ink = (img[..., :3] < tol * full).any(axis=-1)
    if img.shape[-1] == 4:
        ink &= img[..., 3] > 0
    return ink


def _span(mask_1d, pad, size):
    """(start, stop) of the True entries of ``mask_1d`` widened by ``pad``,
    clipped to [0, size]; None if there are none."""
    idx = np.flatnonzero(mask_1d)
    if not len(idx):
        return None
    return max(idx[0] - pad, 0), min(idx[-1] + pad + 1, size)


def trim_whitespace(img, tol=0.98, pad=4):
    """``img`` (H x W x 3 or 4, as ``plt.imread`` returns it) cropped to the box
    around its non-white pixels, plus ``pad`` pixels on each side.

    A pixel is white when every RGB channel is at least ``tol`` (a fraction of
    full scale) or when it is fully transparent. Rendered brains come with wide
    white margins; trimming them lets the brain fill its panel. An image with
    no non-white pixel is returned unchanged."""
    ink = _ink(img, tol)
    rows, cols = _span(ink.any(axis=1), pad, img.shape[0]), _span(ink.any(axis=0), pad, img.shape[1])
    if rows is None:
        return img
    return img[rows[0]:rows[1], cols[0]:cols[1]]


def tighten_panels(img, n_panels=2, gap=0.04, tol=0.98, pad=4):
    """``img`` made of ``n_panels`` equal-width panels side by side (e.g. a
    split-hemisphere brain), with each panel's side margins trimmed and the
    panels rejoined ``gap`` apart.

    The top and bottom are trimmed over the whole image, not per panel, so
    content rendered at the same height stays aligned. ``gap`` is a fraction of
    the mean trimmed panel width; the gap is white (opaque, for RGBA). Panels
    with no non-white pixel are dropped; an image with none is returned
    unchanged. White is as in `trim_whitespace`."""
    ink = _ink(img, tol)
    rows = _span(ink.any(axis=1), pad, img.shape[0])
    if rows is None:
        return img
    edges = np.linspace(0, img.shape[1], n_panels + 1).round().astype(int)
    panels = []
    for c0, c1 in zip(edges[:-1], edges[1:]):
        cols = _span(ink[:, c0:c1].any(axis=0), pad, c1 - c0)
        if cols is not None:
            panels.append(img[rows[0]:rows[1], c0 + cols[0]:c0 + cols[1]])
    width = int(round(gap * np.mean([p.shape[1] for p in panels])))
    white = np.full((rows[1] - rows[0], width, img.shape[2]),
                    255 if np.issubdtype(img.dtype, np.integer) else 1.0, dtype=img.dtype)
    joined = [panels[0]]
    for p in panels[1:]:
        joined += [white, p]
    return np.concatenate(joined, axis=1)
