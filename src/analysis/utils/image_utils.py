"""Helpers for raster images placed inside matplotlib figures."""

import numpy as np


def trim_whitespace(img, tol=0.98, pad=4):
    """``img`` (H x W x 3 or 4, as ``plt.imread`` returns it) cropped to the box
    around its non-white pixels, plus ``pad`` pixels on each side.

    A pixel is white when every RGB channel is at least ``tol`` (a fraction of
    full scale) or when it is fully transparent. Rendered brains come with wide
    white margins; trimming them lets the brain fill its panel. An image with
    no non-white pixel is returned unchanged."""
    full = 255 if np.issubdtype(img.dtype, np.integer) else 1.0
    ink = (img[..., :3] < tol * full).any(axis=-1)
    if img.shape[-1] == 4:
        ink &= img[..., 3] > 0
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if not len(rows):
        return img
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]
