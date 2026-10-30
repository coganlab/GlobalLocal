import numpy as np
from src.analysis.utils.image_utils import tighten_panels, trim_whitespace


def test_tighten_panels_joins_trimmed_panels_at_one_height():
    """Two 100-px panels with ink at different columns and heights: each is
    trimmed to its own columns, both keep the rows spanning all ink, and they
    are joined 4 % of their mean width apart."""
    img = np.full((60, 200, 3), 255, dtype=np.uint8)
    img[10:20, 70:90] = 0            # left panel, near its right edge
    img[30:50, 110:150] = 0          # right panel, near its left edge, lower
    out = tighten_panels(img, n_panels=2, gap=0.5, pad=0)
    assert out.shape == (40, 20 + 15 + 40, 3)         # rows 10-50; gap = 0.5 * mean(20, 40)
    assert (out[:, 20:35] == 255).all()               # the gap is white
    assert (out[0:10, :20] == 0).all() and (out[20:40, 35:] == 0).all()
    assert tighten_panels(np.full((5, 8, 3), 255, np.uint8)).shape == (5, 8, 3)


def test_trim_whitespace_keeps_the_ink_and_its_pad():
    img = np.ones((50, 80, 4))
    img[10:20, 30:40, :3] = 0.2
    out = trim_whitespace(img, pad=2)
    assert out.shape[:2] == (14, 14)
    assert trim_whitespace(np.ones((5, 5, 3))).shape == (5, 5, 3)
