import numpy as np
from src.analysis.utils.image_utils import trim_whitespace


def test_trim_whitespace_keeps_the_ink_and_its_pad():
    img = np.ones((50, 80, 4))
    img[10:20, 30:40, :3] = 0.2
    out = trim_whitespace(img, pad=2)
    assert out.shape[:2] == (14, 14)
    assert trim_whitespace(np.ones((5, 5, 3))).shape == (5, 5, 3)
