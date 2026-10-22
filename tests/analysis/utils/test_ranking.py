"""The top fraction of a score (src/analysis/utils/ranking.py)."""
import numpy as np
import pytest

from src.analysis.utils.ranking import top_fraction


def test_top_fraction_signs_and_rounding():
    v = np.array([3.0, -5.0, 1.0, np.nan, -0.5, 4.0])
    assert list(np.flatnonzero(top_fraction(v, 0.4, 'positive'))) == [0, 5]     # ceil(0.4*5)=2
    assert list(np.flatnonzero(top_fraction(v, 0.4, 'negative'))) == [1, 4]
    assert list(np.flatnonzero(top_fraction(v, 0.4, 'absolute'))) == [1, 5]
    assert not top_fraction(v, 1.0)[3]                                          # NaN never
    with pytest.raises(ValueError, match='sign'):
        top_fraction(v, 0.2, 'both')
