"""Which values are the most extreme: the top fraction of a score.

Shared by the score electrode sets (``decoding/score_electrode_selection.py``)
and Figure 5's top-electrode view (``stats/stability_flexibility_anatomy.py``).
"""

from __future__ import annotations

import math

import numpy as np

SIGNS = ('positive', 'negative', 'absolute')


def top_fraction(values, frac=0.2, sign='positive'):
    """Boolean mask of the top ``frac`` of ``values``: ceil(frac x n) of the n
    finite values, ranked largest first ('positive'), most negative first
    ('negative') or by magnitude ('absolute'). Ties at the cut go to the
    earlier value. Non-finite values are never selected."""
    if sign not in SIGNS:
        raise ValueError(f"sign must be one of {SIGNS}, got {sign!r}")
    v = np.asarray(values, float)
    key = {'positive': v, 'negative': -v, 'absolute': np.abs(v)}[sign]
    ok = np.flatnonzero(np.isfinite(key))
    keep = np.zeros(len(v), bool)
    keep[ok[np.argsort(-key[ok], kind='stable')[:math.ceil(frac * len(ok))]]] = True
    return keep
