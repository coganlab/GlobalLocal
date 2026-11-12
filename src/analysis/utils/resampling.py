"""Resampling helpers shared by the analyses.

`bootstrap_counts` is the one bootstrap draw used for the paper's intervals:
the anatomy's centroid ellipses (`stability_flexibility_anatomy.height_centroids`),
the coding-axes cosines (`coding_axes`), the subspace test (`coding_subspaces`)
and the trajectories and block separations (`coding_trajectories`). Every
statistic there is a (ratio of) sum(s) over electrodes, so a draw is fully
described by how many times each electrode enters it.
"""
import numpy as np

BOOT_UNITS = ('electrode', 'participant')


def bootstrap_counts(groups, n_boot, rng, unit='electrode'):
    """How many times each item enters each bootstrap draw: ``(n_boot, n_items)``.

    `groups` labels each item's group (an electrode's participant), in item
    order. `rng` is a ``numpy.random.Generator``.

    ``unit='electrode'``: items are drawn with replacement within their group,
    and each group keeps its size (the electrode bootstrap within participant).
    ``unit='participant'``: whole groups are drawn with replacement, and an
    item counts as often as its group was drawn (all its electrodes together).

    Groups are visited in sorted order, so the draws for a given `rng` state do
    not depend on the order of the items."""
    groups = np.asarray(groups).astype(str)
    names = np.unique(groups)
    if unit == 'electrode':
        counts = np.zeros((int(n_boot), len(groups)))
        rows = np.arange(int(n_boot))[:, None]
        for name in names:
            members = np.flatnonzero(groups == name)
            picks = members[rng.integers(0, len(members), size=(int(n_boot), len(members)))]
            np.add.at(counts, (rows, picks), 1.0)
        return counts
    if unit == 'participant':
        draws = rng.integers(0, len(names), size=(int(n_boot), len(names)))
        per_group = (draws[:, :, None] == np.arange(len(names))).sum(1)   # (n_boot, groups)
        return per_group[:, np.searchsorted(names, groups)].astype(float)
    raise ValueError(f"unit must be one of {BOOT_UNITS}, not {unit!r}")
