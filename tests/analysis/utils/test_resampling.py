import numpy as np
import pytest

from src.analysis.utils.resampling import bootstrap_counts

GROUPS = np.array(['b', 'a', 'a', 'b', 'b', 'c'])      # 2 a, 3 b, 1 c, items unsorted


def test_electrode_unit_keeps_each_group_size():
    counts = bootstrap_counts(GROUPS, 500, np.random.default_rng(0), unit='electrode')
    assert counts.shape == (500, len(GROUPS))
    for name, size in (('a', 2), ('b', 3), ('c', 1)):
        assert np.all(counts[:, GROUPS == name].sum(1) == size)
    # items are resampled, not just kept: some draws repeat an electrode
    assert (counts[:, GROUPS == 'b'] > 1).any()


def test_participant_unit_moves_a_groups_items_together():
    counts = bootstrap_counts(GROUPS, 500, np.random.default_rng(0), unit='participant')
    for name in ('a', 'b'):
        block = counts[:, GROUPS == name]
        assert np.all(block == block[:, :1])               # every member counted alike
    # three groups drawn per resample
    per_group = np.stack([counts[:, GROUPS == n][:, 0] for n in ('a', 'b', 'c')], 1)
    assert np.all(per_group.sum(1) == 3)


def test_unknown_unit_is_refused():
    with pytest.raises(ValueError):
        bootstrap_counts(GROUPS, 10, np.random.default_rng(0), unit='trial')
