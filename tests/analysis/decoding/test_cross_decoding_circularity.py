"""Tests for the electrode-definition <-> decode double-dipping guard.

The four interaction-defined electrode groups (CPC, SPS, CPS, SPC) each have
exactly one within-block decode cell that would double-dip on them (the cell whose
contrast x block-modulator IS the interaction that defined the group). These pure
predicates name that diagonal cell so the decoding orchestrator can skip it and
keep only the off-diagonal (non-circular) cells.
"""

import pytest

from src.analysis.decoding.cross_decoding import (
    DEFINITION_DECODE_DIAGONAL,
    circular_decode_for_group,
    is_circular_decode,
)


def test_diagonal_covers_all_four_groups():
    assert set(DEFINITION_DECODE_DIAGONAL) == {"CPC", "SPS", "CPS", "SPC"}


@pytest.mark.parametrize("group,contrast,block_col", [
    ("CPC", "congruency", "incongruent_proportion"),   # LWPC
    ("SPS", "switchType", "switch_proportion"),         # LWPS
    ("CPS", "congruency", "switch_proportion"),         # cross
    ("SPC", "switchType", "incongruent_proportion"),    # cross
])
def test_diagonal_cell_is_circular(group, contrast, block_col):
    assert is_circular_decode(group, contrast, block_col)
    assert circular_decode_for_group(group) == (contrast, block_col)


def test_off_diagonal_is_not_circular():
    # CPC electrodes decoded on the flexibility cell -> clean (the workhorse)
    assert not is_circular_decode("CPC", "switchType", "switch_proportion")
    # SPS electrodes decoded on the stability cell -> clean
    assert not is_circular_decode("SPS", "congruency", "incongruent_proportion")
    # a cross group decoded on a construct cell -> clean
    assert not is_circular_decode("CPS", "switchType", "incongruent_proportion")


def test_composite_groups_have_no_diagonal():
    # 'both'/'all' are not a single interaction, so nothing is defined circular
    assert circular_decode_for_group("both") is None
    assert not is_circular_decode("both", "congruency", "incongruent_proportion")


def test_every_group_leaves_three_clean_cells():
    decode_cells = [("congruency", "incongruent_proportion"),
                    ("congruency", "switch_proportion"),
                    ("switchType", "switch_proportion"),
                    ("switchType", "incongruent_proportion")]
    for group in DEFINITION_DECODE_DIAGONAL:
        clean = [c for c in decode_cells if not is_circular_decode(group, *c)]
        assert len(clean) == 3, f"group {group!r} should keep 3 off-diagonal cells"
