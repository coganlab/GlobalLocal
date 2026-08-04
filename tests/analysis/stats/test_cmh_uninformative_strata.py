"""Uninformative subject strata must not contribute to the CMH conjunction.

`StratifiedTable(..., shift_zeros=True)` adds 0.5 to all four cells of any
stratum containing a zero. On an otherwise informative but sparse table that is
the standard continuity fix; on a stratum with a zero MARGINAL it invents an
association out of a subject that carries no information about one. These tests
pin the boundary between the two.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_segregation as sfs


def make_labels(strata):
    """`strata`: list of (subject, n_both, n_S_only, n_F_only, n_neither)."""
    recs = []
    for subj, a, b, c, e in strata:
        for i in range(a):
            recs.append(dict(subject=subj, electrode=f'{subj}_a{i}', S=1, F=1))
        for i in range(b):
            recs.append(dict(subject=subj, electrode=f'{subj}_b{i}', S=1, F=0))
        for i in range(c):
            recs.append(dict(subject=subj, electrode=f'{subj}_c{i}', S=0, F=1))
        for i in range(e):
            recs.append(dict(subject=subj, electrode=f'{subj}_e{i}', S=0, F=0))
    return pd.DataFrame(recs)


INFORMATIVE = [(f'R{i}', 3, 5, 6, 40) for i in range(4)]


# ----------------------------------------------------------------------------
# the invariant: a subject with a zero marginal changes nothing
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('uninformative', [
    [(f'E{i}', 0, 0, 9, 45) for i in range(4)],     # no S electrodes at all
    [(f'E{i}', 0, 9, 0, 45) for i in range(4)],     # no F electrodes at all
    [(f'E{i}', 12, 0, 0, 0) for i in range(2)],     # every electrode is both
    [(f'E{i}', 0, 0, 0, 30) for i in range(3)],     # nothing selected either way
])
def test_uninformative_strata_do_not_move_the_estimate(uninformative):
    base = sfs.cmh_conjunction(make_labels(INFORMATIVE))
    padded = sfs.cmh_conjunction(make_labels(INFORMATIVE + uninformative))

    assert padded['mh_odds_ratio'] == pytest.approx(base['mh_odds_ratio'])
    assert padded['cmh'].pvalue == pytest.approx(base['cmh'].pvalue)
    assert padded['n_informative_strata'] == 4
    assert padded['n_dropped_strata'] == len(uninformative)


def test_the_old_behaviour_did_move_it():
    """Guards the fix itself: without the drop, the same padding shifts both the
    odds ratio and the p-value, so this is a real correction and not a no-op."""
    labels = make_labels(INFORMATIVE + [(f'E{i}', 0, 0, 9, 45) for i in range(4)])
    fixed = sfs.cmh_conjunction(labels)
    old = sfs.cmh_conjunction(labels, drop_uninformative_strata=False)

    assert old['mh_odds_ratio'] > fixed['mh_odds_ratio']
    assert old['cmh'].pvalue < fixed['cmh'].pvalue


# ----------------------------------------------------------------------------
# the degenerate limit: nothing selected anywhere
# ----------------------------------------------------------------------------
def test_nothing_selected_gives_an_undefined_odds_ratio():
    res = sfs.cmh_conjunction(make_labels([(f'Z{i}', 0, 0, 0, 25) for i in range(8)]))
    assert np.isnan(res['mh_odds_ratio'])
    assert np.isnan(res['cmh'].pvalue)
    assert np.isnan(res['homogeneity'].pvalue)
    assert res['n_informative_strata'] == 0


def test_nothing_selected_used_to_fabricate_a_huge_significant_odds_ratio():
    """The exact artifact this fix exists for, held in place so it cannot return:
    8 subjects x 25 unselected electrodes reported OR = 51 at p = 4e-12."""
    labels = make_labels([(f'Z{i}', 0, 0, 0, 25) for i in range(8)])
    old = sfs.cmh_conjunction(labels, drop_uninformative_strata=False)
    assert old['mh_odds_ratio'] > 10
    assert old['cmh'].pvalue < 1e-6


# ----------------------------------------------------------------------------
# what must NOT be dropped
# ----------------------------------------------------------------------------
def test_a_single_empty_cell_is_still_informative_and_still_shifted():
    """`[[2, 0], [1, 30]]` has no zero marginal -- it is sparse, not empty, and
    shift_zeros is the right tool for it. Dropping it would throw away data."""
    res = sfs.cmh_conjunction(make_labels([('A', 2, 0, 1, 30), ('B', 3, 1, 2, 25)]))
    assert res['n_informative_strata'] == 2
    assert res['n_dropped_strata'] == 0
    assert np.isfinite(res['mh_odds_ratio'])


def test_pooled_table_still_counts_every_electrode():
    """The pooled 2x2 is the descriptive readout a reader checks against the
    reported cell counts, so it must stay over ALL strata even though the
    stratified inference drops some."""
    strata = INFORMATIVE + [(f'E{i}', 0, 0, 9, 45) for i in range(4)]
    res = sfs.cmh_conjunction(make_labels(strata))
    total = sum(a + b + c + e for _, a, b, c, e in strata)
    assert np.asarray(res['pooled_table']).sum() == total


def test_per_subject_marks_which_strata_were_used():
    strata = INFORMATIVE + [(f'E{i}', 0, 0, 9, 45) for i in range(4)]
    res = sfs.cmh_conjunction(make_labels(strata))
    ps = res['per_subject']
    assert 'informative' in ps.columns
    assert ps.loc[ps.subject.str.startswith('R'), 'informative'].all()
    assert not ps.loc[ps.subject.str.startswith('E'), 'informative'].any()


# ----------------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------------
def test_sweep_reports_strata_count_and_nan_where_selection_runs_out():
    labels = make_labels(INFORMATIVE)
    labels['q'] = np.where(labels['S'] + labels['F'] > 0, 0.04, 0.9)

    def at(t):
        lab = labels.copy()
        lab['S'] = ((lab['q'] < t) & (labels['S'] == 1)).astype(int)
        lab['F'] = ((lab['q'] < t) & (labels['F'] == 1)).astype(int)
        return lab

    sweep = sfs.conjunction_threshold_sweep(at, [0.01, 0.05])
    assert 'n_informative_strata' in sweep.columns

    strict = sweep[sweep.threshold == 0.01].iloc[0]     # nothing clears 0.01
    assert strict['n_informative_strata'] == 0
    assert np.isnan(strict['mh_odds_ratio'])

    loose = sweep[sweep.threshold == 0.05].iloc[0]
    assert loose['n_informative_strata'] == 4
    assert np.isfinite(loose['mh_odds_ratio'])
