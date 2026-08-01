"""Tests for A6 — stability/flexibility brain–behavior correlation (plan §6).

Covers the behavioral difference-of-differences extraction, the across-subject
correlation with its cross-pairing specificity control, and the within-subject
single-trial mixed model (matched slope must beat the cross slope).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..', '..', '..')))

from src.analysis.stats.stability_flexibility_brain_behavior import (
    behavioral_lwpc_lwps_magnitudes, neural_summary_by_subject,
    subject_level_brain_behavior, trialwise_brain_behavior,
    _synthetic_brain_behavior,
)


# ---------------------------------------------------------------------------
# behavioral magnitude extraction
# ---------------------------------------------------------------------------
def test_behavioral_magnitudes_recover_planted_dod():
    """A planted congruency×proportion RT interaction is recovered as `lwpc`."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(4):
        for _ in range(400):
            cong = rng.choice(['i', 'c'])
            inc = rng.choice([25.0, 75.0])
            sw = rng.choice(['s', 'r'])
            swp = rng.choice([25.0, 75.0])
            # LWPC: congruency effect (+40) grows by +60 in the high-inc block
            rt = 500.0
            rt += 40.0 * (cong == 'i')
            rt += 60.0 * (cong == 'i') * (inc == 75.0)
            rt += rng.normal(0, 5)
            rows.append(dict(subject=s, RT=rt, acc=1, congruency=cong,
                             switchType=sw, incongruent_proportion=inc,
                             switch_proportion=swp))
    mags = behavioral_lwpc_lwps_magnitudes(pd.DataFrame(rows))
    assert np.nanmean(mags['lwpc']) == pytest.approx(60.0, abs=8.0)
    assert abs(np.nanmean(mags['lwps'])) < 15.0           # no planted LWPS effect


def test_behavioral_magnitudes_from_blocktype():
    """blockType is mapped to proportions when explicit columns are absent."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(3):
        for blk in ['A', 'B', 'C', 'D']:
            for _ in range(120):
                rows.append(dict(subject=s, RT=500 + rng.normal(0, 5), acc=1,
                                 congruency=rng.choice(['i', 'c']),
                                 switchType=rng.choice(['s', 'r']),
                                 blockType=blk))
    mags = behavioral_lwpc_lwps_magnitudes(pd.DataFrame(rows))
    assert set(mags['subject']) == {0, 1, 2}
    assert mags['lwpc'].notna().all()


# ---------------------------------------------------------------------------
# across-subject correlation + specificity
# ---------------------------------------------------------------------------
def test_across_subject_matched_beats_cross():
    elec_labels, behavior, _ = _synthetic_brain_behavior(seed=2)
    res = subject_level_brain_behavior(elec_labels, behavior, neural='count')
    assert res['corr_lwpc'] > 0.4 and res['p_lwpc'] < 0.05
    assert res['corr_lwps'] > 0.4 and res['p_lwps'] < 0.05
    # matched stronger than the cross-pairing controls (specificity)
    assert res['corr_lwpc'] > res['corr_cross_stab_lwps']
    assert res['corr_lwps'] > res['corr_cross_flex_lwpc']
    assert 'underpowered' in res['caveat']


def test_neural_summary_counts():
    elec_labels, _, _ = _synthetic_brain_behavior(seed=0)
    summ = neural_summary_by_subject(elec_labels)
    assert (summ['n_S'] <= summ['n_elec']).all()
    assert summ['frac_S'].between(0, 1).all()


# ---------------------------------------------------------------------------
# within-subject single-trial mixed model
# ---------------------------------------------------------------------------
def test_within_subject_matched_slope_beats_cross():
    _, _, trial_df = _synthetic_brain_behavior(seed=4)
    for group, hg in (('LWPC', 'hg_lwpc'), ('LWPS', 'hg_lwps')):
        r = trialwise_brain_behavior(trial_df, group=group, hg_col=hg)
        assert r['p'] < 0.05
        assert abs(r['slope']) > abs(r['slope_cross'])
        assert r['specificity_ok'] is True


def test_trialwise_rejects_bad_group():
    _, _, trial_df = _synthetic_brain_behavior(seed=0)
    with pytest.raises(ValueError):
        trialwise_brain_behavior(trial_df, group='nonsense', hg_col='hg_lwpc')
