"""Tests for A6 — stability/flexibility brain–behavior correlation (plan §6).

Covers the blockType -> proportion map (pinned against the task code and the
behavioral data), the subject-level behavior table (its sign convention pinned
against the file), the behavioral difference-of-differences extraction, the
across-subject correlation with its cross-pairing specificity control, and the
within-subject single-trial mixed model (matched slope must beat the cross slope).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

from src.analysis.stats.stability_flexibility_brain_behavior import (
    behavioral_lwpc_lwps_magnitudes, neural_summary_by_subject,
    subject_level_brain_behavior, trialwise_brain_behavior,
    load_subject_level_behavior, _synthetic_brain_behavior, _BLOCK_PROPORTION_MAP,
    SUBJECT_LEVEL_BEHAVIOR_CSV,
)

TASK_CODE = os.path.join(ROOT, 'src', 'task', 'mainTask.m')
BEHAVIOR_CSV = os.path.join(ROOT, 'combinedData.csv')


# ---------------------------------------------------------------------------
# blockType -> proportion map
# ---------------------------------------------------------------------------
def _task_code_branch(fn_name, var):
    """(letters in the `if` branch, value there, value in the `else`) for the
    `if (blockLetter == 'X' || ...) var = a; else var = b;` in a mainTask.m function."""
    with open(TASK_CODE) as f:
        src = f.read()
    m = re.search(rf"{fn_name}\([^)]*\).*?if\s*\((?P<cond>[^\n]*)\)\s*"
                  rf"{var}\s*=\s*(?P<val>[0-9.]+);\s*else\s*{var}\s*=\s*(?P<other>[0-9.]+);",
                  src, flags=re.DOTALL)
    assert m, f"could not find the {var} branch of {fn_name} in mainTask.m"
    letters = set(re.findall(r"blockLetter\s*==\s*'([A-D])'", m.group('cond')))
    return letters, float(m.group('val')), float(m.group('other'))


@pytest.mark.skipif(not os.path.exists(TASK_CODE), reason="task code not present")
def test_block_map_matches_the_task_code():
    """The map is what `createCongruencyArr` / `createTaskArr` actually build."""
    inc_letters, inc_in, inc_out = _task_code_branch('createCongruencyArr', 'percInc')
    sw_letters, sw_in, sw_out = _task_code_branch('createTaskArr', 'percSwitch')
    for blk in 'ABCD':
        expected_inc = 100 * (inc_in if blk in inc_letters else inc_out)
        expected_sw = 100 * (sw_in if blk in sw_letters else sw_out)
        assert _BLOCK_PROPORTION_MAP[blk]['incongruent_proportion'] == expected_inc, blk
        assert _BLOCK_PROPORTION_MAP[blk]['switch_proportion'] == expected_sw, blk


@pytest.mark.skipif(not os.path.exists(BEHAVIOR_CSV), reason="combinedData.csv not present")
def test_block_map_matches_the_behavioral_data():
    """Each block's observed incongruent / switch fraction matches its label."""
    d = pd.read_csv(BEHAVIOR_CSV, usecols=['logType', 'blockType', 'congruency',
                                          'switchType'])
    d = d[d['logType'] == 'task']
    for blk, g in d.groupby('blockType'):
        frac_inc = (g['congruency'] == 'i').mean()
        sr = g[g['switchType'].isin(['s', 'r'])]
        frac_sw = (sr['switchType'] == 's').mean()
        assert frac_inc == pytest.approx(
            _BLOCK_PROPORTION_MAP[blk]['incongruent_proportion'] / 100, abs=0.03), blk
        assert frac_sw == pytest.approx(
            _BLOCK_PROPORTION_MAP[blk]['switch_proportion'] / 100, abs=0.03), blk


def test_block_map_fully_crosses_the_two_proportions():
    """Every (incongruent, switch) proportion pair occurs in exactly one block, so
    LWPC and LWPS contrast different block pairs (they are not collinear)."""
    pairs = [(v['incongruent_proportion'], v['switch_proportion'])
             for v in _BLOCK_PROPORTION_MAP.values()]
    assert sorted(pairs) == [(25.0, 25.0), (25.0, 75.0), (75.0, 25.0), (75.0, 75.0)]


# ---------------------------------------------------------------------------
# the subject-level behavior table (what the job correlates against)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(SUBJECT_LEVEL_BEHAVIOR_CSV),
                    reason="subject-level behavior table not present")
def test_subject_level_table_is_low_minus_high():
    """LWPC_effect / LWPS_effect are LOW minus HIGH proportion for every measure:
    the orientation of `_dod_rt` and the neural scores. Were either flipped, every
    brain-behavior correlation would change sign."""
    raw = pd.read_csv(SUBJECT_LEVEL_BEHAVIOR_CSV)
    np.testing.assert_allclose(
        raw['LWPC_effect'],
        raw['congruency_effect_25_inc'] - raw['congruency_effect_75_inc'], atol=1e-9)
    np.testing.assert_allclose(
        raw['LWPS_effect'],
        raw['switch_cost_25_switch'] - raw['switch_cost_75_switch'], atol=1e-9)


@pytest.mark.skipif(not os.path.exists(SUBJECT_LEVEL_BEHAVIOR_CSV),
                    reason="subject-level behavior table not present")
def test_load_subject_level_behavior():
    b = load_subject_level_behavior()
    assert list(b.columns[:3]) == ['subject', 'lwpc', 'lwps']
    assert b['subject'].is_unique and len(b) == 25
    # RT: both condition effects shrink in the high-proportion block on average
    assert b['lwpc'].mean() > 0 and b['lwps'].mean() > 0
    err = load_subject_level_behavior(measure='error_mean').set_index('subject')
    acc = load_subject_level_behavior(measure='acc_mean').set_index('subject')
    np.testing.assert_allclose(acc['lwpc'], -err['lwpc'])
    with pytest.raises(ValueError, match='available'):
        load_subject_level_behavior(measure='nonsense')


# ---------------------------------------------------------------------------
# behavioral magnitude extraction from raw trials
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
    # scored LOW minus HIGH, so an effect that GROWS in the 75% block is negative
    # (the real behavioral direction, a shrinking effect, is positive)
    assert np.nanmean(mags['lwpc']) == pytest.approx(-60.0, abs=8.0)
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


def test_blocktype_lwpc_is_not_the_cross_effect():
    """With only blockType to go on, `lwpc` is congruency x INCONGRUENT proportion.

    Trials are drawn with each block's real proportions. RT carries a planted LWPC
    (the congruency effect is 60 ms smaller in 75%-incongruent blocks) and a larger
    planted CROSS effect (congruency x switch proportion, +80 ms). The old map,
    which put A and D in the wrong incongruent-proportion level, returned a mix of
    the cross effect and block-level RT differences instead of the +60."""
    rng = np.random.default_rng(3)
    block_offset = dict(A=0.0, B=40.0, C=-30.0, D=90.0)     # block-level RT shifts
    rows = []
    for s in range(6):
        for blk, props in _BLOCK_PROPORTION_MAP.items():
            inc75 = props['incongruent_proportion'] == 75.0
            sw75 = props['switch_proportion'] == 75.0
            for _ in range(400):
                cong = 'i' if rng.random() < props['incongruent_proportion'] / 100 else 'c'
                sw = 's' if rng.random() < props['switch_proportion'] / 100 else 'r'
                rt = (800.0 + block_offset[blk]
                      + 100.0 * (cong == 'i')
                      - 60.0 * (cong == 'i') * inc75          # LWPC (+60 low - high)
                      + 80.0 * (cong == 'i') * sw75           # cross effect
                      + rng.normal(0, 20))
                rows.append(dict(subject=s, RT=rt, acc=1, congruency=cong,
                                 switchType=sw, blockType=blk))
    mags = behavioral_lwpc_lwps_magnitudes(pd.DataFrame(rows))
    assert np.mean(mags['lwpc']) == pytest.approx(60.0, abs=10.0)
    assert abs(np.mean(mags['lwps'])) < 15.0              # no planted LWPS


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
