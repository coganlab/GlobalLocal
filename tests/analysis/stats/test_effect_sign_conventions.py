"""The LWPC / LWPS sign convention, pinned in one place.

Both interactions are scored LOW-proportion MINUS HIGH-proportion::

    LWPC = (i - c | 25% incongruent) - (i - c | 75% incongruent)
    LWPS = (s - r | 25% switch)      - (s - r | 75% switch)

so a POSITIVE score means the condition effect SHRINKS in the high-proportion
block -- the direction behavior shows. These tests plant an effect with a KNOWN
direction and assert the sign that comes back, on both sides of the brain-behavior
comparison:

  * the neural scores (`stability_flexibility_segregation`), and
  * the behavioral d-o-d (`stability_flexibility_brain_behavior`) plus the
    trial-level adjustment weights the mixed model uses.

The point is not that the neural effect must run this way -- no test here assumes
it does, and the analyses are all two-sided. The point is that "+" means the SAME
thing everywhere, so a correlation between a neural score and a behavioral one
cannot silently be the negative of what its name says.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats.stability_flexibility_brain_behavior import (
    behavioral_lwpc_lwps_magnitudes)
from dcc_scripts.stats.stability_flexibility_brain_behavior_dcc import (
    _adjustment_weight)


def _planted_df(shrink_lwpc=True, shrink_lwps=True, n_trials=1600, seed=0):
    """Single-trial HG where the condition effects shrink (or grow) in the 75% block.

    The congruency effect is +1.0 in the 25%-incongruent block and, when
    ``shrink_lwpc``, +0.2 in the 75% block -- i.e. an adaptation of +0.8 on the
    LOW-minus-HIGH convention. Same construction for switch x switch-proportion.
    """
    rng = np.random.default_rng(seed)
    cong = rng.choice(['c', 'i'], n_trials)
    sw = rng.choice(['r', 's'], n_trials)
    inc = rng.choice([25.0, 75.0], n_trials)
    swp = rng.choice([25.0, 75.0], n_trials)

    cong_eff = np.where(inc == 25.0, 1.0, 0.2 if shrink_lwpc else 1.8)
    sw_eff = np.where(swp == 25.0, 1.0, 0.2 if shrink_lwps else 1.8)
    hg = (cong_eff * (cong == 'i') + sw_eff * (sw == 's')
          + rng.normal(0, 0.5, n_trials))

    return pd.DataFrame(dict(subject='S00', electrode='S00-e0', hg=hg,
                             congruency=cong, switchType=sw,
                             incongruent_proportion=inc, switch_proportion=swp))


@pytest.mark.parametrize('shrink,expected_positive', [(True, True), (False, False)])
def test_neural_scores_are_low_minus_high(shrink, expected_positive):
    """A condition effect that SHRINKS in the 75% block scores POSITIVE."""
    df = _planted_df(shrink_lwpc=shrink, shrink_lwps=shrink)

    elec = sfs.naive_sensitivities(df, contrast_mode='proportion')

    lwpc, lwps = float(elec['x'].iloc[0]), float(elec['y'].iloc[0])
    assert (lwpc > 0) is expected_positive, f"LWPC sign wrong: {lwpc:+.3f}"
    assert (lwps > 0) is expected_positive, f"LWPS sign wrong: {lwps:+.3f}"


def test_neural_score_matches_the_hand_computed_difference_of_differences():
    """The score is the d-o-d of the four CELL means, in the stated order."""
    df = _planted_df()
    m = {(c, p): df[(df.congruency == c) & (df.incongruent_proportion == p)]['hg'].mean()
         for c in ('i', 'c') for p in (25.0, 75.0)}
    expected_numerator = ((m[('i', 25.0)] - m[('c', 25.0)])
                          - (m[('i', 75.0)] - m[('c', 75.0)]))

    lwpc = float(sfs.naive_sensitivities(df, contrast_mode='proportion')['x'].iloc[0])

    # the score divides that numerator by the pooled within-cell SD, so it keeps
    # the sign and stays d-like; check both facts
    assert np.sign(lwpc) == np.sign(expected_numerator)
    assert 0 < abs(lwpc) < abs(expected_numerator) * 10


def test_behavioral_dod_uses_the_same_orientation():
    """RT: a congruency effect that shrinks in the 75% block is POSITIVE too."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(4):
        for _ in range(600):
            cong = rng.choice(['i', 'c'])
            inc = rng.choice([25.0, 75.0])
            sw = rng.choice(['s', 'r'])
            swp = rng.choice([25.0, 75.0])
            # congruency cost 80 ms in the 25% block, 20 ms in the 75% block
            cost = 80.0 if inc == 25.0 else 20.0
            rows.append(dict(subject=s, RT=500.0 + cost * (cong == 'i')
                             + rng.normal(0, 5), acc=1, congruency=cong,
                             switchType=sw, incongruent_proportion=inc,
                             switch_proportion=swp))

    mags = behavioral_lwpc_lwps_magnitudes(pd.DataFrame(rows))

    assert np.nanmean(mags['lwpc']) == pytest.approx(60.0, abs=8.0)


def test_trial_level_adjustment_weights_match_the_subject_level_dod():
    """`adj_congruency`'s weights put +1 on (i, LOW) -- the same diagonal as `_dod_rt`.

    The mixed model's trial-level adjustment and the per-subject d-o-d have to
    agree, or the within-subject and across-subject arms of A6 report effects of
    opposite sign from the same data.
    """
    cond = np.array(['i', 'i', 'c', 'c'])
    mod = np.array([25.0, 75.0, 25.0, 75.0])

    w = _adjustment_weight(cond, mod, 'i', 'c')

    assert list(w) == [1.0, -1.0, -1.0, 1.0]        # (i,low) and (c,high) are +1


def test_both_sides_agree_on_a_shared_adaptation_effect():
    """The end-to-end guarantee: plant adaptation in HG and in RT, get + on both."""
    neural = float(sfs.naive_sensitivities(_planted_df(), contrast_mode='proportion')
                   ['x'].iloc[0])

    rng = np.random.default_rng(2)
    rows = []
    for _ in range(2000):
        cong = rng.choice(['i', 'c'])
        inc = rng.choice([25.0, 75.0])
        cost = 80.0 if inc == 25.0 else 20.0
        rows.append(dict(subject='S00', RT=500.0 + cost * (cong == 'i')
                         + rng.normal(0, 5), acc=1, congruency=cong,
                         switchType=rng.choice(['s', 'r']),
                         incongruent_proportion=inc,
                         switch_proportion=rng.choice([25.0, 75.0])))
    behavioral = float(behavioral_lwpc_lwps_magnitudes(pd.DataFrame(rows))
                       ['lwpc'].iloc[0])

    assert neural > 0 and behavioral > 0
