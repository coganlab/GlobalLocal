"""The block-balanced condition sets must remove the block-mixture confound.

Background. The pooled 4-condition sets define each cell as a raw union of BIDS
events, so a cell mean is weighted by trial count. Congruent trials are 75% of a
25%-incongruent block and 25% of a 75%-incongruent block, which makes the
congruent cell of `congruency x switchProportion` roughly 3:1 inc25-block trials
while the incongruent cell is 1:3. Any tonic difference between block types then
enters the contrast -- at every timepoint, including the pre-stimulus baseline
where no evoked interaction can exist.

The `*_by_block_conditions` sets keep one cell per block type. Run with the
full-factorial `anova_factors` the registry now declares for them, every cell is
its own term, each electrode contributes exactly one row per cell, and the 2-way
terms become equal-weight contrasts in which a block-level offset cancels.

These tests drive the real pipeline on synthetic data whose ONLY structure is a
block-level tonic offset -- no effect of congruency or switch type at all. Any
interaction reported is the confound.
"""

import numpy as np
import pytest

from src.analysis.config import experiment_conditions
from src.analysis.config.condition_registry import (
    get_anova_factors, get_anova_interactions, get_conditions_obj,
)
from src.analysis.power.windowed_anova import run_windowed_anova_cluster_correction


ROI = 'lpfc'
WINDOW_SIZE, STEP_SIZE, N_WINDOWS = 4, 2, 8
N_TIMES = WINDOW_SIZE + STEP_SIZE * (N_WINDOWS - 1)
TIMES = np.linspace(-1.0, 1.5, N_TIMES)
SUBJECTS = ['S1', 'S2', 'S3']
N_ELEC = 3

BLOCK_LABELS = {'MI': '75%', 'MC': '25%', 'MS': '75%', 'MR': '25%'}

# Tonic level per (incongruentProportion, switchProportion) block, deliberately
# NON-additive so it cannot cancel out of a trial-count-weighted contrast.
BLOCK_OFFSET = {('25%', '25%'): 0.0, ('25%', '75%'): 0.9,
                ('75%', '25%'): 0.0, ('75%', '75%'): 0.0}

# Trial counts implied by the proportion manipulation, per block cell.
def _n_trials(level_value, inc_prop):
    """Congruent/repeat trials are the majority in their 'low' block."""
    majority = {'c': '25%', 'i': '75%', 'r': '25%', 's': '75%'}[level_value]
    return 12 if inc_prop == majority else 4


def _windowed(conditions, level_key, seed=0):
    rng = np.random.default_rng(seed)
    out = {c: {ROI: []} for c in conditions}
    for cname, spec in conditions.items():
        inc, swp = spec['incongruentProportion'], spec['switchProportion']
        # switch-type cells are split by switch proportion, congruency cells by
        # incongruent proportion; use whichever governs this factor's frequency.
        governing = swp if level_key == 'switchType' else inc
        n_tr = _n_trials(spec[level_key], governing)
        for _ in SUBJECTS:
            out[cname][ROI].append(
                rng.normal(0, 0.05, size=(n_tr, N_WINDOWS, N_ELEC))
                + BLOCK_OFFSET[(inc, swp)])
    return out


def _run(conditions, factors, windowed):
    elecs = {ROI: {s: [f'{s}_e{j}' for j in range(N_ELEC)] for s in SUBJECTS}}
    results, _ = run_windowed_anova_cluster_correction(
        windowed, conditions, factors, [ROI], SUBJECTS,
        electrodes_per_subject_roi=elecs, times=TIMES,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE, sampling_rate=256,
        n_perm=60, percentile=95, cluster_percentile=95,
        seed=7, n_jobs=1, verbose=False,
    )
    return results[ROI]


@pytest.mark.parametrize('label,level_key,pooled_conditions,cross_effect', [
    ('stimulus_congruency_by_switch_prop_block_balanced_conditions', 'congruency',
     'stimulus_congruency_by_switch_proportion_conditions',
     'C(congruency):C(switchProportion)'),
    ('stimulus_switch_type_by_inc_prop_block_balanced_conditions', 'switchType',
     'stimulus_switch_type_by_incongruent_proportion_conditions',
     'C(switchType):C(incongruentProportion)'),
])
def test_block_balanced_conditions_cancel_the_block_offset(
        label, level_key, pooled_conditions, cross_effect):
    conditions = get_conditions_obj(label)
    factors = get_anova_factors(label)
    windowed = _windowed(conditions, level_key)

    balanced = _run(conditions, factors, windowed)
    assert cross_effect in balanced, f'{cross_effect} missing from {sorted(balanced)}'
    info = balanced[cross_effect]

    assert not info['window_mask'].any(), (
        f"{label}: a block-level tonic offset still produces a significant "
        f"{cross_effect} cluster (peak F = {np.nanmax(info['observed_F']):.2f})")

    # Contrast with the pooled route on the SAME underlying block levels: collapse
    # the 8 cells to the 4 pooled ones by trial count, which is what a raw
    # BIDS-event union does, and confirm the confound reappears.
    other = 'switchProportion' if level_key == 'congruency' else 'incongruentProportion'
    pooled_conds, pooled_windowed = {}, {}
    for cname, spec in conditions.items():
        key = f"{spec[level_key]}_{spec[other]}"
        pooled_conds.setdefault(key, {level_key: spec[level_key], other: spec[other]})
        pooled_windowed.setdefault(key, {ROI: [None] * len(SUBJECTS)})
        for si in range(len(SUBJECTS)):
            arr = windowed[cname][ROI][si]
            cur = pooled_windowed[key][ROI][si]
            pooled_windowed[key][ROI][si] = (
                arr if cur is None else np.concatenate([cur, arr], axis=0))

    pooled = _run(pooled_conds, [level_key, other], pooled_windowed)
    pooled_eff = f'C({level_key}):C({other})'
    assert np.nanmax(pooled[pooled_eff]['observed_F']) > \
        10 * max(np.nanmax(info['observed_F']), 1e-9), (
        'the pooled route should be the one that carries the confound; if this '
        'fails the fixture no longer reproduces the unequal block mixture')


@pytest.mark.parametrize('label', [
    'stimulus_lwpc_block_balanced_conditions',
    'stimulus_lwps_block_balanced_conditions',
    'stimulus_congruency_by_switch_prop_block_balanced_conditions',
    'stimulus_switch_type_by_inc_prop_block_balanced_conditions',
])
def test_registry_entries_are_runnable_for_power_traces(label):
    """Every factor the ANOVA names must exist on every condition."""
    conditions = get_conditions_obj(label)
    factors = get_anova_factors(label)
    assert factors, f'{label} declares no anova_factors'
    for cname, spec in conditions.items():
        missing = [f for f in factors if f not in spec]
        assert not missing, f'{label}/{cname} is missing factor keys {missing}'
    for inter in get_anova_interactions(label):
        assert all(f in factors for f in inter['factors']), (
            f"{label}: interaction {inter['name']} names a factor outside "
            f"anova_factors")


def test_entries_sharing_a_conditions_obj_declare_the_same_model():
    """get_conditions_save_name maps a conditions dict to the FIRST matching
    registry key, so two labels sharing a conditions_obj must not imply two
    different analyses -- otherwise the runs collide in the output directory."""
    pairs = [
        ('stimulus_lwpc_block_balanced_conditions',
         'stimulus_congruency_by_switch_prop_block_balanced_conditions'),
        ('stimulus_lwps_block_balanced_conditions',
         'stimulus_switch_type_by_inc_prop_block_balanced_conditions'),
    ]
    for a, b in pairs:
        assert get_conditions_obj(a) is get_conditions_obj(b)
        assert get_anova_factors(a) == get_anova_factors(b)
        assert ([i['name'] for i in get_anova_interactions(a)]
                == [i['name'] for i in get_anova_interactions(b)])


def test_block_condition_factor_keys_match_their_metadata_query():
    """The added factor keys must agree with each cell's metadata_query."""
    for var, level_key, query_col in [
        ('stimulus_congruency_by_block_conditions', 'congruency', 'congruency'),
        ('stimulus_switch_type_by_block_conditions', 'switchType', 'task_sequence'),
    ]:
        conds = getattr(experiment_conditions, var)
        assert len(conds) == 8, f'{var} should have 8 block cells'
        for cname, spec in conds.items():
            q = spec['metadata_query']
            assert f"{query_col} == '{spec[level_key]}'" in q, (
                f'{var}/{cname}: {level_key}={spec[level_key]!r} contradicts {q!r}')
            inc = spec['incongruentProportion'].rstrip('%')
            swp = spec['switchProportion'].rstrip('%')
            assert f'incongruent_proportion == {inc}' in q, f'{var}/{cname}: {q}'
            assert f'switch_proportion == {swp}' in q, f'{var}/{cname}: {q}'
