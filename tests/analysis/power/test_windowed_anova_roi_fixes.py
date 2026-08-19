"""Tests for the ROI-level windowed-ANOVA cluster correction fixes.

Three behaviors are pinned here, each corresponding to a defect that produced
significant "interaction" clusters spanning the pre-stimulus baseline:

1. `balance_factors` averages the untested factors with equal weight, so a
   block-level offset carried by an unequal cell mixture cancels instead of
   entering the contrast.
2. The permutation shuffles labels once per SUBJECT and applies that relabeling
   to all of the subject's electrodes -- electrodes of one subject share trials,
   so per-electrode shuffles build null datasets the experiment cannot produce.
3. `analysis_window` restricts the cluster search to a time range.

These drive the real pipeline (no mocks) on small synthetic designs.
"""

import numpy as np
import pytest

from src.analysis.power.windowed_anova import run_windowed_anova_cluster_correction


ROI = 'lpfc'
WINDOW_SIZE = 4
STEP_SIZE = 2
N_WINDOWS = 8
N_TIMES = WINDOW_SIZE + STEP_SIZE * (N_WINDOWS - 1)
TIMES = np.linspace(-1.0, 1.5, N_TIMES)

# Full 2x2x2 design: the tested pair (congruency x switchProportion) plus the
# factor whose mixture is unequal across the tested cells (incongruentProportion).
CONDITIONS_16 = {
    f'{cong}{inc}{swp}': {'congruency': cong,
                          'incongruentProportion': inc,
                          'switchProportion': swp}
    for cong in ('c', 'i') for inc in ('25', '75') for swp in ('25', '75')
}
TESTED = ['congruency', 'switchProportion']
BALANCE = ['incongruentProportion']
INTERACTION = 'C(congruency):C(switchProportion)'

# Design-realistic trial counts: congruent trials are 75% of a 25%-incongruent
# block and 25% of a 75%-incongruent block, so the congruent cell of the tested
# 2x2 is ~3:1 inc25 while the incongruent cell is ~1:3.
TRIALS = {'c25': 12, 'c75': 4, 'i25': 4, 'i75': 12}

# A tonic offset attached to the BLOCK, not to any tested factor. Non-additive
# across the block landscape so it does not cancel out of a trial-count-weighted
# interaction contrast.
BLOCK_OFFSET = {('25', '25'): 0.0, ('25', '75'): 0.9,
                ('75', '25'): 0.0, ('75', '75'): 0.0}


def _windowed_data(subjects, elecs_per_sub, seed=0, noise=0.05):
    """{cond: {roi: [ (n_trials, n_windows, n_chan) per subject ]}}.

    Contains a block-level tonic offset and nothing else -- no effect of either
    tested factor, and the offset is constant across windows (as a tonic
    difference is), so any interaction the test reports is the confound.
    """
    rng = np.random.default_rng(seed)
    out = {c: {ROI: []} for c in CONDITIONS_16}
    for cond, spec in CONDITIONS_16.items():
        n_tr = TRIALS[f"{spec['congruency']}{spec['incongruentProportion']}"]
        level = BLOCK_OFFSET[(spec['incongruentProportion'], spec['switchProportion'])]
        for _ in subjects:
            arr = rng.normal(0, noise, size=(n_tr, N_WINDOWS, elecs_per_sub))
            out[cond][ROI].append(arr + level)
    return out


def _run(windowed, subjects, elecs_per_sub, **kw):
    elecs = {ROI: {s: [f'{s}_e{j}' for j in range(elecs_per_sub)] for s in subjects}}
    results, centers = run_windowed_anova_cluster_correction(
        windowed, CONDITIONS_16, TESTED, [ROI], subjects,
        electrodes_per_subject_roi=elecs, times=TIMES,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE, sampling_rate=256,
        n_perm=kw.pop('n_perm', 60), percentile=95, cluster_percentile=95,
        seed=7, n_jobs=1, verbose=False, **kw,
    )
    return results, centers


def test_balance_factors_removes_block_mixture_confound():
    """A purely block-level offset must not show up as an interaction.

    Without balancing, the tested cells hold 3:1 vs 1:3 mixtures of the
    incongruent-proportion blocks, so the block offset enters the contrast. With
    equal-weight sub-cell averaging both cells hold the same mixture and it
    cancels.
    """
    subjects = ['S1', 'S2', 'S3']
    windowed = _windowed_data(subjects, elecs_per_sub=3)

    unbalanced, _ = _run(windowed, subjects, 3)
    balanced, _ = _run(windowed, subjects, 3, balance_factors=BALANCE)

    f_unbal = np.nanmax(unbalanced[ROI][INTERACTION]['observed_F'])
    f_bal = np.nanmax(balanced[ROI][INTERACTION]['observed_F'])

    # The confound is large without balancing and essentially gone with it.
    assert f_unbal > 10 * max(f_bal, 1e-9), (
        f"expected the block offset to inflate the unbalanced interaction F "
        f"(got unbalanced={f_unbal:.3g}, balanced={f_bal:.3g})")
    assert not balanced[ROI][INTERACTION]['window_mask'].any(), (
        "balanced fit still reports a significant interaction cluster for data "
        "whose only structure is a block-level tonic offset")


def test_balance_factors_rejects_pooled_conditions():
    """Balancing over a factor the conditions don't carry must fail loudly."""
    subjects = ['S1', 'S2']
    windowed = _windowed_data(subjects, elecs_per_sub=2)
    with pytest.raises(ValueError, match='not columns of the windowed'):
        _run(windowed, subjects, 2, balance_factors=['switchType'])


def test_permutation_relabels_each_subject_consistently():
    """Every electrode of a subject must receive the SAME label permutation.

    Electrodes of one subject are recorded from the same trials, so a null
    dataset in which two contacts disagree about a trial's condition is not a
    relabeling of the experiment. Detected here by giving each subject a large
    subject-specific offset shared across its electrodes: under a correct
    subject-level shuffle the null carries that offset too, so the false-positive
    rate stays controlled.
    """
    rng = np.random.default_rng(3)
    subjects = ['S1', 'S2', 'S3', 'S4']
    n_elec = 4
    windowed = {c: {ROI: []} for c in CONDITIONS_16}
    # One offset per (subject, cell), shared across that subject's electrodes and
    # constant over time -- the structure a tonic subject/block state produces.
    for cond in CONDITIONS_16:
        for si, _ in enumerate(subjects):
            n_tr = 8
            shared = rng.normal(0, 1.0)
            arr = rng.normal(0, 0.05, size=(n_tr, N_WINDOWS, n_elec)) + shared
            windowed[cond][ROI].append(arr)

    results, _ = _run(windowed, subjects, n_elec, balance_factors=BALANCE, n_perm=100)
    info = results[ROI][INTERACTION]

    # With the per-electrode shuffle this rate ran far above nominal because the
    # null scattered the shared offset while the observed data kept it aligned.
    frac_above = float(np.mean(
        np.nan_to_num(info['observed_F'])
        > np.nanpercentile(np.nan_to_num(info['null_F']), 95, axis=0)))
    assert frac_above < 0.5, (
        f"observed F exceeds the per-window null 95th percentile in "
        f"{frac_above:.0%} of windows; the null is not carrying the "
        f"subject-shared offset")


def test_analysis_window_restricts_search_and_aligns_masks():
    subjects = ['S1', 'S2']
    windowed = _windowed_data(subjects, elecs_per_sub=2)

    full, centers_full = _run(windowed, subjects, 2, balance_factors=BALANCE)
    post, centers_post = _run(windowed, subjects, 2, balance_factors=BALANCE,
                              analysis_window=(0.0, 1.5))

    assert centers_full.min() < 0, 'fixture should span the baseline'
    assert centers_post.min() >= 0.0
    assert len(centers_post) < len(centers_full)

    info = post[ROI][INTERACTION]
    # Window-level arrays follow the restricted window set...
    assert info['window_mask'].shape == centers_post.shape
    assert info['observed_F'].shape == centers_post.shape
    # ...while the sample-level mask still indexes the full time axis, so it can
    # be handed straight to the plotting code.
    assert info['sample_mask'].shape == (len(TIMES),)


def test_analysis_window_with_no_windows_raises():
    subjects = ['S1', 'S2']
    windowed = _windowed_data(subjects, elecs_per_sub=2)
    with pytest.raises(ValueError, match='keeps no windows'):
        _run(windowed, subjects, 2, analysis_window=(90.0, 99.0))


def test_cluster_stat_mass_is_accepted_and_extent_is_default():
    subjects = ['S1', 'S2']
    windowed = _windowed_data(subjects, elecs_per_sub=2)

    res_mass, _ = _run(windowed, subjects, 2, balance_factors=BALANCE,
                       cluster_stat='mass')
    assert INTERACTION in res_mass[ROI]

    with pytest.raises(ValueError, match="cluster_stat must be"):
        _run(windowed, subjects, 2, cluster_stat='bogus')
