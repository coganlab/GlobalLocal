"""Contrast + block definitions for the A4 cross-decoding job.

The Decoder identifies classes by SUBSTRINGS of the condition name, and the two
naming conventions in this project collide on the obvious shorthand:

    real       Stimulus_{c|i}{25|75}{s|r}{25|75}            Stimulus_i75s25
    synthetic  Stimulus_{c|i}_{r|s}_{25|75}inc_{25|75}sw    Stimulus_i_s_75inc_25sw

'75s' is "switch trial in the 75%-incongruent block" in the first and "75%-switch
block" in the second. A hand-written token set that is right for one silently
decodes the WRONG contrast on the other — no exception, just an answer to a
different question. So the class groups are derived from each condition's
declared factor levels instead, and these tests pin that down.

The final test runs the real branch of `main()` end to end over real condition
names, with only the epoch loading and the electrode definition stubbed.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.config import experiment_conditions as ec
from src.analysis.decoding import cross_decoding as cd
from dcc_scripts.decoding import stability_flexibility_cross_decoding_dcc as xd


@pytest.fixture(scope="module")
def real_cells():
    return cd.condition_cells(ec.stimulus_experiment_conditions)


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def test_real_conditions_are_the_full_2x2x2x2(real_cells):
    assert len(real_cells) == 16
    for field, levels in (('congruency', {'i', 'c'}), ('switchType', {'s', 'r'}),
                          ('incongruent_proportion', {25, 75}),
                          ('switch_proportion', {25, 75})):
        assert {c[field] for c in real_cells.values()} == levels


def test_camelcase_proportions_are_normalised(real_cells):
    # the config spells these 'incongruentProportion': '25%'
    assert all(isinstance(c['incongruent_proportion'], int)
               for c in real_cells.values())


def test_partial_conditions_are_dropped():
    """A condition missing a factor cannot be placed in the 2x2x2x2."""
    conditions = dict(ec.stimulus_experiment_conditions)
    conditions['Stimulus_partial'] = {'BIDS_events': ['Stimulus'], 'congruency': 'i'}
    assert 'Stimulus_partial' not in cd.condition_cells(conditions)


def test_a_condition_set_without_the_four_factors_raises():
    with pytest.raises(ValueError, match='full 2x2x2x2'):
        cd.condition_cells(ec.stimulus_congruency_conditions)


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------
def test_every_condition_is_labelled_by_both_contrasts(real_cells):
    """A cross-decode drops any condition only one contrast can label."""
    stab, flex = cd.stability_flexibility_strings(real_cells)
    for name in real_cells:
        assert cd._class_of(name, stab) is not None, name
        assert cd._class_of(name, flex) is not None, name


def test_contrast_classes_match_the_declared_levels(real_cells):
    stab, flex = cd.stability_flexibility_strings(real_cells)
    for name, cell in real_cells.items():
        assert cd._class_of(name, stab) == (0 if cell['congruency'] == 'i' else 1)
        assert cd._class_of(name, flex) == (0 if cell['switchType'] == 's' else 1)


def test_the_two_conventions_really_do_collide():
    """Why the classes are derived rather than hand-written: tokens that are
    correct for the real names mislabel half the synthetic ones."""
    syn = cd.synthetic_condition_cells()
    _, syn_flex = cd.stability_flexibility_strings(syn)
    real_tokens = [['25s', '75s'], ['25r', '75r']]     # right for Stimulus_i25s75
    wrong = [n for n in syn
             if cd._class_of(n, real_tokens) != cd._class_of(n, syn_flex)]
    assert len(wrong) == len(syn) // 2


def test_synthetic_cells_match_the_generator_condition_names():
    arrays = cd.synthetic_roi_labeled_arrays(seed=0)
    assert set(cd.synthetic_condition_cells()) == set(arrays['synthetic'])


# ---------------------------------------------------------------------------
# block sets
# ---------------------------------------------------------------------------
def test_block_sets_split_the_conditions_in_half(real_cells):
    for field in ('incongruent_proportion', 'switch_proportion'):
        sets = cd.block_condition_sets(real_cells, field)
        assert list(sets) == [25, 75]            # low level first
        assert all(len(v) == 8 for v in sets.values())
        assert not set(sets[25]) & set(sets[75])


def test_a_real_block_level_needs_more_than_one_substring(real_cells):
    """The 25%-incongruent block is `i25` OR `c25` — no single substring gets it,
    which is why `filter_conditions` takes a collection."""
    low = cd.block_condition_sets(real_cells, 'incongruent_proportion')[25]
    assert not any(all(tok in n for n in low) for tok in ('i25', 'c25'))

    arrays = {'roi': {n: np.zeros((4, 2, 3)) for n in real_cells}}
    kept = cd.filter_conditions(arrays, 'roi', low)['roi']
    assert set(kept) == set(low)


def test_filter_conditions_still_takes_a_bare_substring(real_cells):
    arrays = {'roi': {n: np.zeros((4, 2, 3)) for n in real_cells}}
    kept = cd.filter_conditions(arrays, 'roi', 'i25')['roi']
    assert set(kept) == {n for n, c in real_cells.items()
                         if c['congruency'] == 'i' and c['incongruent_proportion'] == 25}


def test_filter_conditions_raises_rather_than_returning_empty(real_cells):
    arrays = {'roi': {n: np.zeros((4, 2, 3)) for n in real_cells}}
    with pytest.raises(ValueError, match='no condition'):
        cd.filter_conditions(arrays, 'roi', ['nope'])


# ---------------------------------------------------------------------------
# the real branch, end to end
# ---------------------------------------------------------------------------
def _fake_roi_arrays(cells, n_channels=8, n_trials=24, n_time=16, seed=0):
    """A pseudopopulation keyed by the REAL condition names, with a shared code
    (congruency and switchType on the same axis) so the decodes have signal."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=n_channels)
    axis /= np.linalg.norm(axis)
    arrays = {}
    for name, cell in cells.items():
        x = rng.normal(0, 1.0, (n_trials, n_channels, n_time))
        gain = 1.2 * ((cell['congruency'] == 'i') + (cell['switchType'] == 's'))
        x += (gain * axis)[None, :, None]
        arrays[name] = x
    return arrays


def test_real_branch_runs_over_real_condition_names(tmp_path, monkeypatch):
    cells = cd.condition_cells(ec.stimulus_experiment_conditions)
    channel_names = [f'D{i // 4}-E{i % 4}' for i in range(8)]
    arrays = {'lpfc': _fake_roi_arrays(cells)}

    labels = pd.DataFrame(dict(
        subject=[c.split('-')[0] for c in channel_names],
        electrode=channel_names,
        S=[1, 1, 1, 0, 1, 1, 0, 0], F=[1, 1, 1, 1, 0, 0, 1, 0],
        CPC=[1, 1, 1, 0, 1, 1, 0, 0], SPS=[1, 1, 1, 1, 0, 0, 1, 0],
        CPS=[0, 0, 0, 0, 0, 0, 0, 0], SPC=[0, 0, 0, 0, 0, 0, 0, 0]))

    # stub out only the two things that need the cluster's data
    monkeypatch.setattr(xd, '_resolve_labels', lambda args, df=None: labels)
    monkeypatch.setattr(xd, '_build_roi_arrays',
                        lambda args, lab_root: ('lpfc', arrays, channel_names, cells))
    monkeypatch.setattr('src.analysis.utils.general_utils.resolve_lab_root',
                        lambda explicit=None: '/nonexistent')

    args = SimpleNamespace(
        data_source='real', synthetic_code=None, LAB_root=None, subjects=[],
        task='GlobalLocal', acc_trials_only=True, epochs_root_file='epochs',
        window_tmin=0.0, window_tmax=0.5, conditions=ec.stimulus_experiment_conditions,
        electrodes='sig', rois_dict={'lpfc': []}, alpha=0.05, roi='lpfc',
        electrode_definition='power_traces', power_traces_runs='run',
        power_traces_correction='fdr_bh', power_traces_roi=None,
        reference_group='all', tempgen_groups=(),
        window_size=8, step_size=8, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=10, min_group_size=2,
        seed=0, save_dir=str(tmp_path))
    results = xd.main(args)

    # both contrasts decoded within both levels of their own block factor
    wb = results['within_block']
    assert set(wb) == {'congruency (LWPC)', 'switchType (LWPS)'}
    assert set(wb['congruency (LWPC)']['per_block']) == {'25% incongruent',
                                                         '75% incongruent'}
    assert set(wb['switchType (LWPS)']['per_block']) == {'25% switch', '75% switch'}
    assert wb['congruency (LWPC)']['block_difference'] is not None

    # the reference group is decoded alongside the selected ones
    assert 'all' in results['label_transfer']
    assert results['label_transfer']['all']['stab_to_flex']['n_channels'] == 8

    # the double-dip diagonal is still skipped per interaction group
    for gflag, res in results['within_block_by_group'].items():
        diag = cd.circular_decode_for_group(gflag)
        assert not any(f'{diag[0]} by {diag[1]}' in cell for cell in res['cells'])

    assert (tmp_path / 'cross_decoding.json').exists()
    assert (tmp_path / 'summary.txt').exists()
