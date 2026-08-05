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
    """A condition that declares only ONE of the two transferred factors cannot be
    cross-decoded, so it never enters the cells."""
    conditions = dict(ec.stimulus_experiment_conditions)
    conditions['Stimulus_partial'] = {'BIDS_events': ['Stimulus'], 'congruency': 'i'}
    assert 'Stimulus_partial' not in cd.condition_cells(conditions)


def test_requiring_all_four_factors_still_drops_partial_cells():
    conditions = dict(ec.stimulus_main_effect_conditions)
    assert len(cd.condition_cells(conditions)) == 4
    with pytest.raises(ValueError, match='no condition declares'):
        cd.condition_cells(conditions, required=cd.CELL_FIELDS)


def test_a_single_factor_condition_set_raises():
    """`stimulus_congruency_conditions` / `stimulus_switch_type_conditions` carry
    only one of the two labellings, so no condition can be crossed."""
    for conditions in (ec.stimulus_congruency_conditions,
                       ec.stimulus_switch_type_conditions):
        with pytest.raises(ValueError, match='no condition declares'):
            cd.condition_cells(conditions)


# ---------------------------------------------------------------------------
# the pooled 2x2: ALL congruency vs ALL switch type, collapsed over proportions
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pooled_cells():
    return cd.condition_cells(ec.stimulus_main_effect_conditions)


def test_pooled_conditions_are_the_2x2_over_both_proportions(pooled_cells):
    assert len(pooled_cells) == 4
    assert {c['congruency'] for c in pooled_cells.values()} == {'i', 'c'}
    assert {c['switchType'] for c in pooled_cells.values()} == {'s', 'r'}
    # the proportions are POOLED, not declared
    for field in ('incongruent_proportion', 'switch_proportion'):
        assert all(c[field] is None for c in pooled_cells.values())


def test_pooled_conditions_still_define_both_contrasts(pooled_cells):
    stab, flex = cd.stability_flexibility_strings(pooled_cells)
    for name, cell in pooled_cells.items():
        assert cd._class_of(name, stab) == (0 if cell['congruency'] == 'i' else 1)
        assert cd._class_of(name, flex) == (0 if cell['switchType'] == 's' else 1)


def test_has_block_factor_separates_the_two_condition_sets(real_cells, pooled_cells):
    for field in ('incongruent_proportion', 'switch_proportion'):
        assert cd.has_block_factor(real_cells, field)
        assert not cd.has_block_factor(pooled_cells, field)
        with pytest.raises(ValueError, match='cannot be split'):
            cd.block_condition_sets(pooled_cells, field)


def test_the_pooled_transfer_uses_every_trial(pooled_cells):
    """The point of the pooled set: each class spans all four proportion cells, so
    the transfer is trained and scored on ALL trials, not within a block."""
    arrays = {'roi': {n: np.zeros((4, 2, 3)) for n in pooled_cells}}
    stab, flex = cd.stability_flexibility_strings(pooled_cells)
    built = cd.build_cross_decoding_arrays(arrays, 'roi', stab, flex)
    assert set(built['conditions']) == set(pooled_cells)
    assert built['data'].shape[0] == 4 * len(pooled_cells)
    assert set(built['labels_train']) == set(built['labels_test']) == {0, 1}


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


def _stub_args(tmp_path, conditions):
    return SimpleNamespace(
        data_source='real', synthetic_code=None, LAB_root=None, subjects=[],
        task='GlobalLocal', acc_trials_only=True, epochs_root_file='epochs',
        window_tmin=0.0, window_tmax=0.5, conditions=conditions,
        electrodes='sig', rois_dict={'lpfc': []}, alpha=0.05, roi='lpfc',
        electrode_definition='power_traces', power_traces_runs='run',
        power_traces_correction='fdr_bh', power_traces_roi=None,
        reference_group='all', tempgen_groups=(),
        window_size=8, step_size=8, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=10, min_group_size=2,
        seed=0, save_dir=str(tmp_path))


def _stub_data_loading(monkeypatch, cells, labels, channel_names):
    """Stub out only the two things that need the cluster's data."""
    arrays = {'lpfc': _fake_roi_arrays(cells)}
    monkeypatch.setattr(xd, '_resolve_labels', lambda args, df=None: labels)
    monkeypatch.setattr(xd, '_build_roi_arrays',
                        lambda args, lab_root: ('lpfc', arrays, channel_names, cells))
    monkeypatch.setattr('src.analysis.utils.general_utils.resolve_lab_root',
                        lambda explicit=None: '/nonexistent')


CHANNEL_NAMES = [f'D{i // 4}-E{i % 4}' for i in range(8)]
LABELS = pd.DataFrame(dict(
    subject=[c.split('-')[0] for c in CHANNEL_NAMES],
    electrode=CHANNEL_NAMES,
    S=[1, 1, 1, 0, 1, 1, 0, 0], F=[1, 1, 1, 1, 0, 0, 1, 0],
    CPC=[1, 1, 1, 0, 1, 1, 0, 0], SPS=[1, 1, 1, 1, 0, 0, 1, 0],
    CPS=[0, 0, 0, 0, 0, 0, 0, 0], SPC=[0, 0, 0, 0, 0, 0, 0, 0]))


def test_real_branch_runs_over_real_condition_names(tmp_path, monkeypatch):
    cells = cd.condition_cells(ec.stimulus_experiment_conditions)
    _stub_data_loading(monkeypatch, cells, LABELS, CHANNEL_NAMES)

    results = xd.main(_stub_args(tmp_path, ec.stimulus_experiment_conditions))

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

    # the transfer is POOLED over the proportions even here: its classes span all
    # 16 cells, so the 16-cell set does not restrict it to within a block
    assert len(results['label_transfer']['all']['stab_to_flex']['conditions']) == 16


def test_pooled_branch_transfers_over_all_trials(tmp_path, monkeypatch):
    """`stimulus_main_effect_conditions`: ALL congruency vs ALL switch type, with
    both proportions collapsed. The transfer runs; the block designs are skipped
    rather than split on a factor the conditions pool over."""
    cells = cd.condition_cells(ec.stimulus_main_effect_conditions)
    _stub_data_loading(monkeypatch, cells, LABELS, CHANNEL_NAMES)

    results = xd.main(_stub_args(tmp_path, ec.stimulus_main_effect_conditions))

    assert results['within_block'] == {}
    assert results.get('within_block_by_group') is None

    lt = results['label_transfer']
    assert set(lt) >= {'both', 'all'}
    for group, res in lt.items():
        assert set(res) == {'stab_to_flex', 'flex_to_stab'}
        for direction, r in res.items():
            assert set(r['conditions']) == set(ec.stimulus_main_effect_conditions)

    assert (tmp_path / 'cross_decoding.json').exists()
    assert (tmp_path / 'summary.txt').exists()
