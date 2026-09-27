"""Tests for the task-transfer positive controls (docs/cross_decoding_controls.md §3.5).

The controls are N3b's train-in-one-level / test-in-the-other 2x2
(`block_transfer`) with a trial-level factor in place of the block:

    T1  task (global vs local), congruent trials -> incongruent trials
    T2  task, repeat trials -> switch trials (previous-task confound)
    T3  congruency, global-task trials -> local-task trials
    T4  switch type, global-task trials -> local-task trials

1. The real condition sets and the synthetic generator declare the 2x2 each
   design needs (no `ieeg` needed).
2. Planted answers on synthetic data: a shared task code transfers everywhere,
   a congruency-specific task code fails T1 but passes T2, and previous-task
   carryover breaks T2 but not T1.
3. The DCC job, end to end, with `analysis='task_transfer'`.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.analysis.config import experiment_conditions as ec  # noqa: E402
from src.analysis.decoding import block_transfer as bt  # noqa: E402
from src.analysis.decoding import cross_decoding as cd  # noqa: E402
from dcc_scripts.decoding import stability_flexibility_cross_decoding_dcc as xd  # noqa: E402

CELLS = cd.synthetic_task_condition_cells()
T1 = ('task', 'congruency')
T2 = ('task', 'switchType')

ieeg_required = pytest.mark.skipif(
    __import__('importlib').util.find_spec('ieeg') is None,
    reason="requires the `ieeg` package (cluster environment)")


# ---------------------------------------------------------------------------
# 1. condition sets and cells
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', sorted(xd.TASK_TRANSFER_DESIGNS))
def test_each_design_reads_a_2x2_from_its_real_condition_set(name):
    contrast, block_col, conditions_name = xd.TASK_TRANSFER_DESIGNS[name]
    cells = cd.condition_cells(getattr(ec, conditions_name), required=(contrast, block_col))
    assert len(cells) == 4
    assert {cell[contrast] for cell in cells.values()} == set(bt.CONTRAST_LEVELS[contrast])
    assert {cell[block_col] for cell in cells.values()} == set(bt.CONTRAST_LEVELS[block_col])


def test_task_is_read_only_where_a_condition_declares_it():
    cells = cd.condition_cells(ec.stimulus_experiment_conditions)
    assert all(cell['task'] is None for cell in cells.values())


def test_synthetic_task_cells_match_the_generator_condition_names():
    arrays = cd.synthetic_task_labeled_arrays(seed=0)
    assert set(CELLS) == set(arrays['synthetic'])


def test_prepare_balances_task_within_each_congruency_level():
    p = bt.prepare(cd.synthetic_task_labeled_arrays(seed=0), 'synthetic', CELLS, *T1)
    sizes = dict(zip(*np.unique(p['groups'], return_counts=True)))
    # 2 classes x 2 transfer levels, each pooling both switch types
    assert sizes == {f'{task}|congruency={cong}': 80 for task in 'gl' for cong in 'ci'}
    assert set(p['block']) == {'c', 'i'}


def test_a_contrast_cannot_be_transferred_across_itself():
    with pytest.raises(ValueError, match='other than the contrast'):
        bt.prepare(cd.synthetic_task_labeled_arrays(seed=0), 'synthetic', CELLS,
                   'task', 'task')


# ---------------------------------------------------------------------------
# 2. planted answers
# ---------------------------------------------------------------------------
def _accuracies(arrs, design):
    """{(train_level, test_level): mean accuracy} for one design."""
    from src.analysis.decoding.accuracy_stats import compute_accuracies
    out = bt.run_block_transfer(arrs, 'synthetic', CELLS, *design,
                                n_resamples=2, n_splits=3, window=16, step_size=16)
    return {pair: compute_accuracies(cms['cm_true'], cms['cm_shuffle'])[0].mean()
            for pair, cms in out['cells'].items()}


@ieeg_required
def test_a_shared_task_code_transfers_across_congruency_and_switch_type():
    arrs = cd.synthetic_task_labeled_arrays(seed=0)
    for design in (T1, T2):
        acc = _accuracies(arrs, design)
        lo, hi = sorted({train for train, _ in acc})
        assert min(acc.values()) > 0.7
        assert abs(acc[(lo, hi)] - acc[(hi, hi)]) < 0.07
        assert abs(acc[(hi, lo)] - acc[(lo, lo)]) < 0.07


@ieeg_required
def test_a_congruency_specific_task_code_fails_t1_but_passes_t2():
    """Task on a different axis on incongruent trials: T1 must fail in both
    directions, while T2, whose levels each mix both congruencies, transfers."""
    arrs = cd.synthetic_task_labeled_arrays(seed=0, task_code='congruency_specific')
    t1, t2 = _accuracies(arrs, T1), _accuracies(arrs, T2)
    assert t1[('c', 'c')] > 0.7 and t1[('i', 'i')] > 0.7
    assert t1[('c', 'i')] < 0.6 and t1[('i', 'c')] < 0.6
    assert t2[('r', 's')] > 0.6 and t2[('s', 'r')] > 0.6
    assert t2[('s', 's')] - t2[('r', 's')] < 0.07


@ieeg_required
def test_previous_task_carryover_breaks_t2_but_not_t1():
    """The task x switch type confound: leftover previous-task activity agrees
    with the current task on repeat trials and opposes it on switch trials."""
    arrs = cd.synthetic_task_labeled_arrays(seed=0, carryover=1.2)
    t1, t2 = _accuracies(arrs, T1), _accuracies(arrs, T2)
    assert t2[('r', 'r')] > 0.9 and t2[('s', 's')] > 0.9
    assert t2[('r', 's')] < 0.4 and t2[('s', 'r')] < 0.4
    assert abs(t1[('c', 'i')] - t1[('i', 'i')]) < 0.07
    assert abs(t1[('i', 'c')] - t1[('c', 'c')]) < 0.07


# ---------------------------------------------------------------------------
# 3. the DCC job, end to end on synthetic data
# ---------------------------------------------------------------------------
@ieeg_required
def test_the_task_transfer_job_runs_end_to_end(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        analysis='task_transfer', data_source='synthetic',
        synthetic_code='congruency_specific', electrodes='sig', window_size=16,
        step_size=16, sampling_rate=256, n_splits=3, n_repeats=2,
        explained_variance=0.8, frac_train=None, n_perm=20, seed=0,
        save_dir=str(tmp_path))
    results = xd.main(args)

    assert set(results) == {f'{d}_{c}' for d in xd.TASK_TRANSFER_DESIGNS
                            for c in ('uncentered', 'centered')}
    for name in ('task_transfer.json', 'task_transfer_traces.npz', 'summary.txt',
                 'T1_uncentered_congruent_to_incongruent_synthetic_task_transfer.png'):
        assert (tmp_path / name).exists(), name
    traces = np.load(tmp_path / 'task_transfer_traces.npz')
    assert 'T3_centered_global_to_local_true' in traces

    t1, t2 = results['T1_uncentered']['cells'], results['T2_uncentered']['cells']
    assert t1['c->i']['post_mean_accuracy'] < 0.6 < t1['i->i']['post_mean_accuracy']
    assert t2['r->s']['post_mean_accuracy'] > 0.6

    summary = (tmp_path / 'summary.txt').read_text()
    assert 'TASK-TRANSFER POSITIVE CONTROLS' in summary
    assert summary.count('CEILING:') == 8          # one verdict per design x centering
    assert 'congruent -> incongruent vs within incongruent' in summary
    assert 'EFFECT SIZE' in summary
