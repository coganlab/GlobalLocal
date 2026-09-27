"""`assemble_long_df` carries a shared per-subject `trial` index, `rt` and `acc`.

The A6 per-participant scores split each participant's TRIALS once and apply that
one split to every electrode, and they remove the RT-linked part of single-trial
HG. Both need these columns on the long table:

  * `trial` must be the epoch index, identical across a subject's electrodes, and
    must survive per-electrode row drops (a NaN'd trial on one channel), which a
    row counter would not;
  * `rt` / `acc` come from the metadata aliases the event-name parser emits, and
    are NaN, not an error, when the metadata has neither.
"""
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

_DCC = 'dcc_scripts.stats.stability_flexibility_segregation_dcc'
_GU = 'src.analysis.utils.general_utils'


@pytest.fixture
def seg_dcc():
    """The segregation DCC module, imported with `general_utils` stubbed out.

    `general_utils` pulls in the `ieeg` stack at import time and
    `assemble_long_df` needs none of it. Both modules are restored afterwards so
    other tests see whatever they would have seen."""
    saved = {k: sys.modules.get(k) for k in (_DCC, _GU)}
    stub = types.ModuleType(_GU)
    stub.resolve_lab_root = lambda *a, **k: None
    stub.resolve_electrodes_to_keep = lambda *a, **k: None
    sys.modules[_GU] = stub
    sys.modules.pop(_DCC, None)
    try:
        yield importlib.import_module(_DCC)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _FakeEpochs:
    """The four attributes `assemble_long_df` reads from an mne Epochs."""

    def __init__(self, data, metadata, ch_names, times):
        self._data = data
        self.metadata = metadata
        self.ch_names = ch_names
        self.times = times

    def get_data(self):
        return self._data


def _epochs(with_behavior=True, nan_trial=None):
    """6 trials x 2 channels x 5 samples; trial 0 is first-of-block ('n')."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 2, 5))
    if nan_trial is not None:
        data[nan_trial, 1, :] = np.nan            # one trial lost on channel 'B' only
    md = pd.DataFrame(dict(
        congruency=['c', 'i', 'c', 'i', 'i', 'c'],
        task_sequence=['n', 's', 'r', 'r', 's', 'r'],
        incongruent_proportion=[25.0, 25.0, 75.0, 75.0, 25.0, 75.0],
        switch_proportion=[75.0, 75.0, 25.0, 25.0, 75.0, 25.0]))
    if with_behavior:
        md['reaction_time'] = [900.0, 1010.0, 870.0, 1200.0, 950.0, 1100.0]
        md['accuracy'] = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    return _FakeEpochs(data, md, ['A', 'B'], np.linspace(0.0, 0.4, 5)), data, md


def test_trial_rt_and_acc_columns(seg_dcc):
    ep, data, md = _epochs()
    df = seg_dcc.assemble_long_df({'D1': ep}, 0.0, 0.4)
    assert {'trial', 'rt', 'acc'} <= set(df.columns)
    kept = np.array([1, 2, 3, 4, 5])                  # trial 0 ('n') is dropped
    for elec, g in df.groupby('electrode'):
        np.testing.assert_array_equal(g['trial'].to_numpy(), kept)
        np.testing.assert_allclose(g['rt'].to_numpy(),
                                   md['reaction_time'].to_numpy()[kept])
        np.testing.assert_allclose(g['acc'].to_numpy(), md['accuracy'].to_numpy()[kept])
        ci = ['D1-A', 'D1-B'].index(elec)
        np.testing.assert_allclose(g['hg'].to_numpy(), data[kept, ci, :].mean(axis=1))


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_trial_index_survives_a_dropped_row(seg_dcc):
    """A trial NaN'd on one channel drops that row only; `trial` still aligns the
    remaining rows across channels (a per-electrode row counter would not)."""
    ep, _, _ = _epochs(nan_trial=3)
    df = seg_dcc.assemble_long_df({'D1': ep}, 0.0, 0.4)
    a = df[df['electrode'] == 'D1-A'].set_index('trial')
    b = df[df['electrode'] == 'D1-B'].set_index('trial')
    assert list(a.index) == [1, 2, 3, 4, 5]
    assert list(b.index) == [1, 2, 4, 5]
    np.testing.assert_allclose(a.loc[b.index, 'rt'], b['rt'])


def test_missing_behavior_gives_nan_not_an_error(seg_dcc):
    ep, _, _ = _epochs(with_behavior=False)
    df = seg_dcc.assemble_long_df({'D1': ep}, 0.0, 0.4)
    assert df['rt'].isna().all() and df['acc'].isna().all()
    assert len(df) == 10                              # 5 kept trials x 2 channels
