"""Top-fraction electrode sets on held-out trials."""
import os
import subprocess
import sys

import mne
import numpy as np
import pandas as pd
import pytest

from src.analysis.decoding.score_electrode_selection import (
    held_out_electrode_sets, save_score_selection, score_electrode_sets)
from src.analysis.stats.stability_flexibility_segregation import naive_sensitivities

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _scores():
    # 10 electrodes in one ROI over two subjects; x (LWPC) and y (LWPS) rankings
    # chosen so the top 20 % (2 each) share one electrode
    return pd.DataFrame(dict(
        roi='lpfc', subject=['A'] * 5 + ['B'] * 5, channel=[f'c{i}' for i in range(10)],
        x=[9, 8, 0, 0, 0, 0, 0, 0, 0, 0.], y=[0, 9, 0, 0, 0, 0, 0, 0, 0, 8.],
        mx=np.arange(10.), my=-np.arange(10.)))


def test_score_electrode_sets_builds_each_score_and_each_pair():
    sets, scored = score_electrode_sets(_scores(), frac=0.2, sign='positive')
    assert sets['lwpc'] == {'lpfc': {'A': ['c0', 'c1']}}
    assert sets['lwps'] == {'lpfc': {'A': ['c1'], 'B': ['c9']}}
    assert sets['lwpc_only']['lpfc'] == {'A': ['c0'], 'B': []}
    assert sets['lwps_only']['lpfc'] == {'A': [], 'B': ['c9']}
    assert sets['adaptation_overlap']['lpfc'] == {'A': ['c1'], 'B': []}
    assert sets['congruency'] == {'lpfc': {'B': ['c8', 'c9']}}
    assert sets['switch_type'] == {'lpfc': {'A': ['c0', 'c1']}}
    assert scored['in_lwpc'].sum() == 2 and 'main_effects_overlap' in sets


def _structure(n=40, seed=0):
    """One subject, two conditions, trial_count ids 0..n-1 split over them."""
    rng = np.random.default_rng(seed)
    info = mne.create_info(['c0', 'c1', 'c2'], 100., 'eeg')
    out = {}
    for k, cond in enumerate(('Stimulus_c', 'Stimulus_i')):
        ids = np.arange(k, n, 2).astype(float)
        ep = mne.EpochsArray(rng.normal(size=(len(ids), 3, 20)), info, tmin=-0.1, verbose=False)
        ep.metadata = pd.DataFrame(dict(trial_count=ids))
        out[cond] = {'HG_ev1_power_rescaled': ep}
    return {'A': out}


def test_held_out_sets_keep_only_decode_trials_and_loaded_electrodes(tmp_path):
    structure = _structure()
    partitions = {'A': {'select': set(range(0, 12)), 'decode': set(range(12, 40))}}
    sets = {'lwpc': {'lpfc': {'A': ['c0', 'cX']}}, 'lwps': {'lpfc': {'A': ['c2']}}}
    path = save_score_selection(str(tmp_path / 's.json'), sets, partitions,
                                dict(top_frac=0.2, sign='positive'))
    kept, decode, params = held_out_electrode_sets(path, ['lpfc'], structure, names=['lwpc'])
    assert kept == {'lwpc': {'lpfc': {'A': ['c0']}}}       # cX is not loaded
    for cond in decode['A'].values():
        ids = cond['HG_ev1_power_rescaled'].metadata['trial_count']
        assert ids.min() >= 12                               # no selection trial
    assert params['sign'] == 'positive'
    with pytest.raises(ValueError, match='ROIs'):
        held_out_electrode_sets(path, ['occ'], structure)
    with pytest.raises(ValueError, match='no trial split'):
        held_out_electrode_sets(path, ['lpfc'], {**structure, 'B': structure['A']})


def test_trial_split_does_not_depend_on_the_python_hash_seed():
    code = ("from src.analysis.decoding.anova_electrode_selection import assign_trial_partitions;"
            "import numpy as np;"
            "t={'D0057': (np.arange(50.), np.array(['a','b']*25, dtype=object))};"
            "print(sorted(assign_trial_partitions(t, 0.3, seed=1)['D0057']['select']))")
    outs = {subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, 'PYTHONHASHSEED': str(h)}).stdout
            for h in (1, 2, 3)}
    assert len(outs) == 1 and outs.pop().startswith('[')


def test_naive_sensitivities_main_effects():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(dict(subject='S', electrode='S-e0', trial=np.arange(n),
                           congruency=rng.choice(['c', 'i'], n),
                           switchType=rng.choice(['s', 'r'], n),
                           incongruent_proportion=rng.choice([25.0, 75.0], n),
                           switch_proportion=rng.choice([25.0, 75.0], n)))
    df['hg'] = rng.normal(size=n) + 1.0 * (df['congruency'] == 'i')
    sc = naive_sensitivities(df, contrast_mode='proportion', main_effects=True)
    assert sc.loc[0, 'mx'] > 0.7 and abs(sc.loc[0, 'my']) < 0.3
    with pytest.raises(ValueError, match='proportion'):
        naive_sensitivities(df, contrast_mode='condition', main_effects=True)
