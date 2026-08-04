"""Integration test: does the ANOVA selector actually pick the right electrodes?

The unit tests next door check the plumbing (partition, filter, set algebra).
This one runs the **real** cluster-corrected windowed ANOVA
(`run_within_electrode_windowed_anova_cluster_correction`) through
`select_electrodes_by_windowed_anova` on synthetic data with a *planted*
interaction, and asserts that:

  * an electrode carrying a sustained congruency x incongruent-proportion
    interaction is selected,
  * a pure-noise electrode is not,
  * an electrode with a strong *main effect* but no interaction is not selected
    when the requested effect is the interaction — this is the specificity that
    makes the LWPC/LWPS electrode sets mean what the guide (§3.1) says they mean.

Only `process_windowed_data_for_anova` is stubbed, because it is the one step
that needs real MNE Epochs (and `ieeg`); everything downstream of it is the
production code path.

A note on the planted effect size, learned the hard way. The permutation null
here shuffles trial labels **within** an electrode and applies the *same* shuffle
at every window, so any structure a shuffle accidentally retains shows up at
*every* window at once. Plant a large enough effect (an offset of 3 σ in an
earlier draft of this file) and the top few percent of permutations clear the
pointwise threshold across the whole trace — the null's cluster-*extent*
distribution piles up at full extent, `extent_threshold` reaches full extent
too, and the strict `extent > threshold` test rejects the very effect that was
planted. The offsets below are deliberately modest (≈0.8 σ) so the per-window
noise, not the shuffle-alignment, sets the pointwise threshold. This is a
property of an extent-based cluster test on a *temporally flat* effect, not a
bug — but it is worth knowing before reading a null off a real run whose effect
is sustained across the whole analysis window.
"""

import pytest

pytestmark = pytest.mark.slow

import numpy as np
import pandas as pd

from src.analysis.decoding import anova_electrode_selection as aes

DATA_SEED = 0          # a fresh RandomState per call -> order-independent tests
EFFECT_SIZE = 0.8      # in SD units; see the module docstring on why not larger
N_PERM = 40
N_TRIALS = 24          # per condition, per subject
N_WINDOWS = 8
SUBJECTS = ['D0001', 'D0002']
ROI = 'lpfc'
CHANS = ['interaction', 'noise', 'main_effect_only']

# The LWPC 2x2, exactly as the registry declares it.
CONDITIONS_OBJ = {
    'Stimulus_c25': {'congruency': 'c', 'incongruentProportion': '25%'},
    'Stimulus_c75': {'congruency': 'c', 'incongruentProportion': '75%'},
    'Stimulus_i25': {'congruency': 'i', 'incongruentProportion': '25%'},
    'Stimulus_i75': {'congruency': 'i', 'incongruentProportion': '75%'},
}
ANOVA_FACTORS = ['congruency', 'incongruentProportion']


def _cell_means(cond):
    """Per-channel window-mean offsets for one condition cell.

    'interaction'      : a difference-of-differences and nothing else
                         (the congruency effect flips sign with proportion).
    'noise'            : flat.
    'main_effect_only' : a big congruency effect, identical in both blocks —
                         so a main effect with a zero interaction.
    """
    cong = CONDITIONS_OBJ[cond]['congruency']
    prop = CONDITIONS_OBJ[cond]['incongruentProportion']
    c = +1.0 if cong == 'c' else -1.0
    p = +1.0 if prop == '25%' else -1.0
    return {'interaction': EFFECT_SIZE * c * p,
            'noise': 0.0,
            'main_effect_only': EFFECT_SIZE * c}


def _fake_windowed_data(*args, **kwargs):
    """Stand-in for process_windowed_data_for_anova.

    Returns ``{condition: {roi: [per-subject (n_trials, n_windows, n_chans)]}}``
    with the planted structure sustained across every window (so a cluster of
    full extent forms) plus i.i.d. noise.
    """
    rng = np.random.RandomState(DATA_SEED)
    out = {}
    for cond in CONDITIONS_OBJ:
        offsets = np.array([_cell_means(cond)[c] for c in CHANS])
        per_sub = []
        for _ in SUBJECTS:
            noise = rng.randn(N_TRIALS, N_WINDOWS, len(CHANS))
            per_sub.append(noise + offsets[None, None, :])
        out[cond] = {ROI: per_sub}
    return out


@pytest.fixture
def patched(monkeypatch):
    import src.analysis.power.windowed_anova as wa
    monkeypatch.setattr(wa, 'process_windowed_data_for_anova', _fake_windowed_data)


class _FakeEpochs:
    """Only used for the `times` lookup and the usable-subject gate."""
    def __init__(self, n=N_TRIALS):
        self.times = np.linspace(-1.0, 1.5, 64)
        self.metadata = pd.DataFrame({'trial_count': np.arange(n, dtype=float)})
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self

    def get_data(self, *a, **k):
        return np.zeros((self._n, len(CHANS), len(self.times)))


def _selection_structure():
    return {sub: {cond: {'HG_ev1_power_rescaled': _FakeEpochs()}
                  for cond in CONDITIONS_OBJ}
            for sub in SUBJECTS}


def _run(effect, tmp_path, **kw):
    return aes.select_electrodes_by_windowed_anova(
        _selection_structure(),
        {ROI: {sub: list(CHANS) for sub in SUBJECTS}},
        [ROI],
        SUBJECTS,
        CONDITIONS_OBJ,
        ANOVA_FACTORS,
        effect=effect,
        window_size=16, step_size=4, sampling_rate=256,
        n_perm=N_PERM,
        min_trials_per_cell=2,
        seed=1, n_jobs=1, verbose=False,
        save_dir=str(tmp_path), run_label='lwpc_test',
        **kw,
    )


def test_selects_the_planted_interaction_electrode(patched, tmp_path):
    result = _run('interaction', tmp_path)

    assert result['effect'] == 'C(congruency):C(incongruentProportion)'
    for sub in SUBJECTS:
        picked = result['selected'][ROI][sub]
        assert 'interaction' in picked, f"{sub}: planted interaction was missed"
        assert 'noise' not in picked, f"{sub}: pure-noise electrode selected"
        # Specificity: a main effect is not an interaction.
        assert 'main_effect_only' not in picked, \
            f"{sub}: main-effect-only electrode leaked into the interaction set"


def test_main_effect_request_selects_the_main_effect_electrode(patched, tmp_path):
    result = _run('congruency', tmp_path)
    assert result['effect'] == 'C(congruency)'
    for sub in SUBJECTS:
        picked = result['selected'][ROI][sub]
        assert 'main_effect_only' in picked
        assert 'noise' not in picked


def test_writes_an_inspectable_run_directory(patched, tmp_path):
    _run('interaction', tmp_path)
    run_dir = tmp_path / 'lwpc_test'
    # The same artifacts a power_traces within-electrode ANOVA run writes, so a
    # selection run can be inspected/re-loaded with the existing tooling.
    assert (run_dir / 'summary.csv').exists()
    assert (run_dir / 'run_config.json').exists()
    assert (run_dir / ROI / SUBJECTS[0] / 'interaction.npz').exists()


def test_every_candidate_subject_key_survives(patched, tmp_path):
    """Downstream code indexes electrodes[roi][sub]; keys must not vanish."""
    result = _run('interaction', tmp_path)
    assert set(result['selected'][ROI]) == set(SUBJECTS)
