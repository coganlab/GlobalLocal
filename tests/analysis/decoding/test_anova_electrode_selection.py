"""Unit tests for the ANOVA electrode-selection path.

Covers the pure pieces of ``src/analysis/decoding/anova_electrode_selection.py``:
the trial-id-keyed partition (the property that makes selection and decoding
disjoint *across different condition sets*), the summary-frame filter, the set
algebra, and the naming. The ANOVA call itself needs real epochs and `ieeg`, so
it is smoke-tested on the cluster, not here — same division as
``test_trial_splitting.py``.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.decoding.anova_electrode_selection import (
    apply_trial_partition,
    assign_trial_partitions,
    available_combo_names,
    collect_subject_trial_strata,
    combine_electrode_sets,
    decoding_figure_title,
    describe_electrode_set,
    electrode_set_counts,
    electrode_set_counts_for_roi,
    electrode_set_slug,
    interaction_effect_name,
    pretty_electrode_set,
    resolve_effect,
    safe_effect_slug,
    significant_electrodes_from_summary,
)


# --------------------------------------------------------------------------
# A minimal Epochs stand-in: indexable, has .times / .metadata / len.
# --------------------------------------------------------------------------
class FakeEpochs:
    def __init__(self, metadata, n_times=8):
        self.metadata = metadata.reset_index(drop=True)
        self.times = np.linspace(-1.0, 1.0, n_times)
        self._data = np.zeros((len(self.metadata), 2, n_times))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        return FakeEpochs(self.metadata.iloc[idx], n_times=len(self.times))

    def get_data(self, *a, **k):
        return self._data


def _md(trial_ids, congruency=None, task_sequence=None):
    n = len(trial_ids)
    return pd.DataFrame({
        'trial_count': np.asarray(trial_ids, dtype=float),
        'congruency': congruency if congruency is not None else ['c', 'i'] * (n // 2) + ['c'] * (n % 2),
        'task_sequence': task_sequence if task_sequence is not None else ['s', 'r'] * (n // 2) + ['s'] * (n % 2),
    })


def _structure(cond_to_ids, sub='D0001'):
    return {sub: {cond: {'HG_ev1_power_rescaled': FakeEpochs(_md(ids))}
                  for cond, ids in cond_to_ids.items()}}


# ------------------------------ trial partition ----------------------------
def test_partition_is_disjoint_and_covers_every_trial():
    structure = _structure({'A': list(range(0, 20)), 'B': list(range(20, 40))})
    sub_trials = collect_subject_trial_strata([structure])
    parts = assign_trial_partitions(sub_trials, frac_select=0.3, seed=0)
    sel, dec = parts['D0001']['select'], parts['D0001']['decode']
    assert sel.isdisjoint(dec)
    assert sel | dec == set(float(i) for i in range(40))


def test_partition_respects_frac_select():
    structure = _structure({'A': list(range(100))})
    parts = assign_trial_partitions(
        collect_subject_trial_strata([structure]), frac_select=0.3, seed=0)
    n_sel = len(parts['D0001']['select'])
    # 4 strata cells (c/i x s/r), rounding within each -> allow a few trials of slack
    assert 26 <= n_sel <= 34


def test_partition_is_shared_across_different_condition_sets():
    """The property the whole design rests on.

    The LWPC structure and the LWPS structure slice the SAME physical trials
    into different conditions. A positional split would put trial 7 in the
    selection half of one and the decode half of the other; keying on the trial
    id cannot.
    """
    ids = list(range(40))
    lwpc = _structure({'c25': ids[:20], 'i25': ids[20:]})
    # Same trials, cut a different way.
    lwps = _structure({'s25': ids[::2], 'r25': ids[1::2]})
    decode = _structure({'all': ids})

    parts = assign_trial_partitions(
        collect_subject_trial_strata([decode, lwpc, lwps]),
        frac_select=0.3, seed=0)

    sel_lwpc = apply_trial_partition(lwpc, parts, which='select', verbose=False)
    sel_lwps = apply_trial_partition(lwps, parts, which='select', verbose=False)
    dec = apply_trial_partition(decode, parts, which='decode', verbose=False)

    def ids_of(structure):
        out = set()
        for cond_dict in structure.values():
            for obj in cond_dict.values():
                out |= set(obj['HG_ev1_power_rescaled'].metadata['trial_count'])
        return out

    # Every selection trial, from EITHER selection condition set, is absent from
    # the decode partition.
    assert ids_of(sel_lwpc).isdisjoint(ids_of(dec))
    assert ids_of(sel_lwps).isdisjoint(ids_of(dec))
    # ...and the two selection sets see exactly the same physical trials.
    assert ids_of(sel_lwpc) == ids_of(sel_lwps)


def test_partition_is_deterministic():
    structure = _structure({'A': list(range(30))})
    a = assign_trial_partitions(collect_subject_trial_strata([structure]), seed=5)
    b = assign_trial_partitions(collect_subject_trial_strata([structure]), seed=5)
    assert a['D0001']['select'] == b['D0001']['select']


def test_apply_partition_drops_empty_conditions():
    # Condition B's trials all land on the selection side, so the decode side of
    # B is empty and must be dropped rather than handed on as a 0-trial Epochs.
    structure = _structure({'A': list(range(20)), 'B': [100]})
    parts = {'D0001': {'select': {100.0} | set(float(i) for i in range(10)),
                       'decode': set(float(i) for i in range(10, 20))}}
    dec = apply_trial_partition(structure, parts, which='decode', verbose=False)
    assert set(dec['D0001']) == {'A'}


def test_apply_partition_drops_subject_with_nothing_left():
    structure = _structure({'A': [1, 2, 3]})
    parts = {'D0001': {'select': {1.0, 2.0, 3.0}, 'decode': set()}}
    assert apply_trial_partition(structure, parts, which='decode',
                                 verbose=False) == {}


def test_missing_trial_id_column_raises():
    md = pd.DataFrame({'congruency': ['c', 'i']})
    structure = {'D0001': {'A': {'HG_ev1_power_rescaled': FakeEpochs(md)}}}
    with pytest.raises(ValueError, match="trial_count"):
        collect_subject_trial_strata([structure])


# ------------------------------ effect naming ------------------------------
def test_interaction_effect_name():
    assert (interaction_effect_name(['congruency', 'incongruentProportion'])
            == 'C(congruency):C(incongruentProportion)')


def test_resolve_effect_variants():
    factors = ['congruency', 'incongruentProportion']
    assert resolve_effect('interaction', factors) == 'C(congruency):C(incongruentProportion)'
    assert resolve_effect('any', factors) is None
    assert resolve_effect('congruency', factors) == 'C(congruency)'
    assert resolve_effect('C(weird)', factors) == 'C(weird)'
    with pytest.raises(ValueError):
        resolve_effect('interaction', ['congruency'])


def test_safe_effect_slug():
    assert safe_effect_slug('C(congruency):C(incongruentProportion)') == \
        'congruency_x_incongruentProportion'
    assert safe_effect_slug(None) == 'any_effect'


# --------------------------- summary-frame filter --------------------------
def _summary():
    return pd.DataFrame([
        # (sub, elec, roi, effect, sig_after_fdr, cluster_p)
        ('D1', 'e1', 'lpfc', 'C(a):C(b)', True, 0.01),
        ('D1', 'e2', 'lpfc', 'C(a):C(b)', False, 0.30),
        ('D1', 'e2', 'lpfc', 'C(a)', True, 0.02),
        ('D2', 'e9', 'lpfc', 'C(a):C(b)', True, 0.04),
        ('D2', 'e8', 'lpfc', 'C(a):C(b)', False, 0.06),
    ], columns=['subject', 'electrode', 'roi', 'effect', 'sig_after_fdr',
                'cluster_p_value'])


def test_summary_filter_by_effect_and_fdr():
    candidates = {'lpfc': {'D1': ['e1', 'e2'], 'D2': ['e8', 'e9']}}
    got = significant_electrodes_from_summary(
        _summary(), ['lpfc'], candidates, effect_name='C(a):C(b)', use_fdr=True)
    assert got['lpfc']['D1'] == ['e1']
    assert got['lpfc']['D2'] == ['e9']


def test_summary_filter_any_effect_picks_up_the_main_effect():
    candidates = {'lpfc': {'D1': ['e1', 'e2']}}
    got = significant_electrodes_from_summary(
        _summary(), ['lpfc'], candidates, effect_name=None, use_fdr=True)
    # e2 is not significant on the interaction but is on the main effect.
    assert got['lpfc']['D1'] == ['e1', 'e2']


def test_summary_filter_never_widens_the_candidate_pool():
    candidates = {'lpfc': {'D1': ['e2']}}   # e1 is not a candidate here
    got = significant_electrodes_from_summary(
        _summary(), ['lpfc'], candidates, effect_name='C(a):C(b)', use_fdr=True)
    assert got['lpfc']['D1'] == []


def test_summary_filter_raw_p_route():
    candidates = {'lpfc': {'D2': ['e8', 'e9']}}
    got = significant_electrodes_from_summary(
        _summary(), ['lpfc'], candidates, effect_name='C(a):C(b)',
        use_fdr=False, alpha=0.05)
    assert got['lpfc']['D2'] == ['e9']      # p=0.04 in, p=0.06 out


def test_summary_filter_empty_frame():
    got = significant_electrodes_from_summary(
        pd.DataFrame(), ['lpfc'], {'lpfc': {'D1': ['e1']}})
    assert got == {'lpfc': {}}


# ------------------------------- set algebra -------------------------------
@pytest.fixture
def named_sets():
    return {
        'lwpc': {'lpfc': {'D1': ['a', 'b'], 'D2': ['x']}},
        'lwps': {'lpfc': {'D1': ['b', 'c'], 'D2': []}},
    }


def test_combine_builds_unique_overlap_union(named_sets):
    sets = combine_electrode_sets(named_sets, ['lpfc'])
    assert sets['lwpc_only']['lpfc']['D1'] == ['a']
    assert sets['lwps_only']['lpfc']['D1'] == ['c']
    assert sets['overlap']['lpfc']['D1'] == ['b']
    assert sorted(sets['union']['lpfc']['D1']) == ['a', 'b', 'c']
    # An electrode only one subject has still lands in the right cells.
    assert sets['lwpc_only']['lpfc']['D2'] == ['x']
    assert sets['overlap']['lpfc']['D2'] == []


def test_combine_labels_pass_through(named_sets):
    sets = combine_electrode_sets(named_sets, ['lpfc'], combos=['lwpc', 'lwps'])
    assert set(sets) == {'lwpc', 'lwps'}
    assert sets['lwpc']['lpfc']['D1'] == ['a', 'b']


def test_combine_rejects_unknown_combo(named_sets):
    with pytest.raises(ValueError, match='Unknown electrode-set combination'):
        combine_electrode_sets(named_sets, ['lpfc'], combos=['nonsense'])


def test_available_combo_names():
    assert available_combo_names(['lwpc', 'lwps']) == [
        'lwpc', 'lwps', 'lwpc_only', 'lwps_only', 'overlap', 'union']
    # A single label has no unique/overlap/union to speak of.
    assert available_combo_names(['lwpc']) == ['lwpc']


def test_same_channel_name_in_two_subjects_is_not_confused():
    """Channel names are only unique within a subject."""
    sets = combine_electrode_sets({
        'lwpc': {'roi': {'D1': ['LFMM9'], 'D2': []}},
        'lwps': {'roi': {'D1': [], 'D2': ['LFMM9']}},
    }, ['roi'])
    assert sets['overlap']['roi']['D1'] == []
    assert sets['overlap']['roi']['D2'] == []
    assert sets['lwpc_only']['roi']['D1'] == ['LFMM9']
    assert sets['lwps_only']['roi']['D2'] == ['LFMM9']


# --------------------------------- naming ----------------------------------
def test_counts(named_sets):
    assert electrode_set_counts(named_sets['lwpc']) == (3, 2)
    assert electrode_set_counts_for_roi(named_sets['lwpc'], 'lpfc') == (3, 2)
    assert electrode_set_counts_for_roi(named_sets['lwpc'], 'missing') == (0, 0)


def test_slug_and_pretty():
    assert electrode_set_slug('lwpc_only') == 'lwpc_only'
    assert electrode_set_slug('C(a):C(b)') == 'C_a_C_b'
    assert pretty_electrode_set('lwpc_only') == 'LWPC-only'
    assert pretty_electrode_set('overlap') == 'overlap'


def test_describe_and_title(named_sets):
    desc = describe_electrode_set('lwpc_only', named_sets['lwpc'], roi='lpfc')
    assert 'LWPC-only' in desc and 'n = 3' in desc
    title = decoding_figure_title('c25_vs_i25', 'lpfc', desc)
    assert title.splitlines()[0] == 'c25 vs i25 — lpfc'
    assert 'LWPC-only' in title.splitlines()[1]
