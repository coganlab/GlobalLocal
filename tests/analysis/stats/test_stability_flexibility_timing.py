"""Tests for A5 — stability/flexibility timing (plan §5).

The headline test is the latency–amplitude confound guard: scaling a waveform
must NOT move its 50%-of-peak onset. The rest verify onset recovery, the
Ulrich–Miller (N-1) correction identity, and the planted onset ordering.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..', '..', '..')))

from src.analysis.stats.stability_flexibility_timing import (
    interaction_time_course, onset_50pct_peak, peak_latency,
    jackknife_onset_difference, _synthetic_timecourse_df, _bump,
    _assert_amplitude_invariance,
)


# ---------------------------------------------------------------------------
# the latency–amplitude confound guard (the whole reason for 50%-of-peak)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0.1, 0.5, 2.0, 3.7, 100.0])
def test_onset_invariant_under_amplitude_scaling(k):
    """If stab(t) = k*flex(t), both cross 50%-of-peak at the SAME time."""
    times = np.linspace(-0.2, 0.8, 300)
    wave = _bump(times, 0.25, 0.03, 1.0)
    o1 = onset_50pct_peak(times, wave)
    o2 = onset_50pct_peak(times, k * wave)
    assert np.isclose(o1, o2, atol=1e-9)


def test_baked_in_amplitude_assertion_runs():
    """The module's own baked-in guard returns a finite onset."""
    onset = _assert_amplitude_invariance()
    assert np.isfinite(onset)


def test_onset_recovers_known_crossing():
    """A logistic bump crosses 50%-of-peak near its logistic centre."""
    times = np.linspace(-0.2, 0.8, 500)
    wave = _bump(times, 0.30, 0.01, 1.0)   # sharp rise -> peak ~= amp, mid at 0.30
    assert onset_50pct_peak(times, wave) == pytest.approx(0.30, abs=0.01)


def test_onset_nan_when_no_effect_in_expected_direction():
    """A purely negative-going waveform has no positive onset (expected_sign=+1)."""
    times = np.linspace(-0.2, 0.8, 100)
    wave = -_bump(times, 0.25, 0.03, 1.0)
    assert np.isnan(onset_50pct_peak(times, wave, expected_sign=+1))
    # ...but flips to a valid onset when the expected sign is negative
    assert np.isfinite(onset_50pct_peak(times, wave, expected_sign=-1))


# ---------------------------------------------------------------------------
# direction-agnostic onset (no assumed modulation direction)
# ---------------------------------------------------------------------------
# A neural population's interaction may run either way, so the DEFAULT must read
# an onset off a shrinking (negative-going) d-o-d rather than returning NaN.
def test_onset_default_handles_negative_going_interaction():
    times = np.linspace(-0.2, 0.8, 300)
    wave = -_bump(times, 0.25, 0.03, 1.0)          # effect shrinks with proportion
    onset = onset_50pct_peak(times, wave)          # default expected_sign='auto'
    assert np.isfinite(onset), (
        "a negative-going interaction yielded no onset under the default — the "
        "timing analysis appears to assume a modulation direction"
    )


def test_onset_default_is_sign_symmetric():
    """Flipping the waveform's sign must not move its onset or peak latency."""
    times = np.linspace(-0.2, 0.8, 300)
    wave = _bump(times, 0.25, 0.03, 1.0)
    assert onset_50pct_peak(times, wave) == pytest.approx(
        onset_50pct_peak(times, -wave))
    assert peak_latency(times, wave) == pytest.approx(peak_latency(times, -wave))


def test_jackknife_default_works_when_both_effects_invert():
    """Onset ordering is recovered when BOTH interactions run the other way."""
    df, times = _synthetic_timecourse_df(stab_onset=0.20, flex_onset=0.45, seed=5)
    df = df.copy()
    df['hg'] = df['hg'].apply(lambda a: -np.asarray(a, float))   # invert direction
    jk = jackknife_onset_difference(df, times=times)             # 'auto' signs
    assert jk['diff'] < 0          # stability still leads
    assert jk['p'] < 0.05


def test_peak_latency_after_onset():
    times = np.linspace(-0.2, 0.8, 300)
    wave = _bump(times, 0.25, 0.03, 1.0)
    assert peak_latency(times, wave) >= onset_50pct_peak(times, wave)


# ---------------------------------------------------------------------------
# time course + planted onset ordering
# ---------------------------------------------------------------------------
def test_interaction_time_course_recovers_planted_onsets():
    df, times = _synthetic_timecourse_df(stab_onset=0.20, flex_onset=0.45, seed=3)
    _, stab = interaction_time_course(df, 'stability', times=times)
    _, flex = interaction_time_course(df, 'flexibility', times=times)
    o_stab = onset_50pct_peak(times, stab)
    o_flex = onset_50pct_peak(times, flex)
    assert o_stab == pytest.approx(0.20, abs=0.05)
    assert o_flex == pytest.approx(0.45, abs=0.05)
    assert o_stab < o_flex                      # stability planted to lead


def test_interaction_time_course_requires_interaction_contrast():
    df, times = _synthetic_timecourse_df(seed=0)
    with pytest.raises(ValueError):
        interaction_time_course(df, 'stability', contrast_mode='condition',
                                times=times)


# ---------------------------------------------------------------------------
# jackknife (Ulrich–Miller)
# ---------------------------------------------------------------------------
def test_jackknife_detects_onset_difference_and_correction_identity():
    df, times = _synthetic_timecourse_df(stab_onset=0.20, flex_onset=0.45, seed=5)
    jk = jackknife_onset_difference(df, times=times)
    # stability leads -> negative (LWPC - LWPS) difference, significant
    assert jk['diff'] < 0
    assert jk['p'] < 0.05
    # the (N-1) correction identity: t_corrected == t_raw / (N-1)
    assert jk['t_corrected'] == pytest.approx(jk['t_raw'] / jk['df'], rel=1e-6)
    # CI excludes zero for a strong planted effect
    assert jk['ci'][0] < 0 and jk['ci'][1] < 0


def test_jackknife_requires_enough_electrodes():
    df, times = _synthetic_timecourse_df(n_subj=1, seed=0)
    keep = df[['subject', 'electrode']].drop_duplicates().head(2)
    df = df.merge(keep, on=['subject', 'electrode'])
    with pytest.raises(ValueError, match='electrodes'):
        jackknife_onset_difference(df, times=times)
