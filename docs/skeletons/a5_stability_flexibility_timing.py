"""A5 skeleton — relative onset of stability vs. flexibility information (plan §5).

DROP-IN TARGET
--------------
This IS the new module the plan names: `src/analysis/stats/
stability_flexibility_timing.py`. When implemented, move this file there (drop
the `a5_` prefix). It reuses the time-resolved interaction the segregation
module already computes via the `effect_measure='cluster'` path:

    from src.analysis.stats.stability_flexibility_segregation import (
        _canonical_labels, resolve_contrasts, finalize_contrasts,
        # the per-bin difference-of-differences t lives inside _interaction_cluster;
        # factor it out into a small helper you can call per time bin.
    )

WHY THIS EXISTS
---------------
A SEQUENCE claim — does stability information arise earlier than flexibility (or
vice versa)? — is an axis neither overlap (§2) nor decoding (§4) speaks to.
Theory says proactive/stability may be earlier (tonic/preparatory) and
reactive/flexibility later (phasic), but that's what we MEASURE, not assume.

The whole design defeats the LATENCY-AMPLITUDE CONFOUND: a bigger effect crosses
any absolute threshold sooner. `onset_50pct_peak` normalizes to each process's
OWN peak, so a pure multiplicative amplitude difference does NOT shift onset.
"""

from __future__ import annotations

import numpy as np


def interaction_time_course(df, key, contrast_mode="proportion", contrasts=None):
    """Interaction magnitude as a function of time for one process.

    key='stability' -> LWPC (congruency × incongruent_proportion) difference-of-
    differences over time; key='flexibility' -> LWPS. Grand-averaged over the
    relevant electrodes (or a t-statistic time course).

    Parameters
    ----------
    df : long table where 'hg' holds the per-trial TIME COURSE over the window
        (the effect_measure='cluster' input shape), not a scalar.

    Returns
    -------
    (times, effect) : effect is the signed d-o-d per time bin.

    Steps: attach cell labels via _canonical_labels; for each time bin compute
    the equal-cell-weight difference-of-differences of the four cell means
    (reuse the per-bin logic from `_interaction_cluster`); average across
    electrodes.
    """
    raise NotImplementedError("A5: time-resolved interaction (d-o-d over time)")


def onset_50pct_peak(times, effect, expected_sign=+1):
    """First upward crossing of 50% of the peak, on the rising flank = onset.

    Steps
    -----
    1. Take the peak in the expected direction within the window (or of |effect|):
       peak = max(effect * expected_sign).
    2. threshold = 0.5 * peak.
    3. Walk from the start; return the time of the FIRST sample where the rising
       signal crosses `threshold` upward (linear-interpolate between samples for
       sub-sample resolution).

    WHY 50%-OF-PEAK: if stab(t) = k*flex(t), both cross 50%-of-peak at the SAME
    time -> a pure amplitude difference cannot fake an onset difference. This is
    the confound guard; do NOT additionally amplitude-match.

    Write a unit test: scale a waveform by k and assert the onset is unchanged.
    """
    raise NotImplementedError("A5: 50%-of-peak onset (with the k-scaling unit test)")


def peak_latency(times, effect, expected_sign=+1):
    """Time of the peak — reported ALONGSIDE onset as a shape cross-check.

    A broad plateau vs a sharp transient can move the 50%-of-peak point (a real
    dynamics difference, not an artifact); a consistent onset+peak story is the
    robust claim.
    """
    raise NotImplementedError("A5: peak latency")


def jackknife_onset_difference(df_by_subject, expected_signs=(+1, +1)):
    """Ulrich–Miller jackknifed comparison of LWPC vs LWPS onset.

    Single-subject onsets are too noisy, so measure onset on SMOOTH
    leave-one-subject-out grand-averages instead.

    Parameters
    ----------
    df_by_subject : the long time-course table, groupable by subject.
    expected_signs : (stability_sign, flexibility_sign).

    Returns
    -------
    dict:
        onset_lwpc, onset_lwps : full-sample onsets (descriptive)
        diff : onset_lwpc - onset_lwps (jackknife mean of pseudovalues)
        se   : jackknife SE
        t_corrected : Ulrich–Miller corrected paired t
        p, ci : two-sided p and CI on the difference

    Implementation steps
    ---------------------
    1. For i in subjects: build the leave-one-out grand average; compute
       onset_lwpc[i], onset_lwps[i] via interaction_time_course + onset_50pct_peak;
       d_i = onset_lwpc[i] - onset_lwps[i].  (N leave-one-out values.)
    2. Jackknife SE of the difference:
           d_bar = mean(d_i)
           se = sqrt((N-1)/N * sum((d_i - d_bar)**2))
    3. Ulrich–Miller correction: run an ordinary paired t on the N d_i, then
       t_corrected = t_raw / (N - 1)   (the raw jackknife t is inflated by N-1).
    4. Report the paired onset DIFFERENCE with a CI, not just a p.

    Cross-checks (optional): bootstrap over subjects/electrodes; also report
    peak-latency difference.
    """
    raise NotImplementedError("A5: Ulrich–Miller jackknife of the onset difference")
