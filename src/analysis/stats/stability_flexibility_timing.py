"""A5 — relative onset of stability vs. flexibility information (plan §5).

A *sequence* claim — does stability (LWPC / proactive) information arise EARLIER
than flexibility (LWPS / reactive), or vice versa? — is an axis that neither the
overlap/conjunction test (§2) nor the cross-decoding (§4) speaks to. Theory casts
proactive/stability control as sustained/preparatory (possibly earlier) and
reactive/flexibility control as phasic (possibly later), but that is exactly what
we MEASURE here rather than assume.

What this module provides
-------------------------
- ``interaction_time_course`` — the interaction magnitude as a function of time
  for one process (LWPC or LWPS): the equal-cell-weight difference-of-differences
  of the four cell means, per time bin, grand-averaged over electrodes (or a
  t-statistic across electrodes). This reuses the balanced-interaction machinery
  the segregation module already uses for scalar effects, lifted to run per bin.
- ``onset_50pct_peak`` — onset latency = the first upward crossing of 50 % of the
  effect's own peak, on the rising flank (linear-interpolated for sub-sample
  resolution).
- ``peak_latency`` — the time of the peak, reported ALONGSIDE onset as a
  waveform-shape cross-check.
- ``jackknife_onset_difference`` — the Ulrich–Miller jackknifed comparison of the
  LWPC vs LWPS onset: onsets measured on smooth leave-one-subject-out
  grand-averages, jackknife SE, and the ``(N-1)``-corrected paired t.

The latency–amplitude confound and the guard
---------------------------------------------
A bigger effect crosses any *absolute* threshold sooner, so a naive onset would
make "earlier" mean nothing more than "larger". ``onset_50pct_peak`` normalizes
to each process's OWN peak: if ``stab(t) = k · flex(t)`` for any ``k > 0``, both
cross 50 %-of-peak at the SAME sample, so a pure multiplicative amplitude
difference cannot fake an onset difference. This is baked into a unit test
(``_assert_amplitude_invariance`` below, and ``tests/analysis/stats``).

Drop-in usage
-------------
``df`` is the same long table the segregation module consumes with
``effect_measure='cluster'`` — one row per (electrode, trial), except ``hg`` is
the per-trial TIME COURSE over the window (a 1-D array) rather than a scalar::

    from src.analysis.stats.stability_flexibility_timing import (
        interaction_time_course, onset_50pct_peak, jackknife_onset_difference)

    times, stab = interaction_time_course(df, 'stability',  times=window_times)
    _,     flex = interaction_time_course(df, 'flexibility', times=window_times)
    onset_stab = onset_50pct_peak(times, stab)
    onset_flex = onset_50pct_peak(times, flex)
    jk = jackknife_onset_difference(df, times=window_times)   # LWPC - LWPS onset, CI, t_c

``_synthetic_timecourse_df`` plants a genuine onset structure (stability rising
earlier than flexibility) so the whole path and the tutorial run with no data on
disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist

from src.analysis.stats.stability_flexibility_segregation import (
    resolve_contrasts, finalize_contrasts, _canonical_labels, _is_interaction,
    _dod_cells,
)

# which (cond, mod) sub-factor labels _canonical_labels attaches for each process
_FACTOR_COLS = {
    'stability':   ('_scond', '_smod'),   # congruency × incongruent_proportion (LWPC)
    'flexibility': ('_fcond', '_fmod'),   # switchType  × switch_proportion    (LWPS)
}


# ----------------------------------------------------------------------------
# per-bin difference-of-differences
# ----------------------------------------------------------------------------
def _dod_over_time(cells):
    """Per-time-bin equal-cell-weight difference-of-differences of the four cell
    means. ``cells`` maps (cond, mod) in {1.,0.}×{1.,0.} to a (n_trials, n_time)
    array (from ``_dod_cells``). Returns a (n_time,) signed d-o-d vector:

        [ mean(cond=1,mod=1) - mean(cond=0,mod=1) ]        # cond effect, HIGH block
      - [ mean(cond=1,mod=0) - mean(cond=0,mod=0) ]        # cond effect, LOW  block

    i.e. how much the condition effect (e.g. congruency) CHANGES between the high-
    and low-proportion blocks, as a function of time — signed, so it captures the
    modulation whether the condition effect grows or shrinks (the direction is not
    assumed). Equal weighting of the four cells makes it orthogonal to both main
    effects (the same balance the scalar ``_interaction_cohens_d`` enforces), so a
    pure main effect can't leak in."""
    means = {k: np.asarray(v, float).mean(axis=0) for k, v in cells.items()}
    return ((means[(1.0, 1.0)] - means[(0.0, 1.0)])
            - (means[(1.0, 0.0)] - means[(0.0, 0.0)]))


def _electrode_dod_timecourses(work, key):
    """One d-o-d(t) vector per electrode for the given process. Electrodes with
    any empty (cond, mod) cell are skipped (they can't form a balanced 2x2)."""
    condcol, modcol = _FACTOR_COLS[key]
    traces = []
    for (_subj, _elec), g in work.groupby(['subject', 'electrode']):
        hg = np.vstack([np.asarray(v, float) for v in g['hg'].to_numpy()])
        cells = _dod_cells(hg, g[condcol].to_numpy(), g[modcol].to_numpy())
        if cells is None:
            continue
        traces.append(_dod_over_time(cells))
    return traces


def interaction_time_course(df, key, contrast_mode='proportion', contrasts=None,
                            times=None, statistic='mean'):
    """Interaction magnitude as a function of time for one process.

    Parameters
    ----------
    df : long table; ``hg`` holds the per-trial TIME COURSE over the window (a
        1-D array per row — the ``effect_measure='cluster'`` input shape), not a
        scalar.
    key : 'stability' (LWPC, congruency × incongruent_proportion) or
        'flexibility' (LWPS, switchType × switch_proportion).
    contrast_mode / contrasts : as in the segregation module. Must resolve to an
        INTERACTION contrast (the proportion preset) — a time-resolved onset only
        makes sense for the 2x2 LWPC/LWPS interactions.
    times : the time axis (length n_time). Defaults to ``np.arange(n_time)`` (bin
        indices) when not supplied.
    statistic : 'mean' — grand-average the per-electrode d-o-d(t) over electrodes
        (the information time course); 't' — a t-statistic across electrodes
        (mean / SEM), which is noise-normalized and often has a cleaner flank.

    Returns
    -------
    (times, effect) : ``effect`` is the signed interaction d-o-d per time bin,
        (n_time,). NaNs if no electrode yields a valid 2x2.

    Steps: attach the cell labels via ``_canonical_labels``; for each electrode
    form the four (cond, mod) cells and take the per-bin difference-of-differences
    of their means (``_dod_over_time``); combine across electrodes per
    ``statistic``.
    """
    contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
    if not _is_interaction(contrasts[key]):
        raise ValueError(f"interaction_time_course needs an INTERACTION contrast "
                         f"for {key!r} (use contrast_mode='proportion'); the "
                         f"resolved contrast is simple.")
    work = _canonical_labels(df, contrasts)
    traces = _electrode_dod_timecourses(work, key)
    if not traces:
        n_time = len(np.atleast_1d(df['hg'].iloc[0]))
        eff = np.full(n_time, np.nan)
        return (np.arange(n_time) if times is None else np.asarray(times, float)), eff

    mat = np.vstack(traces)                       # (n_elec, n_time)
    n_time = mat.shape[1]
    if times is None:
        times = np.arange(n_time)
    times = np.asarray(times, float)

    if statistic == 'mean':
        effect = np.nanmean(mat, axis=0)
    elif statistic == 't':
        mean = np.nanmean(mat, axis=0)
        n = mat.shape[0]
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(n_time, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            effect = np.where(sem > 0, mean / sem, 0.0)
    else:
        raise ValueError(f"statistic must be 'mean' or 't'; got {statistic!r}")
    return times, effect


# ----------------------------------------------------------------------------
# onset (50%-of-peak) and peak latency
# ----------------------------------------------------------------------------
def _resolve_sign(effect, expected_sign):
    """Resolve the orientation applied to an interaction time course.

    ``expected_sign='auto'`` (the default) means DO NOT assume a direction: orient
    the waveform by its own dominant deflection (the sign of the finite sample
    with the largest magnitude), so the onset of the interaction is read off
    whichever way it goes. In behavior the congruency effect SHRINKS in high-
    incongruent-proportion blocks (a negative-going d-o-d) while another neural
    population may grow it (positive-going); the sign is a property of the data,
    not something to fix in advance. Pass an explicit ``+1``/``-1`` only to force a
    particular direction (e.g. a hypothesis test on a pre-registered sign)."""
    if expected_sign == 'auto':
        s = np.asarray(effect, float)
        finite = np.isfinite(s)
        if not finite.any():
            return 1.0
        peak = s[finite][np.argmax(np.abs(s[finite]))]
        return -1.0 if peak < 0 else 1.0
    return expected_sign


def onset_50pct_peak(times, effect, expected_sign='auto'):
    """First upward crossing of 50 % of the peak, on the rising flank = onset.

    Steps
    -----
    1. Orient the waveform (``signal = effect * sign``) and take its peak within
       the window. By default (``expected_sign='auto'``) the sign is taken from
       the waveform's OWN dominant deflection, so the direction of the block-
       proportion modulation is NOT assumed — an interaction that grows or shrinks
       the condition effect is treated symmetrically. Pass an explicit ``+1``/``-1``
       to force a direction.
    2. ``threshold = 0.5 * peak``.
    3. Walk from the window start toward the peak; return the time of the FIRST
       sample where the rising signal crosses ``threshold`` upward, linearly
       interpolating between the bracketing samples for sub-sample resolution.

    Why 50 %-of-peak: if ``stab(t) = k · flex(t)`` the threshold scales with the
    peak by the same ``k``, so both waveforms cross at the identical time — a
    pure amplitude difference cannot fake an onset difference. Do NOT additionally
    amplitude-match; that only costs power without adding protection.

    Returns the onset time in the units of ``times``; NaN if the oriented peak is
    not positive (with an explicit ``expected_sign``, no effect in that direction)
    or no upward crossing exists before the peak.
    """
    times = np.asarray(times, float)
    signal = np.asarray(effect, float) * _resolve_sign(effect, expected_sign)
    finite = np.isfinite(signal)
    if finite.sum() < 2:
        return np.nan
    peak_idx = int(np.nanargmax(np.where(finite, signal, -np.inf)))
    peak = signal[peak_idx]
    if not np.isfinite(peak) or peak <= 0:
        return np.nan                         # no effect in the expected direction
    threshold = 0.5 * peak

    # first upward crossing at or before the peak
    for i in range(1, peak_idx + 1):
        lo, hi = signal[i - 1], signal[i]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo < threshold <= hi:              # rising through the threshold
            frac = (threshold - lo) / (hi - lo) if hi != lo else 0.0
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    # already above threshold at the first finite sample -> onset at window start
    if signal[np.argmax(finite)] >= threshold:
        return float(times[np.argmax(finite)])
    return np.nan


def peak_latency(times, effect, expected_sign='auto'):
    """Time of the peak in the effect's direction — reported ALONGSIDE onset as a
    shape cross-check. A broad plateau vs. a sharp transient can shift the
    50 %-of-peak point (a real dynamics difference, not an artifact), so a
    consistent onset+peak story is the robust claim. By default the direction is
    taken from the waveform itself (``expected_sign='auto'``, no assumed sign);
    pass ``+1``/``-1`` to force one. NaN if no finite sample."""
    times = np.asarray(times, float)
    signal = np.asarray(effect, float) * _resolve_sign(effect, expected_sign)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.nan
    peak_idx = int(np.nanargmax(np.where(finite, signal, -np.inf)))
    return float(times[peak_idx])


# ----------------------------------------------------------------------------
# jackknife (Ulrich–Miller) onset-difference test
# ----------------------------------------------------------------------------
def _onsets_from_df(sub_df, times, statistic, contrast_mode, contrasts,
                    expected_signs):
    """(onset_lwpc, onset_lwps, peak_lwpc, peak_lwps) for a slice of the table."""
    t, stab = interaction_time_course(sub_df, 'stability', contrast_mode,
                                      contrasts, times=times, statistic=statistic)
    _, flex = interaction_time_course(sub_df, 'flexibility', contrast_mode,
                                      contrasts, times=times, statistic=statistic)
    s_sign, f_sign = expected_signs
    return (onset_50pct_peak(t, stab, s_sign), onset_50pct_peak(t, flex, f_sign),
            peak_latency(t, stab, s_sign), peak_latency(t, flex, f_sign))


def jackknife_onset_difference(df_by_subject, expected_signs=('auto', 'auto'),
                               contrast_mode='proportion', contrasts=None,
                               times=None, statistic='mean'):
    """Ulrich–Miller jackknifed comparison of the LWPC vs LWPS onset.

    Single-subject latency estimates are too noisy, so onset is measured on
    SMOOTH leave-one-subject-out grand-averages, and the jackknife turns the N
    leave-one-out values into a corrected paired test.

    Parameters
    ----------
    df_by_subject : the long time-course table, groupable by ``subject``.
    expected_signs : (stability_sign, flexibility_sign) passed to
        ``onset_50pct_peak`` for each process. Default ``('auto', 'auto')`` — the
        onset direction of each interaction is read from its own waveform, so no
        sign is assumed for either process.
    times, statistic, contrast_mode, contrasts : threaded to
        ``interaction_time_course``.

    Returns
    -------
    dict with
        onset_lwpc, onset_lwps : full-sample onsets (descriptive)
        peak_lwpc,  peak_lwps  : full-sample peak latencies (shape cross-check)
        diff        : mean of the leave-one-out onset differences (LWPC - LWPS)
        diff_full   : the full-sample onset difference
        se          : jackknife SE of the difference
        t_corrected : the (N-1)-corrected paired t
        t_raw       : the ordinary (uncorrected) paired t on the N differences
        df          : degrees of freedom (N-1)
        p           : two-sided p from ``t_corrected``
        ci          : 95 % CI on the difference
        n_subjects  : N
        onset_lwpc_loo, onset_lwps_loo, diffs : the per-leaveout arrays
        peak_diff   : full-sample peak-latency difference (LWPC - LWPS)

    Implementation
    --------------
    1. For each subject i, drop subject i, build the leave-one-out grand-average
       time course for LWPC and LWPS, and read off each onset -> d_i.
    2. Jackknife SE of the difference:
           d_bar = mean(d_i);  se = sqrt((N-1)/N * sum((d_i - d_bar)**2)).
    3. Ulrich–Miller correction: run an ordinary paired t on the N d_i, then
           t_corrected = t_raw / (N - 1)
       (equivalently t_corrected = d_bar / se). The raw jackknife t is inflated
       by (N-1); the division removes it.
    """
    subjects = list(pd.unique(df_by_subject['subject']))
    N = len(subjects)
    if N < 3:
        raise ValueError(f"jackknife needs >= 3 subjects; got {N}")

    onset_lwpc = np.empty(N)
    onset_lwps = np.empty(N)
    for i, s in enumerate(subjects):
        loo = df_by_subject[df_by_subject['subject'] != s]
        o_pc, o_ps, _, _ = _onsets_from_df(loo, times, statistic, contrast_mode,
                                           contrasts, expected_signs)
        onset_lwpc[i], onset_lwps[i] = o_pc, o_ps

    diffs = onset_lwpc - onset_lwps
    valid = np.isfinite(diffs)
    if valid.sum() < 3:
        raise ValueError("too few leave-one-out subaverages produced a finite "
                         "onset difference; check the effect direction / window")
    d = diffs[valid]
    n = d.size
    d_bar = float(d.mean())

    # jackknife SE (variance inflated by (N-1)); Ulrich–Miller (N-1) correction
    se = float(np.sqrt((n - 1) / n * np.sum((d - d_bar) ** 2)))
    sd = d.std(ddof=1)
    t_raw = d_bar / (sd / np.sqrt(n)) if sd > 0 else np.nan
    t_corrected = t_raw / (n - 1) if np.isfinite(t_raw) else np.nan
    dfree = n - 1
    p = float(2 * _t_dist.sf(abs(t_corrected), dfree)) if np.isfinite(t_corrected) else np.nan
    tcrit = _t_dist.ppf(0.975, dfree)
    ci = (d_bar - tcrit * se, d_bar + tcrit * se)

    # full-sample descriptives (all subjects)
    o_pc_full, o_ps_full, pk_pc_full, pk_ps_full = _onsets_from_df(
        df_by_subject, times, statistic, contrast_mode, contrasts, expected_signs)

    return dict(
        onset_lwpc=o_pc_full, onset_lwps=o_ps_full,
        peak_lwpc=pk_pc_full, peak_lwps=pk_ps_full,
        diff=d_bar, diff_full=(o_pc_full - o_ps_full),
        se=se, t_corrected=t_corrected, t_raw=t_raw, df=dfree,
        p=p, ci=ci, n_subjects=N,
        onset_lwpc_loo=onset_lwpc, onset_lwps_loo=onset_lwps, diffs=diffs,
        peak_diff=(pk_pc_full - pk_ps_full),
    )


# ----------------------------------------------------------------------------
# synthetic ground truth — planted onset structure, runs with no data on disk
# ----------------------------------------------------------------------------
def _bump(times, onset, width, amp):
    """A smooth positive bump (raised-cosine rise into a plateau/decay) whose
    50 %-of-peak crossing sits near ``onset``. Used to plant a known onset."""
    t = np.asarray(times, float)
    # logistic rise centred at `onset` with scale `width`, so f(onset)=0.5*amp
    return amp / (1.0 + np.exp(-(t - onset) / width))


def _synthetic_timecourse_df(n_subj=12, seed=0, n_time=40, t0=-0.2, t1=0.8,
                             stab_onset=0.20, flex_onset=0.40, width=0.03,
                             amp=1.0, noise=1.0):
    """Long time-course table with a planted onset difference.

    The LWPC (stability) interaction rises at ``stab_onset`` and the LWPS
    (flexibility) interaction at ``flex_onset`` (later by default), each entered
    as a PURE 2x2 interaction: the temporal bump is added only to the ``(cond=+,
    mod=high)`` cell, so the equal-cell-weight difference-of-differences recovers
    exactly the bump. Per-electrode gain and additive noise make single-subject
    onsets noisy — which is what the jackknife smooths. Returns a df with the
    ``effect_measure='cluster'`` shape (``hg`` = per-trial time course) plus the
    ``times`` axis it was built on.
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(t0, t1, n_time)
    stab_profile = _bump(times, stab_onset, width, amp)
    flex_profile = _bump(times, flex_onset, width, amp)
    frames = []
    for s in range(n_subj):
        n_tr = int(rng.integers(200, 400))
        cong = rng.choice(['c', 'i'], n_tr)
        sw = rng.choice(['s', 'r'], n_tr)
        inc_prop = rng.choice([25.0, 75.0], n_tr)
        sw_prop = rng.choice([25.0, 75.0], n_tr)
        stab_hi = (cong == 'i') & (inc_prop == 75.0)      # the ++ LWPC cell
        flex_hi = (sw == 's') & (sw_prop == 75.0)          # the ++ LWPS cell
        for e in range(int(rng.integers(15, 30))):
            gain = rng.lognormal(0, 0.4)                    # per-electrode SNR
            tc = rng.normal(0, noise, (n_tr, n_time)) * gain
            tc[stab_hi] += gain * stab_profile[None, :]
            tc[flex_hi] += gain * flex_profile[None, :]
            col = np.empty(n_tr, dtype=object)
            for i in range(n_tr):
                col[i] = tc[i]
            frames.append(pd.DataFrame(dict(
                subject=s, electrode=f'{s}_{e}',
                congruency=cong, switchType=sw,
                incongruent_proportion=inc_prop, switch_proportion=sw_prop,
                hg=col)))
    return pd.concat(frames, ignore_index=True), times


# ----------------------------------------------------------------------------
# baked-in amplitude-invariance check (the latency–amplitude confound guard)
# ----------------------------------------------------------------------------
def _assert_amplitude_invariance(seed=0, k=3.7):
    """The confound guard as an assertion: scaling a waveform by k must NOT move
    its 50%-of-peak onset. Take one waveform, scale it, confirm equal onset."""
    times = np.linspace(-0.2, 0.8, 200)
    wave = _bump(times, 0.25, 0.03, 1.0) + 0.0 * times
    o1 = onset_50pct_peak(times, wave)
    o2 = onset_50pct_peak(times, k * wave)
    assert np.isclose(o1, o2, atol=1e-9), \
        f"onset moved under pure amplitude scaling: {o1} vs {o2}"
    return o1


if __name__ == '__main__':
    # 1) the latency–amplitude guard: onset invariant to a pure k-scaling
    onset = _assert_amplitude_invariance()
    print(f"[guard] 50%-of-peak onset invariant under k-scaling: onset={onset:.4f}s  OK")

    # 2) planted onset structure: stability rises before flexibility
    df, times = _synthetic_timecourse_df(stab_onset=0.20, flex_onset=0.40, seed=1)
    t, stab = interaction_time_course(df, 'stability', times=times)
    _, flex = interaction_time_course(df, 'flexibility', times=times)
    o_stab = onset_50pct_peak(t, stab)
    o_flex = onset_50pct_peak(t, flex)
    print(f"[time course] LWPC onset={o_stab:.3f}s (peak {peak_latency(t, stab):.3f}s), "
          f"LWPS onset={o_flex:.3f}s (peak {peak_latency(t, flex):.3f}s)")
    print(f"              planted stab_onset=0.20s < flex_onset=0.40s; "
          f"recovered LWPC {'<' if o_stab < o_flex else '>='} LWPS")

    # 3) Ulrich–Miller jackknife of the onset difference
    jk = jackknife_onset_difference(df, times=times)
    print(f"[jackknife] LWPC-LWPS onset diff={jk['diff']:.3f}s "
          f"(95% CI [{jk['ci'][0]:.3f}, {jk['ci'][1]:.3f}]s), "
          f"t_c={jk['t_corrected']:.2f}, p={jk['p']:.4g}, N={jk['n_subjects']}")
    print(f"            t_raw={jk['t_raw']:.2f} inflated by (N-1)={jk['df']}; "
          f"t_raw/(N-1)={jk['t_raw']/jk['df']:.2f} == t_c  OK")
