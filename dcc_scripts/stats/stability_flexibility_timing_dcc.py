#!/usr/bin/env python
"""
DCC core for A5 — relative onset of stability vs. flexibility information
(`docs/analysis_guide.md` §18).

Answers a *sequence* question the conjunction (A2) and cross-decoding (A4) layers
are silent on: does the **LWPC** (congruency x incongruent-proportion, stability)
interaction arise EARLIER in the trial than the **LWPS** (switchType x
switch-proportion, flexibility) interaction, or later?

Pipeline:
  1. Assemble the long-format single-trial table with `effect_measure='cluster'`,
     so `hg` holds each trial's HG TIME COURSE over the window rather than the
     window mean (`assemble_long_df`, reused from the A1/A2 job), plus the window
     time axis the courses live on (`window_times`).
  2. `sft.interaction_time_course` for each process: the equal-cell-weight
     difference-of-differences of the four (cond, mod) cell means per time bin,
     combined across electrodes. Equal cell weighting keeps the estimate
     orthogonal to both main effects, so the ~75/25 proportion imbalance cannot
     leak a main effect in as a fake interaction.
  3. `sft.onset_50pct_peak` (onset) and `sft.peak_latency` (shape cross-check) on
     each trace. Normalizing to each effect's OWN peak is what defeats the
     latency-amplitude confound -- a bigger effect crosses any *absolute*
     threshold sooner, so without it "earlier" would just rename "larger".
  4. `sft.jackknife_onset_difference`: onsets read off SMOOTH leave-one-subject-out
     grand averages, jackknife SE, and the Ulrich-Miller `(N-1)`-corrected paired
     t on the LWPC - LWPS difference. That signed difference with its CI is the
     headline.
  5. Figures: the two interaction time courses with their onset/peak markers, the
     leave-one-out onset pairs, and the distribution of leave-one-out differences.

Before anything else the job runs `sft._assert_amplitude_invariance` -- the
latency-amplitude guard as a live assertion (scaling a waveform by k must not move
its 50%-of-peak onset). A failure there invalidates every onset the job would go
on to report, so it is checked first and recorded in the summary.

On the SYNTHETIC path `sft._synthetic_timecourse_df` plants a known onset ordering
(stability rising before flexibility by default), so the whole path is validated
against ground truth in seconds with no data on disk. Set `SYNTHETIC_STAB_ONSET` >
`SYNTHETIC_FLEX_ONSET` to plant the reverse ordering and confirm the sign of the
reported difference flips -- the method reads the order you planted rather than a
fixed prejudice.

Driven by `run_stability_flexibility_timing_dcc.py` (wrapped by
`sbatch_stability_flexibility_timing_dcc.sh`). Not run directly on the cluster;
call `main(args)` with a populated argument namespace.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json

# ---------------------------------------------------------------------------
# PATH SETUP (mirrors the other dcc_scripts entrypoints)
# ---------------------------------------------------------------------------
try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if os.path.exists("/hpc/home"):
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

from src.analysis.stats import stability_flexibility_timing as sft

# NOTE: `general_utils` and the sibling DCC core pull in the mne / ieeg stack at
# import time, and the long-format assembly they provide is only reachable on the
# real-data path. They are therefore imported lazily inside `main` so the
# `DATA_SOURCE=synthetic` dry run validates this job's own logic anywhere the
# analysis modules import — including a laptop with no cluster environment.

# A5 is only defined for the 2x2 LWPC/LWPS interactions, and needs time courses.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cluster'


# ---------------------------------------------------------------------------
# the time axis the windowed courses live on
# ---------------------------------------------------------------------------
def window_times(subjects_epochs, tmin, tmax):
    """The [tmin, tmax] window's time axis, verified identical across subjects.

    `assemble_long_df(..., effect_measure='cluster')` stores each trial's windowed
    HG course but not the axis it was cut on, and A5 reports onsets in SECONDS, so
    we recover it here from the same `_window_indices` slice. Subjects whose window
    has a different length (or a different axis) can't be grand-averaged bin-by-bin
    at all, so that is a hard error rather than a silent truncation.
    """
    from dcc_scripts.stats.stability_flexibility_segregation_dcc import _window_indices

    axes = {}
    for sub, epochs in subjects_epochs.items():
        t = np.asarray(epochs.times, float)
        s_idx, e_idx = _window_indices(t, tmin, tmax)
        axes[sub] = t[s_idx:e_idx]

    lengths = {sub: len(a) for sub, a in axes.items()}
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            "subjects disagree on the number of samples in the analysis window "
            f"[{tmin}, {tmax}]s: {lengths}. A5 grand-averages the interaction "
            "bin-by-bin across electrodes, so every subject must share one time "
            "axis -- resample/crop the epochs to a common axis first.")

    ref_sub, ref = next(iter(axes.items()))
    for sub, a in axes.items():
        if not np.allclose(a, ref, atol=1e-9):
            raise RuntimeError(
                f"subject {sub!r} has the same number of window samples as "
                f"{ref_sub!r} but a different time axis (max deviation "
                f"{np.max(np.abs(a - ref)):.6g}s). Resample to a common axis first.")
    return ref


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    return o


def save_results(times, traces, onsets, jk, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # the interaction time courses themselves (the raw material of every onset)
    pd.DataFrame(dict(time=times,
                      lwpc=traces['stability'],
                      lwps=traces['flexibility'])).to_csv(
        os.path.join(save_dir, 'interaction_time_courses.csv'), index=False)

    # the N leave-one-out onset pairs behind the jackknife SE
    pd.DataFrame(dict(left_out_index=np.arange(len(jk['onset_lwpc_loo'])),
                      onset_lwpc=jk['onset_lwpc_loo'],
                      onset_lwps=jk['onset_lwps_loo'],
                      diff=jk['diffs'])).to_csv(
        os.path.join(save_dir, 'jackknife_leave_one_out.csv'), index=False)

    # the headline test (arrays stripped -- they are in the CSVs above)
    payload = {k: v for k, v in jk.items()
               if k not in ('onset_lwpc_loo', 'onset_lwps_loo', 'diffs')}
    payload.update(onsets)
    with open(os.path.join(save_dir, 'onset_difference.json'), 'w') as f:
        json.dump(_json_safe(payload), f, indent=2)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_plots(times, traces, onsets, jk, save_dir, statistic='mean'):
    ylab = ("interaction d-o-d (grand mean)" if statistic == 'mean'
            else "interaction d-o-d (t across electrodes)")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))

    # (1) the two interaction time courses with onset (50%-of-peak) + peak markers
    for key, name, colour, o_k, p_k in (
            ('stability', 'LWPC (stability)', '#2c7fb8', 'onset_lwpc', 'peak_lwpc'),
            ('flexibility', 'LWPS (flexibility)', '#d95f0e', 'onset_lwps', 'peak_lwps')):
        trace = np.asarray(traces[key], float)
        ax[0].plot(times, trace, color=colour, lw=2, label=name)
        onset, peak_t = onsets[o_k], onsets[p_k]
        finite = trace[np.isfinite(trace)]
        if finite.size:
            # mark the threshold in the trace's own (auto-resolved) direction
            peak = finite[np.argmax(np.abs(finite))]
            ax[0].axhline(0.5 * peak, color=colour, ls=':', lw=1)
            if np.isfinite(onset):
                ax[0].plot(onset, 0.5 * peak, 'o', color=colour, ms=9)
            if np.isfinite(peak_t):
                ax[0].axvline(peak_t, color=colour, ls='--', lw=0.8, alpha=0.6)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set(title="A5 · interaction time courses\n(dot = 50%-of-peak onset, "
                    "dashed = peak latency)",
              xlabel="time (s)", ylabel=ylab)
    ax[0].legend(fontsize=8)

    # (2) the leave-one-out onset pairs -- why the jackknife is stable
    x = np.arange(len(jk['onset_lwpc_loo']))
    ax[1].plot(x, jk['onset_lwpc_loo'], 'o-', color='#2c7fb8', label='LWPC onset')
    ax[1].plot(x, jk['onset_lwps_loo'], 'o-', color='#d95f0e', label='LWPS onset')
    ax[1].set(title="A5 · leave-one-subject-out onsets",
              xlabel="index of the left-out subject", ylabel="onset (s)")
    ax[1].legend(fontsize=8)

    # (3) the leave-one-out differences + the CI on the jackknife estimate
    d = np.asarray(jk['diffs'], float)
    d = d[np.isfinite(d)]
    if d.size:
        ax[2].hist(d, bins=max(5, min(20, d.size)), color="#bbb")
    ax[2].axvline(jk['diff'], color="#d7191c", lw=2,
                  label=f"mean diff = {jk['diff']:+.3f} s")
    ax[2].axvspan(jk['ci'][0], jk['ci'][1], color="#d7191c", alpha=0.12,
                  label=f"95% CI [{jk['ci'][0]:+.3f}, {jk['ci'][1]:+.3f}]")
    ax[2].axvline(0, color='k', lw=0.8, ls=':')
    ax[2].set(title=f"A5 · onset difference (LWPC - LWPS)\n"
                    f"t_corrected = {jk['t_corrected']:.2f}  p = {jk['p']:.4g}  "
                    f"df = {jk['df']}",
              xlabel="leave-one-out onset difference (s)", ylabel="# leave-outs")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'timing_summary.png'), dpi=140,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(times, onsets, jk, save_dir, meta, guard_onset, alpha=0.05):
    lead = ("stability (LWPC) leads" if jk['diff'] < 0 else
            "flexibility (LWPS) leads" if jk['diff'] > 0 else "no ordering")
    ci_excludes_zero = (jk['ci'][0] > 0) or (jk['ci'][1] < 0)
    verdict = ("RELIABLE ordering" if (jk['p'] < alpha and ci_excludes_zero)
               else "ordering NOT reliable")

    # Onset and peak latency should tell the SAME story before an ordering is
    # claimed -- unless an effect is still at its ceiling when the window ends, in
    # which case its "peak" is just the last bin and carries no latency
    # information. Distinguish those two cases rather than crying disagreement.
    times = np.asarray(times, float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    edge = times[-1] - dt
    at_edge = [k for k, v in (('LWPC', onsets['peak_lwpc']),
                              ('LWPS', onsets['peak_lwps']))
               if np.isfinite(v) and v >= edge]
    onset_sign = np.sign(jk['diff_full'])
    peak_sign = np.sign(jk['peak_diff'])
    agree = bool(np.isfinite(onset_sign) and np.isfinite(peak_sign)
                 and onset_sign == peak_sign and onset_sign != 0)
    peak_check = ("UNINFORMATIVE (peak at the window edge for "
                  + "/".join(at_edge) + " — the effect is still rising/plateaued "
                  "at WINDOW_TMAX)") if at_edge else str(agree)

    lines = [
        "=" * 70,
        "STABILITY vs FLEXIBILITY — A5 TIMING (relative onset)",
        "=" * 70,
    ]
    for k, v in meta.items():
        lines.append(f"{k:>22}: {v}")
    lines += [
        "-" * 70,
        f"latency-amplitude guard: PASSED "
        f"(50%-of-peak onset invariant under k-scaling; onset={guard_onset:.4f}s)",
        "-" * 70,
        "FULL-SAMPLE LATENCIES (descriptive):",
        f"      LWPC (stability)   onset = {onsets['onset_lwpc']:.4f} s   "
        f"peak = {onsets['peak_lwpc']:.4f} s",
        f"      LWPS (flexibility) onset = {onsets['onset_lwps']:.4f} s   "
        f"peak = {onsets['peak_lwps']:.4f} s",
        f"      onset difference (LWPC - LWPS) = {jk['diff_full']:+.4f} s",
        f"      peak  difference (LWPC - LWPS) = {jk['peak_diff']:+.4f} s",
        f"      onset and peak agree on the ordering: {peak_check}",
        "-" * 70,
        f"JACKKNIFE (Ulrich-Miller, N = {jk['n_subjects']} subjects):",
        f"      mean leave-one-out difference = {jk['diff']:+.4f} s",
        f"      jackknife SE                  = {jk['se']:.4f} s",
        f"      95% CI                        = [{jk['ci'][0]:+.4f}, {jk['ci'][1]:+.4f}] s",
        f"      t_raw = {jk['t_raw']:.3f}  ->  t_corrected = t_raw/(N-1) = "
        f"{jk['t_corrected']:.3f}   (df = {jk['df']})",
        f"      p = {jk['p']:.4g}   ->  {verdict}   [{lead}]",
        "=" * 70,
        "Reading: the SIGN of the onset difference says which process's information",
        "arises first (negative = stability leads); the CI / (N-1)-corrected t say",
        "whether that ordering is reliable. Onset is measured at 50% of each",
        "effect's OWN peak, so a pure amplitude difference cannot masquerade as an",
        "onset difference. Claim an ordering only when onset AND peak latency agree.",
    ]
    if at_edge:
        lines.append("WARNING: the peak-latency cross-check is uninformative — "
                     + "/".join(at_edge) + " peaks at the window")
        lines.append("         edge, so its 'peak' is just the last bin. Widen "
                     "WINDOW_TMAX until both")
        lines.append("         effects turn over before claiming an ordering.")
    elif not agree:
        lines.append("WARNING: onset and peak latency disagree on the ordering — the")
        lines.append("         waveform shapes differ (plateau vs transient); do not")
        lines.append("         claim an ordering on the onset alone.")
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    alpha = getattr(args, 'alpha', 0.05)
    statistic = getattr(args, 'statistic', 'mean')

    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE} | "
          f"statistic: {statistic}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 0. the latency-amplitude guard, as a live assertion --------------------------
    # If a k-scaled waveform moved its 50%-of-peak onset, every onset below would be
    # confounded with amplitude, so this runs BEFORE any data is touched.
    guard_onset = sft._assert_amplitude_invariance(k=3.7)
    print(f"latency-amplitude guard passed (onset invariant under k-scaling: "
          f"{guard_onset:.4f}s)")

    # 1. long-format TIME-COURSE table + the window time axis ----------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        df, times = sft._synthetic_timecourse_df(
            n_subj=getattr(args, 'synthetic_n_subj', 12),
            seed=getattr(args, 'seed', 0),
            stab_onset=getattr(args, 'synthetic_stab_onset', 0.20),
            flex_onset=getattr(args, 'synthetic_flex_onset', 0.40))
        print(f"synthetic df: {len(df)} rows | {df.subject.nunique()} subjects | "
              f"{df.electrode.nunique()} electrodes | {len(times)} time bins")
        print(f"planted onsets: LWPC={getattr(args, 'synthetic_stab_onset', 0.20)}s "
              f"LWPS={getattr(args, 'synthetic_flex_onset', 0.40)}s")
    else:
        print("DATA SOURCE: real epoched data")
        from src.analysis.utils.general_utils import (
            resolve_lab_root, resolve_electrodes_to_keep,
            load_HG_ev1_rescaled_per_subject)
        # reuse the SAME long-format assembly as the sibling A1/A2/A3 jobs
        from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df

        LAB_root = resolve_lab_root(args.LAB_root)
        print(f"LAB_root: {LAB_root}")
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)
        times = window_times(subjects_epochs, args.window_tmin, args.window_tmax)
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
        print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
              f"{df.electrode.nunique()} electrodes | {len(times)} time bins "
              f"({times[0]:.3f} -> {times[-1]:.3f}s)")
        for col in ('incongruent_proportion', 'switch_proportion'):
            if col not in df.columns or df[col].isna().all():
                raise RuntimeError(
                    f"df is missing usable '{col}' — A5 measures the onset of the "
                    "LWPC/LWPS 2x2 INTERACTIONS, which need the block-proportion "
                    "columns.")
        # The time-course table is far too large to serialize as CSV (one array per
        # trial per electrode); the per-bin interaction traces below are the
        # reusable artifact.

    n_subj = int(df['subject'].nunique())
    if n_subj < 3:
        raise RuntimeError(f"the jackknife needs >= 3 subjects; the assembled "
                           f"table has {n_subj}")

    # 2. interaction magnitude over time, per process ------------------------------
    print("A5: per-bin equal-cell-weight difference-of-differences, per process")
    traces = {}
    for key in ('stability', 'flexibility'):
        _, traces[key] = sft.interaction_time_course(
            df, key, contrast_mode=CONTRAST_MODE, times=times, statistic=statistic)

    # 3. onset (50%-of-peak) + peak latency on the full sample ---------------------
    onsets = dict(
        onset_lwpc=sft.onset_50pct_peak(times, traces['stability']),
        onset_lwps=sft.onset_50pct_peak(times, traces['flexibility']),
        peak_lwpc=sft.peak_latency(times, traces['stability']),
        peak_lwps=sft.peak_latency(times, traces['flexibility']))
    print(f"      LWPC onset={onsets['onset_lwpc']:.4f}s peak={onsets['peak_lwpc']:.4f}s | "
          f"LWPS onset={onsets['onset_lwps']:.4f}s peak={onsets['peak_lwps']:.4f}s")

    # 4. jackknifed onset-difference test ------------------------------------------
    print(f"A5: Ulrich-Miller jackknife over {n_subj} leave-one-subject-out "
          f"grand averages")
    jk = sft.jackknife_onset_difference(
        df, contrast_mode=CONTRAST_MODE, times=times, statistic=statistic)

    # 5. persist + plot + summarize ------------------------------------------------
    save_results(times, traces, onsets, jk, args.save_dir)
    make_plots(times, traces, onsets, jk, args.save_dir, statistic=statistic)
    write_summary(times, onsets, jk, args.save_dir, guard_onset=guard_onset, alpha=alpha,
                  meta=dict(
                      data_source=args.data_source, task=args.task,
                      epochs_root_file=getattr(args, 'epochs_root_file', None),
                      n_subjects=n_subj,
                      n_electrodes=int(df['electrode'].nunique()),
                      window=f"[{times[0]:.3f}, {times[-1]:.3f}]s "
                             f"({len(times)} bins)",
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      statistic=statistic, alpha=alpha, save_dir=args.save_dir))
    return dict(times=times, traces=traces, onsets=onsets, jackknife=jk)
