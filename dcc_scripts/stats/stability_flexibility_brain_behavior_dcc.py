#!/usr/bin/env python
"""
DCC core for A6 — brain-behavior correlation
(`docs/analysis_guide.md` §19).

Ties the A1 neural selectivity to the ACTUAL behavioral control adjustment, so the
substrates are shown to be *functional* rather than incidental. Three levels, all
run here:

  (1) ACROSS PARTICIPANTS, CONTINUOUS SCORES (the across-participant result to
      report): each participant's mean signed per-electrode LWPC / LWPS d against
      its behavioral LWPC / LWPS from the same trials, raw and with the RT-linked
      part of HG removed, with split-half reliabilities (one trial split per
      participant), the reliability ceiling, a joint-regression specificity test
      and a disjoint-half check.
      -> `sbb.participant_scores`, `sbb.participant_brain_behavior`
  (2) ACROSS SUBJECTS, LABEL-BASED (kept for comparison): does a subject with
      more/stronger LWPC electrodes show a larger behavioral LWPC (congruency x
      incongruent-proportion) RT effect, and likewise LWPS?
      -> `sbb.subject_level_brain_behavior`
  (3) WITHIN SUBJECT, SINGLE TRIAL: does trial-by-trial HG in the LWPC electrode
      group predict the trial-by-trial congruency adjustment (LWPS group <-> switch
      adjustment), via a mixed model with a subject random intercept?
      -> `sbb.trialwise_brain_behavior`

The MATCHED pairing should beat the CROSS pairing (LWPC group <-> switch
adjustment, and vice versa), so every level reports the cross pairing alongside the
matched one. RT coupling produces that pattern by itself (see
`sbb.rt_adjust_hg`), which is why level (1) also reports RT-adjusted scores.

Pipeline:
  1. Assemble the same window-mean long table as the A1/A2/A3 jobs
     (`assemble_long_df`, contrast_mode='proportion'; it carries `trial`, `rt`
     and `acc`) and run the A1 electrode definition
     (`sfs.per_electrode_anova_labels`) -> per-electrode S/F flags. The electrode
     set is ROIS x ELECTRODES (task-significant lPFC by default in the submit
     script).
  2. Behavior: per-subject LWPC/LWPS RT magnitudes read from the subject-level
     effects table (BEHAVIOR_CSV, by default
     `src/config/ieeg_behavioral_subject_level_effects.csv`: the `LWPC_effect` /
     `LWPS_effect` of its `key_RT_mean` rows) via `sbb.load_subject_level_behavior`,
     matched to the epochs' subject IDs on their stem. Levels (1) and (2) correlate
     against these. The same contrast scored on the long table's own trials
     (`sbb.behavioral_lwpc_lwps_magnitudes`) is only a cross-check and the
     estimate of the behavioral reliability behind the level-(1) ceiling.
  3. Per-participant continuous scores and their correlations with behavior.
  4. Across-subject correlations for each label-based neural summary ('count',
     'frac', and the mean interaction F), each with its cross-pairing control.
  5. Single-trial table (`assemble_trial_table`): per (subject, trial) RT plus the
     HG averaged over the LWPC and LWPS electrode groups, then the mixed models.

The trial-level adjustment columns
----------------------------------
`sbb.trialwise_brain_behavior` deliberately takes the behavioral adjustment columns
as INPUT -- it does not define them, because the operationalization is a design
choice. This job defines them as each trial's SIGNED CONTRIBUTION to the very
difference-of-differences the rest of the battery is built on (`_adjustment_weight`
/ `add_adjustment_columns`):

    adj_congruency(t) = w(t) * (RT_t - mean RT of that subject)
    w(t) = +1 for (i, low-incongruent) and (c, high-incongruent)
           -1 for (c, low-incongruent) and (i, high-incongruent)

Those are exactly the four cell weights of the LWPC d-o-d, so a trial's `adj` is
large when its RT pushes the interaction in its own direction, and the mean of
`adj` over a subject's trials is that subject's (trial-count-weighted) LWPC/4. A
positive mixed-model slope therefore means "trials with more HG in this electrode
group push the behavioral interaction harder" -- the functional claim A6 makes.
`adj_switch` is the same construction on switchType x switch_proportion.

RT and the group HG are both CENTERED WITHIN SUBJECT, so the slope is a purely
within-subject effect: with an uncentered predictor a between-subject difference in
mean HG would leak into the common slope, which is precisely the confound the
"within subject" framing is supposed to exclude.

On the SYNTHETIC path `sbb._synthetic_brain_behavior` plants a matched
across-subject correlation AND a matched within-subject coupling, each stronger
than its cross control, and `sbb._synthetic_long_df` plants a single-trial table
for level (1) with a chosen brain-behavior link and RT coupling, so the whole path
is validated against ground truth in seconds with no data on disk.

Driven by `run_stability_flexibility_brain_behavior_dcc.py` (wrapped by
`sbatch_stability_flexibility_brain_behavior_dcc.sh`). Not run directly on the
cluster; call `main(args)` with a populated argument namespace.
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

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats import stability_flexibility_brain_behavior as sbb

# NOTE: `general_utils` and the sibling DCC core pull in the mne / ieeg stack at
# import time and are only reachable on the real-data path, so they are imported
# lazily inside `main` -- the `DATA_SOURCE=synthetic` dry run then validates this
# job's own logic anywhere the analysis modules import.

# A6 sits on the A1 electrodes: LWPC/LWPS interactions on window-mean HG.
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
EFFECT_MEASURE = 'cohens_d'

STAB, FLEX = "#2c7fb8", "#d95f0e"

# the neural summaries correlated against behavior, and the label columns each needs
_NEURAL_MODES = (('count', 'n_S / n_F  (# selective electrodes)'),
                 ('frac', 'frac_S / frac_F  (proportion of the subject\'s electrodes)'),
                 ('effect', 'mean_stab / mean_flex  (mean interaction F)'))


# ---------------------------------------------------------------------------
# behavior
# ---------------------------------------------------------------------------
def behavior_from_long_df(df):
    """Per-subject behavioral LWPC/LWPS RT magnitudes from the long table's own
    trials, to cross-check the subject-level table.

    The epochs metadata carry RT, accuracy and the block proportions (parsed from
    the event names), so this scores exactly the trials the HG scores use, with no
    blockType map in the way."""
    if 'rt' not in df.columns or not np.isfinite(
            pd.to_numeric(df['rt'], errors='coerce')).any():
        raise KeyError("the long table has no reaction times; rebuild it with the "
                       "current `assemble_long_df`, which reads them from the "
                       "epochs metadata")
    trials = df.drop_duplicates(['subject', 'trial'])
    if 'acc' in trials.columns and trials['acc'].isna().all():
        trials = trials.drop(columns='acc')        # no accuracy recorded: keep all
    return sbb.behavioral_lwpc_lwps_magnitudes(trials, rt_col='rt')


def csv_behavior_agreement(csv_behavior, behavior):
    """Across-participant agreement of the subject-level table and the behavior
    scored on the long table's trials (both RT).

    Both score the same task sessions, so each participant's LWPC / LWPS should
    agree closely. The long table holds only the correct trials that survived
    preprocessing, so expect close, not identical, values; a sign flip or a poor
    agreement means the two do not measure the same contrast."""
    a = csv_behavior.assign(stem=csv_behavior['subject'].map(sbb.subject_stem))
    b = behavior.assign(stem=behavior['subject'].map(sbb.subject_stem))
    m = a.merge(b, on='stem', suffixes=('_csv', '_epochs')).dropna(
        subset=['lwpc_csv', 'lwpc_epochs', 'lwps_csv', 'lwps_epochs'])
    out = dict(n_participants=len(m))
    for eff in ('lwpc', 'lwps'):
        x, y = m[f'{eff}_csv'], m[f'{eff}_epochs']
        out[f'r_{eff}'] = (float(np.corrcoef(x, y)[0, 1])
                           if len(m) >= 3 and x.std() > 0 and y.std() > 0 else np.nan)
        out[f'mean_abs_diff_{eff}_ms'] = float(np.mean(np.abs(x - y))) if len(m) else np.nan
    return out


# ---------------------------------------------------------------------------
# single-trial table (real data): RT + per-trial HG for each electrode group
# ---------------------------------------------------------------------------
def assemble_trial_table(subjects_epochs, tmin, tmax, electrodes_to_keep=None):
    """Per-(subject, trial) behavior + the per-channel window-mean HG behind it.

    Returns ``(trials, hg_by_subject)``:
      trials         one row per (subject, trial) with `subject`, `trial`, `RT`,
                     `acc`, `congruency`, `switchType`, and the two proportion
                     columns. Rows keep their per-subject order so they align with
                     the HG matrices below.
      hg_by_subject  {subject: (electrode_ids, ndarray (n_trials, n_channels))},
                     the window mean of HG over [tmin, tmax] per trial per channel.
                     `electrode_ids` use the same `f"{subject}-{channel}"` naming as
                     the A1 labels so the two can be joined directly.

    Trials with an unusable congruency label are dropped (they can't contribute to
    either adjustment); a `switchType` outside {'s','r'} — the first-of-block `n` —
    is kept but yields a NaN switch adjustment, which the models drop per-column.
    """
    from dcc_scripts.stats.stability_flexibility_segregation_dcc import (
        _window_indices, _proportion_col, _first_col,
        _RT_COLS, _ACC_COLS, _SWITCH_COLS)

    frames, hg_by_subject = [], {}
    for sub, epochs in subjects_epochs.items():
        md = epochs.metadata
        if md is None or 'congruency' not in md.columns:
            from src.analysis.utils.epoch_metadata_utils import make_metadata_from_event_names
            md = make_metadata_from_event_names(epochs)

        rt_col = _first_col(md, _RT_COLS)
        if rt_col is None:
            raise KeyError(
                f"subject {sub}: epochs metadata has no reaction-time column "
                f"(looked for {_RT_COLS}); the within-subject single-trial level of "
                f"A6 needs per-trial RT. Columns present: {list(md.columns)}")
        acc_col = _first_col(md, _ACC_COLS)
        sw_col = _first_col(md, _SWITCH_COLS)

        cong = md['congruency'].to_numpy().astype(str)
        sw = md[sw_col].to_numpy().astype(str) if sw_col else np.full(len(md), 'n')
        keep = np.isin(cong, ['c', 'i'])

        times = np.asarray(epochs.times, float)
        s_idx, e_idx = _window_indices(times, tmin, tmax)
        hg_mean = np.nanmean(epochs.get_data()[:, :, s_idx:e_idx], axis=2)  # (trial, chan)
        ch_names = list(epochs.ch_names)
        if electrodes_to_keep is not None:
            wanted = electrodes_to_keep.get(sub, set())
            ch_idx = [i for i, ch in enumerate(ch_names) if ch in wanted]
        else:
            ch_idx = list(range(len(ch_names)))

        frames.append(pd.DataFrame(dict(
            subject=sub,
            trial=np.arange(len(md))[keep],
            RT=pd.to_numeric(md[rt_col], errors='coerce').to_numpy()[keep],
            acc=(pd.to_numeric(md[acc_col], errors='coerce').to_numpy()[keep]
                 if acc_col else np.nan),
            congruency=cong[keep],
            switchType=sw[keep],
            incongruent_proportion=_proportion_col(md, 'incongruent_proportion')[keep],
            switch_proportion=_proportion_col(md, 'switch_proportion')[keep])))
        hg_by_subject[sub] = ([f"{sub}-{ch_names[i]}" for i in ch_idx],
                              hg_mean[np.ix_(keep, ch_idx)])

    if not frames:
        raise RuntimeError("assembled 0 trials — check subjects / window / metadata")
    return pd.concat(frames, ignore_index=True), hg_by_subject


def attach_group_hg(trials, hg_by_subject, labels, flag_cols=(('S', 'hg_lwpc'),
                                                             ('F', 'hg_lwps'))):
    """Average each trial's HG over the A1 LWPC (S) and LWPS (F) electrode groups.

    A subject with no electrode in a group gets NaN for that column (the mixed
    model drops those rows), which is the honest outcome — that subject carries no
    information about the group's trial-by-trial coupling."""
    out = trials.copy()
    for flag, col in flag_cols:
        values = np.full(len(out), np.nan)
        for sub, (elec_ids, hg) in hg_by_subject.items():
            sel = labels[(labels['subject'] == sub) & (labels[flag] == 1)]
            wanted = set(sel['electrode'])
            idx = [i for i, e in enumerate(elec_ids) if e in wanted]
            if not idx:
                continue
            rows = (out['subject'] == sub).to_numpy()
            with warnings.catch_warnings():        # all-NaN channel slices are fine
                warnings.simplefilter('ignore', category=RuntimeWarning)
                values[rows] = np.nanmean(hg[:, idx], axis=1)
        out[col] = values
    return out


# ---------------------------------------------------------------------------
# the trial-level behavioral adjustments (see the module docstring)
# ---------------------------------------------------------------------------
def _adjustment_weight(cond, mod, pos, neg):
    """The trial's cell weight in the equal-cell-weight difference-of-differences:
    +1 on the (pos, LOW) and (neg, high) diagonal, -1 on the other, NaN if the trial
    falls in neither (an unusable condition label or a missing proportion).

    LOW-proportion carries the +1, matching `_dod_rt` and the neural SIGN
    CONVENTION (see `stability_flexibility_segregation`): a subject's mean
    `adj_congruency` is their behavioral LWPC / 4 on the SAME orientation, so
    positive = the condition effect shrinks in the high-proportion block."""
    num = pd.to_numeric(pd.Series(mod), errors='coerce').to_numpy()
    finite = num[np.isfinite(num)]
    if finite.size == 0:
        return np.full(len(num), np.nan)
    hi, lo = float(np.max(finite)), float(np.min(finite))
    if hi == lo:                       # only one block level present: no interaction
        return np.full(len(num), np.nan)
    cond = np.asarray(cond).astype(str)
    cond_sign = np.where(cond == pos, 1.0, np.where(cond == neg, -1.0, np.nan))
    mod_sign = np.where(np.isclose(num, lo), 1.0,
                        np.where(np.isclose(num, hi), -1.0, np.nan))
    return cond_sign * mod_sign


def add_adjustment_columns(trials, rt_col='RT', center_cols=('hg_lwpc', 'hg_lwps'),
                           correct_only=True):
    """Attach `adj_congruency` / `adj_switch` and subject-center RT and group HG.

    Each adjustment is the trial's SIGNED CONTRIBUTION to its process's
    difference-of-differences: the d-o-d cell weight times the subject-centered RT.
    Centering RT (and the HG predictors) within subject keeps the mixed-model slope
    a purely WITHIN-subject quantity — an uncentered predictor would let
    between-subject differences in mean HG contribute to the common slope."""
    d = trials.copy()
    if correct_only and 'acc' in d.columns and d['acc'].notna().any():
        d = d[d['acc'] == 1]
    d[rt_col] = pd.to_numeric(d[rt_col], errors='coerce')
    d = d[d[rt_col].notna()].reset_index(drop=True)

    rt_centered = d[rt_col] - d.groupby('subject')[rt_col].transform('mean')
    d['rt_centered'] = rt_centered
    d['w_congruency'] = _adjustment_weight(
        d['congruency'], d['incongruent_proportion'], 'i', 'c')
    d['w_switch'] = _adjustment_weight(
        d['switchType'], d['switch_proportion'], 's', 'r')
    d['adj_congruency'] = d['w_congruency'] * rt_centered
    d['adj_switch'] = d['w_switch'] * rt_centered

    for col in center_cols:
        if col in d.columns:
            d[col] = d[col] - d.groupby('subject')[col].transform('mean')
    return d


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    return o


def save_results(labels, behavior, across, trialwise, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    labels.to_csv(os.path.join(save_dir, 'electrode_labels.csv'), index=False)
    behavior.to_csv(os.path.join(save_dir, 'behavioral_magnitudes.csv'), index=False)

    payload = {}
    for mode, res in across.items():
        res = dict(res)
        table = res.pop('table')
        table.to_csv(os.path.join(save_dir, f'subject_table_{mode}.csv'), index=False)
        payload[mode] = res
    with open(os.path.join(save_dir, 'across_subject.json'), 'w') as f:
        json.dump(_json_safe(payload), f, indent=2)

    with open(os.path.join(save_dir, 'trialwise.json'), 'w') as f:
        json.dump(_json_safe(trialwise), f, indent=2)


def save_participant_results(ps, participant, save_dir, csv_check=None, scores=None):
    """The level-(1) outputs: per-participant and per-electrode scores, their
    reliabilities, and the correlations (both variants) as JSON. `scores`
    replaces `ps['scores']` in participant_scores.csv (the table's behavior
    attached)."""
    os.makedirs(save_dir, exist_ok=True)
    (ps['scores'] if scores is None else scores).to_csv(
        os.path.join(save_dir, 'participant_scores.csv'), index=False)
    ps['electrodes'].to_csv(os.path.join(save_dir, 'participant_electrode_scores.csv'),
                            index=False)
    ps['reliability'].to_csv(os.path.join(save_dir, 'participant_reliability.csv'),
                             index=False)
    payload = {v: {k: val for k, val in res.items() if k != 'table'}
               for v, res in participant.items()}
    payload['settings'] = dict(n_splits=ps['n_splits'], min_elec=ps['min_elec'],
                               notes=ps['notes'])
    if csv_check is not None:
        payload['behavior_csv_agreement'] = csv_check
    with open(os.path.join(save_dir, 'participant_brain_behavior.json'), 'w') as f:
        json.dump(_json_safe(payload), f, indent=2)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_participant_plots(participant, save_dir):
    """Level (1): each participant's neural score against its behavioral score,
    LWPC and LWPS (rows) x RT-adjusted and raw (columns), one dot per participant."""
    variants = [(v, t) for v, t in (('rtadj', 'RT-adjusted (the claim)'),
                                     ('raw', 'raw (upper bound)'))
                if v in participant]
    if not variants:
        return
    ink, muted, rule = '#1f1f1f', '#6b6b6b', '#d9d9d9'
    fig, axes = plt.subplots(2, len(variants), figsize=(5.2 * len(variants), 8.4),
                             squeeze=False)
    for row, (eff, colour, name) in enumerate((('lwpc', STAB, 'LWPC'),
                                               ('lwps', FLEX, 'LWPS'))):
        for col, (variant, vtitle) in enumerate(variants):
            res, a = participant[variant], axes[row, col]
            t = res['table']
            x = t[res['neural_columns'][eff]].to_numpy(float)
            y = t[res['behavior_columns'][eff]].to_numpy(float)
            a.axhline(0, color=rule, lw=0.8, zorder=0)
            a.axvline(0, color=rule, lw=0.8, zorder=0)
            if len(x) >= 3 and np.std(x) > 0:
                b1, b0 = np.polyfit(x, y, 1)
                xs = np.linspace(np.min(x), np.max(x), 2)
                a.plot(xs, b0 + b1 * xs, color=colour, lw=1.5, zorder=2)
            a.scatter(x, y, s=42, color=colour, edgecolor='white', linewidth=0.8,
                      zorder=3)
            lo, hi = res[f'ci_{eff}']
            second = f"ceiling √(rel·rel) = {res[f'ceiling_{eff}']:.2f}"
            if np.isfinite(res[f'corr_{eff}_disjoint_half']):
                second += f" · disjoint-half r = {res[f'corr_{eff}_disjoint_half']:+.2f}"
            # stats sit above the plot area, so no participant is hidden under them
            a.text(0.0, 1.02,
                   f"r = {res[f'corr_{eff}']:+.2f} [{lo:+.2f}, {hi:+.2f}], "
                   f"p = {res[f'p_{eff}']:.3g}, n = {res['n_participants']} "
                   f"(|r| needed {res['r_crit']:.2f})\n{second}",
                   transform=a.transAxes, va='bottom', ha='left', fontsize=8.5,
                   color=ink)
            a.set_title(f"{name} · {vtitle}", fontsize=10, color=ink, loc='left',
                        pad=34)
            a.set_xlabel(f"neural {name}: mean electrode d (low − high)",
                         fontsize=9, color=muted)
            a.set_ylabel(f"behavioral {name}: RT d-o-d, ms (low − high)",
                         fontsize=9, color=muted)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
            a.tick_params(colors=muted, labelsize=8)
    fig.suptitle("A6 · across participants: neural vs behavioral adaptation "
                 "(one dot per participant)", fontsize=11, color=ink, x=0.02,
                 ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'participant_brain_behavior.png'), dpi=140,
                bbox_inches='tight')
    plt.close(fig)


def _slope_ci(slope, z):
    """95% CI from the mixed model's slope and z (SE = |slope / z|)."""
    if not np.isfinite(slope) or not np.isfinite(z) or z == 0:
        return np.nan
    return 1.96 * abs(slope / z)


def make_plots(across, trialwise, save_dir, primary='count'):
    res = across[primary]
    m = res['table']
    s_col, f_col = {'count': ('n_S', 'n_F'), 'frac': ('frac_S', 'frac_F'),
                    'effect': ('mean_stab', 'mean_flex')}[primary]

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    # (1-2) the two MATCHED across-subject scatters
    for a, xcol, ycol, colour, name, r, p in (
            (ax[0], s_col, 'lwpc', STAB, 'LWPC (stability)',
             res['corr_lwpc'], res['p_lwpc']),
            (ax[1], f_col, 'lwps', FLEX, 'LWPS (flexibility)',
             res['corr_lwps'], res['p_lwps'])):
        a.scatter(m[xcol], m[ycol], color=colour)
        a.set(title=f"A6 · matched {name}\nr = {r:.2f}  p = {p:.3g}  "
                    f"n = {res['n_subjects']}",
              xlabel=f"neural: {xcol}", ylabel=f"behavioral {ycol} (RT d-o-d)")

    # (3) across-subject specificity: matched vs cross |r|
    pairs = [('LWPC neural\n↔ LWPC RT', abs(res['corr_lwpc']), STAB, 'matched'),
             ('LWPC neural\n↔ LWPS RT', abs(res['corr_cross_stab_lwps']), STAB, 'cross'),
             ('LWPS neural\n↔ LWPS RT', abs(res['corr_lwps']), FLEX, 'matched'),
             ('LWPS neural\n↔ LWPC RT', abs(res['corr_cross_flex_lwpc']), FLEX, 'cross')]
    xs = np.arange(len(pairs))
    for i, p in enumerate(pairs):                 # one call per bar: alpha is per-bar
        ax[2].bar(i, p[1], color=p[2], alpha=1.0 if p[3] == 'matched' else 0.3,
                  edgecolor='k', linewidth=0.6)
    ax[2].set_xticks(xs); ax[2].set_xticklabels([p[0] for p in pairs], fontsize=7)
    ax[2].set(title=f"A6 · across-subject specificity ({primary})\n"
                    "solid = matched, faded = cross control",
              ylabel="|Pearson r|")

    # (4) within-subject slopes: matched vs cross, with 95% CI
    labels_, vals, errs, colours, alphas = [], [], [], [], []
    for group, colour in (('LWPC', STAB), ('LWPS', FLEX)):
        r = trialwise.get(group)
        if r is None:
            continue
        labels_ += [f"{group}\nmatched", f"{group}\ncross"]
        vals += [r['slope'], r['slope_cross']]
        errs += [_slope_ci(r['slope'], r['z']), _slope_ci(r['slope_cross'], r['z_cross'])]
        colours += [colour, colour]
        alphas += [1.0, 0.3]
    if vals:
        xs = np.arange(len(vals))
        for i in xs:
            ax[3].bar(i, vals[i], yerr=errs[i], color=colours[i], alpha=alphas[i],
                      edgecolor='k', linewidth=0.6, capsize=4)
        ax[3].set_xticks(xs); ax[3].set_xticklabels(labels_, fontsize=7)
        ax[3].axhline(0, color='k', lw=0.6)
        ax[3].set(title="A6 · within-subject single-trial slopes\n"
                        "(mixedlm, subject random intercept; 95% CI)",
                  ylabel="slope: adjustment per unit group HG")
    else:
        ax[3].text(0.5, 0.5, "within-subject level not run\n(see summary.txt)",
                   ha='center', va='center', transform=ax[3].transAxes)
        ax[3].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'brain_behavior_summary.png'), dpi=140,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
_SCORE_LABELS = (('lwpc_neural', 'neural LWPC, raw'),
                 ('lwpc_neural_rtadj', 'neural LWPC, RT-adjusted'),
                 ('lwps_neural', 'neural LWPS, raw'),
                 ('lwps_neural_rtadj', 'neural LWPS, RT-adjusted'),
                 ('lwpc_behav', 'behavioral LWPC'),
                 ('lwps_behav', 'behavioral LWPS'))


def _participant_lines(ps, participant):
    """Summary lines for level (1), the per-participant continuous scores."""
    if ps is None:
        return ["      NOT RUN — see the notes below."]
    s = ps['scores']
    rel = ps['reliability'].set_index('score')
    from_table = any(r['behavior_from'] == 'table' for r in participant.values())
    trial_tag = ' (iEEG trials)' if from_table else ''
    n_ok = int(s[['lwpc_neural', 'lwps_neural']].notna().all(axis=1).sum())
    lines = [
        f"      {len(s)} participants in the long table; {n_ok} with >= "
        f"{ps['min_elec']} usable electrodes (median {np.median(s['n_elec']):.0f} "
        f"electrodes and {np.median(s['n_trials']):.0f} trials per participant)",
        f"      split-half reliability of the participant values (one trial split per "
        f"participant, {ps['n_splits']} splits): full length [half-length r]",
    ]
    for name, label in _SCORE_LABELS:
        if name in rel.index:
            if name.endswith('_behav'):
                label += trial_tag
            lines.append(f"          {label:<30} {rel.loc[name, 'reliability']:5.2f}  "
                         f"[{rel.loc[name, 'r_half']:+.2f} ± {rel.loc[name, 'sd_half']:.2f}]")
    if from_table:
        lines.append("      the behavior table has no trials: the ceiling estimates its "
                     "reliability by the iEEG-trial one")
    if 'rt_hg_r' in s.columns:
        lines.append(f"      within-cell HG-RT correlation, median electrode per "
                     f"participant, median over participants: "
                     f"{np.nanmedian(s['rt_hg_r']):+.3f}")
    for variant, title in (('rtadj', 'RT-ADJUSTED — the claim'),
                           ('raw', 'RAW — upper bound; RT coupling inflates it')):
        res = participant.get(variant)
        if res is None:
            continue
        lines.append(f"      [{title}]  n = {res['n_participants']} participants; "
                     f"|r| needed for p < .05: {res['r_crit']:.2f}")
        for eff, name in (('lwpc', 'LWPC'), ('lwps', 'LWPS')):
            lo, hi = res[f'ci_{eff}']
            lines.append(
                f"          MATCHED  {name} neural ↔ {name} RT : r = {res[f'corr_{eff}']:+.3f} "
                f"[{lo:+.2f}, {hi:+.2f}]  p = {res[f'p_{eff}']:.4g}  "
                f"rho = {res[f'rho_{eff}']:+.3f}  ceiling = {res[f'ceiling_{eff}']:.2f}")
        lines += [
            f"          CROSS    LWPC neural ↔ LWPS RT : r = {res['corr_cross_stab_lwps']:+.3f}  "
            f"p = {res['p_cross_stab_lwps']:.4g}",
            f"          CROSS    LWPS neural ↔ LWPC RT : r = {res['corr_cross_flex_lwpc']:+.3f}  "
            f"p = {res['p_cross_flex_lwpc']:.4g}",
        ]
        for eff, name, other in (('lwpc', 'LWPC', 'LWPS'), ('lwps', 'LWPS', 'LWPC')):
            j = res[f'joint_{eff}']
            lines.append(
                f"          JOINT    {name} RT ~ neural {name} (beta = {j['beta_matched']:+.2f}, "
                f"p = {j['p_matched']:.3g}) + neural {other} (beta = {j['beta_cross']:+.2f}, "
                f"p = {j['p_cross']:.3g})")
        if res['behavior_from'] == 'table':
            lines.append("          half-length r, same half / disjoint halves: n/a "
                         "(the behavior table has no trial halves)")
        else:
            lines.append(
                f"          half-length r, same half / disjoint halves: "
                f"LWPC {res['corr_lwpc_same_half']:+.2f} / {res['corr_lwpc_disjoint_half']:+.2f}   "
                f"LWPS {res['corr_lwps_same_half']:+.2f} / {res['corr_lwps_disjoint_half']:+.2f}")
    res = participant.get('rtadj') or participant.get('raw')
    if res is not None:
        lines.append(f"      {res['caveat']}")
    return lines


def write_summary(labels, behavior, across, trialwise, save_dir, meta,
                  primary='count', notes=(), alpha=0.05, ps=None, participant=None,
                  csv_check=None, behavior_source=None):
    participant = participant or {}
    lines = [
        "=" * 78,
        "STABILITY vs FLEXIBILITY — A6 BRAIN-BEHAVIOR",
        "=" * 78,
    ]
    for k, v in meta.items():
        lines.append(f"{k:>22}: {v}")
    lines += [
        "-" * 78,
        f"electrodes: {len(labels)} | S (LWPC) = {int(labels['S'].sum())} | "
        f"F (LWPS) = {int(labels['F'].sum())} | "
        f"both = {int(((labels['S'] == 1) & (labels['F'] == 1)).sum())}",
        f"behavioral magnitudes ({behavior_source or 'see meta'}): "
        f"{len(behavior)} subjects | mean LWPC = {np.nanmean(behavior['lwpc']):.1f} | "
        f"mean LWPS = {np.nanmean(behavior['lwps']):.1f}",
    ]
    if csv_check is not None:
        lines.append(
            f"behavior cross-check, table vs the long table's own trials "
            f"({csv_check['n_participants']} participants): "
            f"r = {csv_check['r_lwpc']:+.2f} (LWPC), {csv_check['r_lwps']:+.2f} (LWPS); "
            f"mean |difference| = {csv_check['mean_abs_diff_lwpc_ms']:.0f} / "
            f"{csv_check['mean_abs_diff_lwps_ms']:.0f} ms")
    behav_desc = ("the table's LWPC_effect / LWPS_effect"
                  if any(r['behavior_from'] == 'table' for r in participant.values())
                  else "RT d-o-d on the same trials")
    lines += [
        "-" * 78,
        "(1) ACROSS PARTICIPANTS, CONTINUOUS SCORES — the across-participant result",
        f"      neural = mean signed per-electrode d; behavior = {behav_desc};",
        "      both LOW minus HIGH (positive = the effect shrinks in the",
        "      high-proportion block).",
        *_participant_lines(ps, participant),
        "-" * 78,
        "(2) ACROSS SUBJECTS, LABEL-BASED — for comparison only: counts need",
        "    thresholded labels and 'effect' averages an unsigned F",
    ]
    for mode, desc in _NEURAL_MODES:
        res = across.get(mode)
        if res is None:
            continue
        star = " <- primary" if mode == primary else ""
        matched_beats_cross = (abs(res['corr_lwpc']) > abs(res['corr_cross_stab_lwps'])
                               and abs(res['corr_lwps']) > abs(res['corr_cross_flex_lwpc']))
        lines += [
            f"      neural = {mode!r} ({desc}){star}",
            f"          MATCHED  LWPC neural ↔ LWPC RT : r = {res['corr_lwpc']:+.3f}  "
            f"p = {res['p_lwpc']:.4g}",
            f"          MATCHED  LWPS neural ↔ LWPS RT : r = {res['corr_lwps']:+.3f}  "
            f"p = {res['p_lwps']:.4g}",
            f"          CROSS    LWPC neural ↔ LWPS RT : r = {res['corr_cross_stab_lwps']:+.3f}  "
            f"p = {res['p_cross_stab_lwps']:.4g}",
            f"          CROSS    LWPS neural ↔ LWPC RT : r = {res['corr_cross_flex_lwpc']:+.3f}  "
            f"p = {res['p_cross_flex_lwpc']:.4g}",
            f"          specificity (both matched |r| > their cross |r|): "
            f"{matched_beats_cross}",
        ]
    primary_res = across.get(primary)
    if primary_res is not None:
        lines.append(f"      n = {primary_res['n_subjects']} subjects. "
                     f"{primary_res['caveat']}")

    lines += [
        "-" * 78,
        "(3) WITHIN SUBJECT, SINGLE TRIAL",
        "      model: adjustment ~ group HG, subject random intercept (mixedlm);",
        "      adjustment = the trial's signed contribution to its process's",
        "      difference-of-differences (d-o-d cell weight x subject-centered RT).",
    ]
    if trialwise:
        for group in ('LWPC', 'LWPS'):
            r = trialwise.get(group)
            if r is None:
                continue
            lines += [
                f"      [{group}]  n_trials = {r['n_trials']}  n_subjects = {r['n_subjects']}",
                f"          MATCHED ({r['matched_adjustment']}): slope = {r['slope']:+.4f}  "
                f"p = {r['p']:.4g}  z = {r['z']:.2f}",
                f"          CROSS   ({r['cross_adjustment']}): slope = {r['slope_cross']:+.4f}  "
                f"p = {r['p_cross']:.4g}  z = {r['z_cross']:.2f}",
                f"          specificity_ok (|matched| > |cross|): {r['specificity_ok']}",
            ]
    else:
        lines.append("      NOT RUN — see the notes below.")
    for note in notes:
        lines.append(f"      NOTE: {note}")

    lines += [
        "=" * 78,
        "Reading: (1) is the across-participant result. Read the RT-adjusted r against",
        "its ceiling and the |r| needed at this n: a null is uninformative, and a raw r",
        "that shrinks after adjustment was carried by RT coupling. Specificity is the",
        "JOINT beta, not matched |r| > cross |r|: behavioral LWPC and LWPS correlate",
        "across participants, and RT coupling alone makes matched beat cross. (2) is",
        "kept for comparison only. In (3) every slope is 'significant' with thousands",
        "of trials; a plain HG-RT correlation leaks into both slopes, so read it with",
        "docs/a6_brain_behavior.md.",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    alpha = getattr(args, 'alpha', 0.05)
    primary = getattr(args, 'neural_summary', 'count')
    run_trialwise = getattr(args, 'run_trialwise', True)
    min_elec = getattr(args, 'min_elec', 3)
    participant_n_splits = getattr(args, 'participant_n_splits', 200)
    seed = getattr(args, 'seed', 0)
    notes = []

    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE} | fdr_correction: {getattr(args, 'fdr_correction', 'fdr_bh')}")
    os.makedirs(args.save_dir, exist_ok=True)

    trial_df, csv_check, csv_path, participant_behavior = None, None, None, None
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        labels, behavior, trial_df = sbb._synthetic_brain_behavior(
            n_subj=getattr(args, 'synthetic_n_subj', 16), seed=seed,
            across_beta=getattr(args, 'synthetic_across_beta', 1.2),
            within_beta=getattr(args, 'synthetic_within_beta', 0.6),
            cross_frac=getattr(args, 'synthetic_cross_frac', 0.25))
        hg_cols = dict(LWPC='hg_lwpc', LWPS='hg_lwps')
        print(f"synthetic: {len(labels)} electrodes | {len(behavior)} subjects | "
              f"{len(trial_df)} trials (planted matched > cross at both levels)")
        long_df, _ = sbb._synthetic_long_df(
            n_subj=getattr(args, 'synthetic_n_subj', 16), seed=seed,
            link=getattr(args, 'synthetic_link', 0.6),
            rt_coupling=getattr(args, 'synthetic_rt_coupling', 0.3))
        print(f"synthetic long table: {len(long_df)} rows | "
              f"{long_df.electrode.nunique()} electrodes (planted link = "
              f"{getattr(args, 'synthetic_link', 0.6)}, RT coupling = "
              f"{getattr(args, 'synthetic_rt_coupling', 0.3)})")
        behavior_source = 'synthetic, planted'
    else:
        print("DATA SOURCE: real epoched data (behavior from the subject-level table)")
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

        # 1. A1 electrode definition ----------------------------------------------
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
        print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
              f"{df.electrode.nunique()} electrodes")
        for col in ('incongruent_proportion', 'switch_proportion'):
            if col not in df.columns or df[col].isna().all():
                raise RuntimeError(
                    f"df is missing usable '{col}' — A6 correlates behavior against "
                    "the A1 (proportion) electrode definition, which needs the "
                    "block-proportion columns.")
        df.to_csv(os.path.join(args.save_dir, 'long_df.csv'), index=False)
        long_df = df

        print("A1: per-electrode two-way interaction ANOVA (Type III, FDR across electrodes)")
        labels = sfs.per_electrode_anova_labels(
            df, alpha=alpha, contrast_mode=CONTRAST_MODE,
            fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'))

        # 2. behavior, from the subject-level effects table (RT) ----------------------
        csv_path = getattr(args, 'behavior_csv', None) or sbb.SUBJECT_LEVEL_BEHAVIOR_CSV
        table_behavior = sbb.load_subject_level_behavior(csv_path)
        behavior, no_behavior = sbb.match_behavior_to_subjects(
            table_behavior, df['subject'].unique())
        participant_behavior = behavior
        behavior_source = f"{os.path.basename(csv_path)}, key_RT_mean, ms"
        print(f"behavioral magnitudes: {len(behavior)} of {df.subject.nunique()} "
              f"subjects, from {csv_path}")
        if no_behavior:
            notes.append(f"no behavior for {no_behavior} in {os.path.basename(csv_path)}; "
                         "they drop out of every across-participant correlation")
        # cross-check: the same contrast scored on the long table's own trials
        try:
            csv_check = csv_behavior_agreement(table_behavior, behavior_from_long_df(df))
            print(f"behavior cross-check vs the long table's trials: "
                  f"r = {csv_check['r_lwpc']:+.2f} (LWPC), "
                  f"{csv_check['r_lwps']:+.2f} (LWPS) over "
                  f"{csv_check['n_participants']} participants")
        except KeyError as e:
            notes.append(f"behavior cross-check skipped: {e}")

        # 3. single-trial table for the within-subject level -------------------------
        if run_trialwise:
            try:
                trials, hg_by_subject = assemble_trial_table(
                    subjects_epochs, args.window_tmin, args.window_tmax,
                    electrodes_to_keep=keep)
                trials = attach_group_hg(trials, hg_by_subject, labels)
                trial_df = add_adjustment_columns(trials)
                trial_df.to_csv(os.path.join(args.save_dir, 'trial_df.csv'), index=False)
                print(f"single-trial table: {len(trial_df)} trials | "
                      f"{trial_df.subject.nunique()} subjects")
            except (KeyError, RuntimeError) as e:
                notes.append(f"single-trial level skipped: {e}")
                print(f"WARNING: single-trial level skipped: {e}")
                trial_df = None
        else:
            notes.append("single-trial level disabled (RUN_TRIALWISE=0)")
        hg_cols = dict(LWPC='hg_lwpc', LWPS='hg_lwps')

    # 4. per-participant continuous scores (level 1) --------------------------------
    print(f"A6 (1): per-participant continuous scores ({participant_n_splits} shared "
          f"splits, min {min_elec} electrodes)")
    ps, participant = None, {}
    try:
        ps = sbb.participant_scores(long_df, n_splits=participant_n_splits, seed=seed,
                                    min_elec=min_elec)
        notes += [f"participant scores: {n}" for n in ps['notes']]
    except (KeyError, ValueError) as e:
        notes.append(f"per-participant scores skipped: {e}")
        print(f"WARNING: per-participant scores skipped: {e}")
    if ps is not None:
        for variant in ('rtadj', 'raw'):
            try:
                res = sbb.participant_brain_behavior(
                    ps, variant=variant, alpha=alpha, behavior=participant_behavior)
            except KeyError as e:
                notes.append(f"participant correlations ({variant}) skipped: {e}")
                continue
            participant[variant] = res
            print(f"      [{variant}] n={res['n_participants']} | LWPC r={res['corr_lwpc']:+.3f} "
                  f"(ceiling {res['ceiling_lwpc']:.2f}) | LWPS r={res['corr_lwps']:+.3f} "
                  f"(ceiling {res['ceiling_lwps']:.2f}) | |r| needed {res['r_crit']:.2f}")

    # 5. across-subject correlations, label-based (level 2) -------------------------
    print("A6 (2): across-subject correlations (label-based) + cross-pairing controls")
    across = {}
    for mode, _desc in _NEURAL_MODES:
        try:
            across[mode] = sbb.subject_level_brain_behavior(
                labels, behavior[['subject', 'lwpc', 'lwps']], neural=mode,
                stab_effect='F_cong', flex_effect='F_switch')
        except KeyError as e:
            notes.append(f"across-subject neural={mode!r} skipped: {e}")
    if primary not in across:
        raise RuntimeError(f"the primary neural summary {primary!r} could not be "
                           f"computed; available: {sorted(across)}")
    for mode, res in across.items():
        print(f"      [{mode}] matched r: LWPC={res['corr_lwpc']:+.3f} "
              f"LWPS={res['corr_lwps']:+.3f} | cross r: "
              f"{res['corr_cross_stab_lwps']:+.3f} / {res['corr_cross_flex_lwpc']:+.3f}")

    # 6. within-subject single-trial mixed models (level 3) ------------------------
    trialwise = {}
    if trial_df is not None:
        print("A6 (3): within-subject single-trial mixed models (matched vs cross)")
        for group, hg_col in hg_cols.items():
            usable = trial_df[[hg_col, 'adj_congruency', 'adj_switch']].dropna(
                subset=[hg_col])
            if usable.empty:
                notes.append(f"{group} trial-level model skipped: no trial has HG in "
                             f"the {group} electrode group")
                continue
            try:
                trialwise[group] = sbb.trialwise_brain_behavior(
                    trial_df, group=group, hg_col=hg_col)
                r = trialwise[group]
                print(f"      [{group}] matched slope={r['slope']:+.4f} "
                      f"(p={r['p']:.3g}) | cross slope={r['slope_cross']:+.4f} "
                      f"(p={r['p_cross']:.3g}) | specificity_ok={r['specificity_ok']}")
            except Exception as e:                 # a singular fit is informative, not fatal
                notes.append(f"{group} trial-level model failed: {type(e).__name__}: {e}")
                print(f"WARNING: {group} trial-level model failed: {e}")

    # 7. persist + plot + summarize --------------------------------------------------
    save_results(labels, behavior, across, trialwise, args.save_dir)
    if ps is not None:
        save_participant_results(
            ps, participant, args.save_dir, csv_check=csv_check,
            scores=(None if participant_behavior is None
                    else sbb.attach_behavior(ps['scores'], participant_behavior)))
    make_plots(across, trialwise, args.save_dir, primary=primary)
    make_participant_plots(participant, args.save_dir)
    rois = getattr(args, 'rois_dict', None)
    write_summary(labels, behavior, across, trialwise, args.save_dir, notes=notes,
                  primary=primary, alpha=alpha, ps=ps, participant=participant,
                  csv_check=csv_check, behavior_source=behavior_source,
                  meta=dict(
                      data_source=args.data_source, task=args.task,
                      epochs_root_file=getattr(args, 'epochs_root_file', None),
                      behavior_csv=csv_path,
                      window=f"[{getattr(args, 'window_tmin', None)}, "
                             f"{getattr(args, 'window_tmax', None)}]s",
                      electrodes=getattr(args, 'electrodes', None),
                      rois='all' if rois is None else ','.join(rois),
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'),
                      primary_neural_summary=primary, alpha=alpha,
                      min_elec=min_elec, participant_n_splits=participant_n_splits,
                      save_dir=args.save_dir))
    return dict(labels=labels, behavior=behavior, across=across,
                trialwise=trialwise, trial_df=trial_df, participant_scores=ps,
                participant=participant, csv_check=csv_check)
