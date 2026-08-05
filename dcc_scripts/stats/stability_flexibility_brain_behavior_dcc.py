#!/usr/bin/env python
"""
DCC core for A6 — brain-behavior correlation
(`docs/analysis_guide.md` §19).

Ties the A1 neural selectivity to the ACTUAL behavioral control adjustment, so the
substrates are shown to be *functional* rather than incidental. Two levels with
very different power, both run here:

  (1) ACROSS SUBJECTS (n = subjects, honest but underpowered): does a subject with
      more/stronger LWPC electrodes show a larger behavioral LWPC (congruency x
      incongruent-proportion) RT effect, and likewise LWPS?
      -> `sbb.subject_level_brain_behavior`
  (2) WITHIN SUBJECT, SINGLE TRIAL (preferred, far more power): does trial-by-trial
      HG in the LWPC electrode group predict the trial-by-trial congruency
      adjustment (LWPS group <-> switch adjustment), via a mixed model with a
      subject random intercept?
      -> `sbb.trialwise_brain_behavior`

The MATCHED pairing should beat the CROSS pairing (LWPC group <-> switch
adjustment, and vice versa). That gap is the specificity result and the whole point
of A6, so both functions report the cross pairing alongside the matched one.

Pipeline:
  1. Assemble the same window-mean long table as the A1/A2/A3 jobs
     (`assemble_long_df`, contrast_mode='proportion') and run the A1 electrode
     definition (`sfs.per_electrode_anova_labels`) -> per-electrode S/F flags.
  2. Behavior: per-subject LWPC/LWPS RT magnitudes from the raw behavioral table
     (`combinedData.csv` by default) via `sbb.behavioral_lwpc_lwps_magnitudes` --
     the SAME equal-cell-weight difference-of-differences used for the neural
     interaction, so brain and behavior are measured on the identical contrast.
  3. Across-subject correlations for each neural summary ('count', 'frac', and the
     mean interaction F), each with its cross-pairing control.
  4. Single-trial table (`assemble_trial_table`): per (subject, trial) RT plus the
     HG averaged over the LWPC and LWPS electrode groups, then the mixed models.

The trial-level adjustment columns
----------------------------------
`sbb.trialwise_brain_behavior` deliberately takes the behavioral adjustment columns
as INPUT -- it does not define them, because the operationalization is a design
choice. This job defines them as each trial's SIGNED CONTRIBUTION to the very
difference-of-differences the rest of the battery is built on (`_adjustment_weight`
/ `add_adjustment_columns`):

    adj_congruency(t) = w(t) * (RT_t - mean RT of that subject)
    w(t) = +1 for (i, high-incongruent) and (c, low-incongruent)
           -1 for (c, high-incongruent) and (i, low-incongruent)

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
than its cross control, so the whole path is validated against ground truth in
seconds with no data on disk.

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
def load_behavior(csv_path, subject_col='subject_ID', rt_col='RT'):
    """Per-subject behavioral LWPC/LWPS magnitudes from the raw trial table.

    `combinedData.csv` names the subject column `subject_ID`; the analysis module
    expects `subject`, so it is renamed here rather than in the module."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"behavioral table not found at {csv_path}. Point BEHAVIOR_CSV at the "
            "raw trial-level behavior (the repo-root `combinedData.csv`, or the "
            "Box copy on the cluster).")
    raw = pd.read_csv(csv_path)
    if subject_col in raw.columns and 'subject' not in raw.columns:
        raw = raw.rename(columns={subject_col: 'subject'})
    missing = [c for c in ('subject', rt_col, 'congruency', 'switchType')
               if c not in raw.columns]
    if missing:
        raise KeyError(f"behavioral table {csv_path} is missing columns {missing}; "
                       f"it has {list(raw.columns)}")
    behavior = sbb.behavioral_lwpc_lwps_magnitudes(raw, rt_col=rt_col)
    return raw, behavior


# ---------------------------------------------------------------------------
# single-trial table (real data): RT + per-trial HG for each electrode group
# ---------------------------------------------------------------------------
# metadata column aliases -- the event-name parser emits the first of each pair,
# but a metadata CSV attached upstream may already use the short name.
_RT_COLS = ('reaction_time', 'RT', 'rt')
_ACC_COLS = ('accuracy', 'acc')
_SWITCH_COLS = ('task_sequence', 'switchType')


def _first_col(md, names):
    for n in names:
        if n in md.columns:
            return n
    return None


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
        _window_indices, _proportion_col)

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
    +1 on the (pos, high) and (neg, low) diagonal, -1 on the other, NaN if the trial
    falls in neither (an unusable condition label or a missing proportion)."""
    num = pd.to_numeric(pd.Series(mod), errors='coerce').to_numpy()
    finite = num[np.isfinite(num)]
    if finite.size == 0:
        return np.full(len(num), np.nan)
    hi, lo = float(np.max(finite)), float(np.min(finite))
    if hi == lo:                       # only one block level present: no interaction
        return np.full(len(num), np.nan)
    cond = np.asarray(cond).astype(str)
    cond_sign = np.where(cond == pos, 1.0, np.where(cond == neg, -1.0, np.nan))
    mod_sign = np.where(np.isclose(num, hi), 1.0,
                        np.where(np.isclose(num, lo), -1.0, np.nan))
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


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
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
def write_summary(labels, behavior, across, trialwise, save_dir, meta,
                  primary='count', notes=(), alpha=0.05):
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
        f"behavioral magnitudes: {len(behavior)} subjects | "
        f"mean LWPC = {np.nanmean(behavior['lwpc']):.1f} ms | "
        f"mean LWPS = {np.nanmean(behavior['lwps']):.1f} ms",
        "-" * 78,
        "(1) ACROSS SUBJECTS — underpowered by design; supporting, not decisive",
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
        "(2) WITHIN SUBJECT, SINGLE TRIAL — the powered test",
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
        "Reading: the headline is the SPECIFICITY GAP, not a p-value. With thousands",
        "of trials every slope is 'significant', so the claim rests on the matched",
        "pairing being STRONGER than the cross pairing (`specificity_ok`), at both",
        "levels. The across-subject correlation is reported with its n and an honest",
        "underpowered caveat — a null there is uninformative; the within-subject",
        "mixed model is the real test.",
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
    notes = []

    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE} | fdr_correction: {getattr(args, 'fdr_correction', 'fdr_bh')}")
    os.makedirs(args.save_dir, exist_ok=True)

    trial_df = None
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        labels, behavior, trial_df = sbb._synthetic_brain_behavior(
            n_subj=getattr(args, 'synthetic_n_subj', 16),
            seed=getattr(args, 'seed', 0),
            across_beta=getattr(args, 'synthetic_across_beta', 1.2),
            within_beta=getattr(args, 'synthetic_within_beta', 0.6),
            cross_frac=getattr(args, 'synthetic_cross_frac', 0.25))
        hg_cols = dict(LWPC='hg_lwpc', LWPS='hg_lwps')
        print(f"synthetic: {len(labels)} electrodes | {len(behavior)} subjects | "
              f"{len(trial_df)} trials (planted matched > cross at both levels)")
    else:
        print("DATA SOURCE: real epoched data + behavioral table")
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

        print("A1: per-electrode two-way interaction ANOVA (Type III, FDR across electrodes)")
        labels = sfs.per_electrode_anova_labels(
            df, alpha=alpha, contrast_mode=CONTRAST_MODE,
            fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'))

        # 2. behavior ---------------------------------------------------------------
        print(f"behavior: {args.behavior_csv}")
        _, behavior = load_behavior(args.behavior_csv, rt_col=args.behavior_rt_col)
        shared = set(behavior['subject']) & set(labels['subject'])
        print(f"behavioral magnitudes: {len(behavior)} subjects | "
              f"{len(shared)} also have neural labels")
        if not shared:
            raise RuntimeError(
                "no subject appears in BOTH the behavioral table and the neural "
                f"labels. Behavioral subjects: {sorted(set(behavior['subject']))[:8]}... "
                f"neural subjects: {sorted(set(labels['subject']))[:8]}... — check "
                "that the two use the same subject-ID convention.")

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

    # 4. across-subject correlations (every neural summary, each with its cross) ----
    print("A6 (1): across-subject correlations + cross-pairing controls")
    across = {}
    for mode, _desc in _NEURAL_MODES:
        try:
            across[mode] = sbb.subject_level_brain_behavior(
                labels, behavior, neural=mode,
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

    # 5. within-subject single-trial mixed models -----------------------------------
    trialwise = {}
    if trial_df is not None:
        print("A6 (2): within-subject single-trial mixed models (matched vs cross)")
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

    # 6. persist + plot + summarize --------------------------------------------------
    save_results(labels, behavior, across, trialwise, args.save_dir)
    make_plots(across, trialwise, args.save_dir, primary=primary)
    write_summary(labels, behavior, across, trialwise, args.save_dir, notes=notes,
                  primary=primary, alpha=alpha,
                  meta=dict(
                      data_source=args.data_source, task=args.task,
                      epochs_root_file=getattr(args, 'epochs_root_file', None),
                      behavior_csv=getattr(args, 'behavior_csv', None),
                      window=f"[{getattr(args, 'window_tmin', None)}, "
                             f"{getattr(args, 'window_tmax', None)}]s",
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'),
                      primary_neural_summary=primary, alpha=alpha,
                      save_dir=args.save_dir))
    return dict(labels=labels, behavior=behavior, across=across,
                trialwise=trialwise, trial_df=trial_df)
