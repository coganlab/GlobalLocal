"""A6 — brain–behavior correlation (plan §6).

Tie the neural selectivity (A1's LWPC / LWPS electrode groups) to the ACTUAL
behavioral control adjustment, so the substrates are shown to be *functional*,
not incidental. Three levels with very different power:

- **Across subjects, continuous scores** (the across-participant test to report;
  ``participant_scores`` + ``participant_brain_behavior``): each participant's
  MEAN signed per-electrode LWPC / LWPS d against its behavioral LWPC / LWPS.
  Reported with and without the RT-linked part of HG (``rt_adjust_hg``), with
  split-half reliabilities from ONE trial split per participant, and with the
  ceiling those reliabilities put on the correlation. n = participants, so a
  null is uninformative.
- **Across subjects, label-based** (n = subjects, low power): does a subject with
  more (or stronger) LWPC electrodes show a larger behavioral LWPC effect? And
  likewise LWPS? Kept for comparison; the counts need thresholded labels and the
  'effect' summary averages an unsigned F, so prefer the continuous level.
- **Within subject, single-trial** (far more power): does trial-by-trial
  high-gamma in the LWPC electrode group predict the trial-by-trial
  congruency-sequence RT adjustment (and the LWPS group ↔ the switch
  adjustment)? A mixed model with a subject random effect.

The two behavioral constructs mirror the neural ones exactly:

- **LWPC (stability) behavioral magnitude** = the congruency × incongruent-
  proportion interaction on RT — how much the congruency effect (RT_i − RT_c)
  CHANGES in low- vs high-incongruent-proportion blocks. Empirically this is a
  *shrinking* adjustment (the congruency effect is smaller in mostly-incongruent
  blocks), so the signed d-o-d below is typically POSITIVE.
- **LWPS (flexibility) behavioral magnitude** = the switchType × switch-
  proportion interaction on RT — how much the switch cost (RT_s − RT_r) changes in
  low- vs high-switch-proportion blocks; likewise typically a *shrinking* effect,
  and likewise positive.

Both are scored LOW-proportion minus HIGH-proportion, the same orientation the
neural scores use (see SIGN CONVENTION in ``stability_flexibility_segregation``),
so "+" means the same thing on both sides of every correlation below. Flipping one
side without the other silently negates every ``corr_lwpc`` / ``corr_lwps`` here.

The magnitudes are kept SIGNED rather than sign-corrected, and every test built on
them is two-sided, so nothing here assumes which way an effect must run. That
matters most on the neural side: the direction in which a block proportion
modulates a condition effect in a given population is not known a priori, so the
A1 electrode groups these correlate against are themselves direction-agnostic.

The per-subject behavioral magnitudes come precomputed from the subject-level
effects table ``src/config/ieeg_behavioral_subject_level_effects.csv``
(``load_subject_level_behavior``): its ``LWPC_effect`` / ``LWPS_effect`` are the
same equal-cell-weight difference-of-differences the segregation module uses for
the neural interaction, LOW minus HIGH, so brain and behavior are measured on the
identical contrast. ``behavioral_lwpc_lwps_magnitudes`` scores the same contrast
from raw trials; the job uses it only to cross-check the table against the
iEEG trials and for the behavioral split-half reliability.

**Specificity control (the whole point of A6).** The matched pairing
(LWPC group ↔ congruency-sequence adjustment; LWPS group ↔ switch adjustment)
should be stronger than the CROSS pairing (LWPC group ↔ switch adjustment). Both
functions report the cross pairing alongside the matched one. "Matched beats
cross" is also what RT coupling alone produces (see ``rt_adjust_hg``), and
behavioral LWPC and LWPS correlate across participants, so the continuous level
tests specificity with a joint regression on both neural scores instead.

``_synthetic_brain_behavior`` plants a matched across-subject correlation and a
matched within-subject single-trial coupling (both stronger than their cross
controls) so the whole path and the tutorial run with no data on disk.
``_synthetic_long_df`` plants a single-trial long table (real block design, a
brain-behavior link, common-mode noise, optional RT coupling) for the continuous
level.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from scipy.stats import t as t_dist

import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.analysis.stats import stability_flexibility_segregation as sfs


# ----------------------------------------------------------------------------
# behavioral LWPC / LWPS magnitudes (per subject): the subject-level table
# ----------------------------------------------------------------------------
# One row per (subject, measure), measure in {key_RT_mean, acc_mean, error_mean}.
# `LWPC_effect` = congruency_effect_25_inc - congruency_effect_75_inc and
# `LWPS_effect` = switch_cost_25_switch - switch_cost_75_switch: LOW minus HIGH,
# the orientation of `_dod_rt` and the neural scores (a test pins this against the
# file). The subject IDs are stems ('D0107'); the epochs may add a suffix
# ('D0107A'), so match with `subject_stem`.
SUBJECT_LEVEL_BEHAVIOR_CSV = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config',
    'ieeg_behavioral_subject_level_effects.csv'))
_BEHAVIOR_EFFECT_COLS = {'lwpc': 'LWPC_effect', 'lwps': 'LWPS_effect'}


def subject_stem(s):
    """'D0107A' -> 'D0107': the epochs and the behavioral tables spell some IDs
    differently."""
    m = re.match(r'(D\d+)', str(s))
    return m.group(1) if m else str(s)


def load_subject_level_behavior(csv_path=SUBJECT_LEVEL_BEHAVIOR_CSV,
                                measure='key_RT_mean'):
    """Per-subject behavioral LWPC and LWPS from the subject-level effects table.

    Parameters
    ----------
    csv_path : the table, one row per (subject, measure); defaults to
        ``SUBJECT_LEVEL_BEHAVIOR_CSV``.
    measure : which row to take per subject: 'key_RT_mean' (RT, ms), 'error_mean'
        or 'acc_mean' (percent). RT and error rate share the neural orientation
        (positive = the condition effect SHRINKS in the high-proportion block);
        accuracy is error rate negated, so there positive means it GROWS.

    Returns
    -------
    DataFrame: one row per subject with ``subject``, ``lwpc`` (the table's
    ``LWPC_effect``), ``lwps`` (``LWPS_effect``), then the table's other effect
    columns unchanged.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"subject-level behavior table not found at {csv_path}")
    t = pd.read_csv(csv_path)
    missing = [c for c in ('subject', 'measure', *_BEHAVIOR_EFFECT_COLS.values())
               if c not in t.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing columns {missing}; it has "
                       f"{list(t.columns)}")
    if measure not in set(t['measure']):
        raise ValueError(f"measure {measure!r} is not in {csv_path}; available: "
                         f"{sorted(t['measure'].unique())}")
    t = t[t['measure'] == measure].drop(columns='measure')
    dup = sorted(set(t.loc[t['subject'].duplicated(), 'subject']))
    if dup:
        raise ValueError(f"{csv_path} lists {dup} more than once for measure "
                         f"{measure!r}")
    t = t.rename(columns={v: k for k, v in _BEHAVIOR_EFFECT_COLS.items()})
    rest = [c for c in t.columns if c not in ('subject', *_BEHAVIOR_EFFECT_COLS)]
    return t[['subject', *_BEHAVIOR_EFFECT_COLS, *rest]].reset_index(drop=True)


def match_behavior_to_subjects(behavior, subjects):
    """``behavior`` relabelled to the given subject IDs, matched on ``subject_stem``.

    Returns ``(matched, missing)``: one row per subject in ``subjects`` that has
    behavior, in the order given and under its own ID, and the subjects without."""
    stems = behavior['subject'].map(subject_stem)
    if stems.duplicated().any():
        raise ValueError(f"behavior lists subjects {sorted(set(stems[stems.duplicated()]))} "
                         "under more than one ID")
    by_stem = behavior.drop(columns='subject').set_index(stems.to_numpy())
    subjects = list(dict.fromkeys(subjects))
    found = [s for s in subjects if subject_stem(s) in by_stem.index]
    missing = [s for s in subjects if subject_stem(s) not in by_stem.index]
    matched = by_stem.loc[[subject_stem(s) for s in found]].reset_index(drop=True)
    matched.insert(0, 'subject', found)
    return matched, missing


# ----------------------------------------------------------------------------
# behavioral LWPC / LWPS magnitudes (per subject) from raw behavior
# ----------------------------------------------------------------------------
# blockType -> block proportions, as the task builds them (src/task/mainTask.m:
# `createCongruencyArr` makes A and B 75% incongruent, `createTaskArr` makes A
# and C 25% switch). The two proportions are FULLY CROSSED over the four blocks:
#
#   A -> 75% incongruent, 25% switch        B -> 75% incongruent, 75% switch
#   C -> 25% incongruent, 25% switch        D -> 25% incongruent, 75% switch
#
# `general_utils.map_block_type` and combinedData.csv itself agree; the tests pin
# the map against the task code and the data.
#
# This map previously swapped A's and D's incongruent proportions (copied from
# erin_linear_mixed_effects_model.py). That made the two proportions look
# collinear and turned the behavioral "LWPC" into a congruency x SWITCH-proportion
# contrast mixed with block-level RT differences (on combinedData.csv: mean 46 ms,
# r = 0.02 with LWPS, instead of 123 ms and r = 0.44). Behavioral magnitudes
# computed from blockType before the fix -- the archived A6 across-subject
# numbers among them -- must be rerun.
_BLOCK_PROPORTION_MAP = {
    'A': dict(incongruent_proportion=75.0, switch_proportion=25.0),
    'B': dict(incongruent_proportion=75.0, switch_proportion=75.0),
    'C': dict(incongruent_proportion=25.0, switch_proportion=25.0),
    'D': dict(incongruent_proportion=25.0, switch_proportion=75.0),
}


def _dod_rt(sub, cond_col, mod_col, pos, neg, rt_col='RT'):
    """Equal-cell-weight difference-of-differences of mean RT over the 2x2
    (cond × mod) cells: (pos@lo - neg@lo) - (pos@hi - neg@hi). NaN if any of the
    four cells is empty. ``mod_col`` is the proportion column; its two levels are
    taken as the df-wide max ('high') and min ('low').

    LOW minus HIGH, matching the neural SIGN CONVENTION in
    ``stability_flexibility_segregation``: positive = the condition effect
    (RT_i - RT_c, RT_s - RT_r) SHRINKS in the high-proportion block, which is the
    direction behaviour shows. Brain and behaviour have to be on the same
    orientation or the A6 correlations below change sign for no reason."""
    num = pd.to_numeric(sub[mod_col], errors='coerce')
    hi, lo = num.max(), num.min()
    if not np.isfinite(hi) or hi == lo:
        return np.nan
    cells = {}
    for cval, clab in ((pos, 'pos'), (neg, 'neg')):
        for mval, mlab in ((hi, 'hi'), (lo, 'lo')):
            sel = (sub[cond_col] == cval) & np.isclose(num, mval)
            if not sel.any():
                return np.nan
            cells[(clab, mlab)] = sub.loc[sel, rt_col].mean()
    return ((cells[('pos', 'lo')] - cells[('neg', 'lo')])
            - (cells[('pos', 'hi')] - cells[('neg', 'hi')]))


def behavioral_lwpc_lwps_magnitudes(behav_df, rt_col='RT', subject_col='subject',
                                    correct_only=True):
    """Per-subject behavioral LWPC and LWPS magnitudes from a raw trial table.

    Parameters
    ----------
    behav_df : one row per behavioral trial with at least ``subject``, ``RT``,
        ``congruency`` ('i'/'c'), ``switchType`` ('s'/'r'), and either the
        proportion columns (``incongruent_proportion``, ``switch_proportion``) or
        a ``blockType`` column (mapped via ``_BLOCK_PROPORTION_MAP``). An ``acc``
        column, if present and ``correct_only``, restricts to correct trials.
    rt_col, subject_col : column names.

    Returns
    -------
    DataFrame: one row per subject with ``lwpc`` and ``lwps`` RT magnitudes
    (difference-of-differences, in RT units; SIGNED — positive = the condition
    effect is SMALLER in the high-proportion block, which is the usual behavioral
    direction; negative = larger) plus ``n_trials``.
    """
    d = behav_df.copy()
    if correct_only and 'acc' in d.columns:
        d = d[d['acc'] == 1]
    d = d[pd.to_numeric(d[rt_col], errors='coerce').notna()]
    if 'incongruent_proportion' not in d.columns and 'blockType' in d.columns:
        d['incongruent_proportion'] = d['blockType'].map(
            lambda b: _BLOCK_PROPORTION_MAP.get(b, {}).get('incongruent_proportion', np.nan))
    if 'switch_proportion' not in d.columns and 'blockType' in d.columns:
        d['switch_proportion'] = d['blockType'].map(
            lambda b: _BLOCK_PROPORTION_MAP.get(b, {}).get('switch_proportion', np.nan))

    rows = []
    for subj, sub in d.groupby(subject_col):
        rows.append(dict(
            subject=subj,
            lwpc=_dod_rt(sub, 'congruency', 'incongruent_proportion', 'i', 'c', rt_col),
            lwps=_dod_rt(sub, 'switchType', 'switch_proportion', 's', 'r', rt_col),
            n_trials=len(sub)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# per-subject neural summary from A1 electrode labels
# ----------------------------------------------------------------------------
def neural_summary_by_subject(elec_labels, stab_effect=None, flex_effect=None):
    """Reduce per-electrode A1 labels to one neural summary row per subject.

    Parameters
    ----------
    elec_labels : per-electrode table with ``subject``, ``S`` (LWPC-selective),
        ``F`` (LWPS-selective). Optional continuous effect-size columns can be
        named via ``stab_effect`` / ``flex_effect`` (e.g. ``'x'``/``'y'`` from
        ``compute_sensitivities``, or ``'F_cong'``/``'F_switch'`` from the ANOVA
        labels) to also get a per-subject MEAN effect.

    Returns
    -------
    DataFrame per subject: ``n_elec``, ``n_S``, ``n_F``, ``frac_S``, ``frac_F``
    (+ ``mean_stab``/``mean_flex`` when the effect columns are supplied).
    """
    rows = []
    for subj, g in elec_labels.groupby('subject'):
        rec = dict(subject=subj, n_elec=len(g),
                   n_S=int(g['S'].sum()), n_F=int(g['F'].sum()),
                   frac_S=float(g['S'].mean()), frac_F=float(g['F'].mean()))
        if stab_effect is not None and stab_effect in g:
            rec['mean_stab'] = float(np.nanmean(g[stab_effect].to_numpy()))
        if flex_effect is not None and flex_effect in g:
            rec['mean_flex'] = float(np.nanmean(g[flex_effect].to_numpy()))
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# (1) across-subject brain–behavior correlation
# ----------------------------------------------------------------------------
def subject_level_brain_behavior(elec_labels, behavior, neural='count',
                                 stab_effect=None, flex_effect=None):
    """Across-subject correlation of neural selectivity vs behavioral LWPC/LWPS.

    UNDERPOWERED BY DESIGN: n = number of subjects. Reported with n and treated as
    supporting, not decisive — the within-subject test (``trialwise_brain_behavior``)
    is the powered one.

    Parameters
    ----------
    elec_labels : per-electrode A1 labels (subject, S, F [, effect columns]).
    behavior : per-subject behavioral magnitudes with columns ``subject``,
        ``lwpc``, ``lwps`` (from ``load_subject_level_behavior``, relabelled to
        the labels' subject IDs with ``match_behavior_to_subjects``).
    neural : 'count' -> use n_S / n_F; 'frac' -> frac_S / frac_F; 'effect' -> the
        per-subject mean effect (requires ``stab_effect``/``flex_effect``).

    Returns
    -------
    dict: matched correlations (corr_lwpc/p_lwpc for stability, corr_lwps/p_lwps
    for flexibility), the CROSS-pairing specificity controls (corr_cross_*),
    ``n_subjects``, the merged per-subject ``table``, and a ``caveat`` string.
    """
    summ = neural_summary_by_subject(elec_labels, stab_effect, flex_effect)
    merged = summ.merge(behavior, on='subject', how='inner')

    col = {'count': ('n_S', 'n_F'), 'frac': ('frac_S', 'frac_F'),
           'effect': ('mean_stab', 'mean_flex')}[neural]
    s_col, f_col = col
    for c in (s_col, f_col):
        if c not in merged.columns:
            raise KeyError(f"neural={neural!r} needs column {c!r}; supply "
                           f"stab_effect/flex_effect for neural='effect'.")

    def corr(a, b):
        m = merged[[a, b]].dropna()
        if len(m) < 3:
            return (np.nan, np.nan)
        return pearsonr(m[a].to_numpy(), m[b].to_numpy())

    corr_lwpc, p_lwpc = corr(s_col, 'lwpc')        # matched: stability neural ↔ LWPC RT
    corr_lwps, p_lwps = corr(f_col, 'lwps')        # matched: flexibility neural ↔ LWPS RT
    corr_cx1, p_cx1 = corr(s_col, 'lwps')          # cross control
    corr_cx2, p_cx2 = corr(f_col, 'lwpc')          # cross control
    n = len(merged)
    return dict(
        corr_lwpc=corr_lwpc, p_lwpc=p_lwpc,
        corr_lwps=corr_lwps, p_lwps=p_lwps,
        corr_cross_stab_lwps=corr_cx1, p_cross_stab_lwps=p_cx1,
        corr_cross_flex_lwpc=corr_cx2, p_cross_flex_lwpc=p_cx2,
        n_subjects=n, neural=neural, table=merged,
        caveat=(f"Across-subject correlation is underpowered at n={n} subjects; "
                f"treat as supporting evidence, not decisive. The powered test is "
                f"the within-subject single-trial mixed model."))


# ----------------------------------------------------------------------------
# (2) within-subject single-trial mixed model (preferred)
# ----------------------------------------------------------------------------
def trialwise_brain_behavior(trial_df, group='LWPC', hg_col='hg_group',
                             matched_adj=None, cross_adj=None,
                             subject_col='subject'):
    """Within-subject single-trial link between group HG and the matching RT
    adjustment, with the cross pairing as a specificity control.

    Fits ``adjustment ~ hg_group`` with a subject random intercept
    (``statsmodels`` ``mixedlm``) for BOTH the matched and the cross behavioral
    adjustment, so the specificity claim (matched stronger than cross) is explicit.

    Parameters
    ----------
    trial_df : per-trial rows with ``subject``, the single-trial HG averaged over
        the selected electrode group (``hg_col``), and two behavioral adjustment
        columns — the congruency-sequence adjustment and the switch adjustment.
    group : 'LWPC' -> matched = congruency-sequence adjustment, cross = switch;
        'LWPS' -> matched = switch adjustment, cross = congruency-sequence.
        Default matched/cross column names are ``adj_congruency`` / ``adj_switch``;
        override via ``matched_adj`` / ``cross_adj``.

    Returns
    -------
    dict: ``slope``, ``p``, ``z`` for the MATCHED model; ``slope_cross``,
    ``p_cross`` for the CROSS model; ``specificity_ok`` (|matched slope| >
    |cross slope|); ``n_trials``, ``n_subjects``, ``group``.
    """
    if group == 'LWPC':
        m_col = matched_adj or 'adj_congruency'
        c_col = cross_adj or 'adj_switch'
    elif group == 'LWPS':
        m_col = matched_adj or 'adj_switch'
        c_col = cross_adj or 'adj_congruency'
    else:
        raise ValueError(f"group must be 'LWPC' or 'LWPS'; got {group!r}")

    def fit(adj_col):
        d = trial_df[[subject_col, hg_col, adj_col]].dropna()
        d = d.rename(columns={hg_col: 'hg_group', adj_col: 'adj'})
        m = smf.mixedlm('adj ~ hg_group', d, groups=d[subject_col]).fit(reml=False)
        return (float(m.params['hg_group']), float(m.pvalues['hg_group']),
                float(m.tvalues['hg_group']), len(d))

    slope, p, z, n = fit(m_col)
    slope_x, p_x, z_x, _ = fit(c_col)
    return dict(
        group=group, matched_adjustment=m_col, cross_adjustment=c_col,
        slope=slope, p=p, z=z,
        slope_cross=slope_x, p_cross=p_x, z_cross=z_x,
        specificity_ok=bool(abs(slope) > abs(slope_x)),
        n_trials=n, n_subjects=int(trial_df[subject_col].nunique()))


# ----------------------------------------------------------------------------
# (3) across participants, continuous scores (the version to report)
# ----------------------------------------------------------------------------
# One neural LWPC and LWPS per participant: the MEAN of its electrodes' signed
# scores, each the equal-cell-weight difference-of-differences over the pooled
# within-cell SD that the segregation / anatomy scores use
# (`sfs._interaction_cohens_d`). Behavior is scored on the SAME trials, from the
# long table's `rt`, with the same four cells and the same LOW-minus-HIGH sign.
# Equal weights within a participant are fine: its electrodes share its trials,
# so their sampling errors are nearly equal.
#
# Three things decide whether the correlation means anything:
#
#   1. RT coupling. If single-trial HG tracks RT inside a cell (slope b), every
#      electrode's d-o-d contains b x the participant's own behavioral d-o-d.
#      That builds a matched brain-behavior correlation, and a cross one in
#      proportion to how correlated behavioral LWPC and LWPS are, so "matched
#      beats cross" is exactly what RT coupling alone predicts. `rt_adjust_hg`
#      removes the RT-linked part of HG; the scores come with and without it.
#   2. Reliability. It must be measured with ONE split of each participant's
#      trials, shared by all its electrodes. Splitting each electrode separately
#      (as `compute_sensitivities_per_split` does) lets one electrode's half A
#      share trials with another's half B, and noise common to a participant's
#      electrodes then makes the two half-means agree: with a common-mode noise
#      correlation of 0.3 and 10 electrodes, a participant mean with NO real
#      between-participant differences shows a split-half reliability of ~0.57.
#   3. Shared trials. Brain and behavior come from the same trials, so the same
#      split also gives a disjoint-half brain-behavior correlation (neural from
#      half A against behavior from half B), which shared trial noise cannot
#      inflate.
_DESIGN_CELLS = ('congruency', 'switchType', 'incongruent_proportion',
                 'switch_proportion')
# 2x2 cell code = 2*cond + mod (cond 1 = incongruent / switch, mod 1 = the LOW-
# proportion block); the weights are the segregation module's own W_INTERACTION,
# so the sign convention stays in one place.
_W_CODE = np.array([sfs.W_INTERACTION[(c, m)]
                    for c in (0.0, 1.0) for m in (0.0, 1.0)])
_EFFECT_LABELS = (('lwpc', '_scond', '_smod'), ('lwps', '_fcond', '_fmod'))


def _codes(values):
    """Integer codes 0..k-1 (sorted) and the k unique values."""
    codes, uniques = pd.factorize(pd.Series(values), sort=True)
    return codes.astype(np.int64), np.asarray(uniques)


def _cell_codes(df):
    """Per-row 2x2 cell codes (0..3; -1 outside the 2x2) for LWPC and LWPS, from
    the segregation module's proportion-mode labels, so the cells and the sign are
    exactly those of the neural scores."""
    cols = list(_DESIGN_CELLS)
    contrasts = sfs.finalize_contrasts(df[cols], sfs.resolve_contrasts('proportion'))
    lab = sfs._canonical_labels(df[cols], contrasts)
    codes = {}
    for name, condcol, modcol in _EFFECT_LABELS:
        cond, mod = lab[condcol].to_numpy(), lab[modcol].to_numpy()
        ok = np.isfinite(cond) & np.isfinite(mod)
        code = np.full(len(df), -1, dtype=np.int64)
        code[ok] = (2 * cond[ok] + mod[ok]).astype(np.int64)
        codes[name] = code
    return codes


def _dod_scores(values, unit, code, n_units, standardize=True, min_n=2):
    """Equal-cell-weight difference-of-differences per unit, vectorized.

    `unit` (0..n_units-1) and `code` (2x2 cell, -1 = outside) are per row. With
    `standardize` the contrast is divided by the pooled within-cell SD and every
    cell needs >= `min_n` rows: `sfs._interaction_cohens_d`, electrode by
    electrode. Without it, the raw contrast of the four cell means, as `_dod_rt`
    scores behavior. NaN where a cell is short. Centre `values` per unit first
    when standardizing; the sums of squares are then exact."""
    ok = (code >= 0) & np.isfinite(values)
    g = unit[ok] * 4 + code[ok]
    v = values[ok]
    size = n_units * 4
    n = np.bincount(g, minlength=size).reshape(n_units, 4).astype(float)
    s1 = np.bincount(g, weights=v, minlength=size).reshape(n_units, 4)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = s1 / n
        out = (mean * _W_CODE).sum(1)
        if standardize:
            s2 = np.bincount(g, weights=v * v, minlength=size).reshape(n_units, 4)
            ss = np.clip(s2 - s1 * mean, 0.0, None)      # within-cell sums of squares
            sp = np.sqrt(ss.sum(1) / (n - 1).sum(1))
            out = out / sp
            out[~(sp > 0)] = np.nan
    out[(n < min_n).any(1)] = np.nan
    return out


def rt_adjust_hg(df, rt_col='rt', hg_col='hg', cell_cols=_DESIGN_CELLS,
                 electrode_col='electrode'):
    """Remove the RT-linked part of single-trial HG, electrode by electrode.

    hg_adj = hg - b_e * (rt - mean rt_e), with b_e the electrode's POOLED
    WITHIN-CELL slope of HG on RT, the cells being every congruency x switch type
    x incongruent proportion x switch proportion combination (the ANCOVA slope).
    A plain regression of HG on RT would also absorb the condition effects, which
    move both; deviations from the cell means carry none of them. Any contrast of
    cell means then moves by exactly -b_e x the same contrast of cell-mean RT, so
    the adjusted LWPC d-o-d is the raw one minus b_e x the behavioral one.

    This is conservative on purpose. If adaptation in HG reaches RT through the
    same trial-by-trial coupling, the adjustment removes that part too: the
    adjusted score is the neural adaptation not carried by RT, the unadjusted one
    an upper bound. Report both.

    Rows without a finite `rt` get NaN `hg`. Needs window-mean (scalar) HG.

    Returns
    -------
    (adjusted copy of df, per-electrode table with `electrode`, `rt_slope` (HG
    units per RT unit), `rt_r` (the pooled within-cell HG-RT correlation) and
    `n_rt` (trials used)).
    """
    if df[hg_col].dtype == object:
        raise ValueError("rt_adjust_hg needs window-mean (scalar) HG; build the long "
                         "table with effect_measure='cohens_d'")
    hg = pd.to_numeric(df[hg_col], errors='coerce').to_numpy(float)
    rt = pd.to_numeric(df[rt_col], errors='coerce').to_numpy(float)
    ok = np.isfinite(hg) & np.isfinite(rt)
    e_code, e_names = _codes(df[electrode_col].to_numpy())
    cell = df.groupby(list(cell_cols), sort=False, dropna=False).ngroup().to_numpy()
    n_e, n_c = len(e_names), int(cell.max()) + 1
    g = e_code * n_c + cell
    cnt = np.bincount(g[ok], minlength=n_e * n_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        hg_bar = np.bincount(g[ok], weights=hg[ok], minlength=n_e * n_c) / cnt
        rt_bar = np.bincount(g[ok], weights=rt[ok], minlength=n_e * n_c) / cnt
        dh = np.where(ok, hg - hg_bar[g], 0.0)
        dr = np.where(ok, rt - rt_bar[g], 0.0)
    sxy = np.bincount(e_code, weights=dh * dr, minlength=n_e)
    sxx = np.bincount(e_code, weights=dr * dr, minlength=n_e)
    syy = np.bincount(e_code, weights=dh * dh, minlength=n_e)
    n_rt = np.bincount(e_code[ok], minlength=n_e)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(sxx > 0, sxy / sxx, 0.0)
        r = sxy / np.sqrt(sxx * syy)
        rt_mean = np.bincount(e_code[ok], weights=rt[ok], minlength=n_e) / n_rt
        adj = np.where(ok, hg - slope[e_code] * (rt - rt_mean[e_code]), np.nan)
    out = df.copy()
    out[hg_col] = adj
    return out, pd.DataFrame(dict(electrode=e_names, rt_slope=slope, rt_r=r,
                                  n_rt=n_rt))


def _shared_half(strata, sizes, rng):
    """0 (half A) / 1 (half B) per trial. Within each participant x design-cell
    stratum a random floor(n/2) of the trials go to A, as in
    `sfs._stratified_half_split`, but drawn once per participant: every one of its
    electrodes, and its behavior, use the same split."""
    u = rng.random(len(strata))
    order = np.lexsort((u, strata))
    first = np.r_[0, np.flatnonzero(np.diff(strata[order])) + 1]
    run = np.diff(np.r_[first, len(order)])
    rank = np.empty(len(strata), dtype=np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(first, run)
    return (rank >= sizes[strata] // 2).astype(np.int64)


# How a participant's electrode scores become its one neural score, by column
# suffix: '' the signed mean (the claim); '_abs' the mean |d|, so electrodes whose
# adaptation runs in opposite directions no longer cancel; '_pos' the mean over the
# electrodes whose own score is positive (it needs `min_elec` of them). The last
# two fold or select on the same noisy d they average, so noise alone raises
# them, and more for a participant with fewer trials: exploratory.
_SUMMARIES = ('', '_abs', '_pos')
# `participant_brain_behavior` variant -> neural score column suffix
NEURAL_VARIANTS = {'raw': '', 'rtadj': '_rtadj',
                   'abs': '_abs', 'abs_rtadj': '_rtadj_abs',
                   'pos': '_pos', 'pos_rtadj': '_rtadj_pos'}


def participant_scores(df, n_splits=200, seed=0, min_elec=3, rt_adjust=True,
                       correct_only=True):
    """One neural and one behavioral LWPC / LWPS per participant, with reliabilities.

    Parameters
    ----------
    df : the long table from `assemble_long_df` (effect_measure='cohens_d'), one
        row per (electrode, trial): `subject`, `electrode`, `trial`, `hg` (window
        mean), `congruency`, `switchType`, `incongruent_proportion`,
        `switch_proportion`, plus `rt` / `acc` for behavior and the RT adjustment.
        Restrict it to the electrode set you mean (e.g. task-significant lPFC)
        BEFORE calling: every electrode given is averaged.
    n_splits : random half-splits for the reliabilities. Each divides each
        participant's trials once, stratified on the 16 design cells, and applies
        that split to all of its electrodes and to its behavior.
    min_elec : participants with fewer usable electrodes get NaN neural scores
        (they stay in the table for their behavior).
    rt_adjust : also score HG with its RT-linked part removed (`rt_adjust_hg`).
        The slope is fitted once on all trials; within a split it is shared by both
        halves, a leak of one parameter per electrode.
    correct_only : drop trials with `acc` != 1 when accuracy is present, for brain
        and behavior alike. Trials without an RT are dropped too when the table has
        RTs, so brain and behavior always use the same trials.

    Returns
    -------
    dict with
      scores       one row per participant: `n_elec` (usable electrodes),
                   `n_trials`, the neural means `lwpc_neural` / `lwps_neural`
                   (mean per-electrode d, LOW minus HIGH) and their `_rtadj`
                   versions, each also as `_abs` (mean |d|) and `_pos` (mean over
                   its d > 0 electrodes, counted in `_pos_n`; see `_SUMMARIES`),
                   behavioral `lwpc_behav` / `lwps_behav` (RT d-o-d, ms),
                   `mean_rt`, `resp` (mean |HG|, the gain proxy) and `rt_hg_r` (the
                   median within-cell HG-RT correlation of its electrodes).
      electrodes   one row per electrode: its full-data scores, `usable`, `resp`,
                   and `rt_slope` / `rt_r` when adjusted.
      reliability  one row per score: `r_half`, the mean over splits of the
                   across-participant correlation between half-A and half-B values
                   (`sd_half` its spread), and `reliability`, its Spearman-Brown
                   full-length value (NaN when r_half <= 0: no measurable signal).
      halves       {score: array (n_splits, n_participants, 2)}: half-A / half-B
                   participant values per split, for disjoint-half checks.
      subjects     participant order of `halves`.
      notes        what was dropped or skipped, in words.
    """
    need = ['subject', 'electrode', 'hg', *_DESIGN_CELLS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"participant_scores needs columns {missing}; the long table "
                       f"has {list(df.columns)}")
    if df['hg'].dtype == object:
        raise ValueError("participant_scores needs window-mean (scalar) HG; build the "
                         "long table with effect_measure='cohens_d'")
    notes = []
    d = df[need + [c for c in ('trial', 'rt', 'acc') if c in df.columns]].copy()
    d['hg'] = pd.to_numeric(d['hg'], errors='coerce')
    d = d[np.isfinite(d['hg'].to_numpy(float))]

    if 'trial' not in d.columns:
        per_elec = d.groupby(['subject', 'electrode']).size()
        if (per_elec.groupby(level='subject').nunique() > 1).any():
            raise ValueError(
                "the long table has no `trial` column and a participant's electrodes "
                "list different numbers of trials, so their trials cannot be "
                "aligned. Rebuild it with the current `assemble_long_df`, which "
                "writes `trial`.")
        d['trial'] = d.groupby('electrode').cumcount()
        notes.append("no `trial` column: assumed each participant's electrodes list "
                     "the same trials in the same order")

    def _n_trials(mask):
        return int(d.loc[mask, ['subject', 'trial']].drop_duplicates().shape[0])

    if correct_only and 'acc' in d.columns and d['acc'].notna().any():
        bad = (d['acc'] != 1).to_numpy()
        if bad.any():
            notes.append(f"dropped {_n_trials(bad)} trials with acc != 1")
            d = d[~bad]

    has_rt = 'rt' in d.columns and bool(
        np.isfinite(pd.to_numeric(d['rt'], errors='coerce')).any())
    if has_rt:
        d['rt'] = pd.to_numeric(d['rt'], errors='coerce')
        no_rt = ~np.isfinite(d['rt'].to_numpy(float))
        if no_rt.any():
            notes.append(f"dropped {_n_trials(no_rt)} trials without an RT, so brain "
                         "and behavior use the same trials")
            d = d[~no_rt]
    else:
        notes.append("no reaction times in the long table: behavioral scores and the "
                     "RT adjustment were skipped (rebuild it with the current "
                     "`assemble_long_df`)")
        rt_adjust = False
    d = d.reset_index(drop=True)

    codes = _cell_codes(d)
    subj_code, subjects = _codes(d['subject'].to_numpy())
    elec_code, electrodes = _codes(d['electrode'].to_numpy())
    n_p, n_e = len(subjects), len(electrodes)
    subj_of_elec = np.zeros(n_e, dtype=np.int64)
    subj_of_elec[elec_code] = subj_code
    per_elec_n = np.bincount(elec_code, minlength=n_e)

    raw = d['hg'].to_numpy(float)
    values = {'neural': raw}
    slopes = None
    if rt_adjust:
        adj, slopes = rt_adjust_hg(d)
        values['neural_rtadj'] = adj['hg'].to_numpy(float)
    for k, v in values.items():                 # centre per electrode (see _dod_scores)
        fin = np.isfinite(v)
        with np.errstate(divide='ignore', invalid='ignore'):
            centre = (np.bincount(elec_code[fin], weights=v[fin], minlength=n_e)
                      / np.bincount(elec_code[fin], minlength=n_e))
        values[k] = v - centre[elec_code]

    # one row per (participant, trial): behavior and the shared split live here
    trials = d.drop_duplicates(['subject', 'trial']).reset_index(drop=True)
    t_subj = pd.Index(subjects).get_indexer(trials['subject'])
    row_trial = pd.MultiIndex.from_frame(trials[['subject', 'trial']]).get_indexer(
        pd.MultiIndex.from_frame(d[['subject', 'trial']]))
    t_codes = _cell_codes(trials)
    strata = trials.groupby(['subject', *_DESIGN_CELLS], sort=False,
                            dropna=False).ngroup().to_numpy()
    sizes = np.bincount(strata)

    def pmean(x, mask):
        cnt = np.bincount(subj_of_elec[mask], minlength=n_p)
        with np.errstate(divide='ignore', invalid='ignore'):
            m = np.bincount(subj_of_elec[mask], weights=x[mask], minlength=n_p) / cnt
        m[cnt < min_elec] = np.nan
        return m

    def summarize(x, mask):
        """{suffix: participant values} of one electrode score, per `_SUMMARIES`."""
        return {'': pmean(x, mask), '_abs': pmean(np.abs(x), mask),
                '_pos': pmean(x, mask & (x > 0))}

    effects = ('lwpc', 'lwps')
    elec_scores = {f'{eff}_{k}': _dod_scores(v, elec_code, codes[eff], n_e)
                   for k, v in values.items() for eff in effects}
    usable = np.all([np.isfinite(s) for s in elec_scores.values()], axis=0)
    behav = {}
    if has_rt:
        t_rt = trials['rt'].to_numpy(float)
        behav = {f'{eff}_behav': _dod_scores(t_rt, t_subj, t_codes[eff], n_p,
                                             standardize=False, min_n=1)
                 for eff in effects}

    rng = np.random.default_rng(seed)
    halves = {k: np.full((n_splits, n_p, 2), np.nan)
              for k in [*(f'{e}{m}' for e in elec_scores for m in _SUMMARIES), *behav]}
    for s in range(n_splits):
        h_trial = _shared_half(strata, sizes, rng)
        unit = elec_code * 2 + h_trial[row_trial]
        for k, v in values.items():
            for eff in effects:
                ab = _dod_scores(v, unit, codes[eff], n_e * 2).reshape(n_e, 2)
                ok = usable & np.isfinite(ab).all(1)
                half_a, half_b = summarize(ab[:, 0], ok), summarize(ab[:, 1], ok)
                for m in _SUMMARIES:
                    halves[f'{eff}_{k}{m}'][s] = np.column_stack([half_a[m], half_b[m]])
        if has_rt:
            t_unit = t_subj * 2 + h_trial
            for eff in effects:
                halves[f'{eff}_behav'][s] = _dod_scores(
                    t_rt, t_unit, t_codes[eff], n_p * 2, standardize=False,
                    min_n=1).reshape(n_p, 2)

    rel_rows = []
    for name, arr in halves.items():
        rs, ns = [], []
        for s in range(n_splits):
            a, b = arr[s, :, 0], arr[s, :, 1]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
                ns.append(int(ok.sum()))
        r_half = float(np.mean(rs)) if rs else np.nan
        rel_rows.append(dict(
            score=name, r_half=r_half, sd_half=float(np.std(rs)) if rs else np.nan,
            reliability=2 * r_half / (1 + r_half) if r_half > 0 else np.nan,
            n_participants=int(np.median(ns)) if ns else 0, n_splits=len(rs)))

    resp_e = np.bincount(elec_code, weights=np.abs(raw), minlength=n_e) / per_elec_n
    elec_df = pd.DataFrame(dict(subject=subjects[subj_of_elec], electrode=electrodes,
                                n_trials=per_elec_n, usable=usable, resp=resp_e,
                                **elec_scores))
    if slopes is not None:
        elec_df = elec_df.merge(slopes[['electrode', 'rt_slope', 'rt_r']],
                                on='electrode', how='left')

    scores = pd.DataFrame(dict(subject=subjects,
                               n_elec=np.bincount(subj_of_elec[usable], minlength=n_p),
                               n_trials=np.bincount(t_subj, minlength=n_p)))
    for name, x in elec_scores.items():
        for m, val in summarize(x, usable).items():
            scores[f'{name}{m}'] = val
        scores[f'{name}_pos_n'] = np.bincount(subj_of_elec[usable & (x > 0)],
                                              minlength=n_p)
    for name, x in behav.items():
        scores[name] = x
    if has_rt:
        scores['mean_rt'] = pd.Series(trials['rt'].to_numpy(float)).groupby(
            t_subj).mean().reindex(range(n_p)).to_numpy()
    scores['resp'] = pmean(resp_e, usable)
    if slopes is not None:
        scores['rt_hg_r'] = (elec_df[elec_df['usable']].groupby('subject')['rt_r']
                             .median().reindex(subjects).to_numpy())
    return dict(scores=scores, electrodes=elec_df,
                reliability=pd.DataFrame(rel_rows), halves=halves,
                subjects=list(subjects), notes=notes, n_splits=n_splits,
                min_elec=min_elec)


def _fisher_ci(r, n, level=0.95):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    q = norm.ppf(0.5 + level / 2)
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))


def _mean_split_corr(x, y):
    """Mean over splits of the across-participant correlation of two (splits,
    participants) arrays, each split on the participants finite in both."""
    rs = []
    for a, b in zip(x, y):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def attach_behavior(scores, behavior):
    """`participant_scores(...)['scores']` with `lwpc_behav` / `lwps_behav` taken
    from a per-subject table (`subject`, `lwpc`, `lwps`; e.g.
    `load_subject_level_behavior`), matched on `subject_stem`. The long table's own
    trial-scored values, when present, move to `lwpc_behav_trials` /
    `lwps_behav_trials`. A participant the table lacks gets NaN."""
    s = scores.rename(columns={f'{eff}_behav': f'{eff}_behav_trials'
                               for eff in ('lwpc', 'lwps')})
    matched, _ = match_behavior_to_subjects(behavior[['subject', 'lwpc', 'lwps']],
                                            s['subject'])
    return s.merge(matched.rename(columns={'lwpc': 'lwpc_behav', 'lwps': 'lwps_behav'}),
                   on='subject', how='left')


def participant_brain_behavior(ps, variant='rtadj', alpha=0.05, behavior=None,
                               reliability_from_trials=True):
    """Across-participant brain-behavior correlations on `participant_scores` output.

    variant : 'rtadj' (neural scores with the RT-linked part of HG removed: the
        claim) or 'raw' (unadjusted: an upper bound that RT coupling inflates);
        'abs' / 'abs_rtadj' (mean |d|) and 'pos' / 'pos_rtadj' (mean over the
        d > 0 electrodes) are exploratory summaries (`NEURAL_VARIANTS`).
    behavior : optional per-subject table (`subject`, `lwpc`, `lwps`), e.g.
        `load_subject_level_behavior`. When given, every correlation uses it in
        place of the behavior `participant_scores` scored from the long table
        (`attach_behavior`). It has no trials, so the same- and disjoint-half
        correlations are NaN.
    reliability_from_trials : with `behavior`, take the behavioral reliability
        behind the ceiling from the long table's trial-scored RT d-o-d, as an
        estimate for the table's scores (right when the table holds RT effects;
        conservative if it used more trials). False leaves it and the ceiling NaN.

    Returns
    -------
    dict with, for LWPC and LWPS:
      corr_* / p_* / ci_* / rho_*    matched Pearson r, p, 95% CI, Spearman rho
      corr_cross_*                   the cross pairings (neural LWPC vs behavioral
                                     LWPS and the reverse)
      joint_*                        behavioral score ~ BOTH neural scores, all
                                     z-scored: `beta_matched` / `beta_cross` and
                                     their p. The specificity test: cross r's are
                                     not near zero when behavioral LWPC and LWPS
                                     correlate, a matched-vs-cross comparison is not
      reliability_neural_* / reliability_behav_* / ceiling_*
                                     full-length reliabilities and
                                     sqrt(rel_neural x rel_behav), the largest
                                     observable r even for a perfect true link
      corr_*_same_half / corr_*_disjoint_half
                                     half-length r with neural and behavior from
                                     the same trial half, and from opposite halves
                                     (immune to shared trial noise)
    plus `n_participants`, `r_crit` (the |r| needed for p < alpha at this n), the
    merged `table`, `behavior_from` ('table' or 'trials') and a `caveat`.
    """
    if variant not in NEURAL_VARIANTS:
        raise ValueError(f"variant must be one of {list(NEURAL_VARIANTS)}; "
                         f"got {variant!r}")
    ncol = {eff: f'{eff}_neural{NEURAL_VARIANTS[variant]}' for eff in ('lwpc', 'lwps')}
    bcol = {eff: f'{eff}_behav' for eff in ('lwpc', 'lwps')}
    external = behavior is not None
    s = attach_behavior(ps['scores'], behavior) if external else ps['scores']
    missing = [c for c in (*ncol.values(), *bcol.values()) if c not in s.columns]
    if missing:
        raise KeyError(f"participant scores lack {missing}: variant={variant!r} needs "
                       "the behavioral scores (pass `behavior`, or put `rt` in the "
                       "long table) and, for 'rtadj', the RT adjustment, which needs "
                       "`rt` in the long table")
    t = s.dropna(subset=[*ncol.values(), *bcol.values()]).reset_index(drop=True)
    n = len(t)
    out = dict(variant=variant, n_participants=n, neural_columns=dict(ncol),
               behavior_columns=dict(bcol),
               behavior_from='table' if external else 'trials')
    out['r_crit'] = (float(t_dist.ppf(1 - alpha / 2, n - 2)
                           / np.sqrt(t_dist.ppf(1 - alpha / 2, n - 2) ** 2 + n - 2))
                     if n > 2 else np.nan)

    def corr(a, b):
        if n < 3 or t[a].std() == 0 or t[b].std() == 0:
            return np.nan, np.nan
        r, p = pearsonr(t[a], t[b])
        return float(r), float(p)

    for eff in ('lwpc', 'lwps'):
        r, p = corr(ncol[eff], bcol[eff])
        out[f'corr_{eff}'], out[f'p_{eff}'] = r, p
        out[f'ci_{eff}'] = _fisher_ci(r, n)
        if n >= 3:
            rho, prho = spearmanr(t[ncol[eff]], t[bcol[eff]])
            out[f'rho_{eff}'], out[f'p_rho_{eff}'] = float(rho), float(prho)
        else:
            out[f'rho_{eff}'] = out[f'p_rho_{eff}'] = np.nan
    out['corr_cross_stab_lwps'], out['p_cross_stab_lwps'] = corr(ncol['lwpc'], bcol['lwps'])
    out['corr_cross_flex_lwpc'], out['p_cross_flex_lwpc'] = corr(ncol['lwps'], bcol['lwpc'])

    z = (t[[*ncol.values(), *bcol.values()]] - t[[*ncol.values(), *bcol.values()]].mean()) \
        / t[[*ncol.values(), *bcol.values()]].std()
    for eff, other in (('lwpc', 'lwps'), ('lwps', 'lwpc')):
        res = dict(beta_matched=np.nan, p_matched=np.nan, beta_cross=np.nan,
                   p_cross=np.nan)
        if n >= 5 and np.isfinite(z.to_numpy()).all():
            fit = sm.OLS(z[bcol[eff]].to_numpy(),
                         sm.add_constant(z[[ncol[eff], ncol[other]]].to_numpy())).fit()
            res = dict(beta_matched=float(fit.params[1]), p_matched=float(fit.pvalues[1]),
                       beta_cross=float(fit.params[2]), p_cross=float(fit.pvalues[2]))
        out[f'joint_{eff}'] = res

    rel = ps['reliability'].set_index('score')['reliability']
    idx = pd.Index(ps['subjects']).get_indexer(t['subject'])
    for eff in ('lwpc', 'lwps'):
        rn = float(rel.get(ncol[eff], np.nan))
        # with a table, the long table's trial-scored behavior (still keyed
        # `*_behav` in `ps`) only stands in for the reliability, if allowed
        rb = (float(rel.get(bcol[eff], np.nan))
              if not external or reliability_from_trials else np.nan)
        out[f'reliability_neural_{eff}'], out[f'reliability_behav_{eff}'] = rn, rb
        out[f'ceiling_{eff}'] = (float(np.sqrt(rn * rb))
                                 if np.isfinite(rn) and np.isfinite(rb) else np.nan)
        if external:                       # the table's scores have no trial halves
            out[f'corr_{eff}_same_half'] = out[f'corr_{eff}_disjoint_half'] = np.nan
            continue
        hn, hb = ps['halves'][ncol[eff]][:, idx, :], ps['halves'][bcol[eff]][:, idx, :]
        out[f'corr_{eff}_same_half'] = 0.5 * (_mean_split_corr(hn[..., 0], hb[..., 0])
                                             + _mean_split_corr(hn[..., 1], hb[..., 1]))
        out[f'corr_{eff}_disjoint_half'] = 0.5 * (
            _mean_split_corr(hn[..., 0], hb[..., 1])
            + _mean_split_corr(hn[..., 1], hb[..., 0]))

    out['table'] = t[['subject', 'n_elec', 'n_trials', *ncol.values(),
                      *bcol.values()]].copy()
    out['caveat'] = (
        f"n = {n} participants: |r| must reach {out['r_crit']:.2f} for p < {alpha}, "
        "and no observed r can exceed the reliability ceiling. A null is "
        "uninformative. The RT-adjusted variant is the claim; the raw one is an "
        "upper bound that RT coupling inflates.")
    return out


# ----------------------------------------------------------------------------
# the group-level adaptation with and without the RT-linked part of HG
# (Fig. 3's claim; docs/paper_draft.md §1.4, F3)
# ----------------------------------------------------------------------------
def group_adaptation_rt_check(electrodes, min_elec=1, n_perm=10000, seed=0,
                              effects=('lwpc', 'lwps')):
    """Is the mean LWPC / LWPS in HG still in the behavioral direction once the
    RT-linked part of HG is removed?

    Fig. 3's adaptation clusters run past the median RT, and RT coupling alone
    predicts a neural adaptation with behavior's sign: if HG tracks RT within
    cells (slope b), every electrode's difference of differences contains b
    times the behavioral one (`rt_adjust_hg`). This compares the window-mean
    scores before and after removing that part, with PARTICIPANTS as the unit.

    ``electrodes``: `participant_scores(...)['electrodes']`, or the A6 job's
    ``participant_electrode_scores.csv``, with ``lwpc_neural`` /
    ``lwps_neural`` and their ``_rtadj`` versions (signed d, LOW minus HIGH, so
    positive = the behavioral direction). Usable electrodes only. Each
    participant with at least ``min_elec`` of them contributes the mean of
    their scores.

    Per effect and variant (``raw``, ``rtadj``): the electrode mean and share
    positive; the mean over participants with its SEM, one-sample t-test, a
    sign-flip p over participants and how many are positive; a mixed model on
    the electrodes (intercept only, participant random intercept). For the
    adjusted rows, ``retained`` = adjusted / raw participant mean, and
    ``p_change`` the paired t-test of the difference.
    """
    import warnings
    from scipy.stats import ttest_1samp, ttest_rel

    e = electrodes.copy()
    if 'usable' in e.columns:
        e = e[e['usable'].astype(bool)]
    rows = []
    rng = np.random.default_rng(seed)
    for eff in effects:
        means = {}
        for variant, col in (('raw', f'{eff}_neural'), ('rtadj', f'{eff}_neural_rtadj')):
            if col not in e.columns:
                continue
            d = e[['subject', col]].dropna()
            counts = d.groupby('subject')[col].transform('size')
            d = d[counts >= min_elec]
            pm = d.groupby('subject')[col].mean()
            means[variant] = pm
            x = pm.to_numpy(float)
            row = dict(effect=eff.upper(), variant=variant, n_electrodes=int(len(d)),
                       electrode_mean=float(d[col].mean()),
                       electrode_share_positive=float((d[col] > 0).mean()),
                       n_participants=int(len(x)), n_positive=int((x > 0).sum()),
                       participant_mean=float(x.mean()) if len(x) else np.nan)
            if len(x) > 1:
                t = ttest_1samp(x, 0.0)
                flips = rng.choice((-1.0, 1.0), size=(int(n_perm), len(x)))
                null = (flips * x).mean(1)
                row.update(sem=float(x.std(ddof=1) / np.sqrt(len(x))),
                           t=float(t.statistic), p_t=float(t.pvalue),
                           p_signflip=float((np.sum(np.abs(null) >= abs(x.mean())) + 1)
                                            / (n_perm + 1)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # boundary fits; `converged` says so
                    fit = smf.mixedlm(f'{col} ~ 1', d, groups=d['subject']).fit(reml=True)
                row.update(mixed_mean=float(fit.params['Intercept']),
                           mixed_p=float(fit.pvalues['Intercept']),
                           mixed_converged=bool(fit.converged))
            except Exception as exc:                 # too few participants, singular
                row['mixed_note'] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        if {'raw', 'rtadj'} <= set(means):
            both = pd.concat([means['raw'].rename('raw'), means['rtadj'].rename('rtadj')],
                             axis=1).dropna()
            if len(both) > 1:
                rows[-1].update(
                    retained=float(both['rtadj'].mean() / both['raw'].mean())
                    if both['raw'].mean() != 0 else np.nan,
                    p_change=float(ttest_rel(both['raw'], both['rtadj']).pvalue))
    return pd.DataFrame(rows)


def group_adaptation_rt_lines(table):
    """Summary lines for :func:`group_adaptation_rt_check`'s table."""
    if table is None or table.empty:
        return ["      NOT RUN — see the notes below."]
    lines = ["      participants are the unit; positive = the behavioral direction "
             "(LOW minus HIGH)"]
    for r in table.itertuples():
        get = r._asdict().get
        p_mixed = get('mixed_p')
        lines.append(
            f"      {r.effect} {r.variant:<5}  mean d = {r.participant_mean:+.3f} "
            f"± {get('sem', np.nan):.3f} SEM  t-test p = {get('p_t', np.nan):.3g}  "
            f"sign-flip p = {get('p_signflip', np.nan):.3g}  "
            f"{r.n_positive}/{r.n_participants} participants > 0  "
            + (f"mixed p = {p_mixed:.3g}" if p_mixed is not None and np.isfinite(p_mixed)
               else "mixed n/a"))
        if r.variant == 'rtadj' and get('retained') is not None and np.isfinite(get('retained')):
            lines.append(f"            retained after RT adjustment: {get('retained'):.0%} "
                         f"of the raw mean (paired p = {get('p_change'):.3g})")
    lines.append("      the adjusted rows are the check on Fig. 3's direction: a raw "
                 "effect that vanishes here was RT coupling")
    return lines


# ----------------------------------------------------------------------------
# synthetic ground truth — planted matched links stronger than cross
# ----------------------------------------------------------------------------
def _synthetic_brain_behavior(n_subj=16, seed=0, across_beta=1.2,
                              within_beta=0.6, cross_frac=0.25):
    """Planted brain–behavior structure, returns (elec_labels, behavior, trial_df).

    - Across subjects: a subject's behavioral ``lwpc`` grows with its LWPC
      electrode count (slope ``across_beta``), ``lwps`` with its LWPS count; the
      cross links are only ``cross_frac`` as strong.
    - Within subject: single-trial ``adj_congruency`` is driven by the LWPC
      group's single-trial HG (slope ``within_beta``), ``adj_switch`` by the LWPS
      group HG; each group's HG drives the CROSS adjustment only ``cross_frac`` as
      much. So the matched mixed-model slope should beat the cross one.
    """
    rng = np.random.default_rng(seed)
    elec_rows, behav_rows, trial_frames = [], [], []
    for s in range(n_subj):
        subject = f"S{s:02d}"
        n_elec = int(rng.integers(20, 45))
        # latent per-subject selectivity strength drives BOTH counts and behavior
        stab_strength = rng.uniform(0.02, 0.30)
        flex_strength = rng.uniform(0.02, 0.30)
        S = (rng.random(n_elec) < stab_strength).astype(int)
        F = (rng.random(n_elec) < flex_strength).astype(int)
        for e in range(n_elec):
            elec_rows.append(dict(subject=subject, electrode=f"{subject}-e{e}",
                                  S=int(S[e]), F=int(F[e])))
        n_S, n_F = int(S.sum()), int(F.sum())
        # behavioral magnitudes: matched neural count + small cross leak + noise
        lwpc = across_beta * n_S + cross_frac * across_beta * n_F + rng.normal(0, 4)
        lwps = across_beta * n_F + cross_frac * across_beta * n_S + rng.normal(0, 4)
        behav_rows.append(dict(subject=subject, lwpc=lwpc, lwps=lwps))

        # single-trial: group HG drives the matched adjustment (+ weak cross)
        n_tr = int(rng.integers(150, 300))
        hg_lwpc = rng.normal(0, 1, n_tr)          # LWPC-group single-trial HG
        hg_lwps = rng.normal(0, 1, n_tr)          # LWPS-group single-trial HG
        subj_shift_c = rng.normal(0, 2)           # subject random intercepts
        subj_shift_s = rng.normal(0, 2)
        adj_cong = (subj_shift_c + within_beta * hg_lwpc
                    + cross_frac * within_beta * hg_lwps + rng.normal(0, 1, n_tr))
        adj_switch = (subj_shift_s + within_beta * hg_lwps
                      + cross_frac * within_beta * hg_lwpc + rng.normal(0, 1, n_tr))
        trial_frames.append(pd.DataFrame(dict(
            subject=subject, hg_lwpc=hg_lwpc, hg_lwps=hg_lwps,
            adj_congruency=adj_cong, adj_switch=adj_switch)))

    elec_labels = pd.DataFrame(elec_rows)
    behavior = pd.DataFrame(behav_rows)
    trial_df = pd.concat(trial_frames, ignore_index=True)
    return elec_labels, behavior, trial_df


def _synthetic_long_df(n_subj=24, n_elec=(4, 12), trials_per_block=112, seed=0,
                       link=0.6, rt_coupling=0.0, common_noise=0.3,
                       neural_mean=0.15, neural_sd=0.15, elec_sd=0.1):
    """Planted single-trial long table for `participant_scores`; returns (df, truth).

    Blocks follow the real design (`_BLOCK_PROPORTION_MAP`, exact 75/25 counts in
    each block). Each participant has a behavioral LWPC and LWPS (RT d-o-d, ms) and
    a neural LWPC and LWPS (d units; mean `neural_mean`, SD `neural_sd` across
    participants) that correlate with its OWN behavioral effect at `link`, and not
    with the other one. Each electrode carries its participant's neural effects
    plus jitter (`elec_sd`) and a random gain; `common_noise` is the share of trial
    noise variance shared by all of a participant's electrodes. `rt_coupling` adds
    that many noise SDs of HG per SD of RT: the RT confound, which puts
    rt_coupling x (behavioral d-o-d / RT SD) into every electrode's score.

    `truth` holds each participant's planted `lwpc_behav_true` / `lwps_behav_true`
    (ms) and `lwpc_neural_true` / `lwps_neural_true` (d).
    """
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for s in range(n_subj):
        subject = f"S{s:02d}"
        zb = rng.standard_normal(2)
        beh = np.array([120.0, 100.0]) + np.array([60.0, 50.0]) * zb
        neu = neural_mean + neural_sd * (link * zb
                                         + np.sqrt(1 - link ** 2) * rng.standard_normal(2))
        truth.append(dict(subject=subject, lwpc_behav_true=beh[0],
                          lwps_behav_true=beh[1], lwpc_neural_true=neu[0],
                          lwps_neural_true=neu[1]))

        cols = dict(congruency=[], switchType=[], incongruent_proportion=[],
                    switch_proportion=[])
        for props in _BLOCK_PROPORTION_MAP.values():
            n_inc = int(round(trials_per_block * props['incongruent_proportion'] / 100))
            n_sw = int(round(trials_per_block * props['switch_proportion'] / 100))
            cols['congruency'] += list(rng.permutation(
                ['i'] * n_inc + ['c'] * (trials_per_block - n_inc)))
            cols['switchType'] += list(rng.permutation(
                ['s'] * n_sw + ['r'] * (trials_per_block - n_sw)))
            cols['incongruent_proportion'] += [props['incongruent_proportion']] * trials_per_block
            cols['switch_proportion'] += [props['switch_proportion']] * trials_per_block
        inc = np.array(cols['congruency']) == 'i'
        sw = np.array(cols['switchType']) == 's'
        # each condition effect is half its d-o-d larger in the LOW block and half
        # smaller in the HIGH block, so LOW minus HIGH recovers the planted value
        half_c = np.where(np.array(cols['incongruent_proportion']) == 25.0, 0.5, -0.5)
        half_s = np.where(np.array(cols['switch_proportion']) == 25.0, 0.5, -0.5)
        n_tr = len(inc)
        rt = (1100.0 + inc * (120.0 + half_c * beh[0]) + sw * (100.0 + half_s * beh[1])
              + rng.normal(0, 250.0, n_tr))
        z_rt = (rt - rt.mean()) / rt.std()

        common = rng.standard_normal(n_tr)
        for e in range(int(rng.integers(n_elec[0], n_elec[1] + 1))):
            lwpc_e = neu[0] + elec_sd * rng.standard_normal()
            lwps_e = neu[1] + elec_sd * rng.standard_normal()
            hg = rng.uniform(0.5, 1.5) * (
                0.5 + inc * (0.3 + half_c * lwpc_e) + sw * (0.3 + half_s * lwps_e)
                + rt_coupling * z_rt + np.sqrt(common_noise) * common
                + np.sqrt(1 - common_noise) * rng.standard_normal(n_tr))
            frames.append(pd.DataFrame(dict(
                subject=subject, electrode=f"{subject}-e{e}", trial=np.arange(n_tr),
                hg=hg, rt=rt, acc=1.0, **cols)))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth)


if __name__ == '__main__':
    elec_labels, behavior, trial_df = _synthetic_brain_behavior(seed=1)

    # (1) across-subject (underpowered) correlation, with cross controls
    res = subject_level_brain_behavior(elec_labels, behavior, neural='count')
    print(f"[across] LWPC: r={res['corr_lwpc']:.2f} (p={res['p_lwpc']:.3g})  |  "
          f"LWPS: r={res['corr_lwps']:.2f} (p={res['p_lwps']:.3g})  "
          f"[n={res['n_subjects']} subjects]")
    print(f"[across] cross controls (should be weaker): "
          f"stab↔LWPS r={res['corr_cross_stab_lwps']:.2f}, "
          f"flex↔LWPC r={res['corr_cross_flex_lwpc']:.2f}")

    # (2) within-subject single-trial mixed model, matched vs cross
    for group, hg in (('LWPC', 'hg_lwpc'), ('LWPS', 'hg_lwps')):
        r = trialwise_brain_behavior(trial_df, group=group, hg_col=hg)
        print(f"[within {group}] matched slope={r['slope']:.3f} (p={r['p']:.3g})  "
              f"vs cross slope={r['slope_cross']:.3f} (p={r['p_cross']:.3g})  "
              f"specificity_ok={r['specificity_ok']}  "
              f"[n_trials={r['n_trials']}, n_subj={r['n_subjects']}]")

    # (3) continuous per-participant scores: NO planted brain-behavior link in the
    # population, only RT coupling. The raw scores correlate with behavior anyway;
    # the adjusted ones should track the planted correlation, which in a sample of
    # 24 is not exactly zero.
    long_df, truth = _synthetic_long_df(seed=1, link=0.0, rt_coupling=0.4)
    ps = participant_scores(long_df, n_splits=50)
    planted = ps['scores'].merge(truth, on='subject')
    print("[participants] planted neural-behavior r in this sample: "
          + "  ".join(f"{eff.upper()} {np.corrcoef(planted[f'{eff}_neural_true'], planted[f'{eff}_behav_true'])[0, 1]:+.2f}"
                      for eff in ('lwpc', 'lwps')))
    for variant in ('raw', 'rtadj'):
        r = participant_brain_behavior(ps, variant=variant)
        print(f"[participants, {variant}] LWPC r={r['corr_lwpc']:+.2f} "
              f"(ceiling {r['ceiling_lwpc']:.2f})  LWPS r={r['corr_lwps']:+.2f} "
              f"(ceiling {r['ceiling_lwps']:.2f})  [n={r['n_participants']}, "
              f"r_crit={r['r_crit']:.2f}]")
