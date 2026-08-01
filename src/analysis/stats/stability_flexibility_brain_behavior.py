"""A6 — brain–behavior correlation (plan §6).

Tie the neural selectivity (A1's LWPC / LWPS electrode groups) to the ACTUAL
behavioral control adjustment, so the substrates are shown to be *functional*,
not incidental. Two levels with very different power:

- **Across subjects** (n = subjects, low power): does a subject with more (or
  stronger) LWPC electrodes show a larger behavioral LWPC effect? And likewise
  LWPS? This is honest but underpowered at n = subjects.
- **Within subject, single-trial** (preferred, far more power): does trial-by-
  trial high-gamma in the LWPC electrode group predict the trial-by-trial
  congruency-sequence RT adjustment (and the LWPS group ↔ the switch
  adjustment)? A mixed model with a subject random effect.

The two behavioral constructs mirror the neural ones exactly:

- **LWPC (stability) behavioral magnitude** = the congruency × incongruent-
  proportion interaction on RT — how much the congruency effect (RT_i − RT_c)
  GROWS in high-incongruent-proportion blocks.
- **LWPS (flexibility) behavioral magnitude** = the switchType × switch-
  proportion interaction on RT — how much the switch cost (RT_s − RT_r) grows in
  high-switch-proportion blocks.

Both are scored as the same equal-cell-weight difference-of-differences the
segregation module uses for the neural interaction, so brain and behavior are
measured on the identical contrast. (They are the per-subject analogue of the
fixed-effect interactions in ``erin_linear_mixed_effects_model.py`` /
``combinedData.csv``; extract per-subject effects rather than inventing a new RT
contrast.)

**Specificity control (the whole point of A6).** The matched pairing
(LWPC group ↔ congruency-sequence adjustment; LWPS group ↔ switch adjustment)
should be stronger than the CROSS pairing (LWPC group ↔ switch adjustment). Both
functions report the cross pairing alongside the matched one.

``_synthetic_brain_behavior`` plants a matched across-subject correlation and a
matched within-subject single-trial coupling (both stronger than their cross
controls) so the whole path and the tutorial run with no data on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import statsmodels.formula.api as smf


# ----------------------------------------------------------------------------
# behavioral LWPC / LWPS magnitudes (per subject) from raw behavior
# ----------------------------------------------------------------------------
# The default block -> proportion map matches erin_linear_mixed_effects_model.py:
#   A,C -> 75% congruent (25% incongruent), 25% switch
#   B,D -> 25% congruent (75% incongruent), 75% switch
# (Note: in THIS design incongruent-proportion and switch-proportion are
# collinear across blocks — a real limitation of the block structure, worth
# flagging when interpreting which interaction a per-subject effect reflects.)
_BLOCK_PROPORTION_MAP = {
    'A': dict(incongruent_proportion=25.0, switch_proportion=25.0),
    'B': dict(incongruent_proportion=75.0, switch_proportion=75.0),
    'C': dict(incongruent_proportion=25.0, switch_proportion=25.0),
    'D': dict(incongruent_proportion=75.0, switch_proportion=75.0),
}


def _dod_rt(sub, cond_col, mod_col, pos, neg, rt_col='RT'):
    """Equal-cell-weight difference-of-differences of mean RT over the 2x2
    (cond × mod) cells: (pos@hi - neg@hi) - (pos@lo - neg@lo). NaN if any of the
    four cells is empty. ``mod_col`` is the proportion column; its two levels are
    taken as the df-wide max ('high') and min ('low')."""
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
    return ((cells[('pos', 'hi')] - cells[('neg', 'hi')])
            - (cells[('pos', 'lo')] - cells[('neg', 'lo')]))


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
    (difference-of-differences, in RT units; positive = the effect grows in the
    high-proportion block) plus ``n_trials``.
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
        ``lwpc``, ``lwps`` (from ``behavioral_lwpc_lwps_magnitudes``).
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
