#!/usr/bin/env python
"""
DCC core for the A1/A2 layer of the stability vs. flexibility battery: the
parametric per-electrode ANOVA electrode definition (A1) and the overlap /
conjunction inference (A2).

This is the *population-organization* answer from
`docs/stability_flexibility_analysis_plan.md` (§1 electrode definition, §2
conjunction), the parametric sibling of the continuous/CMH job in
`stability_flexibility_segregation_dcc.py`. It:

  A1. labels each electrode LWPC-selective (S) and/or LWPS-selective (F) from a
      two-way Type III interaction ANOVA on window-mean high-gamma
      (`sfs.per_electrode_anova_labels`, contrast_mode='proportion'), FDR across
      electrodes, and records the cross-interaction specificity controls.
  A2. tests whether the co-localization ('both' count) is more/less than chance
      with a within-subject permutation null (`sfs.conjunction_permutation_null`)
      and confirms the verdict is stable across selection thresholds
      (`sfs.conjunction_threshold_sweep`), pooling per subject with CMH
      (`sfs.cmh_conjunction`).

Driven by `run_stability_flexibility_anova_conjunction_dcc.py` (wrapped by
`sbatch_stability_flexibility_anova_conjunction_dcc.sh`). Not run directly on the
cluster; call `main(args)` with a populated argument namespace.

The df assembly and synthetic generator are reused verbatim from the sibling
segregation DCC core, so the input table is built identically. A1 needs the
window-MEAN scalar HG, so this job always runs with effect_measure='cohens_d'
and contrast_mode='proportion' (the LWPC/LWPS interactions).
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
from src.analysis.utils.general_utils import resolve_lab_root, resolve_electrodes_to_keep
# reuse the SAME long-format assembly + synthetic generator as the sibling job
from dcc_scripts.stats.stability_flexibility_segregation_dcc import (
    assemble_long_df, make_synthetic_df)

STAB, FLEX = "#2c7fb8", "#d95f0e"

# A1 is defined on the LWPC/LWPS interactions, scored on window-mean HG.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cohens_d'


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_results(anova_labels, conj, null, sweep, save_dir):
    """Write the A1 label table + A2 conjunction / null / sweep results."""
    os.makedirs(save_dir, exist_ok=True)
    anova_labels.to_csv(os.path.join(save_dir, 'anova_labels.csv'), index=False)
    sweep.to_csv(os.path.join(save_dir, 'threshold_sweep.csv'), index=False)

    conj_json = dict(
        mh_odds_ratio=conj['mh_odds_ratio'],
        cmh_pvalue=conj['cmh'].pvalue, cmh_statistic=conj['cmh'].statistic,
        homogeneity_pvalue=conj['homogeneity'].pvalue,
        homogeneity_statistic=conj['homogeneity'].statistic,
        pooled_table=conj['pooled_table'],
        pooled_fisher_or=conj['pooled_fisher_or'],
        pooled_fisher_p=conj['pooled_fisher_p'])
    if 'or_95ci' in conj:
        conj_json['mh_or_95ci'] = list(conj['or_95ci'])
    with open(os.path.join(save_dir, 'conjunction.json'), 'w') as f:
        json.dump(_json_safe(conj_json), f, indent=2)
    conj['per_subject'].to_csv(
        os.path.join(save_dir, 'conjunction_per_subject.csv'), index=False)

    null_json = dict(observed=null['observed'],
                     null_mean=float(np.mean(null['null'])),
                     null_std=float(np.std(null['null'])),
                     p_two_sided=null['p_two_sided'], z=null['z'],
                     n_perm=int(len(null['null'])))
    with open(os.path.join(save_dir, 'permutation_null.json'), 'w') as f:
        json.dump(_json_safe(null_json), f, indent=2)
    np.save(os.path.join(save_dir, 'permutation_null.npy'), null['null'])


# ---------------------------------------------------------------------------
# figure: A1 label classes + A2 null + threshold sweep + cross-controls
# ---------------------------------------------------------------------------
def make_plots(anova_labels, conj, null, sweep, save_dir):
    lab = anova_labels
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # A: FDR q-value scatter, colored by selectivity class
    sc = ax[0, 0].scatter(lab['q_cong'], lab['q_switch'],
                          c=lab['S'] + 2 * lab['F'], cmap="viridis", s=20, alpha=.75)
    ax[0, 0].axvline(0.05, ls='--', c='k', lw=1); ax[0, 0].axhline(0.05, ls='--', c='k', lw=1)
    ax[0, 0].set(title="A1 · FDR q-values (dashed = alpha)",
                 xlabel="q (stability / LWPC)", ylabel="q (flexibility / LWPS)",
                 xlim=(-.02, 1.02), ylim=(-.02, 1.02))

    # B: selectivity classes
    both = int(((lab.S == 1) & (lab.F == 1)).sum())
    so = int(((lab.S == 1) & (lab.F == 0)).sum())
    fo = int(((lab.S == 0) & (lab.F == 1)).sum())
    ne = int(((lab.S == 0) & (lab.F == 0)).sum())
    ax[0, 1].bar(["neither", "S only", "F only", "both"], [ne, so, fo, both],
                 color=["#ccc", STAB, FLEX, "#31a354"])
    ax[0, 1].set(title="A1 · electrode selectivity classes", ylabel="# electrodes")

    # C: pooled 2x2 with MH OR
    pooled = np.asarray(conj['pooled_table'])
    ax[0, 2].imshow(pooled, cmap="Blues")
    ax[0, 2].set_xticks([0, 1]); ax[0, 2].set_xticklabels(["F=1", "F=0"])
    ax[0, 2].set_yticks([0, 1]); ax[0, 2].set_yticklabels(["S=1", "S=0"])
    for (i, j), v in np.ndenumerate(pooled):
        ax[0, 2].text(j, i, int(v), ha="center", va="center",
                      color="white" if v > pooled.max() / 2 else "black", fontsize=13)
    ax[0, 2].set(title=f"A2 · pooled 2x2 (MH OR={conj['mh_odds_ratio']:.2f}, "
                       f"p={conj['cmh'].pvalue:.3g})")

    # D: within-subject permutation null on the 'both' count
    ax[1, 0].hist(null['null'], bins=40, color="#bbb")
    ax[1, 0].axvline(null['observed'], color="#d7191c", lw=2,
                     label=f"observed = {null['observed']}")
    ax[1, 0].legend()
    ax[1, 0].set(title=f"A2 · within-subject null on overlap\n"
                       f"p={null['p_two_sided']:.3g}, z={null['z']:+.2f}",
                 xlabel="# 'both' electrodes", ylabel="# permutations")

    # E: OR vs threshold sweep
    ax[1, 1].plot(sweep['threshold'], sweep['mh_odds_ratio'], 'o-', color="#31a354")
    ax[1, 1].axhline(1.0, ls='--', c='k', lw=1, label="OR = 1 (independence)")
    ax[1, 1].legend()
    ax[1, 1].set(title="A2 · overlap OR vs selection threshold",
                 xlabel="q-value cutoff", ylabel="MH odds ratio")

    # F: cross-control specificity (should be ~uniform / rejection near alpha)
    have_cs = 'p_cross_cs' in lab.columns
    if have_cs:
        ax[1, 2].hist(lab['p_cross_cs'].dropna(), bins=15, alpha=.6,
                      color="#8856a7", label="cong x switch_prop")
        ax[1, 2].hist(lab['p_cross_si'].dropna(), bins=15, alpha=.6,
                      color="#7fbf7b", label="switch x inc_prop")
        ax[1, 2].axvline(0.05, ls='--', c='k', lw=1)
        ax[1, 2].legend()
    ax[1, 2].set(title="A1 · cross-interaction controls\n(specificity: want ~uniform)",
                 xlabel="interaction p", ylabel="# electrodes")

    fig.tight_layout()
    fig_path = os.path.join(save_dir, 'anova_conjunction_summary.png')
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(anova_labels, conj, null, sweep, save_dir, meta, alpha=0.05):
    lab = anova_labels
    cat_dir = "shared / overlap core" if conj['mh_odds_ratio'] > 1 else "segregated"
    cat_sig = "significant" if conj['cmh'].pvalue < alpha else "n.s."
    null_dir = ("more overlap than chance" if null['observed'] > np.mean(null['null'])
                else "less overlap than chance")
    null_sig = "significant" if null['p_two_sided'] < alpha else "n.s."
    n_both = int(((lab.S == 1) & (lab.F == 1)).sum())
    lines = [
        "=" * 68,
        "STABILITY vs FLEXIBILITY — A1 ANOVA DEFINITION + A2 CONJUNCTION",
        "=" * 68,
    ]
    for key, val in meta.items():
        lines.append(f"{key:>22}: {val}")
    lines += [
        "-" * 68,
        "A1  electrode definition (Type III interaction ANOVA, FDR across elec):",
        f"      LWPC-selective (S): {int(lab.S.sum())}    "
        f"LWPS-selective (F): {int(lab.F.sum())}    both: {n_both}    "
        f"of {len(lab)} electrodes",
    ]
    if 'p_cross_cs' in lab.columns:
        cs = lab['p_cross_cs'].dropna(); si = lab['p_cross_si'].dropna()
        lines += [
            f"      cross-controls (want rejection ~alpha={alpha}): "
            f"cong x switch_prop frac<{alpha} = {np.mean(cs < alpha):.3f} ; "
            f"switch x inc_prop frac<{alpha} = {np.mean(si < alpha):.3f}",
        ]
    lines += [
        "-" * 68,
        f"A2  conjunction (CMH): MH odds ratio = {conj['mh_odds_ratio']:.4f}",
        f"      CMH p = {conj['cmh'].pvalue:.4g}   homogeneity p = {conj['homogeneity'].pvalue:.4g}",
        f"      -> {cat_dir} ({cat_sig})",
        f"A2  overlap permutation null (within-subject): observed 'both' = {null['observed']}, "
        f"null mean = {np.mean(null['null']):.2f}",
        f"      p(two-sided) = {null['p_two_sided']:.4g}   z = {null['z']:+.2f}   "
        f"-> {null_dir} ({null_sig})",
        "-" * 68,
        "A2  threshold sweep (MH OR across q-cutoffs — a robust verdict is stable):",
    ]
    for _, r in sweep.iterrows():
        lines.append(f"      q<{r['threshold']:<5}  n_S={int(r['n_S']):<4} n_F={int(r['n_F']):<4} "
                     f"n_both={int(r['n_both']):<4} OR={r['mh_odds_ratio']:.3f}  CMH p={r['cmh_p']:.3g}")
    lines += [
        "=" * 68,
        "Reading: MH OR < 1 / overlap below null -> segregation (distinct "
        "populations); OR > 1 / overlap above null -> shared core; ~1 -> independent.",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    LAB_root = resolve_lab_root(args.LAB_root)
    alpha = getattr(args, 'alpha', 0.05)

    print(f"LAB_root: {LAB_root}")
    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE} "
          f"(A1 needs window-mean scalar HG)")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. build the long-format df -------------------------------------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        df = make_synthetic_df(rho_true=args.synthetic_rho, effect_measure=EFFECT_MEASURE)
    else:
        print("DATA SOURCE: real epoched data")
        from src.analysis.utils.general_utils import load_HG_ev1_rescaled_per_subject
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)

    print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
          f"{df.electrode.nunique()} electrodes")
    for col in ('incongruent_proportion', 'switch_proportion'):
        if col not in df.columns or df[col].isna().all():
            raise RuntimeError(
                f"df is missing usable '{col}' — the A1/A2 (proportion) analysis "
                "needs the block-proportion columns. Check the epochs metadata.")
    df.to_csv(os.path.join(args.save_dir, 'long_df.csv'), index=False)

    # 2. A1 — parametric per-electrode ANOVA labels -------------------------------
    print("A1: per-electrode two-way interaction ANOVA (Type III, FDR across electrodes)")
    anova_labels = sfs.per_electrode_anova_labels(
        df, alpha=alpha, contrast_mode=CONTRAST_MODE)

    # optional cross-check vs the nonparametric definition (same balanced
    # interaction, permutation estimator) — a sanity check, not part of the result
    if getattr(args, 'crosscheck_nonparametric', False):
        print("     cross-checking against nonparametric per_electrode_labels ...")
        nonpar = sfs.per_electrode_labels(
            df, n_perm=getattr(args, 'n_perm_label', 2000), alpha=alpha,
            contrast_mode=CONTRAST_MODE)
        m = anova_labels.merge(nonpar[['electrode', 'S', 'F']], on='electrode',
                               suffixes=('_anova', '_nonpar'))
        for col in ('S', 'F'):
            a, b = m[f'{col}_anova'], m[f'{col}_nonpar']
            po = (a == b).mean()
            pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else float('nan')
            print(f"     {col}: agreement={po:.3f} kappa={kappa:+.3f} "
                  f"(anova n={int(a.sum())}, nonpar n={int(b.sum())})")
        m.to_csv(os.path.join(args.save_dir, 'anova_vs_nonparametric.csv'), index=False)

    # 3. A2 — conjunction, permutation null, threshold sweep ----------------------
    print("A2: CMH conjunction + within-subject permutation null + threshold sweep")
    conj = sfs.cmh_conjunction(anova_labels)
    null = sfs.conjunction_permutation_null(
        anova_labels, n_perm=args.n_perm_null, seed=getattr(args, 'seed', 0))

    def labels_at_qcut(qcut):
        l = anova_labels.copy()
        l['S'] = (l['q_cong'] < qcut).astype(int)
        l['F'] = (l['q_switch'] < qcut).astype(int)
        return l

    sweep = sfs.conjunction_threshold_sweep(labels_at_qcut, args.thresholds)

    # 4. persist ------------------------------------------------------------------
    save_results(anova_labels, conj, null, sweep, args.save_dir)
    make_plots(anova_labels, conj, null, sweep, args.save_dir)
    write_summary(anova_labels, conj, null, sweep, args.save_dir, alpha=alpha,
                  meta=dict(
                      data_source=args.data_source, task=args.task,
                      epochs_root_file=args.epochs_root_file,
                      n_subjects_requested=len(args.subjects),
                      window=f"[{args.window_tmin}, {args.window_tmax}]s",
                      electrodes=args.electrodes,
                      rois=(list(args.rois_dict.keys()) if args.rois_dict else 'all'),
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      alpha=alpha, n_perm_null=args.n_perm_null,
                      thresholds=list(args.thresholds), save_dir=args.save_dir))
    return dict(anova_labels=anova_labels, conjunction=conj, null=null, sweep=sweep)
