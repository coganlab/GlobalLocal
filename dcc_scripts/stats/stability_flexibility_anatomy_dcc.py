#!/usr/bin/env python
"""
DCC core for A3 — anatomy of the stability/flexibility subpopulations
(`docs/stability_flexibility_analysis_plan.md` §3).

Takes the A1 electrode definition (the parametric two-way interaction ANOVA that
labels each electrode LWPC-selective (S) and/or LWPS-selective (F)) and asks the
descriptive-anatomy question on top of it: *are the distinct subpopulations in
different PLACES?* — while conditioning every claim on iEEG **coverage**, which is
clinically determined and the main confound at this layer.

Pipeline:
  1. Assemble the same long-format single-trial HG table as the A1/A2 job
     (`assemble_long_df` / `make_synthetic_df`, contrast_mode='proportion').
  2. A1: `sfs.per_electrode_anova_labels` -> per-electrode S/F flags.
  3. Map each electrode to a coarse ROI (`build_electrode_roi_map` over the shared
     `subjects_electrodes_to_ROIs_dict` + `config/rois.py`) and derive the 4-way
     group (both / S_only / F_only / neither) with `attach_roi`.
  4. Coverage: subject × ROI matrix (`build_coverage_matrix`).
  5. Coverage-conditioned enrichment test (`roi_group_enrichment_test`):
     chi-square on the group × ROI table with a WITHIN-SUBJECT permutation null,
     restricted to ROIs sampled in >= MIN_SUBJECTS subjects.
  6. Figures: ROI-group histograms (annotated with per-ROI coverage), the
     enrichment null, and a brain-surface figure via the existing vis renderer
     (falls back to the ROI histogram off-cluster).

On the SYNTHETIC path there is no ROI atlas on disk, so a ground-truth
electrode->ROI map with a planted (or null) group×ROI association is used
(`_synthetic_anatomy`) — this validates the whole path and the test's behaviour
(planted association detected; the null not manufacturing significance).

Driven by `run_stability_flexibility_anatomy_dcc.py` (wrapped by
`sbatch_stability_flexibility_anatomy_dcc.sh`). Not run directly on the cluster;
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

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.stats import stability_flexibility_anatomy as sfa
from src.analysis.utils.general_utils import resolve_lab_root, resolve_electrodes_to_keep
# reuse the SAME long-format assembly + synthetic generator as the sibling jobs
from dcc_scripts.stats.stability_flexibility_segregation_dcc import (
    assemble_long_df, make_synthetic_df)

# A3 is defined on the A1 electrodes: LWPC/LWPS interactions on window-mean HG.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cohens_d'


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def save_results(labels_with_roi, coverage, enrich, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    labels_with_roi.to_csv(os.path.join(save_dir, 'anatomy_labels_roi.csv'), index=False)
    coverage.astype(int).to_csv(os.path.join(save_dir, 'coverage_matrix.csv'))
    enrich['contingency'].to_csv(os.path.join(save_dir, 'group_roi_contingency.csv'))
    sfa.roi_group_histogram(labels_with_roi).to_csv(
        os.path.join(save_dir, 'roi_group_histogram.csv'))

    enrich_json = dict(
        rois_tested=list(enrich['rois_tested']),
        observed_stat=enrich['observed_stat'],
        p=enrich['p'],
        n_electrodes=enrich['n_electrodes'],
        per_roi_coverage={k: int(v) for k, v in enrich['per_roi_coverage'].items()})
    if 'note' in enrich:
        enrich_json['note'] = enrich['note']
    with open(os.path.join(save_dir, 'roi_enrichment.json'), 'w') as f:
        json.dump(enrich_json, f, indent=2)
    if 'null' in enrich:
        np.save(os.path.join(save_dir, 'roi_enrichment_null.npy'), enrich['null'])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_plots(labels_with_roi, coverage, enrich, save_dir):
    # (1) ROI-group histogram annotated with per-ROI coverage
    sfa.plot_roi_group_histograms(
        labels_with_roi, out_path=os.path.join(save_dir, 'roi_group_histogram.png'),
        coverage=coverage)
    plt.close('all')

    # (2) coverage heatmap + enrichment null
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cov = coverage.astype(int)
    im = ax[0].imshow(cov.to_numpy(), aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax[0].set_xticks(range(cov.shape[1])); ax[0].set_xticklabels(cov.columns, rotation=45, ha='right')
    ax[0].set_yticks(range(cov.shape[0])); ax[0].set_yticklabels(cov.index, fontsize=7)
    ax[0].set(title="A3 · subject × ROI coverage (green = covered)", xlabel="ROI", ylabel="subject")
    fig.colorbar(im, ax=ax[0], fraction=0.03)

    if 'null' in enrich and len(np.atleast_1d(enrich['null'])) > 1:
        ax[1].hist(enrich['null'], bins=40, color="#bbb")
        ax[1].axvline(enrich['observed_stat'], color="#d7191c", lw=2,
                      label=f"observed chi2 = {enrich['observed_stat']:.1f}")
        ax[1].legend()
    ax[1].set(title=f"A3 · coverage-conditioned enrichment null\n"
                    f"p = {enrich['p']:.4g}  (ROIs >= min_subjects: {len(enrich['rois_tested'])})",
              xlabel="chi-square (group × ROI, within-subject permuted)",
              ylabel="# permutations")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'anatomy_coverage_enrichment.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)

    # (3) brain-surface figure via the existing vis renderer (guarded; falls back)
    brain_path = sfa.plot_selectivity_groups_on_brain(
        labels_with_roi, os.path.join(save_dir, 'selectivity_groups_on_brain.svg'),
        coverage=coverage)
    print(f"brain figure -> {brain_path}")


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(labels_with_roi, coverage, enrich, save_dir, meta,
                  min_subjects, alpha=0.05):
    lab = labels_with_roi
    n_mapped = int(lab['roi'].notna().sum())
    sig = "significant" if enrich['p'] < alpha else "n.s."
    lines = [
        "=" * 70,
        "STABILITY vs FLEXIBILITY — A3 ANATOMY (coverage-conditioned)",
        "=" * 70,
    ]
    for k, v in meta.items():
        lines.append(f"{k:>22}: {v}")
    lines += [
        "-" * 70,
        f"electrodes: {len(lab)} total | {n_mapped} mapped to an ROI | "
        f"{len(lab) - n_mapped} outside the ROI atlas",
        "selectivity groups: "
        + "  ".join(f"{g}={int((lab['group'] == g).sum())}" for g in sfa.GROUPS),
        "-" * 70,
        f"COVERAGE-CONDITIONED ENRICHMENT (min_subjects={min_subjects}):",
        f"      ROIs tested (>= {min_subjects} subjects): {enrich['rois_tested']}",
        f"      chi-square(group × ROI) = {enrich['observed_stat']:.3f}   "
        f"permutation p = {enrich['p']:.4g}  -> {sig}",
        f"      electrodes entering the test: {enrich['n_electrodes']}",
        "      per-ROI coverage (subjects): "
        + ", ".join(f"{r}={int(c)}" for r, c in enrich['per_roi_coverage'].items()),
    ]
    if 'note' in enrich:
        lines.append(f"      NOTE: {enrich['note']}")
    lines += [
        "-" * 70,
        "group × ROI contingency (restricted to covered ROIs):",
        enrich['contingency'].to_string(),
        "=" * 70,
        "Reading: a significant test means selectivity-group membership is",
        "associated with ROI *beyond* what electrode placement (coverage) forces.",
        "Coverage is reported per ROI so no claim rests on where the grid happens to be.",
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
    min_subjects = getattr(args, 'min_subjects', 3)

    print(f"LAB_root: {LAB_root}")
    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. long-format df (+ synthetic ROI map when validating) ---------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        # synthetic labels + planted ROI map: the df is not needed to exercise the
        # anatomy path, but we assemble it too so the A1 call is on the same code.
        labels, e2r = sfa._synthetic_anatomy(
            enrichment=getattr(args, 'synthetic_enrichment', 0.6),
            seed=getattr(args, 'seed', 0))
        print(f"synthetic labels: {len(labels)} electrodes | "
              f"{labels.subject.nunique()} subjects | planted enrichment="
              f"{getattr(args, 'synthetic_enrichment', 0.6)}")
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
                    f"df is missing usable '{col}' — the A3 anatomy layer sits on "
                    "the A1 (proportion) electrode definition, which needs the "
                    "block-proportion columns.")
        df.to_csv(os.path.join(args.save_dir, 'long_df.csv'), index=False)

        # 2. A1 electrode definition ---------------------------------------------
        print("A1: per-electrode two-way interaction ANOVA (Type III, FDR across electrodes)")
        labels = sfs.per_electrode_anova_labels(
            df, alpha=alpha, contrast_mode=CONTRAST_MODE,
            require_sign=getattr(args, 'require_sign', False))

        # 3. electrode -> ROI map from the shared atlas --------------------------
        from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict
        from src.analysis.config.rois import rois_dict
        roi_save_dir = getattr(args, 'roi_dict_dir', None) or args.save_dir
        subjects_rois_dict = make_or_load_subjects_electrodes_to_ROIs_dict(
            subjects=args.subjects, task=args.task, LAB_root=LAB_root,
            save_dir=roi_save_dir)
        e2r = sfa.build_electrode_roi_map(subjects_rois_dict, rois_dict)
        print(f"electrode->ROI map: {len(e2r)} electrodes fall in a known ROI group")

    # 4. attach ROI + coverage ----------------------------------------------------
    lab_roi = sfa.attach_roi(labels, e2r)
    coverage = sfa.build_coverage_matrix(lab_roi)
    print(f"coverage: {coverage.shape[0]} subjects × {coverage.shape[1]} ROIs")

    # 5. coverage-conditioned enrichment test ------------------------------------
    print(f"A3: coverage-conditioned ROI enrichment (min_subjects={min_subjects}, "
          f"within-subject permutation null)")
    enrich = sfa.roi_group_enrichment_test(
        lab_roi, coverage, min_subjects=min_subjects,
        n_perm=args.n_perm, seed=getattr(args, 'seed', 0))

    # 6. persist + plot + summarize ----------------------------------------------
    save_results(lab_roi, coverage, enrich, args.save_dir)
    make_plots(lab_roi, coverage, enrich, args.save_dir)
    write_summary(lab_roi, coverage, enrich, args.save_dir, alpha=alpha,
                  min_subjects=min_subjects,
                  meta=dict(
                      data_source=args.data_source, task=args.task,
                      epochs_root_file=getattr(args, 'epochs_root_file', None),
                      n_subjects=coverage.shape[0],
                      window=f"[{getattr(args, 'window_tmin', None)}, "
                             f"{getattr(args, 'window_tmax', None)}]s",
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      alpha=alpha, min_subjects=min_subjects,
                      n_perm=args.n_perm, save_dir=args.save_dir))
    return dict(labels_with_roi=lab_roi, coverage=coverage, enrichment=enrich)
