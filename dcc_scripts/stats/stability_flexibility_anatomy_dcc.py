#!/usr/bin/env python
"""
DCC core for A3 — anatomy of the stability/flexibility subpopulations
(`docs/analysis_guide.md` §16).

Takes a per-electrode S/F definition and asks the descriptive-anatomy question on
top of it: *are the distinct subpopulations in different PLACES?* — while
conditioning every claim on iEEG **coverage**, which is clinically determined and
the main confound at this layer.

Two electrode definitions (`LABEL_SOURCE`)
------------------------------------------
`a1` (default)
    The parametric two-way interaction ANOVA on window-mean HG
    (`sfs.per_electrode_anova_labels`). Needs the epoched data, so it loads and
    assembles the long-format table first.
`power_traces`
    The within-electrode **windowed ANOVA with cluster correction** run by
    `run_power_traces_dcc.py`, read back through
    `power_traces_conjunction.electrode_labels`. This is the more sensitive
    detector for transient interactions, and it needs NO epoched data — it reads
    the finished run directories — so the job is fast and can run anywhere the
    run dirs are visible.

Pipeline:
  1. Per-electrode S/F labels from whichever source above.
  2. Map each electrode to BOTH anatomical levels: the coarse ROI group
     (`build_electrode_roi_map` + `config/rois.py`) and the raw Destrieux label
     (`build_electrode_anat_map`), then derive the 4-way group
     (both / S_only / F_only / neither) with `attach_roi`.
  3. Optional ROI restriction (`ROI_FILTER`, e.g. `lpfc`) — the same
     restrict-to-ROI option the vis / power jobs have. Applied AFTER labelling,
     so the electrode definition is untouched.
  4. Pick the anatomical level for counting + testing (`ANAT_LEVEL`):
     `group` (coarse ROIs) or `destrieux` (raw labels). `auto` picks `destrieux`
     whenever the analysis is restricted to a single ROI group, because inside
     e.g. an lpfc-only analysis every electrode's ROI is `lpfc` and a group-level
     histogram/test is vacuous.
  5. Coverage: subject × ROI matrix at that level (`build_coverage_matrix`).
  6. Coverage-conditioned enrichment test (`roi_group_enrichment_test`):
     chi-square on the group × ROI table with a WITHIN-SUBJECT permutation null,
     restricted to ROIs sampled in >= MIN_SUBJECTS subjects.
  7. Figures: ROI-group histograms at both levels (annotated with per-ROI
     coverage), the enrichment null, and per-group electrodes on the fsaverage
     brain via the shared `plot_on_average` renderer (falls back to the ROI
     histogram when the surface stack isn't available).

On the SYNTHETIC path there is no ROI atlas on disk, so a ground-truth
electrode->ROI map with a planted (or null) group×ROI association is used
(`_synthetic_anatomy`, including fake Destrieux sublabels) — this validates the
whole path and the test's behaviour (planted association detected; the null not
manufacturing significance).

Two arms (`ARM`)
----------------
`categorical` (default)
    Everything above: binary S/F labels -> 4-way group -> group×ROI enrichment.
`continuous`
    Plan §5–§7 (`docs/analysis_plan_concurrent_regulation.md`): the per-electrode
    LWPC and LWPS *scores* on disjoint trial halves instead of flags, on an
    anatomically- (not effect-) defined electrode set. Tests
    `delta = lwpc_s - lwps_s` against ROI and against MNI coordinates with a
    within-electrode effect-label swap null, reports the §5.4 noise ceiling and
    the §9.2 leave-one-subject-out sweep next to them, and writes the five §6
    brain maps. No significance threshold is applied to electrodes anywhere in
    this arm.
`both`
    Run the categorical arm, then the continuous one into a `continuous/`
    subdirectory of the same save dir.

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

from src.analysis.stats import stability_flexibility_anatomy as sfa
from src.analysis.utils.general_utils import resolve_lab_root, resolve_electrodes_to_keep

# A3 is defined on the A1 electrodes: LWPC/LWPS interactions on window-mean HG.
CONTRAST_MODE = os.environ.get('CONTRAST_MODE', 'proportion')
EFFECT_MEASURE = 'cohens_d'

# column the counting/testing runs on, per ANAT_LEVEL
_LEVEL_TO_COL = {'group': 'roi', 'destrieux': 'anat'}


def resolve_anat_level(anat_level, labels_with_roi, roi_filter=None):
    """Which anatomical level to count/test on: 'roi' (groups) or 'anat' (Destrieux).

    `auto` resolves to Destrieux whenever the analysis is pinned to a single ROI
    group (explicitly via ROI_FILTER, or implicitly because only one group is
    present — e.g. labels read from an lpfc-only power_traces run). At that point
    the coarse column is constant, so a group-level histogram is one bar and the
    group×ROI test has a single column: both are vacuous.
    """
    if anat_level in _LEVEL_TO_COL:
        return _LEVEL_TO_COL[anat_level]
    if anat_level != 'auto':
        raise ValueError(f"ANAT_LEVEL must be 'auto', 'group' or 'destrieux'; "
                         f"got {anat_level!r}")
    single_roi = (roi_filter is not None
                  or labels_with_roi['roi'].dropna().nunique() <= 1)
    if single_roi and 'anat' in labels_with_roi.columns \
            and labels_with_roi['anat'].notna().any():
        return 'anat'
    return 'roi'


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
def save_results(labels_with_roi, coverage, enrich, save_dir, roi_col='roi'):
    os.makedirs(save_dir, exist_ok=True)
    labels_with_roi.to_csv(os.path.join(save_dir, 'anatomy_labels_roi.csv'), index=False)
    coverage.astype(int).to_csv(os.path.join(save_dir, 'coverage_matrix.csv'))
    enrich['contingency'].to_csv(os.path.join(save_dir, 'group_roi_contingency.csv'))

    # Counts at BOTH levels, always: the coarse ROI groups, and (when the
    # Destrieux labels are attached) the raw atlas labels. The second is what to
    # read once the analysis is restricted to one ROI group.
    sfa.roi_group_histogram(labels_with_roi).to_csv(
        os.path.join(save_dir, 'roi_group_histogram.csv'))
    if 'anat' in labels_with_roi.columns and labels_with_roi['anat'].notna().any():
        sfa.roi_group_histogram(labels_with_roi, roi_col='anat').to_csv(
            os.path.join(save_dir, 'destrieux_group_histogram.csv'))

    enrich_json = dict(
        roi_col=enrich.get('roi_col', roi_col),
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
def make_plots(labels_with_roi, coverage, enrich, save_dir, roi_col='roi',
               subjects=None, hemi='both', make_brain=True, hist_top_n=None):
    has_anat = ('anat' in labels_with_roi.columns
                and labels_with_roi['anat'].notna().any())

    # (1a) coarse ROI-group histogram, annotated with per-ROI coverage
    sfa.plot_roi_group_histograms(
        labels_with_roi, out_path=os.path.join(save_dir, 'roi_group_histogram.png'),
        coverage=coverage if roi_col == 'roi' else None)
    plt.close('all')

    # (1b) Destrieux-label histogram — the informative one inside a single ROI
    if has_anat:
        sfa.plot_roi_group_histograms(
            labels_with_roi,
            out_path=os.path.join(save_dir, 'destrieux_group_histogram.png'),
            coverage=coverage if roi_col == 'anat' else None,
            roi_col='anat', top_n=hist_top_n)
        plt.close('all')

    # (2) coverage heatmap + enrichment null
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cov = coverage.astype(int)
    im = ax[0].imshow(cov.to_numpy(), aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax[0].set_xticks(range(cov.shape[1])); ax[0].set_xticklabels(cov.columns, rotation=45, ha='right')
    ax[0].set_yticks(range(cov.shape[0])); ax[0].set_yticklabels(cov.index, fontsize=7)
    level_name = "Destrieux label" if roi_col == 'anat' else "ROI"
    ax[0].set(title=f"A3 · subject × {level_name} coverage (green = covered)",
              xlabel=level_name, ylabel="subject")
    fig.colorbar(im, ax=ax[0], fraction=0.03)

    if 'null' in enrich and len(np.atleast_1d(enrich['null'])) > 1:
        ax[1].hist(enrich['null'], bins=40, color="#bbb")
        ax[1].axvline(enrich['observed_stat'], color="#d7191c", lw=2,
                      label=f"observed chi2 = {enrich['observed_stat']:.1f}")
        ax[1].legend()
    ax[1].set(title=f"A3 · coverage-conditioned enrichment null\n"
                    f"p = {enrich['p']:.4g}  "
                    f"({level_name}s >= min_subjects: {len(enrich['rois_tested'])})",
              xlabel=f"chi-square (group × {level_name}, within-subject permuted)",
              ylabel="# permutations")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'anatomy_coverage_enrichment.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)

    # (3) the selective electrodes themselves, on the fsaverage brain, one colour
    #     per selectivity group (guarded; falls back to the histogram off-cluster)
    brain = {}
    if make_brain:
        brain = sfa.plot_selectivity_groups_on_brain(
            labels_with_roi,
            os.path.join(save_dir, 'selectivity_groups_on_brain.png'),
            coverage=coverage, subjects=subjects, hemi=hemi, roi_col=roi_col)
        print(f"brain figure -> {brain.get('combined')}")
    return brain


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(labels_with_roi, coverage, enrich, save_dir, meta,
                  min_subjects, alpha=0.05, roi_col='roi', brain=None):
    lab = labels_with_roi
    n_mapped = int(lab['roi'].notna().sum())
    level_name = "Destrieux label" if roi_col == 'anat' else "ROI"
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
        f"electrodes: {len(lab)} total | {n_mapped} mapped to an ROI group | "
        f"{len(lab) - n_mapped} outside the ROI atlas",
        "selectivity groups: "
        + "  ".join(f"{g}={int((lab['group'] == g).sum())}" for g in sfa.GROUPS),
    ]
    if 'anat' in lab.columns:
        n_anat = int(lab['anat'].notna().sum())
        lines.append(f"Destrieux labels attached to {n_anat}/{len(lab)} electrodes "
                     f"({lab['anat'].nunique()} distinct labels)")
    lines += [
        "-" * 70,
        f"COVERAGE-CONDITIONED ENRICHMENT at the {level_name} level "
        f"(min_subjects={min_subjects}):",
        f"      {level_name}s tested (>= {min_subjects} subjects): "
        f"{enrich['rois_tested']}",
        f"      chi-square(group × {level_name}) = {enrich['observed_stat']:.3f}   "
        f"permutation p = {enrich['p']:.4g}  -> {sig}",
        f"      electrodes entering the test: {enrich['n_electrodes']}",
        f"      per-{level_name} coverage (subjects): "
        + ", ".join(f"{r}={int(c)}" for r, c in enrich['per_roi_coverage'].items()),
    ]
    if 'note' in enrich:
        lines.append(f"      NOTE: {enrich['note']}")
    lines += [
        "-" * 70,
        f"group × {level_name} contingency (restricted to covered {level_name}s):",
        enrich['contingency'].to_string(),
    ]

    # Per-group counts by raw Destrieux label — the table to read when the
    # analysis is pinned to a single ROI group (where the coarse column is
    # constant and says nothing).
    if 'anat' in lab.columns and lab['anat'].notna().any():
        lines += [
            "-" * 70,
            "per-group counts by RAW DESTRIEUX LABEL (uncorrected counts, "
            "not the test):",
            sfa.roi_group_histogram(lab, roi_col='anat').to_string(),
        ]

    if brain:
        lines += ["-" * 70]
        if brain.get('fallback'):
            lines.append(f"brain figure: NOT rendered ({brain.get('error')}); "
                         f"wrote {brain.get('combined')} instead")
        else:
            lines.append(f"brain figure: {brain.get('combined')}")
            for g, path in (brain.get('per_group') or {}).items():
                lines.append(f"      {g}: {path}")

    lines += [
        "=" * 70,
        f"Reading: a significant test means selectivity-group membership is",
        f"associated with {level_name} *beyond* what electrode placement",
        "(coverage) forces. Coverage is reported per "
        f"{level_name} so no claim rests on where the grid happens to be.",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def load_a1_labels(args, LAB_root, alpha):
    """A1 electrode definition: assemble the long df, fit the per-electrode ANOVA.

    Imported lazily so the `power_traces` route — which reads finished run dirs
    and never touches epoched data — doesn't pay for the epoch-loading stack.
    """
    from src.analysis.stats import stability_flexibility_segregation as sfs
    from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df
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

    print("A1: per-electrode two-way interaction ANOVA (Type III, FDR across electrodes)")
    return sfs.per_electrode_anova_labels(
        df, alpha=alpha, contrast_mode=CONTRAST_MODE,
        fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'))


def load_power_traces_labels(args, alpha):
    """S/F labels from finished `power_traces` within-electrode ANOVA runs.

    Reads the runs through `power_traces_conjunction.electrode_labels`, which
    pivots the four interactions onto one row per electrode and emits exactly the
    `subject, electrode, S, F` contract the anatomy layer needs. No epoched data
    is loaded — the significance decision was made by the cluster-corrected run.

    `args.pt_runs` is either a single run directory (a 4-factor
    `stimulus_experiment_conditions` run containing all four interactions) or a
    dict {'CPC': dir, 'SPS': dir, ...} when the interactions were run separately.
    """
    from src.analysis.stats import power_traces_conjunction as ptc

    runs = args.pt_runs
    correction = getattr(args, 'pt_correction', 'fdr_bh')
    pt_alpha = getattr(args, 'pt_alpha', None) or alpha
    print(f"power_traces label source: correction={correction} alpha={pt_alpha} "
          f"anova_roi={getattr(args, 'pt_roi', None)}")
    print(f"  runs: {runs}")
    labels = ptc.electrode_labels(
        runs, roi=getattr(args, 'pt_roi', None), alpha=pt_alpha,
        correction=correction, require_all=getattr(args, 'pt_require_all', True))
    print(f"power_traces labels: {len(labels)} electrodes | "
          f"{labels.subject.nunique()} subjects | "
          f"S={int(labels['S'].sum())} F={int(labels['F'].sum())} "
          f"(dropped {labels.attrs.get('n_dropped', 0)} electrodes missing from "
          f"some run)")
    funnel = pd.DataFrame(labels.attrs.get('label_funnel', []))
    if not funnel.empty:
        print("power_traces label funnel:")
        print(funnel.to_string(index=False))
    return labels


def load_anatomy_maps(roi_filter=None, roi_dict_dir=None):
    """``(electrode->ROI group, electrode->Destrieux label)`` from the shared atlas.

    Shared by the categorical and the continuous arm — the join and the
    overlapping-ROI bookkeeping are the fiddly parts and they only want writing
    once.
    """
    from src.analysis.utils.general_utils import load_existing_subjects_electrodes_to_ROIs_dict
    from src.analysis.config.rois import rois_dict
    # ROI_DICT_DIR overrides; otherwise fall back to the checked-in dict in
    # src/analysis/config. Never rebuild it here - that needs the ECoG_Recon
    # FreeSurfer files, which the cluster doesn't have.
    subjects_rois_dict = load_existing_subjects_electrodes_to_ROIs_dict(
        save_dir=roi_dict_dir)
    # The ROI groups OVERLAP (dlpfc/lpfc share G_front_middle, S_front_inf, ...;
    # occ/v1 share S_calcarine, ...) and build_electrode_roi_map resolves ties
    # first-group-wins. So when the analysis is scoped to one ROI, subset the
    # dict FIRST — otherwise an earlier group (dlpfc) claims the shared labels
    # and the lpfc filter keeps only lpfc's exclusive ones.
    roi_groups = (sfa.subset_rois_dict(rois_dict, roi_filter)
                  if roi_filter is not None else rois_dict)
    if roi_filter is not None:
        print(f"ROI map restricted to {list(roi_groups)} "
              f"(so shared Destrieux labels are not claimed by another group)")
    e2r = sfa.build_electrode_roi_map(subjects_rois_dict, roi_groups)
    e2a = sfa.build_electrode_anat_map(subjects_rois_dict)
    print(f"electrode->ROI map: {len(e2r)} electrodes fall in a known ROI group; "
          f"electrode->Destrieux map: {len(e2a)} electrodes")
    return e2r, e2a


def main_categorical(args):
    alpha = getattr(args, 'alpha', 0.05)
    min_subjects = getattr(args, 'min_subjects', 3)
    label_source = getattr(args, 'label_source', 'a1')
    roi_filter = getattr(args, 'roi_filter', None) or None
    anat_level = getattr(args, 'anat_level', 'auto')
    if label_source not in ('a1', 'power_traces'):
        raise ValueError("label_source must be 'a1' or 'power_traces'; "
                         f"got {label_source!r}")

    # The power_traces route reads finished run dirs; it needs no LAB_root, and
    # resolving one can fail on a machine with no Box/cluster mount.
    LAB_root = None
    if args.data_source != 'synthetic' and label_source == 'a1':
        LAB_root = resolve_lab_root(args.LAB_root)
        print(f"LAB_root: {LAB_root}")
    print(f"label source: {label_source} | ROI filter: {roi_filter or 'none (whole brain)'} "
          f"| anatomical level: {anat_level}")
    print(f"contrast_mode: {CONTRAST_MODE} | effect_measure: {EFFECT_MEASURE} | fdr_correction: {getattr(args, 'fdr_correction', 'fdr_bh')}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. per-electrode S/F labels (+ electrode -> anatomy maps) -------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        # Ground-truth labels + planted ROI map, plus fake Destrieux sublabels so
        # the ROI-restricted (Destrieux-level) path is exercised too.
        labels, e2r, e2a = sfa._synthetic_anatomy(
            enrichment=getattr(args, 'synthetic_enrichment', 0.6),
            seed=getattr(args, 'seed', 0), return_anat=True)
        print(f"synthetic labels: {len(labels)} electrodes | "
              f"{labels.subject.nunique()} subjects | planted enrichment="
              f"{getattr(args, 'synthetic_enrichment', 0.6)}")
    else:
        if label_source == 'a1':
            print("DATA SOURCE: real epoched data (A1 electrode definition)")
            labels = load_a1_labels(args, LAB_root, alpha)
        else:
            print("DATA SOURCE: finished power_traces runs "
                  "(cluster-corrected electrode definition)")
            labels = load_power_traces_labels(args, alpha)
        labels.to_csv(os.path.join(args.save_dir, 'electrode_labels.csv'),
                      index=False)
        funnel = pd.DataFrame(labels.attrs.get('label_funnel', []))
        if not funnel.empty:
            funnel.to_csv(os.path.join(args.save_dir, 'label_funnel.csv'),
                          index=False)

        # electrode -> anatomy maps from the shared atlas, at BOTH levels -------
        e2r, e2a = load_anatomy_maps(roi_filter,
                                     getattr(args, 'roi_dict_dir', None))

    # 2. attach both anatomical levels -------------------------------------------
    lab_roi = sfa.attach_roi(labels, e2r, electrodes_to_anat=e2a)

    # 3. optional ROI restriction (the lpfc-only analysis) -----------------------
    lab_roi = sfa.restrict_to_roi(lab_roi, roi_filter)
    if roi_filter is not None and lab_roi.empty:
        raise RuntimeError(
            f"ROI filter {roi_filter!r} left no electrodes. Check it is a key of "
            "src/analysis/config/rois.py and that the electrode ids in the labels "
            "match the ROI dict's '{subject}-{channel}' spelling.")

    # 4. anatomical level + coverage ---------------------------------------------
    roi_col = resolve_anat_level(anat_level, lab_roi, roi_filter)
    coverage = sfa.build_coverage_matrix(lab_roi, roi_col=roi_col)
    print(f"counting/testing on '{roi_col}' "
          f"({'raw Destrieux labels' if roi_col == 'anat' else 'coarse ROI groups'})")
    print(f"coverage: {coverage.shape[0]} subjects × {coverage.shape[1]} "
          f"{'labels' if roi_col == 'anat' else 'ROIs'}")

    # 5. coverage-conditioned enrichment test ------------------------------------
    print(f"A3: coverage-conditioned enrichment (min_subjects={min_subjects}, "
          f"within-subject permutation null)")
    enrich = sfa.roi_group_enrichment_test(
        lab_roi, coverage, min_subjects=min_subjects, roi_col=roi_col,
        n_perm=args.n_perm, seed=getattr(args, 'seed', 0))

    # 6. persist + plot + summarize ----------------------------------------------
    save_results(lab_roi, coverage, enrich, args.save_dir, roi_col=roi_col)
    # Index space for the brain figure: the real subject list on real data (so
    # the indices are stable across figures), derived from the labels on the
    # synthetic path (whose subject ids are made up and match no recon).
    brain_subjects = getattr(args, 'brain_subjects', None)
    if brain_subjects is None and args.data_source != 'synthetic':
        brain_subjects = args.subjects
    brain = make_plots(lab_roi, coverage, enrich, args.save_dir, roi_col=roi_col,
                       subjects=brain_subjects,
                       hemi=getattr(args, 'brain_hemi', 'both'),
                       make_brain=getattr(args, 'make_brain', True),
                       hist_top_n=getattr(args, 'hist_top_n', None))
    write_summary(lab_roi, coverage, enrich, args.save_dir, alpha=alpha,
                  min_subjects=min_subjects, roi_col=roi_col, brain=brain,
                  meta=dict(
                      data_source=args.data_source,
                      label_source=label_source,
                      pt_runs=getattr(args, 'pt_runs', None) if label_source == 'power_traces' else None,
                      pt_correction=getattr(args, 'pt_correction', None) if label_source == 'power_traces' else None,
                      roi_filter=roi_filter, anatomical_level=roi_col,
                      task=args.task,
                      epochs_root_file=getattr(args, 'epochs_root_file', None),
                      n_subjects=coverage.shape[0],
                      window=f"[{getattr(args, 'window_tmin', None)}, "
                             f"{getattr(args, 'window_tmax', None)}]s",
                      contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
                      fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'),
                      alpha=alpha, min_subjects=min_subjects,
                      n_perm=args.n_perm, save_dir=args.save_dir))
    return dict(labels_with_roi=lab_roi, coverage=coverage, enrichment=enrich,
                roi_col=roi_col, brain=brain)


# ---------------------------------------------------------------------------
# the CONTINUOUS arm (plan §5–§7): per-electrode LWPC/LWPS scores -> anatomy
# ---------------------------------------------------------------------------
def load_scores(args, LAB_root):
    """``(scores, per_split)`` for the continuous arm. Three routes, cheapest first.

    ``SCORES_CSV``   reuse a finished segregation run's ``electrodes.csv`` (it
                     already carries ``subject, electrode, x, y, resp``). Pair it
                     with ``PER_SPLIT_CSV`` — the per-split table is what the
                     §5.4 noise ceiling and the ``min_elec`` sweep are computed
                     from, and neither is optional in a write-up.
    (compute)        assemble the long df from the epoched data and score it here
                     with ``compute_sensitivities_per_split``. This is the
                     expensive route (N_SPLITS x 4 effect evaluations per
                     electrode); both tables are written out so the next run can
                     take the CSV route.

    Either way the scores are the DISJOINT-HALF ones: LWPC on one half of an
    electrode's trials, LWPS on the other, averaged over splits.
    """
    scores_csv = getattr(args, 'scores_csv', None)
    per_split_csv = getattr(args, 'per_split_csv', None)
    if scores_csv:
        print(f"scores from {scores_csv}")
        scores = pd.read_csv(scores_csv)
        per_split = pd.read_csv(per_split_csv) if per_split_csv else None
        if per_split is None:
            print("  NOTE: no PER_SPLIT_CSV — the §5.4 noise ceiling and the "
                  "min_elec sweep will be skipped. A spatial correlation without "
                  "its ceiling is not reportable.")
        return scores, per_split

    from src.analysis.stats import stability_flexibility_segregation as sfs
    from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df
    from src.analysis.utils.general_utils import load_HG_ev1_rescaled_per_subject

    subjects_epochs = load_HG_ev1_rescaled_per_subject(
        subjects=args.subjects, epochs_root_file=args.epochs_root_file,
        task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
    keep = resolve_electrodes_to_keep(args, LAB_root)
    df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                          electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
    print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
          f"{df.electrode.nunique()} electrodes")

    n_splits = int(getattr(args, 'n_splits', 200))
    print(f"scoring LWPC/LWPS on {n_splits} disjoint half-splits "
          f"(contrast_mode={CONTRAST_MODE}, effect_measure={EFFECT_MEASURE})")
    per_split = sfs.compute_sensitivities_per_split(
        df, n_splits=n_splits, seed=getattr(args, 'seed', 0),
        contrast_mode=CONTRAST_MODE, effect_measure=EFFECT_MEASURE,
        alpha=getattr(args, 'alpha', 0.05))
    scores = sfs.add_responsiveness(sfs.average_over_splits(per_split), df,
                                    getattr(args, 'responsiveness', None))
    return scores, per_split


def run_score_anatomy(args):
    """Plan §5–§7 end to end: scores -> anatomy, with the ceiling and the leverage."""
    from src.analysis.stats import segregation_scatter as scat

    roi_filter = getattr(args, 'roi_filter', None) or None
    min_subjects = getattr(args, 'min_subjects', 3)
    n_perm = int(getattr(args, 'n_perm', 10000))
    seed = int(getattr(args, 'seed', 0))
    save_dir = os.path.join(args.save_dir, 'continuous')
    os.makedirs(save_dir, exist_ok=True)

    # 1. scores + the three anatomy maps -----------------------------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        gradient = getattr(args, 'synthetic_enrichment', 0.6)
        scores, e2r, e2a, e2c = sfa._synthetic_scores(gradient=gradient, seed=seed)
        # there are no trials to split on this route, so fake the per-split table
        # too — otherwise the §5.4 ceiling and the min_elec sweep never run and
        # the dry run does not validate the path it is there to validate
        per_split = sfa._synthetic_per_split(scores, n_splits=20, seed=seed)
        print(f"synthetic scores: {len(scores)} electrodes | "
              f"{scores.subject.nunique()} subjects | planted gradient={gradient}")
    else:
        LAB_root = (None if getattr(args, 'scores_csv', None)
                    else resolve_lab_root(args.LAB_root))
        scores, per_split = load_scores(args, LAB_root)
        e2r, e2a = load_anatomy_maps(roi_filter, getattr(args, 'roi_dict_dir', None))
        # Coordinates are only needed by the §5.2-continuous and §7 panels, and
        # they need the recon files; a partial or empty map just skips those.
        e2c = sfa.build_electrode_coord_map(args.subjects) \
            if getattr(args, 'use_coords', True) else {}
        print(f"electrode->MNI map: {len(e2c)} electrodes")

    scores.to_csv(os.path.join(save_dir, 'scores.csv'), index=False)
    if per_split is not None:
        per_split.to_csv(os.path.join(save_dir, 'per_split.csv'), index=False)

    tab = sfa.attach_scores(scores, e2r, electrodes_to_anat=e2a,
                            electrodes_to_coords=e2c or None)
    tab = sfa.restrict_to_roi(tab, roi_filter)
    if tab.empty:
        raise RuntimeError(f"ROI filter {roi_filter!r} left no electrodes.")
    roi_col = resolve_anat_level(getattr(args, 'anat_level', 'auto'), tab, roi_filter)
    coverage = sfa.build_coverage_matrix(tab, roi_col=roi_col)
    print(f"counting/testing on '{roi_col}' | coverage: {coverage.shape[0]} "
          f"subjects × {coverage.shape[1]} "
          f"{'labels' if roi_col == 'anat' else 'ROIs'}")

    # 2. §5.2 primary: delta ~ roi, within-electrode effect-label swap null -------
    print(f"§5.2 relative-score ROI test (min_subjects={min_subjects}, "
          f"n_perm={n_perm}, within-electrode swap null)")
    roi_res = sfa.relative_score_roi_test(
        tab, coverage, min_subjects=min_subjects, n_perm=n_perm, seed=seed,
        roi_col=roi_col)

    # §9.2: the same statistic with each subject dropped. Fewer permutations —
    # this is a leverage check on the estimate, not a second round of inference.
    loso = sfa.leave_one_subject_out(
        lambda t: sfa.relative_score_roi_test(
            t, coverage, min_subjects=min_subjects, roi_col=roi_col,
            n_perm=max(1000, n_perm // 10), seed=seed), tab)

    # 3. §5.2 secondary: delta ~ MNI coordinates, per hemisphere ------------------
    has_coords = 'mni_y' in tab.columns and tab['mni_y'].notna().any()
    coord_res = centers = None
    if has_coords:
        print("§5.2 coordinate test (delta ~ y + z + x + resp + subject), "
              "per hemisphere")
        coord_res = sfa.relative_score_coordinate_test(tab, n_perm=n_perm, seed=seed)
        print("§7 per-subject weighted medoids (descriptive)")
        centers = sfa.score_centers_per_subject(tab, n_perm=n_perm, seed=seed)
    else:
        print("no MNI coordinates — skipping the coordinate test and the "
              "centroid panel (both are secondary; §5.2-categorical is primary)")

    # 4. §5.4 the ceiling + §5.1 the min_elec sweep + the pooled/within-subject r -
    ceiling = {}
    if per_split is not None:
        from src.analysis.stats import stability_flexibility_segregation as sfs
        parcels = dict(zip(tab['electrode'].astype(str), tab[roi_col]))
        ceiling['electrode'] = sfa.map_reliability(per_split)
        ceiling['parcel'] = sfa.map_reliability(per_split, parcels=parcels)
        resp = tab.drop_duplicates('electrode').set_index('electrode')['resp'] \
            if 'resp' in tab.columns else None
        if resp is not None:
            # §5.1: min_elec drops WHOLE SUBJECTS, so it moves the effective N
            # more than any of the scaling choices. Sweep it and report it.
            rows = []
            for m in (1, 2, 3):
                try:
                    r = sfs.split_resolved_corr(per_split, resp, min_elec=m,
                                                n_perm=2000, seed=seed)
                    rows.append(dict(min_elec=m, corr=r['corr'], p=r['p'],
                                     n_electrodes=r['n_electrodes'],
                                     n_subjects=r['n_subjects'],
                                     reliability_x=r['reliability_x'],
                                     reliability_y=r['reliability_y'],
                                     corr_noise_corrected=r['corr_noise_corrected']))
                except Exception as exc:
                    rows.append(dict(min_elec=m, corr=np.nan,
                                     note=f"{type(exc).__name__}: {exc}"))
            ceiling['min_elec_sweep'] = pd.DataFrame(rows)

    # The pooled vs within-subject correlation of the two scores (plan §5.1: if
    # they agree, say so in Methods and pool). Free — the diagnostics already
    # exist and take the column names as an argument.
    scatter_diag = scat.joint_scatter_diagnostics(
        tab, value_cols=('lwpc_s', 'lwps_s'))

    # 5. figures ------------------------------------------------------------------
    # The §2.5 joint scatter, on the pooled-scaled scores this arm tests, with
    # the noise ceiling annotated on it — same figure and same diagnostics the
    # segregation run draws, so the two are directly comparable.
    note = None
    if ceiling.get('electrode'):
        c = ceiling['electrode']
        note = (f"split-half ceiling: LWPC {c['reliability_lwpc']:+.3f} / LWPS "
                f"{c['reliability_lwps']:+.3f}    LWPC-vs-LWPS spatial r = "
                f"{c['between']:+.3f} (noise-corrected "
                f"{c['between_noise_corrected']:+.3f})")
    fig, _ = scat.plot_joint_scatter(
        tab, save_path=os.path.join(save_dir, 'joint_scatter.png'),
        value_cols=('lwpc_s', 'lwps_s'), diagnostics=scatter_diag,
        xlabel="x = LWPC  (pooled-scaled, disjoint half)",
        ylabel="y = LWPS  (pooled-scaled, disjoint half)", annotate=note)
    plt.close(fig)

    sfa.plot_score_by_roi(tab, out_path=os.path.join(save_dir, 'delta_by_roi.png'),
                          roi_col=roi_col, coverage=coverage,
                          rois=roi_res['rois_tested'] or None)
    plt.close('all')
    maps = {}
    if getattr(args, 'make_brain', True):
        maps = sfa.plot_score_maps(
            tab, save_dir, subjects=getattr(args, 'brain_subjects', None) or args.subjects,
            hemi=getattr(args, 'brain_hemi', 'both'), roi_col=roi_col,
            coverage=coverage)
        plt.close('all')

    # 6. persist + summarise ------------------------------------------------------
    tab.to_csv(os.path.join(save_dir, 'scores_with_anatomy.csv'), index=False)
    coverage.astype(int).to_csv(os.path.join(save_dir, 'coverage_matrix.csv'))
    roi_res['per_roi'].to_csv(os.path.join(save_dir, 'delta_per_roi.csv'), index=False)
    loso.to_csv(os.path.join(save_dir, 'delta_roi_loso.csv'), index=False)
    if centers is not None and len(centers.get('per_group', [])):
        centers['per_group'].to_csv(os.path.join(save_dir, 'score_centers.csv'),
                                    index=False)
    if 'min_elec_sweep' in ceiling:
        ceiling['min_elec_sweep'].to_csv(
            os.path.join(save_dir, 'min_elec_sweep.csv'), index=False)

    summary = dict(
        roi_col=roi_col, value_col='delta',
        rois_tested=list(roi_res['rois_tested']),
        F=roi_res['observed_stat'], p=roi_res['p'],
        n_electrodes=roi_res['n_electrodes'], n_subjects=roi_res['n_subjects'],
        per_roi_coverage={str(k): int(v) for k, v in roi_res['per_roi_coverage'].items()},
        loso_p_range=[float(loso['p'].min()), float(loso['p'].max())] if 'p' in loso else None,
        coordinates={h: dict(F=r['observed_stat'], p=r['p'],
                             n_electrodes=r['n_electrodes'],
                             slopes=r['slopes'].to_dict(orient='records'))
                     for h, r in (coord_res or {}).items()},
        centers=({k: v for k, v in centers.items() if k != 'per_group'}
                 if centers else None),
        ceiling={k: v for k, v in ceiling.items() if k != 'min_elec_sweep'},
        scatter_correlation=dict(corr=scatter_diag['corr'],
                                 corr_within_subject=scatter_diag['corr_within_subject'],
                                 flags=scatter_diag['flags']),
        maps={k: v.get('combined') for k, v in maps.items()})
    with open(os.path.join(save_dir, 'score_anatomy.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    write_score_summary(tab, roi_res, loso, coord_res, centers, ceiling,
                        scatter_diag, save_dir, roi_col=roi_col,
                        alpha=getattr(args, 'alpha', 0.05))
    return dict(scores=tab, coverage=coverage, roi_test=roi_res, loso=loso,
                coordinate_test=coord_res, centers=centers, ceiling=ceiling,
                scatter_diagnostics=scatter_diag, maps=maps, save_dir=save_dir)


def write_score_summary(tab, roi_res, loso, coord_res, centers, ceiling,
                        scatter_diag, save_dir, roi_col='roi', alpha=0.05):
    """The continuous arm's `summary.txt`, written to be read top to bottom."""
    level = "Destrieux label" if roi_col == 'anat' else "ROI"
    sig = "significant" if roi_res['p'] < alpha else "n.s."
    lines = [
        "=" * 70,
        "N4 — CONTINUOUS SCORES -> ANATOMY (plan §5–§7)",
        "=" * 70,
        f"electrodes: {len(tab)} | mapped to an ROI: {int(tab['roi'].notna().sum())} "
        f"| with MNI coords: {int(tab['mni_y'].notna().sum()) if 'mni_y' in tab else 0}",
        "scores: LWPC (x) and LWPS (y) on DISJOINT trial halves, each an "
        "equal-cell-weight",
        "  difference-of-differences / pooled within-cell SD, scaled by ONE pooled "
        "factor per effect.",
        "  delta = lwpc_s - lwps_s  (>0 = LWPC-dominant electrode)",
        "-" * 70,
        f"§5.2 PRIMARY — delta ~ {level} + responsiveness + (1|subject), "
        f"within-electrode swap null:",
        f"      F = {roi_res['observed_stat']:.3f}   permutation p = {roi_res['p']:.4g}"
        f"  -> {sig}",
        f"      {level}s tested (coverage-filtered): {roi_res['rois_tested']}",
        f"      electrodes = {roi_res['n_electrodes']}, subjects = {roi_res['n_subjects']}",
        roi_res['per_roi'].to_string(index=False),
    ]
    if 'note' in roi_res:
        lines.append(f"      NOTE: {roi_res['note']}")
    if 'observed_stat' in loso and len(loso) > 1:
        # Leverage is read off the STATISTIC, not the p-value: p saturates at the
        # permutation floor, so a result carried by one subject and one that is
        # not both print 0.0005 here.
        base = float(loso.loc[loso['dropped'] == '(none)', 'observed_stat'].iloc[0])
        drop = loso[loso['dropped'] != '(none)']
        shift = (drop['observed_stat'] - base).abs()
        worst = drop.loc[shift.idxmax()]
        lines += [
            "-" * 70,
            "§9.2 leverage — the same test with each subject dropped "
            "(fewer permutations):",
            f"      F = {base:.3f} on all subjects; "
            f"{drop['observed_stat'].min():.3f} to {drop['observed_stat'].max():.3f} "
            f"across the {len(drop)} leave-one-out fits",
            f"      largest shift: dropping {worst['dropped']} moves F by "
            f"{worst['observed_stat'] - base:+.3f} (p there = {worst['p']:.4g})",
            f"      p range across folds: {drop['p'].min():.4g} to {drop['p'].max():.4g}",
        ]
    if coord_res:
        lines += ["-" * 70,
                  "§5.2 SECONDARY — delta ~ MNI coordinates (slopes in delta units/mm):"]
        for h, r in coord_res.items():
            if not np.isfinite(r.get('observed_stat', np.nan)):
                lines.append(f"      [{h}] {r.get('note', 'not estimable')}")
                continue
            s = "  ".join(f"{row.axis}={row.slope_per_mm:+.5f} (p={row.p:.4g})"
                          for row in r['slopes'].itertuples())
            lines.append(f"      [{h}] F = {r['observed_stat']:.3f}  p = {r['p']:.4g}"
                         f"  (n={r['n_electrodes']})   {s}")
        lines.append("      a POSITIVE mni_y slope = LWPC dominance increases "
                     "ANTERIORLY")
    if centers and centers.get('n_groups'):
        d = centers['mean_displacement']
        lines += [
            "-" * 70,
            f"§7 DESCRIPTIVE — per-subject × hemisphere {centers['center']}s "
            f"({centers['n_groups']} groups):",
            f"      LWPC sits {d['dy']:+.1f} mm anterior, {d['dx']:+.1f} mm lateral, "
            f"{d['dz']:+.1f} mm superior to LWPS",
            f"      p (anterior) = {centers['p']['dy']:.4g}   "
            f"mean |displacement| = {centers['mean_distance']:.1f} mm",
            "      Descriptive only — the coordinate regression above makes the "
            "same claim without a centroid.",
        ]
    if not ceiling:
        lines += [
            "-" * 70,
            "§5.4 NOISE CEILING: NOT COMPUTED — no per-split table was available "
            "(pass PER_SPLIT_CSV,",
            "      or let this job score the data). A spatial correlation without "
            "its ceiling cannot",
            "      distinguish 'the two maps are distinct' from 'neither map is "
            "measured well enough",
            "      to correlate with anything'. The §5.1 min_elec sweep is "
            "missing for the same reason.",
        ]
    else:
        lines += ["-" * 70,
                  "§5.4 NOISE CEILING — what a null spatial correlation is allowed "
                  "to mean:"]
        for k in ('electrode', 'parcel'):
            c = ceiling.get(k)
            if c:
                lines.append(
                    f"      [{c['unit']}-level, n={c['n_units']}] LWPC-vs-LWPS "
                    f"r = {c['between']:+.3f}   ceiling: LWPC {c['reliability_lwpc']:+.3f}"
                    f" / LWPS {c['reliability_lwps']:+.3f}   noise-corrected "
                    f"{c['between_noise_corrected']:+.3f}")
        if 'min_elec_sweep' in ceiling:
            lines += ["      §5.1 min_elec sweep (it drops WHOLE SUBJECTS):",
                      ceiling['min_elec_sweep'].to_string(index=False)]
    lines += [
        "-" * 70,
        f"pooled LWPC–LWPS correlation r = {scatter_diag['corr']:+.3f}   "
        f"within-subject r = {scatter_diag['corr_within_subject']:+.3f}",
        "      (if these agree, say so in Methods and pool — plan §5.1)",
    ]
    for f in scatter_diag['flags']:
        lines.append(f"      !  {f}")
    lines += [
        "=" * 70,
        "Reading: the test is an effect-type × anatomy INTERACTION, because the "
        "response",
        "is the within-electrode difference between the two effects. It therefore "
        "cannot",
        "commit the difference-of-significance fallacy. Every number above is "
        "conditioned",
        "on coverage, and none of them is interpretable without the §5.4 ceiling.",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


def main(args):
    """Dispatch on ARM: the categorical S/F arm, the continuous score arm, or both."""
    arm = getattr(args, 'arm', 'categorical')
    if arm not in ('categorical', 'continuous', 'both'):
        raise ValueError(f"arm must be 'categorical', 'continuous' or 'both'; "
                         f"got {arm!r}")
    if arm == 'continuous':
        return run_score_anatomy(args)
    out = main_categorical(args)
    if arm == 'both':
        out['continuous'] = run_score_anatomy(args)
    return out
