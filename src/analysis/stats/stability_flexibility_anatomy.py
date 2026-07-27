"""A3 — anatomy of the stability/flexibility subpopulations (plan §3).

Descriptive anatomy of the electrode groups defined in A1 (the parametric
two-way interaction ANOVA in ``stability_flexibility_segregation`` /
``per_electrode_anova_labels``): *are the distinct subpopulations in different
PLACES?* This is the layer most exposed to **coverage bias** — iEEG coverage is
clinically determined, so a raw ROI difference can just reflect where electrodes
happen to be — so every claim here is conditioned on coverage.

What this module provides
-------------------------
- ``build_electrode_roi_map`` — flatten the shared
  ``subjects_electrodes_to_ROIs_dict`` (subject -> {channel -> Destrieux label})
  into a ``{electrode -> ROI group}`` map, using the coarse groups in
  ``src/analysis/config/rois.py``.
- ``attach_roi`` — join the A1 labels (subject, electrode, S, F) to their ROI and
  derive the 4-way ``group`` in {both, S_only, F_only, neither}.
- ``build_coverage_matrix`` — subject × ROI boolean coverage (does a subject have
  ANY electrode in ROI r?), the object every anatomical claim is conditioned on.
- ``roi_group_enrichment_test`` — is selectivity-group membership associated with
  ROI, *conditioned on coverage*? Chi-square on the group × ROI table with a
  within-subject permutation null (so the null respects both nesting and
  coverage), restricted to ROIs sampled in >= ``min_subjects`` subjects.
- ``roi_group_histogram`` / ``plot_roi_group_histograms`` — per-group ROI counts.
- ``plot_selectivity_groups_on_brain`` — thin wrapper that hands the per-group
  electrode lists to the existing Glasser surface renderer in
  ``src/analysis/vis/`` (guarded: falls back to the ROI-histogram figure when the
  heavy surface stack / template is unavailable, e.g. off the cluster).

Drop-in usage
-------------
A1 gives you ``labels`` (subject, electrode, S, F). The real electrode->ROI dict
comes from the shared utils::

    from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict
    from src.analysis.config.rois import rois_dict

    roi_dict = make_or_load_subjects_electrodes_to_ROIs_dict(subjects, task, LAB_root, save_dir)
    e2r      = build_electrode_roi_map(roi_dict, rois_dict)
    lab_roi  = attach_roi(labels, e2r)
    cover    = build_coverage_matrix(lab_roi)
    res      = roi_group_enrichment_test(lab_roi, cover, min_subjects=3)

``_synthetic_anatomy`` builds ground-truth-controlled labels + an electrode->ROI
map with a planted group×ROI association, so the whole path (and the tutorial)
runs without any data on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ["both", "S_only", "F_only", "neither"]
# colour-blind-safe, matched to the segregation summary figure (STAB / FLEX)
GROUP_COLORS = {
    "both": "#31a354",     # green  — carries both processes
    "S_only": "#2c7fb8",   # blue   — stability (LWPC) only
    "F_only": "#d95f0e",   # orange — flexibility (LWPS) only
    "neither": "#cccccc",  # grey
}


# ---------------------------------------------------------------------------
# electrode -> ROI mapping
# ---------------------------------------------------------------------------
def build_electrode_roi_map(subjects_rois_dict, rois_dict, which='default_dict',
                            electrode_fmt="{subject}-{channel}"):
    """Flatten the nested electrodes-to-ROIs dict into ``{electrode -> ROI group}``.

    Parameters
    ----------
    subjects_rois_dict : dict
        ``{subject: {'default_dict': {channel -> anatomical_label}, ...}}`` as
        returned by ``make_or_load_subjects_electrodes_to_ROIs_dict``.
    rois_dict : dict
        Coarse ROI groups -> list of anatomical (Destrieux) labels
        (``src/analysis/config/rois.py``). ROI groups may share labels; the
        FIRST group (in ``rois_dict`` insertion order) that contains a channel's
        label wins, so the mapping is deterministic. Channels whose label is in
        no group are dropped (returned only if they match some group).
    which : str
        Which per-subject sub-dict to read the channel->label map from
        ('default_dict' is the fine-grained Destrieux labelling).
    electrode_fmt : str
        How electrode ids are spelled elsewhere in the pipeline
        (``assemble_long_df`` uses ``"{subject}-{channel}"``).

    Returns
    -------
    dict : ``{electrode_id -> roi_group_name}``. Only electrodes that fall into a
        known ROI group are included.
    """
    # invert rois_dict once: anatomical label -> first ROI group that lists it
    label_to_group = {}
    for group, labels in rois_dict.items():
        for lab in labels:
            label_to_group.setdefault(lab, group)   # first group wins

    e2r = {}
    for subject, sub in subjects_rois_dict.items():
        chan_to_label = sub.get(which, {}) if isinstance(sub, dict) else {}
        for channel, anat_label in chan_to_label.items():
            group = label_to_group.get(anat_label)
            if group is None:
                continue
            e2r[electrode_fmt.format(subject=subject, channel=channel)] = group
    return e2r


def _derive_group(row):
    s, f = int(row['S']), int(row['F'])
    if s and f:
        return "both"
    if s:
        return "S_only"
    if f:
        return "F_only"
    return "neither"


def attach_roi(labels, electrodes_to_rois):
    """Add ``roi`` and the 4-way ``group`` columns to the A1 labels table.

    Parameters
    ----------
    labels : DataFrame with (at least) ``subject, electrode, S, F`` — the A1
        output (``per_electrode_anova_labels``).
    electrodes_to_rois : ``{electrode -> ROI group}`` — either the flat map from
        ``build_electrode_roi_map`` or any dict/Series keyed by electrode id.

    Returns
    -------
    DataFrame : ``labels`` + an ``roi`` column (NaN where the electrode has no
        mapped ROI) and a ``group`` column in {both, S_only, F_only, neither}.
        Electrodes without an ROI are kept (with ``roi=NaN``) so callers can
        report how many selective electrodes fall outside the ROI atlas; the
        coverage-conditioned test drops them.
    """
    out = labels.copy()
    e2r = dict(electrodes_to_rois)
    out['roi'] = out['electrode'].map(e2r)
    out['group'] = out.apply(_derive_group, axis=1)
    return out


# ---------------------------------------------------------------------------
# coverage — the object every anatomical claim is conditioned on
# ---------------------------------------------------------------------------
def build_coverage_matrix(labels_with_roi):
    """Subject × ROI boolean coverage: does subject *s* have ANY electrode in ROI *r*?

    Uses every electrode with a mapped ROI (selective or not) — coverage is about
    where the *grid* is, not where the effects are. Returns a DataFrame indexed by
    subject, columns = ROIs, values bool.
    """
    d = labels_with_roi.dropna(subset=['roi'])
    cov = pd.pivot_table(d, index='subject', columns='roi', values='electrode',
                         aggfunc='count', fill_value=0)
    return cov > 0


# ---------------------------------------------------------------------------
# the coverage-conditioned enrichment test
# ---------------------------------------------------------------------------
def _chi2_stat(table):
    """Pearson chi-square statistic sum((O-E)^2/E) for a counts matrix.

    Computed by hand (not ``scipy.stats.chi2_contingency``) so it is
    permutation-safe: cells with expected 0 contribute 0 and never raise, and the
    statistic is defined identically on every permuted table (same row/col set)."""
    O = np.asarray(table, dtype=float)
    total = O.sum()
    if total <= 0:
        return 0.0
    row = O.sum(axis=1, keepdims=True)
    col = O.sum(axis=0, keepdims=True)
    E = row @ col / total
    with np.errstate(divide='ignore', invalid='ignore'):
        contrib = np.where(E > 0, (O - E) ** 2 / E, 0.0)
    return float(contrib.sum())


def _contingency(groups, rois, group_levels, roi_levels):
    """group × ROI counts on fixed (group_levels × roi_levels) axes."""
    gi = {g: i for i, g in enumerate(group_levels)}
    ri = {r: j for j, r in enumerate(roi_levels)}
    tab = np.zeros((len(group_levels), len(roi_levels)), dtype=float)
    for g, r in zip(groups, rois):
        tab[gi[g], ri[r]] += 1
    return tab


def roi_group_enrichment_test(labels_with_roi, coverage, min_subjects: int = 3,
                              n_perm: int = 10000, seed: int = 0,
                              groups=("both", "S_only", "F_only")):
    """Is selectivity-group membership associated with ROI, conditioned on coverage?

    Parameters
    ----------
    labels_with_roi : output of ``attach_roi`` (needs ``subject, roi, group``).
    coverage : output of ``build_coverage_matrix``.
    min_subjects : keep only ROIs covered in >= this many subjects (the coverage
        condition — an ROI a single patient happens to be wired in cannot support
        a population claim).
    n_perm : within-subject permutations for the null.
    groups : which selectivity groups enter the test. Default excludes
        ``neither`` (the test is about *where the selective cells are*).

    Returns
    -------
    dict with:
        rois_tested : list[str]              — ROIs surviving the coverage filter
        observed_stat : float                — chi-square on the group × ROI table
        p : float                            — within-subject permutation p-value
        contingency : DataFrame              — group × ROI counts (restricted set)
        per_roi_coverage : Series            — n subjects covering each tested ROI
        n_electrodes : int                   — selective electrodes entering the test

    Method
    ------
    1. Restrict to ROIs with ``coverage.sum(axis=0) >= min_subjects``; drop
       electrodes outside them and outside ``groups``.
    2. Build the group × ROI contingency table; statistic = Pearson chi-square.
    3. NULL: permute the ``group`` label WITHIN EACH SUBJECT (so each subject's
       group counts and each electrode's ROI stay fixed — the null respects both
       the subject nesting and the coverage), recompute the statistic ``n_perm``
       times. ``p = (#{null >= observed} + 1) / (n_perm + 1)``.
    4. Report per-ROI coverage alongside, so a reader sees the difference isn't
       pure placement.
    """
    per_roi_cov = coverage.sum(axis=0)
    kept_rois = sorted(str(r) for r in per_roi_cov.index[per_roi_cov >= min_subjects])
    group_levels = list(groups)

    d = labels_with_roi.dropna(subset=['roi'])
    d = d[d['roi'].isin(kept_rois) & d['group'].isin(group_levels)].copy()

    if d.empty or len(kept_rois) < 2 or d['group'].nunique() < 2:
        # not enough structure to test — return a well-formed null result
        tab = _contingency(d['group'].to_numpy(), d['roi'].to_numpy(),
                           group_levels, kept_rois) if not d.empty else \
            np.zeros((len(group_levels), len(kept_rois)))
        return dict(
            rois_tested=kept_rois, observed_stat=0.0, p=1.0,
            contingency=pd.DataFrame(tab, index=group_levels, columns=kept_rois),
            per_roi_coverage=per_roi_cov.reindex(kept_rois),
            n_electrodes=int(len(d)),
            note="insufficient coverage/variation for an enrichment test")

    grp = d['group'].to_numpy()
    roi = d['roi'].to_numpy()
    subj = d['subject'].to_numpy()

    obs_tab = _contingency(grp, roi, group_levels, kept_rois)
    observed = _chi2_stat(obs_tab)

    groups_idx = [np.where(subj == s)[0] for s in np.unique(subj)]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        gp = grp.copy()
        for idx in groups_idx:                 # shuffle group WITHIN each subject
            gp[idx] = grp[rng.permutation(idx)]
        null[i] = _chi2_stat(_contingency(gp, roi, group_levels, kept_rois))
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)

    return dict(
        rois_tested=kept_rois,
        observed_stat=float(observed),
        p=float(p),
        null=null,
        contingency=pd.DataFrame(obs_tab, index=group_levels, columns=kept_rois).astype(int),
        per_roi_coverage=per_roi_cov.reindex(kept_rois).astype(int),
        n_electrodes=int(len(d)))


# ---------------------------------------------------------------------------
# descriptive ROI histograms
# ---------------------------------------------------------------------------
def roi_group_histogram(labels_with_roi, groups=("both", "S_only", "F_only")):
    """Per-group ROI membership counts as a tidy group × ROI DataFrame."""
    d = labels_with_roi.dropna(subset=['roi'])
    d = d[d['group'].isin(groups)]
    tab = (d.groupby(['group', 'roi']).size()
             .unstack(fill_value=0)
             .reindex(index=list(groups), fill_value=0))
    return tab


def plot_roi_group_histograms(labels_with_roi, out_path=None,
                              groups=("both", "S_only", "F_only"), coverage=None):
    """Grouped bar chart of ROI membership per selectivity group.

    If ``coverage`` is given, annotate each ROI with the number of subjects that
    cover it, so the reader can weight raw counts against placement. Returns the
    matplotlib Figure (and saves it when ``out_path`` is given)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tab = roi_group_histogram(labels_with_roi, groups=groups)
    rois = list(tab.columns)
    x = np.arange(len(rois))
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rois)), 4.5))
    for k, g in enumerate(groups):
        ax.bar(x + k * width, tab.loc[g].to_numpy(), width,
               label=g, color=GROUP_COLORS.get(g, None))
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    xlabels = list(rois)
    if coverage is not None:
        cov = coverage.sum(axis=0)
        xlabels = [f"{r}\n(n={int(cov.get(r, 0))} subj)" for r in rois]
    ax.set_xticklabels(xlabels, rotation=30, ha='right')
    ax.set(ylabel="# electrodes", title="A3 · ROI membership by selectivity group")
    ax.legend(title="group")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# brain surface figure (reuse the existing vis renderer; guarded)
# ---------------------------------------------------------------------------
def group_electrode_lists(labels_with_roi, groups=("both", "S_only", "F_only")):
    """{group -> list of electrode ids} — the highlight sets for the brain figure."""
    d = labels_with_roi
    return {g: d.loc[d['group'] == g, 'electrode'].tolist() for g in groups}


def plot_selectivity_groups_on_brain(labels_with_roi, out_path, coverage=None,
                                     groups=("both", "S_only", "F_only"), **vis_kwargs):
    """Render S-only / F-only / both electrodes on the cortical surface.

    Thin wrapper over the existing Glasser SVG renderer in ``src/analysis/vis/``:
    it builds the per-group electrode lists and hands them to that pipeline as
    colour-coded highlight sets rather than writing new surface code. The surface
    renderer needs MNE + PyVista + an fsaverage/Glasser template on disk, which is
    only present on the cluster; when it is unavailable this falls back to the
    ROI-histogram figure so the DCC job still produces a usable anatomical
    summary. Returns the path actually written.
    """
    lists = group_electrode_lists(labels_with_roi, groups=groups)
    try:
        # Reuse the project's surface renderer. It is a script-style module; we
        # call its highlight entrypoint if present. Kept behind a try/except so a
        # missing heavy dependency degrades gracefully rather than crashing the job.
        from src.analysis.vis import (
            brain_figure_glasser_separate_svgs_lateral_medial_view_less_bold as brainfig)
        if hasattr(brainfig, "plot_highlight_groups"):
            brainfig.plot_highlight_groups(
                lists, colors=GROUP_COLORS, out_path=out_path, **vis_kwargs)
            return out_path
        raise AttributeError(
            "vis renderer has no plot_highlight_groups entrypoint; "
            "pass the group lists to it manually on the cluster")
    except Exception as exc:  # pragma: no cover - depends on cluster-only stack
        print(f"[A3] brain-surface render unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to ROI histogram.")
        fallback = out_path
        if fallback.endswith('.svg'):
            fallback = fallback[:-4] + '_roi_hist.png'
        plot_roi_group_histograms(labels_with_roi, out_path=fallback,
                                  groups=groups, coverage=coverage)
        return fallback


# ---------------------------------------------------------------------------
# synthetic ground truth — runs the whole path with no data on disk
# ---------------------------------------------------------------------------
def _synthetic_anatomy(n_subj=12, seed=0, enrichment=0.6,
                       rois=("dlpfc", "lpfc", "acc", "parietal", "occ", "v1")):
    """Ground-truth labels + electrode->ROI map with a planted group×ROI association.

    ``both`` and ``S_only`` electrodes are biased toward frontal ROIs and
    ``F_only`` toward parietal/occipital, with strength ``enrichment`` (0 = no
    association, the null). Coverage is deliberately uneven across subjects so the
    coverage filter has something to bite on. Returns
    ``(labels, electrodes_to_rois)`` ready for ``attach_roi``."""
    rng = np.random.default_rng(seed)
    rois = list(rois)
    frontal = [r for r in rois if r in ("dlpfc", "lpfc", "acc")]
    posterior = [r for r in rois if r in ("parietal", "occ", "v1")]
    if not frontal:
        frontal = rois[:len(rois) // 2]
    if not posterior:
        posterior = rois[len(rois) // 2:]

    def pick_roi(group, covered):
        # bias frontal for S/both, posterior for F; `enrichment` mixes with uniform
        if rng.random() < enrichment:
            pool = frontal if group in ("both", "S_only") else posterior
        else:
            pool = rois
        pool = [r for r in pool if r in covered] or list(covered)
        return rng.choice(pool)

    labels_rows, e2r = [], {}
    for s in range(n_subj):
        subject = f"S{s:02d}"
        # each subject covers a random subset of ROIs (clinical coverage)
        k = rng.integers(3, len(rois) + 1)
        covered = set(rng.choice(rois, size=int(k), replace=False).tolist())
        n_elec = int(rng.integers(20, 45))
        for e in range(n_elec):
            # selectivity: ~15% S, ~15% F, ~5% both, rest neither
            u = rng.random()
            if u < 0.05:
                S, F, group = 1, 1, "both"
            elif u < 0.20:
                S, F, group = 1, 0, "S_only"
            elif u < 0.35:
                S, F, group = 0, 1, "F_only"
            else:
                S, F, group = 0, 0, "neither"
            electrode = f"{subject}-e{e}"
            roi = pick_roi(group, covered)
            e2r[electrode] = roi
            labels_rows.append(dict(subject=subject, electrode=electrode, S=S, F=F))
    labels = pd.DataFrame(labels_rows)
    return labels, e2r


if __name__ == '__main__':
    # smoke test: planted enrichment should be detected; the null (enrichment=0)
    # should not manufacture significance.
    for enr in (0.0, 0.6):
        labels, e2r = _synthetic_anatomy(enrichment=enr, seed=1)
        lab_roi = attach_roi(labels, e2r)
        cover = build_coverage_matrix(lab_roi)
        res = roi_group_enrichment_test(lab_roi, cover, min_subjects=3, n_perm=2000)
        print(f"[enrichment={enr}] ROIs tested={res['rois_tested']} "
              f"chi2={res['observed_stat']:.2f} p={res['p']:.4f} "
              f"(n_elec={res['n_electrodes']})")
        print(res['contingency'])
        print("per-ROI coverage (subjects):", dict(res['per_roi_coverage']))
        print("-" * 60)
