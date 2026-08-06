"""A3 — anatomy of the stability/flexibility subpopulations (plan §3).

Descriptive anatomy of the electrode groups defined upstream — either A1 (the
parametric two-way interaction ANOVA in ``stability_flexibility_segregation`` /
``per_electrode_anova_labels``) or the cluster-corrected ``power_traces`` route
(``stats/power_traces_conjunction.electrode_labels``), which emits the same
``subject, electrode, S, F`` contract. The question: *are the distinct
subpopulations in different PLACES?* This is the layer most exposed to
**coverage bias** — iEEG coverage is clinically determined, so a raw ROI
difference can just reflect where electrodes happen to be — so every claim here
is conditioned on coverage.

Two anatomical granularities
----------------------------
Every function that takes a ``roi_col`` works at either level:

- ``roi_col='roi'``  — the coarse ROI GROUPS of ``src/analysis/config/rois.py``
  (``lpfc``, ``acc``, ``parietal``, ...). The right level for a whole-brain map.
- ``roi_col='anat'`` — the raw **Destrieux** labels the recon assigns
  (``G_front_middle``, ``S_front_inf``, ...). The right level once you have
  RESTRICTED to a single group: inside an lpfc-only analysis the group column is
  constant, so counting/testing on it is vacuous, while the Destrieux labels
  still resolve gyral/sulcal substructure.

What this module provides
-------------------------
- ``build_electrode_roi_map`` — flatten the shared
  ``subjects_electrodes_to_ROIs_dict`` (subject -> {channel -> Destrieux label})
  into a ``{electrode -> ROI group}`` map, using the coarse groups in
  ``src/analysis/config/rois.py``.
- ``build_electrode_anat_map`` — the same flattening WITHOUT the grouping:
  ``{electrode -> Destrieux label}``, optionally dropping white-matter/unknown.
- ``attach_roi`` — join the labels (subject, electrode, S, F) to their ROI (and
  Destrieux label) and derive the 4-way ``group`` in
  {both, S_only, F_only, neither}.
- ``restrict_to_roi`` — keep only electrodes in one (or several) ROI groups, e.g.
  the lpfc-only analysis. Everything downstream then conditions on that subset.
- ``build_coverage_matrix`` — subject × ROI boolean coverage (does a subject have
  ANY electrode in ROI r?), the object every anatomical claim is conditioned on.
- ``roi_group_enrichment_test`` — is selectivity-group membership associated with
  ROI, *conditioned on coverage*? Chi-square on the group × ROI table with a
  within-subject permutation null (so the null respects both nesting and
  coverage), restricted to ROIs sampled in >= ``min_subjects`` subjects.
- ``roi_group_histogram`` / ``plot_roi_group_histograms`` — per-group ROI counts.
- ``plot_selectivity_groups_on_brain`` — renders the per-group electrodes on the
  fsaverage brain through the SAME renderer the rest of the lab uses
  (``vis/jim_mri.plot_on_average``, via the index helpers in
  ``dcc_scripts/vis/plot_sig_electrodes_dcc.py``), one colour per selectivity
  group. Guarded: falls back to the ROI-histogram figure when the heavy surface
  stack / recon templates are unavailable, e.g. off the cluster.

Drop-in usage
-------------
A1 (or ``power_traces_conjunction.electrode_labels``) gives you ``labels``
(subject, electrode, S, F). The real electrode->ROI dict comes from the shared
utils::

    from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict
    from src.analysis.config.rois import rois_dict

    roi_dict = make_or_load_subjects_electrodes_to_ROIs_dict(subjects, task, LAB_root, save_dir)
    e2r      = build_electrode_roi_map(roi_dict, rois_dict)
    e2a      = build_electrode_anat_map(roi_dict)
    lab_roi  = attach_roi(labels, e2r, electrodes_to_anat=e2a)

    # whole-brain: which ROI GROUP are the subpopulations in?
    cover    = build_coverage_matrix(lab_roi)
    res      = roi_group_enrichment_test(lab_roi, cover, min_subjects=3)

    # lpfc-only: which DESTRIEUX label inside lpfc are they in?
    lpfc     = restrict_to_roi(lab_roi, 'lpfc')
    cover_a  = build_coverage_matrix(lpfc, roi_col='anat')
    res_a    = roi_group_enrichment_test(lpfc, cover_a, roi_col='anat')

``_synthetic_anatomy`` builds ground-truth-controlled labels + an electrode->ROI
map with a planted group×ROI association, so the whole path (and the tutorial)
runs without any data on disk.
"""

from __future__ import annotations

import os

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

# Anatomical labels that carry no cortical location. Same list the vis pipeline
# drops in ``plot_sig_electrodes_dcc.electrode_roi_counts``, so the Destrieux
# histograms here and there count the same electrodes.
NON_CORTICAL_LABELS = ('Unknown', 'unknown', 'White-Matter', 'hypointensities')


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


def subset_rois_dict(rois_dict, names):
    """Keep only ``names`` from ``rois_dict`` — do this BEFORE building the ROI map.

    Why this exists. The groups in ``src/analysis/config/rois.py`` OVERLAP:
    ``dlpfc`` and ``lpfc`` share ``G_front_middle``, ``G_front_sup``,
    ``S_front_inf``, ``S_front_middle`` and ``S_front_sup``, and ``occ`` and
    ``v1`` share ``S_calcarine`` / ``G_cuneus`` / ``G_oc-temp_med-Lingual``.
    ``build_electrode_roi_map`` resolves that by insertion order (first group
    wins), so with the full dict ``dlpfc`` claims every shared label and an
    ``roi == 'lpfc'`` filter keeps ONLY the labels unique to lpfc (the inferior
    frontal / anterior insula ones) — silently dropping most of the electrodes
    you meant to analyse.

    Subsetting the dict first makes the ROI you asked for the only claimant, so
    ``lpfc`` gets its full label list. This is the same thing the vis pipeline
    does when it is handed an ROI-restricted ``rois_dict``.

    ``names`` may be a single name or an iterable. Unknown names raise.
    """
    wanted = [names] if isinstance(names, str) else list(names)
    missing = [n for n in wanted if n not in rois_dict]
    if missing:
        raise KeyError(f"unknown ROI group(s) {missing}; "
                       f"known: {sorted(rois_dict)}")
    return {n: list(rois_dict[n]) for n in wanted}


def build_electrode_anat_map(subjects_rois_dict, which='default_dict',
                             electrode_fmt="{subject}-{channel}",
                             drop_labels=NON_CORTICAL_LABELS):
    """Flatten the nested dict into ``{electrode -> raw Destrieux label}``.

    Same walk as :func:`build_electrode_roi_map`, minus the grouping step: the
    label the recon actually assigned is kept verbatim (``G_front_middle``,
    ``S_front_inf``, ...). This is what the histograms should count once the
    analysis has been restricted to a single ROI group — inside an lpfc-only
    analysis every electrode's ``roi`` is ``'lpfc'``, so only the Destrieux level
    still carries information.

    Parameters
    ----------
    subjects_rois_dict, which, electrode_fmt : as in ``build_electrode_roi_map``.
    drop_labels : iterable of str
        Substrings marking a label as non-cortical; matching electrodes are left
        out of the map entirely (so they show up as ``anat=NaN`` downstream
        rather than as a spurious "White-Matter" ROI). Pass ``()`` to keep them.

    Returns
    -------
    dict : ``{electrode_id -> destrieux_label}``.
    """
    drop_labels = tuple(drop_labels or ())
    e2a = {}
    for subject, sub in subjects_rois_dict.items():
        chan_to_label = sub.get(which, {}) if isinstance(sub, dict) else {}
        for channel, anat_label in chan_to_label.items():
            label = str(anat_label)
            if any(bad in label for bad in drop_labels):
                continue
            e2a[electrode_fmt.format(subject=subject, channel=channel)] = label
    return e2a


def electrode_ids(labels, electrode_fmt="{subject}-{channel}"):
    """Fully-qualified ``{subject}-{channel}`` ids for a labels table.

    The two upstream electrode definitions spell the ``electrode`` column
    differently: ``assemble_long_df`` (A1) already writes ``"{subject}-{channel}"``,
    while ``power_traces``' ``summary.csv`` writes the BARE channel name with the
    subject in its own column. The ROI dicts are keyed the first way, so join keys
    are built here: an electrode that already carries its subject prefix is passed
    through untouched, otherwise the prefix is added.

    Returns a pandas Series aligned to ``labels``.
    """
    subs = labels['subject'].astype(str)
    elecs = labels['electrode'].astype(str)
    return pd.Series(
        [e if e.startswith(f"{s}-") else electrode_fmt.format(subject=s, channel=e)
         for s, e in zip(subs, elecs)], index=labels.index, name='electrode_id')


def _derive_group(row):
    s, f = int(row['S']), int(row['F'])
    if s and f:
        return "both"
    if s:
        return "S_only"
    if f:
        return "F_only"
    return "neither"


def attach_roi(labels, electrodes_to_rois, electrodes_to_anat=None,
               electrode_fmt="{subject}-{channel}"):
    """Add ``roi`` (+ optional ``anat``) and the 4-way ``group`` columns.

    Parameters
    ----------
    labels : DataFrame with (at least) ``subject, electrode, S, F`` — the A1
        output (``per_electrode_anova_labels``) or the ``power_traces`` output
        (``power_traces_conjunction.electrode_labels``). If the table already has
        an ``roi`` column (the ``power_traces`` route carries the ANOVA's ROI),
        it is preserved as ``anova_roi`` rather than silently overwritten.
    electrodes_to_rois : ``{electrode -> ROI group}`` — either the flat map from
        ``build_electrode_roi_map`` or any dict/Series keyed by electrode id.
    electrodes_to_anat : ``{electrode -> Destrieux label}``, optional — from
        ``build_electrode_anat_map``. Adds the fine-grained ``anat`` column.
    electrode_fmt : how electrode ids are spelled in the ROI maps; used to
        reconcile the two upstream spellings (see :func:`electrode_ids`).

    Returns
    -------
    DataFrame : ``labels`` + an ``roi`` column (NaN where the electrode has no
        mapped ROI), an ``anat`` column when ``electrodes_to_anat`` is given, and
        a ``group`` column in {both, S_only, F_only, neither}.
        Electrodes without an ROI are kept (with ``roi=NaN``) so callers can
        report how many selective electrodes fall outside the ROI atlas; the
        coverage-conditioned test drops them.
    """
    out = labels.copy()
    if 'roi' in out.columns and 'anova_roi' not in out.columns:
        out = out.rename(columns={'roi': 'anova_roi'})
    ids = electrode_ids(out, electrode_fmt=electrode_fmt)
    out['roi'] = ids.map(dict(electrodes_to_rois))
    if electrodes_to_anat is not None:
        out['anat'] = ids.map(dict(electrodes_to_anat))
    out['group'] = out.apply(_derive_group, axis=1)
    return out


def restrict_to_roi(labels_with_roi, roi, roi_col='roi', verbose=True):
    """Keep only electrodes in ``roi`` (a name, or a list of names).

    The ROI restriction every other script in the repo offers (``rois_dict`` in
    the vis pipeline, ``resolve_electrodes_to_keep`` in the power/stats jobs),
    applied at the anatomy layer. Restricting here rather than upstream means the
    ELECTRODE DEFINITION is untouched — the same S/F flags are used, we just look
    at a subset of the brain.

    Two consequences worth stating in a write-up:
      * coverage and the enrichment test must be recomputed on the subset
        (``build_coverage_matrix(restricted, roi_col='anat')``), and
      * a subject with no electrode in ``roi`` drops out entirely, so subject
        counts fall.

    IMPORTANT for overlapping ROI groups (``dlpfc``/``lpfc``, ``occ``/``v1``):
    the ``roi`` column was resolved first-group-wins when the map was built, so
    filtering it for ``'lpfc'`` against a map built from the FULL ``rois_dict``
    keeps only the labels no earlier group claimed. Build the map from
    ``subset_rois_dict(rois_dict, 'lpfc')`` when lpfc is the analysis scope.

    ``roi=None`` returns the table unchanged (the whole-brain analysis).
    """
    if roi is None:
        return labels_with_roi
    wanted = [roi] if isinstance(roi, str) else list(roi)
    d = labels_with_roi[labels_with_roi[roi_col].isin(wanted)].copy()
    if verbose:
        n_sub_before = labels_with_roi['subject'].nunique()
        print(f"[A3] restrict_to_roi({wanted}, on '{roi_col}'): "
              f"{len(d)}/{len(labels_with_roi)} electrodes, "
              f"{d['subject'].nunique()}/{n_sub_before} subjects kept")
    d.attrs = dict(labels_with_roi.attrs)
    d.attrs['restricted_to'] = wanted
    return d


# ---------------------------------------------------------------------------
# coverage — the object every anatomical claim is conditioned on
# ---------------------------------------------------------------------------
def build_coverage_matrix(labels_with_roi, roi_col='roi'):
    """Subject × ROI boolean coverage: does subject *s* have ANY electrode in ROI *r*?

    Uses every electrode with a mapped ROI (selective or not) — coverage is about
    where the *grid* is, not where the effects are. Returns a DataFrame indexed by
    subject, columns = ROIs, values bool.

    ``roi_col='anat'`` builds the same matrix over raw Destrieux labels, which is
    the coverage object an ROI-restricted (e.g. lpfc-only) analysis conditions on.
    """
    d = labels_with_roi.dropna(subset=[roi_col])
    cov = pd.pivot_table(d, index='subject', columns=roi_col, values='electrode',
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
                              groups=("both", "S_only", "F_only"), roi_col='roi'):
    """Is selectivity-group membership associated with ROI, conditioned on coverage?

    Parameters
    ----------
    labels_with_roi : output of ``attach_roi`` (needs ``subject, group`` and
        ``roi_col``).
    coverage : output of ``build_coverage_matrix`` **at the same level** as
        ``roi_col`` (pass ``build_coverage_matrix(d, roi_col='anat')`` when
        testing Destrieux labels).
    min_subjects : keep only ROIs covered in >= this many subjects (the coverage
        condition — an ROI a single patient happens to be wired in cannot support
        a population claim).
    n_perm : within-subject permutations for the null.
    groups : which selectivity groups enter the test. Default excludes
        ``neither`` (the test is about *where the selective cells are*).
    roi_col : ``'roi'`` (coarse groups) or ``'anat'`` (Destrieux labels). Inside
        an ROI-restricted analysis only the latter is informative.

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

    d = labels_with_roi.dropna(subset=[roi_col])
    d = d[d[roi_col].isin(kept_rois) & d['group'].isin(group_levels)].copy()

    if d.empty or len(kept_rois) < 2 or d['group'].nunique() < 2:
        # not enough structure to test — return a well-formed null result
        tab = _contingency(d['group'].to_numpy(), d[roi_col].to_numpy(),
                           group_levels, kept_rois) if not d.empty else \
            np.zeros((len(group_levels), len(kept_rois)))
        return dict(
            rois_tested=kept_rois, observed_stat=0.0, p=1.0,
            contingency=pd.DataFrame(tab, index=group_levels, columns=kept_rois),
            per_roi_coverage=per_roi_cov.reindex(kept_rois),
            n_electrodes=int(len(d)), roi_col=roi_col,
            note="insufficient coverage/variation for an enrichment test")

    grp = d['group'].to_numpy()
    roi = d[roi_col].to_numpy()
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
        n_electrodes=int(len(d)), roi_col=roi_col)


# ---------------------------------------------------------------------------
# descriptive ROI histograms
# ---------------------------------------------------------------------------
def roi_group_histogram(labels_with_roi, groups=("both", "S_only", "F_only"),
                        roi_col='roi', top_n=None):
    """Per-group ROI membership counts as a tidy group × ROI DataFrame.

    ``roi_col='anat'`` counts raw Destrieux labels instead of the coarse groups —
    the level you want when the analysis is already restricted to one ROI group
    (an lpfc-only table has a single ``roi`` value, so its group histogram is one
    bar per selectivity group and says nothing about location).

    ``top_n`` keeps only the N most-populated ROIs (by total count across
    groups), which keeps a Destrieux histogram readable when many labels are
    represented by one or two electrodes.
    """
    d = labels_with_roi.dropna(subset=[roi_col])
    d = d[d['group'].isin(groups)]
    tab = (d.groupby(['group', roi_col]).size()
             .unstack(fill_value=0)
             .reindex(index=list(groups), fill_value=0))
    if tab.shape[1] == 0:
        return tab
    order = tab.sum(axis=0).sort_values(ascending=False)
    if top_n is not None:
        order = order.head(int(top_n))
    return tab[list(order.index)]


def plot_roi_group_histograms(labels_with_roi, out_path=None,
                              groups=("both", "S_only", "F_only"), coverage=None,
                              roi_col='roi', top_n=None, title=None):
    """Grouped bar chart of ROI membership per selectivity group.

    If ``coverage`` is given, annotate each ROI with the number of subjects that
    cover it, so the reader can weight raw counts against placement. Pass
    ``roi_col='anat'`` for the Destrieux-label version. Returns the matplotlib
    Figure (and saves it when ``out_path`` is given)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tab = roi_group_histogram(labels_with_roi, groups=groups, roi_col=roi_col,
                              top_n=top_n)
    rois = list(tab.columns)
    x = np.arange(len(rois))
    width = 0.8 / max(len(groups), 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rois)), 4.5))
    for k, g in enumerate(groups):
        heights = tab.loc[g].to_numpy() if g in tab.index else np.zeros(len(rois))
        ax.bar(x + k * width, heights, width,
               label=g, color=GROUP_COLORS.get(g, None))
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    xlabels = list(rois)
    if coverage is not None:
        cov = coverage.sum(axis=0)
        xlabels = [f"{r}\n(n={int(cov.get(r, 0))} subj)" for r in rois]
    ax.set_xticklabels(xlabels, rotation=30, ha='right')
    level = "Destrieux label" if roi_col == 'anat' else "ROI group"
    ax.set(ylabel="# electrodes",
           title=title or f"A3 · {level} membership by selectivity group")
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


def group_electrodes_by_subject(labels_with_roi, groups=("both", "S_only", "F_only")):
    """``{group -> {subject -> [channel names]}}`` — the shape the vis stack wants.

    ``plot_sig_electrodes_dcc.electrodes_to_global_indices`` consumes
    ``{subject_with_zeros: [channel, ...]}`` per colour group, with BARE channel
    names (they are looked up in each subject's ``info['ch_names']``). Any
    ``"{subject}-"`` prefix carried by the ``electrode`` column is stripped here.
    """
    out = {}
    for g in groups:
        d = labels_with_roi[labels_with_roi['group'] == g]
        per_subject = {}
        for subject, electrode in zip(d['subject'].astype(str),
                                      d['electrode'].astype(str)):
            channel = electrode[len(subject) + 1:] \
                if electrode.startswith(f"{subject}-") else electrode
            bucket = per_subject.setdefault(subject, [])
            if channel not in bucket:
                bucket.append(channel)
        out[g] = per_subject
    return out


def plot_selectivity_groups_on_brain(labels_with_roi, out_path, coverage=None,
                                     groups=("both", "S_only", "F_only"),
                                     subjects=None, colors=None, hemi='both',
                                     size=0.45, transparency=0.4, rm_wm=False,
                                     per_group_figures=True, roi_col='roi',
                                     **vis_kwargs):
    """Render S-only / F-only / both electrodes on the fsaverage brain.

    Uses the SAME renderer as the rest of the lab's brain figures — the
    ``vis/jim_mri.plot_on_average`` path, addressed through the global-index
    helpers in ``dcc_scripts/vis/plot_sig_electrodes_dcc.py`` — so an A3 figure
    and a ``plot_sig_electrodes`` figure of the same electrodes are directly
    comparable. Each selectivity group gets its own colour; the groups are
    mutually exclusive by construction (an electrode is ``both`` OR ``S_only`` OR
    ``F_only``), so there is no overlap set to reconcile.

    Parameters
    ----------
    labels_with_roi : output of ``attach_roi`` (needs ``subject, electrode, group``).
        Restrict it first (``restrict_to_roi``) to draw only one ROI's electrodes.
    out_path : where to write the combined figure. A non-raster extension is
        coerced to ``.png`` (``Brain.save_image`` writes rasters).
    subjects : subject ids WITH leading zeros to build the index space over.
        Defaults to the subjects present in ``labels_with_roi``. Pass the full
        plotting list if you want a fixed index space across figures.
    colors : ``{group -> matplotlib colour}``; defaults to ``GROUP_COLORS``.
    per_group_figures : also write one brain per group next to the combined one.

    Returns
    -------
    dict with ``combined`` (path written) and ``per_group`` ({group: path}), plus
    ``fallback=True`` and the histogram path when the surface stack is missing.
    The renderer needs MNE + PyVista + the ECoG_Recon/fsaverage templates, which
    only exist on the cluster/Box; off-cluster this degrades to the ROI-histogram
    figure rather than crashing the job.
    """
    colors = dict(GROUP_COLORS if colors is None else colors)
    by_subject = group_electrodes_by_subject(labels_with_roi, groups=groups)
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
        out_path = base + '.png'

    try:
        # Reuse the project's electrode renderer + its index bookkeeping rather
        # than writing new surface code. Kept behind a try/except so a missing
        # heavy dependency degrades gracefully instead of killing the job.
        import matplotlib.colors as mcolors
        from dcc_scripts.vis.plot_sig_electrodes_dcc import (
            strip_leading_zeros, electrodes_to_global_indices)
        # Use the project's MRI helpers for both loading and rendering.  The
        # upstream ieeg helper defaults to ~/Box/ECoG_Recon on the cluster,
        # whereas jim_mri points at the mounted full recon dataset in /cwork.
        from src.analysis.vis.jim_mri import plot_on_average, subject_to_info
        from collections import OrderedDict

        if subjects is None:
            subjects = sorted(labels_with_roi['subject'].astype(str).unique())

        # Build the fsaverage index space subject-by-subject. Some cluster runs
        # have labels for a subject whose ECoG_Recon files are not mounted (for
        # example D57); the previous all-or-nothing map raised immediately and
        # caused A3 to write only the histogram fallback even when the remaining
        # subjects were renderable. Match the old vis behaviour more closely by
        # skipping only the missing subject and plotting every usable one.
        offsets = OrderedDict()
        subjects_no_zeros = []
        running = 0
        for subject in subjects:
            subject_no_zeros = strip_leading_zeros(str(subject))
            try:
                info = subject_to_info(subject_no_zeros)
            except Exception as exc:
                print(f"[A3] Warning: could not load fsaverage info for "
                      f"{subject_no_zeros}; skipping this subject in the brain "
                      f"figure ({type(exc).__name__}: {exc}).")
                continue
            offsets[subject_no_zeros] = (running, info["ch_names"])
            subjects_no_zeros.append(subject_no_zeros)
            running += len(info["ch_names"])

        if not offsets:
            raise RuntimeError("no subjects with loadable fsaverage electrode info")

        indices = {g: electrodes_to_global_indices(by_subject.get(g, {}), offsets)
                   for g in groups}

        def _rgb(g):
            return mcolors.to_rgb(colors.get(g, '#000000'))

        fig = None
        for g in groups:
            if not indices[g]:
                print(f"[A3] {g}: no electrodes to plot.")
                continue
            fig = plot_on_average(subjects_no_zeros, picks=sorted(indices[g]),
                                  rm_wm=rm_wm, hemi=hemi, color=_rgb(g),
                                  size=size, transparency=transparency,
                                  fig=fig, show=False, **vis_kwargs)
        if fig is None:
            raise RuntimeError("no electrodes in any selectivity group to plot")
        fig.save_image(out_path)
        fig.close()
        print(f"[A3] brain figure -> {out_path}")

        per_group = {}
        if per_group_figures:
            for g in groups:
                if not indices[g]:
                    continue
                gfig = plot_on_average(subjects_no_zeros, picks=sorted(indices[g]),
                                       rm_wm=rm_wm, hemi=hemi, color=_rgb(g),
                                       size=size, transparency=transparency,
                                       show=False, **vis_kwargs)
                gpath = f"{base}_{g}.png"
                gfig.save_image(gpath)
                gfig.close()
                per_group[g] = gpath
                print(f"[A3] brain figure ({g}) -> {gpath}")

        counts = {g: sum(len(v) for v in by_subject.get(g, {}).values())
                  for g in groups}
        return dict(combined=out_path, per_group=per_group, counts=counts,
                    fallback=False)

    except Exception as exc:  # pragma: no cover - depends on cluster-only stack
        print(f"[A3] brain-surface render unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to ROI histogram.")
        fallback = f"{base}_roi_hist.png"
        plot_roi_group_histograms(labels_with_roi, out_path=fallback,
                                  groups=groups, coverage=coverage,
                                  roi_col=roi_col)
        return dict(combined=fallback, per_group={}, fallback=True,
                    error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# synthetic ground truth — runs the whole path with no data on disk
# ---------------------------------------------------------------------------
def _synthetic_anatomy(n_subj=12, seed=0, enrichment=0.6,
                       rois=("dlpfc", "lpfc", "acc", "parietal", "occ", "v1"),
                       return_anat=False, n_sublabels=4,
                       sublabel_enrichment=None):
    """Ground-truth labels + electrode->ROI map with a planted group×ROI association.

    ``both`` and ``S_only`` electrodes are biased toward frontal ROIs and
    ``F_only`` toward parietal/occipital, with strength ``enrichment`` (0 = no
    association, the null). Coverage is deliberately uneven across subjects so the
    coverage filter has something to bite on. Returns
    ``(labels, electrodes_to_rois)`` ready for ``attach_roi``.

    ``return_anat=True`` additionally returns a fake **Destrieux-level** map
    ``{electrode -> "<roi>_lab<k>"}`` with ``n_sublabels`` labels per ROI, so the
    ROI-restricted path (restrict to one ROI, then histogram/test on ``anat``)
    can be validated with no atlas on disk. Within an ROI the sublabel carries its
    own planted association at strength ``sublabel_enrichment`` (defaults to
    ``enrichment``): ``both``/``S_only`` favour ``lab0``, ``F_only`` favours the
    last label. Returns ``(labels, electrodes_to_rois, electrodes_to_anat)``.
    """
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

    if sublabel_enrichment is None:
        sublabel_enrichment = enrichment

    def pick_sublabel(group, roi):
        # within-ROI (Destrieux-level) association, same shape as the ROI one
        if n_sublabels < 2 or rng.random() >= sublabel_enrichment:
            k = int(rng.integers(0, max(n_sublabels, 1)))
        else:
            k = 0 if group in ("both", "S_only") else n_sublabels - 1
        return f"{roi}_lab{k}"

    labels_rows, e2r, e2a = [], {}, {}
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
            e2a[electrode] = pick_sublabel(group, roi)
            labels_rows.append(dict(subject=subject, electrode=electrode, S=S, F=F))
    labels = pd.DataFrame(labels_rows)
    if return_anat:
        return labels, e2r, e2a
    return labels, e2r


if __name__ == '__main__':
    # smoke test: planted enrichment should be detected; the null (enrichment=0)
    # should not manufacture significance. Run at BOTH anatomical levels — whole
    # brain on the coarse ROI groups, then lpfc-only on the Destrieux labels.
    for enr in (0.0, 0.6):
        labels, e2r, e2a = _synthetic_anatomy(enrichment=enr, seed=1,
                                              return_anat=True)
        lab_roi = attach_roi(labels, e2r, electrodes_to_anat=e2a)
        cover = build_coverage_matrix(lab_roi)
        res = roi_group_enrichment_test(lab_roi, cover, min_subjects=3, n_perm=2000)
        print(f"[enrichment={enr}] ROIs tested={res['rois_tested']} "
              f"chi2={res['observed_stat']:.2f} p={res['p']:.4f} "
              f"(n_elec={res['n_electrodes']})")
        print(res['contingency'])
        print("per-ROI coverage (subjects):", dict(res['per_roi_coverage']))

        lpfc = restrict_to_roi(lab_roi, 'lpfc', verbose=False)
        cover_a = build_coverage_matrix(lpfc, roi_col='anat')
        res_a = roi_group_enrichment_test(lpfc, cover_a, min_subjects=3,
                                          n_perm=2000, roi_col='anat')
        print(f"  lpfc-only, Destrieux level: labels={res_a['rois_tested']} "
              f"chi2={res_a['observed_stat']:.2f} p={res_a['p']:.4f} "
              f"(n_elec={res_a['n_electrodes']})")
        print(res_a['contingency'])
        print("-" * 60)
