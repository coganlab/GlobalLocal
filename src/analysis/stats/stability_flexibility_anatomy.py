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

The CONTINUOUS arm (plan §5–§7)
-------------------------------
Everything above needs electrodes to individually pass a significance threshold,
of which there are too few, and it throws the effect sizes away. The second half
of this module asks the same anatomical question of the CONTINUOUS per-electrode
scores instead, on an anatomically- (not effect-) defined electrode set:

- ``attach_scores`` — the :func:`attach_roi` join for scores rather than flags,
  plus the pooled per-effect scaling and ``delta = lwpc_s - lwps_s``.
- ``build_electrode_coord_map`` — ``{electrode -> fsaverage/MNI mm}``.
- ``relative_score_roi_test`` — ``delta ~ roi + responsiveness + (1|subject)``
  with a within-electrode effect-label swap null (§5.2, primary).
- ``relative_score_coordinate_test`` — ``delta ~ MNI coords``, per hemisphere
  (§5.2, secondary).
- ``map_reliability`` — the split-half spatial NOISE CEILING every spatial
  number has to be read against (§5.4).
- ``score_centers_per_subject`` — per-subject weighted medoids (§7, descriptive).
- ``leave_one_subject_out`` — the §9.2 leverage sweep for any of the above.
- ``plot_scores_on_brain`` / ``plot_score_maps`` — the five §6 surfaces, same
  renderer and the same graceful degradation as the group figure.

Drop-in usage for that arm, starting from the segregation module's scores::

    from src.analysis.stats import stability_flexibility_segregation as sfs
    per_split = sfs.compute_sensitivities_per_split(df, contrast_mode='proportion')
    elec      = sfs.add_responsiveness(sfs.average_over_splits(per_split), df)
    scores    = attach_scores(elec, e2r, electrodes_to_anat=e2a,
                              electrodes_to_coords=build_electrode_coord_map(subjects))
    cover     = build_coverage_matrix(scores)
    roi_res   = relative_score_roi_test(scores, cover, min_subjects=3)
    ceiling   = map_reliability(per_split, parcels=e2r)   # report next to it

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

import json
import os
import warnings

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


def _roi_lookup_label(label):
    """Return the atlas parcel spelling used by ``config/rois.py``.

    Recon dictionaries are not completely uniform: older files store bare
    Destrieux names (``G_front_middle``), while newer annotation exports store
    FreeSurfer-qualified names (``ctx_lh_G_front_middle`` or
    ``ctx_rh_G_front_middle``).  The coarse ROI configuration intentionally
    omits hemisphere, so remove only that well-defined prefix for lookup.  The
    original label is still retained by :func:`build_electrode_anat_map`.
    """
    label = str(label)
    for prefix in ("ctx_lh_", "ctx_rh_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


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
    Coarse bilateral ROI groups -> list of hemisphere-neutral Destrieux parcel
        selectors (``src/analysis/config/rois.py``). This lookup does not modify
        the raw ``anat`` label or the electrode coordinates used for rendering.
        ROI groups may share labels; the
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
            group = label_to_group.get(_roi_lookup_label(anat_label))
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


def electrodes_by_subject(frame):
    """``{subject -> [bare channel names]}`` for one table of electrodes.

    ``plot_sig_electrodes_dcc.electrodes_to_global_indices`` consumes
    ``{subject_with_zeros: [channel, ...]}`` per colour set, with BARE channel
    names (they are looked up in each subject's ``info['ch_names']``). Any
    ``"{subject}-"`` prefix carried by the ``electrode`` column is stripped here.
    """
    per_subject = {}
    for subject, electrode in zip(frame['subject'].astype(str),
                                  frame['electrode'].astype(str)):
        channel = electrode[len(subject) + 1:] \
            if electrode.startswith(f"{subject}-") else electrode
        bucket = per_subject.setdefault(subject, [])
        if channel not in bucket:
            bucket.append(channel)
    return per_subject


def group_electrodes_by_subject(labels_with_roi, groups=("both", "S_only", "F_only")):
    """``{group -> {subject -> [channel names]}}`` — the shape the vis stack wants."""
    return {g: electrodes_by_subject(labels_with_roi[labels_with_roi['group'] == g])
            for g in groups}


def _fsaverage_index_space(subjects):
    """``({subject_no_zeros: (offset, ch_names)}, [subject_no_zeros, ...])``.

    The fsaverage "global index" space the lab's vis stack addresses electrodes
    in: every subject's channels are concatenated in order, and an electrode is
    named by its position in that concatenation. Built subject-by-subject rather
    than all-or-nothing because some cluster runs have labels for a subject whose
    ECoG_Recon files are not mounted (for example D57) — that subject is skipped
    with a warning instead of killing the whole figure.
    """
    from collections import OrderedDict
    from dcc_scripts.vis.plot_sig_electrodes_dcc import strip_leading_zeros
    from src.analysis.vis.jim_mri import subject_to_info

    offsets, usable, running = OrderedDict(), [], 0
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
        usable.append(subject_no_zeros)
        running += len(info["ch_names"])
    if not offsets:
        raise RuntimeError("no subjects with loadable fsaverage electrode info")
    return offsets, usable


def _looks_blank(path):
    """True when a saved screenshot is one flat colour, i.e. nothing rendered.

    A render window that was never realized screenshots to a solid background
    rather than failing, so ``save_brain_image`` can report success for a figure
    with no brain in it -- which is how a whole cluster run once produced five
    empty score maps. A real fsaverage render covers a large part of the frame,
    so "almost every pixel matches the corner pixel" is a safe blank test.
    Returns False when the image can't be read: never block a figure over a
    check that failed to run.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.image as mpimg
        img = np.asarray(mpimg.imread(path))
    except Exception:
        return False
    if img.size == 0:
        return True
    px = img.reshape(-1, img.shape[-1]) if img.ndim == 3 else img.reshape(-1, 1)
    return float(np.all(px == px[0], axis=-1).mean()) > 0.999


def _render_electrode_sets(sets, out_path, subjects, hemi='both', size=0.45,
                           transparency=0.4, rm_wm=False, per_set_figures=False,
                           **vis_kwargs):
    """Render colour-coded electrode sets on the fsaverage brain. Raises on failure.

    ``sets`` is a list of ``(name, {subject: [channels]}, colour)``. Each set is
    drawn in one ``plot_on_average`` call onto the SAME ``Brain``, which is the
    only thing the renderer supports (one colour per call) and is why both the
    discrete group figure and the continuous score map are expressed as a list of
    single-colour sets — the continuous one just bins its scalar first.

    Callers own the fallback: this function raises when the surface stack or the
    recon templates are missing, and each public plotting function catches that
    and writes its own degraded figure instead.
    """
    # MNE's pyvistaqt backend needs a valid X display while it constructs the
    # scene. Batch jobs provide one through ``xvfb-run``; direct invocations of
    # the Python entrypoint do not, so start the same virtual display here
    # before importing jim_mri (whose module import selects the Qt backend).
    #
    # Which mode we then pick MATTERS, and getting it wrong is what made this
    # fall back to the by-ROI figure while
    # ``dcc_scripts/vis/plot_sig_electrodes_dcc.py`` rendered fine: with a
    # DISPLAY available, forcing ``OFF_SCREEN`` on means the Qt window is never
    # realized, so its OpenGL context is never current and the screenshot dies
    # with ``RenderWindowUnavailable: Render window is not current``. Mirror the
    # vis script: off-screen ONLY when there is genuinely no display.
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()          # sets DISPLAY when it succeeds
        except Exception as exc:
            print(f"[A3] could not start a virtual display "
                  f"({type(exc).__name__}: {exc}); using PyVista off-screen.")
    have_display = bool(os.environ.get("DISPLAY"))
    if have_display:
        os.environ.pop("PYVISTA_OFF_SCREEN", None)
    else:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
    pv.OFF_SCREEN = not have_display
    try:
        pv.global_theme.allow_empty_mesh = True
    except Exception:                # older pyvista has no such theme option
        pass
    print(f"[A3] 3D rendering: "
          f"{'virtual display ' + os.environ['DISPLAY'] if have_display else 'pyvista off-screen (no DISPLAY)'}")

    # The other half of mirroring the vis script, and what made every A3 brain
    # figure come out BLANK once the off-screen half was fixed: showing the
    # window is what realizes its OpenGL context, so with a DISPLAY we must ask
    # for it. ``show=False`` there leaves a hidden Qt window whose framebuffer
    # screenshots to a flat background -- no error, no fallback, just an empty
    # PNG. Without a DISPLAY we are off-screen and there is nothing to show.
    show = have_display

    import matplotlib.colors as mcolors
    from dcc_scripts.vis.plot_sig_electrodes_dcc import (
        electrodes_to_global_indices, save_brain_image)
    from src.analysis.vis.jim_mri import plot_on_average

    offsets, subjects_no_zeros = _fsaverage_index_space(subjects)
    base, _ = os.path.splitext(out_path)

    picks = [(name, sorted(electrodes_to_global_indices(by_subject, offsets)),
              mcolors.to_rgb(color)) for name, by_subject, color in sets]

    fig = None
    for name, idx, rgb in picks:
        if not idx:
            print(f"[A3] {name}: no electrodes to plot.")
            continue
        fig = plot_on_average(subjects_no_zeros, picks=idx, rm_wm=rm_wm,
                              hemi=hemi, color=rgb, size=size,
                              transparency=transparency, fig=fig, show=show,
                              **vis_kwargs)
    if fig is None:
        raise RuntimeError("no electrodes in any set to plot")
    # ``save_brain_image`` (not ``Brain.save_image``) because pyvista < 0.48
    # doesn't make the render window current before grabbing its framebuffer;
    # the helper retries with an explicit ``MakeCurrent()``.
    if not save_brain_image(fig, out_path):
        fig.close()
        raise RuntimeError(f"could not screenshot the brain figure to {out_path}")
    fig.close()
    if _looks_blank(out_path):
        raise RuntimeError(
            f"the brain figure written to {out_path} is blank -- the render "
            f"window produced an empty frame")
    print(f"[A3] brain figure -> {out_path}")

    per_set = {}
    if per_set_figures:
        for name, idx, rgb in picks:
            if not idx:
                continue
            sfig = plot_on_average(subjects_no_zeros, picks=idx, rm_wm=rm_wm,
                                   hemi=hemi, color=rgb, size=size,
                                   transparency=transparency, show=show,
                                   **vis_kwargs)
            path = f"{base}_{name}.png"
            saved = save_brain_image(sfig, path)
            sfig.close()
            if not saved or _looks_blank(path):
                print(f"[A3] per-set panel {name} came out blank or unsaved; "
                      f"skipping it.")
                continue                 # the combined figure already landed --
                                         # a missing per-set panel isn't fatal
            per_set[name] = path
            print(f"[A3] brain figure ({name}) -> {path}")

    return dict(combined=out_path, per_set=per_set,
                counts={name: len(idx) for name, idx, _ in picks})


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
    if subjects is None:
        subjects = sorted(labels_with_roi['subject'].astype(str).unique())

    try:
        # Reuse the project's electrode renderer + its index bookkeeping rather
        # than writing new surface code. Kept behind a try/except so a missing
        # heavy dependency degrades gracefully instead of killing the job.
        rendered = _render_electrode_sets(
            [(g, by_subject.get(g, {}), colors.get(g, '#000000')) for g in groups],
            out_path, subjects=subjects, hemi=hemi, size=size,
            transparency=transparency, rm_wm=rm_wm,
            per_set_figures=per_group_figures, **vis_kwargs)
        return dict(combined=rendered['combined'], per_group=rendered['per_set'],
                    counts=rendered['counts'], fallback=False)

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
# CONTINUOUS ARM (plan §5-§7): per-electrode LWPC/LWPS scores -> anatomy
# ---------------------------------------------------------------------------
# Everything above defines electrodes by a binary S/F flag and asks whether
# GROUP membership is associated with ROI. That throws away the effect sizes and
# needs electrodes to individually survive a threshold, of which there are too
# few. The functions below take the CONTINUOUS per-electrode scores instead
# (`compute_sensitivities_per_split` + `average_over_splits`, i.e. LWPC on one
# trial half and LWPS on the disjoint half) and ask the same anatomical question
# of them, on an anatomically-defined electrode set.
#
# The whole arm turns on one quantity:
#
#     delta = lwpc_s - lwps_s        (both pooled-scaled, see `attach_scores`)
#
# `delta` is the effect-type contrast computed WITHIN an electrode, so
# regressing it on anatomy IS the effect-type x anatomy interaction. Testing
# "LWPC is significant in region A, LWPS is not, therefore a dissociation" is
# the difference-of-significance fallacy (Nieuwenhuis, Forstmann & Wagenmakers
# 2011); a model of `delta` cannot commit it, because the comparison between the
# two effects happens before anything is tested.
#
# The null everywhere in this section is the WITHIN-ELECTRODE SWAP of the two
# effect labels: give this electrode's LWPC score to LWPS and vice versa. Note
# what that does to `delta` -- it negates it. So the swap null is exactly a
# random SIGN FLIP of `delta` per electrode, which is why it is cheap and why it
# preserves, exactly rather than approximately, every nuisance structure:
# subject, coverage, electrode location, the electrode's own responsiveness, and
# the marginal distribution of both effects. Only the assignment of effect type
# moves. Do NOT shuffle locations between LWPC and LWPS instead: that null
# breaks coverage and answers a different question.


def attach_scores(elec_df, electrodes_to_rois, electrodes_to_anat=None,
                  electrodes_to_coords=None, electrode_fmt="{subject}-{channel}",
                  score_cols=('x', 'y')):
    """Join continuous per-electrode scores to anatomy; add the pooled scaling.

    The continuous sibling of :func:`attach_roi` -- same join, same electrode-id
    reconciliation, no S/F flags.

    Parameters
    ----------
    elec_df : DataFrame with ``subject, electrode`` and the two score columns
        (default ``x`` = LWPC, ``y`` = LWPS, the spelling
        ``average_over_splits`` / ``run_joint_distribution_analysis()['electrodes']``
        emits). An existing ``resp`` column (from ``add_responsiveness``) is
        carried through and used as the covariate by the tests below.
    electrodes_to_rois / electrodes_to_anat : as in :func:`attach_roi`.
    electrodes_to_coords : ``{electrode -> (x, y, z)}`` in fsaverage/MNI mm, from
        :func:`build_electrode_coord_map`. Adds ``mni_x/mni_y/mni_z`` and a
        ``hemi`` column (``'lh'`` / ``'rh'`` by the sign of x).

    Returns
    -------
    DataFrame with ``lwpc_score``/``lwps_score`` (raw), ``lwpc_s``/``lwps_s``
    (pooled-scaled), ``abs_lwpc``/``abs_lwps`` (for the |score| maps and the
    centroid weights, which need non-negative weights), ``delta`` and the
    anatomy columns. Main effects (``mx``/``my`` from a ``main_effects=True``
    run) get the same treatment: ``cong_s``, ``switch_s`` and
    ``dm = cong_s - switch_s``, which every test below takes as ``value_col``.

    The scaling is ONE factor per EFFECT, computed across all electrodes pooled
    -- deliberately NOT a within-subject z-score:

    * a within-subject z-score is degenerate at these electrode counts. With 2
      electrodes in a subject ``std(ddof=1)`` forces the two z-scores to exactly
      +/-0.707 whatever the data; with 1 electrode the SD is NaN and the subject
      drops out of the map silently. lPFC has subjects in exactly that range.
    * the scores are already commensurate -- ``_interaction_effect`` divides the
      difference-of-differences by the pooled within-cell SD, so both are d-like
      already. All that is left to equalise is their marginal spread, which one
      pooled factor per effect does.
    * subject gain is handled by the null (a better-SNR subject has larger |LWPC|
      AND larger |LWPS|, and the swap carries that through untouched), by the
      subject term in the models below, and by the leave-one-subject-out sweep.
    """
    out = elec_df.copy()
    xc, yc = score_cols
    out = out.rename(columns={xc: 'lwpc_score', yc: 'lwps_score',
                              'mx': 'cong_score', 'my': 'switch_score'})
    ids = electrode_ids(out, electrode_fmt=electrode_fmt)
    out['electrode_id'] = ids
    out['roi'] = ids.map(dict(electrodes_to_rois))
    if electrodes_to_anat is not None:
        out['anat'] = ids.map(dict(electrodes_to_anat))
    if electrodes_to_coords is not None:
        coords = dict(electrodes_to_coords)
        xyz = np.array([coords.get(e, (np.nan, np.nan, np.nan)) for e in ids],
                       dtype=float)
        out['mni_x'], out['mni_y'], out['mni_z'] = xyz.T
        hemi = np.where(xyz[:, 0] < 0, 'lh', 'rh').astype(object)
        hemi[~np.isfinite(xyz[:, 0])] = np.nan
        out['hemi'] = hemi

    # one scale factor per EFFECT, across all electrodes pooled
    for src, dst in (('lwpc_score', 'lwpc_s'), ('lwps_score', 'lwps_s'),
                     ('cong_score', 'cong_s'), ('switch_score', 'switch_s')):
        if src not in out:
            continue
        sd = out[src].std(ddof=1)
        out[dst] = out[src] / sd if np.isfinite(sd) and sd > 0 else np.nan
    out['abs_lwpc'] = out['lwpc_s'].abs()
    out['abs_lwps'] = out['lwps_s'].abs()
    out['delta'] = out['lwpc_s'] - out['lwps_s']
    if 'cong_s' in out:
        out['dm'] = out['cong_s'] - out['switch_s']
    return out


def build_electrode_coord_map(subjects, subjects_dir=None,
                              electrode_fmt="{subject}-{channel}",
                              to_fsaverage=True):
    """``{electrode -> (x, y, z)}`` in fsaverage/MNI millimetres.

    Same path the brain figures place electrodes with:
    ``jim_mri.subject_to_info(subject)`` -> montage ``ch_pos`` (forced to the
    ``mri`` frame), then the subject's talairach transform to fsaverage so
    coordinates from different subjects are comparable. Keys use the ORIGINAL
    subject spelling (``D0057-LTP1``) even though the recon directories use the
    stripped one (``D57``), so the map joins straight onto the score table.

    A subject whose recon files are not mounted is skipped with a warning
    (same behaviour as the brain figure), so this degrades to a partial map
    rather than killing the job; the coordinate test then runs on whoever is
    left and reports its own n.
    """
    import mne
    from dcc_scripts.vis.plot_sig_electrodes_dcc import strip_leading_zeros
    from src.analysis.vis.jim_mri import subject_to_info, force2frame, get_sub_dir

    out = {}
    for subject in subjects:
        sub = strip_leading_zeros(str(subject))
        try:
            info = subject_to_info(sub, subjects_dir)
            montage = info.get_montage()
            force2frame(montage, 'mri')
            pos = montage.get_positions()['ch_pos']
            trans = (mne.read_talxfm(sub, get_sub_dir(subjects_dir))['trans']
                     if to_fsaverage else None)
        except Exception as exc:
            print(f"[A3] Warning: no coordinates for {subject} "
                  f"({type(exc).__name__}: {exc}); skipping.")
            continue
        for channel, xyz in pos.items():
            v = np.asarray(xyz, float)
            if trans is not None:
                v = mne.transforms.apply_trans(trans, v)
            out[electrode_fmt.format(subject=subject, channel=channel)] = v * 1000.
    return out


# ---------------------------------------------------------------------------
# the swap null and the nuisance model shared by both continuous tests
# ---------------------------------------------------------------------------
def _nuisance_design(d, covariates=('resp',), subject_col='subject'):
    """Design matrix for the terms every model below conditions on.

    Intercept + subject dummies + centred covariates (responsiveness by
    default). The subject dummies are the fixed-effect, no-shrinkage version of
    the plan's ``(1 | subject)``: under a permutation test the shrinkage a mixed
    model would apply buys nothing (the null is built by resampling, not from a
    parametric df), and dummies cannot fail to converge on a subject with two
    electrodes, which a random intercept can.

    Responsiveness enters as a COVARIATE here, not only as the
    pre-residualisation `prepare_continuous` already does, because a region with
    globally larger HG would otherwise show up as a region with larger scores.
    """
    cols, names = [np.ones(len(d))], ['intercept']
    subs = pd.get_dummies(d[subject_col].astype(str), drop_first=True)
    if subs.shape[1]:
        cols.append(subs.to_numpy(float))
        names += [f"subject[{c}]" for c in subs.columns]
    for c in (covariates or ()):
        if c in d.columns and pd.to_numeric(d[c], errors='coerce').notna().any():
            v = pd.to_numeric(d[c], errors='coerce').to_numpy(float)
            v = np.where(np.isfinite(v), v, np.nanmean(v))
            cols.append(v - v.mean())
            names.append(c)
    return np.column_stack(cols), names


def _swap_null(delta, X, stat_fn, n_perm=10000, seed=0, chunk=512):
    """Observed statistic + its within-electrode-swap null distribution.

    `stat_fn` maps an ``(m, n_electrodes)`` block of nuisance-residualised delta
    values to an ``(m, q)`` block of statistics, so the same machinery serves the
    ROI test and the coordinate test. The swap negates delta (see the section
    header), so the null is generated by sign flips; residualisation is a fixed
    linear map, so it is re-applied to every permuted vector rather than to the
    observed one only.

    Permutations run in chunks: the full ``(n_perm, n_electrodes)`` matrix would
    be hundreds of MB at the electrode counts here, and nothing needs it at once.
    """
    delta = np.asarray(delta, float)
    B = np.linalg.pinv(X)

    def resid(D):
        return D - (D @ B.T) @ X.T

    obs = np.asarray(stat_fn(resid(delta[None, :])), float)[0]
    null = np.empty((int(n_perm), obs.size))
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_perm:
        m = int(min(chunk, n_perm - done))
        signs = rng.choice((-1.0, 1.0), size=(m, delta.size))
        null[done:done + m] = stat_fn(resid(signs * delta))
        done += m
    return obs, null


def _perm_p(obs, null, two_sided=True):
    """``(#{null at least as extreme} + 1) / (n_perm + 1)``, column-wise."""
    o, n = (np.abs(obs), np.abs(null)) if two_sided else (obs, null)
    return (np.sum(n >= o, axis=0) + 1) / (null.shape[0] + 1)


def _fdr(p):
    """Benjamini-Hochberg q-values; falls back to raw p if statsmodels is absent."""
    p = np.asarray(p, float)
    try:
        from statsmodels.stats.multitest import multipletests
        return multipletests(p, method='fdr_bh')[1]
    except Exception:
        return p


# ---------------------------------------------------------------------------
# §5.2 categorical (primary): delta ~ roi + responsiveness + (1 | subject)
# ---------------------------------------------------------------------------
def relative_score_roi_test(scores_with_roi, coverage, min_subjects: int = 3,
                            n_perm: int = 10000, seed: int = 0, roi_col='roi',
                            value_col='delta', covariates=('resp',)):
    """Is the RELATIVE score (LWPC - LWPS) associated with ROI, given coverage?

    The continuous counterpart of :func:`roi_group_enrichment_test`, and the
    primary anatomical test of the plan's §5.2. Same coverage bookkeeping (drop
    ROIs covered in fewer than ``min_subjects`` subjects, hold each electrode's
    ROI fixed in the null); different statistic, because the response is a
    number per electrode rather than a category.

    Model, with subject dummies standing in for ``(1 | subject)``::

        delta_ij ~ roi_j + responsiveness_ij + (1 | subject_i)

    Statistic: the one-way F of ROI on the residualised delta -- between-ROI sum
    of squares over within-ROI sum of squares. Reported with a PERMUTATION
    p-value, so the F's parametric assumptions never have to hold; it is used
    only because it is the scale-free way to compare "spread between ROIs"
    against "spread within them" across permutations.

    Null: the within-electrode swap of the two effect labels (= a sign flip of
    delta), which keeps every electrode exactly where it is.

    Returns a dict with ``observed_stat`` (F), ``p``, the ``per_roi`` table
    (n, subjects, raw and adjusted mean delta, per-ROI two-sided p and BH q) and
    the ``null``. A positive mean delta in an ROI means LWPC-dominant there.
    """
    per_roi_cov = coverage.sum(axis=0)
    kept = sorted(str(r) for r in per_roi_cov.index[per_roi_cov >= min_subjects])

    d = scores_with_roi.dropna(subset=[value_col, roi_col]).copy()
    d = d[d[roi_col].astype(str).isin(kept)]
    d[roi_col] = d[roi_col].astype(str)

    def _empty(note):
        return dict(rois_tested=kept, observed_stat=0.0, p=1.0,
                    per_roi=pd.DataFrame(columns=[roi_col, 'n_electrodes',
                                                  'n_subjects', 'mean_delta',
                                                  'mean_delta_adj', 'p', 'q']),
                    per_roi_coverage=per_roi_cov.reindex(kept),
                    n_electrodes=int(len(d)),
                    n_subjects=int(d['subject'].nunique()),
                    roi_col=roi_col, value_col=value_col, note=note)

    if len(kept) < 2 or d[roi_col].nunique() < 2 or len(d) < 4:
        return _empty("insufficient coverage/variation for a relative-score test")

    rois = sorted(d[roi_col].unique())
    idx = {r: i for i, r in enumerate(rois)}
    k, n = len(rois), len(d)
    G = np.zeros((k, n))
    for j, r in enumerate(d[roi_col]):
        G[idx[r], j] = 1.0
    n_r = G.sum(axis=1)
    G = G / n_r[:, None]                      # rows average within an ROI

    X, _ = _nuisance_design(d, covariates=covariates)
    dfe = max(n - k - (X.shape[1] - 1), 1)

    def stat_fn(R):                            # R: (m, n) residualised delta
        means = R @ G.T                        # (m, k) ROI means
        ssb = (means ** 2 * n_r).sum(axis=1)
        ssw = np.maximum((R ** 2).sum(axis=1) - ssb, 1e-12)
        F = (ssb / max(k - 1, 1)) / (ssw / dfe)
        return np.column_stack([F, means])

    obs, null = _swap_null(d[value_col].to_numpy(float), X, stat_fn,
                           n_perm=n_perm, seed=seed)
    p_F = float(_perm_p(obs[:1], null[:, :1], two_sided=False)[0])
    p_roi = _perm_p(obs[1:], null[:, 1:], two_sided=True)

    per_roi = pd.DataFrame({
        roi_col: rois,
        'n_electrodes': n_r.astype(int),
        'n_subjects': [d.loc[d[roi_col] == r, 'subject'].nunique() for r in rois],
        'mean_delta': [d.loc[d[roi_col] == r, value_col].mean() for r in rois],
        'mean_delta_adj': obs[1:],
        'p': p_roi,
        'q': _fdr(p_roi)})

    return dict(rois_tested=rois, observed_stat=float(obs[0]), p=p_F,
                null=null[:, 0], per_roi=per_roi,
                per_roi_coverage=per_roi_cov.reindex(rois),
                n_electrodes=int(n), n_subjects=int(d['subject'].nunique()),
                roi_col=roi_col, value_col=value_col)


# ---------------------------------------------------------------------------
# §5.2 continuous (secondary): delta ~ MNI coordinates
# ---------------------------------------------------------------------------
def _coordinate_fit(d, value_col, coord_cols, covariates, n_perm, seed):
    """One hemisphere's coordinate regression + swap null. Helper for below."""
    n = len(d)
    Z0 = d[list(coord_cols)].to_numpy(float)
    X, _ = _nuisance_design(d, covariates=covariates)
    B = np.linalg.pinv(X)
    Z = Z0 - X @ (B @ Z0)                      # coords residualised on nuisance
    if n < len(coord_cols) + 3 or np.linalg.matrix_rank(Z) < Z.shape[1]:
        return dict(observed_stat=np.nan, p=np.nan, n_electrodes=int(n),
                    n_subjects=int(d['subject'].nunique()),
                    slopes=pd.DataFrame(columns=['axis', 'slope_per_mm', 'p']),
                    note="too few electrodes / rank-deficient coordinates")

    # Frisch-Waugh: with both sides residualised on the nuisance terms, the
    # slopes and the block F are those of the full model, but the permutation
    # only has to touch a 3-column regression.
    Q, _ = np.linalg.qr(Z)
    Zp = np.linalg.pinv(Z)
    kc = Z.shape[1]
    dfe = max(n - kc - (X.shape[1] - 1), 1)

    def stat_fn(R):
        ssf = ((R @ Q) ** 2).sum(axis=1)
        sse = np.maximum((R ** 2).sum(axis=1) - ssf, 1e-12)
        return np.column_stack([(ssf / kc) / (sse / dfe), R @ Zp.T])

    obs, null = _swap_null(d[value_col].to_numpy(float), X, stat_fn,
                           n_perm=n_perm, seed=seed)
    slopes = pd.DataFrame({
        'axis': list(coord_cols),
        'slope_per_mm': obs[1:],
        'p': _perm_p(obs[1:], null[:, 1:], two_sided=True)})
    return dict(observed_stat=float(obs[0]),
                p=float(_perm_p(obs[:1], null[:, :1], two_sided=False)[0]),
                slopes=slopes, null=null[:, 0], n_electrodes=int(n),
                n_subjects=int(d['subject'].nunique()))


def relative_score_coordinate_test(scores_with_coords, n_perm: int = 10000,
                                   seed: int = 0, value_col='delta',
                                   coord_cols=('mni_y', 'mni_z', 'mni_x'),
                                   covariates=('resp',), by_hemisphere=True):
    """``delta ~ y + z + x + responsiveness + (1 | subject)``, per hemisphere.

    The "LWPC sits anterior to LWPS" claim stated as an AXIS rather than as a
    point, which is why the plan leads with this rather than with a centroid
    (§7): a slope uses every electrode and is not moved by where the densest
    implant happens to sit.

    Run per hemisphere because a bilateral fit on the x axis is meaningless
    (mirrored coordinates cancel) and because left and right coverage differ.
    ``y`` is listed first only for readability -- the block F tests all three
    axes jointly and the per-axis slopes come from the same fit.

    Slopes are in delta units (pooled SDs) per millimetre: a positive ``mni_y``
    slope means LWPC dominance increases ANTERIORLY. Same swap null as §5.2.

    Returns ``{'all': result, 'lh': result, 'rh': result}`` (hemispheres only
    when a ``hemi`` column is present and ``by_hemisphere``).
    """
    d = scores_with_coords.dropna(subset=[value_col, *coord_cols]).copy()
    out = {'all': _coordinate_fit(d, value_col, coord_cols, covariates,
                                  n_perm, seed)}
    if by_hemisphere and 'hemi' in d.columns:
        for h in ('lh', 'rh'):
            sub = d[d['hemi'] == h]
            if len(sub):
                out[h] = _coordinate_fit(sub, value_col, coord_cols, covariates,
                                         n_perm, seed)
    return out


# ---------------------------------------------------------------------------
# §5.4 the noise ceiling -- what a null spatial correlation is allowed to mean
# ---------------------------------------------------------------------------
def map_reliability(per_split, parcels=None, method='spearman', min_units=3):
    """Split-half ceiling for the SPATIAL comparison of the two maps.

    ``split_resolved_corr`` already reports this at the electrode level for the
    residualised, within-subject-centred scores it correlates. This is the map
    version of the same argument, computed on the raw per-split effects and,
    optionally, after averaging electrodes within a parcel -- which is the unit
    the anatomical claims are actually made on.

    Per split k, over the units (electrodes, or parcels when ``parcels`` is
    given)::

        between = mean_k  1/2 [ corr(xA_k, yB_k) + corr(xB_k, yA_k) ]
        rel_lwpc = mean_k  corr(xA_k, xB_k)
        rel_lwps = mean_k  corr(yA_k, yB_k)

    `between` never correlates two effects measured on the same trials, so
    shared trial noise cannot inflate it. `rel_*` is how well each map
    correlates with ITSELF across disjoint halves, i.e. the most any correlation
    with it could be. Read together: between 0.35 against a ceiling of 0.40 means
    the maps are as similar as the noise permits (same anatomy); between 0.05
    against 0.40 means they are distinct. Without the ceiling, "the maps do not
    correlate" cannot be told apart from "neither map is measured well enough to
    correlate with anything" (Nili et al. 2014) -- report it next to every
    spatial number.

    ``parcels`` is a dict/Series ``{electrode -> parcel}`` (the same ``roi`` or
    ``anat`` map the tests use).

    Both sides are HALF-LENGTH -- ``between`` correlates two half-trial
    estimates, and so do ``rel_*`` -- so numerator and denominator sit at the
    same trial count and the ratio is the attenuation correction it should be.
    Do NOT Spearman-Brown the reliabilities up here: that would pair a
    full-length ceiling with a half-length numerator and understate the ratio.

    ``between_noise_corrected`` is always computed from PEARSON correlations,
    whatever ``method`` is, because the attenuation formula is classical-test-
    theory algebra for linear measurements. Ranking is a non-linear transform
    that deflates the self-reliabilities more than it deflates the cross term,
    so a Spearman ratio is not an attenuation correction and can sit far above
    1 (on the lPFC run: 1.374 under Spearman, 1.022 under Pearson, from nearly
    identical ``between`` values of 0.172 and 0.174). ``between`` itself is
    still reported under ``method``, which stays Spearman by default for
    robustness to outliers.

    A corrected value above 1 is out of range for a correlation. It means the
    reliabilities are too small to bound anything, not that the maps are
    more-than-perfectly correlated -- ``note`` says so, and
    ``between_noise_corrected_ci`` (bootstrap over splits) shows how unstable
    the ratio is. Report ``between`` and the reliabilities in that case.
    """
    from scipy.stats import spearmanr, pearsonr
    fn = spearmanr if method == 'spearman' else pearsonr

    d = per_split.dropna(subset=['xA', 'xB', 'yA', 'yB']).copy()
    # keep only electrodes defined on EVERY split, so each split's vectors are
    # over the same units and the averages are comparable
    n_sp = per_split['split'].nunique()
    d = d[d.groupby('electrode')['split'].transform('size') == n_sp]
    unit = 'electrode'
    if parcels is not None:
        d['parcel'] = d['electrode'].map(dict(parcels))
        d = d.dropna(subset=['parcel'])
        d = d.groupby(['split', 'parcel'], as_index=False)[['xA', 'xB', 'yA', 'yB']].mean()
        unit = 'parcel'
    if d.empty:
        raise ValueError("no unit survived the completeness / parcel filter")

    rows, rows_p = [], []
    for _, g in d.groupby('split'):
        if len(g) < min_units:
            continue
        xa, xb = g['xA'].to_numpy(), g['xB'].to_numpy()
        ya, yb = g['yA'].to_numpy(), g['yB'].to_numpy()
        rows.append((0.5 * (fn(xa, yb)[0] + fn(xb, ya)[0]),
                     fn(xa, xb)[0], fn(ya, yb)[0]))
        # The attenuation correction below is Pearson algebra, so it is always
        # computed on Pearson correlations even when `method='spearman'` --
        # see `between_noise_corrected` in the docstring.
        rows_p.append((0.5 * (pearsonr(xa, yb)[0] + pearsonr(xb, ya)[0]),
                       pearsonr(xa, xb)[0], pearsonr(ya, yb)[0]))
    if not rows:
        raise ValueError(f"every split has fewer than min_units={min_units} {unit}s")
    between, rel_x, rel_y = np.nanmean(np.array(rows, float), axis=0)
    arr_p = np.array(rows_p, float)
    between_p, rel_xp, rel_yp = np.nanmean(arr_p, axis=0)

    def _corrected(b, rx, ry):
        return b / np.sqrt(rx * ry) if (rx > 0 and ry > 0) else np.nan

    corrected = _corrected(between_p, rel_xp, rel_yp)

    # The ratio is unstable when the reliabilities are small, so report a
    # bootstrap interval over splits next to the point estimate rather than the
    # point estimate alone.
    if np.isfinite(corrected):
        rng = np.random.default_rng(0)
        boot = np.array([_corrected(*arr_p[rng.integers(0, len(arr_p), len(arr_p))].mean(axis=0))
                         for _ in range(2000)], float)
        ci = (tuple(float(v) for v in np.nanpercentile(boot, [2.5, 97.5]))
              if np.isfinite(boot).any() else (np.nan, np.nan))
    else:
        # The point estimate is undefined, so a bootstrap of it is not an
        # interval for anything -- resamples that happen to land on a tiny
        # positive reliability produce arbitrarily large ratios.
        ci = (np.nan, np.nan)

    note = None
    if not np.isfinite(corrected):
        note = ("noise correction undefined: a split-half reliability is <= 0, so "
                "neither map is measured well enough to bound a correlation. "
                "Report `between` and the reliabilities, not a corrected value.")
    elif corrected > 1.0:
        note = (f"noise-corrected value {corrected:.3f} exceeds 1, which is out of "
                "range for a correlation. At reliabilities this low the ratio is "
                "not estimable; report it as 'at the ceiling' and quote `between` "
                "and the reliabilities instead of the ratio.")

    return dict(between=float(between), reliability_lwpc=float(rel_x),
                reliability_lwps=float(rel_y),
                between_noise_corrected=(float(corrected)
                                         if np.isfinite(corrected) else np.nan),
                between_noise_corrected_ci=ci,
                between_pearson=float(between_p),
                reliability_lwpc_pearson=float(rel_xp),
                reliability_lwps_pearson=float(rel_yp),
                note=note,
                unit=unit, n_units=int(d[unit].nunique()),
                n_splits=int(len(rows)), method=method)


# ---------------------------------------------------------------------------
# §7 centroids / medoids -- DESCRIPTIVE ONLY
# ---------------------------------------------------------------------------
def score_centers_per_subject(scores_with_coords, n_perm: int = 10000,
                              seed: int = 0, medoid=True, min_elec: int = 3,
                              weight_cols=('abs_lwpc', 'abs_lwps'),
                              coord_cols=('mni_x', 'mni_y', 'mni_z')):
    """Weighted LWPC vs LWPS centre per subject x hemisphere, compared within subject.

    A single pooled cross-subject centroid is not a defensible test: subjects
    with more electrodes dominate the LOCATION ESTIMATE itself (not merely its
    variance, so no null can undo it), coverage is clinically determined, a
    centroid need not land in cortex, and signed weights cancel. This is the
    defensible version -- per subject, per hemisphere, over the SAME electrodes,
    with NON-NEGATIVE weights (|score|, which is why `attach_scores` emits
    ``abs_lwpc``/``abs_lwps``), compared within subject, and with the same
    within-electrode swap null as §5.2.

    ``medoid=True`` (default) reports the observed electrode minimising the
    weighted distance to the others, so the reported location is a real
    recording site rather than a point in the middle of a hole.

    Statistic: the mean over subject x hemisphere of the LWPC-minus-LWPS
    displacement, reported per axis, with the anterior (y) displacement carrying
    the "LWPC sits N mm anterior to LWPS" claim. Lead with the coordinate
    regression above; this is the descriptive panel next to it.
    """
    w1c, w2c = weight_cols
    d = scores_with_coords.dropna(subset=[w1c, w2c, *coord_cols]).copy()
    if 'hemi' not in d.columns:
        d['hemi'] = np.where(d[coord_cols[0]].to_numpy(float) < 0, 'lh', 'rh')

    rng = np.random.default_rng(seed)
    rows, per_perm = [], []
    for (subject, hemi), g in d.groupby(['subject', 'hemi']):
        if len(g) < min_elec:
            continue
        P = g[list(coord_cols)].to_numpy(float)
        w1 = g[w1c].to_numpy(float)
        w2 = g[w2c].to_numpy(float)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)

        def centers(W):                        # W: (m, n) weights -> (m, 3)
            if medoid:
                return P[np.argmin(W @ D, axis=1)]
            tot = np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
            return (W @ P) / tot

        obs = centers(w1[None, :])[0] - centers(w2[None, :])[0]
        swap = rng.random((int(n_perm), len(g))) < 0.5
        disp = (centers(np.where(swap, w2, w1)) - centers(np.where(swap, w1, w2)))
        p_y = float((np.sum(np.abs(disp[:, 1]) >= abs(obs[1])) + 1) / (n_perm + 1))
        rows.append(dict(subject=subject, hemi=hemi, n_electrodes=int(len(g)),
                         dx=obs[0], dy=obs[1], dz=obs[2],
                         distance=float(np.linalg.norm(obs)), p_anterior=p_y))
        per_perm.append(disp)

    if not rows:
        return dict(per_group=pd.DataFrame(rows), n_groups=0,
                    note=f"no subject x hemisphere had >= {min_elec} electrodes")

    per_group = pd.DataFrame(rows)
    obs_mean = per_group[['dx', 'dy', 'dz']].to_numpy(float).mean(axis=0)
    # one draw per permutation index across ALL groups, so the group mean has a
    # coherent null: every electrode swaps independently, as in §5.2
    null_mean = np.mean(np.stack(per_perm, axis=0), axis=0)        # (n_perm, 3)
    p_axis = [(np.sum(np.abs(null_mean[:, i]) >= abs(obs_mean[i])) + 1)
              / (n_perm + 1) for i in range(3)]
    return dict(per_group=per_group, n_groups=int(len(per_group)),
                center='medoid' if medoid else 'weighted centroid',
                mean_displacement=dict(zip(('dx', 'dy', 'dz'), obs_mean)),
                p=dict(zip(('dx', 'dy', 'dz'), map(float, p_axis))),
                mean_distance=float(per_group['distance'].mean()))


# ---------------------------------------------------------------------------
# §9.2 subject-level sanity on every pooled number
# ---------------------------------------------------------------------------
def leave_one_subject_out(test_fn, table, subject_col='subject',
                          keys=('observed_stat', 'p', 'n_electrodes')):
    """Re-run `test_fn(table)` with each subject dropped in turn.

    Electrode-weighted inference pooled across subjects is standard; what makes
    it safe is checking that two subjects are not supplying the result. Pass a
    one-argument callable, e.g.::

        sfa.leave_one_subject_out(
            lambda t: sfa.relative_score_roi_test(t, cover, n_perm=2000), scores)

    Note the coverage matrix is deliberately NOT recomputed per fold, so every
    fold tests the same ROI set and the rows are comparable.
    """
    def row(tag, res):
        out = {'dropped': tag}
        out.update({k: res[k] for k in keys if k in res})
        return out

    rows = [row('(none)', test_fn(table))]
    for s in sorted(table[subject_col].astype(str).unique()):
        sub = table[table[subject_col].astype(str) != s]
        if sub.empty:
            continue
        rows.append(row(s, test_fn(sub)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main effects as the reference for the delta tilt (docs/analysis_plans.md#closing-figure-plan)
# ---------------------------------------------------------------------------
def _delta_halves(scores, per_split):
    """Per-split table with dm in the x slots and delta in the y slots, each
    half divided by its effect's full-data pooled SD (read back off `scores`)."""
    s = scores.set_index('electrode')
    ps = per_split[per_split['electrode'].isin(s.index)]
    sd = {k: np.nanmedian(s[f'{k}_score'] / s[f'{k}_s'])
          for k in ('lwpc', 'lwps', 'cong', 'switch')}
    return ps[['subject', 'electrode', 'split']].assign(
        xA=ps['mxA'] / sd['cong'] - ps['myA'] / sd['switch'],
        xB=ps['mxB'] / sd['cong'] - ps['myB'] / sd['switch'],
        yA=ps['xA'] / sd['lwpc'] - ps['yA'] / sd['lwps'],
        yB=ps['xB'] / sd['lwpc'] - ps['yB'] / sd['lwps'])


def delta_tracking_test(scores, per_split, n_perm=10000, seed=1, min_elec=3,
                        coord_cols=('mni_y', 'mni_z', 'mni_x')):
    """Test 1: does the main-effect delta (dm) track delta across electrodes?

    The pre-specified co-localization test, ``split_resolved_corr``, on dm and
    delta instead of LWPC and LWPS: one from each disjoint half within every
    split, residualised on responsiveness, centred within participant,
    Spearman, within-participant permutation null. Then with MNI coordinates as
    covariates (does the tracking go beyond a shared gradient?), then per
    process, with the crossed pairings as controls. One row per comparison;
    ``reliability_x``/``reliability_y`` are the two maps' split-half
    reliabilities (within participant, so biased low: §15.4 of
    docs/n4_continuous_anatomy.md).
    """
    from .stability_flexibility_segregation import split_resolved_corr
    s = scores.set_index('electrode')
    ps = per_split[per_split['electrode'].isin(s.index)]
    deltas = _delta_halves(scores, per_split)
    runs = [('dm vs delta', deltas, None)]
    if set(coord_cols) <= set(s.columns) and s[list(coord_cols)].notna().all(axis=1).any():
        xyz = s[list(coord_cols)].dropna()
        runs.append(('dm vs delta, + MNI covariates',
                     deltas[deltas['electrode'].isin(xyz.index)], xyz))
    for name, a, b in (('congruency vs LWPC', 'mx', 'x'), ('switch vs LWPS', 'my', 'y'),
                       ('congruency vs LWPS (crossed)', 'mx', 'y'),
                       ('switch vs LWPC (crossed)', 'my', 'x')):
        runs.append((name, ps[['subject', 'electrode', 'split']].assign(
            xA=ps[a + 'A'], xB=ps[a + 'B'], yA=ps[b + 'A'], yB=ps[b + 'B']), None))
    rows = []
    for name, table, cov in runs:
        r = split_resolved_corr(table, s['resp'], min_elec=min_elec, n_perm=n_perm,
                                seed=seed, covariates=cov)
        rows.append(dict(comparison=name, **{k: r[k] for k in (
            'corr', 'p', 'reliability_x', 'reliability_y', 'n_electrodes', 'n_subjects')}))
    return pd.DataFrame(rows)


def tilt_with_main_effect_covariate(scores, per_split=None, axis='mni_z',
                                    coord_cols=('mni_y', 'mni_z', 'mni_x'),
                                    covariates=('resp',), n_perm=10000, seed=0):
    """Test 2: does delta's tilt survive dm as a covariate?

    ``relative_score_coordinate_test``'s fit (swap null) for dm (do the main
    effects tilt the same way?), delta, and delta + dm. The swap flips only the
    adaptation labels, which also breaks delta's link with dm, so that null is
    conservative. dm and delta share trials there (noise correlation ~+0.05),
    so with ``per_split`` delta is also refitted on each half of every split
    with dm from the OPPOSITE half, and the slopes averaged (these two rows
    have no p-value; read the full-data rows for it).

    Returns ``table`` and ``shrinkage = 1 - with/without`` on ``axis``: near 1
    the tilt is carried by the main effects, near 0 it survives them. Only
    interpretable when delta's tilt without dm is itself distinguishable from 0.
    """
    d = scores.dropna(subset=['delta', 'dm', *coord_cols]).reset_index(drop=True)

    def fit(data, value, covs, n):
        r = _coordinate_fit(data, value, coord_cols, covs, n, seed)
        return tuple(r['slopes'].set_index('axis').loc[axis, ['slope_per_mm', 'p']])

    rows = [('dm', *fit(d, 'dm', covariates, n_perm)),
            ('delta', *fit(d, 'delta', covariates, n_perm)),
            ('delta + dm', *fit(d, 'delta', (*covariates, 'dm'), n_perm))]
    shrinkage = {'same trials': 1 - rows[2][1] / rows[1][1]}
    if per_split is not None and 'mxA' in per_split:
        without, with_ = [], []
        for _, hk in _delta_halves(d, per_split).groupby('split'):
            dk = d.merge(hk.drop(columns=['subject', 'split']), on='electrode').dropna(
                subset=['xA', 'xB', 'yA', 'yB'])
            for da, dm in (('yA', 'xB'), ('yB', 'xA')):
                without.append(fit(dk, da, covariates, 0)[0])
                with_.append(fit(dk, da, (*covariates, dm), 0)[0])
        rows += [('delta, split halves', np.mean(without), np.nan),
                 ('delta + dm from the opposite half', np.mean(with_), np.nan)]
        shrinkage['opposite half'] = 1 - np.mean(with_) / np.mean(without)
    return dict(axis=axis, table=pd.DataFrame(rows, columns=['fit', 'slope_per_mm', 'p']),
                shrinkage=shrinkage)


# ---------------------------------------------------------------------------
# participants as the unit (advisor check 2026-10-02; §19 of
# docs/n4_continuous_anatomy.md)
# ---------------------------------------------------------------------------
# The coordinate test conditions on participant with dummy variables (a fixed
# intercept, `_nuisance_design`) and builds its null by flipping each
# electrode's sign on its own. Electrodes are therefore the units of inference:
# participants with many electrodes spread along the axis weigh most, and
# neighbouring contacts, which share noise and nearly share coordinates, count
# as independent. The functions below ask the same question with participants
# as the units: the slope's decomposition into per-participant slopes, a mixed
# model with a random slope, and the slope with each participant left out.
def _partial_axis_residuals(d, value_col, axis, coord_cols, covariates):
    """``value_col`` and ``axis``, each residualised on the coordinate test's
    nuisance design (participant dummies + covariates) and on the OTHER
    coordinates. By Frisch-Waugh, the pooled slope of the first on the second
    is the coordinate test's slope on ``axis``."""
    X, _ = _nuisance_design(d, covariates=covariates)
    others = [c for c in coord_cols if c != axis]
    N = np.column_stack([X, d[others].to_numpy(float)]) if others else X
    B = np.linalg.pinv(N)
    v = d[value_col].to_numpy(float)
    a = d[axis].to_numpy(float)
    return v - N @ (B @ v), a - N @ (B @ a)


def _mixed_slope_fits(d, value_col, axis, coord_cols, covariates):
    """``value ~ coordinates + covariates`` with a participant random intercept,
    then with a random intercept and a random slope on ``axis``.

    Every predictor is centred within participant, so the fixed slopes are
    within-participant slopes, as in the coordinate test, and the random
    intercept takes the participant means. Coordinates enter in cm for the
    optimiser and are reported per mm. Wald p-values with ~20 participants run
    a little liberal; read them next to the two-stage test.
    """
    import statsmodels.formula.api as smf

    m = d[['subject', value_col]].copy()
    terms = []
    for c, scale in [(c, 10.0) for c in coord_cols] + [(c, 1.0) for c in covariates]:
        if c not in d.columns:
            continue
        v = pd.to_numeric(d[c], errors='coerce')
        m[c + '_c'] = (v - v.groupby(d['subject']).transform('mean')) / scale
        terms.append(c + '_c')
    m = m.dropna()
    formula = f"{value_col} ~ " + " + ".join(terms)
    out = {}
    for name, re_formula in (('random_intercept', None), ('random_slope', f'~{axis}_c')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                fit = smf.mixedlm(formula, m, groups=m['subject'],
                                  re_formula=re_formula).fit(reml=True,
                                                             method=['lbfgs', 'powell'])
            except Exception as exc:                     # singular, did not converge
                out[name] = dict(error=f"{type(exc).__name__}: {exc}")
                continue
        row = dict(slope_per_mm=float(fit.params[f'{axis}_c']) / 10.0,
                   se_per_mm=float(fit.bse[f'{axis}_c']) / 10.0,
                   p=float(fit.pvalues[f'{axis}_c']),
                   converged=bool(fit.converged),
                   n_electrodes=int(len(m)), n_subjects=int(m['subject'].nunique()),
                   warnings=sorted({str(w.message).split('\n')[0] for w in caught})[:3])
        if re_formula:
            cov = fit.cov_re
            key = f'{axis}_c'
            row['random_slope_sd_per_mm'] = (float(np.sqrt(max(cov.loc[key, key], 0.0))) / 10.0
                                             if key in cov.index else np.nan)
        out[name] = row
    return out


def coordinate_slope_by_participant(scores_with_coords, axis='mni_z', value_col='delta',
                                    coord_cols=('mni_y', 'mni_z', 'mni_x'),
                                    covariates=('resp',), min_elec=3, min_spread_mm=5.0,
                                    n_perm=10000, n_boot=2000, seed=0, mixed_model=True):
    """The coordinate test's slope on ``axis`` with participants as the units.

    **Two-stage.** With ``value_col`` and ``axis`` residualised on the test's
    nuisance terms and the other coordinates (:func:`_partial_axis_residuals`),
    each participant s has its own slope ``b_s = sum_s a*v / w_s`` with weight
    ``w_s = sum_s a**2`` (its spread along the axis). The pooled slope the
    coordinate test reports is exactly ``sum w_s b_s / sum w_s``: a weighted
    average of the participants' slopes. Two tests across participants:

    * weighted: that average, with p from flipping the sign of whole
      participants and a 95 % interval from a participant bootstrap;
    * unweighted: the mean of ``b_s`` over participants with at least
      ``min_elec`` electrodes and ``min_spread_mm`` of spread (SD of the
      residualised axis), one-sample t-test, plus how many slopes are negative.

    **Mixed model** (``mixed_model``): :func:`_mixed_slope_fits`, the same
    model with a participant random intercept, and with a random intercept and
    a random slope on ``axis``.

    Returns a dict: ``per_participant`` (subject, n_electrodes, spread_mm,
    weight, slope_per_mm), ``pooled_slope`` (equals the coordinate test's),
    ``weighted``, ``unweighted``, ``top3_weight_share`` (how much of the
    weighted average three participants carry) and ``mixed``.
    """
    from scipy.stats import binomtest, ttest_1samp

    d = scores_with_coords.dropna(subset=[value_col, *coord_cols]).reset_index(drop=True)
    v, a = _partial_axis_residuals(d, value_col, axis, coord_cols, covariates)
    subj = d['subject'].astype(str).to_numpy()
    rows = []
    for s in np.unique(subj):
        m = subj == s
        w = float((a[m] ** 2).sum())
        rows.append(dict(subject=s, n_electrodes=int(m.sum()),
                         spread_mm=float(np.sqrt(w / m.sum())), weight=w,
                         slope_per_mm=float((a[m] * v[m]).sum() / w) if w > 1e-9 else np.nan))
    per = pd.DataFrame(rows)
    pooled = float((a * v).sum() / (a * a).sum())

    ok = per['slope_per_mm'].notna().to_numpy()
    b, w = per.loc[ok, 'slope_per_mm'].to_numpy(float), per.loc[ok, 'weight'].to_numpy(float)
    rng = np.random.default_rng(seed)
    flips = rng.choice((-1.0, 1.0), size=(int(n_perm), len(b)))
    null = (flips * (w * b)).sum(1) / w.sum()
    idx = rng.integers(0, len(b), size=(int(n_boot), len(b)))
    boot = (w[idx] * b[idx]).sum(1) / w[idx].sum(1)
    weighted = dict(slope_per_mm=float((w * b).sum() / w.sum()),
                    p_signflip=float((np.sum(np.abs(null) >= abs(pooled)) + 1) / (n_perm + 1)),
                    ci=tuple(float(x) for x in np.percentile(boot, [2.5, 97.5])),
                    n_participants=int(len(b)))

    elig = (ok & (per['n_electrodes'] >= min_elec).to_numpy()
            & (per['spread_mm'] >= min_spread_mm).to_numpy())
    bu = per.loc[elig, 'slope_per_mm'].to_numpy(float)
    neg = int((bu < 0).sum())
    unweighted = dict(n_participants=int(len(bu)), n_negative=neg,
                      min_elec=min_elec, min_spread_mm=min_spread_mm)
    if len(bu) > 1:
        t = ttest_1samp(bu, 0.0)
        unweighted.update(mean_slope_per_mm=float(bu.mean()),
                          sem=float(bu.std(ddof=1) / np.sqrt(len(bu))),
                          t=float(t.statistic), p_t=float(t.pvalue),
                          p_sign=float(binomtest(neg, len(bu), 0.5).pvalue))
    per['in_unweighted_test'] = elig
    share = np.sort(w)[::-1][:3].sum() / w.sum() if len(w) else np.nan

    out = dict(axis=axis, value_col=value_col, per_participant=per, pooled_slope=pooled,
               weighted=weighted, unweighted=unweighted, top3_weight_share=float(share),
               n_electrodes=int(len(d)), n_subjects=int(len(per)))
    if mixed_model:
        out['mixed'] = _mixed_slope_fits(d, value_col, axis, coord_cols, covariates)
    return out


def coordinate_slope_loso(scores_with_coords, axis='mni_z', value_col='delta',
                          coord_cols=('mni_y', 'mni_z', 'mni_x'), covariates=('resp',),
                          n_perm=2000, seed=0):
    """The coordinate test's ``axis`` slope and swap-null p with each participant
    left out (:func:`leave_one_subject_out`). Fewer permutations than the main
    test: a leverage check on the estimate, not a second round of inference."""
    d = scores_with_coords.dropna(subset=[value_col, *coord_cols])

    def fit(t):
        r = _coordinate_fit(t, value_col, coord_cols, covariates, n_perm, seed)
        sl = r['slopes'].set_index('axis')
        has = axis in sl.index
        return dict(observed_stat=float(sl.loc[axis, 'slope_per_mm']) if has else np.nan,
                    p=float(sl.loc[axis, 'p']) if has else np.nan,
                    block_F=r['observed_stat'], block_p=r['p'],
                    n_electrodes=r['n_electrodes'])

    out = leave_one_subject_out(fit, d, keys=('observed_stat', 'p', 'block_F', 'block_p',
                                              'n_electrodes'))
    return out.rename(columns={'observed_stat': 'slope_per_mm'})


# ---------------------------------------------------------------------------
# what else could make LWPC and LWPS correlate across electrodes?
# (2026-10-05; §19.7 of docs/n4_continuous_anatomy.md)
# ---------------------------------------------------------------------------
# The pre-specified test already removes shared trial noise (the two scores
# come from disjoint halves of an electrode's trials) and linear responsiveness
# (mean |HG|). `overlap_controls` reruns it with each remaining candidate
# removed in turn. A control that leaves r where it was rules that candidate
# out; one that removes r says the overlap goes with that property.
_SAME_HALF_BASE = {'xA': ('mxA', 'myA'), 'xB': ('mxB', 'myB'),
                   'yA': ('mxA', 'myA'), 'yB': ('mxB', 'myB')}


def overlap_controls(scores, per_split, rt_coupling=None, coord_cols=('mni_y', 'mni_z', 'mni_x'),
                     min_elec=3, n_perm=10000, seed=1, loso=True):
    """The pre-specified LWPC-LWPS separate-half r, and the same test with one
    candidate confound removed at a time.

    Rows (``control``):

    * ``pre-specified``: ``split_resolved_corr``, responsiveness (mean |HG|)
      regressed out linearly.
    * ``+ responsiveness, nonlinear``: also log and squared responsiveness, in
      case signal-to-noise co-inflates both scores in a way mean |HG| does not
      capture linearly.
    * ``+ MNI coordinates``: a smooth gradient both maps share.
    * ``+ base effects, same half``: each half's adaptation score with that
      half's congruency and switch-type effects partialled out (needs a
      MAIN_EFFECTS=1 run). This asks whether LWPC and LWPS overlap beyond what
      the overlap of their base effects implies. The base effects are also the
      best available proxy for an electrode's signal-to-noise, so a drop here
      is either "both adaptations scale with the base effects" or residual
      signal-to-noise; the row cannot tell which.
    * ``+ RT coupling``: each electrode's within-cell HG-RT correlation
      (``rt_coupling``: a Series or CSV-loaded table with ``electrode`` and
      ``rt_r``, e.g. the A6 job's ``participant_electrode_scores.csv`` or the
      RT-adjusted segregation run's ``rt_adjustment_slopes.csv``). If HG tracks
      RT, an electrode's LWPC and LWPS each contain its coupling times the
      participant's behavioral effect, so more strongly coupled electrodes
      show more of both. ``rt_r`` is the scale-free version of the slope, which
      is the coupling in d units. Only electrodes with a value enter this row.
    * ``all of the above``.
    * ``responsiveness tertile k``: the test within each tertile of
      responsiveness (descriptive; participants need ``min_elec`` electrodes
      in the tertile).

    With ``loso``, ``leave_one_out`` gives the pre-specified r with each
    participant left out (fewer permutations). Returns ``(table, leave_one_out)``.
    The RT-adjusted segregation run (``RT_ADJUST_HG=1``, §18.10 of the N4 doc) is
    the full version of the RT row: it removes the coupling from every trial
    before scoring.
    """
    from .stability_flexibility_segregation import split_resolved_corr
    s = scores.drop_duplicates('electrode').set_index('electrode')
    ps = per_split[per_split['electrode'].isin(s.index)]
    resp = s['resp']
    rows = []

    def run(name, table=ps, covariates=None, half=None, note=''):
        try:
            r = split_resolved_corr(table, resp, min_elec=min_elec, n_perm=n_perm, seed=seed,
                                    covariates=covariates, half_covariates=half)
            rows.append(dict(control=name, corr=r['corr'], p=r['p'],
                             n_electrodes=r['n_electrodes'], n_subjects=r['n_subjects'],
                             note=note))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(dict(control=name, corr=np.nan, p=np.nan, n_electrodes=0,
                             n_subjects=0, note=f"{type(exc).__name__}: {exc}"))

    run('pre-specified')
    pos = resp.where(resp > 0)
    nonlin = pd.DataFrame({'log_resp': np.log(pos), 'resp_sq': resp ** 2}).dropna()
    run('+ responsiveness, nonlinear', ps[ps['electrode'].isin(nonlin.index)], nonlin)
    xyz = s[[c for c in coord_cols if c in s]].dropna() if set(coord_cols) <= set(s) else None
    if xyz is not None and len(xyz):
        run('+ MNI coordinates', ps[ps['electrode'].isin(xyz.index)], xyz)
    has_base = set(_SAME_HALF_BASE['xA'] + _SAME_HALF_BASE['xB']) <= set(ps.columns)
    if has_base:
        run('+ base effects, same half', half=_SAME_HALF_BASE)
    rt = None
    if rt_coupling is not None:
        rt = (rt_coupling if isinstance(rt_coupling, pd.Series)
              else rt_coupling.drop_duplicates('electrode').set_index('electrode')['rt_r'])
        rt = rt.reindex(s.index).dropna().rename('rt_r').to_frame()
        run('pre-specified, electrodes with RT coupling', ps[ps['electrode'].isin(rt.index)],
            note='the reference for the next row')
        run('+ RT coupling', ps[ps['electrode'].isin(rt.index)], rt)
    every = [t for t in (nonlin, xyz, rt) if t is not None and len(t)]
    if every:
        allcov = pd.concat(every, axis=1, join='inner')
        run('all of the above', ps[ps['electrode'].isin(allcov.index)], allcov,
            half=_SAME_HALF_BASE if has_base else None)
    tert = pd.qcut(resp.rank(method='first'), 3, labels=['low', 'middle', 'high'])
    for t in ('low', 'middle', 'high'):
        run(f'responsiveness tertile: {t}', ps[ps['electrode'].isin(tert.index[tert == t])],
            note='descriptive')
    table = pd.DataFrame(rows)

    out_loso = None
    if loso:
        lrows = []
        for subj in sorted(ps['subject'].astype(str).unique()):
            sub = ps[ps['subject'].astype(str) != subj]
            try:
                r = split_resolved_corr(sub, resp, min_elec=min_elec,
                                        n_perm=max(500, n_perm // 10), seed=seed)
                lrows.append(dict(dropped=subj, corr=r['corr'], p=r['p'],
                                  n_electrodes=r['n_electrodes']))
            except ValueError:
                continue
        out_loso = pd.DataFrame(lrows)
    return table, out_loso


# ---------------------------------------------------------------------------
# Figure 5 (docs/paper_draft.md §1.4): a, the overlap at both levels;
# b, each adaptation against its own and the other base effect
# ---------------------------------------------------------------------------
# F5b's bars in plot order: (adaptation, base effect, matched?, delta_tracking row)
FIG5_BARS = (('LWPC', 'congruency', True, 'congruency vs LWPC'),
             ('LWPC', 'switch', False, 'switch vs LWPC (crossed)'),
             ('LWPS', 'switch', True, 'switch vs LWPS'),
             ('LWPS', 'congruency', False, 'congruency vs LWPS (crossed)'))
# congruency is the stability effect and switch the flexibility one: the
# S_only / F_only blue and orange of the other N4 and segregation figures
FIG5_COLORS = {'congruency': GROUP_COLORS['S_only'], 'switch': GROUP_COLORS['F_only']}
_INK, _MUTED, _RULE = '#262626', '#737373', '#cfcfcf'


def _fmt_p(p):
    return f'{p:.2g}'


def _fmt_r(r, digits=2):
    return f'{r:.{digits}f}'.replace('-', '−')    # the axes' minus sign


def figure5_points(scores, min_elec=3):
    """F5a's points, ``(base, adapt)``: congruency vs switch and LWPC vs LWPS as
    ``x_resid``/``y_resid``, the responsiveness-residualised, participant-centred
    scores each pre-specified test correlates. ``prepare_continuous`` on the raw
    scores, the transform behind the segregation run's ``continuous.csv``."""
    from .stability_flexibility_segregation import prepare_continuous
    cols = ['subject', 'electrode', 'x_resid', 'y_resid']
    return tuple(prepare_continuous(scores[['subject', 'electrode', x, y, 'resp']]
                                    .rename(columns={x: 'x', y: 'y'}), min_elec=min_elec)[cols]
                 .reset_index(drop=True)
                 for x, y in (('cong_score', 'switch_score'), ('lwpc_score', 'lwps_score')))


def _figure5_test(points, scores, json_path, view, per_split, seg_dir, min_elec, n_perm, seed):
    """The pre-specified test behind one half of F5a, from the segregation run
    when it tested these electrodes, else recomputed (see :func:`figure5`)."""
    from .stability_flexibility_segregation import split_resolved_corr
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            r = json.load(f)
        if (r['n_electrodes'], r['n_subjects']) == (len(points), points['subject'].nunique()):
            return dict(r, source=json_path)
    ps_csv = os.path.join(seg_dir, 'per_split.csv') if seg_dir else None
    if per_split is None and ps_csv and os.path.exists(ps_csv):
        per_split = pd.read_csv(ps_csv)
    if per_split is None:
        raise ValueError(f"{json_path} is missing or tested other electrodes, and there "
                         "is no per-split table to recompute it from")
    ps = view(per_split[per_split['electrode'].isin(scores['electrode'])])
    resp = scores.drop_duplicates('electrode').set_index('electrode')['resp']
    r = split_resolved_corr(ps, resp, min_elec=min_elec, n_perm=n_perm, seed=seed)
    return dict(r, source=f'recomputed on these electrodes ({n_perm} permutations)')


def figure5_bars(tracking):
    """F5b's four rows of ``delta_tracking.csv`` (:func:`delta_tracking_test`) in
    plot order, and its dm-vs-delta rows, the matched-minus-crossed test."""
    t = tracking.set_index('comparison')
    missing = [row for *_, row in FIG5_BARS if row not in t.index]
    if missing or 'dm vs delta' not in t.index:
        raise ValueError(f"delta_tracking lacks rows {missing or ['dm vs delta']}")
    bars = pd.DataFrame([dict(adaptation=a, base_effect=b, matched=m, comparison=row,
                              **t.loc[row, ['corr', 'p', 'n_electrodes', 'n_subjects']])
                         for a, b, m, row in FIG5_BARS])
    links = t.loc[[r for r in ('dm vs delta', 'dm vs delta, + MNI covariates') if r in t.index],
                  ['corr', 'p', 'n_electrodes', 'n_subjects']]
    return bars, links


def _figure5_overlap(ax, d, stat, xlabel, ylabel, title, extra=()):
    """One half of F5a: the points, annotated with the pre-specified test's r,
    p and n. No fit line: the points' own correlation is not the test."""
    ax.axhline(0, color=_RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=_RULE, lw=0.6, zorder=0)
    ax.scatter(d['x_resid'], d['y_resid'], s=5, color=_INK, alpha=0.4, linewidths=0,
               rasterized=True)
    lim = 1.08 * np.nanmax(np.abs(d[['x_resid', 'y_resid']].to_numpy(float)))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal', xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc='left', fontsize=7, color=_INK)
    text = [f"r = {_fmt_r(stat['corr'])}, p = {_fmt_p(stat['p'])}",
            f"{stat['n_electrodes']} electrodes, {stat['n_subjects']} participants", *extra]
    ax.text(0.03, 0.97, '\n'.join(text), transform=ax.transAxes, va='top', ha='left',
            fontsize=5.5, color=_INK, linespacing=1.3,
            bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.85))


def _figure5_tracking(ax, bars, links):
    """F5b: matched bars filled, crossed open, grouped by adaptation effect, one
    y axis from 0. The dm-vs-delta correlation's covariance is the matched
    covariances minus the crossed ones (§16.6.5 of n4_continuous_anatomy.md), so
    it annotates both pairs. No error bars: delta_tracking has no interval, and
    the permutation null is not one."""
    from matplotlib.patches import Patch
    xs = np.array([0.0, 1.0, 2.5, 3.5])
    width = 0.6
    for x, b in zip(xs, bars.itertuples()):
        c = FIG5_COLORS[b.base_effect]
        ax.bar(x, b.corr, width=width, color=c if b.matched else 'white', edgecolor=c,
               linewidth=1.0)
    top = max(float(bars['corr'].max()), 0.0) or 0.1
    for x, r in zip(xs, bars['corr']):
        ax.text(x, r + (0.03 if r >= 0 else -0.03) * top, _fmt_r(r), ha='center',
                va='bottom' if r >= 0 else 'top', fontsize=5.5, color=_INK)
    ax.axhline(0, color=_INK, lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(bars['base_effect'], fontsize=5.5)
    ax.tick_params(axis='x', length=0)
    for name in ('LWPC', 'LWPS'):
        ax.text(xs[(bars['adaptation'] == name).to_numpy()].mean(), -0.13, name,
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7,
                fontweight='bold', color=_INK)
    ax.set_ylabel('r, separate halves')

    y, lo, hi = 1.25 * top, xs[0] - width / 2, xs[-1] + width / 2
    ax.plot([lo, lo, hi, hi], [y - 0.05 * top, y, y, y - 0.05 * top], color=_INK, lw=0.7)
    names = {'dm vs delta': 'matched − crossed',
             'dm vs delta, + MNI covariates': 'with MNI coordinates'}
    ax.text(xs.mean(), y + 0.04 * top,
            '\n'.join(f"{names[k]}: r = {_fmt_r(row['corr'], 3)}, p = {_fmt_p(row['p'])}"
                      for k, row in links.iterrows()),
            ha='center', va='bottom', fontsize=5.5, color=_INK, linespacing=1.3)
    ax.set_ylim(min(0.0, 1.3 * float(bars['corr'].min())), 1.75 * top)
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.legend(handles=[Patch(facecolor=_MUTED, edgecolor=_MUTED, label='own base effect (matched)'),
                       Patch(facecolor='white', edgecolor=_MUTED,
                             label='other base effect (crossed)')],
              loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=5.5,
              ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0,
              title=f"each bar: p ≤ {_fmt_p(bars['p'].max())}", title_fontsize=5.5)


def plot_figure5(base, adapt, stat_base, stat_adapt, bars, links, out_stem, centroid=None):
    """Draw F5 and write ``<out_stem>.png`` and ``<out_stem>.pdf`` (text stays
    text in the PDF). Inputs as :func:`figure5` builds them."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rc = {'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.labelcolor': _INK,
          'xtick.labelsize': 6, 'ytick.labelsize': 6, 'xtick.color': _INK, 'ytick.color': _INK,
          'axes.edgecolor': _INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
          'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.2, 2.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.4,
                              left=0.07, right=0.98, top=0.86, bottom=0.3)
        ax_base, ax_adapt, ax_b = (fig.add_subplot(gs[0, i]) for i in range(3))
        _figure5_overlap(ax_base, base, stat_base, 'congruency (d)', 'switch (d)',
                         'Base effects')
        extra = () if centroid is None else (
            f"centroids {centroid['distance']:.1f} mm apart, p = {_fmt_p(centroid['p'])}",)
        _figure5_overlap(ax_adapt, adapt, stat_adapt, 'LWPC (d)', 'LWPS (d)', 'Adaptation',
                         extra)
        _figure5_tracking(ax_b, bars, links)
        for ax, letter in ((ax_base, 'a'), (ax_b, 'b')):
            ax.text(-0.3, 1.13, letter, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='bottom', ha='left', color=_INK)
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        for p in paths:
            fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig)
    return paths


def figure5(scores, tracking, out_dir, per_split=None, seg_dir=None, centroid=None,
            min_elec=3, n_perm=10000, seed=1):
    """Figure 5 from one anatomy run: ``fig5.png``/``.pdf``, ``fig5a_points.csv``
    and ``fig5b_bars.csv`` in ``out_dir``. Returns the tables and summary lines.

    ``scores``: :func:`attach_scores`'s table from a MAIN_EFFECTS=1 run;
    ``tracking``: :func:`delta_tracking_test`'s table. Panel a's r, p and n are
    the segregation run's own (``seg_dir``: ``correlation_main_effects.json``,
    ``correlation.json``) when that run tested as many electrodes and
    participants as there are points; otherwise, e.g. after an ROI filter, they
    are recomputed on these electrodes with ``split_resolved_corr`` from
    ``per_split`` (or ``seg_dir/per_split.csv``). ``centroid``: the
    ``centroid_shuffle_test`` result for panel a's LWPC/LWPS half, if any.
    """
    from .stability_flexibility_segregation import main_effect_view
    base, adapt = figure5_points(scores, min_elec)
    stats = [_figure5_test(pts, scores, os.path.join(seg_dir, name) if seg_dir else None,
                           view, per_split, seg_dir, min_elec, n_perm, seed)
             for pts, name, view in ((base, 'correlation_main_effects.json', main_effect_view),
                                     (adapt, 'correlation.json', lambda ps: ps))]
    bars, links = figure5_bars(tracking)

    os.makedirs(out_dir, exist_ok=True)
    pd.concat([base.assign(panel='congruency vs switch'),
               adapt.assign(panel='LWPC vs LWPS')]).to_csv(
        os.path.join(out_dir, 'fig5a_points.csv'), index=False)
    bars.to_csv(os.path.join(out_dir, 'fig5b_bars.csv'), index=False)
    paths = plot_figure5(base, adapt, *stats, bars, links, os.path.join(out_dir, 'fig5'),
                         centroid=centroid)

    lines = [f"  FIGURE 5 — {', '.join(os.path.basename(p) for p in paths)}, "
             "fig5a_points.csv, fig5b_bars.csv"]
    for name, pts, st in (('congruency vs switch', base, stats[0]),
                          ('LWPC vs LWPS', adapt, stats[1])):
        check = ('' if (len(pts), pts['subject'].nunique())
                 == (st['n_electrodes'], st['n_subjects'])
                 else '   <- points and test differ: check min_elec and dropped electrodes')
        lines.append(f"  a  {name:20s} r = {st['corr']:+.3f}  p = {_fmt_p(st['p'])}  "
                     f"test {st['n_electrodes']} / {st['n_subjects']}, points {len(pts)} / "
                     f"{pts['subject'].nunique()}  [{st['source']}]{check}")
    if centroid is not None:
        lines.append(f"  a  centroid distance {centroid['distance']:.2f} mm, "
                     f"p = {_fmt_p(centroid['p'])} (types shuffled within participant)")
    for b in bars.itertuples():
        lines.append(f"  b  {b.comparison:30s} r = {b.corr:+.3f}  p = {_fmt_p(b.p)}")
    for k, row in links.iterrows():
        lines.append(f"  b  {k:30s} r = {row['corr']:+.3f}  p = {_fmt_p(row['p'])}"
                     "   (matched − crossed)")
    return dict(base=base, adapt=adapt, tests=stats, bars=bars, links=links, files=paths,
                lines=lines)


# ---------------------------------------------------------------------------
# Figure 5, the two anatomy results in one figure (advisor meeting 2026-10-02;
# docs/paper_draft.md §1.4)
# ---------------------------------------------------------------------------
# On the LWPC (x) against LWPS (y) scatter the two results lie along
# perpendicular directions. The overlap (r > 0) is spread ALONG the identity
# line. The gradient is a shift ACROSS it: LWPC - LWPS is each point's signed
# distance from the line (times sqrt 2), so electrodes below the line lean
# LWPC and electrodes above it lean LWPS. Colouring the points by height band
# and marking each band's centroid shows both in one panel; the
# balance-by-height panel shows the gradient in the coordinate test's units.
HEIGHT_BANDS = ('ventral', 'middle', 'dorsal')
# One-hue ordinal ramp, light (ventral) to dark (dorsal), validated as an
# ordinal ramp: monotone lightness, visible steps, light end >= 2:1 on white.
# Violet, so it is not read as the blue/orange congruency/switch identity.
HEIGHT_COLORS = dict(zip(HEIGHT_BANDS, ('#a598e8', '#6a56cf', '#33218a')))


def height_bands(values, edges=None, labels=HEIGHT_BANDS):
    """Tertiles of ``values`` (MNI z) across electrodes, the cut S-N4's panel c
    uses (``n4_section16_followups.band_tables``): fixed by rule, not chosen
    after looking. Pass ``edges`` to apply a cut computed on another set.
    Returns ``(labels, edges)``."""
    v = pd.Series(values, dtype=float)
    if edges is None:
        edges = np.nanpercentile(v, [0, 100 / 3, 200 / 3, 100])
    return pd.cut(v, edges, labels=list(labels), include_lowest=True), np.asarray(edges)


def figure5_height_points(scores, axis='mni_z', min_elec=3):
    """The points of the combined panel, and the band edges.

    LWPC and LWPS are the pre-specified test's scores (responsiveness
    regressed out, participant-centred; ``prepare_continuous``) on the pooled
    scale of ``delta``, with each score's overall mean added back. So the
    identity line means LWPC = LWPS in the units the gradient is tested in,
    and ``x - y`` is delta with participant and responsiveness removed. Adding
    constants does not move any correlation. The bands are tertiles of
    ``axis`` over every electrode with coordinates, as in S-N4; electrodes
    without coordinates stay in the table with no band.
    """
    from .stability_flexibility_segregation import prepare_continuous
    pts = prepare_continuous(
        scores[['subject', 'electrode', 'lwpc_s', 'lwps_s', 'resp']]
        .rename(columns={'lwpc_s': 'x', 'lwps_s': 'y'}), min_elec=min_elec)
    pts['x_plot'] = pts['x_resid'] + pts['x'].mean()
    pts['y_plot'] = pts['y_resid'] + pts['y'].mean()
    coords = [c for c in ('mni_x', 'mni_y', 'mni_z') if c in scores.columns]
    pts = pts[['subject', 'electrode', 'x_resid', 'y_resid', 'x_plot', 'y_plot', 'resp']].merge(
        scores[['electrode', *coords]].drop_duplicates('electrode'), on='electrode', how='left')
    pts['balance'] = pts['x_plot'] - pts['y_plot']
    _, edges = height_bands(scores[axis].dropna())
    band, _ = height_bands(pts[axis], edges=edges)
    pts['band'] = band.astype(object).where(band.notna(), None).to_numpy()
    return pts.reset_index(drop=True), edges


def height_centroids(points, n_boot=2000, seed=0):
    """Each band's centroid on the scatter (electrode means of ``x_plot`` and
    ``y_plot``) with a participant bootstrap: the 2 x 2 covariance of the
    centroid (for its 95 % ellipse) and the 95 % interval of its balance
    ``x - y``. One draw of participants is shared by all bands."""
    d = points.dropna(subset=['band'])
    subjects = np.unique(d['subject'].astype(str))
    s_index = {s: i for i, s in enumerate(subjects)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(int(n_boot), len(subjects)))
    rows = []
    for band in HEIGHT_BANDS:
        g = d[d['band'] == band]
        if g.empty:
            continue
        si = g['subject'].astype(str).map(s_index).to_numpy()
        n = np.bincount(si, minlength=len(subjects)).astype(float)
        sx = np.bincount(si, weights=g['x_plot'].to_numpy(float), minlength=len(subjects))
        sy = np.bincount(si, weights=g['y_plot'].to_numpy(float), minlength=len(subjects))
        nb = n[draws].sum(1)
        okb = nb > 0
        bx = sx[draws].sum(1)[okb] / nb[okb]
        by = sy[draws].sum(1)[okb] / nb[okb]
        cov = np.cov(np.vstack([bx, by]))
        x, y = float(g['x_plot'].mean()), float(g['y_plot'].mean())
        lo, hi = np.percentile(bx - by, [2.5, 97.5])
        rows.append(dict(band=band, n_electrodes=int(len(g)),
                         n_participants=int(g['subject'].nunique()),
                         z_min=float(g['mni_z'].min()) if 'mni_z' in g else np.nan,
                         z_max=float(g['mni_z'].max()) if 'mni_z' in g else np.nan,
                         x=x, y=y, balance=x - y, balance_lo=float(lo), balance_hi=float(hi),
                         cov_xx=float(cov[0, 0]), cov_xy=float(cov[0, 1]),
                         cov_yy=float(cov[1, 1])))
    return pd.DataFrame(rows)


def balance_by_height(scores, edges, axis='mni_z', value_col='delta', covariates=('resp',)):
    """Participant means +/- SEM of the adjusted balance per height band.

    ``value_col`` with participant and responsiveness offsets removed and its
    mean added back (as in ``n4_section16_followups.adjusted``), in the units of
    the coordinate test. Each participant contributes one mean to each band it
    has electrodes in. Returns ``(summary, per_participant)``.
    """
    d = scores.dropna(subset=[value_col, axis]).reset_index(drop=True)
    X, _ = _nuisance_design(d, covariates=covariates)
    v = d[value_col].to_numpy(float)
    d['adjusted'] = v - X @ np.linalg.lstsq(X, v, rcond=None)[0] + v.mean()
    d['band'], _ = height_bands(d[axis], edges=edges)
    per = d.groupby(['band', 'subject'], observed=True)['adjusted'].mean().reset_index()
    summary = (per.groupby('band', observed=True)['adjusted']
               .agg(mean='mean', sem='sem', n_participants='count').reset_index())
    summary = summary.merge(d.groupby('band', observed=True)[axis].median()
                            .rename(f'{axis}_median').reset_index(), on='band')
    summary['band'] = summary['band'].astype(str)
    per['band'] = per['band'].astype(str)
    return summary, per


def _ellipse(ax, x, y, cov, color, n_sd=np.sqrt(5.991)):
    """95 % ellipse of a 2-D normal with covariance ``cov`` (chi-square, 2 df)."""
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(np.asarray(cov, float))
    vals = np.clip(vals, 0.0, None)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    ax.add_patch(Ellipse((x, y), 2 * n_sd * np.sqrt(vals[1]), 2 * n_sd * np.sqrt(vals[0]),
                         angle=angle, facecolor=color, alpha=0.18, edgecolor=color,
                         linewidth=0.9, zorder=3))


def plot_figure5_height(points, centroids, balance, stat, edges, out_stem, axis='mni_z',
                        slope=None, brain_png=None):
    """Draw the combined anatomy figure and write ``<out_stem>.png``/``.pdf``.

    a, the height bands on a sagittal projection of the electrodes (or
    ``brain_png``, a rendered brain in the same colours), with the cuts drawn;
    b, LWPC against LWPS coloured by band, with the identity line; c, the
    band centroids enlarged (the box in b), each with its 95 %
    participant-bootstrap ellipse: the gradient is their spread across the
    identity line; d, the adjusted balance by band, participant means +/- SEM,
    with the slope tests in ``slope`` (a list of text lines). Inputs as
    :func:`figure5_height` builds them.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    rc = {'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.labelcolor': _INK,
          'xtick.labelsize': 6, 'ytick.labelsize': 6, 'xtick.color': _INK, 'ytick.color': _INK,
          'axes.edgecolor': _INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
          'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    d = points.dropna(subset=['band'])
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.2, 2.5))
        gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 1.15, 0.85, 0.8], wspace=0.5,
                              left=0.06, right=0.98, top=0.86, bottom=0.2)
        ax_a, ax_b, ax_z, ax_c = (fig.add_subplot(gs[0, i]) for i in range(4))

        # a: where the bands are
        if brain_png and os.path.exists(brain_png):
            ax_a.imshow(plt.imread(brain_png))
            ax_a.set_axis_off()
        else:
            for band in HEIGHT_BANDS:
                g = d[d['band'] == band]
                ax_a.scatter(g['mni_y'], g[axis], s=4, color=HEIGHT_COLORS[band],
                             linewidths=0, rasterized=True)
            for e in edges[1:-1]:
                ax_a.axhline(e, color=_MUTED, lw=0.6, ls=(0, (3, 2)))
                ax_a.text(0.99, e, f'z = {e:.0f} mm', transform=ax_a.get_yaxis_transform(),
                          va='bottom', ha='right', fontsize=5.5, color=_INK,
                          bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none',
                                    alpha=0.85))
            ax_a.set(xlabel='MNI y (mm), posterior → anterior', ylabel='MNI z (mm)')
            ax_a.set_aspect('equal', adjustable='datalim')
        ax_a.set_title('Height bands', loc='left', fontsize=7, color=_INK)

        # b: the overlap along the identity line, the gradient across it
        for band in HEIGHT_BANDS:
            g = d[d['band'] == band]
            ax_b.scatter(g['x_plot'], g['y_plot'], s=5, color=HEIGHT_COLORS[band], alpha=0.45,
                         linewidths=0, rasterized=True, zorder=2)
        vals = d[['x_plot', 'y_plot']].to_numpy(float)
        lo, hi = np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5)
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax_b.plot([lo, hi], [lo, hi], color=_MUTED, lw=0.7, zorder=1)
        ax_b.text(0.03, 0.97, 'leans LWPS', transform=ax_b.transAxes, ha='left', va='top',
                  fontsize=5.5, color=_MUTED)
        ax_b.text(0.97, 0.03, 'leans LWPC', transform=ax_b.transAxes, ha='right',
                  va='bottom', fontsize=5.5, color=_MUTED)
        ax_b.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal',
                 xlabel='LWPC (SD units)', ylabel='LWPS (SD units)')
        ax_b.set_title('Adaptation by electrode', loc='left', fontsize=7, color=_INK)
        if stat is not None:
            ax_b.text(0.97, 0.97,
                      f"r = {_fmt_r(stat['corr'])}, p = {_fmt_p(stat['p'])}\n"
                      f"{stat['n_electrodes']} electrodes, {stat['n_subjects']} participants",
                      transform=ax_b.transAxes, ha='right', va='top', fontsize=5.5,
                      color=_INK, linespacing=1.3,
                      bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.85))

        # c: the band centroids, enlarged; the box in b marks the region
        ext = [np.sqrt(5.991 * max(v, 0.0)) for v in
               np.r_[centroids['cov_xx'], centroids['cov_yy']]]
        span = max(float(np.ptp(np.r_[centroids['x'], centroids['y']])) + 2 * max(ext, default=0),
                   1e-3)
        mid = float(np.mean(np.r_[centroids['x'], centroids['y']]))
        zlo, zhi = mid - 0.6 * span, mid + 0.6 * span
        ax_b.add_patch(Rectangle((zlo, zlo), zhi - zlo, zhi - zlo, fill=False,
                                 edgecolor=_INK, linewidth=0.6, zorder=6))
        ax_z.plot([zlo, zhi], [zlo, zhi], color=_MUTED, lw=0.7, zorder=1)
        ax_z.text(zhi, zhi, 'LWPC = LWPS ', ha='right', va='top', fontsize=5.5,
                  color=_MUTED, rotation=45, rotation_mode='anchor')
        for c in centroids.itertuples():
            col = HEIGHT_COLORS[c.band]
            _ellipse(ax_z, c.x, c.y, [[c.cov_xx, c.cov_xy], [c.cov_xy, c.cov_yy]], col)
            ax_z.scatter([c.x], [c.y], s=30, color=col, edgecolor='white', linewidth=1.0,
                         zorder=5)
        ax_z.set(xlim=(zlo, zhi), ylim=(zlo, zhi), aspect='equal',
                 xlabel='LWPC (SD units)', ylabel='LWPS (SD units)')
        ax_z.set_title('Band centroids (box in b)', loc='left', fontsize=7, color=_INK)
        # one legend for every panel, under b and c (identity is never colour alone)
        box_b, box_z = ax_b.get_position(), ax_z.get_position()
        fig.legend(handles=[Line2D([], [], marker='o', ls='', color=HEIGHT_COLORS[b],
                                   markersize=4, label=b) for b in HEIGHT_BANDS],
                   loc='upper center', bbox_to_anchor=((box_b.x0 + box_z.x1) / 2, 0.06),
                   ncol=3, frameon=False, fontsize=5.5, handletextpad=0.2,
                   columnspacing=0.9, title='height tertile (MNI z); c: centroid with 95 % '
                   'participant-bootstrap region', title_fontsize=5.5)

        # d: the gradient in the coordinate test's units
        b = balance.set_index('band').reindex([x for x in HEIGHT_BANDS
                                               if x in set(balance['band'])])
        xs = b[f'{axis}_median'].to_numpy(float)
        ax_c.axhline(0, color=_RULE, lw=0.6, zorder=0)
        ax_c.plot(xs, b['mean'], color=_RULE, lw=0.8, zorder=1)
        for band, row in b.iterrows():
            ax_c.errorbar(row[f'{axis}_median'], row['mean'], yerr=row['sem'], fmt='o',
                          color=HEIGHT_COLORS[band], ecolor=_INK, elinewidth=0.7,
                          capsize=0, markersize=4.5, markeredgecolor='white',
                          markeredgewidth=0.6, zorder=3)
        ax_c.set(xlabel='height, MNI z (mm)', ylabel='LWPC − LWPS (SD units)')
        ax_c.set_title('Balance by height', loc='left', fontsize=7, color=_INK)
        if slope:
            # the corner the points leave free: falling balance -> lower left
            falling = len(b) > 1 and b['mean'].iloc[-1] < b['mean'].iloc[0]
            ax_c.text(0.03, 0.03 if falling else 0.97, '\n'.join(slope),
                      transform=ax_c.transAxes, ha='left', va='bottom' if falling else 'top',
                      fontsize=5.5, color=_INK, linespacing=1.3)

        for ax, letter in ((ax_a, 'a'), (ax_b, 'b'), (ax_z, 'c'), (ax_c, 'd')):
            ax.text(-0.3, 1.1, letter, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', va='bottom', ha='left', color=_INK)
        paths = [f'{out_stem}.{ext}' for ext in ('png', 'pdf')]
        for p in paths:
            fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig)
    return paths


def figure5_height(scores, out_dir, per_split=None, seg_dir=None, coord_res=None,
                   participant_res=None, axis='mni_z', coord_cols=('mni_y', 'mni_z', 'mni_x'),
                   min_elec=3, n_boot=2000, n_perm=10000, seed=1, brain_png=None):
    """The combined anatomy figure from one anatomy run: ``fig5_height.png``/
    ``.pdf`` and its tables (``fig5_height_points.csv``,
    ``fig5_height_centroids.csv``, ``fig5_height_balance.csv``) in ``out_dir``.

    ``scores``: :func:`attach_scores`'s table with coordinates. Panel b's r is
    the pre-specified LWPC-LWPS test, taken as in :func:`figure5` (the
    segregation run's ``correlation.json`` in ``seg_dir`` when it tested the
    same electrodes, else recomputed from ``per_split``). Panel d is annotated
    with ``coord_res`` (:func:`relative_score_coordinate_test`; computed here if
    absent) and ``participant_res`` (:func:`coordinate_slope_by_participant`), if
    given. The summary lines include a check that the plotted balance gives the
    coordinate test's slope.
    """
    pts, edges = figure5_height_points(scores, axis, min_elec)
    stat = None
    try:
        stat = _figure5_test(pts, scores, os.path.join(seg_dir, 'correlation.json')
                             if seg_dir else None, lambda ps: ps, per_split, seg_dir,
                             min_elec, n_perm, seed)
    except ValueError as exc:
        print(f"[F5 height] panel b without its test: {exc}")
    plotted = pts.dropna(subset=['band'])
    cents = height_centroids(plotted, n_boot=n_boot, seed=seed)
    balance, per_participant = balance_by_height(scores, edges, axis=axis)

    if coord_res is None:
        coord_res = relative_score_coordinate_test(scores, n_perm=n_perm, seed=seed)
    sl = coord_res['all']['slopes'].set_index('axis').loc[axis]
    slope = [f"{axis[-1]} slope {sl['slope_per_mm']:+.4f} SD/mm".replace('-', '−'),
             f"electrodes: p = {_fmt_p(sl['p'])}"]
    if participant_res is not None:
        slope.append(f"participants: p = {_fmt_p(participant_res['weighted']['p_signflip'])}")
        rs = participant_res.get('mixed', {}).get('random_slope', {})
        if 'p' in rs:
            slope.append(f"random slope: p = {_fmt_p(rs['p'])}")

    # the plotted balance against the coordinate test's own fit
    chk = _coordinate_fit(plotted.dropna(subset=list(coord_cols)), 'balance', coord_cols,
                          ('resp',), 0, seed)
    chk_slope = float(chk['slopes'].set_index('axis').loc[axis, 'slope_per_mm'])

    os.makedirs(out_dir, exist_ok=True)
    pts.to_csv(os.path.join(out_dir, 'fig5_height_points.csv'), index=False)
    cents.to_csv(os.path.join(out_dir, 'fig5_height_centroids.csv'), index=False)
    balance.to_csv(os.path.join(out_dir, 'fig5_height_balance.csv'), index=False)
    per_participant.to_csv(os.path.join(out_dir, 'fig5_height_balance_by_participant.csv'),
                           index=False)
    paths = plot_figure5_height(pts, cents, balance, stat, edges,
                                os.path.join(out_dir, 'fig5_height'), axis=axis, slope=slope,
                                brain_png=brain_png)

    lines = [f"  FIGURE 5 (height) — {', '.join(os.path.basename(p) for p in paths)}, "
             "fig5_height_points.csv, fig5_height_centroids.csv, fig5_height_balance.csv",
             f"  bands: tertiles of {axis} at {edges[1]:.1f} and {edges[2]:.1f} mm; "
             f"{len(plotted)} of {len(pts)} points have coordinates"]
    if stat is not None:
        lines.append(f"  b  LWPC vs LWPS r = {stat['corr']:+.3f}  p = {_fmt_p(stat['p'])}  "
                     f"[{stat['source']}]")
    for c in cents.itertuples():
        lines.append(f"  c  {c.band:8s} centroid LWPC {c.x:+.3f}, LWPS {c.y:+.3f}; balance "
                     f"{c.balance:+.3f} [{c.balance_lo:+.3f}, {c.balance_hi:+.3f}]  "
                     f"({c.n_electrodes} electrodes, {c.n_participants} participants)")
    for r in balance.itertuples():
        lines.append(f"  d  {r.band:8s} adjusted LWPC − LWPS {r.mean:+.3f} ± {r.sem:.3f} SEM "
                     f"({r.n_participants} participants)")
    lines.append(f"  d  {axis} slope {sl['slope_per_mm']:+.5f}/mm (p = {_fmt_p(sl['p'])}); the "
                 f"plotted balance gives {chk_slope:+.5f}/mm"
                 + ("" if np.isclose(chk_slope, sl['slope_per_mm'], rtol=0.25, atol=1e-4)
                    else "   <- differs from the test: check the scaling"))
    return dict(points=pts, centroids=cents, balance=balance, stat=stat, edges=edges,
                files=paths, lines=lines, check_slope=chk_slope)


# ---------------------------------------------------------------------------
# local similarity: is the LWPC-LWPS balance intermixed at the recorded scale?
# (advisor meeting 2026-10-02; §19.3 of docs/n4_continuous_anatomy.md)
# ---------------------------------------------------------------------------
# "Intermixed" is a claim about arrangement: no patches of LWPC-leaning
# electrodes next to patches of LWPS-leaning ones. The overlap correlation does
# not test it, and the height gradient is structure at the scale of the whole
# region. This asks the local question: for pairs of electrodes in the same
# participant, how similar are their scores, as a function of the distance
# between them? Similarity is taken ACROSS trial halves: one electrode's half A
# against the other's half B. At distance 0 the same quantity is the
# electrode's own split-half reliability.
#
# Two things make this valid, and both were learned the hard way (2026-10-05):
#
# 1. The halves must be shared by all of a participant's electrodes
#    (`compute_sensitivities_per_split(shared_split=True)`). With a split drawn
#    per electrode, electrode i's half A shares about half its trials with
#    electrode j's half B, and the trial noise neighbouring contacts share
#    then shows up as near-range "similarity" for every score. On simulated
#    data with no local structure at all, per-electrode splits gave a
#    near-range excess at p < 0.02 for all five scores.
# 2. Inference must take participants, not pairs, as the units. One dataset's
#    estimation noise is itself spatially smooth (neighbours share it), so
#    pairs are not exchangeable and a null that shuffles positions is too
#    narrow: 13 % false positives at alpha = 0.05 on the same simulations. The
#    shuffle is kept only as each participant's baseline; the test flips the
#    sign of each participant's excess over it (2.5 % false positives with six
#    simulated participants).
#
# Patches make near pairs more similar than far ones; intermixing makes the
# curve flat once the linear gradient is removed. A single score (LWPC alone,
# or congruency) is the positive control: if single maps share signal with
# their neighbours but the balance does not, the balance varies on a finer
# scale than the sampling, i.e. it is intermixed there.
LOCAL_BINS_MM = (10.0, 20.0, 40.0)


def _channel_poles(electrode):
    """The contacts a channel is built from: ``'D57-LA1-LA2'`` -> ``{'LA1', 'LA2'}``
    for a bipolar derivative, ``{'LA1'}`` for a monopolar contact. The id is
    ``{subject}-{channel}`` and subject ids carry no hyphen."""
    parts = str(electrode).split('-')
    return set(parts[1:]) if len(parts) > 1 else {parts[0]}


def _local_score_halves(per_split, scores):
    """``{name: (half-A column, half-B column)}`` for the scores the per-split
    table can give, with the balance built as ``delta`` is: each effect's half
    divided by its full-data pooled scale (read off ``scores``)."""
    s = scores.drop_duplicates('electrode').set_index('electrode')
    sd = {k: float(np.nanmedian(s[f'{k}_score'] / s[f'{k}_s']))
          for k in ('lwpc', 'lwps') if f'{k}_score' in s and f'{k}_s' in s}
    if len(sd) < 2:
        sd = {'lwpc': float(np.nanstd(per_split[['xA', 'xB']].to_numpy(), ddof=1)),
              'lwps': float(np.nanstd(per_split[['yA', 'yB']].to_numpy(), ddof=1))}
    ps = per_split.copy()
    for h in ('A', 'B'):
        ps[f'balance{h}'] = ps[f'x{h}'] / sd['lwpc'] - ps[f'y{h}'] / sd['lwps']
    names = {'LWPC − LWPS': ('balanceA', 'balanceB'), 'LWPC': ('xA', 'xB'),
             'LWPS': ('yA', 'yB')}
    if {'mxA', 'mxB', 'myA', 'myB'} <= set(ps.columns):
        names.update({'congruency': ('mxA', 'mxB'), 'switch': ('myA', 'myB')})
    return ps, names


def is_shared_split(per_split):
    """True when every row of the per-split table came from a split shared by
    all of a participant's electrodes (``split_scheme == 'participant'``)."""
    return ('split_scheme' in per_split.columns
            and bool((per_split['split_scheme'] == 'participant').all()))


def local_similarity(per_split, scores_with_coords, bins_mm=LOCAL_BINS_MM,
                     remove_gradient=True, coord_cols=('mni_x', 'mni_y', 'mni_z'),
                     method='spearman', min_elec=3, n_perm=2000, n_boot=2000, seed=0,
                     exclude_shared_contacts=True, require_shared_split=True,
                     min_reliable_share=0.9):
    """Cross-half similarity of electrode pairs within participant, by distance.

    ``per_split`` must come from ``compute_sensitivities_per_split(...,
    shared_split=True)`` (the segregation job with ``SHARED_SPLIT=1``, or
    ``n4_section19_followups.py --long-df``); a table split per electrode is
    refused (see the section comment). ``require_shared_split=False`` is for
    tables whose halves carry independent noise by construction, e.g. planted
    test data.

    For each score (the LWPC - LWPS balance, LWPC, LWPS and, from a
    MAIN_EFFECTS=1 run, congruency and switch):

    1. per split and half, the scores are residualised on responsiveness and,
       with ``remove_gradient``, on the coordinates (centred within
       participant: the linear gradient), then centred within participant
       (the pre-specified test's treatment); with ``method='spearman'`` ranked
       and re-centred within participant; then scaled to unit mean square;
    2. ``C[i, j] = mean_k 1/2 (A_k[i] B_k[j] + B_k[i] A_k[j])`` for electrodes i,
       j of one participant. ``C[i, i]`` is electrode i's split-half
       reliability;
    3. pairs are binned by Euclidean distance (``bins_mm`` edges; the last bin
       is open) and summed per participant.

    **Baseline and test.** Each participant's baseline for a bin is the mean
    of that bin under shuffles of its electrode positions (``n_perm``); its
    excess is observed minus baseline (centring makes pairs slightly
    anti-correlated by construction, and the baseline carries the same bias).
    ``excess`` pools participants by pair count; ``p_greater`` flips the sign
    of whole participants' excesses (one-sided, more similar than baseline);
    intervals come from a participant bootstrap.

    ``relative`` divides the excess by the score's reliability, the share of
    an electrode's reliable signal its neighbours carry. It is only reported
    when the reliability is positive in at least ``min_reliable_share`` of the
    bootstrap draws; at a reliability near or below zero the ratio is not
    defined, and ``notes`` says so. Pairs of bipolar channels sharing a
    contact are dropped (``exclude_shared_contacts``).

    Returns a dict: ``table`` (one row per score and bin), ``contrasts``
    (nearest minus farthest bin, participant sign-flip p), ``comparison`` (the
    balance's relative nearest-bin excess minus each single score's, paired
    participant bootstrap, where both are defined), ``notes`` and the counts.
    """
    from scipy.stats import rankdata
    from .stability_flexibility_segregation import _residualised_split_matrices

    if require_shared_split and not is_shared_split(per_split):
        raise ValueError(
            "local_similarity needs one trial split per participant, shared by all "
            "its electrodes (split_scheme == 'participant'). This per-split table was "
            "split electrode by electrode, so one electrode's half A shares trials "
            "with its neighbours' half B and shared trial noise reads as local "
            "similarity. Rescore with compute_sensitivities_per_split(shared_split="
            "True): SHARED_SPLIT=1 in the segregation job, or "
            "n4_section19_followups.py --long-df <segregation run>/long_df.csv.")

    coord_cols = list(coord_cols)
    s = scores_with_coords.drop_duplicates('electrode').set_index('electrode')
    have = s[coord_cols + ['resp']].dropna().index
    ps, names = _local_score_halves(per_split[per_split['electrode'].isin(have)],
                                    scores_with_coords)
    keys = [k for pair in names.values() for k in pair]
    cov = s.loc[have, coord_cols] if remove_gradient else None
    elecs, subj, groups, splits, mats, _ = _residualised_split_matrices(
        ps, s.loc[have, 'resp'], min_elec=min_elec, covariates=cov, keys=keys)
    P = s.loc[elecs, coord_cols].to_numpy(float)

    def standardise(M):
        M = M.copy()
        for g in groups:
            blk = M[:, g]
            if method == 'spearman':
                blk = rankdata(blk, axis=1)
            M[:, g] = blk - blk.mean(axis=1, keepdims=True)
        ms = np.sqrt((M ** 2).mean(axis=1, keepdims=True))
        return np.divide(M, ms, out=np.zeros_like(M), where=ms > 0)

    edges = np.r_[0.0, np.asarray(bins_mm, float), np.inf]
    labels = ([f'< {edges[1]:g} mm']
              + [f'{a:g}–{b:g} mm' for a, b in zip(edges[1:-2], edges[2:-1])]
              + [f'> {edges[-2]:g} mm'])
    n_bins, n_g = len(labels), len(groups)

    pair_info = []
    for g in groups:
        iu, ju = np.triu_indices(len(g), k=1)
        keep = np.ones(len(iu), bool)
        if exclude_shared_contacts:
            poles = [_channel_poles(e) for e in elecs[g]]
            keep = np.array([not (poles[i] & poles[j]) for i, j in zip(iu, ju)], bool)
        D = np.linalg.norm(P[g][:, None, :] - P[g][None, :, :], axis=-1)
        pair_info.append((iu[keep], ju[keep], D))
    n_excluded = int(sum(len(np.triu_indices(len(g), 1)[0]) - len(pi[0])
                         for g, pi in zip(groups, pair_info)))

    rng = np.random.default_rng(seed)
    perms = [[rng.permutation(len(g)) for _ in range(int(n_perm))] for g in groups]
    draws = rng.integers(0, n_g, size=(int(n_boot), n_g))
    flips = rng.choice((-1.0, 1.0), size=(int(n_perm), n_g))

    def bin_sums(c_pairs, d_pairs):
        b = np.searchsorted(edges, d_pairs, side='right') - 1
        return (np.bincount(b, weights=c_pairs, minlength=n_bins)[:n_bins],
                np.bincount(b, minlength=n_bins)[:n_bins].astype(float))

    def ratio(num, den):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    rows, contrasts, rel_draws, notes = [], [], {}, []
    for name, (ka, kb) in names.items():
        A, B = standardise(mats[ka]), standardise(mats[kb])
        obs = np.zeros((n_g, n_bins))
        cnt = np.zeros((n_g, n_bins))
        base = np.zeros((n_g, n_bins))
        self_sum, self_n = np.zeros(n_g), np.zeros(n_g)
        for gi, (g, (iu, ju, D)) in enumerate(zip(groups, pair_info)):
            C = 0.5 * (A[:, g].T @ B[:, g] + B[:, g].T @ A[:, g]) / len(splits)
            self_sum[gi], self_n[gi] = np.trace(C), len(g)
            c_pairs = C[iu, ju]
            obs[gi], cnt[gi] = bin_sums(c_pairs, D[iu, ju])
            base[gi] = np.mean([bin_sums(c_pairs, D[p[iu], p[ju]])[0] for p in perms[gi]],
                               axis=0)
        dev = obs - base                                   # each participant's excess
        n = cnt.sum(0)
        excess = ratio(dev.sum(0), n)
        null = ratio(flips @ dev, n[None, :])              # participant sign flips
        rel = float(self_sum.sum() / self_n.sum())
        b_excess = ratio(dev[draws].sum(1), cnt[draws].sum(1))
        b_rel = self_sum[draws].sum(1) / self_n[draws].sum(1)
        reliable = (rel > 0) and np.mean(b_rel > 0) >= min_reliable_share
        b_relative = np.where((b_rel > 0)[:, None], b_excess / b_rel[:, None], np.nan)
        rel_draws[name] = b_relative if reliable else None
        if not reliable:
            notes.append(f"{name}: reliability {rel:+.3f} (positive in "
                         f"{np.mean(b_rel > 0):.0%} of bootstrap draws), so its excess is "
                         "not expressed as a share of it")
        rows.append(dict(score=name, bin='same electrode', n_pairs=int(self_n.sum()),
                         n_participants=n_g, similarity=rel,
                         similarity_lo=float(np.percentile(b_rel, 2.5)),
                         similarity_hi=float(np.percentile(b_rel, 97.5))))
        for bi, label in enumerate(labels):
            ok = np.isfinite(null[:, bi])
            row = dict(score=name, bin=label, n_pairs=int(n[bi]),
                       n_participants=int((cnt[:, bi] > 0).sum()),
                       similarity=float(ratio(obs[:, bi].sum(), n[bi])),
                       baseline=float(ratio(base[:, bi].sum(), n[bi])),
                       excess=float(excess[bi]),
                       excess_lo=float(np.nanpercentile(b_excess[:, bi], 2.5)),
                       excess_hi=float(np.nanpercentile(b_excess[:, bi], 97.5)),
                       p_greater=(float((np.sum(null[ok, bi] >= excess[bi]) + 1)
                                        / (ok.sum() + 1)) if n[bi] else np.nan))
            if reliable:
                row.update(relative=float(excess[bi] / rel),
                           relative_lo=float(np.nanpercentile(b_relative[:, bi], 2.5)),
                           relative_hi=float(np.nanpercentile(b_relative[:, bi], 97.5)))
            rows.append(row)
        near, far = 0, n_bins - 1
        if n[near] and n[far]:
            per_g = dev[:, near] / n[near] - dev[:, far] / n[far]
            obs_c = float(per_g.sum())
            null_c = flips @ per_g
            contrasts.append(dict(score=name, nearest=labels[near], farthest=labels[far],
                                  near_minus_far=obs_c,
                                  p_greater=float((np.sum(null_c >= obs_c) + 1)
                                                  / (n_perm + 1))))

    comparison = []
    first = 'LWPC − LWPS'
    for name, d in rel_draws.items():
        if name == first:
            continue
        if rel_draws.get(first) is None or d is None:
            comparison.append(dict(bin=labels[0], score=name, balance_relative=np.nan,
                                   score_relative=np.nan, difference_lo=np.nan,
                                   difference_hi=np.nan,
                                   note='a reliability is not clearly positive'))
            continue
        diff = rel_draws[first][:, 0] - d[:, 0]
        comparison.append(dict(
            bin=labels[0], score=name,
            balance_relative=float(np.nanmedian(rel_draws[first][:, 0])),
            score_relative=float(np.nanmedian(d[:, 0])),
            difference_lo=float(np.nanpercentile(diff, 2.5)),
            difference_hi=float(np.nanpercentile(diff, 97.5)), note=''))

    return dict(table=pd.DataFrame(rows), contrasts=pd.DataFrame(contrasts),
                comparison=pd.DataFrame(comparison), notes=notes, bins=labels,
                n_electrodes=len(elecs), n_subjects=n_g, n_splits=len(splits),
                n_pairs_excluded=n_excluded, method=method,
                remove_gradient=bool(remove_gradient))


def plot_local_similarity(result, out_path):
    """Small multiples, one per score, shared y: each distance bin's excess
    similarity over the participants' position-shuffle baselines, with 95 %
    participant-bootstrap intervals; the score's reliability in the title."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = result['table']
    scores = list(dict.fromkeys(t['score']))
    rc = {'font.size': 6.5, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5,
          'ytick.labelsize': 6, 'axes.spines.top': False, 'axes.spines.right': False,
          'axes.linewidth': 0.6, 'axes.edgecolor': _INK, 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(scores), figsize=(1.45 * len(scores) + 0.4, 2.0),
                                 sharey=True, squeeze=False)
        for ax, name in zip(axes[0], scores):
            g = t[t['score'] == name]
            rel = g.loc[g['bin'] == 'same electrode', 'similarity'].iloc[0]
            g = g[g['bin'] != 'same electrode'].reset_index(drop=True)
            x = np.arange(len(g))
            ax.axhline(0, color=_RULE, lw=0.6, zorder=0)
            ax.errorbar(x, g['excess'],
                        yerr=[g['excess'] - g['excess_lo'], g['excess_hi'] - g['excess']],
                        fmt='o-', color=_INK, ecolor=_MUTED, elinewidth=0.7, capsize=0,
                        ms=3.5, lw=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([b.replace(' mm', '') for b in g['bin']], rotation=45,
                               ha='right')
            ax.set_title(f'{name}\nreliability {_fmt_r(rel)}', fontsize=6.5, color=_INK,
                         loc='left')
            ax.set_xlabel('distance (mm)')
        axes[0, 0].set_ylabel('excess cross-half similarity')
        fig.text(0.99, 0.01, 'excess over each participant\'s position-shuffle baseline; '
                 '95 % participant bootstrap', ha='right', va='bottom', fontsize=5.5,
                 color=_MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# §19 in one call: the anatomy job and n4_section19_followups.py both use it
# ---------------------------------------------------------------------------
def participant_slope_lines(part, loso=None):
    """Summary lines for :func:`coordinate_slope_by_participant` (and its
    leave-one-out table)."""
    ax = part['axis']
    w, u = part['weighted'], part['unweighted']
    lines = [f"  {ax} SLOPE, PARTICIPANTS AS THE UNIT ({part['n_subjects']} participants, "
             f"{part['n_electrodes']} electrodes)",
             f"    pooled slope {part['pooled_slope']:+.5f}/mm = the coordinate test's; it is "
             "the weight-averaged participant slope",
             f"    weighted:   {w['slope_per_mm']:+.5f}/mm  95 % CI [{w['ci'][0]:+.5f}, "
             f"{w['ci'][1]:+.5f}]  sign-flip p = {w['p_signflip']:.4g}  "
             f"({w['n_participants']} participants)",
             f"    three participants carry {part['top3_weight_share']:.0%} of the weight"]
    if 'p_t' in u:
        lines.append(f"    unweighted: {u['mean_slope_per_mm']:+.5f}/mm ± {u['sem']:.5f}  "
                     f"t-test p = {u['p_t']:.4g}  {u['n_negative']}/{u['n_participants']} "
                     f"negative (sign test p = {u['p_sign']:.3g})  [>= {u['min_elec']} "
                     f"electrodes, >= {u['min_spread_mm']:g} mm spread]")
    for name, r in (part.get('mixed') or {}).items():
        if 'error' in r:
            lines.append(f"    mixed, {name.replace('_', ' ')}: failed ({r['error']})")
            continue
        extra = (f"  random-slope SD {r['random_slope_sd_per_mm']:.5f}/mm"
                 if 'random_slope_sd_per_mm' in r else '')
        conv = '' if r['converged'] else '  NOT CONVERGED'
        lines.append(f"    mixed, {name.replace('_', ' ')}: {r['slope_per_mm']:+.5f}/mm "
                     f"± {r['se_per_mm']:.5f}  Wald p = {r['p']:.4g}{extra}{conv}")
    if loso is not None and len(loso) > 1:
        drop = loso[loso['dropped'] != '(none)']
        lines.append(f"    leave one participant out: slope {drop['slope_per_mm'].min():+.5f} "
                     f"to {drop['slope_per_mm'].max():+.5f}/mm, p {drop['p'].min():.3g} to "
                     f"{drop['p'].max():.3g}")
    return lines


def participant_corr_lines(pc):
    """Summary lines for ``participant_split_corr``."""
    lo, hi = pc['ci_weighted']
    return [f"  LWPC–LWPS SEPARATE-HALF r, PARTICIPANTS AS THE UNIT ({pc['n_participants']} "
            f"participants with >= {pc['min_elec']} electrodes, {pc['n_electrodes']} electrodes)",
            f"    weighted (n - 3): r = {pc['corr_weighted']:+.3f}  95 % CI [{lo:+.3f}, {hi:+.3f}]"
            f"  sign-flip p = {pc['p_signflip']:.4g}",
            f"    unweighted: r = {pc['corr_unweighted']:+.3f}  t-test p = {pc['p_t']:.4g}  "
            f"{pc['n_positive']}/{pc['n_participants']} positive"]


def local_similarity_lines(loc):
    """Summary lines for :func:`local_similarity`."""
    t = loc['table']
    lines = [f"  LOCAL SIMILARITY ({loc['n_electrodes']} electrodes, {loc['n_subjects']} "
             f"participants, {loc['n_splits']} shared splits; "
             f"{'linear gradient removed' if loc['remove_gradient'] else 'raw'}; "
             f"{loc['n_pairs_excluded']} contact-sharing pairs dropped)",
             "    excess cross-half similarity over each participant's position-shuffle "
             "baseline [95 % participant bootstrap], participant sign-flip p"]
    for name, g in t.groupby('score', sort=False):
        s_row = g[g['bin'] == 'same electrode'].iloc[0]
        cells = [f"{r.bin.replace(' mm', '')}: {r.excess:+.3f} [{r.excess_lo:+.3f}, "
                 f"{r.excess_hi:+.3f}] p {r.p_greater:.2g}"
                 for r in g[g['bin'] != 'same electrode'].itertuples()]
        lines.append(f"    {name:<12} reliability {s_row.similarity:+.3f} "
                     f"[{s_row.similarity_lo:+.3f}, {s_row.similarity_hi:+.3f}]")
        lines.append("        " + '  '.join(cells))
    for r in loc['contrasts'].itertuples():
        lines.append(f"    {r.score:<12} nearest − farthest {r.near_minus_far:+.3f}  "
                     f"p = {r.p_greater:.3g}")
    for r in loc['comparison'].itertuples():
        if r.note:
            lines.append(f"    nearest bin, share of reliability, LWPC − LWPS vs {r.score}: "
                         f"not defined ({r.note})")
        else:
            lines.append(f"    nearest bin, share of reliability: LWPC − LWPS "
                         f"{r.balance_relative:+.2f} vs {r.score} {r.score_relative:+.2f}  "
                         f"difference 95 % CI [{r.difference_lo:+.2f}, {r.difference_hi:+.2f}]")
    lines += [f"    NOTE: {n}" for n in loc.get('notes', [])]
    lines.append("    read: intermixed = no excess for LWPC − LWPS at short range while the "
                 "single scores show one; if the single scores show none either, the test "
                 "has no power")
    return lines


def overlap_control_lines(table, loso=None):
    """Summary lines for :func:`overlap_controls`."""
    lines = ["  LWPC–LWPS SEPARATE-HALF r WITH EACH CANDIDATE CONFOUND REMOVED"]
    for r in table.itertuples():
        if not np.isfinite(r.corr):
            lines.append(f"    {r.control:<44} not run ({r.note})")
            continue
        note = f"   [{r.note}]" if r.note else ''
        lines.append(f"    {r.control:<44} r = {r.corr:+.3f}  p = {r.p:.3g}  "
                     f"({r.n_electrodes} electrodes, {r.n_subjects} participants){note}")
    if loso is not None and len(loso):
        lines.append(f"    leave one participant out: r {loso['corr'].min():+.3f} to "
                     f"{loso['corr'].max():+.3f}, p {loso['p'].min():.3g} to {loso['p'].max():.3g}")
    return lines


def _shared_split_reliabilities(per_split, per_split_shared, scores, min_elec=3, seed=1):
    """Within-participant split-half reliabilities from both split schemes.
    Per-electrode splits let one electrode's half A share trials with another's
    half B, which biases the within-participant centring; the shared split does
    not."""
    from .stability_flexibility_segregation import (split_resolved_corr, main_effect_view,
                                                    MAIN_EFFECT_COLS)
    resp = scores.drop_duplicates('electrode').set_index('electrode')['resp']
    rows = []
    for scheme, table in (('per electrode', per_split), ('shared by participant', per_split_shared)):
        t = table[table['electrode'].isin(resp.index)]
        r = split_resolved_corr(t, resp, min_elec=min_elec, n_perm=1, seed=seed)
        row = dict(split=scheme, LWPC=r['reliability_x'], LWPS=r['reliability_y'],
                   overlap_r=r['corr'])
        if set(MAIN_EFFECT_COLS) <= set(t.columns):
            m = split_resolved_corr(main_effect_view(t), resp, min_elec=min_elec, n_perm=1,
                                    seed=seed)
            row.update(congruency=m['reliability_x'], switch=m['reliability_y'])
        rows.append(row)
    return pd.DataFrame(rows)


def section19(scores, per_split, out_dir, coord_res=None, seg_dir=None, axis='mni_z',
              n_perm=10000, n_boot=2000, seed=0, sections=(1, 2, 3, 4, 5),
              per_split_shared=None, rt_coupling=None):
    """§19 of docs/n4_continuous_anatomy.md from one anatomy run's tables.

    1. the ``axis`` slope with participants as the unit
       (:func:`coordinate_slope_by_participant`, :func:`coordinate_slope_loso`);
    2. the LWPC-LWPS separate-half r with participants as the unit
       (``participant_split_corr``);
    3. :func:`local_similarity` and its figure. Needs ``per_split_shared``: a
       per-split table scored with one trial split per participant
       (``compute_sensitivities_per_split(shared_split=True)``); ``per_split``
       is used only if it is itself shared. Also compares the
       within-participant reliabilities of the two split schemes;
    4. the combined anatomy figure (:func:`figure5_height`);
    5. :func:`overlap_controls`, with ``rt_coupling`` (electrode -> ``rt_r``)
       for its RT row.

    ``scores``: :func:`attach_scores`'s table (``scores_with_anatomy.csv``);
    ``per_split``: the run's per-split table (2, 3 and 5 need it). Each section
    fails on its own. Writes its tables and figures to ``out_dir``; returns
    ``(summary lines, JSON-able dict)``.
    """
    from .stability_flexibility_segregation import participant_split_corr
    os.makedirs(out_dir, exist_ok=True)
    has_coords = axis in scores.columns and scores[axis].notna().any()
    lines = ["-" * 70, "§19 — PARTICIPANTS AS THE UNIT, LOCAL SIMILARITY, COMBINED FIGURE 5, "
                       "OVERLAP CONTROLS"]
    out, part = {}, None

    def fail(name, exc):
        lines.append(f"  {name}: failed ({type(exc).__name__}: {exc})")

    if 1 in sections and has_coords:
        try:
            part = coordinate_slope_by_participant(scores, axis=axis, n_perm=n_perm,
                                                   n_boot=n_boot, seed=seed)
            loso = coordinate_slope_loso(scores, axis=axis, n_perm=max(1000, n_perm // 10),
                                         seed=seed)
            part['per_participant'].to_csv(os.path.join(out_dir, f'{axis}_slope_by_participant.csv'),
                                           index=False)
            loso.to_csv(os.path.join(out_dir, f'{axis}_slope_loso.csv'), index=False)
            lines += participant_slope_lines(part, loso)
            out['slope_by_participant'] = {k: v for k, v in part.items()
                                           if k != 'per_participant'}
        except Exception as exc:
            fail(f'{axis} slope by participant', exc)
    if per_split is not None:
        ps = per_split[per_split['electrode'].isin(scores['electrode'])]
        if 2 in sections:
            try:
                resp = scores.drop_duplicates('electrode').set_index('electrode')['resp']
                pc = participant_split_corr(ps, resp, n_perm=n_perm, n_boot=n_boot, seed=seed)
                pc['per_participant'].to_csv(os.path.join(out_dir, 'participant_corr.csv'),
                                             index=False)
                lines += participant_corr_lines(pc)
                out['participant_corr'] = {k: v for k, v in pc.items() if k != 'per_participant'}
            except Exception as exc:
                fail('participant-level LWPC–LWPS r', exc)
        shared = per_split_shared if per_split_shared is not None else ps
        if 3 in sections and has_coords and not is_shared_split(shared):
            lines.append("  local similarity: skipped. It needs one trial split per participant "
                         "shared by all its electrodes, and this run split each electrode on "
                         "its own (§19.3 of the N4 doc). Rerun n4_section19_followups.py with "
                         "--long-df <segregation run>/long_df.csv, or the segregation job with "
                         "SHARED_SPLIT=1.")
        elif 3 in sections and has_coords:
            try:
                loc = local_similarity(shared[shared['electrode'].isin(scores['electrode'])],
                                       scores, n_perm=min(n_perm, 5000), n_boot=n_boot,
                                       seed=seed)
                loc['table'].to_csv(os.path.join(out_dir, 'local_similarity.csv'), index=False)
                loc['contrasts'].to_csv(os.path.join(out_dir, 'local_similarity_contrasts.csv'),
                                        index=False)
                loc['comparison'].to_csv(os.path.join(out_dir, 'local_similarity_comparison.csv'),
                                         index=False)
                plot_local_similarity(loc, os.path.join(out_dir, 'local_similarity.png'))
                lines += local_similarity_lines(loc)
                out['local_similarity'] = dict(
                    table=loc['table'].to_dict(orient='records'),
                    contrasts=loc['contrasts'].to_dict(orient='records'),
                    comparison=loc['comparison'].to_dict(orient='records'),
                    notes=loc['notes'])
            except Exception as exc:
                fail('local similarity', exc)
            if per_split_shared is not None and not is_shared_split(ps):
                try:
                    rel = _shared_split_reliabilities(ps, per_split_shared, scores)
                    rel.to_csv(os.path.join(out_dir, 'reliability_by_split_scheme.csv'),
                               index=False)
                    lines += ["  WITHIN-PARTICIPANT SPLIT-HALF RELIABILITY BY SPLIT SCHEME "
                              "(the per-electrode split biases these)",
                              *("    " + l for l in rel.to_string(
                                  index=False, float_format=lambda v: f'{v:+.3f}').split('\n'))]
                    out['reliability_by_split_scheme'] = rel.to_dict(orient='records')
                except Exception as exc:
                    fail('reliabilities by split scheme', exc)
        if 5 in sections:
            try:
                table, oloso = overlap_controls(scores, ps, rt_coupling=rt_coupling,
                                                n_perm=n_perm, seed=seed)
                table.to_csv(os.path.join(out_dir, 'overlap_controls.csv'), index=False)
                if oloso is not None:
                    oloso.to_csv(os.path.join(out_dir, 'overlap_loso.csv'), index=False)
                lines += overlap_control_lines(table, oloso)
                out['overlap_controls'] = table.to_dict(orient='records')
            except Exception as exc:
                fail('overlap controls', exc)
    elif any(k in sections for k in (2, 3, 5)):
        lines.append("  no per-split table: sections 2, 3 and 5 were skipped")
    if 4 in sections and has_coords:
        try:
            fig = figure5_height(scores, out_dir, per_split=per_split, seg_dir=seg_dir,
                                 coord_res=coord_res, participant_res=part, axis=axis,
                                 n_boot=n_boot, n_perm=n_perm, seed=seed)
            lines += fig['lines']
            out['figure5_height'] = dict(centroids=fig['centroids'].to_dict(orient='records'),
                                         balance=fig['balance'].to_dict(orient='records'),
                                         edges=[float(e) for e in fig['edges']])
        except Exception as exc:
            fail('combined figure 5', exc)
    if not has_coords:
        lines.append(f"  no {axis} coordinates: sections 1, 3 and 4 were skipped")
    return lines, out


# ---------------------------------------------------------------------------
# §6 brain maps of the continuous scores
# ---------------------------------------------------------------------------
# The five surfaces of the plan. Map 5 carries the argument; 1-4 are what a
# reader needs in order to check that it is not driven by one effect's magnitude
# alone. The |score| maps use a sequential colormap because they have no sign.
SCORE_MAPS = (
    dict(value_col='lwpc_s', label='signed LWPC (pooled-scaled)',
         cmap='coolwarm', symmetric=True),
    dict(value_col='lwps_s', label='signed LWPS (pooled-scaled)',
         cmap='coolwarm', symmetric=True),
    dict(value_col='abs_lwpc', label='|LWPC|', cmap='viridis', symmetric=False),
    dict(value_col='abs_lwps', label='|LWPS|', cmap='viridis', symmetric=False),
    dict(value_col='delta', label='LWPC - LWPS (relative map)',
         cmap='coolwarm', symmetric=True),
)
MAIN_EFFECT_MAPS = (
    dict(value_col='cong_s', label='signed congruency (pooled-scaled)',
         cmap='coolwarm', symmetric=True),
    dict(value_col='switch_s', label='signed switch type (pooled-scaled)',
         cmap='coolwarm', symmetric=True),
    dict(value_col='dm', label='congruency - switch (relative map)',
         cmap='coolwarm', symmetric=True),
)


def plot_score_by_roi(scores_with_roi, out_path=None, value_col='delta',
                      roi_col='roi', coverage=None, rois=None, title=None):
    """Mean +/- SEM of a per-electrode score by ROI, with the electrodes drawn on.

    The flat companion to the brain map: it is what the §5.2 test is looking at,
    and it is the fallback figure when the surface stack is unavailable.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = scores_with_roi.dropna(subset=[value_col, roi_col])
    if rois is not None:
        d = d[d[roi_col].astype(str).isin([str(r) for r in rois])]
    g = d.groupby(d[roi_col].astype(str))[value_col]
    order = g.mean().sort_values(ascending=False).index.tolist()
    means = g.mean().reindex(order)
    sems = (g.std(ddof=1) / np.sqrt(g.count())).reindex(order)
    counts = g.count().reindex(order)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(order)), 4.5))
    x = np.arange(len(order))
    ax.bar(x, means.to_numpy(), yerr=sems.to_numpy(), width=.65,
           color="#9ecae1", edgecolor="#3182bd", capsize=3, zorder=2)
    rng = np.random.default_rng(0)
    for i, r in enumerate(order):
        v = d.loc[d[roi_col].astype(str) == r, value_col].to_numpy(float)
        ax.scatter(i + rng.uniform(-.18, .18, v.size), v, s=8, alpha=.45,
                   color="#444", zorder=3)
    ax.axhline(0, color='k', lw=.8, zorder=1)
    labels = [f"{r}\n(n={int(counts[r])})" for r in order]
    if coverage is not None:
        cov = coverage.sum(axis=0)
        labels = [f"{l}\n{int(cov.get(r, 0))} subj" for l, r in zip(labels, order)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set(ylabel=value_col,
           title=title or f"A3 · {value_col} by "
                          f"{'Destrieux label' if roi_col == 'anat' else 'ROI'} "
                          f"(>0 = LWPC-dominant)")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig


def _score_colorbar(edges, cmap, label, out_path):
    """Standalone colourbar for a brain map (`plot_on_average` draws none)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    fig, ax = plt.subplots(figsize=(5, 0.9))
    cm = plt.get_cmap(cmap, len(edges) - 1)
    fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(edges, cm.N), cmap=cm),
                 cax=ax, orientation='horizontal', label=label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_scores_on_brain(scores_with_roi, out_path, value_col='delta',
                         cmap='coolwarm', n_bins=9, symmetric=True, vlim=None,
                         clip_pct=98, subjects=None, hemi='both', size=0.45,
                         transparency=0.4, rm_wm=False, coverage=None,
                         roi_col='roi', label=None, **vis_kwargs):
    """Continuous sibling of :func:`plot_selectivity_groups_on_brain`.

    ``plot_on_average`` takes ONE colour per call, so a continuous scalar is
    rendered by binning it into ``n_bins`` colour bins and drawing each bin as
    its own set onto the same brain -- the same global-index path the group
    figure and the coverage figures use, so all of them stay comparable. A
    standalone colourbar is written next to the figure because the renderer
    draws none.

    ``symmetric=True`` centres the scale on zero (the right choice for the
    signed and relative maps, where the sign is the message); the |score| maps
    pass ``symmetric=False``. The scale is clipped at the ``clip_pct``
    percentile of |value| so a couple of extreme electrodes cannot flatten
    everything else, and the clipping is stated on the colourbar.

    Degrades to :func:`plot_score_by_roi` when the surface stack or the recon
    templates are missing, exactly as the group figure degrades to its histogram.
    """
    d = scores_with_roi.dropna(subset=[value_col]).copy()
    if d.empty:
        raise ValueError(f"no electrode has a finite {value_col}")
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
        out_path = base + '.png'
    if subjects is None:
        subjects = sorted(d['subject'].astype(str).unique())

    v = d[value_col].to_numpy(float)
    if vlim is not None:
        lo, hi = vlim
    elif symmetric:
        hi = float(np.nanpercentile(np.abs(v), clip_pct)) or float(np.abs(v).max())
        lo = -hi
    else:
        lo = float(np.nanmin(v))
        hi = float(np.nanpercentile(v, clip_pct)) or float(np.nanmax(v))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v)) + 1e-9

    edges = np.linspace(lo, hi, int(n_bins) + 1)
    bins = np.clip(np.digitize(v, edges) - 1, 0, int(n_bins) - 1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap, int(n_bins))
    sets = [(f"bin{b:02d}", electrodes_by_subject(d[bins == b]), cm(b))
            for b in range(int(n_bins)) if np.any(bins == b)]
    cbar = _score_colorbar(edges, cmap,
                           f"{label or value_col}  (clipped at {clip_pct}th pct)",
                           f"{base}_colorbar.png")

    try:
        rendered = _render_electrode_sets(sets, out_path, subjects=subjects,
                                          hemi=hemi, size=size,
                                          transparency=transparency,
                                          rm_wm=rm_wm, **vis_kwargs)
        return dict(combined=rendered['combined'], colorbar=cbar, vlim=(lo, hi),
                    n_electrodes=int(len(d)), fallback=False)
    except Exception as exc:  # pragma: no cover - depends on cluster-only stack
        print(f"[A3] brain-surface render unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to the by-ROI figure.")
        fallback = f"{base}_by_roi.png"
        if roi_col in d.columns and d[roi_col].notna().any():
            plot_score_by_roi(d, out_path=fallback, value_col=value_col,
                              roi_col=roi_col, coverage=coverage)
            plt.close('all')
        else:
            fallback = None
        return dict(combined=fallback, colorbar=cbar, vlim=(lo, hi),
                    n_electrodes=int(len(d)), fallback=True,
                    error=f"{type(exc).__name__}: {exc}")


def plot_score_maps(scores_with_roi, out_dir, prefix='score_map', maps=SCORE_MAPS,
                    **kwargs):
    """The five §6 surfaces from one score table. Returns {value_col -> result}."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for spec in maps:
        out[spec['value_col']] = plot_scores_on_brain(
            scores_with_roi,
            os.path.join(out_dir, f"{prefix}_{spec['value_col']}.png"),
            value_col=spec['value_col'], cmap=spec['cmap'],
            symmetric=spec['symmetric'], label=spec['label'], **kwargs)
    return out


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


def _synthetic_scores(n_subj=12, seed=0, gradient=0.8, gain_sd=0.6,
                      rois=("dlpfc", "lpfc", "acc", "parietal", "occ", "v1"),
                      anterior=("dlpfc", "lpfc", "acc"), main_effects=None):
    """Continuous scores with a planted anatomy x effect-type interaction.

    The continuous counterpart of :func:`_synthetic_anatomy`: per-electrode LWPC
    and LWPS scores, an ROI map, fake Destrieux sublabels and fake MNI
    coordinates.

    What is planted, at strength ``gradient``: the anterior ROIs carry the LWPC
    effect and the posterior ones carry LWPS, both POSITIVE — which on the
    LOW-minus-HIGH sign convention (see ``stability_flexibility_segregation``) is
    the direction behavioural adaptation predicts, i.e. the condition effect
    shrinks in the high-proportion block. So ``delta = lwpc - lwps`` is positive
    anteriorly and negative posteriorly, |LWPC| is larger anteriorly and |LWPS|
    posteriorly. Planting MAGNITUDE rather than sign is what makes the §7
    centroids testable at all: they weight by |score| and are blind to a purely
    signed dissociation. ``gradient=0`` is the null — the tests must not
    manufacture significance on it.

    A per-subject GAIN multiplies both scores and the responsiveness proxy, so
    the pooled-scaling and covariate paths are exercised rather than assumed.

    ``main_effects`` adds congruency/switch scores (``mx``/``my``) for the two
    worlds of the main-effect check: ``'inherited'`` gives the main effects
    the anterior/posterior layout and makes each adaptation score a share of
    its own main effect; ``'independent'`` gives the main effects no layout and
    leaves ``x``/``y`` as they are.

    Returns ``(scores, electrodes_to_rois, electrodes_to_anat,
    electrodes_to_coords)`` — ready for :func:`attach_scores`.
    """
    rng = np.random.default_rng(seed)
    rois = list(rois)
    # rough MNI centres (mm): anterior ROIs sit at large +y, posterior at -y
    centers = {r: np.array([40.0, (35.0 if r in anterior else -55.0),
                            20.0 + 6 * i]) for i, r in enumerate(rois)}

    rows, e2r, e2a, e2c = [], {}, {}, {}
    for s in range(n_subj):
        subject = f"S{s:02d}"
        gain = float(np.exp(rng.normal(0, gain_sd)))
        covered = rng.choice(rois, size=int(rng.integers(3, len(rois) + 1)),
                             replace=False).tolist()
        for e in range(int(rng.integers(15, 40))):
            roi = str(rng.choice(covered))
            electrode = f"{subject}-e{e}"
            front = 1.0 if roi in anterior else 0.0
            lwpc = gain * (gradient * front + rng.normal(0, 0.6))
            lwps = gain * (gradient * (1.0 - front) + rng.normal(0, 0.6))
            side = 1.0 if rng.random() < 0.5 else -1.0
            e2r[electrode] = roi
            e2a[electrode] = f"{roi}_lab{int(rng.integers(0, 4))}"
            e2c[electrode] = (centers[roi] + rng.normal(0, 8, 3)) * [side, 1, 1]
            rows.append(dict(subject=subject, electrode=electrode,
                             x=lwpc, y=lwps,
                             resp=gain * float(rng.uniform(0.5, 1.5))))
    scores = pd.DataFrame(rows)
    if main_effects not in (None, 'inherited', 'independent'):
        raise ValueError("main_effects must be None, 'inherited' or 'independent'")
    if main_effects:                         # own stream: x/y above are unchanged
        rng = np.random.default_rng(seed + 1)
        n, inherited = len(scores), main_effects == 'inherited'
        g = scores.groupby('subject')['resp'].transform('mean').to_numpy()  # ~ gain
        front = scores['electrode'].map(e2r).isin(anterior).to_numpy(float)
        scores['mx'] = g * (0.5 + 1.5 * inherited * gradient * front
                            + rng.normal(0, 0.3, n))
        scores['my'] = g * (0.5 + 1.5 * inherited * gradient * (1 - front)
                            + rng.normal(0, 0.3, n))
        if inherited:
            scores['x'] = 0.5 * scores['mx'] + g * rng.normal(0, 0.3, n)
            scores['y'] = 0.5 * scores['my'] + g * rng.normal(0, 0.3, n)
    return scores, e2r, e2a, e2c


def _synthetic_per_split(scores, n_splits=20, noise=1.0, seed=0):
    """A fake ``compute_sensitivities_per_split`` table for a score table.

    Each split re-measures both effects on two disjoint halves, i.e. the true
    per-electrode score plus independent measurement noise. That is all
    :func:`map_reliability` needs, so the §5.4 ceiling path can be exercised on
    the synthetic route (where there are no trials to split) and unit-tested with
    a known answer: raising ``noise`` must lower the reliabilities.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(int(n_splits)):
        for r in scores.itertuples():
            rows.append(dict(
                subject=r.subject, electrode=r.electrode, split=k,
                xA=r.x + rng.normal(0, noise), xB=r.x + rng.normal(0, noise),
                yA=r.y + rng.normal(0, noise), yB=r.y + rng.normal(0, noise)))
    out = pd.DataFrame(rows)
    if 'mx' in scores:              # main effects: better measured, own stream
        rng = np.random.default_rng(seed + 1)
        for c in ('mxA', 'mxB', 'myA', 'myB'):
            out[c] = (np.tile(scores[c[:2]].to_numpy(float), int(n_splits))
                      + rng.normal(0, noise / 3, len(out)))
    return out


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

    # the CONTINUOUS arm (§5), same shape of check: a planted anatomy x
    # effect-type interaction must be found, and the null must not manufacture one
    for grad in (0.0, 0.8):
        scores, e2r, e2a, e2c = _synthetic_scores(gradient=grad, seed=1)
        tab = attach_scores(scores, e2r, electrodes_to_anat=e2a,
                            electrodes_to_coords=e2c)
        cover = build_coverage_matrix(tab)
        roi_res = relative_score_roi_test(tab, cover, min_subjects=3, n_perm=2000)
        coord_res = relative_score_coordinate_test(tab, n_perm=2000)['all']
        print(f"[gradient={grad}] delta ~ roi: F={roi_res['observed_stat']:.2f} "
              f"p={roi_res['p']:.4f} (n_elec={roi_res['n_electrodes']})")
        print(roi_res['per_roi'].to_string(index=False))
        y = coord_res['slopes'].set_index('axis').loc['mni_y']
        print(f"  delta ~ coords: F={coord_res['observed_stat']:.2f} "
              f"p={coord_res['p']:.4f} | anterior slope "
              f"{y['slope_per_mm']:+.5f}/mm (p={y['p']:.4f})")
        centers = score_centers_per_subject(tab, n_perm=2000)
        print(f"  {centers['center']}s: LWPC sits "
              f"{centers['mean_displacement']['dy']:+.1f} mm anterior to LWPS "
              f"(p={centers['p']['dy']:.4f}, {centers['n_groups']} subject x hemi)")
        print("-" * 60)
