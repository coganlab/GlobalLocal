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
                              transparency=transparency, fig=fig, show=False,
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
    print(f"[A3] brain figure -> {out_path}")

    per_set = {}
    if per_set_figures:
        for name, idx, rgb in picks:
            if not idx:
                continue
            sfig = plot_on_average(subjects_no_zeros, picks=idx, rm_wm=rm_wm,
                                   hemi=hemi, color=rgb, size=size,
                                   transparency=transparency, show=False,
                                   **vis_kwargs)
            path = f"{base}_{name}.png"
            saved = save_brain_image(sfig, path)
            sfig.close()
            if not saved:                # the combined figure already landed --
                continue                 # a missing per-set panel isn't fatal
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
    anatomy columns.

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
    out = out.rename(columns={xc: 'lwpc_score', yc: 'lwps_score'})
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
    for src, dst in (('lwpc_score', 'lwpc_s'), ('lwps_score', 'lwps_s')):
        sd = out[src].std(ddof=1)
        out[dst] = out[src] / sd if np.isfinite(sd) and sd > 0 else np.nan
    out['abs_lwpc'] = out['lwpc_s'].abs()
    out['abs_lwps'] = out['lwps_s'].abs()
    out['delta'] = out['lwpc_s'] - out['lwps_s']
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

    rows = []
    for _, g in d.groupby('split'):
        if len(g) < min_units:
            continue
        xa, xb = g['xA'].to_numpy(), g['xB'].to_numpy()
        ya, yb = g['yA'].to_numpy(), g['yB'].to_numpy()
        rows.append((0.5 * (fn(xa, yb)[0] + fn(xb, ya)[0]),
                     fn(xa, xb)[0], fn(ya, yb)[0]))
    if not rows:
        raise ValueError(f"every split has fewer than min_units={min_units} {unit}s")
    between, rel_x, rel_y = np.nanmean(np.array(rows, float), axis=0)
    denom = np.sqrt(rel_x * rel_y) if (rel_x > 0 and rel_y > 0) else np.nan

    return dict(between=float(between), reliability_lwpc=float(rel_x),
                reliability_lwps=float(rel_y),
                between_noise_corrected=(float(between / denom)
                                         if np.isfinite(denom) else np.nan),
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
                      anterior=("dlpfc", "lpfc", "acc")):
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
    return pd.DataFrame(rows), e2r, e2a, e2c


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
    return pd.DataFrame(rows)


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
