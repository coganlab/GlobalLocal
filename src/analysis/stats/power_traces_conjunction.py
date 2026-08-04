"""Conjunction tests driven by `power_traces` cluster-corrected electrode definitions.

This is the bridge that lets the **within-electrode windowed ANOVA with cluster
correction** (`src/analysis/power/windowed_anova.py`) serve as the electrode
definition for the stability/flexibility conjunction, in place of
`stability_flexibility_segregation.per_electrode_anova_labels`.

Why this module exists
----------------------
`per_electrode_anova_labels` fits ONE ANOVA on the **time-window-averaged** HG,
which dilutes an interaction that is strong but transient inside the window. The
`power_traces` route instead fits the ANOVA **at every window** and
cluster-corrects across time, so an electrode counts as selective if it has any
surviving cluster. On the narrow question "does this electrode carry the
interaction at all", the temporal route is the more sensitive detector.

What the conjunction stack needs, though, is a **per-electrode record with all
four interactions co-registered on the same row** -- and
`power_traces.windowed_anova.load_significant_electrodes` returns a flat,
per-effect `[(subject, electrode)]` list. This module closes that gap: it reads a
run's `summary.csv` (and, optionally, the per-electrode `.npz` files), pivots the
four interactions onto one row per electrode, and emits a labels DataFrame with
exactly the columns `cmh_conjunction` / `conjunction_permutation_null` /
`conjunction_threshold_sweep` already consume.

The four interaction groups (same naming as `per_electrode_anova_labels`):

    CPC = congruency  x incongruentProportion   -- LWPC / stability   (alias `S`)
    SPS = switchType  x switchProportion        -- LWPS / flexibility (alias `F`)
    CPS = congruency  x switchProportion        -- cross control
    SPC = switchType  x incongruentProportion   -- cross control

Two correction modes, because they answer different questions
-------------------------------------------------------------
`correction='cluster'` reproduces what `load_significant_electrodes` actually
does today: keep any electrode with a surviving cluster at `alpha`, with no
correction across electrodes. It is the like-for-like port of the existing lab
convention.

`correction='fdr_bh'` (default) additionally applies Benjamini-Hochberg **across
electrodes, within (roi, effect)** -- the correction family the conjunction
needs, since the 2x2 table is built by counting electrodes.

A caveat that matters for `fdr_bh`: `summary.csv` only records cluster p-values
for clusters that ALREADY passed the run's cluster-extent threshold; every other
electrode is written with `cluster_p_value = 1.0`. Running BH on that truncated
vector is valid but conservative. `refine_cluster_pvalues_from_npz` recovers the
honest uncorrected cluster p for EVERY electrode by recomputing the cluster test
from the saved `observed_F` / `null_F` arrays, and `electrode_labels` will use it
automatically when `use_npz=True` and the files are present.

Typical use
-----------
    from src.analysis.stats import power_traces_conjunction as ptc

    labels = ptc.electrode_labels(
        runs={'CPC': lwpc_run_dir, 'SPS': lwps_run_dir,
              'CPS': cps_run_dir,  'SPC': spc_run_dir},
        roi='lpfc', alpha=0.05, correction='fdr_bh')

    res = ptc.run_power_traces_conjunction(labels)
    print(res['both_vs_distinct'])

A single 4-factor run (`stimulus_experiment_conditions`) that already contains
all four interaction terms can be passed as one path instead of a dict.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.analysis.power.windowed_anova import (
    _find_contiguous_runs,
    _split_clusters_at_sign_flips,
)
from src.analysis.stats.stability_flexibility_segregation import (
    cmh_conjunction,
    conjunction_permutation_null,
    conjunction_threshold_sweep,
)


# The four two-way interactions, keyed by the group name the conjunction stack
# uses. Values are the (condition_factor, modulator_factor) column names as they
# appear in `config/condition_registry.py`'s `anova_factors`.
DEFAULT_INTERACTIONS = {
    'CPC': ('congruency', 'incongruentProportion'),   # LWPC / stability
    'SPS': ('switchType', 'switchProportion'),        # LWPS / flexibility
    'CPS': ('congruency', 'switchProportion'),        # cross control
    'SPC': ('switchType', 'incongruentProportion'),   # cross control
}

# The two constructs the 2x2 conjunction is actually built on. The cross groups
# are carried for reporting/specificity and for the decoding double-dip
# bookkeeping, but they are not part of the S x F table.
CONSTRUCT_GROUPS = ('CPC', 'SPS')


def anova_effect_name(cond_factor, mod_factor):
    """statsmodels effect name for a two-way interaction, as written by
    `_fit_anova_per_window_per_unit` (which pins names off the formula).

    >>> anova_effect_name('congruency', 'incongruentProportion')
    'C(congruency):C(incongruentProportion)'

    `windowed_anova` builds effect names from `combinations(anova_factors, k)`,
    so the factor ORDER follows the `anova_factors` list, not alphabetical order.
    `_effect_name_variants` handles the reversed spelling so a run whose factors
    were declared in the other order still resolves.
    """
    return f'C({cond_factor}):C({mod_factor})'


def _effect_name_variants(cond_factor, mod_factor):
    """Both orderings of a two-way interaction's effect name."""
    return (anova_effect_name(cond_factor, mod_factor),
            anova_effect_name(mod_factor, cond_factor))


# ----------------------------------------------------------------------------
# reading a within-electrode ANOVA run
# ----------------------------------------------------------------------------
def load_run_summary(run_dir):
    """Read a within-electrode windowed-ANOVA run's `summary.csv`.

    One row per (subject, electrode, roi, effect, cluster). Electrodes with no
    surviving cluster appear with `cluster_idx == -1` and `cluster_p_value == 1.0`
    -- which is what makes `summary.csv` (rather than
    `significant_effects_structure.json`) the right source here: it carries the
    full set of electrodes that were TESTED, i.e. the honest denominator for the
    2x2 table. The JSON only lists the winners.
    """
    path = Path(run_dir) / 'summary.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"no summary.csv in {run_dir!r}; pass the run directory written by "
            "run_within_electrode_windowed_anova_cluster_correction "
            "(save_dir / run_label)")
    df = pd.read_csv(path)
    required = {'subject', 'electrode', 'roi', 'effect', 'cluster_p_value'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    return df


def _best_cluster_per_electrode(summary, effect_names, roi=None):
    """Collapse a run's clusters to ONE row per electrode for a single effect.

    Prefers the `best_cluster_p` / `best_cluster_extent` / `best_cluster_sign`
    columns, which `run_within_electrode_windowed_anova_cluster_correction`
    records for EVERY electrode whether or not its cluster survived the extent
    threshold. That is the graded p an across-electrode correction needs, and it
    is computed at run time where the null sign traces are still in memory, so
    the sign-split is applied symmetrically to observed and null clusters.

    Runs written before those columns existed fall back to the surviving-cluster
    p (`cluster_p_value`), which is floored at exactly 1.0 for every electrode
    that missed the threshold; `refine_cluster_pvalues_from_npz` recovers a
    graded p for those, minus the sign-split. Returns columns
    `subject, electrode, roi, p_cluster, sign, extent, peak_F`.

    `effect_names` is the tuple of acceptable spellings (factor order can differ
    between runs); the first one present in the run is used.
    """
    df = summary
    if roi is not None:
        df = df[df['roi'] == roi]
    present = [e for e in effect_names if (df['effect'] == e).any()]
    if not present:
        available = sorted(summary['effect'].unique())
        raise KeyError(
            f"none of the effect names {list(effect_names)} are in this run "
            f"(roi={roi!r}); available effects: {available}")
    df = df[df['effect'] == present[0]].copy()
    if df.empty:
        raise ValueError(f"effect {present[0]!r} has no rows for roi={roi!r}")

    if 'extent_windows' not in df.columns:
        df['extent_windows'] = 0
    df = df.sort_values(['cluster_p_value', 'extent_windows'],
                        ascending=[True, False])
    best = df.groupby(['subject', 'electrode', 'roi'], as_index=False).first()

    out = best[['subject', 'electrode', 'roi']].copy()
    if 'best_cluster_p' in best.columns and best['best_cluster_p'].notna().any():
        out['p_cluster'] = best['best_cluster_p'].to_numpy()
        out['sign'] = best['best_cluster_sign'].to_numpy()
        out['extent'] = best['best_cluster_extent'].to_numpy()
    else:
        out['p_cluster'] = best['cluster_p_value'].to_numpy()
        out['sign'] = best['sign'].to_numpy() if 'sign' in best else 0
        out['extent'] = best['extent_windows'].to_numpy()
    out['peak_F'] = best['peak_F'].to_numpy() if 'peak_F' in best else np.nan
    return out


# ----------------------------------------------------------------------------
# honest uncorrected cluster p-values, recomputed from the saved null
# ----------------------------------------------------------------------------
def refine_cluster_pvalues_from_npz(run_dir, effect_names, roi=None,
                                    percentile=95, split_clusters_by_sign=False):
    """Recompute an UNCORRECTED cluster p for every electrode from its `.npz`.

    `summary.csv` floors non-surviving electrodes at `cluster_p_value = 1.0`,
    because `run_within_electrode_windowed_anova_cluster_correction` only records
    clusters that already cleared the run's extent threshold. BH-FDR across a
    vector of mostly-exact-1.0 p-values is valid but needlessly conservative, and
    it cannot rank the near-misses.

    This walks `run_dir/<roi>/<subject>/<electrode>.npz` -- written at
    `windowed_anova.py:352` with `observed_F`, `null_F`, `signed_contrast`,
    `effect_names` -- and replays the same cluster test to get the p of each
    electrode's LARGEST cluster, whether or not it survived. Electrodes with no
    supra-threshold window get p = 1.0 (correctly, not by truncation).

    Returns `subject, electrode, roi, p_cluster, sign, extent`, or an empty frame
    if no `.npz` files are present (older runs saved only the summary).
    """
    run_dir = Path(run_dir)
    roi_dirs = [run_dir / roi] if roi is not None else [
        p for p in run_dir.iterdir() if p.is_dir()]

    rows = []
    for roi_dir in roi_dirs:
        if not roi_dir.is_dir():
            continue
        for sub_dir in sorted(p for p in roi_dir.iterdir() if p.is_dir()):
            for npz_path in sorted(sub_dir.glob('*.npz')):
                with np.load(npz_path, allow_pickle=True) as z:
                    names = [str(n) for n in z['effect_names']]
                    match = [n for n in effect_names if n in names]
                    if not match:
                        continue
                    ei = names.index(match[0])
                    obs = np.nan_to_num(z['observed_F'][ei], nan=0.0)
                    null = np.nan_to_num(z['null_F'][:, ei, :], nan=0.0)
                    sgn = (z['signed_contrast'][ei]
                           if 'signed_contrast' in z else None)

                stats = _cluster_pvalue(obs, null, sgn, percentile,
                                        split_clusters_by_sign)
                rows.append(dict(subject=sub_dir.name, electrode=npz_path.stem,
                                 roi=roi_dir.name, **stats))
    return pd.DataFrame(rows, columns=['subject', 'electrode', 'roi',
                                       'p_cluster', 'sign', 'extent'])


def _cluster_pvalue(obs, null, sign_trace, percentile=95,
                    split_clusters_by_sign=False):
    """Uncorrected cluster-extent p for one electrode's F-trace.

    Mirrors `run_within_electrode_windowed_anova_cluster_correction` step 3d --
    pointwise threshold from the null's `percentile` per window, contiguous
    supra-threshold runs, null max-extent distribution -- but returns the p of the
    largest observed cluster instead of thresholding it. p = 1.0 when nothing
    clears the pointwise threshold.

    `split_clusters_by_sign` defaults to False here, unlike the run itself. The
    `.npz` stores `signed_contrast` for the OBSERVED trace only, never for the
    permutations, so splitting would shorten observed clusters while leaving null
    clusters intact -- a one-sided, conservative bias. Keeping both unsplit is the
    symmetric choice. Pass True only if you accept that asymmetry.
    """
    pointwise_thresh = np.nanpercentile(null, percentile, axis=0)
    raw_obs = _find_contiguous_runs(obs > pointwise_thresh)

    do_split = (split_clusters_by_sign and sign_trace is not None
                and np.any(np.isfinite(sign_trace)))
    obs_sub = (_split_clusters_at_sign_flips(raw_obs, sign_trace) if do_split
               else [{'start': s, 'end': e, 'sign': 0} for s, e in raw_obs])
    if not obs_sub:
        return dict(p_cluster=1.0, sign=0, extent=0)

    null_max_extents = np.empty(null.shape[0])
    for pi in range(null.shape[0]):
        raw_pi = _find_contiguous_runs(null[pi] > pointwise_thresh)
        extents = [e - s + 1 for s, e in raw_pi] or [0]
        null_max_extents[pi] = max(extents)

    best = max(obs_sub, key=lambda c: c['end'] - c['start'] + 1)
    ext = best['end'] - best['start'] + 1
    p = float((null_max_extents >= ext).mean())
    return dict(p_cluster=p, sign=int(best['sign']), extent=int(ext))


# ----------------------------------------------------------------------------
# labels: four interactions co-registered on one row per electrode
# ----------------------------------------------------------------------------
def electrode_labels(runs, roi=None, alpha=0.05, correction='fdr_bh',
                     interactions=None, use_npz=True, require_all=True):
    """Per-electrode selectivity labels from `power_traces` cluster-corrected runs.

    Drop-in replacement for
    `stability_flexibility_segregation.per_electrode_anova_labels`: same output
    contract, so `cmh_conjunction`, `conjunction_permutation_null` and
    `conjunction_threshold_sweep` consume it unchanged.

    Parameters
    ----------
    runs : path or dict
        Either ONE run directory whose ANOVA contained all four interaction terms
        (e.g. a `stimulus_experiment_conditions` run), or a dict mapping group
        name -> run directory when each interaction was run separately
        (`{'CPC': ..., 'SPS': ..., 'CPS': ..., 'SPC': ...}`). Only the groups
        present in the dict are labelled; `CPC` and `SPS` are required.
    roi : str or None
        Restrict to one ROI (e.g. `'lpfc'`). Note this also restricts the subject
        strata: a subject with no electrodes in `roi` drops out of the CMH
        entirely, and one whose electrodes are all in a single S/F class
        contributes no information to it.
    alpha : float
        Selection cutoff, applied to `q_<group>` under `fdr_bh` or to the raw
        cluster p under `cluster`.
    correction : {'fdr_bh', 'cluster', 'none'}
        `fdr_bh`  -- BH across electrodes within (roi, effect). The right family
                     for a test that counts electrodes.
        `cluster` -- raw cluster p < alpha, no across-electrode correction. This
                     is what `load_significant_electrodes` does today; use it for
                     a like-for-like comparison against existing lab numbers.
        `none`    -- flag every electrode that had any surviving cluster.
    use_npz : bool
        Only consulted for LEGACY runs whose `summary.csv` predates the
        `best_cluster_p` column. Current runs record a graded cluster p for every
        electrode at run time, which is strictly better (the null sign traces are
        still in memory there, so the sign-split stays symmetric), and this flag
        is then ignored.
    require_all : bool
        Keep only electrodes present in EVERY requested run. Different runs can
        contain different electrode sets (different `min_trials_per_cell`
        outcomes), and a 2x2 table built over inconsistent denominators is not
        interpretable. The number dropped is reported in `.attrs['n_dropped']`.

    Returns
    -------
    DataFrame with `subject`, `electrode`, `roi`, and per group `G`:
    `p_<g>`, `q_<g>`, `<g>_sign`, `<g>_extent`, and the binary flag `G`.
    Aliases `S` = `CPC` and `F` = `SPS` are added for the conjunction stack.
    `.attrs` carries `n_dropped`, `correction`, `alpha`, `roi`, `source`.
    """
    if correction not in ('fdr_bh', 'cluster', 'none'):
        raise ValueError("correction must be 'fdr_bh', 'cluster' or 'none'; "
                         f"got {correction!r}")
    interactions = dict(interactions or DEFAULT_INTERACTIONS)

    if isinstance(runs, (str, Path)):
        runs = {g: runs for g in interactions}
    runs = {g: Path(p) for g, p in runs.items()}
    for g in CONSTRUCT_GROUPS:
        if g not in runs:
            raise ValueError(f"runs must include {g!r} (the S x F conjunction "
                             f"needs both {CONSTRUCT_GROUPS}); got {sorted(runs)}")

    per_group = {}
    for group, run_dir in runs.items():
        if group not in interactions:
            raise KeyError(f"unknown interaction group {group!r}; "
                           f"known: {sorted(interactions)}")
        names = _effect_name_variants(*interactions[group])

        # Preference order: the graded p the run recorded for every electrode;
        # else a recompute from the saved null; else the surviving-cluster p.
        summary = load_run_summary(run_dir)
        if 'best_cluster_p' in summary.columns:
            tbl = _best_cluster_per_electrode(summary, names, roi=roi)
        else:
            tbl = (refine_cluster_pvalues_from_npz(run_dir, names, roi=roi)
                   if use_npz else pd.DataFrame())
            if tbl.empty:
                tbl = _best_cluster_per_electrode(summary, names, roi=roi)
        per_group[group] = tbl.set_index(['subject', 'electrode', 'roi'])

    # Align on a single electrode universe across the four runs.
    indices = [tbl.index for tbl in per_group.values()]
    index = indices[0]
    union = indices[0]
    for idx in indices[1:]:
        index = index.intersection(idx) if require_all else index.union(idx)
        union = union.union(idx)
    n_union = len(union)

    out = pd.DataFrame(index=index).reset_index()
    for group, tbl in per_group.items():
        aligned = tbl.reindex(index)
        g = group.lower()
        out[f'p_{g}'] = aligned['p_cluster'].to_numpy()
        out[f'{g}_sign'] = aligned['sign'].to_numpy()
        out[f'{g}_extent'] = aligned['extent'].to_numpy()

    for group in per_group:
        g = group.lower()
        p = pd.Series(out[f'p_{g}']).fillna(1.0).to_numpy()
        if correction == 'fdr_bh':
            out[f'q_{g}'] = multipletests(p, method='fdr_bh')[1]
            flag = out[f'q_{g}'] < alpha
        elif correction == 'cluster':
            out[f'q_{g}'] = np.nan          # no across-electrode correction applied
            flag = p < alpha
        else:
            out[f'q_{g}'] = np.nan
            flag = pd.Series(out[f'{g}_extent']).fillna(0).to_numpy() > 0
        out[group] = np.asarray(flag).astype(int)

    out['S'] = out['CPC']
    out['F'] = out['SPS']
    out.attrs.update(n_dropped=int(n_union - len(out)), correction=correction,
                     alpha=alpha, roi=roi, source='power_traces')
    return out


# ----------------------------------------------------------------------------
# window alignment: making the independent continuous route comparable
# ----------------------------------------------------------------------------
def load_run_config(run_dir):
    """Read a windowed-ANOVA run's `run_config.json`.

    Falls back to reconstructing the window geometry from any per-electrode
    `.npz` (which stores `window_centers`) for runs written before the manifest
    existed. In that fallback `analysis_tmin`/`analysis_tmax` are the first and
    last window CENTRES, which understates the true extent by half a window at
    each end -- the returned dict flags this as `exact=False`.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / 'run_config.json'
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg['exact'] = True
        return cfg

    for npz_path in sorted(run_dir.glob('*/*/*.npz')):
        with np.load(npz_path, allow_pickle=True) as z:
            if 'window_centers' not in z:
                continue
            centers = np.asarray(z['window_centers'], float)
        return dict(window_centers=[float(c) for c in centers],
                    n_windows=int(centers.size),
                    analysis_tmin=float(centers.min()),
                    analysis_tmax=float(centers.max()),
                    exact=False)

    raise FileNotFoundError(
        f"{run_dir} has neither run_config.json nor any per-electrode .npz; "
        "cannot recover the window geometry")


def analysis_window_from_run(run_dir):
    """`(tmin, tmax)` in seconds covered by a windowed-ANOVA run.

    This is what an independently-estimated analysis should be told to use so it
    describes the same stretch of the epoch. It is NOT a claim that the two
    analyses become equivalent -- the windowed ANOVA fits per sliding window and
    cluster-corrects across them, while a single-window estimator integrates over
    the whole extent. Aligning the extent only removes the trivial objection that
    the two were looking at different data.
    """
    cfg = load_run_config(run_dir)
    return float(cfg['analysis_tmin']), float(cfg['analysis_tmax'])


def describe_window_alignment(run_dir, current_tmin, current_tmax):
    """Human-readable diff between a segregation window and a power_traces run."""
    cfg = load_run_config(run_dir)
    tmin, tmax = float(cfg['analysis_tmin']), float(cfg['analysis_tmax'])
    lines = [
        f"power_traces run: {run_dir}",
        f"  windows        : {cfg.get('n_windows', '?')} "
        f"(size={cfg.get('window_size', '?')} samples, "
        f"step={cfg.get('step_size', '?')})",
        f"  covered extent : [{tmin:.4f}, {tmax:.4f}] s"
        + ("" if cfg.get('exact') else "   (window CENTRES only -- no "
                                        "run_config.json; true extent is wider)"),
        f"  current window : [{current_tmin:.4f}, {current_tmax:.4f}] s",
    ]
    if abs(tmin - current_tmin) > 1e-9 or abs(tmax - current_tmax) > 1e-9:
        lines.append("  -> MISALIGNED; the continuous estimate covers a "
                     "different stretch of the epoch than the labels.")
    else:
        lines.append("  -> aligned.")
    return '\n'.join(lines)


# ----------------------------------------------------------------------------
# the continuous route, in its confound-control role
# ----------------------------------------------------------------------------
def _window_mean_df(df):
    """Collapse an array-valued `hg` column to its per-trial window mean."""
    if df['hg'].dtype != object:
        return df
    return df.assign(hg=df['hg'].apply(lambda a: float(np.nanmean(a))))


def continuous_confound_control(df, responsiveness=None, n_splits=200,
                                n_perm=10000, min_elec=3,
                                contrast_mode='proportion', contrasts=None,
                                effect_measures=('peak_t', 'cluster', 'cohens_d'),
                                alpha=0.05, seed=1):
    """Run the continuous correlation under EVERY effect measure, as a control.

    Why this shape. The count test (`run_power_traces_conjunction`) is the primary
    result, but it has two confounds that both bias it toward "shared", which is
    usually the hypothesis under test:

      1. Per-electrode detection power. An electrode with high SNR or more trials
         is likelier to clear alpha on BOTH interactions from power alone, which
         inflates the "both" cell. Subject stratification does not fix this --
         the variation is across electrodes within a subject.
      2. Shared trial noise. S and F labels are estimated from the same trials, so
         coupled noise inflates co-occurrence.

    The continuous route carries the controls for exactly those two: it
    residualises each sensitivity on the electrode's overall responsiveness
    (`add_responsiveness` + `prepare_continuous`), and it estimates the two
    sensitivities on DISJOINT trial halves (`compute_sensitivities`). Its job here
    is to check that the count result is not manufactured by either -- not to be a
    second headline.

    Why all three measures rather than one pinned choice. Reducing a per-trial HG
    time course to one scalar is under-determined: mass grows with duration and is
    mildly trial-count sensitive, Cohen's d on the window mean attenuates transient
    effects, peak t is amplitude-only. As a headline that arbitrariness is a real
    weakness. As a CONTROL it is not, provided the verdict does not depend on the
    choice -- so run all three and report the range. Divergence across measures is
    itself the finding, and should be reported rather than resolved by picking the
    convenient one.

    Returns dict keyed by effect measure, each with `rho`, `p`, `n_electrodes`,
    `n_subjects`, plus a top-level `agreement` summary.
    """
    from src.analysis.stats.stability_flexibility_segregation import (
        add_responsiveness, compute_sensitivities, prepare_continuous,
        subject_clustered_corr)

    out = {}
    for measure in effect_measures:
        # 'cohens_d' standardises by a scalar pooled SD and is only defined on
        # window-mean HG; 'cluster' and 'peak_t' need the time course. When the
        # table carries time courses, reduce a copy for the window-mean measure
        # -- the same reduction `_anova_interaction_stats` applies.
        use = _window_mean_df(df) if measure == 'cohens_d' else df
        elec = add_responsiveness(
            compute_sensitivities(use, n_splits, contrast_mode=contrast_mode,
                                  contrasts=contrasts, effect_measure=measure,
                                  alpha=alpha),
            use, responsiveness)
        cont = prepare_continuous(elec, min_elec=min_elec)
        corr = subject_clustered_corr(cont, n_perm=n_perm, seed=seed)
        out[measure] = dict(
            rho=float(corr['corr']), p=float(corr['p']),
            n_electrodes=int(corr['n_electrodes']),
            n_subjects=int(corr['n_subjects']))

    rhos = [v['rho'] for v in out.values()]
    ps = [v['p'] for v in out.values()]
    out['agreement'] = dict(
        rho_min=float(min(rhos)), rho_max=float(max(rhos)),
        all_same_sign=bool(all(r > 0 for r in rhos) or all(r < 0 for r in rhos)),
        all_significant=bool(all(p < alpha for p in ps)),
        all_null=bool(all(p >= alpha for p in ps)),
    )
    return out


def summarize_confound_control(res, alpha=0.05):
    """One-screen readout of `continuous_confound_control`, with the verdict."""
    lines = ["continuous correlation (responsiveness-residualised, "
             "disjoint trial halves, within-subject centred):"]
    for measure, v in res.items():
        if measure == 'agreement':
            continue
        lines.append(f"    {measure:<10} rho = {v['rho']:+.3f}  p = {v['p']:.4g}"
                     f"   ({v['n_electrodes']} electrodes, "
                     f"{v['n_subjects']} subjects)")
    a = res['agreement']
    lines.append(f"  rho range across measures: "
                 f"[{a['rho_min']:+.3f}, {a['rho_max']:+.3f}]")
    if not a['all_same_sign']:
        lines.append("  -> MEASURES DISAGREE IN SIGN. The continuous estimate is "
                     "not stable under the scalarisation choice; report this "
                     "rather than picking one.")
    elif a['all_significant'] and a['rho_min'] > 0:
        lines.append("  -> Positive under every measure: co-selectivity survives "
                     "responsiveness residualisation and disjoint halves, so the "
                     "count result is not explained by shared gain or shared "
                     "trial noise.")
    elif a['all_null']:
        lines.append("  -> Null under every measure. Note a null correlation is "
                     "NOT evidence for distinctness; only the count test's "
                     "OR < 1 can supply that.")
    else:
        lines.append("  -> Mixed significance across measures; treat the control "
                     "as inconclusive.")
    return '\n'.join(lines)


# ----------------------------------------------------------------------------
# the count test
# ----------------------------------------------------------------------------
def both_vs_distinct_test(labels, n_perm=10000, seed=0):
    """Is the SHARED electrode count higher than the DISTINCT count, beyond chance?

    Counts, over the electrodes in `labels`:
        both     = S & F          (shared)
        distinct = S-only + F-only
    and tests `D = both - distinct` against a null that shuffles F **within each
    subject**, so every subject's S- and F-marginals are held fixed and only the
    S/F pairing is randomized. That is the same null `conjunction_permutation_null`
    uses, and the one the CMH assumes.

    A note worth having in front of you before reading the output: with the
    marginals fixed, `S_only = n_S - both` and `F_only = n_F - both`, so

        D = both - (n_S - both) - (n_F - both) = 3*both - n_S - n_F

    is a strictly increasing function of `both`. The permutation test on "shared
    vs distinct" is therefore **the same test** as the permutation test on the
    joint count -- identical p-value, by construction. Both are reported so the
    equivalence is visible rather than hidden; do not present them as two
    independent lines of evidence.

    Returns dict with the observed counts, the null distribution of D, one-sided
    p for `shared > distinct`, two-sided p, and z.
    """
    lab = labels.dropna(subset=['S', 'F']).copy()
    S = lab['S'].to_numpy().astype(int)
    F = lab['F'].to_numpy().astype(int)
    subj = lab['subject'].to_numpy()
    groups = [np.where(subj == s)[0] for s in np.unique(subj)]

    def counts(Sv, Fv):
        both = int(((Sv == 1) & (Fv == 1)).sum())
        s_only = int(((Sv == 1) & (Fv == 0)).sum())
        f_only = int(((Sv == 0) & (Fv == 1)).sum())
        return both, s_only, f_only

    both, s_only, f_only = counts(S, F)
    observed = both - (s_only + f_only)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Fp = F.copy()
        for idx in groups:
            Fp[idx] = F[rng.permutation(idx)]
        b, so, fo = counts(S, Fp)
        null[i] = b - (so + fo)

    mean = null.mean()
    sd = null.std()
    p_greater = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(null - mean) >= abs(observed - mean)) + 1)
                  / (n_perm + 1))
    return dict(n_electrodes=int(len(lab)), n_subjects=int(len(groups)),
                n_both=both, n_stab_only=s_only, n_flex_only=f_only,
                n_neither=int(((S == 0) & (F == 0)).sum()),
                observed=int(observed), null=null,
                null_mean=float(mean), null_sd=float(sd),
                p_shared_greater=p_greater, p_two_sided=p_two,
                z=float((observed - mean) / sd) if sd > 0 else np.nan)


def cross_control_counts(labels, interactions=None):
    """Surviving-electrode counts for all four interactions.

    In univariate HG the two CROSS interactions (`CPS`, `SPC`) are expected to be
    near-null. If they recruit as many electrodes as `CPC`/`SPS`, the selection is
    picking up something common to all four contrasts (gain, block-level drift)
    rather than the constructs, and the conjunction should not be interpreted.
    """
    interactions = dict(interactions or DEFAULT_INTERACTIONS)
    rows = []
    for group, (cond, mod) in interactions.items():
        if group not in labels.columns:
            continue
        rows.append(dict(group=group, interaction=f'{cond} x {mod}',
                         n_significant=int(labels[group].sum()),
                         n_electrodes=int(len(labels)),
                         frac=float(labels[group].mean())))
    return pd.DataFrame(rows)


def run_power_traces_conjunction(labels, n_perm=10000, seed=0,
                                 thresholds=(0.01, 0.025, 0.05, 0.10)):
    """Full conjunction battery on `power_traces`-defined electrodes.

    Runs, on the labels produced by `electrode_labels`:
      * `cmh_conjunction`          -- subject-stratified Mantel-Haenszel OR
      * `conjunction_permutation_null` -- within-subject null for the joint count
      * `both_vs_distinct_test`    -- shared vs distinct (see its docstring for
                                      why this is the same test as the above)
      * `cross_control_counts`     -- the two cross interactions as specificity
      * a threshold sweep, when `q_` columns are available

    Reading it: MH OR > 1 with a low CMH p means the two selectivities co-occur
    on the same electrodes more than chance -- a shared core. OR < 1 is
    segregation. The sweep must not flip the sign of that conclusion across
    reasonable cutoffs.
    """
    res = {
        'cmh': cmh_conjunction(labels),
        'joint_count_null': conjunction_permutation_null(labels, n_perm=n_perm,
                                                         seed=seed),
        'both_vs_distinct': both_vs_distinct_test(labels, n_perm=n_perm, seed=seed),
        'cross_controls': cross_control_counts(labels),
        'attrs': dict(labels.attrs),
    }

    if 'q_cpc' in labels.columns and labels['q_cpc'].notna().any():
        def at_threshold(t):
            lab = labels.copy()
            lab['S'] = (lab['q_cpc'] < t).astype(int)
            lab['F'] = (lab['q_sps'] < t).astype(int)
            return lab
        res['sweep'] = conjunction_threshold_sweep(at_threshold, list(thresholds))
    return res


def summarize(res):
    """Human-readable one-screen summary of `run_power_traces_conjunction`."""
    bvd = res['both_vs_distinct']
    cmh = res['cmh']
    lines = [
        f"electrodes: {bvd['n_electrodes']}  subjects: {bvd['n_subjects']}"
        f"  (roi={res['attrs'].get('roi')}, correction={res['attrs'].get('correction')})",
        f"  both={bvd['n_both']}  S-only={bvd['n_stab_only']}  "
        f"F-only={bvd['n_flex_only']}  neither={bvd['n_neither']}",
        f"  MH odds ratio = {cmh['mh_odds_ratio']:.3f}"
        f"   CMH p = {cmh['cmh'].pvalue:.4g}"
        f"   homogeneity p = {cmh['homogeneity'].pvalue:.4g}",
        f"  shared - distinct = {bvd['observed']}"
        f"  (null {bvd['null_mean']:.1f} +/- {bvd['null_sd']:.1f},"
        f" z = {bvd['z']:.2f})",
        f"  p(shared > distinct, one-sided) = {bvd['p_shared_greater']:.4g}"
        f"   two-sided = {bvd['p_two_sided']:.4g}",
        "  cross controls (should be near-null):",
    ]
    for _, r in res['cross_controls'].iterrows():
        lines.append(f"    {r['group']:<4} {r['interaction']:<45} "
                     f"{r['n_significant']:>4}/{r['n_electrodes']} "
                     f"({100 * r['frac']:.1f}%)")
    if 'sweep' in res:
        lines.append("  threshold sweep:")
        for _, r in res['sweep'].iterrows():
            lines.append(f"    q<{r['threshold']:<5} n_S={int(r['n_S']):>3} "
                         f"n_F={int(r['n_F']):>3} n_both={int(r['n_both']):>3} "
                         f"OR={r['mh_odds_ratio']:.3f} p={r['cmh_p']:.4g}")
    return '\n'.join(lines)
