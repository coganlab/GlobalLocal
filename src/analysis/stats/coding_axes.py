"""Coding axes and the cosine between them (docs/analysis_plans.md, subspace
plan §3.1-3.3).

Input is a per-split table of main effects, one row per electrode x split
(x window): congruency on the two trial halves in `mxA` / `mxB`, switch type in
`myA` / `myB`, as `stability_flexibility_segregation.main_effect_mean_diffs`
(or `compute_sensitivities_per_split(..., main_effects=True)`) writes it. A
table without a `window` column is one window, e.g. a segregation run's
`per_split.csv`.

Per window and split, over the pooled electrodes, with c_A, c_B, s_A, s_B the
congruency and switch vectors on halves A and B:

    cross  = 1/2 [cos(c_A, s_B) + cos(c_B, s_A)]   never two effects from the same trials
    ceil_c = cos(c_A, c_B),   ceil_s = cos(s_A, s_B)   each axis against itself

each averaged over splits; then

    ceiling   = sqrt(ceil_c * ceil_s)    (NaN unless both are > 0)
    alignment = cross / ceiling          0 = separate, 1 = as shared as noise allows

a ratio of split averages, not an average of per-split ratios, since one split's
ceiling can be <= 0. The cosine is uncentred (§3.1); the `_pearson` columns
repeat each number on vectors centred across the pooled electrodes, i.e. the
Pearson r the anatomy reports. Their gap is the part of the sharing that is an
across-the-board shift.

The two tests of §3.2, both needed for "partly shared":
  * Anything shared? `cross_cosine_null` rescored with one factor's labels
    shuffled, tested on the RAW cross cosine, cluster-corrected over windows by
    cluster mass (`shared_windows`, which says why not A4's run length).
  * Not one code? A participant bootstrap of ceiling - cross: its CI excludes 0.
Windows whose ceilings' CIs include 0 carry no reliable code and are flagged.
`compare_coding_axes` computes all of it from the in-memory tables.

Controls (§3.3): `rt_control` adjusts or RT-matches the trials before scoring
(with a random-subset control for the matching); `remove_mean` is an option of
the scorer; the region control is a different electrode selection.
"""

import warnings

import numpy as np
import pandas as pd

from . import stability_flexibility_segregation as sfs

AXIS_COLS = ('mxA', 'mxB', 'myA', 'myB')        # congruency A/B, switch type A/B
DESIGN_COLS = ('congruency', 'switchType', 'incongruent_proportion', 'switch_proportion')
NULL_FACTORS = ('congruency', 'switchType')

# The dot products every cosine is built from: name -> (column, column).
_PAIRS = {'cA_sB': ('mxA', 'myB'), 'cB_sA': ('mxB', 'myA'),
          'cA_cB': ('mxA', 'mxB'), 'sA_sB': ('myA', 'myB'),
          'cA_cA': ('mxA', 'mxA'), 'cB_cB': ('mxB', 'mxB'),
          'sA_sA': ('myA', 'myA'), 'sB_sB': ('myB', 'myB')}


def axis_arrays(per_split):
    """``(subjects, {col: (windows, splits, electrodes)})`` for `AXIS_COLS`;
    `subjects` names each electrode's participant, in electrode order.

    Keeps the electrodes defined on every split and window, so every cosine is
    over the same electrodes (the rule `map_reliability` uses). The subspace
    test (`coding_subspaces`) reads its effect matrices from here too."""
    d = per_split if 'window' in per_split else per_split.assign(window=0)
    n_cells = d.groupby(['window', 'split']).ngroups
    d = d.dropna(subset=list(AXIS_COLS))
    d = d[d.groupby('electrode')['split'].transform('size') == n_cells]
    if d.empty:
        raise ValueError("no electrode has all four effects on every split and window")
    d = d.sort_values(['window', 'split', 'electrode'])
    shape = (d['window'].nunique(), d['split'].nunique(), d['electrode'].nunique())
    subjects = d.drop_duplicates('electrode')['subject'].to_numpy()   # electrode order
    return subjects, {c: d[c].to_numpy(float).reshape(shape) for c in AXIS_COLS}


def _pair_sums(arrays, subjects, center=False):
    """Per participant, the sums over its electrodes of every product in
    `_PAIRS`: ``{name: (windows, splits, participants)}``. A cosine over any
    weighted set of participants is a ratio of weighted sums of these, which is
    what makes the participant bootstrap cheap. `center=True` first centres each
    vector across the pooled electrodes (cosine -> Pearson r)."""
    if center:
        arrays = {c: a - a.mean(axis=-1, keepdims=True) for c, a in arrays.items()}
    codes = pd.factorize(subjects)[0]
    onehot = np.eye(codes.max() + 1)[codes]                       # electrodes x participants
    return {name: (arrays[a] * arrays[b]) @ onehot for name, (a, b) in _PAIRS.items()}


def _cosines(sums, weights=None):
    """Split-averaged ``(cross, ceil_c, ceil_s)``, each (windows,) or, with
    `weights` (participants, n), (n, windows): one row per weighting."""
    s = {k: (v.sum(-1) if weights is None else v @ weights) for k, v in sums.items()}

    def cos(ab, aa, bb):
        with np.errstate(divide='ignore', invalid='ignore'):
            return s[ab] / np.sqrt(s[aa] * s[bb])

    per_split = (0.5 * (cos('cA_sB', 'cA_cA', 'sB_sB') + cos('cB_sA', 'cB_cB', 'sA_sA')),
                 cos('cA_cB', 'cA_cA', 'cB_cB'), cos('sA_sB', 'sA_sA', 'sB_sB'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN windows stay NaN
        out = [np.nanmean(v, axis=1) for v in per_split]         # (windows[, n])
    return tuple(o if weights is None else o.T for o in out)


def _ceiling(ceil_c, ceil_s):
    """sqrt(ceil_c * ceil_s), NaN unless both ceilings are positive."""
    ok = (ceil_c > 0) & (ceil_s > 0)
    return np.where(ok, np.sqrt(np.where(ok, ceil_c * ceil_s, 1.0)), np.nan)


def coding_axis_cosines(per_split):
    """The point estimates of §3.2 per window: cross, the two ceilings, ceiling and
    alignment, as cosines and (``_pearson``) as correlations. One row per window."""
    subjects, arrays = axis_arrays(per_split)
    cross, cc, cs = _cosines(_pair_sums(arrays, subjects))
    cross_p, cc_p, cs_p = _cosines(_pair_sums(arrays, subjects, center=True))
    ceiling, ceiling_p = _ceiling(cc, cs), _ceiling(cc_p, cs_p)
    return pd.DataFrame(dict(
        window=np.arange(len(cross)), n_electrodes=len(subjects),
        n_subjects=len(set(subjects)),
        cross=cross, congruency_ceiling=cc, switch_ceiling=cs,
        ceiling=ceiling, alignment=cross / ceiling,
        cross_pearson=cross_p, congruency_ceiling_pearson=cc_p,
        switch_ceiling_pearson=cs_p, ceiling_pearson=ceiling_p,
        alignment_pearson=cross_p / ceiling_p))


RT_CONTROLS = ('none', 'adjust', 'match', 'random')
_RT_GROUPS = ('congruency', 'switchType')        # the factors whose RT is matched


def rt_control(labels, hg_windows, mode='none', n_bins=10, seed=0):
    """The RT control of §3.3, applied to the trials before they are scored.

    'adjust'  removes the RT-linked part of each trial's HG in every window,
              electrode by electrode: `rt_adjust_hg`, the pooled within-design-
              cell slope of HG on RT (as RT_ADJUST_HG in the segregation run).
              Trials without an RT get NaN HG and drop out of the cell means.
    'match'   keeps the subset of each participant's trials in which the four
              congruency x switch-type cells have the same RT distribution
              (`rt_match` with `n_bins` quantile bins: A4's RT_MATCH=rt).
    'random'  the control for 'match': as many trials per participant and cell
              as 'match' keeps, drawn without regard to RT
              (`count_matched_random`). A cross cosine that drops under 'match'
              but not under 'random' is the RT-linked part; a drop under both is
              the smaller sample.
    'none'    the inputs unchanged.

    Trials are kept or dropped per participant, so all of a participant's
    electrodes keep the same trials and the shared halves stay shared.

    `labels` and `hg_windows` are as for
    `stability_flexibility_segregation.main_effect_mean_diffs`; `labels` needs
    an `rt` column. Returns ``(labels, hg_windows, report)`` with `report` a
    dict of tables to save: the per-electrode, per-window slopes for 'adjust';
    the per-participant RT differences before and after, and their summary, for
    'match' and 'random'; empty for 'none'.
    """
    if mode not in RT_CONTROLS:
        raise ValueError(f"mode must be one of {RT_CONTROLS}; got {mode!r}")
    hg_windows = np.asarray(hg_windows, float)
    if mode == 'none':
        return labels, hg_windows, {}
    if mode == 'adjust':
        from .stability_flexibility_brain_behavior import rt_adjust_hg
        columns, slopes = [], []
        for w in range(hg_windows.shape[1]):
            adjusted, slope = rt_adjust_hg(labels.assign(hg=hg_windows[:, w]))
            columns.append(adjusted['hg'].to_numpy(float))
            slopes.append(slope.assign(window=w))
        return (labels, np.column_stack(columns),
                {'rt_adjust_slopes': pd.concat(slopes, ignore_index=True)})

    from src.analysis.utils import rt_matching as rtm
    trials = labels.drop_duplicates(['subject', 'trial']).reset_index(drop=True)
    within = ('subject',)
    keep = rtm.rt_match(trials, _RT_GROUPS, within=within, rt_col='rt',
                        n_bins=n_bins, seed=seed)
    if mode == 'random':
        keep = rtm.count_matched_random(trials, keep, _RT_GROUPS, within=within,
                                        rt_col='rt', seed=seed)
    contrasts = rtm.rt_contrast_table(trials, keep, _RT_GROUPS, within=within, rt_col='rt')
    kept = pd.MultiIndex.from_frame(trials.loc[keep.to_numpy(), ['subject', 'trial']])
    rows = pd.MultiIndex.from_frame(labels[['subject', 'trial']]).isin(kept)
    return (labels.loc[rows].reset_index(drop=True), hg_windows[rows],
            {f'rt_{mode}_contrasts': contrasts,
             f'rt_{mode}_summary': rtm.summarize_rt_contrasts(contrasts)})


def permute_within_cells(df, factor, rng):
    """`df` with `factor`'s labels shuffled among each participant's trials that
    share the other three design factors (§3.2).

    One permutation per participant, applied to all of its electrodes: they share
    trials, so their trial-to-trial noise is correlated, and permuting each
    electrode separately would make that noise independent and the null too
    narrow. Swapping only within cells of the other factors keeps the design
    (cell counts, and congruency's correlation with incProp) and destroys only
    `factor`'s effect."""
    others = [c for c in DESIGN_COLS if c != factor]
    trials = (df.drop_duplicates(['subject', 'trial'])[['subject', 'trial', factor, *others]]
              .reset_index(drop=True))
    orig = trials[factor].to_numpy()
    new = orig.copy()
    for idx in trials.groupby(['subject', *others], dropna=False).indices.values():
        new[idx] = orig[rng.permutation(idx)]
    lookup = pd.Series(new, index=pd.MultiIndex.from_frame(trials[['subject', 'trial']]))
    out = df.copy()
    out[factor] = lookup.reindex(pd.MultiIndex.from_frame(df[['subject', 'trial']])).to_numpy()
    return out


def null_draws(df, hg_windows, factor, n_perm=1000, n_splits=20, seed=0,
               remove_mean=False):
    """Yield `n_perm` per-split tables rescored with `factor` ('congruency' or
    'switchType') shuffled: the draws of the "no code for `factor`" null (§3.2).

    Each draw shuffles `factor` with `permute_within_cells` and rescores every
    window with `main_effect_mean_diffs` on `n_splits` fresh halves; the other
    factor's effects stay real. `df` and `hg_windows` are as for
    `main_effect_mean_diffs`. Every draw is seeded from `seed`, so the null is
    reproducible from it. Any statistic of the real per-split table can be
    recomputed on each draw: `cross_cosine_null` takes the cross cosine, the
    subspace test (`coding_subspaces.subspace_null_stats`) its own numbers, and
    one pass of this generator can feed both."""
    if factor not in NULL_FACTORS:
        raise ValueError(f"factor must be one of {NULL_FACTORS}; got {factor!r}")
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        yield sfs.main_effect_mean_diffs(
            permute_within_cells(df, factor, rng), hg_windows, n_splits=n_splits,
            seed=int(rng.integers(2 ** 31)), remove_mean=remove_mean)


def cross_cosine(per_split):
    """The split-averaged raw cross cosine per window (the `cross` column of
    `coding_axis_cosines`), without the rest of the table: what a null draw
    records."""
    subjects, arrays = axis_arrays(per_split)
    return _cosines(_pair_sums(arrays, subjects))[0]


def cross_cosine_null(df, hg_windows, factor, n_perm=1000, n_splits=20, seed=0,
                      remove_mean=False):
    """The raw cross cosine per window with `factor` shuffled: an
    (n_perm, windows) array, the "no code for `factor`" null of §3.2
    (`null_draws`, scored with `cross_cosine`).

    The observed cross cosine averages more splits (200) than each null draw
    (`n_splits`); averaging fewer splits only widens the null, so the test is
    conservative."""
    return np.array([cross_cosine(per) for per in null_draws(
        df, hg_windows, factor, n_perm=n_perm, n_splits=n_splits, seed=seed,
        remove_mean=remove_mean)])


def _cluster_masses(series, threshold):
    """``[(start, end, mass)]`` for each run of windows where `series` is above
    `threshold`; mass is the summed excess over the threshold."""
    from src.analysis.decoding.accuracy_stats import find_contiguous_clusters
    excess = series - threshold
    return [(s, e, float(excess[s:e + 1].sum()))
            for s, e in find_contiguous_clusters(np.nan_to_num(excess, nan=-1.0) > 0)]


def shared_windows(cross, null, percentile=95, cluster_percentile=95):
    """Boolean per window: inside a run of windows where the cross cosine is above
    the null's pointwise `percentile`, whose cluster MASS (summed excess over that
    threshold) is above the `cluster_percentile` of the largest masses the null
    draws produce, each null draw scored against the other draws' thresholds.

    A4's test (`find_significant_clusters_of_series_vs_distribution_based_on_
    percentile`) has the same pointwise step but judges a run by its LENGTH.
    That fails here: each null draw is one shuffle scored in every window, so
    the draws are strongly correlated across windows, a high draw is high
    everywhere, the null's longest runs span the whole stretch, and no real run
    can be longer. Mass also counts how far above the threshold a run sits,
    which a high draw does not match (Maris & Oostenveld 2007). With a single
    window there is nothing to cluster, so it is the pointwise test."""
    cross, null = np.asarray(cross, float), np.asarray(null, float)
    threshold = np.percentile(null, percentile, axis=0)
    if len(cross) == 1:
        return cross > threshold
    null_max = [max([m for *_, m in _cluster_masses(
                    draw, np.percentile(np.delete(null, i, axis=0), percentile, axis=0))],
                    default=0.0)
                for i, draw in enumerate(null)]
    critical = np.percentile(null_max, cluster_percentile)
    mask = np.zeros(len(cross), bool)
    for start, end, mass in _cluster_masses(cross, threshold):
        if mass > critical:
            mask[start:end + 1] = True
    return mask


def _bootstrap(sums, n_boot, rng, chunk=200):
    """Participant-bootstrap draws of (cross, ceil_c, ceil_s), each (n_boot, windows):
    participants resampled with replacement, keeping their real labels."""
    n_part = next(iter(sums.values())).shape[-1]
    draws = []
    for start in range(0, n_boot, chunk):
        n = min(chunk, n_boot - start)
        weights = rng.multinomial(n_part, np.full(n_part, 1 / n_part), size=n).T
        draws.append(_cosines(sums, weights.astype(float)))
    return tuple(np.concatenate(d) for d in zip(*draws))


def compare_coding_axes(per_split, null=None, n_boot=2000, seed=0, alpha=0.05):
    """Everything §3.2 reports, per window, and its summary.

    `per_split` is the real-label table (module docstring). `null` maps a factor
    to its `cross_cosine_null` array; without it the shuffle test is skipped.
    Returns ``(table, summary)``. `table` has the `coding_axis_cosines` columns
    plus, per window:

        congruency_ceiling_lo/_hi, switch_ceiling_lo/_hi   bootstrap CIs
        reliable        both ceilings' CIs exclude 0 (grey out the rest)
        gap_lo/_hi      CI of ceiling - cross
        not_one_code    reliable and gap_lo > 0
        alignment_lo/_hi
        null_lo_<f>/null_hi_<f>   the `alpha` and 1 - `alpha` percentiles of the
                        null: the band to plot, whose upper edge is the test's
                        threshold
        p_<f>           pointwise one-sided permutation p of the raw cross cosine
        shared_<f>      cluster-corrected, `shared_windows`
        shared          shared under every factor's null: neither "no congruency
                        code" nor "no switch code" explains the cross cosine

    CIs are the central 1 - `alpha` of the bootstrap. `summary` holds the
    alignment averaged over the reliable windows with its bootstrap CI (the
    number for the text), the windows of each marked run, and the settings.
    """
    subjects, arrays = axis_arrays(per_split)
    sums = _pair_sums(arrays, subjects)
    table = coding_axis_cosines(per_split)
    grid_cols = [c for c in ('window_tmin', 'window_tmax', 'window_center') if c in per_split]
    if grid_cols:
        grid = per_split.drop_duplicates('window')[['window', *grid_cols]]
        table = table.merge(grid, on='window', how='left')

    rng = np.random.default_rng(seed)
    b_cross, b_cc, b_cs = _bootstrap(sums, n_boot, rng)
    b_ceiling = _ceiling(b_cc, b_cs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        b_align = b_cross / b_ceiling
        q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
        for name, draws in (('congruency_ceiling', b_cc), ('switch_ceiling', b_cs),
                            ('gap', b_ceiling - b_cross), ('alignment', b_align)):
            table[f'{name}_lo'], table[f'{name}_hi'] = np.nanpercentile(draws, q, axis=0)
    table['reliable'] = (table['congruency_ceiling_lo'] > 0) & (table['switch_ceiling_lo'] > 0)
    table['not_one_code'] = table['reliable'] & (table['gap_lo'] > 0)

    for factor, draws in (null or {}).items():
        draws = np.asarray(draws, float)
        table[f'null_lo_{factor}'], table[f'null_hi_{factor}'] = np.percentile(
            draws, [100 * alpha, 100 * (1 - alpha)], axis=0)
        table[f'p_{factor}'] = ((draws >= table['cross'].to_numpy()).sum(0) + 1) / (len(draws) + 1)
        table[f'shared_{factor}'] = shared_windows(
            table['cross'], draws, percentile=100 * (1 - alpha),
            cluster_percentile=100 * (1 - alpha))
    if null:
        table['shared'] = table[[f'shared_{f}' for f in null]].all(axis=1)

    reliable = table['reliable'].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        b_mean = np.nanmean(b_align[:, reliable], axis=1) if reliable.any() else np.full(n_boot, np.nan)
    summary = dict(
        n_electrodes=int(table['n_electrodes'].iloc[0]),
        n_subjects=int(table['n_subjects'].iloc[0]),
        n_windows=len(table), n_reliable_windows=int(reliable.sum()),
        alignment_reliable_mean=(float(table.loc[reliable, 'alignment'].mean())
                                 if reliable.any() else None),
        alignment_reliable_ci=([float(v) for v in np.nanpercentile(b_mean, q)]
                               if reliable.any() else None),
        not_one_code_windows=table.loc[table['not_one_code'], 'window'].tolist(),
        shared_windows=(table.loc[table['shared'], 'window'].tolist() if null else None),
        n_boot=n_boot, seed=seed, alpha=alpha,
        n_perm={f: int(len(d)) for f, d in (null or {}).items()})
    return table, summary
