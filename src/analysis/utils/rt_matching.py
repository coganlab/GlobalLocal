"""RT matching: subsample trials so the groups being compared have the same RT
distribution.

Why
---
Incongruent and switch trials are slower than congruent and repeat trials. In
stimulus-locked data, anything that tracks time-to-response (response
preparation, sustained activity that ends at the response) then differs between
the levels of both factors, and a decoder or a power contrast can read that
latency difference as a condition effect. Matching RTs asks whether the effect
survives when the compared trials were, on average, equally fast.

How
---
Matching is done separately in each stratum (always per subject, since RT scales
differ across patients; optionally also per block type, etc.):

1. Pool the stratum's RTs over every group and cut them into `n_bins` quantile
   bins, so each bin holds about the same number of trials and the bins are
   finest where the data are densest. Quantile bins are invariant to monotone
   transforms, so RT vs log RT does not matter.
2. Count each group's trials per bin, pick how many to keep per group and bin
   (`balance`, below), and draw that many at random without replacement.

`balance='equal'` keeps the same number of trials in every group within each
bin (the minimum across groups), so every group ends up with the same size and
the same bin profile. Decoders balance classes anyway, so nothing extra is lost
there. `balance='proportional'` keeps the groups' size ratios and gives every
group the same bin profile; use it when group sizes should stay unequal (e.g. a
power contrast inside a 75/25 block).

Matching is only as fine as the bins: inside a bin the slower group can still
sit slightly later. `rt_contrast_table` reports the residual per-subject RT
difference after matching, so check it rather than assume it is zero.

The control run
---------------
Matching throws away roughly half the trials, and a weaker effect after
matching could just be the smaller sample. `count_matched_random` keeps exactly
as many trials per stratum and group as the RT-matched set, drawn at random
with no regard to RT. Run both; an effect that drops under RT matching but not
under the count-matched random subset is the RT-driven part.

Entry points
------------
- `rt_match(trials, groups, ...)`: the pure pandas core. `trials` is any
  one-row-per-trial DataFrame with an RT column; returns a boolean mask.
- `rt_match_subjects_mne_objects(subjects_mne_objects, ...)`: the adapter for
  the `{subject: {condition: {key: Epochs}}}` structure every loader in this
  project returns (`create_subjects_mne_objects_dict`). Reads RT, trial id and
  factor levels from the epochs' metadata, matches, and slices every Epochs
  object to the kept trials. One line after loading plugs it into any analysis:

      subjects_mne_objects, rt_report = rt_match_subjects_mne_objects(
          subjects_mne_objects, groups=('congruency', 'switchType'))

  `mode='random'` returns the count-matched random control instead. The
  loader's `<key>_avg` / `<key>_std_err` Evoked objects are rebuilt from the kept
  trials (`refresh_evoked`), so power-trace plots average the matched set too.
- `count_matched_random(trials, keep, groups, ...)`: the control, as a mask.
- `rt_balance_table` / `rt_contrast_table` / `summarize_rt_contrasts`:
  before/after reports.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

RT_COL = 'reaction_time'
TRIAL_ID_COL = 'trial_count'
SUBJECT_COL = 'subject'
BALANCE_MODES = ('equal', 'proportional')
MATCH_MODES = ('rt', 'random')
# On the GlobalLocal behaviour (26 subjects, correct trials, the four congruency x
# switch-type cells), 10 bins leave a residual per-subject difference of about
# +2 ms (i - c) and +1 ms (s - r), neither significant, from +153 / +197 ms, and
# keep about half the trials. 5 bins keep a little more but leave i - c at
# +8 ms (p = 0.03).
DEFAULT_N_BINS = 10

# Factor names used across the project (condition-config camelCase, the
# cross-decoding snake_case, and the metadata's own names) -> the epochs
# metadata column written by `make_metadata_from_event_names`.
METADATA_COLUMNS = {
    'congruency': 'congruency',
    'switchType': 'task_sequence',
    'switch_type': 'task_sequence',
    'task_sequence': 'task_sequence',
    'incongruentProportion': 'incongruent_proportion',
    'incongruent_proportion': 'incongruent_proportion',
    'switchProportion': 'switch_proportion',
    'switch_proportion': 'switch_proportion',
    'task': 'task',
}


def metadata_columns(factors):
    """Map factor names (any project spelling) to epochs-metadata columns.

    Unknown names pass through unchanged, so a metadata column can be named
    directly (e.g. 'prev_congruency')."""
    if isinstance(factors, str):
        factors = (factors,)
    return tuple(METADATA_COLUMNS.get(f, f) for f in factors)


# ---------------------------------------------------------------------------
# the core
# ---------------------------------------------------------------------------
def _stratum_rng(seed, key):
    """A generator that depends only on the seed and the stratum's key, so a
    stratum's draw does not change when other strata are added or reordered."""
    return np.random.default_rng([int(seed), zlib.crc32(repr(key).encode())])


def _rt_bins(rt, n_bins):
    """Quantile-bin index per trial (0 .. <= n_bins-1), from the pooled RTs."""
    edges = np.quantile(rt, np.linspace(0, 1, n_bins + 1))
    interior = np.unique(edges[1:-1])
    return np.searchsorted(interior, rt, side='right')


def _keep_counts(counts, balance):
    """(groups x bins) available counts -> (groups x bins) counts to keep."""
    if balance == 'equal':
        return np.repeat(counts.min(axis=0, keepdims=True), counts.shape[0], axis=0)
    totals = counts.sum(axis=1, keepdims=True).astype(float)
    shares = counts / totals                        # each group's bin profile
    common = shares.min(axis=0)                     # the profile all groups can meet
    if common.sum() == 0:
        return np.zeros_like(counts)
    common = common / common.sum()
    used = common > 0
    # largest group size r_g with r_g * common_b <= counts_gb in every used bin
    size = (counts[:, used] / common[used]).min(axis=1)
    take = np.zeros_like(counts)
    for g, r in enumerate(size):
        # Largest-remainder rounding of r * common to floor(r) trials. Flooring
        # every bin separately would drop relatively more from the sparse bins
        # and bias the kept RT profile; this keeps the rounding error below one
        # trial per bin. ceil(r * common_b) <= counts_gb, so the cap never binds
        # unless r * common_b is an exact integer.
        target = r * common
        base = np.floor(target + 1e-9).astype(int)
        extra = int(np.floor(r + 1e-9)) - base.sum()
        if extra > 0:
            order = np.argsort(-(target - base), kind='stable')
            base[order[:extra]] += 1
        take[g] = np.minimum(base, counts[g])
    return take


def rt_match(trials, groups, within=(SUBJECT_COL,), rt_col=RT_COL, n_bins=DEFAULT_N_BINS,
             balance='equal', seed=0):
    """Boolean mask over `trials` keeping an RT-matched subset.

    Parameters
    ----------
    trials : DataFrame, one row per trial.
    groups : column name(s) whose level combinations must end up RT-matched,
        e.g. ('congruency', 'task_sequence') matches all four
        congruency x switch-type cells to one another, so neither factor carries
        an RT difference.
    within : column name(s) to match separately inside (default: per subject).
        Add e.g. 'incongruent_proportion' to match within each block level as
        well. Pass () to match over the whole table.
    rt_col : the RT column. Rows with a missing or non-finite RT are dropped.
    n_bins : quantile bins per stratum. More bins match more tightly and keep
        fewer trials; `rt_contrast_table` shows the residual difference.
    balance : 'equal' or 'proportional' (see the module docstring).
    seed : seeds the random draw inside each (stratum, group, bin) cell.

    Rows with a missing value in any `groups` column are dropped. A stratum with
    fewer than two groups present has nothing to match against and is dropped
    entirely.

    Returns
    -------
    pd.Series of bool, aligned to `trials.index`.
    """
    if balance not in BALANCE_MODES:
        raise ValueError(f"balance must be one of {BALANCE_MODES}; got {balance!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    groups = [groups] if isinstance(groups, str) else list(groups)
    within = [within] if isinstance(within, str) else list(within)
    if not groups:
        raise ValueError("groups must name at least one column")
    missing = [c for c in groups + within + [rt_col] if c not in trials.columns]
    if missing:
        raise KeyError(f"trials has no column(s) {missing}; have {list(trials.columns)}")

    keep = pd.Series(False, index=trials.index)
    rt = pd.to_numeric(trials[rt_col], errors='coerce')
    usable = np.isfinite(rt.to_numpy(dtype=float)) & trials[groups].notna().all(axis=1).to_numpy()
    data = trials.loc[usable]
    if data.empty:
        return keep

    strata = data.groupby(within, sort=True, dropna=False) if within else [((), data)]
    for key, stratum in strata:
        group_keys = stratum[groups].apply(tuple, axis=1)
        levels = sorted(group_keys.unique(), key=repr)
        if len(levels) < 2:
            continue
        stratum_rt = pd.to_numeric(stratum[rt_col]).to_numpy(dtype=float)
        bins = _rt_bins(stratum_rt, n_bins)
        n_bin_levels = int(bins.max()) + 1
        members = {}
        counts = np.zeros((len(levels), n_bin_levels), dtype=int)
        for gi, level in enumerate(levels):
            in_group = (group_keys == level).to_numpy()
            for b in range(n_bin_levels):
                idx = stratum.index[in_group & (bins == b)]
                members[gi, b] = idx
                counts[gi, b] = len(idx)
        take = _keep_counts(counts, balance)
        rng = _stratum_rng(seed, key)
        for (gi, b), idx in members.items():
            k = take[gi, b]
            if k:
                keep.loc[rng.choice(idx.to_numpy(), size=k, replace=False)] = True
    return keep


def count_matched_random(trials, keep, groups, within=(SUBJECT_COL,), rt_col=RT_COL,
                         seed=0):
    """The control for `rt_match`: the same number of trials per stratum x group
    as `keep` holds, drawn at random with no regard to RT.

    Candidates are the rows `rt_match` could have used (finite RT, every group
    column present), so both subsets are drawn from the same pool.
    """
    groups = [groups] if isinstance(groups, str) else list(groups)
    within = [within] if isinstance(within, str) else list(within)
    keep = pd.Series(np.asarray(keep, dtype=bool), index=trials.index)
    rt = pd.to_numeric(trials[rt_col], errors='coerce')
    usable = np.isfinite(rt.to_numpy(dtype=float)) & trials[groups].notna().all(axis=1).to_numpy()
    data = trials.loc[usable]
    out = pd.Series(False, index=trials.index)
    for key, cell in data.groupby(within + groups, sort=True, dropna=False):
        k = int(keep.loc[cell.index].sum())
        if k:
            rng = _stratum_rng(seed, ('random',) + (key if isinstance(key, tuple) else (key,)))
            out.loc[rng.choice(cell.index.to_numpy(), size=k, replace=False)] = True
    return out


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def rt_balance_table(trials, keep, groups, within=(SUBJECT_COL,), rt_col=RT_COL):
    """Per stratum x group: trial counts and mean/median RT, before and after."""
    groups = [groups] if isinstance(groups, str) else list(groups)
    within = [within] if isinstance(within, str) else list(within)
    df = trials.assign(_rt=pd.to_numeric(trials[rt_col], errors='coerce'),
                       _keep=np.asarray(keep, dtype=bool))
    rows = []
    for key, g in df.groupby(within + groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        kept = g[g['_keep']]
        rows.append(dict(zip(within + groups, key),
                         n_before=len(g), n_after=len(kept),
                         mean_rt_before=g['_rt'].mean(), mean_rt_after=kept['_rt'].mean(),
                         median_rt_before=g['_rt'].median(),
                         median_rt_after=kept['_rt'].median()))
    return pd.DataFrame(rows)


def _level_key(value):
    if isinstance(value, (int, float, np.number)):
        return (0, float(value), '')
    return (1, 0.0, str(value))


def rt_contrast_table(trials, keep, factors, within=(SUBJECT_COL,), rt_col=RT_COL):
    """Per stratum and two-level factor: RT difference between the levels, before
    and after matching. Levels are sorted (numbers numerically, labels as text)
    and the later one comes first: 'i' - 'c', 's' - 'r', 75 - 25."""
    factors = [factors] if isinstance(factors, str) else list(factors)
    within = [within] if isinstance(within, str) else list(within)
    df = trials.assign(_rt=pd.to_numeric(trials[rt_col], errors='coerce'),
                       _keep=np.asarray(keep, dtype=bool))
    strata = df.groupby(within, sort=True, dropna=False) if within else [((), df)]
    rows = []
    for key, s in strata:
        key = key if isinstance(key, tuple) else (key,)
        for factor in factors:
            levels = sorted(s[factor].dropna().unique(), key=_level_key)
            if len(levels) != 2:
                continue
            lo, hi = levels

            def diff(frame):
                return (frame.loc[frame[factor] == hi, '_rt'].mean()
                        - frame.loc[frame[factor] == lo, '_rt'].mean())
            rows.append(dict(zip(within, key), factor=factor,
                             contrast=f'{hi} - {lo}',
                             diff_before=diff(s), diff_after=diff(s[s['_keep']])))
    return pd.DataFrame(rows)


def summarize_rt_contrasts(contrasts):
    """Across strata: mean, SD and a one-sample t-test of the per-stratum RT
    differences, before and after matching. The 'after' p is the number to report
    ("residual RT difference, t(n-1) = ..., p = ...")."""
    from scipy import stats
    rows = []
    for (factor, contrast), g in contrasts.groupby(['factor', 'contrast'], sort=False):
        row = dict(factor=factor, contrast=contrast, n_strata=len(g))
        for when in ('before', 'after'):
            d = g[f'diff_{when}'].dropna().to_numpy()
            row[f'mean_{when}'] = d.mean() if d.size else np.nan
            row[f'sd_{when}'] = d.std(ddof=1) if d.size > 1 else np.nan
            if d.size > 1 and np.ptp(d) > 0:
                t, p = stats.ttest_1samp(d, 0.0)
            else:
                t, p = np.nan, np.nan
            row[f't_{when}'], row[f'p_{when}'] = t, p
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# adapter: the project's {subject: {condition: {key: Epochs}}} structure
# ---------------------------------------------------------------------------
def _is_epochs_like(obj):
    return (obj is not None and hasattr(obj, 'get_data') and hasattr(obj, 'times')
            and hasattr(obj, '__getitem__') and hasattr(obj, '__len__'))


def trials_table(subjects_mne_objects, id_col=TRIAL_ID_COL):
    """One row per unique (subject, trial id) across every Epochs object in the
    structure, with that trial's metadata. `subject` is the structure's own key
    (the metadata's parsed subject id can be spelled differently). A trial that
    appears in several conditions keeps its first metadata row."""
    frames = []
    for sub, cond_dict in subjects_mne_objects.items():
        for obj_dict in cond_dict.values():
            if not isinstance(obj_dict, dict):
                continue
            for obj in obj_dict.values():
                md = getattr(obj, 'metadata', None) if _is_epochs_like(obj) else None
                if md is None or id_col not in md.columns or len(md) == 0:
                    continue
                frames.append(md.assign(**{SUBJECT_COL: sub}))
                break                  # one Epochs per condition is enough
    if not frames:
        raise ValueError(
            f"no Epochs in the structure carries a '{id_col}' metadata column, so "
            "trials cannot be identified for RT matching. Epochs written by "
            "make_epoched_data.py carry it (make_metadata_from_event_names).")
    table = pd.concat(frames, ignore_index=True)
    return table.drop_duplicates([SUBJECT_COL, id_col]).reset_index(drop=True)


def refresh_evoked(structure):
    """Recompute each Epochs' `<key>_avg` / `<key>_std_err` Evoked from its trials.

    `create_subjects_mne_objects_dict` stores those next to every Epochs, and the
    power-trace plots read the `_avg` ones (`evoked_builders`). After trials are
    dropped they would still average the full set, so rebuild them exactly as the
    loader does: NaN-aware mean with `nave` = trials that are not all-NaN, and
    the NaN-aware standard error (0 where undefined). An Evoked whose Epochs has
    no valid trial left is removed. Returns a new structure; the input is not
    modified.
    """
    import warnings
    out = {}
    for sub, conds in structure.items():
        out[sub] = {}
        for cond, objs in conds.items():
            if not isinstance(objs, dict):
                out[sub][cond] = objs
                continue
            objs = dict(objs)
            for key, epochs in list(objs.items()):
                if not _is_epochs_like(epochs):
                    continue
                data = epochs.get_data()
                n_valid = int((~np.all(np.isnan(data), axis=(1, 2))).sum())
                for suffix in ('_avg', '_std_err'):
                    name = key + suffix
                    if name not in objs:
                        continue
                    if n_valid == 0:
                        del objs[name]
                        continue
                    evoked = objs[name].copy()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        if suffix == '_avg':
                            evoked.data = np.nanmean(data, axis=0)
                            evoked.nave = n_valid
                        else:
                            n = np.maximum(np.sum(~np.isnan(data), axis=0), 1)
                            evoked.data = np.nan_to_num(
                                np.nanstd(data, axis=0, ddof=1) / np.sqrt(n), nan=0.0)
                            evoked.nave = int(np.mean(n))
                    objs[name] = evoked
            out[sub][cond] = objs
    return out


def rt_match_subjects_mne_objects(subjects_mne_objects, groups=('congruency', 'switchType'),
                                  within=(), n_bins=DEFAULT_N_BINS, balance='equal', seed=0,
                                  mode='rt', rt_col=RT_COL, id_col=TRIAL_ID_COL,
                                  verbose=True):
    """RT-match a loaded `{subject: {condition: {key: Epochs}}}` structure.

    `groups` / `within` take any project spelling of a factor (`switchType`,
    `switch_type` and `task_sequence` all mean the metadata's `task_sequence`).
    Matching is always per subject; `within` adds further strata. Every Epochs
    object is restricted to the kept trials by `trial_count`, so the returned
    structure has the same shape as the input and drops straight into the rest
    of the pipeline.

    `mode='rt'` keeps the RT-matched trials; `mode='random'` keeps the
    count-matched random control (`count_matched_random`): the same number of
    trials per subject and group, drawn without regard to RT. Use the same `seed`
    for both runs so the random set mirrors the RT-matched one's counts.

    Returns
    -------
    (matched_structure, report) where `report` is a dict of DataFrames:
    'balance' (per subject x group counts and RTs, before and after),
    'contrasts' (per-subject RT difference per factor) and 'summary'
    (across-subject mean and t-test of those differences).
    """
    from src.analysis.decoding.anova_electrode_selection import apply_trial_partition

    if mode not in MATCH_MODES:
        raise ValueError(f"mode must be one of {MATCH_MODES}; got {mode!r}")
    group_cols = list(metadata_columns(groups))
    within_cols = [SUBJECT_COL] + [c for c in metadata_columns(within) if c != SUBJECT_COL]
    table = trials_table(subjects_mne_objects, id_col=id_col)
    keep = rt_match(table, group_cols, within=within_cols, rt_col=rt_col,
                    n_bins=n_bins, balance=balance, seed=seed)
    if mode == 'random':
        keep = count_matched_random(table, keep, group_cols, within=within_cols,
                                    rt_col=rt_col, seed=seed)

    kept_ids = {sub: set(table.loc[keep & (table[SUBJECT_COL] == sub), id_col].tolist())
                for sub in subjects_mne_objects}
    matched = refresh_evoked(apply_trial_partition(
        subjects_mne_objects, {sub: {'decode': ids} for sub, ids in kept_ids.items()},
        which='decode', id_col=id_col, verbose=False))

    contrasts = rt_contrast_table(table, keep, group_cols, within=within_cols, rt_col=rt_col)
    report = dict(balance=rt_balance_table(table, keep, group_cols, within=within_cols,
                                           rt_col=rt_col),
                  contrasts=contrasts,
                  summary=summarize_rt_contrasts(contrasts) if len(contrasts) else
                  pd.DataFrame())
    if verbose:
        print(format_rt_match_summary(report, groups=group_cols, n_bins=n_bins,
                                      balance=balance, mode=mode))
        dropped = sorted(set(subjects_mne_objects) - set(matched))
        if dropped:
            print(f"[rt-match] subjects with no trials left after matching: {dropped}")
    return matched, report


def format_rt_match_summary(report, groups, n_bins, balance, mode='rt'):
    """A few printable lines: trials kept and the residual RT differences."""
    n_before = int(report['balance']['n_before'].sum())
    n_after = int(report['balance']['n_after'].sum())
    what = 'RT-matched' if mode == 'rt' else 'count-matched RANDOM control'
    lines = [f"[rt-match] {what}, groups={list(groups)} n_bins={n_bins} balance={balance}: "
             f"kept {n_after} of {n_before} trials ({n_after / max(n_before, 1):.0%})"]
    for _, r in report['summary'].iterrows():
        lines.append(
            f"[rt-match]   {r['factor']} ({r['contrast']}): mean per-subject RT difference "
            f"{r['mean_before']:+.0f} ms before -> {r['mean_after']:+.0f} ms after "
            f"(t={r['t_after']:.2f}, p={r['p_after']:.3g}, n={r['n_strata']})")
    return "\n".join(lines)


def save_rt_match_report(report, save_dir, prefix='rt_match'):
    """Write the report's tables as CSVs into `save_dir`; returns the paths."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for name, frame in report.items():
        path = os.path.join(save_dir, f'{prefix}_{name}.csv')
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
