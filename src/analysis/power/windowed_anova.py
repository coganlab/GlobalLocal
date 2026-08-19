"""Windowed ANOVA on power data.

Building the long-form windowed dataframe, per-window OLS/ANOVA fits, the
within-electrode and across-electrode permutation cluster-correction pipelines
(sign-aware cluster splitting), FDR correction, and loading the significant
electrodes a run produced.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# `windower` is imported lazily inside `process_windowed_data_for_anova` (its only
# caller). Importing it at module level pulls in `general_utils` -> `ieeg`, which
# made this whole module -- including the pure-numpy cluster helpers and the
# summary/FDR machinery -- unimportable off-cluster.


def _parse_effect_factors(effect_name):
    """Parse a statsmodels effect name into factor names.

    'C(congruency)'                            -> ('congruency',)
    'C(congruency):C(incongruentProportion)'   -> ('congruency', 'incongruentProportion')
    Anything else / 3-way+ interactions        -> caller decides (we return the tuple as-is).
    """
    factors = re.findall(r'C\(([^)]+)\)', effect_name)
    return tuple(factors) if factors else None


def _signed_contrast_per_window(df_window, effect_name, anova_factors):
    """Signed scalar contrast for one effect in one time window.

    Main effect (one 2-level factor):
        Δ = mean[level0] - mean[level1]              (levels sorted alphabetically)
    Two-way 2x2 interaction:
        Δ = (mean[A0,B0] - mean[A1,B0]) - (mean[A0,B1] - mean[A1,B1])

    Returns NaN for 3-way+ interactions or non-2-level factors (sign is undefined
    without a canonical contrast vector). Cluster correction for those effects
    will fall back to un-split behavior.
    """
    factors = _parse_effect_factors(effect_name)
    if factors is None or len(factors) > 2:
        return np.nan

    if len(factors) == 1:
        f = factors[0]
        if f not in df_window.columns:
            return np.nan
        levels = sorted(df_window[f].dropna().unique())
        if len(levels) != 2:
            return np.nan
        m0 = df_window[df_window[f] == levels[0]]['Activity'].mean()
        m1 = df_window[df_window[f] == levels[1]]['Activity'].mean()
        return m0 - m1

    f1, f2 = factors
    if f1 not in df_window.columns or f2 not in df_window.columns:
        return np.nan
    levels1 = sorted(df_window[f1].dropna().unique())
    levels2 = sorted(df_window[f2].dropna().unique())
    if len(levels1) != 2 or len(levels2) != 2:
        return np.nan
    def cell(l1, l2):
        return df_window[(df_window[f1] == l1) & (df_window[f2] == l2)]['Activity'].mean()
    return ((cell(levels1[0], levels2[0]) - cell(levels1[1], levels2[0]))
          - (cell(levels1[0], levels2[1]) - cell(levels1[1], levels2[1])))


def _find_contiguous_runs(mask):
    """Inclusive (start, end) pairs for every contiguous True run in a 1-D bool array."""
    runs = []
    in_run = False
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs

def _split_clusters_at_sign_flips(raw_clusters, sign_trace):
    """Split each (start, end) cluster at indices where sign(trace) changes.

    Zeros and NaNs are treated as "uncertain" and break runs without contributing
    to either side. Returns list of dicts: {'start', 'end', 'sign': +1/-1}.
    """
    out = []
    for s, e in raw_clusters:
        if e < s:
            continue
        seg = np.sign(np.nan_to_num(sign_trace[s:e + 1], nan=0.0)).astype(int)
        start = None; cur_sign = 0
        for i, sg in enumerate(seg):
            if sg == 0:
                if start is not None:
                    out.append({'start': s + start, 'end': s + i - 1, 'sign': cur_sign})
                    start = None; cur_sign = 0
                continue
            if start is None:
                start = i; cur_sign = sg
            elif sg != cur_sign:
                out.append({'start': s + start, 'end': s + i - 1, 'sign': cur_sign})
                start = i; cur_sign = sg
        if start is not None:
            out.append({'start': s + start, 'end': s + len(seg) - 1, 'sign': cur_sign})
    return out

def _fit_anova_per_window_per_unit(df, formula, anova_factors, unit_cols,
                                   window_col='WindowIndex',
                                   return_sign=False):
    """Fit OLS+ANOVA at every (unit, window) cell. Returns full F-traces.

    Reusable across all ANOVA paths (within-electrode, across-electrode-within-ROI,
    perform_windowed_anova, etc.). The 'unit' is whatever grouping you want -- pass
    unit_cols=('SubjectID', 'Electrode', 'ROI') for within-electrode, ('ROI',) for
    across-electrode-within-ROI, etc.

    Returns
    -------
    f_trace_by_unit    : dict[unit_tuple] -> (n_effects, n_windows) ndarray
    sign_trace_by_unit : dict[unit_tuple] -> (n_effects, n_windows) ndarray
                         (None if return_sign=False)
    effect_names       : list[str] pinned from the formula
    window_indices     : list[int] sorted window indices found in df
    """
    # Pin effect names from the formula -- stable across units even when statsmodels
    # drops a term on a degenerate cell.
    effect_names = []
    for k in range(1, len(anova_factors) + 1):
        for combo in combinations(anova_factors, k):
            effect_names.append(':'.join([f'C({c})' for c in combo]))

    window_indices = sorted(df[window_col].unique())
    n_windows = len(window_indices)
    n_eff = len(effect_names)

    f_by_unit = {}
    sign_by_unit = {} if return_sign else None

    for unit_vals, df_unit in df.groupby(list(unit_cols)):
        # Normalize single-column groupby to a tuple
        if not isinstance(unit_vals, tuple):
            unit_vals = (unit_vals,)
        f_trace = np.full((n_eff, n_windows), np.nan)
        sign_trace = np.full((n_eff, n_windows), np.nan) if return_sign else None
        for wi, w in enumerate(window_indices):
            df_w = df_unit[df_unit[window_col] == w]
            f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
            if f_dict is None:
                continue
            for ei, eff in enumerate(effect_names):
                f_trace[ei, wi] = f_dict.get(eff, np.nan)
                if return_sign:
                    sign_trace[ei, wi] = _signed_contrast_per_window(df_w, eff, anova_factors)
        f_by_unit[unit_vals] = f_trace
        if return_sign:
            sign_by_unit[unit_vals] = sign_trace

    return f_by_unit, sign_by_unit, effect_names, window_indices

def run_within_electrode_windowed_anova_cluster_correction(
    windowed_data, conditions_obj, anova_factors, rois, subjects,
    electrodes_per_subject_roi, times, window_size, step_size, sampling_rate,
    n_perm=1000, percentile=95, cluster_percentile=95,
    min_trials_per_cell=2,
    split_clusters_by_sign=True,
    seed=42, n_jobs=-1, verbose=True,
    save_dir=None, run_label=None,
):
    """Within-electrode windowed ANOVA + cluster correction.

    Differences from run_windowed_anova_cluster_correction:
      - ANOVA fit per (subject, electrode) using trials as observations -- no
        across-electrode aggregation.
      - Permutation null built per electrode (shuffle trial-level factor labels).
      - Cluster correction uses cluster EXTENT (n windows), matching the codebase
        convention. Optionally splits clusters at sign-flips of the contrast.

    Returns
    -------
    results : dict[roi][(subject, electrode)][effect] -> dict of arrays/lists
    window_centers : (n_windows,) array
    summary_df : tidy DataFrame, one row per (sub, elec, roi, effect, cluster)
    skipped : list of dicts -- electrodes skipped due to min_trials_per_cell
    """
    from pathlib import Path
    import pandas as pd
    from joblib import Parallel, delayed

    # === 1. Long-form dataframe (reuse existing builder) ===
    df = create_windowed_anova_dataframe(
        windowed_data, conditions_obj, rois, subjects,
        electrodes_per_subject_roi,
        times=times, window_size=window_size, step_size=step_size,
        sampling_rate=sampling_rate,
    )
    formula = 'Activity ~ ' + ' * '.join([f'C({f})' for f in anova_factors])

    # === 2. Observed pass via shared helper ===
    obs_f_by_unit, obs_sign_by_unit, effect_names, window_indices = \
        _fit_anova_per_window_per_unit(
            df, formula, anova_factors,
            unit_cols=('SubjectID', 'Electrode', 'ROI'),
            return_sign=split_clusters_by_sign,
        )
    n_windows = len(window_indices)
    n_effects = len(effect_names)

    # Window centers + window -> sample-index mapping (for expanding masks)
    window_centers = np.array(
        [df[df['WindowIndex'] == w]['WindowCenter'].iloc[0] for w in window_indices]
    )
    n_times = len(times)
    win_to_samples = []
    for w in window_indices:
        start_sample = int(w * step_size)
        end_sample = min(start_sample + window_size - 1, n_times - 1)
        win_to_samples.append((start_sample, end_sample))

    # === 3. Per-electrode worker (permutation + cluster correction) ===
    def _run_one_electrode(unit_key):
        sub, elec, roi = unit_key
        df_elec = df[(df['SubjectID'] == sub) &
                     (df['Electrode'] == elec) &
                     (df['ROI'] == roi)]

        # 3a. Min-trials gate on the first window (same trials per window)
        df_w0 = df_elec[df_elec['WindowIndex'] == window_indices[0]]
        cell_counts = df_w0.groupby(anova_factors).size()
        if (cell_counts < min_trials_per_cell).any():
            return {'skipped': True, 'reason': 'min_trials_per_cell',
                    'subject': sub, 'electrode': elec, 'roi': roi,
                    'cell_counts': cell_counts.to_dict()}

        # 3b. Observed F and sign from the shared-helper output
        observed_F = obs_f_by_unit[unit_key]                          # (n_eff, n_win)
        observed_sign = (obs_sign_by_unit[unit_key]
                         if split_clusters_by_sign and obs_sign_by_unit is not None
                         else None)

        # 3c. Permutation null (trials shuffled within electrode)
        elec_seed = (seed + hash(unit_key)) % (2**31 - 1)
        rng = np.random.RandomState(elec_seed)
        n_trials_elec = len(df_w0)
        null_F = np.full((n_perm, n_effects, n_windows), np.nan, dtype=np.float32)
        null_sign = (np.full((n_perm, n_effects, n_windows), np.nan, dtype=np.float32)
                     if split_clusters_by_sign else None)

        for pi in range(n_perm):
            perm = rng.permutation(n_trials_elec)
            for wi, w in enumerate(window_indices):
                df_w = df_elec[df_elec['WindowIndex'] == w].copy().reset_index(drop=True)
                for col in anova_factors:
                    df_w[col] = df_w[col].values[perm]
                f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
                if f_dict is None:
                    continue
                for ei, eff in enumerate(effect_names):
                    null_F[pi, ei, wi] = f_dict.get(eff, np.nan)
                    if split_clusters_by_sign:
                        null_sign[pi, ei, wi] = _signed_contrast_per_window(
                            df_w, eff, anova_factors)

        # 3d. Cluster correction per effect (extent statistic, sign-aware)
        per_effect = {}
        for ei, eff in enumerate(effect_names):
            obs = np.nan_to_num(observed_F[ei], nan=0.0)
            null = np.nan_to_num(null_F[:, ei, :], nan=0.0)
            obs_sg = observed_sign[ei] if observed_sign is not None else None
            null_sg = null_sign[:, ei, :] if null_sign is not None else None

            pointwise_thresh = np.nanpercentile(null, percentile, axis=0)
            raw_obs = _find_contiguous_runs(obs > pointwise_thresh)

            do_split = (split_clusters_by_sign and obs_sg is not None
                        and np.any(np.isfinite(obs_sg)))
            obs_sub = (_split_clusters_at_sign_flips(raw_obs, obs_sg)
                       if do_split
                       else [{'start': s, 'end': e, 'sign': 0} for s, e in raw_obs])

            null_max_extents = []
            for pi in range(null.shape[0]):
                raw_pi = _find_contiguous_runs(null[pi] > pointwise_thresh)
                sub_pi = (_split_clusters_at_sign_flips(raw_pi, null_sg[pi])
                          if do_split and null_sg is not None
                          else [{'start': s, 'end': e, 'sign': 0} for s, e in raw_pi])
                extents = [c['end'] - c['start'] + 1 for c in sub_pi] or [0]
                null_max_extents.append(max(extents))
            null_max_extents = np.array(null_max_extents)
            extent_thresh = np.percentile(null_max_extents, cluster_percentile)

            sig_subs = []
            for c in obs_sub:
                ext = c['end'] - c['start'] + 1
                p = float((null_max_extents >= ext).mean())
                if ext > extent_thresh:
                    sig_subs.append({**c, 'extent': ext, 'p_value': p})

            # The electrode's BEST cluster, whether or not it survived the extent
            # threshold. `cluster_p_value` below is the min over SURVIVING
            # clusters and falls back to exactly 1.0 when none survived, which
            # makes every non-surviving electrode indistinguishable -- any
            # across-electrode correction downstream (BH over electrodes, the
            # family a test that COUNTS electrodes needs) is then unable to rank
            # near-misses against true nulls. Recording the best cluster's p
            # unconditionally keeps that information.
            #
            # This has to happen here rather than being reconstructed later from
            # the saved .npz: `null_sign` exists in memory at this point, so the
            # sign-split can be applied symmetrically to the observed AND null
            # clusters. The .npz stores `signed_contrast` for the observed trace
            # only, so a post-hoc recompute must drop sign-splitting entirely to
            # avoid shortening observed clusters while leaving null ones intact.
            if obs_sub:
                best = max(obs_sub, key=lambda c: c['end'] - c['start'] + 1)
                best_extent = best['end'] - best['start'] + 1
                best_cluster_p = float((null_max_extents >= best_extent).mean())
                best_cluster_sign = int(best['sign'])
            else:
                best_extent, best_cluster_p, best_cluster_sign = 0, 1.0, 0

            window_mask = np.zeros(n_windows, dtype=bool)
            pos_window_mask = np.zeros(n_windows, dtype=bool)
            neg_window_mask = np.zeros(n_windows, dtype=bool)
            for c in sig_subs:
                window_mask[c['start']:c['end'] + 1] = True
                if c['sign'] > 0:
                    pos_window_mask[c['start']:c['end'] + 1] = True
                elif c['sign'] < 0:
                    neg_window_mask[c['start']:c['end'] + 1] = True

            def _expand(mask_w):
                m = np.zeros(n_times, dtype=bool)
                for s, e in _find_contiguous_runs(mask_w):
                    sa, _ = win_to_samples[s]
                    _, eb = win_to_samples[e]
                    m[sa:eb + 1] = True
                return m

            peak_idx = (int(np.nanargmax(observed_F[ei]))
                        if np.isfinite(observed_F[ei]).any() else -1)
            min_p = min((c['p_value'] for c in sig_subs), default=1.0)

            per_effect[eff] = {
                'observed_F': observed_F[ei],
                'signed_contrast': (obs_sg if obs_sg is not None
                                    else np.full(n_windows, np.nan)),
                'window_mask': window_mask,
                'pos_window_mask': pos_window_mask,
                'neg_window_mask': neg_window_mask,
                'sample_mask': _expand(window_mask),
                'pos_sample_mask': _expand(pos_window_mask),
                'neg_sample_mask': _expand(neg_window_mask),
                'sig_clusters_with_sign': sig_subs,
                'peak_F': (float(observed_F[ei, peak_idx])
                           if peak_idx >= 0 else np.nan),
                'peak_time': (float(window_centers[peak_idx])
                              if peak_idx >= 0 else np.nan),
                'cluster_p_value': min_p,
                'any_sig_cluster': bool(sig_subs),
                # graded, defined for every electrode (see the comment above)
                'best_cluster_p': best_cluster_p,
                'best_cluster_extent': best_extent,
                'best_cluster_sign': best_cluster_sign,
                'extent_threshold': float(extent_thresh),
            }

        # 3e. Stream-write per-electrode npz (memory: drop null_F after this)
        if save_dir and run_label:
            out_dir = Path(save_dir) / run_label / roi / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_dir / f'{elec}.npz',
                observed_F=observed_F,
                null_F=null_F,
                signed_contrast=(observed_sign if observed_sign is not None
                                 else np.full(observed_F.shape, np.nan)),
                window_mask=np.array([per_effect[e]['window_mask'] for e in effect_names]),
                pos_window_mask=np.array([per_effect[e]['pos_window_mask'] for e in effect_names]),
                neg_window_mask=np.array([per_effect[e]['neg_window_mask'] for e in effect_names]),
                effect_names=np.array(effect_names),
                window_centers=window_centers,
                best_cluster_p=np.array(
                    [per_effect[e]['best_cluster_p'] for e in effect_names]),
                best_cluster_extent=np.array(
                    [per_effect[e]['best_cluster_extent'] for e in effect_names]),
                best_cluster_sign=np.array(
                    [per_effect[e]['best_cluster_sign'] for e in effect_names]),
            )

        return {'skipped': False,
                'subject': sub, 'electrode': elec, 'roi': roi,
                'per_effect': per_effect}

    # === 4. Fan out over electrodes ===
    tasks = list(obs_f_by_unit.keys())
    if verbose:
        print(f"[within-elec] {len(tasks)} (subject, electrode, roi) tasks; "
              f"{n_perm} perms each on {n_jobs} workers")
    raw_results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_run_one_electrode)(k) for k in tasks
    )

    # === 5. Aggregate results + tidy summary df + skipped log ===
    results = {}
    summary_rows = []
    skipped = []
    for r in raw_results:
        if r['skipped']:
            skipped.append(r); continue
        roi = r['roi']
        results.setdefault(roi, {})[(r['subject'], r['electrode'])] = r['per_effect']
        for eff, info in r['per_effect'].items():
            if info['sig_clusters_with_sign']:
                for ci, c in enumerate(info['sig_clusters_with_sign']):
                    summary_rows.append({
                        'subject': r['subject'], 'electrode': r['electrode'], 'roi': roi,
                        'effect': eff, 'cluster_idx': ci,
                        'sign': c['sign'], 'extent_windows': c['extent'],
                        'cluster_onset': float(window_centers[c['start']]),
                        'cluster_offset': float(window_centers[c['end']]),
                        'cluster_p_value': c['p_value'],
                        'peak_F': info['peak_F'], 'peak_time': info['peak_time'],
                        'best_cluster_p': info['best_cluster_p'],
                        'best_cluster_extent': info['best_cluster_extent'],
                        'best_cluster_sign': info['best_cluster_sign'],
                    })
            else:
                summary_rows.append({
                    'subject': r['subject'], 'electrode': r['electrode'], 'roi': roi,
                    'effect': eff, 'cluster_idx': -1,
                    'sign': 0, 'extent_windows': 0,
                    'cluster_onset': np.nan, 'cluster_offset': np.nan,
                    'cluster_p_value': 1.0,
                    'peak_F': info['peak_F'], 'peak_time': info['peak_time'],
                    # NOT 1.0 by fiat -- the real p of this electrode's largest
                    # sub-threshold cluster, so downstream ranking is possible.
                    'best_cluster_p': info['best_cluster_p'],
                    'best_cluster_extent': info['best_cluster_extent'],
                    'best_cluster_sign': info['best_cluster_sign'],
                })
    summary_df = pd.DataFrame(summary_rows)

    # === 6. BH-FDR across electrodes (per roi, per effect) ===
    if not summary_df.empty:
        # Assign in place rather than via groupby().apply(): from pandas 2.2 the
        # grouping columns are excluded from the frame handed to the callable, so
        # the old version silently returned a summary_df with no 'roi'/'effect'
        # columns and every downstream groupby on them raised KeyError.
        summary_df['cluster_p_fdr'] = np.nan
        summary_df['sig_after_fdr'] = False
        for _, idx in summary_df.groupby(['roi', 'effect']).groups.items():
            p = summary_df.loc[idx, 'cluster_p_value'].fillna(1.0).to_numpy()
            reject, p_adj, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
            summary_df.loc[idx, 'cluster_p_fdr'] = p_adj
            summary_df.loc[idx, 'sig_after_fdr'] = reject

    # === 7. Disk writes: summary.csv, skipped.csv, sig_electrodes_*.json,
    #        significant_effects_structure.json (legacy compat) ===
    if save_dir and run_label:
        run_dir = Path(save_dir) / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(run_dir / 'summary.csv', index=False)
        pd.DataFrame(skipped).to_csv(run_dir / 'skipped.csv', index=False)

        if not summary_df.empty:
            sig = summary_df[summary_df['sig_after_fdr']]
            sig_by_re = {}
            for (roi, eff), grp in sig.groupby(['roi', 'effect']):
                sig_by_re.setdefault(eff, {})[roi] = (
                    grp[['subject', 'electrode']].drop_duplicates().to_dict('records'))
            for eff, by_roi in sig_by_re.items():
                safe_eff = eff.replace(':', '_x_').replace('C(', '').replace(')', '')
                with open(run_dir / f'sig_electrodes_{safe_eff}.json', 'w') as f:
                    json.dump(by_roi, f, indent=2)

        # Legacy significant_effects_structure.json
        sig_struct = {}
        for roi, by_elec in results.items():
            for (sub, elec), per_effect in by_elec.items():
                for eff, info in per_effect.items():
                    if not info['any_sig_cluster']:
                        continue
                    cluster_p = info['cluster_p_value']
                    for wi, in_cluster in enumerate(info['window_mask']):
                        if not in_cluster:
                            continue
                        tw = float(window_centers[wi])
                        sig_struct.setdefault(sub, {}) \
                                  .setdefault(elec, {}) \
                                  .setdefault(roi, {}) \
                                  .setdefault(tw, {})[eff] = cluster_p
        with open(run_dir / 'significant_effects_structure.json', 'w') as f:
            json.dump(sig_struct, f, indent=2)

        # Run manifest. The window geometry in particular has to survive the run:
        # any analysis that wants to be comparable to this one (e.g. the
        # continuous segregation correlation, which is estimated independently
        # but must cover the SAME stretch of time to be interpretable alongside
        # these labels) needs to know the exact extent the windows tiled, and
        # `window_centers` alone understates it by half a window at each end.
        with open(run_dir / 'run_config.json', 'w') as f:
            json.dump({
                'window_centers': [float(w) for w in window_centers],
                'window_size': int(window_size),
                'step_size': int(step_size),
                'sampling_rate': float(sampling_rate),
                'n_windows': int(n_windows),
                'times_min': float(times[0]),
                'times_max': float(times[-1]),
                # the extent actually covered by the windows, in seconds
                'analysis_tmin': float(times[win_to_samples[0][0]]),
                'analysis_tmax': float(times[win_to_samples[-1][1]]),
                'anova_factors': list(anova_factors),
                'effect_names': list(effect_names),
                'rois': list(rois),
                'subjects': list(subjects),
                'n_perm': int(n_perm),
                'percentile': float(percentile),
                'cluster_percentile': float(cluster_percentile),
                'min_trials_per_cell': int(min_trials_per_cell),
                'split_clusters_by_sign': bool(split_clusters_by_sign),
                'seed': int(seed),
            }, f, indent=2)

    return results, window_centers, summary_df, skipped

def load_significant_electrodes(within_elec_anova_run_dir, roi=None, effect=None,
                                use_fdr=True, p_thresh=0.05):
    """Filter a within_elec_anova run to a list of (subject, electrode) tuples.

    Prefers significant_effects_structure.json (legacy format) when present;
    falls back to summary.csv with FDR otherwise.
    """
    import pandas as pd
    from pathlib import Path
    run_dir = Path(within_elec_anova_run_dir)
    sig_json = run_dir / 'significant_effects_structure.json'

    if sig_json.exists():
        struct = json.load(open(sig_json))
        out = []
        for sub, by_elec in struct.items():
            for elec, by_roi in by_elec.items():
                if roi is not None and roi not in by_roi:
                    continue
                rois_to_walk = [roi] if roi else list(by_roi.keys())
                for r in rois_to_walk:
                    by_tw = by_roi.get(r, {})
                    matched = any(
                        (effs.get(effect, 1.0) < p_thresh) if effect
                        else any(p < p_thresh for p in effs.values())
                        for _, effs in by_tw.items()
                    )
                    if matched:
                        out.append((sub, elec))
        return sorted(set(out))

    df = pd.read_csv(run_dir / 'summary.csv')
    if roi is not None:
        df = df[df['roi'] == roi]
    if effect is not None:
        df = df[df['effect'] == effect]
    df = df[df['sig_after_fdr']] if use_fdr else df[df['cluster_p_value'] < p_thresh]
    return list(df[['subject', 'electrode']].drop_duplicates()
                  .itertuples(index=False, name=None))

def process_windowed_data_for_anova(subjects_mne_objects, condition_names, rois, subjects,
                                    electrodes_per_subject_roi, window_size=64,
                                    step_size=16, sampling_rate=256):
    """
    Process data with sliding windows for ANOVA analysis.

    Parameters:
    -----------
    window_size : int or None
        Size of window in samples. If None, uses full epoch.
    step_size : int
        Step size for sliding window in samples.
    Slide a window over each subject's trial-level epoch data and average within
    each window to produce per-window per-channel scalars suitable for ANOVA.

    For every (condition, ROI, subject) triple that has at least one electrode in
    the ROI, this function:
      1. Picks the subject's epochs for the given condition.
      2. Restricts channels to ``electrodes_per_subject_roi[roi][subject]``.
      3. Slides a window of ``window_size`` samples with stride ``step_size`` over
         the time axis using ``general_utils.windower``.
      4. Averages within each window (collapsing the time axis inside a window)
         to produce a (n_trials, n_windows, n_channels) array.

    Subjects with **no** electrodes for the ROI are silently skipped (the output
    list for that ROI simply omits them). The downstream
    ``create_windowed_anova_dataframe`` reproduces this same skipping logic so
    list indices line up with the right subject + electrodes — see
    ``create_windowed_anova_dataframe`` for the matching invariant.

    Parameters
    ----------
    subjects_mne_objects : dict
        Nested dict ``[subject][condition_name][mne_object_type]``; we read
        ``[sub][cond]['HG_ev1_power_rescaled']``.
    condition_names : list of str
        Condition keys (e.g. ``['Stimulus_i25s25', ...]``).
    rois : list of str
        ROI names (e.g. ``['lpfc', 'occ', ...]``).
    subjects : list of str
        Subject IDs to iterate. Must be the same ordered list passed to
        ``create_windowed_anova_dataframe`` later, since that function relies on
        the same subject-skipping order.
    electrodes_per_subject_roi : dict
        ``[roi][subject] -> list of electrode names``.
    window_size : int or None, default 64
        Size of window in samples. If None, ``windower`` falls back to a single
        full-epoch window.
    step_size : int, default 16
        Stride between successive windows in samples.
    sampling_rate : int, default 256
        Currently unused (kept for API symmetry with the dataframe builder).

    Returns
    -------
    windowed_data : dict
        ``{condition_name: {roi: list_of_arrays}}`` where each list entry is a
        (n_trials, n_windows, n_channels) ndarray, one per subject that had
        electrodes for that ROI, in the same order as ``subjects``.
    """
    from src.analysis.utils.general_utils import windower   # lazy: pulls in `ieeg`

    windowed_data = {}

    for condition_name in condition_names:
        windowed_data[condition_name] = {}

        for roi in rois:
            roi_data_list = []

            for sub in subjects:
                electrodes = electrodes_per_subject_roi[roi].get(sub, [])
                if not electrodes:
                    continue

                # Get epochs for this condition
                epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy()
                epochs = epochs.pick_channels(electrodes)

                # Get data: (n_trials, n_channels, n_times)
                data = epochs.get_data()

                # Apply windowing to each channel and trial
                # windower expects data with time axis last by default
                windowed_trials = []
                for trial in data:
                    trial_windowed = windower(trial, window_size=window_size,
                                             axis=-1, step_size=step_size, insert_at=0)
                    # Shape: (n_windows, n_channels, window_size)
                    windowed_trials.append(trial_windowed)

                windowed_trials = np.array(windowed_trials)
                # Shape: (n_trials, n_windows, n_channels, window_size)

                # Average within each window
                windowed_avg = np.mean(windowed_trials, axis=-1)
                # Shape: (n_trials, n_windows, n_channels)

                roi_data_list.append(windowed_avg)

            windowed_data[condition_name][roi] = roi_data_list

    return windowed_data

def create_windowed_anova_dataframe(windowed_data, conditions, rois, subjects,
                                    electrodes_per_subject_roi, times,
                                    window_size=None, step_size=1, sampling_rate=256):
    """
    Create DataFrame for windowed ANOVA analysis.
    Flatten the nested ``windowed_data`` structure into a long DataFrame suitable
    for per-window OLS / ANOVA fits.

    Each row is one observation: ``(subject, electrode, ROI, trial, window) ->
    Activity``, with the factor columns from ``conditions[cond]`` attached
    (e.g. ``congruency``, ``incongruentProportion``, ``switchType``,
    ``switchProportion``). ``BIDS_events`` is included verbatim; downstream
    formula-builders should drop it.

    Window centers are precomputed once (in seconds) and attached to every row
    of the corresponding window so the resulting DataFrame can be plotted /
    grouped by ``WindowCenter`` directly.

    Parameters
    ----------
    windowed_data : dict
        Output of :func:`process_windowed_data_for_anova`.
    conditions : dict
        Mapping ``{condition_name: condition_parameters}`` from
        ``experiment_conditions``. The ``condition_parameters`` dict is merged
        into every row so factor columns become available for OLS.
    rois : list of str
    electrodes_per_subject_roi : dict
        ``[roi][subject] -> list of electrode names``.
    times : array-like
        The full per-sample time vector of the epoch (used to compute window
        centers).
    window_size : int or None
        Window length in samples; when None, a single window centered at the
        epoch midpoint is produced.
    step_size : int, default 1
    sampling_rate : int, default 256

    Returns
    -------
    df : pandas.DataFrame
        Columns: ``SubjectID``, ``Electrode``, ``ROI``, ``WindowCenter``,
        ``WindowIndex``, ``Trial``, ``Activity``, plus all keys from
        ``condition_parameters`` (factor columns + ``BIDS_events``).

    Notes
    -----
    Cross-subject channel-name collisions: ``combine_single_channel_evokeds``
    renames duplicates with running suffixes (``LFMM9-0``, ``LFMM9-1``), but the
    ``Electrode`` column here stores the **un-suffixed** original name. Always
    group by ``(SubjectID, Electrode)`` together if you need a unique key.
    """
    data_for_anova = []

    # Window centers
    if window_size is not None:
        n_windows = (len(times) - window_size) // step_size + 1
        window_centers = []
        for i in range(n_windows):
            start_idx = i * step_size
            center_idx = start_idx + window_size // 2
            if center_idx < len(times):
                window_centers.append(times[center_idx])
            else:
                window_centers.append(times[-1])
    else:
        window_centers = [np.mean(times)]

    for condition_name, condition_parameters in conditions.items():
        for roi in rois:
            roi_list = windowed_data.get(condition_name, {}).get(roi, [])
            sub_idx = 0
            for sub in subjects:
                electrodes = electrodes_per_subject_roi[roi].get(sub, [])
                if not electrodes:
                    continue  # process_windowed_data_for_anova also skipped
                if sub_idx >= len(roi_list):
                    break
                subject_data = roi_list[sub_idx]
                sub_idx += 1

                # Defensive: clamp electrode count to actual data shape.
                n_chans_data = subject_data.shape[2]
                n_chans = min(len(electrodes), n_chans_data)
                if n_chans_data != len(electrodes):
                    print(f"[create_windowed_anova_dataframe] WARNING: "
                          f"{sub}/{roi}: electrode list has {len(electrodes)} but "
                          f"data has {n_chans_data} channels; using first {n_chans}.")

                for trial_idx in range(subject_data.shape[0]):
                    for window_idx in range(subject_data.shape[1]):
                        for electrode_idx in range(n_chans):
                            electrode_name = electrodes[electrode_idx]
                            activity = subject_data[trial_idx, window_idx, electrode_idx]
                            data_dict = {
                                'SubjectID': sub,
                                'Electrode': electrode_name,
                                'ROI': roi,
                                'WindowCenter': window_centers[window_idx],
                                'WindowIndex': window_idx,
                                'Trial': trial_idx + 1,
                                'Activity': activity,
                            }
                            data_dict.update(condition_parameters)
                            data_for_anova.append(data_dict)

    return pd.DataFrame(data_for_anova)

def perform_windowed_anova(df, conditions, rois, save_dir, save_name,
                           anova_type='within_electrode'):
    """
    Fit a Type II OLS ANOVA at every time window and persist the results.

    The OLS formula is built dynamically from the keys of any single condition
    in ``conditions`` (excluding ``BIDS_events``), as
    ``Activity ~ C(f1) * C(f2) * ... * C(fn)`` — i.e. all factors plus all
    interactions. This implicitly assumes that every condition shares the same
    factor-key set; mixing condition sets with different keys here will produce
    misleading formulas.

    Two analysis modes:

    - ``'within_electrode'``: per (subject, electrode, ROI), fit OLS on the
      trial-level rows for that electrode in that window. Stores only the
      effects with uncorrected p < 0.05.
    - ``'across_electrode'``: per ROI, average activity within each
      (subject, electrode, factors) cell first, then fit OLS treating each
      electrode-cell as one observation. Stores the full ANOVA table per ROI
      per window.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`create_windowed_anova_dataframe`.
    conditions : dict
        ``{condition_name: condition_parameters}``; only the first entry's keys
        are used to build the formula.
    rois : list of str
        Used only in the ``'across_electrode'`` branch.
    save_dir : str
        Directory to write the JSON output file.
    save_name : str
        Stem for the output filename
        (``{save_name}_windowed_anova_{anova_type}.json``).
    anova_type : {'within_electrode', 'across_electrode'}, default 'within_electrode'

    Returns
    -------
    results_by_window : dict
        - within_electrode: ``{window_center -> list of {SubjectID, Electrode,
          ROI, Effects (DataFrame of significant effects)}}``
        - across_electrode: ``{window_center -> {roi: anova_lm_table}}``

    Notes
    -----
    BUG (across_electrode branch): the inner loop overwrites
    ``results_by_window[window_center]`` for each ROI, so only the last ROI's
    table is preserved per window. To persist all ROIs, that branch should
    accumulate into ``{window_center: {roi: ...}}`` instead of reassigning.

    Output JSON serializes via ``str(result)``, which is lossy and only useful
    for human inspection — re-parsing the JSON back into DataFrames is not
    supported.
    """
    results_by_window = {}

    # Get unique window indices
    window_indices = df['WindowIndex'].unique()

    for window_idx in window_indices:
        df_window = df[df['WindowIndex'] == window_idx]
        window_center = df_window['WindowCenter'].iloc[0]

        if anova_type == 'within_electrode':
            # Perform within-electrode ANOVA for this window
            results = []

            for subject_id in df_window['SubjectID'].unique():
                for electrode in df_window['Electrode'].unique():
                    for roi in df_window['ROI'].unique():
                        df_filtered = df_window[
                            (df_window['SubjectID'] == subject_id) &
                            (df_window['Electrode'] == electrode) &
                            (df_window['ROI'] == roi)
                        ]

                        if df_filtered.empty or len(df_filtered) < 2:
                            continue

                        # Build formula
                        condition_keys = [k for k in conditions[next(iter(conditions))].keys()
                                        if k != 'BIDS_events']
                        formula = 'Activity ~ ' + ' * '.join([f'C({k})' for k in condition_keys])

                        try:
                            model = ols(formula, data=df_filtered).fit()
                            anova_results = anova_lm(model, typ=2)

                            # Store significant effects
                            sig_effects = anova_results[anova_results['PR(>F)'] < 0.05]
                            if not sig_effects.empty:
                                results.append({
                                    'SubjectID': subject_id,
                                    'Electrode': electrode,
                                    'ROI': roi,
                                    'Effects': sig_effects
                                })
                        except:
                            continue

            results_by_window[window_center] = results

        elif anova_type == 'across_electrode':
            # Perform across-electrode ANOVA for this window.
            # Accumulate per-ROI results under this window so multiple ROIs
            # don't overwrite each other.
            if window_center not in results_by_window:
                results_by_window[window_center] = {}

            # Build the factor-key list once per window — exclude BIDS_events
            # because (a) it isn't an ANOVA factor and (b) its values are lists,
            # which are unhashable and would break the groupby below.
            condition_keys = [k for k in conditions[next(iter(conditions))].keys()
                              if k != 'BIDS_events']
            formula = 'Activity ~ ' + ' * '.join([f'C({k})' for k in condition_keys])

            for roi in rois:
                df_roi = df_window[df_window['ROI'] == roi]
                if df_roi.empty:
                    continue

                # Average across trials for each electrode-cell
                df_averaged = df_roi.groupby(
                    ['SubjectID', 'Electrode', 'ROI'] + condition_keys
                )['Activity'].mean().reset_index()

                model = ols(formula, data=df_averaged).fit()
                anova_results = anova_lm(model, typ=2)

                results_by_window[window_center][roi] = anova_results

    # Save results - TODO: this only works for across_electrode right now, need to handle within_electrode too. Check the way I used to store these results for plotting.
    rows = []

    for window_center, roi_dict in results_by_window.items():
        for roi, anova_table in roi_dict.items():
            df_out = anova_table.reset_index().rename(columns={'index': 'term'})
            df_out['window_center'] = window_center
            df_out['roi'] = roi
            rows.append(df_out)

    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(save_dir, f'{save_name}_windowed_anova_{anova_type}.csv'),
        index=False
    )

    return results_by_window

def apply_fdr_correction_to_windowed_results(results_by_window, alpha=0.05):
    """
    Apply FDR correction across all windows and effects.
    """
    # Collect all p-values
    all_pvalues = []
    pvalue_info = []  # Track where each p-value comes from

    for window, results in results_by_window.items():
        if isinstance(results, list):  # within-electrode results
            for result in results:
                effects_df = result.get('Effects', pd.DataFrame())
                if not effects_df.empty:
                    for idx, row in effects_df.iterrows():
                        all_pvalues.append(row['PR(>F)'])
                        pvalue_info.append({
                            'window': window,
                            'subject': result['SubjectID'],
                            'electrode': result['Electrode'],
                            'effect': idx
                        })
        else:  # across-electrode results
            for roi, anova_table in results.items():
                for idx, row in anova_table.iterrows():
                    all_pvalues.append(row['PR(>F)'])
                    pvalue_info.append({
                        'window': window,
                        'roi': roi,
                        'effect': idx
                    })

    # Apply FDR correction
    if all_pvalues:
        rejected, corrected_pvalues, _, _ = multipletests(
            all_pvalues, alpha=alpha, method='fdr_bh'
        )

        # Create corrected results structure
        corrected_results = {}
        for i, info in enumerate(pvalue_info):
            window = info['window']
            if window not in corrected_results:
                corrected_results[window] = []

            if rejected[i]:  # Only keep significant after correction
                info['corrected_pvalue'] = corrected_pvalues[i]
                info['original_pvalue'] = all_pvalues[i]
                corrected_results[window].append(info)

    return corrected_results

def _fit_anova_one_window(df_window, formula, factor_columns):
    """Fit OLS at a single window, return dict[effect_name] -> F-stat.

    `df_window` already aggregated to one row per (electrode × cell).
    Effect names match those returned by anova_lm (e.g., "C(congruency)",
    "C(congruency):C(incongruentProportion)", ...).
    """
    try:
        model = ols(formula, data=df_window).fit()
        table = anova_lm(model, typ=2)
    except Exception:
        return None
    return {idx: row['F'] for idx, row in table.iterrows() if idx != 'Residual'}


def _shuffle_labels_within_electrode(df_one_window, factor_columns, rng):
    """Return a copy of df_one_window with factor columns permuted within each
    electrode (so each electrode keeps its 16 cell means but their factor
    assignment is randomized as a block)."""
    df = df_one_window.copy()
    for (sub, elec), idxs in df.groupby(['SubjectID', 'Electrode']).groups.items():
        idxs = np.asarray(idxs)
        perm = rng.permutation(len(idxs))
        for col in factor_columns:
            df.loc[idxs, col] = df.loc[idxs[perm], col].values
    return df

def run_windowed_anova_cluster_correction(
    windowed_data, conditions_obj, anova_factors, rois, subjects,
    electrodes_per_subject_roi, times, window_size, step_size, sampling_rate,
    n_perm=1000, percentile=95, cluster_percentile=95,
    split_clusters_by_sign=True,
    cluster_stat='extent',
    seed=42, n_jobs=-1, verbose=True,
):
    """Run windowed factorial ANOVAs with permutation-based cluster correction.

    For each ROI and time window, this function fits a full-factorial ordinary
    least-squares ANOVA across electrodes. A permutation null distribution is
    generated by shuffling factor labels within each subject-electrode pair,
    using the same shuffle across all windows in a permutation.

    Candidate clusters are formed from consecutive windows whose observed
    F-statistics exceed a window-specific percentile of the permutation null.
    Clusters are corrected using their extent in windows relative to the
    permutation distribution of maximum cluster extents. When requested and
    available, clusters are additionally split at sign changes in the effect's
    signed contrast.

    Parameters
    ----------
    windowed_data : dict
        Windowed data organized as
        ``windowed_data[condition_name][roi][subject_index]``. Each subject
        entry must be an array with shape
        ``(n_trials, n_windows, n_channels)``. Typically the output of
        ``process_windowed_data_for_anova``.
    conditions_obj : dict
        Mapping from condition names in ``windowed_data`` to dictionaries of
        factor values. Each factor named in ``anova_factors`` must be present
        for every included condition.
    anova_factors : sequence of str
        Factor names used to construct the full-factorial ANOVA formula,
        including all main effects and interactions. For example,
        ``['congruency', 'incongruentProportion', 'switchType',
        'switchProportion']``.
    rois : sequence of str
        ROI names to analyze. Names must correspond to keys in the ROI level of
        ``windowed_data``.
    subjects : sequence
        Subject identifiers ordered consistently with the per-subject arrays
        stored in ``windowed_data``.
    electrodes_per_subject_roi : dict
        Electrode metadata used to associate the channel dimension of each
        subject's ROI data with electrode identifiers. Its structure must match
        that expected by ``create_windowed_anova_dataframe``.
    times : array-like of float
        One-dimensional time axis for the original samples. Its length
        determines the length of the returned sample-level masks.
    window_size : int
        Window length in samples.
    step_size : int
        Number of samples between the starts of consecutive windows.
    sampling_rate : float
        Sampling rate of the original data in Hz, used when constructing the
        windowed ANOVA dataframe and window-center time points.
    n_perm : int, default=1000
        Number of label permutations used to estimate the null distributions.
    percentile : float, default=95
        Percentile of the per-window permutation distribution used as the
        cluster-forming threshold. For example, 95 corresponds to a nominal
        pointwise alpha of 0.05.
    cluster_percentile : float, default=95
        Percentile of the permutation distribution of maximum cluster extents
        used as the cluster-level significance threshold. For example, 95
        corresponds to a nominal cluster-corrected alpha of 0.05.
    split_clusters_by_sign : bool, default=True
        If True, split candidate clusters wherever the signed contrast changes
        sign. Sign splitting is applied only when a finite signed contrast can
        be computed for the effect; otherwise, clusters are treated as
        unsigned.
    cluster_stat : {'extent', 'mass'}, default='extent'
        Statistic used to score a candidate cluster. ``'extent'`` counts
        suprathreshold windows and is magnitude-blind — a long weak cluster
        outscores a short strong one, and with ``window_size=64``/``step_size=16``
        the windows overlap 75%, so one 250 ms effect already spans four of
        them. ``'mass'`` sums each window's excess of F over the threshold.
        Default stays ``'extent'`` so existing runs reproduce; ``'mass'`` is the
        better choice for new ones. Note this is not what produces a
        baseline-spanning cluster — a sustained offset wins under either.
    seed : int, default=42
        Seed for generating reproducible permutation seeds.
    n_jobs : int, default=-1
        Number of parallel workers passed to ``joblib.Parallel``. A value of
        ``-1`` uses all available processors.
    verbose : bool, default=True
        If True, print the ANOVA formula and analysis progress and enable
        joblib progress output.

    Returns
    -------
    results : dict
        Nested mapping ``results[roi][effect_name]``. Each effect dictionary
        contains:

        ``observed_F`` : numpy.ndarray, shape (n_windows,)
            Observed F-statistic at each window.
        ``null_F`` : numpy.ndarray, shape (n_perm, n_windows)
            Permutation null F-statistics.
        ``signed_contrast`` : numpy.ndarray, shape (n_windows,)
            Signed contrast used for sign-aware cluster splitting. Values may
            be NaN when a signed contrast is unavailable for an effect.
        ``window_mask`` : numpy.ndarray of bool, shape (n_windows,)
            Union of all significant positive, negative, and unsigned clusters.
        ``pos_window_mask`` : numpy.ndarray of bool, shape (n_windows,)
            Windows belonging to significant positive-sign clusters.
        ``neg_window_mask`` : numpy.ndarray of bool, shape (n_windows,)
            Windows belonging to significant negative-sign clusters.
        ``sample_mask`` : numpy.ndarray of bool, shape (len(times),)
            Union window mask expanded onto the original sample time axis.
        ``pos_sample_mask`` : numpy.ndarray of bool, shape (len(times),)
            Positive window mask expanded onto the original sample time axis.
        ``neg_sample_mask`` : numpy.ndarray of bool, shape (len(times),)
            Negative window mask expanded onto the original sample time axis.
        ``sig_clusters_with_sign`` : list of dict
            Significant clusters. Each dictionary contains inclusive ``start``
            and ``end`` window indices, ``sign`` (positive, negative, or zero),
            ``extent`` in windows, ``cluster_stat`` (the value actually
            thresholded, per ``cluster_stat``), and a permutation-based
            ``p_value``.

        ROIs with no usable data or no successfully fitted windows are omitted.
    window_centers : numpy.ndarray, shape (n_windows,)
        Time coordinate of the center of each analyzed window.

    Notes
    -----
    A cluster is retained when its statistic (see ``cluster_stat``) is strictly
    greater than the selected percentile of the permutation distribution of
    per-permutation maxima.

    A significant cluster that begins before stimulus onset is a red flag, not a
    result: the pre-stimulus baseline contains no evoked activity, so a
    difference there reflects a tonic or compositional difference between cells.
    Because such an offset is sustained, it produces long suprathreshold runs
    and wins the cluster test under either ``cluster_stat``. Inspect
    ``signed_contrast`` before interpreting: an evoked interaction starts near
    zero in the baseline and grows after onset, whereas a confound shows up as a
    roughly constant offset across the whole epoch.

    Cluster tests do not localize. A cluster spanning -0.9 to +0.9 s does not
    license the claim that the effect begins at -0.9 s (Sassenhagen &
    Draschkow, 2019).
    """
    if cluster_stat not in ('extent', 'mass'):
        raise ValueError(f"cluster_stat must be 'extent' or 'mass', got {cluster_stat!r}")

    rng_master = np.random.RandomState(seed)

    # Build long dataframe (re-uses your existing function)
    df = create_windowed_anova_dataframe(
        windowed_data, conditions_obj, rois, subjects,
        electrodes_per_subject_roi,
        times=times, window_size=window_size, step_size=step_size,
        sampling_rate=sampling_rate,
    )

    # OLS formula over the requested factors
    formula = 'Activity ~ ' + ' * '.join([f'C({f})' for f in anova_factors])
    if verbose:
        print(f"[anova-cluster] Formula: {formula}")

    # Pre-compute window centers (n_windows,) and the corresponding sample mapping
    window_indices = sorted(df['WindowIndex'].unique())
    n_windows = len(window_indices)
    window_centers = np.array(
        [df[df['WindowIndex'] == w]['WindowCenter'].iloc[0] for w in window_indices]
    )

    # Map each window to a (start_sample, end_sample) range, used to expand
    # window-level clusters back to the full sampling-rate mask for plotting.
    n_times = len(times)
    win_to_samples = []
    for w in window_indices:
        start_sample = int(w * step_size)
        end_sample = min(start_sample + window_size - 1, n_times - 1)
        win_to_samples.append((start_sample, end_sample))

    results = {}

    for roi in rois:
        if verbose:
            print(f"[anova-cluster] === ROI: {roi} ===")
        df_roi = df[df['ROI'] == roi]
        if df_roi.empty:
            continue

        # Aggregate to electrode × window × cell (across-electrode ANOVA).
        #
        # NOTE: this is a trial-count-weighted mean. It equals an unweighted
        # (estimated) marginal mean only when every cell of the `anova_factors`
        # design holds the same mixture of whatever it collapses over -- i.e.
        # when `conditions_obj` keeps those cells separate AND each is a term in
        # the model. Run the proportion interactions off a conditions_obj whose
        # cells are one block type each (the *_by_block_conditions sets) with the
        # full-factorial `anova_factors`. A pre-pooled 4-condition set mixes
        # blocks 3:1 vs 1:3 across the cells being contrasted, which puts a
        # block-level offset into the contrast at every timepoint, baseline
        # included.
        group_cols = ['SubjectID', 'Electrode', 'WindowIndex', 'WindowCenter'] + anova_factors
        df_agg = df_roi.groupby(group_cols, as_index=False)['Activity'].mean()

        # === Observed F per effect per window ===
        observed_per_window = {}  # window_idx -> dict(effect -> F)
        for w in window_indices:
            df_w = df_agg[df_agg['WindowIndex'] == w]
            f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
            observed_per_window[w] = f_dict

        # All effects encountered (use first non-None window)
        effect_names = None
        for w in window_indices:
            if observed_per_window[w] is not None:
                effect_names = list(observed_per_window[w].keys())
                break
        if effect_names is None:
            print(f"[anova-cluster] No usable windows for ROI {roi}; skipping.")
            continue

        observed_F = np.full((len(effect_names), n_windows), np.nan)
        for wi, w in enumerate(window_indices):
            f_dict = observed_per_window[w]
            if f_dict is None:
                continue
            for ei, eff in enumerate(effect_names):
                observed_F[ei, wi] = f_dict.get(eff, np.nan)

        # === Compute signed contrast trace per effect alongside observed F ===
        observed_sign = np.full((len(effect_names), n_windows), np.nan)
        if split_clusters_by_sign:
            for wi, w in enumerate(window_indices):
                df_w = df_agg[df_agg['WindowIndex'] == w]
                for ei, eff in enumerate(effect_names):
                    observed_sign[ei, wi] = _signed_contrast_per_window(df_w, eff, anova_factors)

        # === Permutation null ===
        # We shuffle factor labels per electrode once per perm (same shuffle for
        # all windows), since each electrode contributes independent rows per
        # window.
        #
        # This treats every electrode as its own exchangeable unit -- the
        # "super-subject" assumption used throughout this codebase, decoding
        # included. It is a deliberate modelling choice: it assumes no structure
        # is shared across the electrodes of one subject. Where that assumption
        # fails, this null is narrower than the data warrants.
        # Per-perm work: re-fit OLS at every window.
        seeds = rng_master.randint(0, 2**31 - 1, size=n_perm)

        def _one_perm(perm_seed):
            rng = np.random.RandomState(perm_seed)
            shuffle_map = {}
            for (sub, elec), grp in df_agg[df_agg['WindowIndex'] == window_indices[0]] \
                    .groupby(['SubjectID', 'Electrode']):
                n_cells = len(grp)
                shuffle_map[(sub, elec)] = rng.permutation(n_cells)
            null_F_perm = np.full((len(effect_names), n_windows), np.nan)
            null_sign_perm = np.full((len(effect_names), n_windows), np.nan)   # === NEW ===
            for wi, w in enumerate(window_indices):
                df_w = df_agg[df_agg['WindowIndex'] == w].copy().reset_index(drop=True)
                for (sub, elec), idxs in df_w.groupby(['SubjectID', 'Electrode']).groups.items():
                    perm = shuffle_map.get((sub, elec))
                    if perm is None or len(perm) != len(idxs):
                        continue
                    idxs = np.asarray(idxs)
                    for col in anova_factors:
                        df_w.loc[idxs, col] = df_w.loc[idxs[perm], col].values
                f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
                if f_dict is None:
                    continue
                for ei, eff in enumerate(effect_names):
                    null_F_perm[ei, wi] = f_dict.get(eff, np.nan)
                    if split_clusters_by_sign:
                        null_sign_perm[ei, wi] = _signed_contrast_per_window(df_w, eff, anova_factors)
            return null_F_perm, null_sign_perm                                  # === CHANGED ===

        # === unpack tuples from joblib ===
        perm_results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_one_perm)(s) for s in seeds
        )
        null_F   = np.stack([r[0] for r in perm_results], axis=0)   # (n_perm, n_effects, n_windows)
        null_sign = np.stack([r[1] for r in perm_results], axis=0)  # same shape

        # === Per-effect sign-aware cluster correction ===
        results[roi] = {}
        for ei, eff in enumerate(effect_names):
            obs    = np.nan_to_num(observed_F[ei], nan=0.0)
            null   = np.nan_to_num(null_F[:, ei, :], nan=0.0)
            obs_sg = observed_sign[ei]                  # may be all NaN for 3-way+
            null_sg = null_sign[:, ei, :] if split_clusters_by_sign else None

            # 1. Per-window pointwise threshold from the null distribution
            pointwise_thresh = np.nanpercentile(null, percentile, axis=0)  # (n_windows,)

            # 2. Raw observed clusters (contiguous windows above threshold)
            above_thresh = obs > pointwise_thresh
            raw_obs_clusters = _find_contiguous_runs(above_thresh)

            # Branch: do we have a usable sign trace?
            do_split = (split_clusters_by_sign and np.any(np.isfinite(obs_sg)))

            if do_split:
                obs_split = _split_clusters_at_sign_flips(raw_obs_clusters, obs_sg)
            else:
                obs_split = [{'start': s, 'end': e, 'sign': 0} for s, e in raw_obs_clusters]

            # 3. Null cluster-statistic distribution -- split nulls the SAME way
            def _cluster_stat_value(c, trace):
                """Extent = number of suprathreshold windows (magnitude-blind).
                Mass = summed excess of F over the threshold (magnitude-aware)."""
                if cluster_stat == 'extent':
                    return float(c['end'] - c['start'] + 1)
                seg = trace[c['start']:c['end'] + 1]
                return float((seg - pointwise_thresh[c['start']:c['end'] + 1]).sum())

            null_max_stats = []
            for pi in range(null.shape[0]):
                above_pi = null[pi] > pointwise_thresh
                raw_pi = _find_contiguous_runs(above_pi)
                if do_split:
                    sub_pi = _split_clusters_at_sign_flips(raw_pi, null_sg[pi])
                else:
                    sub_pi = [{'start': s, 'end': e, 'sign': 0} for s, e in raw_pi]
                stats_pi = [_cluster_stat_value(c, null[pi]) for c in sub_pi] or [0.0]
                null_max_stats.append(max(stats_pi))
            null_max_stats = np.array(null_max_stats)
            stat_thresh = np.percentile(null_max_stats, cluster_percentile)

            # 4. Filter observed sub-clusters by the cluster statistic, p per cluster
            sig_subs = []
            for c in obs_split:
                m = _cluster_stat_value(c, obs)
                p = float((null_max_stats >= m).mean())
                if m > stat_thresh:
                    sig_subs.append({**c, 'extent': c['end'] - c['start'] + 1,
                                     'cluster_stat': m, 'p_value': p})

            # 5. Build masks for plotting -- split into pos / neg if sign-aware
            window_mask = np.zeros(n_windows, dtype=bool)
            pos_window_mask = np.zeros(n_windows, dtype=bool)
            neg_window_mask = np.zeros(n_windows, dtype=bool)
            for c in sig_subs:
                window_mask[c['start']:c['end'] + 1] = True
                if c['sign'] > 0:
                    pos_window_mask[c['start']:c['end'] + 1] = True
                elif c['sign'] < 0:
                    neg_window_mask[c['start']:c['end'] + 1] = True

            # 6. Expand window-level masks to sample-level (for time-axis plots)
            def _expand(mask_w):
                m = np.zeros(n_times, dtype=bool)
                for s, e in _find_contiguous_runs(mask_w):
                    sa, _ = win_to_samples[s]
                    _, eb = win_to_samples[e]
                    m[sa:eb + 1] = True
                return m

            results[roi][eff] = {
                'observed_F': observed_F[ei],
                'null_F':     null_F[:, ei, :],
                'signed_contrast':  obs_sg,              # (n_windows,) -- the Δ trace
                'window_mask':      window_mask,          # union (back-compat)
                'pos_window_mask':  pos_window_mask,
                'neg_window_mask':  neg_window_mask,
                'sample_mask':      _expand(window_mask),
                'pos_sample_mask':  _expand(pos_window_mask),
                'neg_sample_mask':  _expand(neg_window_mask),
                'sig_clusters_with_sign': sig_subs,       # list of dicts
            }

    return results, window_centers
