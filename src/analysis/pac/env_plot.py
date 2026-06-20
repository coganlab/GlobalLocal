import os
import glob
import argparse
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ieeg.calc.stats import time_perm_cluster
from statsmodels.stats.multitest import multipletests



def sanitize_filename(name: str) -> str:
    return ''.join('_' if c in '<>:"/\\|?*' else c for c in str(name))


def find_summary_pkl(input_dir, subjects, region, condition, prefix='envcorr_summary', verbose=True):
    """Find the envcorr summary pickle written by the envelope-correlation pipeline.

    The saved file name usually looks like:
        envcorr_summary_D0057_lpfc_stimulus_c.pkl
    or, for multiple subjects, something similar with the subject list embedded.

    This helper searches by region + condition and then prefers files whose basename
    contains all requested subject IDs. Returns None if no file contains all subjects
    (to trigger per-subject loading).
    """
    subj_list = [str(s).strip() for s in subjects]
    region_s = sanitize_filename(region)
    cond_s = sanitize_filename(condition)

    pattern = os.path.join(input_dir, f"{prefix}_*_{region_s}_{cond_s}.pkl")
    candidates = sorted(glob.glob(pattern))

    if len(candidates) == 0:
        return None

    # Prefer files whose basename contains all subject IDs.
    preferred = []
    for fp in candidates:
        base = os.path.basename(fp)
        if all(sub in base for sub in subj_list):
            preferred.append(fp)

    if len(preferred) == 1:
        if verbose:
            print(f"[INFO] found merged file containing all subjects: {preferred[0]}")
        return preferred[0]

    if len(preferred) > 1:
        # Choose newest among preferred.
        preferred.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if verbose:
            print(f"[WARN] multiple merged files found; using newest: {preferred[0]}")
        return preferred[0]

    # No file contains all subjects; return None to trigger per-subject loading
    if verbose:
        print(f"[INFO] no single file contains all {len(subj_list)} subjects; will load per-subject files")
    return None


def make_windows(time_start, time_end, window_width, time_step):
    """Generate contiguous or sliding windows across time."""
    wins = []
    cur = float(time_start)
    while cur < time_end:
        end = cur + window_width
        if end > time_end:
            end = time_end
        wins.append((float(cur), float(end)))
        cur += time_step
    return wins


def read_sig_pairs_for_subjects(subj_list, sig_dir, verbose=True):
    allowed = set()

    for sub in subj_list:
        sub = str(sub).strip()
        fp = os.path.join(sig_dir, f"{sub}_sig_pairs.csv")
        if not os.path.exists(fp):
            if verbose:
                print(f"[WARN] no sig pair file: {fp}")
            continue

        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] failed reading {fp}: {e}")
            continue

        cols = [c.lower() for c in df.columns]
        if 'ch1' not in cols or 'ch2' not in cols:
            print(f"[WARN] {fp} missing ch1/ch2")
            continue

        c1 = df.columns[cols.index('ch1')]
        c2 = df.columns[cols.index('ch2')]

        for _, row in df.iterrows():
            a = row[c1]
            b = row[c2]
            allowed.add((a, b))
            allowed.add((b, a))

    return allowed


def load_condition_observations_from_pkl(
    subjects,
    region,
    condition,
    windows,
    input_dir,
    allowed_pairs=None,
    tol=1e-6,
    verbose=False,
    prefix='envcorr_summary'
):
    """Load envelope-correlation summaries and build a matrix of observations.

    Rows = subject/channel-pair observations.
    Columns = windows.
    """
    rows = []
    subj_list = [str(s).strip() for s in subjects]

    # First try to find a single merged file containing all subjects
    pkl_path = find_summary_pkl(
        input_dir=input_dir,
        subjects=subj_list,
        region=region,
        condition=condition,
        prefix=prefix,
        verbose=verbose,
    )

    all_records = []

    if pkl_path is not None:
        # Single merged file found
        if verbose:
            print(f"[INFO] loading merged file: {pkl_path}")
        try:
            data = pd.read_pickle(pkl_path)
            if isinstance(data, pd.DataFrame):
                all_records.extend(data.to_dict('records'))
            elif isinstance(data, list):
                all_records.extend(data)
            else:
                print(f"[WARN] {pkl_path} is not a list or DataFrame")
        except Exception as e:
            print(f"[WARN] failed reading {pkl_path}: {e}")
    else:
        # No merged file found; try loading individual subject files
        if verbose:
            print(f"[INFO] no merged file found; attempting to load individual subject files")
        
        region_s = sanitize_filename(region)
        cond_s = sanitize_filename(condition)
        
        for subj in subj_list:
            pattern = os.path.join(input_dir, f"{prefix}_*{subj}*_{region_s}_{cond_s}.pkl")
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                if verbose:
                    print(f"[WARN] no pkl file for subject {subj}")
                continue
            
            # Use newest file for this subject
            fp = sorted(candidates, key=lambda x: os.path.getmtime(x), reverse=True)[0]
            if verbose:
                print(f"[INFO] loading {subj}: {fp}")
            try:
                data = pd.read_pickle(fp)
                if isinstance(data, pd.DataFrame):
                    all_records.extend(data.to_dict('records'))
                elif isinstance(data, list):
                    all_records.extend(data)
            except Exception as e:
                print(f"[WARN] failed reading {fp}: {e}")

    if not all_records:
        if verbose:
            print(f"[WARN] no records loaded for condition={condition}, region={region}")
        return pd.DataFrame(), None

    pair_window_map = {}

    for rec in all_records:
        subj = str(rec.get("subject", "")).strip()
        if subj not in subj_list:
            if verbose:
                print(f"[DEBUG] skipping subject {subj} not in {subj_list}")
            continue

        ch1 = rec.get("ch1")
        ch2 = rec.get("ch2")
        pair_key = (ch1, ch2)

        if allowed_pairs is not None and pair_key not in allowed_pairs:
            continue

        w_start = rec.get("window_start", np.nan)
        w_end = rec.get("window_end", np.nan)
        envcorr_mean = rec.get("envcorr_mean", np.nan)

        # If older files used another key, try to fall back.
        if envcorr_mean is np.nan or envcorr_mean is None:
            envcorr_mean = rec.get("coh_mean", np.nan)

        for wi, (s, e) in enumerate(windows):
            if np.isclose(w_start, s, atol=tol) and np.isclose(w_end, e, atol=tol):
                if pair_key not in pair_window_map:
                    pair_window_map[pair_key] = {}
                if wi not in pair_window_map[pair_key]:
                    pair_window_map[pair_key][wi] = []
                pair_window_map[pair_key][wi].append(envcorr_mean)

    for (ch1, ch2), win_data in pair_window_map.items():
        vec = []
        for wi in range(len(windows)):
            vals = win_data.get(wi, [])
            vals = [v for v in vals if pd.notna(v)]
            if len(vals) == 0:
                vec.append(np.nan)
            else:
                vec.append(float(np.nanmean(vals)))

        row = {
            "subject": subj_list[0] if len(subj_list) == 1 else "__".join(subj_list),
            "ch1": ch1,
            "ch2": ch2,
            "obs_id": f"{subj_list[0] if len(subj_list) == 1 else '__'.join(subj_list)}__{ch1}__{ch2}",
        }
        for wi, val in enumerate(vec):
            row[f"w{wi}"] = val
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame(), pkl_path

    return pd.DataFrame(rows), pkl_path



def build_paired_matrices(df1, df2, n_windows, verbose=False):
    if df1.empty or df2.empty:
        if verbose:
            print(f"[DEBUG] df1.empty={df1.empty}, df2.empty={df2.empty}")
        return None, None, None

    merge_cols = ["obs_id", "subject", "ch1", "ch2"]
    if verbose:
        print(f"[DEBUG] df1 columns: {df1.columns.tolist()}, shape: {df1.shape}")
        print(f"[DEBUG] df2 columns: {df2.columns.tolist()}, shape: {df2.shape}")
    
    merged = pd.merge(df1, df2, on=merge_cols, suffixes=("_1", "_2"))
    if verbose:
        print(f"[DEBUG] after merge: {merged.shape}")
    
    if merged.empty:
        if verbose:
            print(f"[DEBUG] merged is empty after merge")
        return None, None, None

    x1 = []
    x2 = []
    for wi in range(n_windows):
        x1.append(merged[f"w{wi}_1"].values)
        x2.append(merged[f"w{wi}_2"].values)

    x1 = np.stack(x1, axis=1)
    x2 = np.stack(x2, axis=1)

    if verbose:
        print(f"[DEBUG] before valid filter: x1.shape={x1.shape}, x2.shape={x2.shape}")
        print(f"[DEBUG] nan count in x1: {np.isnan(x1).sum()}, nan count in x2: {np.isnan(x2).sum()}")

    valid = (~np.isnan(x1).any(axis=1)) & (~np.isnan(x2).any(axis=1))
    x1 = x1[valid]
    x2 = x2[valid]
    merged = merged.iloc[valid]

    if verbose:
        print(f"[DEBUG] after valid filter: x1.shape={x1.shape}, x2.shape={x2.shape}")

    return x1, x2, merged



def extract_clusters(mask, pvals, windows):
    clusters = []
    in_cluster = False
    start_idx = None

    for i, sig in enumerate(mask):
        if sig and not in_cluster:
            start_idx = i
            in_cluster = True
        elif not sig and in_cluster:
            end_idx = i - 1
            clusters.append((start_idx, end_idx))
            in_cluster = False

    if in_cluster:
        clusters.append((start_idx, len(mask) - 1))

    cluster_rows = []
    for ci, (sidx, eidx) in enumerate(clusters):
        t0 = windows[sidx][0]
        t1 = windows[eidx][1]
        p_cluster = np.nanmin(pvals[sidx:eidx + 1])
        cluster_rows.append({
            "cluster_id": ci,
            "start_window": sidx,
            "end_window": eidx,
            "time_start": t0,
            "time_end": t1,
            "min_p": p_cluster,
        })

    return cluster_rows


def plot_pair_result(c1, c2, x1, x2, mask, x_centers, out_fig, ylabel="Mean envelope correlation", p_thresh=None):
    mean1 = np.nanmean(x1, axis=0)
    mean2 = np.nanmean(x2, axis=0)

    sem1 = np.nanstd(x1, axis=0, ddof=1) / np.sqrt(max(x1.shape[0], 1))
    sem2 = np.nanstd(x2, axis=0, ddof=1) / np.sqrt(max(x2.shape[0], 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_centers, mean1, marker='o', label=c1)
    ax.plot(x_centers, mean2, marker='o', label=c2)

    ax.fill_between(x_centers, mean1 - sem1, mean1 + sem1, alpha=0.25)
    ax.fill_between(x_centers, mean2 - sem2, mean2 + sem2, alpha=0.25)

    ymin, ymax = ax.get_ylim()
    sig_y = ymax + 0.05 * (ymax - ymin)
    sig_plotted = False
    for i, sig in enumerate(mask):
        if sig:
            label = f"p < {p_thresh}" if not sig_plotted and p_thresh is not None else None
            ax.plot(x_centers[i], sig_y, marker='s', color='black', label=label)
            sig_plotted = True

    if p_thresh is not None:
        ax.set_title(f"{c1} vs {c2} (significant where p < {p_thresh})")
    else:
        ax.set_title(f"{c1} vs {c2}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)



def main():
    p = argparse.ArgumentParser(
        description="2/4-condition cluster permutation timecourse comparison for envelope correlation summaries"
    )

    p.add_argument('--subj', nargs='+', required=True, help='Subject IDs, e.g. D0057 D0063')
    p.add_argument('--region', required=True, help='ROI region name used in the pkl filename, e.g. lpfc')
    p.add_argument('--conditions', nargs='+', required=True, help='Provide 2 or 4 conditions')
    p.add_argument('--input_dir', required=True, help='Directory containing envcorr_summary_*.pkl files')
    p.add_argument('--time_start', type=float, default=-1.0)
    p.add_argument('--time_end', type=float, default=1.5)
    p.add_argument('--window_width', type=float, default=0.5)
    p.add_argument('--time_step', type=float, default=0.5)
    p.add_argument('--sig_pairs_dir', default=None, help='Optional directory with subject-level sig pair CSVs')
    p.add_argument('--n_perm', type=int, default=5000)
    p.add_argument('--p_thresh', type=float, default=0.05)
    p.add_argument('--p_cluster', type=float, default=0.05)
    p.add_argument('--tails', type=int, default=2)
    p.add_argument('--outdir', default='cluster_stats')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if len(args.conditions) not in [2, 4]:
        raise ValueError("You must provide either 2 or 4 conditions.")

    os.makedirs(args.outdir, exist_ok=True)
    # create a dedicated env_plot subfolder inside the provided outdir
    env_outdir = os.path.join(args.outdir, 'env_plot')
    os.makedirs(env_outdir, exist_ok=True)

    # build a short subject label to include in filenames. If the joined
    # subject string is longer than 7 characters, truncate to the first 7.
    subj_joined = "__".join(args.subj)
    subj_label = subj_joined if len(subj_joined) <= 7 else subj_joined[:7]

    windows = make_windows(args.time_start, args.time_end, args.window_width, args.time_step)
    x_centers = [(s + e) / 2.0 for (s, e) in windows]

    allowed_pairs = None
    if args.sig_pairs_dir:
        allowed_pairs = read_sig_pairs_for_subjects(args.subj, args.sig_pairs_dir, verbose=args.verbose)
        if args.verbose:
            print(f"[INFO] loaded {len(allowed_pairs) // 2} unique pairs")

    cond_data = {}
    pkl_paths = {}

    for cond in args.conditions:
        print(f"\n[LOAD] {cond}")
        df, pkl_path = load_condition_observations_from_pkl(
            subjects=args.subj,
            region=args.region,
            condition=cond,
            windows=windows,
            input_dir=args.input_dir,
            allowed_pairs=allowed_pairs,
            verbose=args.verbose,
        )
        cond_data[cond] = df
        pkl_paths[cond] = pkl_path
        print(f"{cond}: {len(df)} observations")
        if pkl_path is not None:
            print(f"  source: {pkl_path}")

    condition_pairs = list(itertools.combinations(args.conditions, 2))
    print("\n====================================")
    print(f"Total comparisons: {len(condition_pairs)}")
    print("====================================")

    all_cluster_rows = []

    for c1, c2 in condition_pairs:
        print("\n------------------------------------")
        print(f"{c1} vs {c2}")
        print("------------------------------------")

        x1, x2, merged = build_paired_matrices(cond_data[c1], cond_data[c2], len(windows), verbose=args.verbose)
        if x1 is None:
            print("[WARN] no matched observations")
            continue

        print(f"matched observations: {x1.shape[0]}")
        print(f"time windows: {x1.shape[1]}")

        _, pvals = time_perm_cluster(
            x1,
            x2,
            p_thresh=args.p_thresh,
            p_cluster=args.p_cluster,
            n_perm=args.n_perm,
            tails=args.tails,
            axis=0,
            permutation_type='pairings',
            seed=42,
            n_jobs=-1,
        )

        pvals = np.asarray(pvals)
        sig_mask = pvals < args.p_thresh

        df_windows = pd.DataFrame({
            'window_idx': np.arange(len(windows)),
            'time_center': x_centers,
            'significant': sig_mask,
            'p_value': pvals,
        })

        out_csv = os.path.join(env_outdir, f"{c1}_vs_{c2}_{subj_label}_window_stats.csv")
        df_windows.to_csv(out_csv, index=False)

        cluster_rows = extract_clusters(sig_mask, pvals, windows)
        for row in cluster_rows:
            row['condition_1'] = c1
            row['condition_2'] = c2
            row['n_obs'] = x1.shape[0]
            all_cluster_rows.append(row)

        out_fig = os.path.join(env_outdir, f"{c1}_vs_{c2}_{subj_label}.png")
        plot_pair_result(
            c1,
            c2,
            x1,
            x2,
            sig_mask,
            x_centers,
            out_fig,
            ylabel='Mean envelope correlation',
            p_thresh=args.p_thresh,
        )
        print(f"saved figure: {out_fig}")

    if len(all_cluster_rows) > 0:
        df_clusters = pd.DataFrame(all_cluster_rows)

        if len(condition_pairs) > 1:
            reject, p_corr, _, _ = multipletests(
                df_clusters['min_p'].values,
                alpha=0.05,
                method='holm'
            )
            df_clusters['p_corrected'] = p_corr
            df_clusters['significant_after_holm'] = reject
        else:
            df_clusters['p_corrected'] = df_clusters['min_p']
            df_clusters['significant_after_holm'] = df_clusters['min_p'] < 0.05

        out_clusters = os.path.join(env_outdir, f"all_cluster_results_{subj_label}.csv")
        df_clusters.to_csv(out_clusters, index=False)
        print(f"\nSaved cluster results -> {out_clusters}")
    else:
        print("\nNo significant clusters found.")


if __name__ == '__main__':
    main()
