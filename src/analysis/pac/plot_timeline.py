import os
import argparse
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ieeg.calc.stats import time_perm_cluster
from statsmodels.stats.multitest import multipletests


BASE_DIR = "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow"


# =========================================================
# FILE HELPERS
# =========================================================

def build_filename(subject, region, condition):

    subj = str(subject).strip()
    region_s = str(region).strip()
    cond_s = str(condition).strip()

    fname = f"coherence_epochs_full_{subj}_{region_s}_{cond_s}.pkl"

    return os.path.join(BASE_DIR, fname)


# =========================================================
# WINDOW HELPERS
# =========================================================

def make_windows(time_start, time_end, window_width, time_step):

    wins = []

    cur = time_start

    while cur < time_end:

        end = cur + window_width

        if end > time_end:
            end = time_end

        wins.append((cur, end))

        cur += time_step

    return wins


# =========================================================
# SIGNIFICANT PAIRS
# =========================================================

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


# =========================================================
# LOAD OBSERVATIONS
# =========================================================

def load_condition_observations(
    subjects,
    region,
    condition,
    windows,
    allowed_pairs=None,
    tol=1e-6,
    verbose=False
):
    """
    Build matrix:

    rows = subject/chpair observations
    cols = windows
    """

    rows = []

    for subj in subjects:

        subj = str(subj).strip()

        path = build_filename(subj, region, condition)

        if not os.path.exists(path):

            if verbose:
                print(f"[WARN] missing file: {path}")

            continue

        try:
            all_records = pd.read_pickle(path)

        except Exception as e:

            print(f"[WARN] failed reading {path}: {e}")
            continue

        if not isinstance(all_records, list):

            print(f"[WARN] {path} not list")
            continue

        pair_window_map = {}

        for rec in all_records:

            ch1 = rec.get("ch1")
            ch2 = rec.get("ch2")

            pair_key = (ch1, ch2)

            if allowed_pairs is not None:

                if pair_key not in allowed_pairs:
                    continue

            w_start = rec.get("window_start", np.nan)
            w_end = rec.get("window_end", np.nan)

            coh_mean = rec.get("coh_mean", np.nan)

            for wi, (s, e) in enumerate(windows):

                if (
                    np.isclose(w_start, s, atol=tol)
                    and np.isclose(w_end, e, atol=tol)
                ):

                    if pair_key not in pair_window_map:
                        pair_window_map[pair_key] = {}

                    if wi not in pair_window_map[pair_key]:
                        pair_window_map[pair_key][wi] = []

                    pair_window_map[pair_key][wi].append(coh_mean)

        # --------------------------------------------
        # convert pair -> vector
        # --------------------------------------------

        for (ch1, ch2), win_data in pair_window_map.items():

            vec = []

            for wi in range(len(windows)):

                vals = win_data.get(wi, [])

                if len(vals) == 0:
                    vec.append(np.nan)
                else:
                    vec.append(np.nanmean(vals))

            row = {
                "subject": subj,
                "ch1": ch1,
                "ch2": ch2,
                "obs_id": f"{subj}__{ch1}__{ch2}"
            }

            for wi, val in enumerate(vec):
                row[f"w{wi}"] = val

            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# MATCH PAIRS BETWEEN CONDITIONS
# =========================================================

def build_paired_matrices(df1, df2, n_windows):

    if df1.empty or df2.empty:
        return None, None, None

    merge_cols = ["obs_id", "subject", "ch1", "ch2"]

    merged = pd.merge(
        df1,
        df2,
        on=merge_cols,
        suffixes=("_1", "_2")
    )

    if merged.empty:
        return None, None, None

    x1 = []
    x2 = []

    for wi in range(n_windows):

        x1.append(merged[f"w{wi}_1"].values)
        x2.append(merged[f"w{wi}_2"].values)

    x1 = np.stack(x1, axis=1)
    x2 = np.stack(x2, axis=1)

    # remove NaN rows
    valid = (
        ~np.isnan(x1).any(axis=1)
        &
        ~np.isnan(x2).any(axis=1)
    )

    x1 = x1[valid]
    x2 = x2[valid]

    merged = merged.iloc[valid]

    return x1, x2, merged


# =========================================================
# CLUSTER EXTRACTION
# =========================================================

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

        p_cluster = np.nanmin(pvals[sidx:eidx+1])

        cluster_rows.append({
            "cluster_id": ci,
            "start_window": sidx,
            "end_window": eidx,
            "time_start": t0,
            "time_end": t1,
            "min_p": p_cluster
        })

    return cluster_rows


# =========================================================
# PLOT
# =========================================================

def plot_pair_result(
    c1,
    c2,
    x1,
    x2,
    mask,
    x_centers,
    out_fig
):

    mean1 = np.nanmean(x1, axis=0)
    mean2 = np.nanmean(x2, axis=0)

    sem1 = np.nanstd(x1, axis=0, ddof=1) / np.sqrt(x1.shape[0])
    sem2 = np.nanstd(x2, axis=0, ddof=1) / np.sqrt(x2.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_centers,
        mean1,
        marker='o',
        label=c1
    )

    ax.plot(
        x_centers,
        mean2,
        marker='o',
        label=c2
    )

    ax.fill_between(
        x_centers,
        mean1 - sem1,
        mean1 + sem1,
        alpha=0.25
    )

    ax.fill_between(
        x_centers,
        mean2 - sem2,
        mean2 + sem2,
        alpha=0.25
    )

    ymin, ymax = ax.get_ylim()

    sig_y = ymax + 0.05 * (ymax - ymin)

    for i, sig in enumerate(mask):

        if sig:

            ax.plot(
                x_centers[i],
                sig_y,
                marker='s'
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean coherence")

    ax.set_title(f"{c1} vs {c2}")

    ax.legend()

    plt.tight_layout()

    fig.savefig(out_fig, dpi=150)

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    p = argparse.ArgumentParser(
        description="2/4-condition cluster permutation timecourse comparison"
    )

    p.add_argument(
        '--subj',
        nargs='+',
        required=True
    )

    p.add_argument(
        '--region',
        required=True
    )

    # ==============================================
    # NOW SUPPORTS:
    # 2 conditions
    # 4 conditions
    # ==============================================

    p.add_argument(
        '--conditions',
        nargs='+',
        required=True,
        help='Provide 2 or 4 conditions'
    )

    p.add_argument(
        '--time_start',
        type=float,
        default=-1.0
    )

    p.add_argument(
        '--time_end',
        type=float,
        default=1.5
    )

    p.add_argument(
        '--window_width',
        type=float,
        default=0.5
    )

    p.add_argument(
        '--time_step',
        type=float,
        default=0.5
    )

    p.add_argument(
        '--sig_pairs_dir',
        default=None
    )

    p.add_argument(
        '--n_perm',
        type=int,
        default=5000
    )

    p.add_argument(
        '--p_thresh',
        type=float,
        default=0.05
    )

    p.add_argument(
        '--p_cluster',
        type=float,
        default=0.05
    )

    p.add_argument(
        '--tails',
        type=int,
        default=2
    )

    p.add_argument(
        '--outdir',
        default='cluster_stats'
    )

    p.add_argument(
        '--verbose',
        action='store_true'
    )

    args = p.parse_args()

    # ==============================================
    # validate number of conditions
    # ==============================================

    if len(args.conditions) not in [2, 4]:

        raise ValueError(
            "You must provide either 2 or 4 conditions."
        )

    os.makedirs(args.outdir, exist_ok=True)

    # ==============================================
    # windows
    # ==============================================

    windows = make_windows(
        args.time_start,
        args.time_end,
        args.window_width,
        args.time_step
    )

    x_centers = [
        (s + e) / 2.0
        for (s, e) in windows
    ]

    # ==============================================
    # sig pairs
    # ==============================================

    allowed_pairs = None

    if args.sig_pairs_dir:

        allowed_pairs = read_sig_pairs_for_subjects(
            args.subj,
            args.sig_pairs_dir,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"[INFO] loaded {len(allowed_pairs)//2} unique pairs")

    # ==============================================
    # load all conditions
    # ==============================================

    cond_data = {}

    for cond in args.conditions:

        print(f"\n[LOAD] {cond}")

        df = load_condition_observations(
            subjects=args.subj,
            region=args.region,
            condition=cond,
            windows=windows,
            allowed_pairs=allowed_pairs,
            verbose=args.verbose
        )

        cond_data[cond] = df

        print(f"{cond}: {len(df)} observations")

    # ==============================================
    # pairwise combinations
    # ==============================================

    condition_pairs = list(
        itertools.combinations(args.conditions, 2)
    )

    print("\n====================================")
    print(f"Total comparisons: {len(condition_pairs)}")
    print("====================================")

    all_cluster_rows = []

    # ==============================================
    # pairwise cluster tests
    # ==============================================

    for c1, c2 in condition_pairs:

        print("\n------------------------------------")
        print(f"{c1} vs {c2}")
        print("------------------------------------")

        x1, x2, merged = build_paired_matrices(
            cond_data[c1],
            cond_data[c2],
            len(windows)
        )

        if x1 is None:

            print("[WARN] no matched observations")
            continue

        print(f"matched observations: {x1.shape[0]}")
        print(f"time windows: {x1.shape[1]}")

        # ==========================================
        # cluster permutation
        # ==========================================

        mask, pvals = time_perm_cluster(
            x1,
            x2,
            p_thresh=args.p_thresh,
            p_cluster=args.p_cluster,
            n_perm=args.n_perm,
            tails=args.tails,
            axis=0,
            permutation_type='pairings',
            seed=42,
            n_jobs=-1
        )

        mask = np.asarray(mask).astype(bool)
        pvals = np.asarray(pvals)

        # ==========================================
        # save window-level results
        # ==========================================

        df_windows = pd.DataFrame({
            'window_idx': np.arange(len(windows)),
            'time_center': x_centers,
            'significant': mask,
            'p_value': pvals
        })

        out_csv = os.path.join(
            args.outdir,
            f"{c1}_vs_{c2}_window_stats.csv"
        )

        df_windows.to_csv(out_csv, index=False)

        # ==========================================
        # extract clusters
        # ==========================================

        cluster_rows = extract_clusters(
            mask,
            pvals,
            windows
        )

        for row in cluster_rows:

            row['condition_1'] = c1
            row['condition_2'] = c2
            row['n_obs'] = x1.shape[0]

            all_cluster_rows.append(row)

        # ==========================================
        # plotting
        # ==========================================

        out_fig = os.path.join(
            args.outdir,
            f"{c1}_vs_{c2}.png"
        )

        plot_pair_result(
            c1,
            c2,
            x1,
            x2,
            mask,
            x_centers,
            out_fig
        )

        print(f"saved figure: {out_fig}")

    # ==============================================
    # MULTIPLE COMPARISON CORRECTION
    # only needed when >1 pairwise test
    # ==============================================

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
            df_clusters['significant_after_holm'] = (
                df_clusters['min_p'] < 0.05
            )

        out_clusters = os.path.join(
            args.outdir,
            "all_cluster_results.csv"
        )

        df_clusters.to_csv(out_clusters, index=False)

        print(f"\nSaved cluster results -> {out_clusters}")

    else:

        print("\nNo significant clusters found.")


if __name__ == '__main__':
    main()