#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# keep same BASE_DIR & filename formatting as your other scripts
BASE_DIR = "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow"


def build_filename(subject, region, condition, tstart, tend):
    """Construct filename identical to sig_test.py convention."""
    if isinstance(subject, (list, tuple)):
        raise ValueError("Subject should be a single ID string, not a list or tuple")
    subj = str(subject).strip()
    tstart_s = str(tstart)
    tend_s = str(tend)
    fname = f"coherence_{subj}_{region}_{condition}_({tstart_s}, {tend_s})_summary.csv"
    return os.path.join(BASE_DIR, fname)


def load_summary_for_condition_window(subjects, region, condition, tstart, tend, allowed_pairs=None, verbose=True):
    """
    Load all summary CSVs for a given condition and time window across subjects.
    Optionally filter to only allowed_pairs (set of (ch1,ch2) tuples).
    Returns combined DataFrame (may be empty if no files found / none match) and list of found_subjects.
    """
    rows = []
    found = []
    for subj in subjects:
        subj = str(subj).strip()
        path = build_filename(subj, region, condition, tstart, tend)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                if verbose:
                    print(f"Warning: failed to read {path}: {e}")
                continue
            df['subject'] = subj
            # filter by allowed_pairs if provided
            if allowed_pairs:
                # define mask where (ch1,ch2) or (ch2,ch1) in allowed_pairs
                mask = df.apply(lambda r: ((r['ch1'], r['ch2']) in allowed_pairs) or ((r['ch2'], r['ch1']) in allowed_pairs), axis=1)
                df = df[mask]
                if df.empty:
                    # nothing left for this subject in this window
                    if verbose:
                        print(f"  Subject {subj}: no matching sig pairs in {path} after filtering")
                    continue
            rows.append(df)
            found.append(subj)
        else:
            # silent skip
            pass
    if not rows:
        return pd.DataFrame(), []
    return pd.concat(rows, ignore_index=True), sorted(found)


def perpair_means(df):
    """
    Same aggregation behavior as the other script:
    - if 'subject' present: compute per-subject per-pair means then average across subjects
    - else: compute mean across all rows per pair
    Returns DataFrame with columns ['ch1','ch2','mean_coh','n','n_subjects']
    """
    if df.empty:
        return pd.DataFrame(columns=['ch1', 'ch2', 'mean_coh', 'n', 'n_subjects'])
    if 'subject' in df.columns:
        g = df.groupby(['subject', 'ch1', 'ch2'], as_index=False).agg(
            mean_coh=('coh_mean', 'mean'),
            n=('coh_mean', 'size')
        )
        g = g.groupby(['ch1', 'ch2'], as_index=False).agg(
            mean_coh=('mean_coh', 'mean'),
            n=('n', 'sum'),
            n_subjects=('subject', 'nunique')
        )
    else:
        g = df.groupby(['ch1', 'ch2'], as_index=False).agg(
            mean_coh=('coh_mean', 'mean'),
            n=('coh_mean', 'size')
        )
        g['n_subjects'] = 1
    return g


def compute_overall_stats(perpair_df):
    """
    From per-pair aggregated table (with mean_coh), compute
    overall mean across pairs, std across pairs, SEM, and n_pairs.
    Returns dict.
    """
    if perpair_df.empty:
        return {'mean': np.nan, 'std': np.nan, 'sem': np.nan, 'n_pairs': 0}
    arr = perpair_df['mean_coh'].values.astype(float)
    n = np.count_nonzero(~np.isnan(arr))
    if n == 0:
        return {'mean': np.nan, 'std': np.nan, 'sem': np.nan, 'n_pairs': 0}
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 0 else np.nan
    return {'mean': mean, 'std': std, 'sem': sem, 'n_pairs': n}


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


def read_sig_pairs_for_subjects(subj_list, sig_dir, verbose=True):
    """
    Read sig pair files for each subject in subj_list from sig_dir.
    Tries several filename patterns. Returns set of allowed pairs {(ch1,ch2),...}.
    """
    allowed = set()
    tried_any = False
    for sub in subj_list:
        sub = str(sub).strip()
        candidates = [
            os.path.join(sig_dir, f"{sub}_sig_pairs.csv"),
        ]
        found_any = False
        for fp in candidates:
            if os.path.exists(fp):
                tried_any = True
                found_any = True
                try:
                    df = pd.read_csv(fp)
                except Exception as e:
                    if verbose:
                        print(f"Warning: failed to read sig pairs file {fp}: {e}")
                    continue
                # Expect columns 'ch1' and 'ch2' (case insensitive)
                cols = [c.lower() for c in df.columns]
                if 'ch1' in cols and 'ch2' in cols:
                    # map back to actual column names
                    c1 = df.columns[cols.index('ch1')]
                    c2 = df.columns[cols.index('ch2')]
                    for _, row in df.iterrows():
                        a = row[c1]
                        b = row[c2]
                        # add both orders to be tolerant
                        allowed.add((a, b))
                        allowed.add((b, a))
                else:
                    if verbose:
                        print(f"Warning: sig file {fp} lacks 'ch1'/'ch2' columns; skipping")
        if not found_any and verbose:
            print(f"Warning: no sig pairs file found for subject {sub} in {sig_dir} (tried {len(candidates)} patterns)")
    if not tried_any and verbose:
        print(f"Warning: no sig pairs files found in {sig_dir} for any provided subjects")
    return allowed


def main():
    p = argparse.ArgumentParser(description="Plot condition timecourses across windows (optionally only sig pairs)")
    p.add_argument('--subj', nargs='+', required=True, help='subjects e.g. D0057 D0059')
    p.add_argument('--region', required=True, help='region e.g. lpfc')
    p.add_argument('--conditions', nargs='+', required=True, help='conditions to compare (space-separated)')
    p.add_argument('--time_start', type=float, default=-1.0)
    p.add_argument('--time_end', type=float, default=1.5)
    p.add_argument('--window_width', type=float, default=0.5)
    p.add_argument('--time_step', type=float, default=0.5)
    p.add_argument('--out_fig', default='conditions_timecourse.png')
    p.add_argument('--out_csv', default='conditions_timecourse_summary.csv')
    p.add_argument('--agg', choices=['per_pair', 'raw'], default='per_pair',
                   help='aggregation method when summarizing across pairs (default per_pair)')
    p.add_argument('--sig_pairs_dir', default=None,
                   help="If provided, read sig pairs files for each subject from this directory and only plot those pairs.")
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    wins = make_windows(args.time_start, args.time_end, args.window_width, args.time_step)
    if len(wins) == 0:
        raise SystemExit("No windows generated; check time_start/time_end/window_width")

    # if sig_pairs_dir provided, build allowed set; if none found, fall back to all pairs
    allowed_pairs = None
    if args.sig_pairs_dir:
        allowed_pairs = read_sig_pairs_for_subjects(args.subj, args.sig_pairs_dir, verbose=args.verbose)
        if not allowed_pairs:
            print(f"Warning: no valid sig pairs found in {args.sig_pairs_dir} for provided subjects; falling back to using all pairs.")
            allowed_pairs = None
        else:
            if args.verbose:
                print(f"Loaded {len(allowed_pairs)//2} unique pairs (expanded with both orders) from sig dir {args.sig_pairs_dir}")

    # Prepare storage for summary table
    records = []

    # For plotting convenience compute x coordinate as window midpoint
    x_centers = [ (s + e) / 2.0 for (s, e) in wins ]

    # For each condition, collect mean/sem per window
    condition_results = {}
    for cond in args.conditions:
        means = []
        sems = []
        n_pairs_list = []
        for (s, e) in wins:
            if args.verbose:
                print(f"Loading {cond} window ({s},{e}) ...")
            df, found = load_summary_for_condition_window(args.subj, args.region, cond, s, e, allowed_pairs=allowed_pairs, verbose=args.verbose)
            if df.empty:
                # record NaNs but continue
                if args.verbose:
                    print(f"  No data found for condition {cond} window ({s},{e}) after filtering")
                perpair = pd.DataFrame(columns=['ch1','ch2','mean_coh','n','n_subjects'])
            else:
                # aggregate (optionally we could support raw pooling, but default per_pair is consistent)
                perpair = perpair_means(df)

            stats = compute_overall_stats(perpair)
            means.append(stats['mean'])
            sems.append(stats['sem'])
            n_pairs_list.append(stats['n_pairs'])

            # record row for CSV
            records.append({
                'condition': cond,
                'tstart': s,
                'tend': e,
                'center': (s+e)/2.0,
                'mean': stats['mean'],
                'std': stats['std'],
                'sem': stats['sem'],
                'n_pairs': stats['n_pairs']
            })

        condition_results[cond] = {'means': np.array(means), 'sems': np.array(sems), 'n_pairs': np.array(n_pairs_list)}

    # save numeric summary to CSV
    df_summary = pd.DataFrame.from_records(records,
                                          columns=['condition','tstart','tend','center','mean','std','sem','n_pairs'])
    df_summary.to_csv(args.out_csv, index=False)
    print(f"Saved numeric summary to {args.out_csv}")

    # Plotting: one figure, one line per condition, shaded SEM
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond, res in condition_results.items():
        y = res['means']
        yerr = res['sems']
        # plot line with markers (matplotlib chooses colors automatically)
        line, = ax.plot(x_centers, y, marker='o', label=cond)
        # shaded error band if sem available
        # get color from line to keep band consistent with line color
        col = line.get_color()
        lower = y - yerr
        upper = y + yerr
        # handle NaNs gracefully
        lower = np.where(np.isnan(lower), np.nan, lower)
        upper = np.where(np.isnan(upper), np.nan, upper)
        ax.fill_between(x_centers, lower, upper, alpha=0.25, facecolor=col, linewidth=0)

    ax.set_xlabel('Time (s) (window center)')
    ax.set_ylabel('Mean coherence (averaged across selected channel-pairs)')
    ax.set_title('Conditions timecourse (mean across selected pairs)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_fig='sig_pairs/' + args.subj[0] + '_' + args.region + '_timecourse.png'
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure to {out_fig}")
    plt.close(fig)


if __name__ == '__main__':
    main()
