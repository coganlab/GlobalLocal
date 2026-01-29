import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats

# base folder where the coherence summary CSVs live (edit if needed)
BASE_DIR = "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow"


def load_csv(path):
    return pd.read_csv(path)


def build_filename(subject, region, condition):
    if isinstance(subject, (list, tuple)):
        raise ValueError("Subject should be a single ID string, not a list or tuple")
    subject = str(subject).strip()
    region_s = str(region).strip()
    cond_s = str(condition).strip()
    fname = f"coherence_epochs_summary_{subject}_{region_s}_{cond_s}.csv"
    return os.path.join(BASE_DIR, fname)


def load_multiple_subjects_data_for_condition(subjects, region, condition, tstart, tend):
    
    try:
        return load_multiple_subjects_data(subjects, region, condition, tstart, tend)
    except SystemExit:
        # load_multiple_subjects_data raised SystemExit if no valid files found
        return pd.DataFrame(), []


def load_multiple_subjects_data(subjects, region, condition, tstart, tend, tol=1e-6):
    all_data = []
    found_subjects = []
    for subject in subjects:
        subject = str(subject).strip()
        try:
            filepath = build_filename(subject, region, condition)
            if os.path.exists(filepath):
                df = load_csv(filepath)
                # Ensure numeric columns exist
                if 'window_start' not in df.columns or 'window_end' not in df.columns:
                    print(f"Warning: file {filepath} does not contain window_start/window_end columns. Skipping subject.")
                    continue
                # select rows matching the requested window (use isclose for floats)
                mask = np.isclose(df['window_start'].astype(float), float(tstart), atol=tol) & \
                       np.isclose(df['window_end'].astype(float), float(tend), atol=tol)
                df_filtered = df[mask].copy()
                if df_filtered.empty:
                    # no matching window rows for this subject
                    # skip quietly (upstream code may warn)
                    continue
                # ensure subject column exists
                df_filtered['subject'] = subject
                all_data.append(df_filtered)
                found_subjects.append(subject)
            else:
                # file missing for this subject/condition -> continue
                pass
        except Exception as e:
            print(f"Error processing subject {subject} cond {condition} window ({tstart},{tend}): {e}")
    if not all_data:
        # Signal upstream that no files were found for this condition/window
        raise SystemExit(f"No valid data files found for condition {condition} in window ({tstart},{tend})")
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data, found_subjects


def perpair_means(df):
    """
    Aggregate dataframe to per-channel-pair means.
    Expects columns at least: 'ch1','ch2','coh_mean' ; optionally 'subject'
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


def compute_perpair_and_overall(A, B):
    Aagg = perpair_means(A)
    Bagg = perpair_means(B)
    merged = pd.merge(Aagg, Bagg, on=['ch1', 'ch2'], how='inner', suffixes=('_A', '_B'))
    if merged.empty:
        raise SystemExit("no matching channel pairs between A and B")
    merged['mean_diff'] = merged['mean_coh_A'] - merged['mean_coh_B']
    x = merged['mean_coh_A'].values
    y = merged['mean_coh_B'].values
    n_pairs = len(x)
    if n_pairs >= 2:
        t_stat, p_val = stats.ttest_rel(x, y, nan_policy='omit')
    else:
        t_stat, p_val = np.nan, np.nan
    overall = {
        'n_pairs': n_pairs,
        'mean_A': float(np.nanmean(x)) if n_pairs > 0 else np.nan,
        'mean_B': float(np.nanmean(y)) if n_pairs > 0 else np.nan,
        'mean_diff': float(np.nanmean(x - y)) if n_pairs > 0 else np.nan,
        't_stat': t_stat, 'p_val': p_val
    }
    perpair_out = merged.rename(columns={'mean_coh_A': 'mean_A', 'mean_coh_B': 'mean_B'})
    # normalize columns
    perpair_out['n_A'] = perpair_out.get('n_A', merged.get('n_A', np.nan))
    perpair_out['n_B'] = perpair_out.get('n_B', merged.get('n_B', np.nan))
    cols = ['ch1', 'ch2', 'n_A', 'n_B', 'mean_A', 'mean_B', 'mean_diff']
    for c in cols:
        if c not in perpair_out.columns:
            perpair_out[c] = np.nan
    perpair_out = perpair_out[cols]
    return perpair_out, overall


def pkl_path_from_summary(csv_path):
    base = os.path.basename(csv_path)
    dirn = os.path.dirname(csv_path)
    if base.startswith('coherence_epochs_summary_') and base.endswith('.csv'):
        new_base = base.replace('coherence_epochs_summary_', 'coherence_epochs_full_').rsplit('.csv', 1)[0] + '.pkl'
        return os.path.join(dirn, new_base)
    # fallback: generic replace
    if csv_path.endswith('_summary.csv'):
        return csv_path.replace('_summary.csv', '_full.pkl')
    return csv_path + '_full.pkl'


def load_trials_for_subjects(subjects, region, cond, tstart, tend, tol=1e-6):
    
    trials_map = {}
    if subjects is None:
        return trials_map
    for subj in subjects:
        subj = str(subj).strip()
        try:
            csvp = build_filename(subj, region, cond)
            pklp = pkl_path_from_summary(csvp)
            if not os.path.exists(pklp):
                # no pickle for this subject
                continue
            d = pd.read_pickle(pklp)
            # ensure window columns exist
            if 'window_start' not in d.columns or 'window_end' not in d.columns:
                print(f"Warning: pkl {pklp} missing window_start/window_end columns; skipping.")
                continue
            # filter for requested window (use tolerance)
            mask = np.isclose(d['window_start'].astype(float), float(tstart), atol=tol) & \
                   np.isclose(d['window_end'].astype(float), float(tend), atol=tol)
            d_filtered = d[mask]
            if d_filtered.empty:
                continue
            for _, row in d_filtered.iterrows():
                key = (row['ch1'], row['ch2'])
                coh_trials = row.get('coh_trials')
                if coh_trials is None:
                    continue
                arr = np.array(coh_trials)
                # arr expected shape: (n_epochs, n_freqs) or (n_freqs,) if averaged
                if arr.ndim == 1:
                    # scalar per-frequency vector: reduce to mean
                    trial_vals = np.atleast_1d(arr).astype(float)
                elif arr.ndim == 2:
                    # prefer per-trial means over frequencies: mean over freq axis -> (n_epochs,)
                    # if arr shape is (n_epochs, n_freqs), arr.mean(axis=1)
                    # if arr shape is (n_freqs, n_epochs) (unexpected), handle accordingly
                    if arr.shape[0] >= arr.shape[1]:
                        trial_means = arr.mean(axis=1)
                    else:
                        trial_means = arr.mean(axis=0)
                    trial_vals = np.atleast_1d(trial_means).astype(float)
                else:
                    # higher dims: collapse last axis (freq) and flatten
                    trial_vals = arr.mean(axis=-1).ravel().astype(float)
                trials_map.setdefault(key, []).extend(list(trial_vals))
        except Exception as e:
            print(f"Error reading pkl for {subj} cond {cond} window ({tstart},{tend}): {e}")
    return trials_map


def load_trials_for_windows_conditions(subjects, region, cond_list, windows):
    """
    Load and merge trial maps for a list of windows and a list of conditions.
    Returns merged trials_map
    """
    merged = {}
    if not windows or not cond_list:
        return merged
    for cond in cond_list:
        for (tstart, tend) in windows:
            tm = load_trials_for_subjects(subjects, region, cond, tstart, tend)
            for k, v in tm.items():
                merged.setdefault(k, []).extend(v)
    return merged


def load_data_for_conditions_windows(subjects, region, cond_list, tstart, tend):
    """
    Load summary CSVs for multiple conditions (cond_list) for a single window,
    concatenate them and return (df, found_subjects_total)
    """
    combined = []
    found_subjects_total = set()
    for cond in cond_list:
        try:
            df, found = load_multiple_subjects_data(subjects, region, cond, tstart, tend)
            if not df.empty:
                combined.append(df)
                found_subjects_total.update(found)
        except SystemExit:
            # no data for this cond/window -> skip quietly
            pass
    if not combined:
        # signal upstream
        raise SystemExit(f"No data found for any of conditions {cond_list} in window ({tstart},{tend})")
    return pd.concat(combined, ignore_index=True), sorted(list(found_subjects_total))


def run_permutation_for_perpair(perpair_df, trials_A, trials_B, args, out_perpair, window_label=None):
    """
    Permutation test for each electrode pair independently.
    """
    perm_pvals = []
    perm_sign = []
    perm_n = []
    perm_methods = []
    rng = np.random.default_rng()

    total_pairs_with_data = 0
    for _, row in perpair_df.iterrows():
        key = (row['ch1'], row['ch2'])
        valsA = np.array(trials_A.get(key, []), dtype=float)
        valsB = np.array(trials_B.get(key, []), dtype=float)
        valsA = valsA[~np.isnan(valsA)]
        valsB = valsB[~np.isnan(valsB)]
        n_A = len(valsA)
        n_B = len(valsB)
        n_total = n_A + n_B
        perm_n.append(n_total)
        if n_total == 0 or n_A == 0 or n_B == 0:
            perm_pvals.append(np.nan)
            perm_sign.append(False)
            perm_methods.append('no_data')
            continue
        total_pairs_with_data += 1
        obs_diff = float(np.mean(valsA) - np.mean(valsB))
        pooled_trials = np.concatenate([valsA, valsB])
        perm_diffs = []
        for _ in range(args.n_perm):
            shuffled = rng.permutation(pooled_trials)
            perm_A = shuffled[:n_A]
            perm_B = shuffled[n_A:]
            perm_diff = np.mean(perm_A) - np.mean(perm_B)
            perm_diffs.append(perm_diff)
        perm_diffs = np.array(perm_diffs)
        if args.alternative == 'greater':
            pval = float(np.mean(perm_diffs >= obs_diff))
        else:
            pval = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
        perm_pvals.append(pval)
        perm_sign.append(bool(pval < args.alpha))
        perm_methods.append('within_pair_shuffling')

    print(f"Permutation test complete: {total_pairs_with_data} pairs tested")

    perpair_df = perpair_df.copy()
    perpair_df['perm_n'] = perm_n
    perpair_df['perm_pval'] = perm_pvals
    perpair_df['perm_method'] = perm_methods
    perpair_df['perm_significant'] = perm_sign
    perpair_df['perm_alternative'] = args.alternative

    # BH FDR
    pvals = np.array(perpair_df['perm_pval'].values, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    mask = ~np.isnan(pvals)
    if mask.sum() > 0:
        pv = pvals[mask]
        m = pv.size
        order = np.argsort(pv)
        q_raw = pv[order] * m / (np.arange(1, m + 1))
        q_adj = np.minimum.accumulate(q_raw[::-1])[::-1]
        qvals_masked = np.empty_like(pv)
        qvals_masked[order] = q_adj
        qvals[mask] = qvals_masked
        qvals = np.minimum(qvals, 1.0)
    perpair_df['perm_q_bh'] = qvals
    perpair_df['perm_significant_bh'] = (perpair_df['perm_q_bh'] < args.alpha).fillna(False)

    base = os.path.splitext(out_perpair)[0]
    if window_label:
        base = f"{base}_{window_label}"
    sig_out_unc = base + '_perm_significant_uncorrected.csv'
    sig_out_bh = base + '_perm_significant_bh.csv'
    sig_pairs_unc = perpair_df[perpair_df['perm_significant']]
    sig_pairs_bh = perpair_df[perpair_df['perm_significant_bh']]
    sig_pairs_unc.to_csv(sig_out_unc, index=False)
    sig_pairs_bh.to_csv(sig_out_bh, index=False)
    print(f"Significant pairs (uncorrected) saved to {sig_out_unc} ({(perpair_df['perm_significant']).sum()} total)")
    print(f"Significant pairs (BH FDR) saved to {sig_out_bh} ({(perpair_df['perm_significant_bh']).sum()} total)")

    return perpair_df


def main():
    p = argparse.ArgumentParser(description="Compare coherence between early (-0.5:0.0) and late (0.0:0.5) time windows using condA+condB combined data")
    p.add_argument('--subj', nargs='+', help='subjects e.g. D0057 D0059 D0063', required=True)
    p.add_argument('--region', help='brain region e.g. lpfc acc', required=True)
    p.add_argument('--condA', help='condition A label (will be combined)', required=True)
    p.add_argument('--condB', help='condition B label (will be combined). If omitted, condA is used', default=None)
    p.add_argument('--time_start', type=float, default=-1.0)
    p.add_argument('--time_end', type=float, default=1.5)
    p.add_argument('--time_step', type=float, default=0.5)
    p.add_argument('--window_width', type=float, default=0.5)
    p.add_argument('-o', '--out_perpair', default='perpair_results.csv')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--perm_trials', action='store_true')
    p.add_argument('--n_perm', type=int, default=200)
    p.add_argument('--alternative', type=str, default='greater', choices=['greater', 'two-sided'])
    args = p.parse_args()

    if args.condB is None:
        args.condB = args.condA
    cond_list = [args.condA, args.condB]

    # Generate windows
    win_list = []
    current_start = args.time_start
    while current_start < args.time_end:
        current_end = current_start + args.window_width
        if current_end > args.time_end:
            current_end = args.time_end
        win_list.append((current_start, current_end))
        current_start += args.time_step

    print("==========================================")
    print("Time Window Comparison Analysis (condA+condB combined)")
    print("==========================================")
    print(f"Subjects: {args.subj}")
    print(f"Region: {args.region}")
    print(f"Conditions combined: {cond_list}")
    print(f"Time windows: {len(win_list)} windows from {args.time_start} to {args.time_end} (step={args.time_step})")
    print(f"Permutation test: {args.perm_trials} (n_perm={args.n_perm})")
    print("==========================================\n")

    # identify baseline and test windows (hard-coded ranges)
    baseline_windows = [(s, e) for s, e in win_list if s >= -0.5 and e <= 0.0]
    test_windows = [(s, e) for s, e in win_list if s >= 0.0 and e <= 0.5]

    if not baseline_windows or not test_windows:
        raise SystemExit("No valid baseline or test windows found within [-0.5,0.0] and [0.0,0.5]")

    # Load and combine summary CSVs for baseline windows across cond_list
    early_data_list = []
    for (tstart, tend) in baseline_windows:
        try:
            df, _ = load_data_for_conditions_windows(args.subj, args.region, cond_list, tstart, tend)
            early_data_list.append(df)
        except SystemExit as e:
            print(f"Warning: {e}")

    if not early_data_list:
        raise SystemExit("Failed to load any baseline data for combined conditions")

    early_data = pd.concat(early_data_list, ignore_index=True)

    # Load and combine summary CSVs for test windows across cond_list
    late_data_list = []
    for (tstart, tend) in test_windows:
        try:
            df, _ = load_data_for_conditions_windows(args.subj, args.region, cond_list, tstart, tend)
            late_data_list.append(df)
        except SystemExit as e:
            print(f"Warning: {e}")

    if not late_data_list:
        raise SystemExit("Failed to load any test data for combined conditions")

    late_data = pd.concat(late_data_list, ignore_index=True)

    # Aggregate per-pair
    early_agg = perpair_means(early_data)
    late_agg = perpair_means(late_data)

    merged = pd.merge(early_agg, late_agg, on=['ch1', 'ch2'], how='inner', suffixes=('_early', '_late'))
    if merged.empty:
        raise SystemExit("No matching electrode pairs found between windows")

    merged['mean_diff'] = merged['mean_coh_late'] - merged['mean_coh_early']

    perpair_df = merged[['ch1', 'ch2']].copy()
    perpair_df['mean_early'] = merged['mean_coh_early'].values
    perpair_df['mean_late'] = merged['mean_coh_late'].values
    perpair_df['mean_diff'] = merged['mean_diff'].values

    # Run permutation test if requested: use trials combined across cond_list and windows
    if args.perm_trials:
        trials_early = load_trials_for_windows_conditions(args.subj, args.region, cond_list, baseline_windows)
        trials_late = load_trials_for_windows_conditions(args.subj, args.region, cond_list, test_windows)

        if not trials_early:
            print("Warning: No trial-level data found for baseline windows (combined conditions).")
        if not trials_late:
            print("Warning: No trial-level data found for test windows (combined conditions).")

        perpair_df = run_permutation_for_perpair(perpair_df, trials_late, trials_early, args, args.out_perpair,
                                                 window_label='timewindow')

    perpair_df.to_csv(args.out_perpair, index=False)
    print(f"Per-pair results saved to: {args.out_perpair}")

    # Determine significant pairs
    if args.perm_trials:
        sig_mask = perpair_df['perm_significant_bh'].astype(bool)
        sig_pairs = perpair_df[sig_mask][['ch1', 'ch2']].values.tolist()
    else:
        sig_mask = perpair_df['mean_late'] > perpair_df['mean_early']
        sig_pairs = perpair_df[sig_mask][['ch1', 'ch2']].values.tolist()

    print(f"Found {len(sig_pairs)} significant pairs (based on selected method)")

    sig_list_out = 'sig_pairs/' + str(args.subj)[2:-2] + '_sig_pairs.csv'
    if sig_pairs:
        sig_df = pd.DataFrame(sig_pairs, columns=['ch1', 'ch2'])
        sig_df.to_csv(sig_list_out, index=False)
        print(f"Significant pairs saved to: {sig_list_out}")
    else:
        print("No significant pairs to save.")


if __name__ == '__main__':
    main()
