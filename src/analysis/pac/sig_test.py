# ...existing code...
import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# base folder where the coherence summary CSVs live (edit if needed)
BASE_DIR = "/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/coh_timewindow"

def load_csv(path):
    return pd.read_csv(path)

def build_filename(subject, region, condition, tstart, tend):
    """
    Build filename like:
    coherence_D0063_acc_stimulus_c_(-0.5, 0.0)_summary.csv
    
    Args:
        subject: A single subject ID (string)
        region: Brain region
        condition: Experimental condition
        tstart: Start time
        tend: End time
    Returns:
        Full path to the file
    """
    # Ensure subject is a single string, not a list or joined string
    if isinstance(subject, (list, tuple)):
        raise ValueError("Subject should be a single ID string, not a list or tuple")
    
    # Remove any spaces that might be in the subject ID
    subject = str(subject).strip()
    
    # keep time formatting similar to example: "(-0.5, 0.0)"
    tstart_s = str(tstart)
    tend_s = str(tend)
    fname = f"coherence_{subject}_{region}_{condition}_({tstart_s}, {tend_s})_summary.csv"
    return os.path.join(BASE_DIR, fname)

def load_multiple_subjects_data(subjects, region, condition, tstart, tend):
    """
    Load data for multiple subjects and combine them.
    
    Args:
        subjects: List of subject IDs (e.g., ['D0057', 'D0059', 'D0063'])
        region: Brain region
        condition: Experimental condition
        tstart: Start time
        tend: End time
        
    Returns:
        Combined DataFrame with an additional 'subject' column
    """
    all_data = []
    found_subjects = []
    
    # Process each subject individually
    for subject in subjects:
        # Clean the subject ID (remove any extra spaces)
        subject = str(subject).strip()
        try:
            filepath = build_filename(subject, region, condition, tstart, tend)
            print(f"Trying file for subject {subject}: {filepath}")
            if os.path.exists(filepath):
                df = load_csv(filepath)
                nrows = len(df.index)
                print(f"Loaded {nrows} rows for subject {subject} from {filepath}")
                df['subject'] = subject  # Add subject identifier
                all_data.append(df)
                found_subjects.append(subject)
            else:
                print(f"Warning: File not found for subject {subject}: {filepath}")
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
    
    if not all_data:
        raise SystemExit("No valid data files found for any subject")
    
    if len(found_subjects) != len(subjects):
        missing = set(subjects) - set(found_subjects)
        print(f"Warning: Data missing for subjects: {', '.join(missing)}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Successfully loaded data for {len(found_subjects)} subjects: {', '.join(found_subjects)}")
    # also return found_subjects so caller can report which were used
    return combined_data, found_subjects

def perpair_means(df):
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
        g = df.groupby(['ch1','ch2'], as_index=False).agg(
            mean_coh=('coh_mean','mean'),
            n=('coh_mean','size')
        )
        g['n_subjects'] = 1
    return g

def compute_perpair_and_overall(A, B):
    Aagg = perpair_means(A)
    Bagg = perpair_means(B)
    merged = pd.merge(Aagg, Bagg, on=['ch1','ch2'], how='inner', suffixes=('_A','_B'))
    if merged.empty:
        raise SystemExit("no matching channel pairs between A and B")
    merged['mean_diff'] = merged['mean_coh_A'] - merged['mean_coh_B']
    # overall paired t-test across pairs
    x = merged['mean_coh_A'].values
    y = merged['mean_coh_B'].values
    n_pairs = len(x)
    if n_pairs >= 2:
        t_stat, p_val = stats.ttest_rel(x, y, nan_policy='omit')
    else:
        t_stat, p_val = np.nan, np.nan
    overall = {
        'n_pairs': n_pairs,
        'mean_A': float(np.nanmean(x)) if n_pairs>0 else np.nan,
        'mean_B': float(np.nanmean(y)) if n_pairs>0 else np.nan,
        'mean_diff': float(np.nanmean(x-y)) if n_pairs>0 else np.nan,
        't_stat': t_stat, 'p_val': p_val
    }
    # prepare per-pair output columns
    perpair_out = merged.rename(columns={
        'mean_coh_A':'mean_A','mean_coh_B':'mean_B'
    })
    # ensure n_A/n_B columns from original aggregation
    # Aagg produced column 'n' -> after merge they are 'n_A' and 'n_B'
    if 'n_A' not in perpair_out.columns and 'n' in Aagg.columns:
        perpair_out['n_A'] = merged['n_A'] if 'n_A' in merged.columns else merged.get('n_A', np.nan)
    if 'n_B' not in perpair_out.columns and 'n' in Bagg.columns:
        perpair_out['n_B'] = merged['n_B'] if 'n_B' in merged.columns else merged.get('n_B', np.nan)
    # fallback: if still missing, try to infer from merged
    perpair_out['n_A'] = perpair_out.get('n_A', merged.get('n_A', np.nan))
    perpair_out['n_B'] = perpair_out.get('n_B', merged.get('n_B', np.nan))
    perpair_out = perpair_out[['ch1','ch2','n_A','n_B','mean_A','mean_B','mean_diff']]
    return perpair_out, overall

def plot_results(perpair_df, overall_res, out_fig, labelA='A', labelB='B', alpha=0.05, annotate_top=10):
    dfp = perpair_df.copy()
    dfp = dfp.sort_values('mean_diff', key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 2]})
    # If permutation significance column exists, color significant pairs differently
    if 'perm_significant' in dfp.columns:
        sig_mask = dfp['perm_significant'].astype(bool).values
        colors = ['C3' if sig else 'gray' for sig in sig_mask]
    else:
        sig_mask = np.zeros(len(dfp), dtype=bool)
        colors = ['gray'] * len(dfp)

    bars = ax1.bar(range(len(dfp)), dfp['mean_diff'], color=colors)
    ax1.set_ylabel('mean_A - mean_B (per pair)')

    # annotate only the single pair with the largest absolute difference
    if len(dfp) > 0:
        # index 0 after sorting by absolute mean_diff is the largest
        i = 0
        y = float(dfp.loc[i, 'mean_diff'])
        # place label slightly above the bar (or below if negative) to avoid overlap
        offset = (abs(y) + 1e-6) * 0.05  # 5% of value magnitude
        if abs(y) < 1e-6:
            offset = 0.01  # small visible offset for near-zero bars
        ypos = y + offset if y >= 0 else y - offset
        txt = f"{dfp.loc[i,'ch1']}-{dfp.loc[i,'ch2']}\nΔ={y:.4f}"
        va = 'bottom' if y >= 0 else 'top'
        ax1.text(i, ypos, txt, rotation=90, fontsize=8, ha='center', va=va,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    ax1.axhline(0, color='black', linewidth=0.6)

    # add legend for significance if present
    if sig_mask.any():
        ax1.plot([], [], color='C3', label='perm-significant (p<{:.3f})'.format(alpha))
        ax1.plot([], [], color='gray', label='not significant')
        ax1.legend(loc='upper right', fontsize=8)

    # overall paired scatter across pairs (same as before)
    x = perpair_df['mean_A'].values
    y = perpair_df['mean_B'].values
    n = len(x)
    # For scatter plot, mark significant pairs with a thicker/different marker
    if 'perm_significant' in perpair_df.columns:
        sig_idx = perpair_df['perm_significant'].astype(bool).values
    else:
        sig_idx = np.zeros(len(perpair_df), dtype=bool)

    # plot non-significant points
    ax2.plot([0]*n, x, 'o', color='C0', label=labelA, markersize=5, alpha=0.8)
    ax2.plot([1]*n, y, 'o', color='C1', label=labelB, markersize=5, alpha=0.8)
    for i, (xi, yi) in enumerate(zip(x, y)):
        lw = 1.5 if sig_idx[i] else 0.6
        col = 'C3' if sig_idx[i] else 'gray'
        ax2.plot([0,1],[xi, yi], color=col, linewidth=lw, alpha=0.7)
    # overlay larger markers for significant points
    if sig_idx.any():
        ax2.plot([0]*n, x, 'o', color='C0', markersize=0)  # ensure legend entry
        ax2.plot([1]*n, y, 'o', color='C1', markersize=0)
        ax2.scatter(np.zeros_like(x[sig_idx]), x[sig_idx], s=40, facecolors='none', edgecolors='C3', label='perm-significant')
        ax2.scatter(np.ones_like(y[sig_idx]), y[sig_idx], s=40, facecolors='none', edgecolors='C3')
    ax2.set_xlim(-0.5,1.5)
    ax2.set_xticks([0,1]); ax2.set_xticklabels([labelA, labelB])
    title = f"across-pairs paired (n_pairs={overall_res['n_pairs']})"
    if not np.isnan(overall_res['t_stat']):
        title += f"  t={overall_res['t_stat']:.3f}  p={overall_res['p_val']:.3g}"
    ax2.set_title(title)
    ax2.legend()

    # add a small summary box on the right with full labels and means
    summary_text = (
        f"A: {labelA}\n"
        f"  mean={overall_res['mean_A']:.4f}\n"
        f"B: {labelB}\n"
        f"  mean={overall_res['mean_B']:.4f}\n"
        f"n_pairs={overall_res['n_pairs']}\n"
    )
    if not np.isnan(overall_res['t_stat']):
        summary_text += f"t={overall_res['t_stat']:.3f}  p={overall_res['p_val']:.3g}\n"
    ax2.text(1.02, 0.5, summary_text, transform=ax2.transAxes, fontsize=8, va='center', ha='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.tight_layout(rect=[0,0,0.95,1.0])
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

def get_built_or_explicit_file(explicit, subj, region, cond, tstart, tend):
    """
    Return explicit if provided; otherwise build from components (all components must be non-None).
    """
    if explicit:
        return explicit
    if None in (subj, region, cond, tstart, tend):
        return None
    return build_filename(subj, region, cond, tstart, tend)

def make_label(subj, region, cond, tstart, tend, fname=None):
        if subj or region or cond:
            # Handle list of subjects
            if isinstance(subj, list):
                if len(subj) <= 3:
                    subj_str = '+'.join(subj)
                else:
                    subj_str = f"{len(subj)} subjects ({subj[0]}...{subj[-1]})" 
            else:
                subj_str = str(subj)
            
            parts = [p for p in [subj_str, region, cond] if p]
            time_part = f"({tstart}, {tend})" if (tstart is not None and tend is not None) else ""
            return "_".join(parts) + ("\n" + time_part if time_part else "")
        return os.path.basename(fname) if fname else "Unknown"

def main():
    p = argparse.ArgumentParser(description="paired test across channel-pairs (use pairs as samples)")
    # legacy explicit file args (keep supported)
    p.add_argument('--fileA', dest='opt_fileA', help='path to first CSV (optional)')
    p.add_argument('--fileB', dest='opt_fileB', help='path to second CSV (optional)')
    p.add_argument('fileA_pos', nargs='?', help=argparse.SUPPRESS)
    p.add_argument('fileB_pos', nargs='?', help=argparse.SUPPRESS)

    # component-based construction for file A
    # allow either a single occurrence with multiple values or repeated --subjA flags
    p.add_argument('--subjA', nargs='+', action='append', help='one or more subjects for A e.g. D0063 D0064 (can repeat)')
    p.add_argument('--regionA', help='brain region for A e.g. acc')
    p.add_argument('--condA', help='condition for A e.g. stimulus_c')
    p.add_argument('--tstartA', help='time window start for A e.g. -0.5')
    p.add_argument('--tendA', help='time window end for A e.g. 0.0')

    # component-based construction for file B
    # allow either a single occurrence with multiple values or repeated --subjB flags
    p.add_argument('--subjB', nargs='+', action='append', help='one or more subjects for B (can repeat)')
    p.add_argument('--regionB', help='brain region for B')
    p.add_argument('--condB', help='condition for B')
    p.add_argument('--tstartB', help='time window start for B')
    p.add_argument('--tendB', help='time window end for B')

    p.add_argument('-o','--out_perpair', default='perpair_results.csv')
    p.add_argument('-O','--out_overall', default='overall_results.csv')
    p.add_argument('-f','--out_fig', default='ttest_across_pairs.png')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--annotate_top', type=int, default=10)
    p.add_argument('--perm_trials', action='store_true',
                   help='If set, read per-trial pkl files and run permutation test per pair')
    p.add_argument('--n_perm', type=int, default=200,
                   help='Number of permutations when --perm_trials is enabled')
    args = p.parse_args()

    # normalize subjA/subjB: argparse with action='append' + nargs='+' gives list-of-lists
    def _flatten_subj(x):
        if x is None:
            return None
        # x can be [['D0057','D0059']] or [['D0057'], ['D0059']] or ['D0057','D0059']
        if isinstance(x, list):
            # if inner elements are lists, flatten
            if any(isinstance(el, (list, tuple)) for el in x):
                flat = []
                for el in x:
                    if isinstance(el, (list, tuple)):
                        flat.extend([str(s).strip() for s in el])
                    else:
                        flat.append(str(el).strip())
                return [s for s in flat if s]
            else:
                return [str(s).strip() for s in x if s]
        # otherwise, return as single-element list
        return [str(x).strip()]

    args.subjA = _flatten_subj(args.subjA)
    args.subjB = _flatten_subj(args.subjB)

    if args.opt_fileA or args.fileA_pos:
        A = load_csv(args.opt_fileA if args.opt_fileA else args.fileA_pos)
        foundA = None
    elif args.subjA and args.regionA and args.condA and args.tstartA and args.tendA:
        A, foundA = load_multiple_subjects_data(
            args.subjA, args.regionA, args.condA, args.tstartA, args.tendA
        )
    else:
        raise SystemExit("must provide explicit file path or full component parameters (subjects, region, condition, tstart, tend) for A")

    if args.opt_fileB or args.fileB_pos:
        B = load_csv(args.opt_fileB if args.opt_fileB else args.fileB_pos)
        foundB = None
    elif args.subjB and args.regionB and args.condB and args.tstartB and args.tendB:
        B, foundB = load_multiple_subjects_data(
            args.subjB, args.regionB, args.condB, args.tstartB, args.tendB
        )
    else:
        raise SystemExit("must provide explicit file path or full component parameters (subjects, region, condition, tstart, tend) for B")
    for col in ['ch1','ch2','coh_mean']:
        if col not in A.columns or col not in B.columns:
            raise SystemExit(f"need column: ch1, ch2, coh_mean")

    labelA = make_label(args.subjA, args.regionA, args.condA, args.tstartA, args.tendA)
    labelB = make_label(args.subjB, args.regionB, args.condB, args.tstartB, args.tendB)

    # report what was loaded
    if isinstance(foundA, list):
        print(f"A: loaded subjects: {', '.join(foundA)} (total rows: {len(A)})")
    else:
        print(f"A: loaded explicit file (total rows: {len(A)})")

    if isinstance(foundB, list):
        print(f"B: loaded subjects: {', '.join(foundB)} (total rows: {len(B)})")
    else:
        print(f"B: loaded explicit file (total rows: {len(B)})")

    perpair_df, overall = compute_perpair_and_overall(A, B)

    # If requested, run permutation test using per-trial pickle files produced by theta_connect
    if args.perm_trials:
        def pkl_path_from_summary(csv_path):
            if csv_path.endswith('_summary.csv'):
                return csv_path.replace('_summary.csv', '_trials.pkl')
            # fallback: append
            return csv_path + '_trials.pkl'

        def load_trials_for_subjects(subjects, region, cond, tstart, tend):
            """Load trial-level mean values (mean across frequencies) for each (ch1,ch2).

            Returns a dict mapping (ch1,ch2) -> list of trial-mean scalars (may span subjects)
            """
            trials_map = {}
            if subjects is None:
                return trials_map
            for subj in subjects:
                subj = str(subj).strip()
                try:
                    csvp = build_filename(subj, region, cond, tstart, tend)
                    pklp = pkl_path_from_summary(csvp)
                    if not os.path.exists(pklp):
                        print(f"Warning: pkl not found for {subj}: {pklp}")
                        continue
                    d = pd.read_pickle(pklp)
                    # d expected: rows per (ch1,ch2) with column 'coh_trials'
                    for _, row in d.iterrows():
                        key = (row['ch1'], row['ch2'])
                        coh_trials = row.get('coh_trials')
                        if coh_trials is None:
                            continue
                        arr = np.array(coh_trials)
                        # heuristics: determine trial axis
                        if arr.ndim == 1:
                            # single-spectrum -> treat as one trial
                            trial_vals = np.atleast_1d(arr.mean())
                        elif arr.ndim == 2:
                            # guess trials are along axis 0 if first dim >= second dim
                            if arr.shape[0] >= arr.shape[1]:
                                # (n_trials, n_freqs)
                                trial_means = arr.mean(axis=1)
                            else:
                                # (n_freqs, n_times) or (n_freqs, n_trials)
                                # interpret second axis as trials
                                trial_means = arr.mean(axis=0)
                            trial_vals = trial_means
                        else:
                            # collapse all but one axis then take mean per trial-like axis
                            # fallback: mean across last axis
                            trial_vals = arr.mean(axis=-1).ravel()

                        trials_map.setdefault(key, []).extend(list(np.atleast_1d(trial_vals)))
                except Exception as e:
                    print(f"Error reading pkl for {subj}: {e}")
            return trials_map

        print("Loading per-trial pkls for A and B (this may take a bit)...")
        trials_A = load_trials_for_subjects(args.subjA, args.regionA, args.condA, args.tstartA, args.tendA) if args.subjA else {}
        trials_B = load_trials_for_subjects(args.subjB, args.regionB, args.condB, args.tstartB, args.tendB) if args.subjB else {}

        # perform permutation per pair
        perm_pvals = []
        perm_sign = []
        perm_nA = []
        perm_nB = []
        rng = np.random.default_rng()
        for _, row in perpair_df.iterrows():
            key = (row['ch1'], row['ch2'])
            valsA = np.array(trials_A.get(key, []), dtype=float)
            valsB = np.array(trials_B.get(key, []), dtype=float)
            nA = len(valsA)
            nB = len(valsB)
            perm_nA.append(nA)
            perm_nB.append(nB)
            if nA == 0 or nB == 0:
                perm_pvals.append(np.nan)
                perm_sign.append(False)
                continue
            obs_diff = float(np.nanmean(valsA) - np.nanmean(valsB))
            pooled = np.concatenate([valsA, valsB])
            n = len(pooled)
            if n <= 1:
                perm_pvals.append(np.nan)
                perm_sign.append(False)
                continue
            perm_diffs = []
            for _ in range(args.n_perm):
                perm = rng.permutation(pooled)
                g1 = perm[:nA]
                g2 = perm[nA: nA + nB]
                perm_diffs.append(np.nanmean(g1) - np.nanmean(g2))
            perm_diffs = np.array(perm_diffs)
            # two-sided p-value: proportion of perm diffs with abs >= abs(obs)
            pval = np.mean(np.abs(perm_diffs) >= abs(obs_diff))
            perm_pvals.append(float(pval))
            perm_sign.append(bool(pval < args.alpha))

        perpair_df['perm_nA'] = perm_nA
        perpair_df['perm_nB'] = perm_nB
        perpair_df['perm_pval'] = perm_pvals
        perpair_df['perm_significant'] = perm_sign
        # save separate list of significant pairs
        sig_pairs = perpair_df[perpair_df['perm_significant']]
        sig_out = os.path.splitext(args.out_perpair)[0] + '_perm_significant.csv'
        sig_pairs.to_csv(sig_out, index=False)
        print(f"Permutation results appended; significant pairs saved to {sig_out}")
    
    # Add number of subjects to the results
    if 'subject' in A.columns:
        overall['n_subjects_A'] = A['subject'].nunique()
    else:
        overall['n_subjects_A'] = 1
        
    if 'subject' in B.columns:
        overall['n_subjects_B'] = B['subject'].nunique()
    else:
        overall['n_subjects_B'] = 1
    
    perpair_df.to_csv(args.out_perpair, index=False)
    pd.DataFrame([overall]).to_csv(args.out_overall, index=False)
    plot_results(perpair_df, overall, args.out_fig, labelA=labelA, labelB=labelB, alpha=args.alpha, annotate_top=args.annotate_top)

    print("per-pair results ->", args.out_perpair)
    print("overall results ->", args.out_overall)
    print("figure ->", args.out_fig)

if __name__ == '__main__':
    main()
