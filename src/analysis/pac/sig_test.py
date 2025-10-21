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
    """
    # keep time formatting similar to example: "(-0.5, 0.0)"
    tstart_s = str(tstart)
    tend_s = str(tend)
    fname = f"coherence_{subject}_{region}_{condition}_({tstart_s}, {tend_s})_summary.csv"
    return os.path.join(BASE_DIR, fname)

def perpair_means(df):
    # 聚合每个 (ch1,ch2)：若有多行则取均值，并记录样本数
    g = df.groupby(['ch1','ch2'], as_index=False).agg(
        mean_coh=('coh_mean','mean'),
        n=('coh_mean','size')
    )
    return g

def compute_perpair_and_overall(A, B):
    Aagg = perpair_means(A)
    Bagg = perpair_means(B)
    merged = pd.merge(Aagg, Bagg, on=['ch1','ch2'], how='inner', suffixes=('_A','_B'))
    if merged.empty:
        raise SystemExit("没有匹配的 (ch1,ch2) 对 —— 无法进行 across-pairs 检验。")
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

    bars = ax1.bar(range(len(dfp)), dfp['mean_diff'], color='gray')
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

    # overall paired scatter across pairs (same as before)
    x = perpair_df['mean_A'].values
    y = perpair_df['mean_B'].values
    n = len(x)
    ax2.plot([0]*n, x, 'o', color='C0', label=labelA)
    ax2.plot([1]*n, y, 'o', color='C1', label=labelB)
    for xi, yi in zip(x, y):
        ax2.plot([0,1],[xi, yi], color='gray', linewidth=0.6, alpha=0.6)
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

def make_label(subj, region, cond, tstart, tend, fname):
        if subj or region or cond:
            parts = [p for p in [subj, region, cond] if p]
            time_part = f"({tstart}, {tend})" if (tstart is not None and tend is not None) else ""
            return "_".join(parts) + ("\n" + time_part if time_part else "")
        return os.path.basename(fname)

def main():
    p = argparse.ArgumentParser(description="paired test across channel-pairs (use pairs as samples)")
    # legacy explicit file args (keep supported)
    p.add_argument('--fileA', dest='opt_fileA', help='path to first CSV (optional)')
    p.add_argument('--fileB', dest='opt_fileB', help='path to second CSV (optional)')
    p.add_argument('fileA_pos', nargs='?', help=argparse.SUPPRESS)
    p.add_argument('fileB_pos', nargs='?', help=argparse.SUPPRESS)

    # component-based construction for file A
    p.add_argument('--subjA', help='subject for A e.g. D0063')
    p.add_argument('--regionA', help='brain region for A e.g. acc')
    p.add_argument('--condA', help='condition for A e.g. stimulus_c')
    p.add_argument('--tstartA', help='time window start for A e.g. -0.5')
    p.add_argument('--tendA', help='time window end for A e.g. 0.0')

    # component-based construction for file B
    p.add_argument('--subjB', help='subject for B')
    p.add_argument('--regionB', help='brain region for B')
    p.add_argument('--condB', help='condition for B')
    p.add_argument('--tstartB', help='time window start for B')
    p.add_argument('--tendB', help='time window end for B')

    p.add_argument('-o','--out_perpair', default='perpair_results.csv')
    p.add_argument('-O','--out_overall', default='overall_results.csv')
    p.add_argument('-f','--out_fig', default='ttest_across_pairs.png')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--annotate_top', type=int, default=10)
    args = p.parse_args()

    # determine fileA/fileB: explicit argument (cli) has priority; otherwise positional; otherwise built from components
    fileA_explicit = args.opt_fileA if args.opt_fileA else args.fileA_pos
    fileB_explicit = args.opt_fileB if args.opt_fileB else args.fileB_pos

    fileA = get_built_or_explicit_file(
        fileA_explicit,
        args.subjA, args.regionA, args.condA, args.tstartA, args.tendA
    )
    fileB = get_built_or_explicit_file(
        fileB_explicit,
        args.subjB, args.regionB, args.condB, args.tstartB, args.tendB
    )

    if not fileA or not fileB:
        raise SystemExit("必须提供两个输入文件：要么显式路径（位置参数或 --fileA/--fileB），要么为 A/B 两个文件分别提供 --subjX --regionX --condX --tstartX --tendX。")

    # sanity: if built path, show which file used
    if (not os.path.isabs(fileA)) and fileA.startswith(BASE_DIR):
        pass
    # load
    if not os.path.exists(fileA):
        raise SystemExit(f"输入文件 A 未找到: {fileA}")
    if not os.path.exists(fileB):
        raise SystemExit(f"输入文件 B 未找到: {fileB}")

    A = load_csv(fileA); B = load_csv(fileB)
    for col in ['ch1','ch2','coh_mean']:
        if col not in A.columns or col not in B.columns:
            raise SystemExit(f"两个文件都必须包含列: ch1, ch2, coh_mean")

    labelA = make_label(args.subjA, args.regionA, args.condA, args.tstartA, args.tendA, fileA)
    labelB = make_label(args.subjB, args.regionB, args.condB, args.tstartB, args.tendB, fileB)

    perpair_df, overall = compute_perpair_and_overall(A, B)
    perpair_df.to_csv(args.out_perpair, index=False)
    pd.DataFrame([overall]).to_csv(args.out_overall, index=False)
    plot_results(perpair_df, overall, args.out_fig, labelA=labelA, labelB=labelB, alpha=args.alpha, annotate_top=args.annotate_top)

    print("per-pair results ->", args.out_perpair)
    print("overall results ->", args.out_overall)
    print("figure ->", args.out_fig)

if __name__ == '__main__':
    main()
# ...existing code...