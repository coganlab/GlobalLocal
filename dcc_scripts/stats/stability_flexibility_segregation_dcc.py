#!/usr/bin/env python
"""
DCC core for the stability vs. flexibility segregation analysis.

Assembles the long-format single-trial high-gamma table the analysis expects
from this project's epoched data, runs the joint-distribution analysis
(`src/analysis/stats/stability_flexibility_segregation.py`), and writes results
+ a summary figure to disk.

Driven by `run_stability_flexibility_segregation_dcc.py` (which is wrapped by
`sbatch_stability_flexibility_segregation_dcc.sh`). Not meant to be run directly
on the cluster; call `main(args)` with a populated argument namespace.

The analysis input `df` is long format, one row per (electrode, trial):
    subject, electrode, hg, congruency in {'c','i'}, switchType in {'s','r'}
We build it by window-averaging HG_ev1_rescaled over [tmin, tmax] and reading
`congruency` and `task_sequence` (-> switchType) from the epochs metadata.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json

# ---------------------------------------------------------------------------
# PATH SETUP (mirrors the other dcc_scripts entrypoints)
# ---------------------------------------------------------------------------
try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if os.path.exists("/hpc/home"):
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

from src.analysis.stats import stability_flexibility_segregation as sfs
from src.analysis.utils.general_utils import resolve_lab_root, resolve_electrodes_to_keep
STAB, FLEX = "#2c7fb8", "#d95f0e"



# ---------------------------------------------------------------------------
# long-format assembly from epoched data
# ---------------------------------------------------------------------------
def _window_indices(times, tmin, tmax):
    idx = np.where((times >= tmin) & (times <= tmax))[0]
    if idx.size == 0:
        raise ValueError(f"analysis window [{tmin}, {tmax}]s falls outside the "
                         f"epoch times [{times[0]:.3f}, {times[-1]:.3f}]s")
    return idx[0], idx[-1] + 1


def assemble_long_df(subjects_epochs, tmin, tmax, electrodes_to_keep=None):
    """Build the (electrode, trial) long table from per-subject Epochs.

    hg          = mean HG_ev1_rescaled over [tmin, tmax]s (one value / trial / channel)
    congruency  = metadata 'congruency'      ('c' / 'i')
    switchType  = metadata 'task_sequence'   ('s' / 'r'); 'n' (first-of-block) dropped
    electrode   = f"{subject}-{channel}"  (unique across subjects)
    """
    frames = []
    for sub, epochs in subjects_epochs.items():
        md = epochs.metadata
        if md is None or 'congruency' not in md.columns or 'task_sequence' not in md.columns:
            # only needed as a fallback; deferred so metadata that is already
            # complete never forces the (heavy) mne-backed import.
            from src.analysis.utils.epoch_metadata_utils import make_metadata_from_event_names
            md = make_metadata_from_event_names(epochs)

        cong = md['congruency'].to_numpy().astype(str)
        sw = md['task_sequence'].to_numpy().astype(str) if 'task_sequence' in md.columns \
            else md['switchType'].to_numpy().astype(str)

        times = epochs.times
        s_idx, e_idx = _window_indices(times, tmin, tmax)
        data = epochs.get_data()                       # (n_trials, n_channels, n_times)
        hg_win = np.nanmean(data[:, :, s_idx:e_idx], axis=2)   # (n_trials, n_channels)
        ch_names = list(epochs.ch_names)

        keep_trials = np.isin(sw, ['s', 'r']) & np.isin(cong, ['c', 'i'])
        if electrodes_to_keep is not None:
            wanted = electrodes_to_keep.get(sub, set())
            ch_idx = [i for i, ch in enumerate(ch_names) if ch in wanted]
        else:
            ch_idx = list(range(len(ch_names)))

        for ci in ch_idx:
            frames.append(pd.DataFrame(dict(
                subject=sub,
                electrode=f"{sub}-{ch_names[ci]}",
                hg=hg_win[keep_trials, ci],
                congruency=cong[keep_trials],
                switchType=sw[keep_trials])))

    if not frames:
        raise RuntimeError("assembled 0 electrodes - check subjects / ROI filter / window")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['hg'])
    return df


def make_synthetic_df(rho_true=0.4, n_subj=10, seed=0, gain_sd=0.5):
    """Ground-truth-controlled synthetic data for validating the pipeline/paths."""
    rng = np.random.default_rng(seed)
    cov = np.array([[0.4**2, rho_true*0.16], [rho_true*0.16, 0.4**2]])
    rows = []
    for s in range(n_subj):
        n_tr = int(rng.integers(280, 420))
        cong = rng.choice(['c', 'i'], n_tr)
        sw = rng.choice(['s', 'r'], n_tr)
        for e in range(int(rng.integers(15, 26))):
            gain = rng.lognormal(0, gain_sd)
            bx, by = rng.multivariate_normal([0, 0], cov)
            hg = (gain * (bx * (cong == 'i') + by * (sw == 's'))
                  + rng.normal(0, 1, n_tr) * gain)
            rows.append(pd.DataFrame(dict(subject=f"S{s:02d}", electrode=f"S{s:02d}-e{e}",
                                          hg=hg, congruency=cong, switchType=sw)))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_results(out, save_dir):
    """Write electrodes/labels/continuous tables + correlation/conjunction JSON."""
    os.makedirs(save_dir, exist_ok=True)
    out['electrodes'].to_csv(os.path.join(save_dir, 'electrodes.csv'), index=False)
    out['labels'].to_csv(os.path.join(save_dir, 'labels.csv'), index=False)
    out['continuous'].to_csv(os.path.join(save_dir, 'continuous.csv'), index=False)

    with open(os.path.join(save_dir, 'correlation.json'), 'w') as f:
        json.dump(_json_safe(out['correlation']), f, indent=2)

    k = out['conjunction']
    conj_json = dict(
        mh_odds_ratio=k['mh_odds_ratio'],
        cmh_pvalue=k['cmh'].pvalue, cmh_statistic=k['cmh'].statistic,
        homogeneity_pvalue=k['homogeneity'].pvalue,
        homogeneity_statistic=k['homogeneity'].statistic,
        pooled_table=k['pooled_table'],
        pooled_fisher_or=k['pooled_fisher_or'], pooled_fisher_p=k['pooled_fisher_p'])
    if 'or_95ci' in k:
        conj_json['mh_or_95ci'] = list(k['or_95ci'])
    with open(os.path.join(save_dir, 'conjunction.json'), 'w') as f:
        json.dump(_json_safe(conj_json), f, indent=2)
    k['per_subject'].to_csv(os.path.join(save_dir, 'conjunction_per_subject.csv'), index=False)


def write_summary(out, save_dir, meta):
    c, k = out['correlation'], out['conjunction']
    a = 0.05
    cont_dir = "shared core" if c['corr'] > 0 else "segregated"
    cont_sig = "significant" if c['p'] < a else "n.s."
    cat_dir = "shared core" if k['mh_odds_ratio'] > 1 else "segregated"
    cat_sig = "significant" if k['cmh'].pvalue < a else "n.s."
    lines = [
        "=" * 68,
        "STABILITY vs FLEXIBILITY SEGREGATION - SUMMARY",
        "=" * 68,
    ]
    for key, val in meta.items():
        lines.append(f"{key:>22}: {val}")
    lines += [
        "-" * 68,
        f"CONTINUOUS  ({c['method']}): corr = {c['corr']:+.4f}  p = {c['p']:.4g}",
        f"            n_electrodes = {c['n_electrodes']}  n_subjects = {c['n_subjects']}",
        f"            -> {cont_dir} ({cont_sig})",
        "-" * 68,
        f"CATEGORICAL (CMH): MH odds ratio = {k['mh_odds_ratio']:.4f}",
        f"            CMH p = {k['cmh'].pvalue:.4g}  homogeneity p = {k['homogeneity'].pvalue:.4g}",
        f"            -> {cat_dir} ({cat_sig})",
        "=" * 68,
        "Reading: corr<=0 / OR<1 -> segregation ; corr>0 / OR>1 -> shared core",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# summary figure
# ---------------------------------------------------------------------------
def make_summary_plots(out, save_dir, n_perm_plot=2000, seed=1):
    from scipy.stats import spearmanr
    elec, cont = out['electrodes'], out['continuous']
    labels, k = out['labels'], out['conjunction']
    c = out['correlation']

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # A: raw joint distribution
    ax[0, 0].scatter(elec.x, elec.y, s=14, alpha=.5, color="#444")
    ax[0, 0].axhline(0, color='k', lw=.6); ax[0, 0].axvline(0, color='k', lw=.6)
    ax[0, 0].set(title="Raw per-electrode (x, y)",
                 xlabel="x = stability d", ylabel="y = flexibility d")

    # B: residualized within-subject
    ax[0, 1].scatter(cont.x_resid, cont.y_resid, s=14, alpha=.5, color="#444")
    ax[0, 1].axhline(0, color='k', lw=.6); ax[0, 1].axvline(0, color='k', lw=.6)
    ax[0, 1].set(title=f"Residualized + within-subject centered\n"
                       f"corr={c['corr']:+.3f}, p={c['p']:.3g}",
                 xlabel="x_resid", ylabel="y_resid")

    # C: within-subject permutation null
    if len(cont) > 2:
        xr, yr = cont.x_resid.to_numpy(), cont.y_resid.to_numpy()
        subj = cont.subject.to_numpy()
        groups = [np.where(subj == s)[0] for s in np.unique(subj)]
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm_plot)
        for i in range(n_perm_plot):
            yp = yr.copy()
            for idx in groups:
                yp[idx] = yr[rng.permutation(idx)]
            null[i] = spearmanr(xr, yp)[0]
        ax[0, 2].hist(null, bins=40, color="#bbb")
        ax[0, 2].axvline(c['corr'], color="#d7191c", lw=2,
                         label=f"observed {c['corr']:+.3f}")
        ax[0, 2].legend()
    ax[0, 2].set(title=f"Within-subject null (p={c['p']:.3g})",
                 xlabel="correlation", ylabel="# permutations")

    # D: selectivity classes
    both = int(((labels.S == 1) & (labels.F == 1)).sum())
    so = int(((labels.S == 1) & (labels.F == 0)).sum())
    fo = int(((labels.S == 0) & (labels.F == 1)).sum())
    ne = int(((labels.S == 0) & (labels.F == 0)).sum())
    ax[1, 0].bar(["neither", "S only", "F only", "both"], [ne, so, fo, both],
                 color=["#ccc", STAB, FLEX, "#31a354"])
    ax[1, 0].set(title="Electrode selectivity classes", ylabel="# electrodes")

    # E: pooled 2x2
    pooled = np.asarray(k['pooled_table'])
    im = ax[1, 1].imshow(pooled, cmap="Blues")
    ax[1, 1].set_xticks([0, 1]); ax[1, 1].set_xticklabels(["F=1", "F=0"])
    ax[1, 1].set_yticks([0, 1]); ax[1, 1].set_yticklabels(["S=1", "S=0"])
    for (i, j), v in np.ndenumerate(pooled):
        ax[1, 1].text(j, i, int(v), ha="center", va="center",
                      color="white" if v > pooled.max()/2 else "black", fontsize=13)
    ax[1, 1].set(title=f"Pooled 2x2 (MH OR={k['mh_odds_ratio']:.2f}, "
                       f"p={k['cmh'].pvalue:.3g})")

    # F: per-subject P(F|S) vs base rate
    ps = k['per_subject'].copy()
    denom = (ps.both + ps.stab_only).replace(0, np.nan)
    ps['p_both_given_S'] = ps.both / denom
    tot = ps[['both', 'stab_only', 'flex_only', 'neither']].sum(1)
    ps['p_F_overall'] = (ps.both + ps.flex_only) / tot
    ax[1, 2].scatter(ps.p_F_overall, ps.p_both_given_S, s=40, color="#31a354")
    top = np.nanmax([ps.p_F_overall.max(), ps.p_both_given_S.max(), 0.05]) * 1.1
    ax[1, 2].plot([0, top], [0, top], 'k--', lw=1, label="independence")
    ax[1, 2].set(title="Per subject: P(F | S) vs base-rate P(F)",
                 xlabel="P(F) overall", ylabel="P(F | S)",
                 xlim=(0, top), ylim=(0, top)); ax[1, 2].legend()

    fig.tight_layout()
    fig_path = os.path.join(save_dir, 'segregation_summary.png')
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# QC / diagnostics figure (the step-by-step panels from the tutorial notebook)
# ---------------------------------------------------------------------------
def _naive_sensitivities(df):
    """Per-electrode (x, y) estimated from ALL trials (shared trial noise).
    Only used for the diagnostic naive-vs-disjoint comparison; the analysis
    itself always uses the disjoint-half estimator in `sfs`."""
    rows = []
    for (subj, elec), g in df.groupby(['subject', 'electrode']):
        x = sfs._cohens_d(g.loc[g.congruency == 'i', 'hg'],
                          g.loc[g.congruency == 'c', 'hg'])
        y = sfs._cohens_d(g.loc[g.switchType == 's', 'hg'],
                          g.loc[g.switchType == 'r', 'hg'])
        rows.append((x, y))
    return np.asarray(rows, float)


def make_diagnostic_plots(out, df, save_dir):
    """Save the notebook's QC panels so a production run also shows *why* the
    corrections were needed and that they did their job. Complements
    segregation_summary.png; writes segregation_diagnostics.png.

    Panels:
      row 0  data structure (unequal trials/electrodes, gain-driven HG spread)
      row 1  gain confound (responsiveness -> |x|,|y|) + naive-vs-disjoint corr
      row 2  residualization stages: raw -> resp-residualized -> +within-subj
      row 3  per-electrode permutation p's + FDR q-value conjunction scatter
    """
    from scipy.stats import pearsonr
    elec, cont, labels = out['electrodes'], out['continuous'], out['labels']

    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return pearsonr(a[m], b[m])[0] if m.sum() > 2 else np.nan

    fig, ax = plt.subplots(4, 3, figsize=(16, 18))

    # --- row 0: data structure ------------------------------------------------
    tr = df.groupby('subject').size()
    ax[0, 0].bar(tr.index.astype(str), tr.values, color="#555")
    ax[0, 0].set(title="Trials per subject", xlabel="subject", ylabel="# trials")
    ax[0, 0].tick_params(axis='x', labelrotation=90, labelsize=7)

    ne = df.groupby('subject')['electrode'].nunique()
    ax[0, 1].bar(ne.index.astype(str), ne.values, color="#777")
    ax[0, 1].set(title="Electrodes per subject", xlabel="subject", ylabel="# electrodes")
    ax[0, 1].tick_params(axis='x', labelrotation=90, labelsize=7)

    gstd = df.groupby('electrode')['hg'].std()
    ax[0, 2].hist(gstd.dropna().values, bins=30, color="#999")
    ax[0, 2].set(title="Per-electrode HG std\n(varies with gain / SNR)",
                 xlabel="HG std", ylabel="# electrodes")

    # --- row 1: gain confound + estimator check -------------------------------
    for a, col, color, name in [(ax[1, 0], 'x', STAB, "|x| stability"),
                                (ax[1, 1], 'y', FLEX, "|y| flexibility")]:
        m = np.isfinite(elec['resp']) & np.isfinite(elec[col])
        rr, cc = elec.loc[m, 'resp'].to_numpy(), elec.loc[m, col].abs().to_numpy()
        a.scatter(rr, cc, s=15, alpha=.5, color=color)
        if m.sum() > 2:
            b1, b0 = np.polyfit(rr, cc, 1)
            xs = np.linspace(rr.min(), rr.max(), 50)
            a.plot(xs, b0 + b1 * xs, color='k', lw=1.2)
        a.set(title=f"responsiveness -> {name}",
              xlabel="responsiveness (resp)", ylabel=name)

    naive = _naive_sensitivities(df)
    r_naive = _corr(naive[:, 0], naive[:, 1]) if len(naive) else np.nan
    r_disj = _corr(elec['x'].to_numpy(), elec['y'].to_numpy())
    ax[1, 2].bar(["naive\n(same trials)", "disjoint\n(pipeline)"],
                 [r_naive, r_disj], color=["#bbb", "#31a354"])
    ax[1, 2].axhline(0, color='k', lw=.6)
    for xi, v in enumerate([r_naive, r_disj]):
        if np.isfinite(v):
            ax[1, 2].text(xi, v, f"{v:+.3f}", ha='center',
                          va='bottom' if v >= 0 else 'top')
    ax[1, 2].set(title="corr(x, y): shared trial noise removed\n"
                       "by the disjoint-half estimator",
                 ylabel="Pearson corr(x, y)")

    # --- row 2: residualization stages ---------------------------------------
    stages = [(elec, 'x', 'y', "1. raw x, y"),
              (cont, 'x1', 'y1', "2. after responsiveness residualization"),
              (cont, 'x_resid', 'y_resid', "3. + within-subject centering")]
    for a, (src, cx, cy, ttl) in zip(ax[2], stages):
        if cx in src.columns and cy in src.columns and len(src):
            a.scatter(src[cx], src[cy], s=15, alpha=.5, color="#444")
        a.axhline(0, color='k', lw=.6); a.axvline(0, color='k', lw=.6)
        a.set(title=ttl, xlabel=cx, ylabel=cy)

    # --- row 3: per-electrode permutation p's + FDR q conjunction ------------
    ax[3, 0].hist(labels['p_cong'].dropna(), bins=20, color=STAB, alpha=.7)
    ax[3, 0].set(title="congruency permutation p", xlabel="p_cong", ylabel="# elec")
    ax[3, 1].hist(labels['p_switch'].dropna(), bins=20, color=FLEX, alpha=.7)
    ax[3, 1].set(title="switch permutation p", xlabel="p_switch", ylabel="# elec")

    sc = ax[3, 2].scatter(labels['q_cong'], labels['q_switch'], s=18, alpha=.6,
                          c=(labels['S'] + 2 * labels['F']), cmap="viridis")
    ax[3, 2].axvline(0.05, color='k', ls='--', lw=1)
    ax[3, 2].axhline(0.05, color='k', ls='--', lw=1)
    ax[3, 2].set(title="FDR q-values (dashed = alpha)",
                 xlabel="q_cong (stability)", ylabel="q_switch (flexibility)",
                 xlim=(-.02, 1.02), ylim=(-.02, 1.02))

    fig.tight_layout()
    fig_path = os.path.join(save_dir, 'segregation_diagnostics.png')
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    LAB_root = resolve_lab_root(args.LAB_root)
        
    print(f"LAB_root: {LAB_root}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. build the long-format df -------------------------------------------------
    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (pipeline / path validation)")
        df = make_synthetic_df(rho_true=args.synthetic_rho)
    else:
        print("DATA SOURCE: real epoched data")
        from src.analysis.utils.general_utils import load_HG_ev1_rescaled_per_subject
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep)

    print(f"assembled df: {len(df)} rows | {df.subject.nunique()} subjects | "
          f"{df.electrode.nunique()} electrodes")
    df.to_csv(os.path.join(args.save_dir, 'long_df.csv'), index=False)

    # 2. run the analysis ---------------------------------------------------------
    out = sfs.run_joint_distribution_analysis(
        df, responsiveness=args.responsiveness,
        n_splits=args.n_splits, n_perm_corr=args.n_perm_corr,
        n_perm_label=args.n_perm_label, alpha=args.alpha, min_elec=args.min_elec)

    # 3. persist ------------------------------------------------------------------
    save_results(out, args.save_dir)
    make_summary_plots(out, args.save_dir)
    make_diagnostic_plots(out, df, args.save_dir)
    write_summary(out, args.save_dir, meta=dict(
        data_source=args.data_source, task=args.task,
        epochs_root_file=args.epochs_root_file,
        n_subjects_requested=len(args.subjects),
        window=f"[{args.window_tmin}, {args.window_tmax}]s",
        electrodes=args.electrodes,
        rois=(list(args.rois_dict.keys()) if args.rois_dict else 'all'),
        n_splits=args.n_splits, n_perm_corr=args.n_perm_corr,
        n_perm_label=args.n_perm_label, save_dir=args.save_dir))
    return out
