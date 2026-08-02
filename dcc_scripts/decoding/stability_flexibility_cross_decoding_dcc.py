#!/usr/bin/env python
"""
DCC core for A4 — cross-decoding of the stability/flexibility subpopulations
(`docs/stability_flexibility_analysis_plan.md` §4).

Co-localization != shared CODE. A1/A2 can show the *same electrodes* are selective
for both stability (LWPC) and flexibility (LWPS); this job asks the
representation-level question those counting analyses cannot: do the "both"
electrodes carry ONE shared code or a mix of two orthogonal codes? It runs the
plan's four designs on pseudo-trials built from the pseudopopulation, with the
§0.8 confound controls (disjoint train/test, trial-count matching, per-condition
mean removal, label-permutation null) baked in:

  (0) within_block_decoding_baseline (Fig 9): decode congruency within
      low/high incongruent-proportion blocks and switchType within low/high
      switch-proportion blocks; the block difference is a neural cross-effect.
  (a) label transfer: train on stability, test on flexibility (and vice versa),
      SEPARATELY on the A1 both / S_only / F_only electrode groups. Prediction:
      only 'both' cross-decodes. Run raw AND per-condition-mean-removed.
  (b) set comparison: same label decoded within each electrode group.
  (c) temporal_generalization (Fig 10): train-time × test-time accuracy matrix,
      within a contrast and across contrasts (on the 'both' group).

Everything runs on the SAME long-format table the A1-A3 jobs use, assembled with
effect_measure='cluster' so each row's `hg` is the trial's HG time course. The A1
electrode groups come from `sfs.per_electrode_anova_labels` (contrast_mode=
'proportion'). On the SYNTHETIC path a ground-truth pseudopopulation with a KNOWN
shared-vs-orthogonal code is used (`cd._synthetic_cross_df`), which validates the
whole path and that the analysis discriminates the two codes.

Driven by `run_stability_flexibility_cross_decoding_dcc.py` (wrapped by
`sbatch_stability_flexibility_cross_decoding_dcc.sh`). Not run directly on the
cluster; call `main(args)` with a populated argument namespace.
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

from src.analysis.decoding import cross_decoding as cd

STAB, FLEX, BOTH = "#2c7fb8", "#d95f0e", "#31a354"

# A4 sits on the A1 (proportion) electrode definition; decoding needs the time
# course, so the df is assembled with the cluster effect measure.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cluster'


# ---------------------------------------------------------------------------
# serialization
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


def _strip_arrays(d):
    """Drop bulky ndarray fields (acc_folds/null/matrix) before JSON."""
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items()
                if not (isinstance(v, np.ndarray) and v.size > 32)}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'cross_decoding.json'), 'w') as f:
        json.dump(_json_safe(_strip_arrays(results)), f, indent=2)
    # temporal-generalization matrices as .npy for re-plotting
    for name, tg in results.get('temporal', {}).items():
        np.save(os.path.join(save_dir, f'tempgen_{name}.npy'), tg['matrix'])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _bar_null(ax, label, acc, null_mean, p, color):
    ax.bar([label], [acc], color=color)
    ax.axhline(0.5, ls='--', c='k', lw=1)
    ax.text(0, acc + 0.01, f"{acc:.2f}\np={p:.3f}", ha='center', va='bottom', fontsize=9)


def make_plots(results, save_dir):
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # A: within-block baseline (Fig 9) — congruency across inc-prop blocks
    wb = results['within_block']
    for col, (cname, res) in enumerate(wb.items()):
        a = ax[0, col] if col < 3 else None
        if a is None:
            break
        blocks = res['blocks']
        accs = [res['per_block'][b]['accuracy'] for b in blocks]
        ps = [res['per_block'][b]['p'] for b in blocks]
        a.bar([str(b) for b in blocks], accs,
              color=[STAB if 'cong' in cname else FLEX] * len(blocks))
        a.axhline(0.5, ls='--', c='k', lw=1)
        for i, (acc, p) in enumerate(zip(accs, ps)):
            a.text(i, acc + 0.01, f"p={p:.3f}", ha='center', fontsize=8)
        a.set(title=f"A4(0) Fig 9 · {cname}\n(decode within each block)",
              ylabel="accuracy", xlabel=res['block_col'], ylim=(0.4, 1.0))

    # C: label transfer per electrode group (raw vs mean-removed)
    lt = results['label_transfer']
    a = ax[0, 2]
    groups = list(lt.keys())
    x = np.arange(len(groups))
    raw = [lt[g]['stab_to_flex']['raw']['accuracy'] for g in groups]
    mr = [lt[g]['stab_to_flex']['mean_removed']['accuracy'] for g in groups]
    a.bar(x - 0.2, raw, 0.4, label='raw', color=BOTH)
    a.bar(x + 0.2, mr, 0.4, label='mean-removed', color="#7fbf7b")
    a.axhline(0.5, ls='--', c='k', lw=1)
    for i, g in enumerate(groups):
        a.text(i - 0.2, raw[i] + 0.01, f"p={lt[g]['stab_to_flex']['raw']['p']:.2f}",
               ha='center', fontsize=7)
    a.set_xticks(x); a.set_xticklabels(groups, rotation=15)
    a.set(title="A4(a) · stability→flexibility transfer\nby electrode group",
          ylabel="cross-decoding accuracy", ylim=(0.4, 1.0))
    a.legend(fontsize=8)

    # D/E/F: temporal generalization matrices
    tg = results.get('temporal', {})
    for col, (name, res) in enumerate(list(tg.items())[:3]):
        a = ax[1, col]
        im = a.imshow(res['matrix'], origin='lower', cmap='viridis',
                      vmin=0.4, vmax=max(0.6, np.nanmax(res['matrix'])))
        a.set(title=f"A4(c) Fig 10 · temporal gen\n{name}",
              xlabel="test time bin", ylabel="train time bin")
        fig.colorbar(im, ax=a, fraction=0.046)

    fig.tight_layout()
    fig_path = os.path.join(save_dir, 'cross_decoding_summary.png')
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(results, save_dir, meta):
    lines = ["=" * 72,
             "STABILITY vs FLEXIBILITY — A4 CROSS-DECODING",
             "=" * 72]
    for k, v in meta.items():
        lines.append(f"{k:>22}: {v}")

    lines += ["-" * 72, "A4(0) within-block decoding baseline (Fig 9):"]
    for cname, res in results['within_block'].items():
        bstr = "  ".join(
            f"{res['block_col']}={b}: acc={res['per_block'][b]['accuracy']:.3f} "
            f"(p={res['per_block'][b]['p']:.3f})" for b in res['blocks'])
        diff = res.get('block_difference')
        lines.append(f"   {cname}: {bstr}"
                     + (f"   Δ(block)={diff:+.3f}" if diff is not None else ""))

    lines += ["-" * 72,
              "A4(a) label transfer (train stability, test flexibility) by group:",
              "      prediction: only the 'both' group cross-decodes."]
    for g, res in results['label_transfer'].items():
        sf = res['stab_to_flex']; fs = res['flex_to_stab']
        lines.append(
            f"   [{g}] stab→flex raw acc={sf['raw']['accuracy']:.3f} (p={sf['raw']['p']:.3f}) | "
            f"mean-removed acc={sf['mean_removed']['accuracy']:.3f} (p={sf['mean_removed']['p']:.3f})")
        lines.append(
            f"        flex→stab raw acc={fs['raw']['accuracy']:.3f} (p={fs['raw']['p']:.3f}) | "
            f"n_channels={sf['raw']['n_channels']}")

    if results.get('set_comparison'):
        lines += ["-" * 72,
                  "A4(b) set comparison (same label decoded within each group):"]
        for label, per_set in results['set_comparison'].items():
            row = "  ".join(f"{g}: acc={r['accuracy']:.3f}(p={r['p']:.3f})"
                            for g, r in per_set.items())
            lines.append(f"   {label}: {row}")

    if results.get('temporal'):
        lines += ["-" * 72, "A4(c) temporal generalization (Fig 10):"]
        for name, res in results['temporal'].items():
            m = np.asarray(res['matrix'])
            diag = float(np.mean(np.diag(m)))
            off = float((m.sum() - np.trace(m)) / (m.size - len(m)))
            lines.append(f"   {name}: mean diagonal={diag:.3f}  mean off-diagonal={off:.3f}  "
                         f"({'sustained/stable' if off > 0.55 else 'diagonal/phasic'} code)")

    lines += ["=" * 72,
              "Reading: cross-decoding above chance (and surviving per-condition mean",
              "removal) on the 'both' group = a SHARED code; chance on 'both' while each",
              "process is individually decodable = ORTHOGONAL codes (segregation at the",
              "representational level). Mean-removed << raw = the transfer was a",
              "univariate offset, not a genuine multivariate code."]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# electrode-group derivation from the A1 labels
# ---------------------------------------------------------------------------
def _electrode_groups(df, alpha):
    from src.analysis.stats import stability_flexibility_segregation as sfs
    labels = sfs.per_electrode_anova_labels(
        df, alpha=alpha, contrast_mode=CONTRAST_MODE)
    S = (labels.S == 1); F = (labels.F == 1)
    groups = {
        'both': labels.loc[S & F, 'electrode'].tolist(),
        'S_only': labels.loc[S & ~F, 'electrode'].tolist(),
        'F_only': labels.loc[~S & F, 'electrode'].tolist(),
    }
    return labels, groups


def _interaction_groups(labels):
    """The FOUR interaction-defined electrode sets (possibly overlapping), keyed by
    the definition-group flag (CPC/SPS/CPS/SPC) so `cd.is_circular_decode` can name
    each set's double-dip cell. Used for the per-group within-block 2x2 that skips
    the diagonal (define==decode) cell."""
    return {flag: labels.loc[labels[flag] == 1, 'electrode'].tolist()
            for flag in ('CPC', 'SPS', 'CPS', 'SPC') if flag in labels.columns}


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(getattr(args, 'seed', 0))
    dec_kw = dict(n_per_cell=args.n_per_cell, n_pseudo=args.n_pseudo,
                  n_folds=args.n_folds, n_perm=args.n_perm)

    # 1. assemble the cluster-mode long df + electrode groups --------------------
    if args.data_source == 'synthetic':
        print(f"DATA SOURCE: synthetic ({args.synthetic_code} code) — validates the path "
              "and that A4 discriminates shared vs orthogonal codes")
        df = cd._synthetic_cross_df(code=args.synthetic_code, seed=getattr(args, 'seed', 0))
        elecs = sorted(df.electrode.unique())
        half = len(elecs) // 2
        groups = {'all': elecs, 'setA': elecs[:half], 'setB': elecs[half:]}
        a1_groups = {'both': elecs}          # the whole population plays 'both'
    else:
        print("DATA SOURCE: real epoched data")
        from src.analysis.utils.general_utils import (
            resolve_lab_root, resolve_electrodes_to_keep, load_HG_ev1_rescaled_per_subject)
        from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df
        LAB_root = resolve_lab_root(args.LAB_root)
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
        print(f"assembled cluster df: {len(df)} rows | {df.subject.nunique()} subjects | "
              f"{df.electrode.nunique()} electrodes")
        labels, a1_groups = _electrode_groups(df, args.alpha)
        labels.to_csv(os.path.join(args.save_dir, 'anova_labels.csv'), index=False)
        groups = dict(a1_groups)
        print("A1 electrode groups: "
              + "  ".join(f"{g}={len(v)}" for g, v in a1_groups.items()))

    results = {}

    # 2. (0) within-block decoding baseline (Fig 9) ------------------------------
    print("A4(0): within-block decoding baseline (Fig 9)")
    results['within_block'] = {
        'congruency (LWPC)': cd.within_block_decoding_baseline(
            df, contrast='congruency', rng=rng, **dec_kw),
        'switchType (LWPS)': cd.within_block_decoding_baseline(
            df, contrast='switchType', rng=rng, **dec_kw),
    }

    # 2b. within-block 2x2 restricted to each interaction-defined electrode group,
    #     SKIPPING the diagonal (define==decode) cell to avoid double-dipping.
    #     Only the OFF-diagonal (cross) cells are computed/kept -- e.g. the LWPC
    #     electrode set is decoded on switchType/switch_prop and the two cross
    #     cells, never on congruency/inc_prop (the interaction that defined it).
    if args.data_source != 'synthetic':
        print("A4(0b): per-group within-block 2x2 (diagonal define==decode cells ignored)")
        interaction_groups = _interaction_groups(labels)
        # the decode 2x2: (what you decode, how you split blocks)
        decode_cells = [('congruency', 'incongruent_proportion'),
                        ('congruency', 'switch_proportion'),
                        ('switchType', 'switch_proportion'),
                        ('switchType', 'incongruent_proportion')]
        per_group = {}
        for gflag, elset in interaction_groups.items():
            if len(elset) < args.min_group_size:
                print(f"     group '{gflag}' has {len(elset)} electrodes "
                      f"(< {args.min_group_size}); skipping")
                continue
            cells = {}
            for contrast, block_col in decode_cells:
                if cd.is_circular_decode(gflag, contrast, block_col):
                    continue                     # double-dipping: ignore this result
                cells[f'{contrast} by {block_col}'] = cd.within_block_decoding_baseline(
                    df, contrast=contrast, block_col=block_col, electrodes=elset,
                    rng=rng, **dec_kw)
            per_group[gflag] = dict(n_electrodes=len(elset),
                                    ignored_cell=cd.circular_decode_for_group(gflag),
                                    cells=cells)
        results['within_block_by_group'] = per_group

    # 3. (a) label transfer per electrode group, raw + mean-removed --------------
    print("A4(a): label transfer (stability<->flexibility) per electrode group")
    lt = {}
    for g, elset in a1_groups.items():
        if len(elset) < args.min_group_size:
            print(f"     group '{g}' has {len(elset)} electrodes (< {args.min_group_size}); skipping")
            continue
        lt[g] = dict(
            stab_to_flex=dict(
                raw=cd.cross_decode(df, 'stability', 'flexibility', electrodes=elset,
                                    strip_condition_means=False, rng=rng, **dec_kw),
                mean_removed=cd.cross_decode(df, 'stability', 'flexibility', electrodes=elset,
                                             strip_condition_means=True, rng=rng, **dec_kw)),
            flex_to_stab=dict(
                raw=cd.cross_decode(df, 'flexibility', 'stability', electrodes=elset,
                                    strip_condition_means=False, rng=rng, **dec_kw)))
    results['label_transfer'] = lt

    # 4. (b) set comparison (same label within each group) -----------------------
    print("A4(b): set comparison (same label decoded within each electrode set)")
    usable = {g: v for g, v in groups.items() if len(v) >= args.min_group_size}
    if len(usable) >= 2:
        results['set_comparison'] = {
            'congruency': cd.cross_decode(df, 'congruency', mode='set_transfer',
                                          electrode_sets=usable, rng=rng, **dec_kw)['per_set'],
            'switchType': cd.cross_decode(df, 'switchType', mode='set_transfer',
                                          electrode_sets=usable, rng=rng, **dec_kw)['per_set'],
        }

    # 5. (c) temporal generalization (Fig 10) on the 'both'/all group ------------
    print("A4(c): temporal generalization (Fig 10)")
    tg_set = a1_groups.get('both') or groups.get('all')
    tg_kw = dict(n_per_cell=args.n_per_cell, n_pseudo=max(30, args.n_pseudo // 2),
                 n_folds=max(2, args.n_folds // 2))
    if tg_set and len(tg_set) >= args.min_group_size:
        results['temporal'] = {
            'stability (within)': cd.temporal_generalization(
                df, 'stability', electrodes=tg_set, rng=rng, **tg_kw),
            'flexibility (within)': cd.temporal_generalization(
                df, 'flexibility', electrodes=tg_set, rng=rng, **tg_kw),
            'stability->flexibility (cross)': cd.temporal_generalization(
                df, 'stability', 'flexibility', electrodes=tg_set, rng=rng, **tg_kw),
        }

    # 6. persist + plot + summarize ----------------------------------------------
    save_results(results, args.save_dir)
    make_plots(results, args.save_dir)
    write_summary(results, args.save_dir, meta=dict(
        data_source=args.data_source,
        synthetic_code=getattr(args, 'synthetic_code', None),
        task=args.task, epochs_root_file=getattr(args, 'epochs_root_file', None),
        window=f"[{getattr(args, 'window_tmin', None)}, {getattr(args, 'window_tmax', None)}]s",
        n_per_cell=args.n_per_cell, n_pseudo=args.n_pseudo, n_folds=args.n_folds,
        n_perm=args.n_perm, alpha=getattr(args, 'alpha', None), save_dir=args.save_dir))
    return results
