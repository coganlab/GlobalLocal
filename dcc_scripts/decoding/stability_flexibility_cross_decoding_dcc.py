#!/usr/bin/env python
"""
DCC core for A4 — cross-decoding of the stability/flexibility subpopulations
(`docs/analysis_guide.md` §17).

Co-localization != shared CODE. A1/A2 can show the *same electrodes* are selective
for both stability (LWPC) and flexibility (LWPS); this job asks the
representation-level question those counting analyses cannot: do the "both"
electrodes carry ONE shared code or a mix of two orthogonal codes?

It runs on the **ordinary decoding pipeline**. The ROI LabeledArray is already a
cross-subject pseudopopulation (subjects are NaN-padded to the per-condition max
and concatenated along the channel axis); `Decoder.cv_cm_jim_window_shuffle`
supplies disjoint train/test folds, a refit shuffle null, and time-resolved
accuracy traces that the usual `time_perm_cluster` machinery corrects across
windows. All A4 adds is a second label vector (`labels_test`) so the classifier is
trained on one contrast and scored against another.

Designs:
  (0) within-block baseline (Fig 9): decode congruency within low/high
      incongruent-proportion blocks and switchType within low/high
      switch-proportion blocks; the block difference is a neural cross-effect.
      Per interaction-defined electrode group, the diagonal (define == decode)
      cell is SKIPPED — see `cd.is_circular_decode`.
  (a) label transfer: train on stability, test on flexibility (and vice versa),
      SEPARATELY on the A1 both / S_only / F_only groups. Prediction: only
      'both' cross-decodes.
  (c) temporal generalization (Fig 10): train-window × test-window accuracy
      matrix, within a contrast and across contrasts (on the 'both' group).

Design (b) "set comparison" is just the same contrast decoded within each
electrode set — an ordinary decode with `electrodes` restricted — and is covered
by (0) per group, so it no longer has its own code path.

The SYNTHETIC path uses a ground-truth pseudopopulation with a KNOWN
shared-vs-orthogonal code (`cd.synthetic_roi_labeled_arrays`), which validates the
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

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

from src.analysis.decoding import cross_decoding as cd
from src.analysis.decoding.accuracy_stats import (
    compute_accuracies, perform_time_perm_cluster_test_for_accuracies)

STAB, FLEX, BOTH = "#2c7fb8", "#d95f0e", "#31a354"

# A4 sits on the A1 (proportion) electrode definition; decoding needs the time
# course, so the long df for the A1 labels is assembled with the cluster measure.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cluster'

# Class definitions in the project's substring convention. These must match how
# the condition names are built (`config/experiment_conditions.py`); the synthetic
# generator uses the same tokens so one set of strings drives both paths.
STAB_STRINGS = [['_i_'], ['_c_']]        # congruency  : incongruent vs congruent
FLEX_STRINGS = [['_s_'], ['_r_']]        # switch type : switch vs repeat


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
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _strip_arrays(d):
    """Drop bulky ndarray fields before JSON (they go to .npz instead)."""
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items()
                if not (isinstance(v, np.ndarray) and v.size > 64)}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'cross_decoding.json'), 'w') as f:
        json.dump(_json_safe(_strip_arrays(results)), f, indent=2)

    # accuracy traces + temporal-generalization matrices for re-plotting
    arrays = {}
    for group, res in results.get('label_transfer', {}).items():
        for direction, r in res.items():
            if 'acc_true' in r:
                arrays[f'labeltransfer_{group}_{direction}_true'] = r['acc_true']
                arrays[f'labeltransfer_{group}_{direction}_shuffle'] = r['acc_shuffle']
    for name, res in results.get('temporal', {}).items():
        np.save(os.path.join(save_dir, f'tempgen_{name}.npy'), res['matrix'])
    if arrays:
        np.savez(os.path.join(save_dir, 'accuracy_traces.npz'), **arrays)


# ---------------------------------------------------------------------------
# one decode -> a summarised result dict
# ---------------------------------------------------------------------------
def _summarise(out, p_thresh=0.05, n_perm=200, seed=42):
    """Confusion matrices -> accuracy traces + a cluster-corrected verdict.

    The true and shuffle traces come straight from the ordinary pipeline, so the
    same `time_perm_cluster` test used everywhere else corrects across windows.
    """
    acc_true, acc_shuffle = compute_accuracies(out['cm_true'], out['cm_shuffle'])
    sig, cluster_p = perform_time_perm_cluster_test_for_accuracies(
        acc_true, acc_shuffle, p_thresh=p_thresh, n_perm=n_perm, seed=seed)
    sig = np.asarray(sig).astype(bool).ravel()
    per_window = acc_true.mean(axis=1)
    return dict(
        acc_true=acc_true, acc_shuffle=acc_shuffle,
        mean_accuracy=float(per_window.mean()),
        peak_accuracy=float(per_window.max()),
        peak_window=int(np.argmax(per_window)),
        shuffle_mean=float(acc_shuffle.mean()),
        n_sig_windows=int(sig.sum()),
        n_windows=int(len(per_window)),
        any_sig=bool(sig.any()),
        cluster_p=_json_safe(cluster_p),
        conditions=out['conditions'],
    )


def _tempgen_matrix(out):
    """(train_win, test_win, samples, cats, cats) -> a (train_win, test_win) accuracy matrix."""
    cm = out['cm_true']
    n_train, n_test = cm.shape[0], cm.shape[1]
    m = np.zeros((n_train, n_test))
    for i in range(n_train):
        # compute_accuracies expects (n_windows, n_samples, cats, cats)
        acc, _ = compute_accuracies(cm[i], cm[i])
        m[i] = acc.mean(axis=1)
    return m


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_plots(results, save_dir):
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # A/B: within-block baseline (Fig 9)
    wb = results.get('within_block', {})
    for col, (cname, res) in enumerate(list(wb.items())[:2]):
        a = ax[0, col]
        blocks = list(res['per_block'].keys())
        accs = [res['per_block'][b]['mean_accuracy'] for b in blocks]
        a.bar([str(b) for b in blocks], accs,
              color=[STAB if 'cong' in cname else FLEX] * len(blocks))
        a.axhline(0.5, ls='--', c='k', lw=1)
        for i, b in enumerate(blocks):
            a.text(i, accs[i] + 0.01,
                   f"sig {res['per_block'][b]['n_sig_windows']}/"
                   f"{res['per_block'][b]['n_windows']}w", ha='center', fontsize=8)
        a.set(title=f"A4(0) Fig 9 · {cname}\n(decode within each block)",
              ylabel="mean accuracy", xlabel="block", ylim=(0.4, 1.0))

    # C: label transfer per electrode group — the time-resolved trace
    a = ax[0, 2]
    for group, res in results.get('label_transfer', {}).items():
        r = res.get('stab_to_flex')
        if r is None or 'acc_true' not in r:
            continue
        trace = np.asarray(r['acc_true']).mean(axis=1)
        a.plot(trace, label=f"{group} (n={r.get('n_channels', '?')})", lw=2)
    a.axhline(0.5, ls='--', c='k', lw=1)
    a.set(title="A4(a) · stability→flexibility transfer\nby electrode group",
          xlabel="time window", ylabel="cross-decoding accuracy")
    a.legend(fontsize=8)

    # D/E/F: temporal generalization matrices
    for col, (name, res) in enumerate(list(results.get('temporal', {}).items())[:3]):
        a = ax[1, col]
        m = np.asarray(res['matrix'])
        im = a.imshow(m, origin='lower', cmap='viridis',
                      vmin=0.4, vmax=max(0.6, np.nanmax(m)))
        a.set(title=f"A4(c) Fig 10 · temporal gen\n{name}",
              xlabel="test time window", ylabel="train time window")
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

    if results.get('within_block'):
        lines += ["-" * 72, "A4(0) within-block decoding baseline (Fig 9):"]
        for cname, res in results['within_block'].items():
            for b, r in res['per_block'].items():
                lines.append(f"   {cname} | block {b}: mean acc={r['mean_accuracy']:.3f} "
                             f"peak={r['peak_accuracy']:.3f} "
                             f"sig windows={r['n_sig_windows']}/{r['n_windows']}")
            if res.get('block_difference') is not None:
                lines.append(f"      Δ(block) on mean accuracy = {res['block_difference']:+.3f}")

    if results.get('within_block_by_group'):
        lines += ["-" * 72,
                  "A4(0b) per-group within-block 2x2 "
                  "(the define==decode diagonal cell is omitted by design):"]
        for gflag, res in results['within_block_by_group'].items():
            lines.append(f"   [{gflag}] n_electrodes={res['n_electrodes']} "
                         f"ignored cell={res['ignored_cell']}")
            for cell, r in res['cells'].items():
                lines.append(f"       {cell}: mean acc={r['mean_accuracy']:.3f} "
                             f"sig={r['n_sig_windows']}/{r['n_windows']}w")

    lines += ["-" * 72,
              "A4(a) label transfer (train stability, test flexibility) by group:",
              "      prediction: only the 'both' group cross-decodes."]
    for g, res in results.get('label_transfer', {}).items():
        for direction, r in res.items():
            lines.append(
                f"   [{g}] {direction}: mean acc={r['mean_accuracy']:.3f} "
                f"peak={r['peak_accuracy']:.3f} (shuffle {r['shuffle_mean']:.3f}) "
                f"sig windows={r['n_sig_windows']}/{r['n_windows']}")

    if results.get('temporal'):
        lines += ["-" * 72, "A4(c) temporal generalization (Fig 10):"]
        for name, res in results['temporal'].items():
            m = np.asarray(res['matrix'])
            diag = float(np.mean(np.diag(m)))
            off = float((m.sum() - np.trace(m)) / max(1, m.size - len(m)))
            lines.append(f"   {name}: mean diagonal={diag:.3f}  mean off-diagonal={off:.3f}  "
                         f"({'sustained/stable' if off > 0.55 else 'diagonal/phasic'} code)")

    lines += ["=" * 72,
              "Reading: cross-decoding above chance on the 'both' group = a SHARED",
              "code; chance on 'both' while each process is individually decodable =",
              "ORTHOGONAL codes (segregation at the representational level). Chance is",
              "the refit shuffle null (train labels permuted), and the window-wise",
              "verdict is cluster-corrected across time, so read `n_sig_windows`",
              "rather than any single window's accuracy.",
              "=" * 72]

    txt = "\n".join(lines)
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


def _restrict_to_electrodes(roi_labeled_arrays, roi, channel_names, keep):
    """Slice an ROI's arrays down to `keep` along the CHANNEL axis (axis 1).

    `channel_names` is the ROI LabeledArray's channel labelling, in order.
    Returns (restricted_dict, n_kept); n_kept == 0 means the group has no channel
    in this ROI and the caller should skip it.
    """
    keep = set(keep)
    idx = [i for i, ch in enumerate(channel_names) if ch in keep]
    if not idx:
        return None, 0
    out = {name: np.asarray(arr)[:, idx, :]
           for name, arr in roi_labeled_arrays[roi].items()}
    return {roi: out}, len(idx)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Decoder settings shared by every decode in this job.
    dec_kw = dict(n_splits=args.n_splits, n_repeats=args.n_repeats,
                  explained_variance=args.explained_variance,
                  window=args.window_size, step_size=args.step_size,
                  frac_train=getattr(args, 'frac_train', None),
                  random_state=getattr(args, 'seed', 42))
    cluster_kw = dict(n_perm=getattr(args, 'n_perm', 200),
                      seed=getattr(args, 'seed', 42))

    # 1. assemble the ROI LabeledArrays + electrode groups -----------------------
    if args.data_source == 'synthetic':
        print(f"DATA SOURCE: synthetic ({args.synthetic_code} code) — validates the path "
              "and that A4 discriminates shared vs orthogonal codes")
        roi = 'synthetic'
        arrays = cd.synthetic_roi_labeled_arrays(
            code=args.synthetic_code, seed=getattr(args, 'seed', 0))
        n_ch = next(iter(arrays[roi].values())).shape[1]
        channel_names = [f'ch{i}' for i in range(n_ch)]
        half = n_ch // 2
        a1_groups = {'both': channel_names,
                     'S_only': channel_names[:half],
                     'F_only': channel_names[half:]}
        interaction_groups, labels = {}, None
    else:
        print("DATA SOURCE: real epoched data")
        from src.analysis.utils.general_utils import (
            resolve_lab_root, resolve_electrodes_to_keep, load_HG_ev1_rescaled_per_subject)
        from src.analysis.utils.labeled_array_utils import (
            put_data_in_labeled_array_per_roi_subject)
        from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df

        LAB_root = resolve_lab_root(args.LAB_root)
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)

        # (i) the A1 electrode definition needs the long single-trial table
        df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                              electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
        print(f"assembled cluster df: {len(df)} rows | {df.subject.nunique()} subjects | "
              f"{df.electrode.nunique()} electrodes")
        labels, a1_groups = _electrode_groups(df, args.alpha)
        labels.to_csv(os.path.join(args.save_dir, 'anova_labels.csv'), index=False)
        interaction_groups = _interaction_groups(labels)
        print("A1 electrode groups: "
              + "  ".join(f"{g}={len(v)}" for g, v in a1_groups.items()))

        # (ii) the decode runs on the ordinary ROI LabeledArray pseudopopulation
        roi = args.roi
        arrays = put_data_in_labeled_array_per_roi_subject(
            args.subjects_mne_objects, args.condition_names, [roi], args.subjects,
            args.electrodes_per_subject_roi, obs_axs=0, chans_axs=1, time_axs=2,
            random_state=getattr(args, 'seed', 42))
        channel_names = list(args.roi_channel_names)

    results = {}

    # 2. (0) within-block decoding baseline (Fig 9) ------------------------------
    # "Decode a contrast within one block level" is an ordinary decode over that
    # block's conditions — restrict the conditions, then train == test contrast.
    print("A4(0): within-block decoding baseline (Fig 9)")
    within_block = {}
    for cname, strings, block_tokens in (
            ('congruency (LWPC)', STAB_STRINGS, ('25inc', '75inc')),
            ('switchType (LWPS)', FLEX_STRINGS, ('25sw', '75sw'))):
        per_block = {}
        for token in block_tokens:
            try:
                sub_arrays = cd.filter_conditions(arrays, roi, token)
            except ValueError as e:
                print(f"     skipping block {token}: {e}")
                continue
            out = cd.run_cross_decoding(sub_arrays, roi, strings, strings, **dec_kw)
            per_block[token] = _summarise(out, **cluster_kw)
        if len(per_block) == 2:
            a, b = (per_block[t]['mean_accuracy'] for t in per_block)
            diff = b - a
        else:
            diff = None
        within_block[cname] = dict(per_block=per_block, block_difference=diff)
    results['within_block'] = within_block

    # 2b. the within-block 2x2 restricted to each interaction-defined group,
    #     SKIPPING the diagonal (define == decode) cell to avoid double-dipping.
    #     Only the OFF-diagonal (cross) cells are computed/kept — e.g. the CPC
    #     electrode set is decoded on switchType/switch_prop and the two cross
    #     cells, never on congruency/inc_prop (the interaction that defined it).
    #     To keep the diagonal cell instead, define the electrodes on a disjoint
    #     set of trials (`trial_splitting.apply_electrode_definition_split`).
    if interaction_groups:
        print("A4(0b): per-group within-block 2x2 (diagonal define==decode cells ignored)")
        decode_cells = [('congruency', 'incongruent_proportion', STAB_STRINGS, ('25inc', '75inc')),
                        ('congruency', 'switch_proportion', STAB_STRINGS, ('25sw', '75sw')),
                        ('switchType', 'switch_proportion', FLEX_STRINGS, ('25sw', '75sw')),
                        ('switchType', 'incongruent_proportion', FLEX_STRINGS, ('25inc', '75inc'))]
        per_group = {}
        for gflag, elset in interaction_groups.items():
            restricted, n_kept = _restrict_to_electrodes(arrays, roi, channel_names, elset)
            if n_kept < args.min_group_size:
                print(f"     group '{gflag}' has {n_kept} electrodes in ROI "
                      f"(< {args.min_group_size}); skipping")
                continue
            cells = {}
            for contrast, block_col, strings, tokens in decode_cells:
                if cd.is_circular_decode(gflag, contrast, block_col):
                    continue                     # double-dipping: ignore this result
                for token in tokens:
                    try:
                        sub_arrays = cd.filter_conditions(restricted, roi, token)
                    except ValueError:
                        continue
                    out = cd.run_cross_decoding(sub_arrays, roi, strings, strings, **dec_kw)
                    cells[f'{contrast} by {block_col} [{token}]'] = _summarise(out, **cluster_kw)
            per_group[gflag] = dict(n_electrodes=n_kept,
                                    ignored_cell=cd.circular_decode_for_group(gflag),
                                    cells=cells)
        results['within_block_by_group'] = per_group

    # 3. (a) label transfer per electrode group ----------------------------------
    print("A4(a): label transfer (stability<->flexibility) per electrode group")
    lt = {}
    for g, elset in a1_groups.items():
        restricted, n_kept = _restrict_to_electrodes(arrays, roi, channel_names, elset)
        if n_kept < args.min_group_size:
            print(f"     group '{g}' has {n_kept} electrodes in ROI "
                  f"(< {args.min_group_size}); skipping")
            continue
        entry = {}
        for direction, (tr, te) in (('stab_to_flex', (STAB_STRINGS, FLEX_STRINGS)),
                                    ('flex_to_stab', (FLEX_STRINGS, STAB_STRINGS))):
            out = cd.run_cross_decoding(restricted, roi, tr, te, **dec_kw)
            entry[direction] = _summarise(out, **cluster_kw)
            entry[direction]['n_channels'] = n_kept
        lt[g] = entry
    results['label_transfer'] = lt

    # 4. (c) temporal generalization (Fig 10) on the 'both' group ----------------
    print("A4(c): temporal generalization (Fig 10)")
    tg_set = a1_groups.get('both')
    restricted, n_kept = (_restrict_to_electrodes(arrays, roi, channel_names, tg_set)
                          if tg_set else (None, 0))
    if n_kept >= args.min_group_size:
        # n_windows^2 predictions — halve the repeats to keep the runtime sane
        tg_kw = dict(dec_kw, n_repeats=max(2, args.n_repeats // 2),
                     temporal_generalization=True)
        results['temporal'] = {}
        for name, (tr, te) in (
                ('stability (within)', (STAB_STRINGS, STAB_STRINGS)),
                ('flexibility (within)', (FLEX_STRINGS, FLEX_STRINGS)),
                ('stability->flexibility (cross)', (STAB_STRINGS, FLEX_STRINGS))):
            out = cd.run_cross_decoding(restricted, roi, tr, te, **tg_kw)
            results['temporal'][name] = dict(matrix=_tempgen_matrix(out),
                                             n_channels=n_kept)

    # 5. persist + plot + summarize ----------------------------------------------
    save_results(results, args.save_dir)
    make_plots(results, args.save_dir)
    write_summary(results, args.save_dir, meta=dict(
        data_source=args.data_source,
        synthetic_code=getattr(args, 'synthetic_code', None),
        task=getattr(args, 'task', None),
        epochs_root_file=getattr(args, 'epochs_root_file', None),
        window=f"[{getattr(args, 'window_tmin', None)}, {getattr(args, 'window_tmax', None)}]s",
        window_size=args.window_size, step_size=args.step_size,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        frac_train=getattr(args, 'frac_train', None),
        alpha=getattr(args, 'alpha', None), save_dir=args.save_dir))
    return results
