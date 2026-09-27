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
      SEPARATELY on the both / S_only / F_only groups, plus the UNSELECTED
      reference group (`args.reference_group`, default 'all' = every electrode in
      the decoded ROI array). Each group also decodes each contrast within
      itself, on the same trials and folds: the ceiling every transfer is read
      against. Prediction: only 'both' cross-decodes; the reference group says
      what the region does before any selection. With main-effect labels
      (`args.contrast_mode == 'condition'`) S_only / F_only are named
      congruency_only / switch_type_only.
      This design is ALREADY pooled: its classes are every 'i' cell vs every 'c'
      cell and every 's' cell vs every 'r' cell, across both block proportions.
      Only design (0)/(0b) splits by proportion.
  (c) temporal generalization (Fig 10): train-window × test-window accuracy
      matrix, within a contrast and across contrasts, on `args.tempgen_groups`
      (default: the 'both' group).

The S/F electrode groups come from either route — see `ELECTRODE_DEFINITIONS`:
`args.electrode_definition='anova'` fits one window-mean ANOVA per electrode in
this job, 'power_traces' reads the finished cluster-corrected windowed-ANOVA runs.

Contrast and block definitions are read off each condition's declared factor
levels (`cd.condition_cells`), not parsed out of condition names — the real and
synthetic naming conventions collide on the obvious shorthand tokens.

A condition set only has to declare congruency AND switchType on the same
condition. With the full 2x2x2x2 (`stimulus_experiment_conditions`, the default)
every design above runs. With the pooled 2x2 (`stimulus_main_effect_conditions`,
which collapses both proportions into 4 cells and so puts ~4x the trials in each)
designs (a) and (c) run unchanged over more trials per cell, and (0)/(0b) are
skipped — see `cd.has_block_factor`.

Design (b) "set comparison" is just the same contrast decoded within each
electrode set — an ordinary decode with `electrodes` restricted — and is covered
by (0) per group, so it no longer has its own code path.

The SYNTHETIC path uses a ground-truth pseudopopulation with a KNOWN
shared-vs-orthogonal code (`cd.synthetic_roi_labeled_arrays`), which validates the
whole path and that the analysis discriminates the two codes.

With `args.analysis == 'block_transfer'` the job runs N3b instead
(`run_block_transfer_job`, docs/n3b_block_transfer.md): each contrast is trained
in one block level and tested in the other, on every electrode of the ROI, with
no electrode groups. `args.analysis == 'task_transfer'` runs the task-transfer
positive controls through the same function (docs/cross_decoding_controls.md
§3.5): task trained on congruent (repeat) trials and tested on incongruent
(switch) ones, and congruency / switch type trained in one task, tested in the
other.

Driven by `run_stability_flexibility_cross_decoding_dcc.py` (wrapped by
`sbatch_stability_flexibility_cross_decoding_dcc.sh`). Not run directly on the
cluster; call `main(args)` with a populated argument namespace.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import re
import json
from types import SimpleNamespace

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

from src.analysis.decoding import block_transfer as bt
from src.analysis.decoding import cross_decoding as cd
from src.analysis.decoding.accuracy_stats import (
    compute_accuracies, perform_time_perm_cluster_test_for_accuracies)
from src.analysis.config import experiment_conditions

STAB, FLEX, BOTH = "#2c7fb8", "#d95f0e", "#31a354"

# A4 sits on the A1 (proportion) electrode definition; decoding needs the time
# course, so the long df for the A1 labels is assembled with the cluster measure.
CONTRAST_MODE = 'proportion'
EFFECT_MEASURE = 'cluster'

# Class definitions are DERIVED from each condition's declared factor levels
# (`cd.condition_cells`), not hand-written as substrings of the condition names.
# The real and synthetic naming conventions collide on shorthand tokens — '75s'
# is "switch trial in the 75%-incongruent block" under
# `stimulus_experiment_conditions` and "75%-switch block" under the synthetic
# generator — and picking the wrong one decodes the wrong contrast silently
# rather than raising. See the note above `cd.condition_cells`.

# how each block factor is spelled in result keys and figures
_BLOCK_TAG = {'incongruent_proportion': 'incongruent', 'switch_proportion': 'switch'}

# each label transfer -> the within decode of the labelling it is SCORED on
_TRANSFER_CEILINGS = {'stab_to_flex': 'flex_to_flex', 'flex_to_stab': 'stab_to_stab'}


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
        # the keys carry '->', spaces and the '[group]' tag — keep them off disk
        safe = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
        np.save(os.path.join(save_dir, f'tempgen_{safe}.npy'), res['matrix'])
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
        significant_windows=sig,
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
def make_plots(results, save_dir, *, first_time_point=-1.0,
               sampling_rate=256, window_size=20, step_size=10):
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

    # Produce the same Nature-style true-vs-shuffle accuracy trace used by the
    # ordinary decoder, once per direction and electrode group. The overview
    # above remains useful for comparing groups; these files provide identical
    # uncertainty bands, chance/onset lines, and significant-cluster markers.
    from src.analysis.decoding.plots.accuracies import plot_accuracies_nature_style
    for group, directions in results.get('label_transfer', {}).items():
        for direction, result in directions.items():
            if 'acc_true' not in result:
                continue
            n_windows = np.asarray(result['acc_true']).shape[0]
            centers = (first_time_point
                       + (np.arange(n_windows) * step_size + window_size / 2)
                       / sampling_rate)
            plot_accuracies_nature_style(
                centers,
                {'true': np.asarray(result['acc_true']),
                 'shuffle': np.asarray(result['acc_shuffle'])},
                significant_clusters=np.asarray(result['significant_windows']),
                window_size=window_size, step_size=step_size,
                sampling_rate=sampling_rate,
                comparison_name=direction, roi=group, save_dir=save_dir,
                title=f'{direction.replace("_", " ")} · {group}',
                samples_axis=1, filename_suffix='_cross_decoding')


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
              "A4(a) label transfer by group. stab = congruency (i vs c), flex = switch",
              "      type (s vs r); stab_to_stab / flex_to_flex are the within-contrast",
              "      ceilings, decoded on the same trials with the same folds.",
              "      prediction: only the 'both' group cross-decodes.",
              f"      '{meta.get('reference_group') or 'all'}' is the UNSELECTED "
              "reference set (every electrode in the",
              "      decoded ROI array, i.e. what `electrodes` above already "
              "restricted it to) —",
              "      no interaction defined it, so it is the honest baseline for "
              "the selected groups."]
    for g, res in results.get('label_transfer', {}).items():
        for direction, r in res.items():
            lines.append(
                f"   [{g}] {direction}: mean acc={r['mean_accuracy']:.3f} "
                f"peak={r['peak_accuracy']:.3f} (shuffle {r['shuffle_mean']:.3f}) "
                f"sig windows={r['n_sig_windows']}/{r['n_windows']}")
            if 'n_below_ceiling' in r:
                ceiling = _TRANSFER_CEILINGS[direction]
                lines.append(
                    f"         vs {ceiling}, its ceiling: below it in {r['n_below_ceiling']} "
                    f"windows; keeps {_retained_text(r, res[ceiling])} of it above chance")

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
# Two interchangeable routes to the same labels table (same columns, same
# CPC/SPS/CPS/SPC + S/F contract), selected by `args.electrode_definition`:
#
#   'anova'        -- `sfs.per_electrode_anova_labels`: ONE two-way ANOVA per
#                     electrode on the window-MEAN HG over [window_tmin,
#                     window_tmax], BH-FDR'd across electrodes. Self-contained
#                     (it only needs the epochs this job already loads), which
#                     is why it was the original and only route here.
#   'power_traces' -- `ptc.electrode_labels`: reads the already-computed
#                     within-electrode WINDOWED ANOVA runs and their permutation
#                     cluster correction, so an electrode counts as selective if
#                     any cluster survives across time. Strictly more sensitive
#                     to strong-but-transient interactions that the window mean
#                     dilutes, and it makes the decoded electrode sets literally
#                     the ones the power-trace figures call significant -- but it
#                     needs finished run directories, which is the only reason it
#                     is not the default.
ELECTRODE_DEFINITIONS = ('anova', 'power_traces', 'csv')


def _channel_keys(labels):
    """Labels rows -> the ROI LabeledArray's channel names (`f'{subject}-{elec}'`).

    `put_data_in_labeled_array_per_roi_subject` labels channels `subject-electrode`
    so they stay unique across the pseudopopulation. `per_electrode_anova_labels`
    already emits that prefixed form in `electrode`; the power-traces route keeps
    `subject` and a BARE `electrode` in separate columns. Normalising here is what
    keeps `_restrict_to_electrodes` from silently matching nothing (which would
    read as "this group has no electrode in the ROI" rather than as a key
    mismatch).
    """
    subj = labels['subject'].astype(str)
    elec = labels['electrode'].astype(str)
    prefix = (subj + '-').to_numpy()
    elec = elec.to_numpy()
    # elementwise, since `Series.str.startswith` only takes a literal prefix
    prefixed = np.fromiter((e.startswith(p) for e, p in zip(elec, prefix)),
                           dtype=bool, count=len(elec))
    return np.where(prefixed, elec, np.char.add(prefix.astype(str), elec.astype(str)))


def _resolve_labels(args, df=None):
    """The per-electrode S/F definition table, from whichever route is selected."""
    definition = getattr(args, 'electrode_definition', 'anova')
    if definition not in ELECTRODE_DEFINITIONS:
        raise ValueError(f"electrode_definition must be one of {ELECTRODE_DEFINITIONS}; "
                         f"got {definition!r}")

    if definition == 'csv':
        path = getattr(args, 'anova_labels_csv', None)
        if not path:
            raise ValueError("electrode_definition='csv' needs ANOVA_LABELS_CSV")
        from src.analysis.utils.anova_label_selection import (
            load_anova_label_electrodes,
            load_anova_labels,
            selected_pairs,
        )
        labels = load_anova_labels(
            path, correction=getattr(args, 'fdr_correction', 'flags'),
            alpha=args.alpha, roi=getattr(args, 'anova_label_roi', None))
        selection = load_anova_label_electrodes(
            path, effect=getattr(args, 'anova_label_effect', 'both'),
            correction=getattr(args, 'fdr_correction', 'flags'),
            alpha=args.alpha, roi=getattr(args, 'anova_label_roi', None))
        pairs = selected_pairs(selection)
        bare_electrodes = [
            electrode.removeprefix(f"{subject}-")
            for subject, electrode in zip(
                labels['subject'].astype(str), labels['electrode'].astype(str))
        ]
        labels = labels[
            [(subject, electrode) in pairs for subject, electrode in zip(
                labels['subject'].astype(str), bare_electrodes)]
        ].copy()
        required = {'subject', 'electrode', 'S', 'F'}
        missing = required - set(labels)
        if missing:
            raise ValueError(f"{path} is missing required columns {sorted(missing)}")
        return labels

    if definition == 'power_traces':
        from src.analysis.stats import power_traces_conjunction as ptc
        runs = getattr(args, 'power_traces_runs', None)
        if not runs:
            raise ValueError(
                "electrode_definition='power_traces' needs `power_traces_runs`: "
                "either one run directory whose ANOVA carried all four "
                "interactions, or {'CPC': dir, 'SPS': dir, 'CPS': dir, 'SPC': dir}. "
                "Set POWER_TRACES_RUN_DIR (or POWER_TRACES_CPC/SPS/CPS/SPC).")
        return ptc.electrode_labels(
            runs=runs,
            roi=getattr(args, 'power_traces_roi', None),
            alpha=args.alpha,
            correction=getattr(args, 'power_traces_correction', 'fdr_bh'))

    from src.analysis.stats import stability_flexibility_segregation as sfs
    return sfs.per_electrode_anova_labels(
        df, alpha=args.alpha, contrast_mode=getattr(args, 'contrast_mode', CONTRAST_MODE),
        fdr_correction=getattr(args, 'fdr_correction', 'fdr_bh'))


# The A1 table keeps one schema in both contrast modes (S = CPC, F = SPS), but
# what the flags MEAN differs: the LWPC / LWPS interactions under 'proportion',
# the congruency / switch-type MAIN effects under 'condition'. Groups are named
# after what they hold, so a main-effect run never reports an "S_only" group.
_GROUP_NAMES = {'proportion': ('both', 'S_only', 'F_only'),
                'condition': ('both', 'congruency_only', 'switch_type_only')}


def _electrode_groups(labels, contrast_mode='proportion'):
    """The three DISJOINT label-transfer groups, as ROI-array channel names."""
    chan = _channel_keys(labels)
    S = (labels['S'] == 1).to_numpy()
    F = (labels['F'] == 1).to_numpy()
    both, s_only, f_only = _GROUP_NAMES[contrast_mode]
    return {
        both: chan[S & F].tolist(),
        s_only: chan[S & ~F].tolist(),
        f_only: chan[~S & F].tolist(),
    }


def _interaction_groups(labels, contrast_mode='proportion'):
    """The FOUR interaction-defined electrode sets (possibly overlapping), keyed by
    the definition-group flag (CPC/SPS/CPS/SPC) so `cd.is_circular_decode` can name
    each set's double-dip cell. Used for the per-group within-block 2x2 that skips
    the diagonal (define==decode) cell.

    Main-effect labels (contrast_mode='condition') carry no interaction: their
    CPC/SPS columns are the congruency / switch-type main effects, so the sets are
    keyed 'congruency' / 'switch_type', whose circular cells are every decode of
    their own contrast (`cd.MAIN_EFFECT_DECODE_CONTRAST`)."""
    chan = _channel_keys(labels)
    if contrast_mode == 'condition':
        flags = {'congruency': 'S', 'switch_type': 'F'}
    else:
        flags = {flag: flag for flag in ('CPC', 'SPS', 'CPS', 'SPC')}
    return {name: chan[(labels[flag] == 1).to_numpy()].tolist()
            for name, flag in flags.items() if flag in labels.columns}


def _add_reference_group(groups, channel_names, args):
    """Add the UNSELECTED reference set: every channel in the decoded ROI array.

    Without it, the label-transfer and temporal-generalization designs only ever
    run on interaction-selected subsets (both / S_only / F_only), so there is no
    baseline for "does this ROI cross-decode at all" -- and every one of those
    subsets was chosen for carrying an interaction, which is exactly the
    selection that inflates within-contrast decodability. The reference group is
    defined by NOTHING the decode is about, so it is the honest comparison.

    What "all" means is set upstream by `args.electrodes`: 'sig' restricts the
    ROI array to the baseline task-significant electrodes, 'all' keeps every
    electrode in the ROI. Either way this group is that array's full channel
    list, so it is never a superset of what was actually loaded. Set
    `args.reference_group` to None/'' to drop it.
    """
    name = getattr(args, 'reference_group', 'all')
    if not name:
        return groups
    if name in groups:
        raise ValueError(f"reference_group={name!r} collides with an existing "
                         f"electrode group; pick another name")
    channel_names = list(channel_names)
    # On the synthetic path 'both' is already every channel by construction;
    # adding a byte-identical group would just decode the same thing twice.
    if any(set(v) == set(channel_names) for v in groups.values()):
        return groups
    groups[name] = channel_names
    return groups


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
# the decoded ROI pseudopopulation (real data)
# ---------------------------------------------------------------------------
def _build_roi_arrays(args, LAB_root, trial_partitions=None, required_fields=None):
    """Load the epochs and build the ROI LabeledArray this job decodes.

    Mirrors `decoding_dcc.main`'s setup so A4 decodes exactly what the ordinary
    decoding job would: the same ROI/significance electrode resolution, the same
    "filter against what actually survived epoching" step, and the same
    pseudopopulation builder.

    Returns `(roi, arrays, channel_names, cells)`, where `channel_names` is the
    array's own channel labelling (`subject-electrode`, in order — the thing
    `_restrict_to_electrodes` slices against) and `cells` is the condition->factor
    table the contrasts are derived from.
    """
    from src.analysis.utils.general_utils import (
        get_sig_chans_per_subject, make_sig_electrodes_per_subject_and_roi_dict,
        load_subjects_electrodes_to_ROIs_dict, create_subjects_mne_objects_dict,
        filter_electrode_lists_against_subjects_mne_objects,
        print_summary_of_dropped_electrodes)
    from src.analysis.utils.labeled_array_utils import (
        put_data_in_labeled_array_per_roi_subject)

    roi = args.roi
    if args.rois_dict is None or roi not in args.rois_dict:
        raise ValueError(
            f"ROI {roi!r} is not in rois_dict (have "
            f"{sorted(args.rois_dict or [])}); set ROI to one of them, or extend "
            "ROIS_DICT in the runner.")

    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = load_subjects_electrodes_to_ROIs_dict(
        save_dir=config_dir, filename='subjects_electrodestoROIs_dict.json')
    sig_chans_per_subject = get_sig_chans_per_subject(
        args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)
    all_elecs, sig_elecs = make_sig_electrodes_per_subject_and_roi_dict(
        args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)

    # A CSV is already the electrode definition; do not silently intersect it
    # with the unrelated baseline-responsiveness (`sig`) list.
    if args.electrodes == 'all' or getattr(args, 'electrode_definition', None) == 'csv':
        raw_electrodes = all_elecs
    elif args.electrodes == 'sig':
        raw_electrodes = sig_elecs
    else:
        raise ValueError(f"electrodes must be 'all' or 'sig'; got {args.electrodes!r}")

    cells = cd.condition_cells(
        args.conditions,
        required=required_fields or cd.CROSS_DECODE_FIELDS)
    condition_names = list(cells)
    print(f"conditions: {len(condition_names)} decodable cells "
          f"(of {len(args.conditions)} in the condition set)")

    subjects_mne_objects = create_subjects_mne_objects_dict(
        subjects=args.subjects, epochs_root_file=args.epochs_root_file,
        conditions={name: args.conditions[name] for name in condition_names},
        task=args.task, just_HG_ev1_rescaled=True, LAB_root=LAB_root,
        acc_trials_only=args.acc_trials_only)
    if trial_partitions is not None:
        from src.analysis.decoding.anova_electrode_selection import apply_trial_partition
        subjects_mne_objects = apply_trial_partition(
            subjects_mne_objects, trial_partitions, which='decode')

    # An electrode in the ROI dict may have been dropped during epoching; asking
    # for it would index past the epochs object.
    electrodes = filter_electrode_lists_against_subjects_mne_objects(
        [roi], raw_electrodes, subjects_mne_objects)
    print_summary_of_dropped_electrodes(raw_electrodes, electrodes)

    arrays = put_data_in_labeled_array_per_roi_subject(
        subjects_mne_objects, condition_names, [roi], args.subjects,
        electrodes, obs_axs=0, chans_axs=1, time_axs=2,
        random_state=getattr(args, 'seed', 42))
    channel_names = _roi_channel_names(arrays, roi)
    print(f"ROI {roi!r} pseudopopulation: {len(channel_names)} channels "
          f"({args.electrodes} electrodes)")
    return roi, arrays, channel_names, cells


def _split_subject_epochs(subjects_epochs, frac_select, seed):
    """Return selection Epochs plus one stable-id partition for later decode.

    The pooled Epochs used by A1 and the condition-specific Epochs used by the
    decoder both carry ``metadata.trial_count``. Building the partition here and
    applying it by that physical-trial id is what prevents the same trial from
    entering electrode selection under one condition and decoding under another.
    """
    from src.analysis.decoding.anova_electrode_selection import assign_trial_partitions
    from src.analysis.decoding.trial_splitting import strata_key_from_metadata

    sub_trials = {}
    for subject, epochs in subjects_epochs.items():
        metadata = getattr(epochs, 'metadata', None)
        if metadata is None or 'trial_count' not in metadata:
            raise ValueError(
                "ELECTRODE_SELECTION_SPLIT requires metadata.trial_count on the "
                f"epochs, but it is absent for {subject}")
        ids = metadata['trial_count'].to_numpy()
        strata = strata_key_from_metadata(
            metadata, ('congruency', 'task_sequence', 'block_type'))
        # The pooled Epochs should already contain one row per physical trial;
        # de-duplicate defensively while preserving the first stratum.
        unique = {}
        for trial_id, stratum in zip(ids, strata):
            unique.setdefault(trial_id, stratum)
        ordered_ids = np.asarray(list(unique))
        sub_trials[subject] = (
            ordered_ids, np.asarray([unique[i] for i in ordered_ids], dtype=object))

    partitions = assign_trial_partitions(
        sub_trials, frac_select=frac_select, seed=seed)
    selection_epochs = {}
    for subject, epochs in subjects_epochs.items():
        keep = partitions[subject]['select']
        idx = np.flatnonzero(epochs.metadata['trial_count'].isin(keep).to_numpy())
        selection_epochs[subject] = epochs[idx]
        print(f"  [trial-split] {subject}: {len(idx)} selection trials / "
              f"{len(epochs) - len(idx)} decoding trials")
    return selection_epochs, partitions


def _roi_channel_names(arrays, roi):
    """The ROI LabeledArray's channel labels, in array order.

    Layout is [Conditions, Trials, Channels, Timepoints], so channels are
    `labels[2]`. Falls back to positional names only if the array carries no
    labelling — in which case the electrode groups cannot match anything and the
    caller is told rather than left with silently empty groups.
    """
    arr = arrays[roi]
    labels = getattr(arr, 'labels', None)
    if labels is not None and len(labels) > 2:
        return [str(c) for c in labels[2]]
    raise ValueError(
        f"the ROI {roi!r} array carries no channel labelling, so the electrode "
        "groups cannot be matched against it; expected a LabeledArray with "
        "[Conditions, Trials, Channels, Timepoints] labels")


# ---------------------------------------------------------------------------
# N3b: block-transfer cross-decoding (docs/n3b_block_transfer.md), and the
# task-transfer positive controls (docs/cross_decoding_controls.md §3.5)
# ---------------------------------------------------------------------------
BLOCK_TRANSFER_DESIGNS = {
    # name: (decoded contrast, transfer factor, pooled condition-set name)
    'X1': ('congruency', 'incongruent_proportion', 'stimulus_lwpc_conditions'),
    'X2': ('switchType', 'switch_proportion', 'stimulus_lwps_conditions'),
    'X3': ('congruency', 'switch_proportion',
           'stimulus_congruency_by_switch_proportion_conditions'),
    'X2b': ('switchType', 'incongruent_proportion',
           'stimulus_switch_type_by_incongruent_proportion_conditions'),
}

# The same train-in-one-level / test-in-the-other 2x2, across a trial-level
# factor instead of a block. T1/T2 are the task x congruency and task x switch
# type controls: task (global vs local) learned on congruent (repeat) trials and
# tested on incongruent (switch) ones. T3/T4 swap the roles, so the contrast is
# the A4 contrast at its own effect size. T2 and T4 carry the previous-task
# confound: on a switch trial the previous task was the other one.
TASK_TRANSFER_DESIGNS = {
    'T1': ('task', 'congruency', 'stimulus_task_by_congruency_conditions'),
    'T2': ('task', 'switchType', 'stimulus_task_by_switch_type_conditions'),
    'T3': ('congruency', 'task', 'stimulus_task_by_congruency_conditions'),
    'T4': ('switchType', 'task', 'stimulus_task_by_switch_type_conditions'),
}

TRANSFER_ANALYSES = {'block_transfer': BLOCK_TRANSFER_DESIGNS,
                     'task_transfer': TASK_TRANSFER_DESIGNS}

# how a transfer factor's levels are printed; block proportions print as '25%'
_LEVEL_NAMES = {'congruency': {'c': 'congruent', 'i': 'incongruent'},
                'switchType': {'r': 'repeat', 's': 'switch'},
                'task': {'g': 'global', 'l': 'local'}}


def _level_label(factor, level):
    return _LEVEL_NAMES.get(factor, {}).get(level, f'{level}%')


def _transfer_tag(factor, train, test):
    """File-name tag for one transfer: '25to75' for a block, 'congruent_to_incongruent'."""
    if factor in _LEVEL_NAMES:
        return f'{_level_label(factor, train)}_to_{_level_label(factor, test)}'
    return f'{train}to{test}'


def _retained(transfer, ceiling, chance=0.5):
    """Share of the ceiling's above-chance accuracy that a transfer keeps, over the
    windows where the ceiling beats its shuffle null; None if it never does.

    Both are summaries from `_summarise`, scored on the same test trials, so this
    is the "transfer as a fraction of within-condition accuracy" of
    docs/cross_decoding_controls.md §2: 1 = full transfer, 0 = none.
    """
    sig = np.asarray(ceiling['significant_windows'], bool)
    if not sig.any():
        return None
    within = np.asarray(ceiling['acc_true']).mean(axis=1)[sig].mean()
    across = np.asarray(transfer['acc_true']).mean(axis=1)[sig].mean()
    if within <= chance:
        return None
    return float((across - chance) / (within - chance))


def _retained_text(transfer, ceiling):
    share = _retained(transfer, ceiling)
    return "n/a (its ceiling never beats shuffle)" if share is None else f"{share:.0%}"


def _window_centers(n_windows, args):
    """Centre of each decoding window, in seconds from stimulus onset."""
    return (getattr(args, 'first_time_point', -1.0)
            + (np.arange(n_windows) * args.step_size + args.window_size / 2)
            / getattr(args, 'sampling_rate', 256))


def _summarise_block_transfer(out, args, cluster_kw):
    """One `bt.run_block_transfer` 2x2 -> a summary per cell, plus the ceiling test.

    Every cell gets the ordinary `_summarise` (cluster test against its own
    shuffle null) and its post-stimulus mean. Each transfer is also compared with
    the within-level accuracy of the level it is SCORED on: `below_ceiling` marks
    the windows where within(test) > transfer(train -> test).
    """
    summary = {}
    for (train, test), cms in out['cells'].items():
        s = _summarise(dict(cms, conditions=[]), **cluster_kw)
        post = _window_centers(s['n_windows'], args) > 0
        s['post_mean_accuracy'] = float(s['acc_true'][post].mean()) if post.any() else None
        s['n_sig_post'] = int(s['significant_windows'][post].sum())
        s['n_sig_pre'] = int(s['significant_windows'][~post].sum())
        summary[(train, test)] = s
    for (train, test), s in summary.items():
        if train != test:
            below, _ = perform_time_perm_cluster_test_for_accuracies(
                summary[(test, test)]['acc_true'], s['acc_true'], **cluster_kw)
            s['below_ceiling'] = np.asarray(below).astype(bool).ravel()
            s['n_below_ceiling'] = int(s['below_ceiling'].sum())
    return summary


def _plot_block_transfer(results, args, roi):
    """For each design, centering and test level: the transfer INTO a level,
    against that level's own within-level accuracy (its ceiling) and the null."""
    from src.analysis.decoding.plots.accuracies import plot_accuracies_nature_style
    for key, res in results.items():
        lo, hi = res['levels']
        factor = res['block_col']
        for train, test in ((lo, hi), (hi, lo)):
            transfer = res['cells'][f'{train}->{test}']
            within = res['cells'][f'{test}->{test}']
            into = f'{_level_label(factor, train)} -> {_level_label(factor, test)}'
            plot_accuracies_nature_style(
                _window_centers(transfer['acc_true'].shape[0], args),
                {f'within {_level_label(factor, test)}': within['acc_true'],
                 into: transfer['acc_true'],
                 'shuffle': transfer['acc_shuffle']},
                significant_clusters=transfer['significant_windows'],
                window_size=args.window_size, step_size=args.step_size,
                sampling_rate=getattr(args, 'sampling_rate', 256),
                comparison_name=f'{key}_{_transfer_tag(factor, train, test)}', roi=roi,
                save_dir=args.save_dir,
                title=f"{res['design']} {res['contrast']}: {into} "
                      f"({'centered' if res['centered'] else 'uncentered'})",
                samples_axis=1,
                filename_suffix=getattr(args, 'analysis', 'block_transfer'))


def _task_effect_size_lines(results):
    """Within-level task accuracy (T1) next to within-level congruency accuracy
    (T3): the effect-size regime docs/closing_figure_plan.md asks the task
    control to be read against."""
    def within(key):
        res = results.get(key)
        if res is None:
            return None
        accs = [res['cells'][f'{level}->{level}']['post_mean_accuracy']
                for level in res['levels']]
        return None if None in accs else float(np.mean(accs))

    task, cong = within('T1_uncentered'), within('T3_uncentered')
    if task is None or cong is None:
        return []
    return ["-" * 72,
            f"EFFECT SIZE: within-level task accuracy {task:.3f} (T1) vs congruency "
            f"{cong:.3f} (T3), post-stimulus, uncentered.",
            "   The further task sits above congruency, the more T1 shows only that the",
            "   pipeline can transfer a strong code; T3 is the control at congruency's",
            "   own effect size."]


def _write_block_transfer_summary(results, meta, save_dir):
    task_controls = meta.get('analysis') == 'task_transfer'

    def cell(s):
        acc = s['post_mean_accuracy']
        return ("n/a" if acc is None else f"{acc:.3f}") + f" ({s['n_sig_post']}/{s['n_sig_pre']})"

    lines = ["=" * 72,
             "TASK-TRANSFER POSITIVE CONTROLS" if task_controls
             else "N3b BLOCK-TRANSFER CROSS-DECODING",
             "=" * 72]
    lines += [f"{k:>22}: {v}" for k, v in meta.items()]
    if meta['n_resamples'] < 5:
        lines.append(f"NOTE: with only {meta['n_resamples']} resamples the cluster tests "
                     "cannot reach p < .05, so every significance count below is 0 by "
                     "construction. Use N_REPEATS >= 10 for a real run.")
    for key, res in results.items():
        lo, hi = res['levels']
        c = res['cells']
        factor = res['block_col']
        lines += ["-" * 72,
                  f"{key}: {res['contrast']}, trained in one {factor} level "
                  "and tested in the other",
                  f"   balanced to {res['n_per_group']} trials per contrast x transfer cell "
                  f"(available: {res['group_sizes']})",
                  "   post-stimulus mean accuracy (significant windows vs shuffle, post/pre):",
                  f"   {'train | test':>14}{_level_label(factor, lo):>18}"
                  f"{_level_label(factor, hi):>18}"]
        for train in (lo, hi):
            lines.append(f"   {_level_label(factor, train):>14}{cell(c[f'{train}->{lo}']):>18}"
                         f"{cell(c[f'{train}->{hi}']):>18}")
        for train, test in ((lo, hi), (hi, lo)):
            transfer, ceiling = c[f'{train}->{test}'], c[f'{test}->{test}']
            lines.append(f"   {_level_label(factor, train)} -> {_level_label(factor, test)} "
                         f"vs within {_level_label(factor, test)}: below that ceiling in "
                         f"{transfer['n_below_ceiling']} windows; keeps "
                         f"{_retained_text(transfer, ceiling)} of it above chance")
        flat = [_level_label(factor, level) for level in (lo, hi)
                if c[f'{level}->{level}']['n_sig_post'] == 0]
        lines.append("   CEILING: " + (f"within {', '.join(flat)} never beats shuffle "
                                        "after stimulus onset -> nothing to transfer; "
                                        "this design is NOT interpretable" if flat else
                                        "both within-level decodes beat shuffle -> interpretable"))
        n_pre = sum(s['n_sig_pre'] for s in c.values())
        if n_pre and res['contrast'] == 'task':
            lines.append(f"   PRE-STIMULUS: {n_pre} significant windows across the 2x2. The "
                         "previous task predicts the current one on repeat trials, so some "
                         "task information can precede the cue.")
        elif n_pre:
            lines.append(f"   ARTIFACT FLAG: {n_pre} significant pre-stimulus windows across "
                         "the 2x2 (congruency/switch information cannot exist there yet)")

    if task_controls:
        lines += _task_effect_size_lines(results)
        lines += ["=" * 72,
                  "Reading (docs/cross_decoding_controls.md §3.5): compare each transfer with",
                  "the within accuracy of the level it is TESTED on ('keeps X of it').",
                  "  T1 task across congruency   -> the clean positive control: transfer ~",
                  "                                 within means the pipeline carries a code",
                  "                                 from one trial population to another",
                  "  T2 task across switch type  -> confounded: on a switch trial the previous",
                  "                                 task was the other one, so a drop is",
                  "                                 expected even with a single task code",
                  "  T3 / T4 congruency / switch -> the A4 contrasts, across task, at their",
                  "  type across task               own effect size; set 'keeps X' beside the",
                  "                                 A4 congruency <-> switch transfer's",
                  "The task cue (frame colour) is drawn with the stimulus, so a task decoder is",
                  "partly visual: T1 validates the code path, not congruency's effect size.",
                  "Resamples are not independent subjects, so window-wise p-values are optimistic.",
                  "=" * 72]
    else:
        lines += ["=" * 72,
                  "Reading (docs/n3b_block_transfer.md §1.6): compare each transfer with the",
                  "within accuracy of the level it is TESTED on.",
                  "  transfer ~ within, centered and uncentered  -> the same code in both levels",
                  "  below within uncentered only                -> same axis, a tonic block shift",
                  "  below within centered, in BOTH directions   -> block context reorganizes the",
                  "                                                 code; check its cross-factor control",
                  "  below in one direction only                 -> the training level's code is",
                  "                                                 weaker, not a different axis",
                  "Resamples are not independent subjects, so window-wise p-values are optimistic.",
                  "=" * 72]
    txt = "\n".join(lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


def run_block_transfer_job(args):
    """N3b, or the task-transfer controls, on every electrode of the decoded ROI
    array, with no electrode groups.

    `args.analysis` picks the design table: 'block_transfer' (the default) runs
    BLOCK_TRANSFER_DESIGNS, 'task_transfer' runs TASK_TRANSFER_DESIGNS.
    `args.electrodes` alone decides whether the decoded electrodes are the
    task-significant ('sig') or all ('all') electrodes of `args.roi`. Every design
    runs uncentered and centered and writes <analysis>.json,
    <analysis>_traces.npz, summary.txt and figures.
    """
    analysis = getattr(args, 'analysis', 'block_transfer')
    designs = TRANSFER_ANALYSES[analysis]
    cluster_kw = dict(n_perm=getattr(args, 'n_perm', 200), seed=getattr(args, 'seed', 42))
    if args.data_source == 'synthetic':
        # Synthetic epochs have no pre-stimulus period, so every window counts as post.
        args = SimpleNamespace(**{**vars(args), 'first_time_point': 0.0})
        code = getattr(args, 'synthetic_code', None)
        roi = 'synthetic'
        if analysis == 'task_transfer':
            # Planted answers: 'congruency_specific' puts task on a different axis
            # on incongruent trials, so T1 must fail while T2 transfers; 'carryover'
            # adds previous-task activity, which T2 and T4 cannot escape and T1 can.
            cells = cd.synthetic_task_condition_cells()
            arrays = cd.synthetic_task_labeled_arrays(
                task_code='congruency_specific' if code == 'congruency_specific' else 'shared',
                carryover=0.6 if code == 'carryover' else 0.0, seed=getattr(args, 'seed', 0))
        else:
            # A planted answer: 'block_specific' puts congruency on a different axis in
            # each incongruent-proportion level, so X1 must fail while X3 transfers.
            cells = cd.synthetic_condition_cells()
            arrays = cd.synthetic_roi_labeled_arrays(
                code='orthogonal', design_proportions=True, seed=getattr(args, 'seed', 0),
                block_code='specific' if code == 'block_specific' else 'same')
        n_channels = next(iter(arrays[roi].values())).shape[1]
    else:
        from src.analysis.utils.general_utils import resolve_lab_root
        # no electrode definition here: ELECTRODES alone picks 'sig' or 'all'
        roi = args.roi
        LAB_root = resolve_lab_root(args.LAB_root)
        n_channels = None

    results, traces, loaded = {}, {}, {}
    for name, (contrast, block_col, conditions_name) in designs.items():
        if args.data_source != 'synthetic':
            # T1/T3 and T2/T4 decode the same condition set; load each set once
            if conditions_name not in loaded:
                design_args = SimpleNamespace(
                    **{**vars(args),
                       'conditions': getattr(experiment_conditions, conditions_name)})
                loaded[conditions_name] = _build_roi_arrays(
                    design_args, LAB_root, required_fields=(contrast, block_col))
            roi, arrays, channel_names, cells = loaded[conditions_name]
            if n_channels is None:
                n_channels = len(channel_names)
            elif n_channels != len(channel_names):
                raise ValueError(f"{analysis} condition sets produced different "
                                 "electrode counts")
        for center in (False, True):
            key = f"{name}_{'centered' if center else 'uncentered'}"
            print(f"{analysis} {key}: {contrast} across {block_col}")
            out = bt.run_block_transfer(
                arrays, roi, cells, contrast, block_col, center=center,
                n_resamples=args.n_repeats, n_splits=args.n_splits,
                explained_variance=args.explained_variance, window=args.window_size,
                step_size=args.step_size, frac_train=getattr(args, 'frac_train', None),
                seed=getattr(args, 'seed', 0))
            print(f"   trials per contrast x transfer cell: {out['group_sizes']} "
                  f"-> {out['n_per_group']} of each kept per resample")
            summary = _summarise_block_transfer(out, args, cluster_kw)
            results[key] = dict(
                design=name, contrast=contrast, block_col=block_col, centered=center,
                conditions_name=conditions_name,
                levels=out['levels'], group_sizes=out['group_sizes'],
                n_per_group=out['n_per_group'],
                cells={f'{train}->{test}': s for (train, test), s in summary.items()})
            for (train, test), s in summary.items():
                tag = _transfer_tag(block_col, train, test)
                traces[f'{key}_{tag}_true'] = s['acc_true']
                traces[f'{key}_{tag}_shuffle'] = s['acc_shuffle']

    meta = dict(analysis=analysis, data_source=args.data_source, roi=roi,
                electrodes=args.electrodes, n_channels=n_channels,
                epochs_root_file=getattr(args, 'epochs_root_file', None),
                window_size=args.window_size, step_size=args.step_size,
                n_splits=args.n_splits, n_resamples=args.n_repeats,
                n_perm=cluster_kw['n_perm'], seed=cluster_kw['seed'],
                save_dir=args.save_dir)
    with open(os.path.join(args.save_dir, f'{analysis}.json'), 'w') as f:
        json.dump(_json_safe(_strip_arrays(dict(meta, designs=results))), f, indent=2)
    np.savez(os.path.join(args.save_dir, f'{analysis}_traces.npz'), **traces)
    _plot_block_transfer(results, args, roi)
    _write_block_transfer_summary(results, meta, args.save_dir)
    return results


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if getattr(args, 'analysis', 'a4') in TRANSFER_ANALYSES:
        return run_block_transfer_job(args)

    # Decoder settings shared by every decode in this job.
    dec_kw = dict(n_splits=args.n_splits, n_repeats=args.n_repeats,
                  explained_variance=args.explained_variance,
                  window=args.window_size, step_size=args.step_size,
                  frac_train=getattr(args, 'frac_train', None),
                  random_state=getattr(args, 'seed', 42))
    cluster_kw = dict(n_perm=getattr(args, 'n_perm', 200),
                      seed=getattr(args, 'seed', 42))

    # 1. assemble the ROI LabeledArrays + electrode groups -----------------------
    trial_partitions = None
    if getattr(args, 'electrode_selection_split', False):
        if args.data_source != 'real' or getattr(args, 'electrode_definition', 'anova') != 'anova':
            raise ValueError(
                "ELECTRODE_SELECTION_SPLIT is supported for real data with "
                "ELECTRODE_DEFINITION=anova. A saved CSV has no trial-membership "
                "record, so this job cannot prove that its defining trials are "
                "disjoint; generate the CSV on a saved selection partition or "
                "use the in-job ANOVA split.")

    if args.data_source == 'synthetic':
        print(f"DATA SOURCE: synthetic ({args.synthetic_code} code) — validates the path "
              "and that A4 discriminates shared vs orthogonal codes")
        roi = 'synthetic'
        arrays = cd.synthetic_roi_labeled_arrays(
            code=args.synthetic_code, seed=getattr(args, 'seed', 0))
        cells = cd.synthetic_condition_cells()
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

        definition = getattr(args, 'electrode_definition', 'anova')
        print(f"ELECTRODE DEFINITION: {definition}")
        LAB_root = resolve_lab_root(args.LAB_root)

        # (i) the electrode definition. The power-traces route reads finished
        #     windowed-ANOVA runs, so it needs neither the epochs nor the long
        #     single-trial table; the in-job ANOVA route builds both.
        if definition == 'power_traces':
            labels = _resolve_labels(args)
        else:
            from dcc_scripts.stats.stability_flexibility_segregation_dcc import assemble_long_df
            subjects_epochs = load_HG_ev1_rescaled_per_subject(
                subjects=args.subjects, epochs_root_file=args.epochs_root_file,
                task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
            if getattr(args, 'electrode_selection_split', False):
                subjects_epochs, trial_partitions = _split_subject_epochs(
                    subjects_epochs,
                    frac_select=args.electrode_selection_frac,
                    seed=args.electrode_selection_seed)
                print(f"ANOVA electrode selection uses {args.electrode_selection_frac:.0%} "
                      f"of trials over [{args.window_tmin}, {args.window_tmax}] s; "
                      "all decoding uses the disjoint remainder")
            keep = resolve_electrodes_to_keep(args, LAB_root)
            df = assemble_long_df(subjects_epochs, args.window_tmin, args.window_tmax,
                                  electrodes_to_keep=keep, effect_measure=EFFECT_MEASURE)
            print(f"assembled cluster df: {len(df)} rows | {df.subject.nunique()} subjects | "
                  f"{df.electrode.nunique()} electrodes")
            labels = _resolve_labels(args, df)

        contrast_mode = getattr(args, 'contrast_mode', CONTRAST_MODE)
        a1_groups = _electrode_groups(labels, contrast_mode)
        labels.to_csv(os.path.join(args.save_dir, 'anova_labels.csv'), index=False)
        interaction_groups = _interaction_groups(labels, contrast_mode)
        print("A1 electrode groups: "
              + "  ".join(f"{g}={len(v)}" for g, v in a1_groups.items()))

        # (ii) the decode runs on the ordinary ROI LabeledArray pseudopopulation
        roi, arrays, channel_names, cells = _build_roi_arrays(
            args, LAB_root, trial_partitions=trial_partitions)

    # The unselected reference set, so label transfer / temporal generalization
    # are not only ever read off interaction-selected subsets.
    a1_groups = _add_reference_group(a1_groups, channel_names, args)
    print("decoded electrode groups: "
          + "  ".join(f"{g}={len(v)}" for g, v in a1_groups.items()))

    # Contrasts and block sets, read off each condition's declared factor levels
    # rather than parsed out of its name (see `cd.condition_cells`).
    # A transfer is only identifiable if the two factors CROSS. Declaring both is
    # not enough: a set like `stimulus_iS_cR_err_conditions` holds only the iS and
    # cR cells, where congruency and switchType split the trials identically, so
    # every "cross" decode below would silently re-report the within-contrast
    # decode as perfect transfer. Fail here rather than emit that number.
    if not cd.factors_are_crossed(cells):
        raise ValueError(
            f"congruency and switchType do not CROSS in this condition set "
            f"(cells: {sorted(cells)}), so a cross-decode is not identifiable — "
            "training on one contrast and scoring the other would measure the "
            "contrast that was trained on. Use a set in which all four "
            "congruency x switchType combinations are present: "
            "stimulus_experiment_conditions (full 2x2x2x2) or "
            "stimulus_main_effect_conditions (pooled over both proportions).")

    stab_strings, flex_strings = cd.stability_flexibility_strings(cells)
    contrast_strings = {'stability': stab_strings, 'congruency': stab_strings,
                        'flexibility': flex_strings, 'switchtype': flex_strings}
    requested_train = getattr(args, 'train_label', None)
    requested_test = getattr(args, 'test_label', None)
    if requested_train:
        train_key, test_key = requested_train.lower(), requested_test.lower()
        unknown = {train_key, test_key} - set(contrast_strings)
        if unknown:
            raise ValueError(f"TRAIN_LABEL/TEST_LABEL must be stability, congruency, "
                             f"flexibility, or switchType; got {sorted(unknown)}")
        transfer_pairs = [(f'{train_key}_to_{test_key}',
                           (contrast_strings[train_key], contrast_strings[test_key]))]
    else:
        # The two within-contrast decodes are each transfer's ceiling: same trials,
        # same folds, scored on the labelling the transfer is scored on
        # (docs/cross_decoding_controls.md §2). A null transfer means nothing
        # unless its ceiling beats chance.
        transfer_pairs = [
            ('stab_to_stab', (stab_strings, stab_strings)),
            ('flex_to_flex', (flex_strings, flex_strings)),
            ('stab_to_flex', (stab_strings, flex_strings)),
            ('flex_to_stab', (flex_strings, stab_strings))]
    # A condition set that POOLS over a proportion (e.g.
    # `stimulus_main_effect_conditions`, the 2x2 for the all-vs-all transfer) has
    # no block contrast to make, so the block-split designs below are skipped
    # rather than run on a constant. Label transfer and temporal generalization
    # are unaffected: they never split by block in the first place.
    blocks = {f: cd.block_condition_sets(cells, f)
              for f in ('incongruent_proportion', 'switch_proportion')
              if cd.has_block_factor(cells, f)}
    if blocks:
        print("block levels: "
              + "  ".join(f"{f}={sorted(levels)}" for f, levels in blocks.items()))
    else:
        print("block levels: none — the condition set pools over both proportions, "
              "so the within-block designs A4(0)/A4(0b) are skipped and only the "
              "pooled label transfer / temporal generalization run")

    results = {}

    # 2. (0) within-block decoding baseline (Fig 9) ------------------------------
    # "Decode a contrast within one block level" is an ordinary decode over that
    # block's conditions — restrict the conditions, then train == test contrast.
    print("A4(0): within-block decoding baseline (Fig 9)")
    within_block = {}
    for cname, strings, block_col in (
            ('congruency (LWPC)', stab_strings, 'incongruent_proportion'),
            ('switchType (LWPS)', flex_strings, 'switch_proportion')):
        if block_col not in blocks:
            print(f"     skipping {cname}: the condition set pools over {block_col}")
            continue
        per_block = {}
        for level, conds in blocks[block_col].items():
            tag = f'{level}% {_BLOCK_TAG[block_col]}'
            try:
                sub_arrays = cd.filter_conditions(arrays, roi, conds)
            except ValueError as e:
                print(f"     skipping block {tag}: {e}")
                continue
            out = cd.run_cross_decoding(sub_arrays, roi, strings, strings, **dec_kw)
            per_block[tag] = _summarise(out, **cluster_kw)
        # low level first (block_condition_sets sorts), so this is high - low
        if len(per_block) == 2:
            low, high = (per_block[t]['mean_accuracy'] for t in per_block)
            diff = high - low
        else:
            diff = None
        within_block[cname] = dict(per_block=per_block, block_difference=diff)
    results['within_block'] = within_block

    # 2b. the within-block 2x2 restricted to each interaction-defined group.
    #     Without a disjoint split, SKIP define == decode to avoid double-dipping;
    #     with the split, selection and decoding trials are independent, so keep it.
    #     Only the OFF-diagonal (cross) cells are computed/kept — e.g. the CPC
    #     electrode set is decoded on switchType/switch_prop and the two cross
    #     cells, never on congruency/inc_prop (the interaction that defined it).
    #     To keep the diagonal cell instead, define the electrodes on a disjoint
    #     set of trials (`trial_splitting.apply_electrode_definition_split`).
    if interaction_groups and blocks:
        diagonal = ("included (disjoint selection/decode trials)"
                    if getattr(args, 'electrode_selection_split', False)
                    else "ignored (same-trial circularity guard)")
        print(f"A4(0b): per-group within-block 2x2 (diagonal {diagonal})")
        decode_cells = [(contrast, block_col, strings)
                        for contrast, block_col, strings in (
                            ('congruency', 'incongruent_proportion', stab_strings),
                            ('congruency', 'switch_proportion', stab_strings),
                            ('switchType', 'switch_proportion', flex_strings),
                            ('switchType', 'incongruent_proportion', flex_strings))
                        if block_col in blocks]
        per_group = {}
        for gflag, elset in interaction_groups.items():
            restricted, n_kept = _restrict_to_electrodes(arrays, roi, channel_names, elset)
            if n_kept < args.min_group_size:
                print(f"     group '{gflag}' has {n_kept} electrodes in ROI "
                      f"(< {args.min_group_size}); skipping")
                continue
            decoded = {}
            for contrast, block_col, strings in decode_cells:
                if (not getattr(args, 'electrode_selection_split', False)
                        and cd.is_circular_decode(gflag, contrast, block_col)):
                    continue                     # double-dipping: ignore this result
                for level, conds in blocks[block_col].items():
                    try:
                        sub_arrays = cd.filter_conditions(restricted, roi, conds)
                    except ValueError:
                        continue
                    out = cd.run_cross_decoding(sub_arrays, roi, strings, strings, **dec_kw)
                    tag = f'{contrast} by {block_col} [{level}%]'
                    decoded[tag] = _summarise(out, **cluster_kw)
            per_group[gflag] = dict(n_electrodes=n_kept,
                                    ignored_cell=cd.circular_decode_for_group(gflag),
                                    cells=decoded)
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
        for direction, (tr, te) in transfer_pairs:
            out = cd.run_cross_decoding(restricted, roi, tr, te, **dec_kw)
            entry[direction] = _summarise(out, **cluster_kw)
            entry[direction]['n_channels'] = n_kept
        for direction, ceiling in _TRANSFER_CEILINGS.items():
            if direction in entry and ceiling in entry:
                below, _ = perform_time_perm_cluster_test_for_accuracies(
                    entry[ceiling]['acc_true'], entry[direction]['acc_true'], **cluster_kw)
                entry[direction]['n_below_ceiling'] = int(np.asarray(below).astype(bool).sum())
                entry[direction]['retained'] = _retained(entry[direction], entry[ceiling])
        lt[g] = entry
    results['label_transfer'] = lt

    # 4. (c) temporal generalization (Fig 10) -----------------------------------
    # Each matrix costs n_windows^2 decodes, so this runs on `args.tempgen_groups`
    # (default: the 'both' group only) rather than on every group above. Add the
    # reference group there to get the unselected comparison matrix too.
    # The runner supplies the default ('both'). Preserve an explicitly empty
    # tuple so TEMPGEN_GROUPS='' really disables the expensive n_windows² stage.
    tempgen_groups = getattr(args, 'tempgen_groups', ('both',))
    print(f"A4(c): temporal generalization (Fig 10) on {list(tempgen_groups)}")
    results['temporal'] = {}
    for g in tempgen_groups:
        if g not in a1_groups:
            # e.g. the reference group was folded into an identical existing one
            print(f"     no electrode group named '{g}' "
                  f"(have {sorted(a1_groups)}); skipping")
            continue
        restricted, n_kept = _restrict_to_electrodes(
            arrays, roi, channel_names, a1_groups[g])
        if n_kept < args.min_group_size:
            print(f"     group '{g}' has {n_kept} electrodes in ROI "
                  f"(< {args.min_group_size}); skipping")
            continue
        # n_windows^2 predictions — halve the repeats to keep the runtime sane
        tg_kw = dict(dec_kw, n_repeats=max(2, args.n_repeats // 2),
                     temporal_generalization=True)
        tempgen_pairs = (transfer_pairs if requested_train else [
            ('stability (within)', (stab_strings, stab_strings)),
            ('flexibility (within)', (flex_strings, flex_strings)),
            ('stability->flexibility (cross)', (stab_strings, flex_strings))])
        for name, (tr, te) in tempgen_pairs:
            out = cd.run_cross_decoding(restricted, roi, tr, te, **tg_kw)
            results['temporal'][f'{name} [{g}]'] = dict(
                matrix=_tempgen_matrix(out), n_channels=n_kept, group=g)
    if not results['temporal']:
        del results['temporal']

    # 5. persist + plot + summarize ----------------------------------------------
    save_results(results, args.save_dir)
    make_plots(
        results, args.save_dir,
        first_time_point=getattr(args, 'first_time_point', -1.0),
        sampling_rate=getattr(args, 'sampling_rate', 256),
        window_size=args.window_size, step_size=args.step_size)
    write_summary(results, args.save_dir, meta=dict(
        data_source=args.data_source,
        synthetic_code=getattr(args, 'synthetic_code', None),
        task=getattr(args, 'task', None),
        epochs_root_file=getattr(args, 'epochs_root_file', None),
        electrodes=getattr(args, 'electrodes', None),
        electrode_definition=getattr(args, 'electrode_definition', 'anova'),
        power_traces_correction=(getattr(args, 'power_traces_correction', None)
                                 if getattr(args, 'electrode_definition', 'anova')
                                 == 'power_traces' else None),
        reference_group=getattr(args, 'reference_group', 'all'),
        electrode_group_sizes={g: len(v) for g, v in a1_groups.items()},
        window=f"[{getattr(args, 'window_tmin', None)}, {getattr(args, 'window_tmax', None)}]s",
        window_size=args.window_size, step_size=args.step_size,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        frac_train=getattr(args, 'frac_train', None),
        alpha=getattr(args, 'alpha', None), save_dir=args.save_dir))
    return results
