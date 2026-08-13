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
      the decoded ROI array). Prediction: only 'both' cross-decodes; the
      reference group says what the region does before any selection.
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

# Class definitions are DERIVED from each condition's declared factor levels
# (`cd.condition_cells`), not hand-written as substrings of the condition names.
# The real and synthetic naming conventions collide on shorthand tokens — '75s'
# is "switch trial in the 75%-incongruent block" under
# `stimulus_experiment_conditions` and "75%-switch block" under the synthetic
# generator — and picking the wrong one decodes the wrong contrast silently
# rather than raising. See the note above `cd.condition_cells`.

# how each block factor is spelled in result keys and figures
_BLOCK_TAG = {'incongruent_proportion': 'incongruent', 'switch_proportion': 'switch'}


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
              "A4(a) label transfer (train stability, test flexibility) by group:",
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


def _electrode_groups(labels):
    """The three DISJOINT label-transfer groups, as ROI-array channel names."""
    chan = _channel_keys(labels)
    S = (labels['S'] == 1).to_numpy()
    F = (labels['F'] == 1).to_numpy()
    return {
        'both': chan[S & F].tolist(),
        'S_only': chan[S & ~F].tolist(),
        'F_only': chan[~S & F].tolist(),
    }


def _interaction_groups(labels):
    """The FOUR interaction-defined electrode sets (possibly overlapping), keyed by
    the definition-group flag (CPC/SPS/CPS/SPC) so `cd.is_circular_decode` can name
    each set's double-dip cell. Used for the per-group within-block 2x2 that skips
    the diagonal (define==decode) cell."""
    chan = _channel_keys(labels)
    return {flag: chan[(labels[flag] == 1).to_numpy()].tolist()
            for flag in ('CPC', 'SPS', 'CPS', 'SPC') if flag in labels.columns}


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
def _build_roi_arrays(args, LAB_root, trial_partitions=None):
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

    cells = cd.condition_cells(args.conditions)
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

        a1_groups = _electrode_groups(labels)
        labels.to_csv(os.path.join(args.save_dir, 'anova_labels.csv'), index=False)
        interaction_groups = _interaction_groups(labels)
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
        transfer_pairs = [
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
