# Refactoring Guide — Making the Codebase Deeper, Not Wider

This document is for anyone (you, a labmate, or an AI assistant) who needs to
break up the large `.py` files in `src/analysis/` so that implementing a new
feature means reading **one small file**, not scrolling through a 4,700-line
monolith.

It is a companion to `docs/analysis_guide.md` (which tells you *where each
analysis lives*). This doc tells you *how to reshape a file once it has grown
too big to hold in your head*.

---

## 0. Status — what has already been done

**`decoding/decoding.py` has been split** (the §4 plan below, executed). The
4,752-line monolith is now a 125-line **facade** that re-exports from focused
submodules:

| Module | Lines | Holds |
|--------|------:|-------|
| `decoding.py` | 125 | facade — re-exports every public name (nothing else) |
| `data_prep.py` | 284 | balancing, `mixup2`, `flatten_features`, `sample_fold` |
| `decoder.py` | 471 | the `Decoder` class + `cv_cm_*` methods |
| `accuracy_stats.py` | 970 | permutation / bootstrap / cluster stats on accuracies |
| `tfr_cluster.py` | 550 | sig-TFR masks + cluster decoding |
| `roi_confusion.py` | 262 | per-ROI confusion-matrix orchestration |
| `context_comparison.py` | 293 | `run_context_comparison_analysis` + overlay |
| `plots/accuracies.py` | 866 | nature-style accuracy plots |
| `plots/confusion.py` | 377 | confusion-matrix + cm-trace plots |
| `plots/trajectories.py` | 673 | PCA / UMAP projections + trajectories |
| `plots/style.py` | 28 | shared `NATURE_STYLE` constant |

Every one of the 46 original functions/classes was moved **verbatim** into
exactly one module, and every existing `from src.analysis.decoding.decoding
import ...` still resolves through the facade — no caller was touched. The two
"cheap wins" (§7) were also applied to this package: the `general_utils`
star-import and the hardcoded `C:/Users/...` path are gone from `decoding.py`.

**Two pre-existing bugs surfaced during the split** (both were already latent in
the old monolith; neither was introduced here, and neither was silently
"fixed" so the refactor stays a pure move):

1. `plot_and_save_tfr_masks` (now in `plots/confusion.py`) calls
   `plot_mask_pages`, which is **never imported** — it lives in
   `spec/wavelet_functions.py`. That code path raises `NameError` if reached.
   Fix: add `from src.analysis.spec.wavelet_functions import plot_mask_pages`
   (verify it doesn't create an import cycle first).
2. `dcc_scripts/spec/get_sig_tfr_differences_dcc.py` and
   `dcc_scripts/decoding/james_sun_cluster_decoding_dcc.py` both
   `import plot_accuracies` from `decoding.decoding`, but **no such function
   exists** (likely a stale rename of `plot_accuracies_nature_style`). Those
   imports were already broken.

**`power/power_traces.py` has been split** (the §5 plan below, executed). The
~2,420-line monolith is now an 81-line **facade** that re-exports from three
focused submodules:

| Module | Lines | Holds |
|--------|------:|-------|
| `power_traces.py` | 81 | facade — re-exports every public name (nothing else) |
| `evoked_builders.py` | 405 | per-electrode / multi-channel evokeds, ROI grand averages, condition subtraction, `time_perm_cluster_between_two_evokeds` |
| `windowed_anova.py` | 1,170 | long-form windowed dataframe, per-window OLS/ANOVA fits, within-/across-electrode permutation cluster correction, FDR, `load_significant_electrodes` |
| `plots.py` | 826 | `plot_power_trace(s)_for_roi(s)`, 2-way / 16-condition interaction plots, `DEFAULT_PLOT_STYLE`, style + color helpers, `anova_results_to_interaction_results_for_plotting` |

All 43 original functions were moved **verbatim** into exactly one module, and
every existing `from src.analysis.power.power_traces import ...` still resolves
through the facade — no caller was touched. The two cheap wins (§7) were applied
here too: the `general_utils` explicit-import list had no star-import to remove,
but the six unused `general_utils` names and the `find_significant_clusters_...`
import were dead and are gone, and the `sys.path`/`__file__` juggling at the top
of the file was deleted (relies on `pip install -e .`).

**One pre-existing bug surfaced during the split and was fixed** (latent in the
old monolith, not introduced by the move): `apply_fdr_correction_to_windowed_results`
(now in `windowed_anova.py`) calls `multipletests`, which the monolith only ever
imported **locally inside** `run_within_electrode_windowed_anova_cluster_correction`
— so it was never in module scope, and that code path raised `NameError` if
reached. Fixed by hoisting `from statsmodels.stats.multitest import multipletests`
to module level (and dropping the now-redundant local import).

**Still monolithic** (not yet split): `utils/general_utils.py` (§6).

### Installation (needed now that the path hacks are gone)

The decoding modules no longer patch `sys.path`, so `src` and `ieeg` must be
importable the normal way — i.e. **`pip install -e .` from the repo root is now
a prerequisite, not a convenience**. The instructions are in the repo-root
[`README.md`](../README.md) ("Python environment setup"); run it once per
environment, not once per session.

The other analysis files (preproc, spec, dcc_scripts) still carry the old
`sys.path.append("C:/Users/jz421/...")` line; sweeping those is the natural
next cheap win now that `pip install -e .` provides `ieeg` on the path.

---

## 1. The goal, stated precisely

The problem with the big files is **not** that they do too much work — it's
that they mix unrelated concerns behind **no interface**. To change any one
thing you must load all of them.

We are borrowing one idea from John Ousterhout's *A Philosophy of Software
Design*: prefer **deep modules** — a *small, obvious interface* hiding a *large
implementation*. A file you import three well-named functions from, and never
otherwise open, is deep. A file where you must read all 4,700 lines to find the
three you need is **wide**, and wide is the thing we are killing.

### The trap to avoid

"Make it deeper" does **not** mean "add layers." A `manager` that calls a
`handler` that calls a `service` is *more* shallow modules stacked up — now you
chase one feature through six files. That is worse than the monolith.

> **The rule:** split by **concern**, not by adding call-tree depth.
> After a split, adding a feature should mean: open *one* file whose name tells
> you it's the right one, import 2–3 named helpers, and never scroll past code
> unrelated to your task.

"Deeper" here = **narrower files behind clear names**, not longer call chains.

---

## 2. Priority order (highest leverage first)

| File | Lines | Why it hurts | Target |
|------|------:|--------------|--------|
| `decoding/decoding.py` | ~4,750 | 8 concerns in one file; imported by 8+ modules | §4 below |
| `power/power_traces.py` | ~2,420 | evoked-building + ANOVA + plotting mixed | §5 below |
| `utils/general_utils.py` | ~2,300 | a domain-agnostic **grab-bag**, star-imported everywhere | §6 below |
| `config/experiment_conditions.py` | ~1,180 | mostly *data*, lower urgency | leave until it blocks you |

**Do not do all of this at once.** The pattern that actually works:
**refactor the file you are about to work in, right before you add the
feature.** Splitting `decoding.py` pays for itself the moment you start
assignment **A4 (cross-decoding)**, which drops a new file into that package.

---

## 3. The safe mechanical recipe (use this for every split)

Eight-plus files import from `decoding.py` alone
(`process_bootstrap.py`, `power_traces.py`, three `dcc_scripts/*`, the
`docs/skeletons/`, etc.). You **cannot** move functions and break every caller.
Do this instead, **one concern at a time**:

1. **Pin behavior first.** Run the existing tests and confirm green *before
   touching anything* (`tests/analysis/decoding/test_decoding.py` is 1,361
   lines of safety net). If a target has no test, that is a signal — add a thin
   smoke test before refactoring it.

2. **Extract one concern.** Start with the **leaf-most, lowest-risk** group —
   the functions nothing else depends on (e.g. the PCA/UMAP trajectory plots).
   Move them **verbatim** into a new module; fix only their imports.

3. **Keep the old file as a facade.** At the bottom of the old file, re-export
   what you moved:

   ```python
   # decoding.py — kept as a thin facade during migration
   from .plots.trajectories import (
       plot_static_pca_projection,
       plot_pca_over_time,
       plot_pca_3d_trajectory,
       plot_high_dim_decision_slice,
       plot_static_umap_projection,
       plot_umap_3d_trajectory,
   )
   ```

   Every existing `from src.analysis.decoding.decoding import ...` keeps working
   **untouched**. This is the key move: it decouples *where code lives* from
   *what callers type*, so you refactor without a big-bang rename.

4. **Run the tests again.** Green → **commit**. One concern = one small,
   reviewable, reversible commit.

5. **Repeat** for the next concern. Migrate callers to the real module paths
   *opportunistically, later*, once things have settled. Delete a re-export only
   after `grep` shows no one imports it from the old location.

> **Prefer explicit re-exports over `from x import *` in the facade.** Star
> re-exports hide what's public and reintroduce the very problem in §7.

---

## 4. Worked plan: `decoding/decoding.py`

The file already clusters into eight concerns. Target layout:

```
src/analysis/decoding/
  decoder.py            # Decoder class + cv_cm_* methods
  data_prep.py          # balancing / mixup / fold sampling / flatten
  roi_confusion.py      # get_confusion_matrices_for_rois_* orchestration
  tfr_cluster.py        # sig-TFR masks + cluster decoding
  accuracy_stats.py     # permutation / bootstrap / cluster stats on accuracies
  plots/
    __init__.py
    accuracies.py       # nature-style accuracy + multi-cluster plots
    confusion.py        # confusion-matrix + cm-trace plots
    trajectories.py     # PCA / UMAP projections + 3D trajectories
  context_comparison.py # run_context_comparison_analysis orchestration
  decoding.py           # thin facade re-exporting the public names
```

Extraction order (leaf-most → most-depended-on), with the functions to move:

| Step | New module | Functions to move (from `decoding.py`) |
|-----:|-----------|----------------------------------------|
| 1 | `plots/trajectories.py` | `plot_static_pca_projection`, `plot_pca_over_time`, `plot_pca_3d_trajectory`, `plot_high_dim_decision_slice`, `plot_static_umap_projection`, `plot_umap_3d_trajectory` |
| 2 | `plots/accuracies.py` | `plot_accuracies_nature_style`, `create_multipanel_nature_figure`, `plot_true_vs_shuffle_accuracies`, `plot_accuracies_with_multiple_sig_clusters`, `find_contiguous_clusters` |
| 3 | `plots/confusion.py` | `get_display_labels_from_cats`, `plot_and_save_confusion_matrix`, `plot_and_save_tfr_masks`, `extract_pooled_cm_traces`, `plot_cm_traces_nature_style` |
| 4 | `data_prep.py` | `concatenate_and_balance_data_for_decoding`, `mixup2`, `flatten_features`, `sample_fold` |
| 5 | `accuracy_stats.py` | `compute_accuracies`, `perform_time_perm_cluster_test_for_accuracies`, `make_pooled_shuffle_distribution`, `find_significant_clusters_of_series_vs_distribution_based_on_percentile`, `find_cluster_lengths`, `get_max_perm_cluster_lengths_based_on_percentile`, `compute_pooled_bootstrap_statistics`, `do_time_perm_cluster_comparing_*` (×2), `do_mne_paired_cluster_test`, `get_time_averaged_confusion_matrix`, `_run_single_permutation`, `cluster_perm_paired_ttest_by_duration`, `run_two_one_tailed_tests_with_time_perm_cluster`, `get_pooled_accuracy_distributions_for_comparison` |
| 6 | `tfr_cluster.py` | `decode_on_sig_tfr_clusters`, `compute_sig_tfr_masks_from_roi_labeled_array`, `compute_sig_tfr_masks_for_specified_channels`, `compute_sig_tfr_masks_from_concatenated_data`, `apply_tfr_masks_and_flatten_to_make_decoding_matrix`, `get_confusion_matrix_for_rois_tfr_cluster` |
| 7 | `roi_confusion.py` | `get_and_plot_confusion_matrix_for_rois_jim`, `get_confusion_matrices_for_rois_time_window_decoding_jim` |
| 8 | `decoder.py` | `Decoder` class (and its `_window_and_predict_minimal`, `cv_cm_*`, `fit_predict`, `calculate_scores` methods) |
| 9 | `context_comparison.py` | `run_context_comparison_analysis`, `plot_cross_block_overlay` |

Do steps 1–3 (the plotting concerns — safest, most self-contained) first and
you've already carved ~1,300 lines of pure plotting out of the hot path.

---

## 5. Worked plan: `power/power_traces.py`

Three concerns, already visible from the `_private` helper clusters:

```
src/analysis/power/
  evoked_builders.py    # combine/extract/make_*_evokeds, ROI grand averages
  windowed_anova.py     # process_windowed_data_for_anova, create/perform_*_anova,
                        #   run_within_electrode_windowed_anova_cluster_correction,
                        #   FDR correction, load_significant_electrodes
  plots.py              # plot_power_trace(s)_for_roi(s), 2way/16-condition
                        #   interaction plots, apply_plot_style, color helpers
  power_traces.py       # facade re-exporting the public names
```

Same recipe as §3. `plots.py` is the safe first extraction.

---

## 6. Worked plan: `utils/general_utils.py`

This one is a **grab-bag**, not a domain module, and it is star-imported
across the codebase — so the facade step (§3.3) matters most here. Split by
domain:

```
src/analysis/utils/
  io.py          # load/save subjects↔ROI dicts, mne objects, sig-chans, acc arrays
  epochs.py      # get_trials(+outlier variants), handle_outliers, NaN imputation,
                 #   filter_and_average_epochs, windower
  rois.py        # make_/filter_/sig_electrodes_per_subject_roi machinery
  stats.py       # permutation_test, within/across-electrode permutation tests,
                 #   ANOVA helpers, extract_significant_effects
  lab_paths.py   # get_default_LAB_root, resolve_lab_root, _subdir
  general_utils.py  # facade re-exporting everything above (keeps `import *` callers alive)
```

Because callers do `from ...general_utils import *`, keep `general_utils.py`
re-exporting **all** public names until you've migrated those callers to
explicit imports (see §7).

---

## 7. Two cheap wins to fold in

These are independent of the splits and each directly reduces how much context
you must load:

1. **Kill `from ...general_utils import *`** (e.g. `decoding.py` line ~82, which
   even carries a `# TODO: fix these` next to it). Star-imports are why you
   can't tell where a name comes from — they force you to mentally load a
   2,300-line file to read any file that stars it. Replace with explicit
   imports; jump-to-definition then works.

2. **Delete the hardcoded `sys.path.append("C:/Users/jz421/Desktop/...")`** and
   the `__file__` path juggling at the top of the analysis files. Run
   `pip install -e .` (there is already a `setup.py`) once, and `src.analysis...`
   is importable everywhere — no per-file path hacks.

---

## 8. Definition of done (per split)

A split is finished when:

- [ ] Each new module is a single concern, roughly 200–600 lines.
- [ ] The old filename still imports and re-exports every previously-public name
      (nothing downstream broke).
- [ ] `tests/analysis/...` is green — same tests, same results as before.
- [ ] No new `import *`; the facade uses **explicit** re-exports.
- [ ] The change is one commit per concern, each independently revertible.

You'll know it worked when the next feature (e.g. `decoding/cross_decoding.py`
for assignment A4) is a new peer file that imports `data_prep` and
`accuracy_stats` — and you never have to open the old monolith to write it.
