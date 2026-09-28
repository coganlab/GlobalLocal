# Code maintenance

Notes on the shape of the codebase rather than on any analysis.

| Part | What it covers | Was |
|---|---|---|
| [Refactoring guide](#refactoring-guide) | How the big modules were split (`decoding/`, `power/`), and the recipe for splitting the next one | `refactoring_guide.md` |
| [Consolidation candidates](#consolidation-candidates) | Duplicated and near-duplicated code, ranked by value ÷ risk. A decision list: nothing in it has been changed | `consolidation_candidates.md` |

Each part keeps its own section numbers.

---

## Refactoring guide

*Refactoring Guide — Making the Codebase Deeper, Not Wider*

This document is for anyone (you, a labmate, or an AI assistant) who needs to
break up the large `.py` files in `src/analysis/` so that implementing a new
feature means reading **one small file**, not scrolling through a 4,700-line
monolith.

It is a companion to `docs/analysis_guide.md` (which tells you *where each
analysis lives*). This doc tells you *how to reshape a file once it has grown
too big to hold in your head*.

---

### 0. Status — what has already been done

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

#### Installation (needed now that the path hacks are gone)

The decoding modules no longer patch `sys.path`, so `src` and `ieeg` must be
importable the normal way — i.e. **`pip install -e .` from the repo root is now
a prerequisite, not a convenience**. The instructions are in the repo-root
[`README.md`](../README.md) ("Python environment setup"); run it once per
environment, not once per session.

The other analysis files (preproc, spec, dcc_scripts) still carry the old
`sys.path.append("C:/Users/jz421/...")` line; sweeping those is the natural
next cheap win now that `pip install -e .` provides `ieeg` on the path.

---

### 1. The goal, stated precisely

The problem with the big files is **not** that they do too much work — it's
that they mix unrelated concerns behind **no interface**. To change any one
thing you must load all of them.

We are borrowing one idea from John Ousterhout's *A Philosophy of Software
Design*: prefer **deep modules** — a *small, obvious interface* hiding a *large
implementation*. A file you import three well-named functions from, and never
otherwise open, is deep. A file where you must read all 4,700 lines to find the
three you need is **wide**, and wide is the thing we are killing.

#### The trap to avoid

"Make it deeper" does **not** mean "add layers." A `manager` that calls a
`handler` that calls a `service` is *more* shallow modules stacked up — now you
chase one feature through six files. That is worse than the monolith.

> **The rule:** split by **concern**, not by adding call-tree depth.
> After a split, adding a feature should mean: open *one* file whose name tells
> you it's the right one, import 2–3 named helpers, and never scroll past code
> unrelated to your task.

"Deeper" here = **narrower files behind clear names**, not longer call chains.

---

### 2. Priority order (highest leverage first)

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

### 3. The safe mechanical recipe (use this for every split)

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

### 4. Worked plan: `decoding/decoding.py`

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

### 5. Worked plan: `power/power_traces.py`

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

### 6. Worked plan: `utils/general_utils.py`

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

### 7. Two cheap wins to fold in

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

### 8. Definition of done (per split)

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

---

## Consolidation candidates

*Consolidation candidates — duplicated and near-duplicated code*

A survey of code that exists in more than one place in this repo. Nothing here
has been changed; this is a decision list.

Each entry gives **what is duplicated**, **how identical it is** (measured by
comparing normalized ASTs, so formatting and comments don't count), **why it
probably happened**, and **what merging it would cost**. They are ordered by
*value ÷ risk*: tier 1 is close to free, tier 4 is a real project.

The measurement pass covered every `.py` file in `src/`, `dcc_scripts/`,
`aaron_code/`, `docs/` and the repo root. Notebooks were not compared to each
other (see §5).

---

### Tier 1 — Safe deletions and one-line imports

These are exact or near-exact copies with an obvious single owner. Each is a
delete-and-import, no behavior change.

#### 1.1 `makeRawBehavioralData.py` is byte-identical in two places

| | |
|---|---|
| **Files** | `makeRawBehavioralData.py` (repo root) and `src/analysis/preproc/makeRawBehavioralData.py` |
| **Identical?** | `diff` reports **zero differences**. All five functions (`load_dataframes`, `combine_dataframes`, `format_subject_ids`, `format_subject_id`, `save_accuracy_arrays`, `main`) match exactly |
| **Why** | The root copy predates the `src/` package; the package copy was added without removing it |
| **Fix** | Delete the root copy. The README's "Post-BIDS Preprocessing" step 3 references a *notebook* (`makeRawBehavioralData.ipynb`) that no longer exists, so no documented workflow points at the root file |
| **Risk** | None found — nothing in the repo imports either by module path |

#### 1.2 `_json_safe` is defined three times, identically

| | |
|---|---|
| **Files** | `dcc_scripts/stats/power_traces_conjunction_dcc.py`, `stability_flexibility_anova_conjunction_dcc.py`, `stability_flexibility_segregation_dcc.py` |
| **Identical?** | Yes — 10 lines, identical AST in all three |
| **Fix** | Move to `src/analysis/utils/general_utils.py` (or a small `dcc_scripts/_common.py`) and import. The other three stats cores (`anatomy`, `timing`, `brain_behavior`) also each define a `_json_safe`, slightly diverged — fold them into the same one |
| **Risk** | Very low. Serialization helper with no state |

#### 1.3 `_is_epochs_like` is defined twice, identically

| | |
|---|---|
| **Files** | `src/analysis/decoding/anova_electrode_selection.py`, `src/analysis/decoding/trial_splitting.py` |
| **Identical?** | Yes — 3 lines |
| **Why** | `anova_electrode_selection.py` was written as "`trial_splitting.py` but with a different selector", and copied the helper along with the split logic |
| **Fix** | Keep the one in `trial_splitting.py` (the older module) and import it |
| **Risk** | None |

#### 1.4 `find_clusters` — the same nested helper in a decoding plot and a power plot

| | |
|---|---|
| **Files** | `src/analysis/decoding/plots/accuracies.py:411`, `src/analysis/power/plots.py:354` |
| **Identical?** | Yes — 16 lines, identical AST. Both are *nested* inside their respective plotting functions, which is why it wasn't obvious |
| **Note** | `accuracies.py` **also** has a module-level `_cluster_spans` and `plots.py` a `_find_cluster_spans` that do the same job again — so contiguous-cluster-span extraction exists in ~4 forms across the two plotting modules |
| **Fix** | One `contiguous_spans(mask) -> list[(start, end)]` in a shared plotting util, used by all four sites |
| **Risk** | Low, but do check each caller's boundary convention (inclusive vs exclusive end index) before merging — that's the one place a silent off-by-one could hide |

#### 1.5 `read_trial_outlier_counts`

| | |
|---|---|
| **Files** | `plot_epoched_data.py` (root), `src/analysis/utils/general_utils.py` |
| **Identical?** | 98% — 5 lines, differ only in a default argument |
| **Fix** | Root script imports from `general_utils` |
| **Risk** | None (root script is legacy anyway — see §4.1) |

---

### Tier 2 — Same file, two copies: `src` vs `dcc_scripts`

The pattern here is "I needed this on the cluster, so I copied it and changed
the paths." Each is a real fork that has since drifted, so merging means picking
which drift was intentional.

#### 2.1 `wavelet_functions.py` vs `wavelet_functions_dcc.py` — the biggest one

| | |
|---|---|
| **Files** | `src/analysis/spec/wavelet_functions.py` (962 lines), `dcc_scripts/spec/wavelet_functions_dcc.py` (638 lines) |
| **Overlap** | **7 functions are byte-identical**: `get_wavelet_baseline` (39L), `load_tfrs` (40L), `plot_mask_pages` (105L), `load_and_get_sig_wavelet_differences` (53L), `load_and_get_sig_wavelet_ratio_differences` (21L), plus `make_and_get_sig_wavelet_differences` (99% match) and `load_wavelets` (93% match). That is **~360 lines duplicated verbatim** |
| **Divergence** | `get_uncorrected_wavelets` is 83% similar (the DCC version drops some kwargs). The `src` version has the multitaper functions the DCC version lacks; the DCC version has `get_trials_for_wavelets` and `get_sig_wavelet_differences` the `src` version doesn't. `load_wavelets` differs in its signature (`layout` vs `bids_root`) |
| **Why** | Classic path-fork. But note: **`wavelet_functions_dcc.py` already imports from `src`** — so the `sys.path` reason for forking is gone |
| **Fix** | Make `wavelet_functions_dcc.py` a thin facade over `src/analysis/spec/wavelet_functions.py`, exactly like `decoding.py` and `power_traces.py` already are. Port the two DCC-only functions into `src` first; reconcile the `layout`/`bids_root` signature by accepting either |
| **Risk** | **Medium.** This is the one with real drift. Worth doing, but read both `get_uncorrected_wavelets` implementations side by side before choosing — if the DCC version's dropped kwargs were a deliberate fix, keeping the `src` version silently changes cluster output |
| **Payoff** | ~360 lines gone, and TFR fixes stop needing to be applied twice |

#### 2.2 `plot_clean.py` vs `plot_clean_dcc.py`

| | |
|---|---|
| **Files** | `src/analysis/preproc/plot_clean.py` (214L), `dcc_scripts/preproc/plot_clean_dcc.py` (217L) |
| **Overlap** | `fix_events_file` (27L) **identical**; `main` **94% identical** |
| **Real differences** | Only four, and all four are configuration, not logic: (a) `LAB_root` — `get_default_LAB_root()` vs a hardcoded `/cwork/$USER`; (b) the `src` version drops a list of EEG channels, the DCC version has that commented out; (c) the DCC version calls `channel_outlier_marker(raw, 3, 2, save=True)`, the `src` version has it commented out with a note that "`get_good_data()` reruns it anyway"; (d) the `src` version still has a `sys.path.append("C:/Users/jz421/...")` line |
| **Fix** | One module, with `--lab-root` and `--drop-eeg-channels` / `--mark-channel-outliers` flags. `plot_clean_dcc.py` becomes a 5-line wrapper that sets the cluster defaults |
| **Risk** | **Low-medium.** Difference (c) is a genuine behavioral divergence — the two copies currently do *different preprocessing*. Merging forces a decision about which is correct, which is itself worth surfacing |

#### 2.3 Block-effect diagnostics exist twice

| | |
|---|---|
| **Files** | `src/analysis/power/block_diagnostics.py` (253L, library), `dcc_scripts/power/diagnose_block_effects.py` (263L, script) |
| **Overlap** | Not textual duplication — the DCC script *does* import the library. But both docstrings describe the same analysis in near-identical prose, and the script re-implements window/label handling the library already has |
| **Fix** | Low priority. Mostly worth a look to confirm the script isn't re-deriving `block_labels_from_metadata` |
| **Risk** | Low |

---

### Tier 3 — Within-package duplication

#### 3.1 `src/analysis/pac/` — five files sharing six helpers

This subpackage has no shared-utilities module, so each script carries its own
copy of the same helpers.

| Helper | Copies | Identical? |
|---|---|---|
| `make_windows` | 4 (`env_correlation`, `theta_connect`, `env_plot`, `plot_timeline`) | Two variants: the correlation pair matches at 91%, the plotting pair at 83%, and the two variants take **different arguments** (`start,end,win_len` vs `time_start,time_end,window_width,time_step`) |
| `read_sig_pairs_for_subjects` | 2 (`env_plot`, `plot_timeline`) | **Identical**, 26L |
| `extract_clusters` | 2 (`env_plot`, `plot_timeline`) | **Identical**, 21L |
| `find_roi_names` | 2 (`env_correlation`, `theta_connect`) | 87% — and **both embed their own copy of `rois_dict`**, which already lives in `src/analysis/config/rois.py` |
| `load_epochs` | 2 (`env_correlation`, `theta_connect`) | 97% |
| `_bh_fdr` | 2 (`env_correlation`, `theta_connect`) | 93% |
| `build_paired_matrices` | 2 (`env_plot`, `plot_timeline`) | 20% — same name, genuinely different code. **Rename one**; a shared name that means two things is worse than a duplicate |
| `plot_pair_result` | 2 (`env_plot`, `plot_timeline`) | 60% |
| `sanitize_filename` | 3 (`env_correlation`, `env_plot`, and `preproc/make_epoched_data_saved.py`) | ~75–98%, all 2 lines |

| | |
|---|---|
| **Fix** | Add `src/analysis/pac/_common.py` with the six shared helpers; have `find_roi_names` import `rois_dict` from `src.analysis.config.rois` instead of embedding it. Rename the diverged `build_paired_matrices` pair |
| **Risk** | **Low mechanically, medium socially.** This code was written by a different contributor and does not currently import from the rest of `src/`. Worth checking with them before restructuring, and worth doing in one commit so their in-flight work rebases cleanly |
| **Payoff** | ~120 lines, and the embedded `rois_dict` copies stop being able to drift from the real one — that one is a correctness risk, not just tidiness |

#### 3.2 `make_epoched_data.py` × 3

| | |
|---|---|
| **Files** | `make_epoched_data.py` (498L), `make_epoched_data_saved.py` (317L), `make_epoched_data_with_phase.py` (319L) |
| **Status** | **Partly done already.** `epoch_helpers.py` was created to hold the three helpers all three shared. But their `main()`s are still 95% identical (`saved` vs `with_phase`), and the bodies still overlap heavily in load → clean → epoch → rescale |
| **Real differences** | `saved` writes epochs to disk and applies a bipolar re-reference; `with_phase` returns amplitude *and* phase; the base version computes stats and significant electrodes |
| **Fix** | One script with `--save-epochs`, `--return-phase`, `--bipolar` flags, sharing one pipeline body. Or, less invasively, extract the common load-clean-epoch-rescale block into `epoch_helpers.py` and leave three thin scripts |
| **Risk** | **Medium-high.** This is the code every downstream result depends on. If you do it, do it with a regression check: run the current and merged versions on one subject and assert the epochs arrays match bit for bit |

#### 3.3 `dcc_scripts/*/run_*_dcc.py` — 15 entry points, ~3250 lines of the same shape

| | |
|---|---|
| **Pattern** | Every entry point reads defaults from environment variables: `os.environ.get` appears **36×** in `run_decoding_dcc.py`, 40× in `run_stability_flexibility_cross_decoding_dcc.py`, 26× in `run_power_traces_conjunction_dcc.py`, and 13–25× in five more. Each hand-writes `int(os.environ.get('X', '12'))` / `float(...)` / a bespoke bool parse |
| **Evidence of the itch** | `run_decoding_dcc.py` already defines a private `_env_bool()` — but only for itself |
| **Similarity** | `run_analysis()` bodies pair up at 74–78% (e.g. `run_make_wavelets_dcc` vs `run_plot_wavelets_dcc`; the two stats entry points) |
| **Fix** | A `dcc_scripts/_env.py` with `env_str / env_int / env_float / env_bool / env_list(name, default)`. Each entry point keeps its own knob list (that's the point of the layer) but stops re-implementing parsing and coercion |
| **Risk** | **Low, and it fixes a real bug class.** Hand-rolled bool parsing is where `FLAG=false` silently becomes `True`. Worth doing for correctness, not just line count |
| **Payoff** | Perhaps 200–300 lines, plus consistent behavior on malformed environment values |

#### 3.4 `src/task/practiceGlobal.m` and `practiceLocal.m` are 98% the same file

| | |
|---|---|
| **Files** | `src/task/practiceGlobal.m` (529L), `src/task/practiceLocal.m` (528L) |
| **Identical?** | 98% by character. `diff` reports **27 changed lines out of ~529** |
| **What actually differs** | The function name; `createTaskArr(nTrials, 'g')` vs `'l'`; the output filename (`GL_Global_Practice_Data_#…` vs `GL_Local_…`); one `TextSize` (24 vs 32); and a swap of the two response-key legend colors/labels (red/"Big" vs blue/"Small"). **Nothing else.** The entire trial loop, timing, saving, pause handling and accuracy logic is byte-identical |
| **Fix** | One `practiceSingleTask(taskType, …)` taking `'g'`/`'l'` and a small style struct. `practiceGlobal`/`practiceLocal` become two-line wrappers, so `Master_Script.m` doesn't change |
| **Risk** | **Low mechanically, but this is participant-facing timing code.** Psychtoolbox timing is the whole point of the task, and a merge must not add a branch inside the trial loop. Test on a real display with a real participant run before it touches data collection |
| **Also** | `practiceGlobalLocal.m` is 93–95% similar to both, and `mainTask.m` is 66–73% similar to all three — the same trial loop, four times. Merging *those* is a bigger job than the pair above, and lower value: start with the 98% pair and see how it feels |

#### 3.5 `save_results` / `make_plots` / `write_summary` across the six stats cores

| | |
|---|---|
| **Files** | The six `dcc_scripts/stats/*_dcc.py` cores |
| **Similarity** | 15–51% pairwise — **too low to merge the functions**, but the *sequence* is identical in all six: build results dict → `_json_safe` → dump JSON → write CSVs → make figures → write a text summary |
| **Fix** | Don't merge the bodies. Extract only the shared scaffolding: `write_json(obj, path)`, `write_summary_header(meta)`, the save-directory convention. Leave each analysis's actual content alone |
| **Risk** | Low if scoped to scaffolding; high if someone tries to unify the bodies. Recommend the narrow version only |

---

### Tier 4 — Dead, superseded, or deliberately duplicated (mostly: leave alone)

#### 4.1 `aaron_code/` — a whole vendored decoding pipeline

`aaron_code/` contains a second `Decoder` class (twice, in fact:
`aaron_decoding_init.py` and `aaron_plot_decoding_ieeg_example.py`), a second
`GroupData`, and duplicates of `flatten_features` (**identical** to
`src/analysis/decoding/data_prep.py`'s), `fit_predict` (**identical** to
`decoder.py`'s), `sample_fold` (32% — diverged), `windower` (19% — diverged) and
`classes_from_labels`.

**Recommendation: leave it, but label it.** Nothing in `src/` or `dcc_scripts/`
imports it — it's reference material, and the value of reference material is
that it is frozen. The README now says so explicitly. The alternative, if the
directory has outlived its usefulness, is deleting it wholesale rather than
merging it — partial merging would give you the worst of both.

#### 4.2 `docs/skeletons/a1…a6_*.py` vs the implemented modules

`docs/skeletons/a1_anova_labels.py` contains `per_electrode_anova_labels` and
`_anova_interaction_stats`, which now also exist (implemented) in
`src/analysis/stats/stability_flexibility_segregation.py`. Same for a2–a6.

**This duplication is the point** — they are assignment stubs whose docstrings
name the drop-in target. Leave them. The only maintenance question is whether a
finished skeleton should be marked "implemented, see `<module>`" so nobody
implements it twice.

#### 4.3 Legacy root scripts and notebooks

`roi_analysis.ipynb`, `whole_brain_analysis.ipynb`, `plot_HG_and_stats.ipynb`,
`plot_clean.ipynb`, `plot_epoched_data.py` overlap heavily with the maintained
`src/analysis/` code, and `src/analysis/power/roi_analysis.py` is an
explicitly-labelled "ongoing refactoring of roi_analysis.ipynb" that still
carries a hardcoded `C:/Users/jz421/...` path.

**Recommendation: don't merge, decide.** Either finish the port and delete the
notebook, or move the notebook to a `legacy/` directory so nobody mistakes it
for current. The in-between state — a half-finished refactor next to the
original — is what makes the tree confusing. `src/analysis/power/roi_analysis.py`
is either the successor to `roi_analysis.ipynb` or it isn't; right now it's
neither.

#### 4.4 `src/analysis/config/group_data.py`

A 32-line `GroupData` class whose own docstring says: *"In progress, dunno if
this will ever be used tbh."* Nothing imports it. `aaron_code/aaron_grouping.py`
has a 228-line `GroupData` that is 1% similar.

**Recommendation: delete**, or move it next to the notebooks as an explicit
sketch. It currently reads as API.

#### 4.5 `make_subjects_electrodes_to_ROIs_dict` in two places

`src/analysis/pac/get_channels_detail.py` (54L) and
`src/analysis/utils/general_utils.py` (86L) share a name but are only 12%
similar — the PAC one builds a simpler dict for its own use.

**Recommendation:** rename the PAC one (e.g. `make_pac_channel_roi_map`) rather
than merging. Same-name-different-behavior is the more dangerous problem here,
and it's a one-line fix.

---

### 5. What this survey did *not* cover

- **Notebook-to-notebook duplication.** 58 notebooks totalling ~50 MB were not
  compared against each other. Judging by filenames, `make_wavelets.ipynb` /
  `make_wavelets_dcc.ipynb`, `plot_wavelets*.ipynb`, `wavelet_differences*.ipynb`,
  `power_traces*.ipynb` and `roi_analysis.ipynb` (root vs `src/analysis/power/`)
  are near-certain pairs. Worth a pass with `nbdime` if you want that number.
- **Copy-pasted blocks inside a single file.** The survey compared whole
  functions across files, so a 40-line block pasted three times inside
  `general_utils.py` (2416L) or `windowed_anova.py` (1391L) would not show up.
  Given that `general_utils.py` is 2416 lines and has no internal structure
  beyond function order, this is the most likely place for undiscovered
  duplication.
- **MATLAB was compared only at whole-file level** (see §3.4 for what that
  found). Function-level comparison inside `src/task/*.m` was not done.

---

### Suggested order, if you want one

1. **Tier 1 in a single commit** (§1.1–1.5) — pure deletions and imports, no
   behavior change, ~60 lines and five confusions gone.
2. **§3.3, the env-var helper** — small, and it fixes a real bug class rather
   than just tidying.
3. **§4.4 delete `group_data.py`, §4.5 rename the PAC function** — two minutes,
   removes two misleading names.
4. **§2.2 `plot_clean`** — because merging it forces the question of which
   preprocessing is actually correct, and that question should be answered
   whether or not you merge.
5. **§2.1 `wavelet_functions`** — the biggest single win (~360 lines), but read
   both versions first.
6. **§3.1 the PAC helpers** — coordinate with whoever owns that code.
7. **§3.4 `practiceGlobal`/`practiceLocal`** — a clean 500-line win, but it is
   participant-facing timing code, so schedule it between data-collection
   sessions, not during one.
8. **§3.2 `make_epoched_data` × 3** — highest risk, do last, with a
   bit-for-bit regression check.

§4 items are decisions, not refactors, and can happen at any time.
