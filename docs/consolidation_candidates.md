# Consolidation candidates — duplicated and near-duplicated code

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

## Tier 1 — Safe deletions and one-line imports

These are exact or near-exact copies with an obvious single owner. Each is a
delete-and-import, no behavior change.

### 1.1 `makeRawBehavioralData.py` is byte-identical in two places

| | |
|---|---|
| **Files** | `makeRawBehavioralData.py` (repo root) and `src/analysis/preproc/makeRawBehavioralData.py` |
| **Identical?** | `diff` reports **zero differences**. All five functions (`load_dataframes`, `combine_dataframes`, `format_subject_ids`, `format_subject_id`, `save_accuracy_arrays`, `main`) match exactly |
| **Why** | The root copy predates the `src/` package; the package copy was added without removing it |
| **Fix** | Delete the root copy. The README's "Post-BIDS Preprocessing" step 3 references a *notebook* (`makeRawBehavioralData.ipynb`) that no longer exists, so no documented workflow points at the root file |
| **Risk** | None found — nothing in the repo imports either by module path |

### 1.2 `_json_safe` is defined three times, identically

| | |
|---|---|
| **Files** | `dcc_scripts/stats/power_traces_conjunction_dcc.py`, `stability_flexibility_anova_conjunction_dcc.py`, `stability_flexibility_segregation_dcc.py` |
| **Identical?** | Yes — 10 lines, identical AST in all three |
| **Fix** | Move to `src/analysis/utils/general_utils.py` (or a small `dcc_scripts/_common.py`) and import. The other three stats cores (`anatomy`, `timing`, `brain_behavior`) also each define a `_json_safe`, slightly diverged — fold them into the same one |
| **Risk** | Very low. Serialization helper with no state |

### 1.3 `_is_epochs_like` is defined twice, identically

| | |
|---|---|
| **Files** | `src/analysis/decoding/anova_electrode_selection.py`, `src/analysis/decoding/trial_splitting.py` |
| **Identical?** | Yes — 3 lines |
| **Why** | `anova_electrode_selection.py` was written as "`trial_splitting.py` but with a different selector", and copied the helper along with the split logic |
| **Fix** | Keep the one in `trial_splitting.py` (the older module) and import it |
| **Risk** | None |

### 1.4 `find_clusters` — the same nested helper in a decoding plot and a power plot

| | |
|---|---|
| **Files** | `src/analysis/decoding/plots/accuracies.py:411`, `src/analysis/power/plots.py:354` |
| **Identical?** | Yes — 16 lines, identical AST. Both are *nested* inside their respective plotting functions, which is why it wasn't obvious |
| **Note** | `accuracies.py` **also** has a module-level `_cluster_spans` and `plots.py` a `_find_cluster_spans` that do the same job again — so contiguous-cluster-span extraction exists in ~4 forms across the two plotting modules |
| **Fix** | One `contiguous_spans(mask) -> list[(start, end)]` in a shared plotting util, used by all four sites |
| **Risk** | Low, but do check each caller's boundary convention (inclusive vs exclusive end index) before merging — that's the one place a silent off-by-one could hide |

### 1.5 `read_trial_outlier_counts`

| | |
|---|---|
| **Files** | `plot_epoched_data.py` (root), `src/analysis/utils/general_utils.py` |
| **Identical?** | 98% — 5 lines, differ only in a default argument |
| **Fix** | Root script imports from `general_utils` |
| **Risk** | None (root script is legacy anyway — see §4.1) |

---

## Tier 2 — Same file, two copies: `src` vs `dcc_scripts`

The pattern here is "I needed this on the cluster, so I copied it and changed
the paths." Each is a real fork that has since drifted, so merging means picking
which drift was intentional.

### 2.1 `wavelet_functions.py` vs `wavelet_functions_dcc.py` — the biggest one

| | |
|---|---|
| **Files** | `src/analysis/spec/wavelet_functions.py` (962 lines), `dcc_scripts/spec/wavelet_functions_dcc.py` (638 lines) |
| **Overlap** | **7 functions are byte-identical**: `get_wavelet_baseline` (39L), `load_tfrs` (40L), `plot_mask_pages` (105L), `load_and_get_sig_wavelet_differences` (53L), `load_and_get_sig_wavelet_ratio_differences` (21L), plus `make_and_get_sig_wavelet_differences` (99% match) and `load_wavelets` (93% match). That is **~360 lines duplicated verbatim** |
| **Divergence** | `get_uncorrected_wavelets` is 83% similar (the DCC version drops some kwargs). The `src` version has the multitaper functions the DCC version lacks; the DCC version has `get_trials_for_wavelets` and `get_sig_wavelet_differences` the `src` version doesn't. `load_wavelets` differs in its signature (`layout` vs `bids_root`) |
| **Why** | Classic path-fork. But note: **`wavelet_functions_dcc.py` already imports from `src`** — so the `sys.path` reason for forking is gone |
| **Fix** | Make `wavelet_functions_dcc.py` a thin facade over `src/analysis/spec/wavelet_functions.py`, exactly like `decoding.py` and `power_traces.py` already are. Port the two DCC-only functions into `src` first; reconcile the `layout`/`bids_root` signature by accepting either |
| **Risk** | **Medium.** This is the one with real drift. Worth doing, but read both `get_uncorrected_wavelets` implementations side by side before choosing — if the DCC version's dropped kwargs were a deliberate fix, keeping the `src` version silently changes cluster output |
| **Payoff** | ~360 lines gone, and TFR fixes stop needing to be applied twice |

### 2.2 `plot_clean.py` vs `plot_clean_dcc.py`

| | |
|---|---|
| **Files** | `src/analysis/preproc/plot_clean.py` (214L), `dcc_scripts/preproc/plot_clean_dcc.py` (217L) |
| **Overlap** | `fix_events_file` (27L) **identical**; `main` **94% identical** |
| **Real differences** | Only four, and all four are configuration, not logic: (a) `LAB_root` — `get_default_LAB_root()` vs a hardcoded `/cwork/$USER`; (b) the `src` version drops a list of EEG channels, the DCC version has that commented out; (c) the DCC version calls `channel_outlier_marker(raw, 3, 2, save=True)`, the `src` version has it commented out with a note that "`get_good_data()` reruns it anyway"; (d) the `src` version still has a `sys.path.append("C:/Users/jz421/...")` line |
| **Fix** | One module, with `--lab-root` and `--drop-eeg-channels` / `--mark-channel-outliers` flags. `plot_clean_dcc.py` becomes a 5-line wrapper that sets the cluster defaults |
| **Risk** | **Low-medium.** Difference (c) is a genuine behavioral divergence — the two copies currently do *different preprocessing*. Merging forces a decision about which is correct, which is itself worth surfacing |

### 2.3 Block-effect diagnostics exist twice

| | |
|---|---|
| **Files** | `src/analysis/power/block_diagnostics.py` (253L, library), `dcc_scripts/power/diagnose_block_effects.py` (263L, script) |
| **Overlap** | Not textual duplication — the DCC script *does* import the library. But both docstrings describe the same analysis in near-identical prose, and the script re-implements window/label handling the library already has |
| **Fix** | Low priority. Mostly worth a look to confirm the script isn't re-deriving `block_labels_from_metadata` |
| **Risk** | Low |

---

## Tier 3 — Within-package duplication

### 3.1 `src/analysis/pac/` — five files sharing six helpers

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

### 3.2 `make_epoched_data.py` × 3

| | |
|---|---|
| **Files** | `make_epoched_data.py` (498L), `make_epoched_data_saved.py` (317L), `make_epoched_data_with_phase.py` (319L) |
| **Status** | **Partly done already.** `epoch_helpers.py` was created to hold the three helpers all three shared. But their `main()`s are still 95% identical (`saved` vs `with_phase`), and the bodies still overlap heavily in load → clean → epoch → rescale |
| **Real differences** | `saved` writes epochs to disk and applies a bipolar re-reference; `with_phase` returns amplitude *and* phase; the base version computes stats and significant electrodes |
| **Fix** | One script with `--save-epochs`, `--return-phase`, `--bipolar` flags, sharing one pipeline body. Or, less invasively, extract the common load-clean-epoch-rescale block into `epoch_helpers.py` and leave three thin scripts |
| **Risk** | **Medium-high.** This is the code every downstream result depends on. If you do it, do it with a regression check: run the current and merged versions on one subject and assert the epochs arrays match bit for bit |

### 3.3 `dcc_scripts/*/run_*_dcc.py` — 15 entry points, ~3250 lines of the same shape

| | |
|---|---|
| **Pattern** | Every entry point reads defaults from environment variables: `os.environ.get` appears **36×** in `run_decoding_dcc.py`, 40× in `run_stability_flexibility_cross_decoding_dcc.py`, 26× in `run_power_traces_conjunction_dcc.py`, and 13–25× in five more. Each hand-writes `int(os.environ.get('X', '12'))` / `float(...)` / a bespoke bool parse |
| **Evidence of the itch** | `run_decoding_dcc.py` already defines a private `_env_bool()` — but only for itself |
| **Similarity** | `run_analysis()` bodies pair up at 74–78% (e.g. `run_make_wavelets_dcc` vs `run_plot_wavelets_dcc`; the two stats entry points) |
| **Fix** | A `dcc_scripts/_env.py` with `env_str / env_int / env_float / env_bool / env_list(name, default)`. Each entry point keeps its own knob list (that's the point of the layer) but stops re-implementing parsing and coercion |
| **Risk** | **Low, and it fixes a real bug class.** Hand-rolled bool parsing is where `FLAG=false` silently becomes `True`. Worth doing for correctness, not just line count |
| **Payoff** | Perhaps 200–300 lines, plus consistent behavior on malformed environment values |

### 3.4 `src/task/practiceGlobal.m` and `practiceLocal.m` are 98% the same file

| | |
|---|---|
| **Files** | `src/task/practiceGlobal.m` (529L), `src/task/practiceLocal.m` (528L) |
| **Identical?** | 98% by character. `diff` reports **27 changed lines out of ~529** |
| **What actually differs** | The function name; `createTaskArr(nTrials, 'g')` vs `'l'`; the output filename (`GL_Global_Practice_Data_#…` vs `GL_Local_…`); one `TextSize` (24 vs 32); and a swap of the two response-key legend colors/labels (red/"Big" vs blue/"Small"). **Nothing else.** The entire trial loop, timing, saving, pause handling and accuracy logic is byte-identical |
| **Fix** | One `practiceSingleTask(taskType, …)` taking `'g'`/`'l'` and a small style struct. `practiceGlobal`/`practiceLocal` become two-line wrappers, so `Master_Script.m` doesn't change |
| **Risk** | **Low mechanically, but this is participant-facing timing code.** Psychtoolbox timing is the whole point of the task, and a merge must not add a branch inside the trial loop. Test on a real display with a real participant run before it touches data collection |
| **Also** | `practiceGlobalLocal.m` is 93–95% similar to both, and `mainTask.m` is 66–73% similar to all three — the same trial loop, four times. Merging *those* is a bigger job than the pair above, and lower value: start with the 98% pair and see how it feels |

### 3.5 `save_results` / `make_plots` / `write_summary` across the six stats cores

| | |
|---|---|
| **Files** | The six `dcc_scripts/stats/*_dcc.py` cores |
| **Similarity** | 15–51% pairwise — **too low to merge the functions**, but the *sequence* is identical in all six: build results dict → `_json_safe` → dump JSON → write CSVs → make figures → write a text summary |
| **Fix** | Don't merge the bodies. Extract only the shared scaffolding: `write_json(obj, path)`, `write_summary_header(meta)`, the save-directory convention. Leave each analysis's actual content alone |
| **Risk** | Low if scoped to scaffolding; high if someone tries to unify the bodies. Recommend the narrow version only |

---

## Tier 4 — Dead, superseded, or deliberately duplicated (mostly: leave alone)

### 4.1 `aaron_code/` — a whole vendored decoding pipeline

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

### 4.2 `docs/skeletons/a1…a6_*.py` vs the implemented modules

`docs/skeletons/a1_anova_labels.py` contains `per_electrode_anova_labels` and
`_anova_interaction_stats`, which now also exist (implemented) in
`src/analysis/stats/stability_flexibility_segregation.py`. Same for a2–a6.

**This duplication is the point** — they are assignment stubs whose docstrings
name the drop-in target. Leave them. The only maintenance question is whether a
finished skeleton should be marked "implemented, see `<module>`" so nobody
implements it twice.

### 4.3 Legacy root scripts and notebooks

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

### 4.4 `src/analysis/config/group_data.py`

A 32-line `GroupData` class whose own docstring says: *"In progress, dunno if
this will ever be used tbh."* Nothing imports it. `aaron_code/aaron_grouping.py`
has a 228-line `GroupData` that is 1% similar.

**Recommendation: delete**, or move it next to the notebooks as an explicit
sketch. It currently reads as API.

### 4.5 `make_subjects_electrodes_to_ROIs_dict` in two places

`src/analysis/pac/get_channels_detail.py` (54L) and
`src/analysis/utils/general_utils.py` (86L) share a name but are only 12%
similar — the PAC one builds a simpler dict for its own use.

**Recommendation:** rename the PAC one (e.g. `make_pac_channel_roi_map`) rather
than merging. Same-name-different-behavior is the more dangerous problem here,
and it's a one-line fix.

---

## 5. What this survey did *not* cover

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

## Suggested order, if you want one

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
