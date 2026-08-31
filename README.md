Global Local Task

Contact: Jim Zhang and Raphael Geddert

jim.zhang@duke.edu, raphael.geddert@duke.edu

---

## What this repository is

Two things live here, and they are almost independent of each other:

1. **The experiment** — a MATLAB/Psychtoolbox global-local task (`src/task/`) that
   runs on participants in the clinic and writes behavioral data.
2. **The analysis pipeline** — a Python package (`src/analysis/`) plus cluster
   launchers (`dcc_scripts/`) that take BIDS-converted intracranial EEG (iEEG)
   from that task and produce power traces, decoding accuracies, time-frequency
   spectra, connectivity, brain figures, and the statistical battery behind the
   manuscript.

Everything downstream of preprocessing starts from **epoched iEEG data**. Each
analysis path consumes epochs and produces a different kind of result. If you
only remember one thing about the layout, remember this:

> **`src/analysis/` is the library. `dcc_scripts/` is how you launch it on the
> cluster. `tests/` is how you check it. `docs/` is why it is designed that way.**

### Where to look first

| If you want… | Read |
|---|---|
| **What each file does** | Part 1 below — the codebase map |
| **How to run an analysis** (motivation, method, knobs, outputs) | [`docs/analysis_guide.md`](docs/analysis_guide.md) — the single guide to every analysis path, including the stability-vs-flexibility A1–A7 battery |
| **What shape the data is at each step** of A1–A7 | [`docs/stability_flexibility_data_flow.md`](docs/stability_flexibility_data_flow.md) |
| **What each output file means** | [`docs/stability_flexibility_outputs_guide.md`](docs/stability_flexibility_outputs_guide.md) |
| **How to break up a file that got too big** | [`docs/refactoring_guide.md`](docs/refactoring_guide.md) |
| **Manuscript Methods text** | [`docs/stability_flexibility_segregation_methods.md`](docs/stability_flexibility_segregation_methods.md) |
| **The figure sequence for the paper** | [`docs/figure_plan.md`](docs/figure_plan.md) |
| **Duplicated code that could be consolidated** | [`docs/consolidation_candidates.md`](docs/consolidation_candidates.md) |
| **Environment setup, BIDS conversion, VPN, cluster access, running the experiment** | Parts 2–6 of this README |

---

# Part 1 — Codebase map

## 1.1 Top level

| Path | What it is |
|---|---|
| `src/` | The importable `src` package: the experiment (`src/task/`) and the analysis library (`src/analysis/`) |
| `dcc_scripts/` | Duke Compute Cluster (DCC) entry points — the scripts you actually launch |
| `tests/` | pytest suite mirroring `src/analysis/`'s layout, plus exploratory testing notebooks |
| `docs/` | Design docs, the analysis guide, assignment skeletons |
| `aaron_code/` | Vendored reference code from Aaron's decoding pipeline. **Not imported by anything** — read-only inspiration |
| `tutorials/` | Three MNE/`ieeg` tutorial notebooks, unrelated to this project's analyses |
| `IEEG_Pipelines/` | Git submodule for the lab's `ieeg` package. Now redundant — `setup.py` installs `ieeg` from PyPI |
| `instruction_images/` | PNGs shown to participants during task instructions (used by `src/task/instructions.m`) |
| `setup.py` | Makes `src` importable and declares runtime dependencies. `pip install -e .` is the whole setup |
| `pytest.ini`, `Makefile` | Test discovery/markers/coverage config, and `make test` / `test-fast` / `test-cov` / `test-parallel` shortcuts |
| `environment.yml`, `environment_fresh.yml`, `environment_send_erin.yml` | Conda environment snapshots. `environment.yml` is minimal; the other two are full frozen exports for reproducing a specific machine |
| `eventsTemplate.json` | BIDS events sidecar template |
| `combinedData.csv` | Pooled behavioral data across subjects — the input to the behavioral models |
| `*.ipynb` at root | **Legacy analysis notebooks** (see §1.13) |
| `README.docx` | Older Word copy of this README. Superseded by this file |

## 1.2 The four-layer pattern (read this before `dcc_scripts/`)

Nearly every cluster analysis is split across four files with predictable names.
Once you know the pattern, you can navigate any analysis path:

| Layer | Name pattern | Job |
|---|---|---|
| **Library** | `src/analysis/<area>/<thing>.py` | The actual statistics/plotting. No paths, no argparse, importable and testable |
| **Core** | `dcc_scripts/<area>/<thing>_dcc.py` | `main(args)` — loads data, calls the library, saves results/figures/summary |
| **Entry point** | `dcc_scripts/<area>/run_<thing>_dcc.py` | Builds the `args` namespace (defaults + environment-variable overrides), then calls the core's `main()` |
| **Job script** | `sbatch_<thing>.sh` / `submit_<thing>.sh` | `sbatch_*` is the single SLURM job; `submit_*` is the loop that submits many `sbatch_*` jobs (one per condition, subject, ROI…) |

So to change **what** is computed you edit the library; to change **how it is
run** you edit `run_*_dcc.py` or the `submit_*.sh` above it.

## 1.3 `src/task/` — the experiment (MATLAB / Psychtoolbox)

The only non-Python code in the repo. Run `Master_Script.m` to run a participant.

| File | Lines | What it does |
|---|---:|---|
| `Master_Script.m` | 658 | Entry point. Prompts for subject ID and practice mode, sets up per-subject data folders, holds all timing/trial-count parameters (stim interval, fixation interval, blocks, trials per block, practice accuracy cutoff), handles counterbalancing and resume-from-partial-session |
| `instructions.m` | 422 | Instruction screens, using the PNGs in `instruction_images/` |
| `practiceGlobal.m` | 529 | Practice block — global task only |
| `practiceLocal.m` | 528 | Practice block — local task only |
| `practiceGlobalLocal.m` | 588 | Practice block — both tasks combined, cued by frame color |
| `mainTask.m` | 716 | The 4-block main experiment |

Operational details (pausing, saving, block types, what the experimenter should
watch for) are in Part 6.

## 1.4 `src/analysis/config/` — what the conditions *are*

Pure data + lookup helpers. No computation. This is the layer you edit to add a
new condition or comparison, and everything else reads from it.

| File | Lines | What it does |
|---|---:|---|
| `experiment_conditions.py` | 1229 | The condition dictionaries. Each named condition maps to its list of `BIDS_events` strings plus its factor levels (`congruency`, `switchType`, `incongruentProportion`, `switchProportion`, `task`). Grouped into condition *sets* (e.g. `stimulus_experiment_conditions`, `stimulus_conditions`, `stimulus_task_by_congruency_conditions`) |
| `condition_registry.py` | 1104 | **Single source of truth** tying each condition set to how it should be used: which comparisons to build, pooled-shuffle settings, context-comparison kwargs, trace labels, display names, balance strata, subtraction pairs, ANOVA factors and interactions. Accessed through `get_comparisons()`, `get_anova_factors()`, `get_display_name()`, etc. Adding a condition means one entry here |
| `plotting_parameters.py` | 274 | Per-condition color, line style and human-readable label, used by every power-trace and decoding figure |
| `rois.py` | 39 | `rois_dict`: coarse bilateral ROIs (`dlpfc`, `lpfc`, `acc`, `parietal`, `v1`, `occ`) as lists of Destrieux parcel names, deliberately without the `ctx_lh_`/`ctx_rh_` prefix so one definition selects both hemispheres. `select_rois()` parses the `--rois` CLI value and fails loudly on typos |
| `group_data.py` | 32 | An abandoned attempt at a `GroupData` class to replace the `subjects_mne_objects` dict. Its own docstring says it may never be used — treat as dead code |

## 1.5 `src/analysis/preproc/` — raw BIDS → epochs

Turns BIDS recordings into the epoched, baseline-rescaled high-gamma data that
every downstream analysis consumes.

| File | Lines | What it does |
|---|---:|---|
| `plot_clean.py` | 214 | **Step 1 of everything.** Line-noise (60 Hz + harmonics) filtering per subject, writes the cleaned raw. Also `fix_events_file()`, which repairs BIDS events files. Run as `python plot_clean.py --subjects D0057 D0059` |
| `make_epoched_data.py` | 498 | **The main preprocessing script.** Bandpass/high-gamma extraction, epoching around Stimulus/Response, baseline rescaling, two-pass outlier handling (`outliers_to_nan` on raw voltage, then optional absolute `max_abs_z` rejection on rescaled power), and a permutation cluster t-test that identifies task-significant electrodes |
| `make_epoched_data_saved.py` | 317 | Variant that **saves** epochs to disk rather than only computing stats, and applies a bipolar re-reference first |
| `make_epoched_data_with_phase.py` | 319 | Variant that returns **amplitude *and* phase** (the input to the PAC/connectivity path) |
| `epoch_helpers.py` | 140 | The three helpers the three `make_epoched_data*` scripts share: `shuffle_array`, `extract_amplitude_and_phase_and_freqs`, `trial_ieeg_rand_offset` |
| `add_experimentStart_to_run1.py` | 242 | One-off fixer: inserts a missing `experimentStart` event into the first run's BIDS events file for named subjects |
| `save_bipolar_derivatives.py` | 224 | Builds bipolar-referenced derivatives: groups contacts by shank, pairs adjacent contacts, computes A−B, and derives bipolar coordinates as contact midpoints |
| `parcellation.py` | 468 | Assigns each electrode to an anatomical parcel from the FreeSurfer recon within a search radius |
| `makeRawBehavioralData.py` | 61 | Combines per-subject behavioral CSVs into one table and writes per-subject accuracy `.npy` arrays (consumed by `general_utils.load_acc_arrays`) |
| `debug_make_epoched_data.ipynb` | 526 | Scratch notebook for debugging the above |

## 1.6 `src/analysis/utils/` — the shared plumbing

Imported by nearly everything. `general_utils.py` is the single biggest file in
the repo and is effectively the project's standard library.

| File | Lines | What it does |
|---|---:|---|
| `general_utils.py` | 2416 | ~60 functions in loose groups: **path/environment resolution** (`get_default_LAB_root`, `resolve_lab_root`, `get_recon_subj_dir`); **electrode↔ROI maps** (`make_or_load_subjects_electrodes_to_ROIs_dict`, `filter_electrodes_by_roi`, `make_sig_electrodes_per_subject_and_roi_dict`); **loading epochs** (`load_mne_objects`, `create_subjects_mne_objects_dict` — the function that builds the `subjects_mne_objects` nested dict everything else passes around); **significant-channel bookkeeping** (`save_sig_chans`, `load_sig_chans`, `get_sig_chans_per_subject`); **permutation tests** (`permutation_test`, within-/across-electrode variants); **outlier handling** (`handle_outliers`, `identify_bad_channels_by_trial_nan_rate`, `impute_trial_nans_by_channel_mean`); **ANOVA + plotting odds and ends** (`perform_modular_anova`, `plot_significance`, `make_plotting_parameters`); and `windower`, the sliding-window helper the decoder uses |
| `labeled_array_utils.py` | 970 | Converts `subjects_mne_objects` into `LabeledArray`s keyed by condition — the format the decoder consumes. Handles per-condition trial subsampling, NaN-trial removal per channel, cross-subject concatenation, and the **bootstrapped** ROI arrays (`make_bootstrapped_roi_labeled_arrays_*`) that the decoding battery resamples over |
| `epoch_metadata_utils.py` | 237 | Parses MNE event-name strings (`Stimulus/i25.0/Taskg/…`) into a proper metadata DataFrame, and adds previous-trial columns |
| `anova_label_selection.py` | 260 | Reads an A1 `anova_labels.csv` back into an electrode selection. Deliberately MNE-free so power, decoding and cross-decoding all select electrodes identically |
| `electrode_exclusion.py` | 76 | Drops named electrodes from a `{roi: {subject: [channels]}}` selection. Exists because one electrode with a large excursion visibly shifts a ~170-electrode ROI mean and no amount of condition balancing removes it |
| `outlier_analysis.ipynb` | 2845 | Exploratory notebook on outlier thresholds |

## 1.7 `src/analysis/power/` — high-gamma power traces and the windowed ANOVA

The main "what does the signal do over time" path.

| File | Lines | What it does |
|---|---:|---|
| `power_traces.py` | 81 | **Facade only.** Was a ~2,400-line monolith; now re-exports the public names from the modules below so old imports keep working |
| `evoked_builders.py` | 413 | Data assembly: per-electrode and multi-channel evokeds, ROI grand averages, condition subtraction, and the cluster-permutation comparison between two evokeds |
| `windowed_anova.py` | 1391 | The statistical core. Builds the long-form windowed dataframe, fits per-window OLS/ANOVA, and runs **two** permutation cluster-correction pipelines — *within-electrode* (each electrode gets its own null; this is what defines "significant electrodes") and *across-electrode* — with sign-aware cluster splitting and FDR correction. `load_significant_electrodes()` reads a finished run back |
| `plots.py` | 1052 | Power-trace figures: per-ROI traces with SD/SEM/CI shading and significance bars, the 2-way interaction plot, the 16-condition mega-plot, shared style/color helpers, and the adapter that reshapes ANOVA cluster results for the interaction plots |
| `block_diagnostics.py` | 253 | Cross-tabulates per-electrode power against **block type**. In a blocked design condition and block are confounded, so a bad recording in one block arrives pre-labelled as a condition effect — this finds those |
| `roi_analysis.py` | 91 | Partially-refactored script version of `roi_analysis.ipynb`. Still carries a hardcoded Windows path |
| `power_traces.ipynb`, `roi_analysis.ipynb` | 1647 / 20763 | Interactive versions of the above |

## 1.8 `src/analysis/decoding/` — classification

The largest subpackage. Already split out of a 4,752-line monolith (see
`docs/refactoring_guide.md`).

**Core pipeline**

| File | Lines | What it does |
|---|---:|---|
| `decoding.py` | 125 | **Facade only** — re-exports every public name from the modules below |
| `data_prep.py` | 284 | Condition balancing/downsampling, `mixup2` augmentation, feature flattening, fold sampling |
| `decoder.py` | 581 | The `Decoder` class (PCA → classifier pipeline) and its cross-validated confusion-matrix methods, including the sliding-window and temporal-generalization variants |
| `accuracy_stats.py` | 978 | Statistics **on** accuracies: permutation/bootstrap cluster tests, pooled shuffle distributions, paired comparisons between two accuracy distributions, cluster-length null distributions |
| `process_bootstrap.py` | 242 | One bootstrap sample end to end — resample trials, build labeled arrays, run sliding-window decoding, run the pooled shuffle. Called in parallel via joblib by `decoding_dcc.py` |
| `roi_confusion.py` | 262 | Per-ROI confusion-matrix orchestration for time-window decoding |
| `tfr_cluster.py` | 550 | Decoding restricted to significant time-frequency clusters: computes the TFR masks on training trials only, then flattens the masked data into a decoding matrix |
| `trial_splitting.py` | 354 | **Disjoint trial splitting** — splits trials so electrode selection and decoding never see the same trials, avoiding double-dipping. Selector here is a cheap task-responsiveness t-test |
| `anova_electrode_selection.py` | 740 | Same split logic, but the selector is the power-traces windowed ANOVA — so the decoded electrodes are literally the ones the power figures call significant. Also the electrode-set algebra (unique / overlap / union) and the label/slug formatting for filenames and figure titles |
| `coupling_electrode_selection.py` | 669 | Electrode sets defined by high-gamma envelope **coupling** (from the PAC path's `high_corr_*.csv`). Expands bipolar pairs to contacts, maps them to ROIs, and — because an unmatched coupled-vs-uncoupled comparison would measure set size, not coupling — builds N degree-matched uncoupled control draws |
| `coupling_comparison.py` | 417 | Reads the per-electrode-set `MASTER_RESULTS` pickles back and tests coupled vs. the distribution over matched control draws |
| `cross_decoding.py` | 615 | **A4.** Train on one contrast, score against another, to ask whether "both"-selective electrodes carry one shared code or two orthogonal ones. Includes circularity guards and a synthetic-data generator with planted geometry |
| `context_comparison.py` | 366 | Top-level context comparisons (e.g. the same contrast decoded in a 25% vs 75% block) and the cross-block overlay plot |

**Thin orchestration wrappers** (called by `decoding_dcc.py`, one concern each):
`run_anova_electrode_selection.py` (261), `run_coupling_electrode_selection.py` (269),
`run_context_comparisons.py` (45), `run_aggregate_and_plot_time_averaged_cms.py` (87),
`run_debug_cm_traces.py` (145), `run_visualization_debug.py` (134).

**Plots** (`decoding/plots/`)

| File | Lines | What it does |
|---|---:|---|
| `accuracies.py` | 939 | Nature-style accuracy time courses, single- and multi-significance-bar variants, multipanel figures |
| `confusion.py` | 377 | Confusion matrices, TFR-mask pages, pooled cm-trace extraction and plotting |
| `trajectories.py` | 673 | PCA/UMAP static projections, PCA-over-time, 3-D trajectories, decision-boundary slices |
| `replot.py` | 639 | Regenerates any decoding figure from its saved `MASTER_RESULTS_*.pkl` — new labels/colors/layout with **no re-decoding** |
| `style.py` | 59 | The shared `NATURE_STYLE` constant |

## 1.9 `src/analysis/spec/` — time-frequency (wavelet / multitaper)

| File | Lines | What it does |
|---|---:|---|
| `wavelet_functions.py` | 962 | Computes uncorrected and baseline-corrected wavelet and multitaper TFRs, saves/loads `-tfr.h5` files, and runs cluster-permutation tests for TFR differences between two conditions. Also `plot_mask_pages()` for paged per-channel mask figures |
| `subjects_tfr_objects_functions.py` | 522 | The multi-subject layer: builds (or loads from cache) a TFR object per subject per condition, then computes significant TFR differences per subject or aggregated per ROI |
| `make_wavelets.ipynb`, `plot_wavelets.ipynb`, `wavelet_differences.ipynb`, `multitaper_spec.ipynb` | — | Interactive counterparts of the cluster scripts in `dcc_scripts/spec/` |

## 1.10 `src/analysis/stats/` — the stability-vs-flexibility battery (A1–A7) and behavior

This is the statistical heart of the manuscript. Each module answers one
question; `docs/analysis_guide.md` Part III is the full treatment.

| File | Lines | Battery step | What it does |
|---|---:|---|---|
| `stability_flexibility_segregation.py` | 1218 | A1/A2 | Do distinct subpopulations support stability (congruency/LWPC) vs flexibility (switch/LWPS), or the same ones? Two complementary subject-aware tests: a **continuous** partial correlation between per-electrode stability and flexibility sensitivity (split-half, to remove shared-gain bias), and a **categorical** per-electrode labeling + Cochran–Mantel–Haenszel conjunction with a permutation null and a threshold sweep. `per_electrode_anova_labels()` is the parametric A1 electrode definition |
| `power_traces_conjunction.py` | 837 | A1/A2 alt | The same conjunction, but with the electrode definition **swapped**: S/F flags are read back from a finished within-electrode windowed-ANOVA cluster-correction run instead of a fresh window-mean ANOVA. Also the continuous confound control (responsiveness) and the both-vs-distinct test |
| `stability_flexibility_anatomy.py` | 828 | A3 | Are the subpopulations in different *places*? ROI histograms, brain plots per group, and a chi-square enrichment test **conditioned on iEEG coverage** (which is clinically determined and the main confound) |
| `stability_flexibility_timing.py` | 491 | A5 | Does stability information arise **earlier** than flexibility information? Per-bin interaction time courses, 50%-of-peak onsets, peak latencies, and the Ulrich–Miller jackknifed onset-difference test |
| `stability_flexibility_brain_behavior.py` | 353 | A6 | Are the substrates *functional*? Across-subject correlation of neural selectivity against the behavioral LWPC/LWPS RT effect, plus the better-powered within-subject single-trial model, each with a cross-pairing specificity control |
| `erin_linear_mixed_effects_model.py` | 50 | behavior | Mixed-effects model of post-error RT on previous-error type × congruency × switch type. Carries a hardcoded absolute path to `combinedData.csv` |
| `erin_stats.ipynb`, `*_tutorial.ipynb`, `*_sandbox.ipynb` | — | | Interactive and teaching versions. The `_tutorial` notebooks walk through A3/A5/A6 step by step |
| `run_erin_stats.sh` | 16 | | Runs the behavioral stats notebook headlessly |

A4 (cross-decoding) lives in `decoding/cross_decoding.py`; A7 is a self-check
assignment in `docs/learning_assignments/`.

## 1.11 `src/analysis/pac/` — envelope coupling and connectivity

Written largely by a different contributor; it uses its own file conventions and
does not go through `dcc_scripts/`. Its `sbatch_*.sh` / `submit_*.sh` scripts sit
alongside the Python.

| File | Lines | What it does |
|---|---:|---|
| `env_correlation.py` | 367 | Per-subject, per-window Pearson correlation between high-gamma **envelopes** of every within-ROI channel pair, with optional orthogonalization and BH-FDR. Writes the `high_corr_<pairtype>_<condition>_<subject>.csv` files that `decoding/coupling_electrode_selection.py` consumes |
| `theta_connect.py` | 431 | The same shape of analysis for **theta coherence** rather than envelope correlation, with a permutation test across trials |
| `sig_test.py` | 445 | Loads the per-pair summaries back, computes per-pair and overall means, and runs permutation tests between conditions or time windows |
| `env_plot.py` | 523 | Plots envelope-correlation results: paired matrices, cluster extraction, per-pair figures |
| `plot_timeline.py` | 723 | Timeline plots of the same results across windows |
| `theta_connect_plot.py` | 53 | Small plotter for the coherence CSVs |
| `get_channels_detail.py` | 133 | Builds this path's own subject→electrode→ROI dictionary |
| `coupling_MI.ipynb` | 294 | Modulation-index (true phase-amplitude coupling) exploration |

Note: this directory also has ~200 committed `.png` result figures and several
result `.csv`s (see §1.15).

## 1.12 `src/analysis/vis/` — brains and diagnostics

| File | Lines | What it does |
|---|---:|---|
| `jim_mri.py` | 899 | The electrode-rendering library: CT/MRI alignment and overlays, `plot_on_average` (electrodes on fsaverage), per-subject significant/non-significant electrode plots, white-matter exclusion, channel grouping and labeling |
| `brain_figure_glasser_separate_svgs_lateral_medial_view_less_bold.py` | 409 | Publication brain figure using the Glasser HCP-MMP1 atlas — highlights dlPFC, ACC, pre-SMA, dmPFC, vmPFC, precuneus, basal ganglia; saves lateral and medial views as separate SVGs |
| `power_traces_anova_f_traces_vis.py` | 177 | Plots the per-window ANOVA **F-traces** saved by `power_traces_dcc.py`, per ROI and per electrode |
| `trial_z_distribution_vis.py` | 438 | Per-trial z-score diagnostics for baseline-rescaled high gamma — the tool for *choosing* the `max_abs_z` outlier threshold from the data instead of guessing. Trial traces over the mean, z distributions, and survival curves per threshold |
| `plot_subjects.ipynb`, `plot_subjects_TUTORIAL.ipynb`, `make_legend.ipynb`, `*_plotting.ipynb` | — | Interactive brain/figure notebooks; the cleaned-up submittable version is `dcc_scripts/vis/plot_sig_electrodes_dcc.py` |

## 1.13 `dcc_scripts/` — cluster entry points

Organized by analysis area, each following the four-layer pattern from §1.2.

| Subdir | Contents |
|---|---|
| `preproc/` | `plot_clean_dcc.py` — the DCC copy of line-noise filtering; `submit_make_epoched_data.sh` / `sbatch_make_epoched_data.sh` fan `make_epoched_data.py` out over subjects |
| `power/` | `power_traces_dcc.py` (479) + `run_power_traces_dcc.py` (340) — the power-trace and windowed-ANOVA job. Plus two diagnostics: `diagnose_block_effects.py` (block-confound cross-tab straight from epochs) and `diagnose_electrode_deviations.py` (attributes a trace excursion to specific electrodes by reading the saved evoked `.npz`s — no re-run needed) |
| `decoding/` | `decoding_dcc.py` (616) + `run_decoding_dcc.py` (556) — the main decoding battery, including the optional electrode-definition split, ANOVA-defined electrode sets, and coupling-defined electrode sets. `stability_flexibility_cross_decoding_dcc.py` (960) is A4. `james_sun_cluster_decoding_dcc.py` is the TFR-cluster decoding job. `report_coupling_counts.py` counts coupled/eligible electrodes **before** you spend cluster time. `run_coupling_comparison_dcc.py` re-runs the coupled-vs-control comparison over saved results |
| `spec/` | `make_wavelets_dcc.py`, `plot_wavelets_dcc.py`, `wavelet_differences_dcc.py`, `get_sig_tfr_differences_dcc.py`, and `wavelet_functions_dcc.py` (a DCC copy of the library — see the consolidation doc) |
| `stats/` | One core + entry point per battery step: `stability_flexibility_anova_conjunction_dcc.py` (A1/A2), `power_traces_conjunction_dcc.py` (A1/A2 via ANOVA-defined electrodes), `stability_flexibility_anatomy_dcc.py` (A3), `stability_flexibility_timing_dcc.py` (A5), `stability_flexibility_brain_behavior_dcc.py` (A6), `stability_flexibility_segregation_dcc.py`. Each assembles the long-form table from epochs, runs the library, and writes results + plots + a human-readable summary |
| `vis/` | `plot_sig_electrodes_dcc.py` (533) — significant electrodes per condition on the fsaverage brain plus ROI histograms; `condition_plot_specs.py` (296) is the registry of *what to compare*, the one file you edit to define a new brain-plot comparison |

## 1.14 `tests/` — the pytest suite

Mirrors `src/analysis/`'s layout. `pytest.ini` sets discovery, the
`slow`/`integration`/`unit` markers, and coverage on `src`; `tests/conftest.py`
provides synthetic MNE fixtures (`create_epochs`, `minimal_subjects_data`,
`simple_electrodes_mapping`) so most tests need no real data.

| Area | Coverage |
|---|---|
| `decoding/` | Decoder and pipeline (1361), figures (643), coupling electrode selection (838), ANOVA electrode selection + integration, cross-decoding incl. a dedicated circularity test, trial splitting |
| `power/` | Windowed-ANOVA cluster p-values and cluster statistic, block-balanced ANOVA conditions, block diagnostics, evoked builders, electrode overlays |
| `stats/` | Power-traces conjunction (607), anatomy, ANOVA labels, timing, brain-behavior, CMH uninformative strata |
| `utils/`, `config/`, `vis/` | Labeled-array utils, general utils, ANOVA label selection, electrode exclusion, ROI definitions, proportion conventions, plotting parameters, condition plot specs, trial-z visualization |
| `preproc/` | Mostly **exploratory notebooks**, not tests: baseline testing, drift testing, bandpass filtering, epochs metadata, permutation-test simulation. Only `test_time_perm_cluster.py` and `baseline_testing.py` are runnable |

Run with `make test` (all), `make test-fast` (skip `slow`), `make test-cov`
(coverage report to `htmlcov/`), or `make test-parallel`.

## 1.15 `docs/`

| File | What it is |
|---|---|
| `analysis_guide.md` (2608) | **The** analysis reference — motivation, method and scripts for every path |
| `stability_flexibility_data_flow.md` (688) | One fake dataset followed through A1–A7 with every intermediate table printed. Generated by `docs/examples/stability_flexibility_data_flow_demo.py`, which runs anywhere with no cluster or iEEG data |
| `stability_flexibility_outputs_guide.md` | What each output PNG/CSV/JSON/TXT means |
| `stability_flexibility_segregation_methods.md` | Manuscript-ready Methods text |
| `refactoring_guide.md` (301) | How the big modules were split, and how to split the next one |
| `figure_plan.md` (253) | The main-text figure sequence and the claim stack behind it |
| `nested_electrode_selection.md` (349) | Design plan for making the diagonal (select on congruency → measure congruency) non-circular |
| `consolidation_candidates.md` | Duplicated/near-duplicated code and what merging each would cost |
| `skeletons/a1…a6_*.py` | Runnable assignment stubs for each battery step, with the drop-in target named at the top |
| `learning_assignments/segregation_bootstrap/` | A7 — a build-a-feature self-check with a pytest grader |

## 1.16 Root-level notebooks and scripts (legacy)

These predate the `src/` package. They still run, but the maintained code is in
`src/analysis/`; most carry hardcoded `C:/Users/jz421/...` paths.

| File | Lines | What it does | Maintained equivalent |
|---|---:|---|---|
| `plot_clean.ipynb` | 686 | Line-noise filtering, interactively | `src/analysis/preproc/plot_clean.py` |
| `plot_rawvsclean.ipynb` | 404 | Sanity-check raw vs. line-filtered traces | — |
| `plot_HG_and_stats.ipynb` | 1555 | High-gamma extraction + permutation cluster stats; the original source of the significant-electrode step | `src/analysis/preproc/make_epoched_data.py` |
| `plot_epoched_data.py` | 929 | Diagnostic plots on epoched data: NaN matrices, per-channel outlier counts, channel grids with windowed or cluster-corrected significance | partly `src/analysis/vis/` |
| `roi_analysis.ipynb` | 5861 | Condition plots and stats for chosen ROIs | `src/analysis/power/` + `dcc_scripts/power/` |
| `whole_brain_analysis.ipynb` | 8400 | The same, for all electrodes | as above |
| `rsa.ipynb` | 32699 | Representational similarity analysis, hand-rolled | — |
| `rsa_using_toolbox.ipynb` | 1784 | RSA via `rsatoolbox`, plus power-trace plotting | — |
| `post_error_slowing_analysis.py` | 430 | Behavioral post-error slowing: difference-score and full-factorial mixed models (statsmodels and pymer4 versions), descriptives, plots | — |
| `makeRawBehavioralData.py` | 61 | **Byte-identical** to `src/analysis/preproc/makeRawBehavioralData.py` | that one |
| `send_erin_this_to_count_error_trials.ipynb` | 978 | One-off error-trial counts shared with a collaborator | — |

## 1.17 Data flow at a glance

```
Natus recording ──(MATLAB: Ecog_preprocessing, makeTrials_GL)──► Trials.mat + EDF
        │
        └──(BIDS_coding repo: BIDS_convert_*.sh)──► BIDS-1.1_GlobalLocal/
                    │
                    ├─ preproc/plot_clean.py ............ line-noise filtering
                    ├─ preproc/parcellation.py .......... electrode → anatomical parcel
                    └─ preproc/make_epoched_data*.py .... epochs + high gamma + baseline
                                    │                     rescale + outliers + sig electrodes
                                    ▼
                        epoched data (derivatives/)
                                    │
      ┌─────────────┬───────────────┼────────────────┬──────────────┐
      ▼             ▼               ▼                ▼              ▼
   power/        decoding/        spec/            pac/          stats/
 power traces   accuracies      wavelet /       envelope       A1–A6 battery
 windowed       cross-decoding   multitaper     correlation    (uses power/ or
 ANOVA →        TFR-cluster      TFR diffs      coherence       its own ANOVA
 significant    coupling sets                   → coupled       for electrode
 electrodes ─────────┘                            pairs ──────────► definition)
      │                                                             │
      └──────────────────────► vis/ (brain figures, ROI histograms) ◄┘
```

## 1.18 Repository hygiene — known issues

Worth knowing before you clone or push:

- **`.gitignore` is not being honored by history.** It lists `*.png`, `*.csv`,
  `*.npz`, `*.out`, `*.err`, `*.json`, `*.pkl` — but the repo currently tracks
  **1584 `.npz`**, **494 `.csv`**, **588 SLURM `.out`/`.err`** logs, **210
  `.png`** figures and 19 `.pyc` files. They were committed before the ignore
  rules landed (ignoring only prevents *new* files being added). Removing them
  with `git rm --cached` would shrink working copies but not history.
- **Large notebooks with outputs committed.** `dcc_scripts/spec/make_wavelets_dcc.ipynb`
  is 26 MB; `src/analysis/spec/make_wavelets.ipynb` 6.2 MB; `rsa.ipynb` 2.8 MB.
  Clearing outputs before commit (or `nbstripout`) would help a lot.
- **`IEEG_Pipelines/` submodule is redundant** — `setup.py` installs `ieeg` from
  PyPI, and the `sys.path.append(".../IEEG_Pipelines/")` hack it existed for has
  been removed from the maintained modules (though it survives in the legacy
  root notebooks and a few `src` files).
- **Duplicated code** — see [`docs/consolidation_candidates.md`](docs/consolidation_candidates.md).

---

# Part 2 — Python environment setup

The analysis code lives in the importable `src` package. Set it up once per
environment (per machine / per conda env) from the repo root:

```bash
pip install -e .
```

This installs the project **and its dependencies** — including `ieeg` (from
PyPI), `mne`, `umap-learn`, etc. — as declared in `setup.py`. After this,
`from src.analysis.decoding.decoding import Decoder` (and the rest) works from
anywhere, with no `sys.path` hacks. `-e` is an *editable* install, so your edits
to `src/...` take effect immediately without reinstalling.

> If you keep a separate conda env for `ieeg`, activate it first; most
> dependencies are already present there and pip will only add what's missing.

---

# Part 3 — Getting data into the pipeline

### Initial Preprocessing (getting the EDF, aligning triggers with events)
1. Nicole wrote docs on this in Box/CoganLab/CRS Resources/Preprocessing. Focus on Global Local Preprocessing and BIDS Guide
2. Data is in Box/CoganLab/ECoG_TaskData. TaskUploadDir is where the edfs are (made by cropping Natus recording) using the start and stop times of the experiment. Cogan_Task_Data is where the behavioral data is, sorted by subject
3. Copy the data (both behavioral and EDF) into Box/CoganLab/D_Data/GlobalLocal. Behavioral data needs to be copied and renamed using the D###_behavioralData format into Box/CoganLab/D_Data/GlobalLocal/rawDataCopies. EDF needs to be copied into Box/CoganLab/D_Data/GlobalLocal/EDFs
4. Ecog_preprocessing.m script. For each subject, need to note their neural_chan_index, trigger_chan_index, and mic_chan_index. This can be gotten by running edfread_fast and grabbing the labels: X = edfread_fast(edf_filename), Labels = x.label
5. Exclude the EEG channels as well as the channels that start with C, EKG, Event, TRIG, OSAT, PR, Pleth

### BIDS Coding (makes BIDS files after initial preprocessing)
1. Run makeTrials_GL.m (/Users/jinjiang-macair/Library/CloudStorage/Box-Box/CoganLab/D_Data/GlobalLocal/makeTrials_GL.m) with the subject id (D##) and date (YYMMDD) to create a Trials.mat file for that subject. You must do this using the makeTrials_GL.m in the D_Data/GlobalLocal folder to include all 29 fields! Otherwise, only 19 fields will be included. To run this, you need to add makeTrials_GL.m to path as well as MATLAB-env folder (/Users/jinjiang-macair/Documents/MATLAB/MATLAB-env). If MATLAB-env isn't there, you can clone it from https://github.com/coganlab/MATLAB-env. To add makeTrials_Gl.m to path, you can click Run and then select add to path. Then, rerun from command line (e.g., makeTrials_GL('D144', '260401').
2. Run BIDS_convert_wsl.sh (within BIDS_coding repository, global local branch). Steps 3-5 go into detail on how to do this. 11/7/25 UPDATE: THIS IS BROKEN USE THE MAC VERSION (CoganLab/BIDS_coding). Also figure out how to push this version to the BIDS_coding remote repository, as it has a fix for df.eval(). And before running this version, need to make relevant Box files available offline (ECoG_Recon_Full and D_Data for the chosen subject). And then make them unavailable again for space afterwards.
3. To install dependencies, need to ```conda create env environment.yml``` on Mac if not already created, and give it an environment name. Or do ```conda env create -f environment.yml``` from the envs folder if on Windows.
4. Need to ```conda activate BIDS_coding``` or whatever you named the conda environment. 
5. Now cd into the BIDS_converter subfolder within BIDS_coding repository (open WSL, cd /git/BIDS_coding/BIDS_converter), and do ```./BIDS_convert_wsl.sh``` after modifying BIDS_convert_wsl.sh with your chosen SUB_IDS (line 18). Or, BIDS_convert_mac or whichever script fits your OS. NOTE: To open the WSL script, do ```explorer.exe .``` to open the file explorer in that location.
6. Copy the BIDS folder into Box (run it locally because it's faster)

---

# Part 4 — Infrastructure access

### Duke Health VPN
1. Get a Duke Health Enterprise Account (send e-mail to dibs-it@duke.edu asking for them to set this up)
2. Set up multi-factor authentication: https://idms-mfa.oit.duke.edu/mfa/help
3. Follow the instructions to set up a Duke VPN: https://oit.duke.edu/service/vpn/
4. You can test your VPN access and download FortiNAC and necessary antivirus: portal.duke.edu, https://duke.service-now.com/kb_view.do?sysparm_article=KB0034098
5. Open Cisco AnyConnect, and connect to the Duke Health VPN using this address: vpn.duhs.duke.edu
6. Enter your netid and netid password, and type 'push' as the Second Password to send a push notification to your MFA app for authentication.
7. Wait a minute or so for the VPN connection to let you through the firewall.

### Microsoft Remote Desktop
1. Download Microsoft Remote Desktop on your local machine
2. On the remote machine that you want to connect to, get your full PC device name by going to Settings -> System -> About -> Full device name (i.e., NEU-7BTXKH2.dhe.duke.edu)
3. On your local machine, follow these instructions to set up the Duke OIT RDS Gateway on Microsoft Remote Desktop: https://oit.duke.edu/help/articles/kb0032645/. NOTE: On Mac, you need to **check** the 'Bypass for local addresses' option, not uncheck it as the instructions say.
4. On your local machine, open Microsoft Remote Desktop, click Add PC, and put in your full device name as the PC name. Put in your netid and netid password as your User account.
5. On your local machine, connect to Duke Health VPN. Wait a bit for it to let you through the firewall.
6. Now try connecting to your remote machine through Microsoft Remote Desktop by double-clicking the icon for it.
   
### Windows FSL
1. Need to open xquartz on windows before running fsl in the ubuntu app. https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows
2. When running XLaunch, it is critical to deselect Native OpenGL and select Disable access control in the Extra Settings panel. https://superuser.com/questions/1372854/do-i-launch-the-app-xlaunch-for-every-login-to-use-gui-in-ubuntu-wsl-in-windows
3. Need to run the line, export DISPLAY=:0 in Ubuntu first before running fsl command for gui to work.
4. Also need to mount the Z: drive on ubuntu every time we open it. Run this command every time: ```sudo mount –t drvfs Z: /mnt/Egner```
5. Now to get to this folder, do ```cd /mnt/Egner```. In the FSL gui, it should also be ```/mnt/Egner```
6. To make the inputs to paste, run the makeInputsForFSL.ipynb script that's in the GlobalLocal folder right now, changing the subjects range.
7. Then, open fsl feat in ubuntu and do emacs fslSecondLevelInputs.txt, and highlight all and do edit -> copy. Then can paste this as input into the fsl feat input window. Also change the number of cope images in the GUI.
8. To make the EV matrix, run the next cell in makeInputsForFSL.ipynb

### Duke Compute Cluster
1. Download the Remote - SSH Extension on VS Code: https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-ssh. Then, set up a remote host from VS Code to dcc-login.oit.duke.edu. Click the >< button on the bottom left and then choose "connect to host", entering dcc-login.oit.duke.edu. It'll ask for a password (enter your net id password) and then 2-step authentication. NOTE: To avoid issues with having to log in multiple times when connecting to remote host from VS code, do ```ssh -Y netid@dcc-login.oit.duke.edu``` from a terminal and then after logging in, open your shell's startup file by doing ```nano ~/.bashrc```, then add ```[[ $- != *i* ]] && return``` as the first line of your shell's startup file, and save/close it by pressing Ctrl+X.
2. ALTERNATIVELY! Set up an ssh key so that you don't have to manually log in: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/login/#ssh-keys - note that you need to ssh into dcc from terminal first before connecting to host on vscode. You need to do this before using jianghao's script to open an interactive session.
   
3. You can do ```ssh -Y netid@dcc-login.oit.duke.edu``` from a terminal to access the DCC.
   
4. To be able to access data from scripts on the DCC using ```LAB_root = os.path.join("cwork", "your_net_id")``` -> Move data from /hpc/home/your_net_id/coganlab/Data/BIDS-1.1_GlobalLocal/ to /cwork/your_net_id/BIDS-1.1_GlobalLocal/ using the Duke Compute Cluster (DCC) Data Transfer Node as the collection on Globus (https://app.globus.org/file-manager?destination_id=1ad66c7c-4f60-11e8-900c-0a6d4e044368&destination_path=%2Fcwork%2Fjz421%2FBIDS-1.1_GlobalLocal%2F&origin_id=1ad66c7c-4f60-11e8-900c-0a6d4e044368&origin_path=%2Fhpc%2Fhome%2Fjz421%2Fcoganlab%2FData%2FBIDS-1.1_GlobalLocal%2F&two_pane=true).<img width="1273" alt="Screenshot 2025-06-06 at 1 07 21 PM" src="https://github.com/user-attachments/assets/dd0204bd-3536-4fb2-9333-ba289f02ec4a" />

5. Use git repositories on the DCC to sync code with local computer code. Use the dcc_scripts folder for scripts that will live on the DCC. Make sure to set up an SSH key for permissions, and pull before making any changes (refer to step 4 here: https://github.com/dward2/BME547/blob/main/Assignments/01_tool_setup_git_intro.md). First, git clone this repository to your netid folder under coganlab on the DCC. Then, do git config --global user.email "your_email@email.com" and do git config --global user.name "your_github_username". 

6. To move files from DCC to a local machine or Box using Terminal, on windows, can run something like: ```scp jz421@dcc-login.oit.duke.edu:/cwork/jz421/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/subjects_tfr_objects/*.png C:Users/jz421/Desktop/tfr_figures/``` but replace the paths with where you've saved the figures on the dcc and where you want to save them to. For mac, do ```scp "jz421@dcc-login.oit.duke.edu:/cwork/jz421/BIDS-1.1_GlobalLocal/BIDS/derivatives/spec/multitaper/subjects_tfr_objects/*.png" ~/Desktop/tfr_figures/```

7. To run an interactive session on the dcc:
   1. Set up ssh host script in my ssh keys file, after putting it in the sshhost folder in home.  
		a. ls -al ~/.ssh/id_*.pub
		b. Id_ed25519.pub
	2. Open a dcc on demand session (choose an amount of time)
		a. https://dcc-ondemand-01.oit.duke.edu/pun/sys/dashboard/batch_connect/sys/bc_jupyter/session_contexts/new
	3. Use jianghao's sshost script with ./sshhost (put this in my base directory in terminal)
		a. Put in the host name from the dcc on demand session
	4. Open vscode, connect to host, choose dcc-compute

---

# Part 5 — Analysis steps

> Full detail — motivation, method, knobs, outputs — is in
> [`docs/analysis_guide.md`](docs/analysis_guide.md). This is the short checklist.

### Post-BIDS Preprocessing
1. Run first three cells of plot_clean.ipynb to do line-noise filtering (for new subjects, will need to run this twice and exclude the eeg channels from the RuntimeWarning). Or just run src/analysis/preproc/plot_clean.py and pass in the subjects. (i.e., python plot_clean.py --subjects D0057 D0059)
2. Copy Trials.csv from Box/CoganLab/D_Data/GlobalLocal/D### for newly run subjects into Box/CoganLab/D_Data/GlobalLocal/rawDataCopies. Rename as D###_behavioralData.csv.
3. Run makeRawBehavioralData.ipynb to generate accuracy arrays for newly run subjects

### Wavelets
1. Run make_wavelets.ipynb to make wavelet tfr files (mne.TimeFrequency.EpochsTFR), saved to filename = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', subj, f'{output_name}-tfr.h5')
2. Run plot_wavelets.ipynb to make wavelet plots for each electrode
3. Run wavelet_differences.ipynb to make wavelet_difference plots for different conditions.
4. All wavelet functions live in wavelet_functions.py (also, copy_wavelet_spec.ipynb is deprecated, that was a copy of Aaron's old code)
5. Alternatively, use the submit/sbatch script on the cluster.
    
### High Gamma Filter and Permutation Testing
1. Run make_epoched_data.py to do the stats without plotting. Run make_epoched_data.py like this: ```(ieeg) PS C:\Users\jz421\Desktop\GlobalLocal> python make_epoched_data.py --passband 4 8 --subjects D0057```. So the passband needs to pass in the lower and then upper bound, and then subjects needs to just be the subject ids, no list brackets.

### Decoding
1. run ```sh submit_specific_conditions_decoding_dcc.sh``` from the dcc_scripts/decoding folder on the dcc, with chosen conditions in this script and other chosen parameters in run_decoding_dcc.py. Testing code is at bottom of sbatch_decoding_dcc.sh, uncomment to test. Make sure your chosen epochs root file is saved in the dcc cwork folder. Using unit of analysis as repeat right now. The decoding_dcc.py script will load in the epoched data of specified subjects, then for each bootstrap, transform it into a LabeledArray where each electrode is randomly downsampled to the lowest number of trials across electrodes in that roi and condition, then for each condition comparison (i.e., congruent vs incongruent), randomly downsample again to the lowest number of trials across conditions for that condition comparison, then run decoding where error bars and stats are calculated using the unit of analysis (bootstrap, repeat, fold). If bootstrap, it will sum accuracies across folds and average across repeats for each bootstrap. If repeat, it will sum accuracies across folds for each repeat. If fold, it will find the variance and stats over all folds.

### RSA
1. rsa.ipynb uses my math to do RSA. rsa_using_toolbox.ipynb uses the rsatoolbox library (and also does power trace plotting too).
 
### Steps for new subjects
1. Run plot_clean.ipynb to preprocess (line noise filtering) for new subjects
2. Get significant electrodes by running plot_hg_and_stats with Stimulus as the event (should be top cell I think). Also run the bottom cells to plot individual electrodes for raw traces and high gamma filtered in this step.
3. Get high gamma of main effect conditions by running plot_hg_and_stats with events as Stimulus/i25 + Stimulus/i75 (inc), and Stimulus/c25 + Stimulus/c75 (con), and Stimulus/i25/s25 + Stimulus/i25/s75 + Stimulus/i75/s25 + Stimulus/i75/s75 + Stimulus/c25/s25 + Stimulus/c25/s75 + Stimulus/c75/s25 + Stimulus/c75/s75 (switch), and Stimulus/i25/r25 + Stimulus/i25/r75 + Stimulus/i75/r25 + Stimulus/i75/r75 + Stimulus/c25/r25 + Stimulus/c25/r75 + Stimulus/c75/r25 + Stimulus/c75/r75 (repeat)
5. Run roi_analysis.ipynb to get condition plots and stats for rois of interest. Choose rois at top of script. Note that the structure used in plot_subjects currently uses all electrodes from all selected rois, so if just want to plot one roi, just select that one.
6. run whole_brain_analysis.ipynb to get condition plots and stats for all electrodes.
7. Run plot_subjects.ipynb to get brain plots for new subjects
8. Run copy_wavelet_spec.ipynb and then plot_wavelets.ipynb for new subjects with Stimulus and Response as the events

---

# Part 6 — Notes for the experimenter

**Experiment Procedure**

Run Master\_Script.m to run the experiment. Specify a subject ID and whether to include practice or proceed immediately to the main task. The practice version will ask if you want to proceed to the main task or exit the experiment when it finishes.

Unique data folders are created for each new participant (i.e., never seen before ID, **not** case sensitive), wherein are saved data files for all practice tasks (each iteration) and main task. If the participant has been run before (e.g., just did practice earlier) a new folder will be created for that participant, with the date and time appended to the folder name. 

Press **escape** at any point during a task to **pause the task**. From there, press any button to resume the task, or press **escape** again to **exit the experiment**.

Press **escape** at any point during the instructions to **exit the experiment.**

Press **escape** at any point during the inter block feedback screens to **exit the experiment.**

**This experiment saves almost all progress.** Task data is saved immediately after a trial is completed, and if you exit the experiment in the middle of the main tasks, any completed blocks won’t need to be repeated next time. The exceptions are that the entire practice needs to be completed, so if you exit the task in the middle of the practice, the whole practice will need to be completed again. Likewise, if you exit the experiment in the middle of a block during the main experiment, you will need to repeat that block.

The practice consists of 3 practice sections, each 16 trials long. First, participants practice one task (either global or local) and then practice the other. Last, participants practice combining both the local and global tasks together, responding based on the color of the frame surrounding the letters. Participants must get at least pracAccCutoff  (set to 75) % correct to move on to the next section, otherwise they will need to repeat that practice section. If the practice task is exited prematurely, participants will need to recomplete all three practice sections.

The main experiment consists of 4 blocks, in a counterbalanced block order. Try to avoid pausing for too long in the middle of a block if possible. For example, if a participant is only able to complete 2 of the 4 blocks in a session, the script will automatically resume with the 3<sup>rd</sup> block the next time that participant is run (**assuming the subject ID is exactly the same**). If the experiment is exited in the middle of a block, however, the participant will have to repeat that block.

If the participant was already completed all 4 blocks previously, you will be asked to confirm rerunning all 4 blocks of the task. The old data will still be saved (in whatever folders they were saved in) but the file that stores which runs have been run before will be overwritten to allow new runs to commence.

**Stimulus/Experiment Details**

Stimulus Timing/Trial Count Details can be specified in Master\_Script.m, starting at line 71. Defaults are as follows:

Stimulus Interval (stimInterval) = 2.5 seconds.

- How long stimulus is on the screen and participant is able to respond. Increasing this time (to 2, or 2.5s at most) will (probably) make the task easier, though this hasn’t been confirmed. The task will automatically proceed to the next screen (feedback) as soon as the participant responds.

Fixation Cross Interval (fixInterval) = 0.5 seconds

- If the trial is the first of a section (or the first after pausing the task), the fixation interval is 2 seconds longer than fixIinterval so the participant has time to prepare.

Number of Blocks (numBlocks) = 4

- This should never be changed due to study design.

Trials Per Block in Main Task (trialsPerBlock) = 112

- This can be changed by increments of 16 trials. With a stiminterval of 1.5s, fixInterval of 0.5 seconds, trials run at longest for 3.4 seconds, so 112 trials x 4 blocks x 3.4 seconds = 25 minutes time spent performing main task (not including block breaks or instructions). This can be considerably shorter however, since trials proceed automatically when the participant responds. Assuming an average RT of ~1000 ms, experiment run time (not including breaks and instructions) = ~21 minutes.

Practice Accuracy Cutoff (pracAccCutoff) = 75

- Minimum accuracy percent (out of 100) needed to move on from one practice task to the next.
  
blockTypes  
A: 25% congruent (or 75% incongruent), 25% switch  
B: 25% congruent (or 75% incongruent), 75% switch  
C: 75% congruent (or 25% incongruent), 25% switch  
D: 75% congruent (or 25% incongruent), 75% switch
  
BIDS EVENTS ARE SAVED IN TERMS OF INCONGRUENT PROPORTION, NOT CONGRUENT PROPORTION. So Stimulus/c25/s25 will grab the stimulus onsets of trials that are congruent and switch, in a 25% incongruent and 25% switch block. AKA a 75% congruent and 25% switch block.
