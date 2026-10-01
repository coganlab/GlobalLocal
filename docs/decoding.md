# Decoding and cross-decoding: how to run them and how to read them

The run-and-read guides for every decoding job, in six self-contained parts.

| Part | What it covers | Was |
|---|---|---|
| [Decoding job](#decoding-job) | The ordinary time-resolved decoding job: choosing the electrodes, what is decoded per condition set, every output, and how to read the true-vs-shuffle and block-comparison figures | `decoding.md` |
| [A4 cross-decoding](#a4-cross-decoding) | Train on congruency, test on switch type (and the reverse): the `anova` / `csv` / `power_traces` / `none` electrode definitions, every knob, the outputs, and the task-transfer positive controls | `a4_cross_decoding.md` |
| [N3b block transfer](#n3b-block-transfer) | Train in one kind of block, test in another: the design choices, what was built, and how to run it | `n3b_block_transfer.md` |
| [Cross-decoding controls](#cross-decoding-controls) | What to run, in what order, when a transfer comes back uninformative, and what each outcome lets you say | `cross_decoding_controls.md` |
| [RT matching](#rt-matching) | The standalone util that subsamples trials to equal RT distributions, its random-subset control, and how to plug it into any analysis | new |
| [Overall-activity control](#overall-activity-control) | Is a decode or transfer carried by a uniform rise in activity or by the pattern across electrodes: `remove_mean` / `mean_only`, how to read them, and their limits | new |

A4 and N3b are two modes of the same job
(`dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py`). The
ordinary job shares the decoder with them, but not the statistics.

Each part keeps its own section numbers. A bare § inside a part refers to that
part's own sections.

Why the jobs are built the way they are: [`analysis_guide.md`](analysis_guide.md)
§7 (the decoder), §17 (A4) and §21 (the circularity controls). Where the results
sit in the paper: [`analysis_plans.md`](analysis_plans.md).

---

## Decoding job

*Decoding — the time-resolved decoding job: how to run it and how to read it*

**What this document is.** A standalone walkthrough of the ordinary decoding job
(`submit_specific_conditions_decoding_dcc.sh` → `run_decoding_dcc.py` →
`decoding_dcc.py`): what it decodes, the ways to choose the electrodes it decodes
from (no table, a saved A1 table, an ANOVA on held-out trials, and the other
launchers), every parameter that changes the answer, the exact commands, every
file it writes, and how to read them.

It is the run-and-read companion to [`analysis_guide.md`](analysis_guide.md) §7
(the module layout and the `Decoder` class) and §21 (the circularity controls, in
depth). For training on one contrast and testing on another, see
[A4 cross-decoding](#a4-cross-decoding); that job shares the decoder but
not the statistics described here.

If you read only one section, read [§3 Choosing the electrodes](#3-choosing-the-electrodes):
the shipped submit script decodes from a saved ANOVA table by default, and turning
that off needs a file edit.

---

### 0. The short version

```bash
cd dcc_scripts/decoding
export EPOCHS_ROOT_FILE=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20

# 1. decode in each population of a saved A1 table (the shipped default:
#    both / congruency_only / switch_type_only x congruency / switch type = 6 jobs)
bash submit_specific_conditions_decoding_dcc.sh

# 2. one population, other condition sets
ANOVA_LABEL_EFFECT=both \
CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
    bash submit_specific_conditions_decoding_dcc.sh

# 3. no table: every task-significant lPFC electrode.
#    First set ANOVA_LABELS_CSVS=("") in submit_specific_conditions_decoding_dcc.sh (§3.1)
CONDITIONS="stimulus_lwpc_block_balanced_conditions" bash submit_specific_conditions_decoding_dcc.sh

# 4. electrodes defined by the windowed ANOVA on 30% of trials, decoded on the other 70%
bash submit_decoding_with_anova_electrode_selection_dcc.sh
```

The results land in `dcc_scripts/decoding/figs/<EPOCHS_ROOT_FILE>/...` (§6). The
figures to look at are `true_v_shfle_<comparison>` (is the contrast decodable?)
and, for condition sets that have one, the context comparison (is it more
decodable in one block type than the other?).

---

### 1. Where the code lives

| Role | File |
|---|---|
| **Job submitter** (one job per condition set × table × population) | `dcc_scripts/decoding/submit_specific_conditions_decoding_dcc.sh` |
| Other launchers, same job | `submit_decoding_with_anova_electrode_selection_dcc.sh`, `submit_decoding_with_electrode_definition_split_dcc.sh`, `submit_decoding_with_coupling_electrode_sets_dcc.sh`, `submit_loo_decoding_dcc.sh` |
| **Cluster wrapper** (5 cores, 225 GB, 48 h) | `dcc_scripts/decoding/sbatch_decoding_dcc.sh` |
| **The knobs**: Python constants, plus a few environment variables | `dcc_scripts/decoding/run_decoding_dcc.py` |
| **The job**: load → electrodes → bootstraps → stats → figures → pickle | `dcc_scripts/decoding/decoding_dcc.py` (`main`, `run_decoding_for_one_electrode_set`) |
| **What gets decoded** for each condition set | `src/analysis/config/condition_registry.py` (`comparisons`, `pooled_shuffle`, `context_comparison`) |
| One bootstrap: pseudo-trials → decode → accuracies | `src/analysis/decoding/process_bootstrap.py` |
| Sliding-window decode per ROI | `src/analysis/decoding/roi_confusion.py` (`get_confusion_matrices_for_rois_time_window_decoding_jim`) |
| Pooling across bootstraps, the percentile cluster test | `src/analysis/decoding/accuracy_stats.py` (`compute_pooled_bootstrap_statistics`) |
| The two-condition comparison (e.g. LWPC) | `src/analysis/decoding/context_comparison.py` (`run_context_comparison_analysis`) |
| Saved-table selection | `src/analysis/utils/anova_label_selection.py` |
| Re-plotting from the saved pickle | `src/analysis/decoding/plots/replot.py` (`replot_master_results`, `replot_all`) |

#### The call path

```
submit_specific_conditions_decoding_dcc.sh    loops tables x populations x CONDITIONS
 └ sbatch_decoding_dcc.sh
    └ run_decoding_dcc.py                      constants + env vars -> args, SAVE_DIR
       └ decoding_dcc.main(args)
          ├ create_subjects_mne_objects_dict   HG epochs per subject x condition (correct trials)
          ├ electrodes: ROI x (sig | all), filtered against what survived epoching
          ├ optional: saved-table filter (csv), held-out split, ANOVA sets, coupling sets
          └ run_decoding_for_one_electrode_set   (once, or once per electrode set)
             ├ joblib over BOOTSTRAPS: process_bootstrap
             │   ├ pseudo-trials: per electrode, drop NaN trials, subsample to the ROI's
             │   │  minimum trial count per condition
             │   ├ time-averaged confusion matrix per comparison
             │   ├ sliding-window decode per comparison: true + shuffle
             │   └ pooled shuffle null (registry 'pooled_shuffle')
             ├ time-averaged confusion matrices, summed over bootstraps
             ├ compute_pooled_bootstrap_statistics: true vs shuffle, percentile clusters
             ├ true_v_shfle_<comparison> figures, CM-trace figures
             ├ context comparison (registry 'context_comparison'): two traces, paired
             │  cluster tests both ways, difference figure
             └ <timestamp>_MASTER_RESULTS_<params>_<condition>.pkl
```

---

### 2. What the job computes

#### 2.1 What is decoded

`CONDITION_NAME` (one per job; the submit script's `CONDITIONS` list) names an
entry of `condition_registry.py`. That entry says which conditions to load and
what to decode:

- **`comparisons`** — one decode each. A comparison is a list of classes, each a
  list of condition-name substrings; e.g. `i_vs_c_at_inc25` is
  `[['Stimulus_i_MC_MR', 'Stimulus_i_MC_MS'], ['Stimulus_c_MC_MR', 'Stimulus_c_MC_MS']]`.
- **`pooled_shuffle`** — an extra shuffle-only decode that pools the classes over
  the block levels, used as the null in the context figure.
- **`context_comparison`** — which two comparisons to set against each other
  (e.g. congruency in 25%- vs 75%-incongruent blocks), with colours and labels.

The condition sets you will use most:

| `CONDITION_NAME` | Comparisons | Context comparison |
|---|---|---|
| `stimulus_congruency_conditions` | `congruency` (c vs i, all trials) | – |
| `stimulus_switch_type_conditions` | `switchType` (r vs s, all trials) | – |
| `stimulus_lwpc_block_balanced_conditions` | `i_vs_c_at_inc25`, `i_vs_c_at_inc75` | `LWPC_block_balanced` |
| `stimulus_lwps_block_balanced_conditions` | `s_vs_r_at_sw25`, `s_vs_r_at_sw75` | `LWPS_block_balanced` |
| `stimulus_congruency_by_switch_prop_block_balanced_conditions` | `i_vs_c_at_sw25`, `i_vs_c_at_sw75` | `congruency_by_switch_prop_block_balanced` |
| `stimulus_switch_type_by_inc_prop_block_balanced_conditions` | `s_vs_r_at_inc25`, `s_vs_r_at_inc75` | `switch_type_by_inc_prop_block_balanced` |
| `stimulus_lwpc_conditions` / `stimulus_lwps_conditions` | six pairs of `c25`/`c75`/`i25`/`i75` (or `s`/`r`) cells | `LWPC` / `LWPS` |
| `stimulus_task_by_congruency_conditions` / `_by_switch_type_` | task within congruent / incongruent (repeat / switch) trials, and the reverse | `task_by_congruency` / `task_by_switch_type` |
| `stimulus_block_pairwise_conditions` / `stimulus_block_multiclass_conditions` | block A–D pairwise / 4-class | – |

**Use the `_block_balanced` sets for the within-block contrasts.** Each of their
classes is the union of two physical block types, and each block type is
subsampled to the smaller one before they are pooled (`balance_strata=True`), so a
tonic block difference cannot ride into the contrast. The 4-cell sets
(`stimulus_lwpc_conditions` etc.) pool raw BIDS events, so the two classes come
from a 3:1 vs 1:3 mix of block types.

> **Gotcha: `stimulus_experiment_conditions` and `stimulus_main_effect_conditions`
> have no `comparisons`.** They exist for A4 and the ANOVAs. Submitted here, the
> job loads everything and decodes nothing.

#### 2.2 One bootstrap

Each of `BOOTSTRAPS` (5) samples, in parallel:

1. **Pseudo-trials.** For every electrode, trials with NaNs are dropped; the
   minimum remaining trial count per condition across the ROI's electrodes is
   found; and every electrode is independently subsampled to that count. Trial *k*
   of electrode A and trial *k* of electrode B are therefore different physical
   trials, even in the same patient: the decoder sees no trial-level covariance
   between electrodes ([`analysis_guide.md`](analysis_guide.md) §7).
2. **Balance.** For each comparison, sub-conditions within a class are subsampled
   to equal size (`balance_strata`), then the classes are subsampled to equal size
   (`BALANCE_METHOD='subsample'`). Chance is 1 / number of classes: 0.5 for every
   set here except the 4-class block decode.
3. **Decode.** PCA keeping `EXPLAINED_VARIANCE` (0.90) → the classifier
   (`MODEL_CHOICE`, LDA), in windows of `WINDOW_SIZE` (64) samples every
   `STEP_SIZE` (16). `N_SPLITS` (5)-fold CV, `N_REPEATS` (5) times; the confusion
   matrices of the folds in a repeat are summed, giving one accuracy trace per
   repeat.
4. **Shuffle.** The same, with the training labels permuted and the model refit,
   `N_SHUFFLE_PERMS` (50) times: 50 null traces.
5. **Pooled shuffle.** For condition sets with `pooled_shuffle`, a further 50 null
   traces on the classes pooled over block levels.
6. **Time-averaged confusion matrix** for each comparison (no windows).

#### 2.3 Statistics

With `UNIT_OF_ANALYSIS='repeat'`, the pooled samples per comparison × ROI are
`BOOTSTRAPS × N_REPEATS` = 25 true traces and `BOOTSTRAPS × N_SHUFFLE_PERMS` = 250
shuffle traces.

- **True vs shuffle** (`compute_pooled_bootstrap_statistics`): a window is a
  candidate when the **mean** true accuracy exceeds the `PERCENTILE` (95th)
  percentile of the 250 shuffle values at that window. Runs of consecutive
  candidate windows form clusters, and a cluster survives if it is longer than the
  `CLUSTER_PERCENTILE` (95th) percentile of the longest run found when one shuffle
  trace is held out and tested against the rest, `N_CLUSTER_PERMS` (100) times.
  The surviving windows are the `> shuffle` bar on every figure.
- **Condition A vs condition B** (`run_context_comparison_analysis`): the 25 true
  traces of each comparison go into two one-tailed `time_perm_cluster` tests
  (A > B, then B > A), with `STAT_FUNC='ttest'`, `PERMUTATION_TYPE='independent'`,
  `P_THRESH_FOR_TIME_PERM_CLUSTER_STATS=0.025`, `P_CLUSTER=0.025` and
  `N_CLUSTER_PERMS` (100) permutations.

> **What the samples are.** The 25 samples are CV repeats and bootstraps of the
> same trials, not subjects, so every p-value here is a within-dataset reliability
> check. `submit_loo_decoding_dcc.sh` (§5.5) is the check that no single subject
> carries a result.

---

### 3. Choosing the electrodes

Two levels decide what is decoded:

1. **The ROIs** — `ROIS_DICT` in `run_decoding_dcc.py` (lPFC only as shipped).
   Override from the environment with `ROIS=lpfc,occ` (names from
   `src/analysis/config/rois.py`; a misspelling raises). Every ROI is decoded
   separately, in the same job.
2. **The electrodes inside each ROI** — one of the routes below.

| Route | Electrodes decoded | How to select it | Output root |
|---|---|---|---|
| **sig / all** (no table) | the ROI's task-significant electrodes, or all of them | `ANOVA_LABELS_CSVS=("")` in the submit script; `ELECTRODES` in `run_decoding_dcc.py` | `figs/<EPOCHS>/` |
| **csv** (shipped default) | every ROI electrode in one population of a saved A1 table | `ANOVA_LABELS_CSV`, `ANOVA_LABEL_EFFECT` | `figs/<EPOCHS>/anova_label_selections/<table>__effect-<pop>__…/<CONDITION>/` |
| **ANOVA sets on held-out trials** | the power-traces windowed-ANOVA sets (lwpc, lwps, …) from 30% of trials, decoded on the other 70% | `submit_decoding_with_anova_electrode_selection_dcc.sh` | `figs/<EPOCHS>/elecset_<set>/` |
| **responsiveness on held-out trials** | electrodes that respond on half the trials, decoded on the other half | `submit_decoding_with_electrode_definition_split_dcc.sh` | `figs/<EPOCHS>/`, `_defsplit` in the names |
| **coupling sets** | electrodes in a significant gamma-envelope correlation pair, against n-matched non-coupled draws | `submit_decoding_with_coupling_electrode_sets_dcc.sh` | `figs/<EPOCHS>/elecset_coupled/`, `elecset_uncoupled_drawNN/` |

The job refuses ANOVA sets combined with either the responsiveness split or
coupling sets. A saved-table filter is applied before any of the held-out routes,
so an `ANOVA_LABELS_CSV` left in the environment narrows them too (§3.1).

#### 3.1 No table: the task-significant (or all) electrodes

This is "decode from the region", with no selection on the decoded effect.

- **`ELECTRODES`** is a constant in `run_decoding_dcc.py` (`'sig'` as shipped; there
  is no environment variable). `sig` keeps the electrodes whose high gamma beats
  their pre-stimulus baseline, read from
  `sig_chans_<subject>_<EPOCHS_ROOT_FILE>.json`; `all` keeps every ROI electrode.
  That significance test says nothing about congruency or switching, so decoding
  those from `sig` electrodes is not circular.
- **To run it**, the submit script's table list must be empty. Uncomment the line
  after the array in `submit_specific_conditions_decoding_dcc.sh`:

  ```bash
  ANOVA_LABELS_CSVS=("")
  ```

  Then `bash submit_specific_conditions_decoding_dcc.sh` submits one job per
  condition set.

> **Gotcha: the environment cannot switch the table off.** `ANOVA_LABELS_CSV=`
> (empty) is ignored and the listed table is used. And do not keep an
> `ANOVA_LABELS_CSV` exported in your shell: every launcher passes
> `--export=ALL`, so the other launchers (§5) would silently filter to it too.

- **Filenames** carry `sig_elecs` or `all_elecs`.

#### 3.2 csv: a saved A1 table

```bash
ANOVA_LABELS_CSV=$REPO/dcc_scripts/stats/results/$EPOCHS_ROOT_FILE/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none \
ANOVA_LABEL_EFFECT=congruency_only \
    bash submit_specific_conditions_decoding_dcc.sh
```

- **What it does:** loads **every** electrode of the ROI (`ELECTRODES` is ignored),
  then keeps the `(subject, electrode)` pairs the table puts in the chosen
  population. If nothing is left, the job stops with an error rather than decode
  an empty set.
- **`ANOVA_LABELS_CSV`** is the table (`anova_labels.csv`) or its folder, and
  replaces the script's `ANOVA_LABELS_CSVS` list. The list holds one
  condition-mode table as shipped; add lines to it to loop over several.
- **Populations** (`ANOVA_LABEL_EFFECT`, one per job): the table's `S` and `F`
  flags define them.

  | Name | Electrodes | Mode |
  |---|---|---|
  | `congruency` / `lwpc` | S = 1 (includes `both`) | condition / proportion |
  | `switch_type` / `lwps` | F = 1 (includes `both`) | condition / proportion |
  | `congruency_only` / `lwpc_only` | S = 1, F = 0 | condition / proportion |
  | `switch_type_only` / `lwps_only` | F = 1, S = 0 | condition / proportion |
  | `both` | S = 1 and F = 1 | either |
  | `union` | S = 1 or F = 1 | either |

  As shipped, the script submits `both`, `congruency_only` and
  `switch_type_only` for each table and condition set.
- **`ANOVA_LABEL_CORRECTION`** (`flags` default, `none`, `fdr_bh`) and
  **`ANOVA_LABEL_ALPHA`** (0.05): `flags` uses the table's own 0/1 flags; `none`
  re-thresholds its raw p at alpha; `fdr_bh` its q.
- **Output:** `figs/<EPOCHS>/anova_label_selections/<table folder>__effect-<population>__correction-<c>__alpha-<a>__roi-all__<hash>/<CONDITION_NAME>/`.
  The table's own folder name (window, electrodes, mode, correction) is kept in
  the path.

> **Gotcha: population names are not checked against the table's mode here.**
> Both A1 modes store their two effects in the same `S`/`F` columns, so
> `congruency_only` on a proportion-mode table silently selects the LWPC-only
> electrodes, under a folder named `effect-congruency_only`. (The A4 job refuses
> this; this job does not.) Match the names to the `_condition_`/`_proportion_`
> in the table's folder.

> **Gotcha: `ANOVA_LABEL_EFFECTS` (plural) in the environment is ignored.** The
> script assigns that array itself. Use `ANOVA_LABEL_EFFECT` (one name) from the
> environment, or edit the array. Emptying the array
> (`ANOVA_LABEL_EFFECTS=()`) makes the script submit every population of the
> table's mode (read off its folder name).

> **Gotcha: selection and decoding share trials.** The table was fit on the same
> trials this job decodes. Decoding congruency from `congruency_only` electrodes
> is inflated by that selection and is descriptive only; decoding switch type from
> them is not selected on. For numbers with no overlap, use §3.3.

#### 3.3 ANOVA sets on held-out trials

```bash
bash submit_decoding_with_anova_electrode_selection_dcc.sh
FRAC_SELECT=0.3 N_PERM=500 SETS=lwpc_only,lwps_only,overlap \
    bash submit_decoding_with_anova_electrode_selection_dcc.sh
```

One job per decoded condition set (`CONDITIONS`, default `stimulus_lwpc_conditions`
and `stimulus_lwps_conditions`). Each job splits every subject's trials on
`metadata.trial_count`, runs the power-traces windowed ANOVA with permutation
cluster correction on the `FRAC_SELECT` (30%) side for each `SEL_LABELS` set,
builds `lwpc`, `lwps`, `lwpc_only`, `lwps_only`, `overlap`, `union`, and decodes
each requested set on the other 70%, into `elecset_<set>/`. `N_PERM` (per
electrode) is the cost driver. The selection report and the ANOVA runs go to
`figs/<EPOCHS>/electrode_selection/`.

Full description, output tree and caveats: [`analysis_guide.md`](analysis_guide.md) §21.3.

#### 3.4 Responsiveness on held-out trials

```bash
bash submit_decoding_with_electrode_definition_split_dcc.sh
FRAC_DEF=0.6 SEED=1 STRATA=congruency,task_sequence,block_type \
    bash submit_decoding_with_electrode_definition_split_dcc.sh
```

Splits each subject's trials `FRAC_DEF` (0.5) / rest, re-selects responsive
electrodes (window vs baseline t-test, BH across channels at `ALPHA`) on the first
part, and decodes the rest. The candidates are the `ELECTRODES` set, so with the
shipped `sig` the upstream significance list (computed on all trials) still
applies first.

> **Gotcha: pass `STRATA=congruency,task_sequence,block_type`.** The default
> `congruency,switchType,blockType` names two columns the metadata does not have,
> so the split is stratified on congruency alone ([`analysis_guide.md`](analysis_guide.md) §21.2).

#### 3.5 Coupling sets

`submit_decoding_with_coupling_electrode_sets_dcc.sh` decodes the electrodes that
take part in a significant gamma-envelope correlation pair (the PAC path's
`high_corr_*.csv`), against `N_DRAWS` n-matched draws of non-coupled electrodes,
and compares the coupled trace with the distribution over draws. Its header
explains the knobs; run `report_coupling_counts.py` first, and
`run_coupling_comparison_dcc.py` after a run split over several jobs.

---

### 4. Parameters

#### 4.1 From the environment

| Variable | Default | Notes |
|---|---|---|
| `CONDITION_NAME` | – (required) | Set per job by the launchers from `CONDITIONS`. |
| `CONDITIONS` | `stimulus_congruency_conditions stimulus_switch_type_conditions` | Launcher only; space-separated, one job each. |
| `EPOCHS_ROOT_FILE` | the `_ttest_zmax_20` file | Also picks the `sig_chans` file. |
| `ROIS` | unset (`lpfc`) | e.g. `lpfc,occ`. |
| `ANOVA_LABELS_CSV`, `ANOVA_LABEL_EFFECT`, `ANOVA_LABEL_CORRECTION`, `ANOVA_LABEL_ALPHA`, `ANOVA_LABEL_ROI` | the script's table; `flags`; `0.05` | §3.2. |
| `LEAVE_OUT` | unset | Drop one subject (§5.5). |
| `N_JOBS` | `SLURM_CPUS_PER_TASK` (5) | Bootstraps run in parallel. |
| `ELECTRODE_DEFINITION_SPLIT*`, `ANOVA_ELECTRODE_SELECTION`, `ELECTRODE_SELECTION_*`, `COUPLING_*` | off | Set by the launchers of §3.3–3.5. |

#### 4.2 Constants in `run_decoding_dcc.py` (edit the file)

| Constant | Value | What it does |
|---|---|---|
| `SUBJECTS` | 24 subjects | |
| `ACC_TRIALS_ONLY` | `True` | Correct trials only. |
| `ROIS_DICT` | `lpfc` | Overridden by `ROIS`. |
| `ELECTRODES` | `'sig'` | `'sig'` or `'all'` (§3.1). Ignored on the csv route. |
| `MODEL_CHOICE` | `'LDA'` | `'SVC'` for a linear SVM. |
| `EXPLAINED_VARIANCE` | `0.90` | PCA variance kept. |
| `N_SPLITS` / `N_REPEATS` | `5` / `5` | CV folds; repeats per bootstrap. |
| `BOOTSTRAPS` | `5` | Independent pseudo-trial draws. |
| `BALANCE_METHOD` | `'subsample'` | Equal classes by subsampling. |
| `WINDOW_SIZE` / `STEP_SIZE` | `64` / `16` | Samples at 256 Hz: 250 ms windows every 62.5 ms. |
| `N_SHUFFLE_PERMS` | `50` | Shuffle refits per bootstrap. |
| `UNIT_OF_ANALYSIS` | `'repeat'` | What one sample is: `'repeat'`, `'fold'` (folds kept separate) or `'bootstrap'` (one mean per bootstrap). Changes the sample count and so every error band and test. |
| `PERCENTILE` / `CLUSTER_PERCENTILE` / `N_CLUSTER_PERMS` | `95` / `95` / `100` | The true-vs-shuffle test (§2.3). `N_CLUSTER_PERMS` is also the permutation count of the context test. |
| `STAT_FUNC_CHOICE`, `P_THRESH_FOR_TIME_PERM_CLUSTER_STATS`, `P_CLUSTER`, `PERMUTATION_TYPE` | `'ttest'`, `0.025`, `0.025`, `'independent'` | The context-comparison test. |
| `RANDOM_STATE` | `42` | Bootstrap *b* uses `42 + b`. |
| `RUN_VISUALIZATION_DEBUG` | `False` | Plot trials and the decision boundary on the first two PCs. |

The commented "testing params" block near the bottom (one subject, 2 splits,
1 repeat, 2 bootstraps) is the smoke test: uncomment it and run one condition
before a full submission.

`FIRST_TIME_POINT` in the file is not used by the decoder: `process_bootstrap.py`
passes −1.0 s, the start of the `Stimulus_-1.0to1.5sec` epochs. A response-locked
or differently cropped epochs file needs that line changed, or the figures' time
axis is shifted.

---

### 5. How to run it

#### 5.1 Set up the shell

```bash
REPO=/hpc/home/$USER/coganlab/$USER/GlobalLocal
cd $REPO/dcc_scripts/decoding
export EPOCHS_ROOT_FILE=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
COND_CSV=$REPO/dcc_scripts/stats/results/$EPOCHS_ROOT_FILE/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none
PROP_CSV=$REPO/dcc_scripts/stats/results/$EPOCHS_ROOT_FILE/anova_conjunction_window_0.0to1.5s_sig_lpfc_proportion_none
```

The table must come from the same epochs file as the decode, which the paths
above guarantee. A missing table is made by
`dcc_scripts/stats/submit_stability_flexibility_anova_conjunction_dcc.sh`
([`analysis_guide.md`](analysis_guide.md) §17.5 step 1).

#### 5.2 Common runs

| Question | Command |
|---|---|
| Main effects in each main-effect population | `ANOVA_LABELS_CSV=$COND_CSV bash submit_specific_conditions_decoding_dcc.sh` (the default: 3 populations × 2 sets) |
| LWPC/LWPS in the main-effect populations | `ANOVA_LABELS_CSV=$COND_CSV CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" bash submit_...` |
| LWPC/LWPS in the LWPC-only / LWPS-only electrodes | run twice with `ANOVA_LABELS_CSV=$PROP_CSV` and `ANOVA_LABEL_EFFECT=lwpc_only`, then `lwps_only` |
| Everything in the region, no selection | `ANOVA_LABELS_CSVS=("")` in the script (§3.1), then `CONDITIONS="..." bash submit_...` |
| Two regions | add `ROIS=lpfc,occ` |
| Selection on held-out trials | `bash submit_decoding_with_anova_electrode_selection_dcc.sh` (§3.3) |

#### 5.3 Before you submit

- The script echoes `condition=… anova_labels=… effect=…` per job. That is what
  will run.
- Logs: `out/slurm_<jobid>_dec_a<table>e<pop>_<condition>.out`. The top prints
  subjects, conditions, ROIs, the selection settings and every decoding parameter;
  the `[anova-labels] decoding with N electrodes` line gives the set size.

#### 5.4 Cost

Per comparison × ROI × bootstrap: `N_REPEATS × N_SPLITS` true fits and
`N_SHUFFLE_PERMS × N_SPLITS` shuffle fits at every window, so the shuffle is ten
times the true decode. A `pooled_shuffle` adds another 50 × 5 per ROI. The wrapper
asks for 48 h and 225 GB; the memory is for loading every subject's epochs. The
selection launchers add the ANOVA permutations (§3.3) on top.

#### 5.5 Leave one subject out

`submit_loo_decoding_dcc.sh` submits one job per subject in its
`LEAVE_OUT_SUBJECTS` list and condition set, each with that subject dropped
(`_loo-<subject>` in the filenames). A result that disappears when one subject is
left out rests on that subject. Edit the two arrays at the top first; the list
must name subjects that are in `SUBJECTS` (as shipped it includes `D0130`, which
is not, so that job decodes everyone).

---

### 6. Outputs

#### 6.1 Where

| Route | Save directory |
|---|---|
| no table | `dcc_scripts/decoding/figs/<EPOCHS_ROOT_FILE>/` (all condition sets share it; the comparison names keep them apart) |
| csv | `…/figs/<EPOCHS_ROOT_FILE>/anova_label_selections/<table>__effect-<pop>__correction-<c>__alpha-<a>__roi-all__<hash>/<CONDITION_NAME>/` |
| ANOVA or coupling sets | `…/figs/<EPOCHS_ROOT_FILE>/elecset_<set>/` per set |

`.png`, `.pdf`, `.pkl` and friends are git-ignored: the results live on the
cluster.

#### 6.2 What

Inside the save directory, with `<params>` =
`job<SLURM id>_<n>_subs_[<electrodes>_]<clf>_<B>bts_<S>splts_<R>rps_<unit>_unit_ev_<ev>`:

| File | Contents | Read it for |
|---|---|---|
| `<comparison>/<roi>/<timestamp>_true_v_shfle_<comparison>_<roi>_<params>.{png,pdf,eps}` | Mean true accuracy ± 1 SD over the 25 samples, the shuffle mean ± 1 SD, and a `> shuffle` bar over the surviving clusters | **Is the contrast decodable, and when** |
| `<context>/<roi>/<timestamp>_<context>_comparison_<roi>_<params>.*` | The two comparisons of the context (e.g. 25% vs 75% incongruent) and the pooled shuffle; a solid and a dashed bar where one beats the other; under them, each trace's own `> chance` bar | **Does decodability differ between the two blocks** |
| `<context>/<roi>/<timestamp>_<context>_ACC_DIFF_plot_<roi>_<params>.*` | Sample-wise difference (first − second) ± 1 SD, with the same two bars | The size and timing of that difference |
| `<comparison>/<roi>/<timestamp>_<condition>_<roi>_<comparison>_SUMMED_<B>boots_…_time_averaged_confusion_matrix.png` | Row-normalised confusion matrix, whole epoch, counts summed over bootstraps | Whether one class dominates the errors |
| `<comparison>/<roi>/…DEBUG_CM_Traces_<comparison>…` | Each confusion-matrix cell over time (correct in green, errors in red) | Debugging: a class the decoder always predicts |
| `<timestamp>_MASTER_RESULTS_<params>_<condition>.pkl` | Everything needed to re-plot or re-test (§6.3) | Numbers and replots |
| `confusion_matrices/` | Created by every bootstrap; normally empty | – |

Every title names the decoded analysis and the electrode set, and every filename
carries the job ID, so figures from different runs in the same folder stay
attributable.

#### 6.3 The results pickle

```python
import pickle
with open(path, 'rb') as f:
    r = pickle.load(f)

r['metadata']['time_window_centers']          # seconds, one per window
r['metadata']['n_electrodes'], r['metadata']['electrodes']   # {roi: {subject: [electrodes]}}
r['metadata']['args']                          # every setting of the run
s = r['stats']['i_vs_c_at_inc25']['lpfc']
s['repeat_true_accs']                          # (25, n_windows)
s['repeat_shuffle_accs']                       # (250, n_windows)
s['significant_clusters']                      # bool per window: the > shuffle bar
r['stats']['pooled_shuffles']['lpfc']['lwpc_block_balanced']      # the context null
r['comparison_clusters']['lpfc']['lwpc_block_balanced']['1_over_2']['clusters']   # 25% > 75%
r['comparison_clusters']['lpfc']['lwpc_block_balanced']['2_over_1']['clusters']   # 75% > 25%
```

The keys under `stats` are the registry's comparison names; the context keys are
the registry's `condition_name`, lower-cased. `replot_master_results(path, save_dir)`
in `src/analysis/decoding/plots/replot.py` redraws every figure a pickle
supports, and `replot_all` does a whole tree; `replot_all_decoding_figures.ipynb`
in `dcc_scripts/decoding/` drives them.

---

### 7. How to read the result

Read in this order.

**Step 1 — the log.** Check the dropped-electrode summary, the electrode count
(`[anova-labels] decoding with N electrodes`, or `DECODING ON ELECTRODE SET … (N
electrodes, M subjects)`), and that every bootstrap finished (`No data generated
for bootstrap` means one did not).

**Step 2 — is it decodable? (`true_v_shfle`)**

| What you see | Reading |
|---|---|
| `> shuffle` bar after stimulus onset, true mean clearly above the shuffle band | Decodable, in those windows. Report onset and extent, not a single window's accuracy. |
| no bar | Not decodable at this threshold. Not evidence of absence: 25 samples and a percentile test are not a power analysis. |
| bar in windows centred before −0.125 s | Nothing about the current stimulus exists there. For congruency or switch type, it is an artifact meter: block-level baseline differences or fold leakage ([Cross-decoding controls](#cross-decoding-controls) §6). Task can legitimately appear early on repeat trials. |
| the shuffle band sits far from 0.5 | Class imbalance or a pipeline problem; check the time-averaged confusion matrix. |

A window centred at *t* covers *t* ± 125 ms (64 samples), so a window centred at
0.05 s already contains post-stimulus data.

**Step 3 — do the two blocks differ? (the context figure)** Only for condition
sets with a `context_comparison`.

- A solid bar in the first trace's colour marks windows where the first
  comparison (e.g. 25% incongruent) decodes better than the second; a dashed bar
  in the second's colour, the reverse. The lower bars are each trace against
  chance.
- **Direction for the adaptation effects.** LWPC predicts a smaller congruency
  effect in 75%-incongruent blocks, so congruency should decode better in
  25%-incongruent blocks: the `25% I > 75% I` bar. The same holds for switch type
  and the `25% S > 75% S` bar. This is the decoding counterpart of the univariate
  direction test ([`n2_direction_tests.md`](n2_direction_tests.md)).
- **Compare, do not count.** "Decodable in 25% blocks, not in 75% blocks" is not a
  difference; only the between-trace bar is. Read the difference figure for its
  size.
- **Which condition set.** Use the `_block_balanced` sets (§2.1). A difference on
  the 4-cell sets can come from their block mix.

**Step 4 — compare electrode sets correctly.** Accuracy grows with the number of
electrodes, and the sets differ in size. Compare sets on their own two traces (the
within-set context comparison, which set size does not bias), not one set's
accuracy against another's. For selected sets, remember which decodes are on the
selecting effect (§3.2).

**Step 5 — report `n_electrodes` and `n_subjects` together.** Both are in the
pickle's metadata and the figure titles. The p-values are over resamples of the
same trials (§2.3); the leave-one-subject-out runs (§5.5) are the protection.

---

### 8. Known issues

1. **`ANOVA_LABELS_CSVS` cannot be cleared from the environment** (§3.1), and the
   shipped default is a saved table, so the no-table run needs a file edit.
2. **`ANOVA_LABEL_EFFECTS` from the environment is ignored** (§3.2).
3. **No mode check on population names** (§3.2), unlike A4.
4. **The def-split launcher's default `STRATA` names columns that do not exist**
   (§3.4).
5. **`FIRST_TIME_POINT` in `run_decoding_dcc.py` is unused**; the decoder assumes
   epochs start at −1.0 s (§4.2).
6. **The pseudo-trials carry no between-electrode covariance** (§2.2), which
   bounds what these accuracies can say about population codes
   ([`analysis_plans.md` › Simplification plan](analysis_plans.md#simplification-plan) §1.1–1.2).
7. **The no-table runs of every condition set share one folder**, kept apart only
   by comparison names and job IDs.

---

### 9. Checklist

```
[ ] export EPOCHS_ROOT_FILE; the table (if any) comes from the same file
[ ] choose the route: no table (edit ANOVA_LABELS_CSVS), csv, or a held-out launcher
[ ] population names match the table's mode (_condition_ vs _proportion_)
[ ] CONDITIONS are registry keys with comparisons (block-balanced sets for block contrasts)
[ ] smoke test: the testing-params block, one condition
[ ] check the echoed submission lines
[ ] log: electrode count, subjects, every bootstrap finished
[ ] true_v_shfle: > shuffle bar after onset, none before -0.125 s
[ ] context figure: read the between-trace bars and the difference plot
[ ] compare sets on within-set contrasts, not raw accuracy
[ ] n_electrodes and n_subjects reported together; LOO for anything headline
```

---

### 10. Tests

`pip install -e . pytest`, then `python -m pytest -o addopts="" tests/analysis/decoding -q`.

| Test file | Pins |
|---|---|
| `test_decoding.py` | balancing, windowing, the bootstrap and pooling, `main` end to end on mocked data, `sig` vs `all` electrodes |
| `test_decoding_figures.py` | significance strip layout, legends and titles from the registry, replotting from the pickle |
| `test_trial_splitting.py` | the responsiveness split: disjointness, stratification, the selector |
| `test_anova_electrode_selection.py`, `..._integration.py` | the ANOVA-set split across condition sets, set algebra, a planted interaction on synthetic data |
| `test_coupling_electrode_selection.py` | coupled sets and matched draws |
| `tests/analysis/utils/test_anova_label_selection.py` | population names, corrections, the output-folder slug |

---

### Related documents

- [`analysis_guide.md`](analysis_guide.md) §7 — the decoding modules and the `Decoder` class; §21 — the held-out selection splits in depth; §17.5 — the runbook for the main-effect populations
- [A4 cross-decoding](#a4-cross-decoding) — train on one contrast, test on another, in the same populations
- [N3b block transfer](#n3b-block-transfer) — train in one block, test in the other
- [Cross-decoding controls](#cross-decoding-controls) §6 — what a pre-stimulus cluster means
- [`n2_direction_tests.md`](n2_direction_tests.md) — the univariate direction of the adaptation effects
- [`analysis_plans.md` › Simplification plan](analysis_plans.md#simplification-plan) §1 — what pseudo-trial decoding can and cannot show

---

## A4 cross-decoding

*A4 — cross-decoding congruency ↔ switch type: how to run it and how to read it*

**What this document is.** A standalone walkthrough of the A4 cross-decoding job
(`submit_stability_flexibility_cross_decoding_dcc.sh`): the question it asks, the
four ways it can define electrode groups (`anova`, `csv`, `power_traces`, `none`),
every parameter that changes the answer, the exact commands, every file it writes,
and how to read them. It also covers the task-transfer positive controls, which
run through the same job (§9), and the confound controls built into it: RT
matching (§6.4), the overall-activity control (§6.5) and the response-locked
condition set (§5). **§13 is the runbook for all of them together**: what to run
in what order, how to compare the runs, and what each outcome lets you say.

It is the run-and-read companion to three other documents, and does not repeat
them:

- [`analysis_guide.md`](analysis_guide.md) §17 — why A4 is built the way it is
  (derived class definitions, why condition sets must cross, the payoff 2×2).
- [Cross-decoding controls](#cross-decoding-controls) — what to do when a
  transfer comes back uninformative.
- [N3b block transfer](#n3b-block-transfer) — the block-transfer analysis,
  which is a third mode of this same job.

For the ordinary (non-transfer) decoding job, see [Decoding job](#decoding-job).

If you read only one section, read [§3 The defaults you actually get](#3-the-defaults-you-actually-get):
the submit script and the Python runner disagree on half the knobs, and the
submit script's defaults changed on 2026-09-28. If you have a transfer and want
to know whether it is reportable, read
[§13 The control battery](#13-the-control-battery-run-it-read-it-report-it).

---

### 0. The short version

```bash
cd dcc_scripts/decoding
export EPOCHS_ROOT_FILE=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20

# no electrode groups: decode every task-significant lPFC electrode as one set
ELECTRODE_DEFINITION=none bash submit_stability_flexibility_cross_decoding_dcc.sh

# groups from a saved A1 table. ELECTRODE_DEFINITION=csv is REQUIRED (§4.2)
ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=<A1 result folder> \
    bash submit_stability_flexibility_cross_decoding_dcc.sh

# groups from an ANOVA fit inside the job (the default route)
bash submit_stability_flexibility_cross_decoding_dcc.sh
# the same, with the ANOVA fit on 30% of trials and everything decoded on the other 70%
# (writes to its own ..._split0.3s0 folder)
ELECTRODE_SELECTION_SPLIT=true bash submit_stability_flexibility_cross_decoding_dcc.sh

# the confound controls (§13 has the full battery and how to read it)
ELECTRODE_DEFINITION=none RT_MATCH=rt     bash submit_stability_flexibility_cross_decoding_dcc.sh  # RT-matched trials
ELECTRODE_DEFINITION=none RT_MATCH=random bash submit_stability_flexibility_cross_decoding_dcc.sh  # its trial-count control
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=remove_mean bash submit_stability_flexibility_cross_decoding_dcc.sh  # pattern only
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=mean_only   bash submit_stability_flexibility_cross_decoding_dcc.sh  # overall activity only
```

Then open `summary.txt` in the save directory printed near the top of
`out/slurm_<jobid>_<jobname>.out`. For each electrode group, the result is three
numbers per transfer direction: how many windows beat the shuffle null, how many
fall below the within-contrast ceiling, and what share of the ceiling it keeps
(§8). Whether a transfer survives the controls is read by comparing runs (§13.3).

---

### 1. Where the code lives

| Role | File |
|---|---|
| **Job submitter** (one job per table × population × condition set) | `dcc_scripts/decoding/submit_stability_flexibility_cross_decoding_dcc.sh` |
| **Cluster wrapper** (8 cores, 128 GB, 16 h) | `dcc_scripts/decoding/sbatch_stability_flexibility_cross_decoding_dcc.sh` |
| **The knobs**: environment variables → `args`, and the save directory | `dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py` |
| **The job**: electrode groups → ROI array → designs → outputs | `dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py` (`main`) |
| Contrasts, condition checks, the two label vectors, the decoder | `src/analysis/decoding/cross_decoding.py` |
| The cross-validated decoder (`labels_test`, `stratify_labels`, `frac_train`, `temporal_generalization`) | `src/analysis/decoding/decoder.py` (`Decoder.cv_cm_jim_window_shuffle`) |
| Accuracy and the cluster test against shuffle | `src/analysis/decoding/accuracy_stats.py` |
| `anova` route: one window-mean ANOVA per electrode | `src/analysis/stats/stability_flexibility_segregation.py` (`per_electrode_anova_labels`) |
| `power_traces` route: finished windowed-ANOVA runs | `src/analysis/stats/power_traces_conjunction.py` (`electrode_labels`) |
| `csv` route: a saved A1 `anova_labels.csv` | `src/analysis/utils/anova_label_selection.py` |
| `RT_MATCH`: RT-matched (or count-matched random) decode trials | `src/analysis/utils/rt_matching.py`, called by `_apply_rt_match` in the job |
| `ACTIVITY_CONTROL`: remove each subject's mean across electrodes, or keep only it | `src/analysis/decoding/activity_control.py`, called by `_activity_controlled` / `_decoded_group` in the job |
| The held-out selection split (`ELECTRODE_SELECTION_SPLIT`) | `src/analysis/decoding/anova_electrode_selection.py` (`assign_trial_partitions`, `apply_trial_partition`) |
| Same job, other analyses | `submit_block_transfer_dcc.sh` (`ANALYSIS=block_transfer`, N3b), `submit_task_transfer_dcc.sh` (`ANALYSIS=task_transfer`, §9) |

#### The call path

```
submit_stability_flexibility_cross_decoding_dcc.sh      loops tables x populations x condition sets
 └ sbatch_stability_flexibility_cross_decoding_dcc.sh
    └ run_stability_flexibility_cross_decoding_dcc.py    env vars -> args; validates; builds SAVE_DIR
       └ stability_flexibility_cross_decoding_dcc.main(args)
          ├ (i)  electrode definition -> a labels table (S, F, CPC, SPS, CPS, SPC)
          │        anova         load epochs -> assemble_long_df -> per_electrode_anova_labels
          │        csv           load_anova_labels + load_anova_label_electrodes(effect)
          │        power_traces  power_traces_conjunction.electrode_labels(runs)
          │        none          no table
          ├ groups: both / S_only / F_only (disjoint), plus the reference group 'all'
          ├ (ii) _build_roi_arrays: the ROI pseudopopulation, channels named '<subject>-<electrode>'
          │        load epochs -> selection split (decode side)  [ELECTRODE_SELECTION_SPLIT]
          │                    -> RT matching of the decode trials [RT_MATCH; writes rt_match_*.csv]
          │                    -> pseudopopulation
          ├ factors_are_crossed check; class definitions read from the conditions' declared levels
          ├ per decode: restrict to the group's electrodes -> activity control [ACTIVITY_CONTROL]
          ├ A4(0)   within-block decodes                               [16-cell condition set only]
          ├ A4(0b)  within-block 2x2 per definition group, circular cells skipped   [16-cell + groups]
          ├ A4(a)   per group: stab_to_stab, flex_to_flex, stab_to_flex, flex_to_stab
          ├ A4(c)   temporal generalization on TEMPGEN_GROUPS
          └ cross_decoding.json, accuracy_traces.npz, tempgen_*.npy, anova_labels.csv,
            rt_match_*.csv, figures, summary.txt
```

Every decode in the job goes through the same four steps:

```
cd.run_cross_decoding(arrays, roi, train_strings, test_strings)
 ├ build_cross_decoding_arrays   concatenate conditions; labels_train, labels_test, strata
 │                               (strata = the source condition, so folds stay balanced on
 │                               the labelling that is SCORED, not only the trained one)
 ├ make_decoder                  PCA (EXPLAINED_VARIANCE) -> LDA with equal class priors
 ├ cv_cm_jim_window_shuffle x2   true labels, then train labels permuted and refit (the null)
 └ _summarise                    confusion matrices -> accuracy per window x repeat
                                 -> time_perm_cluster, true vs shuffle, one-tailed, N_PERM
```

---

### 2. What A4 asks

The A1/A2 counts show whether the same electrodes carry both effects. They cannot
say whether those electrodes carry **one code** or **two codes that happen to
share electrodes**. A4 trains a classifier on one contrast and scores it on the
other, on the same trials:

- **stability** = congruency: incongruent (class 0) vs congruent (class 1);
- **flexibility** = switch type: switch (class 0) vs repeat (class 1).

`stab_to_flex` trains on congruency and scores the predictions against switch
type. If the axis that separates incongruent from congruent also separates switch
from repeat, the predictions track switch type and accuracy beats the shuffle null.

> **The transfer is signed.** Class 0 is paired with class 0: the classifier's
> "incongruent" is scored as "switch". A shared axis on which incongruent and
> switch trials both sit on the same side comes out **above** chance. A shared
> axis with the opposite pairing (incongruent with repeat) comes out reliably
> **below** chance. That is still a shared axis, not a null; see
> [Cross-decoding controls](#cross-decoding-controls) §5 before reporting it.

A transfer means nothing without its **ceiling**: the within-contrast decode of
the labelling it is scored on, on the same trials with the same folds.
`stab_to_flex` is read against `flex_to_flex`, and `flex_to_stab` against
`stab_to_stab`. The job runs all four in every group and compares them for you.

#### The designs

| Design | What is decoded | Runs when |
|---|---|---|
| **A4(0)** within-block | congruency inside 25%- and 75%-incongruent blocks; switch type inside 25%- and 75%-switch blocks, on all loaded electrodes | the condition set declares both proportions (`stimulus_experiment_conditions`) |
| **A4(0b)** within-block per group | the same 2×2 plus the two cross cells, on each definition group (CPC/SPS/CPS/SPC, or congruency/switch_type for main-effect labels), skipping the cells that group was selected on | as A4(0), and the route defines groups |
| **A4(a)** label transfer | the two transfers and the two ceilings, on each group | always |
| **A4(c)** temporal generalization | train at one window, test at every window: congruency within, switch type within, congruency → switch type | `TEMPGEN_GROUPS` is non-empty |

A4(a) is pooled over the block proportions: its classes are every incongruent
cell against every congruent cell, whichever condition set you use. Only
A4(0)/A4(0b) split by block.

---

### 3. The defaults you actually get

`run_stability_flexibility_cross_decoding_dcc.py` has its own defaults, used when
you run it directly (the synthetic dry runs). The submit script sets different
ones and passes them all through `sbatch --export`, so **a submitted job uses the
right-hand column**:

| Knob | Runner default (`python run_...`) | Submit-script default (`bash submit_...`) |
|---|---|---|
| `CONDITIONS` | `stimulus_experiment_conditions` (16 cells) | `stimulus_main_effect_conditions` (4 cells) |
| `ELECTRODE_DEFINITION` | `anova` | `anova` |
| `CONTRAST_MODE` | `proportion`, or read off the CSV folder | `condition`, or read off the CSV folder |
| `FDR_CORRECTION` | `fdr_bh` | `none` (raw p), or `flags` on the csv route |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` s | `0.0` / `1.5` s |
| `WINDOW_SIZE` / `STEP_SIZE` | `20` / `10` samples | `64` / `16` samples (250 / 62.5 ms) |
| `ANOVA_LABEL_EFFECT` | `both` | `union` |
| `EPOCHS_ROOT_FILE` | required for real data | the `..._filterbank_hilbert_stat_func_ttest_zmax_20` file |
| `ROI`, `ELECTRODES`, `ALPHA` | `lpfc`, `sig`, `0.05` | same |
| `N_SPLITS`, `N_REPEATS`, `N_PERM`, `EXPLAINED_VARIANCE` | `5`, `10`, `500`, `0.8` | same |
| `ELECTRODE_SELECTION_SPLIT` | `false` | same |
| `RT_MATCH` (+ `RT_MATCH_BINS`, `_BALANCE`) | `none` (`10`, `equal`) | same |
| `ACTIVITY_CONTROL` | `none` | same |

So `bash submit_stability_flexibility_cross_decoding_dcc.sh` with nothing set is:
**main-effect groups** (congruency and switch-type main effects, raw p < 0.05,
window-mean HG over 0–1.5 s), fit in the job on the task-significant lPFC
electrodes, with the transfer pooled over both proportions. The within-block
designs A4(0)/A4(0b) do **not** run, because the 4-cell condition set has no block
factor. No selection split, no RT matching and no activity control run unless you
ask for them; each one, when set, adds a tag to the output folder (§13.1).

> **Stale defaults elsewhere.** [`analysis_guide.md`](analysis_guide.md) §17.4
> still describes the older submit defaults (csv route, 16-cell set, 0–0.5 s,
> 20/10 samples), and the §17.5 runbook's step 3 command
> (`ANOVA_LABELS_CSV=$COND_CSV bash submit_...`) no longer reads the table,
> because the default route is now `anova` (§4.2). This document describes the
> scripts as they are.

---

### 4. Electrode definitions

Three choices decide which electrodes are decoded, and they are easy to conflate:

1. **`ROI`** — which region (a key of `src/analysis/config/rois.py`: `lpfc`,
   `acc`, `dlpfc`, `parietal`, `occ`, `v1`, …).
2. **`ELECTRODES`** — which of that region's electrodes are loaded. `sig` keeps
   the electrodes whose high gamma beats their pre-stimulus baseline (read from
   `sig_chans_<subject>_<EPOCHS_ROOT_FILE>.json`, so the epochs file also picks the
   significance file); `all` keeps every electrode in the ROI. **The csv route
   ignores this** and always loads every ROI electrode.
3. **`ELECTRODE_DEFINITION`** — how the loaded electrodes are split into groups.

Every route except `none` produces a labels table with a binary `S` and `F` flag
per electrode. The groups are the three disjoint cells of that table, plus the
reference group:

| Group | Electrodes | Named, `CONTRAST_MODE=proportion` | Named, `CONTRAST_MODE=condition` |
|---|---|---|---|
| S and F | carry both effects | `both` | `both` |
| S only | carry the stability effect only | `S_only` | `congruency_only` |
| F only | carry the flexibility effect only | `F_only` | `switch_type_only` |
| reference | every loaded electrode, selected by nothing A4 decodes | `all` (`REFERENCE_GROUP`) | `all` |

`proportion` flags are the LWPC (congruency × incongruent proportion) and LWPS
(switch type × switch proportion) interactions; `condition` flags are the
congruency and switch-type main effects. Groups with fewer than `MIN_GROUP_SIZE`
(5) electrodes are skipped, with a line in the log. If a group happens to equal
the whole array (e.g. `both` in the synthetic data), the reference group is not
added a second time.

#### 4.1 `anova` — fit the ANOVA in the job (default)

```bash
bash submit_stability_flexibility_cross_decoding_dcc.sh                                 # main effects
CONTRAST_MODE=proportion bash submit_stability_flexibility_cross_decoding_dcc.sh        # LWPC / LWPS
WINDOW_TMIN=0.2 WINDOW_TMAX=0.7 FDR_CORRECTION=fdr_bh bash submit_stability_flexibility_cross_decoding_dcc.sh
```

- **What it fits:** one ANOVA per electrode on the window-mean high gamma over
  `[WINDOW_TMIN, WINDOW_TMAX]`, on the `ELECTRODES` set of `ROI`
  (`per_electrode_anova_labels`). An electrode is flagged when its p (or BH q,
  across electrodes) is below `ALPHA`. The flag ignores direction.
- **Knobs that matter:** `CONTRAST_MODE`, `WINDOW_TMIN`/`WINDOW_TMAX`,
  `FDR_CORRECTION` (`none` or `fdr_bh`; `flags` is refused because there is no
  saved table), `ALPHA`, `ELECTRODES`.
- **`ELECTRODE_SELECTION_SPLIT=true`** fits the ANOVA on
  `ELECTRODE_SELECTION_FRAC` (0.3) of each subject's trials, stratified on
  congruency, task sequence and block, and decodes only the other 70%. The split
  is keyed on each trial's `metadata.trial_count`, so a physical trial cannot be
  in the selection half under one condition and the decode half under another.
  With the split, A4(0b) keeps the cells that would otherwise be circular. This is
  the "clean-ceiling" version of the csv run (see §8 step 2).
- **Writes** `anova_labels.csv` (the table it fit) into the save directory.
- **Save directory:**
  `results/<EPOCHS_ROOT_FILE>/cross_decoding_<roi>_window_<tmin>to<tmax>s_<electrodes>_anova_<mode>_<correction>/<CONDITIONS>/`,
  e.g. `cross_decoding_lpfc_window_0.0to1.5s_sig_anova_condition_none/stimulus_main_effect_conditions/`.

> **The split names its folder.** A split run writes to
> `..._anova_<mode>_<correction>_split<frac>s<seed>/` (e.g. `_split0.3s0`), so it
> no longer overwrites the unsplit run of the same settings, and `summary.txt`
> records it on its `electrode_selection_split` line. (Before 2026-09-30 neither
> was true; a split run from then shares the unsplit folder, and only the slurm
> log's `[trial-split]` lines tell them apart.) `SAVE_DIR=...` still overrides
> the whole path.

#### 4.2 `csv` — reuse a saved A1 table

```bash
ELECTRODE_DEFINITION=csv \
ANOVA_LABELS_CSV=$REPO/dcc_scripts/stats/results/$EPOCHS_ROOT_FILE/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none \
    bash submit_stability_flexibility_cross_decoding_dcc.sh
```

- **`ELECTRODE_DEFINITION=csv` is required.** With any other route the submit
  script throws the table list away (it would only name folders after tables the
  job never reads), so `ANOVA_LABELS_CSV=... bash submit_...` on its own silently
  runs the in-job `anova` route instead. The first `echo` line of the submission
  says `anova_labels=none` when this happens.
- **`ANOVA_LABELS_CSV`** is the A1 `anova_labels.csv` or its result folder. Without
  it, the job uses the `ANOVA_LABELS_CSVS` array in the submit script (one
  condition-mode table as shipped), one job per entry.
- **`ANOVA_LABEL_EFFECTS`** (space-separated, default `union`) says which population
  of the table each job starts from; one job per name. The job restricts the
  table to that population, then decodes the disjoint groups left in it:
  - `union` — every electrode with either effect, so `both`, the two `*_only`
    groups and `all` in **one** job. This is the normal choice.
  - `both`, `congruency_only`, … — only that population; the other groups come
    out empty and are skipped. `all` is still decoded.
  - Names must belong to the table's mode: `lwpc`, `lwps`, `lwpc_only`,
    `lwps_only` for a proportion table; `congruency`, `switch_type`,
    `congruency_only`, `switch_type_only` for a condition table. The submit
    script skips the other mode's names with a message, and the runner refuses
    them.
- **`CONTRAST_MODE` is read off the table's folder name** (`..._<roi>_<mode>_<correction>`).
  The submit script passes the folder's mode whatever `CONTRAST_MODE` says; the
  runner, run directly, refuses a contradiction.
- **`ANOVA_LABEL_CORRECTION`** (`flags` default, `none`, `fdr_bh`) and
  **`ANOVA_LABEL_ALPHA`** become the job's `FDR_CORRECTION`/`ALPHA`. `flags`
  keeps the table's own 0/1 flags (whatever correction built it); `none` and
  `fdr_bh` re-threshold its saved p or q columns at `ANOVA_LABEL_ALPHA`.
- **`ELECTRODES` is ignored.** The ROI array holds every ROI electrode, so the
  `all` group is every lPFC electrode, not only the task-significant ones, even
  though the table itself was fit on the `sig` electrodes and the folder name
  still says `sig`. Controls for a csv run (the task transfer, §9) should
  therefore use `ELECTRODES=all`.
- **Unused but still in the folder name:** `WINDOW_TMIN`/`WINDOW_TMAX` and
  `ELECTRODES`. `ELECTRODE_SELECTION_SPLIT` is refused: a saved table has no
  record of which trials fit it.
- **Save directory:**
  ```
  results/<EPOCHS_ROOT_FILE>/cross_decoding_<roi>_window_<tmin>to<tmax>s_<electrodes>_csv_<mode>_<correction>/
      <CONDITIONS>/anova_label_selections/
      <table folder>__effect-<population>__correction-<c>__alpha-<a>__roi-all__<hash>/
  ```
  The folder repeats the table's own name, so the A1 window and correction are
  recoverable from the path.

> **Gotcha: the table was fit on the trials A4 decodes.** Each group's within
> decode of the effect that selected it (`stab_to_stab` on `congruency_only`,
> `flex_to_flex` on `switch_type_only`, both on `both`) is inflated by selection.
> The `all` group's ceilings are not. For numbers with no trial overlap anywhere,
> run the `anova` route with `ELECTRODE_SELECTION_SPLIT=true` (§4.1).

#### 4.3 `none` — no groups, just the loaded electrodes

```bash
ELECTRODE_DEFINITION=none bash submit_stability_flexibility_cross_decoding_dcc.sh                 # task-significant lPFC
ELECTRODE_DEFINITION=none ELECTRODES=all bash submit_stability_flexibility_cross_decoding_dcc.sh  # every lPFC electrode
ELECTRODE_DEFINITION=none ROI=occ bash submit_stability_flexibility_cross_decoding_dcc.sh         # another region
```

- **What it does:** no ANOVA, no table. The only group is the reference group,
  i.e. every electrode `ELECTRODES` loads. With `ELECTRODES=sig` that is every
  task-significant electrode of the ROI. This is the plain question "does this
  region's code transfer?", with no selection on either effect.
- **Temporal generalization** runs on that group by default (`TEMPGEN_GROUPS`
  defaults to `REFERENCE_GROUP` here, instead of `both`).
- **What does not run:** A4(0b) (no definition groups). A4(0) still runs if you
  pass the 16-cell set (`CONDITIONS=stimulus_experiment_conditions`).
- **Unused and left out of the folder name:** `WINDOW_TMIN`/`WINDOW_TMAX`,
  `CONTRAST_MODE`, `FDR_CORRECTION`, `ALPHA`, and every table setting.
- **`REFERENCE_GROUP` must be non-empty**; the runner refuses `''`.
- **No `anova_labels.csv`** is written.
- **Save directory:** `results/<EPOCHS_ROOT_FILE>/cross_decoding_<roi>_<electrodes>_none/<CONDITIONS>/`,
  e.g. `cross_decoding_lpfc_sig_none/stimulus_main_effect_conditions/`.

#### 4.4 `power_traces` — reuse finished windowed-ANOVA runs

```bash
ELECTRODE_DEFINITION=power_traces CONTRAST_MODE=proportion \
POWER_TRACES_RUN_DIR=/path/to/power_traces/run \
    bash submit_stability_flexibility_cross_decoding_dcc.sh
```

- **What it reads:** the within-electrode windowed ANOVA runs from the power-traces
  pipeline and their permutation cluster correction. An electrode is flagged when a
  cluster for the LWPC (LWPS) interaction survives anywhere in time. More sensitive
  to brief interactions than the window mean, and the groups become exactly the
  electrodes the power-trace figures call significant.
- **Run directories:** `POWER_TRACES_RUN_DIR` (one run whose ANOVA carried all
  four interactions, e.g. a `stimulus_experiment_conditions` run), or one per
  interaction with `POWER_TRACES_CPC`, `_SPS`, `_CPS`, `_SPC`.
- **`POWER_TRACES_CORRECTION`**: `fdr_bh` (default; BH across electrodes),
  `cluster` (raw cluster p, the older lab convention) or `none` (any surviving
  cluster). `POWER_TRACES_ROI` restricts the table to one ROI.
- **Needs no epochs** for the definition step (the decode still loads them).

> **Gotcha: pass `CONTRAST_MODE=proportion`.** This route always reads the
> interactions, but the submit script defaults `CONTRAST_MODE` to `condition`.
> Left at that, the groups are misnamed `congruency_only`/`switch_type_only`, and
> the A4(0b) circularity guard treats them as main-effect groups: it skips every
> decode of each group's contrast and drops the CPS/SPC groups.

---

### 5. Condition sets

`CONDITIONS` names a dict in `src/analysis/config/experiment_conditions.py`. A4
needs every condition to declare `congruency` **and** `switchType`, and needs the
two to cross (all four combinations present). Two sets qualify:

| `CONDITIONS` | Cells | Designs that run | Why pick it |
|---|---|---|---|
| `stimulus_main_effect_conditions` (**submit default**) | 4: `Stimulus_{i,c}{r,s}`, both proportions pooled | A4(a), A4(c) | about 4× the trials per cell, so fewer incomplete rows; folds stratified on congruency × switch type |
| `stimulus_experiment_conditions` (runner default) | 16: the full 2×2×2×2 | all four | the only set with block factors; folds stratified on all four factors |
| `response_main_effect_conditions` | 4: `Response_{i,c}{r,s}`, both proportions pooled | A4(a), A4(c) | the response-locked twin of the submit default; needs a `Response_...` epochs file |

`response_main_effect_conditions` and `response_experiment_conditions` are the
response-locked 4- and 16-cell sets. Pair them with a response-locked
`EPOCHS_ROOT_FILE` (`make_epoched_data.py` writes a `Response_...` file next to
each `Stimulus_...` one, with the same settings) and set `FIRST_TIME_POINT` to
that file's first sample (−1.0 for a −1.0 to 1.5 s epoch). With
`ELECTRODES=sig` the electrodes come from the response file's own significance
list, so they are not exactly the stimulus run's; the log prints the count.

Single-factor sets (`stimulus_congruency_conditions`, …) are
refused: they are separate epoch sets over the same trials, so the transfer would
be scored on trials it trained on. Confounded sets (`stimulus_iS_cR_err_conditions`
and siblings) are refused: congruency and switch type split their trials
identically, so a "transfer" would be the within decode. `CONDITIONS="a b"`
submits one job per set.

---

### 6. Parameters

All are environment variables; nothing needs a file edited. Defaults below are
the submit script's (§3).

#### 6.1 Data and electrodes

| Variable | Default | Notes |
|---|---|---|
| `EPOCHS_ROOT_FILE` | the `_ttest_zmax_20` file | Also picks the `sig_chans` file. Export it once so every script agrees. |
| `DATA_SOURCE` | `real` | `synthetic` builds a pseudopopulation with a planted answer (§7.1). |
| `SYNTHETIC_CODE` | `shared` | `shared` must transfer; `orthogonal` must not. |
| `CONDITIONS` | `stimulus_main_effect_conditions` | §5. Space-separated for several jobs. |
| `ROI` | `lpfc` | A key of `config/rois.py`. |
| `ELECTRODES` | `sig` | `sig` or `all`. Ignored on the csv route. |
| `REFERENCE_GROUP` | `all` | Name of the unselected group; `''` drops it (not allowed with `none`). |
| `MIN_GROUP_SIZE` | `5` | Groups with fewer electrodes are skipped. |

#### 6.2 Electrode definition

| Variable | Default | Used by | Notes |
|---|---|---|---|
| `ELECTRODE_DEFINITION` | `anova` | – | `anova`, `csv`, `power_traces`, `none` (§4). |
| `CONTRAST_MODE` | `condition` | anova, power_traces | Read off the folder on csv. Set `proportion` for power_traces. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `1.5` | anova | The ANOVA window, seconds from stimulus onset. Not the decoding window. |
| `FDR_CORRECTION` | `none` (`flags` on csv) | anova, csv | `none` = raw p; `fdr_bh` = BH across electrodes. |
| `ALPHA` | `0.05` | anova, csv, power_traces | |
| `ELECTRODE_SELECTION_SPLIT` | `false` | anova | Fit on 30%, decode 70% (§4.1). |
| `ELECTRODE_SELECTION_FRAC` / `_SEED` | `0.3` / `0` | anova + split | |
| `ANOVA_LABELS_CSV` | the script's list | csv | One table or its folder. |
| `ANOVA_LABEL_EFFECTS` | `union` | csv | Space-separated populations; one job each. `ANOVA_LABEL_EFFECT` (one name) also works. |
| `ANOVA_LABEL_CORRECTION` / `_ALPHA` | `flags` / `0.05` | csv | Become `FDR_CORRECTION` / `ALPHA`. |
| `ANOVA_LABEL_ROI` | unset | csv | Only if the table has a `roi` column. |
| `POWER_TRACES_RUN_DIR` or `POWER_TRACES_CPC`/`_SPS`/`_CPS`/`_SPC` | unset | power_traces | |
| `POWER_TRACES_CORRECTION` / `POWER_TRACES_ROI` | `fdr_bh` / unset | power_traces | |

#### 6.3 Decoding

| Variable | Default | Notes |
|---|---|---|
| `WINDOW_SIZE` / `STEP_SIZE` | `64` / `16` | Samples at 256 Hz: 250 ms windows every 62.5 ms, 37 windows over −1.0 to 1.5 s. |
| `N_SPLITS` | `5` | Folds; or random resamples per repeat when `FRAC_TRAIN` is set. |
| `N_REPEATS` | `10` | Repeats of the fold split. **The samples of the cluster test** and the main runtime lever. Temporal generalization uses half (at least 2). |
| `FRAC_TRAIN` | unset | Unset = `StratifiedKFold`, (N_SPLITS−1)/N_SPLITS train. Set (e.g. `0.5`) for `StratifiedShuffleSplit` at that fraction. Also a probe for fold leakage ([Cross-decoding controls](#cross-decoding-controls) §6). |
| `EXPLAINED_VARIANCE` | `0.8` | PCA variance kept, refit per fold. |
| `N_PERM` | `500` | Permutations for each cluster test. |
| `TEMPGEN_GROUPS` | `both` (`all` under `none`) | Comma-separated. `''` skips A4(c). `both,all` adds the unselected matrix. Each matrix costs `n_windows²` predictions. |
| `TRAIN_LABEL` / `TEST_LABEL` | unset | One decode instead of the battery: `stability`/`congruency`, `flexibility`/`switchType`. Goes to a `train_<x>_test_<y>/` subfolder and has no ceiling comparison. |
| `SAMPLING_RATE` / `FIRST_TIME_POINT` | `256` / `-1.0` | Only label the figure time axes. Change `FIRST_TIME_POINT` for an epoch that does not start at −1.0 s. |
| `SEED` | `0` | Seeds the folds, the ROI-array padding and the cluster tests. |
| `SAVE_DIR` | derived | Overrides the whole output path. |

#### 6.4 RT matching

Incongruent and switch trials are slower (in `combinedData.csv`, correct trials:
+153 ms and +197 ms per subject on average). In stimulus-locked data, activity
that tracks time-to-response then differs between the levels of both factors, and
a transfer can come from that shared latency rather than a shared code.
`RT_MATCH` subsamples the decode trials, per subject, so every combination of the
decoded factors has the same RT distribution
([RT matching](#rt-matching) has the method).

| Variable | Default | Notes |
|---|---|---|
| `RT_MATCH` | `none` | `rt`: the RT-matched subset. `random`: its control, the same number of trials per subject and cell drawn without regard to RT. Compare `rt` with `random`, not with `none`: matching keeps about half the trials, and `random` pays the same cost. |
| `RT_MATCH_BINS` | `10` | Quantile RT bins per subject. 10 leaves ~+2 / +1 ms (i − c / s − r, n.s.) on the behavioural data; 5 leaves i − c at +8 ms (p = 0.03). |
| `RT_MATCH_BALANCE` | `equal` | `equal`: the same count per cell and bin. `proportional`: keep the cells' size ratios. |
| `RT_MATCH_WITHIN` | unset | Extra strata, comma-separated, e.g. `incongruent_proportion,switch_proportion` to match within each block type as well. |
| `RT_MATCH_GROUPS` | the decoded factors | Override which factors are matched (A4: `congruency`, `switchType`; N3b / task transfer: the design's contrast and transfer factor). |
| `RT_MATCH_SEED` | `SEED` | Seeds the draw. Use the same seed for the `rt` and `random` runs. |

- Matching runs on the decode trials only, after `ELECTRODE_SELECTION_SPLIT`; the
  `anova` route's electrode definition still uses its own (unmatched) trials.
- The default matches the four congruency × switch-type cells pooled over blocks,
  which is what A4(a) decodes. The within-block designs A4(0)/A4(0b) (16-cell set)
  compare classes inside one block, so for them add
  `RT_MATCH_WITHIN=incongruent_proportion,switch_proportion`.
- Each mode gets its own folder: `..._rtmatch10/`, `..._rtrandom10/` (plus
  `_proportional`, `_within-...`, `_groups-...`, `_seed<n>` when those are
  non-default).
- It writes `rt_match_<factors>_balance.csv` (per subject × cell: counts and RTs
  before/after), `_contrasts.csv` (per-subject RT difference per factor) and
  `_summary.csv` (across subjects: mean and one-sample t, before/after), and adds
  an `RT matching of the decode trials` block to `summary.txt`. The `after` row
  of `_summary.csv` is the "residual RT difference" to report.
- A subject left with no trials stops the job: its channels are in every
  pseudo-trial, so it cannot drop out. Lower `RT_MATCH_BINS` if that happens.
- The decoder trains only on pseudo-trials that are NaN-free on every decoded
  electrode of every subject (§4.4 under [Cross-decoding controls](#cross-decoding-controls)),
  and matching halves each subject's trials. Each subject's clean trials are
  therefore paired up into the top rows before decoding. The log line
  `complete pseudo-trials per condition: ... fewest clean trials: <subject>` names
  the subject that caps the count. If a fold still has fewer than two complete
  rows per class, the job stops and says so: drop that subject or lower
  `RT_MATCH_BINS`.
- Synthetic data have no RTs, so `RT_MATCH` is refused there.
- It works for `ANALYSIS=block_transfer` and `task_transfer` too (same knobs).

```bash
ELECTRODE_DEFINITION=none RT_MATCH=rt     bash submit_stability_flexibility_cross_decoding_dcc.sh
ELECTRODE_DEFINITION=none RT_MATCH=random bash submit_stability_flexibility_cross_decoding_dcc.sh
```

#### 6.5 Overall-activity control

| Variable | Default | Notes |
|---|---|---|
| `ACTIVITY_CONTROL` | `none` | `remove_mean`: subtract each subject's mean across the decoded electrodes, per pseudo-trial and time point, so only the pattern across electrodes is decoded. `mean_only`: decode each subject's mean alone, the uniform part only. |

- Applied to every decode in the job (A4(0), A4(0b), A4(a), A4(c); N3b and task
  transfer too), per electrode group after restriction, so a subject's mean is over
  the electrodes that group decodes.
- Each mode gets its own folder (`..._remove_mean/`, `..._mean_only/`); the tags
  stack with the split and RT-matching ones.
- `summary.txt` gets an `activity_control` line and, per group, what it did
  (subjects, features, and the single-electrode subjects `remove_mean` empties).
- How to read it, and what it cannot rule out:
  [Overall-activity control](#overall-activity-control).

```bash
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=remove_mean bash submit_stability_flexibility_cross_decoding_dcc.sh
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=mean_only   bash submit_stability_flexibility_cross_decoding_dcc.sh
```

---

### 7. How to run it

#### 7.1 Dry run on synthetic data (minutes, no data)

The synthetic pseudopopulation has 40 channels, 16 conditions × 40 trials and 32
time samples, with congruency and switch type planted on either the same axis
(`shared`) or orthogonal axes (`orthogonal`). Channels 0–19 are labelled `S_only`,
20–39 `F_only`, and all 40 `both`.

```bash
cd dcc_scripts/decoding
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA_SOURCE=synthetic SYNTHETIC_CODE=shared     WINDOW_SIZE=16 STEP_SIZE=8 \
    python run_stability_flexibility_cross_decoding_dcc.py
DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal WINDOW_SIZE=16 STEP_SIZE=8 \
    python run_stability_flexibility_cross_decoding_dcc.py
DATA_SOURCE=synthetic ELECTRODE_DEFINITION=none WINDOW_SIZE=16 STEP_SIZE=8 \
    python run_stability_flexibility_cross_decoding_dcc.py
```

- The window must fit in 32 samples, hence `WINDOW_SIZE=16` (3 windows at
  `STEP_SIZE=8`).
- Single-threaded BLAS matters off the cluster: the LDA fits are small, and with
  the default threading three runs on a 4-core machine were still in the first
  design after 13 minutes; with the `export` line they finished in minutes.
- Output goes to `results/synthetic_<code>/...`, never to a real run's folder.
- **Planted answer:** under `shared` the transfers beat shuffle and keep most of
  their ceiling; under `orthogonal` both ceilings beat shuffle and the transfers
  sit at shuffle. If that does not happen, a real-data null means nothing.
- The synthetic data have no pre-stimulus period, but the time axis still starts
  at `FIRST_TIME_POINT` (−1.0), so ignore the axis labels on these figures.
- `bash submit_... ` with `DATA_SOURCE=synthetic WINDOW_SIZE=16 STEP_SIZE=16`
  runs the same thing as a cluster job.

§8.1 shows what `summary.txt` looks like for these runs.

#### 7.2 The real runs

Set the shell up once (the same as the [`analysis_guide.md`](analysis_guide.md)
§17.5 runbook):

```bash
REPO=/hpc/home/$USER/coganlab/$USER/GlobalLocal
cd $REPO/dcc_scripts/decoding
export EPOCHS_ROOT_FILE=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
COND_CSV=$REPO/dcc_scripts/stats/results/$EPOCHS_ROOT_FILE/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none
```

Then, depending on the question:

| Question | Command |
|---|---|
| Does lPFC's code transfer at all? | `ELECTRODE_DEFINITION=none bash submit_stability_flexibility_cross_decoding_dcc.sh` |
| … on every lPFC electrode? | `ELECTRODE_DEFINITION=none ELECTRODES=all bash submit_...` |
| Does it transfer in each main-effect population (saved table)? | `ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=$COND_CSV bash submit_...` |
| The same, with no selection/decode trial overlap | `ELECTRODE_SELECTION_SPLIT=true bash submit_...` (own `..._split0.3s0` folder) |
| In the LWPC/LWPS populations | `CONTRAST_MODE=proportion bash submit_...` (in-job) or `ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=<a ..._proportion_... table> bash submit_...` |
| With the within-block designs too | add `CONDITIONS=stimulus_experiment_conditions` |
| With the unselected temporal-generalization matrix | add `TEMPGEN_GROUPS=both,all` |
| The task-transfer positive controls | `bash submit_task_transfer_dcc.sh` (§9) |
| Is the transfer a response-time difference? | `ELECTRODE_DEFINITION=none RT_MATCH=rt bash submit_...` **and** `... RT_MATCH=random ...` (§6.4) |
| … seen from the response? | `ELECTRODE_DEFINITION=none CONDITIONS=response_main_effect_conditions EPOCHS_ROOT_FILE=Response_... bash submit_...` (§5) |
| Is it a uniform rise in activity? | `ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=remove_mean bash submit_...` **and** `... ACTIVITY_CONTROL=mean_only ...` (§6.5) |

Run the `none` job first. It is the cheapest (one group) and is the reference
every grouped run is compared with. §13.2 has the confound controls as one
ordered script, with the flags that keep them cheap.

#### 7.3 Before you submit

- Check the submission lines the script echoes: `condition=`, `anova_labels=`,
  `effect=`, `definition=`, `contrast=`, `correction=`. They are what the job will
  run. `definition=anova` when you meant to pass a table means you left out
  `ELECTRODE_DEFINITION=csv`.
- `TEMPGEN_GROUPS` and `SAVE_DIR` are carried by `--export=ALL` from your
  environment, not from the explicit export list (commas would cut
  `TEMPGEN_GROUPS=both,all` at the first comma). Setting them on the command line
  as above works.
- Logs go to `out/slurm_<jobid>_<jobname>.out` in the directory you submitted
  from. The top of the file prints every setting and the save directory.

#### 7.4 Cost

Each group runs 4 decodes (true + shuffle each) of `N_REPEATS × N_SPLITS` fits
per window, plus 6 cluster tests; temporal generalization adds 3 matrices per
group in `TEMPGEN_GROUPS` at `n_windows²` predictions each. A csv `union` job
decodes up to 4 groups and is the heaviest. The wrapper asks for 16 h; if a job
runs out, resubmit with `SBATCH_TIMELIMIT=36:00:00 bash submit_...`, or halve the
work with `N_REPEATS=5`.

---

### 8. Outputs and how to read them

#### 8.0 The files

Everything goes to the save directory of §4.

| File | Contents | Read it for |
|---|---|---|
| `summary.txt` | Settings, then every design's numbers and the reading guide | **Start here** |
| `cross_decoding.json` | The same numbers per design and group: per-window `significant_windows` (cluster-corrected), `cluster_p` (despite the name, the **uncorrected per-window** permutation p; its floor is 1/(N_PERM+1), and an isolated window with a small p is not significant), `n_below_ceiling`, `retained`, and `activity_control` (what the control did to that group); also `rt_match` (the matching summary). Arrays longer than 64 values are dropped | Tables and scripts; §13.3 compares runs from it |
| `accuracy_traces.npz` | Label-transfer accuracy per window × repeat. Keys `labeltransfer_<group>_<direction>_true` / `_shuffle` | Re-plotting, your own statistics |
| `tempgen_<name>.npy` | Temporal-generalization matrix, train window × test window, e.g. `tempgen_stability_flexibility_cross_both.npy` | A4(c) |
| `anova_labels.csv` | The per-electrode definition table the groups came from (`anova`, `csv`, `power_traces`) | Which electrodes are in which group |
| `rt_match_<factors>_{balance,contrasts,summary}.csv` | With `RT_MATCH` set: per subject × cell counts and RTs, per-subject RT differences, and their across-subject test, before/after (§6.4). In a `random` run they show the RT costs left in place | The residual RT difference to report (§13.4) |
| `cross_decoding_summary.png` | Overview: A4(0) bar charts (top left, empty without block factors), the `stab_to_flex` trace per group (top right), up to three temporal-generalization matrices (bottom) | A first look |
| `<direction>_<group>__cross_decoding.{png,pdf,eps}` | One figure per group × direction: true accuracy against shuffle, ±1 SD over repeats, bars where the cluster test is significant | The figures to show |

Window times are window **centres**. With 64-sample windows, a window centred at
*t* covers *t* ± 125 ms, so the first window with no pre-stimulus sample is
centred at +0.125 s, and the last window entirely before the stimulus is centred
at −0.125 s.

#### 8.1 What `summary.txt` looks like

The layout, with `…` for numbers (written by `write_summary`):

```
========================================================================
STABILITY vs FLEXIBILITY — A4 CROSS-DECODING
========================================================================
           data_source: real
  electrode_definition: csv
       reference_group: all
 electrode_group_sizes: {'both': …, 'congruency_only': …, 'switch_type_only': …, 'all': …}
electrode_selection_split: off (electrodes selected on the same trials that are decoded)
              rt_match: off | RT-matched; groups=…, within=['subject'], bins=10, … | count-matched random control …
      activity_control: none | remove_mean (…) | mean_only (…)
                window: [0.0, 1.5]s
               …        (every other setting of the run)
------------------------------------------------------------------------
RT matching of the decode trials:                        <- RT_MATCH runs only
   [rt-match] RT-matched, groups=['congruency', 'task_sequence'] n_bins=10 …: kept … of … trials (…%)
   [rt-match]   congruency (i - c): mean per-subject RT difference +… ms before -> +… ms after (t=…, p=…)
   [rt-match]   task_sequence (s - r): …
------------------------------------------------------------------------
A4(0) within-block decoding baseline (Fig 9):            <- 16-cell runs only
   congruency (LWPC) | block 25% incongruent: mean acc=… peak=… sig windows=k/n
   congruency (LWPC) | block 75% incongruent: mean acc=… peak=… sig windows=k/n
      Δ(block) on mean accuracy = …
   switchType (LWPS) | block 25% switch: …
------------------------------------------------------------------------
A4(0b) per-group within-block 2x2 (…diagonal cell is omitted by design):
   [congruency] n_electrodes=… ignored cell=('congruency', 'any block')
       switchType by switch_proportion [25%]: mean acc=… sig=k/nw
------------------------------------------------------------------------
A4(a) label transfer by group. …
   [both] activity control: remove_mean over … subjects (…)   <- ACTIVITY_CONTROL runs only
   [both] stab_to_stab: mean acc=… peak=… (shuffle …) sig windows=k/n
   [both] flex_to_flex: …
   [both] stab_to_flex: mean acc=… peak=… (shuffle …) sig windows=k/n
         vs flex_to_flex, its ceiling: below it in m windows; keeps X% of it above chance
   [both] flex_to_stab: …
         vs stab_to_stab, its ceiling: …
   [congruency_only] …    [switch_type_only] …    [all] …
------------------------------------------------------------------------
A4(c) temporal generalization (Fig 10):
   stability (within) [both]: mean diagonal=… mean off-diagonal=… (… code)
   flexibility (within) [both]: …
   stability->flexibility (cross) [both]: …
========================================================================
Reading: …
```

#### 8.2 Reading it, in order

**Step 1 — the log.** Before any number, check in the slurm `.out`:

- `ROI 'lpfc' pseudopopulation: N channels (sig electrodes)` and the dropped-electrode
  summary above it;
- `decoded electrode groups: both=… congruency_only=… switch_type_only=… all=…`,
  and any `has N electrodes in ROI (< 5); skipping` line;
- `conditions: K decodable cells` (4 or 16) and the `block levels:` line, which
  says whether A4(0)/A4(0b) ran.

**Step 2 — the ceilings.** For each group, `stab_to_stab` and `flex_to_flex` must
have significant windows. A transfer is scored against the ceiling of the
labelling it predicts, and if that ceiling never beats shuffle the transfer is
uninterpretable: `summary.txt` then prints `keeps n/a (its ceiling never beats
shuffle)`. On the csv and unsplit anova routes, a group's ceiling on its **own**
selecting effect is inflated by selection (§4.2); read the `all` group's
ceilings, or the split run's, for honest ones.

**Step 3 — each transfer against its ceiling.** Three numbers per direction:

- `sig windows=k/n` — windows where the transfer beats its refit shuffle null,
  cluster-corrected over time;
- `below it in m windows` — windows where the ceiling beats the transfer
  (the same cluster test, ceiling vs transfer);
- `keeps X% of it above chance` — over the windows where the ceiling beats
  shuffle, (transfer − 0.5) / (ceiling − 0.5). 100% is full transfer, 0% none.

| Ceiling | Transfer | Reading |
|---|---|---|
| never beats shuffle | anything | **Uninterpretable.** There is no code to transfer. |
| beats shuffle | beats shuffle, 0 windows below ceiling, keeps ≈ 100% | **One shared axis.** |
| beats shuffle | beats shuffle, some windows below ceiling, keeps 20–80% | **Partial overlap.** Report the share; do not round it to yes or no. |
| beats shuffle | 0 sig windows, keeps ≈ 0% | **Separable codes** — both contrasts are decodable but along different axes. |
| beats shuffle | reliably below shuffle, keeps < 0% | **Anti-aligned axis** (incongruent with repeat). Check the class ordering first ([Cross-decoding controls](#cross-decoding-controls) §5). |
| – | transfer above its own ceiling, or significant well before stimulus onset | **Artifact** (F3, [Cross-decoding controls](#cross-decoding-controls) §6). |

**Step 4 — both directions.** `stab_to_flex` and `flex_to_stab` should agree. If
only one transfers, the axis learned from the weaker contrast is noisier, so it
generalizes worse; that is a difference in code strength, not in code geometry.

**Step 5 — groups against `all`.** The prediction for a shared code is that
`both` transfers and the `*_only` groups do not. Compare every group with `all`,
which no effect selected. Groups differ in size and accuracy grows with
electrodes, so compare the `keeps X%` shares (each relative to its own
ceiling), not raw accuracies.

**Step 6 — the pre-stimulus windows.** Congruency and switch type cannot be
decoded before the stimulus. Significant windows centred before −0.125 s are an
artifact meter, not a result ([`analysis_guide.md`](analysis_guide.md) §17 records
them in the earlier real runs). `FRAC_TRAIN=0.5` is the quick probe: a cluster
that shrinks with the training set is fold leakage.

**Step 7 — temporal generalization.** Open the matrices, not just the summary
line. Train time is on the y-axis, test time on the x-axis. A bright diagonal
only is a code that changes over time; a bright square is a stable one. The
`cross` matrix shows whether congruency trained at one time predicts switch type
at another. The summary's `sustained/stable` vs `diagonal/phasic` label is a
threshold (mean off-diagonal > 0.55) over the whole matrix, baseline included, and
the matrices have no shuffle null or statistic, so treat them as descriptive.

**Step 8 — the controls.** A single run cannot rule out that the transfer is a
response-time difference or a uniform rise in activity; that takes comparing runs
(§13.3) and the reading rules of §13.4–13.5. In each control run, first check the
`rt_match` / `activity_control` lines say what you intended, and for `RT_MATCH=rt`
that the residual RT difference in the `RT matching` block is near 0.

**Step 9 — within-block (16-cell runs only).** A4(0) lists each block's mean
accuracy and `Δ(block) = high − low`. A negative Δ for congruency means congruency
is less decodable in 75%-incongruent blocks, the direction LWPC predicts. Δ is a
difference of means over **all** windows, baseline included, with no test of its
own, so read it with the traces. A4(0b) lists the per-group cells, with the
circular one named on each group's `ignored cell=` line.

#### 8.3 Things the numbers do not tell you

- **The p-values are optimistic.** The samples of every cluster test are CV
  repeats of the same trials, not subjects, and the pseudopopulation concatenates
  electrodes from different patients whose trials were never recorded together.
  Treat `n_sig_windows` as a within-dataset reliability check.
- **`mean acc` and `peak` in A4(a) are averaged over every window, including the
  second before the stimulus.** They are diluted and are not the post-stimulus
  accuracy. Read the traces, or average `accuracy_traces.npz` over the windows you
  care about.
- **A transfer at chance is only a result next to a ceiling that is not.**
  [Cross-decoding controls](#cross-decoding-controls) §2 is the rule, and
  the task-transfer T1/T3 controls (§9) show the pipeline can carry a code from
  one trial population to another.

---

### 9. The task-transfer positive controls

The same job, run with `ANALYSIS=task_transfer` by its own launcher. It asks
whether the pipeline can transfer a code at all, using factors that should
transfer. It uses the N3b machinery: each contrast is trained in one level of
another factor and tested in the other level, uncentered and centered, on every
electrode of the ROI, with no groups and no ANOVA.

| Design | Decoded | Train → test | Condition set |
|---|---|---|---|
| T1 | task (global vs local) | congruent → incongruent trials (and back) | `stimulus_task_by_congruency_conditions` |
| T2 | task | repeat → switch trials | `stimulus_task_by_switch_type_conditions` |
| T3 | congruency | global-task → local-task trials | `stimulus_task_by_congruency_conditions` |
| T4 | switch type | global-task → local-task trials | `stimulus_task_by_switch_type_conditions` |

```bash
cd dcc_scripts/decoding
bash submit_task_transfer_dcc.sh                    # lpfc, task-significant electrodes
ELECTRODES=all bash submit_task_transfer_dcc.sh     # every lpfc electrode: pairs with a csv A4 run
DATA_SOURCE=synthetic SYNTHETIC_CODE=congruency_specific N_REPEATS=5 WINDOW_SIZE=16 STEP_SIZE=8 \
    bash submit_task_transfer_dcc.sh                # planted: T1 fails, T2 transfers
```

- **Knobs:** `EPOCHS_ROOT_FILE`, `ROI`, `ELECTRODES`, `DATA_SOURCE`,
  `SYNTHETIC_CODE` (`shared`, `congruency_specific`, `carryover`), `WINDOW_SIZE`,
  `STEP_SIZE`, `N_SPLITS`, `N_REPEATS` (balanced resamples), `N_PERM`, `SEED`.
  Run it on the same `ROI`, `ELECTRODES` and epochs file as the A4 run it controls.
- **Output:** `results/<EPOCHS_ROOT_FILE>/task_transfer_<roi>_<electrodes>_w<W>s<S>/pooled_design_conditions/`
  with `summary.txt`, `task_transfer.json`, `task_transfer_traces.npz` and one
  figure per design × centering × direction. The layout and the 2×2 table in
  `summary.txt` are the N3b ones ([N3b block transfer](#n3b-block-transfer)
  Part 3), with the levels printed as congruent/incongruent, repeat/switch,
  global/local.
- **Reading, in order:**
  1. **T1** is the clean control: task learned on congruent trials should keep
     most of its within accuracy on incongruent trials. If it does, a null A4
     transfer is not a pipeline failure.
  2. **The `EFFECT SIZE` line** prints within-level task accuracy (T1) next to
     within-level congruency accuracy (T3). The task cue is drawn with the
     stimulus, so task is large and partly visual; the further apart the two are,
     the less T1 says about a congruency-sized code.
  3. **T3** is the control at congruency's own effect size. Put its `keeps X%`
     beside A4's `stab_to_flex`: congruency transferring across task while failing
     to transfer to switch type is the "separable codes" result.
  4. **T2/T4** carry a confound: on a switch trial the previous task was the
     other one, so leftover previous-task activity flips between the two levels. A
     drop there is expected even with one task code. A `PRE-STIMULUS` line on a
     task design is expected for the same reason; on T3/T4 it is an
     `ARTIFACT FLAG`.

The rationale for these controls is
[Cross-decoding controls](#cross-decoding-controls) §3.5; the block-transfer
counterpart (X1–X3, X2b) is [N3b block transfer](#n3b-block-transfer).

---

### 10. Known issues and gaps

1. **`ANOVA_LABELS_CSV` is silently ignored unless `ELECTRODE_DEFINITION=csv`**
   (§4.2). The §17.5 runbook command in `analysis_guide.md` predates the default
   change and now runs the `anova` route.
2. ~~`ELECTRODE_SELECTION_SPLIT` is not in the folder name or `summary.txt`~~
   Fixed 2026-09-30 (§4.1); earlier split runs still share the unsplit folder.
3. **On the csv route, `ELECTRODES` and the window are ignored but still name the
   folder** (§4.2).
4. **`power_traces` needs `CONTRAST_MODE=proportion`** by hand (§4.4).
5. **A4(a) `mean acc` includes the baseline** (§8.3). There is no post-stimulus
   summary for label transfer, unlike N3b's table.
6. **Temporal generalization has no null** (§8.2 step 7).
7. **Pre-stimulus significance in selected groups.** The earlier real A4 runs
   showed cross-decode clusters before stimulus onset
   ([`analysis_guide.md`](analysis_guide.md) §17, caveat). In the 2026-09-29 runs
   they appear in the main-effect-selected `both` group (anova and csv routes) and
   in the csv run's 398-electrode `all`, but not in the 171 task-significant
   electrodes of the `none` run. The likeliest route is block structure (pooled
   cells are confounded with block type, and the one-way main-effect ANOVA has no
   block term); [Cross-decoding controls](#cross-decoding-controls) §6 has the
   diagnosis. Any transfer needs its pre-stimulus windows reported next to it.
8. **Resamples are not subjects** (§8.3); there is no leave-one-subject-out for A4.
9. **The transfer sits where RT differs.** In the 2026-09-29 `none` run (171
   task-significant lPFC electrodes) both transfers are significant only from the
   window centred at +0.62 s (median correct RT ~1.17 s), and incongruent and
   switch trials are both slower. Report the RT-matched run (§6.4) against its
   `random` control, and the response-locked run (§5), next to it (§13).
10. **No formal test between runs.** Whether a control "keeps a similar share" is
    judged against the spread across seeds (§13.4), not tested; there is no
    permutation test of one run's transfer against another's.
11. **`ACTIVITY_CONTROL` and `RT_MATCH` are wired into this job only.** The
    functions work on any epochs structure or ROI arrays, but the ordinary decoding
    and power-trace jobs do not call them yet ([RT matching](#rt-matching),
    [Overall-activity control](#overall-activity-control)).

---

### 11. Checklist

```
Running (§13.2 has the commands)
[ ] export EPOCHS_ROOT_FILE once; set every other knob per command, not with export
[ ] synthetic dry runs: shared transfers, orthogonal does not
[ ] ELECTRODE_DEFINITION=none                               (the ungrouped reference run)
[ ] RT_MATCH=rt and RT_MATCH=random, same SEED               (RT control and its trial-count control)
[ ] CONDITIONS=response_main_effect_conditions + a Response_ epochs file   (response-locked)
[ ] ACTIVITY_CONTROL=remove_mean and =mean_only              (pattern vs overall activity)
[ ] RT_MATCH=rt/random + ACTIVITY_CONTROL=remove_mean        (both controls at once)
[ ] bash submit_task_transfer_dcc.sh                         (positive controls, same electrodes)
[ ] SEED=1, SEED=2 with their own SAVE_DIR                   (noise floor)
[ ] groups only if claimed: ELECTRODE_SELECTION_SPLIT=true   (clean ceilings; own folder)
    (or ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=... for the saved table; check the echo lines)

Reading (§8.2, §13.3-13.5)
[ ] log: group sizes, skipped groups, cells, block levels, rt-match / activity-control lines
[ ] every ceiling (stab_to_stab, flex_to_flex) beats shuffle   <- else stop
[ ] no significant windows centred before -0.125 s             <- else stop
[ ] per group and direction: sig windows, windows below ceiling, share kept
[ ] both directions agree (or the asymmetry is reported)
[ ] RT-matched: residual RT ~0; rt vs random share kept, against the seed spread
[ ] remove_mean / mean_only share kept vs baseline
[ ] T1 transfers; T3 share set beside A4's share
[ ] groups compared by share kept, against 'all', split run only
[ ] fill in the report block (§13.7)
```

---

### 12. Tests

Run outside the cluster with `pip install -e . pytest` then
`python -m pytest -o addopts="" tests/analysis/decoding -q`.

| Test file | Pins |
|---|---|
| `test_cross_decoding.py` | the two label vectors, stratification on the condition cell, padding handling, equal priors, shared-transfers/orthogonal-does-not, `frac_train`, temporal generalization |
| `test_cross_decoding_condition_scheme.py` | class definitions read from declared levels, crossed vs confounded sets, 4- vs 16-cell sets, the `none` route decoding only the loaded electrodes |
| `test_cross_decoding_electrode_groups.py` | channel keys, disjoint groups, the reference group, the csv `union` and raw-correction behaviour |
| `test_cross_decoding_circularity.py` | which within-block cell each group double-dips on |
| `test_cross_decoding_runner.py` | contrast mode read from the folder, mode/effect mismatches refused, csv ignored off its route, folder names (incl. the split and RT-matching tags) |
| `test_cross_decoding_rt_match.py` | RT matching in the job: runs after the split, writes its report, keeps the same counts for `rt` and `random`, refuses a subject left empty and synthetic data, lands in `summary.txt` |
| `../utils/test_rt_matching.py` | the RT-matching util itself: planted RT costs removed, equal/proportional balance, unusable rows, reproducibility, the random control, the epochs adapter |
| `test_activity_control.py` | the two transforms (per-subject, NaN-safe, single-electrode subjects), and the planted answers: a shared gain dies under `remove_mean` and survives `mean_only`; a shared pattern does the reverse |
| `test_cross_decoding_activity_control.py` | the job feeds every design the transformed arrays, per group, in A4 and the transfer analyses, and reports it in `summary.txt` |
| `test_task_transfer.py` | T1–T4 on real condition sets, planted synthetic answers, end to end |

---

### 13. The control battery: run it, read it, report it

A transfer above chance says that congruency and switch type share *something*.
Before it can be reported as a shared code, the obvious cheaper explanations have
to be ruled out. Each has a control built into this job. This section is the one
place that says what to run, in what order, what to compare with what, and what
each outcome lets you say. The single controls are documented in §6.4
(RT matching), §6.5 (overall activity), §5 (response-locked) and §9 (task
transfer); the methods are in [RT matching](#rt-matching) and
[Overall-activity control](#overall-activity-control).

Run everything on the ungrouped `none` route first: it is cheap, and no
selection on either effect can inflate it. Do the groups (`both` vs `*_only`)
only if they go in the paper (§13.6).

#### 13.1 What each control asks

| Run | The alternative it tests | Knob | Read it against | Folder (under `results/<EPOCHS_ROOT_FILE>/`) |
|---|---|---|---|---|
| **Baseline** | – (this is the result) | `ELECTRODE_DEFINITION=none` | its own ceilings | `cross_decoding_lpfc_sig_none/` |
| **RT-matched** | *Shared latency.* Incongruent and switch trials are both ~200 ms slower, so late stimulus-locked windows can separate them by time-to-response alone. | `RT_MATCH=rt` | the `random` run, **not** the baseline | `…_none_rtmatch10/` |
| **RT control** | Matching keeps about half the trials; a weaker transfer could just be less data. | `RT_MATCH=random` | the `rt` run | `…_none_rtrandom10/` |
| **Response-locked** | The same question from the other side: align every trial to its response. | `CONDITIONS=response_main_effect_conditions` + a `Response_…` epochs file | where in the epoch the transfer sits | `cross_decoding_lpfc_sig_none/`, under the Response file's folder instead |
| **Pattern only** | *Shared gain.* Hard trials raise HG on most electrodes at once; a decoder can learn "activity is up". | `ACTIVITY_CONTROL=remove_mean` | the baseline, by share kept | `…_none_remove_mean/` |
| **Mean only** | The uniform part alone. | `ACTIVITY_CONTROL=mean_only` | the baseline, by share kept | `…_none_mean_only/` |
| **Both controls** | Latency and gain together (after the two above are read). | `RT_MATCH=rt ACTIVITY_CONTROL=remove_mean` (and `random`) | each other | `…_none_rtmatch10_remove_mean/` |
| **Positive controls** | *A broken pipeline.* If a control kills the transfer, that only means something if the pipeline can carry a code across trial populations at all. | `bash submit_task_transfer_dcc.sh` | T1/T3 share kept beside A4's | `task_transfer_lpfc_sig_w64s16/` |
| **Seeds** | Differences between runs are only the random trial pairing of the pseudopopulation. | `SEED=1`, `SEED=2` (+ `SAVE_DIR`) | the spread of share kept | `…_none_seed<n>/` (set by you) |

Each A4 folder has a `<CONDITIONS>/` subfolder (`stimulus_main_effect_conditions/`,
or `response_main_effect_conditions/` for the response-locked run); the task
transfer's is `pooled_design_conditions/`. `SEED` is not in the folder name, so
seed runs need their own `SAVE_DIR` (§13.2). Every other row gets its own folder
automatically.

#### 13.2 Running it

```bash
REPO=/hpc/home/$USER/coganlab/$USER/GlobalLocal
cd $REPO && git pull                    # RT matching / ACTIVITY_CONTROL are from 2026-09-30
cd dcc_scripts/decoding
export EPOCHS_ROOT_FILE=Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
RESP_FILE=Response_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
A4=submit_stability_flexibility_cross_decoding_dcc.sh

# 1. baseline (the 2026-09-29 run; rerun it if the decoding code changed since)
ELECTRODE_DEFINITION=none bash $A4

# 2. RT: the matched run and its trial-count control, same SEED
ELECTRODE_DEFINITION=none RT_MATCH=rt     TEMPGEN_GROUPS= bash $A4
ELECTRODE_DEFINITION=none RT_MATCH=random TEMPGEN_GROUPS= bash $A4

# 3. response-locked (check that $RESP_FILE exists first)
ELECTRODE_DEFINITION=none CONDITIONS=response_main_effect_conditions \
    EPOCHS_ROOT_FILE=$RESP_FILE bash $A4

# 4. overall activity
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=remove_mean TEMPGEN_GROUPS= bash $A4
ELECTRODE_DEFINITION=none ACTIVITY_CONTROL=mean_only   TEMPGEN_GROUPS= bash $A4

# 5. both controls together (worth it once 2 and 4 are read)
ELECTRODE_DEFINITION=none RT_MATCH=rt     ACTIVITY_CONTROL=remove_mean TEMPGEN_GROUPS= bash $A4
ELECTRODE_DEFINITION=none RT_MATCH=random ACTIVITY_CONTROL=remove_mean TEMPGEN_GROUPS= bash $A4

# 6. positive controls, same electrodes and epochs file as the A4 runs
bash submit_task_transfer_dcc.sh

# 7. seeds: SEED is not in the folder name, so give each its own SAVE_DIR
for s in 1 2; do
  ELECTRODE_DEFINITION=none SEED=$s TEMPGEN_GROUPS= \
  SAVE_DIR=$PWD/results/$EPOCHS_ROOT_FILE/cross_decoding_lpfc_sig_none_seed$s/stimulus_main_effect_conditions \
      bash $A4
done
```

- **Set control knobs per command, never with `export`.** Both launchers submit
  with `--export=ALL`, so an exported `RT_MATCH` or `ACTIVITY_CONTROL` would ride
  along into every later job in that shell, including the task-transfer
  controls. `EPOCHS_ROOT_FILE` is the one variable meant to be exported.
- **`TEMPGEN_GROUPS=` (empty)** skips temporal generalization, which on the `none`
  route otherwise runs on `all` and costs 3 × 37² decodes. The controls do not
  need it; the baseline and response-locked runs keep it.
- **Check the echo lines** before walking away: `definition=none`, and
  `rt_match=` / `activity_control=` as intended.
- **Cost:** one group, so each job is the cheapest A4 run (§7.4). RT-matched runs
  decode about half the trials and are faster still.
- **RT matching is refused** on synthetic data and stops the job if a subject would
  lose every trial (lower `RT_MATCH_BINS`, §6.4).

#### 13.3 Comparing the runs

`summary.txt` in each folder has the numbers for that run. To see the runs side by
side, point this at the save directories (it reads `cross_decoding.json` and, if
present, `accuracy_traces.npz`):

```python
import json, os
import numpy as np

RUNS = {  # label -> save directory (the folder holding cross_decoding.json)
    'baseline':    'results/<EPOCHS_ROOT_FILE>/cross_decoding_lpfc_sig_none/stimulus_main_effect_conditions',
    'rt':          'results/<EPOCHS_ROOT_FILE>/cross_decoding_lpfc_sig_none_rtmatch10/stimulus_main_effect_conditions',
    'random':      'results/<EPOCHS_ROOT_FILE>/cross_decoding_lpfc_sig_none_rtrandom10/stimulus_main_effect_conditions',
    'remove_mean': 'results/<EPOCHS_ROOT_FILE>/cross_decoding_lpfc_sig_none_remove_mean/stimulus_main_effect_conditions',
    'mean_only':   'results/<EPOCHS_ROOT_FILE>/cross_decoding_lpfc_sig_none_mean_only/stimulus_main_effect_conditions',
}
WINDOW, STEP, SRATE, FIRST = 64, 16, 256, -1.0      # WINDOW_SIZE, STEP_SIZE, SAMPLING_RATE, FIRST_TIME_POINT

print(f"{'run':10} {'group':17} {'direction':13} {'n':>4} {'sig':>3} {'from':>6} "
      f"{'pre':>3} {'peak':>5} {'post':>5} {'kept':>5} {'below':>5}")
for label, folder in RUNS.items():
    runs = json.load(open(os.path.join(folder, 'cross_decoding.json')))['label_transfer']
    npz = os.path.join(folder, 'accuracy_traces.npz')
    traces = np.load(npz) if os.path.exists(npz) else None
    for group, directions in runs.items():
        for direction, r in directions.items():
            sig = np.asarray(r['significant_windows'], bool)
            t = FIRST + (np.arange(sig.size) * STEP + WINDOW / 2) / SRATE   # window centres
            post = ''
            if traces is not None:                      # mean over windows fully after 0 s
                acc = traces[f'labeltransfer_{group}_{direction}_true'].mean(axis=1)
                post = f"{acc[t >= WINDOW / 2 / SRATE].mean():.3f}"
            kept = r.get('retained')
            print(f"{label:10} {group:17} {direction:13} {r['n_channels']:4d} {sig.sum():3d} "
                  f"{(f'{t[sig][0]:+.2f}' if sig.any() else '-'):>6} "
                  f"{int(sig[t <= -WINDOW / 2 / SRATE].sum()):3d} {r['peak_accuracy']:5.3f} "
                  f"{post:>5} {('' if kept is None else f'{kept:.0%}'):>5} "
                  f"{r.get('n_below_ceiling', ''):>5}")
```

| Column | Meaning |
|---|---|
| `n` | electrodes decoded (for `mean_only`, the features are one per subject; the summary says how many) |
| `sig`, `from` | significant windows (cluster-corrected against the refit shuffle) and the centre of the first one |
| `pre` | significant windows centred at or before −0.125 s, i.e. entirely before the stimulus: the artifact meter. On the response-locked run these are pre-*response* windows and are not artifacts. |
| `peak` | best window's accuracy |
| `post` | mean accuracy over the windows entirely after 0 s. Use this, not `mean_accuracy` (which averages in the baseline second). |
| `kept` | share of the ceiling's above-chance accuracy the transfer keeps (transfers only) |
| `below` | windows where the ceiling beats the transfer |

The response-locked run has its own time axis (0 = response); keep it in a
separate `RUNS` dict or read its column `from` as time-to-response.

#### 13.4 Reading each control

**First, the prerequisites, in every run:** both ceilings (`stab_to_stab`,
`flex_to_flex`) have significant windows, and `pre` is 0. A run that fails either
cannot say anything about transfer (§8.2 steps 2 and 6).

**RT matching.** Check the matching worked, then compare `rt` with `random`.

1. `rt_match_congruency_task_sequence_summary.csv` in the `rt` folder: the
   `mean_after` column should be near 0 with `p_after` > .05 for both factors (the
   behavioural data give +2 ms and +1 ms at 10 bins). If not, rerun with more bins
   (`RT_MATCH_BINS=15`). The `random` folder's CSV should still show the full RT
   costs; that is what makes it the control.
2. Compare the two runs' transfers:

| `rt` vs `random` | Reading |
|---|---|
| similar share kept and similar `sig` | **Not a shared latency.** The transfer survives when the compared trials are equally fast. |
| `rt` clearly lower, still above chance | **Partly latency.** Report the share that survives. |
| `rt` at chance, `random` transfers | **The late transfer was the RT difference.** The codes are separable once RT is equated. |
| `random` loses the transfer too | **Inconclusive:** halving the trials cost the power. Try `RT_MATCH_BINS=5` (keeps more, matches less tightly) or more electrodes. |

"Similar" has no formal test: judge it against the seed spread (§13.1, the
**Seeds** row). A difference between `rt` and `random` smaller than the difference
between two seeds of the same run is not a difference.

**Response-locked.** Descriptive; read it together with the RT result.

| Where the response-locked transfer sits | Reading |
|---|---|
| only early in the epoch (far before the response), fading towards it | the latency signature: at a fixed time before the response, slow trials are further past their stimulus |
| around the response itself, with both ceilings there | something shared in the response period that is not a timing shift, e.g. effort or response caution |
| nowhere | the shared component is locked to the stimulus, not to the response |

The electrodes come from the response file's own significance list, so they are
not exactly the baseline's 171 (the log prints the count). Windows after the
response include feedback and the next trial's preparation.

**Overall activity.** Compare each run's `kept` with the baseline's, per
direction (the ceilings change too, so do not compare raw accuracy):

| `remove_mean` | `mean_only` | Reading |
|---|---|---|
| keeps a similar share | little or no transfer | **A shared pattern**, not a uniform rise in activity |
| transfer gone, ceilings still above chance | transfers | **A shared gain**: both effects raise overall HG; their specific patterns differ |
| reduced but present | transfers | **Both.** Report the share under each. |
| ceilings gone too | – | the contrasts themselves are mostly overall activity; this control cannot speak to the transfer |

`remove_mean` removes only a shift shared by *all* of a subject's decoded
electrodes; a rise on a subset of them still reads as "pattern"
([Overall-activity control](#overall-activity-control)).

**Positive controls.** T1 (task learned on congruent trials, tested on
incongruent ones) should keep most of its ceiling; if it does, a transfer that a
control kills was killed by the control, not by the pipeline. Put T3's share
(congruency across task, at congruency's own effect size) next to A4's (§9).

**Seeds.** The spread of `kept` across `SEED=0,1,2` is the noise floor for every
comparison above. The grouped runs of 2026-09-29 suggest it is not small: `both`
switch → congruency kept 55% in the `anova` run and 34% in the `csv` run, with the
same group sizes and probably the same electrodes (the csv route loads all 398
electrodes, which changes how trials are paired into pseudo-trials).

#### 13.5 Putting it together: what you can say

| RT (`rt` vs `random`) | Activity (`remove_mean`) | What the transfer supports |
|---|---|---|
| survives | survives | **Congruency and switch type share part of their lPFC code**, not explained by response time or by a uniform rise in activity. The strongest claim. Confirm with run 5 (both controls at once). |
| survives | gone, `mean_only` transfers | They share a **uniform activity increase** (a common difficulty/effort signal) that is not a timing artifact; their specific patterns are separable. |
| gone | – | The late transfer is the **RT difference**; once trials are equally fast the codes are separable. |
| partial | partial | Report the share kept under each control; do not round it to yes or no. |

Whatever the row, also report: the early separability (in the baseline, switch
type is decodable from ~0 s and congruency from ~0.3 s, but nothing transfers
until the window covering 0.5–0.75 s), the direction asymmetry (congruency →
switch keeps more than switch → congruency), and the pre-stimulus windows.

#### 13.6 The groups (`both` vs `*_only`)

Only if the paper makes a claim about the groups. Two extra requirements on top of
§13.4:

- **Held-out selection.** On the csv and unsplit `anova` routes the groups were
  selected on the trials they are decoded on, which inflates each group's own
  ceilings and can inflate the transfer too (electrodes picked because both
  effects push HG the same way favour a shared axis). Run
  `ELECTRODE_SELECTION_SPLIT=true bash $A4` (folder `…_anova_condition_none_split0.3s0/`;
  add `RT_MATCH` / `ACTIVITY_CONTROL` as above). Split and RT matching together
  leave about a third of the trials.
- **The pre-stimulus meter.** In the 2026-09-29 runs, `both` and the csv run's
  398-electrode `all` had significant pre-stimulus windows; the 171-electrode `all`
  had none. [Cross-decoding controls](#cross-decoding-controls) §6 has the
  diagnosis. A group with `pre` > 0 cannot be reported.

Compare groups by `kept`, never by raw accuracy, and against the seed spread.

#### 13.7 What to report, and where things stand

The block to fill in per transfer direction (numbers so far: the 2026-09-29
`none` run, 171 task-significant lPFC electrodes, pooled 2×2, 250 ms windows):

```
electrodes          lpfc, task-significant (ELECTRODES=sig), 171 electrodes / __ subjects
condition set       stimulus_main_effect_conditions (congruency x switch type, blocks pooled)
decoding            PCA(80%) -> LDA, 5-fold x 10 repeats; 250 ms windows, 62.5 ms steps;
                    cluster-corrected against a refit label-shuffle null (500 permutations)
ceilings            stab_to_stab: sig from +0.31 s, peak 0.618 @ 1.31 s
                    flex_to_flex: sig from  0.00 s, peak 0.662 @ 1.12 s
transfer            stab_to_flex: sig from +0.62 s, peak 0.640 @ 1.12 s, keeps 64%, below ceiling 12 windows
                    flex_to_stab: sig from +0.62 s, peak 0.576 @ 0.94 s, keeps 39%, below ceiling 18 windows
pre-stimulus        none
RT matching         residual RT i-c __ ms (p __), s-r __ ms (p __); keeps __% (rt) vs __% (random)
response-locked     __
activity control    keeps __% (remove_mean), __% (mean_only)
both controls       keeps __% (rt + remove_mean) vs __% (random + remove_mean)
positive controls   T1 keeps __%; T3 keeps __%
seeds               keeps __ +/- __ over __ seeds
```

**Done:** the baseline (`none`), and the in-job `anova` and `csv` group runs
(2026-09-29; §13.6 for their caveats).
**To run:** everything else in §13.2.

Two things the controls do not fix, to state in the methods: the cluster tests'
samples are CV repeats of one pseudopopulation, not subjects (§8.3), and there is
no leave-one-subject-out for A4 (§10).

---

### Related documents

- [`analysis_guide.md`](analysis_guide.md) §17 — A4's design and the reasons behind it; §17.5 — the runbook for the main-effect populations and task controls
- [Cross-decoding controls](#cross-decoding-controls) — diagnosing a transfer that did not work, and the report block to print with every transfer
- [RT matching](#rt-matching) and [Overall-activity control](#overall-activity-control) — the methods behind §6.4, §6.5 and §13
- [N3b block transfer](#n3b-block-transfer) — block transfer, the third mode of this job
- [Decoding job](#decoding-job) — the ordinary decoding job, run in the same populations
- [`analysis_plans.md` › Closing figure plan](analysis_plans.md#closing-figure-plan) — where the cross-decoding results sit in the paper
- [`analysis_plans.md` › Concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan) §4 — the plan this implements

---

## N3b block transfer

*N3b: block-transfer cross-decoding*

**Status:** implemented and validated on synthetic data; not yet run on real data.
**Spec:** `analysis_plans.md` › Concurrent-regulation plan §4. Controls and decision rules: [Cross-decoding controls](#cross-decoding-controls).
**Run it:** `cd dcc_scripts/decoding && bash submit_block_transfer_dcc.sh` (Part 3).

N3b trains a classifier in one kind of block and tests it in another:

| Design | Decoded contrast | Train → test | Role |
|---|---|---|---|
| **X1** | congruency | 25%-incongruent blocks → 75%-incongruent blocks | primary (LWPC) |
| **X2** | switch type | 25%-switch blocks → 75%-switch blocks | primary (LWPS) |
| **X3** | congruency | 25%-switch blocks → 75%-switch blocks | positive control for X1 |
| **X2b** | switch type | 25%-incongruent blocks → 75%-incongruent blocks | reciprocal control for X2 |

Every design runs in both directions.

The existing decoder (`Decoder.cv_cm_jim_window_shuffle`) always cuts its train and test sets out of one pool of trials with random folds. It has no way to say "train on these trials, test on those". N3b needs exactly that.

**Goals:**
- the smallest change that reuses the existing pipeline;
- run on all significant electrodes of an ROI (LPFC by default), with no electrode groups;
- explain the design choices well enough that the code can be maintained by someone who didn't write it.

This doc has three parts:
- **Part 1:** the concepts, as answers to the design questions;
- **Part 2:** what was built;
- **Part 3:** how to run it and read the output.

The same change also fixed the A4 padding bug (see 1.1 and "Changes to A4" at the end), and it removed dead code.

### The data this has to work with

From `combinedData.csv`: each subject has 4 physical blocks of 112 trials, one block per type.

| Block | Incongruent | Switch | Accurate incongruent / congruent trials per subject (mean) |
|---|---|---|---|
| A | 75% | 25% | 71 / 26 |
| B | 75% | 75% | 66 / 24 |
| C | 25% | 25% | 22 / 79 |
| D | 25% | 75% | 20 / 73 |

Inside a block, the classes run about **3:1, and the majority class flips between the 25% and 75% levels**. Most of the design choices below follow from that.

---

### Part 1: concepts

#### 1.1 Stratifying vs balancing

- **Stratifying a split:** dealing the trials into folds so that every fold is a small copy of the whole set. If 25% of the trials are incongruent, every fold is about 25% incongruent. It never adds or removes a trial. It only stops a random split from, say, putting most of the rare trials in one fold.
- **`stratify_labels`** (in `cv_cm_jim_window_shuffle`) is the label the folds are kept proportional on. The default is the training labels. A4 passes `strata`, which is the index of the condition each trial came from (0–15 for the 16 cells). So every fold has the same mix of all 16 cells, and therefore of congruency, switch type and both proportions at once. That matters in label transfer, which scores on switch type: folds balanced only on congruency could come out lopsided on switch type.
- **Balancing is a different thing.** It changes the data: you subsample so that groups have equal counts.
  - **A4 subsamples training data instead of using mixup.** Pure-padding rows are removed up front; within each fold, incomplete training pseudo-trials are removed and the remaining classes are subsampled to equal sizes.
  - Partial test rows are still filled with independent noise, as in the normal decoder. This preserves test observations without synthesizing training signal.
- **N3b needs both:** balancing, then stratified folds.

#### 1.2 What to balance

- **Balancing is needed because of the 3:1 ratio inside a block.** LDA's default priors are the training class frequencies. For a weak effect, a 3:1 prior pushes nearly every prediction to the majority class. In the simple 1-D case with d′ = 0.5, balanced accuracy is 0.51 instead of 0.60, even though the signal is there. In X1 the majority also flips between training and testing.
- **Use a design-specific pooled 2×2 condition set and balance its four contrast × transfer-level cells.** X1 uses `stimulus_lwpc_conditions`, X2 uses `stimulus_lwps_conditions`, X3 uses `stimulus_congruency_by_switch_proportion_conditions`, and X2b uses `stimulus_switch_type_by_incongruent_proportion_conditions`. The irrelevant block factor is pooled rather than split into 16 cells, retaining more trials.
- **This is the higher-trial-count version of N3b.** The four-group version keeps about 42 trials per class per level per subject rather than about 40 when balancing all eight full-factor cells.
- **X1 and X3 use their respective pooled condition definitions.** They cover the same physical trial population while grouping it by different transfer factors.
- **Tradeoff for X3:** pooling incongruent proportion means congruency can correlate with the physical A/C or B/D block mix. Treat X3 as a positive control with that caveat; the requested gain in retained trials comes from not balancing the nuisance factor's eight full-factor cells.
- **Do not additionally balance on switch type** in X1 (or on congruency in X2). It is 25/75 inside a block by design, so doing so would cut the data in half. Folds are stratified on the four pooled condition cells.

#### 1.3 Trial loss

- **What gets dropped is only the extra majority trials.** Each repeat draws a new random balanced subsample, so over 10 repeats almost every trial gets used.
- **The real limit is the minority class** (about 20 incongruent trials per subject per 25%-incongruent block), and no method removes it.
- The per-channel minimum in `subsample_to_min_trials_per_condition` (the "~22/class" in `analysis_plans.md` › Simplification plan §1.2) belongs to the ordinary decoding job. A4 and N3b never call it.

#### 1.4 Why there are still folds

- **"Train on all the 25% trials, test on all the 75% trials" is valid for the transfer number on its own.** The two sets share no trials, so there is no double-dipping.
- **Folds are needed for three other reasons:**
  - **The ceiling:** a transfer accuracy only means something next to the within-block accuracy ([Cross-decoding controls](#cross-decoding-controls) §2), and the within-block accuracy has to be cross-validated.
  - **Matching:** if the transfer classifier trains on 100% of the 25% trials but the ceiling classifier trains on 80%, the two numbers aren't comparable.
  - **Repeats:** you need a spread of values for the shuffle null and error bars.
- **The design:** cut folds only inside the training level. Each fold's classifier is scored twice: on its held-out 25% trials (the ceiling), and on all the 75% trials (the transfer). Same folds and same seed, so the ceiling and the transfer come from literally the same classifier.
- **Which ceiling to compare against:** compare transfer 25→75 with **within-75**, because both are scored on the same test trials.
  - The congruency code can simply be weaker in 75%-incongruent blocks (that is the LWPC effect).
  - In that case 25→75 drops while the axis is unchanged.
  - A real change of axis makes **both directions** fall short of their test block's ceiling.
  - A drop in only one direction means the training block's code is weaker, not that the axis changed.
  - The job reports the full 2×2 table (train level × test level).

#### 1.5 Centering

- **Centering subtracts one vector per block level.** It is the same vector for every trial in that level, congruent and incongruent alike. It moves the whole cloud, and cannot rotate the direction that separates C from I inside the block.

```
   same axis, whole block shifted        different axis
   25%:  C●   ●I                         25%:  C●   ●I
   75%:            C●   ●I               75%:       ●I
                                                    ●C
   centering lines them up -> transfers  centering can't fix it -> still fails
```

- **If C25/I25 and C75/I75 separate along different directions** (the reconfiguration hypothesis), that survives centering and X1 still fails.
- **What centering removes is the tonic shift of the whole block.** Examples: all high-gamma higher in 75% blocks, or the pooled-baseline artifact (`analysis_plans.md` › Simplification plan §1.4). A shift like that can make transfer fail even when the axis is identical, which is why an uncentered null can't be read on its own.
- The tonic effect itself is not lost. It is simply a different claim, and X5 or the univariate block effect measures it.
- **Within-level accuracies are unchanged by centering.** The same vector is subtracted from both the training and the test trials. That makes them a built-in check.
- **Pitfall: balance first, then center.**
  - The raw mean of a 3:1 block sits a quarter of the C–I distance away from the class midpoint, on the majority side.
  - The majority flips between levels, so centering on raw means shifts the two levels half the C–I difference apart, exactly along the decoding axis.
  - That fakes "X1 and X2 fail, X3 transfers", which is the pattern the analysis is looking for.

#### 1.6 Reading the results

Read these centered and uncentered, in both directions:

| within (test level) | transfer, uncentered | transfer, centered | meaning |
|---|---|---|---|
| at chance | – | – | Can't interpret: there was nothing to transfer |
| above | ≈ within | ≈ within | Same code |
| above | < within | ≈ within | Same axis; the blocks differ by a tonic shift |
| above | < within in both directions | < within in both directions | Block context reorganizes the code. Only counts if X3, on the same trials, transfers |

Any pre-stimulus windows that come out significant are an artifact flag ([Cross-decoding controls](#cross-decoding-controls) §6).

#### 1.7 Electrodes: all significant electrodes of an ROI

- **The N3b job** (`dcc_scripts/decoding/submit_block_transfer_dcc.sh`) defaults to `ROI=lpfc ELECTRODES=sig`. It runs no CSV, power-trace or ANOVA step and forms no electrode groups. `ELECTRODES=all` keeps every electrode in the ROI.
- **What "sig" means:** the electrode's high-gamma during the stimulus beats its pre-stimulus baseline (a per-electrode cluster test done at epoching). It's read from `sig_chans_<subject>_<EPOCHS_ROOT_FILE>.json`. It says nothing about congruency or switching, so there is no double-dipping with N3b.
- **`EPOCHS_ROOT_FILE` decides which significance file is used.** The A4 submit default has no `_filterbank_hilbert`; the ANOVA-label CSV folders were computed with it. Set it on purpose.
- **Gotcha in the existing A4 job:** with its default `ELECTRODE_DEFINITION=csv`, it loads every electrode in the ROI, significant or not (`_build_roi_arrays` in `stability_flexibility_cross_decoding_dcc.py`), even though output folders say `sig`.

#### 1.8 How to approach changing this codebase

1. **Follow one call path, not files.**
   - `submit_*.sh` → `sbatch_*.sh` → `run_*_dcc.py` (turns environment variables into `args`) → `*_dcc.py main(args)` (loads data → ROI arrays → loops over designs)
   - → `cross_decoding.py` (arrays → label vectors) → `decoder.py` (folds → fit → confusion matrices)
   - → `accuracy_stats.py` (confusion matrices → accuracy → cluster test vs shuffle) → saving and plots
2. **Find the one step that differs.** For N3b, that is which trials train and which test. Everything else is reused.
3. **Add the new behavior as an optional argument that is off by default.** Every existing call stays identical, and the existing tests prove it.
4. **Test on synthetic data with a planted answer before touching real data.** Plant the confounds you're worried about too (3:1 classes, block offsets).

---

### Part 2: how it is built

#### The call path

Everything below the job function is the ordinary cross-decoding pipeline; the new pieces are marked **new**.

```
submit_block_transfer_dcc.sh                       new: ANALYSIS=block_transfer, ROI, ELECTRODES
 └ sbatch_stability_flexibility_cross_decoding_dcc.sh
    └ run_stability_flexibility_cross_decoding_dcc.py     environment variables -> args
       └ stability_flexibility_cross_decoding_dcc.main(args)
          └ run_block_transfer_job(args)                  new: loads the ROI, loops all four designs x centering
             ├ _build_roi_arrays                          the ROI pseudopopulation (sig or all electrodes)
             ├ block_transfer.run_block_transfer          new: balance -> center -> the 2x2
             │  ├ cross_decoding.build_cross_decoding_arrays   remove pure-padding rows
             │  ├ cross_decoding.make_decoder                  PCA -> LDA with equal priors
             │  └ Decoder.cv_cm_jim_window_shuffle(test_only=...)   folds -> confusion matrices
             ├ _summarise                                 accuracy + cluster test vs the shuffle null
             └ summary.txt, block_transfer.json, block_transfer_traces.npz, figures
```

#### What changed, file by file

- **`src/analysis/decoding/decoder.py`: `test_only`.** A new optional argument of `cv_cm_jim_window_shuffle`: a True/False flag per trial. Flagged trials are never trained on. The folds are cut from the unflagged trials only, and every fold's classifier is scored on all the flagged trials. With `test_only=None` (the default) the function behaves exactly as before. The whole change is the few lines that pick `train_idx` and `test_idx` in the fold loop.
- **`src/analysis/decoding/cross_decoding.py`:**
  - `build_cross_decoding_arrays` drops pure-padding rows through `_drop_padding_rows`. Fold preparation subsamples incomplete training rows and balances the surviving classes; test gaps retain the independent-noise fill.
  - `make_decoder` builds the Decoder with equal LDA priors (see 1.2). It passes them as `clf_params`, which also stops `ieeg` printing "No initial parameters" on every fit.
  - `synthetic_roi_labeled_arrays` gains `block_code`, `block_offset` and `design_proportions`, so tests can plant a block-specific code, a tonic block shift and 3:1 cells. Its default output is byte-identical to before.
- **`src/analysis/decoding/block_transfer.py` (new, about 150 lines, reads top to bottom):**
  - `prepare`: trials, labels, each trial's block level and balance group.
  - `balanced_subsample`: equal trials from every group.
  - `center_levels`: subtract each level's mean trial.
  - `run_block_transfer`: the resample loop that produces the 2x2.
- **`dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py`:** `run_block_transfer_job` and its three small helpers (`_summarise_block_transfer`, `_plot_block_transfer`, `_write_block_transfer_summary`). `main()` hands off to it when `args.analysis == 'block_transfer'`, so none of the A4 electrode-group code runs.
- **`dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py`:**
  - reads `ANALYSIS` (default `a4`, so existing submissions are unchanged);
  - gives block-transfer runs their own results folder;
  - always puts synthetic runs under `results/synthetic_<code>/`, so a dry run can no longer overwrite a real run's folder.
- **`dcc_scripts/decoding/submit_block_transfer_dcc.sh` (new):** one job, readable on one screen.
- **Removed dead code:**
  - `decoder.py`: a commented-out older copy of `cv_cm_jim_window_shuffle`, `fit_predict`, `cv_cm_return_scores`, `calculate_scores`, and unused imports. Its comments no longer claim a StandardScaler; the `ieeg` pipeline is PCA → LDA.
  - `cross_decoding.py`: `CONTRASTS`, `resolve_contrast` and their helpers. They were used only by one test, and they coded incongruent as 1 while the rest of the module codes it as 0.

#### Tests

`tests/analysis/decoding/test_block_transfer.py`:

- **The decoder:** `test_only` trials are never trained on and are always the whole test set, and a transfer trains on exactly the folds of the matching within-level decode.
- **Balancing and centering (no `ieeg` needed):** the four design-specific balance groups, equal counts after balancing, and the balance-then-center order. The class midpoint lands at 0; centering the raw 3:1 trials would put it a quarter of the class difference off.
- **Planted answers on synthetic data:**
  - a block-invariant code transfers as well as it decodes;
  - a block-specific code fails X1 in both directions but still passes X3;
  - a tonic block offset breaks only uncentered transfer;
  - within-level accuracies don't move with centering.
- **End to end:** `main()` with `analysis='block_transfer'` on synthetic data writes all its outputs.

`tests/analysis/decoding/test_cross_decoding.py` adds tests for the padding fix and for equal priors on a 3:1 class split.

To run them outside the cluster: `pip install -e . pytest`, then `python -m pytest -o addopts="" tests/analysis/decoding -q`.

---

### Part 3: running it and reading the output

#### Running

- **Synthetic dry run** (runs anywhere): `ANALYSIS=block_transfer DATA_SOURCE=synthetic SYNTHETIC_CODE=block_specific N_REPEATS=10 WINDOW_SIZE=16 STEP_SIZE=8 python dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py`.
  - It takes about 20 minutes on a 4-core machine. `N_REPEATS=2` finishes in a couple of minutes, but then nothing can reach significance, and the summary says so.
  - The planted answer is "X1 fails, X3 transfers". The default `SYNTHETIC_CODE` plants a block-invariant code, so everything should transfer.
  - The results go under `results/synthetic_<code>/`.
- **Real data on the DCC:**
  ```
  cd dcc_scripts/decoding
  bash submit_block_transfer_dcc.sh                                   # lpfc, sig electrodes
  ROI=acc ELECTRODES=all bash submit_block_transfer_dcc.sh            # another region, every electrode
  EPOCHS_ROOT_FILE=<root with the sig_chans you mean> bash submit_block_transfer_dcc.sh
  ```
- **Cost:** 4 designs × 2 centerings × 4 cells × (true + shuffle) × `N_REPEATS` resamples × `N_SPLITS` folds × windows. Check the first real run's runtime before scaling up.
- **Check the log first.** For each design it prints the real trials available per contrast × transfer-level cell, and how many of each are kept per resample. That is the go/no-go of the concurrent-regulation plan §4.4. If the kept number is in the low teens, expect a null and say so up front.

#### Outputs

Written to `results/<EPOCHS_ROOT_FILE>/block_transfer_<ROI>_<ELECTRODES>_w<W>s<S>/pooled_design_conditions/`:

| File | Contents |
|---|---|
| `summary.txt` | Read this first. For every design and centering: the 2×2 table, the ceiling test, the go/no-go line, any artifact flag, and the reading guide |
| `block_transfer.json` | The same numbers per cell, plus the run's settings and the group sizes |
| `block_transfer_traces.npz` | Accuracy traces, windows × resamples. Keys look like `X1_centered_25to75_true` / `..._shuffle` |
| `<design>_<centering>_<train>to<test>_<roi>_block_transfer.{pdf,png,eps}` | The transfer into a level, drawn against that level's own within-level accuracy (its ceiling) and the shuffle null. Bars mark windows where the transfer beats shuffle |

#### Reading `summary.txt`

Each design gets a block like this one. It comes from the synthetic dry run above with `SYNTHETIC_CODE=block_specific`, where the planted answer is that congruency uses a different axis in each incongruent-proportion level:

```
X1_uncentered: congruency, trained in one incongruent_proportion level and tested in the other
   balanced to 80 trials per contrast × transfer-level cell (available: {'c|inc25|sw25': 120, ...})
   post-stimulus mean accuracy (significant windows vs shuffle, post/pre):
     train | test               25%               75%
              25%       0.870 (3/0)       0.513 (2/0)
              75%       0.510 (0/0)       0.879 (3/0)
   25% -> 75% vs within 75%: below that ceiling in 3 windows
   75% -> 25% vs within 25%: below that ceiling in 3 windows
   CEILING: both within-level decodes beat shuffle -> interpretable
```

In that run:
- **X1** transfer sits at chance in both directions, uncentered and centered (0.51–0.53 against ceilings of 0.87–0.88). That is the "reorganizes the code" row of the table in 1.6.
- **X2 and X3** transfer at their within-level accuracy (about 0.88 and 0.75).
- **The 25% → 75% cell is "significant" in 2 windows at 0.513.** Resamples share trials, so tiny departures from the shuffle null can pass the cluster test. Judge a transfer against its ceiling, not against shuffle alone.

- **The table:** rows are the level trained on, columns the level tested on. The diagonal is the within-level ceiling; off the diagonal is transfer. Each cell shows the mean accuracy after stimulus onset, then (significant post / pre windows against the shuffle null).
- **The "vs within" lines** compare each transfer with the ceiling of the level it is tested on (see 1.4).
- **The CEILING line** is the go/no-go (the first row of the table in 1.6).
- **An ARTIFACT FLAG line** appears if any cell is significant before stimulus onset.
- **Read the pairs together:** a design's uncentered and centered blocks, then X1 against X3.

---

### Known issues (flagged, not fixed here)

- **A4(0b) cross cells** (congruency by switch proportion, switch type by incongruent proportion) have the same class-mix confound as unbalanced X3 (see 1.2). This predates the padding fix. For within-block numbers, use N3b's balanced within-level cells.
- **Resamples aren't independent subjects,** so cluster p-values are optimistic. X1-vs-X3 on the same trials is the load-bearing contrast.
- **Follow-ups if transfer sits at chance:** the PCA basis is fit on the training level ([Cross-decoding controls](#cross-decoding-controls) §4.3). X4 and X5 aren't built: X4 needs a letter-identity condition set.
- **Stale material:** `src/analysis/decoding/cross_decoding_tutorial.ipynb` and `docs/skeletons/a4_cross_decoding.py` describe functions that no longer exist.
- **The same job runs the task-transfer controls:** `ANALYSIS=task_transfer` (`submit_task_transfer_dcc.sh`) swaps the block for a trial-level factor — task across congruency and switch type, congruency and switch type across task ([Cross-decoding controls](#cross-decoding-controls) §3.5).

### Changes to A4

The padding fix (1.1) changes every A4 design, so A4 numbers from before this change aren't comparable with new runs.

- `build_cross_decoding_arrays` drops all-NaN padding rows. In each fold, cross-decoding subsamples incomplete training rows and equalizes the surviving class counts instead of applying mixup. Incomplete test rows remain and are filled with independent noise.
- `run_cross_decoding` now uses equal LDA priors (`make_decoder`). Without the padding, the within-block decodes (A4(0)) have their real 3:1 class ratio, and training-frequency priors would lean toward the majority class.

---

## Cross-decoding controls

*Cross-decoding controls — diagnosing a transfer that didn't work*

Companion to
[`analysis_plans.md` › Concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan)
§4 and to [`analysis_guide.md`](analysis_guide.md) §17. This document is the
troubleshooting protocol: what to run, in what order, when a cross-decode comes
back uninformative — and what each outcome licenses you to say.

It covers both shapes of cross-decode in this project:

- **label transfer** — two labellings of the *same* trials (A4: train congruency,
  score switchType). `build_cross_decoding_arrays` + `labels_test=`.
- **block transfer** — one labelling, two *disjoint trial populations* (N3b X1,
  X2, X2b, and X3). The implemented `test_only` path cuts folds only from the
  training level and scores every fold on the other level; the diagnostics below
  apply to it identically, plus §6.

---

### 1. First, name the failure

"It didn't work" is three different problems with three different fixes. Look at
the accuracy trace against the refit shuffle null before doing anything else.

| Signature | What it looks like | Section |
|---|---|---|
| **F1 — at chance** | transfer ≈ shuffle null, everywhere | §4 |
| **F2 — below chance** | transfer reliably *under* the null | §5 |
| **F3 — significant where it cannot be** | above-chance cluster in the pre-stimulus window, or transfer > within-condition accuracy | §6 |

F3 is the one the existing A4 runs actually show (analysis_guide §17's standing
caveat: the two cross panels carry clusters extending into and before the
baseline, for *current-trial congruency*, which is diagnostically impossible).
F1 is the one X1/X2 are most likely to produce. Do not debug them the same way.

---

### 2. The interpretability floor — run this before any diagnosis

**A transfer accuracy is meaningless without the within-condition accuracy on the
same trials, matched for n.** This is the decoding version of the noise ceiling.

Always report the pair:

```
within-condition   (train and test in the same block / same labelling)
transfer           (train in one, test in the other)
both against their own refit shuffle nulls, both with n per class printed
```

Decision rule:

| within-condition | transfer | Reading |
|---|---|---|
| at chance | at chance | **Uninformative.** There was no signal to transfer. Not a result. Fix the signal or report that the analysis is not runnable. |
| well above chance | at chance | **Interpretable null** — the code does not generalize. This is the X1 result that means "block context reconfigures the congruency code." |
| well above chance | above chance but lower | Partial generalization; quantify the drop, do not binarize it. |
| at chance | above chance | Impossible. Go to §6 — something is leaking. |

Concretely for the primary designs: if within-block congruency decoding in the
25%-incongruent blocks sits at 0.57 against a 0.50 null, a null 25 → 75 transfer
tells you nothing, and **no positive control elsewhere in the brain repairs it**.
The control you need is one that runs in the same ROI, at the same trial count,
in the same effect-size regime — that is X3 (§3.4).

---

### 3. The positive-control ladder

Cheapest first. Each rules out a different failure and each is worth running
before concluding anything about a real null.

#### 3.1 Synthetic ground truth (seconds, already implemented)

`cross_decoding.synthetic_roi_labeled_arrays(code="shared" | "orthogonal")` plants
a known answer, and two tests already assert it:

- `test_shared_code_transfers_and_orthogonal_code_does_not` — a planted shared
  axis transfers; a planted orthogonal axis does not, even though both are
  individually decodable (the orthogonal world is in fact the *easier*
  within-contrast decode, which is the point).
- `test_shuffle_null_is_at_chance_for_a_real_cross_decode` — the null is centred.

**This validates the code path, not your data.** Passing it means the transfer
machinery works; it says nothing about whether the lPFC signal is strong enough.
Extend it for block transfer: plant a block-invariant code (must transfer) and a
block-specific code (must not).

#### 3.2 Split-half through the cross-decode code path (minutes)

Run the *same* condition, trained on a random half and tested on the other half,
routed through `run_cross_decoding` / the block-transfer splitter rather than
through ordinary CV. Transfer accuracy must match ordinary cross-validated
accuracy on those trials.

This is the sharpest cheap control, because it isolates the *plumbing* from the
*science*: same trials, same signal, same classifier, only the code path differs.
If a split-half transfer through the new splitter underperforms ordinary CV on
the same data, the splitter (or the subsampling, or the stratification) is
broken — stop and fix it before interpreting X1.

#### 3.3 Occipital big letter across task (cheap, real data)

Train big-letter decoding on `task = global`, test on `task = local`. The
physical stimulus is identical and only attention differs, so visual cortex
should carry the big letter either way.

Caveat to write down and respect: **on congruent trials the big and small letters
are confounded**, so this is a control for the code path on real data, not a
claim about global-specific coding. Restrict to incongruent trials if you want it
clean, and report the trial counts.

#### 3.4 Congruency across switch proportion (the control that matters)

Decode congruency within one switch-proportion level and test in the other,
holding incongruent proportion fixed (design X3).

This is the control that makes a null X1 publishable, because it holds
*everything* constant except which block factor is being crossed: same ROI, same
electrodes, same trial-count regime, same effect-size regime, same number of
block transitions. The result you want:

> congruency **transfers** across switch proportion but **not** across incongruent
> proportion.

That contrast *is* the finding. Its absence — congruency failing to transfer
across both — means the failure is generic (SNR, block nonstationarity, or the
pipeline), not specific to LWPC.

#### 3.5 Task across congruency and switch type (the controls for the A4 transfer)

The positive controls for the congruency ↔ switch **label** transfer
(`analysis_plans.md` › Closing figure plan, "Congruency ↔ switch cross-decoding with task
positive controls"). They reuse the block-transfer 2×2 with a trial-level factor
in place of the block (`ANALYSIS=task_transfer`, `submit_task_transfer_dcc.sh`;
run recipe in [`analysis_guide.md`](analysis_guide.md) §17.5):

| Design | Decoded | Train → test | What a transfer shows |
|---|---|---|---|
| T1 | task | congruent → incongruent | the pipeline carries a code across trial populations — the clean control |
| T2 | task | repeat → switch | same, but confounded (below) |
| T3 | congruency | global task → local task | the A4 contrast transfers at its own effect size |
| T4 | switch type | global task → local task | same for switch type, with T2's confound |

Each design balances its four class × level cells and runs uncentered and
centered, exactly like N3b, so the reading rules of §2 and
[N3b block transfer](#n3b-block-transfer) §1.6 apply. `summary.txt` adds, per
transfer, the share of the ceiling's above-chance accuracy it keeps (over the
windows where the ceiling beats shuffle), and one line setting within-level task
accuracy (T1) against within-level congruency accuracy (T3).

What each can and cannot license:

- **T1 is a code-path control, not an effect-size control.** The frame colour
  that cues the task is drawn with the stimulus (`src/task/mainTask.m:163`), so a
  stimulus-locked task decoder partly decodes colour: large and partly visual. A
  T1 transfer rules out a broken pipeline; it does not show a congruency-sized
  code would survive. The plan's fix — subsample electrodes or trials until
  within-level task accuracy matches congruency's — is not built; the effect-size
  line says how far apart they are.
- **T3 is the matched control.** It *is* a congruency decoder, so it lives in the
  effect-size regime of the A4 transfer. Its task shift (the frame colour again)
  is a tonic offset between the two levels, which centering removes.
- **T2 and T4 carry a real confound.** On a switch trial the previous task was the
  other one. Leftover previous-task activity then agrees with the current task
  on repeat trials and opposes it on switch trials (T2), and flips its relation to
  switch type between the tasks (T4). Centering cannot remove it: it is a class ×
  level interaction, not a level offset. A T2/T4 drop is expected even with one
  code; `SYNTHETIC_CODE=carryover` plants it.
- **Pre-stimulus task windows are not automatically artifacts.** The previous
  task predicts the current one on repeat trials, so some task information can
  exist before the cue. `summary.txt` flags them separately from congruency /
  switch-type pre-stimulus windows, which remain artifacts (§6).

#### 3.6 RT matching (the control for a POSITIVE label transfer)

The rungs above guard a null. A transfer that comes back *above* chance needs the
opposite kind of control: a cheaper reason both labellings could be decoded along
one axis. The first is response time. Incongruent and switch trials are both
slower, so in stimulus-locked windows near the response anything that tracks
time-to-response separates both contrasts the same way.

`RT_MATCH=rt` keeps, per subject, a subset in which the congruency × switch-type
cells have the same RT distribution; `RT_MATCH=random` keeps the same counts
without regard to RT. The transfer survives RT if the `rt` run keeps about the
share the `random` run keeps. Method: [RT matching](#rt-matching); run and read:
[A4 §13](#13-the-control-battery-run-it-read-it-report-it). The response-locked
run (`response_main_effect_conditions`) looks at the same question from the
response side.

#### 3.7 Overall activity (the second control for a positive transfer)

If hard trials raise HG on most electrodes at once, a decoder trained on one
contrast learns "activity is up" and scores the other contrast above chance: a
shared gain rather than a shared pattern. `ACTIVITY_CONTROL=remove_mean` removes
each subject's mean across its decoded electrodes (only the pattern is left);
`ACTIVITY_CONTROL=mean_only` keeps only that mean. A shared pattern survives the
first and fails the second; a shared gain does the reverse. Method:
[Overall-activity control](#overall-activity-control).

---

### 4. F1 — transfer sits at chance

Work through these in order; each is cheap and each rules out a distinct cause.

#### 4.1 Joint-cell trial counts

Congruency × switchType × inc-proportion × switch-proportion cells lose trials
fast, and `subsample_to_min_trials_per_condition` takes the minimum **across
channels in the ROI**, so a single bad electrode caps the whole cell. The
`[NaN filter]` log lines reporting large "% dropped" are padding removal, not
artifact rejection — do not read them as data loss.

Print, per design: n per class in the train population, n per class in the test
population, and the four joint-cell counts.
`tests/analysis/decoding/test_cross_decoding.py::test_all_four_joint_cells_are_populated_and_balanced`
is the shape of the assertion.

**If the counts are in the low teens per class, expect a null and say so up
front.** This is not something a better classifier fixes.

#### 4.2 Block offset / nonstationarity (block transfer only, and it dominates)

Training in one block and testing in another means any tonic block-level HG
difference shifts the test cloud along a direction the classifier did not intend
to use. The baseline carries exactly that confound by construction: a random
0.5 s pre-stimulus baseline z-scored with statistics pooled across all trials, in
a design where `incongruentProportion` *is* the block
(`analysis_plans.md` › Simplification plan §1.4).

**Check:** center features within block — per channel, per block, subtract that
block's mean over trials — and re-run. Report both versions.

- Transfer recovers after centering → the null was a DC shift, not code
  reconfiguration. The centered version is the one that answers the question.
- Transfer still null after centering → the geometric claim survives its most
  likely artifact.

Corollary worth stating in Methods: centering deliberately discards the tonic
block effect, which may itself be the proactive-control signal. That is
undecidable in a blocked design, which is why both versions are reported.

#### 4.3 The PCA basis

`explained_variance=0.8` is **unsupervised** and refit on the training data every
fold. Nothing guarantees the retained components span the discriminant direction
for the *test* labelling — so a shared code can exist and still fail to transfer
because the axis it lives on was discarded as low-variance.

Three re-runs, any of which diagnoses it:

1. PCA off entirely (feasible only with few electrodes / a short window),
2. a fixed, generous `n_components`,
3. PCA fit on the **pooled** data (unsupervised, so no label leakage) rather than
   per fold.

If transfer appears under any of these, the null was a basis artifact. Report the
version with the pre-specified basis and note the sensitivity.

#### 4.4 NaN / train-test asymmetry

Cross-decoding deliberately handles missing train and test values differently.
With its default `oversample=False`, `sample_fold` removes incomplete training
pseudo-trials and deterministically subsamples the surviving classes to the same
count; it does **not** use `mixup2`. Partial test rows are retained and their NaNs
are filled with i.i.d. Gaussian noise, deliberately non-informative so test
imputation cannot leak class information. A transfer whose test population draws
more heavily on sparsely-covered subjects can nevertheless be depressed because
more of its test features are noise.

A pseudo-trial is complete only if every subject's trial in it is NaN-free, and
`make_epoched_data` marks outlier trials NaN per electrode. Under the random
pairing the pseudopopulation builder uses, the complete rows are the
*intersection* of the subjects' clean trials, which shrinks geometrically with the
number of subjects. With 24 subjects that can be one to five rows per condition,
and after RT matching LDA was left with fewer training rows than classes ("The
number of samples must be more than the number of classes"). The job therefore
re-pairs every decode input with `cd.align_complete_pseudotrials`. Each subject's
clean trials move to the top rows, in their existing random order, so the complete
rows become the *minimum* of the subjects' clean-trial counts. This happens once
over all ROI electrodes and again over each electrode group's own electrodes. No
trial is added, dropped or changed; only the arbitrary cross-subject pairing
moves. Runs from before this change trained on far fewer rows, so their
accuracies aren't comparable with new runs.

**Check:** per-subject channel coverage in the train population vs the test
population, and the fraction of test features that were NaN-filled. If the test
side is markedly sparser, restrict both sides to the subjects/channels present in
both and re-run.

#### 4.5 Feature and decoder matching

Two decoders whose accuracies are compared must match on trial count, class
balance, CV folds, feature set, window, and step size. An LWPC decoder with more
trials than the LWPS decoder will look better for that reason alone. Subsample to
the common minimum and average over subsamples, or do not compare them.

---

### 5. F2 — transfer reliably below chance

Below-chance transfer is almost always a **class-ordering flip** between the
training labelling and the scoring labelling — the classifier is right, the
labels are backwards.

**Check first:** `cats_train` vs `cats_test` from `build_cross_decoding_arrays`
(or the block-transfer equivalent). Both are `{tuple(group): class_idx}`; confirm
the same substantive class maps to the same index on both sides. In the block
transfer, confirm the contrast's `pos`/`neg` levels are resolved the same way in
both block populations.

Related traps in this codebase, both already guarded but worth re-checking when
the numbers look strange:

- **Confounded labellings.** If the two contrasts split the surviving trials
  identically, the "transfer" is the within-contrast decode reported as perfect
  generalization — a high number, not an error.
  `build_cross_decoding_arrays` raises on this (`_same_partition`), and
  `cd.factors_are_crossed` is the check to run when filtering conditions by hand.
- **Sign instability across folds.** LDA's class order is not guaranteed stable
  when a fold is missing a class. Pin it explicitly; this matters most for the
  Haufe patterns (plan §8.2 step 4), but it also produces noisy-looking accuracy
  when folds disagree.

A genuinely below-chance transfer, after ordering is verified, is an *anti*-code
(the two conditions use opposed axes). That is a real and reportable result — but
verify the ordering twice before claiming it.

---

### 6. F3 — significant where it cannot be

The diagnostic case: a congruency decode with an above-chance cluster **before
the stimulus**. Current-trial congruency cannot be known pre-stimulus, so any
such cluster is a confound readout. Use the pre-stimulus window as an **artifact
meter**: whatever drives it back to chance is the right fix.

Suspects, in the order worth testing:

1. **Fold structure ignores time.** `StratifiedKFold(shuffle=True)` draws random
   folds with no regard for trial order or run boundaries, so slow drift
   correlated with a temporally clustered label leaks across folds. **Fix:**
   time-/run-aware folds — leave-one-run-out or `GroupKFold` on run/block id.
   This is the same recommendation as simplification plan §2.8's
   leave-one-block-out.
2. **Block-level baseline leakage.** The pooled-statistics z-score puts tonic
   block differences into the pre-stimulus window by construction, and
   `incongruentProportion` is the block. The switchType panel (which varies
   *within* block) shows no pre-stimulus cluster while the proportion panel shows
   one spanning the whole baseline — that asymmetry is the signature. **Fix:**
   per-trial baseline (simplification plan §2.4) and/or within-block centering
   (§4.2 above).
3. **Tiny min-balanced samples** on the rare cell, which make accuracy estimates
   unstable enough to produce spurious clusters.
4. **Sequence carryover.** Legitimate for switch type (the previous trial defines
   it); a confound for congruency.

**Quick probe:** sweep `frac_train`. If the pre-stimulus cluster shrinks as the
training set shrinks, it is fold leakage rather than signal.

**Where it shows up now (A4, 2026-09-29).** Pre-stimulus windows appear in the
electrodes *selected* for the congruency and switch-type main effects (`both`, on
both the in-job `anova` and the `csv` route) and in the csv run's `all` (every
lPFC electrode, 398, including non-responsive ones), but not in the 171
task-significant electrodes decoded without selection (`none`). That pattern
points at suspect 2 through the selection:

- In the pooled 2×2 (`stimulus_main_effect_conditions`), 75% of incongruent
  trials come from 75%-incongruent blocks and 75% of switch trials from
  75%-switch blocks, so any tonic block difference separates the classes before
  the stimulus.
- The main-effect selection is a one-way ANOVA on congruency (or switch type)
  alone (`per_electrode_anova_labels`), with no block term, over the window mean.
  An electrode with a tonic block offset shows a "congruency effect", so the
  selection can enrich for exactly these electrodes.
- Non-responsive electrodes (in the 398) carry slow drift and tonic offsets with
  no stimulus response to dilute them.

Checks, cheapest first: decode block type with the ordinary decoding job on the
same electrodes (`CONDITIONS=stimulus_block_multiclass_conditions`, blocks A–D as
four classes, or `stimulus_block_pairwise_conditions`; with no table, as in
[Decoding job](#decoding-job) §3.1) and look at its pre-stimulus windows. If block
type is decodable before the stimulus, the leak is confirmed. Then the
selection split (`ELECTRODE_SELECTION_SPLIT=true`), and, if the groups are to be
reported, tonic block centering (§4.2) — which is not built for A4 and which
removes any genuine proactive (block-level) signal along with the artifact.

Also treat **transfer > within-condition accuracy** as an F3: a transferred axis
cannot beat an axis trained on the labelling it is scored against. That
combination means the two labellings are not actually crossed, or the test
population is contaminated with training trials.

---

### 7. Report this block with every cross-decode

Make it a fixed table in the output directory, not something reconstructed later.

```
design                     X1: congruency, 25%inc -> 75%inc
electrode set              lpfc, anatomical, n = ___ channels / ___ subjects
n per class (train)        ___ / ___
n per class (test)         ___ / ___
joint cell counts          ___ ___ ___ ___
feature centering          within-block: yes / no
PCA                        explained_variance = 0.8, refit per fold
fold structure             PredefinedSplit on block; ___ subsamples
within-condition acc       ___  (null ___, p ___)      <- the ceiling
transfer acc               ___  (null ___, p ___)
pre-stimulus cluster       none / [t0, t1]             <- artifact meter
reverse direction          75%inc -> 25%inc: ___
positive control X3        congruency across switch proportion: ___
positive control T1 / T3   task across congruency: ___ ; congruency across task: ___
RT matching (positives)    residual RT ___ ms; transfer ___ (rt) vs ___ (random)
activity (positives)       transfer ___ (remove_mean) vs ___ (mean_only) vs ___ (none)
response-locked            transfer windows ___ relative to the response
seeds                      share kept ___ +/- ___
```

The last four lines apply to a transfer that came back above chance; A4 §13.7 has
the same block filled in for the label transfer.

The two lines that carry all the interpretive weight are **within-condition acc**
and **pre-stimulus cluster**. A reader who sees the first can tell whether a null
means anything; a reader who sees the second can tell whether a positive means
anything.

---

### 8. Decision tree

```
transfer at chance?
├── within-condition also at chance ........... not runnable — report counts, stop (§2)
└── within-condition above chance
    ├── block transfer? → center within block and re-run ......... (§4.2)
    │   └── still null → check PCA basis (§4.3), NaN asymmetry (§4.4)
    ├── counts in the low teens? → underpowered, say so .......... (§4.1)
    └── all checks pass + X3 transfers → INTERPRETABLE NULL:
        the code is reconfigured by block context

transfer below chance? .......................... check class ordering first (§5)

transfer above chance?
├── pre-stimulus cluster present → artifact; fix folds/baseline .. (§6)
├── transfer > within-condition → labellings not crossed ......... (§6)
└── clean → rule out the cheaper shared axes (A4 §13)
    ├── RT_MATCH=rt keeps ≈ the share RT_MATCH=random keeps? ..... (§3.6)
    │   └── no → the transfer is (partly) a response-time difference
    ├── ACTIVITY_CONTROL=remove_mean keeps the share? ............ (§3.7)
    │   └── no, and mean_only transfers → a shared gain, not a pattern
    └── both yes → report with its ceiling, reverse direction, and the controls
```

---

### 9. What a clean result looks like

For the primary question, the reportable pattern is:

| Design | Expected if stability and flexibility are concurrently but separably regulated |
|---|---|
| within-block congruency decode | above chance in both incongruent-proportion blocks |
| X1 congruency 25% ↔ 75% inc | **fails to transfer** (block context reconfigures the congruency code) |
| X3 congruency 25% ↔ 75% switch | **transfers** (a block factor that does not reconfigure it) |
| within-block switchType decode | above chance in both switch-proportion blocks |
| X2 switchType 25% ↔ 75% switch | **fails to transfer** |
| X5 inc-proportion axis ↔ switch-proportion axis | at chance, with both within-axis decodes significant → concurrent but separable regulation |

X1-fails-while-X3-transfers is the load-bearing contrast. Either one alone is not
a result.

---

## RT matching

*A standalone util: subsample trials so the groups being compared have the same
RT distribution, in any analysis.*

**Code:** `src/analysis/utils/rt_matching.py`. **Tests:**
`tests/analysis/utils/test_rt_matching.py`. Wired into A4 / N3b / task transfer
through `RT_MATCH` ([A4 §6.4](#64-rt-matching)); any other analysis can call it
in one line.

### Why

Incongruent and switch trials are slower than congruent and repeat trials. On the
behavioural data (`combinedData.csv`, correct trials, 26 subjects) the mean
per-subject costs are +153 ms (i − c) and +197 ms (s − r). In stimulus-locked
data, anything that tracks time-to-response then differs between the levels of
both factors: response preparation arrives later on the slow trials, and activity
that lasts until the response lasts longer. A decoder or a power contrast can
read that latency as a condition effect, and a cross-decode can read it as a
shared code. Matching asks whether the effect survives when the compared trials
were, on average, equally fast. Reviewers commonly ask for it on any
conflict/switch-cost effect.

### How

Separately for each subject (and any extra `within` strata):

1. Pool the subject's RTs over every group being matched and cut them into
   `n_bins` quantile bins (equal trial counts per bin; invariant to RT vs log RT).
2. Count each group's trials per bin, choose how many to keep per group and bin,
   and draw that many at random without replacement.
   - `balance='equal'` (default): the minimum across groups, so every group ends
     up the same size with the same bin profile. Decoders balance classes anyway.
   - `balance='proportional'`: every group gets the same bin profile but keeps its
     size relative to the others (rounding spread across bins, so it stays
     unbiased). Use it when unequal group sizes should survive, e.g. a power
     contrast inside a 75/25 block.

Rows with no RT, or a missing value in a matched factor, are dropped. A stratum
with fewer than two groups is dropped. Each stratum's draw depends only on the
seed and that stratum, so adding a subject never changes another's trials.

**The control.** Matching keeps about half the trials, so a weaker effect after
matching could just be the smaller sample. `count_matched_random` (or
`mode='random'`) keeps exactly as many trials per subject and group as the
RT-matched set, drawn with no regard to RT. Compare the RT-matched result with the
random one: the difference between them is the RT-driven part.

**On the behavioural data** (the four congruency × switch-type cells matched to
each other, `balance='equal'`):

| `n_bins` | kept | i − c after | s − r after |
|---|---|---|---|
| 3 | 56% | +14 ms (p = .06) | +22 ms (p = .001) |
| 5 | 54% | +8 ms (p = .03) | +4 ms (p = .16) |
| 10 (default) | 48% | +2 ms (p = .26) | +1 ms (p = .59) |
| 10, `mode='random'` | 48% | +142 ms (p < .001) | +208 ms (p < .001) |

Kept per subject × cell at the default: median 47 trials, minimum 20. The epoched
data can hold fewer trials than the behaviour file, so read the job's own
`rt_match_*_balance.csv`.

### Which factors to match

Match every factor whose levels the analysis compares, together, so none of them
keeps an RT difference:

| Analysis | `groups` | Notes |
|---|---|---|
| A4 congruency ↔ switch-type transfer | `('congruency', 'switchType')` | the job's default |
| Congruency decoded within each incongruent-proportion block (LWPC) | `('congruency', 'incongruentProportion')` | matches i vs c *and* the two blocks to one another |
| Same, but leave block RT differences alone | `('congruency',)`, `within=('incongruentProportion',)` | only i vs c inside each block |
| Switch type within switch-proportion blocks (LWPS) | `('switchType', 'switchProportion')` | |
| A power trace of i vs c | `('congruency',)`, `balance='proportional'` | keeps more trials when the cells differ in size |

Factor names can use any project spelling (`switchType`, `switch_type`,
`task_sequence`; `incongruentProportion`, `incongruent_proportion`); metadata
columns can be named directly too (e.g. `prev_congruency`).

### Plugging it into another analysis

Every loader returns the `{subject: {condition: {key: Epochs}}}` structure, and
the epochs' metadata already carries `reaction_time` and `trial_count` (parsed
from the event names by `make_metadata_from_event_names`). So right after
loading:

```python
from src.analysis.utils.rt_matching import (
    rt_match_subjects_mne_objects, save_rt_match_report)

subjects_mne_objects = create_subjects_mne_objects_dict(...)          # as now
subjects_mne_objects, rt_report = rt_match_subjects_mne_objects(
    subjects_mne_objects, groups=('congruency', 'incongruentProportion'),
    mode='rt')                                   # 'random' for the control run
save_rt_match_report(rt_report, save_dir)        # rt_match_{balance,contrasts,summary}.csv
```

In the ordinary decoding job that is after `create_subjects_mne_objects_dict` in
`dcc_scripts/decoding/decoding_dcc.py` (~line 388); in the power traces, after
the same call in `dcc_scripts/power/power_traces_dcc.py` (~line 124). Neither is
wired yet. Give the matched run its own save directory.

The loader also stores a trial-averaged `<key>_avg` and `<key>_std_err` Evoked
next to every Epochs, and the power-trace plots read the `_avg` ones
(`src/analysis/power/evoked_builders.py`). The adapter rebuilds both from the
kept trials (`refresh_evoked`, the loader's own formulas), so a matched power
trace plots the matched average, not the full one.

For anything that is not an epochs structure (a behavioural table, a long
per-trial DataFrame), call the core directly:

```python
from src.analysis.utils.rt_matching import rt_match, count_matched_random
keep = rt_match(trials, ['congruency', 'task_sequence'], within=['subject'])
control = count_matched_random(trials, keep, ['congruency', 'task_sequence'])
```

### Reporting it

- The method: "Within each participant, trials were subsampled so that
  [the four congruency × switch-type cells] had matched RT distributions (10
  quantile bins of the participant's pooled RTs; equal trials per cell per bin).
  A control analysis used an equal number of randomly chosen trials per cell."
- The residual: the `after` row of `rt_match_*_summary.csv`, e.g. "residual RT
  difference i − c = +2 ms, t(25) = 1.2, p = .26".
- The result against the random control, not against the full data.

### What it does not do

- It equates mean timing, not trial-by-trial processing: within a bin a slow
  group can still sit slightly later (the residual line says how much).
- It does not remove effects that scale with difficulty rather than with time
  (e.g. more activity on hard trials at every RT). If an effect survives matching,
  "shared difficulty signal" is still open; RT matching only rules out
  "shared latency".
- Error trials and trials with no response have no usable RT and are never kept.

---

## Overall-activity control

*Is a decode, or a transfer, carried by a uniform rise in activity on hard trials,
or by the pattern across electrodes?*

**Code:** `src/analysis/decoding/activity_control.py`. **Tests:**
`tests/analysis/decoding/test_activity_control.py`,
`test_cross_decoding_activity_control.py`. Wired into A4 / N3b / task transfer
through `ACTIVITY_CONTROL` ([A4 §6.5](#65-overall-activity-control)).

### Why

Incongruent and switch trials are both harder. If being on a hard trial simply
raises high gamma on most electrodes, a decoder trained on congruency learns
"activity is up", and that axis separates switch from repeat too. The transfer is
then real but says "both effects raise overall activity", not "the two effects
share a representational code". A4's transfer cannot tell these apart on its
own.

### What the two modes do

Both are per subject: a pseudo-trial row holds a *different* physical trial from
each subject, so a mean across all of a row's channels would mix trials. The
subject comes from the channel name (`<subject>-<electrode>`).

- **`remove_mean`** — for each subject, subtract the mean across that subject's
  decoded electrodes, per pseudo-trial and time point. Anything that moves all of
  a subject's electrodes together is gone; the pattern across electrodes is left.
- **`mean_only`** — replace each subject's electrodes by their mean: one feature
  per subject, the uniform part alone.

NaNs stay where they are (a subject absent from a row stays absent). A subject
with a single decoded electrode has nothing left after `remove_mean`; the summary
lists them.

### How to read them

Compare the three runs (`none`, `remove_mean`, `mean_only`) by each transfer's
**share of its own ceiling** (`keeps X%`), since the ceilings change with the
transform too:

| `remove_mean` | `mean_only` | Reading |
|---|---|---|
| transfer survives, similar share | transfers little or not at all | **A shared pattern.** Not explained by a uniform rise in activity. |
| transfer gone, ceilings still above chance | transfers | **A shared gain.** The two effects share "more activity on hard trials", while their specific patterns differ. |
| transfer reduced but present | transfers | **Both.** Report the share under each. |
| ceilings gone too | — | The contrasts are themselves mostly overall activity; `remove_mean` cannot speak to the transfer. |

**Planted answers** (the test): two subjects × 8 electrodes; congruency and switch
type each have their own zero-mean pattern plus a shared component. Congruency's
ceiling / congruency→switch transfer:

| Shared component | full | `remove_mean` | `mean_only` |
|---|---|---|---|
| uniform gain | 0.81 / 0.66 | 0.60 / **0.52** | 0.74 / **0.76** |
| zero-mean pattern | 0.77 / 0.68 | 0.78 / **0.65** | 0.48 / **0.51** |

Keep the planted effects weak when extending these tests: with large effects LDA
treats the other factor's effect as within-class variance and projects the shared
direction out, and nothing transfers in either world.

### What it cannot rule out

- `remove_mean` removes only a shift shared by **all** of a subject's decoded
  electrodes. A rise on a subset of them (say, the most task-responsive third)
  survives and reads as "pattern". Restricting to more homogeneous groups, or
  adding `mean_only` on subgroups, narrows this but does not close it.
- It is additive. The HG is z-scored per channel against baseline, so a
  proportional gain mostly looks additive, but not exactly.
- It says nothing about RT: combine with `RT_MATCH=rt` (and its `random`
  control) for the strongest version. The folder tags stack
  (`..._rtmatch10_remove_mean/`).

### Using it elsewhere

```python
from src.analysis.decoding.activity_control import apply_activity_control
arrays, channel_names, info = apply_activity_control(
    roi_labeled_arrays, roi, channel_names, 'remove_mean')   # or 'mean_only'
```

`roi_labeled_arrays` is the `{roi: {condition: (trials, channels, time)}}` dict
the decoders use; `channel_names` its channel labels in order (for a LabeledArray,
`roi_labeled_arrays[roi].labels[2]`). It returns plain arrays. The ordinary
decoding job reads only the condition keys and arrays
(`gather_class_data_by_stratum`), so they should drop in, but it is neither wired
nor tested there. Apply it after restricting to the electrodes you decode, so each
subject's mean is over those electrodes.
