# Decoding — the time-resolved decoding job: how to run it and how to read it

**What this document is.** A standalone walkthrough of the ordinary decoding job
(`submit_specific_conditions_decoding_dcc.sh` → `run_decoding_dcc.py` →
`decoding_dcc.py`): what it decodes, the ways to choose the electrodes it decodes
from (no table, a saved A1 table, an ANOVA on held-out trials, and the other
launchers), every parameter that changes the answer, the exact commands, every
file it writes, and how to read them.

It is the run-and-read companion to [`analysis_guide.md`](analysis_guide.md) §7
(the module layout and the `Decoder` class) and §21 (the circularity controls, in
depth). For training on one contrast and testing on another, see
[`a4_cross_decoding.md`](a4_cross_decoding.md); that job shares the decoder but
not the statistics described here.

If you read only one section, read [§3 Choosing the electrodes](#3-choosing-the-electrodes):
the shipped submit script decodes from a saved ANOVA table by default, and turning
that off needs a file edit.

---

## 0. The short version

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

## 1. Where the code lives

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

### The call path

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

## 2. What the job computes

### 2.1 What is decoded

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

### 2.2 One bootstrap

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

### 2.3 Statistics

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

## 3. Choosing the electrodes

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

### 3.1 No table: the task-significant (or all) electrodes

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

### 3.2 csv: a saved A1 table

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

### 3.3 ANOVA sets on held-out trials

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

### 3.4 Responsiveness on held-out trials

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

### 3.5 Coupling sets

`submit_decoding_with_coupling_electrode_sets_dcc.sh` decodes the electrodes that
take part in a significant gamma-envelope correlation pair (the PAC path's
`high_corr_*.csv`), against `N_DRAWS` n-matched draws of non-coupled electrodes,
and compares the coupled trace with the distribution over draws. Its header
explains the knobs; run `report_coupling_counts.py` first, and
`run_coupling_comparison_dcc.py` after a run split over several jobs.

---

## 4. Parameters

### 4.1 From the environment

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

### 4.2 Constants in `run_decoding_dcc.py` (edit the file)

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

## 5. How to run it

### 5.1 Set up the shell

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

### 5.2 Common runs

| Question | Command |
|---|---|
| Main effects in each main-effect population | `ANOVA_LABELS_CSV=$COND_CSV bash submit_specific_conditions_decoding_dcc.sh` (the default: 3 populations × 2 sets) |
| LWPC/LWPS in the main-effect populations | `ANOVA_LABELS_CSV=$COND_CSV CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" bash submit_...` |
| LWPC/LWPS in the LWPC-only / LWPS-only electrodes | run twice with `ANOVA_LABELS_CSV=$PROP_CSV` and `ANOVA_LABEL_EFFECT=lwpc_only`, then `lwps_only` |
| Everything in the region, no selection | `ANOVA_LABELS_CSVS=("")` in the script (§3.1), then `CONDITIONS="..." bash submit_...` |
| Two regions | add `ROIS=lpfc,occ` |
| Selection on held-out trials | `bash submit_decoding_with_anova_electrode_selection_dcc.sh` (§3.3) |

### 5.3 Before you submit

- The script echoes `condition=… anova_labels=… effect=…` per job. That is what
  will run.
- Logs: `out/slurm_<jobid>_dec_a<table>e<pop>_<condition>.out`. The top prints
  subjects, conditions, ROIs, the selection settings and every decoding parameter;
  the `[anova-labels] decoding with N electrodes` line gives the set size.

### 5.4 Cost

Per comparison × ROI × bootstrap: `N_REPEATS × N_SPLITS` true fits and
`N_SHUFFLE_PERMS × N_SPLITS` shuffle fits at every window, so the shuffle is ten
times the true decode. A `pooled_shuffle` adds another 50 × 5 per ROI. The wrapper
asks for 48 h and 225 GB; the memory is for loading every subject's epochs. The
selection launchers add the ANOVA permutations (§3.3) on top.

### 5.5 Leave one subject out

`submit_loo_decoding_dcc.sh` submits one job per subject in its
`LEAVE_OUT_SUBJECTS` list and condition set, each with that subject dropped
(`_loo-<subject>` in the filenames). A result that disappears when one subject is
left out rests on that subject. Edit the two arrays at the top first; the list
must name subjects that are in `SUBJECTS` (as shipped it includes `D0130`, which
is not, so that job decodes everyone).

---

## 6. Outputs

### 6.1 Where

| Route | Save directory |
|---|---|
| no table | `dcc_scripts/decoding/figs/<EPOCHS_ROOT_FILE>/` (all condition sets share it; the comparison names keep them apart) |
| csv | `…/figs/<EPOCHS_ROOT_FILE>/anova_label_selections/<table>__effect-<pop>__correction-<c>__alpha-<a>__roi-all__<hash>/<CONDITION_NAME>/` |
| ANOVA or coupling sets | `…/figs/<EPOCHS_ROOT_FILE>/elecset_<set>/` per set |

`.png`, `.pdf`, `.pkl` and friends are git-ignored: the results live on the
cluster.

### 6.2 What

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

### 6.3 The results pickle

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

## 7. How to read the result

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
| bar in windows centred before −0.125 s | Nothing about the current stimulus exists there. For congruency or switch type, it is an artifact meter: block-level baseline differences or fold leakage ([`cross_decoding_controls.md`](cross_decoding_controls.md) §6). Task can legitimately appear early on repeat trials. |
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

## 8. Known issues

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
   ([`analysis_simplification_plan.md`](analysis_simplification_plan.md) §1.1–1.2).
7. **The no-table runs of every condition set share one folder**, kept apart only
   by comparison names and job IDs.

---

## 9. Checklist

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

## 10. Tests

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

## Related documents

- [`analysis_guide.md`](analysis_guide.md) §7 — the decoding modules and the `Decoder` class; §21 — the held-out selection splits in depth; §17.5 — the runbook for the main-effect populations
- [`a4_cross_decoding.md`](a4_cross_decoding.md) — train on one contrast, test on another, in the same populations
- [`n3b_block_transfer.md`](n3b_block_transfer.md) — train in one block, test in the other
- [`cross_decoding_controls.md`](cross_decoding_controls.md) §6 — what a pre-stimulus cluster means
- [`n2_direction_tests.md`](n2_direction_tests.md) — the univariate direction of the adaptation effects
- [`analysis_simplification_plan.md`](analysis_simplification_plan.md) §1 — what pseudo-trial decoding can and cannot show
