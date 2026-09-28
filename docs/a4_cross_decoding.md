# A4 — cross-decoding congruency ↔ switch type: how to run it and how to read it

**What this document is.** A standalone walkthrough of the A4 cross-decoding job
(`submit_stability_flexibility_cross_decoding_dcc.sh`): the question it asks, the
four ways it can define electrode groups (`anova`, `csv`, `power_traces`, `none`),
every parameter that changes the answer, the exact commands, every file it writes,
and how to read them. It also covers the task-transfer positive controls, which
run through the same job (§9).

It is the run-and-read companion to three other documents, and does not repeat
them:

- [`analysis_guide.md`](analysis_guide.md) §17 — why A4 is built the way it is
  (derived class definitions, why condition sets must cross, the payoff 2×2).
- [`cross_decoding_controls.md`](cross_decoding_controls.md) — what to do when a
  transfer comes back uninformative.
- [`n3b_block_transfer.md`](n3b_block_transfer.md) — the block-transfer analysis,
  which is a third mode of this same job.

For the ordinary (non-transfer) decoding job, see [`decoding.md`](decoding.md).

If you read only one section, read [§3 The defaults you actually get](#3-the-defaults-you-actually-get):
the submit script and the Python runner disagree on half the knobs, and the
submit script's defaults changed on 2026-09-28.

---

## 0. The short version

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
ELECTRODE_SELECTION_SPLIT=true SAVE_DIR=<a folder of its own> \
    bash submit_stability_flexibility_cross_decoding_dcc.sh
```

Then open `summary.txt` in the save directory printed near the top of
`out/slurm_<jobid>_<jobname>.out`. For each electrode group, the result is three
numbers per transfer direction: how many windows beat the shuffle null, how many
fall below the within-contrast ceiling, and what share of the ceiling it keeps
(§8).

---

## 1. Where the code lives

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
| Same job, other analyses | `submit_block_transfer_dcc.sh` (`ANALYSIS=block_transfer`, N3b), `submit_task_transfer_dcc.sh` (`ANALYSIS=task_transfer`, §9) |

### The call path

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
          ├ factors_are_crossed check; class definitions read from the conditions' declared levels
          ├ A4(0)   within-block decodes                               [16-cell condition set only]
          ├ A4(0b)  within-block 2x2 per definition group, circular cells skipped   [16-cell + groups]
          ├ A4(a)   per group: stab_to_stab, flex_to_flex, stab_to_flex, flex_to_stab
          ├ A4(c)   temporal generalization on TEMPGEN_GROUPS
          └ cross_decoding.json, accuracy_traces.npz, tempgen_*.npy, anova_labels.csv,
            figures, summary.txt
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

## 2. What A4 asks

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
> [`cross_decoding_controls.md`](cross_decoding_controls.md) §5 before reporting it.

A transfer means nothing without its **ceiling**: the within-contrast decode of
the labelling it is scored on, on the same trials with the same folds.
`stab_to_flex` is read against `flex_to_flex`, and `flex_to_stab` against
`stab_to_stab`. The job runs all four in every group and compares them for you.

### The designs

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

## 3. The defaults you actually get

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

So `bash submit_stability_flexibility_cross_decoding_dcc.sh` with nothing set is:
**main-effect groups** (congruency and switch-type main effects, raw p < 0.05,
window-mean HG over 0–1.5 s), fit in the job on the task-significant lPFC
electrodes, with the transfer pooled over both proportions. The within-block
designs A4(0)/A4(0b) do **not** run, because the 4-cell condition set has no block
factor.

> **Stale defaults elsewhere.** [`analysis_guide.md`](analysis_guide.md) §17.4
> still describes the older submit defaults (csv route, 16-cell set, 0–0.5 s,
> 20/10 samples), and the §17.5 runbook's step 3 command
> (`ANOVA_LABELS_CSV=$COND_CSV bash submit_...`) no longer reads the table,
> because the default route is now `anova` (§4.2). This document describes the
> scripts as they are.

---

## 4. Electrode definitions

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

### 4.1 `anova` — fit the ANOVA in the job (default)

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

> **Gotcha: the split is not in the folder name.** A split run and an unsplit
> run with the same settings write to the same directory, and the second one to
> finish overwrites the first. `summary.txt` does not record the split either
> (the slurm log does: look for `[trial-split]` lines). Give the split run its own
> `SAVE_DIR=...`.

### 4.2 `csv` — reuse a saved A1 table

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

### 4.3 `none` — no groups, just the loaded electrodes

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

### 4.4 `power_traces` — reuse finished windowed-ANOVA runs

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

## 5. Condition sets

`CONDITIONS` names a dict in `src/analysis/config/experiment_conditions.py`. A4
needs every condition to declare `congruency` **and** `switchType`, and needs the
two to cross (all four combinations present). Two sets qualify:

| `CONDITIONS` | Cells | Designs that run | Why pick it |
|---|---|---|---|
| `stimulus_main_effect_conditions` (**submit default**) | 4: `Stimulus_{i,c}{r,s}`, both proportions pooled | A4(a), A4(c) | about 4× the trials per cell, so fewer incomplete rows; folds stratified on congruency × switch type |
| `stimulus_experiment_conditions` (runner default) | 16: the full 2×2×2×2 | all four | the only set with block factors; folds stratified on all four factors |

`response_experiment_conditions` is the response-locked 16-cell set (pair it with
a response-locked `EPOCHS_ROOT_FILE` and set `FIRST_TIME_POINT` to that file's
first sample). Single-factor sets (`stimulus_congruency_conditions`, …) are
refused: they are separate epoch sets over the same trials, so the transfer would
be scored on trials it trained on. Confounded sets (`stimulus_iS_cR_err_conditions`
and siblings) are refused: congruency and switch type split their trials
identically, so a "transfer" would be the within decode. `CONDITIONS="a b"`
submits one job per set.

---

## 6. Parameters

All are environment variables; nothing needs a file edited. Defaults below are
the submit script's (§3).

### 6.1 Data and electrodes

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

### 6.2 Electrode definition

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

### 6.3 Decoding

| Variable | Default | Notes |
|---|---|---|
| `WINDOW_SIZE` / `STEP_SIZE` | `64` / `16` | Samples at 256 Hz: 250 ms windows every 62.5 ms, 37 windows over −1.0 to 1.5 s. |
| `N_SPLITS` | `5` | Folds; or random resamples per repeat when `FRAC_TRAIN` is set. |
| `N_REPEATS` | `10` | Repeats of the fold split. **The samples of the cluster test** and the main runtime lever. Temporal generalization uses half (at least 2). |
| `FRAC_TRAIN` | unset | Unset = `StratifiedKFold`, (N_SPLITS−1)/N_SPLITS train. Set (e.g. `0.5`) for `StratifiedShuffleSplit` at that fraction. Also a probe for fold leakage ([`cross_decoding_controls.md`](cross_decoding_controls.md) §6). |
| `EXPLAINED_VARIANCE` | `0.8` | PCA variance kept, refit per fold. |
| `N_PERM` | `500` | Permutations for each cluster test. |
| `TEMPGEN_GROUPS` | `both` (`all` under `none`) | Comma-separated. `''` skips A4(c). `both,all` adds the unselected matrix. Each matrix costs `n_windows²` predictions. |
| `TRAIN_LABEL` / `TEST_LABEL` | unset | One decode instead of the battery: `stability`/`congruency`, `flexibility`/`switchType`. Goes to a `train_<x>_test_<y>/` subfolder and has no ceiling comparison. |
| `SAMPLING_RATE` / `FIRST_TIME_POINT` | `256` / `-1.0` | Only label the figure time axes. Change `FIRST_TIME_POINT` for an epoch that does not start at −1.0 s. |
| `SEED` | `0` | Seeds the folds, the ROI-array padding and the cluster tests. |
| `SAVE_DIR` | derived | Overrides the whole output path. |

---

## 7. How to run it

### 7.1 Dry run on synthetic data (minutes, no data)

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

### 7.2 The real runs

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
| The same, with no selection/decode trial overlap | `ELECTRODE_SELECTION_SPLIT=true SAVE_DIR=$PWD/results/$EPOCHS_ROOT_FILE/cross_decoding_lpfc_split_anova_condition/stimulus_main_effect_conditions bash submit_...` |
| In the LWPC/LWPS populations | `CONTRAST_MODE=proportion bash submit_...` (in-job) or `ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=<a ..._proportion_... table> bash submit_...` |
| With the within-block designs too | add `CONDITIONS=stimulus_experiment_conditions` |
| With the unselected temporal-generalization matrix | add `TEMPGEN_GROUPS=both,all` |
| The task-transfer positive controls | `bash submit_task_transfer_dcc.sh` (§9) |

Run the `none` job first. It is the cheapest (one group) and is the reference
every grouped run is compared with.

### 7.3 Before you submit

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

### 7.4 Cost

Each group runs 4 decodes (true + shuffle each) of `N_REPEATS × N_SPLITS` fits
per window, plus 6 cluster tests; temporal generalization adds 3 matrices per
group in `TEMPGEN_GROUPS` at `n_windows²` predictions each. A csv `union` job
decodes up to 4 groups and is the heaviest. The wrapper asks for 16 h; if a job
runs out, resubmit with `SBATCH_TIMELIMIT=36:00:00 bash submit_...`, or halve the
work with `N_REPEATS=5`.

---

## 8. Outputs and how to read them

### 8.0 The files

Everything goes to the save directory of §4.

| File | Contents | Read it for |
|---|---|---|
| `summary.txt` | Settings, then every design's numbers and the reading guide | **Start here** |
| `cross_decoding.json` | The same numbers per design and group, with per-window `significant_windows`, `cluster_p`, `n_below_ceiling`, `retained`; arrays longer than 64 values are dropped | Tables and scripts |
| `accuracy_traces.npz` | Label-transfer accuracy per window × repeat. Keys `labeltransfer_<group>_<direction>_true` / `_shuffle` | Re-plotting, your own statistics |
| `tempgen_<name>.npy` | Temporal-generalization matrix, train window × test window, e.g. `tempgen_stability_flexibility_cross_both.npy` | A4(c) |
| `anova_labels.csv` | The per-electrode definition table the groups came from (`anova`, `csv`, `power_traces`) | Which electrodes are in which group |
| `cross_decoding_summary.png` | Overview: A4(0) bar charts (top left, empty without block factors), the `stab_to_flex` trace per group (top right), up to three temporal-generalization matrices (bottom) | A first look |
| `<direction>_<group>__cross_decoding.{png,pdf,eps}` | One figure per group × direction: true accuracy against shuffle, ±1 SD over repeats, bars where the cluster test is significant | The figures to show |

Window times are window **centres**. With 64-sample windows, a window centred at
*t* covers *t* ± 125 ms, so the first window with no pre-stimulus sample is
centred at +0.125 s, and the last window entirely before the stimulus is centred
at −0.125 s.

### 8.1 What `summary.txt` looks like

The layout, with `…` for numbers (written by `write_summary`):

```
========================================================================
STABILITY vs FLEXIBILITY — A4 CROSS-DECODING
========================================================================
           data_source: real
  electrode_definition: csv
       reference_group: all
 electrode_group_sizes: {'both': …, 'congruency_only': …, 'switch_type_only': …, 'all': …}
                window: [0.0, 1.5]s
               …        (every other setting of the run)
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

### 8.2 Reading it, in order

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
| beats shuffle | reliably below shuffle, keeps < 0% | **Anti-aligned axis** (incongruent with repeat). Check the class ordering first ([`cross_decoding_controls.md`](cross_decoding_controls.md) §5). |
| – | transfer above its own ceiling, or significant well before stimulus onset | **Artifact** (F3, [`cross_decoding_controls.md`](cross_decoding_controls.md) §6). |

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

**Step 8 — within-block (16-cell runs only).** A4(0) lists each block's mean
accuracy and `Δ(block) = high − low`. A negative Δ for congruency means congruency
is less decodable in 75%-incongruent blocks, the direction LWPC predicts. Δ is a
difference of means over **all** windows, baseline included, with no test of its
own, so read it with the traces. A4(0b) lists the per-group cells, with the
circular one named on each group's `ignored cell=` line.

### 8.3 Things the numbers do not tell you

- **The p-values are optimistic.** The samples of every cluster test are CV
  repeats of the same trials, not subjects, and the pseudopopulation concatenates
  electrodes from different patients whose trials were never recorded together.
  Treat `n_sig_windows` as a within-dataset reliability check.
- **`mean acc` and `peak` in A4(a) are averaged over every window, including the
  second before the stimulus.** They are diluted and are not the post-stimulus
  accuracy. Read the traces, or average `accuracy_traces.npz` over the windows you
  care about.
- **A transfer at chance is only a result next to a ceiling that is not.**
  [`cross_decoding_controls.md`](cross_decoding_controls.md) §2 is the rule, and
  the task-transfer T1/T3 controls (§9) show the pipeline can carry a code from
  one trial population to another.

---

## 9. The task-transfer positive controls

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
  `summary.txt` are the N3b ones ([`n3b_block_transfer.md`](n3b_block_transfer.md)
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
[`cross_decoding_controls.md`](cross_decoding_controls.md) §3.5; the block-transfer
counterpart (X1–X3, X2b) is [`n3b_block_transfer.md`](n3b_block_transfer.md).

---

## 10. Known issues and gaps

1. **`ANOVA_LABELS_CSV` is silently ignored unless `ELECTRODE_DEFINITION=csv`**
   (§4.2). The §17.5 runbook command in `analysis_guide.md` predates the default
   change and now runs the `anova` route.
2. **`ELECTRODE_SELECTION_SPLIT` is not in the folder name or `summary.txt`**
   (§4.1). Split and unsplit runs overwrite each other.
3. **On the csv route, `ELECTRODES` and the window are ignored but still name the
   folder** (§4.2).
4. **`power_traces` needs `CONTRAST_MODE=proportion`** by hand (§4.4).
5. **A4(a) `mean acc` includes the baseline** (§8.3). There is no post-stimulus
   summary for label transfer, unlike N3b's table.
6. **Temporal generalization has no null** (§8.2 step 7).
7. **The earlier real A4 runs showed cross-decode clusters before stimulus
   onset** ([`analysis_guide.md`](analysis_guide.md) §17, caveat). Until that is
   explained, any transfer needs its pre-stimulus windows reported next to it.
8. **Resamples are not subjects** (§8.3); there is no leave-one-subject-out for A4.

---

## 11. Checklist

```
[ ] export EPOCHS_ROOT_FILE once; every job below uses it
[ ] synthetic dry runs: shared transfers, orthogonal does not
[ ] ELECTRODE_DEFINITION=none                     (the ungrouped reference run)
[ ] ELECTRODE_DEFINITION=csv ANOVA_LABELS_CSV=...  (or the anova route; check the echo lines)
[ ] ELECTRODE_SELECTION_SPLIT=true with its own SAVE_DIR   (clean ceilings)
[ ] ELECTRODES=all bash submit_task_transfer_dcc.sh        (controls, matching electrodes)
[ ] log: group sizes, skipped groups, cells, block levels
[ ] every ceiling (stab_to_stab, flex_to_flex) beats shuffle   <- else stop
[ ] per group and direction: sig windows, windows below ceiling, share kept
[ ] both directions agree
[ ] groups compared by share kept, against 'all'
[ ] no significant windows centred before -0.125 s
[ ] T1 transfers; T3 share set beside A4's share
```

---

## 12. Tests

Run outside the cluster with `pip install -e . pytest` then
`python -m pytest -o addopts="" tests/analysis/decoding -q`.

| Test file | Pins |
|---|---|
| `test_cross_decoding.py` | the two label vectors, stratification on the condition cell, padding handling, equal priors, shared-transfers/orthogonal-does-not, `frac_train`, temporal generalization |
| `test_cross_decoding_condition_scheme.py` | class definitions read from declared levels, crossed vs confounded sets, 4- vs 16-cell sets, the `none` route decoding only the loaded electrodes |
| `test_cross_decoding_electrode_groups.py` | channel keys, disjoint groups, the reference group, the csv `union` and raw-correction behaviour |
| `test_cross_decoding_circularity.py` | which within-block cell each group double-dips on |
| `test_cross_decoding_runner.py` | contrast mode read from the folder, mode/effect mismatches refused, csv ignored off its route, folder names |
| `test_task_transfer.py` | T1–T4 on real condition sets, planted synthetic answers, end to end |

---

## Related documents

- [`analysis_guide.md`](analysis_guide.md) §17 — A4's design and the reasons behind it; §17.5 — the runbook for the main-effect populations and task controls
- [`cross_decoding_controls.md`](cross_decoding_controls.md) — diagnosing a transfer that did not work, and the report block to print with every transfer
- [`n3b_block_transfer.md`](n3b_block_transfer.md) — block transfer, the third mode of this job
- [`decoding.md`](decoding.md) — the ordinary decoding job, run in the same populations
- [`closing_figure_plan.md`](closing_figure_plan.md) — where the cross-decoding results sit in the paper
- [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md) §4 — the plan this implements
