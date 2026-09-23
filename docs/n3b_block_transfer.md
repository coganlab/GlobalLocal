# N3b: block-transfer cross-decoding

**Status:** planned; implementation in progress.
**Spec:** `docs/analysis_plan_concurrent_regulation.md` §4. Controls and decision rules: `docs/cross_decoding_controls.md`.

N3b trains a classifier in one kind of block and tests it in another:

| Design | Decoded contrast | Train → test | Role |
|---|---|---|---|
| **X1** | congruency | 25%-incongruent blocks → 75%-incongruent blocks | primary (LWPC) |
| **X2** | switch type | 25%-switch blocks → 75%-switch blocks | primary (LWPS) |
| **X3** | congruency | 25%-switch blocks → 75%-switch blocks | positive control for X1 |

Every design runs in both directions.

The existing decoder (`Decoder.cv_cm_jim_window_shuffle`) always cuts its train and test sets out of one pool of trials with random folds. It has no way to say "train on these trials, test on those". N3b needs exactly that.

**Goals:**
- the smallest change that reuses the existing pipeline;
- run on all significant electrodes of an ROI (LPFC by default), with no electrode groups;
- explain the design choices well enough that the code can be maintained by someone who didn't write it.

**Decisions:**
- implement with a walkthrough (this doc);
- delete dead code in a separate commit;
- fix the A4 NaN-padding bug as well (see 1.1).

## The data this has to work with

From `combinedData.csv`: each subject has 4 physical blocks of 112 trials, one block per type.

| Block | Incongruent | Switch | Accurate incongruent / congruent trials per subject (mean) |
|---|---|---|---|
| A | 75% | 25% | 71 / 26 |
| B | 75% | 75% | 66 / 24 |
| C | 25% | 25% | 22 / 79 |
| D | 25% | 75% | 20 / 73 |

Inside a block, the classes run about **3:1, and the majority class flips between the 25% and 75% levels**. Most of the design choices below follow from that.

---

## Part 1: concepts

### 1.1 Stratifying vs balancing

- **Stratifying a split:** dealing the trials into folds so that every fold is a small copy of the whole set. If 25% of the trials are incongruent, every fold is about 25% incongruent. It never adds or removes a trial. It only stops a random split from, say, putting most of the rare trials in one fold.
- **`stratify_labels`** (in `cv_cm_jim_window_shuffle`) is the label the folds are kept proportional on. The default is the training labels. A4 passes `strata`, which is the index of the condition each trial came from (0–15 for the 16 cells). So every fold has the same mix of all 16 cells, and therefore of congruency, switch type and both proportions at once. That matters in label transfer, which scores on switch type: folds balanced only on congruency could come out lopsided on switch type.
- **Balancing is a different thing.** It changes the data: you subsample so that groups have equal counts.
  - **A4 does not balance.** Its IR/IS/CR/CS cells look equal only because `LabeledArray.from_dict` pads every condition to the same height with all-NaN rows.
  - At test time those rows are filled with random noise and scored. Rare cells such as I25 are mostly padding. This is the bug fixed alongside N3b.
- **N3b needs both:** balancing, then stratified folds.

### 1.2 What to balance

- **Balancing is needed because of the 3:1 ratio inside a block.** LDA's default priors are the training class frequencies. For a weak effect, a 3:1 prior pushes nearly every prediction to the majority class. In the simple 1-D case with d′ = 0.5, balanced accuracy is 0.51 instead of 0.60, even though the signal is there. In X1 the majority also flips between training and testing.
- **Balance incongruent vs congruent within each of the four physical blocks A–D.** That's 8 groups (class × incongruent proportion × switch proportion), all equal. It keeps C25, I25, C75 and I75 equal, and it also stops X3 from being confounded:
  - In the 25%-switch blocks (A+C), about 76% of incongruent trials come from block A and about 76% of congruent trials from block C.
  - So without per-block balancing, the "congruency" classifier can learn the A-vs-C difference, which is the incongruent-proportion block effect.
  - That effect carries over to B-vs-D, so the control would look good for the wrong reason.
- **This costs about 5% more trials than balancing only C25/I25/C75/I75.** The 4-group version keeps about 42 trials per class per level per subject; the 8-group version keeps about 40.
- **X1 and X3 then run on exactly the same trials in every resample.** That makes X3 a clean control.
- **Do not balance on switch type** (or on congruency for X2). It is 25/75 inside a block by design, so balancing it would cut the data in half. The folds are stratified on the full 16-cell condition instead.

### 1.3 Trial loss

- **What gets dropped is only the extra majority trials.** Each repeat draws a new random balanced subsample, so over 10 repeats almost every trial gets used.
- **The real limit is the minority class** (about 20 incongruent trials per subject per 25%-incongruent block), and no method removes it.
- The per-channel minimum in `subsample_to_min_trials_per_condition` (the "~22/class" in `analysis_simplification_plan.md` §1.2) belongs to the ordinary decoding job. A4 and N3b never call it.

### 1.4 Why there are still folds

- **"Train on all the 25% trials, test on all the 75% trials" is valid for the transfer number on its own.** The two sets share no trials, so there is no double-dipping.
- **Folds are needed for three other reasons:**
  - **The ceiling:** a transfer accuracy only means something next to the within-block accuracy (`cross_decoding_controls.md` §2), and the within-block accuracy has to be cross-validated.
  - **Matching:** if the transfer classifier trains on 100% of the 25% trials but the ceiling classifier trains on 80%, the two numbers aren't comparable.
  - **Repeats:** you need a spread of values for the shuffle null and error bars.
- **The design:** cut folds only inside the training level. Each fold's classifier is scored twice: on its held-out 25% trials (the ceiling), and on all the 75% trials (the transfer). Same folds and same seed, so the ceiling and the transfer come from literally the same classifier.
- **Which ceiling to compare against:** compare transfer 25→75 with **within-75**, because both are scored on the same test trials.
  - The congruency code can simply be weaker in 75%-incongruent blocks (that is the LWPC effect).
  - In that case 25→75 drops while the axis is unchanged.
  - A real change of axis makes **both directions** fall short of their test block's ceiling.
  - A drop in only one direction means the training block's code is weaker, not that the axis changed.
  - The job reports the full 2×2 table (train level × test level).

### 1.5 Centering

- **Centering subtracts one vector per block level.** It is the same vector for every trial in that level, congruent and incongruent alike. It moves the whole cloud, and cannot rotate the direction that separates C from I inside the block.

```
   same axis, whole block shifted        different axis
   25%:  C●   ●I                         25%:  C●   ●I
   75%:            C●   ●I               75%:       ●I
                                                    ●C
   centering lines them up -> transfers  centering can't fix it -> still fails
```

- **If C25/I25 and C75/I75 separate along different directions** (the reconfiguration hypothesis), that survives centering and X1 still fails.
- **What centering removes is the tonic shift of the whole block.** Examples: all high-gamma higher in 75% blocks, or the pooled-baseline artifact (`analysis_simplification_plan.md` §1.4). A shift like that can make transfer fail even when the axis is identical, which is why an uncentered null can't be read on its own.
- The tonic effect itself is not lost. It is simply a different claim, and X5 or the univariate block effect measures it.
- **Within-level accuracies are unchanged by centering.** The same vector is subtracted from both the training and the test trials. That makes them a built-in check.
- **Pitfall: balance first, then center.**
  - The raw mean of a 3:1 block sits a quarter of the C–I distance away from the class midpoint, on the majority side.
  - The majority flips between levels, so centering on raw means shifts the two levels half the C–I difference apart, exactly along the decoding axis.
  - That fakes "X1 and X2 fail, X3 transfers", which is the pattern the analysis is looking for.

### 1.6 Reading the results

Read these centered and uncentered, in both directions:

| within (test level) | transfer, uncentered | transfer, centered | meaning |
|---|---|---|---|
| at chance | – | – | Can't interpret: there was nothing to transfer |
| above | ≈ within | ≈ within | Same code |
| above | < within | ≈ within | Same axis; the blocks differ by a tonic shift |
| above | < within in both directions | < within in both directions | Block context reorganizes the code. Only counts if X3, on the same trials, transfers |

Any pre-stimulus windows that come out significant are an artifact flag (`cross_decoding_controls.md` §6).

### 1.7 Electrodes: all significant electrodes of an ROI

- **The N3b job** (`dcc_scripts/decoding/submit_block_transfer_dcc.sh`) defaults to `ROI=lpfc ELECTRODES=sig`. It runs no CSV, power-trace or ANOVA step and forms no electrode groups. `ELECTRODES=all` keeps every electrode in the ROI.
- **What "sig" means:** the electrode's high-gamma during the stimulus beats its pre-stimulus baseline (a per-electrode cluster test done at epoching). It's read from `sig_chans_<subject>_<EPOCHS_ROOT_FILE>.json`. It says nothing about congruency or switching, so there is no double-dipping with N3b.
- **`EPOCHS_ROOT_FILE` decides which significance file is used.** The A4 submit default has no `_filterbank_hilbert`; the ANOVA-label CSV folders were computed with it. Set it on purpose.
- **Gotcha in the existing A4 job:** with its default `ELECTRODE_DEFINITION=csv`, it loads every electrode in the ROI, significant or not (`_build_roi_arrays` in `stability_flexibility_cross_decoding_dcc.py`), even though output folders say `sig`.

### 1.8 How to approach changing this codebase

1. **Follow one call path, not files.**
   - `submit_*.sh` → `sbatch_*.sh` → `run_*_dcc.py` (turns environment variables into `args`) → `*_dcc.py main(args)` (loads data → ROI arrays → loops over designs)
   - → `cross_decoding.py` (arrays → label vectors) → `decoder.py` (folds → fit → confusion matrices)
   - → `accuracy_stats.py` (confusion matrices → accuracy → cluster test vs shuffle) → saving and plots
2. **Find the one step that differs.** For N3b, that is which trials train and which test. Everything else is reused.
3. **Add the new behavior as an optional argument that is off by default.** Every existing call stays identical, and the existing tests prove it.
4. **Test on synthetic data with a planted answer before touching real data.** Plant the confounds you're worried about too (3:1 classes, block offsets).

---

## Part 2: implementation plan (one commit per step)

### Step 1: `decoder.py`, new `test_only` option on `cv_cm_jim_window_shuffle`

- New keyword `test_only=None`: a boolean per trial.
- Trials marked True are never trained on. Folds are cut from the other trials, and every fold's classifier is scored on **all** the marked trials.
- The fold loop becomes:
  ```python
  trainable = np.arange(len(labels)) if test_only is None else np.flatnonzero(~test_only)
  fixed_test = None if test_only is None else np.flatnonzero(test_only)
  for f, (tr, te) in enumerate(splitter.split(np.zeros(len(trainable)), strat[trainable])):
      train_idx = trainable[tr]
      test_idx = trainable[te] if fixed_test is None else fixed_test
  ```
  plus a length check and a docstring paragraph.
- **Unchanged:** shuffle (still permutes training labels only), `frac_train`, `folds_as_samples`, temporal generalization (it gives cross-block temporal generalization for free), and the output shapes.
- With `test_only=None` the folds are identical to before, because the splitter only reads the number of trials from X.

### Step 2: fix A4 padding (`cross_decoding.py`)

- **`build_cross_decoding_arrays`:** drop rows that are NaN on every channel, which is pure padding, and skip conditions left with no rows. Rows missing only some subjects are kept and imputed as before (mixup when training, noise when testing).
  - This fixes every A4 design.
  - It also covers the all-NaN rows that `_restrict_to_electrodes` creates.
- **Equal LDA priors in a shared `make_decoder(...)` factory**, used by `run_cross_decoding` and N3b.
  - Why this is needed: the padding was hiding the 3:1 class ratio in A4's within-block decodes. Without equal priors those numbers would lean toward the majority class.
  - Confirm against the installed `ieeg` that `PcaEstimateDecoder` uses the `clf` instance as-is.

### Step 3: N3b library, new `src/analysis/decoding/block_transfer.py`

**`prepare(roi_labeled_arrays, roi, cells, strings, block_col)`** (no `ieeg` needed):
- Calls `build_cross_decoding_arrays(arrays, roi, strings, strings)`.
- Looks up each trial's factor levels through `cells`.
- Returns the data, labels, strata, each trial's `block_col` level, and its balance group (class × incongruent proportion × switch proportion, 8 groups).

**`balanced_subsample(groups, rng)`:** the same number of trials from every group (the smallest group's size), in their original order.

**`run_block_transfer(...)`:**
```python
for r in range(n_resamples):
    k = balanced_subsample(groups, rng)             # new balanced draw every repeat
    Xk = X[k].copy()
    if center:                                      # AFTER balancing -> mean = class midpoint
        for lvl in (lo, hi): Xk[block[k] == lvl] -= np.nanmean(Xk[block[k] == lvl], axis=0)
    dec = make_decoder(cats, 2, n_splits=n_splits, n_repeats=1, random_state=seed + r)
    for tr, te in ((lo, lo), (lo, hi), (hi, hi), (hi, lo)):
        use = np.isin(block[k], (tr, te))
        test_only = None if tr == te else (block[k][use] == te)   # within = ordinary CV
        for shuffle in (False, True):
            cm = dec.cv_cm_jim_window_shuffle(Xk[use], y[k][use], normalize='true', obs_axs=0,
                     window=window, step_size=step_size, stratify_labels=strata[k][use],
                     test_only=test_only, shuffle=shuffle)
```
- Same seed ⇒ within(lo) and lo→hi use identical folds.
- X1 and X3 have identical groups, so they draw identical resamples.
- The output feeds straight into `compute_accuracies` and the cluster test.

**Synthetic generator:** `synthetic_roi_labeled_arrays` gets three new keywords.
- `block_code='same'|'specific'`: with `'specific'`, the congruency axis in 75%-incongruent cells is a third axis at right angles to the first.
- `block_offset`: a tonic shift along the congruency axis applied to every 75%-incongruent trial.
- `design_proportions`: when True, cell sizes follow the 25/75 design, i.e. 3:1 classes within a block.
- The default output stays byte-identical.

### Step 4: the N3b job

**`stability_flexibility_cross_decoding_dcc.py`:**
- `main()` dispatches to `run_block_transfer_job(args)` when `ANALYSIS=block_transfer`.
- The job loads data with the existing `_build_roi_arrays` (`ELECTRODES` alone decides sig vs all).
- It prints the 8 group sizes and the balanced n; this is the go/no-go from the concurrent-regulation plan §4.4.
- It runs X1, X2 and X3, each uncentered and centered.
- Each cell goes through the existing `_summarise` (cluster test vs shuffle).
- It adds a ceiling test (within the test level > transfer into it).
- **Outputs:**
  - `block_transfer.json`
  - `block_transfer_traces.npz`
  - `summary.txt`
  - figures made with `plot_accuracies_nature_style`

**Runner:** `run_stability_flexibility_cross_decoding_dcc.py` reads `ANALYSIS=a4|block_transfer` (default `a4`). For N3b, `N_REPEATS` means the number of balanced resamples.

**Submit script:** new `submit_block_transfer_dcc.sh`, one job per call through the existing sbatch wrapper.

**Tests:** new `tests/analysis/decoding/test_block_transfer.py` checks that:
- `test_only` trials are never trained on;
- a block-invariant code transfers;
- a block-specific code fails X1 but passes X3;
- a block offset breaks only uncentered X1;
- within-level cells are identical centered and uncentered;
- balancing and centering do what 1.2 and 1.5 require;
- an end-to-end synthetic run of the job writes its outputs.

### Step 5: remove dead code

- **`decoder.py`:** the commented-out old method, `fit_predict`, `cv_cm_return_scores`, `calculate_scores`, and unused imports. Also fix the comments claiming a StandardScaler: the `ieeg` pipeline is PCA → LDA.
- **`cross_decoding.py`:** `DEFAULT_CELL_COLS`, `_congruency_label`, `_switch_label`, `CONTRASTS`, `resolve_contrast`, and their test. They coded incongruent as 1 while the rest of the module codes it as 0.

### Step 6: update this doc

Replace this plan with a walkthrough of what was built, how to run it, and how to read `summary.txt`.

---

## Known issues (flagged, not fixed here)

- **A4(0b) cross cells** (congruency by switch proportion, switch type by incongruent proportion) have the same class-mix confound as unbalanced X3 (see 1.2). This predates the padding fix. For within-block numbers, use N3b's balanced within-level cells.
- **Resamples aren't independent subjects,** so cluster p-values are optimistic. X1-vs-X3 on the same trials is the load-bearing contrast.
- **`mixup2` crashes** if a subject has no same-class trial in a training fold. This is rare, but possible for low-accuracy subjects.
- **Seeds don't control mixup or the test-noise fill,** which use the global `np.random`.
- **Follow-ups if transfer sits at chance:** the PCA basis is fit on the training level (`cross_decoding_controls.md` §4.3). X4 and X5 aren't built: X4 needs a letter-identity condition set.
- **Stale material:** `src/analysis/decoding/cross_decoding_tutorial.ipynb` and `docs/skeletons/a4_cross_decoding.py` describe functions that no longer exist.
- **`TEMPGEN_GROUPS=both,all`** is cut at the comma by `sbatch --export` in the A4 submit script.

## Verification

1. Run the decoding tests: `python -m pytest -o addopts="" tests/analysis/decoding/test_cross_decoding*.py tests/analysis/decoding/test_block_transfer.py -q`.
2. Run a synthetic end-to-end job: `ANALYSIS=block_transfer DATA_SOURCE=synthetic SYNTHETIC_CODE=block_specific N_REPEATS=2 N_PERM=50 WINDOW_SIZE=16 STEP_SIZE=8 python dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py`. `summary.txt` should show X1 transfer ≈ 0.5, X3 transfer ≈ within, and identical within-level cells centered vs uncentered.
3. On the DCC: `cd dcc_scripts/decoding && bash submit_block_transfer_dcc.sh`. Check the group-size and go/no-go lines in the log first.
