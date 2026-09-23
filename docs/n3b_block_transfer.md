# N3b: block-transfer cross-decoding

**Status:** implemented and validated on synthetic data; not yet run on real data.
**Spec:** `docs/analysis_plan_concurrent_regulation.md` §4. Controls and decision rules: `docs/cross_decoding_controls.md`.
**Run it:** `cd dcc_scripts/decoding && bash submit_block_transfer_dcc.sh` (Part 3).

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

This doc has three parts:
- **Part 1:** the concepts, as answers to the design questions;
- **Part 2:** what was built;
- **Part 3:** how to run it and read the output.

The same change also fixed the A4 padding bug (see 1.1 and "Changes to A4" at the end), and it removed dead code.

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
  - **A4 does not balance.** Before this change, its IR/IS/CR/CS cells looked equal only because `LabeledArray.from_dict` pads every condition to the same height with all-NaN rows.
  - At test time those rows were filled with random noise and scored, so rare cells such as I25 were mostly padding. A4 now drops them (see "Changes to A4" at the end).
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

## Part 2: how it is built

### The call path

Everything below the job function is the ordinary cross-decoding pipeline; the new pieces are marked **new**.

```
submit_block_transfer_dcc.sh                       new: ANALYSIS=block_transfer, ROI, ELECTRODES
 └ sbatch_stability_flexibility_cross_decoding_dcc.sh
    └ run_stability_flexibility_cross_decoding_dcc.py     environment variables -> args
       └ stability_flexibility_cross_decoding_dcc.main(args)
          └ run_block_transfer_job(args)                  new: loads the ROI, loops X1-X3 x centering
             ├ _build_roi_arrays                          the ROI pseudopopulation (sig or all electrodes)
             ├ block_transfer.run_block_transfer          new: balance -> center -> the 2x2
             │  ├ cross_decoding.build_cross_decoding_arrays   trials + labels, padding rows dropped
             │  ├ cross_decoding.make_decoder                  PCA -> LDA with equal priors
             │  └ Decoder.cv_cm_jim_window_shuffle(test_only=...)   folds -> confusion matrices
             ├ _summarise                                 accuracy + cluster test vs the shuffle null
             └ summary.txt, block_transfer.json, block_transfer_traces.npz, figures
```

### What changed, file by file

- **`src/analysis/decoding/decoder.py`: `test_only`.** A new optional argument of `cv_cm_jim_window_shuffle`: a True/False flag per trial. Flagged trials are never trained on. The folds are cut from the unflagged trials only, and every fold's classifier is scored on all the flagged trials. With `test_only=None` (the default) the function behaves exactly as before. The whole change is the few lines that pick `train_idx` and `test_idx` in the fold loop.
- **`src/analysis/decoding/cross_decoding.py`:**
  - `build_cross_decoding_arrays` drops padding rows (see 1.1) through `_drop_padding_rows`.
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

### Tests

`tests/analysis/decoding/test_block_transfer.py`:

- **The decoder:** `test_only` trials are never trained on and are always the whole test set, and a transfer trains on exactly the folds of the matching within-level decode.
- **Balancing and centering (no `ieeg` needed):** the 8 balance groups, equal counts after balancing, and the balance-then-center order. The class midpoint lands at 0; centering the raw 3:1 trials would put it a quarter of the class difference off.
- **Planted answers on synthetic data:**
  - a block-invariant code transfers as well as it decodes;
  - a block-specific code fails X1 in both directions but still passes X3;
  - a tonic block offset breaks only uncentered transfer;
  - within-level accuracies don't move with centering.
- **End to end:** `main()` with `analysis='block_transfer'` on synthetic data writes all its outputs.

`tests/analysis/decoding/test_cross_decoding.py` adds tests for the padding fix and for equal priors on a 3:1 class split.

To run them outside the cluster: `pip install -e . pytest`, then `python -m pytest -o addopts="" tests/analysis/decoding -q`.

---

## Part 3: running it and reading the output

### Running

- **Synthetic dry run** (seconds to minutes, anywhere): `ANALYSIS=block_transfer DATA_SOURCE=synthetic SYNTHETIC_CODE=block_specific N_REPEATS=10 WINDOW_SIZE=16 STEP_SIZE=8 python dcc_scripts/decoding/run_stability_flexibility_cross_decoding_dcc.py`. The planted answer is "X1 fails, X3 transfers". The default `SYNTHETIC_CODE` plants a block-invariant code, so everything should transfer.
- **Real data on the DCC:**
  ```
  cd dcc_scripts/decoding
  bash submit_block_transfer_dcc.sh                                   # lpfc, sig electrodes
  ROI=acc ELECTRODES=all bash submit_block_transfer_dcc.sh            # another region, every electrode
  EPOCHS_ROOT_FILE=<root with the sig_chans you mean> bash submit_block_transfer_dcc.sh
  ```
- **Cost:** 3 designs × 2 centerings × 4 cells × (true + shuffle) × `N_REPEATS` resamples × `N_SPLITS` folds × windows. That is about as many classifier fits as the A4 battery, so it fits the same 16 h allocation.
- **Check the log first.** For each design it prints the real trials available per class per physical block, and how many of each are kept per resample. That is the go/no-go of the concurrent-regulation plan §4.4. If the kept number is in the low teens, expect a null and say so up front.

### Outputs

Written to `results/<EPOCHS_ROOT_FILE>/block_transfer_<ROI>_<ELECTRODES>_w<W>s<S>/stimulus_experiment_conditions/`:

| File | Contents |
|---|---|
| `summary.txt` | Read this first. For every design and centering: the 2×2 table, the ceiling test, the go/no-go line, any artifact flag, and the reading guide |
| `block_transfer.json` | The same numbers per cell, plus the run's settings and the group sizes |
| `block_transfer_traces.npz` | Accuracy traces, windows × resamples. Keys look like `X1_centered_25to75_true` / `..._shuffle` |
| `<design>_<centering>_<train>to<test>_<roi>_block_transfer.{pdf,png,eps}` | The transfer into a level, drawn against that level's own within-level accuracy (its ceiling) and the shuffle null. Bars mark windows where the transfer beats shuffle |

### Reading `summary.txt`

Each design gets a block like this one:

```
X1_uncentered: congruency, trained in one incongruent_proportion level and tested in the other
   balanced to 40 trials per class per physical block (available: {...})
   post-stimulus mean accuracy (significant windows vs shuffle, post/pre):
     train | test               25%               75%
              25%       0.812 (3/0)       0.515 (0/0)
              75%       0.506 (0/0)       0.803 (3/0)
   25% -> 75% vs within 75%: below that ceiling in 3 windows
   75% -> 25% vs within 25%: below that ceiling in 3 windows
   CEILING: both within-level decodes beat shuffle -> interpretable
```

- **The table:** rows are the level trained on, columns the level tested on. The diagonal is the within-level ceiling; off the diagonal is transfer. Each cell shows the mean accuracy after stimulus onset, then (significant post / pre windows against the shuffle null).
- **The "vs within" lines** compare each transfer with the ceiling of the level it is tested on (see 1.4).
- **The CEILING line** is the go/no-go (the first row of the table in 1.6).
- **An ARTIFACT FLAG line** appears if any cell is significant before stimulus onset.
- **Read the pairs together:** a design's uncentered and centered blocks, then X1 against X3.

---

## Known issues (flagged, not fixed here)

- **A4(0b) cross cells** (congruency by switch proportion, switch type by incongruent proportion) have the same class-mix confound as unbalanced X3 (see 1.2). This predates the padding fix. For within-block numbers, use N3b's balanced within-level cells.
- **Resamples aren't independent subjects,** so cluster p-values are optimistic. X1-vs-X3 on the same trials is the load-bearing contrast.
- **`mixup2` crashes** if a subject has no same-class trial in a training fold. This is rare, but possible for low-accuracy subjects.
- **Seeds don't control mixup or the test-noise fill,** which use the global `np.random`.
- **Follow-ups if transfer sits at chance:** the PCA basis is fit on the training level (`cross_decoding_controls.md` §4.3). X4 and X5 aren't built: X4 needs a letter-identity condition set.
- **Stale material:** `src/analysis/decoding/cross_decoding_tutorial.ipynb` and `docs/skeletons/a4_cross_decoding.py` describe functions that no longer exist.
- **`TEMPGEN_GROUPS=both,all`** is cut at the comma by `sbatch --export` in the A4 submit script.

## Changes to A4

The padding fix (1.1) changes every A4 design, so A4 numbers from before this change aren't comparable with new runs.

- `build_cross_decoding_arrays` now drops the all-NaN padding rows. Before, they were filled by mixup when training and scored as noise when testing, which pulled accuracies toward chance, most of all for rare cells.
- `run_cross_decoding` now uses equal LDA priors (`make_decoder`). Without the padding, the within-block decodes (A4(0)) have their real 3:1 class ratio, and training-frequency priors would lean toward the majority class.
