# Nested electrode selection — design plan

How to make the drill-down **diagonal** (select on congruency → measure
congruency) reportable instead of circular, by estimating selection and effect on
disjoint trials. Covers both the decoding (nested inside the CV loop) and the
power traces (repeated split-half), plus how to reconcile them with the
full-trial selection used for the anatomy panel.

Companion to [`analysis_guide.md`](analysis_guide.md) §14.1 (the "ignore the
diagonal" rule), §17 (the decoding designs), and §21 (disjoint trial splits).
Figure context is in [`figure_plan.md`](figure_plan.md) F3/F4.

## Which cells actually need a split

The rule is the same for decoding and for power traces, so apply one table, not
two procedures. A cell is circular only when the **selection contrast and the
measured contrast are the same**:

| Selection | Measured | Circular? | Why |
|---|---|---|---|
| task-responsiveness | congruency or switch main effect | **no** | responsiveness is the condition *mean*, the contrast is the *difference* — orthogonal |
| congruency main effect | congruency main effect | **yes** | selection contrast = measured contrast |
| congruency main effect | switch main effect | **no** | off-diagonal |
| congruency main effect | LWPC / LWPS interaction | **no** | Type III interaction row is orthogonal to both main effects (§14.1) |

Two consequences worth internalizing before writing any code:

- **F3 needs nothing.** Its electrodes are selected on task-responsiveness, which
  is orthogonal to every condition contrast plotted there.
- **Under the hierarchical design** (define groups on main effects, test
  adaptation within them — see `figure_plan.md`), the **adaptation traces carry
  the paper's claim and are already clean.** Only main-effect-on-its-own-selection
  needs a split: roughly 2 cells out of 12.

The orthogonality claims rest on balanced cells, and the design is deliberately
75/25. Verify empirically with the permutation check in `figure_plan.md` rather
than assuming it.

### Where the bias sits

Selection bias is concentrated in the **selection window**. Selecting on a 0–1 s
window mean and plotting −1 to 1.5 s leaves the pre-stimulus portion only
indirectly inflated (through overall electrode responsiveness), while the 0–1 s
portion is directly inflated. This is why an uncorrected trace can look entirely
plausible and still be unusable for statistics — the distortion is local and does
not announce itself.

## Why this is correct, and why it is the *conservative* choice

Selection is part of the fitting procedure. If the entire procedure — selection
**and** classifier training — sees only the training trials, and the held-out
trials are used solely for scoring, the accuracy estimate is unbiased. Training
the classifier on the same trials used to select electrodes is fine; only the
*test* trials must be untouched.

**Nested feature selection is the standard correction, not a novel risk.** The
pattern reviewers flag is selection performed *outside* the CV loop, on all
trials — which is what the pipeline does today. One sentence in Methods settles
it:

> Electrode selection was nested within the cross-validation loop: for each fold,
> electrodes were selected using only that fold's training trials, and the
> held-out trials were used solely for evaluation.

Standard reference for the failure mode this fixes: Kriegeskorte et al. (2009),
circular analysis in systems neuroscience.

**5-fold already gives the 80% train fraction.** `StratifiedKFold(n_splits=5)`
trains on 80% and tests every trial exactly once. Do **not** reach for
`frac_train` / `StratifiedShuffleSplit` — those resamples are not a partition, so
trials are tested a variable number of times for no benefit here.

## The insertion point

`Decoder.cv_cm_jim_window_shuffle` in `src/analysis/decoding/decoder.py`
(fold loop at ~line 317). The loop already forms exactly what is needed:

```python
for f, (train_idx, test_idx) in enumerate(splitter.split(data, strat)):
    x_train = data[train_idx]
    y_train = labels[train_idx].copy()
    x_test  = data[test_idx]
    y_test  = (labels if labels_test is None else labels_test)[test_idx]

    if shuffle:
        rng.shuffle(y_train)

    cm_windowed = self._window_and_predict_minimal(...)
```

**Electrode selection is a channel-axis mask; the CV split is a trial-axis
partition.** They are orthogonal, so nothing about the fold structure changes.
Insert a mask computation after the train/test arrays exist and apply it to both.

Two facts make this much cheaper than a restructure:

**1. `stratify_labels` already carries the condition cell.** Per §17,
`build_cross_decoding_arrays` returns `strat` = the joint
`congruency × switchType × incongruent_proportion × switch_proportion` cell. So
`strat[train_idx]` *is* the selection ANOVA's design, already in scope. The only
new argument is the selection callback itself.

**2. Compute the ANOVA from `x_train` directly — not from A1's long table.**
`put_data_in_labeled_array_per_roi_subject` **randomizes trial ordering within
each subject** before NaN-padding and concatenating along channels. Decoding-array
rows therefore do **not** map back to the segregation module's table by index, and
trying to join them is where this task would turn expensive. Computing a
window-mean 2×2 ANOVA per channel straight off `x_train` avoids the mapping
entirely, and has the side benefit that selection and decoding share
preprocessing.

### Proposed API

Keep the ANOVA logic where it lives and pass it in:

```python
def cv_cm_jim_window_shuffle(..., select_fn=None, select_window=None):
    """
    select_fn : callable or None
        ``select_fn(x_train, cells_train) -> bool ndarray (n_channels,)``
        Called once per fold on training trials only. ``None`` keeps the
        current behaviour (decode every channel passed in).
    select_window : (start, stop) sample indices, or None
        Time window the selection statistic is computed over. Fixed a priori;
        see "One selection per fold" below.
    """
```

Inside the loop:

```python
    if select_fn is not None:
        cells_train = strat[train_idx]
        mask = select_fn(x_train[..., sl], cells_train)   # sl = select_window
        if mask.sum() == 0:
            n_empty_folds += 1
            continue                     # record, don't silently average it in
        x_train = x_train[:, mask]
        x_test  = x_test[:, mask]
        fold_masks.append(mask)          # for the stability report
```

## Four design decisions that are not optional

### 1. The null must run selection too

`shuffle=True` currently permutes `y_train` *after* the split. If `select_fn`
reads `cells_train`, permuting `y_train` leaves selection untouched — the null
holds selection fixed while the observed pipeline does not.

Point estimates stay unbiased either way. The problem is **variance**: the guide's
whole rationale for the refit-under-shuffle null is that it "carries the variance
of the entire estimation pipeline," and `time_perm_cluster` forms its cluster
threshold from that variance. A null missing the selection step is too tight, and
significance is inflated.

Fix: draw **one permutation of trial indices per fold and apply it to both**
`y_train` and `cells_train`. Because `y_train` is a component of the cell,
permuting them together keeps the two coherent while decoupling both from the
neural data:

```python
    if shuffle:
        perm = rng.permutation(len(y_train))
        y_train = y_train[perm]
        cells_train = cells_train[perm]      # select_fn sees the permuted cells
```

For the **off-diagonal** cells, selection is already orthogonal to the scored
contrast, so holding selection fixed and permuting only the decoded labels is a
valid and slightly more powerful null. Permuting both is conservative and uniform
across cells — prefer it unless power becomes limiting, and say which you used.

### 2. Top-k, not a threshold

An FDR-thresholded mask yields a different channel count per fold — sometimes
zero. Consequences: PCA-at-X%-variance operates on a different feature space each
fold, and the null's dimensionality can differ systematically from the observed
(shuffled labels select fewer channels), which biases the comparison the null
exists to make.

**Select the top-k channels by F statistic instead.** Constant `k` across folds
*and* across the null removes both problems and makes the empty-mask case
impossible. Choose `k` from the full-data selection (e.g. the count at your
reported α) and state it as fixed a priori. Report the threshold-based version in
supplement for continuity with the classification panel.

### 3. One selection per fold, on a fixed a priori window

Do **not** re-select per sliding time window. Per-window selection multiplies cost
by `n_windows` and makes the electrode set time-varying, so the accuracy trace no
longer describes a single population.

Select once per fold on a fixed window — use A1's post-stimulus window (0–1 s) so
the selection matches the taxonomy — and apply that mask across all time windows,
including pre-stimulus. Evaluating pre-stimulus accuracy on post-stimulus-selected
electrodes is still unbiased (selection used training trials only), and it is
*diagnostically useful*: it gives a clean read on §17's impossible pre-stimulus
congruency decoding without selection as a confound.

### 4. NaN padding makes per-fold selection noisy, asymmetrically

Subjects are NaN-padded to the per-condition max trial count. A low-trial
subject's fold-training rows can be mostly padding, so its per-electrode ANOVA in
that fold rests on very few real trials — and this is **worse in the 25% cells**,
which have fewer real trials to begin with.

Top-k contains the downstream damage but not the underlying noise. Before
trusting any diagonal result:

- log the per-fold count of real (non-NaN) trials per channel per condition cell,
- set a minimum-real-trials floor per channel and drop channels below it *before*
  ranking,
- report the distribution.

## Power traces — repeated split-half, not cross-validation

The diagonal power traces need the same independence, but **not** the same
machinery. Nothing is being fitted, so there are no folds and no train/test
asymmetry — just repeated random splits with the held-out traces averaged.

```
for split in range(n_splits):
    sel_trials, trace_trials = stratified_half_split(trials, cells, rng)
    mask  = select_electrodes(x[sel_trials],   cells[sel_trials])
    trace = condition_effect(x[trace_trials, mask], cells[trace_trials])
    traces.append(trace)
grand = mean(traces)
```

Three differences from the decoding case:

**Use 50/50, not 80/20.** For decoding, more training data buys a better
classifier, so the (k−1)/k of 5-fold is right. For a trace there is no model —
selection only needs enough trials to rank electrodes stably, and the trace wants
as many trials as it can get. 50/50 balances those far better. Stratify the split
on the full condition cell so both halves keep the 75/25 structure.

**The null goes through the same procedure.** As with the decoding null: if the
observed trace is select-then-average-over-splits, the permutation null must be
too, or `time_perm_cluster` forms its threshold from variance the observed
statistic does not have. Permute the cells once per split and run the identical
select-then-average path.

**Check the rare cells survive halving first.** The 25%-incongruent-within-
25%-switch cell is the binding constraint. Count real (non-NaN) trials per cell
per subject before choosing the fraction; if halving empties that cell for several
subjects, either raise the selection fraction or restrict the split-based
treatment to the cells that can support it and mark the rest descriptive.

Splits overlap — each trial lands in many held-out halves — so the averaged trace
is unbiased but its across-split spread is **not** an independent-sample variance.
Do not build error bars from the split distribution; get inference from the
matched permutation null above.

## What changes in the reporting

With per-split selection there is no single "n=27 congruency-sensitive electrodes"
being measured — there is a distribution of per-split sets. This creates a real
inconsistency with the anatomy panel, which uses full-trial selection, and it has
to be addressed rather than hoped past.

**The inconsistency is principled.** The two panels are doing different jobs:

- The **anatomy/count panel** claims something about the *electrodes themselves* —
  how many are selective and where they sit. Full-trial selection is the *correct*
  estimator here, and it is not circular: the selection contrast is never
  re-tested, and the spatial distribution being tested is orthogonal to the
  selection statistic.
- The **decoding and trace panels** claim something about *effect magnitude within
  selected electrodes*. That is circular on the diagonal, hence the split.

Selection-for-description and selection-for-testing have genuinely different data
requirements. The problem is not using two procedures; it is showing two electrode
sets in one figure without saying so.

**The fix that does the most work: encode selection stability in the anatomy
panel.** Rather than binary membership, size or color each electrode by how often
it is selected across splits. This unifies the panels — the anatomy now displays
the same per-split selection the decoding consumes — adds information, and
answers the reviewer question before it is asked. It is nearly free once
`fold_masks` is stored.

It is also a result in its own right. If full-trial-selected electrodes are
recovered in ~90% of splits, the two panels describe the same population and the
mismatch is cosmetic. If it is ~40%, the taxonomy is much less stable than a
binary map implies, and you need to know that before writing conclusions from it.

The continuous effect-size correlation (§14) is the other bridge: it never
thresholds, so it has no set to mismatch. Reporting it alongside makes the
threshold-dependence of the taxonomy visible.

Concretely:

- **Classification / anatomy panel (F4a):** full-trial selection, electrodes
  encoded by selection frequency.
- **Trace and decoding panels (F4b/c):** per-split selection. Report median
  [min, max] set size, and state in the caption that selection differs from F4a
  and why.
- **Methods:** one sentence distinguishing the two uses. See the Methods sentence
  in the first section.

## Cost

Selection runs `n_repeats × n_splits × n_bootstraps × 2` (observed + null) times.
A window-mean 2×2 ANOVA per channel is cheap and fully vectorizable over channels
— budget it as negligible against the existing PCA→LDA fits.

**Use the window-mean ANOVA, not the per-timepoint cluster variant** (§14.2). The
cluster version inside a nested loop is not worth the compute, and the selection
statistic does not need to be the same one used for the taxonomy figure — it needs
to be cheap, fixed a priori, and computed on training trials.

## Acceptance tests

The first is the one that actually proves the change works; write it first.

1. **Planted selection bias returns to chance.** Generate data with *no* real
   condition effect. The current non-nested pipeline (select on all trials, then
   decode) must come out **above chance** — that is the bug being fixed. The
   nested pipeline must come out **at chance**. If the nested version is above
   chance on pure noise, the nesting is broken somewhere.
2. **Planted real effect is recovered.** Plant an effect on a known channel
   subset. Nested decoding must exceed chance, and `fold_masks` must overlap the
   planted set well above the rate expected by chance.
3. **Constant dimensionality.** With top-k, assert every fold's mask sums to `k`.
4. **Null matches the pipeline.** With `shuffle=True`, confirm `select_fn` is
   called with permuted cells — a spy/counter assertion is enough. Guards against
   silently regressing to a fixed-selection null.
5. **Off-diagonal unchanged.** Cross-decode results with `select_fn=None` must
   match the current implementation bit-for-bit. This change must not perturb the
   results that were already valid.

## Fallback if this is not built

The **off-diagonal is the load-bearing result and needs none of this** — selection
on congruency is already orthogonal to decoding switch type, so those cells are
valid today. The specificity claim ("process-specific electrodes fail to decode
the other process") is fully supported without any code change.

If the nested selection is skipped, report the diagonal explicitly as
selection-inflated and descriptive — label it in the panel, not just the caption —
and make the off-diagonal carry the inference. That is defensible; it just makes
F4 a weaker payoff figure, because the diagonal is the half a reader finds
intuitive.

**Recommendation: build it.** Roughly a day given that `stratify_labels` already
supplies the cells and the fold loop already forms `x_train`, and it converts the
intuitive half of the paper's punchline figure from "descriptive" to "reportable."
