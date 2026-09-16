# Cross-decoding controls — diagnosing a transfer that didn't work

Companion to
[`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md)
§4 and to [`analysis_guide.md`](analysis_guide.md) §17. This document is the
troubleshooting protocol: what to run, in what order, when a cross-decode comes
back uninformative — and what each outcome licenses you to say.

It covers both shapes of cross-decode in this project:

- **label transfer** — two labellings of the *same* trials (A4: train congruency,
  score switchType). `build_cross_decoding_arrays` + `labels_test=`.
- **block transfer** — one labelling, two *disjoint trial populations* (X1–X4:
  train congruency in 25%-incongruent blocks, test in 75%). Needs the new
  splitter (plan §4.2); the diagnostics below apply to it identically, plus §6.

---

## 1. First, name the failure

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

## 2. The interpretability floor — run this before any diagnosis

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

## 3. The positive-control ladder

Cheapest first. Each rules out a different failure and each is worth running
before concluding anything about a real null.

### 3.1 Synthetic ground truth (seconds, already implemented)

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

### 3.2 Split-half through the cross-decode code path (minutes)

Run the *same* condition, trained on a random half and tested on the other half,
routed through `run_cross_decoding` / the block-transfer splitter rather than
through ordinary CV. Transfer accuracy must match ordinary cross-validated
accuracy on those trials.

This is the sharpest cheap control, because it isolates the *plumbing* from the
*science*: same trials, same signal, same classifier, only the code path differs.
If a split-half transfer through the new splitter underperforms ordinary CV on
the same data, the splitter (or the subsampling, or the stratification) is
broken — stop and fix it before interpreting X1.

### 3.3 Occipital big letter across task (cheap, real data)

Train big-letter decoding on `task = global`, test on `task = local`. The
physical stimulus is identical and only attention differs, so visual cortex
should carry the big letter either way.

Caveat to write down and respect: **on congruent trials the big and small letters
are confounded**, so this is a control for the code path on real data, not a
claim about global-specific coding. Restrict to incongruent trials if you want it
clean, and report the trial counts.

### 3.4 Congruency across switch proportion (the control that matters)

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

---

## 4. F1 — transfer sits at chance

Work through these in order; each is cheap and each rules out a distinct cause.

### 4.1 Joint-cell trial counts

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

### 4.2 Block offset / nonstationarity (block transfer only, and it dominates)

Training in one block and testing in another means any tonic block-level HG
difference shifts the test cloud along a direction the classifier did not intend
to use. The baseline carries exactly that confound by construction: a random
0.5 s pre-stimulus baseline z-scored with statistics pooled across all trials, in
a design where `incongruentProportion` *is* the block
(`analysis_simplification_plan.md` §1.4).

**Check:** center features within block — per channel, per block, subtract that
block's mean over trials — and re-run. Report both versions.

- Transfer recovers after centering → the null was a DC shift, not code
  reconfiguration. The centered version is the one that answers the question.
- Transfer still null after centering → the geometric claim survives its most
  likely artifact.

Corollary worth stating in Methods: centering deliberately discards the tonic
block effect, which may itself be the proactive-control signal. That is
undecidable in a blocked design, which is why both versions are reported.

### 4.3 The PCA basis

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

### 4.4 NaN / mixup asymmetry

Train and test are imputed **differently**: `sample_fold` fills training NaNs
with `mixup2` (informed combinations) and test NaNs with i.i.d. Gaussian noise
(`decoder.py:71`, deliberately non-informative so imputation cannot leak class
information). That asymmetry is correct for ordinary CV, but it bites a transfer
whose test population draws more heavily on sparsely-covered subjects: the test
features are then substantially noise.

**Check:** per-subject channel coverage in the train population vs the test
population, and the fraction of test features that were NaN-filled. If the test
side is markedly sparser, restrict both sides to the subjects/channels present in
both and re-run.

### 4.5 Feature and decoder matching

Two decoders whose accuracies are compared must match on trial count, class
balance, CV folds, feature set, window, and step size. An LWPC decoder with more
trials than the LWPS decoder will look better for that reason alone. Subsample to
the common minimum and average over subsamples, or do not compare them.

---

## 5. F2 — transfer reliably below chance

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

## 6. F3 — significant where it cannot be

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

Also treat **transfer > within-condition accuracy** as an F3: a transferred axis
cannot beat an axis trained on the labelling it is scored against. That
combination means the two labellings are not actually crossed, or the test
population is contaminated with training trials.

---

## 7. Report this block with every cross-decode

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
```

The two lines that carry all the interpretive weight are **within-condition acc**
and **pre-stimulus cluster**. A reader who sees the first can tell whether a null
means anything; a reader who sees the second can tell whether a positive means
anything.

---

## 8. Decision tree

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
└── clean → report with its ceiling and its reverse direction
```

---

## 9. What a clean result looks like

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
