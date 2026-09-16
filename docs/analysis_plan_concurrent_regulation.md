# Analysis plan — concurrent regulation of stability and flexibility

**Status:** active wrap-up plan (2026-09). One-week scope: close out anatomy, add
the cross-decoding analyses that are cheap, and stop.

**Relationship to the other docs.** This supersedes the *framing* of
[`analysis_simplification_plan.md`](analysis_simplification_plan.md) (which
argued shared-vs-independent mechanisms) and the *narrative* of
[`figure_plan.md`](figure_plan.md) (which was built around a mixed-selectivity
drill-down). Everything those documents say about **estimators and their biases**
still holds and is not re-derived here — §2.2/§2.2b/§2.2c of the simplification
plan in particular. [`analysis_guide.md`](analysis_guide.md) remains the
description of the pipelines as built;
[`stability_flexibility_data_flow.md`](stability_flexibility_data_flow.md)
remains the A1–A7 walk-through. Cross-decoding troubleshooting lives in its own
document: [`cross_decoding_controls.md`](cross_decoding_controls.md).

---

## 0. The narrative

**We see concurrent regulation of stability and flexibility. The paper
characterizes that concurrent regulation in the brain.**

Four beats, in order:

| # | Beat | Carried by | Status |
|---|---|---|---|
| N1 | **Behavior.** Both adaptations are present in the same subjects, in the same sessions. | RT / error LWPC and LWPS | done |
| N2 | **lPFC high gamma carries both adaptation effects.** | LWPC and LWPS power traces, plus the direction tests (§2) | traces done, directions **missing** |
| N3 | **Those adaptation effects are decodable from distributed lPFC activity**, including information no single electrode supplies. | LWPC / LWPS decoding (§3), block-transfer cross-decoding (§4) | LWPC/LWPS decoding done; transfer **new** |
| N4 | **Are the two effects organized differently across cortex?** | per-electrode continuous scores → anatomy (§5–§8) | **new** |

Three framing rules that follow from it, and they change what gets written:

1. **Not combative.** The absence of cross-effects (congruency × switch
   proportion, switchType × incongruent proportion) is a *scoping* statement —
   "we therefore focus on the two within-process adaptation effects" — not a
   dissociation claim. Do not build a figure or a paragraph around the null
   cross-effects. Do not run the independent-vs-dependent argument at all.
2. **Decoding is not a multivariate-superiority claim.** The claim is the weaker
   and true one: *adding electrode b to electrode a yields information that
   neither provides alone.* That is what a pooled decoder shows, and it is not
   undercut by §1.1 of the simplification plan (cross-electrode covariance is
   destroyed by the pseudopopulation, so there is no trial-level-covariance claim
   to make — and we are not making one). Donos et al. 2022 is the citation for
   "a classifier detects structure the cluster test misses."
3. **"Overlapping tissue, separable codes" is a publishable answer**, and it is
   the most likely one in frontal cortex. Pre-commit to reporting it. The anatomy
   analyses are powered enough to say "we conditioned on coverage and found no
   spatial reorganization" *only if* the noise ceiling is reported alongside
   (§5.4).

---

## 1. What already exists

| Piece | Where | Note |
|---|---|---|
| Behavioral LWPC/LWPS | `stats/erin_linear_mixed_effects_model.py`, `combinedData.csv` | done |
| LWPC / LWPS power traces | `power/power_traces.py`, `dcc_scripts/power/` | done |
| LWPC / LWPS decoding | `decoding/`, `dcc_scripts/decoding/run_decoding_dcc.py` | done |
| Per-electrode continuous LWPC/LWPS scores | `stats/stability_flexibility_segregation.py` → `compute_sensitivities_per_split` with `contrast_mode='proportion'` | done, **this is the input to all of §5–§8** |
| Split-half noise ceiling | `split_resolved_corr` → `reliability_x`, `reliability_y` | done |
| Scatter + leverage diagnostics | `stats/segregation_scatter.py` | done — this is "my lwpc vs lwps scatterplot" |
| Responsiveness (power) control | `add_responsiveness` + `prepare_continuous` | **already implemented**; see §9.1 |
| Categorical group × ROI enrichment, coverage-conditioned | `stats/stability_flexibility_anatomy.py` → `roi_group_enrichment_test` | done, categorical only |
| Brain rendering of electrode groups | `plot_selectivity_groups_on_brain` | done, categorical colours only |
| Cross-decode machinery (same trials, two labellings) | `decoding/cross_decoding.py` | done — but **not** what §4 needs |
| Two-accuracy-trace comparison | `accuracy_stats.do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions` | done |
| Timing (onsets, jackknifed difference) | `stats/stability_flexibility_timing.py` | done, optional this week |

Everything marked **new** below is the week's work.

---

## 2. N2 — direction tests on the adaptation effects

**Do this first. It is hours, and everything downstream assumes its answer.**

The power traces establish that an LWPC/LWPS interaction exists in lPFC. They do
not say which way it goes, and "concurrent regulation" is a claim about
*direction*: a list-wide manipulation that increases control should **shrink** the
congruency effect in mostly-incongruent blocks and **shrink** the switch cost in
mostly-switch blocks.

Report, per effect, the two simple effects and their difference:

```
LWPC:  (i − c | 25% incongruent)   vs   (i − c | 75% incongruent)
LWPS:  (s − r | 25% switch)        vs   (s − r | 75% switch)
```

- **Unit of inference is the subject**, not the electrode. Average each simple
  effect within subject over the lPFC electrode set and the pre-specified window,
  then a paired t-test (or a cluster-permutation over time if you want the time
  course; `stability_flexibility_timing.interaction_time_course` already produces
  the difference-of-differences trace per electrode and
  `_combine_electrode_traces` collapses it).
- **Report each simple effect's own sign and significance**, not just the
  interaction. Two simple effects with the same sign and different magnitude is
  the expected adaptation pattern; a sign flip is a different (and more
  interesting, and more suspicious) result.
- Use the **cell-balanced** scoring (`_interaction_effect` with equal cell
  weights — the default in `contrast_mode='proportion'`). The 75/25 design makes
  trial-count-weighted contrasts leak the main effect into the interaction; see
  simplification plan §2.2b.

**Kill switch.** If the two adaptation directions disagree with the behavioral
ones, stop and re-read the epoch metadata before running anything in §3–§8. This
check costs an afternoon and protects the whole week.

---

## 3. N3a — LWPC / LWPS decoding (already run; what to add)

Nothing structural. Three reporting additions:

1. **Report trial counts per decoder.** LWPC and LWPS decoders that differ in n,
   class balance, fold count, or feature set cannot have their accuracies
   compared. The limiting cells are known (simplification plan §1.2: `i_MC_MS`
   at 11/class caps the 25%-incongruent decode). If the two decoders are not
   matched, either match them by subsampling to the common minimum and averaging
   over subsamples, or state that the accuracies are not compared to each other.
2. **The two-true-traces test is legitimate.** Comparing two above-chance
   accuracy traces is fine —
   `do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions`
   already does it. One caveat that decides whether the p-value means anything:
   the exchangeability unit must be the resample that both traces share, and
   folds/repeats/bootstraps are **resampling replicates, not biological
   observations**. Do not report n = (folds × repeats × bootstraps).
3. **Pre-stimulus window is the artifact meter.** Congruency cannot be decodable
   before the stimulus. Any pre-stimulus cluster in a congruency decode is a
   confound readout, not a result — see `cross_decoding_controls.md` §5 and
   analysis_guide §17's standing caveat.

---

## 4. N3b — block-transfer cross-decoding (the new analysis)

This is the analysis worth adding, because it is the *decoding analogue of the
adaptation effect itself*, which the existing A4 (train congruency → test
switchType) is not. A4 asks whether conflict and switching share a coding axis;
that is a base-effect question and it belongs in the supplement.

### 4.1 The designs

| Design | Train | Test | Reads as |
|---|---|---|---|
| **X1 (primary)** | congruency, in 25%-incongruent blocks | congruency, in 75%-incongruent blocks (and reverse) | LWPC as a **cross-condition generalization failure**: if block context reconfigures the congruency code, transfer drops below the within-block ceiling |
| **X2 (primary)** | switchType, in 25%-switch blocks | switchType, in 75%-switch blocks (and reverse) | the same for LWPS |
| **X3 (positive control)** | congruency, in 25%-switch blocks | congruency, in 75%-switch blocks | congruency across a factor that should **not** reconfigure it. Same ROI, same trial-count regime, same effect-size regime as X1 — this is what makes a null X1 interpretable |
| **X4 (positive control)** | big letter, task = global | big letter, task = local (occipital) | validates the transfer **code path** on a signal that must be there. Caveat: on congruent trials big and small letter are confounded, so this is a code-path control, not a claim about global-specific coding |
| **X5 (optional)** | incongruent proportion (25 vs 75, collapsing congruency) | switch proportion (25 vs 75, collapsing switchType) | do the two block-context signals share an axis? Orthogonal factors of the same 2×2, so it is well posed |

X1–X3 are the week's target. X4 is cheap insurance. X5 only if X1/X2 land clean.

### 4.2 The implementation gap — read this before scheduling

`build_cross_decoding_arrays` constructs **two labellings of the same trials**
and lets `StratifiedKFold` inside `cv_cm_jim_window_shuffle` make train and test
disjoint. X1–X4 are the other shape: **one labelling, two disjoint trial
populations** (the blocks), where which trials train and which test is *fixed by
the design*, not by the fold.

So the fold splitter must be replaced, not reused. Concretely:

```
new in cross_decoding.py:
  build_block_transfer_arrays(roi_labeled_arrays, roi, contrast, block_col,
                              train_level, test_level)
      -> data, labels, group (train/test membership), strata (condition cell)

new in decoder.py (or a sibling of cv_cm_jim_window_shuffle):
  accept `groups=` and split with PredefinedSplit / LeaveOneGroupOut so the
  train fold is exactly the train-level trials and the test fold the test-level
  trials; repeats then resample *within* those populations (subsample the
  larger side to the smaller, average over subsamples) rather than repartitioning
  across them.
```

`shuffle=True` keeps working unchanged and remains the right null (permute train
labels, refit). `filter_conditions` + `block_condition_sets` already give the
condition name sets for each block level, so the *condition bookkeeping* is done;
it is the splitter that is missing. Budget **half a day plus tests**, and write
the test against `synthetic_roi_labeled_arrays` first (a planted shared code must
transfer across blocks; a planted block-specific code must not).

### 4.3 Two things that will decide whether X1/X2 mean anything

**(a) Block offset is the dominant threat, and it is specific to this design.**
Training in one block and testing in another means any tonic, block-level HG
difference shifts the test cloud along a direction the classifier never intended
to use. The baseline in this dataset carries exactly that confound by
construction (simplification plan §1.4: a random 0.5 s pre-stimulus baseline,
z-scored with statistics pooled across all trials, in a design where
`incongruentProportion` *is* the block). So:

> **Center features within block** — per channel, per block, subtract that
> block's mean over trials — before the transfer, and report the transfer with
> and without.

Without centering, a null transfer is uninterpretable: it could be code
reconfiguration or it could be a DC shift. With centering, the block-identity
information is removed by construction and what remains is the geometry
question. Note that this deliberately discards the tonic block effect, which may
itself be the proactive-control signal — that is undecidable in a blocked design,
which is why both versions get reported.

**(b) The within-block ceiling is mandatory.** A transfer accuracy is only
readable against the within-block accuracy on the *same* trials, matched for n:

```
report, always, as a pair:
  within-block congruency decoding, 25% blocks     (trained and tested in 25%)
  transfer 25% → 75%
```

If within-block congruency decoding sits at 0.57 against a 0.5 null, a null
transfer says nothing — there was not enough signal to transfer. This is the same
logic as the split-half noise ceiling in the segregation analysis (§5.4), and
`cross_decoding_controls.md` §2 states the decision rule.

### 4.4 Check the joint cell counts before running anything

X1 splits an already-thin design. The four-way joint cells (congruency ×
switchType × inc-proportion × switch-proportion) lose trials fast, and
`subsample_to_min_trials_per_condition` takes the minimum **across channels**, so
one bad electrode caps the ROI. Known limiting cells put the 25%-incongruent
congruency decode at ~22/class *pooled over switch proportion*; X1 restricted
within block is where that binds.

Run the count first (`tests/analysis/decoding/test_cross_decoding.py::test_all_four_joint_cells_are_populated_and_balanced`
is the shape of the check; the real version reads the per-stratum counts out of
the run log). **If the within-block decode is itself at chance, X1 is not
runnable and no amount of control analysis fixes that** — say so and move the
week's remaining time to §5–§8.

---

## 5. N4 — per-electrode continuous scores → anatomy (the primary anatomical test)

### 5.1 Scores

Already implemented. `compute_sensitivities_per_split(df, contrast_mode='proportion')`
gives, per electrode, an LWPC score (`x`) and an LWPS score (`y`) on disjoint
halves, and `average_over_splits` collapses them for mapping and plotting.

Requirements that are already met and should be stated in Methods rather than
re-engineered:

- **signed, balanced effect size**, not a p-value (sample-size dependent) and not
  a raw F (unsigned). `_interaction_effect` is an equal-cell-weight
  difference-of-differences divided by the pooled within-cell SD.
- **cross-validated / disjoint-half estimation**, so the score is not read at a
  peak selected by the same contrast.
- **electrode set is anatomical**, not effect-selected (simplification plan §2.6).
  Do not select LWPC- or LWPS-significant electrodes before asking where the
  effects are — that makes the answer partly a property of the selection rule.
  Note also the practical reason: there are too few individually significant
  LWPC/LWPS electrodes to do anatomy on directly.

Standardize within subject before combining across subjects:

```python
def within_subject_zscore(series):
    sd = series.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return series * np.nan
    return (series - series.mean()) / sd

brain_table["lwpc_z"] = brain_table.groupby("subject")["lwpc_score"].transform(within_subject_zscore)
brain_table["lwps_z"] = brain_table.groupby("subject")["lwps_score"].transform(within_subject_zscore)
```

This is what makes `lwpc_z − lwps_z` a meaningful *relative* score: it does not
claim the raw LWPC and LWPS units are comparable, only their within-subject
standing. (`prepare_continuous` already does within-subject centering for the
correlation path; the z-score is the version the anatomy model needs, because it
puts the two effects on a common scale before differencing.)

### 5.2 The test, and the fallacy it has to avoid

**The question is an interaction: effect type × anatomy.** Showing LWPC is
significant in region A while LWPS is not, and concluding a dissociation, is the
difference-of-significance fallacy (Nieuwenhuis, Forstmann & Wagenmakers 2011).
Whatever the model, effect type and anatomy go in the same model with an explicit
interaction term.

Two forms, both worth reporting:

**Categorical (primary).** Relative score `Δ = lwpc_z − lwps_z` per electrode,
tested against ROI / Destrieux parcel:

```
Δ_ij  ~  roi_j  +  responsiveness_ij  +  (1 | subject_i)
```

conditioned on coverage exactly as `roi_group_enrichment_test` already does:
drop ROIs covered in fewer than `min_subjects` subjects, and build the null by
permutation that holds each electrode's ROI fixed.

**The null is a within-electrode swap of the two effect labels.** For each
electrode, randomly exchange its LWPC and LWPS score, recompute Δ, recompute the
statistic. This is the right null because it preserves — exactly, not
approximately — every nuisance structure: subject, coverage, electrode location,
the electrode's overall responsiveness, and the marginal distribution of both
effects. Only the *assignment of effect type* moves, which is precisely the
interaction being tested. Do **not** shuffle locations between LWPC and LWPS;
that null breaks coverage and answers a different question.

**Continuous (secondary).** Regress Δ on MNI coordinates:

```
Δ_ij ~ y_ij + z_ij + x_ij + responsiveness_ij + (1 | subject_i)
```

separately per hemisphere. This is the "LWPC sits anterior to LWPS" claim stated
as an axis rather than as a point, and it is more robust than any centroid (§7).
Coordinates come from `jim_mri.subject_to_info(subject)` → montage `ch_pos`
(after `force2frame`), the same path `plot_selectivity_groups_on_brain` uses to
place electrodes.

### 5.3 Implementation gap

`stability_flexibility_anatomy.py` is categorical throughout — `attach_roi`
derives a 4-way `group` from binary S/F flags, and `roi_group_enrichment_test`
takes a contingency table. The continuous arm needs:

```
new in stability_flexibility_anatomy.py:
  attach_scores(elec_df, electrodes_to_rois, electrodes_to_anat=None,
                electrodes_to_coords=None)      # same join as attach_roi, no S/F
  relative_score_roi_test(scores_with_roi, coverage, min_subjects=3,
                          n_perm=10000)          # Δ ~ roi, within-electrode swap null
  relative_score_coordinate_test(scores_with_roi, n_perm=10000)  # Δ ~ coords
```

Reuse `electrode_ids`, `build_electrode_roi_map`, `build_electrode_anat_map`,
`build_coverage_matrix` and `restrict_to_roi` unchanged — the join and the
coverage bookkeeping are the fiddly parts and they already work.

### 5.4 The noise ceiling decides what a null means

`split_resolved_corr` already returns `reliability_x` and `reliability_y`. The
spatial version of the same argument:

> If within-LWPC split-half spatial r = 0.40 and LWPC-vs-LWPS spatial r = 0.35,
> the two maps are **as similar as the noise permits** — same anatomy. If within
> is 0.40 and between is 0.05, they are distinct.

Without the ceiling, "the maps do not correlate" is indistinguishable from
"neither map is measured well enough to correlate with anything" (Nili et al.
2014). Report the ceiling next to every spatial comparison in §5–§8. This is
non-optional and it is the single most common way this kind of analysis gets
rejected.

---

## 6. N4 — brain maps of the continuous scores

Five surfaces, from the same table:

1. signed LWPC effect
2. signed LWPS effect
3. |LWPC|
4. |LWPS|
5. `lwpc_z − lwps_z` (the relative map)

Map 5 is the one that carries the argument; 1–4 are what a reader needs to check
that 5 is not being driven by one effect's magnitude alone.

**Implementation.** `plot_selectivity_groups_on_brain` colours by discrete group.
It needs a continuous sibling that takes a per-electrode scalar and a diverging
colormap, reusing the same `plot_on_average` / global-index path so the figures
stay comparable with the existing coverage figures. Keep the graceful
off-cluster degradation (the renderer needs MNE + PyVista + the recon templates).

Note for the centroid step: a center of mass needs **non-negative** weights, so
it uses maps 3–4, never 1–2.

---

## 7. Centroids — descriptive only

Demote this. A single pooled cross-subject centroid is not a defensible primary
test, for four reasons that all apply here:

1. Subjects with more electrodes dominate — it is electrode-weighted, not
   subject-level, inference.
2. Coverage is clinically determined, so a pooled centroid can reflect
   implantation strategy rather than physiology.
3. A centroid need not land in cortex at all (bilateral distributions put it near
   the midline; it can fall in white matter or a ventricle).
4. Signed scores cancel — as the denominator approaches zero the "center" leaves
   the electrode distribution entirely.

The defensible version, as a **secondary descriptive** panel:

- one LWPC center and one LWPS center **per subject**,
- **separately per hemisphere**,
- over the **same electrodes**, with **non-negative weights** (|scores|),
- compared **within subject**,
- null = **within-electrode swap of the two effect labels** (same null as §5.2),
- prefer a **medoid** — the observed electrode with the smallest weighted
  distance to the others — so the reported location is a real recording site.

Report it as "LWPC's weighted center sits N mm anterior to LWPS's, within
subject" with the permutation p. The coordinate regression in §5.2 makes the same
claim without a centroid, and it is the one to lead with.

---

## 8. Haufe-transformed decoder patterns — convergent evidence, run last

Worth doing (it is what the advisor asked for) and worth labelling honestly:
**complementary evidence that distributed populations carry each form of
adaptation, not the primary anatomical localization.**

### 8.1 Why raw weights will not do

Raw LDA weights answer "which linear combination best separates the classes,"
not "which electrodes carry the activity." With correlated features a decoder can
assign a large weight to a noise-cancelling electrode with no class information,
a small weight to a strongly informative but redundant one, opposite signs to
correlated neighbours, and unstable signs across folds. The fix is the forward
model (Haufe et al. 2014): multiply the covariance of the training features by
the weight vector, in the original input space.

### 8.2 The procedure, in the order the pipeline forces

Per fold, per window:

1. **Unwind PCA**: `w_scaled = clf.coef_ @ pca.components_` → length n_channels ×
   n_timepoints.
2. **Unwind the scaler**: divide by the scaler's per-feature scale
   (`scaler.scale_`). The project pipeline is `scaler → pca → clf`
   (`decoder.py:82`, `named_steps['scaler'|'pca'|'clf']`), so both steps are
   needed.
3. **Haufe transform**: `a = Σ_X w`, with `Σ_X` the covariance of that fold's
   **training** features in raw electrode × time space.
4. **Pin the sign**: fix the LDA class order explicitly (the same class is
   always +1) so patterns are sign-comparable across folds.
5. **Average across folds in raw feature space, never in PC space.** The PCA is
   refit every fold, so the basis rotates and flips between folds and averaging
   PCA coefficients is meaningless.
6. Reshape to channels × time, reduce over the analysis window to **one scalar
   per electrode**, and L2-normalize the pattern to unit norm — this removes the
   overall accuracy/scale difference between the LWPC and LWPS decoders, which
   otherwise dominates any comparison.

Compare the two patterns by **spatial correlation** and by the §5 anatomy tests,
against the **within-electrode swap null** and the **split-half ceiling**. You
cannot back-project a *difference of accuracies* — that has no weight vector. Two
decoders, two patterns, normalize, then compare.

Back-project each effect at **its own peak window** (the effects have different
timing) and report a common-window version as a robustness check. Match the two
decoders on trial count, class balance, CV folds, and feature set before
comparing.

### 8.3 Implementation gap and the standing caveat

`_window_and_predict_minimal` calls `self.fit(...)` then `self.predict(...)` and
discards the model each fold. Collecting patterns needs a variant of that loop
that stores the unwound, sign-pinned pattern per (fold, window). Budget most of a
day including the round-trip test: plant a known pattern in
`synthetic_roi_labeled_arrays`, back-project, confirm recovery.

**Caveat to write into the figure caption.** PCA at 80% explained variance mixes
channels, so the back-projection is spatially blurred and is therefore *weaker*
evidence about anatomy than the per-electrode effect maps. Keep it convergent.

---

## 9. Controls that apply across the whole plan

### 9.1 Power / responsiveness

Already implemented, and the answer to "this needs to be controlled for power" is
that it largely already is: `add_responsiveness` computes `mean |HG|` per
electrode (a gain proxy, and deliberately **not** `|mean HG|`, which is a
function of the very effects being controlled — simplification plan §2.2c), and
`prepare_continuous` residualizes both scores on it before correlating.

Two upgrades for the manuscript:

- Pass an **explicit** `responsiveness=` — the baseline-vs-signal cluster
  statistic — rather than relying on the default proxy.
- Add responsiveness as a **covariate** in the §5.2 anatomy models, not only as a
  pre-residualization of the scores. A region with globally larger HG would
  otherwise show up as a region with larger scores.

If a reviewer asks for a stronger version: regress the electrode's mean HG out of
each trial's HG before scoring. Report it as a robustness check, not the primary.

### 9.2 Subject-level sanity on every pooled number

`segregation_scatter.joint_scatter_diagnostics` already computes per-subject
correlations, leave-one-subject-out range, the most influential subject, the
drop-top-2%-of-electrodes value, the within-subject correlation, and the maximum
subject share of electrodes. Run it on the anatomy tables too, or at least
report per-subject counts and a leave-one-subject-out sweep next to every pooled
anatomical statistic. Electrode-weighted inference across subjects is acceptable
and standard — but only after checking that two subjects are not supplying the
result.

### 9.3 Baseline / block effects

Two checks, both already specified:

- **Per-trial baseline re-run** of the power traces as a robustness check
  (simplification plan §2.4): subtract each trial's own baseline mean, keep a
  pooled per-channel SD. Do not read the flattened pre-stimulus window as
  evidence — the baseline is drawn from inside it.
- **Show the block effect directly** rather than treating it only as nuisance:
  compare blocks in the power traces and in decoding, and compare the first ~10
  trials of a block against the last ~10 as a within-block adaptation-time
  descriptive. `power/block_diagnostics.py` and
  `dcc_scripts/power/diagnose_block_effects.py` already exist for this.

The block-centering decision in §4.3(a) is the same issue reaching the decoder.
Keep the treatment consistent across §3, §4 and §5, and say which one the primary
numbers use.

---

## 10. Schedule (one week)

Sequenced so the cheap things that can invalidate expensive things run first.

| Day | Work | Gate |
|---|---|---|
| 1 | §2 direction tests. §4.4 joint-cell trial counts. §9.2 per-subject sanity on the existing scatter. | Directions match behavior? Is the within-block decode above chance? |
| 2 | §5.1 build the score table; §5.3 `attach_scores` + the join to ROI/anat/coords. §6 continuous brain maps (maps 1–5). | Score table joins cleanly to coordinates for every subject |
| 3 | §5.2 the categorical ROI interaction test + within-electrode swap null; §5.4 the ceiling. | Ceiling reported next to every r |
| 4 | §4.2 implement the block-transfer splitter + synthetic test. | Planted shared code transfers; planted block-specific code does not |
| 5 | Run X1, X2, X3 with the §4.3 pair (within-block ceiling + transfer), centered and uncentered. X4 if time. | `cross_decoding_controls.md` checklist filled in |
| 6 | §8 Haufe back-projection + round-trip test; spatial comparison against the §5 maps. | Recovery test passes before touching real data |
| 7 | §5.2 continuous coordinate model, §7 descriptive centroids/medoids, figures, Methods paragraphs. | — |

**Cut in this order if the week compresses:** §7 centroids → §5.2-continuous
(coordinates) → §8 Haufe → X4/X5. Do not cut §2, §5.4, or the §4.3 ceiling —
those are what make the rest reportable.

---

## 11. What is deliberately not in this plan

| Dropped | Why |
|---|---|
| Independent-vs-dependent framing; cross-effect nulls as a result | Not the narrative. Scope statement only. |
| Electrode-count pie charts per group | Weak evidence, and counts are small. The continuous score map answers the same question better. |
| Anatomy restricted to individually-significant LWPC / LWPS electrodes | Too few electrodes, and selecting on the effect biases the location question. |
| Pooled cross-subject centroid as a primary claim | §7's four failure modes. |
| Per-electrode single-channel decoding accuracy maps | Answers §5's question at ~n_electrodes× the compute. |
| ROI-restricted decoding with electrode-count matching; ROI ablation / knockout | Genuinely good analyses (they capture multivariate contribution the univariate map misses) — but each is a full decoding sweep, and the week does not have room. Note them as the obvious extension if a reviewer asks "which electrodes *uniquely* drive decoding." |
| A4 label transfer (congruency ↔ switchType) in the main text | Base-effect geometry, not adaptation. Supplement, honestly labelled. |
| Low-frequency bands | Needs the longer / pre-block baseline re-run first (figure_plan.md). Not this week. |

---

## 12. What to claim, and what not to

**Can claim, if the results support it:**

- Both adaptations are present concurrently, in behavior and in lPFC high gamma,
  with their directions reported.
- Distributed lPFC activity carries decodable information about each adaptation —
  including information not available from any single electrode.
- Where a transfer is run *and* its within-block ceiling is above chance: block
  context does (or does not) reconfigure the congruency / switch code.
- Where anatomy is tested as an interaction, conditioned on coverage, with the
  ceiling reported: the two effects are (or are not) organized differently across
  lPFC.

**Cannot claim:**

- "The mechanisms are independent." A null correlation plus two significant main
  effects is not a dissociation. Independence needs reliable within-domain
  patterns for both effects *and* an interval excluding a meaningful shared
  effect.
- A dissociation from two separate significance tests without the interaction
  term (Nieuwenhuis et al. 2011).
- Anything about trial-level cross-electrode covariance. The pseudopopulation
  destroys it by construction (simplification plan §1.1).
- That back-projected decoder patterns localize the effects. They are convergent
  evidence, blurred by the PCA.
- That a null transfer means no shared code, without the within-block ceiling.

**Most likely honest headline:** *overlapping lPFC tissue, separable coding
axes* — concurrent regulation implemented in an intermixed population rather than
in segregated substrates, with the anatomical analysis (continuous scores,
coverage-conditioned, interaction-tested) reported as convergent evidence that
the overlap is not an artifact of electrode coverage.

---

## 13. References this plan leans on

- **Haufe et al. 2014, NeuroImage** — forward activation patterns are
  interpretable, extraction filters (LDA weights) are not. §8.
- **Nili et al. 2014, PLoS Comput Biol** — noise ceiling; why a null pattern
  correlation needs one before it means "same anatomy." §5.4.
- **Nieuwenhuis, Forstmann & Wagenmakers 2011** — the difference-of-significance
  fallacy; why the anatomy test is an interaction. §5.2.
- **Bernardi et al. 2020, Cell** — cross-condition generalization performance,
  operationally defined. §4.
- **Kriegeskorte, Goebel & Bandettini 2006, PNAS** — "where is the information"
  framing, if ROI-restricted decoding is added later. §11.
- **Donos et al. 2022** — uni- vs multivariate methods on iEEG; the citation for
  "decoding detects structure the cluster test misses." §0.
- **Fu et al. 2022** — same/different populations for multiple conflict types,
  framed geometrically. The nearest template for the geometry arm.
