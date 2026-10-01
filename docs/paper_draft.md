# Paper draft: figure plan, Methods and Results

*Assembled 2026-10-01 from the results written up so far.*

This document has three parts:

1. **Figure plan:** the main-text figures (F1–F5) and the supplement, panel by
   panel, with where each one stands after the 2026-10-01 results.
2. **Methods text,** in paper order.
3. **Results text,** in paper order, with the Discussion sentences and the
   figure captions.

It supersedes three parts of the
[closing figure plan](analysis_plans.md#closing-figure-plan) where they differ:
"Proposed closing figure", "Supplement placement" and the weekly figure table.
That plan still holds the reasoning behind each choice. The
[concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan)
still holds the N1–N4 framework.

**What is copied, and from where.** The Methods and Results text was copied on
2026-10-01 from the sections named above each block. The source sections stay
the reference: if one changes, recopy it here. Two kinds of text were not
copied:

- **Skeletons** (marked ✏️): drafted here from the runbooks, for analyses with
  no written-up result yet. Check every parameter against the run you report.
- **Placeholders** (marked ⬜): results not yet in the docs. The power traces
  (F3) and the LWPC/LWPS decoding (F4) stay in the plan; their results only need
  writing up.

Bracketed text (**[…]**) has to come from the archived run or from elsewhere in
the paper.

---

## Contents

- [1. Figure plan](#1-figure-plan)
  - [1.1 The arc](#11-the-arc)
  - [1.2 What changed with the 2026-10-01 results](#12-what-changed-with-the-2026-10-01-results)
  - [1.3 Status at a glance](#13-status-at-a-glance)
  - [1.4 Main-text figures](#14-main-text-figures)
  - [1.5 Supplement](#15-supplement)
  - [1.6 Left out of the paper](#16-left-out-of-the-paper)
  - [1.7 Before submission](#17-before-submission)
- [2. Methods](#2-methods)
- [3. Results](#3-results)
- [4. Discussion sentences](#4-discussion-sentences)
- [5. Figure captions](#5-figure-captions)

---

## 1. Figure plan

### 1.1 The arc

Behavior shows that stability and flexibility are regulated at the same time, in
the same participants (F1). lPFC high gamma carries both adaptation effects in
the behavioral direction (F3), and both are decodable from distributed lPFC
activity (F4). The anatomy then shows how the two are organized (F5). They are
carried by one intermixed population. Electrode by electrode, their balance
follows the local balance of the demands each regulates.

**The ending sentence** (from the all-lPFC main-effect run, §16.6 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)):

> Each adaptation tracks the local strength of the demand it regulates; whether
> the spatial gradient in their balance is inherited from the base effects could
> not be determined.

Framing rules that still apply (concurrent-regulation plan §0):

- The absent cross-effects (congruency × switch proportion, switch type ×
  incongruent proportion) are a scoping sentence, not a figure.
- Decoding is not a multivariate-superiority claim. The claim is that adding
  electrodes yields information no single electrode supplies.
- Organize figures by claim, not by measure. Power and decoding for the same
  claim go in the same figure.

### 1.2 What changed with the 2026-10-01 results

| Result | Effect on the figures |
|---|---|
| **N4 all-lPFC main-effect run written up and checked** ([`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) §17) | F5 is final in design: four panels a–d, led by the overlap result, with the gradient as a modest secondary finding. The numbers are checked against the run's outputs. Still needed: the task-significant rerun with main effects, and the plots. |
| **A4 congruency ↔ switch cross-decoding run** ([`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01)) | Supplement S5, not the closing figure. The transfer is partial and late, and not yet shown to be specific to lPFC or free of RT. It earns one Discussion sentence. The main-effect group decoding is dropped. |
| **A6 brain–behavior run** ([`a6_brain_behavior.md`](a6_brain_behavior.md) §13–§14) | Supplement S-BB only, framed as "could not be tested at this n", plus one Discussion sentence. |
| **Weighted medoids** (§17.4 of the N4 doc) | Dropped from the supplement (old S9). They are null on every axis, and the two centre definitions disagree in sign. |
| **Power traces and LWPC/LWPS decoding** | Unchanged. F3 and F4 stay in the main text; their results are not yet written up in the docs. |

### 1.3 Status at a glance

| Fig | Claim | Status | Next step |
|---|---|---|---|
| F1 | Both adaptations are present concurrently in behavior | ⬜ Figure done; numbers not in the docs. Confirm it was not made with the swapped block map (§1.4, F1). | Confirm the source script; write the Results paragraph |
| F2 | Coverage and signal validation | Needs the coverage table (S1) | Build the per-ROI, per-participant table |
| F3 | lPFC high gamma carries LWPC and LWPS in the behavioral direction | ⬜ Traces done; direction tests implemented; results not in the docs | Run or collect the direction tests; write up |
| F4 | Both adaptations are decodable from distributed lPFC activity | ⬜ Done; results not in the docs | Report trial counts per decoder; write up |
| F5 | One intermixed population at both levels; the adaptation balance tracks the base-effect balance and has a dorsomedial gradient | Results and text final for all lPFC (N4 §17); panels not plotted | Plot a–d; task-significant main-effect rerun |

### 1.4 Main-text figures

#### F1 — Task, manipulation, behavior (N1)

| Panel | Content |
|---|---|
| a | Paradigm |
| b | The 2 × 2 block-proportion design. Blocks: A 75 % incongruent / 25 % switch, B 75 / 75, C 25 / 25, D 25 / 75, fully crossed. |
| c | RT: the congruency effect by incongruent proportion (LWPC), the switch cost by switch proportion (LWPS) |
| d | Error rate, the same layout |

**The point:** both adaptations are present in the same participants and the
same sessions. The absent behavioral cross-effects go in the text as scope, and
in S4.

**Numbers so far** (a quick check from `combinedData.csv`, correct trials, 23
participants, with the corrected block map;
[`a6_brain_behavior.md`](a6_brain_behavior.md) §1.1): LWPC 123 ms (SD 147),
positive in 20 of 23, t(22) = 4.0, p = 0.0006; LWPS 97 ms, positive in 20 of 23,
t(22) = 4.5, p = 0.0002. These are not the paper's behavioral model.

⚠️ **Check the source of F1 first.** Until 2026-09-27 the `blockType` map
swapped blocks A and D. That turned "LWPC" into a congruency × switch-proportion
contrast (mean 46 ms instead of 123 ms). `erin_linear_mixed_effects_model.py`,
which the concurrent-regulation plan names as the N1 source, carried the swap
and has no congruency × proportion term. Confirm which script produced the
reported F1 numbers, and remake F1 if it used the old map.

#### F2 — Coverage and signal validation

| Panel | Content |
|---|---|
| a | All electrodes on the MNI surface, coloured by ROI |
| b | Per-electrode high-gamma traces for one example participant, task-responsive electrodes outlined |
| c | Example spectrogram |

Cite the per-ROI, per-participant coverage table (S1) here, with the inclusion
threshold. It answers "why only lPFC?". The counts in the old figure plan are
stale.

#### F3 — Adaptation effects in lPFC high gamma (N2) ⬜

| Panel | Content | Data |
|---|---|---|
| a | Task-responsive lPFC electrodes on the surface, with electrode and participant counts | `sig` electrodes, lPFC |
| b | High-gamma traces for LWPC: the congruency effect in 25 % vs 75 % incongruent blocks | power traces |
| c | High-gamma traces for LWPS: the switch cost in 25 % vs 75 % switch blocks | power traces |
| d | Direction tests: both simple effects and their difference (low − high proportion), each with its own cluster bar | `n2_direction_tests/<roi>/` figures and npz ([`n2_direction_tests.md`](n2_direction_tests.md) §6) |

**Panel d makes this figure a claim.** It reports the sign of each adaptation
against the behavioral sign. Positive = the condition effect shrinks in
high-proportion blocks, the behavioral direction.

**To report with it** ([`n2_direction_tests.md`](n2_direction_tests.md) §7–§8):

- `n_electrodes` and `n_subjects` together.
- Each simple effect's sign and significance, not only the interaction.
- The per-participant direction tally and a leave-one-participant-out sweep.
  Neither is implemented on the N2 path; both can be rebuilt from the saved
  `_evoked.npz` files.
- `N_PERM` above the default 500 (which floors p at ~0.002) for a reported p.
- A matching sign is not independent evidence on its own: if high gamma tracks
  RT within cells, the neural difference of differences inherits behavior's
  sign. The RT-adjusted electrode scores in A6's
  `participant_electrode_scores.csv` are the check.

Main-effect traces (incongruent vs congruent, switch vs repeat) go to S2.

**What would change it:** a direction opposite to behavior. That stops
everything downstream until the epoch metadata are re-read.

#### F4 — Decoding the two adaptations (N3) ⬜

| Panel | Content |
|---|---|
| a | LWPC: congruency decoded within 25 %- and within 75 %-incongruent blocks, each against its refit shuffle null, and their difference |
| b | LWPS: switch type decoded within 25 %- and within 75 %-switch blocks, the same layout |

Use the `_block_balanced` condition sets
([`decoding.md` › Decoding job](decoding.md#decoding-job) §2.1), so a tonic block
difference cannot enter the contrast.

**To report with it** (concurrent-regulation plan §3):

- Trial counts per class for each decoder. If the LWPC and LWPS decoders differ
  in n, either subsample to the common minimum or say that their accuracies are
  not compared.
- Pre-stimulus windows as the artifact meter: congruency cannot be decodable
  before the stimulus.
- The samples are CV repeats and bootstraps of one pseudopopulation, not
  participants. The leave-one-subject-out run (`submit_loo_decoding_dcc.sh`)
  checks that no participant carries a result.

**Layout:** a trellis with shared axes and one legend reads as one panel.

**Dropped:** the block-transfer panel (old F4c). The cross-proportion transfer
is shelved: the within-block ceiling sits near chance with ~20 rare-class trials
per participant per block, and a drop in transfer is confounded with LWPC itself.

#### F5 — How the two adaptations are organized across lPFC (N4)

All lPFC (398 electrodes, 22 participants) is the primary set, as specified
before analysis. The task-significant set is reported in full in S-N4. Data and
status per panel: §16.7.3 and §17.4 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md).

| Panel | Shows | Key numbers | Data | Status |
|---|---|---|---|---|
| a | **Shared electrodes at both levels:** congruency vs switch beside LWPC vs LWPS, on the responsiveness-residualized, participant-centred scores each pre-specified test correlates | r = 0.23, p = 0.0001; r = 0.10, p = 0.0005 (397 electrodes, 21 participants); centroid distance 1.4 mm, p = 0.95 | LWPC/LWPS: `x_resid`/`y_resid` in the segregation run's `continuous.csv`. Congruency/switch: `prepare_continuous` on its `electrodes.csv` with `mx`/`my`. r from the segregation `summary.txt` and `correlation_main_effects.json`. | Not plotted; no rerun needed |
| b | **The two balances by parcel:** adjusted congruency − switch (x) against adjusted LWPC − LWPS (y), one dot per Destrieux parcel, sized by electrodes, coloured by distance from the midline | omnibus F = 1.70, p = 0.017 and F = 1.91, p = 0.010; label r = 0.73, same sign in 13 of 19 | `panel_b_label_means.csv` (`n4_section16_followups.py` §4) | Data ready; not plotted |
| c | **The gradient, by score:** congruency, switch, LWPC and LWPS by tertile of distance from the midline, participant mean ± SEM, as matched small multiples (base effects, adaptation) | LWPC \|x\| slope p = 0.006; the other three p ≥ 0.34. Exploratory. | `panel_c_midline.csv` (script §3) | Data ready; not plotted |
| d | **The link between levels (Test 1):** r for congruency − switch vs LWPC − LWPS, with the two matched and the two crossed pairings | 0.092 (p = 0.0003); with coordinates 0.087; matched 0.22 / 0.17; crossed 0.13 / 0.12 | `delta_tracking.csv` | Data ready; not plotted |

**Design notes.**

- Panel a plots the scores the test correlates, not `joint_scatter.png`, whose
  numbers are not the pre-specified test. Specification and caption:
  [closing figure plan › Panel a](analysis_plans.md#panel-a).
- Panel c: matched small multiples with shared axes and one legend, participant
  means ± SEM, no electrode scatter. It replaces the "by height" panels; the
  height version goes to S-N4.
- Test 2 (is the gradient inherited?) has no panel. It cannot say how much of
  the gradient is inherited. It gets one sentence in the text and goes to S-N4.
- Brain maps appear only as coverage or illustration, with a legend line saying
  single electrodes are not interpretable.

**Placement of each result** (§17.4 of the N4 doc):

| Result | Where |
|---|---|
| LWPC and LWPS share electrodes; no centroid separation | Main text, headline (F5a) |
| Congruency and switch share electrodes | Main text, as the reference (F5a) |
| Adaptation balance differs across parcels | Main text; omnibus only, never a named parcel |
| Base-effect balance differs across parcels, in step | Main text (F5b); the label correlation is descriptive |
| Electrode-level tracking, process-specific (Test 1) | Main text (F5d); the most defensible link between levels |
| Coordinate gradient | Main text, secondary |
| Dorsomedial vs ventrolateral; carried by LWPC | Main text, labelled exploratory (F5c) |
| Inheritance (Test 2) | One sentence; details in S-N4 |

**What would change it:** the task-significant rerun with main effects. Settling
whether the gradient is inherited would need a more reliable measure of the
base-effect balance than these trial counts give (split-half reliability 0.08).

**Open for the advisor** (closing figure plan, "Open questions"): all lPFC as
the primary set; lPFC-only scope; whether to name the axis "dorsomedial versus
ventrolateral" or keep height as the headline; whether the gradient appears in
the abstract (recommendation: at most one clause).

#### Timing (A5): fold in or drop

A row in F3 only if the LWPC/LWPS onset ordering is significant, with the
latency–amplitude guard. No timing result is written up in the docs; a null
folds it away.

### 1.5 Supplement

Numbers kept from the earlier plans so the cross-references in other docs still
work. Renumber at submission.

| S | Content | Status | Text |
|---|---|---|---|
| S1 | Per-ROI, per-participant coverage table with the inclusion threshold | To build | – |
| S2 | Main effects (congruency, switch type) in lPFC high gamma: traces and decoding | ⬜ Not in the docs | – |
| S2b | Main-effect electrodes on the brain (one pre-specified method), other methods as robustness | Not run | – |
| S2c | LWPC and LWPS traces within congruency, switch and both groups, selected on half A and tested on half B, with the group × effect-type test | Not run | – |
| S3 | Low-frequency bands, rerun with a ≥ 1 s baseline that predates the block context | Not run. Do not use the current low-band results: the 0.5 s pre-stimulus baseline is 2–4 cycles at theta and may contain the block-level effect. | – |
| S4 | Absent cross-effects, as scope | – | One sentence in Results |
| S5 | Congruency ↔ switch cross-decoding, labelled as base-effect geometry: the unselected lPFC transfer with its ceilings, `remove_mean`, RT-matched vs random, occipital | Run 2026-10-01. Seeds, the electrode-matched region comparison, the positive controls, `mean_only` and response-locked runs still to do. | §2.6, §3.6 |
| S6 | Haufe-transformed decoder patterns, as convergent evidence | Optional; not run | – |
| S7 | Per-participant high-gamma traces; demographics, electrode counts, exclusions | To build | – |
| S8 | Cross-decoding control table for every transfer reported | The A4 block is filled in ([`decoding.md` › A4 §13.7](decoding.md#137-what-to-report-and-where-things-stand)); pseudo-trial counts missing | – |
| S10 | Per-trial-baseline robustness rerun of the power traces; direct block comparisons | Not run | – |
| S-N4 | Anatomy in full: the task-significant set; per-parcel bars, coverage, leave-one-out tables; both coordinate tables and the hemisphere fits; the height vs midline models and panel c by height; Test 2 with the reliability and bootstrap; map similarity | All-lPFC parts ready. The task-significant set has LWPC/LWPS only; its main-effect rerun is pending. | §3.5 |
| S-BB | Brain–behavior across participants: RT-adjusted (primary) and raw correlations, reliabilities, ceiling, power | Run 2026-09-30; text final | §2.7, §3.7, §5.2 |

### 1.6 Left out of the paper

| Result | Why |
|---|---|
| Weighted medoids (`score_centers.csv`) | Null on every axis; 6 of 25 groups have both medoids on the same contact; the two centre definitions disagree in sign on z |
| Noise-corrected LWPC–LWPS ratio (1.03 electrode, 3.38 parcel) | Not estimable at these reliabilities |
| `joint_scatter.png` correlations (pooled 0.32, within-participant 0.22) | Full-data correlations, not the pre-specified test |
| A4 main-effect groups (`both`, `*_only`) | Opposite to the shared-code prediction, pre-stimulus windows, groups change with each run's selection trials |
| A4 with RT matching and `remove_mean` together | Uninterpretable: the congruency ceiling is gone, in the random control too |
| Cross-proportion (block) transfer | Shelved: no within-block ceiling, and confounded with LWPC itself |
| A6 positive-electrode summary (r ≈ 0.53–0.56, p ≈ 0.07) | Post hoc, 11–12 participants, at or above its own ceiling, selects on the sign it averages |
| A6 mean \|d\| joint regression; level-2 counts; level-3 slopes | Suppression from shared noise; uncorrected and coverage-driven; the slopes measure the high gamma–RT link |
| "Correlated at the noise ceiling"; "LWPC dominance increases ventrally"; "the tilt survives the main effects" | Withdrawn or not supported; see §16.6.7 of the N4 doc for the wording to use instead |

### 1.7 Before submission

**F5 and S-N4** (N4 §17.4):

- [ ] Copy the all-lPFC anatomy folder before the task-significant rerun; the
      rerun writes to the same path (§17.1 of the N4 doc).
- [ ] Rerun the task-significant set with main effects; report it in full.
- [ ] Plot F5 panels a–d.
- [ ] Leave-one-participant-out on the pre-specified LWPC–LWPS correlation, and
      its p from 10,000 permutations.
- [ ] Advisor decisions: primary set, scope, axis name, abstract.

**F1–F4:**

- [ ] F1: confirm the behavioral source script uses the corrected block map.
- [ ] F2: build the coverage table (S1).
- [ ] F3: collect the direction-test results; add the per-participant tally and
      leave-one-participant-out; raise `N_PERM`.
- [ ] F4: trial counts per class for each decoder; check pre-stimulus windows.

**S5** ([`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01)):

- [ ] Seeds for the baseline, `rt` and `random`; recompute `kept` over
      post-stimulus windows.
- [ ] lPFC subsampled to occipital's 54 electrodes.
- [ ] Task positive controls; `mean_only`; response-locked.
- [ ] Pseudo-trial counts per cell from each run.

**Everywhere:**

- [ ] Fill in the preprocessing paragraph (§2.1) and check it describes the
      epochs file every run used.
- [ ] Archive the git commit, submission command and Slurm log with each run.

---

## 2. Methods

### 2.1 Participants, recordings and preprocessing

*Copied from [`methods.md` › N4](methods.md#participants-recordings-and-electrode-population), first paragraph.*

> Intracranial EEG was recorded from **[N participants]** while they performed the
> Global/Local task. Broadband high-gamma activity (70–150 Hz) was extracted from
> cleaned, average-referenced recordings using a filterbank–Hilbert transform and
> epoched from −1.0 to 1.5 s relative to stimulus onset. The preprocessing
> pipeline supplied signal epochs and a separately constructed 0.5-s
> prestimulus baseline object to a z-score rescaling operation. Because alignment
> of individual signal trials to individual baseline segments has not been
> verified in the current implementation, this normalization is not described as
> trial-by-trial baseline correction. Epochs were decimated by a factor of eight.
> Trials exceeding 10 SD were marked as outliers, and channels for which more
> than 5% of trials were outliers were excluded. Analyses included correct trials
> only and omitted the first trial of each block, for which task sequence was
> undefined.

**[Add: task-responsive electrode definition (high gamma against the
pre-stimulus baseline), used by F3, F4, S5 and S-BB.]**

### 2.2 Task design and behavior (F1) ⬜

**[To write.]** It needs to state:

- The four block types, fully crossed: A 75 % incongruent / 25 % switch, B 75 /
  75, C 25 / 25, D 25 / 75 (`src/task/mainTask.m`).
- LWPC = (I − C)₂₅ − (I − C)₇₅ and LWPS = (S − R)₂₅ − (S − R)₇₅ on mean RT, low
  minus high proportion, so positive means the predicted adaptation: a smaller
  congruency effect or switch cost in high-proportion blocks. The neural scores
  use the same orientation.
- Trial inclusion and the model or test behind the reported numbers.

### 2.3 Power traces and direction tests (F3) ✏️

*Skeleton, drafted from [`n2_direction_tests.md`](n2_direction_tests.md) §1–§4.
Check every parameter against the reported run.*

> **[Describe the trace analysis behind F3b–c: electrode set, conditions, and
> the test between traces.]** To establish the direction of each adaptation
> effect, we compared the congruency effect (incongruent − congruent) between
> 25 %- and 75 %-incongruent blocks, and the switch cost (switch − repeat)
> between 25 %- and 75 %-switch blocks, in **[n]** task-responsive lPFC
> electrodes from **[N]** participants. For each effect we ran three tests on
> the trial-averaged high-gamma time courses: the condition effect in the
> low-proportion blocks, the condition effect in the high-proportion blocks,
> and their difference (low minus high proportion; positive values indicate a
> smaller condition effect in high-proportion blocks, the direction of
> behavioral adaptation). Each test was a two-sided, paired cluster-based
> permutation test over time, with electrodes pooled across participants as
> the unit of observation (cluster-forming threshold p = 0.05, cluster
> threshold p = 0.05, **[N_PERM]** permutations). Because electrodes within a
> participant are correlated, these p-values are optimistic. We therefore also
> report the number of participants whose electrode-averaged effect points in
> each direction, and the test repeated with each participant left out.

### 2.4 Decoding the adaptation effects (F4) ✏️

*Skeleton, drafted from the decoding job's defaults
([`decoding.md` › Decoding job](decoding.md#decoding-job) §2). Check every
parameter against the reported run.*

> We decoded each condition contrast separately within each block type from
> **[n]** task-responsive lPFC electrodes (**[N]** participants). For LWPC,
> congruency (incongruent vs congruent) was decoded within 25 %- and within
> 75 %-incongruent blocks. For LWPS, switch type (switch vs repeat) was decoded
> within 25 %- and within 75 %-switch blocks. Each class pooled two block types,
> each subsampled to the smaller before pooling, so that a sustained difference
> between blocks could not enter the contrast. For each condition, every
> electrode's trials with missing values were dropped, and each electrode was
> independently subsampled to the smallest trial count of any electrode in the
> region (**[n]** pseudo-trials per class). Electrodes from all participants
> were concatenated into one pseudopopulation, and classes were balanced by
> subsampling. The classifier was principal component analysis (components
> explaining 90 % of the training variance) followed by linear discriminant
> analysis, applied to 250-ms windows (64 samples) stepped by 62.5 ms, with
> five-fold cross-validation repeated five times, over five independent draws
> of the pseudopopulation. The null distribution refit the classifier on
> permuted training labels (50 permutations per draw). A window was
> significant when the mean true accuracy exceeded the 95th percentile of the
> null, and runs of significant windows were kept when longer than the 95th
> percentile of the longest run under the null. The two block types' accuracy
> traces were compared with a cluster-based permutation test in each
> direction (α = 0.025 each). The samples entering these tests are
> cross-validation repeats of the pseudopopulation, not participants, so the
> p-values measure within-dataset reliability; a leave-one-participant-out
> analysis checked that no participant carried a result.

### 2.5 Anatomy: segregation and continuous anatomy (F5, S-N4)

*Copied from §17.2 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#172-methods).
It replaces the bracketed N4 template in `methods.md` for the all-lPFC analysis.*

> **Electrode population.** Anatomical analyses used every electrode in lateral
> prefrontal cortex (lPFC; 398 electrodes, 254 left and 144 right, from 22
> participants, 1–55 per participant). The set was defined by atlas label, not
> by task responsiveness or by any effect, because these analyses ask how
> effects are distributed across the region rather than whether they are
> present. This choice was recorded before analysis. Results in the
> task-responsive subset are reported in full in the Supplement. Each electrode
> was assigned a Destrieux parcel and fsaverage (MNI) coordinates from the
> participant's reconstruction.
>
> **Per-electrode scores.** High-gamma power (70–150 Hz) on correct trials was
> averaged over 0–1.5 s after stimulus onset. Each process was scored from the
> four cell means of its 2 × 2 design, congruency (incongruent, I; congruent, C)
> × incongruent proportion (25 %, 75 %) and switch type (switch, S; repeat, R)
> × switch proportion, with equal weight on every cell. The adaptation effects
> were differences of differences, LWPC = (I − C)₂₅ − (I − C)₇₅ and
> LWPS = (S − R)₂₅ − (S − R)₇₅. The main effects were the mean simple effects,
> congruency = ½[(I − C)₂₅ + (I − C)₇₅] and switch = ½[(S − R)₂₅ + (S − R)₇₅].
> Each was divided by the pooled within-cell standard deviation (Cohen's *d*).
> Positive LWPC and LWPS mean the predicted adaptation: a smaller condition
> effect in high-proportion blocks. Equal cell weights stop the unequal trial
> counts of the proportion design from leaking the main effect into the
> interaction, or block-level differences into the main effect.
>
> **Disjoint trial halves.** Each electrode's trials were divided into random
> halves 1,000 times, stratified on congruency, incongruent proportion, switch
> type and switch proportion. All four scores were computed on each half. An
> electrode's score was the mean over halves and splits. Every test that relates
> two scores across electrodes took them from opposite halves within each split
> and then averaged over splits, so that shared trial noise could not create an
> association. A map's split-half reliability was the mean correlation between
> electrodes' two half estimates.
>
> **Balances.** Each score was divided by its standard deviation across
> electrodes, without centring or within-participant standardization. The
> adaptation balance was Δ_adapt = LWPC − LWPS (positive: relatively
> LWPC-dominant) and the base-effect balance Δ_main = congruency − switch
> (positive: relatively congruency-dominant). Because each balance is a
> difference within an electrode, a test of how it varies with anatomy is an
> effect-type × anatomy interaction, and anything an electrode contributes
> equally to both scores (gain, responsiveness, participant) cancels.
>
> **Co-localization.** To test whether two effects share electrodes, one score
> from one half was correlated with the other score from the other half (both
> directions, every split) after both were regressed on responsiveness (mean
> absolute high-gamma) and centred within participant. Spearman ρ was averaged
> over splits. Participants with fewer than three electrodes were excluded
> (397 electrodes, 21 participants remained). The null permuted electrode
> correspondence within participant, identically across splits. Electrodes
> positive for each adaptation effect were compared in location by the distance
> between their centroids, with electrode labels shuffled within participant.
>
> **Anatomical test.** Each balance was modelled as
> balance = parcel + responsiveness + participant, with participants as fixed
> effects. The statistic was the partial *F* for the parcel block. Destrieux
> parcels sampled in at least three participants entered (19 parcels, 396
> electrodes). The null exchanged the two effect labels independently within
> every electrode, which flips the sign of the balance while keeping each
> electrode's participant, location, responsiveness and pair of scores
> (10,000 permutations; *p* = (*b* + 1)/(*N* + 1)). Adjusted parcel means, with
> Benjamini–Hochberg correction over the 19 parcels, were used only to describe
> a significant omnibus result. The test was repeated with each participant
> left out (1,000 permutations per fold).
>
> **Coordinate test.** Each balance was also modelled as
> balance = MNI *y* + *z* + *x* + responsiveness + participant, across all
> electrodes and within each hemisphere, with the same null. We report the
> block *F*, and slopes in standard deviations per millimetre with *p* values
> Bonferroni-corrected over the three axes.
>
> **Main effects as a reference** (specified before this run). Test 1 asked
> whether the balances track each other: Δ_main from one half against Δ_adapt
> from the other, by the co-localization procedure above, with and without MNI
> coordinates partialled out within participant. The matched (congruency–LWPC,
> switch–LWPS) and crossed (congruency–LWPS, switch–LWPC) pairings were
> computed the same way. Test 2 asked whether the adaptation gradient is
> inherited: the coordinate model for Δ_adapt was refitted with Δ_main as a
> covariate, and the change in slope (shrinkage = 1 − slope with
> covariate / slope without) was read against Δ_main's split-half reliability,
> because a covariate measured with reliability λ can remove only about λ of a
> slope it fully carries. Uncertainty in the shrinkage came from 2,000
> participant bootstrap resamples.
>
> **Exploratory follow-ups** (chosen after the coordinate result). Because
> height and distance from the midline are correlated across lPFC electrodes,
> the coordinate model was refitted with |*x*| in place of signed *x*. Each
> single score was fitted on *y* + *z* + |*x*| + responsiveness + participant.
> A single score has no partner to exchange labels with, so its null shuffled
> coordinates among each participant's electrodes (2,000 shuffles). Adjusted
> Cohen's *d* (participant and responsiveness removed) is shown by tertile of
> |*x*| for display only.
>
> **Reporting.** All tests were two-sided with α = 0.05.
> Electrode maps and per-participant weighted centres were descriptive and
> were not used for inference.

⚠️ The last sentence mentions weighted centres, which §1.6 leaves out of the
paper. Drop "and per-participant weighted centres" if they are not shown.

### 2.6 Congruency ↔ switch-type cross-decoding (S5)

*Copied from [`methods.md` › A4](methods.md#a4-congruency--switch-type-cross-decoding), Methods.*

> **Data and electrodes.** We used the stimulus-locked high-gamma epochs described
> above (70–150 Hz, −1.0 to 1.5 s, decimated to 256 Hz; correct trials only; the
> first trial of each block omitted). We decoded all lateral prefrontal electrodes
> whose high gamma exceeded their pre-stimulus baseline (171 electrodes from 21
> participants). Electrodes were not selected for a congruency or switch-type
> effect. As a regional comparison we repeated every analysis on the task-responsive
> occipital electrodes (54 electrodes from **[N]** participants).
>
> **Conditions and pseudopopulation.** Trials were sorted into the four congruency
> × switch-type cells, each pooled over the four block types. For each cell, each
> electrode's outlier trials were removed and the electrode was randomly subsampled
> to the smallest number of clean trials of any electrode in that cell
> (**[n]** pseudo-trials per cell). Electrodes from all participants were then
> concatenated into one pseudopopulation. Because electrodes were sampled
> independently, a pseudo-trial combines different trials of the same condition
> across electrodes, including electrodes from the same participant.
>
> **Decoding.** Each 250-ms window (64 samples, stepped by 62.5 ms; 37 windows) was
> decoded separately. The features were every electrode's samples in that window.
> The classifier was principal component analysis (components explaining 80% of the
> training variance, refit in each fold) followed by linear discriminant analysis
> with equal class priors. We used stratified five-fold cross-validation, repeated
> ten times. Folds were stratified on the four cells, so every test fold was
> balanced on both labellings. Accuracy was the mean of the two classes' hit rates.
>
> **Cross-decoding.** On each fold we trained one classifier on congruency
> (incongruent vs congruent) and one on switch type (switch vs repeat). Each was
> scored on held-out trials against both labellings. Scoring against the training
> labelling gives the within-contrast accuracy (the *ceiling*). Scoring against the
> other labelling gives the *transfer*. Incongruent was paired with switch and
> congruent with repeat, so a shared axis on which the harder condition of each
> contrast falls on the same side yields above-chance transfer. Each transfer was
> expressed as the share of its ceiling's above-chance accuracy that it retained,
> (transfer − 0.5) / (ceiling − 0.5), averaged over the windows in which the ceiling
> was significant.
>
> **Statistics.** For each decode, the null distribution came from permuting the
> training labels within each fold and refitting. True-label accuracies (ten CV
> repeats) were compared with the null by a one-tailed cluster-based permutation
> test over windows (500 permutations, α = 0.05). The same test compared each
> transfer with its ceiling. Windows centred at or before −0.125 s, which contain no
> post-stimulus sample, served as a check on artifacts. Because the samples entering
> these tests are CV repeats of a single pseudopopulation rather than participants,
> we treat the window-wise results as a within-dataset reliability measure, not as
> population inference.
>
> **Controls.**
>
> - *Response time.* Incongruent and switch trials were slower. In each
>   participant, RTs were pooled across the four cells and cut into ten quantile
>   bins. Within each bin we kept, at random, the same number of trials from each
>   cell. This left 48% of trials and removed the RT costs (incongruent − congruent:
>   +160 → +2 ms, p = 0.45; switch − repeat: +195 → +2 ms, p = 0.39; across 24
>   participants). A control drew the same number of trials per participant and
>   cell without regard to RT, keeping the RT costs (+156 and +208 ms). The
>   RT-matched result is compared with this control, not with the full-trial result.
> - *Overall activity.* To ask whether the transfer reflected a uniform change in
>   activity, we subtracted each participant's mean across its electrodes, per
>   pseudo-trial and time point, before decoding. Participants contributing a single
>   electrode carry no information after this step (2 of 21).
>
> **[Positive controls, the response-locked analysis and the seed repeats go here
> once run.]**

Limitations to state with it (same source): the samples are CV repeats of one
pseudopopulation with no estimate across participants; a pseudopopulation code
is an upper bound on what any one participant's lPFC shares; the activity
control removes only a shift common to all of a participant's electrodes; and
these are the base effects, not their adaptation.

### 2.7 Brain–behavior (S-BB)

*Copied from [`methods.md` › A6](methods.md#a6-brainbehavior-supplement-s-bb).*

> **Participants and electrodes.** We asked whether participants whose lateral
> prefrontal high-gamma activity adapted more to the proportion manipulations
> also adapted more in their behavior. The analysis used the task-significant
> electrodes in lateral prefrontal cortex (171 electrodes from 21 participants;
> task significance as defined in **[main Methods section]**). These electrodes
> were selected for overall task responsiveness, not for an LWPC or LWPS effect.
> The high-gamma epochs and preprocessing were those of the anatomical analyses.
> Only correct trials with a recorded response time (RT) were used. A participant
> entered the analysis if it had at least three usable electrodes (18
> participants; median 7 electrodes and 397 trials each) and a behavioral score;
> one of the 18 was absent from the behavioral summary table **[reason]**,
> leaving 17 participants.
>
> **Neural and behavioral scores.** For each electrode, single-trial high gamma
> was averaged over 0–1.5 s after stimulus onset. LWPC and LWPS were scored as in
> the anatomical analyses: the equal-cell-weighted difference of differences (the
> condition effect in the low-proportion blocks minus that in the high-proportion
> blocks), divided by the pooled within-cell standard deviation. Here each score
> was computed once from all of the electrode's trials rather than on split
> halves. An electrode was usable if all four of its scores (LWPC and LWPS,
> unadjusted and RT-adjusted, below) were defined. A participant's neural LWPC
> and LWPS were the unweighted means of its usable electrodes' scores. Equal
> weights are appropriate because a participant's electrodes share its trials
> and so have similar sampling error.
>
> The behavioral LWPC and LWPS were the same differences of differences computed
> on mean RT, in milliseconds, from the behavioral analysis **[section reference;
> trial inclusion as described there]**. Both scores were oriented so that
> positive values indicate the predicted adaptation: a smaller congruency effect
> or switch cost in the high-proportion blocks.
>
> **Removing the RT-linked component of high gamma.** The analysis window covers
> most responses (median RT 1.19 s), so single-trial high gamma may track RT. If
> it does, every cell mean of high gamma carries the same multiple of that cell's
> mean RT. Each electrode's neural difference-of-differences then contains its
> slope on RT times the participant's own behavioral difference of differences.
> That term would correlate neural with behavioral adaptation across participants
> with no link between them beyond the trial-by-trial coupling, and it would also
> pass the specificity checks below.
>
> We therefore removed the RT-linked component of high gamma separately for each
> electrode. The slope of high gamma on RT was estimated from deviations around
> each of the 16 cell means of the design (congruency × task sequence ×
> incongruent proportion × switch proportion), pooled across cells, so that
> condition effects, which move both high gamma and RT, did not enter it. Each
> trial's high gamma was replaced by its value minus the slope times the trial's
> deviation from the electrode's mean RT, and the scores were recomputed. This
> removes exactly the slope times the behavioral difference of differences from
> each electrode's score. The adjustment is conservative: if neural adaptation
> reaches behavior through the same trial-by-trial coupling, that part is
> removed too. We therefore treat the RT-adjusted correlation as the primary test
> and report the unadjusted correlation as an upper bound.
>
> **Statistical analysis.** For each effect, the neural and behavioral scores
> were related across participants by Pearson correlation (two-sided, α = .05;
> |r| ≥ 0.48 needed at n = 17), with a 95% confidence interval from Fisher's *z*
> and Spearman's ρ as a rank-based check. Behavioral LWPC and LWPS are
> correlated across participants, so a neural score could relate to both. To
> test specificity, each behavioral score was regressed (ordinary least squares,
> all variables *z*-scored) on both neural scores together, and the coefficient
> of the matched neural score was compared with that of the other. The two
> RT-adjusted matched correlations were the primary tests. **[If the cross
> pairings are reported, state that p-values are uncorrected across the eight
> correlations: matched and cross, adjusted and unadjusted.]**
>
> **Reliability and the correlation ceiling.** An observed correlation cannot
> exceed the square root of the product of the two scores' reliabilities. We
> estimated each score's reliability by splitting every participant's trials
> into random halves, stratified on the 16 design cells, 200 times. Each split
> was drawn once per participant and applied to all of its electrodes. Drawing
> it separately per electrode would let noise common to a participant's
> electrodes masquerade as reliability. All scores were recomputed in each half.
> The half-length reliability was the across-participant correlation of half-A
> with half-B values, averaged over splits, and was stepped up to full length
> with the Spearman–Brown formula. It was treated as unmeasurable when the
> half-length correlation was zero or negative. The behavioral summary table has
> no trial-level data, so its reliability was estimated from the same contrasts
> computed on the RTs of the recorded trials. These agreed closely with the
> table across participants (r = 0.90 for LWPC and 0.76 for LWPS, 20
> participants).
>
> The ceiling on each brain–behavior correlation was the square root of the
> product of the neural and behavioral reliabilities. To express what the
> ceiling means for detection, we simulated 40,000 samples of 17 participants
> from a bivariate normal distribution whose correlation was the ceiling times
> an assumed true correlation, and counted the samples reaching |r| ≥ 0.48. The
> sample size needed for 80% power was obtained from Fisher's *z*. Both treat
> the estimated reliabilities as known, so they are approximate.

The optional paragraph on the exploratory participant summaries (mean |d|,
positive electrodes only) is in the source. Use it only together with the
optional Results sentence in §3.7.

---

## 3. Results

### 3.1 Concurrent adaptation in behavior (F1) ⬜

**[To write.]** Report both adaptations in RT and errors, in the same
participants and sessions, then one scoping sentence: "Neither proportion
manipulation affected the other process (congruency × switch proportion,
switch type × incongruent proportion; Supplementary S4), so we focus on the two
within-process adaptation effects." Check the source script first (§1.4, F1).

### 3.2 Adaptation effects in lPFC high gamma (F3) ⬜

**[To write from the power traces and the direction tests.]** For each effect:
the two simple effects' signs and clusters, the difference of differences and
its cluster, the direction against behavior, `n_electrodes` and `n_subjects`,
the per-participant tally and the leave-one-out range.

### 3.3 Decoding the adaptation effects (F4) ⬜

**[To write from the decoding runs.]** For each effect: when congruency (switch
type) is decodable in each block type, peak accuracy, the cluster where the two
block types differ, trial counts per class, and any pre-stimulus windows.

### 3.4 How the two adaptations are organized across lPFC (F5)

*Copied from §17.3 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#173-results).
Every number was checked against the run's outputs on 2026-10-01; §17.5 there
gives each number's source file.*

> **LWPC and LWPS share electrodes, and their balance varies across lPFC.** We
> scored every lPFC electrode (398 electrodes, 22 participants) for LWPC and
> LWPS. Across this anatomically defined set both adaptation effects were small
> on average (mean Cohen's *d* = 0.01 for LWPC and 0.05 for LWPS; 49 % and 56 %
> of electrodes positive), and single-electrode estimates were noisy
> (full-data split-half reliability 0.27–0.30), so we tested only
> population-level summaries. The two effects shared electrodes. LWPC and LWPS
> scores from separate halves of the trials were positively correlated within
> participants (Spearman *r* = 0.10, *p* < 0.001; 397 electrodes, 21
> participants), and electrodes positive for each effect did not differ in
> location (centroid distance 1.4 mm, *p* = 0.95).
>
> The balance between the two effects (LWPC − LWPS within each electrode)
> nevertheless differed across Destrieux parcels (19 parcels sampled in at
> least three participants; *F* = 1.91, label-exchange permutation
> *p* = 0.010; *p* = 0.003–0.057 with each participant left out). No single
> parcel differed from zero after FDR correction (all *q* ≥ 0.13).
> Descriptively, the superior frontal gyrus and sulcus leaned toward LWPS. The
> balance also varied with position (MNI coordinates: block *F* = 2.85,
> *p* = 0.031). Relative to LWPS, LWPC was weaker dorsally (*z* slope −0.0077
> SD/mm, *p* = 0.008; Bonferroni-corrected over three axes, *p* = 0.023). The
> predicted anterior–posterior axis showed no effect (*p* = 0.58). Height
> explained about 2 % of the balance's variance, and the slope replicated
> across independent halves of the trials (*p* = 0.005). Height and distance
> from the midline are correlated across lPFC electrodes (within-participant
> *r* = −0.58), so the two cannot be fully separated. In an exploratory model,
> distance from the midline described the gradient better than height
> (distance *p* = 0.019, height *p* = 0.53), so we describe it as dorsomedial
> versus ventrolateral. In an exploratory breakdown by score, LWPC carried the
> gradient: it was slightly reversed within about 27 mm of the midline
> (adjusted *d* = −0.07) and positive farther out (0.04–0.06; slope
> *p* = 0.006), whereas LWPS did not vary (*p* = 0.75).
>
> **The adaptation balance tracks the balance of the demands.** To ask whether
> this organization follows the demands being regulated, we scored the
> congruency and switch-type main effects from the same trials and halves,
> weighting the proportion blocks equally. The base effects also shared
> electrodes (*r* = 0.23, *p* < 0.001). Their balance (congruency − switch)
> also differed across parcels (*F* = 1.70, *p* = 0.017; *p* ≤ 0.058 with any
> one participant left out), in step with the adaptation balance (*r* = 0.73
> across the 19 parcel means; same sign in 13). Electrode by electrode, on
> separate trial halves, the two balances were linked: where congruency
> dominated, LWPC dominated (*r* = 0.09, *p* < 0.001), and the link held with
> coordinates partialled out (*r* = 0.09, *p* < 0.001). It was process-specific:
> each adaptation effect tracked its own main effect more closely than the
> other one (congruency–LWPC *r* = 0.22 vs switch–LWPC 0.12; switch–LWPS 0.17
> vs congruency–LWPS 0.13). The base-effect balance showed no significant
> gradient (*F* = 1.81, *p* = 0.13); its slope pointed the same way at about
> half the size. Neither congruency nor switch varied detectably with distance
> from the midline (*p* ≥ 0.34). Adding the base-effect balance as a covariate
> reduced the adaptation gradient by 10 % (*z* slope still *p* = 0.015).
> Because the base-effect balance was measured with low reliability
> (split-half *r* = 0.08), full inheritance would produce a reduction of only
> about 15 %, and the participant-bootstrap interval (−20 % to 40 %) includes
> both no inheritance and full inheritance.
>
> Together, LWPC and LWPS adaptation are carried by one intermixed lPFC
> population. Electrode by electrode, their balance follows the local balance
> of the demands each regulates, and it shifts modestly between dorsomedial and
> ventrolateral lPFC. Whether that shift is inherited from the base effects
> could not be determined. In the task-responsive subset (171 electrodes, 21
> participants) the estimates were similar but not significant (Supplementary
> **[S-N4]**).

### 3.5 Anatomy supplement (S-N4)

*The first paragraph is adapted from the task-significant draft in §15.12 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#1512-reporting),
recast as the consistency check now that all lPFC is primary. The others are
written here from the numbers in §16.6.6 and the §17.3 supplement additions of
that doc. Add the task-significant main-effect results once the rerun is done.*

> **Task-responsive subset.** We repeated the anatomical analyses on the
> task-responsive lPFC electrodes (171 electrodes, 21 participants). Both
> effects were positive on average (mean Cohen's *d* = 0.14 for LWPC and 0.18
> for LWPS), and single-electrode estimates were again noisy (split-half
> reliability 0.26–0.39). LWPC and LWPS scores measured on separate halves of
> the trials were weakly correlated (Spearman *r* = 0.08, *p* = 0.057; 167
> electrodes from 18 participants with at least three electrodes), electrodes
> positive for each effect did not differ in location (centroid distance
> 3.3 mm, *p* = 0.44), and the balance between the two effects did not differ
> across Destrieux parcels (*F* = 0.91, *p* = 0.29). The height slope of the
> balance matched that in all lPFC (−0.0075 SD/mm, *p* = 0.24; difference from
> the remaining electrodes, *p* = 0.67). With the observed slope planted, this
> electrode layout reaches *p* < 0.05 in 26 % of simulations, against 66 % for
> all lPFC electrodes. **[Main-effect results for this subset, once run.]**
>
> **Hemispheres.** Fitted within each hemisphere, the adaptation balance varied
> with position in the left (*F* = 4.22, *p* = 0.005; 254 electrodes) but not
> detectably in the right (*F* = 2.33, *p* = 0.12; 144 electrodes). The
> base-effect balance likewise varied in the left (*F* = 2.45, *p* = 0.039) but
> not the right (*F* = 0.68, *p* = 0.45).
>
> **Is the gradient inherited?** Adding the base-effect balance as a covariate
> reduced the height slope of the adaptation balance from −0.0077 to −0.0069
> SD/mm (*p* = 0.015 with the covariate) and the distance-from-midline slope
> from +0.0139 to +0.0125 SD/mm (*p* = 0.030), a 10 % reduction in both. With
> the base-effect balance taken from the opposite trial half, the height slope
> fell by 5 %. Because the base-effect balance had a split-half reliability of
> only 0.08 (0.15 on full data), a gradient carried entirely by it would shrink
> by only about 8–15 %. The participant-bootstrap 95 % intervals for the
> reduction (height −20 % to 40 %; distance −15 % to 46 %) include both no
> inheritance and full inheritance.
>
> **Map similarity.** Pooled across participants, the separate-half LWPC–LWPS
> correlation (Pearson *r* = 0.17) was about the size of the two maps' half-data
> reliabilities (0.18 for LWPC, 0.16 for LWPS). At these reliabilities a
> noise-corrected ratio is not estimable, so we report the three values side by
> side.

Figures and tables for S-N4: per-parcel bars (`delta_by_roi.png`,
`dm_by_roi.png`), coverage (`coverage_matrix.csv`), leave-one-out tables
(`delta_roi_loso.csv`, `dm_roi_loso.csv`), both coordinate tables
(`score_anatomy.json` → `coordinates`, `dm_coordinates.csv`), the height vs
midline models (script §1), panel c by height (`panel_c_height.csv`), and Test 2
(`tilt_with_dm.csv`, script §5).

### 3.6 Congruency ↔ switch-type cross-decoding (S5)

*Copied from [`methods.md` › A4](methods.md#results-draft), Results (draft). It
states only what the full-trial runs support. Choose the bracketed variants once
the seeds and the electrode-matched region comparison are in.*

> Congruency and switch type were each decodable from the task-responsive lPFC
> population. Congruency was decodable from the window centred at +0.12 s, with
> peak accuracy 0.76. Switch type was decodable from +0.31 s, with peak 0.76. A
> congruency decoder also predicted switch type, and a switch-type decoder
> predicted congruency, but only from the window centred at +0.62 s (covering
> 0.50–0.75 s) onward, and well below the within-contrast accuracy. The
> congruency decoder retained 47% of switch type's above-chance accuracy, and
> was below it in 19 of 37 windows. The switch-type decoder retained 26% of
> congruency's, and was below it in 28 windows. No window before the stimulus
> was significant in any of the four decodes. Removing each participant's mean
> activity lowered both within-contrast accuracies (peaks 0.63 and 0.65) but
> left the transfers nearly unchanged (15 and 14 windows; 81% and 77%
> retained). The shared component is therefore not a uniform rise in activity.
> The two contrasts thus engage the same electrodes along largely distinct
> population codes, sharing a component that appears only late in the trial.
>
> **[One of the following, depending on the seeds:]** *(if the RT-matched −
> random gap exceeds the seed spread)* After matching RTs across the four
> cells, the transfer was reduced relative to a trial-count control
> (congruency → switch: 43% vs 70% retained; switch → congruency: 12% vs 39%)
> and confined to 0.5–1.1 s after the stimulus, before most responses (median
> RT 1.17 s). Part of the shared component therefore reflects the RT difference
> shared by incongruent and switch trials. *(otherwise)* RT matching did not
> change the transfer beyond the variability between pseudopopulation draws.
>
> **[Region sentence, once lPFC has been subsampled to 54 electrodes:]**
> Occipital electrodes showed a transfer retaining a similar share of their
> ceilings (54% and 23%), but at +1.0 s and later rather than at 0.5–1.0 s.

⚠️ Before quoting the RT-matched switch → congruency share (12 %), recompute it
over post-stimulus windows only: 6 of that ceiling's 20 significant windows are
pre-stimulus ([`decoding.md` › A4 §13.8.3](decoding.md#138-results-2026-10-01)).

### 3.7 Brain–behavior (S-BB)

*Copied from §14.2 of [`a6_brain_behavior.md`](a6_brain_behavior.md#142-results-supplement-s-bb).*

> **Brain–behavior.** We asked whether participants whose lateral prefrontal
> high gamma adapted more also adapted more in behavior. Of the 21 participants
> with task-significant lPFC electrodes, 17 had at least three usable
> electrodes and behavioral scores (median 7 electrodes and 397 trials per
> participant). Behavioral adaptation was reliable across participants
> (split-half reliability, estimated on the recorded trials: 0.72 for LWPC,
> 0.56 for LWPS). Participant-level neural adaptation was not: neural LWPC had
> no measurable between-participant reliability, and neural LWPS had a
> reliability of 0.38. These reliabilities cap the observable LWPS correlation
> at 0.46, below the |r| = 0.48 needed for significance at n = 17. Even a
> perfect underlying relationship would have reached significance in fewer than
> half of samples of this size (simulated power 0.49), and a moderate one
> (r = 0.5) in 14 %. For LWPC, no correlation was detectable at all. Neither
> correlation differed from zero after removing the RT-linked component of high
> gamma (LWPC: r = 0.10, 95 % CI [−0.40, 0.55], p = 0.70; LWPS: r = −0.06
> [−0.52, 0.44], p = 0.83; Supplementary Fig. S-BB) or without that adjustment
> (LWPC: r = 0.24 [−0.27, 0.65], p = 0.35; LWPS: r = 0.19 [−0.32, 0.61],
> p = 0.47). The unadjusted correlations were somewhat larger, as expected if
> part of each neural score reflects trial-by-trial coupling between high gamma
> and RT (median within-cell r = 0.13). With these intervals, the data neither
> support nor rule out a relationship between individual differences in neural
> and behavioral adaptation.

Optional sentence, only if the exploratory summaries are in the Methods:

> Two alternative participant summaries, the mean absolute score and the mean
> over positively scoring electrodes, gave no interpretable result: the first
> tracked participants' trial counts (LWPS: r = −0.61), as expected of a score
> driven by noise, and the second retained only 11 participants.

---

## 4. Discussion sentences

**Base-effect codes (S5).** *Written here from the "supported now" statement in
[`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01). It goes next
to the N4 overlap result.*

> The same task-responsive lPFC electrodes carried both base effects,
> congruency and switch type, along largely distinct population codes: a
> decoder trained on one recovered only part of the other's decodable
> information, and only from about 0.5 s after the stimulus (Supplementary
> S5).

**Brain–behavior (S-BB).** *Copied from §14.3 of
[`a6_brain_behavior.md`](a6_brain_behavior.md#143-discussion-sentence).*

> We could not test whether individual differences in lPFC adaptation track
> individual differences in behavioral adaptation. With 17 participants, the
> split-half reliability of participant-level difference-of-differences scores
> left any across-participant correlation undetectable, even for a strong
> underlying relationship (Supplementary Note S-BB). Difference scores are
> characteristically unreliable across individuals (Hedge et al., 2018), so such
> a test needs many more participants, many more trials per participant, or
> both.

**Anatomy limits to state** (§17.4 of the N4 doc): the effects are small (height
explains ~2 % of the balance's variance; no parcel survives FDR); LWPC averages
*d* ≈ 0.01 across all lPFC, so the gradient is LWPC slightly reversed medially
and slightly positive laterally; the axis was not predicted and the predicted
anterior–posterior axis is null; and inheritance could not be resolved.

---

## 5. Figure captions

### 5.1 Fig. 5 (draft)

*Drafted here from the panel specifications in §1.4 and the numbers in §3.4.
Panel a's caption text follows the
[closing figure plan › Panel a](analysis_plans.md#panel-a).*

> **Fig. 5 | LWPC and LWPS adaptation share lPFC electrodes, and their balance
> follows the balance of the demands they regulate.** All lPFC electrodes (398
> electrodes, 22 participants). **a**, Each point is one electrode's score from
> all trials, after regressing out overall responsiveness and subtracting each
> participant's mean (397 electrodes from the 21 participants with at least
> three). Left, congruency against switch-type main effect; right, LWPC against
> LWPS. *r* is the pre-specified test: one score from one half of the trials
> against the other score from the other half, averaged over 1,000 random
> splits (Spearman; within-participant permutation null). Because each half has
> half the trials, *r* is smaller than the correlation among the plotted
> points. **b**, The two balances by Destrieux parcel: adjusted mean
> congruency − switch (*x*) against adjusted mean LWPC − LWPS (*y*), one point
> per parcel sampled in at least three participants (19 parcels), sized by
> electrode count and coloured by distance from the midline. *F* and *p* are
> each balance's parcel test (label-exchange permutation). The parcel means
> share trials, so their correlation is descriptive. **c**, Congruency, switch,
> LWPC and LWPS (Cohen's *d*, participant and responsiveness removed) by
> tertile of distance from the midline; mean ± SEM across participants.
> Exploratory. **d**, Electrode-level correlations on separate trial halves
> between congruency − switch and LWPC − LWPS, with and without MNI
> coordinates partialled out, and for the matched (congruency–LWPC,
> switch–LWPS) and crossed (congruency–LWPS, switch–LWPC) pairings. Spearman
> *r*, within-participant permutation *p*.

### 5.2 Supplementary Fig. S-BB

*Copied from §14.4 of [`a6_brain_behavior.md`](a6_brain_behavior.md#144-figure-caption-supplementary-fig-s-bb).
Use `participant_brain_behavior_scatter_rtadj`, annotated with r, its 95 % CI
and the ceiling instead of R² (R² drops the sign; LWPS is r = −0.06).*

> **Supplementary Fig. S-BB | Neural against behavioral adaptation across
> participants.** Each point is one participant (n = 17). *x*: behavioral LWPC
> (left) and LWPS (right), the difference of differences in mean RT (ms), low-
> minus high-proportion blocks. *y*: the same contrast in high gamma (0–1.5 s
> after stimulus onset), as Cohen's *d* averaged over the participant's
> task-significant lPFC electrodes, after removing the component of high gamma
> linearly related to RT within condition cells. Positive values on both axes
> indicate the predicted adaptation. Lines are least-squares fits. Each panel
> gives Pearson's r with its 95 % CI and p. The ceiling (LWPS: 0.46) is the
> largest correlation expected for a perfect underlying relationship, given the
> split-half reliabilities of the two scores. Neural LWPC had no measurable
> reliability, so no ceiling is given. Significance at α = .05 required
> |r| ≥ 0.48.

### 5.3 Still to write

F1–F4 captions, Supplementary S5 (cross-decoding traces with ceilings, transfer
and control runs), and S-N4.
