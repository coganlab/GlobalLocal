# Stability vs. Flexibility: Shared or Distinct Neural Substrates?

**Analysis plan.** This document lays out the full battery of analyses for the
question: *do stability (LWPC / proactive control) and flexibility (LWPS /
reactive control) rely on shared or distinct iEEG substrates?* Specifically:
are there **distinct subpopulations** supporting one process but not the other,
or only **shared populations** carrying both?

The battery has three logical layers, each answering a different sub-question,
plus cross-cutting statistics notes:

| Layer | Sub-question | Analyses |
|---|---|---|
| **Population organization** | Are the *same electrodes* selective for both, or distinct sets? | ANOVA electrode definition → overlap/conjunction null; anatomy (brain maps, ROI histograms) |
| **Representation / information** | Do the two processes share a *code*, or just co-locate? | Cross-decoding (label-transfer, electrode-set-transfer) |
| **Dynamics** | Does one process's information arise *earlier* than the other's? | Time-series characterization; onset-latency comparison (jackknife) |
| **Behavioral link** | Does this selectivity track the actual control adjustment? | Brain–behavior correlation |

Two constructs, defined as two-way interactions on single-trial high-gamma (HG):

- **LWPC (stability)** = `congruency × incongruent_proportion` interaction
  (congruency effect grows in high-incongruent-proportion blocks).
- **LWPS (flexibility)** = `switchType × switch_proportion` interaction
  (switch effect grows in high-switch-proportion blocks).

> **Frequency scope.** Constructs are defined on **high-gamma** as a proxy for
> local activity. Conflict (theta) and switching (beta) have well-known
> low-frequency signatures, and the neural cross-effects (Fig 9) may live there.
> HG is the primary analysis; re-run the conjunction and decoding once in
> low-frequency bands as a robustness / "where does the cross-effect live" check.

> **Why not one four-way ANOVA?** The four-way
> (`congruency × switchType × inc_prop × switch_prop`) has uninterpretable,
> underpowered high-order terms. Two focused two-way interactions map directly
> onto the theoretical constructs. The two *cross* interactions
> (`congruency × switch_prop`, `switchType × inc_prop`) are **not**
> characterizations of interest — keep them only as **specificity controls**
> (LWPC should ride on inc-prop, not switch-prop). *In univariate HG these
> cross-terms should be null.* Note the contrast with **decoding** (§4, Fig 9),
> where cross-effects **do** appear: the ANOVA cross-terms staying null while a
> classifier finds cross-structure is itself informative — see the central
> dissociation below.

Much of the population-organization layer is already implemented in
`src/analysis/stats/stability_flexibility_segregation.py` (per-electrode
labels, subject-stratified conjunction / CMH). This plan reframes it around the
ANOVA electrode definitions and adds the representation and dynamics layers.

---

## Where this fits in the paper (figure plan)

This battery is the analytic core of the paper. "Shared vs distinct" is not one
question but **three levels**, and the answer need not be the same at each:

1. **Anatomical / electrode overlap** — are the *same sites* selective for both?
   → §2, §3 (Figs 5, 7).
2. **Single-channel tuning** — within a site, does the same channel carry both
   signals? → §2 continuous correlation, §4a on the *both* group (Figs 7, 8).
3. **Population representational format** — is it the *same code*? → §4 / §5
   (Figs 9, 10).

Full figure sequence and how each maps onto the layers:

| Fig | Content | Role | Section |
|---|---|---|---|
| 1 | Behavior: LWPC + LWPS effects, **no behavioral cross-effects** | Establishes both adaptations exist and are *behaviorally independent* — the puzzle | motivation |
| 2 | Time–frequency (wavelet): congruency (inc−con), switch cost (switch−repeat) | Signal validation; motivates HG focus | §1 setup |
| 3 | High-gamma rises after stimulus onset | Signal validation | §1 setup |
| 4 | HG power traces: LWPC & LWPS within-trial; **pre-trial cross-effects** | Effects + the tonic/baseline issue (§0.7) | §1, §5 |
| 5 | 2×2 conjunction (electrode counts) + stats | Are the same sites selective for both? | §2 |
| 6 | Onset latency (jackknife, 50%-of-peak) | Sequence: does one precede the other? | §5 |
| 7 | Segregation: conjunction OR **+ continuous effect-size correlation** | Core anatomical answer | §2 |
| 8 | Orthogonal power traces (define on LWPC → LWPS trace, and vice versa) | Cross-contrast functional confirmation | §2 / §4 (univariate) |
| 9 | Within-block decoding (inc/con and switch/repeat by block type), incl. neural cross-effects | Readable information + behavior/neural dissociation | §4 |
| 10 | Cross-decoding (label transfer) + temporal generalization matrices | Shared code vs merely co-located | §4, §5 |

**Central dissociation to foreground.** Fig 1 shows *no behavioral crossover*,
yet Fig 9 shows *neural* cross-effects (congruency decoding differs by
switch-proportion block, and vice versa). This **behavior-independent /
neural-interacting** dissociation is a headline result, not a nuisance —
*provided* it survives the decoding confounds in §0.8 (RT / difficulty leakage,
trial-count matching, univariate mean-offset removal). Treat the *behavioral*
cross-interactions as specificity controls (they should be null); treat the
*neural* cross-effects as a finding to confound-proof.

Where the segregation analysis "slots in": Figs 5, 7, 8 are one anatomical-answer
block (§2), with the **continuous effect-size correlation** (Fig 7) as its
robust, threshold-free headline; the decoding (Figs 9–10, §4) is the
representational-level answer. The paper's conclusion is the *conjunction* of the
two levels (the §4 payoff 2×2).

---

## 0. Cross-cutting statistical principles (read first)

These apply to every analysis below; they are the difference between a real
result and an artifact.

1. **Double-dipping / selection bias.** Defining electrodes on contrast A and
   then reporting contrast A's effect (or A's decoding, or A's onset) in that
   group is circular — the estimate is biased by the winner's curse.
   - The **cross-contrast** direction is clean: define on **LWPC**, test **LWPS**
     (and vice versa). This is the workhorse.
   - Anything reported on the **selection** contrast (e.g. the LWPC time series
     of the LWPC group) must come from **held-out trials** (disjoint-half; see
     `_stratified_half_split` in the segregation module) or be labeled
     descriptive-only.
2. **Disjoint trial halves.** Even the cross-contrast test can be coupled by
   shared trial-level noise, since LWPC and LWPS are estimated from the same
   trials. For airtight tests, estimate the selection contrast and the test
   contrast on disjoint halves.
3. **Power matching.** LWPC and LWPS almost certainly differ in effect size, so
   the stronger contrast will "recruit" more electrodes at a fixed α. A raw
   overlap asymmetry is then partly a power artifact. Report electrode counts
   and effects **as a function of threshold**, and report each contrast's power
   (e.g. via the effect-size distribution), not a single-α snapshot.
4. **Multiple comparisons.** FDR (Benjamini–Hochberg) across electrodes for the
   per-electrode selection tests (already done in `per_electrode_labels`).
5. **Coverage bias.** iEEG coverage is clinically determined. Any anatomical
   claim (ROI differences) must be conditioned on coverage, or it reflects
   *where electrodes are*, not function.
6. **Latency–amplitude confound.** A larger effect crosses any onset threshold
   sooner. Any "X is earlier than Y" claim must guard against X simply being
   bigger than Y (see §5).
7. **Tonic / pre-trial baseline (do not handwave).** List-wide manipulations
   induce a *sustained* block-level state that is present **before** stimulus
   onset. Pre-trial "cross-effects" (Fig 4) may therefore be genuine tonic
   proactive-control signals, not artifacts — but they poison any baseline
   correction that spans them (you would subtract the very effect you study).
   Use a baseline that predates the block context (pre-block, or a robust common
   baseline), report the pre-trial effect explicitly rather than normalizing it
   away, and separate the **tonic** (sustained, block-level) from the **phasic**
   (stimulus-evoked) component. This is a substantive result about proactive
   control, not a cleanup step.
8. **Decoding confounds (§4, Figs 9–10).** Block types differ in difficulty and
   RT, so a classifier can exploit RT-correlated power or a univariate mean
   offset instead of a genuine control code. Before interpreting any decoding
   effect — especially the neural *cross*-effects — match trial counts across
   blocks, regress out or match RT, and confirm the effect survives
   per-condition mean removal.

---

## 1. Electrode definition (per-electrode two-way ANOVAs)

**Goal.** Label each electrode as LWPC-selective and/or LWPS-selective.

**Method.**
- For each electrode, fit a two-way ANOVA on single-trial HG (aggregated over
  the analysis window, baseline-normalized):
  - LWPC electrode ⇐ significant `congruency × incongruent_proportion`
    interaction term.
  - LWPS electrode ⇐ significant `switchType × switch_proportion` interaction.
- Significance via **within-electrode permutation** of the interaction contrast
  (shuffle the ±1 interaction labels), then **FDR across electrodes**. This is
  exactly what `per_electrode_labels` already does with the interaction
  contrast (`contrast_mode='proportion'`), producing binary `S` (stability) and
  `F` (flexibility) flags per electrode.
- Also fit the two **cross** interactions as specificity controls (report, do
  not select on them).

**Output.** Per-electrode table: `{subject, electrode, ROI, S (LWPC), F (LWPS),
effect sizes, q-values}`, plus the four groups: **LWPC-only**, **LWPS-only**,
**both**, **neither**.

**Code hooks.** `per_electrode_labels(df, contrast_mode='proportion', ...)`
already returns `S`/`F`. The ANOVA framing is a thin wrapper — the interaction
contrast in `_CONTRAST_PRESETS['proportion']` is the same 2×2 difference-of-
differences an interaction F-term tests.

---

## 2. Overlap / conjunction test — *is the co-localization more or less than chance?*

**Goal.** The core "distinct vs shared populations" test.

**Method.**
- Build the 2×2 conjunction per subject: `both / LWPC-only / LWPS-only /
  neither`, then pool with **Cochran–Mantel–Haenszel** (subject-stratified).
  - MH odds ratio **< 1** → segregation (fewer double-selective electrodes than
    chance); **> 1** → shared/overlap core; **≈ 1** → independent.
- **Null distribution of counts.** Two equivalent framings:
  - *Analytic:* expected overlap under independence ≈ `(n_LWPC/N)·(n_LWPS/N)·N`;
    compare observed to hypergeometric.
  - *Permutation (preferred, respects nesting):* shuffle the LWPC / LWPS
    selection labels **within each subject** and recompute the overlap count →
    empirical null. Report observed overlap vs. null with a p-value.
- **Threshold sweep.** Recompute the overlap OR across a range of selection
  thresholds and plot it — a segregation claim should be stable across
  thresholds, not an artifact of one α.
- **Continuous (threshold-free) segregation — Fig 7 headline.** The conjunction
  thresholds each contrast, so its verdict can hinge on α. The threshold-free
  complement correlates the **per-electrode LWPC effect size** against the
  **per-electrode LWPS effect size** across all electrodes (using the effect
  sizes `per_electrode_labels` already returns). *Positive* correlation → shared
  tuning (channels selective for one tend to be selective for the other); *≈ 0*
  → segregation; report per-ROI as well as pooled. Estimate the two effect sizes
  on **disjoint trial halves** so shared trial-level noise cannot inflate the
  correlation, and get a null by within-subject permutation of the block labels.
  This is the robust version of the same question the OR asks categorically.

**What it answers.** Whether stability- and flexibility-selective electrodes
are the *same sites* more/less than chance. **Positive evidence for
segregation** is possible here (OR<1) — something a decoding null *cannot*
provide.

**Limitation (why we also need §4).** Co-localization ≠ shared code. "Both"
electrodes can be a genuinely shared representation **or** mixed selectivity
with orthogonal codes. This test cannot tell them apart.

**Code hooks.** `cmh_conjunction(labels)` — returns MH OR, CMH test,
homogeneity test, per-subject tables, pooled Fisher. Add the within-subject
permutation null and the threshold sweep as wrappers.

---

## 3. Anatomy — brain maps & ROI distributions

**Goal.** Are the distinct subpopulations in *different places*? (Greg's plots.)

**Method.**
- Plot LWPC-only, LWPS-only, and both electrodes on the brain (existing vis
  pipeline, `src/analysis/vis`), color-coded by group.
- Histogram of ROI membership for each group.
- **Test** ROI-distribution differences between groups with a permutation /
  χ² on the ROI counts — but **condition on coverage** (restrict to ROIs
  sampled in ≥ k subjects; report per-subject coverage) so a difference isn't
  just electrode placement.

**What it answers.** Descriptive anatomical dissociation. Supports, but does not
by itself establish, distinct substrates.

---

## 4. Cross-decoding — *shared code or just co-located?*

**Goal.** The representation-level test that disambiguates the "both" group.
This is the piece the counting analyses cannot do.

**Three complementary designs (plus a within-block baseline):**

**(0) Within-block decoding baseline (Fig 9).** Before any transfer, establish
that each contrast is decodable and compare across blocks: decode inc/con within
mostly-congruent vs mostly-incongruent blocks (and switch/repeat within
mostly-repeat vs mostly-switch blocks), comparing accuracies. This is the
decoding analog of the univariate LWPC/LWPS effects. It is also where the
**neural cross-effects** surface — congruency decoding differing by
switch-proportion block, and switch decoding by inc-proportion block — the
Fig-1-vs-Fig-9 dissociation. Interpret only after the §0.8 confound controls.

**(a) Label-transfer (within an electrode set).** Train a decoder on the
stability contrast, test on the flexibility contrast (and vice versa), on the
*same* electrodes.
- Successful transfer ⇒ the two processes share a representational geometry (a
  common coding axis).
- Run separately on LWPC-only, LWPS-only, and both groups. Prediction: the
  *both* group cross-decodes; the distinct groups do not.

**(b) Electrode-set-transfer.** Train on one electrode set, test on the other
(e.g. train on LWPC electrodes, test on LWPS electrodes) for the *same* label.
- Tests whether the two subpopulations carry interchangeable information.

**(c) Temporal generalization (Fig 10).** Train a decoder at time *t*, test at
time *t′*, filling a train-time × test-time matrix — run it both **within** a
contrast (does the code stay stable or move across the trial?) and **across**
contrasts (cross-temporal label transfer).
- Off-diagonal generalization ⇒ a *sustained / stable* code; a narrow diagonal
  ⇒ a *moving / phasic* code that recodes over time.
- Reading the across-contrast matrix together with §5's univariate onset gives
  the full temporal-format picture: *when* a shared component (if any) exists and
  whether it is stationary. Same pseudo-trial construction and disjoint
  train/test discipline as (a)/(b).

**Pooling / pseudopopulation.** We pool electrodes across subjects into a
pseudopopulation (standard in the decoding literature). The one construction
detail that matters: since subjects don't share trials, build **pseudo-trials**
by aligning within condition cells (match on congruency × inc-prop ×
switchType × switch-prop), and keep **train/test trials disjoint** so accuracy
isn't inflated by trial reuse. Cross-validate over pseudo-trial folds; get a
null by permuting the transferred labels.

**Circularity note.** Selection bias only bites the **test** side: as long as
electrodes/trials used to *define* a group aren't the ones you compute the
transferred (test) accuracy on, the generalization estimate is unbiased. So
define groups and select decoding-test trials independently.

**What it answers.** Positive transfer = shared information (strong, directional
evidence). A transfer *null* is weaker (power-limited) — which is exactly why
we pair it with §2's positive segregation evidence.

**Code hooks.** `src/analysis/decoding/decoding.py` (existing decoding
machinery) for the base classifier + CV; add the cross-condition train/test
split and the pseudo-trial construction.

**The payoff 2×2.** Reading §2 and §4 together:

| | Cross-decodes | Doesn't cross-decode |
|---|---|---|
| **Co-localized (OR>1)** | shared substrate / shared code | mixed selectivity, orthogonal codes |
| **Not co-localized (OR<1)** | (rare) shared low-D code across sites | **distinct substrates** |

---

## 5. Temporal dynamics — relative onset of stability vs. flexibility information

> **Status: specified.** Information signal = univariate interaction time course
> (Step 1); onset = 50%-of-peak latency via jackknife (Step 2–3); multivariate
> decoding onset from §4 as a secondary corroborating check.

**Goal.** Does stability information arise *earlier or later* than flexibility
information? Theory motivates it: proactive/stability control is often cast as
sustained/tonic and preparatory (earlier), reactive/flexibility control as
phasic (later) — but this is exactly what we want to measure, not assume.

**Step 1 — define an "information time course" for each process: univariate
(decided).** The primary signal is the **interaction magnitude over time** — the
LWPC and LWPS difference-of-differences as functions of time (grand average over
the relevant electrodes, or a t-statistic time course). This measures *when the
response emerges* and is the cleaner "onset of information," less entangled with
classifier choices; it reuses the `effect_measure='cluster'` time-resolved path
already in the segregation module.
- *Corroborating check (secondary):* the time-resolved decoding-accuracy
  courses from §4 (train/test per time bin) measure *when the information
  becomes readable* — a subtly different quantity (readout latency can lag
  response onset). Since the decoders are built for §4 anyway, agreement between
  the univariate and multivariate onsets makes the sequence claim bulletproof.

**Step 2 — define onset latency: 50%-of-peak (decided).** For each process,
find the peak of the effect in the expected direction (or of |effect|) within
the window, then take the **first upward crossing of 50% of that peak** on the
rising flank.
- **Why this defeats the latency–amplitude confound.** Normalizing to each
  process's *own* peak removes a pure multiplicative amplitude difference: if
  `stab(t) = k·flex(t)`, both cross 50%-of-peak at the *same* time, whereas an
  absolute threshold would make the larger effect cross spuriously early. So we
  do **not** additionally amplitude-match (subsampling to equalize peaks would
  only cost power without adding protection).
- It is still a single-point crossing on the rising edge, hence noise-sensitive
  — which is exactly what the jackknife (Step 3) absorbs by measuring it on
  smooth leave-one-subject-out averages, not per subject.
- **Cross-check:** also report **peak latency** alongside onset. Residual
  differences in waveform *shape* (broad plateau vs. sharp transient) can move
  the 50%-of-peak point — a genuine dynamics difference, not an artifact — so a
  consistent onset+peak story is the robust claim.
- Note: this is *not* fractional-area latency (50% of cumulative area), which is
  a center-of-mass measure of the whole waveform rather than an onset;
  50%-of-peak reads more naturally as onset and is the standard ERP measure.

**Step 3 — jackknife the latency comparison (Miller/Ulrich).** Single-subject
latency estimates are too noisy; the jackknife estimates latency on smooth
leave-one-subject-out grand-averages instead:
- For each of N leave-one-out subaverages, measure LWPC onset and LWPS onset.
- Jackknife SE from the pseudovalues (variance inflated by `(N−1)`); compare the
  two onsets with the **Ulrich–Miller corrected paired t** (`t_c = t / (N−1)`).
- Cross-checks: bootstrap over subjects/electrodes; report the paired
  onset *difference* with a CI, not just a p-value.

**Latency–amplitude confound — handled by the 50%-of-peak measure (Step 2).**
The peak-normalization neutralizes a pure "one effect is bigger" difference, so
no separate amplitude-matching step is needed; reporting peak latency alongside
onset guards the residual shape-difference case.

**What it answers.** A *sequence* claim (stability precedes flexibility, or
vice versa) that neither the overlap nor the decoding analysis speaks to — an
independent axis of "shared vs distinct."

**Code hooks.** New module (e.g. `stability_flexibility_timing.py`); reuse the
per-electrode interaction time courses (the `effect_measure='cluster'` path in
the segregation module already produces time-resolved contrasts) and the
decoding time-course machinery for the multivariate variant.

---

## 6. Brain–behavior correlation

**Goal.** Tie the neural selectivity to the actual control adjustment, so the
substrates are shown to be *functional*, not incidental.

**Two levels (power differs sharply):**
- *Across subjects (low power, n = subjects):* correlate a subject's
  LWPC/LWPS electrode count or mean effect with their behavioral LWPC/LWPS
  magnitude (the congruency-sequence and switch-proportion RT effects).
- *Within subject, single-trial (preferred):* does trial-by-trial HG in the
  LWPC group predict the trial-by-trial congruency-sequence RT adjustment
  (and LWPS group ↔ switch adjustment)? Directly links selectivity to control,
  with far more power.

---

## Summary — how the pieces combine

- **§2 (overlap null)** gives *positive evidence for segregation* if OR<1 —
  distinct populations. Cross-decoding cannot do this.
- **§4 (cross-decoding)** disambiguates the *both* group: shared code vs. mixed
  selectivity. The counting cannot do this.
- **§5 (timing)** adds an orthogonal *sequence* dimension.
- **§3 / §6** anchor the result anatomically and behaviorally.

No single analysis settles "shared vs distinct"; the **conjunction of §2 and
§4** (the 2×2 above) is the strong, specific claim, with §5 and §6 as
independent support. Layered onto that is the **behavior-independent /
neural-interacting dissociation** (Fig 1 vs Fig 9): behavior treats the two
adaptations as independent, but the neural data lets us ask *at what level* they
interact — same sites (§2), same code (§4), same timing (§5) — which is the
paper's throughline.

---

## Open questions to discuss

1. **Analysis window** for the aggregate-HG ANOVA electrode definition (§1) —
   fixed window vs. data-driven.
2. **Pseudopopulation construction (§4):** how to build pseudo-trials and how
   many folds, given per-subject trial counts.
3. **Baseline choice (§0.7):** which baseline avoids subtracting the tonic
   block-level state — pre-block vs. robust common baseline — and how to
   separate the tonic from the phasic component cleanly.
4. **Low-frequency robustness (frequency scope):** which bands (theta / beta) to
   re-run the conjunction and decoding in, and whether the neural cross-effects
   are stronger there than in HG.

*(§5 timing is fully specified: univariate interaction time course, 50%-of-peak
onset, jackknife.)*
