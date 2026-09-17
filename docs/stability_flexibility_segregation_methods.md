# Methods — stability/flexibility segregation analysis

Two interchangeable, self-contained Methods write-ups for the analysis implemented in
`src/analysis/stats/stability_flexibility_segregation.py` (driven by
`dcc_scripts/stats/stability_flexibility_segregation_dcc.py`):

| Version | Effect measure | Launcher setting | HG input per trial |
|---|---|---|---|
| **A** | signed supra-threshold *t* mass over the analysis window | `EFFECT_MEASURE=cluster` | time course over `[tmin, tmax]` |
| **B** | Cohen's *d* on the window-mean HG | `EFFECT_MEASURE=cohens_d` | scalar window mean |

Everything else — contrasts, disjoint-half estimator, gain control, subject-aware
inference, conjunction — is identical between them, so the two sections are
deliberately parallel and can be swapped one-for-one in a manuscript. Both are
written for the **interaction** (LWPC / LWPS) contrasts, i.e.
`CONTRAST_MODE=proportion`; a note at the end of each says what changes if the
main-effect (`condition`) contrasts are used instead.

Bracketed `[…]` items are run-dependent numbers to fill in from
`results/<tag>/…/summary.txt`, `labels.csv`, `correlation.json`, and
`conjunction.json`.

> **Implementation status (2026-09-10).** Earlier versions of this document
> carried a warning not to submit the disjoint-half paragraph: the code averaged
> *x* and *y* over the 200 splits *before* correlating them, which forfeited the
> disjoint-half correction (the average is dominated by cross terms
> `cov(x_j, y_k)`, *j ≠ k*, whose trial sets overlap ~50%). **That is fixed** —
> the correlation is now computed within each split and averaged, and the
> split-half reliability is reported alongside it. This fixes the shared-trial
> aggregation problem, but it does **not** make the current pipeline a complete
> confirmatory population analysis. In particular, splitting is still by trials
> rather than intact blocks, the interaction-label permutation still shuffles a
> block-level modulator trial by trial, inference is still electrode-weighted,
> and the `cluster` measure still returns one scalar for the requested interval
> rather than a similarity curve over time. The status table below distinguishes
> implemented improvements from remaining work; manuscript prose must retain
> these qualifications.
>
> Two things to carry into a manuscript. (i) Main-effect contrasts are now scored
> with equal cell weights, like the interactions, which matters if congruency and
> task sequence are correlated in the trial table; report the cross-tab. (ii) The
> per-electrode S/F labels behind the categorical test still use the older
> trial-count-weighted scoring, so in `CONTRAST_MODE=condition` the continuous
> and categorical arms are not scored identically. Both are detailed in
> `analysis_simplification_plan.md` §2.2–§2.2b.

## Review status of the revised implementation

This table maps the September 2026 methodological review to the code currently
on this branch. “Resolved” means the requested computation is implemented, not
that every inferential assumption has thereby been established.

| Review item | Current status | Consequence |
|---|---|---|
| Balanced adaptation contrasts | **Resolved.** Proportion mode estimates equal-cell LWPC and LWPS differences-of-differences. | Retain as the primary construct definition. |
| Four half-specific estimates and cross-fitted similarity | **Resolved.** `compute_sensitivities_per_split` returns `xA`, `xB`, `yA`, and `yB`; `split_resolved_corr` averages the two cross-half correlations within each split. | Shared trial noise is not reintroduced by averaging effects before correlation. Splits are resampling replicates, not independent observations. |
| Split-half reliability | **Partly resolved.** Half-data reliabilities and an attenuation-corrected correlation are returned. | Treat raw similarity as primary. Reliabilities are companions, not proof of a “noise ceiling”; no bootstrap confidence intervals or full-data Spearman–Brown estimates are implemented. The corrected value is omitted when either reliability is non-positive, but remains potentially unstable when reliabilities are small and positive. |
| Block-aware splitting | **Open.** The long table has no required session/run/block/trial-position contract, and halves are stratified trial splits rather than intact-block splits. | Slow block-level dependence is not handled. Confirmatory use requires identifiers and block-respecting resampling or cluster-aware modelling. |
| Interaction-label permutation | **Open / invalid for confirmatory use.** It still permutes the block-proportion modulator trial by trial within condition. | Do not use categorical electrode *p*/FDR labels or CMH overlap as confirmatory evidence until the null preserves actual block exchangeability. |
| Participant-level population inference | **Open.** The primary statistic is one pooled, within-subject-centred electrode correlation; subjects with more electrodes receive more weight. | Label it “within-subject electrode-level association.” Add per-subject estimates when coverage permits or a hierarchical subject bootstrap. `min_elec=3` is only an eligibility default, not a defensible subject-level correlation threshold. |
| Responsiveness adjustment | **Partly resolved.** The fallback bug (`|mean HG|` rather than `mean |HG|`) is fixed, but adjustment remains mandatory in the primary function and uses one pooled slope. | Report an unadjusted primary analysis and adjustment using an independent responsiveness/SNR measure as sensitivity analysis before making a mechanism claim. |
| Time-resolved similarity | **Open.** `effect_measure='cluster'` and `'peak_t'` each collapse the full interval to one electrode scalar. | Neither produces `S(t)`. Implement fixed time bins, reliability curves, and a subject-respecting across-time null before making timing claims. |
| Categorical conjunction | **Unchanged as secondary/descriptive.** CMH is subject-stratified and uninformative strata are now removed, but the labels inherit the invalid block-modulator permutation. | Anatomical S-only/F-only/both/neither maps are useful descriptively; they must not decide the shared-versus-independent conclusion. |
| Interpretation language | **Open in generated output.** `write_summary` still chooses “shared core” or “segregated” from the sign alone and uses an ad hoc reliability threshold of 0.2. | Interpret positive/negative nonsignificant estimates neutrally. “Distinct” needs reliable patterns plus equivalence to a prespecified margin; otherwise report “inconclusive.” |
| Baseline description | **Open pending verification.** The preprocessing call uses a separate baseline epochs object; paired trial-by-trial behaviour has not been demonstrated here. | Do not describe the transform as trial-by-trial unless the `ieeg.rescale` implementation and trial alignment are verified. Run local per-trial baseline subtraction as a sensitivity analysis. |

### Confirmatory priority

The current implementation is a substantially improved **continuous,
cross-fitted electrode-pattern analysis**, and the raw cross-half similarity is
the appropriate primary statistic among the outputs it currently produces. It
is nevertheless an interim analysis. The next load-bearing changes are, in
order: (1) block/run identifiers and block-respecting splits/nulls, (2)
subject-level or hierarchical subject-bootstrap uncertainty, and (3) fixed-bin
`S(t)` with reliability and across-time correction. Responsiveness adjustment
and categorical conjunction should be sensitivity/descriptive analyses rather
than prerequisites for the primary claim.

---

## Version A — time-resolved ("cluster") effect measure (`effect_measure='cluster'`)

### Single-trial high-gamma

Analyses were performed on stimulus-locked single-trial high-gamma (HG,
70–150 Hz) from [N] patients performing the Global/Local task. Broadband HG was
extracted from the cleaned, average-referenced recordings with a filterbank–Hilbert
decomposition and epoched from −1.0 to 1.5 s relative to stimulus onset. The
current preprocessing passes the signal epochs and a separately constructed
0.5-s pre-stimulus baseline epochs object to `ieeg.rescale(..., mode='zscore')`.
Whether that operation pairs each signal trial with its own aligned baseline has
not yet been verified, so it must not be described as “trial-by-trial” in a
manuscript without that verification. Epochs were decimated by a factor of 8; trials
exceeding 10 SD were treated as outliers and channels with more than 5% outlier
trials were dropped. Only correct trials were analysed, and trials whose task
sequence was undefined (first trial of a block) were excluded, leaving trials
labelled by congruency (congruent/incongruent), task sequence
(switch/repeat), and the two block-level proportions (incongruent proportion and
switch proportion). Electrode identifiers were scoped by subject, so channels
with the same name in different patients were never pooled.

For each electrode and trial we retained the **HG time course over the analysis
window** ([0.0, 0.5] s post-stimulus), i.e. the time dimension was *not*
averaged away; trials with any non-finite sample in the window were discarded.
The resulting long-format table (one row per electrode × trial) was the input to
all analyses below.

### Constructs and contrasts

Stability and flexibility were each operationalised as a two-way interaction on
single-trial HG:

* **Stability (LWPC)** = congruency × incongruent proportion — the congruency
  effect as a function of the block's incongruent proportion.
* **Flexibility (LWPS)** = task sequence × switch proportion — the switch effect
  as a function of the block's switch proportion.

Each interaction was scored as a **balanced (equal-cell-weight)
difference-of-differences** over the four 2×2 cells,
`d-o-d = (M₁₁ − M₀₁) − (M₁₀ − M₀₀)`, rather than as a pooled contrast between
the two "+1" and the two "−1" cells. This matters because the proportion design
makes the four cells deliberately unequal in trial count (≈75/25): a
trial-count-weighted pooled contrast is dominated by the frequent cells, so a
pure congruency or switch **main** effect leaks into the estimate, whereas the
equal-cell difference-of-differences is orthogonal to both main effects.
Electrodes missing any of the four cells (or with fewer than two trials in a
cell) were assigned a missing value rather than zero, so they entered neither the
correlation nor the significance count.

Interactions were treated as **two-sided**: an electrode counts as
LWPC- (or LWPS-) selective whenever its condition effect is modulated by block
proportion, whether the effect grows or shrinks in high-proportion blocks. The
behavioural adjustment has a known direction, but no neural population is
required to mirror it, so no sign was imposed at any selection step. The signed
direction of every electrode's interaction was nevertheless recorded and is
reported descriptively; it is oriented **low-proportion minus high-proportion**,
so that a positive value denotes the behavioural adaptation direction (the
condition effect is smaller in the high-proportion block) for both LWPC and LWPS,
and for the behavioural difference-of-differences it is compared against.

### Effect measure: signed supra-threshold *t* mass over time

Because the interaction may be transient within the analysis window, each
electrode's interaction magnitude was quantified with a **time-resolved**
statistic rather than a window average. For each electrode and each contrast, the
balanced difference-of-differences was computed **at every time bin** in the
window and converted to a per-bin *t* statistic (`d-o-d(t) / SE(t)`, with SE
pooled across the four cells). Bins whose |*t*| exceeded the two-tailed critical
*t* at α = 0.05 were retained, and the effect was defined as the **signed mass**
of those bins, i.e. the sum of the signed *t* over all supra-threshold bins
(0 when no bin survived). This yields a single **signed, graded scalar per
electrode** — the property the continuous correlation and the 2×2 conjunction
both require, and which a per-effect pass/fail mask cannot provide.

Three properties of this statistic should be stated explicitly. First, **no
contiguity is imposed**: every bin clearing the threshold contributes, whether or
not it is adjacent to another such bin, so the measure is a thresholded integral
over the window rather than a contiguous-cluster statistic in the
Maris–Oostenveld sense. Second, the sum is **signed**, so bins of opposite sign
partially cancel; the measure is therefore the *net* signed evidence for the
interaction across the window, which is what allows it to substitute for a signed
effect size in the correlation and conjunction. Third, the per-bin α = 0.05
threshold is a **statistic-forming** threshold, not a correction: it applies no
cluster-level or across-time multiple-comparison control by itself. Inference is
performed one level up, on this statistic as a whole — its null distribution is
obtained per electrode by permuting the block-proportion modulator within each
level of the condition factor (below), and the resulting *p* values are
FDR-corrected across electrodes.

One implementation note. A 2×2 interaction is a four-cell
difference-of-differences, not a two-sample contrast, so the pipeline's
two-condition permutation cluster test (`ieeg.calc.stats.time_perm_cluster`) does
not apply to it: permuting a two-group label would null the main effects rather
than the interaction. The interaction path therefore always uses the parametric
per-bin threshold described above, and the module's `USE_TIME_PERM_CLUSTER`
switch — which substitutes the permutation-derived cluster mask — takes effect
only for the two-group (main-effect) contrasts described at the end of this
section.

As a robustness complement, the whole analysis was repeated with an
**amplitude-only** measure (`effect_measure='peak_t'`): the signed per-bin d-o-d
*t* at the instant of maximal |*t*|. The mass statistic grows with an effect's
*duration* as well as its amplitude and is mildly trial-count sensitive; peak
*t* is timing- and duration-invariant, so a segregation verdict that holds under
both measures is not an artifact of one contrast simply lasting longer.

### Per-electrode sensitivities on disjoint trial halves

For every electrode we estimated a stability sensitivity *x* and a flexibility
sensitivity *y*. Because both are estimated from the same trials, shared trial
noise would inflate their correlation, so *x* and *y* were computed on **disjoint
halves of that electrode's trials**. Trials were split into two halves stratified
on the full crossing of the contrast factors (congruency, incongruent proportion,
task sequence, switch proportion), so neither half was confounded with a
condition, and both contrasts were computed on both halves.

Crucially, the *x*–*y* correlation was computed **within each split and then
averaged over the 200 splits**, rather than averaging the sensitivities across
splits and correlating once. Averaging first would reintroduce the shared-trial
noise the split removes, because the average is dominated by cross terms pairing
*x* from one split with *y* from another, whose trial sets overlap by
approximately half. For each split the two cross directions were averaged,
½[ρ(*x*<sub>A</sub>, *y*<sub>B</sub>) + ρ(*x*<sub>B</sub>, *y*<sub>A</sub>)], so
the two halves enter symmetrically.

Effects were scored with **equal cell weights**: an interaction as the
difference-of-differences of the four cell means, and a main effect as the
equal-weight mean of the within-cell differences. This makes the stability and
flexibility contrasts orthogonal in cell-mean space irrespective of cell counts,
so that neither contrast leaks into the other when the design factors are
correlated in the trial table. A naive same-trial estimate was computed as a
diagnostic only, to show the magnitude of the shared-noise inflation the disjoint
estimator removes; it is not used for inference.

### Gain control and subject nesting

The current implementation removes two nuisance sources before testing. (i)
**Shared gain/SNR**: an
electrode with a high signal-to-noise ratio shows larger effects for *both*
contrasts, which by itself produces a positive *x*–*y* correlation. Each
electrode's overall task responsiveness was therefore computed (the mean |HG| over
trials and time bins, or, where available, the electrode's baseline-versus-signal
cluster statistic) and *x* and *y* were each linearly residualised on it.
(ii) **Subject nesting**: residualised sensitivities were centred within subject,
so the estimate reflects within-subject co-selectivity and matches the
within-subject permutation null used for inference. Subjects contributing fewer
than three usable electrodes were excluded from the continuous test.

This is an implementation description, not an endorsement of responsiveness
residualisation as the primary specification. Responsiveness can be genuine
shared biological variance, the fallback proxy is estimated from the same data,
and the current regression uses one slope pooled across subjects. The preferred
report is therefore the unadjusted cross-fitted similarity as primary and an
adjusted analysis using an independently estimated responsiveness or precision
measure as sensitivity analysis. That preferred unadjusted path is **not yet an
option in `run_joint_distribution_analysis`**.

### Continuous test: is stability sensitivity related to flexibility sensitivity?

The association between the residualised, within-subject-centred *x* and *y* was
quantified with Spearman's ρ across all electrodes. Significance was assessed
with a permutation null in which *y* was shuffled **within each subject** (10,000
permutations), which preserves between-subject structure and therefore isolates
the within-subject association; the same permutation was applied across all
splits, so it breaks the *x*–*y* electrode correspondence while leaving each
split's internal structure intact. The two-tailed *p* value is the proportion of
permutations with |ρ| at least as large as observed. As a parametric cross-check,
a linear mixed model with a subject random intercept was fitted to the
responsiveness-residualised sensitivities. A significantly positive estimate,
when both patterns are reliable, supports shared alignment; a significantly
negative estimate supports opposing organization. A nonsignificant estimate of
either sign does not establish either conclusion. A near-zero estimate supports
distinct patterns only when reliability is adequate and an equivalence interval
excludes a prespecified meaningful positive association; that equivalence test
is not currently implemented.

**Split-half reliability and attenuation correction.** A correlation near zero
is only interpretable if both
effects are measured reliably in the first place, so from the same disjoint
halves we computed the split-half reliability of each sensitivity,
*r*<sub>stab</sub> = ρ(*x*<sub>A</sub>, *x*<sub>B</sub>) and *r*<sub>flex</sub> =
ρ(*y*<sub>A</sub>, *y*<sub>B</sub>), averaged over splits on the same
residualised and within-subject-centred values. These diagnose pattern
reproducibility, and ρ<sub>corrected</sub> = ρ ⁄ √(*r*<sub>stab</sub> ·
*r*<sub>flex</sub>) is reported alongside the raw estimate; the permutation *p*
applies unchanged to both, since they differ only by a fixed positive
denominator. The corrected estimate is secondary and is reported only when both
reliabilities are positive; it can nevertheless be unstable or exceed ±1 when
either reliability is small. It must not be clipped or treated as a guaranteed
noise ceiling. Reliable within-domain estimates (*r*<sub>stab</sub> = […],
*r*<sub>flex</sub> = […]) make a near-zero estimate interpretable, but positive
evidence for spatially distinct populations additionally requires an equivalence
interval excluding a prespecified meaningful shared association. Low reliability
means the correlation is uninformative and no independence claim is licensed.
Electrodes whose effect was
undefined on any split were excluded from the continuous test ([…] electrodes).

### Categorical test: 2×2 conjunction

Each electrode was independently labelled as stability-selective (S) and/or
flexibility-selective (F). Labels were obtained from a within-electrode
permutation test on the same supra-threshold *t* mass: the block-proportion
modulator was permuted **within each level of the condition factor** (2,000
permutations), which holds both main effects and all cell counts fixed and nulls
the interaction alone — unlike a free label shuffle, which under unequal cells
lets a main effect masquerade as an interaction. The resulting two-tailed
*p* values were corrected across electrodes with the Benjamini–Hochberg FDR
separately for each construct, and electrodes with *q* < 0.05 were flagged.
Electrodes with an undefined statistic were carried as non-significant rather than
dropped, so the FDR denominator remains honest.

This permutation is retained for exploratory reproduction only. Because the
modulator is constant within an experimental block, trial-wise reassignment
creates block configurations that could not occur and ignores within-block
dependence. Consequently the categorical *p* values, FDR flags, and downstream
CMH test are not confirmatory. They should be replaced by a null that permutes
the randomized trial-level factor within actual blocks, permutes intact block
identities only where the design makes them exchangeable, or uses a block-aware
regression/bootstrap.

As a parametric cross-check on the labels, the same 2×2 interaction was tested
per electrode with a Type III, sum-coded two-way ANOVA (FDR-corrected across
electrodes in the same way); note that this cross-check necessarily operates on
the window-mean HG, so agreement between it and the cluster-based labels
indicates that the primary result does not depend on the temporal statistic. The
two cross interactions (congruency × switch proportion and task sequence ×
incongruent proportion) were computed in the same framework as specificity
controls, and are expected to be near-null in univariate HG.

The S × F contingency was then evaluated with a **Cochran–Mantel–Haenszel** test
stratified by subject (one 2×2 table per patient) — the subject-aware analogue of
Fisher's exact test. The Mantel–Haenszel odds ratio is the key quantity:
OR < 1 indicates segregation (fewer joint-selective electrodes than expected),
OR > 1 a shared core, and OR ≈ 1 independence. Homogeneity of the odds ratio
across subjects was tested (Breslow–Day/Tarone), and the pooled table is reported
descriptively. Two additional checks accompany it: (i) an **empirical null for the
number of jointly selective electrodes**, obtained by shuffling the F labels
within each subject (10,000 permutations), which fixes each subject's S and F
marginals and randomises only the pairing; and (ii) a **threshold sweep**, in
which the selection cutoff is varied and the odds ratio and electrode counts are
recomputed, since LWPC and LWPS effects need not be equally strong and the
stronger one would otherwise recruit more electrodes at any fixed α.

### Reporting

We report, for the continuous test, ρ, its within-subject permutation *p*, and
the numbers of electrodes and subjects entering it; for the categorical test, the
per-class electrode counts (both / stability-only / flexibility-only / neither),
the MH odds ratio with its 95% CI, the CMH *p*, the homogeneity *p*, the
permutation *p* for the joint count, and the threshold sweep. Results are
reported at α = 0.05 throughout.

*If the main-effect contrasts are used instead* (`CONTRAST_MODE=condition`),
stability is the congruency contrast (incongruent − congruent) and flexibility
the task-sequence contrast (switch − repeat); each is then a two-group contrast,
the effect is the signed cluster mass of the per-bin two-sample *t*, the
disjoint-half split is stratified on congruency and task sequence only, and the
per-electrode null is a free permutation of the two-group label. All other steps
are unchanged.

---

## Version B — window-mean effect measure (`effect_measure='cohens_d'`)

### Single-trial high-gamma

Analyses were performed on stimulus-locked single-trial high-gamma (HG,
70–150 Hz) from [N] patients performing the Global/Local task. Broadband HG was
extracted from the cleaned, average-referenced recordings with a filterbank–Hilbert
decomposition and epoched from −1.0 to 1.5 s relative to stimulus onset. The
current preprocessing passes the signal epochs and a separately constructed
0.5-s pre-stimulus baseline epochs object to `ieeg.rescale(..., mode='zscore')`.
Whether that operation pairs each signal trial with its own aligned baseline has
not yet been verified, so it must not be described as “trial-by-trial” in a
manuscript without that verification. Epochs were decimated by a factor of 8; trials
exceeding 10 SD were treated as outliers and channels with more than 5% outlier
trials were dropped. Only correct trials were analysed, and trials whose task
sequence was undefined (first trial of a block) were excluded, leaving trials
labelled by congruency (congruent/incongruent), task sequence
(switch/repeat), and the two block-level proportions (incongruent proportion and
switch proportion). Electrode identifiers were scoped by subject, so channels
with the same name in different patients were never pooled.

For each electrode and trial, HG was **averaged over the analysis window**
([0.0, 0.5] s post-stimulus) to yield a single scalar per trial; trials with a
non-finite value were discarded. The resulting long-format table (one row per
electrode × trial) was the input to all analyses below.

### Constructs and contrasts

Stability and flexibility were each operationalised as a two-way interaction on
single-trial HG:

* **Stability (LWPC)** = congruency × incongruent proportion — the congruency
  effect as a function of the block's incongruent proportion.
* **Flexibility (LWPS)** = task sequence × switch proportion — the switch effect
  as a function of the block's switch proportion.

Each interaction was scored as a **balanced (equal-cell-weight)
difference-of-differences** over the four 2×2 cells,
`d-o-d = (M₁₁ − M₀₁) − (M₁₀ − M₀₀)`, rather than as a pooled contrast between
the two "+1" and the two "−1" cells. This matters because the proportion design
makes the four cells deliberately unequal in trial count (≈75/25): a
trial-count-weighted pooled contrast is dominated by the frequent cells, so a
pure congruency or switch **main** effect leaks into the estimate, whereas the
equal-cell difference-of-differences is orthogonal to both main effects.
Electrodes missing any of the four cells (or with fewer than two trials in a
cell) were assigned a missing value rather than zero, so they entered neither the
correlation nor the significance count.

Interactions were treated as **two-sided**: an electrode counts as
LWPC- (or LWPS-) selective whenever its condition effect is modulated by block
proportion, whether the effect grows or shrinks in high-proportion blocks. The
behavioural adjustment has a known direction, but no neural population is
required to mirror it, so no sign was imposed at any selection step. The signed
direction of every electrode's interaction was nevertheless recorded and is
reported descriptively; it is oriented **low-proportion minus high-proportion**,
so that a positive value denotes the behavioural adaptation direction (the
condition effect is smaller in the high-proportion block) for both LWPC and LWPS,
and for the behavioural difference-of-differences it is compared against.

### Effect measure: standardised difference-of-differences (Cohen's *d*)

Each electrode's interaction magnitude was quantified as the balanced
difference-of-differences of the four cell means, standardised by the pooled
within-cell standard deviation — a Cohen's-*d*-scaled interaction effect size.
Standardising makes effect sizes comparable across electrodes with different HG
variance, and the equal cell weighting keeps the estimate orthogonal to both main
effects (see above). Electrodes whose pooled within-cell SD was zero, or with a
cell containing fewer than two trials, were assigned a missing value.

Because this measure averages HG over the analysis window before contrasting
conditions, an interaction that is present only transiently within the window is
attenuated in proportion to its duration relative to the window length. The
window-mean measure is therefore reported as a simple, assumption-light
quantification of interaction magnitude, and the same analysis was repeated with
a time-resolved measure (`effect_measure='cluster'`; per-bin
difference-of-differences *t*, thresholded at α = 0.05 and summed with sign over
all supra-threshold bins) as a sensitivity analysis for transient effects.

### Per-electrode sensitivities on disjoint trial halves

For every electrode we estimated a stability sensitivity *x* and a flexibility
sensitivity *y*. Because both are estimated from the same trials, shared trial
noise would inflate their correlation, so *x* and *y* were computed on **disjoint
halves of that electrode's trials**. Trials were split into two halves stratified
on the full crossing of the contrast factors (congruency, incongruent proportion,
task sequence, switch proportion), so neither half was confounded with a
condition, and both contrasts were computed on both halves.

Crucially, the *x*–*y* correlation was computed **within each split and then
averaged over the 200 splits**, rather than averaging the sensitivities across
splits and correlating once. Averaging first would reintroduce the shared-trial
noise the split removes, because the average is dominated by cross terms pairing
*x* from one split with *y* from another, whose trial sets overlap by
approximately half. For each split the two cross directions were averaged,
½[ρ(*x*<sub>A</sub>, *y*<sub>B</sub>) + ρ(*x*<sub>B</sub>, *y*<sub>A</sub>)], so
the two halves enter symmetrically.

Effects were scored with **equal cell weights**: an interaction as the
difference-of-differences of the four cell means, and a main effect as the
equal-weight mean of the within-cell differences. This makes the stability and
flexibility contrasts orthogonal in cell-mean space irrespective of cell counts,
so that neither contrast leaks into the other when the design factors are
correlated in the trial table. A naive same-trial estimate was computed as a
diagnostic only, to show the magnitude of the shared-noise inflation the disjoint
estimator removes; it is not used for inference.

### Gain control and subject nesting

The current implementation removes two nuisance sources before testing. (i)
**Shared gain/SNR**: an
electrode with a high signal-to-noise ratio shows larger effects for *both*
contrasts, which by itself produces a positive *x*–*y* correlation. Each
electrode's overall task responsiveness was therefore computed (the mean
absolute HG over trials, or, where available, the electrode's baseline-versus-signal
cluster statistic) and *x* and *y* were each linearly residualised on it.
(ii) **Subject nesting**: residualised sensitivities were centred within subject,
so the estimate reflects within-subject co-selectivity and matches the
within-subject permutation null used for inference. Subjects contributing fewer
than three usable electrodes were excluded from the continuous test.

This is an implementation description, not an endorsement of responsiveness
residualisation as the primary specification. Responsiveness can be genuine
shared biological variance, the fallback proxy is estimated from the same data,
and the current regression uses one slope pooled across subjects. The preferred
report is therefore the unadjusted cross-fitted similarity as primary and an
adjusted analysis using an independently estimated responsiveness or precision
measure as sensitivity analysis. That preferred unadjusted path is **not yet an
option in `run_joint_distribution_analysis`**.

### Continuous test: is stability sensitivity related to flexibility sensitivity?

The association between the residualised, within-subject-centred *x* and *y* was
quantified with Spearman's ρ across all electrodes. Significance was assessed
with a permutation null in which *y* was shuffled **within each subject** (10,000
permutations), which preserves between-subject structure and therefore isolates
the within-subject association; the same permutation was applied across all
splits, so it breaks the *x*–*y* electrode correspondence while leaving each
split's internal structure intact. The two-tailed *p* value is the proportion of
permutations with |ρ| at least as large as observed. As a parametric cross-check,
a linear mixed model with a subject random intercept was fitted to the
responsiveness-residualised sensitivities. A significantly positive estimate,
when both patterns are reliable, supports shared alignment; a significantly
negative estimate supports opposing organization. A nonsignificant estimate of
either sign does not establish either conclusion. A near-zero estimate supports
distinct patterns only when reliability is adequate and an equivalence interval
excludes a prespecified meaningful positive association; that equivalence test
is not currently implemented.

**Split-half reliability and attenuation correction.** A correlation near zero
is only interpretable if both
effects are measured reliably in the first place, so from the same disjoint
halves we computed the split-half reliability of each sensitivity,
*r*<sub>stab</sub> = ρ(*x*<sub>A</sub>, *x*<sub>B</sub>) and *r*<sub>flex</sub> =
ρ(*y*<sub>A</sub>, *y*<sub>B</sub>), averaged over splits on the same
residualised and within-subject-centred values. These diagnose pattern
reproducibility, and ρ<sub>corrected</sub> = ρ ⁄ √(*r*<sub>stab</sub> ·
*r*<sub>flex</sub>) is reported alongside the raw estimate; the permutation *p*
applies unchanged to both, since they differ only by a fixed positive
denominator. The corrected estimate is secondary and is reported only when both
reliabilities are positive; it can nevertheless be unstable or exceed ±1 when
either reliability is small. It must not be clipped or treated as a guaranteed
noise ceiling. Reliable within-domain estimates (*r*<sub>stab</sub> = […],
*r*<sub>flex</sub> = […]) make a near-zero estimate interpretable, but positive
evidence for spatially distinct populations additionally requires an equivalence
interval excluding a prespecified meaningful shared association. Low reliability
means the correlation is uninformative and no independence claim is licensed.
Electrodes whose effect was
undefined on any split were excluded from the continuous test ([…] electrodes).

### Categorical test: 2×2 conjunction

Each electrode was independently labelled as stability-selective (S) and/or
flexibility-selective (F). Labels were obtained from a within-electrode
permutation test on the same standardised difference-of-differences: the
block-proportion modulator was permuted **within each level of the condition
factor** (2,000 permutations), which holds both main effects and all cell counts
fixed and nulls the interaction alone — unlike a free label shuffle, which under
unequal cells lets a main effect masquerade as an interaction. The resulting
two-tailed *p* values were corrected across electrodes with the
Benjamini–Hochberg FDR separately for each construct, and electrodes with
*q* < 0.05 were flagged. Electrodes with an undefined statistic were carried as
non-significant rather than dropped, so the FDR denominator remains honest.

This permutation is retained for exploratory reproduction only. Because the
modulator is constant within an experimental block, trial-wise reassignment
creates block configurations that could not occur and ignores within-block
dependence. Consequently the categorical *p* values, FDR flags, and downstream
CMH test are not confirmatory. They should be replaced by a null that permutes
the randomized trial-level factor within actual blocks, permutes intact block
identities only where the design makes them exchangeable, or uses a block-aware
regression/bootstrap.

As a parametric cross-check on the labels, the same 2×2 interaction was tested
per electrode on the window-mean HG with a Type III, sum-coded two-way ANOVA
(FDR-corrected across electrodes in the same way); sum coding with Type III sums
of squares makes the interaction term orthogonal to the main effects by
construction, matching the equal-cell difference-of-differences used by the
permutation route. The two cross interactions (congruency × switch proportion and
task sequence × incongruent proportion) were computed in the same framework as
specificity controls, and are expected to be near-null in univariate HG.

The S × F contingency was then evaluated with a **Cochran–Mantel–Haenszel** test
stratified by subject (one 2×2 table per patient) — the subject-aware analogue of
Fisher's exact test. The Mantel–Haenszel odds ratio is the key quantity:
OR < 1 indicates segregation (fewer joint-selective electrodes than expected),
OR > 1 a shared core, and OR ≈ 1 independence. Homogeneity of the odds ratio
across subjects was tested (Breslow–Day/Tarone), and the pooled table is reported
descriptively. Two additional checks accompany it: (i) an **empirical null for the
number of jointly selective electrodes**, obtained by shuffling the F labels
within each subject (10,000 permutations), which fixes each subject's S and F
marginals and randomises only the pairing; and (ii) a **threshold sweep**, in
which the selection cutoff is varied and the odds ratio and electrode counts are
recomputed, since LWPC and LWPS effects need not be equally strong and the
stronger one would otherwise recruit more electrodes at any fixed α.

### Reporting

We report, for the continuous test, ρ, its within-subject permutation *p*, and
the numbers of electrodes and subjects entering it; for the categorical test, the
per-class electrode counts (both / stability-only / flexibility-only / neither),
the MH odds ratio with its 95% CI, the CMH *p*, the homogeneity *p*, the
permutation *p* for the joint count, and the threshold sweep. Results are
reported at α = 0.05 throughout.

*If the main-effect contrasts are used instead* (`CONTRAST_MODE=condition`),
stability is the congruency contrast (incongruent − congruent) and flexibility
the task-sequence contrast (switch − repeat); each is then a two-group contrast,
the effect is the ordinary two-sample Cohen's *d* on window-mean HG, the
disjoint-half split is stratified on congruency and task sequence only, and the
per-electrode null is a free permutation of the two-group label. All other steps
are unchanged.

---

## Parameter appendix (both versions)

| Parameter | Value | Where set |
|---|---|---|
| Analysis window | [0.0, 0.5] s post-stimulus | `WINDOW_TMIN` / `WINDOW_TMAX` |
| HG passband | 70–150 Hz, filterbank–Hilbert | epochs file (`EPOCHS_ROOT_FILE`) |
| Baseline | 0.5-s window drawn from [−1.0, 0.0] s, z-scored | epochs file |
| Trials | correct only; undefined task sequence dropped | `ACC_TRIALS_ONLY`, `assemble_long_df` |
| Contrasts | `proportion` (LWPC / LWPS interactions) | `CONTRAST_MODE` |
| Effect measure | `cluster` (A) / `cohens_d` (B) | `EFFECT_MEASURE` |
| Per-bin statistic-forming threshold (measure A) | two-tailed *t* at α = 0.05; no contiguity requirement, no cluster-level correction | `alpha` (`_interaction_cluster`) |
| Disjoint-half resamples | 200 | `N_SPLITS` |
| Continuous-test permutations | 10,000 (within-subject) | `N_PERM_CORR` |
| Per-electrode label permutations | 2,000 (modulator within condition level) | `N_PERM_LABEL` |
| Conjunction-null permutations | 10,000 (F within subject) | `conjunction_permutation_null` |
| Multiple comparisons | Benjamini–Hochberg FDR across electrodes, per construct | `per_electrode_labels` |
| α | 0.05 | `ALPHA` |
| Min. electrodes per subject (continuous test) | 3 | `MIN_ELEC` |
| Correlation | Spearman ρ | `subject_clustered_corr` |

Software: Python, NumPy, SciPy, pandas, statsmodels
(`StratifiedTable` for the CMH test, `multipletests` for FDR, `smf.ols` /
`anova_lm` for the Type III ANOVA cross-check), and `ieeg` for HG extraction and
the optional permutation cluster mask.
