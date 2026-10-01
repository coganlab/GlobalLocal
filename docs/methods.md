# Methods text

Manuscript-ready Methods for the segregation, anatomy and cross-decoding analyses. Bracketed
quantities are run-dependent: fill them in from the archived run.

| Part | What it covers | Was |
|---|---|---|
| [N4: segregation and continuous anatomy](#n4-segregation-and-continuous-anatomy) | The combined N4 Methods, written for the primary configuration of the continuous anatomy pipeline: per-electrode LWPC/LWPS scores on disjoint halves, the coverage-conditioned anatomical test, the coordinate follow-ups, the maps, and the descriptive medoids | `n4_anatomy_segregation_methods.md` |
| [Segregation: cluster and Cohen's d versions](#segregation-cluster-and-cohens-d-versions) | Two interchangeable write-ups of the segregation analysis on its own, one per effect measure (`cluster`, `cohens_d`), with the implementation-status notes and a parameter appendix | `stability_flexibility_segregation_methods.md` |
| [A6: brain–behavior (supplement S-BB)](#a6-brainbehavior-supplement-s-bb) | The across-participant correlation of neural with behavioral LWPC/LWPS: participant scores, the RT adjustment, the reliability ceiling and the power it implies. Filled in from the 2026-09-30 run | new (2026-10-01) |
| [A4: congruency ↔ switch-type cross-decoding](#a4-congruency--switch-type-cross-decoding) | Methods, a draft results paragraph and the limitations for supplement S5: the unselected lPFC transfer, its RT and overall-activity controls, the occipital comparison | new |

The N4 text is the newer one (2026-09-17) and matches the current primary
configuration. The segregation versions date from 2026-08 (last edited
2026-09-17). The A6 text (2026-10-01) is for the supplement only; the Results
paragraph, figure caption and placement verdict that go with it are in §14 of
[`a6_brain_behavior.md`](a6_brain_behavior.md). The A4 text dates from 2026-10-01.

---

## N4: segregation and continuous anatomy

*Methods — N4 segregation and continuous anatomy*

This section provides manuscript-ready Methods text for the N4 analysis. It is
written for the primary configuration used by the continuous anatomy pipeline:
an anatomically defined electrode set, `CONTRAST_MODE=proportion`, window-mean
high-gamma effect sizes (`EFFECT_MEASURE=cohens_d`), 200 disjoint-half
resamples, and 10,000 inferential permutations. Bracketed quantities should be
replaced with values from the archived run. The categorical conjunction is
included as an explicitly exploratory companion analysis because its current
trial-wise interaction null does not preserve the block structure of the
proportion manipulation.

For the all-lPFC main-effect run (1,000 splits, with congruency and switch as
the reference), the filled-in Methods are §17.2 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#172-methods).

### Participants, recordings, and electrode population

Intracranial EEG was recorded from **[N participants]** while they performed the
Global/Local task. Broadband high-gamma activity (70–150 Hz) was extracted from
cleaned, average-referenced recordings using a filterbank–Hilbert transform and
epoched from −1.0 to 1.5 s relative to stimulus onset. The preprocessing
pipeline supplied signal epochs and a separately constructed 0.5-s
prestimulus baseline object to a z-score rescaling operation. Because alignment
of individual signal trials to individual baseline segments has not been
verified in the current implementation, this normalization is not described as
trial-by-trial baseline correction. Epochs were decimated by a factor of eight.
Trials exceeding 10 SD were marked as outliers, and channels for which more
than 5% of trials were outliers were excluded. Analyses included correct trials
only and omitted the first trial of each block, for which task sequence was
undefined.

The primary population comprised all recording-quality electrodes within the
predeclared anatomical scope (**[whole brain / specified ROI]**), rather than
electrodes selected for a significant LWPC or LWPS effect. Electrode names were
combined with participant identifiers before any join or aggregation so that
identically named contacts from different participants remained distinct. The
final score analysis included **[E electrodes from N participants]**. We report
participant and electrode counts together because electrodes within a
participant are not independent biological replicates.

### Per-electrode LWPC and LWPS scores

For each electrode and trial, high-gamma activity was averaged over the
predeclared **[tmin–tmax s]** poststimulus window. Stability was operationalized
as the list-wide proportion congruency (LWPC) interaction between congruency
(incongruent versus congruent) and the block's incongruent-trial proportion.
Flexibility was operationalized as the list-wide proportion switch (LWPS)
interaction between task sequence (switch versus repeat) and the block's
switch-trial proportion.

Each interaction was estimated using the equal-cell-weighted
difference-of-differences

```text
(condition effect in the low-proportion context)
  − (condition effect in the high-proportion context).
```

The contrast therefore compared the incongruent-minus-congruent effect across
incongruent-proportion contexts for LWPC and the switch-minus-repeat effect
across switch-proportion contexts for LWPS. Positive values indicate the
predicted adaptation direction: a smaller condition effect in the
high-proportion context. Equal weighting of the four 2 × 2 cell means was used
because the proportion manipulation deliberately produces unequal cell counts;
a trial-count-weighted contrast would allow the condition main effect to leak
into the interaction estimate. Each difference-of-differences was divided by
the pooled within-cell standard deviation to obtain a signed, Cohen's-*d*-scaled
effect size. Scores were undefined when any cell contained fewer than two
trials or when the pooled within-cell standard deviation was zero.

### Disjoint-half estimation and map reliability

LWPC and LWPS were estimated from the same trial pool, so calculating both on
identical observations could induce a positive association through shared
trial noise. We therefore generated 200 random splits within each electrode,
stratified on the full crossing of congruency, incongruent proportion, task
sequence, and switch proportion. Both effects were estimated separately in
both halves, yielding `LWPC_A`, `LWPC_B`, `LWPS_A`, and `LWPS_B` for every
split. An electrode's score for anatomical mapping was the mean of its two
half-estimates and then the mean across splits.

For the segregation analysis, cross-effect similarity was calculated within
each split before averaging:

```text
0.5 × [rho(LWPC_A, LWPS_B) + rho(LWPC_B, LWPS_A)].
```

This order is essential: averaging scores over splits before correlating them
would reintroduce overlapping-trial terms from different splits. Before each
correlation, both scores were linearly residualized on electrode
responsiveness and centred within participant. Responsiveness was quantified as
mean absolute high-gamma activity across trials **[or describe the independent
responsiveness measure if one was supplied]**. Participants with fewer than
three usable electrodes were excluded from this correlation. Spearman's rho
was averaged across splits, and a two-sided null distribution was generated by
shuffling the LWPS electrode correspondence within participant while applying
the same shuffle to every split (10,000 permutations). Thus, inference tested
the within-participant electrode-level association while retaining
between-participant structure.

Split-half reproducibility was estimated in parallel as the mean across splits
of `rho(LWPC_A, LWPC_B)` and `rho(LWPS_A, LWPS_B)`, after the same nuisance
adjustments. We also report the attenuation-corrected cross-effect correlation,
`rho / sqrt(reliability_LWPC × reliability_LWPS)`, when both reliabilities were
positive. This corrected value was treated as a diagnostic, not a bounded
estimate or formal noise ceiling; it can be unstable or exceed ±1 when either
reliability is small. A near-zero cross-effect association was interpreted as
evidence for distinct spatial patterns only when both maps were sufficiently
reliable. Because the current implementation does not perform an equivalence
test against a prespecified smallest shared association, a nonsignificant
correlation alone was not interpreted as proof of segregation.

The pipeline currently requires responsiveness adjustment and uses one pooled
slope across participants. Because responsiveness may contain genuine shared
biological variance and the default proxy is estimated from the same data, we
treat this result as a responsiveness-adjusted electrode-pattern analysis and
do not imply that residualization is assumption-free. The resampling splits
also operate on trials rather than intact blocks; consequently, the analysis
does not model slow within-block dependence and is interpreted with that
qualification.

### Exploratory categorical conjunction

As a secondary visualization and threshold-dependent check, electrodes were
classified independently as LWPC-selective and/or LWPS-selective. For each
effect, a two-sided per-electrode permutation test was applied to the signed
standardized difference-of-differences (2,000 permutations), and the resulting
*p* values were controlled across electrodes with the Benjamini–Hochberg
procedure separately for LWPC and LWPS. The four resulting classes were
LWPC-only, LWPS-only, both, and neither. Their association was summarized with
a Cochran–Mantel–Haenszel test stratified by participant and a Mantel–Haenszel
odds ratio; an odds ratio below one denotes less overlap than expected from the
within-participant marginals, whereas an odds ratio above one denotes greater
overlap. We additionally permuted LWPS labels within participant 10,000 times to
form an empirical null for the number of jointly selective electrodes and
repeated the contingency analysis over a range of selection thresholds.

This categorical analysis was considered exploratory. The implemented
per-electrode null permutes the block-proportion label trial by trial within
condition level. Because proportion is constant within a block, those
reassignments do not preserve the experimental exchangeability unit and can
generate configurations that could not occur in the design. The categorical
*p* values, FDR labels, and downstream conjunction test therefore did not
determine the N4 conclusion.

### Anatomical localization and score scaling

Per-electrode scores were joined to each participant's atlas lookup using the
participant-scoped electrode identifier. We retained both a coarse ROI grouping
and the native Destrieux parcel label. The primary whole-brain analysis used
the predeclared **[coarse ROI groups / Destrieux parcels]**; **[describe any
regional Destrieux analysis as a prespecified follow-up]**. Anatomical units
represented in fewer than three participants were excluded before inference.
A participant-by-anatomical-unit coverage matrix was retained and reported with
the results. This restriction prevents units represented by only one or two
participants from defining the tested family, but it does not render clinically
determined electrode coverage random.

To put the two effects on comparable pooled scales without distorting small
within-participant samples, each score column was divided by its standard
deviation across all eligible electrodes:

```text
LWPC_s = LWPC / SD(LWPC)
LWPS_s = LWPS / SD(LWPS)
delta  = LWPC_s − LWPS_s.
```

Scores were not z-scored within participant and pooled means were not
subtracted. The primary anatomical outcome, `delta`, is positive where an
electrode is relatively more LWPC-dominant and negative where it is relatively
more LWPS-dominant. A value near zero indicates similar scaled scores and does
not imply that both effects are absent.

### Primary coverage-conditioned anatomical test

The primary N4 question was whether the within-electrode balance of LWPC and
LWPS varied among coverage-eligible anatomical units. We modelled

```text
delta_e = anatomical_unit_e + responsiveness_e + participant_e + error_e,
```

where participant was represented by fixed-effect indicator variables. The
omnibus statistic was the partial *F* for the anatomical-unit block after
accounting for responsiveness and participant. Its significance was evaluated
nonparametrically: for each of 10,000 permutations, the LWPC and LWPS labels
were independently exchanged within every electrode, which is equivalent to a
random sign flip of `delta`. This null retains each electrode's participant,
anatomy, responsiveness, coverage, and pair of observed scores while removing
their effect-type assignment. Monte Carlo *p* values were calculated as
`(number at least as extreme + 1) / (number of permutations + 1)`.

The omnibus test was followed by anatomical-unit summaries of the raw and
nuisance-adjusted mean `delta`. Two-sided label-swap *p* values for individual
units were corrected over the retained anatomical family using
Benjamini–Hochberg FDR. Unit-level results were used to localize a supported
omnibus effect rather than as substitutes for that test. We repeated the
omnibus analysis after excluding each participant in turn, using at least 1,000
permutations per fold, and evaluated robustness from changes in the observed
*F* statistic as well as the permutation *p* value.

### Secondary coordinate analysis

Where electrode reconstructions were available, coordinates were transformed
to fsaverage/MNI millimetres. We fitted a secondary spatial-gradient model to
all electrodes and separately within each hemisphere:

```text
delta_e = MNI-y_e + MNI-z_e + MNI-x_e
          + responsiveness_e + participant_e + error_e.
```

The coordinate-block *F* and its within-electrode label-swap permutation *p*
tested whether location predicted relative effect dominance. Axis-specific
slopes and permutation *p* values described direction and were treated as
follow-ups to the block test. A positive anterior–posterior (`MNI-y`) slope
indicates increasing LWPC dominance anteriorly, and a positive
inferior–superior (`MNI-z`) slope indicates increasing LWPC dominance
superiorly. Left–right slopes were interpreted primarily within hemisphere to
avoid cancellation across mirrored bilateral coordinates.

### Visualization and descriptive centres

Five electrode maps displayed signed `LWPC_s`, signed `LWPS_s`, their absolute
magnitudes, and `delta`. Signed maps used zero-centred diverging scales;
magnitude maps used sequential scales. Display limits were clipped at the 98th
percentile to prevent isolated extremes from compressing the colour range, and
each map was accompanied by its own colour bar. These pooled maps were treated
as visualizations rather than independent inferential tests because dense
implants contribute more displayed electrodes. A flat anatomical summary showed
mean `delta` ± SEM and individual electrodes together with electrode and
participant coverage counts.

For descriptive localization only, we calculated one LWPC and one LWPS centre
for every participant × hemisphere group containing at least three electrodes.
Both centres used the same electrodes, absolute scaled effect sizes as
non-negative weights, and a weighted medoid: the observed electrode minimizing
the weighted distance to all other electrodes in the group. We summarized the
within-group LWPC-minus-LWPS displacement in millimetres. These medoids were not
used to establish anatomical separation; directional claims were based on the
coverage-conditioned anatomy and coordinate models.

### Statistical reporting and interpretation

All tests were two-sided with alpha = .05. For segregation, we report the raw
cross-fitted Spearman correlation, within-participant permutation *p*, both
split-half reliabilities, the attenuation-corrected diagnostic, and the numbers
of electrodes and participants. For anatomy, we report the omnibus *F* and
label-swap permutation *p*, the retained units and their participant coverage,
adjusted unit means with FDR-corrected *q* values, and leave-one-participant-out
estimates. Coordinate-block and axis results are identified as secondary, and
medoids as descriptive. The primary anatomical claim is phrased as variation
in the **relative LWPC/LWPS balance** across cortex; it is not inferred from an
LWPC test being significant in one location while an LWPS test is not.

The archived analysis record comprised the git commit, exact submission
command, input score and split-resolved tables, time window, electrode and ROI
scope, anatomical level, coverage threshold, permutation counts, random seed,
coverage matrix, result tables, figures, and warnings concerning missing atlas
labels, coordinates, or surface rendering.

---

## A6: brain–behavior (supplement S-BB)

*Methods: across-participant brain–behavior correlation*

Manuscript-ready text for the supplement. It describes the primary configuration
(task-significant lPFC, 0–1.5 s, `MIN_ELEC=3`, 200 shared splits, `SEED=0`) and
is filled in from the 2026-09-30 run
(`brain_behavior_window_0.0to1.5s_sig_lpfc_count/`). Bracketed text has to come
from elsewhere in the paper. Levels (2) and (3) of the job, the label-based and
single-trial versions, are left out on purpose: [`a6_brain_behavior.md`](a6_brain_behavior.md)
§13.3 explains why neither measures what it is named for. The two exploratory
summaries are in an optional last paragraph.

### Participants and electrodes

We asked whether participants whose lateral prefrontal high-gamma activity
adapted more to the proportion manipulations also adapted more in their
behavior. The analysis used the task-significant electrodes in lateral
prefrontal cortex (171 electrodes from 21 participants; task significance as
defined in **[main Methods section]**). These electrodes were selected for
overall task responsiveness, not for an LWPC or LWPS effect. The high-gamma
epochs and preprocessing were those of the anatomical analyses. Only correct
trials with a recorded response time (RT) were used. A participant entered the
analysis if it had at least three usable electrodes (18 participants; median 7
electrodes and 397 trials each) and a behavioral score; one of the 18 was absent
from the behavioral summary table **[reason]**, leaving 17 participants.

### Neural and behavioral scores

For each electrode, single-trial high gamma was averaged over 0–1.5 s after
stimulus onset. LWPC and LWPS were scored as in the anatomical analyses: the
equal-cell-weighted difference of differences (the condition effect in the
low-proportion blocks minus that in the high-proportion blocks), divided by the
pooled within-cell standard deviation. Here each score was computed once from
all of the electrode's trials rather than on split halves. An electrode was
usable if all four of its scores (LWPC and LWPS, unadjusted and RT-adjusted,
below) were defined. A participant's neural LWPC and LWPS were the unweighted
means of its usable electrodes' scores. Equal weights are appropriate because a
participant's electrodes share its trials and so have similar sampling error.

The behavioral LWPC and LWPS were the same differences of differences computed
on mean RT, in milliseconds, from the behavioral analysis **[section reference;
trial inclusion as described there]**. Both scores were oriented so that positive
values indicate the predicted adaptation: a smaller congruency effect or switch
cost in the high-proportion blocks.

### Removing the RT-linked component of high gamma

The analysis window covers most responses (median RT 1.19 s), so single-trial
high gamma may track RT. If it does, every cell mean of high gamma carries the
same multiple of that cell's mean RT. Each electrode's neural
difference-of-differences then contains its slope on RT times the participant's
own behavioral difference of differences. That term would correlate neural with
behavioral adaptation across participants with no link between them beyond the
trial-by-trial coupling, and it would also pass the specificity checks below.

We therefore removed the RT-linked component of high gamma separately for each
electrode. The slope of high gamma on RT was estimated from deviations around
each of the 16 cell means of the design (congruency × task sequence ×
incongruent proportion × switch proportion), pooled across cells, so that
condition effects, which move both high gamma and RT, did not enter it. Each
trial's high gamma was replaced by its value minus the slope times the trial's
deviation from the electrode's mean RT, and the scores were recomputed. This
removes exactly the slope times the behavioral difference of differences from
each electrode's score. The adjustment is conservative: if neural adaptation
reaches behavior through the same trial-by-trial coupling, that part is removed
too. We therefore treat the RT-adjusted correlation as the primary test and
report the unadjusted correlation as an upper bound.

### Statistical analysis

For each effect, the neural and behavioral scores were related across
participants by Pearson correlation (two-sided, α = .05; |r| ≥ 0.48 needed at
n = 17), with a 95% confidence interval from Fisher's *z* and Spearman's ρ as a
rank-based check. Behavioral LWPC and LWPS are correlated across participants,
so a neural score could relate to both. To test specificity, each behavioral
score was regressed (ordinary least squares, all variables *z*-scored) on both
neural scores together, and the coefficient of the matched neural score was
compared with that of the other. The two RT-adjusted matched correlations were
the primary tests. **[If the cross pairings are reported, state that p-values
are uncorrected across the eight correlations: matched and cross, adjusted and
unadjusted.]**

### Reliability and the correlation ceiling

An observed correlation cannot exceed the square root of the product of the two
scores' reliabilities. We estimated each score's reliability by splitting every
participant's trials into random halves, stratified on the 16 design cells, 200
times. Each split was drawn once per participant and applied to all of its
electrodes. Drawing it separately per electrode would let noise common to a
participant's electrodes masquerade as reliability. All scores were recomputed
in each half. The half-length reliability was the across-participant correlation
of half-A with half-B values, averaged over splits, and was stepped up to full
length with the Spearman–Brown formula. It was treated as unmeasurable when the
half-length correlation was zero or negative. The behavioral summary table has
no trial-level data, so its reliability was estimated from the same contrasts
computed on the RTs of the recorded trials. These agreed closely with the table
across participants (r = 0.90 for LWPC and 0.76 for LWPS, 20 participants).

The ceiling on each brain–behavior correlation was the square root of the
product of the neural and behavioral reliabilities. To express what the ceiling
means for detection, we simulated 40,000 samples of 17 participants from a
bivariate normal distribution whose correlation was the ceiling times an assumed
true correlation, and counted the samples reaching |r| ≥ 0.48. The sample size
needed for 80% power was obtained from Fisher's *z*. Both treat the estimated
reliabilities as known, so they are approximate.

### Exploratory participant summaries (optional)

The signed mean lets electrodes whose adaptation runs in opposite directions
cancel. As an exploratory check we also summarized each participant by the mean
absolute score over its electrodes and by the mean over only its electrodes with
a positive score (chosen separately for LWPC and LWPS; at least three such
electrodes required). Both summaries fold or select on the same noisy score they
average, so noise alone raises them, and more so in participants with fewer
trials. We therefore correlated each with the participant's trial count as a
check on whether noise drove it.

---

## Segregation: cluster and Cohen's d versions

*Methods — stability/flexibility segregation analysis*

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
> `analysis_plans.md` › Simplification plan §2.2–§2.2b.

### Review status of the revised implementation

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

#### Confirmatory priority

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

### Version A — time-resolved ("cluster") effect measure (`effect_measure='cluster'`)

#### Single-trial high-gamma

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

#### Constructs and contrasts

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

#### Effect measure: signed supra-threshold *t* mass over time

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

#### Per-electrode sensitivities on disjoint trial halves

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

#### Gain control and subject nesting

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

#### Continuous test: is stability sensitivity related to flexibility sensitivity?

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

#### Categorical test: 2×2 conjunction

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

#### Reporting

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

### Version B — window-mean effect measure (`effect_measure='cohens_d'`)

#### Single-trial high-gamma

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

#### Constructs and contrasts

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

#### Effect measure: standardised difference-of-differences (Cohen's *d*)

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

#### Per-electrode sensitivities on disjoint trial halves

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

#### Gain control and subject nesting

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

#### Continuous test: is stability sensitivity related to flexibility sensitivity?

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

#### Categorical test: 2×2 conjunction

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

#### Reporting

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

### Parameter appendix (both versions)

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

---

## A4: congruency ↔ switch-type cross-decoding

*Methods and results text — supplementary S5*

Written 2026-10-01 from the runs in
[`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01), for supplement
S5 with the control table S8
([`analysis_plans.md` › Closing figure plan](analysis_plans.md#closing-figure-plan)).
Bracketed quantities come from the slurm log or from runs not yet done. The
pseudo-trial count is the `Decoded pseudo-trials` block of `summary.txt` (or, for
runs made before that block existed, the `subsampling to N trials` log lines). The
results paragraph states only what the full-trial runs support. Revise it once
the seeds, the electrode-matched region comparison and the positive controls are
in (§13.8.4 of that section).

### Methods

**Data and electrodes.** We used the stimulus-locked high-gamma epochs described
above (70–150 Hz, −1.0 to 1.5 s, decimated to 256 Hz; correct trials only; the
first trial of each block omitted). We decoded all lateral prefrontal electrodes
whose high gamma exceeded their pre-stimulus baseline (171 electrodes from 21
participants). Electrodes were not selected for a congruency or switch-type
effect. As a regional comparison we repeated every analysis on the task-responsive
occipital electrodes (54 electrodes from **[N]** participants).

**Conditions and pseudopopulation.** Trials were sorted into the four congruency
× switch-type cells, each pooled over the four block types. For each cell, each
electrode's outlier trials were removed and the electrode was randomly subsampled
to the smallest number of clean trials of any electrode in that cell
(**[n]** pseudo-trials per cell). Electrodes from all participants were then
concatenated into one pseudopopulation. Because electrodes were sampled
independently, a pseudo-trial combines different trials of the same condition
across electrodes, including electrodes from the same participant.

**Decoding.** Each 250-ms window (64 samples, stepped by 62.5 ms; 37 windows) was
decoded separately. The features were every electrode's samples in that window.
The classifier was principal component analysis (components explaining 80% of the
training variance, refit in each fold) followed by linear discriminant analysis
with equal class priors. We used stratified five-fold cross-validation, repeated
ten times. Folds were stratified on the four cells, so every test fold was
balanced on both labellings. Accuracy was the mean of the two classes' hit rates.

**Cross-decoding.** On each fold we trained one classifier on congruency
(incongruent vs congruent) and one on switch type (switch vs repeat). Each was
scored on held-out trials against both labellings. Scoring against the training
labelling gives the within-contrast accuracy (the *ceiling*). Scoring against the
other labelling gives the *transfer*. Incongruent was paired with switch and
congruent with repeat, so a shared axis on which the harder condition of each
contrast falls on the same side yields above-chance transfer. Each transfer was
expressed as the share of its ceiling's above-chance accuracy that it retained,
(transfer − 0.5) / (ceiling − 0.5), averaged over the windows in which the ceiling
was significant.

**Statistics.** For each decode, the null distribution came from permuting the
training labels within each fold and refitting. True-label accuracies (ten CV
repeats) were compared with the null by a one-tailed cluster-based permutation
test over windows (500 permutations, α = 0.05). The same test compared each
transfer with its ceiling. Windows centred at or before −0.125 s, which contain no
post-stimulus sample, served as a check on artifacts. Because the samples entering
these tests are CV repeats of a single pseudopopulation rather than participants,
we treat the window-wise results as a within-dataset reliability measure, not as
population inference.

**Controls.**

- *Response time.* Incongruent and switch trials were slower. In each
  participant, RTs were pooled across the four cells and cut into ten quantile
  bins. Within each bin we kept, at random, the same number of trials from each
  cell. This left 48% of trials and removed the RT costs (incongruent − congruent:
  +160 → +2 ms, p = 0.45; switch − repeat: +195 → +2 ms, p = 0.39; across 24
  participants). A control drew the same number of trials per participant and
  cell without regard to RT, keeping the RT costs (+156 and +208 ms). The
  RT-matched result is compared with this control, not with the full-trial result.
- *Overall activity.* To ask whether the transfer reflected a uniform change in
  activity, we subtracted each participant's mean across its electrodes, per
  pseudo-trial and time point, before decoding. Participants contributing a single
  electrode carry no information after this step (2 of 21).

**[Positive controls, the response-locked analysis and the seed repeats go here
once run.]**

### Results (draft)

Congruency and switch type were each decodable from the task-responsive lPFC
population. Congruency was decodable from the window centred at +0.12 s, with
peak accuracy 0.76. Switch type was decodable from +0.31 s, with peak 0.76. A
congruency decoder also predicted switch type, and a switch-type decoder predicted
congruency, but only from the window centred at +0.62 s (covering 0.50–0.75 s)
onward, and well below the within-contrast accuracy. The congruency decoder
retained 47% of switch type's above-chance accuracy, and was below it in 19 of 37
windows. The switch-type decoder retained 26% of congruency's, and was below it in
28 windows. No window before the stimulus was significant in any of the four
decodes. Removing each participant's mean activity lowered both within-contrast
accuracies (peaks 0.63 and 0.65) but left the transfers nearly unchanged (15 and
14 windows; 81% and 77% retained). The shared component is therefore not a uniform
rise in activity. The two contrasts thus engage the same electrodes along largely
distinct population codes, sharing a component that appears only late in the
trial.

**[One of the following, depending on the seeds:]** *(if the RT-matched − random
gap exceeds the seed spread)* After matching RTs across the four cells, the
transfer was reduced relative to a trial-count control (congruency → switch: 43%
vs 70% retained; switch → congruency: 12% vs 39%) and confined to 0.5–1.1 s after
the stimulus, before most responses (median RT 1.17 s). Part of the shared
component therefore reflects the RT difference shared by incongruent and switch
trials. *(otherwise)* RT matching did not change the transfer beyond the
variability between pseudopopulation draws.

**[Region sentence, once lPFC has been subsampled to 54 electrodes:]** Occipital
electrodes showed a transfer retaining a similar share of their ceilings (54% and
23%), but at +1.0 s and later rather than at 0.5–1.0 s.

### Limitations to state

- The samples in every test are CV repeats of one pseudopopulation, and there is
  no estimate across participants (no leave-one-participant-out).
- Electrodes from different participants were never recorded together. A
  pseudopopulation code is an upper bound on what any one participant's lPFC
  shares.
- The overall-activity control removes only a shift common to all of a
  participant's decoded electrodes. A change on a subset of them still counts as
  pattern.
- These are the base effects, not their adaptation (LWPC, LWPS). The analysis
  says nothing about whether the adaptation effects share a code.

### Parameter appendix

| Parameter | Value | Where set |
|---|---|---|
| ROI, electrodes | `lpfc` (control: `occ`), task-significant (`sig`), no selection | `ROI`, `ELECTRODES`, `ELECTRODE_DEFINITION=none` |
| Condition set | 4 cells, congruency × switch type, blocks pooled | `CONDITIONS=stimulus_main_effect_conditions` |
| Window, step | 64 / 16 samples at 256 Hz (250 / 62.5 ms), 37 windows | `WINDOW_SIZE`, `STEP_SIZE` |
| Classifier | PCA (80% variance) → LDA, equal priors | `EXPLAINED_VARIANCE`, `make_decoder` |
| Cross-validation | stratified 5-fold on the four cells, 10 repeats | `N_SPLITS`, `N_REPEATS` |
| Null | training labels permuted, refit per fold | `cv_cm_jim_window_shuffle(shuffle=True)` |
| Cluster test | one-tailed, 500 permutations, α = 0.05 | `N_PERM`, `ALPHA` |
| RT matching | per participant, 10 quantile bins, equal counts per cell | `RT_MATCH=rt`, `RT_MATCH_BINS`, `RT_MATCH_BALANCE` |
| RT control | same counts, drawn without regard to RT | `RT_MATCH=random` |
| Activity control | participant mean subtracted per pseudo-trial and time point | `ACTIVITY_CONTROL=remove_mean` |
| Seed | 0 (folds, pseudo-trial draw, RT draw, cluster tests) | `SEED` |
