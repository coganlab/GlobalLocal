# Methods — N4 segregation and continuous anatomy

This section provides manuscript-ready Methods text for the N4 analysis. It is
written for the primary configuration used by the continuous anatomy pipeline:
an anatomically defined electrode set, `CONTRAST_MODE=proportion`, window-mean
high-gamma effect sizes (`EFFECT_MEASURE=cohens_d`), 200 disjoint-half
resamples, and 10,000 inferential permutations. Bracketed quantities should be
replaced with values from the archived run. The categorical conjunction is
included as an explicitly exploratory companion analysis because its current
trial-wise interaction null does not preserve the block structure of the
proportion manipulation.

## Participants, recordings, and electrode population

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

## Per-electrode LWPC and LWPS scores

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

## Disjoint-half estimation and map reliability

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

## Exploratory categorical conjunction

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

## Anatomical localization and score scaling

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

## Primary coverage-conditioned anatomical test

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

## Secondary coordinate analysis

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

## Visualization and descriptive centres

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

## Statistical reporting and interpretation

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

