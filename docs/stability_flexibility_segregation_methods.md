# Methods — stability/flexibility segregation analysis

Two interchangeable, self-contained Methods write-ups for the analysis implemented in
`src/analysis/stats/stability_flexibility_segregation.py` (driven by
`dcc_scripts/stats/stability_flexibility_segregation_dcc.py`):

| Version | Effect measure | Launcher setting | HG input per trial |
|---|---|---|---|
| **A** | signed cluster mass over the analysis window | `EFFECT_MEASURE=cluster` | time course over `[tmin, tmax]` |
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

---

## Version A — cluster-mass effect measure (`effect_measure='cluster'`)

### Single-trial high-gamma

Analyses were performed on stimulus-locked single-trial high-gamma (HG,
70–150 Hz) from [N] patients performing the Global/Local task. Broadband HG was
extracted from the cleaned, average-referenced recordings with a filterbank–Hilbert
decomposition, epoched from −1.0 to 1.5 s relative to stimulus onset, and
z-scored trial-by-trial against a 0.5-s pre-stimulus baseline drawn from the
−1.0 to 0.0 s interval. Epochs were decimated by a factor of 8; trials
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
reported descriptively.

### Effect measure: signed cluster mass over time

Because the interaction may be transient within the analysis window, each
electrode's interaction magnitude was quantified with a **time-resolved,
cluster-based statistic** rather than a window average. For each electrode and
each contrast, the balanced difference-of-differences was computed **at every
time bin** in the window and converted to a per-bin *t* statistic
(`d-o-d(t) / SE(t)`, with SE pooled across the four cells). Bins whose |*t*|
exceeded the two-tailed critical *t* at α = 0.05 were retained, and the effect
was defined as the **signed cluster mass**, i.e. the sum of *t* over
supra-threshold bins (0 when no bin survived). This is a per-timepoint,
cluster-corrected interaction test that returns a single **signed, graded scalar
per electrode** — the property the continuous correlation and the 2×2
conjunction both require, and which a per-effect pass/fail mask cannot provide.

Two implementation notes. First, the significance mask can optionally be
obtained from the pipeline's permutation cluster test
(`ieeg.calc.stats.time_perm_cluster`, `USE_TIME_PERM_CLUSTER = True`,
1000 permutations, two-tailed) instead of the parametric threshold; the fast
parametric threshold is used by default because the effect is re-evaluated
inside the disjoint-half resampling and the per-electrode permutation null, where
a nested permutation test would be prohibitive. Second, a 2×2 interaction is a
four-cell difference-of-differences, not a two-sample contrast, so the generic
two-condition permutation cluster test does not apply to it directly; the
per-bin d-o-d *t* above is its four-cell analogue.

As a robustness complement, the whole analysis was repeated with an
**amplitude-only** measure (`effect_measure='peak_t'`): the signed per-bin d-o-d
*t* at the instant of maximal |*t*|. Cluster mass grows with an effect's
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
condition; which half supplied *x* and which supplied *y* was randomised on each
draw so the data are used symmetrically. Sensitivities were averaged over 200
independent random disjoint splits. A naive same-trial estimate was computed as a
diagnostic only, to show the magnitude of the shared-noise inflation the disjoint
estimator removes; it is not used for inference.

### Gain control and subject nesting

Two nuisance sources were removed before testing. (i) **Shared gain/SNR**: an
electrode with a high signal-to-noise ratio shows larger effects for *both*
contrasts, which by itself produces a positive *x*–*y* correlation. Each
electrode's overall task responsiveness was therefore computed (the mean |HG| over
trials and time bins, or, where available, the electrode's baseline-versus-signal
cluster statistic) and *x* and *y* were each linearly residualised on it.
(ii) **Subject nesting**: residualised sensitivities were centred within subject,
so the estimate reflects within-subject co-selectivity and matches the
within-subject permutation null used for inference. Subjects contributing fewer
than three usable electrodes were excluded from the continuous test.

### Continuous test: is stability sensitivity related to flexibility sensitivity?

The association between the residualised, within-subject-centred *x* and *y* was
quantified with Spearman's ρ across all electrodes. Significance was assessed
with a permutation null in which *y* was shuffled **within each subject** (10,000
permutations), which preserves between-subject structure and therefore isolates
the within-subject association; the two-tailed *p* value is the proportion of
permutations with |ρ| at least as large as observed. As a parametric cross-check,
a linear mixed model with a subject random intercept was fitted to the
responsiveness-residualised sensitivities. A positive correlation indicates
shared tuning (a domain-general core); a correlation at or below zero indicates
segregation.

### Categorical test: 2×2 conjunction

Each electrode was independently labelled as stability-selective (S) and/or
flexibility-selective (F). Labels were obtained from a within-electrode
permutation test on the same cluster-mass statistic: the block-proportion
modulator was permuted **within each level of the condition factor** (2,000
permutations), which holds both main effects and all cell counts fixed and nulls
the interaction alone — unlike a free label shuffle, which under unequal cells
lets a main effect masquerade as an interaction. The resulting two-tailed
*p* values were corrected across electrodes with the Benjamini–Hochberg FDR
separately for each construct, and electrodes with *q* < 0.05 were flagged.
Electrodes with an undefined statistic were carried as non-significant rather than
dropped, so the FDR denominator remains honest.

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
decomposition, epoched from −1.0 to 1.5 s relative to stimulus onset, and
z-scored trial-by-trial against a 0.5-s pre-stimulus baseline drawn from the
−1.0 to 0.0 s interval. Epochs were decimated by a factor of 8; trials
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
reported descriptively.

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
a time-resolved cluster-mass measure (`effect_measure='cluster'`; per-bin
difference-of-differences *t*, thresholded at α = 0.05 and summed over
supra-threshold bins) as a sensitivity analysis for transient effects.

### Per-electrode sensitivities on disjoint trial halves

For every electrode we estimated a stability sensitivity *x* and a flexibility
sensitivity *y*. Because both are estimated from the same trials, shared trial
noise would inflate their correlation, so *x* and *y* were computed on **disjoint
halves of that electrode's trials**. Trials were split into two halves stratified
on the full crossing of the contrast factors (congruency, incongruent proportion,
task sequence, switch proportion), so neither half was confounded with a
condition; which half supplied *x* and which supplied *y* was randomised on each
draw so the data are used symmetrically. Sensitivities were averaged over 200
independent random disjoint splits. A naive same-trial estimate was computed as a
diagnostic only, to show the magnitude of the shared-noise inflation the disjoint
estimator removes; it is not used for inference.

### Gain control and subject nesting

Two nuisance sources were removed before testing. (i) **Shared gain/SNR**: an
electrode with a high signal-to-noise ratio shows larger effects for *both*
contrasts, which by itself produces a positive *x*–*y* correlation. Each
electrode's overall task responsiveness was therefore computed (the absolute
mean HG over trials, or, where available, the electrode's baseline-versus-signal
cluster statistic) and *x* and *y* were each linearly residualised on it.
(ii) **Subject nesting**: residualised sensitivities were centred within subject,
so the estimate reflects within-subject co-selectivity and matches the
within-subject permutation null used for inference. Subjects contributing fewer
than three usable electrodes were excluded from the continuous test.

### Continuous test: is stability sensitivity related to flexibility sensitivity?

The association between the residualised, within-subject-centred *x* and *y* was
quantified with Spearman's ρ across all electrodes. Significance was assessed
with a permutation null in which *y* was shuffled **within each subject** (10,000
permutations), which preserves between-subject structure and therefore isolates
the within-subject association; the two-tailed *p* value is the proportion of
permutations with |ρ| at least as large as observed. As a parametric cross-check,
a linear mixed model with a subject random intercept was fitted to the
responsiveness-residualised sensitivities. A positive correlation indicates
shared tuning (a domain-general core); a correlation at or below zero indicates
segregation.

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
| Cluster-forming threshold | two-tailed *t* at α = 0.05 | `alpha` (`_interaction_cluster`) |
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
