# Paper figure plan: stability and flexibility in iEEG

## The organizing principle

The main figures should follow the paper's inferential chain rather than catalog every analysis:

1. **The manipulation worked behaviorally.**
2. **The sampled neural population and univariate signals are suitable for the question.**
3. **Stability and flexibility have overlapping but partly selective LPFC substrates.**
4. **Context changes the strength and/or timing of those representations.**
5. **Cross-process effects test whether the two control demands interact.**

The key design rule is **one visual representation per scientific claim**. A time-resolved decoding trace and its temporal-generalization matrix should not both be in the main text unless the off-diagonal generalization pattern supports a distinct claim. Electrode-subset replications are likewise validation or mechanism analyses, not three complete repetitions of the population result.

## Recommended main-text structure (six figures)

### Figure 1 — Task, factorial manipulation, and behavior

**Claim:** The task independently manipulates stability and flexibility, and behavior shows the expected within-process adaptations without cross-process behavioral interactions.

- **a.** Trial schematic (cue, Navon stimulus, response, and timing).
- **b.** The 2 × 2 block design: incongruent proportion (25%, 75%) × switch proportion (25%, 75%).
- **c.** Reaction-time switch cost as a function of switch proportion, plus the corresponding cross-process comparison by incongruent proportion.
- **d.** Reaction-time congruency effect as a function of incongruent proportion, plus the corresponding cross-process comparison by switch proportion.
- **e–f (optional).** Accuracy/error-rate analogues only if they add a meaningful result; otherwise move them to the supplement and report the summary statistics in the text.

Plot subject-level estimates or distributions behind the group estimate. Use 95% confidence intervals when the goal is estimation; if the interval is within-subject, state the normalization method in the caption. Label the difference scores explicitly (for example, `RT_switch − RT_repeat`) rather than relying only on “switch cost.”

### Figure 2 — Coverage, preprocessing yield, and signal validity

**Claim:** The study samples the relevant anatomy and yields robust task-responsive neural signals.

- **a.** All retained electrodes on left/right cortical surfaces, with LPFC contacts emphasized and non-LPFC contacts visually de-emphasized.
- **b.** Participant-level coverage or coverage density, so one heavily sampled participant cannot be mistaken for broad cross-participant coverage.
- **c.** One representative electrode's time-frequency response, with the high-gamma band and analysis windows marked.
- **d.** One representative high-gamma trace or compact summary of task responsiveness.
- **e.** A small attrition/count flow: recorded → quality-controlled → LPFC → task-responsive LPFC.

Do not show every electrode trace in the main figure. Put participant-wise contact maps and per-electrode quality-control traces in supplementary figures or a browsable data supplement. If the paper ultimately analyzes only high gamma, use the spectrogram once as signal validation and keep subsequent figures focused on high gamma.

### Figure 3 — Anatomical organization of selectivity

**Claim:** LPFC contains congruency-selective, switch-selective, and mixed-selective sites, with their degree of anatomical segregation quantified rather than inferred from a map.

- **a.** All LPFC contacts.
- **b.** Task-responsive LPFC contacts.
- **c.** Selectivity map: congruency only, switch only, and both/mixed (use purple for both).
- **d.** Counts/proportions with participant-level points or uncertainty intervals.
- **e.** A compact, noncircular spatial-segregation statistic or observed-versus-null result.

Choose the classification scheme that matches the paper's primary claim before finalizing this figure. If the claim is about **representations of stability and flexibility**, classify by congruency and switch main effects. If the claim is specifically about **adaptation mechanisms**, classify by LWPC and LWPS interactions. Avoid combining the two schemes into one crowded taxonomy; the non-primary scheme belongs in the supplement.

### Figure 4 — Univariate high-gamma signatures and timing

**Claim:** Stability and flexibility demands, and their within-process adaptations, have distinguishable high-gamma magnitude and/or timing profiles.

Use a two-row structure: stimulus-locked above, response-locked below.

- **a.** Congruency main effect.
- **b.** Switch main effect.
- **c.** Direct comparison of congruency versus switch timing (peak/latency statistic and null distribution as an inset, not a separate figure).
- **d.** LWPC interaction contrast.
- **e.** LWPS interaction contrast.
- **f.** Direct comparison of LWPC versus LWPS timing (again as an inset or compact summary).

Show condition traces only where they aid interpretation; otherwise plot the relevant difference wave with its uncertainty interval and significant time clusters. The two timing tests naturally belong beside the traces from which they are derived. Cross-process univariate interactions can occupy a supplementary figure unless they are a central positive result.

### Figure 5 — Primary population decoding

**Claim:** LPFC population activity represents congruency and switch state, and the relevant block context changes those representations.

- **a.** A small analysis schematic defining features, labels, cross-validation unit, balancing, and chance level.
- **b.** Time-resolved congruency decoding in low- versus high-incongruency contexts.
- **c.** Their direct context-difference trace or prespecified-window effect size.
- **d.** Time-resolved switch decoding in low- versus high-switch contexts.
- **e.** Their direct context-difference trace or prespecified-window effect size.
- **f.** Compact participant/bootstrap summary in the inferential window(s), if the time-course panels do not already communicate uncertainty adequately.

This is the main decoding figure. Do **not** duplicate every panel with a temporal-generalization matrix. Include temporal generalization only if it answers a separate question—such as transient versus sustained coding—and then show one matrix pair per decoded variable, preferably in a dedicated supplementary figure.

### Figure 6 — Cross-process decoding and specificity

**Claim:** Context in one control domain changes representation of the other domain, and this result survives the analyses needed to rule out block identity and trial-composition explanations.

- **a.** Switch decoding across low- versus high-incongruency contexts.
- **b.** Direct cross-context difference/effect size.
- **c.** Congruency decoding across low- versus high-switch contexts.
- **d.** Direct cross-context difference/effect size.
- **e.** Block-balanced or leave-one-block/run-out control.
- **f.** A concise model-comparison/interpretive schematic distinguishing a resource tradeoff from task-information interdependence, but only if the analyses discriminate between them.

Because prestimulus decoding may reveal block identity, the control is part of the main claim, not supplementary housekeeping. The strongest design would ensure folds are independent at the block/run and patient levels as appropriate, balance trial labels within each block, and explicitly show that the context effect remains when block identity cannot drive classification.

## What moves to the supplement

1. **S1:** Participant demographics, electrode counts, and full participant-wise coverage.
2. **S2:** Preprocessing/quality-control examples and all participant/electrode high-gamma traces.
3. **S3:** Behavioral accuracy/error-rate results if they do not change the behavioral conclusion.
4. **S4:** Alternative electrode-classification scheme and sensitivity to task-responsive/contact-selection thresholds.
5. **S5:** Cross-process univariate high-gamma effects.
6. **S6:** Electrode-subset decoding displayed as a **summary grid**, not three full copies of Figure 5. Rows are electrode groups; columns are the four primary decoding effects; cells show prespecified-window effect sizes and confidence intervals.
7. **S7:** Temporal-generalization matrices for congruency and switch decoding, restricted to comparisons with an interpretable off-diagonal hypothesis.
8. **S8:** Frequency-band specificity (theta, alpha, beta) if these analyses are confirmatory or null; promote them only if the paper becomes a multiband story.
9. **S9:** Decoder robustness: alternative balancing, PCA thresholds, windows, metrics, and block-confound controls.

## How this solves the 20- and 30-panel problem

Use three levels of visual detail:

- **Main time courses:** only the population-level contrasts that carry the paper's central claims.
- **Compact summaries:** electrode-group results as forest plots/heatmaps of effect sizes in prespecified windows.
- **Diagnostic matrices:** temporal generalization and full confusion matrices in the supplement.

This avoids treating every crossing of **decoded variable × context × electrode group × locking × visualization type** as a separate main-text panel. The full factorial analysis can still be reported, but the paper should display each dimension in the form best suited to it: time courses for temporal claims, maps for anatomical claims, and interval/forest plots for subgroup comparisons.

## Noncircularity requirements for electrode-group decoding

Electrode-group decoding is only interpretable if selection is independent of evaluation. Use one of these strategies and state it directly in the figure schematic:

1. **Nested selection (preferred):** identify selective contacts using training folds only, then evaluate on held-out trials/runs.
2. **Independent localizer/window:** select contacts using independent data or a nonoverlapping, preregistered time window, then test decoding elsewhere.
3. **Leave-one-participant-out definition:** derive group selection criteria without the held-out participant where the inferential target supports this design.

If none is feasible, label electrode-subset decoding exploratory and keep it supplementary. Do not use the same condition contrast and trials both to select electrodes and to claim above-chance decoding or enhanced accuracy in that selected group.

## Decisions needed before locking the storyboard

1. **Primary neural claim:** Is the headline about partially distinct *representations* (main-effect selectivity), partially distinct *adaptation mechanisms* (LWPC/LWPS selectivity), or context-dependent population coding? This determines whether Figure 3 or Figure 5 is the paper's centerpiece.
2. **Status of temporal generalization:** Is there a directional hypothesis and a meaningful off-diagonal result (for example, more sustained coding for stability than flexibility)? If not, omit it from the main text.
3. **Status of frequency bands:** Are theta/alpha/beta prespecified tests with interpretable effects, or exploratory completeness analyses? This determines whether Figure 2 should introduce a multiband story.
4. **Cross-process controls:** After balancing within block and using block/run-aware cross-validation, do the cross-process decoding effects and prestimulus effects remain? Figure 6 should not be finalized until this is known.
5. **Inference unit:** Are uncertainty and tests based on patients, electrodes, bootstrap resamples, or some combination? Main figures should foreground patient-level generalization and avoid presenting bootstrap precision as between-patient evidence.

## A leaner five-figure fallback

If the journal strongly limits figures, merge Figures 2 and 3 into **coverage and selectivity**, and keep the remaining order unchanged. Do not merge Figures 5 and 6: within-process and cross-process decoding support different claims, and separating them makes the logic—and the block-confound control—much easier to understand.
