# iEEG final figure plan: main effects as a reference for LWPC/LWPS

**Status:** working plan, 2026-09-25. Companion to
[`figure_plan.md`](figure_plan.md),
[`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md)
and §15 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) (the
all-lPFC and task-significant rewrite). Section numbers below (§15.5, §15.12,
§15.13) refer to that rewrite.

## Summary

Use the main effects (congruency, switch type) as a reference for the adaptation effects, not as results of their own. Main effects in lPFC are well covered in the literature; LWPC and LWPS are what is new in this paper.

The closing question becomes: **is the dorsoventral tilt in LWPC − LWPS inherited from how the base effects are organized?** The planned main-effect anatomy rerun ("go from this to the delta") answers it directly. Every outcome gives a one-sentence, non-combative ending:

| Result | Ending sentence |
| --- | --- |
| Main-effect delta tracks adaptation delta, and the tilt shrinks when it is a covariate | Each adaptation scales with the local strength of the demand it regulates. |
| Main effects show no matching tilt, or the tilt survives the covariate | Adaptation has spatial structure of its own, beyond the base effects. |
| Main effects are clearly organized, but adaptation delta is unrelated to them | The demands are organized in space; their adaptation is shared. |

Keep the overlap result (one intermixed population) as the anatomy headline. The tilt is modest and was not predicted, so the closing figure explains it rather than carrying the paper.

The other ideas are worth running but belong in the supplement unless they come out strong: main-effect electrode maps, adaptation within main-effect groups, brain–behavior, and congruency ↔ switch cross-decoding with task controls. The cross-proportion transfer stays shelved.

## Where the paper stands

The paper now characterizes concurrent regulation rather than arguing that stability and flexibility use independent substrates. The absent cross-effects (congruency × switch proportion, switch type × incongruent proportion) are a scoping statement, not a figure. This follows `docs/analysis_plan_concurrent_regulation.md`.

| Beat | Claim | Evidence | Status |
| --- | --- | --- | --- |
| N1 | Both adaptations are present in the same subjects and sessions | Behavioral LWPC and LWPS (RT, errors) | Done |
| N2 | lPFC high gamma carries both adaptation effects | LWPC and LWPS power traces in task-significant lPFC electrodes; direction tests | Traces done; direction tests implemented (`docs/n2_direction_tests.md`) |
| N3 | Both adaptations are decodable from distributed lPFC activity | LWPC and LWPS decoding from task-significant lPFC electrodes | Done; block-transfer cross-decoding shelved |
| N4 | How the two effects are organized across lPFC | Continuous per-electrode scores, anatomy (§15 of `docs/n4_continuous_anatomy.md`) | Done: overlap plus a dorsoventral tilt |

The gap is a closing figure that ties N2–N4 together. The candidates are the ideas assessed below.

## What the updated anatomy results say

LWPC and LWPS are carried by one intermixed lPFC population, and the balance between them tilts along the dorsoventral axis. Both findings are significant only in all lPFC; the task-significant set gives the same estimates at lower power.

| Result | All lPFC (398 electrodes, 22 participants) | Task-significant (171 electrodes, 21 participants) |
| --- | --- | --- |
| LWPC–LWPS correlation, separate trial halves (pre-specified) | r = +0.097, p ≤ 0.001 | r = +0.077, p = 0.057 |
| Distance between LWPC+ and LWPS+ centroids | 1.4 mm, p = 0.95 | 3.3 mm, p = 0.44 |
| z slope of delta (LWPC − LWPS), SD/mm | −0.0077, p = 0.0075 (Bonferroni over 3 axes: 0.022) | −0.0075, p = 0.24 |
| Tilt replicates across disjoint trial halves | p = 0.005 | p = 0.15 |
| Anterior–posterior slope | p = 0.58 | p = 0.30 |
| Full-data split-half reliability | 0.28–0.30 | 0.26–0.39 |
| Electrodes FDR-significant for LWPC / LWPS | 0 / 0 | 0 / 8 |
| Responsiveness vs LWPC / LWPS, within participant | +0.14 / +0.24 | +0.16 / +0.33 |

**Shape of the tilt.** Ventral lPFC is roughly balanced (mean delta −0.02) and dorsal lPFC leans LWPS (−0.24). Among task-significant electrodes, LWPC falls from d = 0.15 ventrally to 0.08 dorsally, while LWPS stays at 0.17–0.20. It is an effect-type × height interaction: neither effect's own slope is significant, and height explains 2.1 % of delta's variance.

**Two earlier readings no longer hold.**

- "Correlated at the noise ceiling" is withdrawn. The ratio cannot be estimated at these reliabilities; a participant bootstrap gives [0.54, 4.71]. Report the separate-half correlation and the two reliabilities instead.
- "LWPC dominance increases ventrally" is wrong. Delta is negative on average at every height, because LWPS is larger overall. The accurate phrasing is that LWPC is weaker, relative to LWPS, in dorsal lPFC.

**What this means for the new analyses.**

1. Adaptation scales with responsiveness on both effects. Electrodes selected for a main effect are the responsive ones, so they will show more of both adaptations. Any specificity claim must be relative (LWPC versus LWPS within the same electrodes), which cancels responsiveness.
2. Single electrodes are mostly noise (sign agreement between halves is 51–59 %), and no electrode passes FDR for LWPC. Tests must be population-level. Continuous tests across all electrodes beat thresholded groups.
3. The tilt is the one piece of spatial structure found. The base effects are the natural candidate explanation, which is why the main-effect anatomy is the priority.

## Each planned analysis

Only the main-effect anatomy rerun belongs in the main text as a test. The rest are descriptive panels, supplement material, or shelved.

| Analysis | Question it answers | Placement | Priority |
| --- | --- | --- | --- |
| Anatomy on congruency − switch delta | Is the adaptation tilt inherited from the base effects? | Main text, closing figure | 1 |
| Main-effect electrodes on the brain | Where are congruency, switch and both electrodes? | Supplement; defines groups for the next row | 3 |
| Adaptation traces and decoding within main-effect groups | Is each adaptation expressed where its demand is processed? | Supplement, as the picture of the continuous test | 2 |
| Brain–behavior | Does neural adaptation track behavioral adaptation? | Supplement unless striking | 4 |
| Cross-decoding within main-effect groups | Do congruency and switch share a coding axis inside each group? | Drop unless the next row works | 6 |
| Congruency ↔ switch cross-decoding with task controls | Are the base-effect codes separable? | Supplement (S5) | 5 |
| Cross-proportion (block) transfer | Does block context reconfigure the code? | Shelved | – |

### Anatomy on the congruency − switch delta

This extends the §15 pipeline to the main effects and asks whether the tilt in LWPC − LWPS follows the organization of congruency − switch. It is cheap and reuses existing code; the implementation details are in the next section.

- **If the two deltas track each other and the tilt shrinks with the main-effect delta as a covariate:** each adaptation scales with the local strength of its demand. This is the simplest ending, and it also explains why responsive electrodes adapt more.
- **If the main effects have no matching tilt, or the tilt survives the covariate:** adaptation has spatial organization of its own. More novel, but it rests on a modest, unpredicted tilt.
- **If the main effects are organized but unrelated to adaptation:** the demands are organized; their adaptation is shared.
- **Risk:** main-effect maps will be far more reliable than adaptation maps (more signal, not a difference of differences). Do not read "main effects are more segregated" off raw correlations across the two levels.

### Main-effect electrodes on the brain (congruency, switch, both)

A useful descriptive panel, and it defines the groups for the next analysis. It is not a claim on its own.

- Thresholded maps look disjoint even under independence. At 49 % and 56 % positive rates, independence alone leaves about 28 % of electrodes in both maps (§15.5).
- The "both" group is inflated by responsiveness: high-signal electrodes pass both thresholds.
- Pick one selection method in advance. The windowed-ANOVA within-electrode clusters already define the significant electrodes, so use them; put other methods in the supplement as robustness. Several methods in the main text invite a garden-of-forking-paths comment.
- Make sure the main effects in that model are balanced over the proportion factors (Type III, with proportion in the model); see the leakage trap in the next section.

### Adaptation within main-effect groups (traces and decoding)

This is the "select on the main effect, test adaptation inside" design from the retired section of `docs/figure_plan.md`. It reuses the existing LWPC and LWPS traces and decoders with a different electrode set.

- **If LWPC appears in congruency electrodes and LWPS in switch electrodes, more than in the crossed pairing:** adaptation happens on the substrate of the demand it regulates.
- **If both adaptations appear in every group:** adaptation is a broad modulation of task-responsive lPFC, consistent with the overlap result.
- **Responsiveness:** every main-effect group is more responsive than average, so both adaptations will be larger there. Test the group × effect-type interaction (is LWPC − LWPS larger in congruency electrodes than in switch electrodes?), never "significant here, not there".
- **Selection leakage:** under the 75/25 design, the congruency main effect and LWPC share noise (correlation about +0.05). For electrodes selected on incongruent > congruent, that nudges LWPC toward the predicted sign. Select on trial half A and test on half B (`trial_splitting.py`).
- **Power:** the task-significant tilt is already p = 0.24 with 171 electrodes. Splitting into three groups cuts power further.
- **Verdict:** use it as the picture of the continuous test. Main-effect traces and decoding on their own go to supplement S2.

### Brain–behavior

Reviewers will ask for it, but it is the least likely to give a clean result.

- **Across participants:** n = 21–22. Each participant's behavioral LWPC and LWPS rest on the rare cells (about 40–50 trials each), so individual scores are unreliable and any correlation is attenuated (the "reliability paradox", Hedge et al., 2018). A null is uninformative; a positive result is fragile.
- **Trial-wise version, a trap in the current code:** the adjustment is w(t) · RT with w = +1 on the rare cells, so w averages about −0.5. Any plain HG–RT correlation leaks into the "matched" slope. The cross-pairing control cancels it only if both electrode groups are equally tied to RT.
- **Safer model:** RT ~ congruency × incongruent proportion × HG, with a participant random effect; test the three-way term. Same for switch type × switch proportion.
- **Verdict:** supplement unless striking. The direction tests, where the neural sign matches the behavioral sign, already link brain and behavior at the group level.

### Cross-decoding within main-effect groups

Training a congruency decoder and testing it on switch-type labels inside each group asks about base-effect geometry, not adaptation. The diagonal cells are circular unless selection and decoding use disjoint trial halves. Drop it unless the whole-lPFC version below produces an interpretable result.

### Congruency ↔ switch cross-decoding with task positive controls

- **Upside:** if congruency ↔ switch transfer fails while a real positive control transfers, the paper can say "separable codes". With the anatomy's "overlapping population", that is the plan's pre-committed headline: overlapping tissue, separable codes.
- **Level mismatch:** this is about the base effects, while N4 is about adaptation. State that explicitly if both appear together.
- **The task controls are weaker than they look.** The colored frame that cues the task is drawn with the stimulus (`src/task/mainTask.m:163`). A stimulus-locked task decoder partly decodes frame color, so it transfers because the signal is large and partly visual. That validates the code path, not the effect-size regime (`docs/cross_decoding_controls.md` §2).
- **To make the control meaningful:** subsample electrodes or trials until within-condition task accuracy matches congruency's, then report transfer as a fraction of within-condition accuracy.
- **Task × switch type has a real confound:** on switch trials the previous task was the other one, so leftover previous-task activity differs between training and test trials. Task × congruency is the cleaner control.
- **Prerequisite:** within-condition accuracy for both congruency and switch type must clear chance, or a failed transfer means nothing.
- **Verdict:** supplement S5.

### Cross-proportion (block) transfer

Keep it shelved; Tobias's objection holds.

- The rare class holds about 20 trials per participant per block, so within-block accuracy sits near chance and a failed transfer has no ceiling to compare against.
- A drop in transfer is confounded with LWPC itself: a weaker congruency code in one block type is the adaptation effect, not a change of axis.
- If any of it is reported, put it in the supplement with the within-block ceilings and the control table (`docs/cross_decoding_controls.md` §7).

## How to run the main-effect anatomy cleanly

Compute all four scores per electrode from the same trial halves, with the main effects weighted equally across the proportion levels. Running the existing `contrast_mode='condition'` job separately gets both of those wrong.

### The four scores

For each process, the same four cell means give both the main effect and the adaptation effect. For congruency, with the simple effect at each incongruent-proportion level:

```math
\text{Congruency} = \tfrac{1}{2}\left[(i-c)_{25\%} + (i-c)_{75\%}\right] \qquad \text{LWPC} = (i-c)_{25\%} - (i-c)_{75\%}
```

Switch type and LWPS follow the same pattern over switch proportion. These are the existing `W_MAIN` and `W_INTERACTION` weights (`stability_flexibility_segregation.py:519`), applied to the congruency × incongruent-proportion cells. Standardize each score across electrodes as §15 does, then define two deltas:

- **Main-effect delta (dm):** congruency − switch. Positive means relatively congruency-dominant.
- **Adaptation delta (da):** LWPC − LWPS, as in §15.

### Two traps in a separate condition-mode run

1. **The halves won't line up.** `_strata_columns` (`stability_flexibility_segregation.py:274`) builds the split strata from the contrasts. Proportion mode stratifies on congruency, incongruent proportion, switch type and switch proportion; condition mode stratifies on congruency and switch type only. The two runs therefore draw different splits, and "half A" in one is not disjoint from "half B" in the other.
2. **Block effects leak into the main effect.** Condition mode balances congruency over switch type (`BALANCE_MAIN_EFFECTS`, `:610`), but weights it by trial count over incongruent proportion. About 77 % of incongruent trials come from 75 %-incongruent blocks (137 of 179 per participant), and about 75 % of congruent trials from 25 % blocks (152 of 202). So the "congruency" score absorbs about half of any overall HG difference between block types. That is a signal confound; disjoint halves do not remove it.

### Shared noise on the same trials

Even with equal cell weights, the main effect and the adaptation effect share noise when cell counts are unequal. The covariance is proportional to:

```math
\tfrac{1}{2}\left(\frac{1}{n_{i,25}} + \frac{1}{n_{c,25}} - \frac{1}{n_{i,75}} - \frac{1}{n_{c,75}}\right)
```

With the per-participant counts (42, 152, 137, 50), the noise correlation is about +0.05. That sounds small, but it is half the size of the cross-effect correlations (about 0.1), with reliabilities of only about 0.3. Taking the main effect from one half and the adaptation effect from the other removes it.

### Code change

- [x] Extend `compute_sensitivities_per_split` (`stability_flexibility_segregation.py:707`) to also return main-effect columns on the same `g1`/`g2` halves: congruency on halves A and B, and switch type on halves A and B. Score them with `W_MAIN` over the proportion cells, in the proportion-mode run, so the splits are shared.
- [x] Add a synthetic check with two planted worlds: an adaptation tilt inherited from a main-effect tilt, and an adaptation tilt with no main-effect tilt. Test 2 below must shrink the slope in the first world and leave it in the second.
- [ ] Rerun both electrode sets (all lPFC and task-significant) with the new columns.

Implemented; how to run it and read the outputs is §16 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md).

### Test 1: do the two deltas track each other?

Correlate dm from half A with da from half B, and dm from B with da from A, within each split, then average over splits. Mirror `split_resolved_corr`: residualize on responsiveness, centre within participant, Spearman, participants with at least 3 electrodes, within-participant permutation null.

A positive correlation could come from both maps sharing one smooth gradient. That is the "inherited" hypothesis, not a problem. To ask whether the tracking goes beyond shared geography, refit with MNI coordinates as extra covariates.

Secondary, per process: correlate congruency with LWPC, and switch with LWPS, on separate halves. Use the crossed pairings (congruency with LWPS, switch with LWPC) as controls. Residualize on responsiveness here, because it inflates every pairing.

### Test 2: does the tilt survive the main-effect delta?

1. Check that the main effects have a tilt in the same direction: `relative_score_coordinate_test(value_col='dm')`. If inherited, congruency should be weaker than switch in dorsal lPFC. The swap null is valid for dm because it is a paired difference.
2. Refit the adaptation tilt with dm as a covariate: `relative_score_coordinate_test(value_col='delta', covariates=('resp', 'dm'))`. The function already takes covariates (`stability_flexibility_anatomy.py:1202`).
3. Compare the z slope with and without dm. Use dm from the half opposite to da where you can.

The swap null flips only the adaptation labels, which also destroys the dm relationship. The null is therefore wider than the true null, so the test is conservative, not anticonservative.

### Reporting cautions

- Report each map's split-half reliability. The main-effect maps will be far more reliable, so compare the raw correlations and reliabilities side by side, never a noise-corrected ratio.
- Treat the adaptation tilt as the thing being explained. It was not predicted, so the three-axis block test (p = 0.032) remains the protected result.

## Which electrode set to report

Recommendation: make all lPFC primary for the anatomy, and report the task-significant set in full as the consistency check. The choice was written down before any results, and the git history dates it.

| Date | Commit | What it says |
| --- | --- | --- |
| 2026-09-16 | `e69d97e` (analysis plan) | "electrode set is anatomical, not effect-selected" |
| 2026-09-17 | `3efaf71` (N4 guide §2.3) | "all electrodes in the predeclared anatomical scope" |
| 2026-09-21 | `0f98504` | First lPFC findings recorded (the all-lPFC run) |

**It is a different question from the traces.** The power traces and decoding ask whether an effect is present, where responsive electrodes make sense. The anatomy asks how an effect is distributed across a region, which needs the whole region.

**The task-significant set agrees; it is just smaller.**

- Same estimates: co-localization r = 0.08 versus 0.10; z slope −0.0075 versus −0.0077 SD/mm.
- The two groups' slopes do not differ (p = 0.67).
- Task significance is unrelated to height (within-participant r = +0.005, p = 0.93), so restricting to it changes the electrode count, not the spatial sampling.
- With the observed slope planted, p < 0.05 is reached 26 % of the time on the task-significant layout versus 66 % on all lPFC. The task-significant null is what that power predicts.

**Consequences.**

- Swap the order of the two draft Results paragraphs in §15.12, and drop "Because this subset was small".
- Use the same set (all lPFC) for the main-effect anatomy, so the comparison with adaptation is like for like.
- Add a Methods sentence along these lines: "Anatomical analyses used all lPFC electrodes, as specified before analysis, because they ask how effects are distributed across the region rather than whether they are present. Results in the task-responsive subset are reported in full."

Without that justification in Methods, switching populations for one section reads as choosing the set that gives significance.

## Proposed closing figure

One anatomy figure (F5) with four panels: the overlap, the tilt, the base effects on the same axis, and the link between them. It ends the paper on the arc: both adaptations in behavior, both in lPFC high gamma, decodable, one shared population, and a balance that does (or does not) follow the base demands.

| Panel | Content | Status |
| --- | --- | --- |
| a | LWPC against LWPS across electrodes (`joint_scatter.png`), annotated with the pre-specified separate-half r and the centroid test. Remove the noise-corrected value it currently prints. | Re-annotate |
| b | LWPC and LWPS against height: band or binned means ± SEM across participants, one line per effect. This is the figure for the tilt. | Not yet made by the pipeline |
| c | Congruency and switch main effects against height, same bands and axes as b. If the tilt is inherited, it shows here. | Needs the new main-effect columns |
| d | Main-effect delta against adaptation delta across electrodes (separate halves, within participant), with the z slope of the adaptation delta before and after the covariate. | Needs Tests 1 and 2 |

**Design notes.**

- Show b and c as matched small multiples with shared axes and one legend, so they read as one comparison.
- Plot participant-level means with SEM across participants, not electrode-level scatter. Single electrodes are not interpretable.
- If the tilt turns out inherited, panel d carries the ending. If not, panel c becomes the "base effects differ" contrast and d the null.
- Per-electrode dot maps appear only as coverage or illustration, with a legend line saying single electrodes are not interpretable.

### Supplement placement

| Item | Content |
| --- | --- |
| S2 | Main effects in lPFC high gamma: traces and decoding |
| S2b | Main-effect electrodes on the brain (one pre-specified method), plus other selection methods as robustness |
| S2c | LWPC and LWPS traces within congruency, switch and both groups, selected on half A and tested on half B, with the group × effect-type test |
| S5 | Congruency ↔ switch cross-decoding with the accuracy-matched task × congruency control, labelled as base-effect geometry |
| S8 | Cross-decoding control table for every transfer reported |
| S-BB | Brain–behavior: across-participant correlation with its n and reliability caveat, and the three-way mixed model |
| S-N4 | Task-significant anatomy in full; parcel test (`delta_by_roi.png`, reordered by mean z); anterior–posterior null |

## Priority order and weekly figure plan

Do the main-effect anatomy first: it is the only new analysis that can change the ending, and it reuses the N4 pipeline.

### This week, in order

- [x] Add matched-half main-effect columns to `compute_sensitivities_per_split`, with the synthetic inherited/independent check.
- [ ] Rerun the proportion-mode score jobs for all lPFC and task-significant with the new columns.
- [ ] Run Test 1 (delta–delta correlation) and Test 2 (tilt with and without dm), plus the dm coordinate test.
- [ ] Make the height figure: panels b and c.
- [ ] Carry over the §15.13 open items: rerun segregation with `N_PERM_CORR=10000`; rerun the anatomy jobs so `summary.txt` carries the Pearson-based value; re-annotate `joint_scatter.png`; switch `between_noise_corrected_ci` to a participant bootstrap.

### Next, if time

- [ ] Main-effect electrodes on the brain, and adaptation traces within groups on disjoint halves (supplement).
- [ ] Brain–behavior with the three-way mixed model.
- [ ] Congruency ↔ switch cross-decoding with the accuracy-matched task × congruency control.

### Weekly figure-plan template

One row per figure, updated each week: the claim it carries, where it stands, the next step, and the result that would change it. This week's version:

| Figure | Claim | Status | Next step | What would change it |
| --- | --- | --- | --- | --- |
| F1 | Both adaptations present concurrently in behavior | Done | None | – |
| F2 | Coverage and signal validation | Needs coverage table (S1) | Build per-ROI, per-participant table | – |
| F3 | lPFC high gamma carries LWPC and LWPS in the expected directions | Traces done | Confirm direction tests match behavior | A direction opposite to behavior |
| F4 | Both adaptations decodable from distributed lPFC activity | Done; transfer panel dropped | Report trial counts per decoder | – |
| F5 | One intermixed population; balance tilts dorsoventrally | Overlap and tilt done | Main-effect anatomy (Tests 1 and 2), height figure | Whether the tilt is inherited from the base effects decides the closing sentence |

Keep this table in `docs/figure_plan.md` so the repo stays the source of truth.

## Open questions for the advisor meeting

- [ ] **Primary electrode set for anatomy:** all lPFC (recommended, pre-specified) or task-significant (consistent with the traces)?
- [ ] **Scope of N4:** the N4 guide (§2.3) names whole-brain N4 as primary and lPFC as a separate, legitimate analysis. Is lPFC-only the paper's scope, given coverage outside lPFC?
- [ ] **Pre-commit to the ending:** are we content if the tilt turns out inherited from the base effects ("adaptation scales with its demand")? Agree now to report whichever outcome appears.
- [ ] **Main-effect electrode definition:** which single method defines congruency, switch and both electrodes (windowed-ANOVA clusters recommended)?
- [ ] **Weight of the tilt:** it was not predicted and explains 2 % of delta's variance. Does it appear in the abstract, or only in Results?
- [ ] **Brain–behavior:** supplement, or main text if the three-way mixed model is clear?
- [ ] **A4 cross-decoding:** keep in the supplement with the accuracy-matched task control, or drop?
