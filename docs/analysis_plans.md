# Analysis plans

Every analysis plan for the paper in one place, **newest first**. Each plan is
kept as written, with links updated, so the reasoning behind each decision stays
on record. Where two plans disagree, the newer one wins. The table says what
still holds in each.

**The current plan is the [closing figure plan](#closing-figure-plan)
(2026-09-25).** It does not replace the
[concurrent-regulation plan](#concurrent-regulation-plan). It keeps that plan's
N1–N4 narrative and its specs, and adds the main effects (congruency, switch
type) as the reference for the N4 anatomy and the closing figure. Read the two
together: the closing figure plan for what to do next and where each result
goes, and the concurrent-regulation plan for why each analysis is specified the
way it is.

**The figure-by-figure plan after the 2026-10-01 results, with the Methods and
Results text assembled in paper order, is [`paper_draft.md`](paper_draft.md).**
Where it differs from this document's "Proposed closing figure", "Supplement
placement" or weekly figure table, it wins.

| Plan | Written | Last edited | What still holds | Was |
|---|---|---|---|---|
| [Closing figure plan](#closing-figure-plan) | 2026-09-25 | 2026-10-01 | **Current.** Main effects as the reference for LWPC/LWPS, the all-lPFC vs task-significant electrode-set choice, the reasoning behind F5 and the supplement placement, open questions for the advisor. The panel-level F1–F5 plan, supplement table and to-do list moved to [`paper_draft.md`](paper_draft.md) on 2026-10-01. Its four-panel F5 is superseded there by a two-panel F5, with the tilt moved to the supplement | `closing_figure_plan.md` |
| [Concurrent-regulation plan](#concurrent-regulation-plan) | 2026-09-16 | 2026-09-27 | **Current framework.** The N1–N4 narrative, the framing rules, and the specs the runbooks implement. The status columns (§0, §1) and the one-week schedule (§10) are out of date; the closing figure plan has the current status | `analysis_plan_concurrent_regulation.md` |
| [Simplification plan](#simplification-plan) | 2026-09-10 | 2026-09-16 | The shared-vs-independent framing is retired. The estimator diagnoses (§1.1–§1.4) and bias fixes (§2.2–§2.2c) still hold, and the later plans rely on them | `analysis_simplification_plan.md` |
| [Nested electrode selection plan](#nested-electrode-selection-plan) | 2026-08-19 | 2026-08-19 | Never built as designed; it served the subpopulation drill-down that the figure plan retired. A single held-out split was built instead ([`decoding.md`](decoding.md) §3.3–§3.4, [`analysis_guide.md`](analysis_guide.md) §21). Still the reference if a select-then-test figure returns (closing figure plan, S2c) | `nested_electrode_selection.md` |
| [Figure plan](#figure-plan) | 2026-08-18 | 2026-09-23 | Narrative revised on 2026-09-16 to the concurrent-regulation framing. F5 and the supplement list are superseded by the closing figure plan, whose weekly table is the current F1–F5 status. The reviewer objections (coverage, low-frequency baseline) still apply | `figure_plan.md` |

Each plan keeps its own section numbers. A bare § inside a plan refers to that
plan's own sections, unless the plan's first paragraph names another document.

How to run each analysis lives elsewhere: [`analysis_guide.md`](analysis_guide.md)
for every pipeline, and the runbooks [`n2_direction_tests.md`](n2_direction_tests.md),
[`decoding.md`](decoding.md), [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)
and [`a6_brain_behavior.md`](a6_brain_behavior.md).

---

## Closing figure plan

*iEEG final figure plan: main effects as a reference for LWPC/LWPS*

**Status:** working plan, 2026-09-25. Companion to the
[figure plan](#figure-plan), the
[concurrent-regulation plan](#concurrent-regulation-plan)
and §15 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) (the
all-lPFC and task-significant rewrite). Section numbers below (§15.5, §15.12,
§15.13) refer to that rewrite.

### Summary

Use the main effects (congruency, switch type) as a reference for the adaptation effects, not as results of their own. Main effects in lPFC are well covered in the literature; LWPC and LWPS are what is new in this paper.

The closing question becomes: **is the dorsoventral tilt in LWPC − LWPS inherited from how the base effects are organized?** The planned main-effect anatomy rerun ("go from this to the delta") answers it directly. Every outcome gives a one-sentence, non-combative ending:

| Result | Ending sentence |
| --- | --- |
| Main-effect delta tracks adaptation delta, and the tilt shrinks when it is a covariate | Each adaptation scales with the local strength of the demand it regulates. |
| Main effects show no matching tilt, or the tilt survives the covariate | Adaptation has spatial structure of its own, beyond the base effects. |
| Main effects are clearly organized, but adaptation delta is unrelated to them | The demands are organized in space; their adaptation is shared. |

**Outcome, all lPFC (2026-09-27; §16.6 of `n4_continuous_anatomy.md`).** None of the three rows fits cleanly.
- **Label level:** the main-effect balance differs across Destrieux labels (p = 0.017), in step with the adaptation balance (label r = 0.73).
- **Electrode level:** the main-effect balance tracks the adaptation balance, process-specifically (Test 1, r = 0.09, p = 0.0003).
- **The tilt:** whether it is inherited cannot be determined. The main-effect balance is too noisy (split-half reliability 0.08) for the covariate test to separate none from all.
- **The axis:** the tilt is better described as dorsomedial versus ventrolateral than as dorsoventral, and it is carried by LWPC.

Ending sentence: *each adaptation tracks the local strength of the demand it regulates; whether the spatial gradient in their balance is inherited from the base effects could not be determined.*

Keep the overlap result (one intermixed population) as the anatomy headline. The tilt is modest and was not predicted, so the closing figure explains it rather than carrying the paper.

**Update 2026-10-01: the closing figure is cut to two panels, and the tilt goes to the supplement.** F5 is now the overlap at both levels (a) and Test 1 drawn as matched vs crossed correlations (b). The dm–delta correlation is kept as the test behind b, because its covariance is the two matched covariances minus the two crossed ones. The main text drops "balance" language and Test 2. Reasons: the tilt explains about 2 % of delta's variance, lies on an unpredicted axis, and its inheritance cannot be resolved, so it cannot carry a figure panel. Score maps cannot replace the correlations either (single-electrode reliability ~0.3). New ending sentence: *LWPC and LWPS are carried by one intermixed lPFC population, and each tracks the local strength of the demand it regulates.* Layout, placement and text: [`paper_draft.md`](paper_draft.md) §1.4 (F5), §3.4 and §3.5.

The other ideas are worth running but belong in the supplement unless they come out strong: main-effect electrode maps, adaptation within main-effect groups, brain–behavior, and congruency ↔ switch cross-decoding with task controls. The cross-proportion transfer stays shelved. The cross-decoding came back partial on 2026-10-01 and stays in the supplement (see its outcome below).

### Where the paper stands

The paper now characterizes concurrent regulation rather than arguing that stability and flexibility use independent substrates. The absent cross-effects (congruency × switch proportion, switch type × incongruent proportion) are a scoping statement, not a figure. This follows the [concurrent-regulation plan](#concurrent-regulation-plan).

*Update 2026-10-01:* the neural cross-effects now appear in F3, as the off-diagonal of a 2 × 2 grid of traces whose diagonal is LWPS and LWPC. They are still scope, not a dissociation claim. They are labelled neutrally and reported as "no cluster survived", not "not modulated" ([`paper_draft.md`](paper_draft.md) §1.4, F3).

| Beat | Claim | Evidence | Status |
| --- | --- | --- | --- |
| N1 | Both adaptations are present in the same subjects and sessions | Behavioral LWPC and LWPS (RT, errors) | Done |
| N2 | lPFC high gamma carries both adaptation effects | LWPC and LWPS power traces in task-significant lPFC electrodes; direction tests | Traces done; direction tests implemented (`docs/n2_direction_tests.md`) |
| N3 | Both adaptations are decodable from distributed lPFC activity | LWPC and LWPS decoding from task-significant lPFC electrodes | Done; block-transfer cross-decoding shelved |
| N4 | How the two effects are organized across lPFC | Continuous per-electrode scores, anatomy (§15 and §16 of `docs/n4_continuous_anatomy.md`) | Done for all lPFC: overlap at both levels, a dorsomedial gradient in the adaptation balance, main effects as the reference (§16.6) |

The gap is a closing figure that ties N2–N4 together. The candidates are the ideas assessed below.

### What the updated anatomy results say

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

### Each planned analysis

Only the main-effect anatomy rerun belongs in the main text as a test. The rest are descriptive panels, supplement material, or shelved.

| Analysis | Question it answers | Placement | Priority |
| --- | --- | --- | --- |
| Anatomy on congruency − switch delta | Is the adaptation tilt inherited from the base effects? | Main text, closing figure | 1 |
| Main-effect electrodes on the brain | Where are congruency, switch and both electrodes? | Supplement; defines groups for the next row | 3 |
| Adaptation traces and decoding within main-effect groups | Is each adaptation expressed where its demand is processed? | Supplement, as the picture of the continuous test | 2 |
| Brain–behavior | Does neural adaptation track behavioral adaptation? | Supplement unless striking. Run 2026-09-30: not striking, so supplement (S-BB) | 4 |
| Cross-decoding within main-effect groups | Do congruency and switch share a coding axis inside each group? | Dropped (2026-10-01) | – |
| Congruency ↔ switch cross-decoding with task controls | Are the base-effect codes separable? | Supplement (S5); largely separable, partial late transfer (2026-10-01) | 5 |
| Cross-proportion (block) transfer | Does block context reconfigure the code? | Shelved | – |

#### Anatomy on the congruency − switch delta

This extends the §15 pipeline to the main effects and asks whether the tilt in LWPC − LWPS follows the organization of congruency − switch. It is cheap and reuses existing code; the implementation details are in the next section.

- **If the two deltas track each other and the tilt shrinks with the main-effect delta as a covariate:** each adaptation scales with the local strength of its demand. This is the simplest ending, and it also explains why responsive electrodes adapt more.
- **If the main effects have no matching tilt, or the tilt survives the covariate:** adaptation has spatial organization of its own. More novel, but it rests on a modest, unpredicted tilt.
- **If the main effects are organized but unrelated to adaptation:** the demands are organized; their adaptation is shared.
- **Risk:** main-effect maps will be far more reliable than adaptation maps (more signal, not a difference of differences). Do not read "main effects are more segregated" off raw correlations across the two levels.

#### Main-effect electrodes on the brain (congruency, switch, both)

A useful descriptive panel, and it defines the groups for the next analysis. It is not a claim on its own.

- Thresholded maps look disjoint even under independence. At 49 % and 56 % positive rates, independence alone leaves about 28 % of electrodes in both maps (§15.5).
- The "both" group is inflated by responsiveness: high-signal electrodes pass both thresholds.
- Pick one selection method in advance. The windowed-ANOVA within-electrode clusters already define the significant electrodes, so use them; put other methods in the supplement as robustness. Several methods in the main text invite a garden-of-forking-paths comment.
- Make sure the main effects in that model are balanced over the proportion factors (Type III, with proportion in the model); see the leakage trap in the next section.

#### Adaptation within main-effect groups (traces and decoding)

This is the "select on the main effect, test adaptation inside" design from the retired section of the [figure plan](#figure-plan). It reuses the existing LWPC and LWPS traces and decoders with a different electrode set.

- **If LWPC appears in congruency electrodes and LWPS in switch electrodes, more than in the crossed pairing:** adaptation happens on the substrate of the demand it regulates.
- **If both adaptations appear in every group:** adaptation is a broad modulation of task-responsive lPFC, consistent with the overlap result.
- **Responsiveness:** every main-effect group is more responsive than average, so both adaptations will be larger there. Test the group × effect-type interaction (is LWPC − LWPS larger in congruency electrodes than in switch electrodes?), never "significant here, not there".
- **Selection leakage:** under the 75/25 design, the congruency main effect and LWPC share noise (correlation about +0.05). For electrodes selected on incongruent > congruent, that nudges LWPC toward the predicted sign. Select on trial half A and test on half B (`trial_splitting.py`).
- **Power:** the task-significant tilt is already p = 0.24 with 171 electrodes. Splitting into three groups cuts power further.
- **Verdict:** use it as the picture of the continuous test. Main-effect traces and decoding on their own go to supplement S2.

#### Brain–behavior

Reviewers will ask for it, but it is the least likely to give a clean result. How to run it and read the outputs: [`a6_brain_behavior.md`](a6_brain_behavior.md).

- **Across participants:** one neural score per participant, the mean of its signed per-electrode d, against its behavioral d-o-d on the same trials (`participant_scores`). With n ≈ 18–21, significance needs |r| ≥ 0.43–0.47. Behavioral split-half reliability (full length, `combinedData.csv`) is 0.69 for LWPC but only 0.37 for LWPS, so the "reliability paradox" (Hedge et al., 2018) bites mainly on LWPS; the neural reliability comes from the run. A null is uninformative; a positive result is fragile.
- **The behavioral LWPC was the wrong contrast until 2026-09-27.** The `blockType` map swapped blocks A and D, which turned "LWPC" into a congruency × switch-proportion contrast (mean 46 ms instead of 123 ms). It is fixed and pinned by tests; rerun anything computed with it.
- **RT coupling, across participants.** If HG tracks RT within cells, every electrode's LWPC contains that slope × the participant's own behavioral LWPC. That builds a matched correlation which also passes "matched beats cross" and the joint regression. The job also reports scores with the RT-linked part of HG removed (`rt_adjust_hg`, the pooled within-cell slope): report those, with the raw ones as an upper bound.
- **Trial-wise version, a trap in the current code:** the adjustment is w(t) · RT with w = +1 on the rare cells, so w averages about −0.5. Any plain HG–RT correlation leaks into the "matched" slope. The cross-pairing control cancels it only if both electrode groups are equally tied to RT.
- **Safer model:** RT ~ congruency × incongruent proportion × HG, with a participant random effect; test the three-way term. Same for switch type × switch proportion.
- **Verdict:** supplement unless striking. The direction tests, where the neural sign matches the behavioral sign, link brain and behavior at the group level, but RT coupling predicts that sign too; recheck them on the RT-adjusted electrode scores (`participant_electrode_scores.csv`).
- **Outcome, task-significant lPFC (2026-09-30; §13 of `a6_brain_behavior.md`):** null and uninformative. At n = 17, RT-adjusted LWPC r = 0.10 [−0.40, 0.55] and LWPS r = −0.06 [−0.52, 0.44]. Neural LWPC has no measurable between-participant reliability. The LWPS ceiling (0.46) is below the 0.48 needed, so even a perfect link has about 49 % power. **Supplement S-BB only**, framed as "could not be tested at this n". Manuscript text: §14 of `a6_brain_behavior.md` and [`methods.md` › A6](methods.md#a6-brainbehavior-supplement-s-bb).

#### Cross-decoding within main-effect groups

Training a congruency decoder and testing it on switch-type labels inside each group asks about base-effect geometry, not adaptation. The diagonal cells are circular unless selection and decoding use disjoint trial halves. Drop it unless the whole-lPFC version below produces an interpretable result.

How to run it, with the main-effect decoding and power traces in the same groups: §17.5 of [`analysis_guide.md`](analysis_guide.md).

**Outcome (2026-10-01): drop.** With held-out selection, every group transfers, and `congruency_only` keeps more of its ceiling than `both` (108% against 65%). That is the opposite of the shared-code prediction. Several groups have pre-stimulus windows. The `both` group is 5–9 electrodes and changes with each run's selection trials. Under RT matching its congruency ceiling is gone. Details: [`decoding.md` › A4 §13.8.3](decoding.md#138-results-2026-10-01), "Groups".

#### Congruency ↔ switch cross-decoding with task positive controls

- **Upside:** if congruency ↔ switch transfer fails while a real positive control transfers, the paper can say "separable codes". With the anatomy's "overlapping population", that is the plan's pre-committed headline: overlapping tissue, separable codes.
- **Level mismatch:** this is about the base effects, while N4 is about adaptation. State that explicitly if both appear together.
- **The task controls are weaker than they look.** The colored frame that cues the task is drawn with the stimulus (`src/task/mainTask.m:163`). A stimulus-locked task decoder partly decodes frame color, so it transfers because the signal is large and partly visual. That validates the code path, not the effect-size regime (`decoding.md` › Cross-decoding controls §2).
- **To make the control meaningful:** subsample electrodes or trials until within-condition task accuracy matches congruency's, then report transfer as a fraction of within-condition accuracy.
- **Task × switch type has a real confound:** on switch trials the previous task was the other one, so leftover previous-task activity differs between training and test trials. Task × congruency is the cleaner control.
- **Prerequisite:** within-condition accuracy for both congruency and switch type must clear chance, or a failed transfer means nothing.
- **Verdict:** supplement S5.
- **Implemented** (not yet run on real data): the A4 transfer now reports its within-contrast ceilings, and `submit_task_transfer_dcc.sh` runs task × congruency, task × switch type, and congruency / switch type across task. The accuracy matching is not built; the summary prints task against congruency within-level accuracy instead. Run recipe: §17.5 of [`analysis_guide.md`](analysis_guide.md); rationale: §3.5 of [`decoding.md` › Cross-decoding controls](decoding.md#cross-decoding-controls).

**Outcome (2026-10-01; [`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01)).** These runs cover the 171 task-significant lPFC electrodes, unselected: the baseline, RT matching with its random control, `remove_mean`, and both together; plus occipital.

| Question | Answer so far |
| --- | --- |
| Are both base effects decodable? | Yes. Peaks 0.76 and 0.76; congruency from +0.12 s, switch type from +0.31 s (window centres). |
| Do they transfer? | Partly, and only late. From +0.62 s (window 0.5–0.75 s) onward; congruency → switch keeps 47% of its ceiling, switch → congruency 26%. Both sit below their ceilings in most windows. No pre-stimulus windows. |
| Is the transfer a uniform rise in activity? | No. With each subject's mean removed, the transfers barely change while the ceilings fall (81% and 77% kept). `mean_only` not run. |
| Is it response time? | Partly, probably. RT matching cuts it relative to its random control (43% vs 70%, 12% vs 39%), leaving only 0.5–1.1 s, before most responses (median RT 1.17 s). But the half-trial runs have pre-stimulus clusters as large as the surviving transfer, and `kept` moves ~20 points between runs that should agree. Needs seeds. |
| Both controls at once? | Uninterpretable: the congruency ceiling is gone, in the random control as well. |
| Is it specific to lPFC? | Not shown. Occipital keeps a similar share (54%, 23%), though at +1.0 s and later rather than 0.5–1.0 s, and its own ceilings have pre-stimulus windows. Needs lPFC subsampled to occipital's 54 electrodes. |
| Positive controls, response-locked, seeds | Not run. |

What it supports: *the same lPFC electrodes carry both base effects along largely distinct codes, sharing a component that appears only from ~0.5 s and is not a uniform activity increase.* That fits the pre-committed "overlapping tissue, separable codes" headline in a weaker form: *largely* separable. Nothing yet says the shared part is a control code rather than RT or a signal found outside lPFC too.

**Verdict: supplement S5, not the closing figure.**

- **It answers a different question from the paper's.** N2–N4 are about LWPC and LWPS; this is about the base effects. A closing figure on another level of the question opens a thread instead of closing one. F5 already ties N2–N4 together.
- **Its own answer needs qualifiers.** Partial, partly RT-linked, and not shown to be specific to lPFC. A closing figure needs a one-sentence claim, and every honest sentence here needs a caveat.
- **Reviewers will see the statistics.** The inference is across CV repeats of one pseudopopulation, not across subjects. The control runs show artifact clusters at the size of the effect being compared.

The full-trial result supports one sentence in the Discussion, next to the N4 overlap result: same tissue, largely distinct base-effect codes.

**What would move it into the main text:** the RT-matched − random gap larger than the seed spread; lPFC at 54 electrodes still transferring at 0.5–1.0 s where occipital does not; T1 keeping most of its ceiling; and an estimate across subjects. Even then it would be a panel about the base effects, best placed beside the N4 overlap result, not after it.

#### Cross-proportion (block) transfer

Keep it shelved; Tobias's objection holds.

- The rare class holds about 20 trials per participant per block, so within-block accuracy sits near chance and a failed transfer has no ceiling to compare against.
- A drop in transfer is confounded with LWPC itself: a weaker congruency code in one block type is the adaptation effect, not a change of axis.
- If any of it is reported, put it in the supplement with the within-block ceilings and the control table (`decoding.md` › Cross-decoding controls §7).

### How to run the main-effect anatomy cleanly

Compute all four scores per electrode from the same trial halves, with the main effects weighted equally across the proportion levels. Running the existing `contrast_mode='condition'` job separately gets both of those wrong.

#### The four scores

For each process, the same four cell means give both the main effect and the adaptation effect. For congruency, with the simple effect at each incongruent-proportion level:

```math
\text{Congruency} = \tfrac{1}{2}\left[(i-c)_{25\%} + (i-c)_{75\%}\right] \qquad \text{LWPC} = (i-c)_{25\%} - (i-c)_{75\%}
```

Switch type and LWPS follow the same pattern over switch proportion. These are the existing `W_MAIN` and `W_INTERACTION` weights (`stability_flexibility_segregation.py:519`), applied to the congruency × incongruent-proportion cells. Standardize each score across electrodes as §15 does, then define two deltas:

- **Main-effect delta (dm):** congruency − switch. Positive means relatively congruency-dominant.
- **Adaptation delta (da):** LWPC − LWPS, as in §15.

#### Two traps in a separate condition-mode run

1. **The halves won't line up.** `_strata_columns` (`stability_flexibility_segregation.py:274`) builds the split strata from the contrasts. Proportion mode stratifies on congruency, incongruent proportion, switch type and switch proportion; condition mode stratifies on congruency and switch type only. The two runs therefore draw different splits, and "half A" in one is not disjoint from "half B" in the other.
2. **Block effects leak into the main effect.** Condition mode balances congruency over switch type (`BALANCE_MAIN_EFFECTS`, `:610`), but weights it by trial count over incongruent proportion. About 77 % of incongruent trials come from 75 %-incongruent blocks (137 of 179 per participant), and about 75 % of congruent trials from 25 % blocks (152 of 202). So the "congruency" score absorbs about half of any overall HG difference between block types. That is a signal confound; disjoint halves do not remove it.

#### Shared noise on the same trials

Even with equal cell weights, the main effect and the adaptation effect share noise when cell counts are unequal. The covariance is proportional to:

```math
\tfrac{1}{2}\left(\frac{1}{n_{i,25}} + \frac{1}{n_{c,25}} - \frac{1}{n_{i,75}} - \frac{1}{n_{c,75}}\right)
```

With the per-participant counts (42, 152, 137, 50), the noise correlation is about +0.05. That sounds small, but it is half the size of the cross-effect correlations (about 0.1), with reliabilities of only about 0.3. Taking the main effect from one half and the adaptation effect from the other removes it.

#### Code change

- [x] Extend `compute_sensitivities_per_split` (`stability_flexibility_segregation.py:707`) to also return main-effect columns on the same `g1`/`g2` halves: congruency on halves A and B, and switch type on halves A and B. Score them with `W_MAIN` over the proportion cells, in the proportion-mode run, so the splits are shared.
- [x] Add a synthetic check with two planted worlds: an adaptation tilt inherited from a main-effect tilt, and an adaptation tilt with no main-effect tilt. Test 2 below must shrink the slope in the first world and leave it in the second.
- [x] Rerun all lPFC with the new columns (2026-09-26; results in §16.6 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)).
- [ ] Rerun the task-significant set with the new columns.

Implemented; how to run it and read the outputs is §16 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md).

#### Test 1: do the two deltas track each other?

Correlate dm from half A with da from half B, and dm from B with da from A, within each split, then average over splits. Mirror `split_resolved_corr`: residualize on responsiveness, centre within participant, Spearman, participants with at least 3 electrodes, within-participant permutation null.

A positive correlation could come from both maps sharing one smooth gradient. That is the "inherited" hypothesis, not a problem. To ask whether the tracking goes beyond shared geography, refit with MNI coordinates as extra covariates.

Secondary, per process: correlate congruency with LWPC, and switch with LWPS, on separate halves. Use the crossed pairings (congruency with LWPS, switch with LWPC) as controls. Residualize on responsiveness here, because it inflates every pairing.

#### Test 2: does the tilt survive the main-effect delta?

1. Check that the main effects have a tilt in the same direction: `relative_score_coordinate_test(value_col='dm')`. If inherited, congruency should be weaker than switch in dorsal lPFC. The swap null is valid for dm because it is a paired difference.
2. Refit the adaptation tilt with dm as a covariate: `relative_score_coordinate_test(value_col='delta', covariates=('resp', 'dm'))`. The function already takes covariates (`stability_flexibility_anatomy.py:1202`).
3. Compare the z slope with and without dm. Use dm from the half opposite to da where you can.

The swap null flips only the adaptation labels, which also destroys the dm relationship. The null is therefore wider than the true null, so the test is conservative, not anticonservative.

#### Reporting cautions

- Report each map's split-half reliability. The main-effect maps will be far more reliable, so compare the raw correlations and reliabilities side by side, never a noise-corrected ratio.
- Treat the adaptation tilt as the thing being explained. It was not predicted, so the three-axis block test (p = 0.032) remains the protected result.

### Which electrode set to report

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

### Proposed closing figure

*Superseded 2026-10-01 by the two-panel F5 in [`paper_draft.md`](paper_draft.md) §1.4 (see the update under "Summary"). Kept as written below; panels b and c here are now S-N4 panels, and d is the new F5b.*

One anatomy figure (F5) with four panels: the overlap at both levels, the two balances by label, the gradient, and the link between the levels. It ends the paper on the arc: both adaptations in behavior, both in lPFC high gamma, decodable, one shared population, and a balance that tracks the base demands electrode by electrode.

Revised 2026-09-27 after the all-lPFC main-effect run. The data behind each panel, and which parts still need plotting code, are in §16.7.3 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md).

| Panel | Content | Status |
| --- | --- | --- |
| a | Congruency against switch beside LWPC against LWPS, on the scores each pre-specified test correlates: LWPC/LWPS from `x_resid`/`y_resid` in the segregation run's `continuous.csv`, congruency/switch the same transform of `mx`/`my`. Each half is annotated with its separate-half r, p and n, and the LWPC/LWPS half with the centroid test. Not `joint_scatter.png`; see "Panel a" below. | Not yet made; needs no rerun |
| b | The two balances by Destrieux label: adjusted dm against adjusted delta, one dot per label, with each omnibus test and the label correlation (r = 0.73) | Data ready (`n4_section16_followups.py`); not plotted |
| c | Congruency, switch, LWPC and LWPS by distance from the midline, participant means ± SEM per tertile, as matched small multiples. Replaces the "by height" panels; the height version goes to the supplement. | Data ready; not plotted |
| d | Test 1: the dm–delta correlation with the matched and crossed pairings | Data ready (`delta_tracking.csv`); not plotted |

**Design notes.**

- Show c as matched small multiples (base effects, adaptation) with shared axes and one legend, so they read as one comparison.
- For c, plot participant-level means with SEM across participants, not electrode-level scatter. Single electrodes are not interpretable; panel a is the one electrode scatter, captioned as in "Panel a".
- Test 2 stays in the text and supplement. It cannot say how much of the gradient is inherited (§16.6.6), so it does not carry a panel.
- Per-electrode dot maps appear only as coverage or illustration, with a legend line saying single electrodes are not interpretable.

#### Panel a

Plot the scores the pre-specified test correlates, not `joint_scatter.png`.

| | `joint_scatter.png` | Panel a | Pre-specified test |
| --- | --- | --- | --- |
| Trials behind each score | all (split-averaged) | all (split-averaged) | LWPC and LWPS from opposite halves of each split |
| Responsiveness removed | no | yes | yes |
| Centred within participant | no | yes | yes |
| Participants with < 3 electrodes | kept | dropped | dropped |

- **Points:** `x_resid` and `y_resid` from the segregation run's `continuous.csv`. They are the points in `segregation_summary.png`'s residualized panel, so no rerun is needed. Check the count against the test's (all lPFC: 397 electrodes, 21 participants).
- **Congruency/switch half:** the same points for the main effects, which the job does not write: `prepare_continuous` (`stability_flexibility_segregation.py`) on the segregation run's `electrodes.csv` with `mx`/`my` as `x`/`y`. Annotate it with the r, p and n in `correlation_main_effects.json` (all lPFC: r = 0.23, p = 0.0001, 397 electrodes, 21 participants). The caption logic below applies unchanged. Take both halves from the same `_main_effects` run, so they share splits.
- **Annotation:** the pre-specified r, p and n, and the centroid test. No fit line, second r or noise-corrected value.
- **Caption:** the points correlate about twice as strongly as r, mostly because half-trial scores are noisier (§15.5), so the caption has to say what r is. Draft: "Each point is one electrode's LWPC and LWPS score from all trials, after regressing out overall responsiveness and subtracting each participant's mean. r is the pre-specified test: LWPC from one half of the trials against LWPS from the other, averaged over 1,000 random splits. Because each half has half the trials, r is smaller than the correlation among the plotted points."
- **Leverage:** if a reviewer asks whether one participant drives r, answer with the pre-specified test rerun leaving out each participant (§15.13). The leave-one-out range on `joint_scatter.png` is for its own, uncorrected correlation.

Why not the existing figures:

- **`joint_scatter.png`:** no number on it is the pre-specified test. The r values, the leave-one-out range and the fit line's slope describe the plotted points (both axes are scaled to SD 1, so the slope is essentially their pooled Pearson r). The ceiling line's r uses separate halves, but it is pooled across participants and not residualised. Its axis labels say "disjoint half", but the points are split-averaged, which rebuilds the full-data scores (§15.2). Keep it as a pipeline diagnostic, in neither the main text nor the supplement.
- **`segregation_summary.png`:** its residualized panel has the right points, but the r in its title is the separate-half test, not those points' own correlation. Its null panel permutes the plotted points, not the statistic whose p it prints. Its categorical panels need FDR labels, and no electrode passes FDR for LWPC.

No scatter can show the separate-half r itself. The r averages two correlations per split (half A against half B and the reverse) over all the run's splits (1,000 in the all-lPFC main-effect run; 200 in the earlier runs), and a plot with one point per electrode shows either one of them or, averaged over splits, the full-data scores. To draw the separate-half relationship directly, bin instead: within each split, residualise and centre the half scores as the test does, bin electrodes by half-A LWPC and average half-B LWPS in each bin, then average over splits and both directions, with participant-bootstrap error bars. That matches the means ± SEM style of b and c but needs new code.

#### Supplement placement

| Item | Content |
| --- | --- |
| S2 | Main effects in lPFC high gamma: traces and decoding |
| S2b | Main-effect electrodes on the brain (one pre-specified method), plus other selection methods as robustness |
| S2c | LWPC and LWPS traces within congruency, switch and both groups, selected on half A and tested on half B, with the group × effect-type test |
| S5 | Congruency ↔ switch cross-decoding, labelled as base-effect geometry: the unselected lPFC transfer with its ceilings, the `remove_mean` and RT-matched / random runs, the occipital comparison, and the accuracy-matched task × congruency control once run (methods and draft text: [`methods.md` › A4](methods.md#a4-congruency--switch-type-cross-decoding)) |
| S8 | Cross-decoding control table for every transfer reported |
| S-BB | Brain–behavior: per-participant correlation, RT-adjusted and raw, with its n, reliabilities, ceiling and the power they imply (text drafted: `a6_brain_behavior.md` §14); the three-way mixed model if it is built |
| S-N4 | Task-significant anatomy in full; parcel test (`delta_by_roi.png`, reordered by mean z); anterior–posterior null |

### Priority order and weekly figure plan

Do the main-effect anatomy first: it is the only new analysis that can change the ending, and it reuses the N4 pipeline.

#### This week, in order

- [x] Add matched-half main-effect columns to `compute_sensitivities_per_split`, with the synthetic inherited/independent check.
- [x] Rerun the proportion-mode score job for all lPFC with the new columns.
- [ ] Rerun it for the task-significant set.
- [x] Run Test 1 (delta–delta correlation) and Test 2 (tilt with and without dm), plus the dm label and coordinate tests (all lPFC; §16.6 of `n4_continuous_anatomy.md`).
- [ ] Make panel a from `continuous.csv` (see "Panel a"), and run the leave-one-participant-out check on its r.
- [ ] Plot the revised F5 panels b–d (§16.7.3 of `n4_continuous_anatomy.md`).
- [ ] Carry over the §15.13 open items: rerun segregation with `N_PERM_CORR=10000`; fix `joint_scatter.png`'s axis labels and drop its noise-corrected value (it stays a pipeline diagnostic); switch `between_noise_corrected_ci` to a participant bootstrap. (The Pearson-based value is in the all-lPFC main-effect run's `summary.txt`.)

#### Next, if time

- [ ] Main-effect electrodes on the brain, and adaptation traces within groups on disjoint halves (supplement).
- [x] Per-participant brain–behavior scores (raw and RT-adjusted, shared-split reliability) and the behavioral block-map fix (`a6_brain_behavior.md`).
- [x] Run A6 on task-significant lPFC (2026-09-30; null and uninformative, `a6_brain_behavior.md` §13). Supplement text written (§14 there; `methods.md` › A6).
- [ ] All lPFC and the 0–0.5 s window as checks: only if a reviewer asks (`a6_brain_behavior.md` §14.1).
- [ ] Brain–behavior with the three-way mixed model.
- [x] Congruency ↔ switch cross-decoding on unselected lPFC, with RT matching, `remove_mean` and occipital (2026-10-01; supplement S5, see the outcome above).
- [ ] Its remaining controls: seeds for the baseline, `rt` and `random`; lPFC subsampled to 54 electrodes against occipital; the task positive controls; `mean_only`; response-locked ([`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01)).

#### Weekly figure-plan template

One row per figure, updated each week: the claim it carries, where it stands, the next step, and the result that would change it. This week's version:

| Figure | Claim | Status | Next step | What would change it |
| --- | --- | --- | --- | --- |
| F1 | Both adaptations present concurrently in behavior | Done | None | – |
| F2 | Coverage and signal validation | Needs coverage table (S1) | Build per-ROI, per-participant table | – |
| F3 | lPFC high gamma carries LWPC and LWPS in the expected directions | Traces done | Confirm direction tests match behavior | A direction opposite to behavior |
| F4 | Both adaptations decodable from distributed lPFC activity | Done; transfer panel dropped | Report trial counts per decoder | – |
| F5 | One intermixed population at both levels; the adaptation balance tracks the base-effect balance and has a dorsomedial gradient | Main-effect anatomy done for all lPFC (§16.6) | Panel a from `continuous.csv`; plot panels b–d; task-significant main-effect rerun | The task-significant replication. Settling whether the gradient is inherited would need a more reliable measure of the base-effect balance than these trial counts give. |

Keep this table here, updated each week, so the repo stays the source of truth. It is the current F1–F5 status; the [figure plan](#figure-plan) below keeps the original sequence.

*Updated 2026-10-01:* the current version of this table, with the panels and the to-do list for each figure, is §1.3 of [`paper_draft.md`](paper_draft.md).

### Open questions for the advisor meeting

- [ ] **Primary electrode set for anatomy:** all lPFC (recommended, pre-specified) or task-significant (consistent with the traces)?
- [ ] **Scope of N4:** the N4 guide (§2.3) names whole-brain N4 as primary and lPFC as a separate, legitimate analysis. Is lPFC-only the paper's scope, given coverage outside lPFC?
- [ ] **Pre-commit to the ending:** are we content if the tilt turns out inherited from the base effects ("adaptation scales with its demand")? Agree now to report whichever outcome appears.
- [ ] **Main-effect electrode definition:** which single method defines congruency, switch and both electrodes (windowed-ANOVA clusters recommended)?
- [ ] **Weight of the tilt:** it was not predicted and explains 2 % of delta's variance. Does it appear in the abstract, or only in Results? *Recommendation (2026-10-01): neither the abstract nor a figure panel. One Results paragraph with the pre-specified parcel and coordinate tests, and the rest in S-N4 ([`paper_draft.md`](paper_draft.md) §1.4, F5).*
- [ ] **Naming the axis:** the pre-specified coordinate model reports height (z), but height and distance from the midline correlate r = −0.58 in lPFC, and a follow-up favours distance (§16.6.4). Report "dorsomedial versus ventrolateral" with both models, or keep height as the headline? *If the tilt recommendation is taken, this only affects S-N4.*
- [ ] **Brain–behavior:** supplement, or main text if the three-way mixed model is clear? The across-participant test came out null and uninformative (2026-09-30). Recommendation: one supplementary note plus one Discussion sentence (`a6_brain_behavior.md` §14). Build the three-way model only if a main-text brain–behavior claim is wanted; it is the only version with plausible power.
- [ ] **A4 cross-decoding:** keep in the supplement with the accuracy-matched task control, or drop? *Recommendation (2026-10-01): supplement S5, not the closing figure, plus one Discussion sentence. The groups are dropped. See the outcome under "Congruency ↔ switch cross-decoding with task positive controls".*

---

## Concurrent-regulation plan

*Analysis plan — concurrent regulation of stability and flexibility*

**Status:** active wrap-up plan (2026-09). One-week scope: close out anatomy, add
the cross-decoding analyses that are cheap, and stop.

**Relationship to the other docs.** This supersedes the *framing* of the
[simplification plan](#simplification-plan) (which
argued shared-vs-independent mechanisms) and the *narrative* of the
[figure plan](#figure-plan) (which was built around a mixed-selectivity
drill-down). Everything those documents say about **estimators and their biases**
still holds and is not re-derived here — §2.2/§2.2b/§2.2c of the simplification
plan in particular. [`analysis_guide.md`](analysis_guide.md) remains the
description of the pipelines as built;
[`stability_flexibility_battery.md` › Data flow walk-through](stability_flexibility_battery.md#data-flow-walk-through)
remains the A1–A7 walk-through. Cross-decoding troubleshooting lives in its own
document: [`decoding.md` › Cross-decoding controls](decoding.md#cross-decoding-controls).

---

### 0. The narrative

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

### 1. What already exists

| Piece | Where | Note |
|---|---|---|
| Behavioral LWPC/LWPS | `stats/erin_linear_mixed_effects_model.py`, `combinedData.csv` | done; **check the source**: that script is a post-error model with no congruency × proportion term, and its `blockType` map swapped blocks A and D until 2026-09-27. Per-participant values come from `behavioral_lwpc_lwps_magnitudes` (both positive on `combinedData.csv`: LWPC +123 ms, LWPS +97 ms; [`a6_brain_behavior.md`](a6_brain_behavior.md) §1.1) |
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

### 2. N2 — direction tests on the adaptation effects

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

- **Unit of inference is the electrode, pooled across subjects — no subject term
  in the null.** Score both simple effects per electrode over the lPFC electrode
  set and the pre-specified window, then a one-sample **sign-flip permutation on
  the per-electrode difference-of-differences**, electrodes exchangeable.

  This matches the power traces §2 exists to interrogate, and that consistency is
  the argument.
  `create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition`
  (`power/evoked_builders.py`) extracts one trial-averaged evoked per electrode
  and `extend`s them into a flat list — subject identity is discarded at that
  line — and `time_perm_cluster_between_two_evokeds` then runs
  `time_perm_cluster(..., axis=0, permutation_type='independent')` on the
  resulting `(n_electrodes, n_times)` array, permuting condition labels along the
  electrode axis. §2's statistic is the same object: a mean over electrodes whose
  SE comes from between-electrode spread. Holding the kill switch to a stricter
  null than the traces it validates would let it fail a direction those traces
  already reported — an incoherent thing to build.

  Note the §2 test is the **paired** form of that null (the
  difference-of-differences is formed *within* electrode before the test), so it
  is more powerful than the trace null as currently configured, not a relaxation
  of it. §9.4 shows the traces can be switched to the same paired null with one
  line, at which point §2 becomes simply the windowed case of the trace test.
  For the time course, `stability_flexibility_timing.interaction_time_course`
  already produces the per-electrode DoD trace and `_combine_electrode_traces`
  collapses it.
- **State what pooling costs, rather than pretending it is free.** Electrodes
  within a subject are correlated, so the effective N sits below the electrode
  count by roughly the design effect `1 + (m̄ − 1)·ICC`. At ~174 lPFC electrodes
  across 12 subjects (m̄ ≈ 14.5), an ICC of 0.1 puts the effective N near 74 and
  an ICC of 0.3 near 34. **The consequence is an optimistically small p-value,
  not a wrong sign** — and §2's output is a *direction*, cross-checked against
  behavior and against each simple effect's own sign, so an inflated p changes
  nothing about the verdict. This is the general rule for the plan: pooling
  without a subject term costs an optimistic p-value in §2, the power traces and
  the decoding, and that is acceptable. It is **not** acceptable on the
  LWPC–LWPS correlation (§5.1), where between-subject SNR offsets can reverse the
  sign and manufacture the effect outright. Optimistic p on a sanity check and a
  fabricated effect on a headline claim are different categories of error.
- **Report `n_electrodes` and `n_subjects` together**, plus two cheap leverage
  checks — kept as *descriptives*, since they are what actually protects this
  result, not the p-value: the per-subject direction tally (how many subjects'
  electrode averages point the expected way), and a leave-one-subject-out sweep
  (§9.2). lPFC coverage is skewed, so LOSO is what rules out a one-subject
  result. Subject-level aggregation is not the alternative: a paired t over ~12
  subjects has ~11 df and too little power to serve as a kill switch.
- **Report each simple effect's own sign and significance**, not just the
  interaction. Two simple effects with the same sign and different magnitude is
  the expected adaptation pattern; a sign flip is a different (and more
  interesting, and more suspicious) result.
- Use the **cell-balanced** scoring (`_interaction_effect` with equal cell
  weights — the default in `contrast_mode='proportion'`). The 75/25 design makes
  trial-count-weighted contrasts leak the main effect into the interaction; see
  simplification plan §2.2b.

- **Implemented** as `statistical_method='time_perm_cluster_interaction'`
  (`dcc_scripts/power/power_traces_dcc.py`). One run emits all three tests per
  ROI: both simple effects and the difference-of-differences, each cluster-
  corrected across time on the pooled-electrode unit. Orientation is pinned
  LOW-proportion-minus-HIGH by selecting the subtraction pairs by name, so a
  POSITIVE effect is the predicted shrinkage. Note this is the opposite sign
  from `windowed_anova._signed_contrast_per_window` and from
  `W_INTERACTION` in the segregation module, both of which compute high−low.
  Requires `STAT_FUNC_CHOICE = 'ttest_rel'` (→ `permutation_type='samples'`);
  the design is paired and `'independent'` throws the pairing away. Each of the
  three tests gets its own figure (traces + that test's own cluster bar) under
  `n2_direction_tests/<roi>/`, and its own npz with the mask, cluster p-values
  and the signed delta; see `n2_direction_tests.md` §6.

**Kill switch.** If the two adaptation directions disagree with the behavioral
ones, stop and re-read the epoch metadata before running anything in §3–§8. This
check costs an afternoon and protects the whole week.

---

### 3. N3a — LWPC / LWPS decoding (already run; what to add)

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
   confound readout, not a result — see `decoding.md` › Cross-decoding controls §5 and
   analysis_guide §17's standing caveat.

---

### 4. N3b — block-transfer cross-decoding (the new analysis)

> **Implemented for X1, X2, X2b, and X3.** See `decoding.md` › N3b block transfer for what was
> built, how to run it (`dcc_scripts/decoding/submit_block_transfer_dcc.sh`)
> and how to read the output. The build differs from the §4.2 sketch in two
> ways. It uses a `test_only` mask on the existing decoder instead of a
> `groups=` splitter. It loads a pooled, design-specific 2×2 condition set and
> balances the four contrast × transfer-level cells.

This is the analysis worth adding, because it is the *decoding analogue of the
adaptation effect itself*, which the existing A4 (train congruency → test
switchType) is not. A4 asks whether conflict and switching share a coding axis;
that is a base-effect question and it belongs in the supplement.

#### 4.1 The designs

| Design | Train | Test | Reads as |
|---|---|---|---|
| **X1 (primary)** | congruency, in 25%-incongruent blocks | congruency, in 75%-incongruent blocks (and reverse) | LWPC as a **cross-condition generalization failure**: if block context reconfigures the congruency code, transfer drops below the within-block ceiling |
| **X2 (primary)** | switchType, in 25%-switch blocks | switchType, in 75%-switch blocks (and reverse) | the same for LWPS |
| **X2b (reciprocal control)** | switchType, in 25%-incongruent blocks | switchType, in 75%-incongruent blocks (and reverse) | switch coding across the block factor used by X1; reciprocal counterpart to X3 |
| **X3 (positive control)** | congruency, in 25%-switch blocks | congruency, in 75%-switch blocks | congruency across a factor that should **not** reconfigure it. Same ROI, same trial-count regime, same effect-size regime as X1 — this is what makes a null X1 interpretable |
| **X4 (positive control)** | big letter, task = global | big letter, task = local (occipital) | validates the transfer **code path** on a signal that must be there. Caveat: on congruent trials big and small letter are confounded, so this is a code-path control, not a claim about global-specific coding |
| **X5 (optional)** | incongruent proportion (25 vs 75, collapsing congruency) | switch proportion (25 vs 75, collapsing switchType) | do the two block-context signals share an axis? Orthogonal factors of the same 2×2, so it is well posed |

X1, X2, X2b, and X3 are implemented. The letter-identity X4 remains a proposed
code-path control; X5 remains optional.

#### 4.2 Implementation

`build_cross_decoding_arrays` constructs **two labellings of the same trials**
and lets `StratifiedKFold` inside `cv_cm_jim_window_shuffle` make train and test
disjoint. N3b is the other shape: **one labelling, two disjoint trial
populations** (the blocks), where which trials train and which test is *fixed by
the design*, not by the fold.

The implemented path keeps the existing splitter but restricts it to the
training population. Concretely:

```
block_transfer.prepare(...)
    -> data, labels, transfer level, condition strata, balance group

Decoder.cv_cm_jim_window_shuffle(..., test_only=mask)
    -> cut folds only among unmasked training-level trials
    -> score every fold's classifier on all masked test-level trials
```

`shuffle=True` keeps working unchanged and remains the right null (permute train
labels, refit). Each resample balances the four contrast × transfer-level cells,
then optionally centers each transfer level. Synthetic tests assert that a
planted shared code transfers and a planted block-specific code does not.

#### 4.3 Two things that will decide whether X1/X2 mean anything

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
`decoding.md` › Cross-decoding controls §2 states the decision rule.

#### 4.4 Check the joint cell counts before running anything

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

### 5. N4 — per-electrode continuous scores → anatomy (the primary anatomical test)

#### 5.1 Scores

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

**Pool electrodes across subjects for the maps and the anatomy model, and put
the two effects on a common scale with ONE pooled scaling per effect — not a
within-subject z-score:**

```python
# one scale factor per EFFECT, computed across all electrodes pooled
for src, dst in (("lwpc_score", "lwpc_s"), ("lwps_score", "lwps_s")):
    sd = brain_table[src].std(ddof=1)
    brain_table[dst] = brain_table[src] / sd if np.isfinite(sd) and sd > 0 else np.nan
```

Three reasons this is the right shape, in descending order of how much they cost
if ignored:

1. **A within-subject z-score is degenerate at these electrode counts.** With
   **2 electrodes** in a subject, `std(ddof=1)` forces the two z-scores to
   *exactly* ±0.707 whatever the data — all magnitude information in that
   subject is destroyed and replaced by a symmetric pair of extremes. With **1
   electrode** the SD is `NaN`, so the subject is **silently dropped from the
   brain map**. lPFC has several subjects in exactly that range, so this is not a
   hypothetical. Pooled scaling has neither failure mode.
2. **The scores are already commensurate.** `_interaction_effect` is a
   difference-of-differences divided by the pooled within-cell SD — a
   standardized, unit-free, *d*-like effect size for both LWPC and LWPS. The only
   thing left to equalize before differencing is their overall marginal spread,
   and a single pooled scale factor per effect does exactly that.
3. **Subject gain is handled by the null, not by rescaling.** §5.2's null is a
   within-electrode swap of the two effect labels, which preserves subject,
   coverage, location and the electrode's own responsiveness *exactly*. A subject
   with better SNR has larger |LWPC| **and** larger |LWPS|, and the swap carries
   that through untouched, so it cannot manufacture an anatomy × effect-type
   interaction. The `(1 | subject)` term in §5.2 and the LOSO sweep in §9.2 are
   the remaining guards, and they are the right ones.

This is consistent with §9.2: electrode-weighted inference pooled across subjects
is acceptable and standard — what makes it safe is the leverage check, not
per-subject normalization.

**The one place a within-subject operation stays is the LWPC–LWPS correlation
(§5.4 / the scatter), and there it is centering, not z-scoring.** That is the one
number pooling can genuinely invent: a subject high on both axes for SNR reasons
produces a positive pooled correlation that is a subject-level gain effect, not
electrode-level co-localization. `prepare_continuous` already centers within
subject, and centering is safe at small n in a way z-scoring is not — a
1-electrode subject lands at (0, 0), contributing zero to the covariance
numerator and to both variance sums, so it leaves `r` numerically unchanged
instead of being dropped. You do not have to take a position on this in the
abstract: `joint_scatter_diagnostics` already returns `corr` and
`corr_within_subject` side by side. Report both; if they agree, say so in
Methods and pool.

**Check `min_elec` before reading any correlation.** `prepare_continuous` and
`split_resolved_corr` both default to `min_elec=3`, which drops **whole
subjects**, not marginal electrodes. That filter is a bigger lever on the
effective N than anything in the pooling question above, so sweep it over
{1, 2, 3} and report the sensitivity alongside the primary correlation.

#### 5.2 The test, and the fallacy it has to avoid

**The question is an interaction: effect type × anatomy.** Showing LWPC is
significant in region A while LWPS is not, and concluding a dissociation, is the
difference-of-significance fallacy (Nieuwenhuis, Forstmann & Wagenmakers 2011).
Whatever the model, effect type and anatomy go in the same model with an explicit
interaction term.

Two forms, both worth reporting:

**Categorical (primary).** Relative score `Δ = lwpc_s − lwps_s` per electrode
(the pooled-scaled scores of §5.1), tested against ROI / Destrieux parcel:

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

#### 5.3 Implementation gap

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

#### 5.4 The noise ceiling decides what a null means

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

### 6. N4 — brain maps of the continuous scores

Five surfaces, from the same table:

1. signed LWPC effect
2. signed LWPS effect
3. |LWPC|
4. |LWPS|
5. `lwpc_s − lwps_s` (the relative map)

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

### 7. Centroids — descriptive only

Demote this. A single pooled cross-subject centroid is not a defensible primary
test, for four reasons that all apply here:

1. Subjects with more electrodes dominate the **location estimate itself** — a
   pooled centroid is pulled toward wherever the densest implant happens to sit.
   Note this is *not* the electrode-weighting question settled in §2 and §5.1: a
   pooled effect score with a within-subject null and a LOSO sweep (§9.2) is
   fine, whereas coverage moves a centroid's estimate and not merely its
   variance, which no null can undo.
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

### 8. Haufe-transformed decoder patterns — convergent evidence, run last

Worth doing (it is what the advisor asked for) and worth labelling honestly:
**complementary evidence that distributed populations carry each form of
adaptation, not the primary anatomical localization.**

#### 8.1 Why raw weights will not do

Raw LDA weights answer "which linear combination best separates the classes,"
not "which electrodes carry the activity." With correlated features a decoder can
assign a large weight to a noise-cancelling electrode with no class information,
a small weight to a strongly informative but redundant one, opposite signs to
correlated neighbours, and unstable signs across folds. The fix is the forward
model (Haufe et al. 2014): multiply the covariance of the training features by
the weight vector, in the original input space.

#### 8.2 The procedure, in the order the pipeline forces

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

#### 8.3 Implementation gap and the standing caveat

`_window_and_predict_minimal` calls `self.fit(...)` then `self.predict(...)` and
discards the model each fold. Collecting patterns needs a variant of that loop
that stores the unwound, sign-pinned pattern per (fold, window). Budget most of a
day including the round-trip test: plant a known pattern in
`synthetic_roi_labeled_arrays`, back-project, confirm recovery.

**Caveat to write into the figure caption.** PCA at 80% explained variance mixes
channels, so the back-projection is spatially blurred and is therefore *weaker*
evidence about anatomy than the per-electrode effect maps. Keep it convergent.

---

### 9. Controls that apply across the whole plan

#### 9.1 Power / responsiveness

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

#### 9.2 Subject-level sanity on every pooled number

`segregation_scatter.joint_scatter_diagnostics` already computes per-subject
correlations, leave-one-subject-out range, the most influential subject, the
drop-top-2%-of-electrodes value, the within-subject correlation, and the maximum
subject share of electrodes. Run it on the anatomy tables too, or at least
report per-subject counts and a leave-one-subject-out sweep next to every pooled
anatomical statistic. Electrode-weighted inference across subjects is acceptable
and standard — but only after checking that two subjects are not supplying the
result.

#### 9.3 Baseline / block effects

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

#### 9.4 The power traces run an independent permutation on a paired design

**Verified against `ieeg.calc.stats.time_perm_cluster`.** `permutation_type` is
forwarded unchanged into `scipy.stats.permutation_test`, so `'independent'`
pools the observations of both samples along `axis` and randomly re-partitions
them. Every electrode contributes an evoked to **both** conditions, so the design
is paired and the null is carrying between-electrode variance — electrodes differ
enormously in overall HG amplitude — that pairing would cancel. The traces are
therefore **losing sensitivity**. This is conservative, not anti-conservative:
nothing already reported is called into question by it.

**This is a configuration choice, not a bug, and the paired path is already
wired.** `dcc_scripts/power/run_power_traces_dcc.py` sets
`STAT_FUNC_CHOICE = 'ttest'`, which resolves to `PERMUTATION_TYPE =
'independent'`. The `'ttest_rel'` branch in the same block resolves to
`PERMUTATION_TYPE = 'samples'`, and the `mean_diff` branch carries the comment
"Choose based on whether the observations are matched." One line switches the
statistic and the permutation type together.

**The switch also makes §2 and the traces the same procedure.** For two samples,
scipy's `'samples'` permutation randomly swaps the paired observations, which for
a difference statistic is exactly a sign-flip on the per-pair difference — the
test §2 specifies. Under `'ttest_rel'` / `'samples'`, §2 stops being "the paired
form of the trace null" and becomes the windowed case of it: same unit (electrode,
pooled across subjects), same null family.

Two checks before flipping, neither yet done:

1. **Assert the pairing is real.** `time_perm_cluster` calls
   `make_data_same(sig2, sig1.shape, axis, -1, True, rng)` before testing, and
   `'samples'` pairs whatever sits at matching indices. The two evokeds are built
   by the same loop over the same subjects and electrode dict so the order should
   match, but a mismatch would make the paired test *wrong* rather than merely
   conservative. Add `assert evoked_cond1.ch_names == evoked_cond2.ch_names` to
   `time_perm_cluster_between_two_evokeds` — worth having under either mode.
   
   **Check 1 is now implemented** as a `ValueError` at the top of
  `time_perm_cluster_between_two_evokeds`, raised only under
  `permutation_type='samples'`.

2. **Smoke-test `ttest_rel` through the vectorized path** on one ROI before a
   full sweep. `_handle_stat_func` will take its "stat_func returns a tuple"
   branch for scipy's `ttest_rel`, and `vectorized=True` batches the input.

Minor, unrelated, and not worth fixing on its own: `time_perm_cluster_between_two_evokeds`
defaults to `stat_func=None` and forwards it, overriding `time_perm_cluster`'s own
`stat_func=ttest` default and reaching `inspect.signature(None)`. The DCC path
always passes `stat_func` explicitly, so this only bites a direct call with
defaults from a notebook.

---

### 10. Schedule (one week)

Sequenced so the cheap things that can invalidate expensive things run first.

| Day | Work | Gate |
|---|---|---|
| 1 | §2 direction tests. §4.4 joint-cell trial counts. §9.2 per-subject sanity on the existing scatter. | Directions match behavior? Is the within-block decode above chance? |
| 2 | §5.1 build the score table; §5.3 `attach_scores` + the join to ROI/anat/coords. §6 continuous brain maps (maps 1–5). | Score table joins cleanly to coordinates for every subject |
| 3 | §5.2 the categorical ROI interaction test + within-electrode swap null; §5.4 the ceiling. | Ceiling reported next to every r |
| 4 | §4.2 implement the block-transfer splitter + synthetic test. | Planted shared code transfers; planted block-specific code does not |
| 5 | Run X1, X2, X3 with the §4.3 pair (within-block ceiling + transfer), centered and uncentered. X4 if time. | `decoding.md` › Cross-decoding controls checklist filled in |
| 6 | §8 Haufe back-projection + round-trip test; spatial comparison against the §5 maps. | Recovery test passes before touching real data |
| 7 | §5.2 continuous coordinate model, §7 descriptive centroids/medoids, figures, Methods paragraphs. | — |

**Cut in this order if the week compresses:** §7 centroids → §5.2-continuous
(coordinates) → §8 Haufe → X4/X5. Do not cut §2, §5.4, or the §4.3 ceiling —
those are what make the rest reportable.

---

### 11. What is deliberately not in this plan

| Dropped | Why |
|---|---|
| Independent-vs-dependent framing; cross-effect nulls as a result | Not the narrative. Scope statement only. |
| Electrode-count pie charts per group | Weak evidence, and counts are small. The continuous score map answers the same question better. |
| Anatomy restricted to individually-significant LWPC / LWPS electrodes | Too few electrodes, and selecting on the effect biases the location question. |
| Pooled cross-subject centroid as a primary claim | §7's four failure modes. |
| Per-electrode single-channel decoding accuracy maps | Answers §5's question at ~n_electrodes× the compute. |
| ROI-restricted decoding with electrode-count matching; ROI ablation / knockout | Genuinely good analyses (they capture multivariate contribution the univariate map misses) — but each is a full decoding sweep, and the week does not have room. Note them as the obvious extension if a reviewer asks "which electrodes *uniquely* drive decoding." |
| A4 label transfer (congruency ↔ switchType) in the main text | Base-effect geometry, not adaptation. Supplement, honestly labelled. |
| Low-frequency bands | Needs the longer / pre-block baseline re-run first (the [figure plan](#figure-plan)). Not this week. |

---

### 12. What to claim, and what not to

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

### 13. References this plan leans on

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

---

## Simplification plan

*Analysis simplification plan — stability/flexibility shared vs. independent mechanisms*

Working plan produced from a diagnostic pass over the decoding and power-trace
pipelines (2026-09). Records what the diagnosis found, what to do instead, and
why. Companion to `analysis_guide.md` (which describes the pipelines as built);
this document argues for changing which of them is primary.

> **Framing superseded (2026-09).** The paper's narrative has moved from
> *shared vs. independent mechanisms* to **concurrent regulation of stability and
> flexibility, characterized in the brain** — see the
> [concurrent-regulation plan](#concurrent-regulation-plan),
> which is the active plan. Two consequences for this document: the
> independent-vs-dependent argument is retired (the absent cross-effects are a
> scoping statement, not a result), and §2.8's "optional, and probably omit"
> verdict on cross-decoding no longer holds — the *block-transfer* form of it
> (train congruency in one incongruent-proportion block, test in the other) is
> now a planned analysis, for the reason §2.8 itself gives. **Everything else
> here still stands**, in particular the estimator diagnoses in §1.1–§1.4 and the
> three bias fixes in §2.2/§2.2b/§2.2c, which the new plan depends on and does
> not re-derive.

**Scientific question (as originally posed).** Do stability adaptation (LWPC) and
flexibility adaptation (LWPS) rely on shared or independent neural mechanisms in
lPFC? Under the current narrative this becomes the descriptive question of
whether the two adaptation effects are expressed by overlapping or distinct
electrode populations — asked and reported without the dissociation framing.

---

### TL;DR

1. **The decoding pipeline, as constructed, cannot answer the question.** It
   builds synthetic pseudotrials by sampling each channel's trials
   independently, which destroys the trial-level cross-electrode covariance that
   would make it multivariate. What remains is close to a weighted univariate
   analysis wearing MVPA clothing.
2. **The analysis that does answer the question already exists in this repo**:
   `src/analysis/stats/stability_flexibility_segregation.py`, run via
   `run_joint_distribution_analysis(..., contrast_mode='proportion')`. Make it
   primary.
3. **Four estimator improvements are now implemented, but they do not close the
   confirmatory-analysis checklist.** Two were planned: the split aggregation is
   fixed (§2.2 — the
   disjoint-half correction was being forfeited by averaging the estimates
   before correlating them) and split-half reliabilities are reported (§2.3 —
   without them a null result is uninterpretable; with them, an equivalence test
   can potentially supply positive evidence for distinct patterns). Two more
   turned up while testing those, both
   invisible to the split because they bias the *signal* rather than the noise:
   main-effect contrasts are now cell-balanced (§2.2b), and the default
   responsiveness proxy no longer double-counts the effects it is meant to
   control for (§2.2c). Under a simulated true null the estimator moved from
   −0.25 … +0.62 to roughly ±0.07. The remaining load-bearing issues are
   block-aware splitting/permutation, subject-level population inference,
   time-binned similarity with across-time correction, optional rather than
   mandatory responsiveness adjustment, and calibrated interpretation of null
   estimates. The categorical arm also still scores main effects the old way
   (§2.2b, "What is still open"). See the implementation-status table in
   `methods.md` › Segregation Methods.
4. **Leave the power traces alone.** Their structure is fine; the pain we found
   is specific to the pseudopopulation, which they don't use.
5. **Retire the block-context accuracy comparisons.** They are confounded by
   trial count and SNR, and they were never a direct test of shared mechanism.

---

### Part 1 — Diagnostic findings

Each finding is tagged with how confident to be: **[verified]** = read in the
code and traced; **[derived]** = arithmetic from verified values; **[open]** =
needs checking before relying on it.

#### 1.1 The pseudopopulation destroys cross-electrode covariance — [verified]

`labeled_array_utils.py:822-826`, inside the constructor that
`process_bootstrap` actually calls:

```python
for channel in all_channels_in_roi:
    channel_data = nan_removed_data_dict[condition_name][channel]
    sample_indices = rng.choice(len(channel_data), size=n_samples, replace=False)
    resampled_channels_for_condition.append(channel_data[sample_indices])
concatenated_chans = np.stack(resampled_channels_for_condition, axis=chans_axs)
```

Each channel draws its own trial indices. Pseudotrial row *i* is therefore a
chimera — channel A's trial 17 beside channel B's trial 4 — **even for two
electrodes in the same patient recorded simultaneously**.

What survives: each channel's condition-dependent mean and variance, and the
temporal structure within that channel's selected trial.

What is destroyed: simultaneous trial-by-trial covariance between electrodes,
coordinated population fluctuations, trial-level network states.

**Why this is the decisive finding.** A linear classifier on data with no
cross-channel covariance can only exploit each channel's marginal tuning. Its
information content is therefore close to that of a weighted sum of
per-electrode effects — which is what the power traces already measure, minus
the weighting. It is a legitimate form of pseudopopulation decoding, but it
answers "do electrodes collectively have condition-dependent tuning?" rather
than "does the trial-level population state carry the information?" Only the
second question justifies MVPA over a univariate contrast.

This also explains, retrospectively, why the block-split decoding accuracies
tracked trial count and SNR so closely: there was little genuine multivariate
signal for them to track instead.

#### 1.2 Trial counts are set by the worst channel, and differ across the cells being compared — [verified/derived]

`subsample_to_min_trials_per_condition` (`labeled_array_utils.py:761`) takes the
minimum valid trial count **across all channels in the ROI** for each condition.
One bad electrode caps the trial count for the whole ROI in that cell.
Conditions then land at different heights, `LabeledArray.from_dict` pads them to
a common height with NaN, and `data_prep.py` strips the padding — which is why
the `[NaN filter]` log lines report "79.4% dropped" for rare cells. That figure
is padding removal, not artifact rejection.

Because the surviving count per stratum is a property of the dataset rather than
of the run, the final n for any block-restricted split can be reconstructed by
replaying the balancing (`data_prep.py` steps 3–4) over the per-stratum counts
printed in the run logs. Doing so reproduces four actual runs exactly, so the
two unobserved cells below are trustworthy:

| contrast | block | final n/class | limiting cell |
|---|---|---|---|
| c vs i | MC (25% I) | 22 | `i_MC_MS` (11) |
| c vs i | MI (75% I) | 30 | `c_MI_MR` (15) |
| c vs i | MR (25% S) | 26 | `i_MC_MR` (13) |
| c vs i | MS (75% S) | 22 | `i_MC_MS` (11) |

Consequences for the four decoding panels:

- **Congruency × switchProportion** — MR = 26 vs MS = 22. The observed effect
  (25% S decodes better) runs *the same direction as the n difference*. That
  panel also carries a significant **pre-stimulus** cluster, in a window where
  congruency information cannot exist. Treat as a trial-count artifact.
- **Congruency × incongruentProportion** — MC = 22 vs MI = 30. The observed
  effect (25% I decodes better) runs *against* the n difference, so it survives
  this confound and is if anything understated.

#### 1.3 The parameter surface — [verified]

~20 parameters materially change the result before counting the optional
electrode-selection modes; ~45 with them. Two specific problems:

- `N_SHUFFLE_PERMS = 50` bounds the smallest resolvable p at 1/51 ≈ 0.02, while
  inference runs at one-tailed α = 0.025. That is at the resolution floor.
- `PERCENTILE`/`CLUSTER_PERCENTILE` (both 95) and `P_THRESH`/`P_CLUSTER` (both
  0.025) express two decisions in four parameters.

Feature count is also extreme: 174 electrodes × 64 samples = **11,136 features**
per pseudotrial against 22–30 observations per class. PCA is not optional at
that ratio, which makes `EXPLAINED_VARIANCE` consequential rather than cosmetic.

#### 1.4 The baseline carries a block-level confound — [verified]

`make_epoched_data.py:195` draws a random 0.5 s baseline window from within
−1.0–0 s; line 322 rescales with `mode='zscore'` using statistics **pooled
across all trials** per channel. So tonic block-level differences in HG survive
into the pre-stimulus window by construction. Since `incongruentProportion` *is*
the block in a blocked design, a sustained pre-stimulus separation is the
expected result, not a finding.

The asymmetry in the power traces confirms it: the switchType panel (varies
*within* block) shows no pre-stimulus cluster; the incongruentProportion panel
(block-level) shows one spanning the entire baseline. A single bad electrode
would contaminate both.

**Recommended control, not replacement:** per-trial baseline mean subtraction
with a pooled SD (see §2.4). Run it as a robustness check, because you cannot
determine from these data whether the tonic block difference is nuisance or is
itself the proactive-control effect — that is undecidable in a blocked design.

#### 1.5 Corrections — do NOT act on these earlier suggestions

Recorded so they aren't repeated:

- **"Uncomment the trial shuffle at `labeled_array_utils.py:236-240`."** Wrong.
  That code sits in `process_single_subject_labeled_array`, which
  `process_bootstrap` does not call. Trial sampling in the active path is
  already random. The fix would be a no-op.
- **"Surviving trials are each subject's chronologically first *k*."** Wrong,
  and it followed from the same misidentified path. Sampling is random
  (`rng.choice`).
- **"The minimum is across subjects."** It is across *channels* — stricter.
- **"Bootstraps are redundant with `N_REPEATS`."** Wrong. `N_REPEATS` re-splits
  the same trials into folds; `BOOTSTRAPS` changes which trials enter. They
  measure different variance components.
- **"Decode per subject instead of pooling."** Too aggressive for this dataset.
  lPFC electrode counts are heavily skewed and a per-subject decoder on 1–2
  electrodes sits at chance. The plan below keeps electrodes pooled and moves
  the *inference* unit to subject instead.

---

### Part 2 — The plan

#### 2.1 Primary analysis: joint-distribution segregation (already implemented)

`src/analysis/stats/stability_flexibility_segregation.py`, entry point
`run_joint_distribution_analysis`, with **`contrast_mode='proportion'`**.

This is **A2-continuous** in the existing battery. What it computes and why the
estimator is shaped the way it is are already documented — do not re-derive them
here:

| For | Read |
|---|---|
| what A2 does, step by step | `stability_flexibility_battery.md` › Data flow walk-through §3b, §8 |
| how it relates to (and differs from) RSA | `stability_flexibility_battery.md` › Data flow walk-through §10 |
| manuscript-ready Methods prose | `methods.md` › Segregation Methods |
| where it sits among A1–A7 | `stability_flexibility_battery.md` › Data flow walk-through §11 |

This document adds only the argument for making it **primary**, plus two defects
found in the implementation (§2.2, §2.3).

Why it is the right primary test:

- **It targets the adaptation question directly.** LWPC and LWPS are
  interactions — how the base effect changes with block proportion. That is what
  "stability/flexibility adaptation" means. Accuracy-in-context-A vs.
  accuracy-in-context-B is an indirect proxy for the same thing, mediated by
  signal, noise, trial count, and estimator behaviour.
- **It recovers what the ROI mean throws away — without needing trial-level
  covariance.** The ROI mean cancels when some electrodes push positive and
  others negative; asking whether the two effects land on the *same electrodes,
  in the same direction* does not. Note the vocabulary: this is **not**
  multivariate — each electrode's sensitivity is a scalar and there is no
  pattern dimension (`stability_flexibility_battery.md` › Data flow walk-through §10 is right about this, and the
  representational-geometry question lives in A4). Its virtue here is precisely
  that it needs no within-trial cross-electrode structure, so §1.1 does not
  touch it.
- **It already handles two confounds that would otherwise dominate** — shared
  trial noise via disjoint halves, shared gain/SNR via responsiveness
  residualisation. *(As originally implemented the shared-noise correction did
  not survive the aggregation; fixed — see §2.2.)*
- **Inference already respects subjects** — within-subject centering plus
  within-subject permutation (`subject_clustered_corr`), and CMH stratification
  for the categorical conjunction. This is the piece the decoding pipeline never
  had: CV folds, repeats and bootstraps are resampling replicates, not
  independent biological observations, and treating ~25 accuracy curves as n=25
  overstates evidence badly.

Interpretation:

| result | reading |
|---|---|
| corr > 0 | shared / domain-general core |
| corr ≈ 0 | no evidence of a common spatial pattern |
| corr < 0 | segregated subpopulations |
| both effects reliable, corr ≈ 0 | adaptation happens in lPFC via spatially distinct electrode sets |

That last row is precisely what an ROI-mean power trace cannot reveal, and it is
the scientifically interesting outcome.

**Verified in code** (`compute_sensitivities_per_split`, `_stratified_half_split`,
`split_resolved_corr`): the half-split is stratified on the contrast cells and
genuinely disjoint; both cross directions are used, so the halves enter
symmetrically; the permutation null shuffles y within subject, preserving
between-subject structure. All as documented. The aggregation problem found in
the original implementation is fixed — see §2.2, and §2.2b for a further bias
found while testing that fix.

#### 2.2 The split aggregation — **fixed**

**Status: implemented.** `compute_sensitivities_per_split` + `split_resolved_corr`,
wired into `run_joint_distribution_analysis` as the primary `correlation`.

**The bug.** `compute_sensitivities` averaged x over the 200 splits and y over
the 200 splits, then returned one `(x, y)` per electrode which
`subject_clustered_corr` correlated. That aggregation forfeited the
disjoint-half correction entirely.

Within one split *k*, `x_k` and `y_k` come from disjoint trial sets, so their
sampling noise is independent — which is the whole design. But the average
contains K·(K−1) cross terms `cov(x_j, y_k)` with *j ≠ k*, computed on trial
sets that overlap by ~50%. Those terms do not vanish and they swamp the K
disjoint ones.

How completely they swamp them is the striking part. Across three simulated
regimes, the split-averaged estimator and the naive same-trial estimator agree
to three decimal places:

| regime | naive (all trials) | split-averaged (old) | within-split (new) |
|---|---|---|---|
| A | −0.2777 | −0.2780 | −0.2074 |
| B | +0.6153 | +0.6157 | +0.4866 |
| C | +0.0591 | +0.0609 | +0.0535 |

200 splits bought nothing at all. (Regimes defined in §2.2b; truth is 0 in all
three.)

**The fix.** Correlate *within* each split, then average the correlations:

```
S = mean_k  ½ [ corr(x_A,k , y_B,k) + corr(x_B,k , y_A,k) ]
```

Both cross directions are used, so the halves enter symmetrically and the coin
flip in the old code is unnecessary. Residualisation on responsiveness and
within-subject centring are applied *per split*, matching `prepare_continuous`.
The permutation null uses one within-subject electrode permutation applied
identically across all splits. Because each split's vectors are centred and
unit-normed, a correlation is a dot product, so the whole thing collapses to a
lookup in one precomputed n_elec × n_elec matrix — 10,000 permutations cost
O(n_elec) each rather than O(n_splits × n_elec).

**What the fix does and does not buy.** Isolated in simulation (`bx = by = 0`,
so the *only* coupling is shared sampling noise, with a non-proportional 2×2
cross-tab so that noise actually covaries):

| | naive | within-split |
|---|---|---|
| pure shared-trial noise | −0.46 … +0.40 | +0.002 … +0.068 |

That is the mechanism the disjoint split was designed for, and it is removed
essentially completely. It is *not* the only mechanism — see §2.2b.

**A correction to an earlier version of this section.** It said the bias
"appears when the cells are unbalanced." That is imprecise. Unequal *marginals*
(25%/75%) are harmless on their own: what matters is whether the 2×2 cross-tab
is **proportional**. Writing the per-trial contrast weights

```
w_x(t) = +1/n_i if incongruent else −1/n_c
w_y(t) = +1/n_s if switch      else −1/n_r
```

the shared-noise covariance of the two effects is `σ² · Σ_t w_x(t) w_y(t)`,
which is exactly zero when `n_is = n_i·n_s/N` and so on, however lopsided the
marginals are. Simulated with independent 25%/75% factors, all estimators sit at
0 and there is nothing to fix. The bias needs congruency and switchType to be
*correlated in the trial table*. **Check this in your own data before assuming
either way** — compute the four cell counts per subject and test the cross-tab
for proportionality. That number, not the marginals, says how much of this
mattered.

#### 2.2b Cell-balanced main effects — a third bias the split cannot remove

**Status: implemented** (`BALANCE_MAIN_EFFECTS`, `W_MAIN`). Found while testing
§2.2; not in the original plan.

Testing the §2.2 fix against a true null in three regimes turned up two further
couplings, both of which survive the disjoint split untouched, because both are
**signal** confounds rather than noise confounds — they are present identically
in every trial and therefore in every half.

- **Regime B — design non-orthogonality.** If congruency and switchType are
  correlated in the trial table, an electrode with a purely congruency-driven
  response still scores a switch effect, because its incongruent trials are
  disproportionately switch trials. Every electrode inherits the same leakage,
  which is precisely what a spurious across-electrode correlation is made of.
- **Regime A — the shared pooled SD.** Cohen's *d* divides by an SD estimated
  from the same trials. A large stability effect inflates the within-group
  variance that the flexibility contrast divides by, deflating it — a *negative*
  coupling, which is the direction that would masquerade as segregation. (Regime
  A turned out to have a second, larger contributor as well; see §2.2c.)

Both are fixed at the contrast level, not the split level. Scoring a main effect
as the equal-weight mean of the within-cell differences makes the two contrasts
orthogonal in cell-mean space *by construction*, whatever the cell counts:

```
x = ½[(m_is − m_cs) + (m_ir − m_cr)]        # congruency, equal weight over switch
y = ½[(m_cs − m_cr) + (m_is − m_ir)]        # switch, equal weight over congruency
```

Simulated under a true null, ~330 electrodes, 12 subjects:

| regime | scoring | naive | within-split |
|---|---|---|---|
| B (non-orthogonal design) | pooled two-group (old) | +0.58 | +0.44 |
| B | cell-weighted (new) | −0.21 | **−0.07** |
| A (shared pooled SD) | pooled two-group (old) | −0.23 | −0.18 |
| A | cell-weighted (new) | −0.09 | **−0.09** |
| C (clean control) | either | ≈ 0 | ≈ 0 |

Neither change alone suffices: cell-weighting removes the signal-mediated
coupling, the split removes the noise-mediated coupling, and only together do
all four regimes land near zero.

**Note this only applies to `contrast_mode='condition'`.** The interaction path
(`contrast_mode='proportion'`, the manuscript's primary mode) has always used an
equal-cell-weight difference-of-differences, and is protected already. This
extends the same treatment one level down, to main effects.

Also tested and **rejected**: dropping the denominator entirely (a raw
cell-balanced mean difference). It fixes regime B equally well but is markedly
*worse* in regime A (−0.25 … −0.29 vs −0.09), because without standardisation
the sensitivities scale with per-electrode gain and the linear responsiveness
residualisation does not fully remove a multiplicative gain term. Keep the
pooled within-cell SD.

**One cost: it needs more trials per cell.** The balanced form requires ≥ 2
trials in each of the four 2×2 cells *of each half*, i.e. **≥ 4 per cell before
splitting**, where the old two-group form needed only 2 per group. Your cells
run 11–63, so halves give roughly 5–31 and this is not close to binding — but it
binds hard in sparser configurations, and when it does the electrode is dropped
from the correlation entirely (an effect must be defined on *every* split for the
per-split vectors to be comparable). `split_resolved_corr` reports
`n_electrodes_dropped` and warns when more than 10% go. If that warning fires,
the correlation is being computed on a non-random subset of electrodes and the
cell counts need looking at before the number means anything.

#### 2.2c The responsiveness proxy was a function of the effects — **fixed**

**Status: implemented** (`add_responsiveness`). A plain bug, found by chasing the
residual regime-A bias after §2.2b.

`prepare_continuous` residualises *x* and *y* on `resp` to remove the shared
gain/SNR confound. That only works if `resp` measures **gain** and nothing else.
The scalar-HG branch computed `|mean HG|` — the absolute value of the
electrode's mean — while the docstring, and the time-resolved branch, said
`mean |HG|`. Those are very different quantities. When both contrasts push the
electrode's mean the same way (which is what a population-level effect *means*),
`|mean HG|` behaves like `x + y`. Regressing *x* and *y* on their own sum drives
the two residuals apart, so the correction manufactures a **negative**
correlation — spurious *segregation*, the more publishable direction.

Regime A, naive estimator, three seeds:

| responsiveness proxy | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `\|mean HG\|` (scalar branch, as written) | −0.120 | −0.209 | −0.161 |
| `mean \|HG\|` (as documented; now both branches) | +0.025 | −0.066 | −0.020 |
| true per-electrode gain (oracle) | +0.058 | −0.052 | +0.012 |

The documented proxy lands level with the oracle. Both branches now compute
`mean |HG|`.

This only affects runs that used the **default** proxy. `add_responsiveness`
takes an explicit `responsiveness=` argument and the guide already recommends
passing a baseline-vs-signal cluster statistic; runs that did so are unaffected.

Taken together over regime A, the three fixes compose:

```
−0.25   original
−0.12   + cell-weighted main effects  (§2.2b)
+0.03   + corrected responsiveness proxy  (§2.2c)
```

**What is still open.** `per_electrode_labels` — the categorical (A3) arm —
scores simple contrasts through `_effect_from_arrays`, not `_effect_for`, so it
still uses the old pooled two-group form and is still exposed to the regime-B
leakage. It was left alone deliberately: changing it would change the S/F label
counts and the conjunction table, i.e. published numbers, and its
within-electrode permutation null is a separate design with its own documented
caveats. The consequence is an asymmetry worth knowing about when reading
`electrodes.csv`: `x`/`y` are now cell-balanced, `S`/`F` are not. Resolve before
the categorical arm is used for anything load-bearing in `condition` mode.

#### 2.3 Split-half noise ceiling — **added**

**Status: implemented.** `split_resolved_corr` returns `reliability_x`,
`reliability_y` and `corr_noise_corrected`; the launcher writes them to
`correlation.json` and prints them in `summary.txt` with an explicit
"a null corr is NOT evidence of independence" warning when either reliability is
low.

The trials are already split into disjoint halves A and B to estimate the two
sensitivities. From those same halves, also compute:

```
r_LWPC = corr(β_LWPC,A , β_LWPC,B)     # is the stability pattern reliably measured?
r_LWPS = corr(β_LWPS,A , β_LWPS,B)     # is the flexibility pattern?
S_corrected = S / sqrt(r_LWPC × r_LWPS)
```

**Why it is essential.** Without it, `S ≈ 0` cannot be distinguished from
"neither pattern is measured well enough to correlate with anything." With it,
high `r_LWPC` and `r_LWPS` alongside `S ≈ 0` is *positive evidence* that the
patterns are reliable **and** unrelated — which is the only honest route to an
"independent mechanisms" claim (see Part 6).

This matters more, not less, under the recommended anatomical electrode set
(§2.6), because including mostly-unresponsive electrodes attenuates `S` toward
zero and the ceiling is what makes the attenuated number readable.

Cost: it came free. Computing each contrast on *both* halves rather than one
(four effect evaluations per split instead of two) yields the reliabilities
directly and removes the need for the coin flip. The only price is 2× the effect
evaluations, which matters solely for `effect_measure='cluster'`, where each
evaluation runs its own permutation test — halve `n_splits` there.

`p` is reported for `S`; it applies to `S_corrected` unchanged, since the two
differ only by a fixed positive denominator and so order the null identically.

One operational note: an electrode whose effect is undefined on even a single
split (a contrast cell emptied by that split) is dropped from the correlation
entirely, so that the per-split vectors are comparable and the permutation
applies to a fixed electrode set. `n_electrodes_dropped` reports how many. With
few trials per cell this can bite; check it before interpreting `n_electrodes`.

#### 2.4 Supporting: power traces — keep as they are

No structural change needed. The pseudopopulation is a *decoding-specific*
requirement: decoding needs one trial carrying a value on all 174 channels
simultaneously, so it must synthesise them. Power traces average within each
electrode first, so heterogeneous trial counts never have to line up and none of
§1.1–1.2 applies.

Their role in the paper: establish that each adaptation exists in mean activity.
That is a different and complementary claim from whether they share a pattern.

Two changes worth making anyway:

1. **Re-run with per-trial baseline** as a robustness check (§1.4). Subtract
   each trial's own baseline mean, keep a pooled per-channel SD:

   ```python
   base = HG_base._data                                       # (trials, ch, base_times)
   per_trial_mean = np.nanmean(base, axis=-1, keepdims=True)
   pooled_sd = np.nanstd(base - per_trial_mean, axis=(0, -1), keepdims=True)
   HG_ev1_rescaled._data = (HG_ev1._data - per_trial_mean) / pooled_sd
   ```

   Mean-centre before taking the SD so the denominator describes the same
   quantity the numerator now contains (within-trial fluctuation), otherwise
   each channel's normalised baseline lands on a different scale set by how much
   tonic drift it happened to carry. Note Response epochs have fewer trials than
   the baseline array (434 vs 448 for D0121), so align on `TrialCount` metadata
   rather than position.

   Report the post-stimulus effect *with and without*. Do not read the flattened
   pre-stimulus window as evidence — the baseline is drawn from inside it, so
   flatness there is near-tautological.

2. **Report effect sizes alongside cluster presence.** The current
   double-dissociation claim rests on two *non-significant* interaction
   clusters, and non-significance is not evidence of specificity.

#### 2.5 Supporting: the scatterplot — **implemented**

**Status: implemented.** `src/analysis/stats/segregation_scatter.py`
(`plot_joint_scatter`, `joint_scatter_diagnostics`), wired into the launcher as
`segregation_joint_scatter.png`. It is written by every run, and can be produced
*on its own* — no splits, no permutations, minutes instead of hours — with
`SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh`.

Before any inference machinery:

- x-axis: each electrode's LWPC interaction
- y-axis: the same electrode's LWPS interaction
- colour by subject; marginal histograms on both axes

Read it as: positive diagonal → shared; spread on both axes with no
correspondence → independent; spread on one axis only → one mechanism; negative
diagonal → opponent; **all the structure in one colour or a few points →
artifact**.

This is an afternoon's work and will tell you most of the answer before you
commit to anything. It also gives reviewers a direct view of the data instead of
asking them to trust a pipeline.

**The last reading is a number, not an impression.** "One colour or a few
points" is exactly the kind of judgement that gets made generously when you
already like the answer, so `joint_scatter_diagnostics` computes it and the
figure prints it:

| quantity | catches |
|---|---|
| `per_subject` correlations + the bar panel | structure carried by one subject |
| `loso_min` / `loso_max`, `most_influential_subject` | how much the pooled r moves when any one subject is dropped |
| `corr_drop_top` | how much survives without the most influential ~2% of electrodes |
| `corr_within_subject` | whether the structure is within subjects or *between* them — a pooled r much larger than this one is a subject-level offset, which the pipeline centres out and the raw scatter does not |
| `max_subject_share` | one subject supplying most of the electrodes |

Any of these crossing a threshold is raised as a `flags` entry and drawn on the
figure. They are gated on |r| ≥ 0.1: below that there is no apparent structure
to attribute to anything, and a flat cloud is a *result* (the "independent"
reading), not a suspect figure — what makes it interpretable is the §2.3 noise
ceiling, not this panel.

**Two cautions on reading the scatter's number.** It is descriptive: no
responsiveness residualisation, no within-subject centring, no permutation, and
— on the default `SCATTER_N_SPLITS=0` path — x and y are scored on *all* of the
electrode's trials, so they share trial noise (§2.2). Expect it to sit above the
pipeline's estimate, and read it as an upper bound; the figure annotates the
pipeline's corrected `corr` beside it when both are available.
`SCATTER_N_SPLITS=200` scores the sensitivities on disjoint halves instead, at
the full estimator's cost. Second, the default correlation is Spearman, which is
rank-based and therefore resists exactly the few-extreme-points failure the last
reading warns about; a large Pearson/Spearman gap is itself the outlier signal.

#### 2.6 Electrode set: anatomical, not condition-selected

Use one anatomically defined lPFC set with recording-quality exclusions only.

Do **not** pre-select LWPC-significant, LWPS-significant, their union, or their
intersection before measuring co-localization — selecting on the effects whose
overlap you are about to test makes the overlap partly a property of the
selection rule. The current decoding runner defaults to `ELECTRODES='sig'` with
`ELECTRODE_DEFINITION_SPLIT` off, and offers several further selection modes;
none of them belong upstream of this analysis.

The bias this avoids, and the nested-selection machinery for cases where you
*must* select, are worked out in the [nested electrode selection plan](#nested-electrode-selection-plan) — see "Where
the bias sits" and "The null must run selection too". The recommendation here is
simply to sidestep it: an anatomical set needs no nested selection at all.

Trade-off: including unresponsive electrodes attenuates `S`. That is
conservative and acceptable — and it is exactly why §2.3 is not optional.

#### 2.7 Timing

You can keep timing without the decoding machinery. `S` is a scalar per time
window, so compute it in sliding windows and plot a time-resolved
shared-pattern curve. Correcting over time for one scalar is far simpler than
cluster-correcting differences between accuracy curves.

For the primary *inferential* claim, use one pre-specified window and show the
time course descriptively: "we show the full time course for completeness;
statistical inference was performed on a pre-specified window." If you want an
actual timing claim ("stability adapts earlier than flexibility"), onset latency
with a bootstrap CI on the difference is a sharper instrument than a cluster bar.

#### 2.8 Cross-decoding — ~~optional confirmatory~~ **now planned, in a different form**

> **Updated 2026-09.** The caveat at the end of this section (a transfer between
> *congruency* and *switchType* asks a base-effect question, not the adaptation
> question) is correct and is exactly why the analysis has been re-specified
> rather than dropped. The planned form transfers **across block levels within one
> contrast** — train congruency in the 25%-incongruent blocks, test in the 75%
> blocks — which makes LWPC a cross-condition generalization failure and *is* the
> adaptation question, stated multivariately. Designs, the mandatory
> within-block ceiling, the block-centering control, and the implementation gap
> (train and test come from different trials, so the fold splitter has to be
> replaced) are in the
> [concurrent-regulation plan](#concurrent-regulation-plan)
> §4; the troubleshooting protocol is in
> [`decoding.md` › Cross-decoding controls](decoding.md#cross-decoding-controls). The
> leave-one-block-out recommendation below applies to it unchanged, and matters
> more, not less. The A4 congruency ↔ switchType transfer stays optional and
> supplementary, labelled as the base-effect question this section says it is.

This is **A4** in the existing battery, and it is already specified in depth —
designs, the double-dipping guard, the within-block 2×2, temporal generalization
— in `stability_flexibility_battery.md` › Data flow walk-through §5, implemented in
`dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py`. Read those
first; this section only says what to strip out and one framing problem.

Only if reviewers expect MVPA. Keep it small: per-subject, one mean value per
electrode in the fixed window (no time samples as separate features), shrinkage
LDA (`solver='lsqr', shrinkage='auto'`, no PCA, no hyperparameter search),
**leave-one-block-out CV**, subject as the unit of inference, 1000+ permutations
within valid exchangeability strata.

Leave-one-block-out matters specifically: random trial splits within a block let
the classifier exploit block-level drift — the same tonic confound as §1.4.

**Caveat on framing.** Training i-vs-c and testing s-vs-r asks whether
*congruency* and *switch type* share a coding axis. That is **not** the
adaptation question — a region could code conflict and switching on a shared
axis while their block-level adaptations are independent. Cross-decoding
adaptation directly is awkward in principle, because an interaction is not a
trial-level class label; that awkwardness is itself a reason the
pattern-correlation approach is the better primary instrument. Label this
analysis honestly as the base-effect question, or omit it.

---

### Part 3 — What to retire

From the primary result:

- cross-subject synthetic pseudotrials and independent per-channel sampling
- PCA explained-variance threshold
- 250 ms sliding windows at 62.5 ms steps (75% overlap) as an inferential object
- block-context accuracy comparisons (`C/I (25% S) > C/I (75% S)` etc.)
- folds / repeats / bootstraps as inferential sample size
- the multiple electrode-selection variants
- two-stage percentile + cluster-percentile procedures
- accuracy plots carrying several different significance bars

Remaining major decisions, all statable in a short Methods paragraph:
anatomical ROI · time window · baseline definition · minimum trial/electrode
criterion · similarity metric · permutation scheme.

---

### Part 4 — Open questions to resolve first

1. **What does `ELECTRODES='sig'` actually mean?** [open] If significance was
   defined on the same trials, contrast, or window subsequently analysed, any
   downstream effect is partly circular. If it is a condition-independent
   responsiveness criterion, the problem is much smaller. Resolve before
   reporting anything conditioned on it. (§2.6 sidesteps this for the primary
   analysis by using anatomical electrodes.)
2. **Which subjects cap the trial counts, and how many electrodes do they
   contribute?** A subject contributing 1–2 of 174 electrodes while capping a
   cell at 11 is worth excluding; one contributing 8 is not. Needs a
   leave-one-subject-out sweep on the real per-subject counts (the electrode
   distribution in `sig_electrodes_per_subject_roi.json` is stale — 44
   electrodes / 12 subjects vs. the 174 actually used).
3. **Does the incongruentProportion decoding crossover survive per-trial
   baselining?** It is the one decoding result that ran against its trial-count
   confound, so it is worth one clean check before discarding the decoding
   results wholesale.

---

### Part 5 — Order of operations

Within the A1–A7 sequence of `stability_flexibility_battery.md` › Data flow walk-through §11 this is a
re-prioritisation, not a new pipeline: A2-continuous is promoted to primary, A4
demoted to optional confirmation.

1. ~~Make the §2.5 scatterplot~~ — **done, and it is the cheapest thing to
   run.** `SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh`
   writes `segregation_joint_scatter.png` plus its leverage diagnostics without
   running any inference. Look at it, and at the `flags` it prints, before
   spending a full run.
2. ~~Fix the §2.2 split aggregation~~ — **done.** Correlation is now computed
   per split and averaged.
3. ~~Add the §2.3 noise ceiling~~ — **done.** Reliabilities and the
   noise-corrected estimate come back in `correlation.json` and `summary.txt`.
4. **Check the 2×2 cross-tab in your own data** (§2.2, end). Per subject, the
   four congruency × switchType cell counts, tested for proportionality. This is
   a few lines and it tells you how much §2.2b was actually buying — cheap, and
   it should be in the Methods either way.
5. **Run** `run_joint_distribution_analysis(..., contrast_mode='proportion')` on
   anatomical lPFC electrodes. Read `reliability_x`/`reliability_y` *before*
   reading `corr`: if either is near zero, the correlation is uninterpretable
   whatever it says, and the electrode set or the effect measure is the problem
   to fix first. Also check `n_electrodes_dropped` (§2.3).
6. Re-run power traces with per-trial baseline as a robustness check (§2.4).
7. Add the time-resolved `S` curve if the timing claim is wanted (§2.7).
8. Only then, if desired, the §2.8 minimal cross-decoder.

Steps 1 and 4–5 are the paper. Everything after is support.

Before the categorical arm is used for anything load-bearing in `condition`
mode, resolve the §2.2b open item (`per_electrode_labels` still scores main
effects the old way).

---

### Part 6 — What not to claim

If pattern similarity or cross-decoding comes out null, the honest statement is:

> We found no evidence that the measured linear HG pattern was shared between
> stability and flexibility adaptation.

**Not:** "the mechanisms are independent." Independence requires positive
evidence — reliable within-domain patterns for *both* effects (this is what the
§2.3 noise ceiling supplies, now reported as `reliability_x`/`reliability_y`),
adequate measurement quality, and a confidence interval excluding a
theoretically meaningful shared-pattern effect. Two significant main effects
plus a non-significant correlation is not a dissociation.

One further caution now that §2.2b is known: a *negative* correlation is the
easiest result to over-read, because two of the three biases documented there
push in that direction. Before reading `corr < 0` as segregation, confirm it
survives on the cell-weighted contrast (it now does by default) and check the
cross-tab proportionality per §2.2 step 4.

The same caution applies to the existing power-trace double dissociation, which
currently rests on two non-significant interaction clusters (§2.4).

---

## Nested electrode selection plan

*Nested electrode selection — design plan*

How to make the drill-down **diagonal** (select on congruency → measure
congruency) reportable instead of circular, by estimating selection and effect on
disjoint trials. Covers both the decoding (nested inside the CV loop) and the
power traces (repeated split-half), plus how to reconcile them with the
full-trial selection used for the anatomy panel.

Companion to [`analysis_guide.md`](analysis_guide.md) §14.1 (the "ignore the
diagonal" rule), §17 (the decoding designs), and §21 (disjoint trial splits).
Figure context is in the [figure plan](#figure-plan), F3/F4.

### Which cells actually need a split

The rule is the same for decoding and for power traces, so apply one table, not
two procedures. A cell is circular only when the **selection contrast and the
measured contrast are the same**:

| Selection | Measured | Circular? | Why |
|---|---|---|---|
| task-responsiveness | congruency or switch main effect | **no** | responsiveness is the condition *mean*, the contrast is the *difference* — orthogonal |
| congruency main effect | congruency main effect | **yes** | selection contrast = measured contrast |
| congruency main effect | switch main effect | **no** | off-diagonal |
| congruency main effect | LWPC / LWPS interaction | **no** | Type III interaction row is orthogonal to both main effects (§14.1) |

Two consequences worth internalizing before writing any code:

- **F3 needs nothing.** Its electrodes are selected on task-responsiveness, which
  is orthogonal to every condition contrast plotted there.
- **Under the hierarchical design** (define groups on main effects, test
  adaptation within them — see the [figure plan](#figure-plan)), the **adaptation traces carry
  the paper's claim and are already clean.** Only main-effect-on-its-own-selection
  needs a split: roughly 2 cells out of 12.

The orthogonality claims rest on balanced cells, and the design is deliberately
75/25. Verify empirically with the permutation check in the [figure plan](#figure-plan) rather
than assuming it.

#### Where the bias sits

Selection bias is concentrated in the **selection window**. Selecting on a 0–1 s
window mean and plotting −1 to 1.5 s leaves the pre-stimulus portion only
indirectly inflated (through overall electrode responsiveness), while the 0–1 s
portion is directly inflated. This is why an uncorrected trace can look entirely
plausible and still be unusable for statistics — the distortion is local and does
not announce itself.

### Why this is correct, and why it is the *conservative* choice

Selection is part of the fitting procedure. If the entire procedure — selection
**and** classifier training — sees only the training trials, and the held-out
trials are used solely for scoring, the accuracy estimate is unbiased. Training
the classifier on the same trials used to select electrodes is fine; only the
*test* trials must be untouched.

**Nested feature selection is the standard correction, not a novel risk.** The
pattern reviewers flag is selection performed *outside* the CV loop, on all
trials — which is what the pipeline does today. One sentence in Methods settles
it:

> Electrode selection was nested within the cross-validation loop: for each fold,
> electrodes were selected using only that fold's training trials, and the
> held-out trials were used solely for evaluation.

Standard reference for the failure mode this fixes: Kriegeskorte et al. (2009),
circular analysis in systems neuroscience.

**5-fold already gives the 80% train fraction.** `StratifiedKFold(n_splits=5)`
trains on 80% and tests every trial exactly once. Do **not** reach for
`frac_train` / `StratifiedShuffleSplit` — those resamples are not a partition, so
trials are tested a variable number of times for no benefit here.

### The insertion point

`Decoder.cv_cm_jim_window_shuffle` in `src/analysis/decoding/decoder.py`
(fold loop at ~line 317). The loop already forms exactly what is needed:

```python
for f, (train_idx, test_idx) in enumerate(splitter.split(data, strat)):
    x_train = data[train_idx]
    y_train = labels[train_idx].copy()
    x_test  = data[test_idx]
    y_test  = (labels if labels_test is None else labels_test)[test_idx]

    if shuffle:
        rng.shuffle(y_train)

    cm_windowed = self._window_and_predict_minimal(...)
```

**Electrode selection is a channel-axis mask; the CV split is a trial-axis
partition.** They are orthogonal, so nothing about the fold structure changes.
Insert a mask computation after the train/test arrays exist and apply it to both.

Two facts make this much cheaper than a restructure:

**1. `stratify_labels` already carries the condition cell.** Per §17,
`build_cross_decoding_arrays` returns `strat` = the joint
`congruency × switchType × incongruent_proportion × switch_proportion` cell. So
`strat[train_idx]` *is* the selection ANOVA's design, already in scope. The only
new argument is the selection callback itself.

**2. Compute the ANOVA from `x_train` directly — not from A1's long table.**
`put_data_in_labeled_array_per_roi_subject` **randomizes trial ordering within
each subject** before NaN-padding and concatenating along channels. Decoding-array
rows therefore do **not** map back to the segregation module's table by index, and
trying to join them is where this task would turn expensive. Computing a
window-mean 2×2 ANOVA per channel straight off `x_train` avoids the mapping
entirely, and has the side benefit that selection and decoding share
preprocessing.

#### Proposed API

Keep the ANOVA logic where it lives and pass it in:

```python
def cv_cm_jim_window_shuffle(..., select_fn=None, select_window=None):
    """
    select_fn : callable or None
        ``select_fn(x_train, cells_train) -> bool ndarray (n_channels,)``
        Called once per fold on training trials only. ``None`` keeps the
        current behaviour (decode every channel passed in).
    select_window : (start, stop) sample indices, or None
        Time window the selection statistic is computed over. Fixed a priori;
        see "One selection per fold" below.
    """
```

Inside the loop:

```python
    if select_fn is not None:
        cells_train = strat[train_idx]
        mask = select_fn(x_train[..., sl], cells_train)   # sl = select_window
        if mask.sum() == 0:
            n_empty_folds += 1
            continue                     # record, don't silently average it in
        x_train = x_train[:, mask]
        x_test  = x_test[:, mask]
        fold_masks.append(mask)          # for the stability report
```

### Four design decisions that are not optional

#### 1. The null must run selection too

`shuffle=True` currently permutes `y_train` *after* the split. If `select_fn`
reads `cells_train`, permuting `y_train` leaves selection untouched — the null
holds selection fixed while the observed pipeline does not.

Point estimates stay unbiased either way. The problem is **variance**: the guide's
whole rationale for the refit-under-shuffle null is that it "carries the variance
of the entire estimation pipeline," and `time_perm_cluster` forms its cluster
threshold from that variance. A null missing the selection step is too tight, and
significance is inflated.

Fix: draw **one permutation of trial indices per fold and apply it to both**
`y_train` and `cells_train`. Because `y_train` is a component of the cell,
permuting them together keeps the two coherent while decoupling both from the
neural data:

```python
    if shuffle:
        perm = rng.permutation(len(y_train))
        y_train = y_train[perm]
        cells_train = cells_train[perm]      # select_fn sees the permuted cells
```

For the **off-diagonal** cells, selection is already orthogonal to the scored
contrast, so holding selection fixed and permuting only the decoded labels is a
valid and slightly more powerful null. Permuting both is conservative and uniform
across cells — prefer it unless power becomes limiting, and say which you used.

#### 2. Top-k, not a threshold

An FDR-thresholded mask yields a different channel count per fold — sometimes
zero. Consequences: PCA-at-X%-variance operates on a different feature space each
fold, and the null's dimensionality can differ systematically from the observed
(shuffled labels select fewer channels), which biases the comparison the null
exists to make.

**Select the top-k channels by F statistic instead.** Constant `k` across folds
*and* across the null removes both problems and makes the empty-mask case
impossible. Choose `k` from the full-data selection (e.g. the count at your
reported α) and state it as fixed a priori. Report the threshold-based version in
supplement for continuity with the classification panel.

#### 3. One selection per fold, on a fixed a priori window

Do **not** re-select per sliding time window. Per-window selection multiplies cost
by `n_windows` and makes the electrode set time-varying, so the accuracy trace no
longer describes a single population.

Select once per fold on a fixed window — use A1's post-stimulus window (0–1 s) so
the selection matches the taxonomy — and apply that mask across all time windows,
including pre-stimulus. Evaluating pre-stimulus accuracy on post-stimulus-selected
electrodes is still unbiased (selection used training trials only), and it is
*diagnostically useful*: it gives a clean read on §17's impossible pre-stimulus
congruency decoding without selection as a confound.

#### 4. NaN padding makes per-fold selection noisy, asymmetrically

Subjects are NaN-padded to the per-condition max trial count. A low-trial
subject's fold-training rows can be mostly padding, so its per-electrode ANOVA in
that fold rests on very few real trials — and this is **worse in the 25% cells**,
which have fewer real trials to begin with.

Top-k contains the downstream damage but not the underlying noise. Before
trusting any diagonal result:

- log the per-fold count of real (non-NaN) trials per channel per condition cell,
- set a minimum-real-trials floor per channel and drop channels below it *before*
  ranking,
- report the distribution.

### Power traces — repeated split-half, not cross-validation

The diagonal power traces need the same independence, but **not** the same
machinery. Nothing is being fitted, so there are no folds and no train/test
asymmetry — just repeated random splits with the held-out traces averaged.

```
for split in range(n_splits):
    sel_trials, trace_trials = stratified_half_split(trials, cells, rng)
    mask  = select_electrodes(x[sel_trials],   cells[sel_trials])
    trace = condition_effect(x[trace_trials, mask], cells[trace_trials])
    traces.append(trace)
grand = mean(traces)
```

Three differences from the decoding case:

**Use 50/50, not 80/20.** For decoding, more training data buys a better
classifier, so the (k−1)/k of 5-fold is right. For a trace there is no model —
selection only needs enough trials to rank electrodes stably, and the trace wants
as many trials as it can get. 50/50 balances those far better. Stratify the split
on the full condition cell so both halves keep the 75/25 structure.

**The null goes through the same procedure.** As with the decoding null: if the
observed trace is select-then-average-over-splits, the permutation null must be
too, or `time_perm_cluster` forms its threshold from variance the observed
statistic does not have. Permute the cells once per split and run the identical
select-then-average path.

**Check the rare cells survive halving first.** The 25%-incongruent-within-
25%-switch cell is the binding constraint. Count real (non-NaN) trials per cell
per subject before choosing the fraction; if halving empties that cell for several
subjects, either raise the selection fraction or restrict the split-based
treatment to the cells that can support it and mark the rest descriptive.

Splits overlap — each trial lands in many held-out halves — so the averaged trace
is unbiased but its across-split spread is **not** an independent-sample variance.
Do not build error bars from the split distribution; get inference from the
matched permutation null above.

### What changes in the reporting

With per-split selection there is no single "n=27 congruency-sensitive electrodes"
being measured — there is a distribution of per-split sets. This creates a real
inconsistency with the anatomy panel, which uses full-trial selection, and it has
to be addressed rather than hoped past.

**The inconsistency is principled.** The two panels are doing different jobs:

- The **anatomy/count panel** claims something about the *electrodes themselves* —
  how many are selective and where they sit. Full-trial selection is the *correct*
  estimator here, and it is not circular: the selection contrast is never
  re-tested, and the spatial distribution being tested is orthogonal to the
  selection statistic.
- The **decoding and trace panels** claim something about *effect magnitude within
  selected electrodes*. That is circular on the diagonal, hence the split.

Selection-for-description and selection-for-testing have genuinely different data
requirements. The problem is not using two procedures; it is showing two electrode
sets in one figure without saying so.

**The fix that does the most work: encode selection stability in the anatomy
panel.** Rather than binary membership, size or color each electrode by how often
it is selected across splits. This unifies the panels — the anatomy now displays
the same per-split selection the decoding consumes — adds information, and
answers the reviewer question before it is asked. It is nearly free once
`fold_masks` is stored.

It is also a result in its own right. If full-trial-selected electrodes are
recovered in ~90% of splits, the two panels describe the same population and the
mismatch is cosmetic. If it is ~40%, the taxonomy is much less stable than a
binary map implies, and you need to know that before writing conclusions from it.

The continuous effect-size correlation (§14) is the other bridge: it never
thresholds, so it has no set to mismatch. Reporting it alongside makes the
threshold-dependence of the taxonomy visible.

Concretely:

- **Classification / anatomy panel (F4a):** full-trial selection, electrodes
  encoded by selection frequency.
- **Trace and decoding panels (F4b/c):** per-split selection. Report median
  [min, max] set size, and state in the caption that selection differs from F4a
  and why.
- **Methods:** one sentence distinguishing the two uses. See the Methods sentence
  in the first section.

### Cost

Selection runs `n_repeats × n_splits × n_bootstraps × 2` (observed + null) times.
A window-mean 2×2 ANOVA per channel is cheap and fully vectorizable over channels
— budget it as negligible against the existing PCA→LDA fits.

**Use the window-mean ANOVA, not the per-timepoint cluster variant** (§14.2). The
cluster version inside a nested loop is not worth the compute, and the selection
statistic does not need to be the same one used for the taxonomy figure — it needs
to be cheap, fixed a priori, and computed on training trials.

### Acceptance tests

The first is the one that actually proves the change works; write it first.

1. **Planted selection bias returns to chance.** Generate data with *no* real
   condition effect. The current non-nested pipeline (select on all trials, then
   decode) must come out **above chance** — that is the bug being fixed. The
   nested pipeline must come out **at chance**. If the nested version is above
   chance on pure noise, the nesting is broken somewhere.
2. **Planted real effect is recovered.** Plant an effect on a known channel
   subset. Nested decoding must exceed chance, and `fold_masks` must overlap the
   planted set well above the rate expected by chance.
3. **Constant dimensionality.** With top-k, assert every fold's mask sums to `k`.
4. **Null matches the pipeline.** With `shuffle=True`, confirm `select_fn` is
   called with permuted cells — a spy/counter assertion is enough. Guards against
   silently regressing to a fixed-selection null.
5. **Off-diagonal unchanged.** Cross-decode results with `select_fn=None` must
   match the current implementation bit-for-bit. This change must not perturb the
   results that were already valid.

### Fallback if this is not built

The **off-diagonal is the load-bearing result and needs none of this** — selection
on congruency is already orthogonal to decoding switch type, so those cells are
valid today. The specificity claim ("process-specific electrodes fail to decode
the other process") is fully supported without any code change.

If the nested selection is skipped, report the diagonal explicitly as
selection-inflated and descriptive — label it in the panel, not just the caption —
and make the off-diagonal carry the inference. That is defensible; it just makes
F4 a weaker payoff figure, because the diagonal is the half a reader finds
intuitive.

**Recommendation: build it.** Roughly a day given that `stratify_labels` already
supplies the cells and the fold loop already forms `x_train`, and it converts the
intuitive half of the paper's punchline figure from "descriptive" to "reportable."

---

## Figure plan

*Figure plan — Intracranial EEG correlates of concurrent demands on stability and flexibility*

Working plan for the main-text figure sequence. Companion to
[`analysis_guide.md`](analysis_guide.md) §12 (the analysis-side figure sequence),
§14.1 (the four interaction groups), and §21 (disjoint trial splits).

### The narrative

> **Revised 2026-09** to match the
> [concurrent-regulation plan](#concurrent-regulation-plan).
> The earlier spine — characterize LPFC as a whole, then drill into
> process-specific vs. process-general subpopulations — is retired along with
> the independent-vs-dependent framing. What follows is the current sequence.
> The retired version is preserved below under "Retired: the subpopulation
> drill-down", because its two structural arguments (select on main effects, test
> interactions within them; rescue the diagonal with disjoint halves) are still
> correct and still apply if any subpopulation figure comes back.

**Concurrent regulation, then characterization.** Behavior shows stability and
flexibility being regulated at the same time in the same subjects → lPFC high
gamma carries both adaptation effects → decoding shows distributed lPFC activity
carries information about each adaptation → anatomy asks whether the two effects
are organized differently across cortex.

**Organize figures by claim, not by measure.** Power and decoding for the same
claim belong in the same figure.

### The claim stack

| # | Claim | Carried by |
|---|---|---|
| C1 | Stability and flexibility are regulated **concurrently in behavior** | F1 |
| C2 | lPFC high gamma carries **both adaptation effects**, in the expected directions | F3 |
| C3 | **Distributed lPFC activity carries decodable information about each adaptation**, including information no single electrode supplies | F4 |
| C4 | The two adaptation effects are **(not) organized differently across lPFC**, conditioned on coverage and read against a noise ceiling | F5 |

**"Independent" is a behavioral word in this paper — and now it is barely used at
all.** C1 is a concurrency claim, not an independence claim. For the neural
results, describe what was measured: adaptation effects, their directions, their
decodability, their spatial organization. The **absent cross-effects**
(congruency × switch proportion, switchType × incongruent proportion) are a
*scoping* statement in the text — "we therefore focus on the two within-process
adaptation effects" — and do not get a figure or a dissociation claim.

### Retired: the subpopulation drill-down

*Kept for the reasoning, not as the plan. The two decisions below govern any
figure that defines electrode groups and then tests something within them.*

#### Two structural decisions

##### Define groups on main effects, test adaptation within them

The torn-ness between main-effect and interaction electrodes resolves
hierarchically, and the resolution is better than either option alone:

- **Main effects define the subpopulations.** Congruency-sensitive,
  switch-sensitive, both. Well-powered — this is where the electrode counts are.
- **Interactions are tested *within* those groups.** "Do the congruency-sensitive
  electrodes show LWPC? Do the switch-sensitive ones show LWPS?"

**This is non-circular by construction.** Under sum coding, main-effect and
interaction contrasts are orthogonal, and §14.1 already uses Type III SS for
exactly this reason — the interaction row is orthogonal to both main effects. So
selecting on a main effect and testing the interaction does not double-dip.

*Caveat, and it needs checking:* the cells are deliberately unbalanced (75/25),
so the orthogonality is approximate rather than exact. Verify empirically before
relying on it — permute labels, run the full select-on-main-effect →
test-interaction pipeline, and confirm the false-positive rate is nominal. Cheap
to run, and it converts an assumption into a reported control.

The payoff: this keeps the **adaptation** framing (which is the novel claim)
while selecting on the **main effects** (which is where the power is). Report the
interaction-defined counts in the supplement as convergent evidence — with the
threshold sweep and the continuous effect-size correlation (§14), which is the
real answer to low counts. The counting analysis is what's underpowered; the
correlation is not, because it never thresholds.

##### The drill-down's diagonal is circular — fix it with disjoint halves

The expected result as stated — *congruency electrodes decode congruency but not
switch type; switch electrodes the reverse; both electrodes decode both* — is half
guaranteed and half a real test:

| Cell | Status |
|---|---|
| congruency electrodes → decode congruency | **circular** (selection contrast = decode contrast) |
| congruency electrodes → decode switch type | **real test** — this is the specificity claim |
| switch electrodes → decode switch type | **circular** |
| switch electrodes → decode congruency | **real test** |
| both electrodes → decode both | **circular on both** |

This is §14.1's "ignore the diagonal" rule. The load-bearing result is the
**off-diagonal**: process-specific electrodes *fail* to decode the other process.

But don't just drop the diagonal — the diagonal is the intuitive half of the
story and a reader will want it. **Rescue it with disjoint trial halves** (§21,
`_stratified_half_split`): select electrodes on half the trials, decode on the
other half. The diagonal then becomes legitimate and the full 3×2 reads cleanly.
Cross-validation alone does *not* fix this — selection happened before the CV
split, on every trial.

### Main-text sequence

#### F1 — Task, manipulation, behavior *(C1)*
`a` paradigm · `b` 2×2 block proportion manipulation · `c` RT · `d` error rate.

Unchanged, except in emphasis: the point is that **both adaptations are present
in the same subjects and the same sessions** — concurrent regulation. The absent
behavioral cross-effects belong in the text as scope (and in S4), not as a
visual centrepiece.

#### F2 — Coverage and signal validation
`a` all electrodes on the MNI surface, colored by ROI · `b` per-electrode HG
traces for one example subject, task-responsive electrodes outlined · `c` example
spectrogram.

Add a per-ROI, per-subject coverage table to the supplement and cite it here
(see "Anticipated reviewer objections" below).

#### F3 — Adaptation effects in lPFC high gamma *(C2)*
`a` task-responsive lPFC electrodes on the surface, with counts · `b` HG traces:
LWPC — congruency effect in 25% vs. 75% incongruent blocks · `c` HG traces:
LWPS — switch cost in 25% vs. 75% switch blocks · `d` the two simple effects per
subject with their difference (the **direction tests**, plan §2).

Panel `d` is what makes this figure a claim rather than a display: it reports the
*sign* of each adaptation, subject by subject, against the behavioral direction.
An interaction cluster without a direction is not a regulation result.

Main effects (incongruent vs. congruent, switch vs. repeat) move to the
supplement unless space allows a row — they are context, and the paper is about
the adaptation effects now.

#### F4 — Decoding the two adaptations *(C3)*
`a` LWPC decoding · `b` LWPS decoding, both across anatomically-defined lPFC
electrodes, each against its refit shuffle null and with n per class printed ·
`c` block-transfer: within-block accuracy beside 25% ↔ 75% transfer, for both
primary effects — congruency across incongruent proportion (X1) and switch type
across switch proportion (X2) — plus their reciprocal controls, congruency
across switch proportion (X3) and switch type across incongruent proportion
(X2b).

The caption's job is the claim in C3: adding electrodes yields information
individual electrodes do not carry. Not "multivariate beats univariate" — the
pseudopopulation cannot support that (simplification plan §1.1).

Panel `c` only appears if the within-block ceiling clears chance
([`decoding.md` › Cross-decoding controls](decoding.md#cross-decoding-controls) §2). If it does not,
drop the panel rather than showing a null with no ceiling.

**Layout is what controls the bloat here, not panel count.** A trellis with
shared axes, one row label, one column label, and no per-cell legends or titles
reads as *one panel*. The same plots given individual titles, axes, and legends
read as six subpanels and look like bloat. Small multiples are cheap;
independently-decorated subpanels are expensive.

#### F5 — Anatomy of the two effects *(C4)*
`a` per-electrode signed LWPC and LWPS scores on the MNI surface · `b` the
relative map (`lwpc_s − lwps_s`) · `c` the ROI × effect-type interaction test,
coverage-conditioned, with the within-electrode swap null · `d` split-half
spatial reliability beside the between-effect similarity — **the noise ceiling,
which is what makes `c` readable in either direction**.

Haufe-transformed decoder patterns go in the supplement as convergent evidence,
labelled as such (plan §8.3: PCA blurs the back-projection, so it is weaker
evidence about anatomy than the per-electrode maps).

#### Timing — fold in or drop
`a` LWPC vs. LWPS interaction onsets, each normalized to its own peak (the
latency–amplitude guard, §12.1 principle 6) · `b` jackknife onset difference with
the Ulrich–Miller corrected test, overlaid on the permutation null.

Under the current narrative this is a row in F3, not a figure — and only if the
ordering is significant. A null folds it away entirely.

### Anticipated reviewer objections

#### "Why only LPFC?"

Coverage genuinely does not support more, but **show it, don't hand-wave it.**
From `sig_electrodes_per_subject_roi.json` (an older run — the relative picture
holds, the absolute counts are stale):

| ROI | sig. electrodes | subjects with ≥1 |
|---|---|---|
| lpfc | 44 | 12/17 |
| dlpfc | 25 | 8/17 |
| occ | 18 | 5/17 |
| acc | 8 | 4/17 |
| v1 | 6 | 3/17 |
| parietal | 5 | 3/17 |

That is a defensible answer *as a table*. State the minimum coverage you required
and show the ROIs that failed it. Reviewers accept coverage limits; they do not
accept unexamined ones.

**Better: turn it into a specificity control.** If any control ROI clears your
threshold, run the same partition there. "The partition is LPFC-specific, not a
global property of task-responsive cortex" converts your weakest point into a
result. Occipital is the natural choice — decent counts, and no one expects
control-signal structure in visual cortex, so a null there is exactly what you
want. ACC would be the more interesting positive control but is likely too thin.

#### "Why only high gamma?"

Your suspicion that the low bands are a preprocessing artifact is probably
right, and there are two specific mechanisms in the current pipeline. Both are
worth resolving *before* deciding what the low-band supplement says, because
right now you cannot distinguish "no low-frequency effect" from "the pipeline
removed it."

**1. The baseline is too short for low frequencies.**
`make_epoched_data.py` uses `base_times_length=0.5` — a 0.5 s baseline. That is
35–75 cycles at 70–150 Hz, and **2–4 cycles at 4–8 Hz**. Z-scoring against a
two-cycle baseline puts enormous variance in the denominator for theta, which
would flatten exactly the effects you are looking for while leaving HG untouched.
This is arithmetic, not speculation. Fix: use a longer baseline for the low bands
(≥1 s, ideally scaled to cycles rather than fixed seconds).

**2. The baseline may be subtracting the signal itself.** `within_base_times=(-1, 0)`
draws the baseline from the pre-stimulus period. Your own §12.1 principle 7 notes
that list-wide manipulations induce a *sustained block-level state present before
stimulus onset* — and sustained state is, by definition, low-frequency. So for
theta/alpha/beta the baseline is not neutral: it plausibly contains the effect,
and normalizing against it removes it. This bites the low bands far harder than
HG, and the guide flags the mechanism for HG without noting that it is worse
downstream. Fix: baseline against `experimentStart` (the code already supports
`baseline_event="experimentStart"`), which predates the block context.

Re-run one low band with both fixes. Then:

- **Still null** → report it in the supplement with the fixed pipeline. A clean
  null in theta costs you nothing, and "we checked, with an appropriate baseline"
  is a complete answer. HG being the informative band is the expected result and
  is well-precedented.
- **Not null** → you have a new result, and you would have shipped without it.

Either way you are answering from evidence rather than hand-waving, which is the
entire point. Do not put the *current* low-band results in the supplement — a
reviewer who spots the 0.5 s baseline will discount the whole supplement.

### Compression points

Five main figures (F1–F5). To adjust:

- **→ 4:** fold F5's panels `a`/`b` (the score maps) into F4 and move the
  interaction test to the supplement — only if the anatomy result is null *and*
  the ceiling says the null is uninterpretable.
- **→ 4:** drop F4`c` if the within-block ceiling does not clear chance.
- **→ 6:** split F5 into maps and inference, if `a`/`b` crowd `c`/`d`.

### Supplement

| S | Content |
|---|---|
| S1 | Per-ROI, per-subject coverage table with the inclusion threshold |
| S2 | Main effects (congruency, switch type) in lPFC HG; continuous LWPC/LWPS effect-size correlation and its leverage diagnostics |
| S3 | Low-frequency bands, re-run with the fixed baseline |
| S4 | Absent cross-effects (congruency × switch proportion, switchType × incongruent proportion) — reported as scope, not as a dissociation |
| S5 | A4 label transfer (congruency ↔ switchType), labelled as the base-effect geometry question, with the pre-stimulus caveat stated |
| S6 | Haufe-transformed decoder patterns and their spatial comparison with the univariate maps |
| S7 | Per-subject HG traces; demographics, electrode counts, exclusions |
| S8 | Cross-decoding control table ([`decoding.md` › Cross-decoding controls](decoding.md#cross-decoding-controls) §7) for every transfer reported |
| S9 | Descriptive within-subject centroids/medoids per hemisphere, with the within-electrode swap null |
| S10 | Per-trial-baseline robustness re-run of the power traces; direct block comparisons |

### Open items before this plan freezes

1. Direction tests on both adaptation effects (plan §2) — F3`d` stands on them.
2. Joint-cell trial counts (plan §4.4) — F4`c` stands or falls on the within-block
   ceiling.
3. Implement the block-transfer splitter (plan §4.2) and its synthetic test.
4. Implement the continuous-score anatomy arm (plan §5.3) and report the spatial
   noise ceiling (plan §5.4) — F5`d`.
5. Re-run one low band with a longer, pre-block baseline before deciding what S3
   says.
6. Check whether any control ROI clears threshold for the specificity analysis.
