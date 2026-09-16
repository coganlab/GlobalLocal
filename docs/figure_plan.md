# Figure plan — Intracranial EEG correlates of concurrent demands on stability and flexibility

Working plan for the main-text figure sequence. Companion to
[`analysis_guide.md`](analysis_guide.md) §12 (the analysis-side figure sequence),
§14.1 (the four interaction groups), and §21 (disjoint trial splits).

## The narrative

> **Revised 2026-09** to match
> [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md).
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

## The claim stack

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

## Retired: the subpopulation drill-down

*Kept for the reasoning, not as the plan. The two decisions below govern any
figure that defines electrode groups and then tests something within them.*

### Two structural decisions

#### Define groups on main effects, test adaptation within them

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

#### The drill-down's diagonal is circular — fix it with disjoint halves

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

## Main-text sequence

### F1 — Task, manipulation, behavior *(C1)*
`a` paradigm · `b` 2×2 block proportion manipulation · `c` RT · `d` error rate.

Unchanged, except in emphasis: the point is that **both adaptations are present
in the same subjects and the same sessions** — concurrent regulation. The absent
behavioral cross-effects belong in the text as scope (and in S4), not as a
visual centrepiece.

### F2 — Coverage and signal validation
`a` all electrodes on the MNI surface, colored by ROI · `b` per-electrode HG
traces for one example subject, task-responsive electrodes outlined · `c` example
spectrogram.

Add a per-ROI, per-subject coverage table to the supplement and cite it here
(see "Anticipated reviewer objections" below).

### F3 — Adaptation effects in lPFC high gamma *(C2)*
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

### F4 — Decoding the two adaptations *(C3)*
`a` LWPC decoding · `b` LWPS decoding, both across anatomically-defined lPFC
electrodes, each against its refit shuffle null and with n per class printed ·
`c` block-transfer: within-block accuracy beside 25% ↔ 75% transfer, for
congruency across incongruent proportion (X1) and the positive control,
congruency across switch proportion (X3).

The caption's job is the claim in C3: adding electrodes yields information
individual electrodes do not carry. Not "multivariate beats univariate" — the
pseudopopulation cannot support that (simplification plan §1.1).

Panel `c` only appears if the within-block ceiling clears chance
([`cross_decoding_controls.md`](cross_decoding_controls.md) §2). If it does not,
drop the panel rather than showing a null with no ceiling.

**Layout is what controls the bloat here, not panel count.** A trellis with
shared axes, one row label, one column label, and no per-cell legends or titles
reads as *one panel*. The same plots given individual titles, axes, and legends
read as six subpanels and look like bloat. Small multiples are cheap;
independently-decorated subpanels are expensive.

### F5 — Anatomy of the two effects *(C4)*
`a` per-electrode signed LWPC and LWPS scores on the MNI surface · `b` the
relative map (`lwpc_s − lwps_s`) · `c` the ROI × effect-type interaction test,
coverage-conditioned, with the within-electrode swap null · `d` split-half
spatial reliability beside the between-effect similarity — **the noise ceiling,
which is what makes `c` readable in either direction**.

Haufe-transformed decoder patterns go in the supplement as convergent evidence,
labelled as such (plan §8.3: PCA blurs the back-projection, so it is weaker
evidence about anatomy than the per-electrode maps).

### Timing — fold in or drop
`a` LWPC vs. LWPS interaction onsets, each normalized to its own peak (the
latency–amplitude guard, §12.1 principle 6) · `b` jackknife onset difference with
the Ulrich–Miller corrected test, overlaid on the permutation null.

Under the current narrative this is a row in F3, not a figure — and only if the
ordering is significant. A null folds it away entirely.

## Anticipated reviewer objections

### "Why only LPFC?"

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

### "Why only high gamma?"

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

## Compression points

Five main figures (F1–F5). To adjust:

- **→ 4:** fold F5's panels `a`/`b` (the score maps) into F4 and move the
  interaction test to the supplement — only if the anatomy result is null *and*
  the ceiling says the null is uninterpretable.
- **→ 4:** drop F4`c` if the within-block ceiling does not clear chance.
- **→ 6:** split F5 into maps and inference, if `a`/`b` crowd `c`/`d`.

## Supplement

| S | Content |
|---|---|
| S1 | Per-ROI, per-subject coverage table with the inclusion threshold |
| S2 | Main effects (congruency, switch type) in lPFC HG; continuous LWPC/LWPS effect-size correlation and its leverage diagnostics |
| S3 | Low-frequency bands, re-run with the fixed baseline |
| S4 | Absent cross-effects (congruency × switch proportion, switchType × incongruent proportion) — reported as scope, not as a dissociation |
| S5 | A4 label transfer (congruency ↔ switchType), labelled as the base-effect geometry question, with the pre-stimulus caveat stated |
| S6 | Haufe-transformed decoder patterns and their spatial comparison with the univariate maps |
| S7 | Per-subject HG traces; demographics, electrode counts, exclusions |
| S8 | Cross-decoding control table ([`cross_decoding_controls.md`](cross_decoding_controls.md) §7) for every transfer reported |
| S9 | Descriptive within-subject centroids/medoids per hemisphere, with the within-electrode swap null |
| S10 | Per-trial-baseline robustness re-run of the power traces; direct block comparisons |

## Open items before this plan freezes

1. Direction tests on both adaptation effects (plan §2) — F3`d` stands on them.
2. Joint-cell trial counts (plan §4.4) — F4`c` stands or falls on the within-block
   ceiling.
3. Implement the block-transfer splitter (plan §4.2) and its synthetic test.
4. Implement the continuous-score anatomy arm (plan §5.3) and report the spatial
   noise ceiling (plan §5.4) — F5`d`.
5. Re-run one low band with a longer, pre-block baseline before deciding what S3
   says.
6. Check whether any control ROI clears threshold for the specificity analysis.
