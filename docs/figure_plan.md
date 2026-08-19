# Figure plan — Intracranial EEG correlates of concurrent demands on stability and flexibility

Working plan for the main-text figure sequence. Companion to
[`analysis_guide.md`](analysis_guide.md) §12 (the analysis-side figure sequence),
§14.1 (the four interaction groups), and §21 (disjoint trial splits).

## The narrative

**Zoom-in.** Characterize LPFC as a whole → show it carries both signals (mixed
selectivity) → drill into subpopulations → show the mixture is driven by the
`both` electrodes.

This is the right spine, and it beats a partition-first ordering for two reasons:
it front-loads a result the reader can evaluate before being asked to accept a
selection procedure, and it makes the all-LPFC level do real inferential work
rather than serve as description.

**Organize figures by level, not by measure.** Put power *and* decoding together
at each level (F3 = all of LPFC, F4 = subpopulations) rather than splitting into a
power figure and a decoding figure. The claim is about levels, so the figures
should be too — and it collapses the drill-down into one figure instead of three.

## The claim stack

| # | Claim | Level | Carried by |
|---|---|---|---|
| C1 | Stability and flexibility are adapted **independently in behavior** | behavior | F1 |
| C2 | LPFC as a whole carries **both** conflict and switching signals — mixed selectivity | population | F3 |
| C3 | LPFC contains **process-specific** and **process-general** sites, and the population-level mixture is **driven by the process-general sites** | subpopulation | F4 |
| C4 | One adaptation arises **earlier** in the trial than the other | timing | F5 |

**"Independent" is a behavioral word in this paper.** C1 earns it; nothing neural
does. For the neural results use **process-specific** / **process-general**. This
keeps C3 decoupled from the cross-process effects: a site carrying
`congruency × switch_proportion` is a *different interaction* from
`congruency × incongruent_proportion`, so its existence is not evidence against
"some sites are congruency-only."

## Two structural decisions

### Define groups on main effects, test adaptation within them

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

### The drill-down's diagonal is circular — fix it with disjoint halves

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

Unchanged. Make the **absent behavioral cross-effects** visually obvious — it is
the contrast against which any neural cross-effect is read.

### F2 — Coverage and signal validation
`a` all electrodes on the MNI surface, colored by ROI · `b` per-electrode HG
traces for one example subject, task-responsive electrodes outlined · `c` example
spectrogram.

Add a per-ROI, per-subject coverage table to the supplement and cite it here
(see "Anticipated reviewer objections" below).

### F3 — LPFC as a whole *(C2)*
`a` task-responsive LPFC electrodes on the surface, with counts · `b` HG power
traces: main effects (switch vs. repeat, incongruent vs. congruent) · `c` HG
traces: within-process adaptation (LWPC, LWPS) · `d` decoding of congruency and
switch type across all task-responsive LPFC electrodes.

The point of this figure is **mixed selectivity at the population level**: LPFC
carries both signals, and nothing here tells you whether that is one mixed
population or two overlaid specific ones. That question is what F4 answers, so
state it explicitly in the last line of the caption — it is the hinge of the paper.

This level is also the **reference group** for everything in F4 (§17.1's
`REFERENCE_GROUP`, default `all`). Every subpopulation in F4 was *chosen* for
carrying an effect, so none of them is a baseline for "does LPFC decode this at
all." F3 is that baseline. Framing it this way is what makes it earn a figure
rather than read as throat-clearing — and it is why you should keep it even
though the narrative could technically skip straight to subpopulations.

### F4 — Subpopulations *(C3 — the payoff figure)*
`a` classification of task-responsive LPFC electrodes: congruency-sensitive,
switch-sensitive, both — counts, conjunction vs. chance (CMH odds ratio), and
anatomical distribution with the coverage-conditioned test · `b` HG power traces
per group, **trellis layout** · `c` decoding, trellis: rows = electrode group
(congruency / switch / both), columns = decode target (congruency / switch type),
diagonal estimated on held-out trials per §21.

**Layout is what controls the bloat here, not panel count.** A 3×2 trellis with
shared axes, one row label, one column label, and no per-cell legends or titles
reads as *one panel*. The same six plots given individual titles, axes, and
legends read as six subpanels and look like bloat. Small multiples are cheap;
independently-decorated subpanels are expensive. This is a design decision, not
an analysis decision, and it is the difference between an 18-panel figure and a
3-panel figure showing identical data.

If space is still tight, replace the trellis cells with scalars — late-window
accuracy with bootstrap CIs — and move the traces to supplement.

### F5 — Timing *(C4)*
`a` congruency vs. switch main-effect onsets · `b` LWPC vs. LWPS interaction
onsets, each normalized to its own peak (the latency–amplitude guard, §12.1
principle 6) · `c` jackknife onset difference with the Ulrich–Miller corrected
test, overlaid on the permutation null.

Restrict to the two within-process comparisons. If the ordering comes back null,
fold this into F3/F4 as a row and the paper goes to four figures.

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

Five main figures. To adjust:

- **→ 4:** fold F5 into F4 as a row (do this automatically if the onset ordering
  is null).
- **→ 6:** split F4 into partition and drill-down, if `a` crowds `b`/`c`.
- **→ 7:** promote cross-process effects to their own figure, but only if they
  survive the §12.1 principle-8 controls (trial-count matching, RT
  matching/regression, per-condition mean removal, run-aware folds).

## Supplement

| S | Content |
|---|---|
| S1 | Per-ROI, per-subject coverage table with the inclusion threshold |
| S2 | Interaction-defined electrode counts, threshold sweep, continuous LWPC/LWPS effect-size correlation |
| S3 | Low-frequency bands, re-run with the fixed baseline |
| S4 | Cross-process decoding cells and their confound controls |
| S5 | Cross-decoding (A4 label transfer), with the pre-stimulus caveat stated |
| S6 | Temporal generalization matrices |
| S7 | Per-subject HG traces; demographics, electrode counts, exclusions |
| S8 | Permutation check that main-effect selection does not inflate the interaction test |

## Open items before this plan freezes

1. Run the permutation check on main-effect-selection → interaction-test
   orthogonality. The hierarchical design rests on it.
2. Implement the disjoint-half split (§21) so F4's diagonal is reportable.
3. Re-run one low band with a longer, pre-block baseline before deciding what S3
   says.
4. Check whether any control ROI clears threshold for the specificity analysis.
5. Confirm the A5 onset ordering is significant — F5 stands or folds on it.
