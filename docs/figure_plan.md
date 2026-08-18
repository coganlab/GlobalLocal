# Figure plan — Intracranial EEG correlates of concurrent demands on stability and flexibility

Working plan for the main-text figure sequence. Companion to
[`analysis_guide.md`](analysis_guide.md) §12 (the analysis-side figure sequence) and
§14.1 (the four interaction groups). Where the two disagree, this file is the
paper's plan and `analysis_guide.md` is the analysis battery's plan; the mapping
between them is in the table at the bottom.

## The claim stack

The paper makes four claims, in this order. Every main-text panel exists to
support one of them; anything that supports none is supplementary.

| # | Claim | Level | Carried by |
|---|---|---|---|
| C1 | Stability and flexibility are adapted **independently in behavior** — a double dissociation of LWPC and LWPS, with null cross-effects | behavior | F1 |
| C2 | LPFC contains **process-specific** sites (CPC-only, SPS-only) **and process-general** sites (both) | electrodes | F3 |
| C3 | Those populations show the expected HG dynamics, confirmed **non-circularly** by cross-contrast definition | univariate | F4 |
| C4 | The population readout reproduces the partition, and **additionally shows neural cross-process effects that behavior does not** | multivariate | F5 |
| C5 | One adaptation arises **earlier** in the trial than the other | timing | F6 |

### Vocabulary discipline

**"Independent" is a behavioral word in this paper.** C1 earns it; nothing
neural does. For the neural results use **process-specific** and
**process-general** populations. This matters because it decouples C2 from the
cross-process effects: a site carrying `congruency × switch_proportion` (CPS) is
a *different interaction* from `congruency × incongruent_proportion` (CPC), so
its existence is not evidence against "some sites are CPC-only." C2 is a claim
about the partition, and the partition holds whatever CPS/SPC turn out to be.

Consequence: the cross-process effects are **a designated result with a stated
caveat**, not a threat to the headline. They get labeled cells in F5 and a
control panel, and the text says plainly that behavior shows no cross-effects
while the neural readout does — flagged as pending the §12.1 principle-8
controls until those are run.

## The panel budget rule

> A panel earns main-text space if **flipping its result would change a sentence
> in the abstract.** Otherwise it is a scalar in a summary panel, a supplementary
> figure, or cut.

The decoding bloat (~45 panels under the original plan) comes from budgeting
every decode cell as a time-resolved accuracy trace. A trace is the right unit
only when the *shape over time* is the claim. When the claim is "this cell is
above chance and larger than that one," the unit is a **scalar with a bootstrap
CI**, and a dozen of them fit in one panel.

Four collapse rules, applied below:

1. **One panel per comparison, not per condition.** The 25% and 75% block levels
   of a decode cell are overlaid on one axis, not split across two panels. The
   pooled ("main effect") decode is a light reference trace *inside* that panel,
   not two panels of its own. → 6 panels become 4.
2. **The electrode-group breakdown is a summary panel, not a repeated grid.** Do
   not re-render the trace grid per group. Plot a single `group × decode-cell`
   panel of late-window cluster-corrected accuracy (or cluster mass) with
   bootstrap CIs. One panel replaces 12–16 traces, and it makes the group
   comparison directly readable instead of requiring the reader to eyeball across
   a grid. Full traces → supplement. **This deletes the original Figure 7.**
3. **Temporal generalization is a claim, not a display.** It answers exactly one
   question: sustained vs. transient code. Two matrices answer it. If more are
   wanted, reduce each matrix to a scalar generalization index (mean off-diagonal
   / mean diagonal, or diagonal width at half-max) and plot the indices as a
   strip. Note `TEMPGEN_GROUPS` already defaults to `both` (§17.1), so this is the
   intended scope anyway.
4. **Cross-decoding (A4 label transfer) comes out of the main text.** It adds no
   claim the partition doesn't already make, and it is the one analysis whose
   baseline is diagnostically impossible — significant pre-stimulus *congruency*
   decoding (§17's observed-status caveat). Leading with it invites a reviewer to
   discredit the whole decoding section. Supplement with the caveat, or hold for
   a follow-up once the principle-8 controls are in.

Net: **~45 decoding panels → 7.**

## Main-text sequence

### F1 — Task, manipulation, behavior *(C1)*
`a` paradigm · `b` 2×2 block proportion manipulation · `c` RT: switch cost and
congruency effect by both proportions · `d` error rate, same layout.

Unchanged from the draft. The double dissociation in `c` is the premise the rest
of the paper answers; make the **absent cross-effects** visually obvious here,
because F5's off-diagonal is the payoff.

### F2 — Coverage and signal validation
`a` all electrodes on the MNI surface, colored by ROI · `b` per-electrode HG
traces for one example subject, task-responsive electrodes outlined · `c` example
spectrogram showing the post-stimulus HG increase.

Unchanged. Keep it lean — this figure convinces the reader the recording and
pipeline work, and nothing more. Per-subject versions → supplement.

### F3 — The electrode partition *(C2 — the headline figure)*
`a` counts for all four interaction groups (CPC/SPS/CPS/SPC) with the conjunction
test: observed CPC∩SPS overlap vs. chance, with the CMH odds ratio · `b` the
threshold sweep (counts and overlap as a function of α), per §12.1 principle 3,
so the result is not one α snapshot · `c` continuous, threshold-free version: per
electrode LWPC effect size vs. LWPS effect size, with the correlation · `d`
anatomical distribution of the groups on the surface, with the
coverage-conditioned test (A3).

This is the figure the abstract's main sentence points at. Report the four
groups, not two — that is what makes the cross-process effects a described
feature of the taxonomy rather than a hole in it. Also report the **signed
direction** breakdown per group (`<g>_sign`, §14.1): "of N LWPC electrodes, k
showed a larger and N−k a smaller congruency effect in mostly-incongruent
blocks." Selection is two-sided by design, so the sign split is a result worth
stating, not a filter.

### F4 — HG dynamics in the partitioned populations *(C3)*
`a` main effects on task-responsive LPFC electrodes: switch vs. repeat,
incongruent vs. congruent *(non-circular — selection is on baseline
responsiveness, orthogonal to all four interactions)* · `b` within-process
adaptation: LWPS and LWPC traces · `c` **the orthogonal/cross-contrast panel** —
define on CPC, plot the LWPS trace; define on SPS, plot the LWPC trace · `d`
pre-stimulus / tonic effects, plotted explicitly rather than baselined away.

Two notes.

**The draft's Figure 5 is circular as written.** "HG power in LPFC electrodes
sensitive to main and adaptation effects" defines electrodes on an effect and
then plots that effect (§12.1 principle 1). Two fixes, take either: estimate
selection and trace on disjoint trial halves (§21), or replace it with panel `c`
above. Panel `c` is the better paper: non-circular, a stronger claim, and 2
panels instead of 6. The draft Figure 4 and Figure 5 merge into this one figure.

**Panel `d` is a result, not cleanup** (§12.1 principle 7). List-wide
manipulations induce a sustained block-level state present before stimulus onset.
Report it, use a baseline that predates the block context, and separate tonic
from phasic. Doing this in the main text also pre-empts the obvious reviewer
question about F5's pre-stimulus decoding.

### F5 — Population readout *(C4)*
`a–d` the within-block decoding 2×2, one panel per cell, 25% and 75% block levels
overlaid within each, pooled decode as a gray reference trace. Lay the panels out
as an actual 2×2 grid — **diagonal = matched (CPC, SPS), off-diagonal = cross
(CPS, SPC)** — with the axes labeled so the geometry is self-evident. A reader
who saw F1 immediately sees that behavior has no off-diagonal and the neural
readout does.

`e` the electrode-group summary: late-window accuracy per `group × decode cell`,
bootstrap CIs, circular diagonal cells omitted per §14.1's rule (each defined
group yields three usable cells and one ignored). This single panel is the
entirety of the original Figure 7.

`f–g` temporal generalization for the two matched decodes on the `both` group.
Keep these only if the sustained-vs-transient distinction is doing work in the
proactive/reactive discussion — it currently is, so keep them.

`h` **control panel**: block-identity decoding, and the count-matched /
mean-removed version of the cross cells. Whether the off-diagonal survives this
is what decides whether the text calls the cross-effects a finding or a caveat.
If the panel is too crowded, this moves to supplement and the main text cites it
— but it must exist somewhere before the cross cells are interpreted.

### F6 — Timing *(C5)*
`a` LWPC and LWPS interaction time courses, each normalized to its own peak (the
latency–amplitude guard, §12.1 principle 6) · `b` jackknife onset difference with
the Ulrich–Miller corrected test · `c` observed onset difference overlaid on the
permutation null.

You flagged timing as one of the paper's four claims, so it gets a real figure
rather than being appended to the power traces. **Contingency:** if the ordering
comes back null, demote this to a row of F4 and the paper goes to five figures.
Restrict the comparison to LWPC vs. LWPS — cross-process timing is not a claim
anyone is making.

### F7 — Brain–behavior *(optional, A6)*
If A6 lands, this is the strongest available demonstration that the partition is
*functional*: across-subject LWPC/LWPS neural summary vs. behavioral magnitude
with its cross-pairing control, plus the within-subject trialwise mixed model.
Add it as F7 or fold the trialwise result into F3 as a fifth panel. Not currently
in hand.

## Supplement

| S | Content | Displaced from |
|---|---|---|
| S1 | Full per-electrode-group decoding trace grids | original Fig 7 |
| S2 | All remaining temporal-generalization matrices | original Fig 6/7 |
| S3 | Cross-decoding (A4 label transfer) by group, with the pre-stimulus caveat stated | original Fig 6 |
| S4 | Decoding confound controls: trial-count matching, RT matching/regression, per-condition mean removal, run-aware folds | §12.1 principle 8 |
| S5 | Per-subject HG traces and electrode tables | Fig 2 |
| S6 | Low-frequency (theta/alpha/beta) replications of the conjunction and decoding | §12 frequency scope |
| S7 | Subject demographics, electrode counts, exclusions | Methods |

## Budget and compression points

Six main figures (seven with A6). To compress or expand:

- **→ 5 figures:** merge F6 into F4 as a bottom row (do this automatically if the
  onset comparison is null).
- **→ 4 figures:** additionally merge F2 into F3 — coverage panel becomes F3`a`,
  validation panels go to supplement.
- **→ 8 figures:** split F5 into "matched decoding" and "cross-process decoding +
  controls." Only worth it if the cross-effects survive S4 and become
  main-text-load-bearing.

## Mapping to the analysis guide

| `analysis_guide.md` §12 Fig | Here | Note |
|---|---|---|
| 1 behavior | F1 | unchanged |
| 2 time–frequency | F2`c` | reduced to one example spectrogram |
| 3 HG rises | F2`b` | merged into validation |
| 4 HG traces + pre-trial cross-effects | F4`a`,`b`,`d` | |
| 5 2×2 conjunction counts | F3`a` | |
| 6 onset latency | F6 | promoted to its own figure |
| 7 segregation: conjunction + continuous correlation | F3`b`,`c` | |
| 8 orthogonal power traces | F4`c` | the non-circular confirmation |
| 9 within-block decoding 2×2 | F5`a–e` | collapsed per rules 1–2 |
| 10 cross-decoding + tempgen | F5`f`,`g` + S2, S3 | tempgen trimmed to 2, transfer to supplement |

## Open items before this plan freezes

1. Run the §12.1 principle-8 controls on the cross cells (S4). Their outcome
   decides F5's framing and whether the paper is 6 or 8 figures.
2. Decide draft-Figure-5's fix: disjoint trial halves (§21) or the orthogonal
   panel. The orthogonal panel is recommended.
3. Confirm the A5 onset ordering is significant — F6 stands or folds on it.
4. Confirm whether A6 is reachable for this paper or the next.
