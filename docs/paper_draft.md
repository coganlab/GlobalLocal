# Paper draft: figure plan, Methods and Results

*Assembled 2026-10-01 from the results written up so far. Revised the same day:
F3 is described as made, and F5 is cut to two panels with the gradient moved to
the supplement (§1.2). Later that day the task-significant N4 rerun with main
effects came in; it is in §3.4's last sentence and §3.5's first paragraph.*

*Revised 2026-10-02 after the advisor meeting (§1.2, last rows): the paper is
framed as a characterization; F5 combines the overlap scatter with the height
gradient and adds the congruency ↔ switch cross-decoding; the matched-vs-crossed
bars leave the main figure; participant-level versions of the anatomy tests, a
local-similarity test of "intermixed" and an RT-adjusted check on F3 are coded
and waiting for runs.*

This document has three parts:

1. **Figure plan:** the main-text figures (F1–F5) and the supplement, panel by
   panel, with where each one stands after the 2026-10-01 results.
2. **Methods text,** in paper order.
3. **Results text,** in paper order, with the Discussion sentences and the
   figure captions.

It supersedes three parts of the
[closing figure plan](analysis_plans.md#closing-figure-plan) where they differ:
"Proposed closing figure", "Supplement placement" and the weekly figure table.
That plan still holds the reasoning behind each choice. The
[concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan)
still holds the N1–N4 framework.

**What is copied, and from where.** The Methods and Results text was copied on
2026-10-01 from the sections named above each block. The source sections stay
the reference: if one changes, recopy it here. Two kinds of text were not
copied:

- **Skeletons** (marked ✏️): drafted here from the runbooks, for analyses with
  no written-up result yet. Check every parameter against the run you report.
- **Placeholders** (marked ⬜): results not yet in the docs. The power traces
  (F3) and the LWPC/LWPS decoding (F4) stay in the plan; their results only need
  writing up.

Bracketed text (**[…]**) has to come from the archived run or from elsewhere in
the paper.

---

## Contents

- [1. Figure plan](#1-figure-plan)
  - [1.1 The arc](#11-the-arc)
  - [1.2 What changed with the 2026-10-01 results](#12-what-changed-with-the-2026-10-01-results)
  - [1.3 Status at a glance](#13-status-at-a-glance)
  - [1.4 Main-text figures](#14-main-text-figures)
  - [1.5 Supplement](#15-supplement)
  - [1.6 Left out of the paper](#16-left-out-of-the-paper)
  - [1.7 Before submission](#17-before-submission)
- [2. Methods](#2-methods)
- [3. Results](#3-results)
- [4. Discussion sentences](#4-discussion-sentences)
- [5. Figure captions](#5-figure-captions)

---

## 1. Figure plan

### 1.1 The arc

**Framing (advisor meeting, 2026-10-02):** a characterization by systems
neuroscientists, not a test of one cognitive hypothesis. The paper describes how
lPFC carries two concurrent adaptations: whether (F3), in what population signal
(F4), and where and in what format (F5). It makes no dissociation or
independence claim.

Behavior shows that stability and flexibility are regulated at the same time, in
the same participants (F1). lPFC high gamma carries both adaptation effects in
the behavioral direction (F3), and both are decodable from distributed lPFC
activity (F4). F5 then characterizes their organization. The two adaptations
share electrodes, intermixed locally, with a shallow dorsal–ventral bias in
their balance. Over the same electrodes, congruency and switch type are carried
by largely distinct population codes that share a component only late in the
trial.

**The ending sentence** (provisional; the participant-level tests and the local
similarity of §19 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)
decide its last clause):

> LWPC and LWPS are carried by one intermixed lPFC population, with a shallow
> dorsal–ventral bias in their balance, in which congruency and switch type are
> coded along largely separable axes.

The previous ending, "each tracks the local strength of the demand it
regulates" (the matched-vs-crossed result, §16.6 of the N4 doc), stays as one
Results sentence; its bars left the figure.

The earlier ending had a second clause: whether the spatial gradient is
inherited from the base effects could not be determined. It went to the
supplement with the gradient (§1.4, F5).

Framing rules that still apply (concurrent-regulation plan §0):

- The absent cross-effects (congruency × switch proportion, switch type ×
  incongruent proportion) are scope, not a dissociation claim. The neural ones
  are shown as F3b's off-diagonal subplots, labelled neutrally; the behavioral
  ones are a sentence in the text and S4.
- Decoding is not a multivariate-superiority claim. The claim is that adding
  electrodes yields information no single electrode supplies.
- Organize figures by claim, not by measure. Power and decoding for the same
  claim go in the same figure.

### 1.2 What changed with the 2026-10-01 results

| Result | Effect on the figures |
|---|---|
| **N4 all-lPFC main-effect run written up and checked** ([`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) §17) | The numbers are checked against the run's outputs. Still needed: the plots. |
| **N4 task-significant main-effect rerun** (2026-10-01; [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) §18) | F5 is unchanged, and all lPFC stays primary. The subset agrees in sign and size with both F5 panels: overlap at both levels (LWPC–LWPS *r* = 0.08, *p* = 0.055; congruency–switch 0.17, *p* = 0.001), and matched above crossed by the same margins (test *r* = 0.08, *p* = 0.049). Neither balance is spatially organized there, and the all-lPFC midline description does not repeat. One sentence in the main text (§3.4); the full subset paragraph is in S-N4 (§3.5). |
| **F5 cut to two panels** (2026-10-01, later) | a: the overlap at both levels. b: each adaptation against its own and the other base effect (matched vs crossed). The parcel scatter (old b) and the gradient by score (old c) move to S-N4. The pre-specified parcel and coordinate tests keep one short paragraph in the main text. The balance-vs-balance correlation stays as the test behind b, but the main text no longer talks in balances. |
| **A4 congruency ↔ switch cross-decoding run** ([`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01)) | Supplement S5, not the closing figure. The transfer is partial and late, and not yet shown to be specific to lPFC or free of RT. It earns one Discussion sentence. The main-effect group decoding is dropped. |
| **A6 brain–behavior run** ([`a6_brain_behavior.md`](a6_brain_behavior.md) §13–§14) | Supplement S-BB only, framed as "could not be tested at this n", plus one Discussion sentence. |
| **Weighted medoids** (§17.4 of the N4 doc) | Dropped from the supplement (old S9). They are null on every axis, and the two centre definitions disagree in sign. |
| **F3 made** (2026-10-01) | Panel a plus one 2 × 2 grid of traces (b) that includes the two cross-effects as scope. Its bars are the windowed ANOVA's sign-split interaction clusters, so they give the direction themselves; the old direction-test panel moves to S2d. |
| **Power traces and LWPC/LWPS decoding** | Unchanged. F3 and F4 stay in the main text; their results are not yet written up in the docs. |
| **Advisor meeting, 2026-10-02: framing** | A characterization (§1.1). The absent cross-effects stay scope. The decoding cross-effects (congruency by switch proportion, switch type by incongruent proportion) do show results, so F4 is not a clean 2 × 2 and cannot carry a "but not" claim either; report what they show, in the supplement. |
| **Advisor meeting: F5** | One figure for both anatomy results: the LWPC–LWPS scatter coloured by height tertile, with tertile centroids and the balance by height (§1.4, F5). The matched-vs-crossed bars leave it ("too confusing, not necessary"); their result is one Results sentence and S-N4. The congruency ↔ switch cross-decoding joins F5 as the population-code counterpart of the overlap. |
| **Advisor meeting: is participant a random effect?** | No. Both anatomy tests treat participant as a fixed effect and electrodes as the units of inference (§19 of the N4 doc). Participant-level versions are coded: the z slope decomposed into per-participant slopes, mixed models with a random slope, leave-one-participant-out, and the overlap r per participant. |
| **Advisor meeting: a stronger test of "intermixed"** | Local similarity (§19.3 of the N4 doc): cross-half similarity of electrode pairs by distance, with single scores as the positive control. Coded and validated on planted intermixed and patchy worlds; not run. |
| **Advisor meeting: brain–behavior** | Not working. Stays a supplementary note (S-BB) with one Discussion sentence. |
| **RT-adjusted check on F3** (2026-10-02) | Coded: `sbb.group_adaptation_rt_check`; new A6 runs write `group_adaptation_rt_check.csv`, and `dcc_scripts/stats/f3_rt_adjusted_check.py` reads an existing run's `participant_electrode_scores.csv` (§1.4, F3). |
| **First §19 run and the F3 check** (2026-10-05; N4 §19.8) | F3: both adaptations survive the RT adjustment with participants as the unit (72 % and 77 % retained, p ≈ 0.007). The z gradient holds with participants as the unit at p ≈ 0.04 (weighted sign-flip, random slope) but varies across participants. The overlap r holds when participants are weighted by electrode count (p = 0.03), not unweighted (p = 0.35). The local-similarity numbers are **not usable**: the per-split table splits each electrode on its own, and shared trial noise between neighbours made every score look locally similar (N4 §19.3). Fixed: shared splits plus participant-level inference; needs a rerun with `--long-df`. The within-participant reliabilities quoted in §3.4 share the bias. |
| **Overlap confound controls** (2026-10-05; N4 §19.7) | Coded, not run: nonlinear responsiveness, coordinates, same-half base effects, RT coupling, leave-one-participant-out. The full RT control is the `RT_ADJUST_HG=1` run (N4 §18.10). |

### 1.3 Status at a glance

| Fig | Claim | Status | Next step |
|---|---|---|---|
| F1 | Both adaptations are present concurrently in behavior | ⬜ Figure done; numbers not in the docs. Confirm it was not made with the swapped block map (§1.4, F1). | Confirm the source script; write the Results paragraph |
| F2 | Coverage and signal validation | Needs the coverage table (S1) | Build the per-ROI, per-participant table |
| F3 | lPFC high gamma carries LWPC and LWPS in the behavioral direction | ⬜ Figure made (a, b); both adaptation clusters have the behavioral sign; numbers not in the docs. RT-adjusted check run 2026-10-05: both survive (§3.2 item 4). | Optional 0–0.5 s A6 rerun; unify the trace encoding; save the cluster p-values; collect the simple effects; write up |
| F4 | Both adaptations are decodable from distributed lPFC activity | ⬜ Done; results not in the docs. The cross-effect decoders show results too. | Report trial counts per decoder; write up, cross-effects included (supplement) |
| F5 | One intermixed population with a shallow height bias in the balance; separable base-effect codes over it | Anatomy results final for all lPFC (N4 §17) and the subset (§18). Combined figure, participant-level tests and local similarity coded (N4 §19), not run. Cross-decoding run 2026-10-01; controls pending. | Check the all-lPFC folder survived (§1.7); run `n4_section19_followups.py` on both anatomy runs; finish the A4 controls |

### 1.4 Main-text figures

#### F1 — Task, manipulation, behavior (N1)

| Panel | Content |
|---|---|
| a | Paradigm |
| b | The 2 × 2 block-proportion design. Blocks: A 75 % incongruent / 25 % switch, B 75 / 75, C 25 / 25, D 25 / 75, fully crossed. |
| c | RT: the congruency effect by incongruent proportion (LWPC), the switch cost by switch proportion (LWPS) |
| d | Error rate, the same layout |

**The point:** both adaptations are present in the same participants and the
same sessions. The absent behavioral cross-effects go in the text as scope, and
in S4.

**Numbers so far** (a quick check from `combinedData.csv`, correct trials, 23
participants, with the corrected block map;
[`a6_brain_behavior.md`](a6_brain_behavior.md) §1.1): LWPC 123 ms (SD 147),
positive in 20 of 23, t(22) = 4.0, p = 0.0006; LWPS 97 ms, positive in 20 of 23,
t(22) = 4.5, p = 0.0002. These are not the paper's behavioral model.

⚠️ **Check the source of F1 first.** Until 2026-09-27 the `blockType` map
swapped blocks A and D. That turned "LWPC" into a congruency × switch-proportion
contrast (mean 46 ms instead of 123 ms). `erin_linear_mixed_effects_model.py`,
which the concurrent-regulation plan names as the N1 source, carried the swap
and has no congruency × proportion term. Confirm which script produced the
reported F1 numbers, and remake F1 if it used the old map.

#### F2 — Coverage and signal validation

| Panel | Content |
|---|---|
| a | All electrodes on the MNI surface, coloured by ROI |
| b | Per-electrode high-gamma traces for one example participant, task-responsive electrodes outlined |
| c | Example spectrogram |

Cite the per-ROI, per-participant coverage table (S1) here, with the inclusion
threshold. It answers "why only lPFC?". The counts in the old figure plan are
stale.

#### F3 — Adaptation effects in lPFC high gamma (N2) ⬜

*Layout as made on 2026-10-01: a brain and one 2 × 2 grid of traces.*

| Panel | Content | Data |
|---|---|---|
| a | Task-responsive lPFC electrodes on the surface, with electrode and participant counts | `sig` electrodes, lPFC |
| b | High-gamma traces in a 2 × 2 grid. Rows: the condition effect (switch cost; congruency effect). Columns: the proportion manipulation (switch proportion; incongruent proportion). Each subplot shows its four cell means. The diagonal holds the two adaptation effects (LWPS top left, LWPC bottom right), each with its interaction cluster. The off-diagonal holds the two cross-effects, as scope. | The windowed-ANOVA run on the block-balanced sets: `stimulus_lwps_block_balanced_conditions` (top row) and `stimulus_lwpc_block_balanced_conditions` (bottom row); the `_2way_` figures and `anova_F_traces/*.npz`. **[Confirm the figure came from these two sets.]** |

**One letter for the grid, not four.** The four subplots are one 2 × 2 design,
and the grid is read as a whole: a cluster on each diagonal subplot, none off
it. Lettering them b–e reads as four separate results. In the text,
name a subplot by its row and column ("Fig. 3b, switch cost by switch
proportion"). If the journal wants a letter on every axis, use b–e in reading
order.

**What the old panels became.** Old b and c are the diagonal of b. Old d (the
N2 difference waves) moves to the supplement as S2d: the bars in b are split by
sign, so they already give each interaction's direction. The simple effects go
in the Results text.

**What the bars are.** They are not the N2 direction tests. They are the
significant clusters of the windowed ANOVA's two-way interaction term
(`run_windowed_anova_cluster_correction`, drawn by
`plot_2way_interaction_for_roi`), split wherever the interaction's signed
contrast changes sign. Blue marks a negative contrast.
`_signed_contrast_per_window` orders factor levels alphabetically, so its
contrast is high − low proportion. A negative value means the condition effect
is smaller in high-proportion blocks, which is the behavioral direction. Both
bars are blue, so both adaptation effects point the predicted way. Confirm with
`neg_window_mask` in the `anova_F_traces` npz. The traces agree by eye: in both
diagonal subplots, the gap between the two conditions is wider in the
low-proportion blocks.

**Before the figure is final:**

1. **One encoding in all four subplots.** At present the diagonal colours by
   proportion and dashes the harder condition. The off-diagonal colours by
   condition and dashes the 75 % blocks, and its blue means switch in one
   subplot and congruent in the other. Use colour for the proportion block
   (light 25 %, dark 75 %) and line style for the condition (solid repeat or
   congruent, dashed switch or incongruent) everywhere. Then in every subplot
   the effect is the gap between a colour's two lines, and adaptation is that
   gap narrowing from light to dark. The off-diagonal style comes from the
   fallback branch at `src/analysis/power/plots.py:902-904`.
2. **Row titles that do not claim a null.** "Modulated by demands on
   flexibility but not stability" asserts that the cross-effects are absent.
   A missing cluster does not show that, and the framing rule (§1.1) keeps the
   cross-effects as scope. Label the rows "Switch cost" and "Congruency effect"
   and the columns "Switch proportion" and "Incongruent proportion"; the claim
   goes in the caption title. A "but not" claim would need a direct test of
   each matched effect against its crossed one. In the block-balanced design
   both reduce to block A against block D: LWPS − cross-effect = switch cost
   in A − switch cost in D, and LWPC − cross-effect = congruency effect in D −
   congruency effect in A.
3. **Shared axes and aligned subplots.** The top-right axis now sits lower than
   the top-left one. Use one legend per column instead of four.
4. **The bar colour** encodes sign, and the same blue is a trace colour. Draw
   the bars black and give the sign in the caption, or keep blue and say what
   it means.
5. **Panel a:** say whether right-hemisphere electrodes are mirrored onto the
   left hemisphere.

**To report with it** ([`n2_direction_tests.md`](n2_direction_tests.md) §7–§8):

- `n_electrodes` and `n_subjects` together, from the run's `ch_names`. If this
  is the same task-responsive set as S5 and S-BB, expect 171 electrodes from
  21 participants.
- Each cluster's window and p. The ANOVA computes a p per cluster
  (`sig_clusters_with_sign`) but does not save it: `power_traces_dcc.py`
  writes only the masks to `anova_F_traces/*.npz`, and the interaction npz gets
  an empty `cluster_p_values`. Save it, and raise `N_PERM` above the default
  500 (which floors p at ~0.002) for a reported p.
- Each simple effect's sign and significance (the N2 tests). By eye, in both
  diagonal subplots the easier condition in the low-proportion blocks (repeat
  in 25 %-switch blocks, congruent in 25 %-incongruent blocks) sits lowest late
  in the trial, and the two high-proportion traces nearly meet. If the
  simple-effect tests agree, say so: the effect is present in low-proportion
  blocks and smaller in high-proportion blocks.
- The N2 tests run on the 4-cell sets, the figure's ANOVA on the 8-cell
  block-balanced sets. Say so, or recompute the simple effects from the
  block-balanced `_evoked.npz` files so that both come from the same cells.
- The per-participant direction tally and a leave-one-participant-out sweep.
  Neither is implemented on either path; both can be rebuilt from the saved
  `_evoked.npz` files.
- Bar starts are not onset estimates: a cluster test does not localize in time
  (Sassenhagen & Draschkow, 2019). Do not write that LWPC began at 0.25 s and
  LWPS at 0.45 s; an ordering claim belongs to A5 (Timing, below).
- Both clusters run to the end of the epoch, past the median RT (1.19 s). A
  matching sign is not independent evidence on its own: if high gamma tracks
  RT within cells, the neural difference of differences inherits behavior's
  sign. **The check is coded (2026-10-02):** `sbb.group_adaptation_rt_check`
  takes the A6 electrode scores with and without the RT-linked part of high
  gamma (`rt_adjust_hg`, §2.7) and tests the mean LWPC and LWPS with
  participants as the unit (t-test, sign-flip, a mixed model with a participant
  random intercept), plus the share retained after adjustment. On the existing
  A6 run:

  ```bash
  python dcc_scripts/stats/f3_rt_adjusted_check.py \
      --electrodes <A6 run>/participant_electrode_scores.csv
  ```

  New A6 runs write `group_adaptation_rt_check.csv` and block (0) of
  `summary.txt` themselves. Rerun A6 with `WINDOW_TMAX=0.5` for a window before
  most responses. Reading: adjusted means still positive → F3's direction is
  not RT coupling alone; adjusted means near zero → the window-mean adaptation
  goes with the HG–RT coupling, which the adjustment removes together with any
  adaptation expressed through it (§2.7). On planted data an RT-only
  "adaptation" keeps −5 % to 25 % of its raw size and a real one about 100 %.
  It is a window-mean check on the task-significant set, not a test of the
  time-resolved clusters. **Result (2026-10-05, N4 §19.8):** LWPC keeps 72 %
  (adjusted mean d = 0.080, p = 0.007, 16/21 participants positive), LWPS 77 %
  (0.122, p = 0.007, 15/21). RT coupling carries about a quarter of each; the
  rest is not RT coupling.

Main-effect traces (incongruent vs congruent, switch vs repeat) go to S2.

**What would change it:** a direction opposite to behavior. That stops
everything downstream until the epoch metadata are re-read.

#### F4 — Decoding the two adaptations (N3) ⬜

| Panel | Content |
|---|---|
| a | LWPC: congruency decoded within 25 %- and within 75 %-incongruent blocks, each against its refit shuffle null, and their difference |
| b | LWPS: switch type decoded within 25 %- and within 75 %-switch blocks, the same layout |

Use the `_block_balanced` condition sets
([`decoding.md` › Decoding job](decoding.md#decoding-job) §2.1), so a tonic block
difference cannot enter the contrast.

**To report with it** (concurrent-regulation plan §3):

- Trial counts per class for each decoder. If the LWPC and LWPS decoders differ
  in n, either subsample to the common minimum or say that their accuracies are
  not compared.
- Pre-stimulus windows as the artifact meter: congruency cannot be decodable
  before the stimulus.
- The samples are CV repeats and bootstraps of one pseudopopulation, not
  participants. The leave-one-subject-out run (`submit_loo_decoding_dcc.sh`)
  checks that no participant carries a result.

**Layout:** a trellis with shared axes and one legend reads as one panel.

**Dropped:** the block-transfer panel (old F4c). The cross-proportion transfer
is shelved: the within-block ceiling sits near chance with ~20 rare-class trials
per participant per block, and a drop in transfer is confounded with LWPC itself.

#### F5 — How the two adaptations are organized across lPFC (N4, with A4)

All lPFC (398 electrodes, 22 participants) is the primary set, as specified
before analysis. The task-significant set is reported in full in S-N4 (N4 §18).
Data and status per panel: §16.7.3 and §17.4 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md).

*Revised 2026-10-02 after the advisor meeting.* The overlap scatter and the
height gradient go in one figure, the matched-vs-crossed bars leave it, and the
congruency ↔ switch cross-decoding joins it. (The 2026-10-01 version had two
panels: the overlap at both levels, and the bars. Its code still runs:
`fig5.png`.)

| Panel | Shows | Key numbers | Data / code | Status |
|---|---|---|---|---|
| a | **Where the height bands are:** the electrodes on a sagittal projection (MNI y against z), coloured by height tertile, with the two cuts drawn; or a rendered brain in the same colours (`brain_png`) | tertile cuts (mm) | `sfa.figure5_height` → `fig5_height.png` | Coded, not run |
| b | **The overlap:** LWPC against LWPS, coloured by tertile, with the identity line and the pre-specified separate-half r. A box marks panel c's region. | r = 0.10, p = 0.0005 (397 electrodes, 21 participants); participant-level r **[N4 §19.2]** | `fig5_height_points.csv`; r from the segregation run's `correlation.json` | Coded, not run |
| c | **Overlap along the line, gradient across it:** the three tertile centroids enlarged, each with its 95 % participant-bootstrap ellipse. Dorsal should sit above the identity line (leaning LWPS), ventral near it. | centroids and balance CIs **[run]** | `fig5_height_centroids.csv` | Coded, not run |
| d | **The gradient in the test's units:** LWPC − LWPS by tertile, participant means ± SEM, with the z slope's electrode-level p and participant-level p | z slope −0.0077 SD/mm, p = 0.008; participant-level p and random-slope p **[N4 §19.1]** | `fig5_height_balance.csv`, `mni_z_slope_by_participant.csv` | Coded, not run |
| e | **Separable codes over the same electrodes:** congruency ↔ switch cross-decoding over time in task-responsive lPFC: both within-contrast ceilings and both transfers, with the early window (both decodable, no transfer) and the late one (partial transfer) marked; the RT-matched transfer once the seeds are in | ceilings 0.76 / 0.76; transfer from +0.62 s, 47 % and 26 % of the ceilings | A4 run ([`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01)) | Run 2026-10-01; controls pending (below) |

Optional, beside b: the congruency-against-switch scatter (r = 0.23) with the
same colouring, as the anatomical partner of e (both are about the base
effects).

**Why the two anatomy results fit one scatter.** On LWPC (x) against LWPS (y)
they lie along perpendicular directions. The overlap is spread along the
identity line; the gradient is a shift across it, because LWPC − LWPS is each
point's signed distance from the line. Colour by height shows both, and the
centroids (c) make a 2 %-of-variance shift visible where 397 coloured dots
cannot. The separating line in score space is the identity line; the
dorsal/ventral cuts are drawn on the anatomy (a), where they mean something.
The figure's choices are argued in §19.4 of the N4 doc.

**What the bars became.** "Each adaptation tracks its own base effect more than
the other" (matched − crossed r = 0.09, p < 0.001) is one Results sentence; the
bars move to S-N4 (`fig5.png` b). The advisors found them confusing and not
necessary for the story.

**Panel e, before it carries a panel.**

- Its level differs: e is about the base effects, b–d about adaptation. One
  caption sentence says so; under the characterization framing that is a
  description of the same population at two levels.
- The positive test is the time course: the share of each ceiling that
  transfers rises from about 0 early to 26–47 % late (a window × within-vs-cross
  interaction). Do not claim "no transfer early", which is a null.
- RT: matching RTs cuts the transfer relative to its random control (43 % vs
  70 %, 12 % vs 39 %). Needs the seeds before it is a claim. If it holds, the
  late shared component tracks the slowing incongruent and switch trials share,
  and the separate components carry the process-specific information.
- Occipital transfers too, later (+1.0 s on). Read as: the late shared
  component is not lPFC-specific, consistent with a general slowing or response
  signal. What lPFC should own is early separable codes for both variables and
  the adaptations themselves. Check whether occipital decodes switch type early
  at all (congruency may be partly visual), and, if coverage allows, run the F3
  and F4 adaptation tests in occipital.
- Still to run (S5 list): seeds for the baseline, `rt` and `random`; lPFC
  subsampled to occipital's 54 electrodes; post-stimulus-only kept shares; the
  task positive controls.

**Design notes.**

- Panel b plots the scores the test correlates (responsiveness out,
  participant-centred, each score's mean added back, on delta's pooled scale),
  not `joint_scatter.png`. The job prints the z slope of the plotted balance
  next to the coordinate test's as a check that b–d and the test agree.
- Height tertiles use a one-hue violet ramp (validated as an ordinal ramp), so
  they are not read as the blue/orange congruency/switch identity of e.
- No "electrodes driving the z effect": at single-electrode reliability ~0.3
  the most influential electrodes are partly selected on noise. The centroids
  carry the gradient.
- Brain maps of the scores appear only as coverage or illustration.
  Single-electrode reliability is ~0.3, so a map cannot carry a claim.

**Placement of each result** (supersedes the 2026-10-01 table and §17.4 of the
N4 doc):

| Result | Where |
|---|---|
| LWPC and LWPS share electrodes; no centroid separation | Main text, F5b, with the participant-level r in the same sentence |
| Locally intermixed (local similarity, N4 §19.3) | Main text, one sentence (the balance's near-range share against the single scores'); curves in S-N4. Only with the positive control beside it. |
| Height gradient in the balance | F5a, c, d, with the participant-level and random-slope p; the parcel test as one sentence. Omnibus only, never a named parcel. |
| Congruency and switch share electrodes | Main text sentence; optional scatter beside F5b |
| Separable congruency and switch codes, late partial transfer | F5e if its controls hold; otherwise S5 and one Discussion sentence |
| Each adaptation tracks its own base effect more than the other (Test 1) | One Results sentence; bars in S-N4 |
| Base-effect balance by parcel (label r = 0.73) | S-N4; descriptive |
| Dorsomedial vs ventrolateral; carried by LWPC | S-N4, exploratory |
| Inheritance (Test 2) | S-N4 only |
| Leave-one-participant-out range of the z slope | S-N4 |
| Task-responsive subset, with main effects (N4 §18) | One sentence at the end of §3.4; in full in S-N4 (§3.5) |

**What would change it.**

- The z slope's participant-level p (weighted sign-flip) or the random-slope p
  above 0.05: a few participants carry the gradient. Keep it as one Results
  sentence and S-N4, and drop panels a, c and d's claim to a gradient (b stays).
- Local similarity: the balance rises at short range → "patchy", not
  "intermixed"; say "overlapping". The single scores flat as well → no power;
  say "overlapping" and report the null with its control.
- e's controls fail (lPFC at 54 electrodes transfers like occipital in the same
  window, and the early separation is not lPFC's) → e goes back to S5.

**Open for the advisor:** all lPFC as the primary set; lPFC-only scope. The
gradient's weight is settled: it stays in F5.

#### Timing (A5): fold in or drop

A panel (F3c) only if the LWPC/LWPS onset ordering is significant, with the
latency–amplitude guard. No timing result is written up in the docs; a null
folds it away.

### 1.5 Supplement

Numbers kept from the earlier plans so the cross-references in other docs still
work. Renumber at submission.

| S | Content | Status | Text |
|---|---|---|---|
| S1 | Per-ROI, per-participant coverage table with the inclusion threshold | To build | – |
| S2 | Main effects (congruency, switch type) in lPFC high gamma: traces and decoding | ⬜ Not in the docs | – |
| S2b | Main-effect electrodes on the brain (one pre-specified method), other methods as robustness | Not run | – |
| S2c | LWPC and LWPS traces within congruency, switch and both groups, selected on half A and tested on half B, with the group × effect-type test | Not run | – |
| S2d | N2 direction tests (old F3d): for LWPC and LWPS, each simple effect and the difference waves, each with its own cluster bar | Implemented; results not collected | §2.3 |
| S3 | Low-frequency bands, rerun with a ≥ 1 s baseline that predates the block context | Not run. Do not use the current low-band results: the 0.5 s pre-stimulus baseline is 2–4 cycles at theta and may contain the block-level effect. | – |
| S4 | Absent behavioral cross-effects, as scope (the neural ones are F3b's off-diagonal) | – | One sentence in Results |
| S5 | Congruency ↔ switch cross-decoding, labelled as base-effect geometry: the unselected lPFC transfer with its ceilings, `remove_mean`, RT-matched vs random, occipital. The main lPFC traces become F5e if the controls hold (§1.4, F5); the controls stay here. | Run 2026-10-01. Seeds, the electrode-matched region comparison, the positive controls, `mean_only` and response-locked runs still to do. | §2.6, §3.6 |
| S5b | LWPC/LWPS decoding cross-effects: congruency decoded within switch-proportion blocks, switch type within incongruent-proportion blocks. They show results, so F4 is not a clean 2 × 2; describe them. | Run; not in the docs | – |
| S6 | Haufe-transformed decoder patterns, as convergent evidence | Optional; not run | – |
| S7 | Per-participant high-gamma traces; demographics, electrode counts, exclusions | To build | – |
| S8 | Cross-decoding control table for every transfer reported | The A4 block is filled in ([`decoding.md` › A4 §13.7](decoding.md#137-what-to-report-and-where-things-stand)); pseudo-trial counts missing | – |
| S10 | Per-trial-baseline robustness rerun of the power traces; direct block comparisons | Not run | – |
| S-N4 | Anatomy in full: the task-significant set; the matched-vs-crossed bars (the 2026-10-01 F5b, `fig5.png` b); the two balances by parcel; the four scores by distance from the midline and by height; per-parcel bars, coverage, leave-one-out tables; both coordinate tables and the hemisphere fits; the height vs midline models; Test 2 with the reliability and bootstrap; map similarity; **§19 of the N4 doc:** per-participant z slopes and their leave-one-out range, the mixed models, the per-participant overlap r, the local-similarity curves | All-lPFC parts ready; the parcel and band panels have data, not plots. The task-significant set is complete with main effects (2026-10-01, N4 §18). §19 coded 2026-10-02, not run. | §3.5 |
| S-BB | Brain–behavior across participants: RT-adjusted (primary) and raw correlations, reliabilities, ceiling, power | Run 2026-09-30; text final | §2.7, §3.7, §5.3 |

### 1.6 Left out of the paper

| Result | Why |
|---|---|
| Weighted medoids (`score_centers.csv`) | Null on every axis; 6 of 25 groups have both medoids on the same contact; the two centre definitions disagree in sign on z |
| Noise-corrected LWPC–LWPS ratio (1.03 electrode, 3.38 parcel) | Not estimable at these reliabilities |
| `joint_scatter.png` correlations (pooled 0.32, within-participant 0.22) | Full-data correlations, not the pre-specified test |
| A4 main-effect groups (`both`, `*_only`) | Opposite to the shared-code prediction, pre-stimulus windows, groups change with each run's selection trials |
| A4 with RT matching and `remove_mean` together | Uninterpretable: the congruency ceiling is gone, in the random control too |
| Cross-proportion (block) transfer | Shelved: no within-block ceiling, and confounded with LWPC itself |
| A6 positive-electrode summary (r ≈ 0.53–0.56, p ≈ 0.07) | Post hoc, 11–12 participants, at or above its own ceiling, selects on the sign it averages |
| A6 mean \|d\| joint regression; level-2 counts; level-3 slopes | Suppression from shared noise; uncorrected and coverage-driven; the slopes measure the high gamma–RT link |
| "Correlated at the noise ceiling"; "LWPC dominance increases ventrally"; "the tilt survives the main effects" | Withdrawn or not supported; see §16.6.7 of the N4 doc for the wording to use instead |

### 1.7 Before submission

**F5 and S-N4** (N4 §17.4):

- [ ] **Check that the all-lPFC anatomy folder survived.** The task-significant
      rerun wrote to `anatomy_a1_lpfc_window_0.0to1.5s_sig/`, the all-lPFC
      folder (its `score_anatomy.json` → `maps`). If no copy was made first,
      the matched-vs-crossed `delta_tracking.csv` and the S-N4 tables are
      gone; rerun the anatomy job on the all-lPFC `_main_effects` segregation
      run to get them back (N4 §18.1). That rerun also draws the combined F5
      and everything in N4 §19. Then copy both folders to names that say which
      set they hold.
- [x] Rerun the task-significant set with main effects; report it in full
      (2026-10-01; N4 §18, text in §3.4 and §3.5).
- [ ] Confirm the subset's segregation run name, and that its
      `correlation_main_effects.json` gives congruency–switch *r* = 0.167 (N4
      §18.3 recomputed it from `per_split.csv`).
- [x] Run `dcc_scripts/stats/n4_section19_followups.py` on the all-lPFC run
      (2026-10-05; N4 §19.8): participant-level z slope and overlap r in §3.4.
- [ ] Rerun it with `--long-df <segregation run>/long_df.csv` (shared splits;
      the first local-similarity numbers are not usable, N4 §19.3) and
      `--rt-coupling` (§19.7); replace the within-participant reliabilities in
      §3.4 with the shared-split ones. Then the task-significant run.
- [ ] Run the segregation job with `RT_ADJUST_HG=1` (N4 §18.10) and the
      anatomy job on it: the full RT control for the overlap and the gradient.
- [ ] Decide F5e with the A4 controls below; draw it from the A4 run's traces.
- [ ] Plot the S-N4 panels (the bars, the balances by parcel, the bands).
      *The bars are `fig5.png` b; the anatomy job already draws them.*
- [ ] Leave-one-participant-out on the pre-specified LWPC–LWPS correlation, and
      its p from 10,000 permutations.
- [ ] Advisor decisions: primary set and scope. *The gradient's weight was
      decided on 2026-10-02: it stays in F5 with the scatter.*

**F1–F4:**

- [ ] F1: confirm the behavioral source script uses the corrected block map.
- [ ] F2: build the coverage table (S1).
- [ ] F3: run `dcc_scripts/stats/f3_rt_adjusted_check.py` on the A6 run's
      `participant_electrode_scores.csv`, and rerun A6 with `WINDOW_TMAX=0.5`
      (§1.4, F3).
- [ ] F3: one trace encoding in all four subplots and neutral row labels
      (§1.4, F3); save the ANOVA cluster p-values and raise `N_PERM`; collect
      the simple effects; add the per-participant tally and
      leave-one-participant-out.
- [ ] F4: trial counts per class for each decoder; check pre-stimulus windows.

**S5** ([`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01)):

- [ ] Seeds for the baseline, `rt` and `random`; recompute `kept` over
      post-stimulus windows.
- [ ] lPFC subsampled to occipital's 54 electrodes.
- [ ] Task positive controls; `mean_only`; response-locked.
- [ ] Pseudo-trial counts per cell from each run.

**Everywhere:**

- [ ] Fill in the preprocessing paragraph (§2.1) and check it describes the
      epochs file every run used.
- [ ] Archive the git commit, submission command and Slurm log with each run.

---

## 2. Methods

### 2.1 Participants, recordings and preprocessing

*Copied from [`methods.md` › N4](methods.md#participants-recordings-and-electrode-population), first paragraph.*

> Intracranial EEG was recorded from **[N participants]** while they performed the
> Global/Local task. Broadband high-gamma activity (70–150 Hz) was extracted from
> cleaned, average-referenced recordings using a filterbank–Hilbert transform and
> epoched from −1.0 to 1.5 s relative to stimulus onset. The preprocessing
> pipeline supplied signal epochs and a separately constructed 0.5-s
> prestimulus baseline object to a z-score rescaling operation. Because alignment
> of individual signal trials to individual baseline segments has not been
> verified in the current implementation, this normalization is not described as
> trial-by-trial baseline correction. Epochs were decimated by a factor of eight.
> Trials exceeding 10 SD were marked as outliers, and channels for which more
> than 5% of trials were outliers were excluded. Analyses included correct trials
> only and omitted the first trial of each block, for which task sequence was
> undefined.

**[Add: task-responsive electrode definition (high gamma against the
pre-stimulus baseline), used by F3, F4, S5 and S-BB.]**

### 2.2 Task design and behavior (F1) ⬜

**[To write.]** It needs to state:

- The four block types, fully crossed: A 75 % incongruent / 25 % switch, B 75 /
  75, C 25 / 25, D 25 / 75 (`src/task/mainTask.m`).
- LWPC = (I − C)₂₅ − (I − C)₇₅ and LWPS = (S − R)₂₅ − (S − R)₇₅ on mean RT, low
  minus high proportion, so positive means the predicted adaptation: a smaller
  congruency effect or switch cost in high-proportion blocks. The neural scores
  use the same orientation.
- Trial inclusion and the model or test behind the reported numbers.

### 2.3 Power traces and adaptation tests (F3) ✏️

*Skeleton, drafted from `run_power_traces_dcc.py`,
`windowed_anova.run_windowed_anova_cluster_correction` and
[`n2_direction_tests.md`](n2_direction_tests.md) §1–§4. Check every parameter
against the reported run.*

> We analysed **[n]** task-responsive lPFC electrodes from **[N]**
> participants. For each electrode, high gamma on correct trials was averaged
> within each cell of a block-balanced design: congruency × incongruent
> proportion × switch proportion for the congruency effect, and switch type ×
> incongruent proportion × switch proportion for the switch cost. Each
> combination of the two proportions is one block type, so every cell came from
> a single block type. The traces show the mean across electrodes of each
> condition × proportion cell, weighting the two levels of the other proportion
> equally, ± 1 SEM across electrodes.
>
> Each condition effect was tested against each proportion manipulation with a
> full-factorial ANOVA fitted across electrodes (ordinary least squares) in
> sliding 250-ms windows (64 samples at 256 Hz, stepped by 62.5 ms). Because
> every cell is one block type and every factor is in the model, each two-way
> term is an equal-weight contrast, so a sustained difference between blocks
> cannot enter it. The null distribution permuted cell labels within each
> electrode, with the same permutation in every window (**[N_PERM]**
> permutations). Windows whose *F* exceeded the 95th percentile of their null
> formed candidate clusters. Candidates were split wherever the interaction's
> signed contrast changed sign, and a cluster was kept when its extent in
> windows **[or summed excess *F*, if the run set `CLUSTER_STAT=mass`]**
> exceeded the 95th percentile of the largest null cluster, with null clusters
> split the same way. The sign of a kept cluster gives the direction of the
> interaction: whether the condition effect was smaller in the high-proportion
> blocks, the direction of behavioral adaptation, or larger. The adaptation
> effects were congruency × incongruent proportion (LWPC) and switch type ×
> switch proportion (LWPS). The cross-effects, congruency × switch proportion
> and switch type × incongruent proportion, were tested in the same models.
>
> To read each adaptation effect, we also tested the condition effect within
> the low-proportion and within the high-proportion blocks (two-sided, paired
> cluster-based permutation tests over time; cluster-forming and cluster
> thresholds p = 0.05; **[N_PERM]** permutations). In all of these tests the
> unit of observation was the electrode, pooled across participants. Because
> electrodes within a participant are correlated, the p-values are optimistic.
> We therefore also report the number of participants whose electrode-averaged
> effect points in each direction, and the tests repeated with each participant
> left out.

✏️ **RT-adjusted check** (skeleton, 2026-10-02; code
`sbb.group_adaptation_rt_check`, run as in §1.4, F3). Check the window and the
electrode set against the A6 run it reads.

> Because the adaptation clusters extended past the median response time, we
> asked whether their direction could arise from trial-by-trial coupling
> between high gamma and RT alone. For each task-responsive lPFC electrode, we
> removed the RT-linked component of window-averaged high gamma (0–1.5 s;
> slope estimated within the 16 design cells, as in the brain–behavior
> analysis, §2.7) and recomputed LWPC and LWPS. We averaged each participant's
> electrodes and tested the mean across participants against zero (one-sample
> *t*-test and sign-flip permutation; a mixed model with a participant random
> intercept on the electrode scores gave the same conclusion **[check]**), and
> report the share of the unadjusted mean that remained. **[If run: the same
> with a 0–0.5 s window, before most responses.]**

### 2.4 Decoding the adaptation effects (F4) ✏️

*Skeleton, drafted from the decoding job's defaults
([`decoding.md` › Decoding job](decoding.md#decoding-job) §2). Check every
parameter against the reported run.*

> We decoded each condition contrast separately within each block type from
> **[n]** task-responsive lPFC electrodes (**[N]** participants). For LWPC,
> congruency (incongruent vs congruent) was decoded within 25 %- and within
> 75 %-incongruent blocks. For LWPS, switch type (switch vs repeat) was decoded
> within 25 %- and within 75 %-switch blocks. Each class pooled two block types,
> each subsampled to the smaller before pooling, so that a sustained difference
> between blocks could not enter the contrast. For each condition, every
> electrode's trials with missing values were dropped, and each electrode was
> independently subsampled to the smallest trial count of any electrode in the
> region (**[n]** pseudo-trials per class). Electrodes from all participants
> were concatenated into one pseudopopulation, and classes were balanced by
> subsampling. The classifier was principal component analysis (components
> explaining 90 % of the training variance) followed by linear discriminant
> analysis, applied to 250-ms windows (64 samples) stepped by 62.5 ms, with
> five-fold cross-validation repeated five times, over five independent draws
> of the pseudopopulation. The null distribution refit the classifier on
> permuted training labels (50 permutations per draw). A window was
> significant when the mean true accuracy exceeded the 95th percentile of the
> null, and runs of significant windows were kept when longer than the 95th
> percentile of the longest run under the null. The two block types' accuracy
> traces were compared with a cluster-based permutation test in each
> direction (α = 0.025 each). The samples entering these tests are
> cross-validation repeats of the pseudopopulation, not participants, so the
> p-values measure within-dataset reliability; a leave-one-participant-out
> analysis checked that no participant carried a result.

### 2.5 Anatomy: segregation and continuous anatomy (F5, S-N4)

*Copied from §17.2 of [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#172-methods).
It replaces the bracketed N4 template in `methods.md` for the all-lPFC analysis.*

> **Electrode population.** Anatomical analyses used every electrode in lateral
> prefrontal cortex (lPFC; 398 electrodes, 254 left and 144 right, from 22
> participants, 1–55 per participant). The set was defined by atlas label, not
> by task responsiveness or by any effect, because these analyses ask how
> effects are distributed across the region rather than whether they are
> present. This choice was recorded before analysis. Results in the
> task-responsive subset are reported in full in the Supplement. Each electrode
> was assigned a Destrieux parcel and fsaverage (MNI) coordinates from the
> participant's reconstruction.
>
> **Per-electrode scores.** High-gamma power (70–150 Hz) on correct trials was
> averaged over 0–1.5 s after stimulus onset. Each process was scored from the
> four cell means of its 2 × 2 design, congruency (incongruent, I; congruent, C)
> × incongruent proportion (25 %, 75 %) and switch type (switch, S; repeat, R)
> × switch proportion, with equal weight on every cell. The adaptation effects
> were differences of differences, LWPC = (I − C)₂₅ − (I − C)₇₅ and
> LWPS = (S − R)₂₅ − (S − R)₇₅. The main effects were the mean simple effects,
> congruency = ½[(I − C)₂₅ + (I − C)₇₅] and switch = ½[(S − R)₂₅ + (S − R)₇₅].
> Each was divided by the pooled within-cell standard deviation (Cohen's *d*).
> Positive LWPC and LWPS mean the predicted adaptation: a smaller condition
> effect in high-proportion blocks. Equal cell weights stop the unequal trial
> counts of the proportion design from leaking the main effect into the
> interaction, or block-level differences into the main effect.
>
> **Disjoint trial halves.** Each electrode's trials were divided into random
> halves 1,000 times, stratified on congruency, incongruent proportion, switch
> type and switch proportion. All four scores were computed on each half. An
> electrode's score was the mean over halves and splits. Every test that relates
> two scores across electrodes took them from opposite halves within each split
> and then averaged over splits, so that shared trial noise could not create an
> association. A map's split-half reliability was the mean correlation between
> electrodes' two half estimates.
>
> **Balances.** Each score was divided by its standard deviation across
> electrodes, without centring or within-participant standardization. The
> adaptation balance was Δ_adapt = LWPC − LWPS (positive: relatively
> LWPC-dominant) and the base-effect balance Δ_main = congruency − switch
> (positive: relatively congruency-dominant). Because each balance is a
> difference within an electrode, a test of how it varies with anatomy is an
> effect-type × anatomy interaction, and anything an electrode contributes
> equally to both scores (gain, responsiveness, participant) cancels.
>
> **Co-localization.** To test whether two effects share electrodes, one score
> from one half was correlated with the other score from the other half (both
> directions, every split) after both were regressed on responsiveness (mean
> absolute high-gamma) and centred within participant. Spearman ρ was averaged
> over splits. Participants with fewer than three electrodes were excluded
> (397 electrodes, 21 participants remained). The null permuted electrode
> correspondence within participant, identically across splits. Electrodes
> positive for each adaptation effect were compared in location by the distance
> between their centroids, with electrode labels shuffled within participant.
>
> **Anatomical test.** Each balance was modelled as
> balance = parcel + responsiveness + participant, with participants as fixed
> effects. The statistic was the partial *F* for the parcel block. Destrieux
> parcels sampled in at least three participants entered (19 parcels, 396
> electrodes). The null exchanged the two effect labels independently within
> every electrode, which flips the sign of the balance while keeping each
> electrode's participant, location, responsiveness and pair of scores
> (10,000 permutations; *p* = (*b* + 1)/(*N* + 1)). Adjusted parcel means, with
> Benjamini–Hochberg correction over the 19 parcels, were used only to describe
> a significant omnibus result. The test was repeated with each participant
> left out (1,000 permutations per fold).
>
> **Coordinate test.** Each balance was also modelled as
> balance = MNI *y* + *z* + *x* + responsiveness + participant, across all
> electrodes and within each hemisphere, with the same null. We report the
> block *F*, and slopes in standard deviations per millimetre with *p* values
> Bonferroni-corrected over the three axes.
>
> **Main effects as a reference** (specified before this run). Test 1 asked
> whether the balances track each other: Δ_main from one half against Δ_adapt
> from the other, by the co-localization procedure above, with and without MNI
> coordinates partialled out within participant. The matched (congruency–LWPC,
> switch–LWPS) and crossed (congruency–LWPS, switch–LWPC) pairings were
> computed the same way. Test 2 asked whether the adaptation gradient is
> inherited: the coordinate model for Δ_adapt was refitted with Δ_main as a
> covariate, and the change in slope (shrinkage = 1 − slope with
> covariate / slope without) was read against Δ_main's split-half reliability,
> because a covariate measured with reliability λ can remove only about λ of a
> slope it fully carries. Uncertainty in the shrinkage came from 2,000
> participant bootstrap resamples.
>
> **Exploratory follow-ups** (chosen after the coordinate result). Because
> height and distance from the midline are correlated across lPFC electrodes,
> the coordinate model was refitted with |*x*| in place of signed *x*. Each
> single score was fitted on *y* + *z* + |*x*| + responsiveness + participant.
> A single score has no partner to exchange labels with, so its null shuffled
> coordinates among each participant's electrodes (2,000 shuffles). Adjusted
> Cohen's *d* (participant and responsiveness removed) is shown by tertile of
> |*x*| for display only.
>
> **Reporting.** All tests were two-sided with α = 0.05.
> Electrode maps and per-participant weighted centres were descriptive and
> were not used for inference.

⚠️ The last sentence mentions weighted centres, which §1.6 leaves out of the
paper. Drop "and per-participant weighted centres" if they are not shown.

With F5 cut to two panels (§1.4), two changes when this goes into the
manuscript:

- After the Test 1 sentence, add: "Because the covariance of the two balances
  equals the two matched covariances minus the two crossed ones, Test 1 tests
  whether each adaptation effect tracks its own base effect more closely than
  the other." The Results (§3.4) lean on this.
- The coordinate test, the exploratory follow-ups and Test 2 now report
  results that sit mainly in S-N4. Keep them here or move them to the
  supplementary methods, to match where the journal puts S-N4.
- The task-responsive subset was scored on 200 splits, not 1,000, and its
  label tests used the 15 parcels sampled in at least three of its
  participants (N4 §18.1). Add to the supplementary methods: "In the
  task-responsive subset, trials were split 200 times rather than 1,000."

✏️ **Participants as the unit, and local similarity** (skeleton, 2026-10-02;
§19 of the N4 doc). Add after the coordinate test.

> **Participant-level tests.** The tests above treat participant as a fixed
> effect and electrodes as the units of inference. We therefore repeated the two
> main tests with participants as the units. For the overlap, each participant's
> separate-half LWPC–LWPS correlation was computed over its own electrodes
> (participants with at least four), on the same scores and splits, and the
> correlations were averaged in Fisher *z* with weights *n* − 3 and tested by
> flipping the sign of whole participants. For the height gradient, the pooled
> slope equals a weighted average of the participants' own slopes (each
> participant weighted by its electrodes' spread in height, after the same
> covariates); we tested that average by flipping the sign of whole
> participants, tested the unweighted mean of the slopes of participants with
> at least three electrodes and 5 mm of spread, and fitted a linear mixed model
> with a participant random intercept and random height slope (REML;
> predictors centred within participant). We also report the slope with each
> participant left out.
>
> **Local similarity.** To ask whether electrodes leaning toward LWPC or LWPS
> form local patches, we compared each pair of electrodes within a participant
> across trial halves: one electrode's score from one half against the other's
> from the other half, averaged over both directions and all splits, after
> removing responsiveness, the linear coordinate trend and each participant's
> mean (Spearman, on unit-variance scores). Pairs were binned by distance
> (< 10, 10–20, 20–40, > 40 mm). For this analysis the trials were split once
> per participant and repetition, and the same halves were used for all of its
> electrodes, so that no electrode's half shared trials with another
> electrode's other half (**[n]** splits). Each participant's baseline for a
> bin was its mean over shuffles of the participant's electrode positions;
> we tested the excess over that baseline by flipping the sign of whole
> participants, with intervals from a participant bootstrap. The same analysis
> of LWPC, LWPS, congruency and switch type alone served as the positive
> control. An electrode's similarity with itself across halves is its
> split-half reliability, against which the neighbours' similarity is
> expressed where that reliability is clearly positive.
>
> **Alternative explanations of the overlap.** We repeated the overlap test
> with, in turn, log and squared responsiveness, MNI coordinates, each half's
> congruency and switch-type effects (from the same half as each adaptation
> score), and each electrode's within-cell correlation between high gamma and
> RT as additional covariates **[N4 §19.7; report what was run]**.

### 2.6 Congruency ↔ switch-type cross-decoding (S5)

*Copied from [`methods.md` › A4](methods.md#a4-congruency--switch-type-cross-decoding), Methods.*

> **Data and electrodes.** We used the stimulus-locked high-gamma epochs described
> above (70–150 Hz, −1.0 to 1.5 s, decimated to 256 Hz; correct trials only; the
> first trial of each block omitted). We decoded all lateral prefrontal electrodes
> whose high gamma exceeded their pre-stimulus baseline (171 electrodes from 21
> participants). Electrodes were not selected for a congruency or switch-type
> effect. As a regional comparison we repeated every analysis on the task-responsive
> occipital electrodes (54 electrodes from **[N]** participants).
>
> **Conditions and pseudopopulation.** Trials were sorted into the four congruency
> × switch-type cells, each pooled over the four block types. For each cell, each
> electrode's outlier trials were removed and the electrode was randomly subsampled
> to the smallest number of clean trials of any electrode in that cell
> (**[n]** pseudo-trials per cell). Electrodes from all participants were then
> concatenated into one pseudopopulation. Because electrodes were sampled
> independently, a pseudo-trial combines different trials of the same condition
> across electrodes, including electrodes from the same participant.
>
> **Decoding.** Each 250-ms window (64 samples, stepped by 62.5 ms; 37 windows) was
> decoded separately. The features were every electrode's samples in that window.
> The classifier was principal component analysis (components explaining 80% of the
> training variance, refit in each fold) followed by linear discriminant analysis
> with equal class priors. We used stratified five-fold cross-validation, repeated
> ten times. Folds were stratified on the four cells, so every test fold was
> balanced on both labellings. Accuracy was the mean of the two classes' hit rates.
>
> **Cross-decoding.** On each fold we trained one classifier on congruency
> (incongruent vs congruent) and one on switch type (switch vs repeat). Each was
> scored on held-out trials against both labellings. Scoring against the training
> labelling gives the within-contrast accuracy (the *ceiling*). Scoring against the
> other labelling gives the *transfer*. Incongruent was paired with switch and
> congruent with repeat, so a shared axis on which the harder condition of each
> contrast falls on the same side yields above-chance transfer. Each transfer was
> expressed as the share of its ceiling's above-chance accuracy that it retained,
> (transfer − 0.5) / (ceiling − 0.5), averaged over the windows in which the ceiling
> was significant.
>
> **Statistics.** For each decode, the null distribution came from permuting the
> training labels within each fold and refitting. True-label accuracies (ten CV
> repeats) were compared with the null by a one-tailed cluster-based permutation
> test over windows (500 permutations, α = 0.05). The same test compared each
> transfer with its ceiling. Windows centred at or before −0.125 s, which contain no
> post-stimulus sample, served as a check on artifacts. Because the samples entering
> these tests are CV repeats of a single pseudopopulation rather than participants,
> we treat the window-wise results as a within-dataset reliability measure, not as
> population inference.
>
> **Controls.**
>
> - *Response time.* Incongruent and switch trials were slower. In each
>   participant, RTs were pooled across the four cells and cut into ten quantile
>   bins. Within each bin we kept, at random, the same number of trials from each
>   cell. This left 48% of trials and removed the RT costs (incongruent − congruent:
>   +160 → +2 ms, p = 0.45; switch − repeat: +195 → +2 ms, p = 0.39; across 24
>   participants). A control drew the same number of trials per participant and
>   cell without regard to RT, keeping the RT costs (+156 and +208 ms). The
>   RT-matched result is compared with this control, not with the full-trial result.
> - *Overall activity.* To ask whether the transfer reflected a uniform change in
>   activity, we subtracted each participant's mean across its electrodes, per
>   pseudo-trial and time point, before decoding. Participants contributing a single
>   electrode carry no information after this step (2 of 21).
>
> **[Positive controls, the response-locked analysis and the seed repeats go here
> once run.]**

Limitations to state with it (same source): the samples are CV repeats of one
pseudopopulation with no estimate across participants; a pseudopopulation code
is an upper bound on what any one participant's lPFC shares; the activity
control removes only a shift common to all of a participant's electrodes; and
these are the base effects, not their adaptation.

### 2.7 Brain–behavior (S-BB)

*Copied from [`methods.md` › A6](methods.md#a6-brainbehavior-supplement-s-bb).*

> **Participants and electrodes.** We asked whether participants whose lateral
> prefrontal high-gamma activity adapted more to the proportion manipulations
> also adapted more in their behavior. The analysis used the task-significant
> electrodes in lateral prefrontal cortex (171 electrodes from 21 participants;
> task significance as defined in **[main Methods section]**). These electrodes
> were selected for overall task responsiveness, not for an LWPC or LWPS effect.
> The high-gamma epochs and preprocessing were those of the anatomical analyses.
> Only correct trials with a recorded response time (RT) were used. A participant
> entered the analysis if it had at least three usable electrodes (18
> participants; median 7 electrodes and 397 trials each) and a behavioral score;
> one of the 18 was absent from the behavioral summary table **[reason]**,
> leaving 17 participants.
>
> **Neural and behavioral scores.** For each electrode, single-trial high gamma
> was averaged over 0–1.5 s after stimulus onset. LWPC and LWPS were scored as in
> the anatomical analyses: the equal-cell-weighted difference of differences (the
> condition effect in the low-proportion blocks minus that in the high-proportion
> blocks), divided by the pooled within-cell standard deviation. Here each score
> was computed once from all of the electrode's trials rather than on split
> halves. An electrode was usable if all four of its scores (LWPC and LWPS,
> unadjusted and RT-adjusted, below) were defined. A participant's neural LWPC
> and LWPS were the unweighted means of its usable electrodes' scores. Equal
> weights are appropriate because a participant's electrodes share its trials
> and so have similar sampling error.
>
> The behavioral LWPC and LWPS were the same differences of differences computed
> on mean RT, in milliseconds, from the behavioral analysis **[section reference;
> trial inclusion as described there]**. Both scores were oriented so that
> positive values indicate the predicted adaptation: a smaller congruency effect
> or switch cost in the high-proportion blocks.
>
> **Removing the RT-linked component of high gamma.** The analysis window covers
> most responses (median RT 1.19 s), so single-trial high gamma may track RT. If
> it does, every cell mean of high gamma carries the same multiple of that cell's
> mean RT. Each electrode's neural difference-of-differences then contains its
> slope on RT times the participant's own behavioral difference of differences.
> That term would correlate neural with behavioral adaptation across participants
> with no link between them beyond the trial-by-trial coupling, and it would also
> pass the specificity checks below.
>
> We therefore removed the RT-linked component of high gamma separately for each
> electrode. The slope of high gamma on RT was estimated from deviations around
> each of the 16 cell means of the design (congruency × task sequence ×
> incongruent proportion × switch proportion), pooled across cells, so that
> condition effects, which move both high gamma and RT, did not enter it. Each
> trial's high gamma was replaced by its value minus the slope times the trial's
> deviation from the electrode's mean RT, and the scores were recomputed. This
> removes exactly the slope times the behavioral difference of differences from
> each electrode's score. The adjustment is conservative: if neural adaptation
> reaches behavior through the same trial-by-trial coupling, that part is
> removed too. We therefore treat the RT-adjusted correlation as the primary test
> and report the unadjusted correlation as an upper bound.
>
> **Statistical analysis.** For each effect, the neural and behavioral scores
> were related across participants by Pearson correlation (two-sided, α = .05;
> |r| ≥ 0.48 needed at n = 17), with a 95% confidence interval from Fisher's *z*
> and Spearman's ρ as a rank-based check. Behavioral LWPC and LWPS are
> correlated across participants, so a neural score could relate to both. To
> test specificity, each behavioral score was regressed (ordinary least squares,
> all variables *z*-scored) on both neural scores together, and the coefficient
> of the matched neural score was compared with that of the other. The two
> RT-adjusted matched correlations were the primary tests. **[If the cross
> pairings are reported, state that p-values are uncorrected across the eight
> correlations: matched and cross, adjusted and unadjusted.]**
>
> **Reliability and the correlation ceiling.** An observed correlation cannot
> exceed the square root of the product of the two scores' reliabilities. We
> estimated each score's reliability by splitting every participant's trials
> into random halves, stratified on the 16 design cells, 200 times. Each split
> was drawn once per participant and applied to all of its electrodes. Drawing
> it separately per electrode would let noise common to a participant's
> electrodes masquerade as reliability. All scores were recomputed in each half.
> The half-length reliability was the across-participant correlation of half-A
> with half-B values, averaged over splits, and was stepped up to full length
> with the Spearman–Brown formula. It was treated as unmeasurable when the
> half-length correlation was zero or negative. The behavioral summary table has
> no trial-level data, so its reliability was estimated from the same contrasts
> computed on the RTs of the recorded trials. These agreed closely with the
> table across participants (r = 0.90 for LWPC and 0.76 for LWPS, 20
> participants).
>
> The ceiling on each brain–behavior correlation was the square root of the
> product of the neural and behavioral reliabilities. To express what the
> ceiling means for detection, we simulated 40,000 samples of 17 participants
> from a bivariate normal distribution whose correlation was the ceiling times
> an assumed true correlation, and counted the samples reaching |r| ≥ 0.48. The
> sample size needed for 80% power was obtained from Fisher's *z*. Both treat
> the estimated reliabilities as known, so they are approximate.

The optional paragraph on the exploratory participant summaries (mean |d|,
positive electrodes only) is in the source. Use it only together with the
optional Results sentence in §3.7.

---

## 3. Results

### 3.1 Concurrent adaptation in behavior (F1) ⬜

**[To write.]** Report both adaptations in RT and errors, in the same
participants and sessions, then one scoping sentence: "Neither proportion
manipulation affected the other process (congruency × switch proportion,
switch type × incongruent proportion; Supplementary S4), so we focus on the two
within-process adaptation effects." Check the source script first (§1.4, F1).

### 3.2 Adaptation effects in lPFC high gamma (F3) ⬜

**[To write from the ANOVA clusters and the direction tests.]** In this order:

1. The electrode set: `n_electrodes` and `n_subjects`.
2. LWPS, then LWPC: the interaction cluster's window, sign and p; each simple
   effect's sign and cluster; the per-participant tally and the leave-one-out
   range; the direction against behavior.
3. One sentence for the off-diagonal that states what was tested and no more,
   e.g. "No cluster survived for congruency × switch proportion or for switch
   type × incongruent proportion (Fig. 3b, off-diagonal)." Not "was not
   modulated by" (§1.4, F3).
4. The RT check (run 2026-10-05; N4 §19.8): "After removing the component of
   high gamma linearly related to RT within condition cells, both adaptation
   effects remained in the behavioral direction across participants (LWPC:
   mean *d* = 0.08, *t*-test *p* = 0.007, 16 of 21 participants positive;
   LWPS: 0.12, *p* = 0.007, 15 of 21), keeping 72 % and 77 % of their
   unadjusted size." Window-mean (0–1.5 s), task-significant electrodes; add
   the 0–0.5 s rerun if made.

Do not read onset times off the bars (§1.4, F3).

### 3.3 Decoding the adaptation effects (F4) ⬜

**[To write from the decoding runs.]** For each effect: when congruency (switch
type) is decodable in each block type, peak accuracy, the cluster where the two
block types differ, trial counts per class, and any pre-stimulus windows.

### 3.4 How the two adaptations are organized across lPFC (F5)

*Restructured on 2026-10-01 from §17.3 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#173-results) for the
two-panel F5 (§1.4). It no longer copies §17.3: the order and emphasis differ.
The numbers are §17.3's and the new ones are from §16.6.2 and §16.6.5 of that
doc. All were checked against the run's outputs on 2026-10-01; §17.5 there
gives each number's source file. The closing subset sentence is from §18 of
that doc (sources in §18.9). The parcel and gradient detail that left the
main text is in §3.5 below.*

*2026-10-02, for the combined F5 (§1.4): the overlap is Fig. 5b; the
matched-vs-crossed paragraph is cut to its first sentence, with the bars in
S-N4; the gradient paragraph points to Fig. 5a, c and d. Bold brackets mark the
numbers §19 of the N4 doc will supply. If F5e goes in, its paragraph is §3.6's
first one, shortened.*

> **LWPC and LWPS share electrodes.** We scored every lPFC electrode (398
> electrodes, 22 participants) for LWPC and LWPS. Across this anatomically
> defined set both adaptation effects were small on average (mean Cohen's
> *d* = 0.01 for LWPC and 0.05 for LWPS; 49 % and 56 % of electrodes positive),
> and single-electrode estimates were noisy (full-data split-half reliability
> 0.27–0.30), so we tested only population-level summaries. The two effects
> shared electrodes. LWPC and LWPS scores from separate halves of the trials
> were positively correlated within participants (Spearman *r* = 0.10,
> *p* < 0.001; 397 electrodes, 21 participants; Fig. 5b), and electrodes
> positive for each effect did not differ in location (centroid distance
> 1.4 mm, *p* = 0.95). With participants as the unit, the correlation held
> when each participant's own correlation was weighted by its electrode count
> (*r* = 0.10, 95 % CI [0.03, 0.16], sign-flip *p* = 0.031; 20 participants
> with at least four electrodes) but not when every participant counted
> equally (*r* = 0.04, *p* = 0.35; 11 of 20 positive), so it describes the
> electrode population rather than every participant (N4 §19.8).
> **[Local similarity, worded by its outcome; rerun needed with shared splits,
> N4 §19.3 (the 2026-10-05 numbers are not usable):
> neighbouring electrodes (< 10 mm) shared … of their reliable LWPC signal and
> … of their LWPS signal, and … of the balance between the two (difference
> 95 % CI […, …]). Only if the balance is flat while the single scores rise:
> "so the two effects are intermixed at the scale of the recordings".]** The base effects that the two adaptations act on,
> congruency and switch type, were scored from the same trials and halves with
> the proportion blocks weighted equally. They also shared electrodes
> (*r* = 0.23, *p* < 0.001).
>
> **Each adaptation tracks the effect it regulates.** Electrode by electrode,
> on separate trial halves, each adaptation effect correlated more with its
> own base effect than with the other one (congruency–LWPC *r* = 0.22 vs
> switch–LWPC 0.12; switch–LWPS 0.17 vs congruency–LWPS 0.13; Supplementary
> **[S-N4]**). We
> tested this difference with the correlation between congruency − switch and
> LWPC − LWPS, whose covariance equals the two matched covariances minus the
> two crossed ones, so that anything common to all four scores cancels. It was
> positive (*r* = 0.09, *p* < 0.001) and held with MNI coordinates partialled
> out (*r* = 0.09, *p* < 0.001), so the link is local rather than a gradient
> that the two maps share. For LWPS the matched base effect was the less
> reliable map (within-participant split-half reliability 0.23 for switch,
> 0.35 for congruency **[replace with the shared-split values,
> `reliability_by_split_scheme.csv`; N4 §19.3: the per-electrode split biases
> these]**) and still correlated more, so that difference is not
> one of reliability. All four correlations were positive: electrodes with
> larger base effects of either kind adapted more on both.
>
> **The balance between the two adaptations varies modestly across lPFC.** The
> balance between the two effects (LWPC − LWPS within each electrode) differed
> across Destrieux parcels (19 parcels sampled in at least three participants;
> *F* = 1.91, label-exchange permutation *p* = 0.010; *p* = 0.003–0.057 with
> each participant left out), although no single parcel differed from zero
> after FDR correction (all *q* ≥ 0.13). It also varied with position (MNI
> coordinates: block *F* = 2.85, *p* = 0.031): relative to LWPS, LWPC was
> weaker dorsally (*z* slope *p* = 0.008; Bonferroni-corrected over three
> axes, *p* = 0.023). The predicted anterior–posterior axis showed no effect
> (*p* = 0.58), and height explained about 2 % of the balance's variance
> (Fig. 5a, c, d). With participants as the unit the gradient held, though
> less strongly and unevenly across participants: the participants' own
> slopes, weighted by their electrodes' spread in height, averaged −0.0077
> SD/mm (95 % CI −0.0147 to −0.0013; sign-flip *p* = 0.041), and a mixed model
> with a random height slope gave −0.0083 SD/mm (*p* = 0.037; between-
> participant SD of the slope 0.010 SD/mm). With each participant left out,
> *p* ranged from 0.002 to 0.099 (N4 §19.8). This gradient and its relation to
> the base effects are described in Supplementary **[S-N4]**.
>
> Together, LWPC and LWPS adaptation are carried by one intermixed lPFC
> population, and each tracks the local strength of the effect it regulates.
> The task-responsive subset (171 electrodes, 21 participants) gave the same
> picture at lower power: both pairs of effects shared electrodes (LWPC–LWPS
> *r* = 0.08, *p* = 0.055; congruency–switch *r* = 0.17, *p* = 0.001), and
> each adaptation effect tracked its own base effect more closely than the
> other by the same margins as in all lPFC (*r* = 0.08, *p* = 0.049). The
> balance between LWPC and LWPS did not vary detectably across parcels or with
> position there, as expected at that set's power (Supplementary **[S-N4]**).

### 3.5 Anatomy supplement (S-N4)

*The first paragraph is copied from §18.8 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#188-results-text-supplement-s-n4),
the task-significant run with main effects (2026-10-01); §18.9 there gives
each number's source. The next two paragraphs ("The adaptation gradient", "The
base-effect balance") are the §17.3 text that left the main text on 2026-10-01
(§3.4 above). The others are written here from the numbers in §16.6.6 and the
§17.3 supplement additions of that doc.*

> **Task-responsive subset.** We repeated the anatomical analyses on the
> task-responsive lPFC electrodes (171 electrodes, 21 participants), with the
> trials split 200 times. All four effects were positive on average (mean
> Cohen's *d* = 0.17 for congruency, 0.12 for switch type, 0.14 for LWPC and
> 0.18 for LWPS), and single-electrode adaptation estimates were again noisy
> (full-data split-half reliability 0.26–0.39). At both levels the two effects
> shared electrodes: congruency and switch-type scores from separate halves of
> the trials were correlated (Spearman *r* = 0.17, *p* = 0.001), and LWPC and
> LWPS scores weakly so (*r* = 0.08, *p* = 0.055; 167 electrodes from the 18
> participants with at least three). Electrodes positive for each adaptation
> effect did not differ in location (centroid distance 3.3 mm, *p* = 0.44).
> Each adaptation effect again correlated more with its own base effect than
> with the other one (congruency–LWPC *r* = 0.13 vs switch–LWPC 0.04;
> switch–LWPS 0.15 vs congruency–LWPS 0.12). The differences were the size
> they were in all lPFC electrodes (0.09 and 0.03, against 0.10 and 0.04), and
> the test of them was at the threshold of significance (*r* = 0.08,
> *p* = 0.049; with MNI coordinates partialled out, *r* = 0.08, *p* = 0.052).
> In this subset the congruency and switch-type maps were equally reliable
> (within-participant split-half reliability 0.25 and 0.26), so the larger
> difference for LWPC does not reflect a more reliable congruency map. Neither
> balance differed across Destrieux parcels (15 parcels; LWPC − LWPS:
> *F* = 0.91, *p* = 0.29; congruency − switch: *F* = 1.15, *p* = 0.12; with
> each participant left out, *p* = 0.10–0.45 and 0.03–0.30) or varied with
> position (MNI coordinates: *F* = 1.44, *p* = 0.28, and *F* = 0.88,
> *p* = 0.48). Their height slopes matched those in all lPFC (LWPC − LWPS:
> −0.0075 SD/mm, *p* = 0.23; congruency − switch: −0.0035 SD/mm, *p* = 0.53;
> all lPFC −0.0077 and −0.0037). The adaptation slope did not differ from that
> of the remaining electrodes (*p* = 0.67), and with the all-lPFC slope
> planted, this electrode layout reaches *p* < 0.05 in 26 % of simulations,
> against 66 % for all lPFC electrodes. In this subset, distance from the
> midline did not describe the gradient better than height (both *p* ≥ 0.44),
> and LWPC did not vary with distance from the midline (*p* = 0.82). Because
> the adaptation balance had no detectable gradient here, we did not test
> whether it is inherited from the base effects.
>
> **The adaptation gradient.** In all lPFC, the dorsal weakening of LWPC
> relative to LWPS had a *z* slope of −0.0077 SD/mm, and it replicated across
> independent halves of the trials (*p* = 0.005). Descriptively, the superior
> frontal gyrus and sulcus leaned toward LWPS. Height and distance from the
> midline are correlated across lPFC electrodes (within-participant
> *r* = −0.58), so the two cannot be fully separated. In an exploratory model,
> distance from the midline described the gradient better than height
> (distance *p* = 0.019, height *p* = 0.53), so we describe it as dorsomedial
> versus ventrolateral. In an exploratory breakdown by score, LWPC carried the
> gradient: it was slightly reversed within about 27 mm of the midline
> (adjusted *d* = −0.07) and positive farther out (0.04–0.06; slope
> *p* = 0.006), whereas LWPS did not vary (*p* = 0.75; Supplementary Fig.
> **[S-N4, scores by distance from the midline]**).
>
> **The base-effect balance.** The balance between the base effects
> (congruency − switch) also differed across parcels (*F* = 1.70, *p* = 0.017;
> *p* ≤ 0.058 with any one participant left out), in step with the adaptation
> balance (*r* = 0.73 across the 19 parcel means; same sign in 13;
> Supplementary Fig. **[S-N4, balances by parcel]**). The parcel means share
> trials, so this correlation is descriptive. The base-effect balance showed no
> significant gradient (*F* = 1.81, *p* = 0.13); its slope pointed the same way
> at about half the size. Neither congruency nor switch varied detectably with
> distance from the midline (*p* ≥ 0.34).
>
> **Hemispheres.** Fitted within each hemisphere, the adaptation balance varied
> with position in the left (*F* = 4.22, *p* = 0.005; 254 electrodes) but not
> detectably in the right (*F* = 2.33, *p* = 0.12; 144 electrodes). The
> base-effect balance likewise varied in the left (*F* = 2.45, *p* = 0.039) but
> not the right (*F* = 0.68, *p* = 0.45).
>
> **Is the gradient inherited?** Adding the base-effect balance as a covariate
> reduced the height slope of the adaptation balance from −0.0077 to −0.0069
> SD/mm (*p* = 0.015 with the covariate) and the distance-from-midline slope
> from +0.0139 to +0.0125 SD/mm (*p* = 0.030), a 10 % reduction in both. With
> the base-effect balance taken from the opposite trial half, the height slope
> fell by 5 %. Because the base-effect balance had a split-half reliability of
> only 0.08 (0.15 on full data), a gradient carried entirely by it would shrink
> by only about 8–15 %. The participant-bootstrap 95 % intervals for the
> reduction (height −20 % to 40 %; distance −15 % to 46 %) include both no
> inheritance and full inheritance.
>
> **Map similarity.** Pooled across participants, the separate-half LWPC–LWPS
> correlation (Pearson *r* = 0.17) was about the size of the two maps' half-data
> reliabilities (0.18 for LWPC, 0.16 for LWPS). At these reliabilities a
> noise-corrected ratio is not estimable, so we report the three values side by
> side.

Figures and tables for S-N4: the two balances by parcel (old F5b;
`panel_b_label_means.csv`, script §4), the four scores by distance from the
midline (old F5c; `panel_c_midline.csv`, script §3) and by height
(`panel_c_height.csv`), per-parcel bars (`delta_by_roi.png`, `dm_by_roi.png`),
coverage (`coverage_matrix.csv`), leave-one-out tables (`delta_roi_loso.csv`,
`dm_roi_loso.csv`), both coordinate tables (`score_anatomy.json` →
`coordinates`, `dm_coordinates.csv`), the height vs midline models (script §1),
and Test 2 (`tilt_with_dm.csv`, script §5). For the task-responsive subset, the
same tables from its own anatomy folder, and its matched-vs-crossed bars beside
the all-lPFC ones (its `delta_tracking.csv`; N4 §18.4). The design notes for the two old F5
panels still apply: participant means ± SEM, no electrode scatter, and the
scores as matched small multiples with shared axes and one legend
(§16.7.3 of the N4 doc).

### 3.6 Congruency ↔ switch-type cross-decoding (S5)

*Copied from [`methods.md` › A4](methods.md#results-draft), Results (draft). It
states only what the full-trial runs support. Choose the bracketed variants once
the seeds and the electrode-matched region comparison are in.*

> Congruency and switch type were each decodable from the task-responsive lPFC
> population. Congruency was decodable from the window centred at +0.12 s, with
> peak accuracy 0.76. Switch type was decodable from +0.31 s, with peak 0.76. A
> congruency decoder also predicted switch type, and a switch-type decoder
> predicted congruency, but only from the window centred at +0.62 s (covering
> 0.50–0.75 s) onward, and well below the within-contrast accuracy. The
> congruency decoder retained 47% of switch type's above-chance accuracy, and
> was below it in 19 of 37 windows. The switch-type decoder retained 26% of
> congruency's, and was below it in 28 windows. No window before the stimulus
> was significant in any of the four decodes. Removing each participant's mean
> activity lowered both within-contrast accuracies (peaks 0.63 and 0.65) but
> left the transfers nearly unchanged (15 and 14 windows; 81% and 77%
> retained). The shared component is therefore not a uniform rise in activity.
> The two contrasts thus engage the same electrodes along largely distinct
> population codes, sharing a component that appears only late in the trial.
>
> **[One of the following, depending on the seeds:]** *(if the RT-matched −
> random gap exceeds the seed spread)* After matching RTs across the four
> cells, the transfer was reduced relative to a trial-count control
> (congruency → switch: 43% vs 70% retained; switch → congruency: 12% vs 39%)
> and confined to 0.5–1.1 s after the stimulus, before most responses (median
> RT 1.17 s). Part of the shared component therefore reflects the RT difference
> shared by incongruent and switch trials. *(otherwise)* RT matching did not
> change the transfer beyond the variability between pseudopopulation draws.
>
> **[Region sentence, once lPFC has been subsampled to 54 electrodes:]**
> Occipital electrodes showed a transfer retaining a similar share of their
> ceilings (54% and 23%), but at +1.0 s and later rather than at 0.5–1.0 s.

⚠️ Before quoting the RT-matched switch → congruency share (12 %), recompute it
over post-stimulus windows only: 6 of that ceiling's 20 significant windows are
pre-stimulus ([`decoding.md` › A4 §13.8.3](decoding.md#138-results-2026-10-01)).

### 3.7 Brain–behavior (S-BB)

*Copied from §14.2 of [`a6_brain_behavior.md`](a6_brain_behavior.md#142-results-supplement-s-bb).*

> **Brain–behavior.** We asked whether participants whose lateral prefrontal
> high gamma adapted more also adapted more in behavior. Of the 21 participants
> with task-significant lPFC electrodes, 17 had at least three usable
> electrodes and behavioral scores (median 7 electrodes and 397 trials per
> participant). Behavioral adaptation was reliable across participants
> (split-half reliability, estimated on the recorded trials: 0.72 for LWPC,
> 0.56 for LWPS). Participant-level neural adaptation was not: neural LWPC had
> no measurable between-participant reliability, and neural LWPS had a
> reliability of 0.38. These reliabilities cap the observable LWPS correlation
> at 0.46, below the |r| = 0.48 needed for significance at n = 17. Even a
> perfect underlying relationship would have reached significance in fewer than
> half of samples of this size (simulated power 0.49), and a moderate one
> (r = 0.5) in 14 %. For LWPC, no correlation was detectable at all. Neither
> correlation differed from zero after removing the RT-linked component of high
> gamma (LWPC: r = 0.10, 95 % CI [−0.40, 0.55], p = 0.70; LWPS: r = −0.06
> [−0.52, 0.44], p = 0.83; Supplementary Fig. S-BB) or without that adjustment
> (LWPC: r = 0.24 [−0.27, 0.65], p = 0.35; LWPS: r = 0.19 [−0.32, 0.61],
> p = 0.47). The unadjusted correlations were somewhat larger, as expected if
> part of each neural score reflects trial-by-trial coupling between high gamma
> and RT (median within-cell r = 0.13). With these intervals, the data neither
> support nor rule out a relationship between individual differences in neural
> and behavioral adaptation.

Optional sentence, only if the exploratory summaries are in the Methods:

> Two alternative participant summaries, the mean absolute score and the mean
> over positively scoring electrodes, gave no interpretable result: the first
> tracked participants' trial counts (LWPS: r = −0.61), as expected of a score
> driven by noise, and the second retained only 11 participants.

---

## 4. Discussion sentences

**Base-effect codes (S5).** *Written here from the "supported now" statement in
[`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01). It goes next
to the N4 overlap result.*

> The same task-responsive lPFC electrodes carried both base effects,
> congruency and switch type, along largely distinct population codes: a
> decoder trained on one recovered only part of the other's decodable
> information, and only from about 0.5 s after the stimulus (Supplementary
> S5).

**Brain–behavior (S-BB).** *Copied from §14.3 of
[`a6_brain_behavior.md`](a6_brain_behavior.md#143-discussion-sentence).*

> We could not test whether individual differences in lPFC adaptation track
> individual differences in behavioral adaptation. With 17 participants, the
> split-half reliability of participant-level difference-of-differences scores
> left any across-participant correlation undetectable, even for a strong
> underlying relationship (Supplementary Note S-BB). Difference scores are
> characteristically unreliable across individuals (Hedge et al., 2018), so such
> a test needs many more participants, many more trials per participant, or
> both.

**What the matched correlations mean (Results sentence; bars in S-N4).** *Written here from §16.6.5 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md). It goes after the
overlap result.*

> Each adaptation effect was larger at electrodes with a larger base effect of
> its own kind. A proportional reduction would produce this: if each
> electrode's congruency effect shrinks by a similar fraction in mostly
> incongruent blocks, electrodes with larger congruency effects lose more in
> absolute terms. The result argues against adaptation being expressed mainly
> by electrodes that do not carry the effect being regulated.

**Anatomy limits to state** (§17.4 of the N4 doc):

- The effects are small: single-electrode reliability is ~0.3, LWPC averages
  *d* ≈ 0.01 across all lPFC, and the matched–crossed gaps are 0.04–0.10 in r.
- For LWPC the matched base effect (congruency) is also the more reliable map
  in all lPFC, so reliability could contribute to that pair's gap there; the
  LWPS pair rules reliability out only for LWPS. In the task-responsive subset
  the two base-effect maps are equally reliable (0.25 and 0.26 within
  participant) and LWPC's gap is the same size (0.09 against 0.10), which
  argues against reliability for LWPC too (N4 §18.2). That is descriptive: no
  test compares the two pairs' gaps, and the subset's overall test is at the
  threshold (*p* = 0.049).
- The task-responsive subset reproduces F5 at lower power, but not the
  gradient's midline description (N4 §18.6), which stays exploratory.
- The gradient (main text one paragraph, detail in S-N4): no parcel survives
  FDR, height explains ~2 % of the balance's variance, the axis was not
  predicted and the predicted anterior–posterior axis is null. Inheritance
  could not be resolved; it is stated in S-N4 only.

---

## 5. Figure captions

### 5.1 Fig. 3 (draft)

*Drafted here from the F3 specification in §1.4. It describes the single trace
encoding of fix 1 there, not the current figure.*

> **Fig. 3 | lPFC high gamma carries both adaptation effects in the behavioral
> direction.** **a**, Task-responsive lPFC electrodes (**[n]** electrodes from
> **[N]** participants) on the MNI template **[; right-hemisphere electrodes
> mirrored onto the left]**. **b**, High gamma (z) relative to stimulus onset,
> mean across electrodes ± SEM. Rows: switch cost (switch vs repeat, top) and
> congruency effect (incongruent vs congruent, bottom). Columns: switch
> proportion (left) and incongruent proportion (right). Colour: proportion in
> the block (light, 25 %; dark, 75 %). Line style: condition (dashed, switch or
> incongruent; solid, repeat or congruent). Each trace weights the two levels
> of the other proportion equally. The diagonal shows the two adaptation
> effects, LWPS (top left) and LWPC (bottom right); the off-diagonal shows the
> cross-effects. Bars mark clusters in which the condition × proportion
> interaction was significant (ANOVA in 250-ms windows, permutation null
> within electrodes, cluster-corrected). Both have the sign of behavioral
> adaptation: a smaller condition effect in high-proportion blocks. No cluster
> survived off the diagonal. Electrodes are pooled across participants, so the
> cluster p-values are optimistic; per-participant tallies are given in the
> text.

### 5.2 Fig. 5 (draft)

*Redrafted 2026-10-02 for the combined F5 (§1.4). The title is the provisional
ending sentence (§1.1): "intermixed" waits for the local-similarity result and
"separable axes" for the A4 controls. Numbers in bold brackets come from the
§19 run. The 2026-10-01
caption (overlap at both levels, and the bars) is kept below for S-N4.*

> **Fig. 5 | LWPC and LWPS adaptation share an intermixed lPFC population with
> a shallow dorsal–ventral bias, over which congruency and switch type are
> coded along largely separable axes.** **a–d**, All lPFC electrodes (398
> electrodes, 22 participants). **a**, Electrodes on a sagittal projection,
> coloured by tertile of height (MNI *z*; cuts at **[…]** and **[…]** mm).
> **b**, LWPC against LWPS: each point is one electrode's score from all trials,
> after regressing out overall responsiveness and subtracting each
> participant's mean, with each score's overall mean added back (397
> electrodes from the 21 participants with at least three; SD units). Colour as
> in **a**. Points above the identity line lean toward LWPS, points below it
> toward LWPC. *r* is the pre-specified test: one score from one half of the
> trials against the other score from the other half, averaged over 1,000
> random splits (Spearman; within-participant permutation null). Because each
> half has half the trials, *r* is smaller than the correlation among the
> plotted points. The box marks **c**. **c**, Each tertile's centroid, with its
> 95 % region from a participant bootstrap. The two effects' shared variation
> runs along the identity line; the height gradient is the centroids' spread
> across it. **d**, LWPC − LWPS by height tertile, with participant and
> responsiveness offsets removed; mean ± SEM across participants. The slope is
> from the pre-specified coordinate model (within-electrode label-swap null),
> with **[its participant-level test]**. **e**, Congruency and switch type
> decoded from task-responsive lPFC (**[n]** electrodes, **[N]** participants):
> each within its own labels (ceiling) and each decoder tested on the other's
> labels (transfer), in 250-ms windows. **[Bars: windows above the shuffle
> null. Shading: early windows where both are decodable and the transfer is
> not; late windows of partial transfer.]**

**S-N4 (the 2026-10-01 panels):**

> **The overlap at both levels and each adaptation against its base effects.**
> **a**, Congruency against switch-type main effect, and LWPC against LWPS,
> scores and test as in Fig. 5b. **b**, Electrode-level correlations, on
> separate trial halves, between each adaptation effect and its own base effect
> (filled; congruency–LWPC, switch–LWPS) or the other base effect (open;
> switch–LWPC, congruency–LWPS). The annotated test is the correlation between
> congruency − switch and LWPC − LWPS, whose covariance is the two matched
> covariances minus the two crossed ones; it is given with and without MNI
> coordinates partialled out. Spearman *r*, within-participant permutation *p*.

### 5.3 Supplementary Fig. S-BB

*Copied from §14.4 of [`a6_brain_behavior.md`](a6_brain_behavior.md#144-figure-caption-supplementary-fig-s-bb).
Use `participant_brain_behavior_scatter_rtadj`, annotated with r, its 95 % CI
and the ceiling instead of R² (R² drops the sign; LWPS is r = −0.06).*

> **Supplementary Fig. S-BB | Neural against behavioral adaptation across
> participants.** Each point is one participant (n = 17). *x*: behavioral LWPC
> (left) and LWPS (right), the difference of differences in mean RT (ms), low-
> minus high-proportion blocks. *y*: the same contrast in high gamma (0–1.5 s
> after stimulus onset), as Cohen's *d* averaged over the participant's
> task-significant lPFC electrodes, after removing the component of high gamma
> linearly related to RT within condition cells. Positive values on both axes
> indicate the predicted adaptation. Lines are least-squares fits. Each panel
> gives Pearson's r with its 95 % CI and p. The ceiling (LWPS: 0.46) is the
> largest correlation expected for a perfect underlying relationship, given the
> split-half reliabilities of the two scores. Neural LWPC had no measurable
> reliability, so no ceiling is given. Significance at α = .05 required
> |r| ≥ 0.48.

### 5.4 Supplementary S-N4: the two panels that left F5 (draft)

*The old Fig. 5 caption's panels b and c, kept when they moved to S-N4.*

> **The two balances by Destrieux parcel.** Adjusted mean congruency − switch
> (*x*) against adjusted mean LWPC − LWPS (*y*), one point per parcel sampled
> in at least three participants (19 parcels), sized by electrode count and
> coloured by distance from the midline. *F* and *p* are each balance's parcel
> test (label-exchange permutation). The parcel means share trials, so their
> correlation is descriptive.
>
> **The four scores by distance from the midline.** Congruency, switch, LWPC
> and LWPS (Cohen's *d*, participant and responsiveness removed) by tertile of
> distance from the midline; mean ± SEM across participants. Exploratory.

### 5.5 Still to write

F1, F2 and F4 captions, Supplementary S5 (cross-decoding traces with ceilings,
transfer and control runs), and the rest of S-N4.
