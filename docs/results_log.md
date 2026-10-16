# Results log

Newest entry first. One entry per batch of cluster jobs: what finished, what it
showed, what it means for the paper, and what to do with it. Details live in
the owning docs linked from each block.

---

## 2026-10-16: coding axes, response-locked

Drafted by Claude; not yet reviewed.

Jobs: `coding_axes_real` 58783103–58783108 (6), all COMPLETED, exit 0, in
17–58 min. Owning doc: [`coding_axes.md`](coding_axes.md) §11. The controls
figure (`coding_axes_controls.png` / `.csv`) was made the same day from these
six runs.

### Short version

1. **The codes and their shared part are present before the response,
   strongest 0.4–0.75 s before it, fading toward it, and gone after it.** All
   lPFC: shared from −0.875 to −0.06 s, alignment 0.59 [0.42, 0.83] (0.75–0.78
   at its peak); θ₁ 61° at 0.76 [0.70, 0.93] of its ceiling, as
   stimulus-locked.
2. **RT matching removes the codes in this frame too** (3 reliable windows
   against 8 in the random subset; neither variable has a reproducible
   dimension). `rt_adjust` again keeps the sharing (0.52).
3. **On the A4 response-locked reading table this is closest to the latency
   signature, but not cleanly**: sharing lasts to the last pre-response window,
   and the ramp's peak height differs by cell. Supports "tied to RT, present
   before the response", not "a timing artifact".
4. **Occipital: no reliable code at all** (83 electrodes, 14 participants), so
   no answer on lPFC specificity.

### coding_axes_real (response-locked): main, remove_mean, rt_adjust, rt_match10, rt_random10, occ

- **Status:** all six COMPLETED. `Response_-1.0to1.5sec_…_zmax_20` epochs,
  windows −1.0 to 1.5 s (37, 0 = response), trajectory axes −1.0–0, −1.0–−0.5
  and −0.5–0 s, all lPFC (398 / 22) or all occipital (83 / 14), seed 0, 200
  splits, 1000 × 20 shuffles, 2000 bootstraps, `K_MAX` 3. 58783106–08 started
  after today's block-separation edits and also wrote `block_*` files;
  58783103–05 did not. The cosine and subspace code was the same for all six
  (coding_axes.md §11.1).
- **Result:**
  - Main: 16 reliable windows (−0.875 to +0.06 s), cross 0.25 against a
    ceiling of 0.41, **alignment 0.59 [0.42, 0.83]**, shared −0.875 to
    −0.06 s, not one code −0.81 to −0.125 s and 0.0–0.06 s. Nothing reliable
    after +0.06 s. Subspace: k = 3 / 3 (capped), θ₁ 0.48 (61°) against 0.64,
    **0.76 [0.70, 0.93]**, shared, not one code; alignment index 0.56
    [0.50, 0.85] (C → S), 0.58 [0.47, 0.91] (S → C). At k = 1, θ₁ 0.735 against
    0.72 stimulus-locked. Trajectories (−1.0–0 s axes): signal fraction 0.44
    [0.33, 0.59]; cos(c, s) 0.37 for early-half axes vs 0.30 late; switch
    moves along the congruency axis by 52–76 % as much as congruency does.
  - `remove_mean`: 0.45 [0.25, 0.67], shared −0.875 to −0.06 s; θ₁ 0.74
    [0.67, 0.89].
  - `rt_adjust`: 0.52 [0.30, 0.80], shared −0.75 to −0.25 s, ceilings about
    halved; θ₁ 0.79 [0.72, 1.00], shared.
  - `rt_match10` / `rt_random10` (seed 0): RT gaps −0.35 / −0.38 ms after
    matching (p = 0.86), 170 / 182 ms kept in the random subset; 47 % of
    trials each. Matched: 3 reliable windows, ceiling 0.14, k = 0 / 0.
    Random: 8 windows, ceiling 0.24, alignment 0.36 [−0.04, 0.93], shared
    −0.75 to −0.375 s; θ₁ 0.51, not shared at the primary k = 2.
  - Occipital: 0 reliable windows, k = 0 / 0.
  - Descriptive, computed for this entry from existing outputs: where the codes
    are shared, each code's electrode pattern has a cosine of 0.55–0.68 with
    the within-cell HG-on-RT slope pattern, in both frames (no null).
- **Interpretation:** Response locking does not make the codes vanish. It
  makes them cleaner and puts them, with their sharing, in the second before
  the response, fading as the response approaches. That is the latency
  signature's shape (A4 table, [`decoding.md`](decoding.md) §13.4): at a fixed
  time before the response the slow cells, above all incongruent-switch, sit
  higher on a common ramp. Matching removing the codes again, and the
  shared pattern resembling the within-cell RT pattern, point the same way.
  Two things keep me from calling it a pure timing shift. The sharing
  persists to the last pre-response window. And the ramp peaks highest in
  the slowest cell, which a shifted fixed waveform would not do and activity
  growing with time on task would. The regression/matching disagreement
  survives response locking, so it more likely lies in how the two controls
  work than in stimulus alignment. Random-subset seed 0 is again the weak
  draw, so match vs random on the *sharing* can't be read from one seed.
  Settling checks: response-locked RT seeds 1–4; a latency-only model run
  through the same job; a split-half RT-axis test with a null
  ([`coding_axes.md`](coding_axes.md) §11.7).
- **Effect on the paper's story:** **Supports** the "tied to RT" clause of F5
  d–e and **adds** "present before the response"
  ([`paper_draft.md`](paper_draft.md) §1.1 last clause, §1.4 F5 "What would
  change it", §3.4 last bracket, §4 "Base-effect codes" bracket). The geometry
  numbers (θ₁ about 0.75 of its ceiling) are unchanged, so "partly aligned"
  stands. Note that the two pre-written decision rules disagree: paper_draft
  §1.4 reads codes that persist before the response as "present before the
  response", while decoding.md §13.4 reads sharing that sits early and fades
  as latency. The data meet both descriptions, so the wording below uses both.
  No regional claim (occipital has no code).
- **Recommendation:** **Supplement (S5)** for the response-locked figures:
  `coding_axes_controls.png` plus the main run's `coding_axis_cosines.png`.
  Add **one main-text sentence** in §3.4 beside d–e. Keep d–e
  stimulus-locked. **Rerun first** before any sentence comparing matched with
  random subsets in response time: seeds 1–4 (8 jobs, ~50 min each, 8 cores /
  64 GB). The occipital result cannot be reported as a region control; a
  stimulus-locked occipital run is still missing.

### Oddities

- `block_separations*` / `block_*` exist only in the `rt_match10`,
  `rt_random10` and `occ` folders: their jobs started after the
  block-separation code was saved (09:12–09:15). Not interpreted; those runs'
  trajectories are unreadable (10–11 participants under the subsets).
- Every log has "Mean of empty slice" warnings from
  `stability_flexibility_segregation_dcc.py:181`, as in the stimulus-locked
  runs (21–22 per log there). Not a failure.
- In the `rt_match10` and `occ` runs the raw plane fraction sits below its own
  bootstrap CI (0.035 vs [0.037, 0.116]; 0.070 vs [0.073, 0.208]). That is a
  small-sample bias of the bootstrap in unreadable runs.
- `subspace_summary.json` and `summary.json` report `seed` 3 / 5. These are
  the derived bootstrap seeds; the run seed is 0 (`run_config.json`).

### Not documented here

- 58783375 (`coding_axes_real`, COMPLETED 09:54): a stimulus-locked all-lPFC
  (`ROIS=lpfc`) `rt_adjust` rerun, not in this request.
- Still running or pending: `dec_a0e0_*_feat-channels` 58738829–58738830
  (running), 58738831–58738834 (pending, `AssocGrpMemLimit`).
- Finished on 2026-10-15/16 but not written up yet: `pwr_a0e0_*`
  58739128–58739143, `dec_scoresel_*` 58739251–58739254, `dec_wmgroups_*`
  58739922–23, 58739928–29, 58757457–58, `sf_anatomy_a1_real` 58741204–07,
  58741689, 58741692, 58741779, 58741783 (FAILED, not inspected).
- Interactive and outside the repo: 58782513, 58752835 (Jupyter); 58739111
  (`svd_debug`, `claude_debug/`).

### Proposed `paper_draft.md` edits (not applied)

1. **§1.1, the candidate sentence's bracket** "[, tied to RT: the
   response-locked run decides this clause (§1.7)]" →
   ", tied to RT and present before the response".
2. **§1.2 "What changed", new row:** "**Response-locked coding axes**
   (2026-10-16; [`coding_axes.md`](coding_axes.md) §11) | The codes and their
   sharing sit in the second before the response (shared −0.875 to −0.06 s,
   alignment 0.59 [0.42, 0.83]; θ₁ 0.76 [0.70, 0.93] of its ceiling), fade
   toward it and are gone after it; RT matching again removes them (seed 0).
   Closest to the latency signature, not cleanly. 'Tied to RT, present before
   the response'; not 'response-related'. Occipital: no reliable code."
3. **§3.4, replacing "[Response-locked result, once run (§1.7).]":** "> Aligned
   to the response instead, both codes and their alignment were present
   before the response, strongest 0.4–0.75 s before it (alignment 0.59
   [0.42, 0.83] over the reliable windows; shared from 0.88 to 0.06 s before
   the response), and absent after it; RT matching again removed the codes
   (Supplementary **[S5]**)."
4. **§4 "Base-effect codes", replacing "[After the response-locked run: whether
   they persist before the response.]":** "> They were present in the second
   before the response and faded as it approached, the pattern expected if
   slower trials sit further along a common ramp of activity; whether that is
   response timing or a process that takes longer on slow trials, these data
   do not say."
5. **§1.4 F5, "What would change it", response-locked bullet, append:** "Run
   2026-10-16: the codes persist before the response and fade toward it, and
   matching still removes them; the clause becomes 'tied to RT, present before
   the response' ([`coding_axes.md`](coding_axes.md) §11.7)."
6. **§1.5 S5 row, append:** "the response-locked controls figure and main
   cosine panel (coding_axes.md §11)". Status cell: "response-locked run
   2026-10-16 (seed 0); response-locked RT seeds 1–4 and a stimulus-locked
   occipital run still to do".
7. **§1.7 checklist:** tick "Response-locked coding-axes run" (the x labels
   now read "s from response"). Under "Occipital coding axes", note that the
   response-locked occipital run has no reliable code and a stimulus-locked
   one is still missing. Add "[ ] Response-locked `rt_match10`/`rt_random10`,
   `SEED=1..4`".

### Downloads

Manifest: `/hpc/home/jz421/download_manifests/2026-10-16.txt` (139 files,
~46 MB: figures, csv, json and null npz files of the six folders, plus the
controls figure and table). Left out: the six `per_split_windows.csv` tables (420–423 MB each for
lPFC, 88 MB for occipital). They can be regenerated and are only needed to
recompute the cosines.

---

## 2026-10-15 (afternoon and evening): coding-axes RT seeds, subspace test, trajectories

Drafted by Claude; not yet reviewed.

Jobs: `coding_axes_real` 58740222–58740241 (20), 58741497–58741498 (2),
58758317–58758318 (2), 58759949 (1), all COMPLETED in 14–39 min. Owning doc:
[`coding_axes.md`](coding_axes.md) §10.3–10.8.

### For the PI meeting (2026-10-16): what changed since the morning entry

1. **The subspace test confirms "partly shared" over the whole trial.** In all
   lPFC the first principal angle between the congruency and switch subspaces
   is 63°, at 0.74 [0.66, 0.93] of its ceiling. It is shared (p = 0.001 against
   both shuffles) and not one code. Each code is essentially one electrode
   pattern, and the shared part is a single direction.
2. **The trajectories show it directly.** On held-out trials, switch moves
   along the congruency axis by about 60 % as much as congruency does. The c–s
   plane holds about half the reproducible condition-dependent signal (0.52;
   0.69 in sig lPFC).
3. **RT matching removing the codes holds over five seeds, and it is not the
   halved sample.** Matched: 1–6 reliable windows, ceilings 0.10–0.16. Random
   subsets of the same size: 9–12 windows, ceilings 0.22–0.28, sharing kept
   (median alignment 0.79). The morning's weak random subset (0.36) was the
   seed-0 draw. This replaces item 3 of the morning entry.

Question to raise: given item 3, should the late shared component (and the
late codes generally) be framed as RT/response-related, with a response-locked
run as the next step?

### coding_axes_real: RT seeds, subspace test, trajectories

- **Status:** RT matching and random subsets at seeds 1–4 and at 5 bins, for
  all and sig lPFC (58740222–41). Main all and sig reruns with the subspace
  test and trajectories (58741497–98). Seed-0 `rt_match10`/`rt_random10` all
  lPFC rerun with the current code (58758317–18; cosine numbers identical to
  the morning runs). `remove_mean` all lPFC rerun the same way (58759949).
- **Result:**
  - Subspace test. All lPFC: k = 3 (congruency) / 2 (switch), primary k = 2,
    θ₁ 0.45 (63°) against ceiling 0.61, normalized 0.74 [0.66, 0.93];
    alignment index 0.58 [0.46, 0.89] (C → S) and 0.54 (S → C). Sig lPFC:
    primary k = 1, θ₁ 59°, 0.69 [0.57, 0.86]. Dimension 1 carries 16–21 % of
    the held-out sum of squares; dimensions 2–3 about 1 % each. `remove_mean`
    (all lPFC): θ₁ 66°, 0.69 [0.51, 0.87] at k = 1, against 0.72 without it.
  - Trajectories: plane signal fraction 0.52 [0.40, 0.63] (all), 0.69 (sig).
    Axes from 0.1–0.5 s carry ≈1 %.
  - RT, all lPFC, 10 bins, five seeds: matched 1–6 reliable windows, one
    variable with k = 0 in every seed. Random 9–12 windows, alignment
    0.36–0.88 (median 0.79), θ₁ shared in 4 of 5 seeds (not seed 0). Sig lPFC is the same picture. Matching removed the RT gaps in
    every seed (≤ 2 ms left).
- **Interpretation:** The geometry claim is now solid in the unmatched data:
  one shared direction at about 0.7–0.75 of the noise ceiling, and the rest
  separate. The RT result is no longer a sampling worry. With RT equated
  across the four cells, the late congruency and switch mean-difference codes
  themselves mostly disappear, not only their shared part. It still disagrees
  with `rt_adjust` (sharing kept, 0.58) and with A4's matched decoders.
  Late stimulus-locked HG that lasts until the response would explain the
  pattern, but that is untested.
- **Effect on the paper's story:** **Supports** F5 d "partly shared, not a
  uniform rise", now with a trial-wide test and a picture
  ([`paper_draft.md`](paper_draft.md) §1.4 F5 d; §3.6). **Weakens** any
  process-specific reading of the late codes: the RT result now looks real.
  Plan §7's first row ("what they share is a common slowing component") is
  the closest wording, but the data say more than that.
- **Recommendation:** **Main text for the geometry** (F5 d: cosine panel plus
  the trajectory plane; subspace numbers in the text), **supplement for the
  RT seeds table**. Before writing the RT sentence: (1) a response-locked run;
  (2) trial counts per participant × block × condition in the matched
  subsets; (3) the occipital control and the controls figure, still not run.

---

## 2026-10-15

Drafted by Claude; not yet reviewed.

Jobs: everything submitted by jz421 on 2026-10-15 (`sacct -u jz421 -X -S
2026-10-15T00:00`). This entry covers the families that had finished; the rest
are listed under "Still outstanding".

### For the PI meeting (2026-10-16): the short version

1. **Task-significant electrodes carry more of each base effect than the
   others.** At a matched electrode count, sig electrodes decode congruency and
   switch type better than nonsig ones in lPFC and DLPFC (by 0.17–0.24
   accuracy, p = 0.001 in all four). Nonsig electrodes still decode well above
   chance (0.58–0.67). Keeping all lPFC as the anatomy set is still defensible,
   but we can't argue that the extra electrodes are as informative as the sig
   ones.
2. **The congruency and switch coding axes are partly shared, late in the
   trial, and not as a common shift.** All lPFC: the cross cosine beats the
   shuffle null from 0.56 s on, at about 0.7 of the noise ceiling (alignment
   0.69 [0.57, 0.89]). It is unchanged by `remove_mean` (0.67). This is the same
   picture as A4, now with a null and a ceiling.
3. **Open problem: RT matching removes the codes themselves**, not only the
   sharing: after matching, almost no window has a reliable congruency or
   switch axis, while the random-subset control keeps them. This is stronger
   than A4's RT result and the two disagree in kind. It needs seeds and a
   cell-count check before anyone draws a conclusion from it. *(Updated in the
   afternoon/evening entry above: five seeds confirm it, and the weak random
   subset was the seed-0 draw.)*
4. **Top-20 % score sets are built** (positive, negative, absolute; lPFC and
   DLPFC). The decoding on them is still queued. The LWPC/LWPS scores on 30 %
   of trials are close to noise level, so expect weak separation between sets.
5. **The channel-feature decoding (`dec_a0e0`) has not produced anything yet.**
   Three attempts failed (SVD non-convergence, then two rounds of OOM at
   225 GB); the 500 GB rerun is running.

Questions to raise: Is the RT-matching loss of the codes a reason to frame the
late shared component as RT/response-related (the A4 reading), and should we
add a response-locked version? Is "sig carries more per electrode" worth a
supplement panel, or one Methods sentence?

### dec_wmgroups: window-mean decoding in sig, nonsig and all electrodes

Owning doc: [`new_analyses_walkthrough.md`](new_analyses_walkthrough.md) §2.6
(pointer in [`decoding.md`](decoding.md) §3.6).

- **Status:** 2 jobs, both COMPLETED (58726163 congruency, 2 h 03 min;
  58726164 switch type, 3 h 12 min). ROIs lPFC and DLPFC, 0–1.5 s means, 20
  bootstraps, 1000 membership permutations, seed 42.
- **Result:** At matched count (lPFC 171, DLPFC 121 electrodes per group),
  sig > all > nonsig in every case. lPFC congruency 0.86 / 0.79 / 0.67, switch
  type 0.83 / 0.76 / 0.66; DLPFC congruency 0.82 / 0.74 / 0.64, switch type
  0.81 / 0.70 / 0.58 (chance 0.50). All 12 pairwise differences have Holm
  p ≤ 0.004; sig − nonsig is 0.17–0.24 against a null range of about ±0.09.
- **Interpretation:** Per electrode, task-significant electrodes carry more of
  each base effect. Nonsig electrodes are not empty. The analysis was meant to
  test "nonsig decodes at least as well as sig", which would justify using all
  electrodes on data grounds; that reading is ruled out.
- **Effect on the paper's story:** Neither supports nor weakens a Results claim
  directly. It bears on the electrode-set choice in F5
  ([`paper_draft.md`](paper_draft.md) §1.4 F5, and §2.5 Methods). All lPFC was
  pre-specified as primary and stays so, but the stated reason can't be that
  nonsig electrodes are equally informative. Covers the base effects only, not
  LWPC/LWPS.
- **Recommendation:** **Supplement** (one bar panel, e.g. in S-N4) plus one
  Methods sentence. Not main text: it is a design justification, not a
  finding about adaptation.

### sbatch_score_electrode_selection: top / negative / absolute 20 % sets

Owning doc: [`new_analyses_walkthrough.md`](new_analyses_walkthrough.md) §3.5
(pointer in [`decoding.md`](decoding.md) §3.7).

- **Status:** 3 jobs COMPLETED, ~10 min each (58726651 positive, 58726655
  negative, 58726659 absolute), `--rois lpfc,dlpfc`, 30 % selection, top 20 %,
  0–1.5 s, seed 0. This is step 1 only; the decoding on these sets
  (`dec_scoresel_*`) is still pending and no power-trace run was submitted.
- **Result:** 80-electrode sets in lPFC (of 398, 22 participants) and
  56-electrode sets in DLPFC (of 277), each drawing on 11–20 participants; one
  participant supplies up to 25 % of a positive set (41 % in the DLPFC
  negative `lwps` set). Positive LWPC ∩ LWPS overlap is 13 / 80 in lPFC,
  slightly under the 16 expected for independent sets; LWPC–LWPS r on the
  selection trials is 0.05 (lPFC) and 0.00 (DLPFC).
- **Interpretation:** The sets are built correctly. Positive and negative sets
  are disjoint, and the absolute sets are the union of the two tails. The
  adaptation scores on 30 % of trials have a spread close to what sampling
  noise alone gives (a rough estimate, in the owning doc), so the `lwpc` and
  `lwps` sets will be partly chosen by noise and their held-out effects will
  shrink.
- **Effect on the paper's story:** None yet. Bears on the "does the top-LWPC
  set carry LWPC but not LWPS" question, which is not in
  [`paper_draft.md`](paper_draft.md) yet.
- **Recommendation:** **Exclude for now**: a selection step has no result of
  its own. Report the set sizes and participant counts with the step-2
  decoding when it finishes. Before reading step 2, compute each score's
  30 % vs 70 % reliability.

### coding_axes_real: congruency vs switch coding axes over time

Owning doc: [`coding_axes.md`](coding_axes.md) §10.

- **Status:** 11 jobs COMPLETED, 10–20 min each (58726994–58727165 and
  58727101): all lPFC main; DLPFC main; all lPFC with `remove_mean`,
  `rt_adjust`, `rt_match10`, `rt_random10`; task-significant lPFC main plus the
  same four controls. Not run: occipital control, controls figure.
- **Result:** All lPFC (398 electrodes, 22 participants): shared from 0.56 to
  1.38 s, cross 0.27 vs ceiling 0.37, alignment **0.69 [0.57, 0.89]**, "not
  one code" at 0.50 and 0.81–1.00 s. DLPFC 0.66 [0.52, 0.85]; sig 0.61
  [0.48, 0.78]. `remove_mean` 0.67; `rt_adjust` 0.58 [0.37, 0.96]. RT matching:
  1 (all) and 0 (sig) reliable windows, so the codes are gone; the random
  subset keeps 9–11 reliable windows, with alignment 0.36 and sharing only
  after 1.06 s.
- **Interpretation:** The two codes are partly shared after ~0.55 s, as the
  plan and A4 predicted, and the sharing is a pattern rather than a shift
  common to all electrodes. Nothing can be said about the early period: no
  window before 0.5 s has a reliable mean-difference code in the main run, so
  "separable early" is untested here. The RT-matching result means the
  congruency and switch mean differences in HG largely disappear when RT is
  equated across the four cells. Taken at face value, the late codes, shared or
  not, are mostly RT-linked. Because the single random-subset draw also moves
  the result a lot, and A4's matched decoders still decoded, I'm not confident
  in this reading yet.
- **Effect on the paper's story:** **Supports** the F5 panel-d claim "partly
  shared, emerging late, not a uniform rise"
  ([`paper_draft.md`](paper_draft.md) §1.4 F5 d; §3.6), and gives the plan's
  geometry figure its first number. **Weakens** any reading of the late shared
  component as process-specific. It also puts at risk the separate (non-shared)
  part being cognitive rather than RT, if the matching result holds. Does not
  support "separable early".
- **Recommendation:** **Rerun first.** Needed before it goes into F5 d or
  S5: (1) `rt_match`/`rt_random` with 3–5 seeds; (2) per-participant ×
  block × condition trial counts in the matched subset; (3) the occipital run;
  (4) `plot_coding_axes_controls.py`. If the main result holds and the RT
  question is settled, the main all-lPFC panel can sit beside A4 in F5 d as
  the null-and-ceiling version of the same claim.

### Failures

| Family | Jobs | Cause | Resubmitted? |
|---|---|---|---|
| `dec_a0e0_*_feat-channels` (channel-feature decoding, no ANOVA table, 6 conditions), attempt 1 | 58724122–58724126 FAILED; 58724121 CANCELLED | `LinAlgError: SVD did not converge` in the decoder fit for the pooled shuffle null (loky worker), after 17–30 min | Yes, see below |
| same, attempt 2 (DLPFC) | 58725544–58725549 | OUT_OF_MEMORY at 225 GB, right after loading (~20 GB RSS) in the first PCA fit, or while building the visualization debug arrays | Yes |
| same, attempt 3 (DLPFC + lPFC) | 58725697–58725702 | OUT_OF_MEMORY at 225 GB while building the first bootstrap's / visualization arrays | Yes: **58738829–58738834** (500 GB, 50 CPUs), running/pending, **outstanding**. Whether the SVD failure recurs is unknown: attempts 2–3 never reached that step |
| `sf_anatomy_a1_real` (A3/N4 anatomy; sig lPFC, sig DLPFC, all DLPFC, all lPFC) | 58726853, 58726855, 58726860, 58726861 | **Not an analysis failure.** Each log ends with "✓ Analysis completed successfully!" and the figures and tables were written. The job's exit code 1 comes from `xvfb-run` cleanup: its own Xvfb server had already exited, so `kill $XVFBPID` failed under `set -e` and `xvfb-run` returned 1 instead of Python's 0. All four jobs ran at once on the same node (dcc-coganlab-gpu-ferc-s-aa32-29) and all report display `:99`, so a display-number collision is the likely reason the Xvfb servers died early (not verified). The two sig runs also skipped the §19 RT-adjusted follow-up (`RT_LONG_DF_CSV not found`). | No. The outputs were not reviewed in this entry; check the brain renders are not blank before using them |

Owning doc for the `dec_a0e0` failures:
[`new_analyses_walkthrough.md`](new_analyses_walkthrough.md) §1.7.

### Still outstanding (not documented)

- `dec_a0e0_*_feat-channels` rerun: 58738829–58738834 (2 running, 4 pending).
- `pwr_a0e0_*` power traces: 58739128–58739143 (2 running, 14 pending).
- `dec_scoresel_*` decoding on the score sets: 58739251–58739254 (pending).

Not documented on purpose: 58721702 (Jupyter session), 58739111 (`svd_debug`,
outside the repo in `claude_debug/`). Also not covered here: the cancelled
`dec_scoresel_*` attempts 58739124–58739127 and 58739210–58739213 (cancelled
before starting, superseded by 58739251–54).

### Proposed `paper_draft.md` edits (not applied)

1. **§1.2 "What changed", new row:** "**Coding axes run** (2026-10-15;
   [`coding_axes.md`](coding_axes.md) §10) | F5 d: the congruency and switch
   coding axes are partly shared from +0.56 s (alignment 0.69 [0.57, 0.89] of
   the split-half ceiling, all lPFC), unchanged by `remove_mean` (0.67). Agrees
   with A4. No reliable code before 0.5 s, so it says nothing about early
   separation. RT matching removes the codes themselves; needs seeds before
   any claim."
2. **§1.4 F5, panel d, "Key numbers" cell, append:** "coding-axes alignment
   0.69 [0.57, 0.89], shared 0.56–1.38 s (coding_axes.md §10)". **Status cell,
   append:** "coding axes run 2026-10-15; RT-matching seeds and occipital
   pending".
3. **§3.6, after the `remove_mean` sentence, a candidate sentence (once the
   seeds are in):** "> The per-electrode coding axes gave the same picture
   with a split-half ceiling and a label-shuffle null: the congruency and
   switch-type axes were aligned beyond chance from +0.56 s, at 0.69 (95 % CI
   0.57–0.89) of the ceiling, and not fully (the CI excluded 1 at 0.81–1.00 s).
   Removing each participant's mean activity left this unchanged (0.67)."
4. **§2.5 Methods (electrode set), one sentence:** "At a matched electrode
   count, task-significant electrodes decoded congruency and switch type
   better than the remaining electrodes (window-mean decoding, 0–1.5 s;
   accuracy difference 0.17–0.24 in lPFC and DLPFC, membership-permutation
   p = 0.001), although the remaining electrodes also decoded both above chance
   (0.58–0.67); we nonetheless analyze all lPFC electrodes, as specified
   before analysis."
5. **§1.5 Supplement, S-N4 row, append:** "the sig / nonsig / all window-mean
   decoding bars (new_analyses_walkthrough.md §2.6)".

### Downloads

Manifest: `/hpc/home/jz421/download_manifests/2026-10-15.txt` (95 files,
~16 MB: the window-mean-group and score-selection outputs, and every
coding-axes figure, csv, json and null file). Left out: the eleven
`per_split_windows.csv` tables of the coding-axes runs (100–240 MB each; five
are over 200 MB). They can be regenerated and are only needed to recompute
the cosines.
