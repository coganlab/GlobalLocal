# Results log

Newest entry first. One entry per batch of cluster jobs: what finished, what it
showed, what it means for the paper, and what to do with it. Details live in
the owning docs linked from each block.

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
   cell-count check before anyone draws a conclusion from it.
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
