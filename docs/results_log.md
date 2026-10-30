# Results log

Newest entry first. One entry per batch of cluster jobs: what finished, what it
showed, what it means for the paper, and what to do with it. Details live in
the owning docs linked from each block.

---

## 2026-10-30: anatomy vs coding-axes overlap; block-projection null submitted

Drafted by Claude; not yet reviewed. **Partial:** the block-null runs are
still running.

Jobs: `overlap_measures` 60565385 COMPLETED (10 min); `coding_axes_real`
60598899–904 (all24: stimulus- and response-locked, main / `remove_mean` /
`rt_adjust`) and 60598938–40 (`SUBJECT_SET=good`, stimulus-locked, the same
three), all with `N_BLOCK_PERM=500`, submitted 2026-10-30, running.

### Short version

1. **The anatomy and the coding axes agree on the main effects.** The
   congruency–switch similarity, noise-corrected, is 0.65–0.86 under every
   combination of scoring (raw difference or Cohen's d) and comparison
   (cosine, Pearson, within-participant centring); d with within-participant
   centring gives 0.79, the anatomy's own noise-corrected r. An earlier
   estimate of 0.42 used mismatched inputs (a task-significant r with all-lPFC
   reliabilities).
2. **Early-window axes cannot test the congruency question.** With axes from
   0.1–0.5 s (cos(c, s) ≈ 0) every block change is near 0, including switch's
   (+0.01 [−0.21, 0.19]), because no reliable code exists then. These outputs
   were already in the 10-22 folders. Late axes (0.6–1.4 s) reproduce the main
   result: congruency × incProp along switch +0.33 [0.16, 0.52], switch ×
   switchProp along switch +0.88 [0.34, 1.61].
3. **New in the coding-axes job** (not yet in results): a shuffled-blocks null
   for every block contrast and proportion change
   (`coding_trajectories.block_null`; `N_BLOCK_PERM`, default 0 = off), and each
   separation projected on the other variable's axis too (`axis` column), so
   congruency's change on switch's unique axis is reported directly.

### overlap_measures: raw vs d, cosine vs r

- **Status:** COMPLETED. `dcc_scripts/subspace/overlap_measures.py` on the
  all-lPFC main-effects segregation run's `long_df.csv` (0–1.5 s window means;
  200 per-electrode splits, seed 0). Output
  `stats/results/<epochs>/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects/overlap_measures/`.
- **Result** (398 electrodes, 22 participants; alignment = cross / ceiling):

  | Scores | cosine | Pearson | within participant |
  |---|---|---|---|
  | mean difference | 0.77 [0.58, 1.03] | 0.72 | 0.86 [0.65, 1.21] |
  | Cohen's d | 0.70 [0.53, 0.93] | 0.65 | 0.79 [0.60, 1.05] |

- **Effect on the paper's story:** F5 and F6 measure the same thing for the
  main effects, so the anatomy–subspace question has no conflict to resolve
  there (`paper_draft.md` §1.1, 2026-10-29 narrative).
- **Recommendation:** one Methods/Results sentence; the table for reviewers.

### Downloads

Figure package for choosing and redrawing panels (every current candidate
figure, with a README of figure, panel, source and draft caption):
`/hpc/home/jz421/paper_figures/paper_figures_2026-10-30.zip` (24 MB, 80
figures). It holds the 10-22 coding-axes figures; the block-null runs will
overwrite those in place with p values added.

```bash
scp jz421@dcc-login.oit.duke.edu:/hpc/home/jz421/paper_figures/paper_figures_2026-10-30.zip ~/Downloads/
```

---

## 2026-10-29: rerun on the quality-checked subject set (`SUBJECT_SET=good`)

Drafted by Claude; not yet reviewed. **Partial:** decoding (F4) and six of the
ten RT-matched / random coding-axes seeds are still running; this entry will be
completed when they finish.

Jobs: 60508448–60508470 (23) and 60508606 (anatomy, resubmitted), all
submitted by `dcc_scripts/submit_subject_set_rerun.sh`, log
`dcc_scripts/out/subject_set_rerun_good_20261029_095756.log` (commit e63d0a6
plus uncommitted changes). Settings are the paper runs' recorded ones; every
output folder ends in `_subjects-good`, so the all24 results are untouched.

**The set** (`src/analysis/config/subjects.py`, `good`, 23 participants):
all24 minus D0144 (missing block D) and D0117 (accuracy 0.63: correct over
task trials in `combinedData.csv`, first trial of each block dropped, as in the
behavioral notebook; the equal-cell-weight `acc_mean` in
`ieeg_behavioral_subject_level_effects.csv` is a different measure and gives
0.61), plus D0130. D0071 (abnormal high-gamma traces) was already out. **D0130 has no
frontal electrodes** (parietal, temporal, occipital and subcortical contacts
only), so for every lPFC analysis the change is just dropping D0117 (11 lPFC
electrodes) and D0144 (16).

### Short version

1. **Every main-text claim checked so far holds**, with slightly larger effects
   in most places.
2. **F3:** both adaptation clusters remain, in the behavioral direction. The
   congruency × switch-proportion cluster that the all24 run had is gone.
3. **F5 a–c:** overlap, height gradient and its participant-level tests all
   hold. The parcel test weakens to p = 0.047.
4. **F5 d–e:** alignment, shared windows, θ₁ and plane fraction are unchanged.
   One fewer "not one code" window. RT matching removes the codes again
   (seeds 0–2).

### pwr (F3): windowed ANOVA, block-balanced sets, task-responsive lPFC

- **Status:** 60508448–49 COMPLETED (18–19 min). Folder
  `power/figs/<epochs>/anova_within_roi_subjects-good/`.
- **Result:** 152 task-responsive lPFC electrodes from 19 participants (all24:
  171 from 21). Significant clusters, window centres; "−" means the condition
  effect is smaller in high-proportion blocks, the behavioral direction:

  | Term | all24 | good |
  |---|---|---|
  | congruency × incongruent proportion (LWPC) | − 0.56–1.38 s | − 0.69–1.38 s |
  | switch type × switch proportion (LWPS) | − 0.56–1.38 s | − 0.75–1.38 s |
  | congruency × switch proportion | − 0.56–1.06 s | none |
  | switch type × incongruent proportion | none | none |

- **Interpretation:** both adaptation effects hold. Their clusters start a
  window or two later, which is not an onset difference (cluster tests do not
  localize in time). The traces of the lost cross-effect cluster look almost
  identical; it fell below threshold.
- **Effect on the paper's story:** F3 now matches its framing: both adaptation
  effects present and both cross-effects absent. Report n = 152 / 19.
- **Recommendation: main text,** as planned.

### segreg + sf_anatomy (F5 a–c): all-lPFC scores and continuous anatomy

- **Status:** 60508454–56 COMPLETED (3–19 min); 60508457 FAILED (it started
  before its inputs existed: this cluster's `sbatch` ignores
  `SBATCH_DEPENDENCY`, so the hold never applied); resubmitted with
  `--dependency` as 60508606, COMPLETED (8 min). Folders
  `stats/results/<epochs>/segregation_results/*_subjects-good/` and
  `anatomy_a1_lpfc_window_0.0to1.5s_all_subjects-good/`.
- **Result:** 370 electrodes from 19 participants enter the overlap (all24:
  397 / 21); the anatomy has 371 electrodes from 20.

  | Test | all24 | good |
  |---|---|---|
  | LWPC–LWPS overlap (separate halves) | r = 0.097, p = 0.0001 | r = 0.106, p = 0.0001 |
  | congruency–switch overlap | r = 0.224 | r = 0.237 |
  | split-half reliability, LWPC / LWPS | 0.069 / −0.094 | 0.095 / −0.110 |
  | z slope of the balance | −0.0077/mm [−0.0124, −0.0032], p = 0.0074 | −0.0087/mm [−0.0140, −0.0039], p = 0.0028 |
  | z slope, weighted sign-flip (participants) | p = 0.041 | p = 0.033 |
  | z slope, random-slope mixed model | p = 0.037 | p = 0.043 |
  | z slope, leave one participant out | p 0.002–0.099 | p 0.001–0.055 |
  | overlap, participants weighted (n − 3) | r = 0.098, p = 0.031 | r = 0.105, p = 0.029 |
  | overlap, participants unweighted | p = 0.35 | p = 0.34 |
  | matched − crossed (Test 1) | r = 0.088, p = 0.0005 | r = 0.082, p = 0.0022 |
  | parcel test, adaptation balance | F = 1.90, p = 0.010 | F = 1.69, p = 0.047 |
  | parcel test, base-effect balance | F = 1.70, p = 0.017 | F = 1.51, p = 0.037 |

- **Interpretation:** the overlap and the height gradient do not depend on
  D0117 or D0144; the gradient's worst leave-one-out p even improves. The parcel
  test loses most, as an omnibus over 19 parcels would when 27 electrodes go.
- **Effect on the paper's story:** none for F5 a–c. The parcel sentence in
  §3.4 needs the new p, which is now borderline.
- **Recommendation: main text,** as planned; quote the parcel test as
  borderline.

### coding_axes_real (F5 d–e, S5): all dlPFC + lPFC

- **Status:** main, `remove_mean`, `rt_adjust`, `rt_match10` seeds 0–2 and
  `rt_random10` seed 0 COMPLETED (26–33 min); 60508464, 66–70 (the other
  seeds) still running.
- **Result:** 371 electrodes from 20 participants (all24: 398 / 22).

  | Run | all24 | good |
  |---|---|---|
  | main: alignment over reliable windows | 0.69 [0.57, 0.89], 15 windows | 0.72 [0.59, 0.97], 15 windows |
  | main: shared windows | 0.56–1.38 s | 0.56–1.38 s |
  | main: "not one code" windows | 5 (6, 11–14) | 4 (11–14) |
  | main: θ₁ | 63°, 0.74 [0.66, 0.93] of ceiling, shared | 63°, 0.75 [0.67, 0.94], shared |
  | main: plane signal fraction; cos(c, s) | 0.52; 0.38 | 0.53; 0.38 |
  | `remove_mean` alignment | 0.67 | 0.72 |
  | `rt_adjust` alignment | 0.58 | 0.61 |
  | `rt_match10` reliable windows, seeds 0/1/2 | 1 / 3 / 3 | 0 / 3 / 3 |
  | `rt_random10` seed 0 reliable windows | 9 | 9 |

- **Interpretation:** the geometry is unchanged. "Not one code" rests on one
  fewer window, and the alignment CI now reaches 0.97, so "partly aligned"
  is a little less firmly separated from "one code".
- **Effect on the paper's story:** none, pending the remaining seeds.
- **Recommendation: main text,** as planned; update the n.

### dec (F4, S5b): within-block decoding — pending

60508450–53 running (up to 48 h).

### Oddities

- **F3 baseline vs the draft.** The all24 congruency × switch-proportion
  cluster is in the saved task-responsive-electrode figure
  (`lpfc_..._24_subjects_2way_congruency_x_switchProportion_sig_elecs_sem_shading.png`,
  2026-10-08), not only in the all-electrode run, yet `paper_draft.md` §1.4 F3
  says there is no cluster off the diagonal. Check which figure the draft
  describes.
- The all24 `anova_F_traces/*.npz` carry no electrode-set tag, so a sig run
  and an all run overwrite each other; the 10-08 files are whichever ran last.

### Downloads

Manifest: `/hpc/home/jz421/download_manifests/2026-10-29_good_subjects.txt`
(586 files, 203 MB: figures, csv, json, txt, pdf, pkl and npz from the finished
`_subjects-good` folders). Left out: `per_split*.csv` and `long_df.csv`.
Regenerate when decoding finishes.

```bash
rsync -avz --files-from=:/hpc/home/jz421/download_manifests/2026-10-29_good_subjects.txt \
  jz421@dcc-login.oit.duke.edu:/hpc/group/coganlab/jz421/GlobalLocal/dcc_scripts/ \
  ~/GlobalLocal_results/
```

---

## 2026-10-22 (evening): proportion trajectories, raw-axis trajectories, unique-axis separations

Drafted by Claude; not yet reviewed.

Jobs: `coding_axes_real` 59312137–59312142 (stimulus-locked) and
59312143–59312148 (response-locked), 12, all COMPLETED, exit 0, in 15–64 min.
An earlier batch, 59305401–59305413 (12, all COMPLETED), had only the proportion
trajectories and was overwritten by this one. Owning doc:
[`coding_axes.md`](coding_axes.md) §13 (method §1.2–1.3, figures §6.3).

### Short version

1. **Switch adaptation is specific in both senses.** The switch separation
   changes with switch proportion and not incongruent proportion, and the
   change lies along the switch axis, not the congruency axis: +0.70
   [0.23, 1.30] along switch, +0.02 along congruency, stimulus-locked. It holds
   under `remove_mean`, RT regression (+0.46 [0.12, 0.86]) and on the unique
   switch axis, and before the response.
2. **Congruency adaptation is not along the congruency axis.** The extra
   congruency separation in 25 % incongruent blocks lies along the switch axis
   (+0.32 [0.16, 0.47]), not the congruency axis (+0.21 [−0.02, 0.41]). The
   same holds under RT regression and before the response. The prediction was
   the reverse.
3. **The response-locked frame looks like a clean double dissociation**
   (congruency × incProp along congruency +0.52 [0.20, 0.85]), **but that part
   is after the response**: −0.09 before the response under RT regression.
4. **Reruns reproduce §12:** cosines and subspace byte-identical; trajectories
   and block separations to < 10⁻⁹.

### coding_axes_real: proportion trajectories, raw axes, unique axes (both frames, main and five controls)

- **Status:** all twelve COMPLETED; settings as 2026-10-22 (afternoon),
  `RT_CONTROL=none` explicit, the RT regression run included (59312139,
  59312145). New outputs per folder: `proportion_trajectories.csv`,
  `proportion_changes.csv`, `proportion_summary.json`,
  `proportion_trajectories_<axes>.png`, `trajectories_raw.csv`,
  `trajectories_<axes>[_uncentred]_raw.png`, `block_*_unique.*`.
- **Result:** see the short version; tables in `coding_axes.md` §13.2–13.3.
  The cross sets are near 0, except congruency × switchProp along the switch
  axis under `remove_mean` (+0.15 [0.04, 0.28]) and before the response (+0.40).
- **Interpretation:** the switch half of the predicted dissociation holds in
  its strongest form so far: right proportion, right axis, robust to the
  controls. The congruency half does not hold as stated. Congruency's
  incProp-related change is real (raw-axis +0.33), but in the population it
  rides along the switch direction, and the switch axis also picks up some
  congruency × switchProp change. Two readings remain open: a shared direction
  tracking block-level difficulty or surprise, or an artifact of splitting a
  partly out-of-plane change between two oblique axes.
- **Effect on the paper's story:** strengthens the switch-specific claim of
  the §6 supplement (`paper_draft.md` §1.5, §3.4). Argues against claiming a
  population-level double dissociation for congruency. Adds a possible finding
  (congruency adaptation on the switch axis) that needs a null before it can be
  mentioned.
- **Recommendation: supplement.** Panel d of `proportion_trajectories` (switch ×
  switchProp) with its cross panel c is the figure to show. Hold the
  congruency panels until (1) a shuffle null exists for the along-axis
  components (shuffle the proportion labels within participant and cell), and
  (2) the plot is redrawn with early-window axes (0.1–0.5 s, cos(c, s) ≈ 0),
  where the oblique split is not needed. Not main text.

### Oddities

- Response-locked, congruency × incProp along congruency flips sign around the
  response under RT regression (−0.09 before, +0.49 after). The post-response
  windows include feedback and the next trial (§11.1).
- Stimulus-locked occipital: switch × switchProp along the congruency axis is
  +0.09 [0.0001, 0.20], borderline, in a region with no reliable per-window
  code.
- `rt_match10` / `rt_random10` disagree with each other again (e.g. congruency
  × switchProp along congruency −0.42 match vs +0.21 random); not interpreted.

### Proposed `paper_draft.md` edits (not applied)

1. **§1.5 Supplement row from the 2026-10-22 (afternoon) entry, replace the
   result cell with:** "The switch separation changes with switch proportion,
   not incongruent proportion, and along the switch axis only (+0.70
   [0.23, 1.30]; survives `remove_mean`, RT regression and unique axes);
   congruency's incProp change lies along the switch axis, not its own; also
   weakly in occipital | Pending a shuffle null."
2. **§3.4 sentence (only if the null holds):** "> At the population level, the
   switch separation was larger in mostly-repeat than mostly-switch blocks,
   and the difference lay along the switch coding axis, not the congruency
   axis (Supplementary **[S?]**)."

### Downloads

Manifest: `/hpc/home/jz421/download_manifests/2026-10-22_evening.txt` (538
files, 110 MB: every figure, csv, json and null npz written by this batch in the
twelve folders; it supersedes `2026-10-22.txt`). Left out: the
`per_split_windows.csv` tables (1.8 GB, unchanged).

```bash
rsync -avz --files-from=:/hpc/home/jz421/download_manifests/2026-10-22_evening.txt \
  jz421@dcc-login.oit.duke.edu:/hpc/group/coganlab/jz421/GlobalLocal/dcc_scripts/ \
  ~/GlobalLocal_results/
```

---

## 2026-10-22: block-split adaptation in the coding axes (subspace plan §6)

Drafted by Claude; not yet reviewed.

Jobs: `coding_axes_real` 59283148–59283153 (stimulus-locked) and
59283438–59283443 (response-locked), 12, all COMPLETED, exit 0, in 16–62 min.
Owning doc: [`coding_axes.md`](coding_axes.md) §12 (method §1.3, figure
§6.3); plan: [`analysis_plans.md`](analysis_plans.md) subspace plan §6.

### Short version

1. **Switch: the predicted pattern holds.** Projected on the switch axis, the
   switch separation is larger in low- than high-switch-proportion blocks and
   does not change with incongruent proportion: own − other +0.80
   [0.31, 1.47] stimulus-locked, +0.69 [0.10, 1.33] response-locked. It
   survives `remove_mean`, and survives RT adjustment stimulus-locked (+0.52
   [0.13, 1.04]).
2. **Congruency: no clean specificity.** The congruency separation is larger
   in low-incProp blocks (+0.33 [0.08, 0.57]), but not reliably more than it
   changes with switchProp (own − other +0.21 [−0.17, 0.62]), and the two are
   equal under RT adjustment. The response-locked own − other (+0.61) comes
   from post-response windows with no reliable code.
3. **Two reasons for caution on switch**: one block (low incProp, high
   switchProp) has almost no switch separation and carries much of the effect,
   and occipital cortex shows the same switch pattern (+0.39 [0.03, 0.68]).
4. **Reruns reproduce §10–11 byte for byte.** New besides §6: stimulus-locked
   occipital (no reliable window) and the stimulus-locked `rt_adjust` subspace
   test and trajectories.

### coding_axes_real: §6 block separations, stimulus- and response-locked, main and five controls

- **Status:** all twelve COMPLETED. Same settings and seeds as the 2026-10-15
  and 2026-10-16 runs (stimulus-locked 0–1.5 s; response-locked −1.0 to 1.5 s
  with −1.0–0 s axes; all lPFC or occipital; seed 0). Submitted with
  `RT_CONTROL=none` explicitly, because commit ff91524 changed the submit
  script's default to `adjust`. They overwrote the seed-0 folders. Every
  pre-existing output is byte-identical except `run_config.json` (timestamp,
  new `block_bootstrap` seed).
- **Result:** see the short version. Full table, per-block separations and
  timing in `coding_axes.md` §12.2. Stimulus-locked, the switch effect starts
  at about 0.625 s. The congruency effect appears only in the last one or two
  windows, past the median RT.
- **Interpretation:** at the population level, the switch code's adaptation is
  specific to switch proportion. This is the trajectory form of Test 1, and it
  agrees for switch. The congruency code changes with incongruent proportion,
  but the data do not show that this change is specific. Neither half is
  lPFC-specific: occipital shows the switch pattern, with weak codes.
- **Effect on the paper's story:** adds a supplementary population-level
  version of Test 1 (`paper_draft.md` §1.4 F5 panel b, §3.4) for switch only.
  It does not support a "double dissociation in the population geometry"
  claim, because the congruency half is missing.
- **Recommendation: supplement, switch half only, after two checks.**
  (1) Trial counts per cell in the low-incProp, high-switchProp block, and
  whether behavioural switch costs are smallest there. (2) A label-shuffle
  null for own − other (shuffle switchProp within participant, cell and
  incProp). Today only bootstrap CIs exist, and the per-window marks are
  uncorrected. Do not put it in the main text.

### Oddities

- The low-incProp, high-switchProp block has a near-zero switch separation in
  both frames (0.12 against 0.32–1.10), which implies an incProp × switchProp
  interaction on the switch separation. Unexplained.
- Stimulus-locked occipital has no reliable window in the per-window cosine
  test, yet the time-pooled subspace test finds one reproducible dimension per
  variable (θ₁ 77°, shared). Pooling 21 windows gives the subspace test more
  power than any single window.
- `rt_match10` / `rt_random10` (10–11 participants) and response-locked
  occipital disagree with each other on the block contrasts; not interpreted
  (trajectory signal fractions unreadable; `coding_axes.md` §11.8).

### Not documented here

- `dec_a0e0_*` 58738829–58738834: TIMEOUT (2026-10-15 to 10-19 starts), not
  inspected.
- Submitted 2026-10-22 13:15–13:23, outside this batch: `fig5_top_*` 59288971–59288974 FAILED
  (13:15), not inspected. `scoresel_sel1` 59288978 COMPLETED. `pwr_a0e0_*`
  59288979–59289022 CANCELLED, resubmitted as 59289276–59289287 (running /
  pending). `dec_wmgroups_*` 59289268–59289269 running.
- Interactive: 58782513, 59145395, 59279561 (Jupyter).

### Proposed `paper_draft.md` edits (not applied)

1. **§1.5 Supplement, new row:** "Block-split adaptation along the coding
   axes ([`coding_axes.md`](coding_axes.md) §12) | The switch separation
   changes with switch proportion and not incongruent proportion (own − other
   +0.80 [0.31, 1.47]; survives `remove_mean` and RT adjustment); congruency
   shows no specific change; also in occipital | Pending the
   low-incProp/high-switchProp check and a shuffle null."
2. **§3.4, after the Test 1 sentence (only if the checks pass):** "> Projected
   onto the population switch axis, the switch separation was larger in
   mostly-repeat than mostly-switch blocks and did not depend on incongruent
   proportion (Supplementary **[S?]**); the corresponding effect for
   congruency was not specific."

### Downloads

Manifest: `/hpc/home/jz421/download_manifests/2026-10-22.txt` (310 files,
53 MB: figures, csv, json and null npz files of the twelve folders). Left out:
the twelve `per_split_windows.csv` tables (1.8 GB together), which are
unchanged from the earlier runs and can be regenerated.

```bash
rsync -avz --files-from=:/hpc/home/jz421/download_manifests/2026-10-22.txt \
  jz421@dcc-login.oit.duke.edu:/hpc/group/coganlab/jz421/GlobalLocal/dcc_scripts/ \
  ~/GlobalLocal_results/
```

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
