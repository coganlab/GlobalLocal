# N4 — continuous electrode scores → anatomy, brain maps, and descriptive centres

**What this document is.** The runbook and the record of results for N4, beats
**5–7** of [`analysis_plans.md` › Concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan):
how LWPC and LWPS adaptation are organized across lPFC, with the congruency and
switch-type main effects as the reference.

**How it is organized.**

| Part | Sections | Read it for |
|---|---|---|
| **Start here** | [§0](#0-start-here-current-results-how-they-were-computed-and-how-to-write-them-up) | The current results; how each was computed; which figure shows it; how to read it in the paper; manuscript-ready Methods and Results; what is still open |
| Method and runbook | §1–§14 | What the tests compute, how to run them, every output, and how to read it |
| Detailed results | §16–§19 | The main-effect reference and the all-lPFC run (§16–§17), the task-significant subset (§18), participants as the unit, local similarity, the overlap controls and Figure 5 (§19) |
| [Archive](#archive-superseded-results-and-drafts) | §15 and superseded parts of §16, §17 and §19 | Earlier runs, drafts and interpretations that the current results replace, kept because older notes and scripts cite them |

Section numbers have not changed, so references from the scripts and other docs
(`n4_section15_followups.py`, "N4 §19.8", …) still resolve. For a line-by-line
guide to the code, see [`n4_code_walkthrough.md`](n4_code_walkthrough.md) and
its notebook, `dcc_scripts/stats/n4_code_walkthrough.ipynb`, which runs on the
real all-lPFC outputs.

---

## 0. Start here: current results, how they were computed, and how to write them up

*Status as of 2026-10-05, the date of the latest runs. Each number points to the
section that holds it, and that section names its source file.*

⚠️ *2026-10-08: the DLPFC anatomy run wrote into
`anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` (the folder is named after
`ROI_FILTER`, which was `lpfc`). Until it is rerun, the files there dated
2026-10-08 are DLPFC (277 electrodes), not all lPFC; the numbers in this doc
are still the lPFC ones. The DLPFC results, set against this section's, are in
[`dlpfc_results.md`](dlpfc_results.md) §5–§6, and the fix is in its §8.1.*

### 0.1 The question, the scores and the population

N4 asks how the two adaptation effects, LWPC and LWPS, are organized across
lateral prefrontal cortex (lPFC):

1. **Do they share electrodes?** Is an electrode's LWPC score correlated with
   its LWPS score, when the two are measured on separate halves of its trials?
2. **Does the balance between them vary with location?** For each electrode,
   `delta = lwpc_s − lwps_s` (positive = relatively LWPC-dominant). Does delta
   differ across Destrieux parcels, or change along MNI axes?

**Scores.** For every electrode, four Cohen's *d* on window-mean high gamma
(70–150 Hz, 0–1.5 s after stimulus onset, correct trials), each from the four
cell means of a 2 × 2 design with equal weight per cell (§2.1, §16.1):

```text
LWPC       = (I − C | 25 % incongruent) − (I − C | 75 % incongruent)
LWPS       = (S − R | 25 % switch)      − (S − R | 75 % switch)
congruency = ½[(I − C | 25 %) + (I − C | 75 %)]
switch     = ½[(S − R | 25 %) + (S − R | 75 %)]
```

each divided by the electrode's pooled within-cell SD. Positive LWPC or LWPS is
the predicted adaptation: a smaller condition effect in high-proportion blocks.
For the anatomy, each score is divided by its SD across electrodes
(`lwpc_s`, `lwps_s`, `cong_s`, `switch_s`; "SD units"), and the two balances are
`delta = lwpc_s − lwps_s` and `dm = cong_s − switch_s` (§2.2).

**Trial splits.** Each electrode's trials are split at random into two halves,
stratified on the four design factors, 1,000 times, and all four scores are
computed on each half (`per_split.csv`: `xA xB yA yB mxA mxB myA myB`). Every
test that relates two scores takes them from opposite halves and averages over
splits, so trial noise shared within an electrode cannot create an
association. An electrode's score for the anatomy tests is the mean over halves
and splits.

**Population.** All lPFC electrodes: 398 electrodes (254 left, 144 right) from
22 participants (1–55 each), defined by atlas label, a choice stated in
Methods. The task-significant subset (171 electrodes, 21 participants) is the
replication (§0.9). Single electrodes are mostly noise (split-half reliability
0.27–0.30 at full length), so only population-level summaries are interpreted.

### 0.2 The pipeline as run

| Step | What it does | Command | Writes | Details |
|---|---|---|---|---|
| 1. Score | The four scores for every electrode, on both halves of each split; the overlap tests | `cd dcc_scripts/stats && ROIS=lpfc MAIN_EFFECTS=1 RT_ADJUST_HG=0 N_SPLITS=1000 N_PERM_CORR=10000 bash submit_stability_flexibility_segregation_dcc.sh` | `segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects/` | §16.1–§16.2 |
| 2. Anatomy | Destrieux labels and MNI coordinates; parcel and coordinate tests on delta and dm; Test 1 and Test 2; participant-level tests, overlap controls and Figure 5 | `ARM=continuous ROI_FILTER=lpfc ANAT_LEVEL=destrieux N_PERM=10000 bash submit_stability_flexibility_anatomy_dcc.sh` with `SEG_RUN` set to step 1's folder | `anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` (the `_sig` suffix does not select electrodes, §13) | §16.2–§16.3 |
| 3. §16 follow-ups | Height against distance from the midline, single-score slopes, bands, label means, Test 2 bootstrap | `python dcc_scripts/stats/n4_section16_followups.py --scores … --seg-dir … --tilt … --out-dir …` | `continuous/section16/` | §16.6 |
| 4. Long table with trial ids | The trial-level table that step 5 rescores with one split per participant | `RT_ADJUST_HG=0 SCATTER_N_SPLITS=0 SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh`, and again with `RT_ADJUST_HG=1` | `…_scatter_only_splits0/long_df.csv` | §19.5 |
| 5. §19 follow-ups | Shared-split reliabilities and local similarity, the RT-coupling control (and everything else in §19 for runs older than 2026-10-02) | `python dcc_scripts/stats/n4_section19_followups.py --anatomy-dir … --seg-dir … --long-df … [--rt-coupling …]` | `continuous/section19/`, or `section19_rt_adjusted/` | §19.5 |
| 6. RT sensitivity | Step 1 with the RT-linked part of high gamma removed | step 1 with `RT_ADJUST_HG=1` | `…_main_effects_rt_adjusted/` | §18.10 |

*Since 2026-10-07* step 2 runs steps 3 and 5 itself, and the §15 script too
(`FOLLOWUPS=15,16,19`, the default). Make step 4's two long tables first; the
submitter finds them under step 1's `segregation_results/`
(`LONG_DF_CSV`, `RT_LONG_DF_CSV`, `RT_COUPLING_CSV`). §19 then has local
similarity and the RT row in `continuous/` with the rest of §19, and the
RT-adjusted companion in `continuous/section19_rt_adjusted/`; §15 and §16 go
to `continuous/section15/` and `continuous/section16/`. A missing table skips
only its part, and `summary.txt` ends with a FOLLOW-UPS block saying what ran
(§19.5).

The segregation submitter defaults to `RT_ADJUST_HG=1` and 200 splits, so step 1
sets both. Code: scoring and the overlap tests are in
`src/analysis/stats/stability_flexibility_segregation.py`; everything after is
in `src/analysis/stats/stability_flexibility_anatomy.py` (§1 maps the
functions; [`n4_code_walkthrough.md`](n4_code_walkthrough.md) goes line by
line). DCC paths are in §17.1.

### 0.3 The results at a glance (all lPFC)

| | Result | Numbers | Source |
|---|---|---|---|
| 1 | **LWPC and LWPS share electrodes.** | separate-half *r* = 0.097, *p* = 0.0001 (397 electrodes, 21 participants); 0.111, *p* = 0.0002 with shared splits | §16.6.2, §19.8.4 |
| | … with participants as the unit | weighted *r* = 0.098 [0.025, 0.157], *p* = 0.031; unweighted *r* = 0.043, *p* = 0.35 (11 of 20 positive) | §19.8.3 |
| | … and the two kinds of electrode sit in the same place | centroids of LWPC-positive and LWPS-positive electrodes 1.4 mm apart, *p* = 0.95 | §15.6 |
| 2 | **The overlap follows the base effects.** | *r* 0.097 → 0.028 (*p* = 0.23) with each half's congruency and switch effects partialled out; 0.094 with nonlinear responsiveness, 0.100 with coordinates, 0.081–0.113 with any one participant left out | §19.8.5 |
| 3 | **It is not RT.** | RT-coupling covariate 0.082 (*p* = 0.001); RT-adjusted high gamma 0.103 (*p* = 0.0004) | §19.8.5, §18.10 |
| 4 | **Congruency and switch share electrodes.** | *r* = 0.226, *p* = 0.0001 | §16.6.2 |
| 5 | **Each adaptation tracks the effect it regulates.** | dm vs delta *r* = 0.092, *p* = 0.0003 (0.087 with coordinates partialled out); matched 0.217 and 0.169 against crossed 0.121 and 0.133 | §16.6.5 |
| 6 | **The balance differs across parcels.** | *F* = 1.91, *p* = 0.010 (19 parcels; *p* = 0.003–0.057 with each participant left out); no parcel *q* < 0.05 | §16.6.3 |
| 7 | **The balance shifts with height**: relative to LWPS, LWPC is weaker dorsally. | block *F* = 2.85, *p* = 0.031; *z* slope −0.0077 SD/mm, *p* = 0.008 (Bonferroni 0.023); ~2 % of variance; replicates across trial halves (*p* = 0.005); anterior–posterior null (*p* = 0.58) | §16.6.4, §15.7 |
| | … with participants as the unit | weighted −0.0077 [−0.0147, −0.0013], *p* = 0.041; random-slope model −0.0083, *p* = 0.037 (between-participant SD 0.010); *p* = 0.002–0.099 with each participant left out | §19.8.2 |
| | … in Figure 5 | dorsal tertile's centroid balance −0.34 [−0.49, −0.20]; ventral −0.04 [−0.19, 0.12] | §19.8.2 |
| | … and which axis | height and distance from the midline correlate *r* = −0.58; distance fits better (*p* = 0.019 against 0.53); LWPC carries it (*p* = 0.006); exploratory | §16.6.4 |
| 8 | **Beyond that gradient, electrodes do not reliably differ in their balance.** | balance reliability −0.041 [−0.119, 0.027], against LWPC's 0.170, which neighbours within 10 mm share (+0.170, *p* = 0.004) | §19.8.6 |
| 9 | The base-effect balance | across parcels *F* = 1.70, *p* = 0.017; no significant gradient (block *p* = 0.13); its parcel means track delta's (*r* = 0.73) | §16.6.3–§16.6.4 |
| 10 | Is the gradient inherited from the base effects? | unresolved: shrinkage 0.10, bootstrap −0.20 to 0.40 | §16.6.6 |

**In one sentence** (the candidate ending in [`paper_draft.md`](paper_draft.md)
§1.1, with the 2026-10-05 addition): LWPC and LWPS share lPFC electrodes
because each is expressed where the signal it regulates is, and congruency and
switch type share electrodes; beyond a shallow dorsal–ventral bias, electrodes
do not reliably differ in the balance between the two adaptations, which
describes one population rather than two intermixed ones.

### 0.4 How each result was computed, what to look at, and how to read it

#### A. LWPC and LWPS share electrodes (results 1 and 4)

**Computed.** `split_resolved_corr` in the segregation job. In every split,
LWPC from half A is correlated with LWPS from half B, and LWPC from B with LWPS
from A. Before correlating, each half's scores are regressed on responsiveness
(mean |high gamma|) across electrodes and centred within participant, then
ranked (Spearman). The two directions and the 1,000 splits are averaged.
Participants with fewer than three electrodes are dropped. The null permutes
which LWPS goes with which LWPC within each participant, with the same
permutation in every split, 10,000 times; *p* = (*b* + 1)/(*N* + 1), two-sided.
The same test on congruency and switch gives result 4.

*Participants as the unit* (`participant_split_corr`, §19.2): each participant's
own separate-half correlation over its electrodes (participants with ≥ 4), on
exactly the same values; the correlations are averaged in Fisher *z* with
weights *n* − 3 and tested by flipping the sign of whole participants (10,000),
with a participant-bootstrap interval (2,000). The unweighted version is a
one-sample *t*-test on *z*.

*Location* (`centroid_shuffle_test`, §15.6): the distance between the centroid
of LWPC-positive electrodes and that of LWPS-positive electrodes, against
electrode labels shuffled within participant.

**Look at.** Fig. 5b (`fig5_height.png`, panel b; points in
`fig5_height_points.csv`); per-participant *r* in `participant_corr.csv`.

**Read it as.** A small, robust positive overlap: electrodes that adapt more for
one demand adapt more for the other. It holds in the electrode population,
weighted by electrode count, but not in every participant, so word it as a
property of the population. The scatter's points correlate more strongly than
*r* (they are full-data scores, §15.5), so the caption must say that *r*
compares separate trial halves.

#### B. Why they share electrodes: the base effects (results 2 and 5)

**Computed.** `overlap_controls` (§19.7) reruns the overlap test with one more
set of covariates regressed out at the residualisation step: log and squared
responsiveness; MNI coordinates (centred within participant); each half's
congruency and switch effects, taken from the same half as the adaptation score
they adjust, so the two sides of the correlation still share no trials; and
each electrode's RT coupling (C, below). It also drops each participant in turn
(1,000 permutations per fold).

Test 1 (`delta_tracking_test`, §16.4) is the same overlap test applied to dm
(one half) against delta (the other half), with and without coordinates
partialled out. Its covariance equals the two matched covariances
(congruency–LWPC, switch–LWPS) minus the two crossed ones (congruency–LWPS,
switch–LWPC), so anything common to all four scores cancels: it tests whether
each adaptation tracks its own base effect more than the other one.

**Look at.** `section19/overlap_controls.csv` and `overlap_loso.csv`;
`delta_tracking.csv`; the matched-vs-crossed bars (`fig5.png` panel b, an S-N4
panel).

**Read it as.** Most of the LWPC–LWPS overlap goes with the base effects: each
adaptation is expressed where the effect it regulates is (matched > crossed),
and congruency and switch share electrodes, so the adaptations do too. Nothing
else tested moves it. The caveat: the base effects are also the best proxy for
an electrode's signal-to-noise, and that row cannot tell the two apart; the
process-specific matched > crossed pattern argues against pure
signal-to-noise. A sharper control (each adaptation partialled on its own base
effect only) is still to run.

#### C. It is not RT (result 3)

**Computed.** Two ways. (1) A covariate: each electrode's pooled within-cell
correlation between high gamma and RT (`rt_r`), added to the overlap test.
(2) The full control: per electrode, the pooled within-cell slope of high gamma
on RT is estimated and removed from every trial before any score is computed
(`RT_ADJUST_HG=1`, `rt_adjust_hg`), and everything is rescored. It is
conservative: it also removes any adaptation that reaches RT through the same
trial-by-trial coupling.

**Look at.** `section19_rt_adjusted/summary_section19.txt`; the RT-adjusted
segregation run's `summary.txt` (§18.10).

**Read it as.** RT coupling does not produce the overlap: 0.082 with the
covariate, 0.103 on RT-adjusted high gamma (shared splits), 0.090 on the
pipeline's split, against 0.097 raw. RT adjustment does roughly halve the base
effects' overlap (0.224 → 0.127), so RT coupling contributes to how strongly
electrodes carry congruency and switch, not to their adaptation.

#### D. The balance differs across parcels (result 6)

**Computed.** `relative_score_roi_test` (§4): delta = parcel + responsiveness +
participant (participants as fixed effects). The statistic is the one-way *F*
of parcel on delta with responsiveness and participant removed. The null swaps
the LWPC and LWPS labels independently within every electrode, which flips the
sign of delta and keeps each electrode's participant, location,
responsiveness and pair of scores (10,000 permutations). Destrieux parcels
sampled in at least three participants enter (19 parcels, 396 electrodes).
Adjusted parcel means with Benjamini–Hochberg *q* describe a significant
omnibus. Leave-one-participant-out reruns use 1,000 permutations.

**Look at.** `delta_by_roi.png`, `delta_per_roi.csv`, `delta_roi_loso.csv`
(S-N4).

**Read it as.** The balance varies modestly across lPFC. Report the omnibus only:
no parcel survives FDR, one participant can move *p* to 0.057, and the subset
is null. The superior frontal gyrus and sulcus lean toward LWPS, descriptively.

#### E. The balance shifts with height (result 7)

**Computed.** `relative_score_coordinate_test` (§5): delta = MNI *y* + *z* + *x* +
responsiveness + participant, same swap null. The block *F* tests the three axes
together; per-axis slopes are in SD units per mm, Bonferroni-corrected over the
three axes. The slope was refitted on each trial half separately to check it
is not trial noise (§15.7).

*Participants as the unit* (`coordinate_slope_by_participant`, §19.1): with delta
and *z* both residualised on the nuisance terms and the other two axes, the
pooled slope is exactly a weighted average of the participants' own slopes,
each weighted by its electrodes' spread in height. That average is tested by
flipping the sign of whole participants (10,000) with a participant bootstrap
(2,000); the unweighted mean of participants with ≥ 3 electrodes and ≥ 5 mm of
spread by a *t*-test; and a linear mixed model with a participant random
intercept and random *z* slope (REML, predictors centred within participant).
`coordinate_slope_loso` drops each participant in turn.

*Figure 5* (`figure5_height`, §19.4): electrodes split into tertiles of MNI *z*;
each tertile's centroid on the LWPC-against-LWPS scatter with a 95 %
participant-bootstrap region; and the adjusted balance per tertile as
participant means ± SEM.

**Look at.** Fig. 5a (where the tertiles are), 5c (tertile centroids against the
identity line) and 5d (balance by height); `fig5_height_centroids.csv`,
`fig5_height_balance.csv`, `mni_z_slope_by_participant.csv`,
`mni_z_slope_loso.csv`.

**Read it as.** A shallow gradient: relative to LWPS, LWPC is weaker dorsally.
It explains about 2 % of the balance's variance, was not predicted (the
predicted anterior–posterior axis is null), and varies across participants
(the slope's between-participant SD equals the slope). It is a statement about
the balance: neither effect's own height slope is significant (§15.7). Height
and distance from the midline cannot be separated in lPFC, so naming the axis
belongs in the supplement as exploratory.

#### F. Beyond the gradient, no reliable electrode-by-electrode balance (result 8)

**Computed.** The trial table is rescored with one random split per
participant, shared by all of its electrodes (200 splits), because the
pipeline's per-electrode split lets one electrode's half A share trials with
its neighbours' half B, which biases within-participant reliabilities and
creates spurious local similarity (§19.3). Within-participant split-half
reliabilities come from this table.

Local similarity (`local_similarity`, §19.3): per split and half, each score
(the balance, LWPC, LWPS, congruency, switch) is regressed on responsiveness and
coordinates (removing the linear gradient), centred within participant, ranked
and scaled. For every pair of electrodes in a participant, similarity is one
electrode's half A against the other's half B and the reverse, averaged over
splits; an electrode paired with itself gives its split-half reliability. Pairs
are binned by distance (< 10, 10–20, 20–40, > 40 mm). Each participant's
baseline is the bin mean over 5,000 shuffles of its electrode positions, and
the excess over baseline is tested by flipping the sign of whole participants
(one-sided), with participant-bootstrap intervals. The single scores are the
positive control.

**Look at.** `section19/local_similarity.png`, `local_similarity.csv`,
`reliability_by_split_scheme.csv` (S-N4).

**Read it as.** The positive control works: neighbours within 10 mm share
LWPC's reliable signal, and switch type's. The balance has no local excess, but
also no reliable within-participant variation once the gradient is removed
(−0.04, interval up to 0.03), so this is not a positive test of "intermixed".
What it supports: electrodes do not come in two kinds, LWPC-leaning and
LWPS-leaning; they carry more or less of both together. "One population rather
than two intermixed ones."

#### G. Supplement only: the base-effect balance and inheritance (results 9 and 10)

**Computed.** The parcel and coordinate tests on dm instead of delta (§16.6.3–
§16.6.4). Test 2 (`tilt_with_main_effect_covariate`, §16.4) refits delta's
coordinate model with dm as a covariate; the shrinkage of the slope is read
against dm's split-half reliability (a covariate measured with reliability λ
removes about λ of a slope it fully carries), with a 2,000-resample
participant bootstrap.

**Look at.** `dm_by_roi.png`, `dm_coordinates.csv`, `tilt_with_dm.csv`,
`section16/panel_b_label_means.csv`.

**Read it as.** The base-effect balance also differs across parcels, in step
with the adaptation balance, but has no significant gradient of its own. Whether
the adaptation gradient is inherited from the base effects cannot be
determined: dm is measured too unreliably for the shrinkage to tell.

### 0.5 Figures to look at

Files are in the all-lPFC anatomy run's `continuous/` folder (§17.1) unless
marked. The 2026-10-05 Figure 5 is in `continuous/section19/`; a new anatomy
run writes it to `continuous/`.

| Figure | File | Shows | In the paper |
|---|---|---|---|
| Fig. 5a | `fig5_height.png`, panel a; or `fig5_height_brain.png` | Electrodes on a sagittal projection (MNI *y* against *z*), coloured by height tertile; cuts at *z* = 16.3 and 38.1 mm. The brain version (added 2026-10-06, §19.4) shows the same tertiles on fsaverage with each tertile's centroid per hemisphere; positions in `fig5_height_brain_centroids.csv` | Main |
| Fig. 5b | panel b | LWPC against LWPS (the test's residualised scores), coloured by tertile, identity line, *r* = 0.097, *p* = 0.0001 | Main |
| Fig. 5c | panel c | The tertile centroids, enlarged, with 95 % participant-bootstrap regions; dorsal sits above the identity line | Main |
| Fig. 5d | panel d | Adjusted LWPC − LWPS by tertile, participant means ± SEM, with the slope's electrode- and participant-level *p* | Main |
| Fig. 5e | A4 cross-decoding traces ([`decoding.md`](decoding.md#138-results-2026-10-01)) | Congruency ↔ switch codes over the same electrodes | Main if its controls hold ([`paper_draft.md`](paper_draft.md) §1.4) |
| S-N4 | `fig5.png`, panel b | Matched vs crossed base-effect × adaptation correlations (Test 1) | Supplement |
| S-N4 | `section19/local_similarity.png` | Excess similarity by distance for the balance and each single score | Supplement |
| S-N4 | `delta_by_roi.png`, `dm_by_roi.png` | Adjusted balance per parcel | Supplement |
| S-N4 | `section16/panel_c_height.csv`, `panel_c_midline.csv` | The four scores by tertile of height and of distance from the midline (data; not yet plotted) | Supplement |
| — | `joint_scatter.png`, `score_map_*.png`, `score_centers.csv` | Diagnostics on uncorrected scores; maps; weighted medoids | Not as evidence. Maps only as coverage or illustration. |

### 0.6 How to read the results in the paper

**The story, in order.** (1) LWPC and LWPS share electrodes. (2) They do
because each adaptation is expressed where the effect it regulates is, and
those effects share electrodes. (3) Their balance shifts modestly with height
(and differs across parcels). (4) Beyond that shallow shift, electrodes do not
come in LWPC-leaning and LWPS-leaning kinds: one population, not two
intermixed ones. Under the characterization framing of
[`paper_draft.md`](paper_draft.md) §1.1, none of this is a dissociation or
independence claim.

**What to claim and what not to.**

| Write | Do not write | Why |
|---|---|---|
| "LWPC and LWPS share electrodes" | "LWPC and LWPS are co-regulated", "a shared adaptation mechanism" | The overlap goes with the base effects (result 2) |
| "One population rather than two intermixed ones" | "Intermixed", "local patches were ruled out" | The balance has no reliable variation for local similarity to test (result 8) |
| "Relative to LWPS, LWPC adaptation was weaker dorsally" | "Dorsal lPFC supports flexibility", "LWPC is absent dorsally" | It is a shift in the balance; neither effect's own slope is significant |
| "A shallow gradient, about 2 % of the variance, not predicted" | "lPFC is organized dorsoventrally for control" | The size and the unpredicted axis |
| "Across the electrode population" | "In every participant" | Unweighted participant-level tests are not significant |
| The parcel omnibus | Any named parcel as the driver | No parcel survives FDR |
| "Whether the gradient is inherited from the base effects could not be determined" | "The gradient survives the main effects, so it is adaptation-specific" | dm is too unreliable for Test 2 to tell |

More in §16.6.7.

**What a reviewer will ask, and the answer.**

- *Why map LWPC across all lPFC when its mean is near zero there* (mean
  *d* = 0.01; 49 % of electrodes positive)? Because N2/N3 establish LWPC in
  task-responsive electrodes; here the question is how scores are distributed
  across the region, and the population was fixed by anatomy before analysis.
- *Are electrodes the right unit?* Every key result was repeated with
  participants as the unit (results 1 and 7).
- *Is the overlap just signal-to-noise?* Responsiveness (linear and nonlinear)
  does not move it; the base effects do, and the process-specific matched >
  crossed pattern is not what signal-to-noise alone would give.
- *Is it RT?* No (result 3), and the adaptation means in Fig. 3 survive the RT
  adjustment too (§19.8.8).
- *Why does the subset not show the anatomy?* Power: the same estimates on fewer
  electrodes, and a planted slope of the observed size reaches *p* < 0.05 only
  26 % of the time on its layout (§0.9).

### 0.7 Methods (manuscript text)

This supersedes §17.2 (archived), adding the participant-level tests, the
shared trial splits, the overlap controls, the RT adjustment, local similarity
and Figure 5. Preprocessing is not repeated here; take it from the general iEEG
Methods and check it describes the epochs file these runs used
(`…_drop_and_nan_thresh_perc_5.0_…_stat_func_ttest_zmax_20`). Paragraphs
marked *(Supplementary)* can move to the supplementary methods.

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
> and then averaged over splits, so that trial noise shared within an electrode
> could not create an association. Within-participant split-half reliabilities,
> and the local-similarity analysis below, compare one electrode's half with
> another electrode's; for these the trials were instead split once per
> participant and repetition (200 splits), with the same halves used for all of
> its electrodes, so that no electrode's half shared trials with another
> electrode's other half.
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
> **Overlap.** To test whether two effects share electrodes, one score from one
> half was correlated with the other score from the other half (both
> directions, every split) after both were regressed on responsiveness (mean
> absolute high gamma) and centred within participant. Spearman ρ was averaged
> over splits. Participants with fewer than three electrodes were excluded
> (397 electrodes, 21 participants remained). The null permuted electrode
> correspondence within participant, identically across splits (10,000
> permutations; *p* = (*b* + 1)/(*N* + 1)). To take participants as the units,
> each participant's own separate-half correlation was computed over its
> electrodes (participants with at least four), on the same values; the
> correlations were averaged in Fisher *z* with weights *n* − 3 and tested by
> flipping the sign of whole participants (10,000 flips; 95 % interval from
> 2,000 participant bootstrap resamples), and the unweighted mean was tested
> with a one-sample *t*-test. Electrodes positive for each adaptation effect
> were compared in location by the distance between their centroids, with
> electrode labels shuffled within participant.
>
> **Alternative explanations of the overlap.** We repeated the overlap test with,
> in turn, each of the following added to the covariates removed before
> correlating: log and squared responsiveness; MNI coordinates; each half's
> congruency and switch-type effects, taken from the same half as the
> adaptation score they adjusted, so that the two sides of the correlation
> still shared no trials; and each electrode's RT coupling, the pooled
> within-cell correlation between its high gamma and RT. We also repeated it
> with each participant left out (1,000 permutations per fold). As a full
> control for RT, we removed from each electrode's single-trial high gamma its
> pooled within-cell linear dependence on RT before computing any score, and
> repeated the analyses. This adjustment is conservative, because it also
> removes any adaptation that reaches behavior through the same trial-by-trial
> coupling.
>
> **Tracking of the base effects.** To ask whether each adaptation follows the
> effect it regulates, Δ_main from one half was correlated with Δ_adapt from
> the other by the overlap procedure above, with and without MNI coordinates
> partialled out within participant. Because the covariance of the two balances
> equals the two matched covariances (congruency–LWPC, switch–LWPS) minus the
> two crossed ones (congruency–LWPS, switch–LWPC), this tests whether each
> adaptation tracks its own base effect more closely than the other; the four
> pairings were computed the same way.
>
> **Anatomical test.** Each balance was modelled as
> balance = parcel + responsiveness + participant, with participants as fixed
> effects. The statistic was the *F* for the parcel block. Destrieux parcels
> sampled in at least three participants entered (19 parcels, 396 electrodes).
> The null exchanged the two effect labels independently within every
> electrode, which flips the sign of the balance while keeping each electrode's
> participant, location, responsiveness and pair of scores (10,000
> permutations). Adjusted parcel means, with Benjamini–Hochberg correction over
> the 19 parcels, were used only to describe a significant omnibus result. The
> test was repeated with each participant left out (1,000 permutations per
> fold).
>
> **Coordinate test.** Each balance was also modelled as
> balance = MNI *y* + *z* + *x* + responsiveness + participant, across all
> electrodes and within each hemisphere, with the same null. We report the
> block *F*, and slopes in standard deviations per
> millimetre with *p* values Bonferroni-corrected over the three axes. To check
> that the height slope was not trial noise, it was refitted on each half of
> every split separately. To take participants as the units, we used the fact
> that, with the balance and height both residualised on the other terms, the
> pooled slope is a weighted average of the participants' own slopes, each
> weighted by its electrodes' spread in height. We tested that average by
> flipping the sign of whole participants (10,000 flips; 2,000 bootstrap
> resamples), tested the unweighted mean slope of participants with at least
> three electrodes and 5 mm of spread with a *t*-test, and fitted a linear
> mixed model with a participant random intercept and random height slope
> (REML; predictors centred within participant). The slope was also refitted
> with each participant left out.
>
> **Figure 5.** For display, electrodes were divided into tertiles of MNI *z*.
> Each tertile's centroid was computed on the overlap scatter (the
> responsiveness-residualised, participant-centred scores that the overlap test
> correlates, on the balance's scale), with a 95 % region from 2,000
> participant bootstrap resamples. The balance by tertile was shown as
> participant means ± SEM after participant and responsiveness were removed.
> The tertiles were not used for inference.
>
> **Local similarity** *(Supplementary)*. To ask whether electrodes leaning
> toward LWPC or LWPS form local patches, we compared each pair of electrodes
> within a participant across trial halves: one electrode's score from one half
> against the other's from the other half, averaged over both directions and
> all 200 shared splits, after removing responsiveness, the linear coordinate
> trend and each participant's mean (Spearman, on unit-variance scores). Pairs
> were binned by distance (< 10, 10–20, 20–40, > 40 mm). Each participant's
> baseline for a bin was its mean over 5,000 shuffles of the participant's
> electrode positions; we tested the excess over that baseline by flipping the
> sign of whole participants (one-sided), with intervals from 2,000 participant
> bootstrap resamples. An electrode's similarity with itself across halves is
> its split-half reliability. The same analysis of LWPC, LWPS, congruency and
> switch type alone served as the positive control.
>
> **Inheritance of the gradient** *(Supplementary)*. The coordinate model for
> Δ_adapt was refitted with Δ_main as a covariate, and the change in slope
> (shrinkage = 1 − slope with covariate / slope without) was read against
> Δ_main's split-half reliability, because a covariate measured with
> reliability λ can remove only about λ of a slope it fully carries.
> Uncertainty in the shrinkage came from 2,000 participant bootstrap resamples.
>
> **Exploratory follow-ups** *(Supplementary; chosen after the coordinate
> result)*. Because height and distance from the midline are correlated across
> lPFC electrodes, the coordinate model was refitted with |*x*| in place of
> signed *x*. Each single score was fitted on *y* + *z* + |*x*| +
> responsiveness + participant. A single score has no partner to exchange
> labels with, so its null shuffled coordinates among each participant's
> electrodes (2,000 shuffles).
>
> **Reporting.** All tests were two-sided with α = 0.05 unless stated.
> Electrode maps were descriptive and were not used for inference.

Supplementary methods also need: "In the task-responsive subset, trials were
split 200 times rather than 1,000, and the label tests used the 15 parcels
sampled in at least three of its participants."

### 0.8 Results (manuscript text)

Main text first, then the supplement. Every number is in §0.3 with its source.
[`paper_draft.md`](paper_draft.md) §3.4 holds an earlier version with bracketed
placeholders; this text fills them.

> **LWPC and LWPS share electrodes.** We scored every lPFC electrode (398
> electrodes, 22 participants) for LWPC and LWPS. Across this anatomically
> defined set both adaptation effects were small on average (mean Cohen's
> *d* = 0.01 for LWPC and 0.05 for LWPS; 49 % and 56 % of electrodes positive),
> and single-electrode estimates were noisy (full-data split-half reliability
> 0.27–0.30), so we tested only population-level summaries. LWPC and LWPS
> scores from separate halves of the trials were positively correlated within
> participants (Spearman *r* = 0.10, *p* < 0.001; 397 electrodes, 21
> participants; Fig. 5b), and electrodes positive for each effect did not
> differ in location (centroid distance 1.4 mm, *p* = 0.95). With participants
> as the unit, the correlation held when each participant's own correlation
> was weighted by its electrode count (*r* = 0.10, 95 % CI [0.03, 0.16],
> *p* = 0.031) but not when every participant counted equally (*r* = 0.04,
> *p* = 0.35; 11 of 20 positive), so it describes the electrode population
> rather than every participant.
>
> **The overlap follows the effects the two adaptations regulate.** The base
> effects, congruency and switch type, scored from the same trials and halves,
> also shared electrodes (*r* = 0.23, *p* < 0.001), and each adaptation
> correlated more with its own base effect than with the other
> (congruency–LWPC *r* = 0.22 vs switch–LWPC 0.12; switch–LWPS 0.17 vs
> congruency–LWPS 0.13; test of the difference *r* = 0.09, *p* < 0.001, and
> *r* = 0.09 with MNI coordinates partialled out). For LWPS the matched base
> effect was the less reliable map (within-participant split-half reliability
> 0.35 for switch, 0.46 for congruency), so this difference is not one of
> reliability. Partialling each half's congruency and switch-type effects out
> of that half's adaptation scores removed most of the LWPC–LWPS overlap
> (*r* = 0.03, *p* = 0.23). The overlap did not depend on nonlinear
> responsiveness (*r* = 0.09), location (MNI coordinates partialled out,
> *r* = 0.10), RT coupling (*r* = 0.08, *p* = 0.001) or any one participant
> (*r* = 0.08–0.11 with each left out), and it was unchanged when the
> RT-linked part of high gamma was removed before scoring (*r* = 0.10,
> *p* < 0.001). LWPC and LWPS thus share electrodes because each is expressed
> where the effect it regulates is, and those effects share electrodes.
>
> **The balance between the two adaptations shifts modestly with height.** The
> balance between the two effects (LWPC − LWPS within each electrode) differed
> across Destrieux parcels (19 parcels sampled in at least three participants;
> *F* = 1.91, label-exchange permutation *p* = 0.010; *p* = 0.003–0.057 with
> each participant left out), although no single parcel differed from zero
> after FDR correction (all *q* ≥ 0.13). It also varied with position (MNI
> coordinates: block *F* = 2.85, *p* = 0.031): relative to LWPS, LWPC was
> weaker dorsally (*z* slope −0.0077 SD/mm, *p* = 0.008; Bonferroni-corrected
> over three axes, *p* = 0.023; Fig. 5a, c, d). The predicted
> anterior–posterior axis showed no effect (*p* = 0.58), height explained
> about 2 % of the balance's variance, and the slope replicated across
> independent halves of the trials (*p* = 0.005). In the highest third of
> electrodes the balance leaned toward LWPS (centroid −0.34 SD, 95 % CI −0.49
> to −0.20), while the lowest third was balanced (−0.04, −0.19 to 0.12; Fig.
> 5c). With participants as the unit the gradient held, less strongly and
> unevenly across participants: the participants' own slopes, weighted by
> their electrodes' spread in height, averaged −0.0077 SD/mm (95 % CI −0.0147
> to −0.0013; sign-flip *p* = 0.041), and a mixed model with a random height
> slope gave −0.0083 SD/mm (*p* = 0.037; between-participant SD of the slope
> 0.010 SD/mm). With each participant left out, *p* ranged from 0.002 to
> 0.099.
>
> **One population rather than two intermixed ones.** Beyond this shallow
> gradient, electrodes did not reliably differ in their balance. With trial
> halves shared by each participant's electrodes, the within-participant
> split-half reliability of the balance, after the linear gradient was
> removed, was −0.04 (95 % CI −0.12 to 0.03), whereas LWPC alone was reliable
> (0.17) and neighbouring electrodes within 10 mm shared its reliable signal
> (excess similarity 0.17, *p* = 0.004; Supplementary **[S-N4]**). Electrodes
> therefore did not divide into LWPC-leaning and LWPS-leaning kinds; they
> carried more or less of both adaptations together, in step with the effects
> each regulates. The task-responsive subset (171 electrodes, 21 participants)
> gave the same picture at lower power: both pairs of effects shared electrodes
> (LWPC–LWPS *r* = 0.08, *p* = 0.055; congruency–switch *r* = 0.17,
> *p* = 0.001), and each adaptation tracked its own base effect more closely
> than the other by the same margins as in all lPFC (*r* = 0.08, *p* = 0.049).
> The balance did not vary detectably across parcels or with position there,
> as expected at that set's power (Supplementary **[S-N4]**).

**Supplement (S-N4).** Five paragraphs, in this order:

1. *Task-responsive subset*: §18.8, as is.
2. *The adaptation gradient* (axis and per-score breakdown, exploratory):

   > Height and distance from the midline are correlated across lPFC
   > electrodes (within-participant *r* = −0.58), so the two cannot be fully
   > separated. In an exploratory model, distance from the midline described
   > the gradient better than height (distance *p* = 0.019, height *p* = 0.53).
   > In an exploratory breakdown by score, LWPC carried the gradient: it was
   > slightly reversed within about 27 mm of the midline (adjusted *d* = −0.07)
   > and positive farther out (0.04–0.06; slope *p* = 0.006), whereas LWPS did
   > not vary (*p* = 0.75). Fitted separately, the gradient was significant in
   > the left hemisphere (block *F* = 4.22, *p* = 0.005; 254 electrodes) but
   > not the right (*F* = 2.33, *p* = 0.12; 144 electrodes). Of the
   > participants with at least three electrodes and 5 mm of spread in height,
   > 13 of 20 had a negative slope (unweighted mean −0.0110 SD/mm, *p* = 0.078).

3. *The base-effect balance and inheritance*:

   > The base-effect balance (congruency − switch) also differed across parcels
   > (*F* = 1.70, *p* = 0.017; *p* ≤ 0.058 with any one participant left out),
   > in step with the adaptation balance (*r* = 0.73 across the 19 parcel
   > means; same sign in 13). It showed no significant gradient (*F* = 1.81,
   > *p* = 0.13); its slope pointed the same way at about half the size, and it
   > was significant in the left hemisphere alone (*F* = 2.45, *p* = 0.039).
   > Adding the base-effect balance as a covariate reduced the adaptation
   > gradient by 10 % (*z* slope still *p* = 0.015). Because the base-effect
   > balance was measured with low reliability (split-half *r* = 0.08), full
   > inheritance would produce a reduction of only about 15 %, and the
   > participant-bootstrap interval (−20 % to 40 %) includes both no
   > inheritance and full inheritance.

4. *Local similarity*:

   > Local similarity had a working positive control on all lPFC electrodes:
   > neighbours within 10 mm shared LWPC's reliable signal (excess 0.17, 95 %
   > CI 0.06–0.29, *p* = 0.004; nearest minus farthest bin *p* = 0.001) and
   > switch type's (nearest minus farthest *p* = 0.015), and congruency leaned
   > the same way (*p* = 0.06–0.09). The balance showed no local excess (0.01,
   > −0.05 to 0.08, *p* = 0.41) and no reliable within-participant variation
   > (−0.04, −0.12 to 0.03). The same held on RT-adjusted high gamma. In the
   > task-responsive subset the adaptation scores were too unreliable for the
   > analysis to detect local structure (reliabilities 0.04–0.14).

5. *Reliabilities and RT*:

   > Within-participant split-half reliabilities, from trial halves shared by
   > each participant's electrodes, were 0.21 for LWPC, 0.04 for LWPS, 0.46 for
   > congruency and 0.35 for switch type (task-responsive subset: 0.17, 0.14,
   > 0.39, 0.39). Splitting each electrode's trials separately biased them
   > downward (0.07, −0.09, 0.34, 0.24) but left the overlap unchanged
   > (*r* = 0.10 and 0.11). Removing the RT-linked part of high gamma left the
   > adaptation reliabilities nearly unchanged (LWPC 0.20) and lowered those of
   > the base effects (congruency 0.34, switch 0.29), and it roughly halved the
   > congruency–switch overlap (*r* = 0.22 to 0.13) while leaving the
   > LWPC–LWPS overlap intact.

### 0.9 The task-significant subset

The same signs and sizes at lower power (§18; 171 electrodes, 21 participants,
200 splits). LWPC–LWPS *r* = 0.077, *p* = 0.055; congruency–switch *r* = 0.167,
*p* = 0.001; Test 1 *r* = 0.080, *p* = 0.049, with the same matched-minus-crossed
gaps as all lPFC. Neither balance varies across parcels (*p* = 0.29 and 0.12)
or with position (*p* = 0.28 and 0.48), although delta's height slope is the
same size (−0.0075 SD/mm); a planted slope of that size reaches *p* < 0.05 only
26 % of the time on this layout. Local similarity has no power here (§19.8.6).
The supplement paragraph is §18.8.

### 0.10 What the current results replace

| Earlier statement | Now | Where |
|---|---|---|
| "LWPC and LWPS are carried by one intermixed lPFC population" (§15.1, §17.3) | "One population rather than two intermixed ones." Local similarity cannot test intermixing here: the balance has no reliable variation to arrange. | §19.8.6–§19.8.7 |
| The overlap shows the two adaptations co-localize as such | The overlap follows the base effects (0.097 → 0.028) | §19.8.5 |
| "The tilt is dorsoventral" (§15.1, §15.7) | The pre-specified test reports height, and Figure 5 uses height tertiles. Height and distance from the midline cannot be separated in lPFC; distance fits better (exploratory). | §16.6.4, §19.4 |
| Within-participant reliabilities LWPC 0.07, LWPS −0.09, congruency 0.34, switch 0.24 (subset 0.01, 0.00, 0.25, 0.26) | Biased low by the per-electrode split. With shared splits: 0.21, 0.04, 0.46, 0.35 (subset 0.17, 0.14, 0.39, 0.39) | §19.8.4 |
| "Switch is the less reliable map, 0.23 vs 0.35" | 0.35 vs 0.46; the argument holds | §19.8.4 |
| The noise-corrected overlap ratio (1.02–1.37 per electrode, 3.38 per parcel); "correlated at the noise ceiling" | Not estimable; not reported | §15.4 |
| The first local-similarity numbers (2026-10-02 code) | Not usable: the per-electrode split turned shared trial noise into similarity | §19.3, [Archive](#198-archived-parts-superseded-run-details) |
| The primary electrode set is open (§15.11) | All lPFC, stated in Methods; the subset is the replication | §0.7 |
| Figure 5 with four panels (2026-09-27), then two (2026-10-01) | One combined figure: the LWPC–LWPS scatter coloured by height tertile, with tertile centroids and the balance by height (`fig5_height`) | §19.4, §19.8.2 |
| Methods in §17.2; draft Results in §15.12, §16.7.2 and §17.3 | §0.7 and §0.8 | |
| LWPC–LWPS *p* = 0.0005 (2,000 permutations) | *p* = 0.0001 with 10,000 | §16.6.2 |

### 0.11 Open items

Collected from the archived lists (§15.13, §16.7.4, §17.4) and
[`paper_draft.md`](paper_draft.md) §1.7.

- [ ] **Confirm which folder holds which run.** The all-lPFC anatomy folder held
      the all-lPFC scores on 2026-10-05 (§18.1). Confirm its `per_split.csv` is
      the all-lPFC one too, find the subset's outputs, and copy each run to a
      folder name that says which set it holds.
- [ ] Confirm the subset's segregation run name, and that its
      `correlation_main_effects.json` gives congruency–switch *r* = 0.167
      (§18.1, §18.3).
- [ ] Run the anatomy job on the RT-adjusted segregation run (§18.10): the
      gradient and Test 1 on RT-adjusted scores.
- [ ] Recompute dm's split-half reliability from shared splits, and with it the
      scale for Test 2 (§16.6.2, §16.6.6). It needs the shared-split
      congruency–switch *r*. The S-N4 text in §0.8 quotes the old 0.08 until
      then.
- [ ] The sharper base-effect control: partial each adaptation on its own base
      effect only, LWPC on congruency and LWPS on switch type (§19.8.5).
- [ ] Plot the S-N4 panels: the matched-vs-crossed bars (already `fig5.png`
      panel b), the two balances by parcel, the four scores by band.
- [ ] Copy §0.7 and §0.8 into [`paper_draft.md`](paper_draft.md) §2.5 and §3.4,
      which still carry bracketed placeholders for the §19 results.
- [ ] Advisor decisions: all lPFC as the primary set; lPFC-only scope.
- [ ] Fill in the preprocessing paragraph, and archive the git commit,
      submission command and Slurm log with each run (§14).
- Not blocking, code clean-up: `joint_scatter.png` still labels its axes
  "disjoint half" and prints the noise-corrected ratio; `map_reliability`'s
  `between_noise_corrected_ci` still bootstraps splits rather than
  participants; the pooled coordinate model still uses signed *x*; Test 2's
  summary line still says "near 1 = inherited" instead of printing dm's
  reliability; optionally, a participant-bootstrap interval for the subset's
  shared-split overlap ratio (0.49, §19.8.4).

Done since those lists: the LWPC–LWPS *p* from 10,000 permutations (0.0001);
leave-one-participant-out on the overlap (*r* 0.081–0.113, §19.8.5); the subset
rerun with main effects (§18); the §19 runs with shared splits and RT coupling
(§19.8); Figure 5 drawn on the real scores (§19.8.2).

---

**Method and runbook (§1–§14).** What follows describes what each test
computes, how to run it and how to read every output. It was written for a
whole-brain run; §0.2 has the commands as run for lPFC.

## 1. Where the code lives

| Role | File / function |
|---|---|
| Analysis specification | `analysis_plans.md` › Concurrent-regulation plan, §§5–7 |
| Score estimation | `src/analysis/stats/stability_flexibility_segregation.py`: `compute_sensitivities_per_split`, `average_over_splits`, `add_responsiveness` |
| Core N4 statistics | `src/analysis/stats/stability_flexibility_anatomy.py`: `attach_scores`, `relative_score_roi_test`, `relative_score_coordinate_test`, `map_reliability`, `leave_one_subject_out` |
| Brain maps | same file: `SCORE_MAPS`, `plot_score_by_roi`, `plot_scores_on_brain`, `plot_score_maps` |
| Descriptive centres | same file: `score_centers_per_subject` |
| Overlap test and its participant-level version | `stability_flexibility_segregation.py`: `split_resolved_corr`, `participant_split_corr`, `_residualised_split_matrices`; shared splits: `compute_sensitivities_per_split(shared_split=True)` |
| Main-effect reference (§16) | `stability_flexibility_anatomy.py`: `delta_tracking_test`, `tilt_with_main_effect_covariate`; follow-ups `dcc_scripts/stats/n4_section16_followups.py` |
| §19 | `stability_flexibility_anatomy.py`: `coordinate_slope_by_participant`, `coordinate_slope_loso`, `overlap_controls`, `local_similarity`, `figure5_height`, `section19`; script `dcc_scripts/stats/n4_section19_followups.py`; line by line in [`n4_code_walkthrough.md`](n4_code_walkthrough.md) |
| DCC orchestration and output writing | `dcc_scripts/stats/stability_flexibility_anatomy_dcc.py`: `load_scores`, `run_score_anatomy`, `write_score_summary` |
| Environment-variable entry point | `dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py` |
| Slurm submitter / display wrapper | `dcc_scripts/stats/submit_stability_flexibility_anatomy_dcc.sh`, `sbatch_stability_flexibility_anatomy_dcc.sh` |
| Recommended upstream score-producing run | `dcc_scripts/stats/submit_stability_flexibility_segregation_dcc.sh` |
| Ground-truth regression tests | `tests/analysis/stats/test_stability_flexibility_anatomy.py` |
| §19 with electrodes as the unit (2026-10-06; main text, participant level to the supplement) | `coordinate_slope_by_electrode`, `electrode_split_corr`, `local_similarity`'s `*_electrode` columns, `figure5_height(unit=...)`; written up in [`n4_section19_electrode_level.md`](n4_section19_electrode_level.md) |

### Do not choose the wrong arm

`run_stability_flexibility_anatomy_dcc.py` defaults to `ARM=categorical`. That is
the older thresholded S/F-group enrichment analysis. **For this document, always
set `ARM=continuous`**. `ARM=both` runs categorical first and then writes N4 into
a `continuous/` subdirectory, but it costs more and can obscure which result is
primary.

`LABEL_SOURCE` is relevant to the categorical arm. It does not define the N4
continuous electrode set when `SCORES_CSV` is supplied. The continuous arm uses
the electrodes present in that score CSV and then applies the anatomical
`ROI_FILTER`.

---

## 2. The estimand and sign convention

### 2.1 What the two scores are

With the required `CONTRAST_MODE=proportion`, the two scores are:

```text
LWPC = (incongruent - congruent | low incongruent proportion)
     - (incongruent - congruent | high incongruent proportion)

LWPS = (switch - repeat | low switch proportion)
     - (switch - repeat | high switch proportion)
```

The implementation uses four cell means with equal weights and divides their
difference-of-differences by the pooled within-cell SD. It therefore avoids
letting the frequent cells dominate an unbalanced proportion design.

> **Positive LWPC or LWPS means adaptation:** the condition effect is smaller in
> the high-proportion block. Negative means the condition effect grew rather
> than shrank.

For each split, both effects are evaluated on both halves (`xA`, `xB`, `yA`,
`yB`). `average_over_splits` averages the two half-estimates and all requested
splits to give `x` (LWPC) and `y` (LWPS) per electrode. Cross-effect reliability
calculations continue to use the split-resolved values, so same-trial noise
cannot masquerade as co-localization.

### 2.2 Scaling and the relative score

`attach_scores` renames `x/y` to `lwpc_score/lwps_score`, then divides each
entire column by that effect's SD across all pooled electrodes:

```text
lwpc_s = lwpc_score / SD(lwpc_score across all electrodes)
lwps_s = lwps_score / SD(lwps_score across all electrodes)
delta  = lwpc_s - lwps_s
```

**Units.** `lwpc_score`/`lwps_score` are Cohen's *d* — a difference-of-differences
of cell means over the pooled within-cell SD of single-trial high-gamma — so the
raw scores are already unit-free. `lwpc_s`/`lwps_s`/`delta` are that *d* divided
by its own across-electrode SD, so their unit is **"SDs of this effect across
this dataset's electrodes"**: `lwpc_s = 1.5` means an electrode 1.5 cross-electrode
SDs above the mean LWPC *d*, not *d* = 1.5 and not a z-score (the mean is not
removed, and nothing is standardized within subject). They are a *relative*,
sample-dependent scale: the same electrode rescored against a different electrode
set gets a different number, so compare them within a run, never across runs, and
quote `lwpc_score`/`lwps_score` when an absolute effect size is wanted.

There is no subtraction of the pooled mean because the interaction asks about
relative spatial variation; a global offset cannot create between-ROI spread
after the nuisance intercept. There is deliberately no within-subject z-score.
With two electrodes it would force values to ±0.707 regardless of magnitude,
and with one electrode it would silently lose the subject. The scores are
already standardized, so one pooled spread adjustment per effect is sufficient.

Interpret `delta` as:

| Value | Meaning |
|---|---|
| `delta > 0` | relatively more LWPC-dominant |
| `delta < 0` | relatively more LWPS-dominant |
| `delta ≈ 0` | similar scaled LWPC and LWPS scores; it does **not** imply both effects are absent |

Always inspect the signed and magnitude maps alongside `delta`. For example,
`delta > 0` could mean strong positive LWPC, weak LWPS, negative LWPS, or both.

### 2.3 The electrode set is anatomical, not effect-selected

The defensible N4 population is all electrodes in the predeclared anatomical
scope (`ROIS=all` upstream for whole-brain N4, or a predeclared ROI scope), not
electrodes selected for an LWPC or LWPS effect. Selecting significant effects
before asking where those effects occur would make the anatomical answer partly
a consequence of the selection rule and would discard the many weak but
informative continuous observations.

For that reason, use `ELECTRODES=all`, `ROIS=all`, and
`CONTRAST_MODE=proportion` in the upstream score run for the whole-brain primary
analysis. A predeclared `ROI_FILTER=lpfc` is a legitimate separate lPFC analysis,
but is not a substitute for whole-brain N4.

---

## 3. Data flow through the code

The recommended path computes expensive scores once in the segregation job and
reuses its CSVs in the anatomy job:

```text
high-gamma Epochs on disk
  EPOCHS_ROOT_FILE
       │
       │ load_HG_ev1_rescaled_per_subject(..., acc_trials_only=True)
       ▼
per-subject MNE epochs + trial metadata
       │
       │ assemble_long_df(window_tmin, window_tmax,
       │                  effect_measure='cohens_d')
       ▼
long trial table
  subject, electrode, congruency, switchType,
  incongruent_proportion, switch_proportion, hg
       │
       │ compute_sensitivities_per_split(...,
       │     contrast_mode='proportion', n_splits=200)
       │ each electrode's trials are divided into disjoint halves
       ▼
per_split.csv
  one row / electrode / split: xA, xB, yA, yB
       │
       ├─ average_over_splits() + add_responsiveness()
       ▼
electrodes.csv
  subject, electrode, x, y, resp, ...
       │
       │ N4 anatomy job: load_scores()
       │ load atlas maps and optional fsaverage/MNI coordinates
       ▼
attach_scores()
  join ROI + Destrieux + coordinates
  pooled scaling; abs_lwpc, abs_lwps; delta = lwpc_s - lwps_s
       │
       ├─ restrict_to_roi(ROI_FILTER)
       ├─ build_coverage_matrix()
       │
       ├─ PRIMARY: relative_score_roi_test()
       │      delta ~ anatomy + responsiveness + subject dummies
       │      within-electrode score-label-swap null
       │
       ├─ LEVERAGE: leave_one_subject_out()
       │
       ├─ SECONDARY: relative_score_coordinate_test()
       │      delta ~ y + z + x + responsiveness + subject dummies
       │      all electrodes and each hemisphere
       │
       ├─ RELIABILITY: map_reliability()
       │      electrode and parcel levels
       │
       ├─ DESCRIPTIVE: score_centers_per_subject()
       │      subject × hemisphere weighted medoids
       │
       └─ FIGURES + CSV + JSON + summary.txt
```

### 3.1 Anatomy joins

The atlas cache supplies both:

- `roi`: a coarse group defined by `src/analysis/config/rois.py`; and
- `anat`: the raw Destrieux label.

`ANAT_LEVEL=group` tests `roi`; `ANAT_LEVEL=destrieux` tests `anat`.
`ANAT_LEVEL=auto` uses coarse groups for whole brain, but automatically switches
to Destrieux labels when `ROI_FILTER` contains one coarse group—otherwise every
electrode would have the same uninformative coarse label.

Coordinates follow the same reconstruction path used for brain figures:
`jim_mri.subject_to_info` → channel montage → forced MRI frame → subject
Talairach transform → fsaverage/MNI millimetres. Missing reconstruction files
produce warnings and partial coordinates rather than killing the primary ROI
test.

### 3.2 Responsiveness and subject nuisance terms

When the score CSV is produced by the full segregation job, `resp` is carried
into N4. In the current pipeline it defaults to mean absolute HG unless an
external responsiveness dictionary is supplied in code. It controls general
electrode gain so “large response everywhere” is not mistaken for preferential
LWPC or LWPS anatomy.

The implementation uses subject dummy variables as the fixed-effect,
no-shrinkage equivalent of the plan's `(1 | subject)` nuisance intercept. The
permutation supplies inference, so a parametric mixed-model degrees-of-freedom
approximation is not used.

---

## 4. The primary anatomical test

### 4.1 The question

At the selected anatomical level, the conceptual model is:

```text
delta_ij ~ anatomy_j + responsiveness_ij + subject_i
```

The omnibus statistic is a one-way `F`: between-anatomy variation in
nuisance-adjusted `delta` divided by within-anatomy variation. The `F` is only a
scale-free statistic; its reported significance comes from permutations, not a
parametric F distribution.

Because `delta` is formed within each electrode before the model, anatomy is
being tested against the difference between effect types. That is the required
**effect type × anatomy interaction**.

### 4.2 Coverage conditioning

iEEG sampling is clinically determined. `coverage_matrix.csv` has one row per
subject and one column per anatomical unit; a `1` means that subject has at least
one electrode in the unit. Anatomical units covered in fewer than
`MIN_SUBJECTS` are excluded before testing (default `3`).

This does not make implant coverage random. It prevents a label represented by
only one or two people from driving the tested ROI family. Report the retained
units and their subject counts with every anatomical claim.

### 4.3 The null

For each permutation, every electrode independently keeps its subject,
coordinates, anatomy, responsiveness, and pair of observed scores, but LWPC and
LWPS are randomly exchanged. Algebraically this is a sign flip of `delta`.

The Monte Carlo p-value is `(extreme + 1) / (N_PERM + 1)`. With the default
10,000 permutations its smallest possible value is about `0.0001`. Never report
`p = 0`.

### 4.4 Omnibus and per-anatomy rows

The omnibus permutation `p` answers whether relative LWPC/LWPS balance varies
anywhere across the retained anatomical units. `delta_per_roi.csv` then provides
unit-level descriptions:

- `mean_delta`: raw mean `delta`;
- `mean_delta_adj`: mean after removing subject and responsiveness nuisance
  effects—the quantity used by the statistic;
- `p`: two-sided within-electrode-swap permutation p for that adjusted mean;
- `q`: Benjamini–Hochberg value across the retained units;
- `n_electrodes`, `n_subjects`: the sampling behind the row.

Lead with the omnibus test. Use `q`, not an isolated raw row-level `p`, when
identifying the units that explain a significant omnibus result. A positive
adjusted mean is LWPC-dominant; a negative adjusted mean is LWPS-dominant.

### 4.5 Leave-one-subject-out leverage

The same ROI test is rerun after dropping each subject. These folds use fewer
permutations (`max(1000, N_PERM/10)`) because this is an estimate-leverage check,
not a second inferential family.

Read shifts in `observed_stat`, not only shifts in `p`: permutation p-values can
saturate at their resolution floor. A large F collapse when one subject is
removed means the pooled result is fragile even if every fold still prints a
small p-value.

---

## 5. Secondary coordinate test

When coordinates are available, the pipeline also fits, separately by
hemisphere as well as in the pooled coordinate table:

```text
delta ~ mni_y + mni_z + mni_x + responsiveness + subject
```

The block `F/p` tests whether the three-coordinate block predicts relative
effect dominance. Per-axis slopes and permutation p-values describe direction:

| Axis | Positive slope means |
|---|---|
| `mni_y` | LWPC dominance increases anteriorly |
| `mni_z` | LWPC dominance increases superiorly |
| `mni_x` | LWPC dominance increases toward more positive/rightward x; interpret only within hemisphere |

The code reports an `all` fit and, when possible, `lh` and `rh`. Prefer the
hemisphere-specific x slopes because mirrored bilateral x coordinates can
cancel. Treat axis p-values as follow-ups to the coordinate-block test and be
explicit about any multiplicity policy used in the manuscript.

The coordinate model is secondary to the coverage-conditioned categorical
anatomy test, but it is the preferred analysis for a directional claim such as
“LWPC dominance is more anterior.” It uses all positions rather than reducing a
spatial distribution to one point.

---

## 6. Brain maps of continuous scores

The job writes five maps from the same `scores_with_anatomy.csv`:

| Base filename | Values | Colour scale | What it shows |
|---|---|---|---|
| `score_map_lwpc_s.png` | signed pooled-scaled LWPC | symmetric `coolwarm` | direction and location of LWPC adaptation |
| `score_map_lwps_s.png` | signed pooled-scaled LWPS | symmetric `coolwarm` | direction and location of LWPS adaptation |
| `score_map_abs_lwpc.png` | `abs(lwpc_s)` | sequential `viridis` | LWPC magnitude regardless of sign |
| `score_map_abs_lwps.png` | `abs(lwps_s)` | sequential `viridis` | LWPS magnitude regardless of sign |
| `score_map_delta.png` | `lwpc_s - lwps_s` | symmetric `coolwarm` | the relative map tested by N4 |

Each map bins a continuous value into nine colours because the shared brain
renderer accepts one colour per electrode set. Its scale is clipped at the 98th
percentile so a few extremes do not flatten the remaining electrodes. The
adjacent `*_colorbar.png` records the actual range; do not compare hues between
figures without their own colourbars.

Signed and delta scales are centred on zero. Magnitude scales are sequential.
The maps pool electrodes for display, so dense implants are visually prominent.
They are **visualizations, not independent evidence**, and should be read with
`coverage_matrix.csv`, the ROI test, and the leave-one-subject-out table.

If PyVista/MNE/reconstruction/display dependencies are unavailable, the code
writes the colourbar and a `*_by_roi.png` fallback instead. That is a successful
statistical run, but it is not a rendered cortical surface. The Slurm wrapper
uses `xvfb-run` to provide a virtual display; missing recon/template data can
still trigger the fallback.

### `delta_by_roi.png`

This flat companion figure is always attempted. Bars are mean `delta` ± SEM,
dots are electrodes, and labels report electrode and covered-subject counts.
The zero line separates relative LWPC dominance from relative LWPS dominance.
The error bars are descriptive; significance comes from the swap test and the
subject/responsiveness-adjusted statistics, not whether a bar's SEM crosses zero.

### `joint_scatter.png`

Every point is an electrode (`lwpc_s` on x, `lwps_s` on y). It reports both the
pooled and within-subject correlation and, when `per_split.csv` is supplied,
annotates the split-half ceiling. It is a descriptive diagnostic, useful for
checking whether one participant or a few electrodes carry the cloud, and it is
not the primary anatomy interaction.

It is not the co-localization figure to report either. The points are
split-averaged scores, which equal the full-data scores (§15.2), although the
axis labels say "disjoint half", and they are not residualised on responsiveness
or centred within participant. So neither the points nor any number printed on
them is the pre-specified test (`split_resolved_corr`). The figure to report is
the combined Figure 5, `fig5_height.png` (§19.4).

---

## 7. Reliability and the noise ceiling

A low LWPC–LWPS spatial correlation is uninterpretable unless each map is
reliable enough to correlate with anything. `map_reliability` uses the
split-resolved table:

```text
between = mean over splits of ½[corr(xA, yB) + corr(xB, yA)]
LWPC reliability = mean corr(xA, xB)
LWPS reliability = mean corr(yA, yB)
noise-corrected between = between / sqrt(rel_LWPC × rel_LWPS)
```

It computes these at both electrode and parcel level. The cross-half pairing
prevents shared trial noise from inflating similarity.

Read the result comparatively:

- `between` near the two within-effect reliabilities: maps are about as similar
  as measurement permits;
- `between` well below healthy positive reliabilities: evidence for different
  maps is more credible;
- either reliability near zero or negative: the between-map correlation and
  attenuation correction are not interpretable; collect/aggregate better or
  soften the claim;
- corrected values can exceed ±1 under noisy finite-sample estimates; treat them
  as an instability warning, not a literal correlation.

Three things to know before quoting any of these numbers (worked through on
real data in [§15.4](#154-how-reliable-are-the-maps), archived, and settled in
§19.8.4):

- `between_noise_corrected` is computed from **Pearson** correlations whatever
  `method` is. The attenuation formula is Pearson algebra, and a Spearman ratio
  can sit far above 1.
- `between_noise_corrected_ci` bootstraps the **splits**. The splits all
  re-divide the same trials and electrodes, so it only measures which trials fell
  in which half; it is not a confidence interval. Resample participants for
  sampling uncertainty.
- The within-participant reliabilities that `split_resolved_corr` reports (the
  `min_elec` sweep) are biased low when `compute_sensitivities_per_split`
  draws a new trial split for each electrode, which is the pipeline's default.
  Confirmed on the real data (§19.8.4: all-lPFC LWPS −0.09 with per-electrode
  splits, +0.04 with shared ones). Quote within-participant reliabilities only
  from a table scored with `shared_split=True` (`SHARED_SPLIT=1`, or
  `n4_section19_followups.py --long-df`); the overlap r itself is not affected
  (§19.7).

The anatomy job also reruns the disjoint-half electrode correlation with
`min_elec` equal to 1, 2, and 3. `min_elec` drops whole subjects that have too few
eligible electrodes, so changes across `min_elec_sweep.csv` reveal an important
sample-definition sensitivity.

**Always pass `PER_SPLIT_CSV`.** The job is allowed to run without it, but then
the ceiling and min-electrode sweep are absent and the summary explicitly marks
the spatial correlation as unreportable.

---

## 8. Centres / medoids — descriptive only

The centre panel exists to describe the maps in millimetres, not to establish
anatomical separation. A single pooled centroid is invalid here because subjects
with more electrodes move the location estimate, clinical coverage can mimic a
physiological centre, bilateral distributions can put a centroid in the
midline/white matter, and signed weights can cancel.

The implemented version avoids the worst failures:

- one LWPC centre and one LWPS centre per **subject × hemisphere**;
- the same electrodes for both effects;
- non-negative weights `abs_lwpc` and `abs_lwps`;
- at least three electrodes in a subject × hemisphere group;
- a weighted **medoid** by default—the observed electrode minimizing weighted
  distance to the other electrodes;
- LWPC-minus-LWPS displacement within group; and
- the same within-electrode score-label-swap null.

`score_centers.csv` contains:

| Column | Meaning |
|---|---|
| `subject`, `hemi` | independent descriptive group |
| `n_electrodes` | sites available to locate both medoids |
| `dx`, `dy`, `dz` | LWPC medoid minus LWPS medoid in mm |
| `distance` | Euclidean separation in mm |
| `p_anterior` | within-group swap p for `dy`; descriptive diagnostic |

The summary averages displacement over the eligible subject × hemisphere
groups. Positive `dy` means the LWPC medoid is anterior to the LWPS medoid;
positive `dz` means superior; positive `dx` means more positive x. Lead with the
coordinate regression for an anterior/posterior claim and present the medoids as
an intuitive secondary panel. Do not call their p-values the primary anatomical
test.

---

## 9. How to run it

Run commands from the repository root unless a command explicitly changes
directory.

This section is the general runbook, written for a whole-brain run. The exact
sequence that produced the current all-lPFC results is §0.2. Two submitter
defaults to know: `submit_stability_flexibility_segregation_dcc.sh` now has
`RT_ADJUST_HG=1` and `MAIN_EFFECTS=1` as defaults, so set `RT_ADJUST_HG=0` for
the primary (raw high gamma) scores.

### 9.1 First validate the entire path with synthetic data

This exercises scores, atlas joins, primary and coordinate tests, reliability,
maps, centres, persistence, and summary writing without accessing real epochs:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

The synthetic default plants a spatial gradient. Also run the null:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0 N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Use 1,000 permutations only as a path check. Use the default 10,000 for the
reported result.

### 9.2 Produce reusable real-data scores

Edit the hard-coded `EPOCHS_ROOT_FILE`, `WINDOW_TMIN`, and `WINDOW_TMAX` near the
top of `submit_stability_flexibility_segregation_dcc.sh` so they match the
predeclared analysis. Then run the anatomically defined whole-brain score set:

```bash
cd dcc_scripts/stats
ROIS=all ELECTRODES=all CONTRAST_MODE=proportion EFFECT_MEASURE=cohens_d \
N_SPLITS=200 N_PERM_CORR=10000 N_PERM_LABEL=2000 \
  bash submit_stability_flexibility_segregation_dcc.sh
```

Why these values matter:

- `ROIS=all`, `ELECTRODES=all`: no effect-based or lPFC-only selection before
  the whole-brain anatomy question;
- `CONTRAST_MODE=proportion`: score the LWPC/LWPS interactions, not congruency
  and switch main effects;
- `EFFECT_MEASURE=cohens_d`: the signed, balanced, unit-free measure specified
  for N4;
- `N_SPLITS=200`: stable disjoint-half averages and a split-resolved ceiling.

Do **not** use `SCATTER_ONLY=1` as the N4 input. That path is diagnostic and does
not write the complete full-run product set expected here.

The upstream directory is printed as `Save dir:` in the Slurm log and normally
has this shape:

```text
dcc_scripts/stats/results/<epochs>/segregation_results/
  window_<tmin>to<tmax>s_all_all_rois_proportion_cohens_d_fdr_bh/
```

Verify that it contains at least `electrodes.csv` and `per_split.csv`.

### 9.3 Run N4 from those finished scores — recommended route

Set both paths to the files from the **same** segregation run:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/segregation_results/window_..."

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER='' ANAT_LEVEL=group \
MIN_SUBJECTS=3 N_PERM=10000 SEED=0 \
MAKE_BRAIN=1 BRAIN_HEMI=both USE_COORDS=1 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Environment variables not explicitly repeated in the submitter's
`--export=...` list still reach Slurm through `--export=ALL`; the shell-prefix
form above is therefore intentional. Use absolute CSV paths because the batch
job's working directory may differ.

The score-CSV route does not load epochs. It still needs the electrode atlas and,
for coordinate/centre panels, reconstruction files. Set `ROI_DICT_DIR` if the
atlas JSON cache is not in its normal lab location.

### 9.4 Restricted lPFC / Destrieux follow-up

For a predeclared lPFC analysis using raw labels inside lPFC:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/the/same/wholebrain/segregation/run"

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER=lpfc ANAT_LEVEL=destrieux HIST_TOP_N=20 \
MIN_SUBJECTS=3 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

`ANAT_LEVEL=auto` would also select Destrieux here. `HIST_TOP_N` belongs to the
categorical histogram path and does not change the continuous primary test or
`delta_by_roi.png`; setting it here is harmless but not necessary.

### 9.5 Compute scores inside the anatomy job — supported but expensive

If no upstream score run exists, omit `SCORES_CSV` and `PER_SPLIT_CSV` and let
the anatomy job load epochs and score them:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=real ROI_FILTER='' ANAT_LEVEL=group \
CONTRAST_MODE=proportion ELECTRODES=all N_SPLITS=200 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Before doing this, edit the anatomy submitter's hard-coded `EPOCHS_ROOT_FILE`
and window. The submitter also sets `SCORES_CSV=${SCORES_CSV:-"$SEG_RUN/electrodes.csv"}`,
which treats an empty value as unset, so blank the `SEG_RUN` default in the file
rather than on the command line. The DCC core pins the in-job estimator to
`EFFECT_MEASURE='cohens_d'`; it is not an environment variable exposed by the
anatomy runner. Reusing a verified segregation `electrodes.csv` is nevertheless
clearer and safer.

**`ELECTRODES` does not select electrodes in the anatomy job, on either route.**
On the CSV route the electrode set is whatever the CSV contains
(`load_scores`, `stability_flexibility_anatomy_dcc.py`). On this in-job route,
`run_stability_flexibility_anatomy_dcc.py` hard-codes `ROIS_DICT = None`, and
`resolve_electrodes_to_keep` keeps every channel when that is `None`. The only
visible effect of `ELECTRODES=sig` is the `_sig` suffix on the output directory.
To analyse task-significant electrodes, produce their scores upstream: set
`ELECTRODES=sig` in `submit_stability_flexibility_segregation_dcc.sh` (it is
hard-coded there too, so edit the line), run it, and point `SCORES_CSV` and
`PER_SPLIT_CSV` at the resulting `window_<tmin>to<tmax>s_sig_<rois>_...`
directory.

### 9.6 Run without reconstruction/display support

The primary ROI test needs anatomy labels but not coordinates or 3-D rendering:

```bash
cd dcc_scripts/stats
ARM=continuous SCORES_CSV="/abs/path/electrodes.csv" \
PER_SPLIT_CSV="/abs/path/per_split.csv" USE_COORDS=0 MAKE_BRAIN=0 \
ANAT_LEVEL=group N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

This intentionally omits the coordinate test, cortical surfaces, and centres.
It still produces the primary test, coverage table, ROI figure, reliability,
scatter, leverage sweep, CSVs, JSON, and text summary.

### 9.7 Local invocation for debugging

The entry point can be run directly from the repository root, although real
data/recon path discovery is lab-specific:

```bash
ARM=continuous DATA_SOURCE=synthetic N_PERM=200 MAKE_BRAIN=0 \
python dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py
```

The Slurm wrapper is preferable for real maps because it launches Python under
`xvfb-run`.

---

## 10. Output directory and every output

The runner constructs:

```text
dcc_scripts/stats/results/<tag>/
  anatomy_<label_source>_<scope>_window_<tmin>to<tmax>s_<electrodes>/
    continuous/
```

On the CSV route `<tag>` is not derived from the score filename. With the default
`LABEL_SOURCE=a1` and no epoch directory it currently resolves to the synthetic
fallback tag; with `LABEL_SOURCE=power_traces` it resolves to `power_traces`.
This is only an output-path naming quirk—the loaded CSV remains the data source.
Trust the `Save dir:` line in the job log and archive the exact command/CSV paths
with the result. All files below live in `continuous/`.

### 10.1 Read these first

| File | Contents | How to use it |
|---|---|---|
| `summary.txt` | human-readable primary F/p, per-anatomy rows, LOSO, coordinates, centres, ceiling, correlations | Start here; it states skipped components explicitly. |
| `score_anatomy.json` | machine-readable compact summary of the same analysis | Manuscript/table automation and provenance. It excludes the full permutation arrays. |
| `delta_by_roi.png` | mean ± SEM and electrode dots for the tested relative score | Flat visual companion to the primary model. |
| `coverage_matrix.csv` | subject × tested anatomy presence/absence | Required context for every anatomical conclusion. |

### 10.2 Input/intermediate audit tables

| File | Important columns / interpretation |
|---|---|
| `scores.csv` | loaded or freshly computed per-electrode input (`x`, `y`, `resp`). This is a copy inside the run for provenance. |
| `per_split.csv` | `xA`, `xB`, `yA`, `yB` for every split/electrode; source of reliability and `min_elec` sensitivity. |
| `scores_with_anatomy.csv` | canonical audit table: raw scores, scaled scores, magnitudes, `delta`, ROI, Destrieux label, and coordinates. Use this to reproduce plots or inspect exclusions. |

In `scores_with_anatomy.csv`, check:

- unique `electrode_id` and sensible `subject` counts;
- fraction missing `roi` and `anat`;
- fraction missing MNI coordinates (coordinates may be missing without harming
  the primary ROI test);
- finite `lwpc_s`, `lwps_s`, and `delta`;
- that the requested `ROI_FILTER` was actually applied; and
- extreme values before accepting a percentile-clipped display.

### 10.3 Primary test and robustness tables

| File | Columns | Interpretation |
|---|---|---|
| `delta_per_roi.csv` | anatomy label, electrode/subject N, raw and adjusted means, p, q | Follow-up anatomy rows after the omnibus test. Sign gives relative dominance. |
| `delta_roi_loso.csv` | dropped subject, observed F, p, electrode N | Subject leverage; compare F against `(none)`. |
| `min_elec_sweep.csv` | threshold, cross-effect correlation/p, N, reliabilities, corrected r | Sensitivity to dropping subjects with fewer than 1/2/3 electrodes. Not the primary ROI test. |

### 10.4 Map and scatter figures

| Files | Interpretation |
|---|---|
| `joint_scatter.png` | electrode-level LWPC–LWPS relationship with pooled/within-subject diagnostics and ceiling annotation; a diagnostic on uncorrected scores, not the figure to report (§6) |
| `score_map_<value>.png` | rendered surface for each of the five values |
| `score_map_<value>_colorbar.png` | required scale for that map |
| `score_map_<value>_by_roi.png` | fallback only when the surface cannot render |

Depending on `BRAIN_HEMI` and the rendering helper, additional view-specific
files may accompany the combined map. `score_anatomy.json` records the combined
path actually returned, or the fallback path.

### 10.5 Centre table

`score_centers.csv` is written only if at least one subject × hemisphere group
has coordinates and at least three eligible electrodes. Its absence can simply
mean insufficient coordinates/coverage; read `summary.txt` and the job log
before treating absence as a pipeline error.

---

## 11. Interpretation checklist

Use this order. It prevents an attractive surface or centroid from outrunning
the actual test.

### A. Validate the population and coverage

- [ ] `ARM=continuous`, `CONTRAST_MODE=proportion`, and the intended time window.
- [ ] Scores came from an anatomically defined set (`all` for whole brain), not
      LWPC/LWPS-significant electrodes.
- [ ] `SCORES_CSV` and `PER_SPLIT_CSV` came from the same upstream run.
- [ ] Mapped electrode and subject counts are plausible.
- [ ] Retained anatomical units meet `MIN_SUBJECTS`; report their coverage.
- [ ] Missing coordinates affect only coordinate/maps/centres, not the primary
      label-based ROI test.

### B. Read the primary interaction

- [ ] Read the omnibus `F` and permutation `p` in `summary.txt`.
- [ ] If significant, use adjusted means and BH `q` in `delta_per_roi.csv` to
      describe where relative dominance lies.
- [ ] Say “relative LWPC/LWPS balance varies by anatomy,” not “LWPC is present
      here and LWPS absent there.”
- [ ] Do not interpret `delta` without checking both signed component maps.

### C. Check robustness

- [ ] Inspect LOSO changes in **F**, especially the largest shift.
- [ ] Compare pooled and within-subject LWPC–LWPS correlations.
- [ ] Inspect `min_elec=1,2,3` rather than reporting only the default survivor
      set.
- [ ] Interpret between-map similarity only relative to both split-half
      reliabilities.
- [ ] Flag undefined/unstable noise correction when reliability is non-positive.

### D. Add secondary spatial descriptions

- [ ] Coordinate block test first, then directional slopes.
- [ ] Prefer hemisphere-specific interpretation of `mni_x`.
- [ ] Use maps to show the data and coverage, not as independent tests.
- [ ] Describe weighted medoids in millimetres as convergent/descriptive only.

### Suggested result language

If the omnibus test is significant and robust:

> The within-electrode balance of pooled-scaled LWPC and LWPS scores varied
> across coverage-eligible anatomical units (swap-permutation omnibus F = …,
> p = …). Adjusted relative scores were LWPC-dominant in … and LWPS-dominant in
> … (BH q = …). The statistic remained similar in leave-one-subject-out fits,
> and the spatial comparison was interpretable given split-half reliabilities of
> … and … .

If it is null with a healthy ceiling:

> Relative LWPC/LWPS balance did not vary detectably across coverage-eligible
> anatomical units (F = …, swap-permutation p = …). Split-half spatial
> reliabilities were … and …, and cross-effect similarity was …; thus the null
> is not readily explained by wholly unreliable maps.

If it is null with a poor ceiling:

> The anatomy interaction was not significant, but the LWPC and/or LWPS map had
> low split-half reliability. The analysis therefore cannot distinguish a
> shared anatomical organization from insufficient precision, and the null is
> inconclusive.

For the lPFC runs, the current worked version of this language is the paper's
text: [`paper_draft.md`](paper_draft.md) §3.4 (all lPFC) and §3.5 (the
task-significant subset, from §18.8 here).

---

## 12. Parameters that change the answer

| Variable | Default in runner | Recommendation / effect |
|---|---:|---|
| `ARM` | `categorical` | **Set `continuous`.** |
| `SCORES_CSV` | unset | Recommended: full segregation run's `electrodes.csv`. |
| `PER_SPLIT_CSV` | unset | Recommended/needed for reportable ceiling and min-electrode sweep. |
| `N_SPLITS` | `200` | Used only if scoring inside anatomy; more splits stabilize estimates but cost time. |
| `CONTRAST_MODE` | `proportion` | Must remain `proportion` for LWPC/LWPS N4. |
| `ROI_FILTER` | whole brain | Changes the tested population; predeclare whole-brain primary vs regional follow-up. |
| `ANAT_LEVEL` | `auto` | Whole brain → coarse groups; one-group restriction → Destrieux. Explicit values improve provenance. |
| `MIN_SUBJECTS` | `3` | Coverage threshold; higher is more conservative but drops anatomical units. |
| `N_PERM` | `10000` | Primary swap-null resolution; lower only for dry runs. |
| `SEED` | `0` | Reproducibility of permutations and plot jitter. |
| `USE_COORDS` | `1` | Controls coordinate test and centres, not the primary ROI test. |
| `MAKE_BRAIN` | `1` | Controls five surface/fallback maps, not statistics. |
| `BRAIN_HEMI` | `both` | Display choice (`both`, `lh`, `rh`, `split`). `split` draws one hemisphere per panel in a window twice as wide. |
| `BRAIN_ZOOM` | renderer default | Per-panel camera zoom (`<1` zooms out). Display choice only; lower it to widen the gap between the `split` hemispheres. |
| `ELECTRODES` | runner `all`, submitter `sig` | CSV route inherits its input population; in-job scoring must be changed to `all` for N4. |

Two non-obvious distinctions:

1. `ALPHA` labels significance in the text and is used by some score estimators;
   the N4 primary decision still comes from the reported permutation `p`.
2. `FDR_CORRECTION` and `LABEL_SOURCE` largely describe the categorical arm;
   do not mistake them for filtering continuous scores loaded from CSV.

---

## 13. Common failure modes

### “My job ran categorical anatomy”

`ARM` was omitted. Rerun with `ARM=continuous`; do not interpret the categorical
S/F table as beats 5–7.

### “The anatomy job says the ceiling was not computed”

`PER_SPLIT_CSV` was omitted or unreadable. Point it to `per_split.csv` from the
same segregation run as `SCORES_CSV`.

### “Only lPFC electrodes are present in my whole-brain run”

The upstream segregation submitter defaults to `ROIS=lpfc`. Recompute scores
with `ROIS=all`; changing only `ROI_FILTER` cannot restore electrodes that never
entered `electrodes.csv`.

### “I set `ELECTRODES=sig` but the run used every electrode”

`ELECTRODES` has no effect on which electrodes the anatomy job analyses. With
`SCORES_CSV` set, the electrodes are the ones in that CSV; computing scores in
the job keeps every channel because the runner hard-codes `ROIS_DICT = None`.
The output directory still ends in `_sig`, so rename or delete it. To analyse
task-significant electrodes, run the segregation job with `ELECTRODES=sig` and
point `SCORES_CSV` / `PER_SPLIT_CSV` at that run ([§9.5](#95-compute-scores-inside-the-anatomy-job--supported-but-expensive)).

### “No cortical surfaces appeared”

Read the Slurm log for the renderer exception and look for `*_by_roi.png` files.
Statistics may be complete. Confirm `xvfb-run`, PyVista/MNE, fsaverage/template
surfaces, and subject reconstructions are available.

### “No coordinate result or `score_centers.csv`”

Either `USE_COORDS=0`, reconstruction lookup failed, coordinates were missing,
or no subject × hemisphere had three eligible electrodes. This does not
invalidate the label-based primary test.

### “The omnibus is significant but no row has q < .05”

The omnibus asks whether any between-unit variation exists collectively;
row-level BH tests ask which adjusted unit means differ from zero. These are not
identical questions. Report the omnibus and avoid naming a specific unit as the
driver unless its corrected follow-up supports that claim.

### “The delta map looks dramatic but the primary test is null”

The display pools electrodes, clips extremes, and does not account visually for
subject leverage or coverage. Trust the prespecified swap test and LOSO result;
describe the map as exploratory/descriptive.

### “A medoid p-value is smaller than the ROI-test p-value”

Do not promote it. The centre statistic answers a narrower, coverage-sensitive
location-summary question. The plan explicitly makes it descriptive; the ROI
interaction and coordinate regression remain the inferential analyses.

---

## 14. Reproducibility and tests

Before a production run:

```bash
pytest tests/analysis/stats/test_stability_flexibility_anatomy.py -v
pytest tests/analysis/stats/test_main_effect_anatomy.py tests/analysis/stats/test_section19_anatomy.py -v
```

The second line covers the main-effect block (§16) and §19.

The continuous-arm tests verify pooled scaling without losing tiny subjects,
electrode-ID/anatomy/coordinate joins, recovery of planted and null ROI effects,
coverage filtering, sign invariance of the swap test, coordinate-gradient
recovery, reliability degradation with noise, real-electrode medoids, LOSO
coverage, and surface-render fallback behavior.

For every reported run archive:

- the git commit;
- the exact submission command and Slurm log;
- absolute upstream CSV paths;
- epoch directory, window, electrode and ROI scope;
- `ANAT_LEVEL`, `MIN_SUBJECTS`, `N_SPLITS`, `N_PERM`, and seed;
- `summary.txt`, `score_anatomy.json`, all CSVs, and all figures; and
- warnings about missing atlas labels, coordinates, or render fallback.

Related reading:

- [`n2_direction_tests.md`](n2_direction_tests.md) — the direction test that
  should be settled before N4;
- [`analysis_plans.md` › Concurrent-regulation plan](analysis_plans.md#concurrent-regulation-plan)
  §§5–7 — the statistical rationale and reporting hierarchy;
- [`stability_flexibility_battery.md` › Data flow walk-through](stability_flexibility_battery.md#data-flow-walk-through) —
  broader A1–A6 data flow; and
- [`stability_flexibility_battery.md` › Outputs guide](stability_flexibility_battery.md#outputs-guide)
  — the wider segregation/anatomy output family.

---

## 16. Main effects as the reference for the tilt

The question from [`analysis_plans.md` › Closing figure plan](analysis_plans.md#closing-figure-plan): is the
dorsoventral tilt in delta (LWPC − LWPS) inherited from how the base effects,
congruency and switch type, are organized?

### 16.1 How the scores are made

`compute_sensitivities_per_split(..., contrast_mode='proportion',
main_effects=True)` also scores each process's main effect, from the same four
cells as its adaptation effect with equal weight over the proportion levels
(`W_MAIN`), on the same trial halves:

```text
congruency = ½[(i − c | 25 %) + (i − c | 75 %)]      per_split: mxA, mxB   electrodes: mx
switch     = ½[(s − r | 25 %) + (s − r | 75 %)]      per_split: myA, myB   electrodes: my
```

LWPC and LWPS come out identical with or without it; the split draws do not
change. Do not use a separate `CONTRAST_MODE=condition` run instead. Its halves
do not line up with these. It also weights congruency by trial count over the
proportion levels, so block effects leak into it (see "Two traps in a separate
condition-mode run" in the plan).

`attach_scores` turns `mx`/`my` into `cong_s`, `switch_s` and
`dm = cong_s − switch_s` (positive = relatively congruency-dominant), with the
same pooled scaling as `lwpc_s`/`lwps_s`. The swap null is valid for dm because
it is a paired difference, so every test in this document takes
`value_col='dm'`.

### 16.2 How to run it

1. Segregation, once per electrode set. `MAIN_EFFECTS=1` is the default in
   the submitter. The output directory gets a `_main_effects` suffix, so the
   archived LWPC/LWPS-only runs are never overwritten. Scoring takes about
   twice as long.

   ```bash
   cd dcc_scripts/stats
   ROIS=lpfc CONTRAST_MODE=proportion EFFECT_MEASURE=cohens_d N_SPLITS=200 \
   N_PERM_CORR=10000 bash submit_stability_flexibility_segregation_dcc.sh
   ```

   For the task-significant set, edit `ELECTRODES=sig` in the submitter as
   before (§9.5).

2. Anatomy. `SEG_RUN` in `submit_stability_flexibility_anatomy_dcc.sh` now
   points at the all-lPFC `_main_effects` directory; change `all` to `sig` in it
   for the task-significant set. When the scores carry `mx`/`my`, the
   continuous arm runs the main-effect block after the usual N4 outputs. With
   older CSVs it runs exactly as before.

   ```bash
   ARM=continuous ROI_FILTER=lpfc ANAT_LEVEL=destrieux N_PERM=10000 \
     bash submit_stability_flexibility_anatomy_dcc.sh
   ```

### 16.3 What it writes (in `continuous/`)

| Output | Contents |
|---|---|
| `summary.txt`, block `MAIN EFFECTS` | dm's own anatomy, laid out like delta's blocks: the primary label test with its per-label rows, the leave-one-subject-out sweep, and the coordinate test (pooled, lh, rh). Then the main-effect reliabilities, Test 1 and Test 2. |
| `score_anatomy.json`, key `main_effects` | the same numbers |
| `dm_per_roi.csv`, `dm_by_roi.png` | the label test on dm; columns as in `delta_per_roi.csv`, with `mean_dm`/`mean_dm_adj` for `mean_delta`/`mean_delta_adj` |
| `dm_roi_loso.csv` | the label test on dm with each subject dropped (as `delta_roi_loso.csv`) |
| `dm_coordinates.csv` | `dm ~ y + z + x + resp + subject`: one row per fit (`all`, `lh`, `rh`) and axis, with the slope, its p, and the fit's block F and p |
| `score_map_cong_s.png`, `score_map_switch_s.png`, `score_map_dm.png` | main-effect maps (with `MAKE_BRAIN=1`) |
| `delta_tracking.csv` | Test 1 |
| `tilt_with_dm.csv` | Test 2 |
| `fig5.png`, `fig5.pdf`, `fig5a_points.csv`, `fig5b_bars.csv` | The 2026-10-01 two-panel Figure 5 (overlap at both levels; matched-vs-crossed bars), drawn by `sfa.figure5` after Test 1. Its bars are now an S-N4 panel. The current Figure 5 is `fig5_height` (§19.4). |

Read dm's anatomy like delta's (§4, §5), with congruency in place of LWPC:
a positive adjusted mean or slope means relatively congruency-dominant there, or
increasingly so along that axis.

The segregation run's own `summary.txt` and `correlation_main_effects.json`
report the congruency–switch co-localization on the same halves.

### 16.4 The two tests

**Test 1, `delta_tracking_test`: does dm track delta?** This is
`split_resolved_corr`, the pre-specified co-localization test, applied to dm
and delta: dm from one half against delta from the other within every split,
residualized on responsiveness, centred within participant, Spearman,
participants with at least three electrodes, within-participant permutation
null. Rows in `delta_tracking.csv`:

- `dm vs delta`: the primary number. A shared smooth gradient counts here,
  because that is the "inherited" hypothesis.
- `dm vs delta, + MNI covariates`: the same with coordinates partialled out
  within participant (`split_resolved_corr(covariates=...)`). Does the tracking
  go beyond shared geography?
- `congruency vs LWPC`, `switch vs LWPS`, and the two crossed pairings as
  controls.

The reliabilities in each row are within participant, so they are biased low
(§15.4). Compare them, and never divide by them.

**Test 2, `tilt_with_main_effect_covariate`: does delta's z tilt survive dm?**
Rows in `tilt_with_dm.csv`, all `relative_score_coordinate_test`'s fit with the
swap null:

| fit | question |
|---|---|
| `dm` | Do the main effects tilt? An inherited tilt needs a negative z slope: congruency weaker than switch dorsally. |
| `delta` | The §15.7 tilt, on the same electrodes. |
| `delta + dm` | The tilt with dm as a covariate, on full-data scores. dm and delta share trials here, with a noise correlation of about +0.05. |
| `delta, split halves` / `delta + dm from the opposite half` | Each half of every split refitted with dm from the other half, then averaged. This is the clean version. No p-value; read the full-data rows for it. |

`shrinkage = 1 − with/without`. It is only meaningful when the `delta` row
itself is significant. The swap null flips only the adaptation labels, which
also breaks delta's link with dm, so the null is conservative.

⚠️ **The shrinkage is not on a 0-to-1 scale.** A covariate measured with error
removes only the reliable part of what it explains. A tilt carried entirely by
dm therefore shrinks by about dm's split-half reliability, not by 1. In lPFC
that reliability is about 0.08 (§16.6.6; computed from per-electrode-split
reliabilities, which are biased low, so the true value is probably somewhat
higher), so the shrinkage cannot tell an
inherited tilt from one that is not. The "near 1 = inherited" hint printed in
`summary.txt` came from the synthetic check, where dm's reliability is 0.85.
Read the shrinkage next to dm's reliability and the bootstrap interval from
`n4_section16_followups.py` (section 5).

### 16.5 Reading the outcome

| Result | Ending (from the plan) |
|---|---|
| Test 1 positive, dm tilts the same way, shrinkage near dm's reliability | Each adaptation scales with the local strength of the demand it regulates. |
| dm has no matching tilt, or shrinkage near 0 with a reliable dm | Adaptation has spatial structure of its own. |
| dm organized (parcel or coordinate test) but Test 1 null | The demands are organized; their adaptation is shared. |

The all-lPFC outcome fits none of these rows cleanly; see §16.6.

The main-effect maps will be far more reliable than the adaptation maps. Put the
two levels' correlations next to their reliabilities, never as "the main effects
are more segregated", and never as a noise-corrected ratio.

The synthetic check is in `tests/analysis/stats/test_main_effect_anatomy.py`.
It uses `_synthetic_scores(main_effects='inherited' | 'independent')`: the
inherited world must shrink the tilt and show tracking, and the independent
world must do neither. `DATA_SOURCE=synthetic` runs the inherited world through
the job; its planted layout is anterior–posterior, so its z rows are nulls.

### 16.6 Findings from the all-lPFC run

**The run.** Segregation run
`window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects` (1,000
splits), then the anatomy job on it (`ROI_FILTER=lpfc`, `ANAT_LEVEL=destrieux`,
`N_PERM=10000`): 398 electrodes, 22 participants. The anatomy output folder is
`anatomy_a1_lpfc_window_0.0to1.5s_sig/`. The `_sig` suffix comes from the
submitter's `ELECTRODES` default (§13), but the folder holds all lPFC
electrodes. The task-significant set was rerun with main effects on
2026-10-01; its results are in §18.

**Where the numbers come from.** Each table names its source:

- the anatomy job's outputs (`summary.txt`, `dm_coordinates.csv`,
  `delta_tracking.csv`, `tilt_with_dm.csv`);
- the segregation run's `correlation_main_effects.json`;
- `dcc_scripts/stats/n4_section16_followups.py` for everything the job does not
  compute ("script §k" below). It takes about four minutes:

```bash
python dcc_scripts/stats/n4_section16_followups.py \
    --scores    <anatomy run>/continuous/scores_with_anatomy.csv \
    --seg-dir   <segregation run> \
    --tilt      <anatomy run>/continuous/tilt_with_dm.csv \
    --out-dir   <anatomy run>/continuous/section16
```

`--seg-dir` supplies `correlation_main_effects.json` (section 5) and panel a's
r and p (section 6, which redraws Figure 5). `--main-json` still overrides the
JSON. `--sections 1,2` runs a subset.

#### 16.6.1 Takeaway

Every test below is about the **balance** between two effects, within each
electrode: dm = congruency − switch, delta = LWPC − LWPS. None is about one
effect on its own, except where marked as a follow-up.

| | Base effects: congruency, switch | Adaptation: LWPC, LWPS |
|---|---|---|
| The two effects share electrodes (separate halves, within participant) | r = 0.23, p = 0.0001 | r = 0.10, p = 0.0005 |
| The balance differs across Destrieux labels | F = 1.70, p = 0.017 | F = 1.91, p = 0.010 |
| … with any one participant left out | p ≤ 0.058 | p ≤ 0.057 |
| The balance changes along MNI axes (pipeline model, y + z + x) | block p = 0.13 | block p = 0.031 (z slope p = 0.008) |
| … with distance from the midline instead of x (script §1) | \|x\| slope p = 0.16 | \|x\| slope p = 0.019 |

1. **At both levels, the two effects share electrodes.** Congruency and switch
   effects sit on largely the same electrodes, and so do LWPC and LWPS.
2. **At both levels, the balance differs across labels, and the two balances
   line up.** Labels that lean congruency in their base effects lean LWPC in
   their adaptation (label r = 0.73). The same holds electrode by electrode on
   separate trial halves (Test 1, r = 0.09), and it is process-specific.
3. **The adaptation balance also changes along a spatial gradient; the
   base-effect balance shows none that is significant.** dm's point estimate
   runs the same way at about half the slope. These are two separate
   statements; no test compares the two gradients.
4. **The adaptation gradient runs from the midline outward as much as from top
   to bottom, and it is carried by LWPC.** Height and distance from the midline
   cannot be separated in lPFC; distance from the midline fits better. LWPC is
   absent near the midline and present laterally, while LWPS and both base
   effects do not vary detectably.
5. **Whether the adaptation gradient comes from the base effects is
   unresolved.** Test 2 cannot tell, because dm is too noisy to work as a
   covariate.

#### 16.6.2 The base effects share electrodes

`correlation_main_effects.json` is the pre-specified co-localization test
(`split_resolved_corr`, §15.5) run on congruency and switch instead of LWPC
and LWPS: one effect from each disjoint half, responsiveness removed, centred
within participant, Spearman, participants with at least three electrodes.

| | congruency vs switch | LWPC vs LWPS |
|---|---|---|
| r, separate halves | 0.226 | 0.096 |
| p | 0.0001 (10,000 permutations) | 0.0001 (10,000; segregation `summary.txt`, r = 0.0971); 0.0005 with 2,000 in `min_elec_sweep.csv` |
| split-half reliabilities, within participant (pipeline's per-electrode split) | 0.345 / 0.234 | 0.072 / −0.098 |
| … with splits shared by a participant's electrodes (§19.8.4) | 0.462 / 0.351 | 0.207 / 0.035 |
| r as a share of √(rel · rel), per-electrode split | 0.79 | not estimable |
| electrodes / participants | 397 / 21 | 397 / 21 |

- Congruency and switch effects mostly land on the same electrodes. The 0.79
  leans high, because the per-electrode-split reliabilities are biased low
  (§19.8.4). The shared-split ratio needs the shared-split congruency–switch r,
  which has not been recorded.
- Do not compare the levels on raw r ("the base effects overlap more than the
  adaptation effects"). The adaptation maps' within-participant ceiling cannot
  be estimated, and pooled, they sit at their ceiling (§15.4).
- This overlap is why dm is noisy. dm's split-half reliability works out to
  (0.345 + 0.234 − 2 × 0.226) / (2 − 2 × 0.226) = 0.08 (script §5), matching
  Test 1's own 0.09. That number sets the scale for Test 2 (§16.6.6). It uses
  the biased per-electrode-split reliabilities; recomputing it from shared
  splits is an open item (§0.6).

#### 16.6.3 Both balances differ across labels, in step

From `summary.txt` (the label tests, 19 labels, 396 electrodes, 22
participants) and script §4 (the label means side by side):

| | dm (congruency − switch) | delta (LWPC − LWPS) |
|---|---|---|
| omnibus | F = 1.70, p = 0.017 | F = 1.91, p = 0.010 |
| leave one participant out | F 1.47–2.20; p 0.001–0.058 (worst: drop D0103) | F 1.60–2.39; p 0.003–0.057 (worst: drop D0144) |
| labels with q < 0.05 | none | none |
| most extreme labels | lh S_circular_insula_sup +0.53 (q = 0.07); lh G_front_sup −0.24 (q = 0.12) | lh S_front_sup −0.54, lh G_front_sup −0.30, rh G_front_middle +0.41 (all q = 0.13) |

- The adjusted label means of dm and delta correlate r = 0.73 (Spearman 0.65),
  with the same sign in 13 of 19 labels. All four superior frontal labels lean
  switch and LWPS, or sit near zero; rh MFG and IFS, rh IFG (orbital,
  triangular) and the lh superior insular sulcus lean congruency and LWPC. The
  clear exceptions are lh IFG orbital and triangular and lh MFG.
- This correspondence is descriptive. Full-data dm and delta share trials (noise
  correlation about +0.05). Test 1 (§16.6.5) is the separate-half version.
- The omnibus is the claim at both levels. Do not name a single label as the
  driver.

#### 16.6.4 The adaptation gradient: which axis, and which effect

**Height and distance from the midline are tangled.** Within participant,
MNI z and |x| correlate r = −0.58: the superior frontal gyrus and sulcus are
both dorsal and medial. The pooled pipeline model uses signed x, which cancels
between hemispheres (§5), so z absorbed the lateral part. Script §1, swap null:

| value | model | block F, p | z slope (p) | \|x\| slope (p) |
|---|---|---|---|---|
| delta | y + z + x (pipeline) | 2.85, 0.031 | −0.0077 (0.008) | |
| delta | y + z + \|x\| | 5.24, 0.001 | −0.0023 (0.53) | +0.0139 (0.019) |
| delta | z alone | 7.90, 0.003 | −0.0077 (0.007) | |
| delta | \|x\| alone | 13.9, 0.0002 | | +0.0154 (0.0007) |
| dm | y + z + x (pipeline) | 1.81, 0.13 | −0.0037 (0.15) | |
| dm | y + z + \|x\| | 1.84, 0.09 | −0.0009 (0.79) | +0.0071 (0.16) |
| dm | \|x\| alone | 4.54, 0.018 | | +0.0084 (0.033) |

- With |x| in the model, distance from the midline carries delta's gradient and
  height drops out. The pipeline's per-hemisphere fits already showed this.
  Within one hemisphere, x is distance from the midline, and it beats z in both
  (lh: x p = 0.068, z p = 0.21; rh: x p = 0.17, z p = 0.97).
- The data cannot fully separate the two axes. Describe the gradient as
  **dorsomedial versus ventrolateral**, which is true of both.
- dm leans the same way at about half the slope. Its |x| slope is significant
  only when fitted alone (p = 0.033).

**The gradient is carried by LWPC.** Script §2: single scores on y + z + |x|,
within-participant coordinate shuffle, because a single score has no partner
to swap with (§15.9):

| score | \|x\| slope (SD/mm) | p | z slope | p |
|---|---|---|---|---|
| LWPC | +0.0126 | 0.006 | +0.0011 | 0.71 |
| LWPS | −0.0013 | 0.75 | +0.0034 | 0.17 |
| congruency | +0.0032 | 0.45 | +0.0023 | 0.40 |
| switch | −0.0039 | 0.34 | +0.0032 | 0.25 |

Adjusted Cohen's d by tertile of distance from the midline (script §3;
participant and responsiveness removed; descriptive):

| band | \|x\| (mm) | n | congruency | switch | LWPC | LWPS |
|---|---|---|---|---|---|---|
| medial | 0–27 | 133 | 0.03 | 0.05 | −0.07 | 0.07 |
| middle | 28–40 | 132 | 0.12 | 0.08 | 0.06 | 0.05 |
| lateral | 40–71 | 133 | 0.06 | 0.03 | 0.04 | 0.03 |

- LWPC is absent, slightly reversed, within about 27 mm of the midline and
  present laterally. LWPS is flat.
- Neither base effect varies detectably. Congruency is lowest near the midline
  in the band means but stays positive there while LWPC turns negative.
- In the pipeline model, switch's z slope reaches p = 0.04 (script §2). It drops
  to p = 0.25 once |x| is in the model; do not report it.
- The axis was chosen after looking. Report it as a description of the
  pre-specified coordinate result (block p = 0.031), not as a separate finding.

This updates §15.1 (item 2) and §15.7, which describe the tilt as dorsoventral.

#### 16.6.5 The two balances track each other (Test 1)

From `delta_tracking.csv`: one map from each disjoint half, responsiveness
removed, within participant, Spearman, 397 electrodes, 21 participants.

| comparison | r | p |
|---|---|---|
| dm vs delta | 0.092 | 0.0003 |
| dm vs delta, + MNI covariates | 0.087 | 0.0007 |
| congruency vs LWPC (matched) | 0.217 | 0.0001 |
| switch vs LWPS (matched) | 0.169 | 0.0001 |
| congruency vs LWPS (crossed) | 0.133 | 0.0001 |
| switch vs LWPC (crossed) | 0.121 | 0.0002 |

- **The balances are linked electrode by electrode.** Electrodes that lean
  congruency lean LWPC. Partialling out coordinates barely changes it, so the
  link is local, not a gradient the two maps share.
- **It is process-specific.** The dm–delta covariance equals the two matched
  covariances minus the two crossed ones, so anything common to all four scores
  cancels. For LWPS the matched map (switch) is the less reliable one (0.35 vs
  0.46 with shared splits, §19.8.4; 0.23 vs 0.35 with the pipeline's split)
  and still wins, so reliability does not explain the gap.
- **There is also a shared component.** The crossed pairings are positive too,
  after the responsiveness covariate: electrodes with larger base effects of
  either kind adapt more on both.
- **Why a matched correlation means scaling.**
  cov(congruency, LWPC) = ½[var((i − c)₂₅) − var((i − c)₇₅)] across electrodes.
  It is positive when each electrode's congruency effect is scaled down in
  75 %-incongruent blocks. So an electrode with i < c that adapts gets a
  negative LWPC; §2.1's "negative means the effect grew" holds only where the
  base effect is positive.
- The reliability columns are within participant and biased low (§15.4); the
  negative ones cannot be real. Compare them; never divide by them.

#### 16.6.6 Is the gradient inherited? (Test 2)

From `tilt_with_dm.csv` (height) and script §5 (both axes, with a 2,000-resample
participant bootstrap of the shrinkage):

| axis | dm slope (p) | delta slope (p) | delta + dm slope (p) | shrinkage | bootstrap 95 % |
|---|---|---|---|---|---|
| height (pipeline model) | −0.0037 (0.15) | −0.0077 (0.008) | −0.0069 (0.015) | 0.10 | −0.20 to 0.40 |
| distance from the midline | +0.0071 (0.16) | +0.0139 (0.019) | +0.0125 (0.030) | 0.10 | −0.15 to 0.46 |

With dm from the opposite trial half (height only, `tilt_with_dm.csv`), the
shrinkage is 0.05.

**The scale.** A covariate measured with reliability λ removes about λ of the
slope it carries. dm's split-half reliability is 0.08, or 0.15 on full data
(§16.6.2). A gradient carried entirely by dm would therefore shrink by about
0.08 (opposite half) to 0.15 (same trials).

| shrinkage | observed | expected if fully inherited | implied share |
|---|---|---|---|
| same trials, height | 0.095 | 0.15 | 0.63 |
| same trials, midline | 0.097 | 0.15 | 0.64 |
| opposite half, height | 0.050 | 0.08 | 0.61 |

- At face value, about 60 % of the gradient could run through the base effects.
  That leans high: the reliabilities are biased low, and the same-trial rows
  also carry shared-trial noise.
- The bootstrap intervals include both 0 and the ~0.15 that full inheritance
  would give. The data allow anything from none of the gradient to all of it.
- The only other evidence is the single-score pattern (§16.6.4). The gradient
  sits in LWPC, and congruency shows no matching gradient. That argues against
  LWPC's gradient being a scaled copy of congruency's, but those slopes are
  noisy too.

**Conclusion.** Each adaptation tracks the local strength of the demand it
regulates. Whether the spatial gradient in their balance is inherited from the
base effects cannot be determined from these data.

#### 16.6.7 What not to write

| Tempting sentence | Why not | Write instead |
|---|---|---|
| "Only the adaptation balance is spatially organized" | No test compares the two gradients; dm points the same way at half the slope | Two separate statements (16.6.1, item 3) |
| "The base effects overlap more than the adaptation effects" | The adaptation ceiling cannot be estimated | "At both levels the two effects share electrodes" |
| "The tilt survives the main effects, so it is adaptation-specific" | dm is too noisy for the shrinkage to show that | "Whether the gradient is inherited could not be determined" |
| "The balance tilts dorsoventrally" | Height and distance from the midline are tangled; distance fits better | "dorsomedial versus ventrolateral" |
| "Label X drives the effect" | No label survives FDR at either level | Report the omnibus |
| "Negative LWPC means no adaptation" | Adapting a negative congruency effect gives negative LWPC | Describe the sign relative to the base effect |
| "LWPC and LWPS are carried by one intermixed population" | Local similarity cannot test "intermixed": the balance has no reliable within-participant variation to be patchy or intermixed (§19.8.6) | "One population rather than two intermixed ones" (wording in §19.8.7) |
| "LWPC and LWPS share electrodes because they are co-regulated" | Partialling each half's base effects removes most of the overlap (0.097 → 0.028, §19.8.5) | "They share electrodes because each is expressed where the signal it regulates is, and those signals share electrodes" |

### 16.7 Putting it into the paper

#### 16.7.1 Where each number comes from

All files are in the anatomy run's `continuous/` folder unless marked
"segregation".

| Claim | File | Number |
|---|---|---|
| LWPC and LWPS share electrodes | segregation `summary.txt`; `min_elec_sweep.csv` (`min_elec = 3`) | r = 0.096, p = 0.0005 |
| Congruency and switch share electrodes | segregation `correlation_main_effects.json` | r = 0.226, p = 0.0001 |
| The adaptation balance differs across labels | `summary.txt` (§5.2 primary), `delta_per_roi.csv`, `delta_roi_loso.csv` | F = 1.91, p = 0.010 |
| The base-effect balance differs across labels | `summary.txt` (MAIN EFFECTS), `dm_per_roi.csv`, `dm_roi_loso.csv` | F = 1.70, p = 0.017 |
| The two label patterns line up | script §4, `panel_b_label_means.csv` | r = 0.73, 13 of 19 same sign |
| The adaptation balance has a spatial gradient | `summary.txt` (§5.2 secondary), `score_anatomy.json` → `coordinates` | block F = 2.85, p = 0.031 |
| The base-effect balance has no significant gradient | `dm_coordinates.csv` | block F = 1.81, p = 0.13 |
| The gradient is dorsomedial vs ventrolateral | script §1 | r(z, \|x\|) = −0.58; \|x\| p = 0.019, z p = 0.53 |
| It is carried by LWPC | script §2 and §3, `panel_c_midline.csv` | LWPC \|x\| slope p = 0.006; others p ≥ 0.34 |
| The balances track each other | `delta_tracking.csv` | r = 0.092, p = 0.0003 |
| … process-specifically | `delta_tracking.csv`, rows 3–6 | 0.22 and 0.17 vs 0.13 and 0.12 |
| Inheritance is unresolved | `tilt_with_dm.csv`; script §5 | shrinkage 0.10 (0.05 opposite half); dm reliability 0.08; bootstrap −0.20 to 0.40 |

The LWPC–LWPS *p* is 0.0001 with 10,000 permutations in the segregation
`summary.txt` (§16.6.2). §16.7.2–§16.7.4 (draft paragraphs, the earlier Figure
5 plans and their open items) are in the
[Archive](#archive-superseded-results-and-drafts); the current versions are §0.5,
§0.8 and §0.11.

---

## 17. Write-up for the paper: the all-lPFC main-effect run

This section turns the run in §16.6 into manuscript text and says whether it is
ready for the paper. Every number was checked on 2026-10-01 against the run's
own outputs (`summary.txt`, `score_anatomy.json`, `scores_with_anatomy.csv`,
`dm_coordinates.csv`, `score_centers.csv`). The follow-up numbers (script §1–§4)
were recomputed from that `scores_with_anatomy.csv` with
`n4_section16_followups.py`, and they match §16.6 exactly. Numbers that come
from files outside the anatomy folder are marked with their source in §17.5.

What is current here: where the run is (§17.1) and the source of each number
(§17.5). The Methods (§17.2), the first Results draft (§17.3) and the
2026-10-01 placement table (§17.4) are superseded by §0.7, §0.8 and §0.5–§0.6
and are in the [Archive](#archive-superseded-results-and-drafts).

### 17.1 Where the run is on the DCC

The anatomy job writes every output into one `continuous/` folder
(`run_score_anatomy`, `stability_flexibility_anatomy_dcc.py`). The runner names
the folder `results/<EPOCHS_ROOT_FILE>/anatomy_<LABEL_SOURCE>_<ROI_FILTER>_window_<tmin>to<tmax>s_<ELECTRODES>/`
(`run_stability_flexibility_anatomy_dcc.py`). `score_anatomy.json` → `maps`
records the absolute path of the brain maps, which confirms it:

| What | Path on the DCC |
|---|---|
| Epochs root (`RUN`) | `/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/` |
| **Anatomy outputs** (the five files above and everything else in §16.3) | `$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` |
| Upstream scores (`electrodes.csv`, `per_split.csv`, `continuous.csv`, `summary.txt`, `correlation_main_effects.json`) | `$RUN/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects/` |
| Follow-up tables (`panel_b_label_means.csv`, `panel_c_*.csv`), if the §16.6 command was run as written | `$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/section16/` |

The folder ends in `_sig`, but it holds **all 398 lPFC electrodes**. The suffix
comes from the submitter's `ELECTRODES=sig` default, which does not select
electrodes (§13).

```bash
RUN=/hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20
ls -lt "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/"
head -12 "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/summary.txt"   # F = 1.907, p = 0.0102, 398 electrodes

# if the folder has moved, find every continuous-arm run by its JSON
find /hpc/home/jz421/coganlab/jz421/GlobalLocal/dcc_scripts/stats/results \
     -path '*continuous*' -name score_anatomy.json
```

⚠️ **Copy this folder before the task-significant rerun.** The folder name does
not depend on `SCORES_CSV`. Pointing `SEG_RUN` at the task-significant
`_main_effects` run and submitting with the same defaults writes to the same
`anatomy_a1_lpfc_window_0.0to1.5s_sig/` path and overwrites these results:

```bash
cp -rp "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig" \
       "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_ALL_lpfc_main_effects"
```

*Update 2026-10-01:* the task-significant rerun did write to this path (§18.1).
Check whether a copy was made before it; if not, §18.1 says how to get the
all-lPFC files back.

### 17.5 Where each number in 17.3 comes from

`continuous/` is the anatomy folder in §17.1. "By hand" means computed on
2026-10-01 from `scores_with_anatomy.csv` and not printed by any script. §17.3
itself is archived, but its numbers are the ones
[`paper_draft.md`](paper_draft.md) §3.4 quotes, so this table is still their
source.

| Number | Source |
|---|---|
| 398 electrodes (254 lh / 144 rh), 22 participants, 1–55 per participant | `continuous/scores_with_anatomy.csv` (by hand) |
| Mean *d* LWPC 0.01 / LWPS 0.05; 49 % / 56 % positive | `continuous/scores_with_anatomy.csv`, `lwpc_score`, `lwps_score` (by hand) |
| Full-data reliability 0.27–0.30 | Spearman–Brown of the Pearson half-data reliabilities in `continuous/score_anatomy.json` → `ceiling.electrode` (0.179, 0.159) |
| LWPC–LWPS *r* = 0.10, *p* < 0.001; 397 / 21 | segregation `summary.txt` (0.0971, *p* = 0.0001, 10,000 permutations); also `continuous/summary.txt`, `min_elec` sweep (`min_elec = 3`: 0.096, *p* = 0.0005, 2,000 permutations) |
| Centroid distance 1.4 mm, *p* = 0.95 | §15.6 (earlier 200-split scores; `n4_section15_followups.py` section 10) |
| Parcel *F* = 1.91, *p* = 0.010; 19 parcels; leave-one-out *p* 0.003–0.057 | `continuous/summary.txt` (§5.2 primary, §9.2), `score_anatomy.json` |
| All *q* ≥ 0.13 | `continuous/summary.txt` per-label rows (lowest *q* = 0.126) |
| Block *F* = 2.85, *p* = 0.031; *z* −0.0077, *p* = 0.008; *y* *p* = 0.58 | `continuous/score_anatomy.json` → `coordinates.all` |
| ~2 % of variance; replication across halves *p* = 0.005 | §15.7 (`n4_section15_followups.py` sections 5 and 12) |
| *r*(*z*, \|*x*\|) = −0.58; distance *p* = 0.019, height *p* = 0.53 | `n4_section16_followups.py` §1 |
| LWPC −0.07 medial, 0.04–0.06 farther out; slope *p* = 0.006; LWPS *p* = 0.75 | `n4_section16_followups.py` §2 and §3 |
| Congruency–switch *r* = 0.23, *p* < 0.001 | segregation `correlation_main_effects.json` (§16.6.2) |
| Base-effect parcel *F* = 1.70, *p* = 0.017; leave-one-out *p* ≤ 0.058 | `continuous/summary.txt` (MAIN EFFECTS), `dm_roi_loso.csv` (§16.6.3) |
| Label *r* = 0.73, 13 of 19 | `n4_section16_followups.py` §4 |
| Test 1 *r* values | `continuous/score_anatomy.json` → `main_effects.tracking` (`delta_tracking.csv`) |
| Base-effect gradient *F* = 1.81, *p* = 0.13; \|*x*\| *p* ≥ 0.34 | `continuous/dm_coordinates.csv`; `n4_section16_followups.py` §2 |
| 10 % reduction, *p* = 0.015 | `continuous/score_anatomy.json` → `main_effects.tilt` (`tilt_with_dm.csv`) |
| Split-half *r* = 0.08; bootstrap −20 % to 40 % | `n4_section16_followups.py` §5 with the segregation JSON (§16.6.6) |
| Task-responsive subset | §18 (171 electrodes, 21 participants; with main effects) |

---

## 18. The task-significant main-effect run

The rerun that §16.7.4 and §17.4 asked for: the task-significant set
(`ELECTRODES=sig`) scored with main effects, then the anatomy job. Its outputs
arrived on 2026-10-01: `summary.txt`, `score_anatomy.json`,
`scores_with_anatomy.csv`, `per_split.csv` and `dm_coordinates.csv`. Every
number below comes from one of three places:

- those files;
- `n4_section16_followups.py` sections 1–5, run on this
  `scores_with_anatomy.csv` ("script §k");
- `split_resolved_corr` run on this `per_split.csv` ("recomputed"), for the
  numbers that would be in the segregation run's `correlation_main_effects.json`,
  which was not among the files. The recomputation reproduces the anatomy job's
  own congruency–LWPC row exactly (r = 0.131925), so it uses the same inputs.

### 18.1 The run

| | |
|---|---|
| Electrodes | 171 (120 left / 51 right), 21 participants, 1–19 per participant |
| Splits | **200** (`per_split.csv` has splits 0–199), not the 1,000 of the all-lPFC run |
| Permutations | 10,000 for the label, coordinate and Test 1 nulls; 1,000 per leave-one-out fold |
| Co-localization and Test 1 | 167 electrodes, 18 participants (D0065, D0110 and D0145 have fewer than three electrodes) |
| Label tests | 15 Destrieux labels sampled in at least three participants; 167 electrodes, 21 participants |
| Anatomy folder | `$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` (`score_anatomy.json` → `maps`) |
| Upstream scores | Presumably `$RUN/segregation_results/window_0.0to1.5s_sig_lpfc_proportion_cohens_d_fdr_bh_main_effects/`. **[Confirm.]** |

**The LWPC and LWPS scores are the ones in §15.** The half-data reliabilities
(Spearman 0.145 / 0.200, Pearson 0.148 / 0.241) and the label test
(*F* = 0.907) reproduce §15.4 and §15.7 exactly. Adding the main effects did
not change the split draws (§16.1), so every task-significant LWPC/LWPS number
in §15 stands. Swap-null p-values differ from §15 in the second decimal because
the permutations were drawn again (coordinate block p 0.28 vs 0.29, z p 0.23 vs
0.24). Quote this run's values from now on.

⚠️ **This run wrote into the all-lPFC folder.** `score_anatomy.json` → `maps`
points at `anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/`, the folder that
§17.1 says holds the all-lPFC run. Unless that folder was copied first
(§17.1's `cp -rp`), the all-lPFC outputs have been overwritten. Check:

```bash
head -4 "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/summary.txt"   # "electrodes: 171" = overwritten
ls -d "$RUN"/anatomy_a1_lpfc_window_0.0to1.5s_*                              # is there a copy?
```

The all-lPFC numbers are recorded in §16–§17, and no text needs the files
again. F5b's `delta_tracking.csv` and the S-N4 tables do. If they are gone,
rerun the anatomy job on the all-lPFC `_main_effects` segregation run (§16.2),
which should reproduce §16.6 because the job's seeds are fixed. Copy each
folder to a name that says which set it holds before the next submission:

```bash
cp -rp "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_sig" \
       "$RUN/anatomy_a1_lpfc_window_0.0to1.5s_SIG_lpfc_main_effects"
```

*2026-10-05:* the first §19 run read this folder and found 398 electrodes from
22 participants (its `section19/summary_section19.txt` is committed in the
repo), so on that date the folder held the all-lPFC scores again. Confirm that
its `per_split.csv` is the all-lPFC one as well (1,000 splits), and record where
the subset's outputs now live.

### 18.2 Takeaway

| | All lPFC (398 electrodes, 22 participants; §16.6) | Task-significant (171, 21) |
|---|---|---|
| Congruency and switch share electrodes (separate halves) | r = 0.23, p = 0.0001 | r = 0.17, p = 0.001 |
| LWPC and LWPS share electrodes (separate halves) | r = 0.10, p = 0.0005 | r = 0.08, p = 0.055 |
| Test 1: dm vs delta | r = 0.092, p = 0.0003 | r = 0.080, p = 0.049 |
| … with MNI coordinates partialled out | r = 0.087, p = 0.0007 | r = 0.078, p = 0.052 |
| Matched − crossed gap, LWPC / LWPS | 0.10 / 0.04 | 0.09 / 0.03 |
| Adaptation balance across labels | F = 1.91, p = 0.010 | F = 0.91, p = 0.29 |
| Base-effect balance across labels | F = 1.70, p = 0.017 | F = 1.15, p = 0.12 |
| Label means of the two balances | r = 0.73, same sign in 13 of 19 | r = 0.42, same sign in 12 of 15 |
| Adaptation balance on MNI coordinates | block F = 2.85, p = 0.031 | block F = 1.44, p = 0.28 |
| Base-effect balance on MNI coordinates | block F = 1.81, p = 0.13 | block F = 0.88, p = 0.48 |
| z slope of delta / of dm (SD/mm) | −0.0077 (p = 0.008) / −0.0037 (p = 0.15) | −0.0075 (p = 0.23) / −0.0035 (p = 0.53) |
| LWPC slope with distance from the midline (script §2) | +0.0126, p = 0.006 | +0.0022, p = 0.82 |

1. **At both levels the two effects share electrodes, as in all lPFC.** The
   base effects clearly (r = 0.17, p = 0.001), the adaptation effects at the
   same size as before and just short of significance (r = 0.08, p = 0.055).
2. **Each adaptation tracks its own base effect by the same margins as in all
   lPFC.** LWPC correlates with congruency 0.09 more than with switch (all
   lPFC 0.10), and LWPS with switch 0.03 more than with congruency (0.04). The
   test of the difference sits at the threshold: p = 0.049, and 0.052 with
   coordinates partialled out. With 10,000 permutations each p-value carries
   a Monte Carlo error of about ±0.004 (95 %), so read them as one result at
   p ≈ 0.05, not as one pass and one fail.
3. **Reliability does not explain the LWPC gap here.** Within participant the
   congruency and switch maps are equally reliable in this set (0.39 and 0.39
   with shared splits, §19.8.4; 0.250 and 0.257 with the pipeline's split),
   and the LWPC gap is the same size as in all lPFC, where congruency is the
   more reliable map (0.46 vs 0.35 shared; 0.345 vs 0.234 per electrode). This
   answers the limit stated
   for LWPC in [`paper_draft.md`](paper_draft.md) §4, descriptively: no test
   compares the two pairs' gaps on their own.
4. **Neither balance is spatially organized in this set.** No label test and no
   coordinate fit is significant (§18.5). The z slopes of both balances are the
   same size as in all lPFC. For delta this is the power shortfall that §15.7
   predicted (a planted slope reaches p < 0.05 26 % of the time here, 66 % in
   all lPFC), not a different pattern; dm's power was not simulated.
5. **The all-lPFC description of the gradient does not repeat.** Here distance
   from the midline does not fit better than height, and LWPC shows no slope
   with it (§18.6). Those two descriptions were exploratory in all lPFC
   (§16.6.4), and the subset has too little power to confirm or refute them.
6. **Inheritance (Test 2) cannot be asked here.** delta's own slope is not
   significant, so its shrinkage has nothing to measure (§16.4).

### 18.3 The four scores and their reliabilities

From `scores_with_anatomy.csv` (raw Cohen's *d*), `score_anatomy.json` →
`ceiling` and `main_effects.reliability` (pooled, half data, Spearman), and the
Test 1 rows (within participant):

| | congruency | switch | LWPC | LWPS |
|---|---|---|---|---|
| mean Cohen's *d* | 0.17 | 0.12 | 0.14 | 0.18 |
| electrodes positive | 76 % | 74 % | 67 % | 75 % |
| split-half reliability, pooled | 0.53 | 0.46 | 0.15 | 0.20 |
| split-half reliability, within participant (per-electrode split) | 0.250 | 0.257 | 0.014 | −0.000 |
| … with splits shared by a participant's electrodes (§19.8.4) | 0.392 | 0.388 | 0.174 | 0.141 |

- Pooled, the base-effect maps are two to four times as reliable as the
  adaptation maps. Do not compare the two levels on raw r (§16.5).
- The per-electrode-split row is biased low (§19.3, confirmed in §19.8.4).
  Quote the shared-split row.

**The base effects share electrodes** (recomputed; `split_resolved_corr` on the
main-effect halves, responsiveness removed, within participant, Spearman,
10,000 permutations, seed 1):

| | congruency vs switch | LWPC vs LWPS |
|---|---|---|
| r, separate halves | 0.167 | 0.077 |
| p (10,000 permutations) | 0.001 | 0.055 |
| r as a share of √(rel · rel), within participant | 0.66 | not estimable |
| electrodes / participants | 167 / 18 | 167 / 18 |

- The LWPC–LWPS p from 10,000 permutations is 0.055, against 0.057 from 1,000
  in the segregation `summary.txt` (§15.5) and 0.051 from 2,000 in this run's
  `min_elec` sweep. If the segregation run was submitted with
  `N_PERM_CORR=10000` (§16.2), its `summary.txt` shows the same 0.055, because
  the seed is the same.
- dm's split-half reliability works out to (0.250 + 0.257 − 2 × 0.167) /
  (2 − 2 × 0.167) = 0.10 (0.19 at full length), against 0.08 in all lPFC. Test
  1's own dm reliability is 0.097. Both use the per-electrode-split
  reliabilities (see §0.6).

### 18.4 Test 1

From `score_anatomy.json` → `main_effects.tracking` (`delta_tracking.csv`):
separate halves, responsiveness removed, within participant, Spearman, 167
electrodes, 18 participants, 10,000 permutations.

| comparison | r | p | all lPFC r |
|---|---|---|---|
| dm vs delta | 0.080 | 0.049 | 0.092 |
| dm vs delta, + MNI covariates | 0.078 | 0.052 | 0.087 |
| congruency vs LWPC (matched) | 0.132 | 0.003 | 0.217 |
| switch vs LWPS (matched) | 0.147 | 0.0009 | 0.169 |
| congruency vs LWPS (crossed) | 0.121 | 0.005 | 0.133 |
| switch vs LWPC (crossed) | 0.041 | 0.33 | 0.121 |

- Same sign and similar size as in all lPFC on every row. The LWPC pairs are
  smaller here, the LWPS pairs about the same, and the two gaps (0.09 and
  0.03) match all lPFC's (0.10 and 0.04).
- The one crossed pairing that is not significant is switch–LWPC. So in this
  set LWPC is associated with congruency and hardly with switch, while LWPS is
  associated with both.
- Three of four pairings are positive and significant, so the shared component
  of §16.6.5 is here too: electrodes with larger base effects adapt more.

### 18.5 The two balances by label and position

From `summary.txt`, `score_anatomy.json` and `dm_coordinates.csv`:

| | delta (LWPC − LWPS) | dm (congruency − switch) |
|---|---|---|
| label omnibus (15 labels) | F = 0.91, p = 0.29 | F = 1.15, p = 0.12 |
| leave one participant out | F 0.78–1.23; p 0.099–0.45 | F 0.79–1.58; p 0.029–0.30 |
| … largest shift | drop D0133: p = 0.099 | drop D0133: p = 0.029 |
| labels with q < 0.05 | `rh_S_front_sup` (7 electrodes, 4 participants), q = 0.012 | none (all q ≥ 0.74) |
| coordinates, all (block F, p) | 1.44, 0.28 | 0.88, 0.48 |
| … left hemisphere (120) | 1.27, 0.26 | 0.47, 0.69 |
| … right hemisphere (51) | 1.25, 0.48 | 0.63, 0.46 |
| z slope (SD/mm) | −0.0075, p = 0.23 | −0.0035, p = 0.53 |
| y slope | +0.0084, p = 0.30 | −0.0001, p = 0.99 |
| x slope | −0.0002, p = 0.97 | +0.0048, p = 0.40 |

- Both omnibus tests are null. A label row means something only after a
  significant omnibus (§15.7), so do not report `rh_S_front_sup`.
- dm's leave-one-out range reaches p = 0.029 when D0133 is dropped: one
  participant can move this test across 0.05 in either direction. Report the
  range, not the fold.
- The predicted anterior–posterior axis is null for both balances.

### 18.6 Follow-ups: axis, single scores, bands and labels (script §1–§4)

**Height against distance from the midline** (script §1, swap null). Within
participant, z and |x| correlate r = −0.50 (all lPFC −0.58).

| value | model | block F, p | z slope (p) | \|x\| slope (p) |
|---|---|---|---|---|
| delta | y + z + x (pipeline) | 1.44, 0.28 | −0.0075 (0.23) | |
| delta | y + z + \|x\| | 1.54, 0.25 | −0.0059 (0.44) | +0.0057 (0.68) |
| delta | z alone | 2.91, 0.11 | −0.0085 (0.17) | |
| delta | \|x\| alone | 1.87, 0.21 | | +0.0124 (0.27) |
| dm | y + z + x (pipeline) | 0.88, 0.48 | −0.0035 (0.53) | |
| dm | y + z + \|x\| | 0.53, 0.64 | −0.0017 (0.79) | +0.0079 (0.52) |
| dm | \|x\| alone | 1.47, 0.22 | | +0.0093 (0.36) |

With both in the model, height and distance from the midline split delta's
slope evenly here. In all lPFC, distance took it (§16.6.4).

**Single scores** (script §2, within-participant coordinate shuffle):

| score | z slope, pipeline model (p) | \|x\| slope, y + z + \|x\| (p) |
|---|---|---|
| LWPC | −0.0046 (0.29) | +0.0022 (0.82) |
| LWPS | +0.0029 (0.41) | −0.0035 (0.66) |
| congruency | +0.0002 (0.94) | −0.0040 (0.57) |
| switch | +0.0037 (0.33) | −0.0119 (0.12) |

No single score varies detectably with position. The pipeline-model z slopes
of LWPC and LWPS are §15.7's.

**Adjusted Cohen's *d* by tertile** (script §3; participant and responsiveness
removed; electrode means, as in §16.6.4; descriptive):

| band | mm | n | congruency | switch | LWPC | LWPS |
|---|---|---|---|---|---|---|
| medial (\|x\|) | 1–30 | 57 | 0.16 | 0.15 | 0.10 | 0.25 |
| middle (\|x\|) | 30–39 | 57 | 0.19 | 0.12 | 0.16 | 0.10 |
| lateral (\|x\|) | 39–60 | 57 | 0.16 | 0.11 | 0.15 | 0.18 |
| ventral (z) | −6 to 18 | 57 | 0.17 | 0.12 | 0.17 | 0.19 |
| middle (z) | 18–33 | 57 | 0.16 | 0.11 | 0.13 | 0.15 |
| dorsal (z) | 34–76 | 57 | 0.17 | 0.14 | 0.11 | 0.20 |

- Every score is positive in every band. LWPC falls from ventral to dorsal
  while LWPS does not, as in §15.7.
- Medially, LWPC is lowest and LWPS highest. Averaged per participant
  (`panel_c_midline.csv`), LWPC is flat (0.13–0.14) and only LWPS is higher
  medially (0.26 vs 0.12–0.18). So in this set the medial lean toward LWPS
  does not come from LWPC, unlike the all-lPFC breakdown (§16.6.4). The
  tertile edges are this set's own, not all lPFC's.

**Label means** (script §4): the adjusted label means of dm and delta
correlate r = 0.42 (Spearman 0.43), with the same sign in 12 of 15 labels (all
lPFC 0.73, 13 of 19). As in all lPFC, all four superior frontal labels lean
toward switch and LWPS (adjusted delta −0.12 to −0.70, dm −0.03 to −0.35).
Descriptive only: both omnibus tests are null, and the label means share
trials.

### 18.7 Test 2

From `score_anatomy.json` → `main_effects.tilt` and script §5:

| axis | dm slope (p) | delta slope (p) | delta + dm slope (p) | shrinkage | bootstrap 95 % |
|---|---|---|---|---|---|
| height (pipeline model) | −0.0035 (0.53) | −0.0075 (0.23) | −0.0068 (0.28) | 0.09 | −0.93 to 1.01 |
| distance from the midline | +0.0079 (0.52) | +0.0057 (0.68) | +0.0042 (0.76) | 0.26 | −2.6 to 3.0 |

With dm from the opposite trial half (height), the shrinkage is 0.04. Full
inheritance would give about 0.19 on the same trials and 0.10 on opposite
halves (dm's reliability, §18.3).

Not interpretable: delta's own slope is not significant (§16.4), and the
bootstrap intervals show it. Report that Test 2 was not run to a conclusion in
this set.

### 18.8 Results text (Supplement S-N4)

For the "Task-responsive subset" paragraph of
[`paper_draft.md`](paper_draft.md) §3.5. It replaces the §15.12 draft and the
earlier LWPC/LWPS-only paragraph.

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
> (within-participant split-half reliability 0.39 and 0.39, shared trial
> splits), so the larger
> difference for LWPC does not reflect a more reliable congruency map. Neither
> balance differed across Destrieux parcels (15 parcels; LWPC − LWPS:
> *F* = 0.91, *p* = 0.29; congruency − switch: *F* = 1.15, *p* = 0.12; with
> each participant left out, *p* = 0.10–0.45 and 0.03–0.30) or varied with position
> (MNI coordinates: *F* = 1.44, *p* = 0.28, and *F* = 0.88, *p* = 0.48). Their
> height slopes matched those in all lPFC (LWPC − LWPS: −0.0075 SD/mm,
> *p* = 0.23; congruency − switch: −0.0035 SD/mm, *p* = 0.53; all lPFC −0.0077
> and −0.0037). The adaptation slope did not differ from that of the remaining
> electrodes (*p* = 0.67), and with the all-lPFC slope planted, this electrode
> layout reaches *p* < 0.05 in 26 % of simulations, against 66 % for all lPFC
> electrodes. In this subset, distance from the midline did not describe the
> gradient better than height (both *p* ≥ 0.44), and LWPC did not vary with
> distance from the midline (*p* = 0.82). Because the adaptation balance had no
> detectable gradient here, we did not test whether it is inherited from the
> base effects.

Two numbers in it come from §15, not from this run: the full-data
reliabilities (0.26–0.39, §15.4) and the centroid test (§15.6). The
LWPC–LWPS scores are the same, so both still apply.

Supplementary Methods need one sentence: "In the task-responsive subset,
trials were split 200 times rather than 1,000."

### 18.9 Where each number comes from

`continuous/` is this run's anatomy folder (§18.1). "Recomputed" means
`split_resolved_corr` on this run's `per_split.csv` with `scores_with_anatomy.csv`
→ `resp`, `min_elec = 3`, 10,000 permutations, seed 1.

| Number | Source |
|---|---|
| 171 electrodes (120 / 51), 21 participants, 1–19 each; 200 splits | `scores_with_anatomy.csv`, `per_split.csv` (by hand) |
| Mean *d*, % positive | `scores_with_anatomy.csv`, `*_score` columns (by hand) |
| Pooled reliabilities 0.53 / 0.46 / 0.15 / 0.20 | `summary.txt` (MAIN EFFECTS; §5.4 ceiling) |
| Within-participant reliabilities 0.250 / 0.257 / 0.014 / −0.000 (per-electrode split) | `score_anatomy.json` → `main_effects.tracking` |
| Within-participant reliabilities 0.392 / 0.388 / 0.174 / 0.141 (shared split; the 0.39 / 0.39 in §18.8) | `reliability_by_split_scheme.csv`, §19 second run (§19.8.4) |
| Congruency–switch *r* = 0.167, *p* = 0.001 | recomputed |
| LWPC–LWPS *r* = 0.077, *p* = 0.055 | recomputed (0.057 with 1,000 permutations in the segregation `summary.txt`) |
| dm reliability 0.10 / 0.19 | from the recomputed congruency–switch row, as in §16.6.2 |
| Test 1 rows | `summary.txt` (TEST 1), `score_anatomy.json` → `main_effects.tracking` |
| Label tests, leave-one-out, per-label q | `summary.txt` (§5.2 primary, §9.2, both blocks), `score_anatomy.json` |
| Coordinate fits, hemispheres | `score_anatomy.json` → `coordinates`; `dm_coordinates.csv` |
| Axis models, single scores, bands, label means | script §1–§4 on `scores_with_anatomy.csv` |
| Test 2, bootstrap | `score_anatomy.json` → `main_effects.tilt`; script §5 |
| Centroid test 3.3 mm, *p* = 0.44; reliabilities 0.26–0.39; planted power 26 % vs 66 %; subset vs rest *p* = 0.67 | §15.4, §15.6, §15.7 (same LWPC/LWPS scores) |

Left out, as in all lPFC: the medoids (`summary.txt` §7: 20 groups, *p* ≥ 0.43
on every axis), the noise-corrected ratio (0.684 electrode level, undefined at
parcel level; §15.4), and the full-data `joint_scatter.png` correlations
(pooled 0.20, within participant 0.18).

### 18.10 RT-adjusted F5 sensitivity run

The segregation job can remove the RT-linked component of high gamma **before**
`compute_sensitivities_per_split` makes any maps. Run the score-producing job
with scalar window-mean HG and the adjustment enabled:

```bash
cd dcc_scripts/stats
RT_ADJUST_HG=1 EFFECT_MEASURE=cohens_d MAIN_EFFECTS=1 \
  bash submit_stability_flexibility_segregation_dcc.sh
```

Keep `CONTRAST_MODE=proportion` (the submitter default) for LWPC/LWPS. The run is
written to a distinct `_rt_adjusted` directory and includes
`rt_adjustment_slopes.csv` (one within-cell HG-on-RT slope and correlation per
electrode). `long_df.csv`, `per_split.csv`, and the other segregation outputs in
that directory are RT-adjusted. The job stops with an explicit error if the
epochs metadata have no finite reaction times or if a time-resolved effect
measure (`cluster` or `peak_t`) is requested.

Then point the continuous anatomy job at that run's `per_split.csv` exactly as
for the unadjusted analysis (set `SEG_RUN` in
`submit_stability_flexibility_anatomy_dcc.sh`) and submit it:

```bash
cd dcc_scripts/stats
ARM=continuous bash submit_stability_flexibility_anatomy_dcc.sh
```

Compare the resulting F5/Test 1 outputs against the raw run; do not overwrite or
silently replace the raw estimate. The adjustment is conservative: it removes
neural adaptation carried through the same trial-level HG–RT relationship as
well as nuisance RT-linked HG.

**Status (2026-10-05).** The all-lPFC segregation step has run
(`window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects_rt_adjusted`,
200 splits, per-electrode split; its `summary.txt` is committed):

| | raw (`…_main_effects`) | RT-adjusted |
|---|---|---|
| LWPC–LWPS separate-half r | +0.097, p = 0.0001 | +0.090, p = 0.0001 |
| congruency–switch separate-half r | +0.224, p = 0.0001 | +0.127, p = 0.0001 |

397 electrodes, 21 participants, 10,000 permutations in both. The adaptation
overlap survives the RT adjustment, as it does on shared splits (§19.8.5); the
base effects' overlap roughly halves, consistent with RT coupling contributing to
how strongly electrodes carry the base effects (§19.8.4). The anatomy job on this
run (the gradient and Test 1 on RT-adjusted scores) has not been recorded.

## 19. Participants as the unit, local similarity, and the combined Figure 5

*Added 2026-10-02 after the advisor meeting; run on the real data on 2026-10-05
(results in §19.8). A line-by-line guide to this code is
[`n4_code_walkthrough.md`](n4_code_walkthrough.md), and
`dcc_scripts/stats/n4_code_walkthrough.ipynb` steps through it on the real
all-lPFC outputs.*

The advisors asked three things of the anatomy:

1. **Is participant a random effect?** Not in the pre-specified tests. Both
   treat participant as a **fixed** effect: `split_resolved_corr` centres within
   participant and shuffles electrodes within participant; the coordinate test
   (`relative_score_coordinate_test`) uses participant dummy variables
   (`_nuisance_design`; its docstring's `(1 | subject)` is notation, chosen
   deliberately because a random intercept can fail on participants with two
   or three electrodes) and flips each electrode's sign on its own. Fixed and
   random intercepts give the same within-participant slope here. What neither
   test does is treat **participants as the units of inference**: a few
   participants with many electrodes, or neighbouring contacts that share noise
   and nearly share coordinates, can carry the p-value. §19.1 and §19.2 add the
   participant-level versions.
2. **Combine the two anatomy results** (the overlap scatter and the height
   gradient) in one figure, without the matched-vs-crossed bars. §19.4.
3. **A stronger test of "intermixed".** §19.3.

### 19.1 The height slope with participants as the unit

`sfa.coordinate_slope_by_participant(scores, axis='mni_z')`.

**Two-stage.** Residualise the balance (`delta`) and height on the coordinate
test's nuisance terms (participant dummies, responsiveness) and on the other two
coordinates. By Frisch–Waugh, the pooled slope of one on the other is the
coordinate test's z slope. Within participant *s* the same residuals give a
slope b_s with weight w_s = Σ_s (residual height)², the participant's spread
along z. The pooled slope is **exactly** Σ w_s b_s / Σ w_s: a weighted average of
the participants' own slopes. The test checks this identity to machine
precision. Two tests across participants:

- **weighted** (the pooled slope): p from flipping the sign of whole
  participants, 95 % interval from a participant bootstrap;
- **unweighted**: the mean b_s over participants with ≥ 3 electrodes and
  ≥ 5 mm of spread, one-sample t-test, and how many slopes are negative (sign
  test).

`top3_weight_share` says how much of the weighted average three participants
carry.

**Mixed models.** `delta ~ y + z + x + resp`, every predictor centred within
participant, with (a) a participant random intercept and (b) a random intercept
and a random slope on z (statsmodels `mixedlm`, REML). With within-centred
predictors the random-intercept model's slope equals the pooled one; the
random-slope model is the advisors' "subject as a random effect" in the strong
sense. Its Wald p runs a little liberal with ~20 participants; read it next to
the sign-flip p. Convergence warnings are kept in the output.

**Leave one participant out.** `sfa.coordinate_slope_loso`: the z slope and its
swap-null p with each participant dropped (fewer permutations; a leverage check).

### 19.2 The overlap correlation with participants as the unit

`sfs.participant_split_corr(per_split, resp)`. Each participant's own
separate-half LWPC–LWPS correlation over its electrodes, on exactly the values
the pooled test correlates (both now go through
`_residualised_split_matrices`, so they cannot drift apart; the refactor leaves
`split_resolved_corr`'s output bit-identical). The per-participant r are
averaged in Fisher z:

- weighted by n − 3 (the usual inverse-variance weight), sign-flip p over
  participants and a participant-bootstrap interval;
- unweighted, one-sample t-test, and how many participants are positive.

Participants need ≥ 4 electrodes (the pooled test uses 3); with one participant
the two tests agree exactly (tested).

### 19.3 Local similarity: is the balance intermixed at the recorded scale?

`sfa.local_similarity(per_split_shared, scores)`, figure `plot_local_similarity`.
(The first version, of 2026-10-02, lacked the first two requirements below; its
numbers from the first real run are not usable and are in the Archive.)

"Intermixed" is a claim about arrangement: no patches of LWPC-leaning electrodes
next to patches of LWPS-leaning ones. The overlap r does not test that, and the
height gradient is structure at the scale of the whole region. This asks the
local question.

1. Per split and half, each score is residualised on responsiveness and on the
   coordinates (the linear gradient, `remove_gradient=True`), centred within
   participant, ranked and re-centred within participant (Spearman), and scaled
   to unit mean square.
2. For electrodes i and j of one participant, C[i, j] = mean over splits of
   ½ (A_i B_j + B_i A_j): one electrode's half A against the other's half B and
   the reverse. C[i, i] is the electrode's split-half reliability.
3. Pairs are binned by distance (< 10, 10–20, 20–40, > 40 mm) and summed per
   participant. Each participant's baseline is the mean of each bin over
   shuffles of its electrode positions; its **excess** is observed minus
   baseline. Within-participant centring makes pairs slightly anti-correlated
   by construction, and the baseline carries the same bias.
4. **Inference takes participants as the units:** a sign flip of each
   participant's excess (one-sided), and participant-bootstrap intervals.

**Three requirements, and why.**

- **The halves must be shared by all of a participant's electrodes.** The
  segregation job splits each electrode's trials on its own
  (`compute_sensitivities_per_split`), so electrode i's half A shares about half
  its trials with electrode j's half B. Neighbouring contacts share trial noise,
  and that noise then reads as near-range "similarity" for every score. The
  first real run showed the tell-tale pattern: every score, the balance
  included, had a near-range excess larger than its own reliability, and some
  reliabilities came out negative. A neighbour cannot share more of an
  electrode's signal than the electrode shares with itself, but shared noise
  does exactly this. On simulated trial-level data with no local structure,
  per-electrode splits gave near-range p < 0.02 for all five scores; shared
  splits removed it. **So:** `compute_sensitivities_per_split(shared_split=True)`
  draws one stratified split per participant and repetition and gives it to
  every electrode (the table gets `split_scheme = 'participant'`), and
  `local_similarity` refuses any other table.
- **Participants, not pairs, are the units of inference.** Even with shared
  splits, one dataset's estimation noise is spatially smooth, so a
  position-shuffle p is too liberal: 8 of 60 null tests (13 %) at α = 0.05 on
  simulated data, with too many p near 1 as well. The participant sign-flip
  gave 3 of 120 (2.5 %) with six simulated participants. The shuffle survives
  only as each participant's baseline.
- **No ratios to an unreliable map.** The "share of reliability" is reported
  only when the reliability is positive in ≥ 90 % of bootstrap draws.

**The same bias touches the within-participant reliabilities.** With
per-electrode splits, the within-participant centring mixes in other
electrodes' halves, which share trials with this one's other half. That biases
the within-participant split-half reliabilities downward, which the real data
confirm (§19.8.4: all-lPFC LWPC 0.07 → 0.21, LWPS −0.09 → 0.04, congruency
0.34 → 0.46, switch 0.24 → 0.35). These are the reliabilities
`split_resolved_corr` reports. The overlap r itself is not affected, as §19.7
explains and §19.8.4 confirms. The §19 script prints the reliabilities from
both split schemes side by side (`reliability_by_split_scheme.csv`); quote the
shared-split ones.

Scores: the balance (LWPC − LWPS, built as `delta` is), LWPC, LWPS, and from a
`MAIN_EFFECTS=1` run congruency and switch. **The single scores are the positive
control.**

| Pattern | Reading |
|---|---|
| Single scores show a near-range excess, the balance does not | Intermixed at the recorded scale. The claim. |
| The balance shows one too | Patches: nearby electrodes lean the same way |
| No score shows one | No power: the analysis cannot tell intermixed from patchy at these reliabilities. Say so. |

On planted data: an intermixed world keeps ~10 % of LWPC's near-range excess in
the balance (the two pooled scale factors differ by sampling); a patchy world
keeps ~90 %.

Bipolar channel pairs that share a contact are dropped
(`exclude_shared_contacts`). The high-gamma electrodes are named as single
contacts (`D0057-LTP1`), so nothing should be dropped there; `n_pairs_excluded`
confirms it.

### 19.4 The combined Figure 5

`sfa.figure5_height(scores, out_dir, ...)` writes `fig5_height.png`/`.pdf`.

On the LWPC (x) against LWPS (y) scatter the two results lie along perpendicular
directions: the overlap is spread **along** the identity line, the gradient is a
shift **across** it (LWPC − LWPS is each point's signed distance from the line).
Points **above** the line lean LWPS, **below** it LWPC.

| Panel | Content |
|---|---|
| a | Height tertiles on a sagittal projection of the electrodes (MNI y against z), with the two cuts drawn. Pass `brain_png` for a rendered brain in the same colours instead. |
| b | LWPC against LWPS, coloured by tertile, identity line, the pre-specified r. A box marks panel c's region. |
| c | The three tertile centroids enlarged, each with its 95 % participant-bootstrap ellipse. With a dorsal LWPS lean the dorsal centroid sits above the line and the ventral one near it. |
| d | The balance by tertile: participant means ± SEM of `delta` with participant and responsiveness removed (the coordinate test's units), annotated with the z slope's electrode-level and participant-level p. |

Choices, and why:

- **Tertiles of z** over every electrode with coordinates: the S-N4 panel c cut,
  fixed by rule. Three ordered bands get a one-hue violet ramp (validated as
  ordinal), so they are not read as the blue/orange congruency/switch identity.
- **The separating line is the identity line**, not a dorsal/ventral boundary.
  At ~2 % of variance the tertiles overlap almost completely in score space; a
  classifier line would suggest a separation that is not there. The z cuts are
  drawn where they mean something: on the anatomy in panel a.
- **Centroids, not "the electrodes driving z".** At single-electrode reliability
  ~0.3, the most influential electrodes are partly picked for their noise.
- **Units.** The points are the pre-specified test's scores (responsiveness out,
  participant-centred) on delta's pooled scale with each score's mean added
  back, so the identity line means LWPC = LWPS. The summary prints the z slope
  of the plotted balance next to the coordinate test's as a check; they agree
  exactly on the real data (−0.00770/mm both, §19.8.2).

**The tertiles on the brain** (`fig5_height_brain.png`, added 2026-10-06;
`sfa.plot_height_bands_on_brain`). Panel b's electrodes on the fsaverage brain,
each in its tertile's colour, with each tertile's centroid as a larger sphere in
a darker shade of that colour (ventral `#635b8b`, middle `#40347c`, dorsal
`#1f1453`: the band colours × 0.6). The renderer is the one the score maps use,
so the electrodes sit where they do in those maps. The renderer draws no
legend; `fig5_height_brain_legend.png` is the legend.

- **A centroid is the mean MNI position of its tertile's electrodes, per
  hemisphere** (sign of x). A bilateral mean would put x near the midline,
  outside both hemispheres. Electrode means, like panel c's centroids, so an
  electrode-rich participant pulls its tertile's centroid. They describe where
  the sampled electrodes are and test nothing (§8). A tertile with fewer than
  three electrodes in a hemisphere gets no centroid there.
  `fig5_height_brain_centroids.csv` has the positions, and the summary prints
  them (`brain <band> <hemi> centroid MNI (x, y, z)`). Both are written on every
  run, whether or not the brain is drawn.
- **Drawn on the lateral surface.** A centroid of electrodes on curved cortex
  lies under the surface, where the translucent brain washes it out. So each one
  is drawn at its y and z on its hemisphere's lateral pial surface: the most
  lateral fsaverage vertex within 4 mm of that y, z. y and z, what a lateral
  view shows, are exact; only x moves. `centroids_on_surface=False` draws the
  true mean. The CSV and the summary always give the true mean.
- **What makes a centroid stand out.** Opacity, not size. The electrodes are
  drawn at 40 % opacity (`electrode_alpha`), the centroids opaque and only 1.5×
  the electrode diameter (`centroid_size`). Opacity also separates a centroid
  from the next band's electrodes, which a darkened colour comes close to (the
  middle centroid and the dorsal electrodes). The electrode opacity comes from
  `jim_mri.plot_on_average(elec_alpha=...)`, which defaults to 1, so the other
  brain figures are unchanged.
- Without the recon files or a display, `fig5_height_brain_sagittal.png` is
  written instead: one sagittal projection per hemisphere with the same
  colours, the cuts and the centroids. The summary marks it `FALLBACK` and gives
  the reason.

### 19.5 How to run it

**New anatomy runs** compute everything in the `continuous/` folder (block "§19"
of `summary.txt`, `section19` in `score_anatomy.json`). The tertile brain
follows `MAKE_BRAIN`, `BRAIN_HEMI` and `BRAIN_ZOOM`, as the score maps do.
`BRAIN_HEMI=split` shows both lateral surfaces, the clearest view of a
dorsal–ventral layout.

Since 2026-10-07 that includes the parts that need more than the run's own
tables, when `FOLLOWUPS` contains 19 (the default). The submitter points them
at the scatter-only runs below, under `SEG_RUN`'s `segregation_results/`:

| Variable | Default | Gives |
|---|---|---|
| `LONG_DF_CSV` | `…_fdr_bh_scatter_only_splits0/long_df.csv` (raw) | section 3: rescored with `SHARED_N_SPLITS` (200) shared splits, `per_split_shared.csv`, local similarity and the reliabilities by split scheme, in `continuous/` |
| `RT_COUPLING_CSV` | `…_fdr_bh_rt_adjusted_scatter_only_splits200/rt_adjustment_slopes.csv` | section 5's RT row |
| `RT_LONG_DF_CSV` | `…_fdr_bh_rt_adjusted_scatter_only_splits200/long_df.csv` | section 3 on RT-adjusted high gamma, `continuous/section19_rt_adjusted/` |

A file that is not there skips only its part; `summary.txt` says which and how
to make it.

**Existing runs**, from their outputs alone (no epochs, atlases or recon files):

```bash
python dcc_scripts/stats/n4_section19_followups.py \
    --anatomy-dir <anatomy run>/continuous \
    --seg-dir     <the _main_effects segregation run it read> \
    --long-df     <the same segregation run>/long_df.csv \
    --rt-coupling <A6 run>/participant_electrode_scores.csv
```

The script's five sections: 1, the height slope with participants as the unit
(§19.1); 2, the overlap r with participants as the unit (§19.2); 3, local
similarity and the reliabilities by split scheme (§19.3); 4, the combined
Figure 5 (§19.4); 5, the overlap controls (§19.7). `--sections 3,5` runs a
subset.

**Section 3 needs a split shared by each participant's electrodes**, which the
pipeline's `per_split.csv` is not. Give it one of:

- `--long-df <segregation run>/long_df.csv`: the script rescores with one trial
  split per participant (200 splits by default, `--shared-n-splits`; 5–10
  minutes for all lPFC) and writes `per_split_shared.csv`;
- `--per-split-shared <run>/per_split.csv` from a `SHARED_SPLIT=1` segregation
  run, or the `per_split_shared.csv` an earlier script run wrote.

Without either, section 3 is skipped and the rest runs.

`--brain` adds the tertile brain (§19.4). It is the one step that reads recon
files (`ECOG_RECON_DIR`, else `/cwork/$USER/ECoG_Recon`), so run it under a
virtual display, as the Slurm wrapper does:

```bash
xvfb-run -a python dcc_scripts/stats/n4_section19_followups.py \
    --anatomy-dir <anatomy run>/continuous --seg-dir <segregation run> \
    --sections 1,4 --brain --brain-hemi split
```

Keep section 1 with section 4: it supplies panel d's participant-level p, and
`fig5_height.png` is redrawn without those lines if it is left out.
`--brain-zoom` sets the camera zoom (`<1` zooms out).

**The long table needs a `trial` column**, added to `assemble_long_df` on
2026-09-27. Older tables cannot be patched from row order: rows with NaN high
gamma were dropped electrode by electrode, so two electrodes' k-th rows need
not be the same trial. The all-lPFC `_main_effects` segregation run
(2026-09-26) has no `trial` column. Tables that do:

| Long table | Electrodes | How to make it |
|---|---|---|
| **All lPFC, raw** (primary; fourth run in §19.8) | 398 | `cd dcc_scripts/stats && RT_ADJUST_HG=0 SCATTER_N_SPLITS=0 SCATTER_ONLY=1 bash submit_stability_flexibility_segregation_dcc.sh`; writes `<segregation_results>/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_scatter_only_splits0/long_df.csv` in minutes. Set both variables: the submitter defaults to `RT_ADJUST_HG=1` and 200 scatter splits. |
| All lPFC, RT-adjusted (third run) | 398 | the same command with `RT_ADJUST_HG=1` (folder `..._rt_adjusted_scatter_only_splits200`). The script sees the `rt_adjustment_slopes.csv` beside it, labels the output RT-adjusted and writes to `section19_rt_adjusted/`. |
| Task-significant lPFC (second run) | 171 | the A6 run's `long_df.csv` (2026-09-30; same epochs file, correct trials only) |

`--rt-coupling` adds the RT row of section 5: a CSV with `electrode` and `rt_r`.
For all lPFC use the RT-adjusted run's `rt_adjustment_slopes.csv`; the A6 run's
`participant_electrode_scores.csv` covers the task-significant electrodes only.

The current all-lPFC results come from running the script on the all-lPFC
anatomy folder (§17.1) with the raw long table (section 3) and, separately,
with the RT-adjusted long table and its `rt_adjustment_slopes.csv` (sections 3
and 5); see §19.8.1.

| File | What |
|---|---|
| `mni_z_slope_by_participant.csv` | per participant: electrodes, spread (mm), weight, slope, whether it entered the unweighted test |
| `mni_z_slope_loso.csv` | z slope and p with each participant left out |
| `participant_corr.csv` | per participant: separate-half LWPC–LWPS r, reliabilities, Fisher z, weight |
| `local_similarity.csv`, `_contrasts.csv`, `_comparison.csv`, `local_similarity.png` | §19.3 (with `--long-df`) |
| `per_split_shared.csv`, `reliability_by_split_scheme.csv` | §19.3: the shared-split table, and the reliabilities from both schemes |
| `overlap_controls.csv`, `overlap_loso.csv` | §19.7 |
| `fig5_height.png/.pdf`, `fig5_height_points.csv`, `_centroids.csv`, `_balance.csv`, `_balance_by_participant.csv` | §19.4 |
| `fig5_height_brain_centroids.csv` | §19.4: each tertile's mean MNI position per hemisphere, with electrode and participant counts |
| `fig5_height_brain.png`, `_legend.png` (or `fig5_height_brain_sagittal.png`) | §19.4: the tertiles and their centroids on the brain (`MAKE_BRAIN` / `--brain`) |
| `summary_section19.txt`, `section19.json` | everything above in words and numbers |

### 19.6 What to report

- **Main text, one sentence each,** next to the pooled numbers: the z slope's
  participant-level p (weighted sign-flip) and the random-slope model's p; the
  overlap r's participant-level r and p.
  On the real data both participant-level p are about 0.04 (§19.8.2), and the
  overlap holds weighted (p = 0.031) but not unweighted (p = 0.35, §19.8.3):
  word the overlap as a property of the electrode population, not of every
  participant.
- **If the participant-level p for the z slope is not below 0.05:** the
  gradient is carried by a few participants. Say so, and keep it at the weight
  the advisors gave it (one Results paragraph, S-N4).
- **Local similarity:** only from a shared-split table (§19.3), and only with
  the positive control beside it. On the real data it is not a test of
  "intermixed": the balance has no reliable within-participant variation to
  arrange (§19.8.6). Report that instead, in the wording of §19.8.7, with the
  curves in S-N4.
- **Overlap controls (§19.7):** one Methods sentence listing what was removed,
  one Results sentence with the rows that change r (only the same-half base
  effects do, §19.8.5), the rest in S-N4.
- **Leave-one-out range** of the z slope in S-N4.

### 19.7 Could something other than co-localised adaptation make LWPC and LWPS correlate?

*Added 2026-10-05.* `sfa.overlap_controls(scores, per_split, rt_coupling)`, section
5 of the script.

**What the scores are.** Each electrode's LWPC (LWPS) is a Cohen's d: the
equal-cell-weighted difference of differences of window-mean high gamma
(0–1.5 s), (incongruent − congruent) in 25 %-incongruent blocks minus the same in
75 % blocks, divided by that electrode's pooled within-cell SD of single-trial
high gamma (`_interaction_cohens_d`). So it is in units of the electrode's own
trial-to-trial variability. For the anatomy, each score is then divided by one
pooled factor, the SD of that score across all electrodes (`lwpc_s`, the "SD
units" of the figures). It is not a within-participant z-score. The pre-specified
test then regresses out responsiveness and subtracts each participant's mean
before correlating, Spearman.

**Ruled out by design.**

- **Shared trial noise** within an electrode: the two scores come from disjoint
  halves of its trials.
- **Gain:** a d is unchanged by multiplying an electrode's high gamma by a
  constant.
- **Participant offsets:** centring within participant, and a within-participant
  null.
- **Shared trials across electrodes** (the per-electrode split, §19.3): this
  enters the overlap r only through noise shared by one electrode's LWPC and
  another electrode's LWPS on common trials. With equal cell weights the two
  contrasts are orthogonal over trials (the congruency weights sum to zero
  within each incongruent-proportion level, whatever the switch composition),
  so that noise cancels in expectation. On simulated data with spatially
  correlated noise, both split schemes gave the same r (−0.039 and −0.041).

**Candidates the pre-specified test does not rule out, and the controls.**

| Candidate | How it would make r > 0 | Control (row of `overlap_controls.csv`) |
|---|---|---|
| Signal-to-noise beyond linear mean \|HG\| | A d is scale-free but not SNR-free. Where both adaptations are positive on average, better-SNR electrodes show more of both. Mean \|HG\| is one proxy, entered linearly. | `+ responsiveness, nonlinear` (log and square); `responsiveness tertile` rows (descriptive) |
| RT coupling | If high gamma tracks RT within cells, each electrode's LWPC and LWPS contain its coupling times the participant's behavioral LWPC and LWPS, both positive in most participants. More strongly coupled electrodes show more of both. Mean \|HG\| does not remove this. The same mechanism makes all four base-effect × adaptation correlations positive (the "shared component" of §16.6.5). | `+ RT coupling` (each electrode's within-cell HG–RT r, the coupling in d units); in full, the `RT_ADJUST_HG=1` run (§18.10) |
| Both adaptations scale with the base effects | If adaptation is a proportional shrink, LWPC tracks congruency and LWPS tracks switch, and congruency and switch share electrodes (r = 0.23). That puts the base effects' overlap into the adaptations. | `+ base effects, same half`: each half's adaptation scores with that half's base effects partialled out. A drop means "no overlap beyond the base effects'". It cannot separate this from residual SNR, since the base effects are also the best SNR proxy. |
| A shared spatial gradient | Both maps follow one smooth trend | `+ MNI coordinates` |
| A few participants | Electrode-rich participants dominate a pooled r | §19.2 and `overlap_loso.csv` |

On planted data, each control removes the confound it targets and leaves the
others alone. A base-driven overlap fell from 0.10 to −0.04 under the base-effect
control and was untouched by the RT row; an RT-driven overlap fell from 0.27 to
−0.06 under the RT row and was untouched by the base-effect control (tests in
`test_section19_anatomy.py`). Real-data results: §19.8.5.

### 19.8 Results on the real data (2026-10-05)

Organized by result rather than by run. Superseded run details (the first
run's local similarity, the subset-only RT rows, an earlier reading of the
reliabilities) are in the [Archive](#198-archived-parts-superseded-run-details).

#### 19.8.1 The runs

All four ran `n4_section19_followups.py` on the all-lPFC main-effect anatomy
folder (398 electrodes, 22 participants) and its per-split table. They differ in
the long table given for section 3 and the RT-coupling file given for section 5.

| Run | `--long-df` (section 3) | `--rt-coupling` (section 5) | Sections | What is current from it |
|---|---|---|---|---|
| First | none (the `_main_effects` long table has no trial ids) | none | 1–4 | The height slope and the overlap *r* with participants as the unit, and Figure 5's numbers (§19.8.2–§19.8.3). Its local-similarity numbers used the per-electrode split and are not usable. `section19/summary_section19.txt` is committed in the repo. |
| Second | the A6 run's (task-significant, 171 electrodes) | the A6 run's | 3, 5 | The subset's reliabilities by split scheme and local similarity (§19.8.4, §19.8.6); the all-lPFC overlap controls other than RT (§19.8.5) |
| Third | all-lPFC scatter-only run, RT-adjusted (`…_rt_adjusted_scatter_only_splits200`) | that run's `rt_adjustment_slopes.csv` | 3, 5 | The overlap on RT-adjusted high gamma, the RT-coupling row on every electrode (§19.8.5), local similarity on RT-adjusted high gamma (§19.8.6) |
| Fourth | all-lPFC scatter-only run, raw (`RT_ADJUST_HG=0 SCATTER_N_SPLITS=0`) | none | 3 | **The primary shared-split reliabilities and local similarity** (§19.8.4, §19.8.6) |

The Fig. 3 RT check (§19.8.8) is `f3_rt_adjusted_check.py` on the A6 run.

#### 19.8.2 The height slope with participants as the unit (first run)

| Test | Slope (SD/mm) | p |
|---|---|---|
| Electrodes (coordinate test, as before) | −0.0077 | 0.0074 |
| Participants, weighted (sign-flip; 21 participants) | −0.0077, 95 % CI [−0.0147, −0.0013] | 0.041 |
| Participants, unweighted (20 with ≥ 3 electrodes and ≥ 5 mm spread; t-test) | −0.0110 ± 0.0059 | 0.078; 13/20 negative, sign test 0.26 |
| Mixed model, random intercept | −0.0077 ± 0.0028 | 0.0052 |
| Mixed model, random slope | −0.0083 ± 0.0040; random-slope SD 0.0101 | 0.037 |
| Leave one participant out | −0.0093 to −0.0050 | 0.002 to 0.099 |

Reading: the gradient holds with participants as the unit, at p ≈ 0.04 (weighted
sign-flip and random slope). It is heterogeneous: the between-participant SD of
the slope (0.010/mm) is as large as the slope, three participants carry 38 % of
the weight, and one participant's removal takes p to 0.099. The random-intercept
model's p = 0.005 is not a participant-level test: with predictors centred
within participant it reproduces the electrode-level slope and its precision.
Quote the weighted sign-flip and random-slope p.

**Figure 5's numbers** (`fig5_height`, same run). Height tertiles cut at
MNI *z* = 16.3 and 38.1 mm; all 397 points have coordinates. Panel b's *r* is
the pre-specified +0.097 (*p* = 0.0001, from the segregation run's
`correlation.json`).

| Tertile | Electrodes / participants | Centroid LWPC, LWPS (panel c) | Centroid balance [95 % participant bootstrap] | Adjusted balance, participant mean ± SEM (panel d) |
|---|---|---|---|---|
| ventral | 132 / 19 | +0.016, +0.052 | −0.036 [−0.193, +0.119] | −0.151 ± 0.085 (20 participants) |
| middle | 132 / 18 | +0.220, +0.289 | −0.068 [−0.258, +0.058] | −0.141 ± 0.134 (18) |
| dorsal | 133 / 19 | −0.140, +0.197 | −0.337 [−0.487, −0.201] | −0.375 ± 0.121 (19) |

- The dorsal centroid sits above the identity line (leaning LWPS) with an
  interval that excludes zero; the ventral and middle centroids sit on it.
  Panel d shows the same step in participant means.
- Panels c and d average differently (electrodes vs participants, and d
  removes responsiveness as the coordinate test does), so their levels differ.
- The plotted balance gives the coordinate test's slope exactly (−0.00770/mm
  both), so the figure and the test agree.

#### 19.8.3 The overlap r with participants as the unit (first run)

Weighted *r* = +0.098, 95 % CI [+0.025, +0.157], sign-flip *p* = 0.031 (20
participants with ≥ 4 electrodes, 394 electrodes); unweighted *r* = +0.043,
t-test *p* = 0.35, 11/20 positive. The pooled *r* is carried by electrode-rich
participants; the typical participant shows little. Per-participant *r* from a
handful of electrodes is very noisy, so the unweighted test is weak, but the
result should be worded as an overlap across the electrode population, not as
one every participant shows.

#### 19.8.4 Within-participant reliabilities by split scheme

`reliability_by_split_scheme.csv`; the same electrodes in every row of a table.

All lPFC (fourth run, raw high gamma; last row from the third run):

| Split | LWPC | LWPS | congruency | switch | LWPC–LWPS r |
|---|---|---|---|---|---|
| per electrode (the pipeline's) | +0.069 | −0.094 | +0.341 | +0.235 | +0.097 |
| **shared by participant** | **+0.207** | **+0.035** | **+0.462** | **+0.351** | **+0.111** (p = 0.0002) |
| shared, RT-adjusted high gamma | +0.202 | +0.006 | +0.337 | +0.294 | +0.103 (p = 0.0004) |

Task-significant subset (second run):

| Split | LWPC | LWPS | congruency | switch | LWPC–LWPS r |
|---|---|---|---|---|---|
| per electrode (the pipeline's) | +0.009 | +0.001 | +0.247 | +0.252 | +0.073 |
| shared by participant | +0.174 | +0.141 | +0.392 | +0.388 | +0.076 |

- **The split bias is confirmed.** Every reliability rises with a shared split,
  in both sets, and the overlap *r* does not move (all lPFC 0.097 → 0.111,
  subset 0.073 → 0.076), as §19.3 and §19.7 predicted.
- **The paper's LWPS reliability argument survives**, with new numbers: in all
  lPFC switch type is still the less reliable base-effect map (0.35 against
  0.46 for congruency), and LWPS still tracks it more (§16.6.5). In the subset
  the two are equally reliable (0.39 and 0.39), and the gaps are the same
  (§18.2).
- **RT adjustment leaves the adaptation reliabilities nearly unchanged** (LWPC
  0.21 → 0.20) and lowers the base effects' (congruency 0.46 → 0.34, switch
  0.35 → 0.29). RT coupling contributes to how strongly electrodes carry the
  base effects, not to their adaptation. The RT-adjusted segregation run shows
  the same at the level of overlap: congruency–switch *r* 0.224 → 0.127,
  LWPC–LWPS 0.097 → 0.090 (§18.10).
- **LWPS has little reliable electrode-to-electrode variation within
  participant on all lPFC** (0.035; 0.14 in the subset). The overlap (0.111) is
  larger than √(0.207 × 0.035) = 0.085, which only sampling error in the small
  LWPS reliability allows, so the noise-corrected ratio is not estimable on all
  lPFC. What can be said: LWPS's measurable reliable variation is no larger
  than what it shares with LWPC. In the subset the ratio is
  0.076 / √(0.174 × 0.141) ≈ 0.49; it needs a participant-bootstrap interval
  before it is quoted, and the interval will be wide.

#### 19.8.5 Overlap controls

All lPFC: the second run, with the RT-coupling and "all of the above" rows from
the third run, which has an RT coupling value for every electrode.

| Control | r | p | Electrodes / participants |
|---|---|---|---|
| pre-specified | +0.097 | 0.0002 | 397 / 21 |
| + responsiveness, nonlinear | +0.094 | 0.0003 | 397 / 21 |
| + MNI coordinates | +0.100 | 0.0001 | 397 / 21 |
| **+ base effects, same half** | **+0.028** | **0.23** | 397 / 21 |
| + RT coupling (third run) | +0.082 | 0.001 | 397 / 21 |
| all of the above (third run) | +0.036 | 0.12 | 397 / 21 |
| responsiveness tertile low / middle / high | +0.071 / +0.131 / +0.084 | 0.10 / 0.005 / 0.06 | descriptive |
| leave one participant out | +0.081 to +0.113 | 0.001 to 0.004 | |

**The full RT control** (third run). On RT-adjusted high gamma with shared
splits, LWPC–LWPS *r* = **+0.103, *p* = 0.0004** (397 electrodes, 21
participants), against +0.097 on raw high gamma (+0.111 raw with shared
splits). **RT coupling does not produce the overlap.** With the Fig. 3 check
(§19.8.8: 72–77 % of the adaptation means survive), RT is ruled out as the
explanation of either result.

Reading:

- **Not** nonlinear responsiveness, a shared gradient, one participant, or RT
  coupling. The first three leave *r* within ±0.02; the RT-coupling covariate
  lowers it to 0.082 (still *p* = 0.001), and on RT-adjusted high gamma it is
  0.103. The overlap is also present in every responsiveness tertile.
- **The base effects account for most of it** (0.097 → 0.028). LWPC and LWPS
  share electrodes largely because each scales with the effect it regulates
  (congruency, switch type), and those two effects share electrodes
  (r = 0.23). This fits Test 1: LWPC tracks congruency more than switch type,
  and LWPS the reverse. It is the proportional reading of §16.6.5: adaptation
  is expressed where the regulated signal is.
- **What this row cannot separate:** the base effects are also the best proxy
  for an electrode's signal-to-noise. "Both adaptations scale with residual
  signal-to-noise" would also vanish here. Two observations argue against
  that being all of it. The nonlinear-responsiveness control did not move r.
  And Test 1 is process-specific (matched > crossed), which pure
  signal-to-noise would not produce. A sharper test partials each adaptation on
  its own base effect only (LWPC on congruency, LWPS on switch type): if that
  removes the overlap, it is inherited from the base effects' overlap through
  each adaptation's own signal.
- **For the paper:** "LWPC and LWPS share electrodes" stays true and robust.
  Its explanation is the base effects: the two adaptations overlap where the
  signals they regulate overlap. That is a stronger ending than "intermixed",
  which local similarity could not support (§19.8.6).

#### 19.8.6 Local similarity

Excess cross-half similarity over each participant's position-shuffle baseline
[95 % participant bootstrap], participant sign-flip *p* (§19.3).

**All lPFC, raw high gamma** (fourth run; 397 electrodes, 21 participants;
primary):

| Score | Reliability | < 10 mm | Nearest − farthest |
|---|---|---|---|
| LWPC − LWPS | −0.041 [−0.119, +0.027] | +0.009 [−0.051, +0.077], p 0.41 | +0.004, p 0.47 |
| LWPC | +0.170 [+0.075, +0.246] | **+0.170 [+0.056, +0.289], p 0.004** | **+0.227, p 0.001** |
| LWPS | −0.007 [−0.120, +0.106] | −0.011, p 0.57 | −0.018, p 0.58 |
| congruency | +0.269 [+0.058, +0.490] | +0.196, p 0.088 (10–20 mm p 0.017) | +0.237, p 0.062 |
| switch | +0.195 [+0.127, +0.275] | +0.128, p 0.057 | **+0.173, p 0.015** |

**All lPFC, RT-adjusted high gamma** (third run; 397 electrodes, 21
participants):

| Score | Reliability | < 10 mm | 10–20 mm | Nearest − farthest |
|---|---|---|---|---|
| LWPC − LWPS | −0.046 [−0.107, +0.017] | +0.005 [−0.042, +0.064], p 0.46 | −0.105, p 0.98 | −0.000, p 0.50 |
| LWPC | +0.161 [+0.063, +0.252] | **+0.165 [+0.040, +0.303], p 0.007** | +0.136, p 0.36 | **+0.227, p 0.007** |
| LWPS | −0.009 [−0.121, +0.114] | −0.005, p 0.53 | −0.004, p 0.59 | −0.004, p 0.52 |
| congruency | +0.228 [+0.054, +0.410] | +0.140, p 0.09 | +0.041, p 0.03 | +0.178, p 0.075 |
| switch | +0.195 [+0.107, +0.264] | +0.112, p 0.08 | +0.052, p 0.07 | +0.165, p 0.074 |

**Task-significant subset** (second run; 167 electrodes from 18 participants
with ≥ 3, 200 shared splits):

| Score | Reliability | < 10 mm | 10–20 mm | Nearest − farthest |
|---|---|---|---|---|
| LWPC − LWPS | +0.079 [−0.007, +0.157] | +0.053 [−0.027, +0.114], p 0.15 | +0.024, p 0.22 | +0.085, p 0.16 |
| LWPC | +0.135 [+0.004, +0.256] | +0.049, p 0.33 | +0.053, p 0.21 | +0.095, p 0.27 |
| LWPS | +0.037 [−0.096, +0.188] | −0.029, p 0.71 | −0.043, p 0.89 | −0.051, p 0.76 |
| congruency | +0.356 [+0.132, +0.592] | +0.023, p 0.41 | +0.052, p 0.24 | +0.025, p 0.41 |
| switch | +0.342 [+0.174, +0.487] | **+0.235 [+0.066, +0.361], p 0.009** | −0.087, p 0.86 | **+0.257, p 0.003** |

Reading:

- **The positive control works on all lPFC.** Electrodes within 10 mm share
  LWPC's reliable signal (+0.170, about its whole reliability) and switch
  type's; congruency leans the same way. RT adjustment does not change this.
- **The balance shows no local excess, and no reliable variation either.**
  Once the linear gradient is removed, its within-participant reliability is
  −0.041 (interval up to +0.027). With nothing reliable to be patchy, the null
  is not a positive test of intermixing.
- **The numbers hang together.** A difference score keeps
  rel(LWPC) + rel(LWPS) − 2 × overlap = 0.207 + 0.035 − 0.222 ≈ 0.02 of
  reliable variance, which is what the balance shows. LWPC has reliable
  variation and the balance has none, so LWPS's reliable variation is largely
  shared with LWPC.
- **The subset has no power.** Its adaptation reliabilities (0.04–0.14) are too
  low for any score built from them to show local structure, and the positive
  control works only for switch type. It is the "no power" row of §19.3's table.

#### 19.8.7 What this supports

Within participant, beyond the shallow dorsal–ventral gradient, electrodes do
not reliably differ in their LWPC-versus-LWPS balance, while they do reliably
differ in LWPC (and neighbours share that). There is no evidence of two kinds of
electrodes, LWPC-leaning and LWPS-leaning, mixed together. The data look like
one population whose electrodes carry more or less adaptation of both kinds, in
step with how much they carry the base effects (§19.8.5). Suggested wording:
"one population rather than two intermixed ones: beyond a shallow dorsal–ventral
bias, electrodes did not reliably differ in the balance between the two
adaptations (within-participant split-half reliability of the balance −0.04,
95 % CI −0.12 to 0.03), although they reliably differed in LWPC (0.17) and
neighbouring electrodes shared it". The interval allows a small reliable
balance (up to ~0.03), so say "not reliably", not "none".

#### 19.8.8 Fig. 3, RT-adjusted

`f3_rt_adjusted_check.py`, A6 window 0–1.5 s, task-significant lPFC, 21
participants.

| | Raw mean d | p (t; sign-flip; mixed) | RT-adjusted mean d | p (t; sign-flip; mixed) | Retained |
|---|---|---|---|---|---|
| LWPC | +0.111 ± 0.029; 17/21 > 0 | 0.0012; 0.001; 0.0002 | +0.080 ± 0.027; 16/21 > 0 | 0.0066; 0.0061; 0.0033 | 72 % (paired p = 0.008) |
| LWPS | +0.157 ± 0.040; 18/21 > 0 | 0.0009; 0.0008; 0.0001 | +0.122 ± 0.040; 15/21 > 0 | 0.0069; 0.0047; 0.0016 | 77 % (paired p = 0.022) |

Reading: both adaptations stay positive and significant with participants as
the unit after the RT-linked part of high gamma is removed. RT coupling carries
about a quarter of each, and the rest is not RT coupling. This is a window-mean
check on the task-significant electrodes, not a test of the time-resolved
clusters. A 0–0.5 s rerun of A6 would check a window before most responses.

---

## Archive: superseded results and drafts

*Everything below is superseded by §0 and §16–§19. It keeps its original
section numbers because older notes, other docs and
`n4_section15_followups.py` cite it. Where a current result still rests on a
number here, §0.3 cites it.*

| Section | What it was | Superseded by |
|---|---|---|
| §15 | Findings from the first lPFC runs (200 splits, LWPC and LWPS only) | §16–§19. Still the source of §15.6, §15.7, §15.9 and §15.10's numbers. |
| §16.7.2 | Draft Results paragraphs with the main effects | §0.8 |
| §16.7.3 | Figure 5 with four, then two panels | §19.4, §0.5 |
| §16.7.4 | Open items of 2026-09-27 | §0.11 |
| §17.2 | Methods without the §19 analyses | §0.7 |
| §17.3 | The first full Results draft | §0.8 |
| §17.4 | Can it go in the paper? (placement and readiness, 2026-10-01) | §0.5–§0.6, [`paper_draft.md`](paper_draft.md) §1.4 |
| §19.8 (parts) | Run-by-run details of 2026-10-05 that later runs replaced | §19.8.1–§19.8.6 |

## 15. Findings from the first lPFC runs (archived)

*Archived. The first two continuous-arm runs: 200 splits, LWPC and LWPS only, before the main effects (§16), the participant-level tests and the shared splits (§19). Still the source of the centroid test (§15.6), the slope's replication across trial halves and the power simulation (§15.7), the sign subsets and unsigned measures (§15.9) and the weighted centres (§15.10). Its interpretations are superseded: see §0.10.*

Two runs of the continuous arm on lPFC (`roi == 'lpfc'`), both scored on 200
disjoint half-splits with `CONTRAST_MODE=proportion` and
`EFFECT_MEASURE=cohens_d`:

| run | electrodes | participants | upstream segregation run |
|---|---|---|---|
| **all lPFC** | 398 (254 lh / 144 rh) | 22 | `window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh` |
| **task-significant lPFC** | 171 (120 lh / 51 rh) | 21 | `window_0.0to1.5s_sig_lpfc_proportion_cohens_d_fdr_bh` |

"Task-significant" is `ELECTRODES=sig`: electrodes significant against baseline
in the epochs file's stimulus test, the set the power-trace and decoding
analyses use. It is not selected on LWPC or LWPS. Every task-significant
electrode is also in the all-lPFC run with identical scores, so the smaller run
is a subset of the larger.

Every number below is reproduced from the runs' `scores_with_anatomy.csv` and
`per_split.csv` by

```bash
python dcc_scripts/stats/n4_section15_followups.py \
    --scores        <all-lPFC run>/scores_with_anatomy.csv \
    --per-split     <all-lPFC run>/per_split.csv \
    --subset-scores <task-significant run>/scores_with_anatomy.csv
```

and by the same command on the task-significant run without
`--subset-scores`. Each block of output is labelled `[pipeline: <function>]` or
`[follow-up]`. Swap-null p-values use 20,000 permutations and coordinate-shuffle
p-values 2,000. The co-localization test in §15.5 is quoted from the segregation
runs' own `summary.txt`. The few numbers that were computed by hand and are not
in the script are marked as such.

### 15.1 Takeaway

1. **LWPC and LWPS are carried by one intermixed lPFC population, not separate
   modules.** Across all lPFC electrodes the two effects are positively
   correlated (pre-specified test: r = +0.097, p ≤ 0.001), and electrodes
   positive for each effect sit in the same place (centroids 1.4 mm apart,
   p = 0.95). The task-significant set shows the same pattern at lower power
   (r = +0.077, p = 0.057; centroids p = 0.44).
2. **The balance between them tilts along the dorsoventral axis.** Ventral lPFC
   is roughly balanced and dorsal lPFC leans LWPS. Across all lPFC the z slope of
   delta is −0.0077 SD/mm (p = 0.0075), and it replicates across disjoint trial
   halves (p = 0.005). Task-significant electrodes have the same slope
   (−0.0075 SD/mm) but too few electrodes for significance (p = 0.24), and their
   slope does not differ from the other electrodes' (p = 0.67).
3. **The tilt is an effect-type × height interaction, not segregation.**
   Electrodes positive for each effect occur at every height, the correlation
   between the effects is positive in every height band, and height explains 2 %
   of delta's variance. Among task-significant electrodes both effects are
   positive at every height; LWPC is simply lower dorsally.
4. **The dorsoventral axis was not predicted.** The predicted anterior–posterior
   axis is null in both sets (p = 0.58 and 0.30). z is one of three axes, so the
   three-axis block test (p = 0.032) is the protected result, and the
   Bonferroni-corrected z p is 0.022.

Single electrodes are mostly noise: full-data split-half reliability is
0.26–0.39, and an electrode's sign agrees between its two trial halves only
51–59 % of the time. Interpret population summaries only.

⚠️ **Update to item 2 (§16.6.4).** Height and distance from the midline
correlate r = −0.58 in lPFC, and with |x| in the model the gradient follows
distance from the midline (p = 0.019), not height (p = 0.53). Describe it as
dorsomedial versus ventrolateral, and see §16.6 for the main-effect reference.

### 15.2 Corrections to the previous revision

The previous version of this section made five claims that do not hold.

| previous claim | correction |
|---|---|
| The two maps are "correlated at their noise ceiling" (noise-corrected r = 1.02, 95 % interval [0.998, 1.046]) | That interval resamples the 200 splits, which re-divide the same trials and electrodes; it measures only which trials fell in which half. Resampling participants gives [0.54, 4.71], with 9 % of resamples undefined. The ratio is not estimable at these reliabilities (§15.4). |
| On the split-averaged maps the disjoint-half within-subject correlation is r = +0.243 | Averaging each half over 200 splits rebuilds the full-data map (corr(mean xA, mean xB) = 0.987), so this correlation shares trials. It is the same quantity as `summary.txt`'s within-subject r = +0.224. |
| The §5.1 sweep's negative `reliability_y` comes from within-subject centring removing variance | Centring pushes a reliability toward zero, not below it, and cannot explain a cross-map r larger than √(rel_x · rel_y). The likely cause is the per-electrode split (§15.4). |
| Testing the separation of the sign-defined centres would be circular | A test is valid when the two sets are re-formed inside every permutation (§15.10). It restates the slope rather than adding evidence. |
| LWPC dominance increases ventrally | delta is negative on average in every height band, because LWPS is larger overall. Ventral lPFC is roughly balanced (mean delta −0.02) and dorsal lPFC leans LWPS (−0.24). |

The magnitude null was also called "settled". It is better described as
uninformative, because absolute values cannot register a small shift of scores
centred near zero (§15.9).

### 15.3 Vocabulary used in this section

Each electrode carries two **signed** scores: positive means the effect runs in
the direction behaviour predicts, negative means it runs the other way.

```text
delta = lwpc_s - lwps_s        # positive = LWPC-dominant, negative = LWPS-dominant
```

Electrodes are grouped by whether their two scores point the same way:

- **concordant**: both scores have the **same** sign (`++` or `--`).
- **discordant**: the scores have **opposite** signs (`+-` or `-+`).

This matters because `delta` mixes two things a reader might not want mixed:

| electrode | `lwpc_s` | `lwps_s` | `delta` | group | bigger in magnitude |
|---|---|---|---|---|---|
| A | +1.5 | +0.5 | **+1.0** | concordant `++` | LWPC |
| B | −1.5 | −0.5 | **−1.0** | concordant `--` | LWPC |
| C | +1.0 | −1.0 | **+2.0** | discordant `+-` | neither (tied) |
| D | +1.0 | +1.0 | **0.0** | concordant `++` | neither (tied) |

A and B have identical magnitude relationships but **opposite deltas**; C has the
largest `delta` despite its two effects being equal in size. None of this needs
resolving to read the results, because the tests are about how `delta` changes
across electrodes, not about any one electrode's value (§15.7).

### 15.4 How reliable are the maps?

Split-half reliability is computed within each split, where `xA` and `xB` are
disjoint trial halves, and averaged over splits (`map_reliability`):

| | all lPFC: LWPC | all lPFC: LWPS | task-sig: LWPC | task-sig: LWPS |
|---|---|---|---|---|
| Spearman, half data | +0.162 | +0.097 | +0.145 | +0.200 |
| Pearson, half data | +0.178 | +0.163 | +0.148 | +0.241 |
| Pearson, full data (Spearman–Brown) | 0.30 | 0.28 | 0.26 | 0.39 |
| sign agreement between halves | 55.5 % | 51.0 % | 56.7 % | 59.4 % |

⚠️ **Do not compute reliability after averaging over splits.** The A-half of one
split overlaps the B-half of another, so `corr(x̄A, x̄B)` across split-averaged
maps returns **+0.987**, an artefact, not a ceiling.

⚠️ **Do not Spearman-Brown the reliabilities for the attenuation correction.**
Both sides of `map_reliability` are half-length (`between` correlates two
half-trial estimates, and so do the reliabilities), so the ratio is already at
matched trial counts. The full-data row above is only for reading the averaged
map.

⚠️ **The +1.374 in the archived `summary.txt` was a rank-transform artefact.**
The attenuation formula is algebra for **Pearson** correlations, while `method`
defaults to Spearman, and ranking deflates the self-reliabilities much more than
the cross term:

| all lPFC | `between` | ceiling √(rel·rel) | ratio |
|---|---|---|---|
| Spearman (archived) | +0.172 | 0.125 | +1.374 |
| Pearson | +0.174 | 0.170 | +1.022 |

`map_reliability` now computes `between_noise_corrected` from Pearson whatever
`method` is, and sets `note` when the ratio is out of range or undefined.

⚠️ **The noise-corrected ratio is not estimable at these reliabilities.** The
Pearson ratio is 1.022 in all lPFC and 0.684 in the task-significant run. The
`between_noise_corrected_ci` that `map_reliability` returns bootstraps the
splits, so it only reflects which trials fell in which half (all lPFC
[0.998, 1.046]; task-significant [0.654, 0.716]). Resampling participants
gives the sampling uncertainty:

| | participant bootstrap, 95 % | resamples undefined |
|---|---|---|
| all lPFC | [0.54, 4.71] | 9 % |
| task-significant | [0.25, 2.64] | 10 % |

Report `between` and the two reliabilities, never the ratio or the split
interval. The parcel-level ratio is undefined in both runs, because the LWPC
parcel map's Pearson reliability is negative (−0.026 all lPFC, −0.201
task-significant).

⚠️ **Within-participant reliabilities are biased low by the split design.**
`split_resolved_corr`, the pre-specified co-localization test (§15.5), centres
each score within participant before computing reliabilities. There they come
out at +0.069 / −0.094 for LWPC / LWPS in all lPFC (Pearson: +0.095 / −0.079)
and +0.014 / −0.000 in the task-significant run. Two
independent halves of a real map cannot have a negative expected correlation,
and a cross-map correlation cannot exceed √(rel_x · rel_y). In all lPFC the
cross-map r (+0.105, Pearson) breaks that bound in 99.75 % of participant
resamples, so this is not sampling noise.

The likely cause: `compute_sensitivities_per_split` draws a new trial split for
**each electrode**. Within a participant, electrode i's half A therefore shares
about a quarter of the trials with electrode j's half B. Trial-level noise shared
across a participant's electrodes then correlates one electrode's half A with its
neighbours' half B, and within-participant centring turns that into a negative
correlation between each electrode's own two halves. In a simulation with
common-mode noise correlation 0.2–0.4, within-participant reliabilities that are
+0.06 to +0.10 with one split per participant fall to −0.04 to −0.15 with one
split per electrode.

This does not affect any reported result:

| result | affected by per-electrode splits? | why |
|---|---|---|
| parcel test, coordinate slopes, sign subsets, \|score\| | no | these use scores averaged over all splits, which equal the full-data scores however the splits were drawn |
| cross-validated μ² | no | it multiplies an electrode's own two halves, which never share trials |
| co-localization r | negligibly | it compares two different contrasts; same-half and separate-half correlations agree (§15.5) |
| cross-validated slope | negligibly, toward zero | simulation: 0.0041 vs 0.0040 for a true 0.004 |
| pooled (uncentred) reliabilities | negligibly | the bias is about 18 times smaller than within participant |
| within-participant reliabilities | yes, pushed down | not reported |

One split per participant, shared by all its electrodes, is needed only to report
a within-participant reliability or ceiling, or to confirm this explanation on
the real trials. It has not been implemented. (The bound check and both
simulations were run by hand and are not in the script.)

**What low reliability does and does not invalidate.** A reliability of ~0.3
means individual electrode scores are mostly noise, so no single electrode
should be interpreted and any dot map is largely noise. It does **not**
invalidate a slope or a correlation across electrodes: those summaries average
over hundreds of noisy electrodes, and §15.7 checks the slope directly on
separate trial halves.

### 15.5 Do the two effects share electrodes?

**Pre-specified test.** `split_resolved_corr` in the segregation runs: LWPC on one
trial half against LWPS on the other, residualised on responsiveness, centred
within participant, Spearman, participants with at least three electrodes
(`min_elec = 3`). From each segregation run's `summary.txt`:

| | all lPFC | task-significant |
|---|---|---|
| r | **+0.097** | +0.077 |
| p (1,000 permutations) | **≤ 0.001** | 0.057 |
| electrodes / participants | 397 / 21 | 167 / 18 |
| participants dropped (< 3 electrodes) | D0069 | D0065, D0110, D0145 |

With 1,000 permutations p cannot go below 0.001; the anatomy run's `min_elec`
sweep (2,000 permutations) gives p = 0.0005 for the same all-lPFC value. For an
exact p, rerun the segregation job with `N_PERM_CORR=10000`. Report the Spearman
values: the Pearson version (task-significant r = +0.091, p = 0.034) is not the
pre-specified test.

Reading: across all lPFC the two effects share signal. In the task-significant
set the association is not significant, and because its within-participant
reliabilities are about zero, that null is not evidence of independence either.
The two estimates are close (0.08 vs 0.10).

**Supporting checks** (follow-ups, section 4 of the script):

| | all lPFC | task-significant |
|---|---|---|
| electrodes positive on both effects | 135 vs 121.6 ± 4.2 expected, p = 0.001 | 96 vs 90.0 ± 2.3 expected, p = 0.01 |
| responsiveness vs LWPC / LWPS, within participant | +0.14 / +0.24 | +0.16 / +0.33 |

The expected counts come from shuffling the LWPS signs among each participant's
electrodes. The count uses only the signs of full-data scores, so it is weaker
than the pre-specified test and should not outrank it. More responsive
electrodes adapt more on both effects, but controlling for responsiveness only
trims the all-lPFC separate-half correlation from about +0.12 (participant only)
to +0.105 (participant and responsiveness, Pearson).

**Not shared-trial noise.** Correlations per split, averaged over splits:

| | all lPFC: pooled r | all lPFC: within participant | task-sig: within participant |
|---|---|---|---|
| `xA` vs `yA` (same trials) | +0.169 | +0.116 | +0.097 |
| `xB` vs `yB` (same trials) | +0.200 | +0.146 | +0.126 |
| `xA` vs `yB` (separate trials) | +0.174 | +0.126 | +0.118 |
| `xB` vs `yA` (separate trials) | +0.174 | +0.114 | +0.115 |

Same-trial and separate-trial correlations agree, so sharing trials does not
inflate the association.

**Why a scatter of the scores shows a stronger correlation than r.** The
pre-specified r has no scatter of its own: it averages 400 correlations (half A
against half B and the reverse, over 200 splits). A plot with one point per
electrode shows either one of them or, averaged over splits, the full-data
scores (§15.2). Those correlate about twice as strongly: +0.22 within participant
before responsiveness is removed (§15.2; with it removed, the value is in the
segregation run's `correlation_split_averaged.json`). Shared trials add little,
as the table shows. Most of the gap is attenuation: a score from half the trials
is noisier (Pearson reliability about 0.17, against 0.29 from all trials, §15.4),
so correlations between half-trial scores are smaller. A figure of the scores
must therefore say which r it prints (§15.12).

**The categorical (CMH) test is not computable in either run.** After FDR, no
electrode is individually significant for LWPC in either run (all lPFC: 0 LWPC,
0 LWPS; task-significant: 0 LWPC, 8 LWPS), so there are no groups to compare and
the odds ratio is `nan`. The segregation summary's "segregated (n.s.)" label
comes from the missing odds ratio, not from evidence of segregation. Report the
test as not computable (see `stability_flexibility_battery.md` › Outputs guide).

**Thresholded maps look disjoint whatever the truth.** With 49 % of all-lPFC
electrodes LWPC-positive and 56 % LWPS-positive, independence alone would leave
only 28 % in both maps. Apparent segregation in a thresholded dot map is not
evidence of segregation; §15.6 tests location directly.

### 15.6 Are LWPC-positive and LWPS-positive electrodes in different places?

`centroid_shuffle_test` (section 10 of the script). Only electrodes positive on
at least one score enter. The statistic is the distance between the centroid of
the LWPC-positive electrodes and the centroid of the LWPS-positive ones; an
electrode positive on both counts toward both. The null shuffles the electrode
types among electrodes of the same participant, which keeps every position and
each participant's mix of types.

| | all lPFC | task-significant |
|---|---|---|
| electrodes (LWPC+ / LWPS+ / both) | 284 (195 / 224 / 135) | 147 (114 / 129 / 96) |
| centroid distance | 1.4 mm | 3.3 mm |
| shuffle within participant | **p = 0.95** | **p = 0.44** |
| shuffle within participant × hemisphere | p = 0.78 | p = 0.45 |
| left hemisphere only (within participant) | 3.4 mm, p = 0.063 | 2.4 mm, p = 0.65 |
| right hemisphere only (within participant) | 3.8 mm, p = 0.41 | 0.4 mm, p = 0.99 |

No separation in either set. Design notes:

- **Shuffle within participant.** The mix of types differs a lot between
  participants, and participants cover different parts of lPFC. A shuffle across
  all electrodes would let "this participant is LWPS-heavy and happens to have
  dorsal electrodes" pass for anatomy. Here it happens not to change the answer
  (across all electrodes: p = 0.79 all lPFC, 0.17 task-significant).
- **Read the within-hemisphere rows.** The pooled 3-D distance in all lPFC is
  almost all x (1.39 of 1.40 mm), so it mostly reflects how the two sets split
  between hemispheres.
- **It tests location, not balance.** Only signs enter, electrodes negative on
  both effects are left out, and electrodes positive on both pull the two
  centroids together.

**Why it cannot see the dorsoventral tilt.** Take a dorsal electrode with LWPS 0.5
and LWPC 0.2, and a ventral one with LWPC 0.5 and LWPS 0.2. Both are in both sets,
so the centroids coincide, yet the balance flips from dorsal to ventral. "Positive
electrodes of each kind sit in the same place" (this section) and "the balance
shifts with height" (§15.7) are both true.

### 15.7 The balance tilts along the dorsoventral axis

⚠️ The axis is revisited in §16.6.4: height and distance from the midline cannot
be separated here, and distance fits better. The numbers below stand; the
"dorsoventral" label does not.

`relative_score_coordinate_test(value_col='delta')`:
`delta ~ mni_y + mni_z + mni_x + responsiveness + participant`, with the swap
null. The parcel test (`relative_score_roi_test(roi_col='anat')`) is the plan's
primary anatomical test and is shown alongside.

| | all lPFC | task-significant |
|---|---|---|
| parcel omnibus (primary) | F = 1.90, **p = 0.010** (19 parcels) | F = 0.91, p = 0.29 (15 parcels) |
| three-axis block F | 2.87, **p = 0.032** | 1.44, p = 0.29 |
| z slope (SD/mm) | **−0.0077, p = 0.0075** | −0.0075, p = 0.24 |
| y slope (SD/mm) | +0.0024, p = 0.58 | +0.0084, p = 0.30 |
| x slope (SD/mm) | +0.0013, p = 0.62 | −0.0002, p = 0.97 |

Units: each score is divided by its SD across the run's electrodes, so a slope is
in those SDs per mm. Across the central 90 % of the all-lPFC z range (67 mm)
delta changes by about 0.44 SD of delta.

**Read it as an interaction.** The z slope of delta is exactly the LWPC slope minus
the LWPS slope (all lPFC: −0.0037 − 0.0040 = −0.0077). So the test asks one
question: do the two effects change differently with height? That is an
effect-type × height interaction, tested by swapping the effect labels within
electrodes. Subtracting within an electrode is the paired version of it: it
cancels anything an electrode contributes to both scores (gain, responsiveness,
participant). Neither effect's own slope is significant (all lPFC: LWPC −0.0037,
p = 0.12; LWPS +0.0040, p = 0.06; task-significant: −0.0046, p = 0.29 and
+0.0029, p = 0.41), so describe the result as LWPC **relative to** LWPS, not as
either effect changing on its own.

**What it looks like.** Height bands (tertiles of the all-lPFC z; Cohen's d
adjusted for participant and responsiveness; section 12 of the script):

| band (MNI z, mm) | all lPFC: n | LWPC d | LWPS d | mean delta | LWPC–LWPS r | task-sig: n | LWPC d | LWPS d |
|---|---|---|---|---|---|---|---|---|
| ventral (−22 to 16) | 133 | 0.01 | 0.02 | −0.02 | +0.10 | 52 | 0.15 | 0.17 |
| middle (16 to 38) | 132 | 0.06 | 0.08 | −0.18 | +0.10 | 72 | 0.16 | 0.17 |
| dorsal (38 to 76) | 133 | −0.04 | 0.05 | −0.24 | +0.13 | 47 | 0.08 | 0.20 |

LWPC–LWPS r is computed on separate trial halves, within participant. The band
means are descriptive; the test is the slope.

- **Negative scores are not needed for the tilt.** Among task-significant
  electrodes both effects are positive at every height; LWPC is lower dorsally
  and LWPS is not.
- **A positive correlation and a tilt coexist.** In every band the effects are
  positively correlated while the balance shifts: a shared component plus a
  small height-dependent shift. Height explains 2.1 % of delta's variance within
  participant.

**Replication across trial halves.** The slope refitted separately on each half of
every split (all lPFC; task-significant in brackets):

| | result |
|---|---|
| mean z slope, half A / half B | −0.00780 / −0.00759 (−0.00784 / −0.00710) |
| same sign in both halves | 98.5 % of splits (79.5 %) |
| E[slope_A × slope_B] | 4.96 × 10⁻⁵ (2.34 × 10⁻⁵) |
| share of the observed slope that is signal | 91 % (65 %) |
| within-participant coordinate shuffle | **p = 0.005** (p = 0.15) |

Because the halves share no trials, E[slope_A · slope_B] is unbiased for the
squared slope. This controls trial noise, not participant sampling.

**Robustness (all lPFC).**

| check | result |
|---|---|
| add a hemisphere term | z slope −0.0074, p = 0.009 |
| z × hemisphere interaction | +0.0031, p = 0.65 (one shared slope) |
| drop the responsiveness covariate | z p = 0.007 |
| leave one participant out | z p < 0.05 in 21/22 folds (worst: drop D0146, p = 0.09) |

The separate hemisphere fits (left z p = 0.21, right p = 0.98) are underpowered,
and the interaction finds no difference to explain. In the task-significant run
no leave-one-out fold reaches p < 0.05 (worst p = 0.45), as expected at its
power.

**Parcels (all lPFC).** The extremes order dorsoventrally, matching the slope:

| Destrieux label | n | adj. mean `delta` | p | q |
|---|---|---|---|---|
| `lh_S_front_sup` | 33 | −0.54 | 0.009 | 0.126 |
| `lh_G_front_sup` | 47 | −0.30 | 0.013 | 0.126 |
| `rh_G_front_middle` | 34 | +0.41 | 0.020 | 0.129 |
| `lh_G_front_inf-Triangul` | 26 | +0.31 | 0.124 | 0.337 |

No label survives FDR, so the omnibus is the claim. In the task-significant run
one parcel (`rh_S_front_sup`, 7 electrodes from 4 participants) has q = 0.012,
but a parcel row means something only after a significant omnibus, and that
omnibus is p = 0.29. Do not report it.

**Task-significant electrodes vs the rest** (section 11, fitted in the all-lPFC
run's units):

| | electrodes | z slope (SD/mm) | p |
|---|---|---|---|
| task-significant | 171 | −0.0076 | 0.24 |
| other lPFC | 227 | −0.0103 | 0.001 |
| difference | | +0.0027 | 0.67 |

Task significance itself is not organised by height (within participant
r = +0.005, p = 0.93), so restricting to it changes the number of electrodes, not
the spatial sampling. Planting the all-lPFC slope on the task-significant layout
at the observed reliability, p < 0.05 is reached 26 % of the time, against 66 %
on the all-lPFC layout (section 9). The task-significant null is what that power
predicts.

### 15.8 The anterior–posterior hypothesis is null

`delta ~ y`: all lPFC p = 0.58 (left 0.77, right 0.22); task-significant
p = 0.30 (left 0.62, right 0.69). The §8 centre machinery is built around this
axis (`p_anterior`). Report it as a null.

### 15.9 Why unsigned and subset versions do not show it

| value regressed on z | all lPFC | task-significant | null |
|---|---|---|---|
| `delta` (signed) | −0.0077, p = 0.0075 | −0.0075, p = 0.24 | swap |
| `abs_lwpc − abs_lwps` | +0.0023, p = 0.32 | −0.0068, p = 0.17 | swap |
| cross-validated μ²: E[xA·xB] − E[yA·yB] | +0.0046, p = 0.40 | −0.0163, p = 0.22 | swap |
| `lwpc_s` alone | −0.0037, p = 0.12 | −0.0046, p = 0.29 | coordinate shuffle |
| `lwps_s` alone | +0.0040, p = 0.06 | +0.0029, p = 0.41 | coordinate shuffle |
| `delta`, positive-on-both electrodes only | −0.0059, p = 0.26 (n = 135) | −0.0047, p = 0.54 (n = 96) | swap |

**Absolute values cannot register a small shift around zero.** Across all lPFC
both scores are centred near zero (51 % of LWPC and 44 % of LWPS scores are
negative), and across the covered z range each score shifts by only about
0.25 SD. For a score distributed N(μ, 1), E|x| ≈ 0.80 + 0.4μ², which is nearly
flat near zero: moving μ from 0 to 0.25 changes E|x| by about 0.025. In the
task-significant set, where most scores are positive (67 % LWPC, 75 % LWPS; mean
d 0.14 and 0.18), |score| ≈ score, and the absolute version gives nearly the
signed slope (−0.0068 vs −0.0075). So the all-lPFC magnitude null reflects the
transform, not evidence that negative scores drive the tilt.

The cross-validated μ² (the mean over splits of xA·xB) removes the noise floor
that biases |score| upward (E|score| ≈ 0.8σ for an electrode with no effect), but
it is also quadratic in μ and just as insensitive near zero; its point estimate
points one way in all lPFC and the other in the task-significant set. Neither
unsigned version is informative. Say that unsigned measures did not detect the
tilt, not that "neither effect's magnitude changes".

**Fitting only the electrodes positive on both.** The test is valid, because
swapping the labels keeps an electrode in the `++` set, but selecting on noisy
signs shrinks the slope and discards most of the data. Planting the observed
gradient on the all-lPFC layout at the observed reliability (section 9):

| analysis | mean slope (true −0.0077) | P(p < 0.05) |
|---|---|---|
| all electrodes | −0.0079 | 66 % |
| `++` electrodes, selected on the same data | −0.0033 | 10 % |
| `++` selected on half A, fitted on half B | −0.0073 | 13 % |

With no gradient planted, all three give p < 0.05 about 5 % of the time. Only
about 68 % of electrodes that look `++` are truly `++`. The observed `++` result
(p = 0.26, same direction) is what a real gradient would produce. Selecting on
one score alone (for example LWPC > 0) is different: the swap moves electrodes in
and out of the set, so the swap null no longer applies.

**Is `delta` just picking up sign disagreement?** Concordance partitions are
invariant under the label swap, so the null stays valid inside each (all lPFC):

| subset | n | `delta` z slope | p |
|---|---|---|---|
| all | 398 | −0.0077 | 0.0075 |
| concordant (`++` or `--`) | 249 | −0.0082 | 0.003 |
| discordant (`+-` or `-+`) | 149 | −0.0149 | 0.029 |
| `++` only | 135 | −0.0059 | 0.26 |
| `--` only | 114 | −0.0079 | 0.023 |

No: the tilt is present with the same sign in every subset. In the
task-significant set every subset is non-significant (concordant p = 0.32), as
expected at its size.

**Which effect dominates moves with height; neither effect's sign does**
(within-participant correlation with z, coordinate shuffle):

| | all lPFC | task-significant |
|---|---|---|
| P(`lwpc_s` > 0) | r = −0.021, p = 0.67 | r = +0.012, p = 0.89 |
| P(`lwps_s` > 0) | r = +0.043, p = 0.42 | r = +0.131, p = 0.12 |
| P(`delta` > 0) | **r = −0.144, p = 0.006** | r = −0.096, p = 0.21 |

⚠️ **Null validity.** The sign-flip null in `_swap_null` is valid only for a
paired difference, where negation equals the label swap. It is valid for
`delta`, `abs_lwpc − abs_lwps` and the μ² difference, and **invalid** for any
single score. Passing `value_col='lwpc_s'` or `'abs_lwpc'` to
`relative_score_coordinate_test` tests nothing; the single-score rows use a
within-participant coordinate shuffle instead.

### 15.10 Weighted centres

`score_centers_per_subject` is null on every axis under every weighting:

| weighting / centre | all lPFC: dx (p) | dy (p) | dz (p) | task-sig: dz (p) |
|---|---|---|---|---|
| `abs`, `medoid=True` | +0.69 (0.59) | +0.46 (0.80) | −3.26 (0.15) | −2.35 (0.44) |
| `abs`, `medoid=False` | +0.61 (0.35) | +0.89 (0.33) | +1.42 (0.26) | −1.22 (0.35) |
| positive-clipped, `medoid=True` | +2.08 (0.29) | −0.86 (0.76) | −1.95 (0.56) | +0.09 (0.98) |

This is expected. The centres weight by |score|, which cannot see a shift of
signed scores around zero (§15.9). A null centre does not qualify §15.7.
Implementation hazards:

- `medoid=True` returns **exactly zero** displacement for 6 of 25 participant ×
  hemisphere groups in all lPFC (7 of 20 in the task-significant run). The
  medoid takes only *n* discrete values, and on clustered depth shafts both
  labels snap to the same contact. `medoid=False` produces none.
- The two centre definitions disagree in sign on dz in all lPFC (−3.26 vs
  +1.42).
- Positive clipping leaves some groups with all-zero weights for one effect;
  those centres are undefined and returned as zero.

**Centres of the sign-defined sets** describe the tilt directly:

| all lPFC | LWPC-dominant (`delta > 0`) | LWPS-dominant (`delta < 0`) | Δz |
|---|---|---|---|
| pooled | n = 186, z̄ = 24.6 | n = 212, z̄ = 29.3 | −4.6 mm |
| lh | n = 118, z̄ = 21.0 | n = 136, z̄ = 28.3 | −7.3 mm |
| rh | n = 68, z̄ = 30.9 | n = 76, z̄ = 31.1 | −0.1 mm |

Averaged over participant × hemisphere groups (both sets ≥ 2 electrodes, 21
groups) Δz is −1.83 mm, 14 of 21 in the expected direction; the pooled rows are
coverage-inflated. As a test, re-forming the two sets inside every swap and
centring heights within participant, Δz = −5.1 mm, p = 0.009 (task-significant:
−2.7 mm, p = 0.29). The two Δz values differ because one averages groups and the
other averages electrodes. The test is valid, but it restates §15.7 rather than
adding independent evidence.

### 15.11 Which electrode set to report

The choice of primary population is open. The facts that bear on it:

- **Consistency.** The power-trace and decoding analyses use the
  task-significant electrodes.
- **The plan.** §5.1 of `analysis_plans.md` › Concurrent-regulation plan and §2.3 of
  this document specify an anatomical electrode set, "not effect-selected". The
  task-significant selection is not made on LWPC or LWPS, so it does not create
  the circularity those sections warn about, but "anatomical" most naturally
  means every electrode in the region.
- **Power, not a different result.** Both positive findings (co-localization and
  the tilt) are significant only in all lPFC. Their estimates are the same in the
  task-significant set (r 0.08 vs 0.10; slope −0.0075 vs −0.0077), the two
  groups' slopes do not differ (p = 0.67), and task significance is unrelated to
  height (p = 0.93).

Options:

1. **Task-significant as primary.** Consistent with the rest of the paper. The
   anatomy section then claims no evidence of segregation and a non-significant
   trend toward overlap, with the all-lPFC results as the larger-sample check.
2. **All lPFC as primary.** Justified by the plan's anatomical population, stated
   in Methods. The task-significant set becomes the consistency check. Without
   that justification written down, switching populations for one section will
   read as choosing the set that gives significance.

Whichever is primary, report the other in full.

### 15.12 Reporting

**Tests to cite.**

| claim | test | code |
|---|---|---|
| the effects share signal | pre-specified continuous correlation on separate halves | `split_resolved_corr` (segregation `summary.txt`) |
| no anatomical separation | centroid shuffle within participant | `centroid_shuffle_test` (script section 10) |
| balance differs by parcel | parcel omnibus, swap null | `relative_score_roi_test` |
| balance tilts with height | coordinate regression, swap null; Bonferroni over 3 axes | `relative_score_coordinate_test` |
| the tilt is not trial noise | slope refitted on each half, coordinate shuffle | script section 5 |
| same tilt in both electrode sets | subset × z interaction, swap null | script section 11 |
| anterior–posterior null | coordinate regression, y | `relative_score_coordinate_test` |

**Figures.**

- LWPC and LWPS against height: band or binned means ± SEM across participants,
  two lines, one per effect. This is the figure for the tilt; the pipeline does
  not make it yet.
- LWPC against LWPS: `x_resid`/`y_resid` from the segregation run's
  `continuous.csv`, the responsiveness-residualised, participant-centred scores
  the pre-specified test correlates, labelled with its r, p and n. The points
  correlate more strongly than r (§15.5), so the caption must say that r compares
  separate trial halves. Not `joint_scatter.png`: neither its points nor its
  numbers are the pre-specified test (§6). Specification and caption: "Panel a"
  in [`analysis_plans.md` › Closing figure plan](analysis_plans.md#panel-a). The pipeline
  does not make it yet.
- `delta_by_roi.png`: adjusted mean delta per parcel, reordered by mean z.
- Per-electrode dot maps only as coverage or illustration, with a legend line
  saying single electrodes are not interpretable. Never as evidence.

**Draft Results paragraph** (task-significant electrodes as primary; for all lPFC
as primary, swap the order of the two paragraphs and drop "Because this subset
was small"). The all-lPFC version with the main effects and the revised axis
wording is in §16.7.2.

> To test whether LWPC and LWPS adaptation are carried by separate lPFC
> populations, we scored each task-responsive lPFC electrode (171 electrodes, 21
> participants) for both effects as a signed, standardized difference-of-differences
> in high-gamma power. Both effects were positive on average (mean Cohen's
> d = 0.14 for LWPC and 0.18 for LWPS). Single-electrode estimates were noisy
> (split-half reliability 0.26–0.39), so we tested only population-level
> summaries. We found no evidence that the two effects occupy different
> electrodes or regions. LWPC and LWPS scores measured on separate halves of the
> trials were weakly and non-significantly correlated (Spearman r = 0.08,
> p = 0.057; 167 electrodes from 18 participants with at least three
> electrodes). Electrodes positive for LWPC and for LWPS did not differ in
> location (centroid distance 3.3 mm, p = 0.44, electrode labels shuffled within
> participant), and the balance between the two effects (LWPC − LWPS) did not
> differ across Destrieux parcels (F = 0.91, p = 0.29).
>
> Because this subset was small, we repeated the analyses on all lPFC electrodes
> (398 electrodes, 22 participants). LWPC and LWPS were positively correlated
> (r = 0.10, p ≤ 0.001), and electrodes positive for each were again not
> spatially separated (centroid distance 1.4 mm, p = 0.95). The balance between
> the two effects, however, differed across parcels (F = 1.90, p = 0.010) and
> varied along the dorsoventral axis (three-axis F = 2.87, p = 0.03; z slope
> −0.0077 SD/mm, p = 0.007, Bonferroni-corrected p = 0.022). We had not
> predicted this axis, and the predicted anterior–posterior axis showed no effect
> (p = 0.58). Relative to LWPS, LWPC adaptation was weaker in dorsal lPFC. Among
> task-responsive electrodes, both effects were positive at every height, but
> LWPC fell from d = 0.15 ventrally to 0.08 dorsally while LWPS stayed at
> 0.17–0.20. This tilt replicated across independent halves of the trials
> (p = 0.005) and was the same size among task-responsive electrodes
> (−0.0075 SD/mm, p = 0.24; difference from the remaining electrodes, p = 0.67).
> It did not appear in the unsigned magnitude of either effect (p ≥ 0.32).
> Together, these results suggest that LWPC and LWPS adaptation are carried by an
> overlapping, intermixed lPFC population whose balance shifts modestly along the
> dorsoventral axis.

### 15.13 Open items

- [ ] Choose the primary electrode set (§15.11) and state the reason in Methods.
- [ ] Rerun both segregation jobs with `N_PERM_CORR=10000` for exact
      co-localization p-values.
- [ ] Rerun both anatomy jobs so `summary.txt` carries the Pearson-based
      noise-corrected value; the archived all-lPFC summary still shows +1.374 /
      +1.369.
- [ ] Make the LWPC-and-LWPS-by-height figure.
- [ ] Make the LWPC-against-LWPS figure from `continuous.csv` (§15.12). No rerun
      is needed.
- [ ] Leave-one-participant-out on the pre-specified correlation
      (`split_resolved_corr` with each participant dropped). The leave-one-out
      range on `joint_scatter.png` is for its own, uncorrected correlation.
- [ ] Keep `joint_scatter.png` as a pipeline diagnostic, but fix its axis labels
      (they say "disjoint half"; the points are split-averaged) and drop the
      noise-corrected value from its annotation (code not yet changed).
- [ ] Change `map_reliability`'s `between_noise_corrected_ci` from a split
      bootstrap to a participant bootstrap (code not yet changed).
- [ ] Optional: one trial split per participant in
      `compute_sensitivities_per_split`, only to report within-participant
      reliabilities or confirm §15.4's explanation.
- [ ] Deferred: confirmation across held-out participants. The trial-half
      replication controls trial noise only; leave-one-participant-out is
      reassuring but is not a held-out test.

## Archived drafts from §16, §17 and §19

### 16.7.2 Draft Results paragraphs

*Archived: superseded by §17.3, then by §0.8.*

All lPFC as the primary set (see `analysis_plans.md` › Closing figure plan, "Which electrode set
to report"). This replaces the §15.12 draft for the all-lPFC version.

⚠️ Superseded by §17.3, which adds the effect sizes, the leave-one-out range and
the hemisphere fits. §17.2 has the matching Methods, and §17.4 says what is
ready for the paper.

> We scored each lPFC electrode (398 electrodes, 22 participants) for LWPC and
> LWPS as signed, standardized differences of differences in high-gamma power,
> measured on separate halves of the trials. The two effects shared electrodes:
> LWPC and LWPS scores from separate halves were positively correlated
> (Spearman r = 0.10, p < 0.001). The balance between them (LWPC − LWPS)
> differed across Destrieux labels (F = 1.91, permutation p = 0.010; p ≤ 0.057
> with any one participant left out) and varied with position (MNI coordinates,
> F = 2.85, p = 0.031). LWPC was weaker relative to LWPS in dorsomedial lPFC,
> the superior frontal gyrus and sulcus. Because height and distance from the
> midline are correlated across lPFC electrodes (r = −0.58), the two cannot be
> fully separated; distance from the midline described the gradient better
> (joint model: distance p = 0.019, height p = 0.53). In a follow-up breakdown,
> the gradient was carried by LWPC, which was absent within about 27 mm of the
> midline and present laterally (p = 0.006), while LWPS did not vary (p = 0.75).
>
> To ask whether this organization follows the demands being regulated, we
> scored the congruency and switch-type main effects from the same trials and
> halves, weighting the proportion blocks equally. These effects also shared
> electrodes (r = 0.23, p < 0.001), and their balance (congruency − switch) also
> differed across labels (F = 1.70, p = 0.017), in step with the adaptation
> balance (r = 0.73 across 19 labels). The base-effect balance showed no
> significant spatial gradient (F = 1.81, p = 0.13); its slope along the
> adaptation gradient pointed the same way at about half the size. Neither
> congruency nor switch effects varied detectably with distance from the
> midline (p ≥ 0.34). Electrode by electrode, the two balances were linked: on
> separate trial halves, electrodes where congruency dominated were those where
> LWPC dominated (r = 0.09, p < 0.001; r = 0.09 with coordinates partialled out),
> and each adaptation effect tracked its own main effect more closely than the
> other (congruency–LWPC r = 0.22 vs switch–LWPC 0.12; switch–LWPS 0.17 vs
> congruency–LWPS 0.13). Adding the base-effect balance as a covariate reduced
> the adaptation gradient by 10 % (p = 0.015 with the covariate). Because the
> base-effect balance was measured with low reliability (split-half r = 0.08),
> the participant-bootstrap interval for that reduction (−20 % to 40 %) includes
> both no reduction and the ~15 % that full inheritance would produce. Each
> adaptation thus tracks the local strength of the demand it regulates, but
> whether the spatial gradient in their balance is inherited from the base
> effects could not be determined.

Methods needs one sentence each on: the main effects' equal weighting over
proportion blocks (§16.1), the swap null for both balances (§4.3), the
coordinate-shuffle null for single scores (§15.9), and the bootstrap (script §5).

### 16.7.3 Figure (F5, revised)

*Archived: Figure 5 is now the combined height figure (§19.4); figures to look at are in §0.5.*

*Update 2026-10-01: F5 is now two panels.* a stays. d becomes b, drawn as
matched vs crossed bars with the dm–delta correlation as the test that
compares them (§16.6.5: its covariance is the matched covariances minus the
crossed ones). b and c move to the supplement (S-N4) with the specifications
below. Current layout and reasons: [`paper_draft.md`](paper_draft.md) §1.4, F5.

*Code (2026-10-01):* the anatomy job now draws the two-panel F5 itself
(`continuous/fig5.png` and `.pdf`, §16.3), from `sfa.figure5` in
`stability_flexibility_anatomy.py`. Panel a's points are `prepare_continuous` on
the run's scores, which reproduces the segregation run's `continuous.csv`. Its r,
p and n are read from the segregation run's `correlation.json` and
`correlation_main_effects.json` (beside `SCORES_CSV`) when that run tested the
same electrodes, and recomputed otherwise; `summary.txt` says which. The
centroid annotation is `centroid_shuffle_test` (within participant) rerun on
this run's scores, so it may differ slightly from §15.6's 1.4 mm, which came
from the 200-split scores. Panel b reads Test 1's rows. To restyle the figure
without rerunning the job, edit `sfa.plot_figure5`, then run section 6 of
`n4_section16_followups.py` (§16.6).

Four panels, as specified on 2026-09-27. Plot participant means ± SEM across participants wherever a
panel summarizes electrodes; single electrodes are not interpretable (§15.4).

| Panel | Shows | Data | Status |
|---|---|---|---|
| a | Shared electrodes at both levels: congruency vs switch beside LWPC vs LWPS, on the responsiveness-residualised, participant-centred scores each pre-specified test correlates, annotated with its separate-half r, p and n | LWPC/LWPS: `x_resid`/`y_resid` in the segregation run's `continuous.csv`; congruency/switch: the same transform of `mx`/`my` in its `electrodes.csv`; r from the segregation `summary.txt` and `correlation_main_effects.json` | Not plotted; no rerun needed. Not `joint_scatter.png` (§6); specification and caption in `analysis_plans.md` › Closing figure plan, "Panel a". |
| b | The two balances by label: adjusted dm (x) against adjusted delta (y), one dot per label, sized by electrodes, coloured by distance from the midline; each omnibus F and p, and r = 0.73 | `panel_b_label_means.csv` (script §4), or `dm_per_roi.csv` + `delta_per_roi.csv` | Data ready; not plotted |
| c | The four scores by distance from the midline: base effects and adaptation as matched small multiples, participant mean ± SEM per tertile | `panel_c_midline.csv` (script §3) | Data ready; not plotted. Replaces the planned "by height" panels. |
| d | The link: r for dm vs delta, the two matched and the two crossed pairings, with p | `delta_tracking.csv` | Data ready; not plotted |

**Supplement:**

- per-label bars (`delta_by_roi.png`, `dm_by_roi.png`), coverage
  (`coverage_matrix.csv`) and leave-one-out tables (`delta_roi_loso.csv`,
  `dm_roi_loso.csv`);
- both coordinate tables (`score_anatomy.json` → `coordinates`,
  `dm_coordinates.csv`), the height-vs-midline models (script §1), and panel c
  by height (`panel_c_height.csv`);
- Test 2 (`tilt_with_dm.csv`, script §5) with dm's reliability and the
  bootstrap interval;
- the task-significant replication (§18).

### 16.7.4 Still open

*Archived: the current open items are §0.11.*

- [x] Rerun the task-significant set with main effects (segregation with
      `ELECTRODES=sig`, then the anatomy job), and run the script on it
      (2026-10-01; §18).
- [ ] Decide how to name the axis: the pre-specified model reports height, the
      follow-up favours distance from the midline (advisor question in
      `analysis_plans.md` › Closing figure plan). Since 2026-10-01 this is an
      S-N4 detail only.
- [ ] Make panel a, both halves, as specified in `analysis_plans.md` › Closing figure plan
      ("Panel a"), and plot panel d as the new F5b and panels b–c as S-N4
      panels from the script's tables. *F5a–b: code done (the anatomy job's
      `fig5.png`, §16.7.3); not yet run on the real scores. The S-N4 panels
      are still to plot.*
- [ ] Optional pipeline changes: fit |x| in the pooled coordinate model, and
      have Test 2 print dm's reliability, the implied share and the bootstrap
      interval next to the shrinkage.

### 17.2 Methods

*Archived: superseded by §0.7, which adds the §19 analyses.*

Manuscript-ready text for this run. It replaces the bracketed N4 template in
[`methods.md`](methods.md#n4-segregation-and-continuous-anatomy) for the
all-lPFC analysis, which still describes 200 splits and does not cover the main
effects. Preprocessing is not repeated here. Take it from the general iEEG
Methods, and check that it describes the epochs file this run used
(`…_drop_and_nan_thresh_perc_5.0_…_stat_func_ttest_zmax_20`).

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

Not covered above: the §19 analyses (participants as the unit, shared trial
splits, local similarity, the overlap controls) and the RT-adjusted rescoring.
§0.7 is the complete version.

### 17.3 Results

*Archived: superseded by §0.8. Its numbers stand; §17.5 gives their sources.*

This replaces the §16.7.2 draft. It adds the effect sizes, the leave-one-out
range and the hemisphere fits, and it states that the axis and the per-score
breakdown were exploratory.

*Update 2026-10-01: the paper's version is restructured.* F5 was cut to two
panels (§16.7.3), so the main-text Results in
[`paper_draft.md`](paper_draft.md) §3.4 reorder this text. The overlap result
leads, then Test 1 framed as matched vs crossed, then one paragraph with the
pre-specified parcel and coordinate tests. The gradient detail, the
base-effect balance and Test 2 moved to the supplement text
([`paper_draft.md`](paper_draft.md) §3.5). The numbers below are unchanged and
stay the source; §17.5 still gives each one's file.

> **LWPC and LWPS share electrodes, and their balance varies across lPFC.** We
> scored every lPFC electrode (398 electrodes, 22 participants) for LWPC and
> LWPS. Across this anatomically defined set both adaptation effects were small
> on average (mean Cohen's *d* = 0.01 for LWPC and 0.05 for LWPS; 49 % and 56 %
> of electrodes positive), and single-electrode estimates were noisy
> (full-data split-half reliability 0.27–0.30), so we tested only
> population-level summaries. The two effects shared electrodes. LWPC and LWPS
> scores from separate halves of the trials were positively correlated within
> participants (Spearman *r* = 0.10, *p* < 0.001; 397 electrodes, 21
> participants), and electrodes positive for each effect did not differ in
> location (centroid distance 1.4 mm, *p* = 0.95).
>
> The balance between the two effects (LWPC − LWPS within each electrode)
> nevertheless differed across Destrieux parcels (19 parcels sampled in at
> least three participants; *F* = 1.91, label-exchange permutation
> *p* = 0.010; *p* = 0.003–0.057 with each participant left out). No single
> parcel differed from zero after FDR correction (all *q* ≥ 0.13).
> Descriptively, the superior frontal gyrus and sulcus leaned toward LWPS. The
> balance also varied with position (MNI coordinates: block *F* = 2.85,
> *p* = 0.031). Relative to LWPS, LWPC was weaker dorsally (*z* slope −0.0077
> SD/mm, *p* = 0.008; Bonferroni-corrected over three axes, *p* = 0.023). The
> predicted anterior–posterior axis showed no effect (*p* = 0.58). Height
> explained about 2 % of the balance's variance, and the slope replicated
> across independent halves of the trials (*p* = 0.005). Height and distance
> from the midline are correlated across lPFC electrodes (within-participant
> *r* = −0.58), so the two cannot be fully separated. In an exploratory model,
> distance from the midline described the gradient better than height
> (distance *p* = 0.019, height *p* = 0.53), so we describe it as dorsomedial
> versus ventrolateral. In an exploratory breakdown by score, LWPC carried the
> gradient: it was slightly reversed within about 27 mm of the midline
> (adjusted *d* = −0.07) and positive farther out (0.04–0.06; slope
> *p* = 0.006), whereas LWPS did not vary (*p* = 0.75).
>
> **The adaptation balance tracks the balance of the demands.** To ask whether
> this organization follows the demands being regulated, we scored the
> congruency and switch-type main effects from the same trials and halves,
> weighting the proportion blocks equally. The base effects also shared
> electrodes (*r* = 0.23, *p* < 0.001). Their balance (congruency − switch)
> also differed across parcels (*F* = 1.70, *p* = 0.017; *p* ≤ 0.058 with any
> one participant left out), in step with the adaptation balance (*r* = 0.73
> across the 19 parcel means; same sign in 13). Electrode by electrode, on
> separate trial halves, the two balances were linked: where congruency
> dominated, LWPC dominated (*r* = 0.09, *p* < 0.001), and the link held with
> coordinates partialled out (*r* = 0.09, *p* < 0.001). It was process-specific:
> each adaptation effect tracked its own main effect more closely than the
> other one (congruency–LWPC *r* = 0.22 vs switch–LWPC 0.12; switch–LWPS 0.17
> vs congruency–LWPS 0.13). The base-effect balance showed no significant
> gradient (*F* = 1.81, *p* = 0.13); its slope pointed the same way at about
> half the size. Neither congruency nor switch varied detectably with distance
> from the midline (*p* ≥ 0.34). Adding the base-effect balance as a covariate
> reduced the adaptation gradient by 10 % (*z* slope still *p* = 0.015).
> Because the base-effect balance was measured with low reliability
> (split-half *r* = 0.08), full inheritance would produce a reduction of only
> about 15 %, and the participant-bootstrap interval (−20 % to 40 %) includes
> both no inheritance and full inheritance.
>
> Together, LWPC and LWPS adaptation are carried by one intermixed lPFC
> population. Electrode by electrode, their balance follows the local balance
> of the demands each regulates, and it shifts modestly between dorsomedial and
> ventrolateral lPFC. Whether that shift is inherited from the base effects
> could not be determined. In the task-responsive subset (171 electrodes, 21
> participants), the overlap at both levels and the electrode-level link
> between the balances had the same sign and similar size, and neither balance
> varied detectably across parcels or with position (Supplementary
> **[S-N4]**).

*Update 2026-10-01:* the last sentence now uses the task-significant
main-effect run (§18). The paper's version is in
[`paper_draft.md`](paper_draft.md) §3.4.

Supplement additions from this run, beyond §16.7.3:

- **Hemisphere fits.** Adaptation balance: left *F* = 4.22, *p* = 0.005
  (254 electrodes), right *F* = 2.33, *p* = 0.12 (144). Base-effect balance:
  left *F* = 2.45, *p* = 0.039, right *F* = 0.68, *p* = 0.45
  (`dm_coordinates.csv`). The left-hemisphere base-effect fit is why the paper
  must not say the base effects are spatially unorganized.
- **Map similarity.** Report the pooled separate-half LWPC–LWPS correlation
  and the two half-data reliabilities (Pearson 0.17, against 0.18 for LWPC and
  0.16 for LWPS), not the noise-corrected ratio. `summary.txt` flags that ratio (1.03 at electrode
  level, 3.38 at parcel level) as not estimable.

### 17.4 Can it go in the paper?

*Archived: superseded by §0.5–§0.6 and [`paper_draft.md`](paper_draft.md) §1.4.*

**Yes. It belongs in the N4 anatomy section and the closing figure (F5), led by
the overlap result, with the gradient as a modest secondary finding. It is not
ready to submit until the items at the end of this section are done.**

*Update 2026-10-01:* the gradient is now further down. F5 shows the overlap
(a) and the matched-vs-crossed tracking (b). The pre-specified parcel and
coordinate tests keep one Results paragraph, and everything else about the
gradient is in S-N4. The placement table below is updated to match;
[`paper_draft.md`](paper_draft.md) §1.4, F5 has the reasons.

**Why it can go in.**

- The main tests were specified before the data were seen. The plan's
  anatomical set and primary test are dated 2026-09-16 and 2026-09-17, and
  Tests 1 and 2 were written down on 2026-09-25, before the main-effect run on
  2026-09-26 (`analysis_plans.md` › Closing figure plan).
- The null respects the design. Exchanging the effect labels within each
  electrode keeps coverage, participant and responsiveness fixed, and the code
  recovers planted effects and returns nulls on null data
  (`tests/analysis/stats/`).
- Each positive result was checked against trial noise (separate halves),
  participant leverage (leave-one-out) and coverage (≥ 3 participants per
  parcel).
- The numbers are stable. The follow-ups reproduce exactly from this run's
  scores, and the parcel means match the earlier 200-split run to two decimals.

**What limits it, and how the text handles it.**

| Limit | Consequence for the text |
|---|---|
| The effects are small. Height explains ~2 % of the balance's variance, single-electrode reliability is ~0.3, and no parcel survives FDR. | Population-level claims only. Never name a parcel or an electrode as the driver. |
| LWPC averages *d* ≈ 0.01 across all lPFC (49 % positive). The gradient is LWPC slightly reversed medially and slightly positive laterally. | Say so in Results (the draft does). A reviewer will ask why we map an effect that is near zero on average. The answer is that N2/N3 establish LWPC in task-responsive electrodes, and here the question is how scores are distributed across the region. |
| The primary parcel test is sensitive to single participants (leave-one-out *p* up to 0.057) and is null in the task-responsive subset (*F* = 0.91, *p* = 0.29). | Report the leave-one-out range. Report the subset in full, with the power argument (a planted slope reaches *p* < 0.05 26 % of the time there, 66 % in all lPFC; §15.7). |
| The axis was not predicted, the predicted anterior–posterior axis is null, and the midline description came after looking. | Keep the gradient out of the title and the abstract (recommendation of 2026-10-01; was "at most one clause"). Label the \|*x*\| model and the per-score breakdown as exploratory, in S-N4. |
| Inheritance cannot be resolved with a base-effect balance this unreliable. | Supplement only (was one sentence in the main text). |

**Where each result goes.**

| Result | Where | Note |
|---|---|---|
| LWPC and LWPS share electrodes; no centroid separation | Main text, headline (F5a) | Pre-specified. The task-responsive subset agrees in size (*r* = 0.08, *p* = 0.055 with 10,000 permutations; §18.3). |
| Congruency and switch share electrodes | Main text, as the reference (F5a) | |
| Adaptation balance differs across parcels (*F* = 1.91, *p* = 0.010) | Main text, one paragraph with the coordinate test | Pre-specified primary test. Omnibus only. |
| Base-effect balance differs across parcels, in step (label *r* = 0.73) | S-N4 (was F5b) | The label correlation is descriptive (it shares trials). |
| Electrode-level tracking, process-specific (Test 1) | Main text (F5b, was F5d), as matched vs crossed | Pre-specified, survives coordinates. The most defensible link between the levels. The task-responsive subset has the same gaps (*r* = 0.08, *p* = 0.049; §18.4). |
| Task-responsive subset, with main effects | S-N4 in full (§18.8); one sentence in the main text | Agrees in sign and size on everything in F5; neither balance is spatially organized there. |
| Coordinate gradient (block *p* = 0.031, *z* *p* = 0.008) | Main text, in the same paragraph as the parcel test | Unpredicted, small. Slope and replication details in S-N4. |
| Dorsomedial vs ventrolateral; carried by LWPC | S-N4, exploratory (was F5c) | Naming the axis is now a supplement detail. |
| Inheritance (Test 2) | S-N4 only | Bootstrap interval spans none to all. |
| Hemisphere fits, base-effect coordinate test, leave-one-out tables, coverage | Supplement | See the 17.3 additions. |
| Weighted medoids (`score_centers.csv`) | Leave out | Null on every axis. In 6 of 25 groups both medoids land on the same contact, so the displacement is exactly zero, and the two centre definitions disagree in sign on *z* (§15.10). |
| Noise-corrected ratio (1.03 electrode, 3.38 parcel) | Leave out | Not estimable (§15.4). |
| Pooled *r* = 0.32 / within-participant *r* = 0.22 (`joint_scatter.png`) | Leave out | Full-data correlations, not the pre-specified test (§6, §15.5). |
| Brain maps | Illustration only | Caption: single electrodes are not interpretable. |

**Before it goes into a submitted draft.**

- [x] Copy this run's folder (§17.1), then rerun the task-significant set with
      main effects. Report it in full whatever it shows. *Rerun 2026-10-01
      (§18); it wrote to this folder, so check that the copy was made (§18.1).*
- [ ] Get the advisor's answers to the open questions in `analysis_plans.md` ›
      Closing figure plan: all lPFC as the primary set, lPFC-only scope, how to
      name the axis, and whether the gradient appears in the abstract.
- [ ] Make F5 panels a–b, and the two S-N4 panels that were F5b–c (§16.7.3).
- [ ] Run leave-one-participant-out on the pre-specified LWPC–LWPS
      correlation. Quote its *p* from 10,000 permutations: check whether the
      main-effect segregation run's `summary.txt` already has it
      (`N_PERM_CORR=10000` in §16.2), and rerun if not.
- [ ] Fill in the preprocessing paragraph, and archive the git commit,
      submission command and Slurm log with the run (§14).

The other §15.13 items are code clean-up or deferred, and do not block the paper.

### 19.8 (archived parts): superseded run details

*Superseded by §19.8.1–§19.8.6. Kept for the record.*

**First run, local similarity (per-electrode split; not usable).** Every score,
the balance included, had a near-range excess (+0.16 to +0.29, all
p = 0.0002) larger than its own reliability, and the reliabilities came out
negative (LWPS −0.15, balance −0.18). That is the signature of shared trial
noise between neighbours' halves (§19.3). The "share of reliability" of 34.33,
with a degenerate interval, came from dividing by reliabilities near or below
zero. The full block is in the committed
`anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/section19/summary_section19.txt`.

**Second run, RT rows on the subset.** Before RT coupling was available for all
lPFC, the RT rows used the A6 values, which cover the task-significant
electrodes only. Superseded by the all-lPFC rows of §19.8.5.

| Control | r | p | Electrodes / participants |
|---|---|---|---|
| pre-specified, RT-coupling electrodes | +0.073 | 0.068 | 167 / 18 |
| + RT coupling | +0.065 | 0.10 | 167 / 18 |
| all of the above | +0.043 | 0.28 | 167 / 18 |

The second run's reading ended: "The all-lPFC run, with 2.3 times the
electrodes, may do better; the reliabilities are the limit." The fourth run
answered it (§19.8.6).

**Third run, "Reliabilities: two changes at once".** In the third run's
`reliability_by_split_scheme` table the "per electrode" row is raw high gamma
with the pipeline's split, and the "shared, RT-adjusted" row differs in both
split and RT adjustment (LWPC 0.069 → 0.202, LWPS −0.094 → 0.006, congruency
0.341 → 0.337, switch 0.235 → 0.294). Do not read the difference as the split
bias alone. An earlier version of this paragraph guessed that RT coupling
supplies much of LWPS's reliable variation; the fourth run showed otherwise:
raw LWPS is already near zero on all lPFC. The third run also said to repeat
local similarity on raw high gamma before quoting it; the fourth run did.
