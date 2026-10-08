# DLPFC: results and what is still needed to report it

*Written 2026-10-08 from the DLPFC runs of 2026-10-07 and 2026-10-08; anatomy
updated the same day from the reruns into DLPFC-named folders (§1). Every
number below was read from the run outputs named in §1; the all-lPFC numbers
it is compared against come from the sections of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md),
[`decoding.md`](decoding.md) and [`paper_draft.md`](paper_draft.md) named in
each table.*

This doc asks one question: could the paper report DLPFC instead of all lPFC?
It collects the DLPFC power traces, decoding, cross-decoding, segregation and
anatomy results next to their lPFC counterparts, flags what differs, and lists
what has to run before DLPFC can carry the paper (§8).

---

## 0. The short version

**What DLPFC is here.** `rois_dict['dlpfc']` in `src/analysis/config/rois.py`:
`G_front_middle`, `G_front_sup`, `S_front_inf`, `S_front_middle`,
`S_front_sup`. It is lPFC without the inferior frontal gyrus (opercular,
triangular, orbital), the anterior lateral fissure and the circular insular
sulcus. It is a strict subset of lPFC: 277 of 398 electrodes (21 of 22
participants) for anatomy, 121 of 171 task-significant electrodes (20 of 21
participants) for the traces and decoding.

**What carries over.** Every main-text claim has a DLPFC counterpart of about
the same size:

- both adaptation effects in high gamma, in the behavioral direction (F3);
- both adaptations decodable, each better in its low-proportion blocks (F4);
- LWPC and LWPS share electrodes (r = 0.094, p = 0.0015; 0.121 with shared
  splits), and the overlap follows the base effects (F5b);
- the LWPC–LWPS balance differs across parcels, more strongly than in lPFC
  (F = 2.65, p = 0.002);
- congruency and switch type are decodable, with partial late transfer (F5e /
  S5).

**What does not carry over, or gets weaker:**

1. **The height gradient does not hold with participants as the unit.** The
   electrode-level z slope is steeper (−0.0101/mm, p = 0.033), but the weighted
   participant sign-flip gives p = 0.117. Under the paper's own rule (paper
   draft §1.4, F5, "What would change it"), that drops the gradient claim from
   panels a, c and d. In DLPFC the gradient is **medial–lateral**: distance
   from the midline gives F = 19.1, p = 0.0001, and with it in the model the
   z slope is p = 0.86 (§6.3).
2. **The overlap does not hold with participants as the unit** (weighted
   r = 0.086, p = 0.12). In lPFC it does (0.098, p = 0.031).
3. **LWPS no longer tracks its own base effect more than the other.** Switch–LWPS
   r = 0.100 against congruency–LWPS r = 0.109 (lPFC: 0.168 against 0.137).
   The "each adaptation tracks the signal it regulates" reading holds for LWPC
   only (§6.2).
4. **Coordinates absorb more of the overlap** (0.094 → 0.065; lPFC
   0.097 → 0.100).
5. **Congruency is decodable before the stimulus** in the DLPFC cross-decoding
   run (−1.0 to −0.5 s, cluster p = 0.002), and two of the DLPFC block-contrast
   decoders have pre-stimulus difference clusters. The F4 plan uses
   pre-stimulus decoding as the artifact meter, so this needs explaining (§3,
   §4).

**Problems with the runs themselves:**

- **DLPFC anatomy now has its own folder, and the rerun changes nothing.** The
  submit script now passes `ROI` through, so the rerun wrote
  `anatomy_a1_dlpfc_window_0.0to1.5s_all/continuous/`. Every number in it
  matches the first run, including §19, which now reads DLPFC-only long tables
  (§1). §6 stands as written.
- ⚠️ **There is no task-significant DLPFC anatomy yet.** The `_sig` and `_all`
  DLPFC folders are byte-identical (277 electrodes in both). In this route
  `ELECTRODES` only names the folder. The electrodes come from `SCORES_CSV`,
  and the submit script points that at the `all_${ROI}` segregation run whatever
  `ELECTRODES` is. A real sig run needs the sig segregation tables passed in
  (§8.1).
- ⚠️ **Both all-lPFC anatomy folders still hold DLPFC.**
  `anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` (the folder the N4 doc
  cites) still has the first DLPFC run. `anatomy_a1_lpfc_window_0.0to1.5s_all/`
  has an intermediate DLPFC run from before `ROI` was passed through (10-08
  09:36–09:42). Both have 277 electrodes. Commit 3461b20 committed the DLPFC
  `summary.txt` and `summary_section19.txt` into the `_sig` lPFC folder, so
  **git HEAD now holds DLPFC numbers under the lPFC path**. The last lPFC
  version is at `3461b20^` (398 electrodes). §8.1 restores it.
- The F3-equivalent DLPFC power traces used the 4-cell sets
  (`stimulus_lwpc_conditions`, `stimulus_lwps_conditions`), not the 8-cell
  `_block_balanced` sets F3 is drawn from. They are not yet the F3 analysis
  (§2, §8.2).

**Reading.** DLPFC replicates lPFC's adaptation, decoding and overlap results
at the electrode level. It is weaker wherever participants are the unit. Its
spatial story differs: medial–lateral rather than dorsal–ventral. All lPFC
was specified as the primary set before analysis (paper draft §1.4, F5).
Switching to DLPFC now is a post hoc choice, and reviewers will ask why (§7).

---

## 1. The runs

All runs use the epochs file
`Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20`
(`<root>` below). Paths are relative to `dcc_scripts/`.

| Analysis | Output | Date | Electrodes |
|---|---|---|---|
| Power traces, 4-cell sets (4 jobs) | `power/figs/<root>/anova_within_roi/dlpfc/`, F traces in `…/anova_within_roi/anova_F_traces/*_dlpfc_*.npz` | 10-07 20:14–20:29 | sig, 121 |
| Decoding, `_block_balanced` sets, sig (jobs 58108333–6) | `decoding/figs/<root>/<comparison>/dlpfc/`, `…/<root>/20261007_19284*_MASTER_RESULTS_job5810833[3-6]_…pkl` | 10-07 19:28 | sig, 121 / 20 participants |
| Decoding, `_block_balanced` sets, all (jobs 58108354–7) | same folders, `…MASTER_RESULTS_job5810835[4-7]_…pkl` | 10-07 19:31–19:55 | all, 277 / 21 |
| A4 cross-decoding, no groups | `decoding/results/<root>/cross_decoding_dlpfc_sig_none/stimulus_main_effect_conditions/` | 10-07 19:57 | sig, 121 |
| Segregation, main effects (4 runs) | `stats/results/<root>/segregation_results/window_0.0to1.5s_{all,sig}_dlpfc_proportion_cohens_d_fdr_bh_main_effects{,_rt_adjusted}/` | 10-07 | all 275 / 19; sig 113 / 14 |
| Segregation, scatter only (long tables) | `…/window_0.0to1.5s_sig_dlpfc_proportion_cohens_d_fdr_bh{,_rt_adjusted}_scatter_only_splits0/` | 10-07 | sig |
| Segregation, scatter only, all electrodes (DLPFC long tables for §19) | `…/window_0.0to1.5s_all_dlpfc_proportion_cohens_d_fdr_bh{_scatter_only_splits0,_rt_adjusted_scatter_only_splits200}/` | 10-08 | all |
| N4 anatomy + §15, §16, §19 follow-ups (**the one §6 reports**) | `stats/results/<root>/anatomy_a1_dlpfc_window_0.0to1.5s_all/continuous/` | 10-08 09:38–10:08 | all, 277 / 21 |
| Same, `ELECTRODES=sig` | `…/anatomy_a1_dlpfc_window_0.0to1.5s_sig/continuous/` | 10-08 09:38–10:08 | **all, 277 / 21**: identical to `_all`, not a sig run (§0) |
| First DLPFC anatomy run (superseded) | `…/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous/` (wrong name; §0) | 10-08 08:53–09:00 | all, 277 / 21 |

Anatomy inputs (from the end of its `summary.txt`): scores and per-split table
from the `all_dlpfc…_main_effects` segregation run; §19's shared-split long
table and RT coupling from the DLPFC scatter-only runs
(`all_dlpfc…_scatter_only_splits0/long_df.csv`,
`all_dlpfc…_rt_adjusted_scatter_only_splits200/{long_df,rt_adjustment_slopes}.csv`).
The first run read the **lPFC** long tables instead. Every number in
`summary.txt`, `summary_section19.txt`, `section15/`, `section16/` and
`section19_rt_adjusted/` is the same in both runs; only the input paths
differ. So using the lPFC tables made no difference, as expected: DLPFC is a
subset and the trial split is drawn per participant.

---

## 2. Power traces (F3)

Windowed ANOVA, cluster-corrected over windows, task-significant electrodes,
4-cell sets. The window spans are the significant samples of the term's
cluster mask (`sample_mask` in the F-trace npz). The sign comes from
`neg_window_mask`/`pos_window_mask`; negative is the behavioral direction
(paper draft §1.4, F3, "What the bars are").

| Term | DLPFC (121 el.) | lPFC, sig (171 el.; 09-10 figure) | lPFC, all (398 el.; 10-06 npz) |
|---|---|---|---|
| **LWPC** (congruency × incongruent proportion) | **0.88–1.50 s, negative** (7 windows) | ~0.31–1.50 s, negative (read from the figure) | none |
| **LWPS** (switch type × switch proportion) | **0.44–1.50 s, negative** (14 windows) | (figure only) | 0.75–1.43 s, negative |
| congruency × switch proportion (cross-effect) | none | – | none |
| switch type × incongruent proportion (cross-effect) | none | – | none |
| congruency main effect | 0.19–1.50 s | – | 0.31–1.50 s |
| switch type main effect | 0.44–1.50 s | – | 0.62–1.50 s |
| incongruent proportion main effect | −0.81 to −0.13 s, positive | – | −0.81 to 0.00 s |

Figures: `dlpfc/dlpfc_stimulus_{lwpc,lwps}_conditions_24_subjects_2way_*_sig_elecs_sem_shading.png`.
By eye, in both the effect (dashed minus solid) is larger in the 25 % blocks
than in the 75 % blocks late in the trial, the behavioral direction.

Reading:

- Both adaptations are present in DLPFC in the behavioral direction, and both
  cross-effects are absent, as in lPFC. F3's pattern carries over.
- The DLPFC LWPC cluster is shorter than lPFC's task-significant one (from
  0.88 s against ~0.31 s). Cluster extents are not onset estimates, so do not
  read this as a timing difference.
- **The lPFC task-significant F-trace npz files are gone** for the 4-cell sets:
  the 10-06 all-electrode run wrote the same file names (the names carry no
  `sig`/`all` tag). Only the 09-10 figure is left. The same will happen with
  DLPFC sig/all.
- **These are not F3's sets.** F3 uses `stimulus_lwpc_block_balanced_conditions`
  and `stimulus_lwps_block_balanced_conditions`. The submit script's own comment
  says why: in the 4-cell sets the contrasted cells hold 3:1 against 1:3
  mixtures of the other proportion's blocks, so a block-level offset enters
  the contrast at every time point. The only lPFC block-balanced
  F traces are on a different epochs file
  (`Stimulus_-1.0to1.5sec_0.5sec_within-1.0-0.0sec_base_…_ttest_ind_…`,
  2026-08-19). Neither ROI has a block-balanced run on `<root>`. §8.2.
- As for lPFC: `N_PERM = 500` floors cluster p at ~0.002, and the interaction
  npz's `cluster_p_values` is empty (paper draft §1.4, F3).

---

## 3. LWPC/LWPS decoding (F4, S5b)

LDA, `_block_balanced` sets, 5 bootstraps × 5 folds × 5 repeats. Mean accuracy
over post-stimulus windows (shuffle mean in brackets), peak, and the windows
above the shuffle null (cluster-corrected). The block contrast is the
comparison test between the two blocks of a context (`comparison_clusters`).
All windows are centres in seconds.

**Adaptation decoders (F4)**

| Decoder | DLPFC sig (121 / 20) | DLPFC all (277 / 21) | lPFC sig (171 / 21) | lPFC all (398 / 22) |
|---|---|---|---|---|
| i vs c, 25 % incongruent | 0.648 (0.495), peak 0.78; > null 0.75–1.38 | 0.660, peak 0.78; 0.56–1.38 | 0.682, peak 0.87; 0.50–1.38 | 0.667, peak 0.81; 0.56–1.38 |
| i vs c, 75 % incongruent | 0.556, peak 0.61; none | 0.528; none | 0.549; none | 0.529; none |
| **LWPC contrast** (25 % > 75 %) | **0.75–1.38** | 0.50–1.38 | 0.56–1.38 | 0.44–1.38 |
| s vs r, 25 % switch | 0.674 (0.499), peak 0.82; 0.44–1.38 | 0.643, peak 0.81; 0.50–1.38 | 0.687, peak 0.86; 0.62–1.38 | 0.660, peak 0.87; 0.56–1.38 |
| s vs r, 75 % switch | 0.525, peak 0.62; none | 0.530; none | 0.522; none | 0.538; none |
| **LWPS contrast** (25 % > 75 %) | **0.25–1.38** | 0.50–1.38; **also 75 % > 25 % at −0.31 to +0.38** | 0.50–1.38 | 0.50–1.38 |

**Cross-effect decoders (S5b)**

| Decoder | DLPFC sig | DLPFC all | lPFC sig | lPFC all |
|---|---|---|---|---|
| i vs c, 25 % / 75 % switch | 0.588 / 0.561; > null 0.62–1.19 / 1.19–1.38 | 0.593 / 0.579 | 0.609 / 0.570 | 0.639 / 0.570 |
| contrast, 25 % > 75 % switch | 0.56–1.06; **also −0.38 to −0.12** | 0.69–1.06; **also −0.44 to +0.12** | 0.62–1.06 | 0.62–1.19 |
| s vs r, 25 % / 75 % incongruent | 0.553 / 0.630 | 0.541 / 0.625 | 0.554 / 0.620 | 0.548 / 0.608 |
| contrast, 75 % > 25 % incongruent | 0.31–1.38 | 0.06–1.38 | 0.62–1.25 | 0.69–1.19 |

lPFC sig numbers are from jobs 57902167–70 (rerun 57902184–7 identical); lPFC
all from 57897841–4.

Reading:

- **F4 carries over.** In DLPFC both adaptation contexts decode only in the
  low-proportion block, and the block contrast is significant late in the trial.
  Accuracies are within ~0.03 of lPFC's.
- **The cross-effects carry over too**, as in lPFC. Congruency decodes better
  in 25 %-switch blocks, and switch type better in 75 %-incongruent blocks. F4
  is not a clean 2 × 2 in either ROI; report what they show.
- **Pre-stimulus clusters (new in DLPFC).** The congruency-by-switch-proportion
  contrast has a 25 % > 75 % cluster before the stimulus in both electrode
  sets, and the all-electrode LWPS contrast has a 75 % > 25 % cluster from
  −0.31 s. lPFC has none. Congruency cannot be decodable before the stimulus,
  so these are artifact-meter failures. Check the `true_v_shfle` figures to see
  whether either block's trace rises above chance before 0 s, or whether the
  comparison test is picking up two chance-level traces. Report them either
  way.

---

## 4. Congruency ↔ switch cross-decoding (F5e, S5)

A4, no electrode groups, task-significant electrodes, no RT matching, no
activity control. Pseudo-trials: ir 58, is 44, cr 77, cs 76 (255 in total).
Window centres = −0.875 + 0.0625 × index.

| | DLPFC (121 el.) | lPFC (171 el.; [`decoding.md` › A4 §13.8](decoding.md#138-results-2026-10-01)) |
|---|---|---|
| congruency within (ceiling) | mean 0.592, peak 0.755; 22/37 windows | 0.598, peak 0.757; 21/37 |
| switch type within (ceiling) | mean 0.587, peak 0.735; 22/37 | 0.601, peak 0.764; 18/37 |
| congruency → switch | mean 0.532, peak 0.639; 12/37 from +0.69 s; keeps 39 % of its ceiling | 0.540, peak 0.670; 13/37 from +0.62 s; 47 % |
| switch → congruency | mean 0.509, peak 0.593; 12/37 from +0.69 s; 17 % | 0.515, peak 0.608; 13/37 from +0.62 s; 26 % |
| congruency within, **pre-stimulus** | **windows 0–4 (−0.875 to −0.625 s centres; −1.0 to −0.5 s), cluster p = 0.002** | none |
| temporal generalization | diagonal/phasic for all three | same |

Reading:

- The pattern carries over: both variables decodable, partial transfer late
  (from ~0.7 s), a little weaker than in lPFC.
- **Congruency is decodable before the stimulus in DLPFC.** A likely cause:
  `stimulus_main_effect_conditions` pools blocks, so incongruent trials come
  mostly from 75 %-incongruent blocks. Any tonic block difference is then
  decodable as "congruency". The incongruent-proportion main effect is
  significant before the stimulus in the DLPFC power traces (§2). This
  undermines "both decodable early, no transfer early" unless it is explained.
  Check it with a block-balanced main-effect condition set before F5e uses the
  DLPFC run.
- None of the A4 controls (seeds, RT matching, `remove_mean`, positive
  controls) has been run on DLPFC (§8.4).

---

## 5. Segregation: the overlap (F5b)

Pre-specified split-resolved Spearman correlation of LWPC and LWPS on disjoint
trial halves (per-electrode splits), and the congruency–switch counterpart. The
categorical CMH is `nan` in every run, DLPFC and lPFC alike; ignore its
"segregated (n.s.)" label.

| Run | Electrodes / participants | LWPC–LWPS r | p | Reliability LWPC / LWPS | Congruency–switch r | p |
|---|---|---|---|---|---|---|
| **DLPFC all** | 275 / 19 | **+0.094** | 0.0015 | 0.049 / −0.148 | +0.209 | 0.0001 |
| DLPFC all, RT-adjusted HG | 275 / 19 | +0.091 | 0.0011 | 0.039 / −0.138 | +0.128 | 0.0006 |
| DLPFC sig | 113 / 14 | +0.057 | 0.22 | 0.011 / −0.102 | +0.190 | 0.0023 |
| DLPFC sig, RT-adjusted HG | 113 / 14 | +0.066 | 0.16 | 0.011 / −0.080 | +0.089 | 0.15 |
| lPFC all | 397 / 21 | +0.097 | 0.0001 | 0.069 / −0.094 | +0.224 | 0.0001 |
| lPFC all, RT-adjusted HG | 397 / 21 | +0.090 | 0.0001 | 0.065 / −0.121 | +0.127 | 0.0001 |
| lPFC sig | 167 / 18 | +0.077 | 0.055 | 0.014 / −0.000 | +0.167 | 0.001 |
| lPFC sig, RT-adjusted HG | 167 / 18 | +0.076 | 0.062 | 0.030 / −0.026 | +0.091 | 0.060 |

Reading:

- **All DLPFC matches all lPFC** in size (0.094 against 0.097), raw and on
  RT-adjusted high gamma. Its p is larger because it has 122 fewer electrodes.
- The task-significant DLPFC subset (113 electrodes, 14 participants) is
  underpowered (p = 0.22), as the lPFC subset was at the margin (p = 0.055).
  It would be a supplement row, as now.

---

## 6. Anatomy (F5, S-N4)

All DLPFC, `all_dlpfc…_main_effects` scores, 277 electrodes, 21 participants,
10 Destrieux labels after the coverage filter, from
`anatomy_a1_dlpfc_window_0.0to1.5s_all/continuous/` (the 10-08 rerun; the same
numbers as the first run, §1). The lPFC column is the 2026-10-01 summary (at
`3461b20^`) and N4 §19.8. There is no task-significant DLPFC column: the
`_sig` folder repeats the all-electrode run (§0).

### 6.1 Pre-specified tests

| Test | DLPFC | lPFC |
|---|---|---|
| **Balance (delta) ~ parcel**, swap null | **F = 2.65, p = 0.0018**; 0.001–0.011 leaving out each participant | F = 1.90, p = 0.010; 0.003–0.057 |
| … parcels with q < 0.05 | one (right middle frontal gyrus, q = 0.048) | none |
| **Balance ~ MNI coordinates** | block F = 2.38, **p = 0.066**; z −0.0101/mm, p = 0.033; x, y n.s. | F = 2.87, p = 0.030; z −0.0077/mm, p = 0.0074 |
| … per hemisphere | lh F = 4.76, p = 0.003 (x p = 0.007); rh F = 3.87, p = 0.014 (x p = 0.023), opposite x signs = more LWPC laterally in both | lh p = 0.005; rh p = 0.12 |
| Medoids (descriptive) | LWPC 3.8 mm anterior of LWPS (p = 0.031) | 0.5 mm |
| Base-effect balance (dm) ~ parcel | F = 1.22, p = 0.25 | F = 1.70, p = 0.017 |
| dm ~ coordinates | F = 1.30, p = 0.27 | F = 1.84, p = 0.12 |

Under the paper's rule (omnibus only, never a named parcel), the q = 0.048
parcel goes in the supplement table only.

### 6.2 Overlap, Test 1 and controls

| | DLPFC | lPFC |
|---|---|---|
| LWPC–LWPS r, electrodes (pre-specified) | +0.094 [0.034, 0.151], p = 0.0013 | +0.097, p = 0.0001 |
| … participants, weighted (n − 3) | **+0.086 [−0.004, 0.167], p = 0.12** (17 participants) | +0.098 [0.025, 0.157], p = 0.031 |
| … participants, unweighted | +0.050, p = 0.20, 8/17 positive | +0.043, p = 0.35 |
| … shared splits | +0.121 [0.053, 0.184], p = 0.0002 | +0.111, p = 0.0002 |
| … shared splits, RT-adjusted HG | +0.116 [0.049, 0.184], p = 0.0006 | +0.103, p = 0.0004 |
| Shared-split reliability LWPC / LWPS / congruency / switch | 0.203 / −0.000 / 0.471 / 0.323 | 0.207 / 0.035 / 0.462 / 0.351 |
| Congruency–switch, shared splits | +0.230, noise-corrected 0.59 [0.39, 0.81] | – |
| + nonlinear responsiveness | +0.089, p = 0.002 | +0.094 |
| **+ MNI coordinates** | **+0.065, p = 0.015** | +0.100 |
| **+ base effects, same half** | **+0.043, p = 0.099** | +0.028, p = 0.23 |
| + RT coupling | +0.095, p = 0.0014 | +0.082 |
| all of the above | +0.037, p = 0.15 | +0.036, p = 0.12 |
| leave one participant out | 0.070 to 0.117, p 0.001–0.011 | 0.081 to 0.113 |
| **Test 1** dm vs delta (matched − crossed) | +0.074, p = 0.0098 | +0.088, p = 0.0005 |
| congruency–LWPC (matched) / switch–LWPC (crossed) | 0.230 / 0.145 | 0.217 / 0.122 |
| **switch–LWPS (matched) / congruency–LWPS (crossed)** | **0.100 / 0.109** | 0.168 / 0.137 |
| Centroid distance, LWPC- vs LWPS-positive | 3.38 mm, p = 0.15 | 1.40 mm, p = 0.95 |

Reading:

- The overlap and its explanation (the base effects) carry over at the
  electrode level, and survive RT on both routes.
- **Not with participants as the unit.** lPFC's weighted test was the reason
  the paper could say the overlap generalizes across participants (paper draft
  §1.2); DLPFC does not support that.
- **Test 1 is carried by LWPC alone.** LWPS tracks congruency as much as switch
  type in DLPFC. The dm-vs-delta test still passes because LWPC's specificity is
  large. "Each adaptation tracks the effect it regulates" would have to become
  "LWPC tracks congruency more than switch type".
- **Coordinates take a third of the overlap** in DLPFC (0.094 → 0.065), but
  none in lPFC. The overlap partly reflects a shared spatial gradient within
  DLPFC (§6.3).

### 6.3 Height, distance from the midline, Figure 5

| | DLPFC | lPFC |
|---|---|---|
| z slope, electrodes | −0.0101/mm [−0.0186, −0.0014], p = 0.033 | −0.0077/mm, p = 0.0074 |
| … participants, weighted sign-flip | **−0.0101 [−0.0208, +0.0014], p = 0.117** (19) | −0.0077 [−0.0147, −0.0013], p = 0.041 |
| … participants, unweighted | −0.0205 ± 0.0114, p = 0.090, 12/18 negative | −0.0110, p = 0.078 |
| … mixed, random slope | −0.0124 ± 0.0060, p = 0.037 | −0.0083, p = 0.037 |
| … mixed, random intercept | failed (singular matrix) | p = 0.005 |
| … leave one participant out | p 0.008–0.13 | 0.002–0.099 |
| r(z, \|x\|) within participant | −0.39 | −0.58 |
| **delta ~ \|x\| alone** | **F = 19.1, p = 0.0001** | \|x\| fits better than z (p = 0.019 against 0.53) |
| delta ~ y + z + \|x\| | **\|x\| +0.023/mm, p = 0.0006; z p = 0.86** | – |
| Which score carries \|x\| | LWPC +0.019/mm, p = 0.0015; congruency p = 0.002; LWPS, switch n.s. | LWPC, p = 0.006 |
| Figure 5 tertile cuts | z = 29.6 and 46.0 mm | 16.3 and 38.1 mm |
| Centroid balance ventral / middle / dorsal | −0.06 [−0.24, 0.12] / **−0.22 [−0.43, −0.01]** / **−0.31 [−0.48, −0.11]** | −0.04 / −0.07 / −0.34 [−0.49, −0.20] |
| Balance by \|x\| tertile (adjusted mean delta), medial / middle / lateral | −0.37 / −0.25 / +0.01 | – |
| Test 2: z slope with dm partialled | −0.0096, p = 0.041; shrinkage 0.05 | shrinkage 0.10 |

Reading:

- **In DLPFC the balance is organized medial–lateral, not dorsal–ventral.**
  Distance from the midline beats height outright, and height adds nothing once
  it is in the model. That is the same direction as lPFC's exploratory finding,
  but much clearer, because DLPFC drops the ventrolateral IFG, where height and
  laterality move together. LWPC is weak near the midline and present laterally.
  That is a cleaner anatomical statement than lPFC's height gradient, **but it
  is exploratory** (§16 follow-up, not the pre-specified coordinate model).
- The pre-specified coordinate block is n.s. in DLPFC (p = 0.066). Its z slope
  is nominally significant at the electrode level only.
- **By the paper's rule, the DLPFC height gradient does not go in F5.** The
  weighted participant p is 0.117. F5a, c and d would have to change to the
  medial–lateral axis, labelled exploratory, or move to the supplement.

### 6.4 Local similarity (§19.3)

Raw high gamma, 275 electrodes, 19 participants, 200 shared splits from the
DLPFC long table, linear gradient removed, electrodes as the unit [95 % CI],
one-sided p.

| Score | Reliability | < 10 mm excess | Nearest − farthest |
|---|---|---|---|
| LWPC − LWPS | −0.010 [−0.202, 0.182] | +0.026, p 0.37 | +0.030, p 0.38 |
| LWPC | +0.157 [0.010, 0.303] | **+0.147, p 0.0095** | **+0.253, p 0.0003** |
| LWPS | −0.063 [−0.268, 0.141] | −0.074, p 0.80 | −0.088, p 0.77 |
| congruency | +0.330 [0.216, 0.444] | **+0.285, p 1e-6** | **+0.375, p 6e-8** |
| switch | +0.155 [0.006, 0.303] | **+0.133, p 0.018** | **+0.158, p 0.027** |

RT-adjusted high gamma gives the same picture (`section19_rt_adjusted/`: LWPC
nearest − farthest +0.259, p = 0.0002; balance +0.026, p = 0.40).

Reading: as in lPFC. The positive control works (better than in lPFC:
congruency and switch both show it), and the balance has no reliable
within-participant variation beyond the gradient. "One population rather than
two intermixed ones" carries over.

---

## 7. Should the paper switch?

What argues for it:

- DLPFC is a conventional, nameable region, and its spatial result is cleaner:
  a medial–lateral balance that height does not explain.
- The parcel test is stronger (p = 0.002 against 0.010).
- Every electrode-level result in F3, F4 and F5 replicates.

What argues against it:

- **All lPFC was specified as the primary set before analysis.** Choosing
  DLPFC after seeing both sets is a forking path, even though DLPFC is a
  subset. If you switch, give an anatomical reason that does not depend on
  these results (e.g. excluding IFG and the insular sulcus as a different
  functional territory). Say in the Methods that the region was narrowed after
  the lPFC analyses, and keep lPFC as a supplement row for every result.
- The two participant-level results the paper leans on (the overlap and the
  gradient) do not hold in DLPFC.
- LWPS's specificity for switch type (half of Test 1) is gone.
- The pre-stimulus congruency decoding has to be explained before F5e uses
  DLPFC.

A middle route: keep lPFC primary and report DLPFC as the anatomical
follow-up. The medial–lateral result is a natural S-N4 paragraph ("within
DLPFC, the balance follows distance from the midline, not height").

---

## 8. What to run before DLPFC can be reported

Run each from a compute node with the preamble in
[`CLAUDE.md`](../CLAUDE.md). `cd` into the script's folder first. Read each
submit script's arrays before running: several are edited in the working tree
right now (`git diff dcc_scripts/`).

### 8.1 Anatomy: restore lPFC, and add a real task-significant DLPFC run

Done on 10-08: DLPFC into its own folder, with DLPFC-only long tables for §19
(§1). Two things are left. `ELECTRODES` is fixed in
`submit_stability_flexibility_anatomy_dcc.sh` (line 39, now `all`) and is not
read from the environment. It only names the output folder, so both runs below
set it to `sig` first and back to `all` afterwards.

1. **Restore all lPFC (required).** Set `ELECTRODES=sig` so the run lands in
   the folder the N4 doc cites. That overwrites the superseded first DLPFC run
   there (§1); the `_all` DLPFC folder has the same numbers. `ROI` now picks
   the scores, the long tables and the folder:

   ```bash
   cd dcc_scripts/stats
   ROI=lpfc bash submit_stability_flexibility_anatomy_dcc.sh
   ```

   Then compare the tracked `summary.txt` and `summary_section19.txt` with
   the last lPFC version, not with HEAD (HEAD holds DLPFC, §0):
   `git diff 3461b20^ -- 'results/<root>/anatomy_a1_lpfc_window_0.0to1.5s_sig/'`.
   Only paths should differ (398 electrodes; F = 1.907, p = 0.0102). The stray
   DLPFC run in `anatomy_a1_lpfc_window_0.0to1.5s_all/` can be deleted, or
   overwritten by an `ELECTRODES=all` lPFC run.

2. **Task-significant DLPFC (needed for a sig supplement row).** Pass the
   `sig_dlpfc` segregation tables, because the defaults point at `all_${ROI}`:

   ```bash
   cd dcc_scripts/stats
   SEG=/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/stats/results/<root>/segregation_results
   SIG=$SEG/window_0.0to1.5s_sig_dlpfc_proportion_cohens_d_fdr_bh
   ROI=dlpfc \
     SCORES_CSV=${SIG}_main_effects/electrodes.csv \
     PER_SPLIT_CSV=${SIG}_main_effects/per_split.csv \
     LONG_DF_CSV=${SIG}_scatter_only_splits0/long_df.csv \
     RT_LONG_DF_CSV=${SIG}_rt_adjusted_scatter_only_splits0/long_df.csv \
     RT_COUPLING_CSV=${SIG}_rt_adjusted_scatter_only_splits0/rt_adjustment_slopes.csv \
     bash submit_stability_flexibility_anatomy_dcc.sh
   ```

   This replaces the duplicate in `anatomy_a1_dlpfc_window_0.0to1.5s_sig/`.
   Check that its `summary.txt` reports 121 electrodes and 20 participants
   (the sig `electrodes.csv`; 113 / 14 in the split tests), not 277. The sig RT-adjusted scatter-only run is
   `splits0`, while the all-electrode one is `splits200`. The job rescores the
   shared splits itself (`SHARED_N_SPLITS=200`), so this should not matter;
   check that `section19_rt_adjusted/` was written. Expect it to be
   underpowered, like the sig segregation (§5: 14 participants).

   Do steps 1 and 2 with the same `ELECTRODES=sig` edit, then set it back to
   `all`. Note that the `_sig` suffix means different things in the two
   folders: task-significant electrodes for DLPFC after step 2, all electrodes
   for lPFC (N4 doc, §13).

### 8.2 Power traces on F3's sets (required for F3)

F3 needs the 8-cell `_block_balanced` sets on `<root>`, for DLPFC and for lPFC
(lPFC has them only on the August epochs file).

- In `dcc_scripts/power/submit_specific_conditions_power_traces_dcc.sh`,
  comment out the active 4-cell `CONDITIONS=(…)` block (lines 8–13) and
  uncomment the `_block_balanced` block (lines 42–45).
- `ROIS_DICT` in `run_power_traces_dcc.py` is `dlpfc` in the working tree. Run
  once, then set it to `lpfc` and run again; keep `ELECTRODES=sig` (the default).
- Then `cd dcc_scripts/power && bash submit_specific_conditions_power_traces_dcc.sh`
  (2 jobs per ROI).
- Before it is a reported number: raise `N_PERM` above 500 and save the cluster
  p-values (already on the F3 list, paper draft §1.7).
- Move the 4-cell DLPFC outputs aside first if you also want an all-electrode
  run: sig and all write the same npz names (§2).

### 8.3 F3's RT check and A6 on DLPFC

```bash
cd dcc_scripts/stats
ROIS=dlpfc bash submit_stability_flexibility_brain_behavior_dcc.sh
```

One job. New A6 runs write `group_adaptation_rt_check.csv` (F3's RT-adjusted
check, paper draft §1.4, F3) and the S-BB numbers for DLPFC. Optionally add
`WINDOW_TMAX=0.5` for the pre-response window.

### 8.4 Decoding and cross-decoding

- **Explain the pre-stimulus clusters** (§3, §4). Open the DLPFC `true_v_shfle`
  and `comparison` figures for `i_vs_c_at_sw25/75` and `LWPS_block_balanced`
  (all electrodes). For A4, rerun with a block-balanced main-effect condition
  set if one exists in `config/condition_registry.py`, and check whether the
  pre-stimulus congruency decoding goes away.
- **Leave one participant out** for F4: `submit_loo_decoding_dcc.sh` reads
  `ROIS` from the environment, so `ROIS=dlpfc bash submit_loo_decoding_dcc.sh`
  works for the region. Its `CONDITIONS` are the two cross-effect sets: add
  `stimulus_lwpc_block_balanced_conditions` and
  `stimulus_lwps_block_balanced_conditions`. Check `LEAVE_OUT_SUBJECTS` (14
  participants) against the 20 that have DLPFC task-significant electrodes. Also
  check `ELECTRODES` in `run_decoding_dcc.py` (currently `'all'`). Jobs =
  conditions × participants.
- **A4 controls on DLPFC** (S5 list; `ROI=dlpfc` is the working-tree default):

  ```bash
  cd dcc_scripts/decoding
  RT_MATCH=rt bash submit_stability_flexibility_cross_decoding_dcc.sh
  RT_MATCH=random bash submit_stability_flexibility_cross_decoding_dcc.sh
  ACTIVITY_CONTROL=remove_mean bash submit_stability_flexibility_cross_decoding_dcc.sh
  for s in 1 2 3 4; do SEED=$s bash submit_stability_flexibility_cross_decoding_dcc.sh; done
  ```

  Check that the seeded runs land in separate folders before the loop;
  otherwise they overwrite each other.

### 8.5 Coverage (F2, S1)

Add DLPFC rows to the coverage table: 277 electrodes / 21 participants (all),
121 / 20 (task-significant); segregation keeps 275 / 19 and 113 / 14
(participants with too few electrodes drop out).

### 8.6 Already done for DLPFC

Segregation (all and sig, raw and RT-adjusted; scatter-only long tables for
both), anatomy with §15/§16/§19 on all electrodes in
`anatomy_a1_dlpfc_window_0.0to1.5s_all/` (10-08 rerun), LWPC/LWPS and
cross-effect decoding (sig and all), A4 baseline cross-decoding, 4-cell power
traces.
