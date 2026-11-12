# Paper draft: figure plan, Methods and Results

*2026-11-12: F4 decisions (§1.1, §1.5; draft text in §3.4 and §5.2). The
anatomy panel is the LWPC–LWPS scatter; its overlap is shown in the text to go
with the base effects (controls and matched-vs-crossed), which motivates the
coding-axes panels on the base codes. Controls are reported in the text, with
the table or forest plot in S-N4. F4 stays on all lPFC. Scatter restyled; the
coding-axes figures are now in `figures_2026-11-11/` (task-seq. block
separations were empty before).*

*2026-11-10: new story, four main figures (§1). The previous figure plan (F1–F6, 2026-10-29) moved, unedited, to [`analysis_plans.md`](analysis_plans.md#paper-figure-plan-of-2026-10-29-superseded-2026-11-10). The Methods, Results and captions below (§2–§5) still use the old figure numbers; §1 maps them to the new ones.*

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

*2026-10-08: DLPFC runs of F3–F5 are in. Reporting DLPFC instead of all lPFC
is under consideration; the comparison, and what still has to run, is in
[`archive/dlpfc_results.md`](archive/dlpfc_results.md) (§1.2, last row). Nothing below has been
switched to DLPFC.*

*2026-10-16: the coding-axes runs are in: the subspace test, the trajectories,
and RT matching over five seeds ([`coding_axes.md`](coding_axes.md) §10). They
replace the A4 cross-decoding as F5's base-effect panels (d–e; A4 moves to S5),
withdraw "largely distinct / separable" for the base-effect codes, and add an
RT sentence to the main text (§1.1; §1.2, last rows; §1.4, F5; §2.6; §3.4).
Drafted by Claude; not yet reviewed.*

This document has three parts:

1. **Figure plan:** the main-text figures (F1–F4) and the supplement, panel by
   panel, with where each figure file is and the code that makes it.
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
  - [1.1 Status at a glance](#11-status-at-a-glance)
  - [1.2 F1 — Task, proportion manipulation, behavior](#12-f1--task-proportion-manipulation-behavior)
  - [1.3 F2 — lPFC: coverage, signal, and why lPFC](#13-f2--lpfc-coverage-signal-and-why-lpfc)
  - [1.4 F3 — Both adaptations in power and in decoding](#14-f3--both-adaptations-in-power-and-in-decoding)
  - [1.5 F4 — One population, partly shared codes, adapting to block proportion](#15-f4--one-population-partly-shared-codes-adapting-to-block-proportion)
  - [1.6 Not in this story](#16-not-in-this-story)
- [2. Methods](#2-methods)
- [3. Results](#3-results)
- [4. Discussion sentences](#4-discussion-sentences)
- [5. Figure captions](#5-figure-captions)

---

## 1. Figure plan

**The story as of 2026-11-10.** Four main figures. The plan before this one
(the 2026-10-29 F1–F6 arc and everything behind it) is kept, unedited, in
[`analysis_plans.md` › Paper figure plan of 2026-10-29](analysis_plans.md#paper-figure-plan-of-2026-10-29-superseded-2026-11-10).
Section numbers §1.1–§1.7 quoted in older docs and in §2–§5 below refer to that
plan.

All runs use the quality-checked participant set (`SUBJECT_SET=good`, 23
participants; [`analysis_guide.md`](analysis_guide.md) §3) unless a row says
otherwise. `<E>` stands for the epochs folder name
`Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20`.
Paths are relative to the repo root. **Redraw** changes only the look and runs in
seconds from saved tables. **Rerun** recomputes the analysis on the cluster. How to change
a figure's look is in [`figure_editing_guide.md`](figure_editing_guide.md), and
the open items from the figure discussion are in
[`subspace_figure_todos.md`](subspace_figure_todos.md).

**Old names.** The code and output files still use the old figure numbers:

| New | Old name in files and in §2–§5 below |
|---|---|
| F2 | F2 (coverage and signal) |
| F3a power traces | F3, `fig3_*` in `dcc_scripts/vis/paper_panels/` |
| F3b decoding | F4, `fig4_*` / `sfig4_*` |
| F4a scatter | F5 a–b, `fig5_height_scatter` |
| F4b–d coding axes | F5 d–e / F6, `coding_axis_cosines`, `trajectories_*`, `proportion_trajectories_*`, `block_separations_*` |

### 1.1 Status at a glance

| Fig | Claim | Status |
|---|---|---|
| F1 | Both proportion manipulations change behavior (RT and error rate) | Made outside this repo |
| F2 | lPFC is covered, task-responsive, and the region that carries the task | a–c made. Supplement: window-mean task and letter decoding made, including occipital; time-resolved version submitted 2026-11-10 (jobs 61504712 task, 61504713 big letter) |
| F3 | lPFC high gamma carries LWPC and LWPS in power and in decoding | Panels made, good set. Power traces from one pooled two-way ANOVA per term. LWPC (0.50–1.50 s) and LWPS (0.62–1.50 s) hold, and they also hold in the block-balanced three-way and the four-way ([`analysis_plans.md` › Power-trace ANOVA design](analysis_plans.md#power-trace-anova-design-two-way-three-way-or-four-way-2026-11-10) §6). Cross-effects shown, with no claim of independence. **Open:** subtractions or four-trace panels as the main figure |
| F4 | The two adaptations sit on one intermixed population; their overlap goes with the base effects, whose congruency and task-seq. codes are partly shared and change with block proportion | Panels made (scatter 2026-11-12, coding axes 2026-11-11). **Decided 2026-11-12:** LWPC–LWPS scores in a; all lPFC. **Open:** one figure (a–d) or two (Fig A "partly shared" / Fig B "adapts", as in [`subspace_figure_todos.md`](subspace_figure_todos.md)) |

### 1.2 F1 — Task, proportion manipulation, behavior

a, the task; b, the 2 × 2 block-proportion manipulation (blocks A 75 %
incongruent / 25 % switch, B 75 / 75, C 25 / 25, D 25 / 75); c, RT and error
rate: the congruency effect by incongruent proportion (LWPC) and the switch cost
by switch proportion (LWPS). Made outside this repo.

### 1.3 F2 — lPFC: coverage, signal, and why lPFC

| Panel | Content | Figure | Code |
|---|---|---|---|
| a | All analyzed lPFC electrodes and the task-significant ones, on the brain (the render also draws every localized contact in grey) | `dcc_scripts/vis/figs_subjects-good/brain_all_recon_elecs__all_lpfc_elecs__sig_lpfc_elecs.png` / `.tif`, electrode counts in the `electrode_counts_*.json` beside it | `dcc_scripts/vis/plot_sig_electrodes_dcc.py` (sets and colours: `condition_plot_specs.py`, `ELECTRODE_SET_COLORS`) |
| b | Example spectrogram, one lPFC electrode: D0094 LFPM12 (`ctx_lh_S_front_inf`), wavelet, correct trials, −0.5 to 1.5 s | `dcc_scripts/spec/figs/example_spectrogram_D0094_LFPM12_Stimulus_c+Stimulus_i_wavelet_-0.5to1.5s_vlim3.0dB.png` / `.pdf` | `dcc_scripts/spec/plot_example_spectrogram_dcc.py` |
| c | High-gamma power trace over every stimulus-locked trial, all task-significant lPFC electrodes, mean ± SEM | `dcc_scripts/power/figs/<E>/anova_within_roi_subjects-good/lpfc/lpfc_stimulus_all_conditions_23_subjects_sig_elecs_sem_shading.png` | `dcc_scripts/power/power_traces_dcc.py`, condition label `stimulus_all_conditions` (one condition, so no test is run) |
| S | Task (global/local) and big-letter decoding in lPFC and occipital cortex, to motivate lPFC (read the result off the runs before writing the claim) | Window-mean bars: `dcc_scripts/decoding/figs/<E>_subjects-good/window_mean_groups/stimulus_{task,big_letter}_conditions_0to1.5s_20bts_1000perm_seed42/{Task,BigLetter}/{lpfc,occ}/group_bars.png`. Time-resolved: submitted 2026-11-10 (jobs 61504712 / 61504713; `ELECTRODES=sig`, `ROIS=lpfc,occ`, `FEATURE_MODE=channels_x_time`); results will be `*_MASTER_RESULTS_job6150471{2,3}_*.pkl` in `dcc_scripts/decoding/figs/<E>_subjects-good/` | `dcc_scripts/decoding/decoding_dcc.py`; window-mean: `submit_window_mean_groups_dcc.sh` |

**Rerun / redraw:**

```bash
# a: re-render (the launcher hard-codes PLOT_SETS; the full command is in
#    figure_editing_guide.md §11.5)
# b: set SUBJECT_ID='D0094', CHANNEL='LFPM12', SPEC_TYPE='wavelet' at the top
#    of the script (its defaults are D0057 RAI7, multitaper), then
python dcc_scripts/spec/plot_example_spectrogram_dcc.py
# c (from dcc_scripts/power):
CONDITIONS=stimulus_all_conditions ELECTRODES=sig bash submit_specific_conditions_power_traces_dcc.sh
# S, time-resolved (from dcc_scripts/decoding). FEATURE_MODE matches the F3b runs;
# the submitter's own default is now `channels`, and the runner's ELECTRODES default is `all`:
SUBJECT_SET=good ROIS=lpfc,occ ELECTRODES=sig FEATURE_MODE=channels_x_time \
  CONDITIONS="stimulus_task_conditions stimulus_big_letter_conditions" \
  bash submit_specific_conditions_decoding_dcc.sh
# S, window-mean bars: figure_editing_guide.md §11.1 and §11.6 (ROIS=occ)
```

### 1.4 F3 — Both adaptations in power and in decoding

**Open decision:** the main figure shows either the **subtractions** (the
per-block effect, i − c or s − r, one line per block) or the **four-trace**
panels (each condition in each block). The other version goes to the
supplement. Both exist for every panel.

| Panel | Content | Figure (`dcc_scripts/vis/paper_panels/`) |
|---|---|---|
| a | Power traces, task-significant lPFC, mean ± SEM across electrodes, with the cluster-corrected interaction bar. LWPC: `fig3_congruency_x_incprop_LWPC`; LWPS: `fig3_switch_x_switchprop_LWPS` | four-trace: `fig3_*`; subtraction: `fig3_*_difference` |
| b | Decoding, task-significant lPFC: each condition decoded within the 25 % and 75 % blocks, the pooled shuffle, the 25 % vs 75 % contrast and each block against chance | `fig4_LWPC`, `fig4_LWPS`; difference: `fig4_*_difference` |
| c | **Cross-effects, pooled two-way (in the paper; placement open; decided 2026-11-10).** Congruency × switch proportion and switch type × incongruent proportion, in power and in decoding. Shown without a claim of independence: the text must not say the cross-effects are absent | power: `fig3_congruency_x_switchprop_cross`, `fig3_switch_x_incprop_cross`; decoding: `sfig4_*` |
| S | Decoding accuracy by electrode count, in sig, non-sig and all lPFC electrodes, for congruency and switch type **and for task and big letter**. Motivates using all electrodes in F4 | `dcc_scripts/decoding/figs/<E>_subjects-good/window_mean_groups/<run>/<comparison>/lpfc/electrode_curve.png`, with `group_bars.png` beside it (matched counts, membership permutation tests). Runs: [`figure_editing_guide.md`](figure_editing_guide.md) §11.1 |

No independence claim is made from the cross-effects. That drops the
"behavior-independent, neurally interacting" dissociation that
[`analysis_guide.md`](analysis_guide.md) §12 calls the headline result, so the
Results and Discussion text should not rely on it.

Every panel is the same size, with label-less variants for assembly
(`utils/panel_layout.py`). The inputs and every setting are recorded in
`paper_panels/paper_panels_config.json`.

- **Power inputs:** the windowed-ANOVA runs in
  `dcc_scripts/power/figs/<E>/anova_within_roi_subjects-good` (good set). Each
  term comes from its own pooled 2 × 2 run (`PAPER_DESIGN = 'two_way'` in
  `paper_panels.py`; jobs 61512114–20):
  - `stimulus_lwpc_conditions`
  - `stimulus_lwps_conditions`
  - `stimulus_congruency_by_switch_proportion_conditions`
  - `stimulus_switch_type_by_incongruent_proportion_conditions`

  Each cell holds every trial of that condition at that proportion level. The
  block-balanced three-way and the four-way versions are not in the paper;
  `--power-design three_way` or `four_way` draws them to
  `dcc_scripts/vis/paper_panels_<design>/`. The three designs compared:
  [`analysis_plans.md` › Power-trace ANOVA design](analysis_plans.md#power-trace-anova-design-two-way-three-way-or-four-way-2026-11-10).
- **Decoding inputs:** the good-set results files, jobs 60508450–53
  (`dcc_scripts/decoding/figs/<E>_subjects-good/`), set by `DECODING_JOBS` in
  `paper_panels.py`. They have been in use since the 2026-11-10 redraw, so F3a
  and F3b use the same participants.

**Redraw:** `python dcc_scripts/vis/paper_panels.py --subject-set good`.
**Rerun:**
- **Power:** from `dcc_scripts/power`, run with
  `CONDITIONS="stimulus_lwpc_conditions stimulus_lwps_conditions stimulus_congruency_by_switch_proportion_conditions stimulus_switch_type_by_incongruent_proportion_conditions"`
  (jobs 61512114–20).
- **Decoding:** from `dcc_scripts/decoding`, run the four block-balanced
  labels.
- `dcc_scripts/submit_subject_set_rerun.sh` submits both: the four pooled
  power sets and the four block-balanced decoding labels.
The electrode-count curves: [`figure_editing_guide.md`](figure_editing_guide.md) §11.1.
How the decoding job works: [`decoding.md`](decoding.md).

### 1.5 F4 — One population, partly shared codes, adapting to block proportion

**The argument (decided 2026-11-12).** The anatomy is shown on the
adaptations: LWPC and LWPS share electrodes (a). The text then shows that this
overlap goes with the base effects: it is not detectable once each half's
congruency and task-seq. effects are partialled out, while responsiveness,
location and RT coupling leave it in place; and each adaptation tracks its
own base effect more than the other (matched − crossed). That is the bridge to
b–d, which look at the base codes themselves. Wording: "not detectable beyond
the base effects" or "largely accounted for", not "goes away" (the partialled
r is a null, p = 0.19). Caveat for the Discussion: the base effects are also
the best proxy for an electrode's signal-to-noise; the matched > crossed
pattern argues against pure signal-to-noise, and the sharper control (each
adaptation partialled on its own base effect only, N4 §19 B) has not been run.

**Controls: text, not a figure (decided 2026-11-12).** One sentence in §3.4
with the good-set numbers (`overlap_controls.csv`, all lPFC; all24 values in
brackets for reference): none 0.106 (0.097); + responsiveness 0.103 (0.094);
+ MNI 0.109 (0.100); + RT coupling 0.091 (0.082); + congruency and task-seq.
effects 0.031, p = 0.19 (0.028, p = 0.23). Matched − crossed r = 0.082,
p = 0.002 (0.079 with MNI). In S-N4: the control table **or** the forest plot
(`continuous/overlap_controls.png`), not both; and the matched-vs-crossed bars.

**Electrode set: all lPFC (decided 2026-11-12).** The anatomy needs it: in the
task-significant subset (148 electrodes, 16 participants) the overlap is
0.073, p = 0.086 and matched − crossed 0.061, p = 0.16, at about 60–75 % of
the all-lPFC estimates on 40 % of the electrodes (lost power, not a different
result), so the pivot cannot be shown there. The coding-axes results hold in
the subset (alignment 0.63 [0.49, 0.85]; switch-axis block effect p = 0.006).
Report the subset in S-N4 and the coding-axes supplement. F3 uses
task-significant electrodes and F4 all lPFC: Methods needs one sentence on why
(the pre-specified anatomy population, and the electrode-count curves in S-F3).

| Panel | Content | Figure | Code |
|---|---|---|---|
| a | Each electrode's LWPC score against its LWPS score (full-data scores, SD units). No identity line, no fit line (the points' own correlation, 0.23, is not the test's 0.106). Only "\*p = 0.0001", small, above the top-right corner; the separate-half Spearman *r* goes in the text (decided 2026-11-12) | `dcc_scripts/stats/results/<E>/anatomy_a1_lpfc_window_0.0to1.5s_all_subjects-good/continuous/section19_paper_2026-10-30/fig5_height_scatter.png` / `.pdf` (370 electrodes, 19 participants) | `plot_figure5_scatter` in `src/analysis/stats/stability_flexibility_anatomy.py`; launcher `dcc_scripts/stats/fig5_scatter.py`; runbook [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) §19 |
| b | Cosine between the congruency and task-seq. coding axes over time, against the shuffle null and the split-half ceiling, with error bands | `coding_axis_cosines` | `plot_coding_axes` |
| c | Trajectories of the four condition means (c/i × r/s) in the congruency–task-seq. plane, with x(t) and y(t) | `trajectories_0.0-1.5s` | `plot_trajectories` |
| d | The same plane per proportion block, and the block separations along each axis | `proportion_trajectories_0.0-1.5s`, `block_separations_0.0-1.5s` | `plot_proportion_trajectories`, `plot_block_separations` |
| S | The subspace test (principal angles, alignment index); also the alignment over time (`coding_axis_cosines_alignment`) and the 25 % − 75 % separation changes (`block_separations_0.0-1.5s_changes`, statistics in `block_contrasts.csv`) | `subspace_test` | `plot_subspace_test` |
| S-N4 | The congruency–task-seq. scatter (r = 0.237, same style as a; supplement rather than main, since b shows the shared base codes more directly and over time, and its r is quoted in the text; dropping it is also fine): `continuous/fig5_height_main_effects_scatter.png` / `.pdf`. The overlap controls (table or forest plot) and the matched-vs-crossed bars (`continuous/fig5.png` panel b, old colours). The task-significant versions | as named | `fig5_scatter.py --pair main_effects`; `sfa.overlap_controls`, `sfa.figure5` |

b–d and S are PNG + PDF in
`dcc_scripts/subspace/results/<E>/coding_axes/windows_0.0to1.5s_w64s16_all_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good/figures_2026-11-11/`.
Don't use `figures_2026-11-09/`: its task-seq. block separations are empty (a
label-rename bug, fixed 2026-11-11). The PNGs one folder up are the old
2026-10-30 previews.

**Open, to check before writing d.** For congruency by incongruent
proportion, all lPFC, the bootstrap CI excludes 0 ([0.06, 0.56]) but the
block-permutation p is 0.13 (`block_summary.json`). Decide which test the claim
rests on. The plotting functions
are in `dcc_scripts/subspace/subspace_figures.py`. The analysis is
`src/analysis/stats/coding_axes.py`, `coding_subspaces.py` and
`coding_trajectories.py`; its runbook and results are in [`coding_axes.md`](coding_axes.md),
and the math in [`walkthroughs/subspace_math_walkthrough.md`](walkthroughs/subspace_math_walkthrough.md).
Occipital cortex, run the same way, has no reliable code
([`figure_editing_guide.md`](figure_editing_guide.md) §11.6).

```bash
# a, redraw (--seg-dir supplies the p; the main-effects points are in continuous/):
S=dcc_scripts/stats/results/<E>/segregation_results/window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh_main_effects_subjects-good
python dcc_scripts/stats/fig5_scatter.py --dir <the section19_paper_2026-10-30 folder> --seg-dir $S
python dcc_scripts/stats/fig5_scatter.py --dir <the anatomy run>/continuous --pair main_effects --seg-dir $S
# b-d and S, redraw:
R=dcc_scripts/subspace/results/<E>/coding_axes/windows_0.0to1.5s_w64s16_all_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good
python dcc_scripts/subspace/redraw_figures.py --run-dir $R --out-dir $R/figures_<date>
# b-d and S, rerun (from dcc_scripts/subspace, about 40 min):
RT_CONTROL=none N_BLOCK_PERM=500 SUBJECT_SET=good ROIS=lpfc,dlpfc bash submit_coding_axes_dcc.sh
```

### 1.6 Not in this story

These analyses are out of the main text. Their docs stay as the record:
- the height gradient and the parcel tests ([`n4_continuous_anatomy.md`](n4_continuous_anatomy.md));
- the A4 congruency ↔ switch cross-decoding ([`decoding.md`](decoding.md));
- brain–behavior ([`archive/a6_brain_behavior.md`](archive/a6_brain_behavior.md));
- the N2 direction tests ([`archive/n2_direction_tests.md`](archive/n2_direction_tests.md));
- DLPFC as the reported region ([`archive/dlpfc_results.md`](archive/dlpfc_results.md)).

Any of them can come back as a supplement or for reviewers.

---

## 2. Methods

### 2.1 Participants, recordings and preprocessing

*Copied from [`archive/methods.md` › N4](archive/methods.md#participants-recordings-and-electrode-population), first paragraph.*

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
[`archive/n2_direction_tests.md`](archive/n2_direction_tests.md) §1–§4. Check every parameter
against the reported run.*

> We analysed **[n]** task-responsive lPFC electrodes from **[N]**
> participants. Each condition effect was tested against each proportion
> manipulation in a separate 2 × 2 design: congruency × incongruent proportion,
> switch type × switch proportion, congruency × switch proportion and switch
> type × incongruent proportion. For each electrode, high gamma on correct
> trials was averaged over all trials in each condition × proportion cell. The
> traces show the mean across electrodes ± 1 SEM.
>
> Each 2 × 2 was tested with a two-way ANOVA fitted across electrodes (ordinary
> least squares) in sliding 250-ms windows (64 samples at 256 Hz, stepped by
> 62.5 ms). Electrode was not a term in the model, so between-electrode
> variance stays in the residual; the within-electrode permutation below keeps
> the test valid.
>
> **[Author note, not paper text. Three things to settle. (1) In the two
> cross-effect designs, each cell mixes the blocks of the other proportion
> unequally. For example, at each switch-proportion level, congruent trials
> come about 3:1 from the 25 % incongruent block and incongruent trials about
> 1:3 (`condition_registry.py`, the comment on
> `stimulus_lwpc_block_balanced_conditions`). (2) The behavioral analysis (F1)
> weights all 16 cells equally, so the neural and behavioral cross terms are
> defined differently (`analysis_plans.md` › Power-trace ANOVA design §2).
> Decide how Methods states both. (3) Do not describe the cross-effects as
> absent. LWPC and LWPS hold in the pooled, block-balanced and full-factorial
> designs alike; the cross-effects depend on the design (same section, §6).]** The null distribution permuted cell labels within each
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
> analysis, §2.8) and recomputed LWPC and LWPS. We averaged each participant's
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
It replaces the bracketed N4 template in `archive/methods.md` for the all-lPFC analysis.*

*Placement decided 2026-11-10: the anatomy on RT-adjusted scores and both
confound-control tables (LWPC–LWPS, `overlap_controls.csv`; congruency–switch,
`overlap_controls_main_effects.csv`) go to S-N4. Good-set numbers, all lPFC and
task-significant: [`results_log.md`](results_log.md), 2026-11-10.*

*2026-11-12: counts and settings updated to the good set (anatomy run
`anatomy_a1_lpfc_window_0.0to1.5s_all_subjects-good`, its `summary.txt` and
`score_anatomy.json`; segregation run `..._all_lpfc_..._main_effects_subjects-good`,
200 splits).*

> **Electrode population.** Anatomical analyses used every electrode in lateral
> prefrontal cortex (lPFC; 371 electrodes, 238 left and 133 right, from 20
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
> halves 200 times, stratified on congruency, incongruent proportion, switch
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
> (370 electrodes, 19 participants remained). The null permuted electrode
> correspondence within participant, identically across splits. Electrodes
> positive for each adaptation effect were compared in location by the distance
> between their centroids, with electrode labels shuffled within participant.
>
> **Anatomical test.** Each balance was modelled as
> balance = parcel + responsiveness + participant, with participants as fixed
> effects. The statistic was the partial *F* for the parcel block. Destrieux
> parcels sampled in at least three participants entered (19 parcels, 370
> electrodes, 20 participants). The null exchanged the two effect labels independently within
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
- On the good set both electrode sets were scored on 200 splits, so the old
  "200 rather than 1,000" sentence for the subset is no longer needed. The
  subset's label test used the 13 parcels sampled in at least three of its
  participants (141 electrodes, 19 participants). Add that to the
  supplementary methods.

✏️ **Units of inference, intervals, and local similarity** (skeleton,
2026-10-02; electrode level first since 2026-10-06; §19 of the N4 doc and
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#1910-19-with-electrodes-as-the-unit)). Add
after the coordinate test. The participant-level paragraph can move to the
supplementary methods with S-N4.

> **Units of inference.** The tests above treat participant as a fixed effect
> and electrodes as the units of inference. Intervals at that level came from
> an electrode bootstrap within participant: each participant kept its number
> of electrodes, its electrodes were drawn with replacement, and the statistic
> was recomputed (2,000 draws; for the separate-half correlations, on 100 of
> the splits).
>
> **Participant-level tests (Supplementary).** We repeated the two main tests
> with participants as the units. For the overlap, each participant's
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
> electrode's other half (200 splits). Each participant's baseline for a
> bin was its mean over shuffles of the participant's electrode positions, and
> the excess over it was pooled over all electrode pairs. Because neighbouring
> contacts share trial noise, pairs are not exchangeable, and a test that
> shuffles positions is too liberal. We therefore tested the excess with
> electrodes as the units by its standard error: every estimate is a quadratic
> form in the electrodes' scores, whose covariance under no local structure we
> modelled as exchangeable signal plus trial noise, with the noise covariance
> between each pair of electrodes estimated from the differences between their
> trial halves (one-sided normal *p*, intervals ± 1.96 SE; the per-split
> standardisation carried through by the delta method). On simulated data with
> noise shared between neighbouring contacts and no local structure, this test
> rejected at 4.0–4.7 % at α = 0.05 for reliabilities like ours. With
> participants as the units (Supplementary), we flipped the sign of each
> participant's excess, with intervals from a participant bootstrap. The same
> analysis of LWPC, LWPS, congruency and switch type alone served as the
> positive control. An electrode's similarity with itself across halves is its
> split-half reliability, against which the neighbours' similarity is
> expressed where that reliability is clearly positive.
>
> **Alternative explanations of the overlap.** We repeated the overlap test
> with, in turn, log and squared responsiveness, MNI coordinates, each half's
> congruency and switch-type effects (from the same half as each adaptation
> score), each electrode's within-cell correlation between high gamma and RT,
> and all of these together, as additional covariates at the residualisation
> step (10,000 permutations each), and with each participant left out
> (1,000 permutations per fold).

### 2.6 Coding axes, subspace test and trajectories (F5 d–e, S5) ✏️

*Drafted 2026-10-16 from [`coding_axes.md`](coding_axes.md) §1, §4 and §9 and
[`analysis_plans.md` › Subspace trajectory plan](analysis_plans.md#subspace-trajectory-plan)
§3–4. A skeleton: check every parameter against the run you report. Inserted
here, so the cross-decoding and brain–behavior sections are now §2.7 and
§2.8.*

> **Coding axes.** For every lateral prefrontal electrode (371 electrodes,
> 20 participants; the anatomy set; correct trials only) we computed the
> congruency effect (mean high gamma on incongruent minus congruent trials,
> the two incongruent-proportion levels weighted equally) and the switch-type
> effect (switch minus repeat, the two switch-proportion levels weighted
> equally) in 250-ms windows stepped by 62.5 ms over 0–1.5 s after stimulus
> onset (21 windows). Effects were mean differences in high-gamma units, not
> standardized, so that electrodes were not reweighted by their own
> variability. An electrode's trial was dropped if any sample in 0–1.5 s was
> missing. Across electrodes, pooled over participants, each variable's
> effects in a window form a coding axis.
>
> **Split halves, ceiling and alignment.** We split each participant's trials
> into halves, stratified on congruency, switch type and both proportions,
> 200 times. One split per participant was shared by all of its electrodes, so
> no trial contributed to both halves of any pair of electrodes. In each window
> we computed the cosine between the congruency axis from one half and the
> switch-type axis from the other, averaged over both pairings (cross), and the
> cosine between the two halves of each variable (its ceiling). Alignment was
> the cross cosine divided by the geometric mean of the two ceilings: 0 for
> separate codes, 1 for codes as shared as noise allows. Windows were
> interpreted only where both ceilings' 95 % CIs excluded 0.
>
> **Tests.** To test whether anything was shared, we shuffled the congruency
> labels within each participant's switch type × proportion cells (one
> permutation per participant, applied to all of its electrodes), rescored the
> data and recorded the cross cosine, and did the same for switch type
> (**[1,000]** permutations per factor, each averaged over **[20]** splits). A
> window was shared where the cross cosine exceeded both nulls, cluster-mass
> corrected across windows (Maris & Oostenveld, 2007). To test whether the two
> codes were one, we resampled participants with replacement (**[2,000]**
> resamples) and marked the windows where the 95 % CI of ceiling − cross
> excluded 0 (pointwise).
>
> **Subspace test.** On each half we stacked each variable's per-window axes
> into an electrode × window matrix and took its leading left singular vectors
> (uncentred SVD). Each variable's number of dimensions k was the number of
> leading dimensions, tested in order, whose held-out variance exceeded the
> shuffle null; the primary k was the smaller of the two. We report the
> smallest principal angle θ₁ between the congruency and switch-type subspaces
> (from different halves) and the alignment index in both directions (Elsayed
> et al., 2016), each against the same shuffles, as a fraction of its
> split-half ceiling, and with participant-bootstrap CIs. The ceiling
> normalization is exact at k = 1 and approximate above it, so raw values are
> given with their ceilings.
>
> **Trajectories.** On one half we averaged each axis over 0–1.5 s and made the
> pair orthonormal with congruency first: x, the congruency axis; y, the part
> of the switch-type axis orthogonal to it. On the other half we averaged each
> electrode's high gamma in the four congruency × switch-type cells in every
> window, each cell the equal-weight mean of its four proportion blocks,
> subtracted the mean of the four cells, and projected each cell onto x and y.
> We swapped the halves, averaged over both directions and the 200 splits, and
> obtained CIs by participant bootstrap. Electrodes without a trial in every
> congruency × switch type × block cell of each half were left out of the
> trajectories (362 of 371 electrodes, 19 participants). The
> fraction of the centred cells' reproducible variance that lies in the plane
> was computed from cross-half products, which removes independent noise.
>
> **Controls.** We repeated every analysis (i) after subtracting, on every
> trial and time point, the mean across each participant's electrodes; (ii)
> after removing the RT-linked component of high gamma by regression within
> design cells **[check: the same cells as §2.8]**; and (iii) on trial subsets
> in which RT was matched across the four congruency × switch-type cells within
> each participant (10 RT quantile bins), against random subsets of the same
> size, five draws of each. **[The response-locked analysis and the occipital
> comparison go here once run.]**

### 2.7 Congruency ↔ switch-type cross-decoding (S5)

*Copied from [`archive/methods.md` › A4](archive/methods.md#a4-congruency--switch-type-cross-decoding), Methods.*

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

### 2.8 Brain–behavior (S-BB)

*Copied from [`archive/methods.md` › A6](archive/methods.md#a6-brainbehavior-supplement-s-bb).*

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

*2026-10-16: F5 d–e are now the coding axes (§1.4), and their paragraph is the
last one below, drafted from [`coding_axes.md`](coding_axes.md) §10.2–10.5 and
§10.7. A4 (§3.6) stays in S5. The gradient paragraph points to Fig. 5a–c, the
layout since 2026-10-10.*

*2026-11-12: rewritten on the good set and for the 2026-11-12 argument (§1.5):
overlap, then "it goes with the base effects", then the base codes. Sources:
the all-lPFC anatomy run's `continuous/summary.txt`, `score_anatomy.json`,
`overlap_controls.csv`, `section19_controls_2026-11-10/overlap_loso.csv` and
`scores_with_anatomy.csv` (means); the sig anatomy run's `summary.txt`; the
coding-axes runs' `summary.json`, `subspace_summary.json`,
`subspace_dimensions.csv`, `trajectory_summary.json` and `trajectories.csv`
(window 14, 1.0 s). The gradient paragraph moved to §3.5 (S-N4), as §1.6
decided. Figure letters are the new F4's. The prose still says "switch type";
the figures say "task seq.", so pick one name for the paper.*

> **LWPC and LWPS share electrodes.** We scored every lPFC electrode (371
> electrodes, 20 participants) for LWPC and LWPS. Across this anatomically
> defined set both adaptation effects were small on average (mean Cohen's
> *d* = 0.00 for LWPC and 0.03 for LWPS; 47 % and 55 % of electrodes
> positive), and single-electrode estimates were noisy (full-data split-half
> reliability 0.34 for LWPC and 0.09 for LWPS), so we tested only
> population-level summaries. LWPC and LWPS scores from separate halves of the
> trials were positively correlated within participants (Spearman *r* = 0.11,
> 95 % CI [0.06, 0.16], *p* < 0.001; 370 electrodes, 19 participants;
> Fig. 4a), and electrodes positive for each effect did not differ in location
> (centroid distance 3.0 mm, *p* = 0.80). The overlap did not depend on
> nonlinear responsiveness (*r* = 0.10), on location (MNI coordinates
> partialled, *r* = 0.11), on each electrode's coupling to RT (*r* = 0.09) or
> on any one participant (*r* = 0.09–0.12 with each left out, all
> *p* ≤ 0.003). It was not detectable, however, once each half's congruency
> and switch-type effects were partialled out (*r* = 0.03, *p* = 0.19;
> Supplementary **[S-N4]**).
>
> **Each adaptation tracks the effect it regulates.** The base effects that the
> two adaptations act on, congruency and switch type, were scored from the same
> trials and halves with the proportion blocks weighted equally. They shared
> electrodes more strongly than the adaptations did (*r* = 0.24, *p* < 0.001),
> and each adaptation effect correlated more with its own base effect than with
> the other one (congruency–LWPC *r* = 0.23 vs switch–LWPC 0.13; switch–LWPS
> 0.16 vs congruency–LWPS 0.13; Supplementary **[S-N4]**). We tested this
> difference with the correlation between congruency − switch and
> LWPC − LWPS, whose covariance equals the two matched covariances minus the
> two crossed ones, so that anything common to all four scores cancels. It was
> positive (*r* = 0.08, *p* = 0.002) and held with MNI coordinates partialled
> out (*r* = 0.08, *p* = 0.002), so the link is local rather than a gradient
> that the two maps share. For LWPS the matched base effect was the less
> reliable map (within-participant split-half reliability 0.34 [0.25, 0.41]
> for switch type, 0.48 [0.41, 0.53] for congruency, from trial splits shared
> by all of a participant's electrodes) and still correlated more, so that
> difference is not one of reliability **[CI of the reliability difference,
> run]**. Because the base effects are also the best available proxy for an
> electrode's signal-to-noise ratio, partialling them cannot by itself separate
> "inherited from the base effects" from "shared signal quality"; the
> process-specific matched > crossed pattern argues for the former.
>
> **[Local similarity, draft:]** Electrodes leaning toward LWPC or toward LWPS
> did not form local patches: their balance (LWPC − LWPS) showed no excess
> similarity between nearby electrodes (nearest minus farthest distance bin
> −0.01, *p* = 0.55), whereas LWPC, congruency and switch type alone did
> (0.17–0.23, *p* ≤ 0.003). The balance itself had no measurable reliability
> (−0.06 [−0.24, 0.13]), so this is an absence of evidence for patches rather
> than evidence that the two are intermixed.
>
> The task-responsive subset (152 electrodes, 19 participants; 148 and 16 in
> the overlap tests) pointed the same way at lower power: the point estimates
> were about 60–75 % of those in all lPFC on 40 % of the electrodes. The base
> effects shared electrodes (*r* = 0.15, *p* = 0.006), whereas the adaptation
> overlap (*r* = 0.07, *p* = 0.086) and the matched-minus-crossed test
> (*r* = 0.06, *p* = 0.16) did not reach significance (Supplementary
> **[S-N4]**).
>
> Together, LWPC and LWPS share lPFC electrodes because each is expressed where
> the effect it regulates is, and those base effects share electrodes. We
> therefore asked how the base effects themselves are coded across the
> population.
>
> **Congruency and switch type are coded by partly aligned population
> patterns.** We treated each electrode's congruency and switch-type effects in
> every 250-ms window as two population coding axes and compared them across
> separate halves of the trials, against a label-shuffle null and the
> split-half ceiling that noise sets (all lPFC, 371 electrodes,
> 20 participants; Methods §2.6). Both codes were reliable from about 0.5 s
> after the stimulus (15 of 21 windows, 0.50–1.38 s). From 0.56 s on, the
> congruency axis from one half was more aligned with the switch-type axis from
> the other than either shuffle allowed (cluster-mass corrected), at about 0.7
> of the most that noise allows (cross-half cosine 0.27 against a ceiling of
> 0.36; alignment 0.72, 95 % participant-bootstrap CI [0.59, 0.97]; each
> averaged over the reliable windows; Fig. 4b).
> Over the whole trial each code was dominated by one pattern (the first
> dimension held 13–23 % of the held-out variance, the next two about 1 % each),
> and the smallest principal angle between the two codes' subspaces was 63°
> (cosine 0.45 against a ceiling of 0.60, or 0.75 [0.67, 0.94] of it): more
> aligned than either shuffle allowed (*p* = 0.001), and below its ceiling. On
> held-out trials, the switch conditions moved along the congruency axis by
> about half as much as the congruency conditions did (0.86 against 1.65 at
> 1.0 s), and the plane of the two axes held about half of the reproducible
> condition-dependent activity (0.53 [0.39, 0.64]; 362 electrodes,
> 19 participants; Fig. 4c). Subtracting each trial's mean across a
> participant's electrodes left the alignment unchanged (0.72 [0.55, 1.00]), so
> it is not a uniform rise in activity. When RT was equated across the four
> cells, the codes themselves largely disappeared (0–4 of 21 windows reliable,
> against 8–13 in random subsets of the same size; five draws each), so the
> late codes and their alignment are tied to RT. Removing the RT-linked
> component by regression instead left the alignment in place (0.61 [0.35,
> 1.00]; Supplementary **[S5]**). The task-responsive subset gave the same
> alignment (0.63 [0.49, 0.85]; 152 electrodes, 19 participants).
> **[Response-locked result, once run (§1.7).]**
>
> **[F4d paragraph, to write: the block separations. All lPFC, 0–1.5 s axes
> (`block_summary.json`): the switch-type separation was larger in 25 % than
> in 75 % switch blocks (0.41, 95 % CI [0.11, 0.70], *p* = 0.020; own minus
> other proportion 0.52 [0.22, 0.85], *p* = 0.06); the congruency separation
> in 25 % than in 75 % incongruent blocks, 0.32 [0.06, 0.56], *p* = 0.13.
> Decide whether the claim rests on the bootstrap CI or the block-permutation
> *p* (§1.5) before writing. Task-responsive subset: switch 0.41, *p* = 0.006;
> own minus other *p* = 0.008.]**

### 3.5 Anatomy supplement (S-N4)

*The first paragraph is copied from §18.8 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#188-results-text-supplement-s-n4),
the task-significant run with main effects (2026-10-01); §18.9 there gives
each number's source. "Participants as the unit" is the §3.4 text that left the
main text on 2026-10-06, when the main text settled on electrodes as the unit
([`n4_continuous_anatomy.md`](n4_continuous_anatomy.md#1910-19-with-electrodes-as-the-unit)). The
next two paragraphs ("The adaptation gradient", "The
base-effect balance") are the §17.3 text that left the main text on 2026-10-01
(§3.4 above). The others are written here from the numbers in §16.6.6 and the
§17.3 supplement additions of that doc.*

*2026-11-12: the paragraph below left §3.4 (the gradient is out of the main
text, §1.6) and is on the good set (all-lPFC anatomy run, `summary.txt`;
height share from `section15/summary_section15.txt`). The rest of this section
is still all24 and still needs the same update.*

> **The balance between the two adaptations varies modestly across lPFC.** The
> balance between the two effects (LWPC − LWPS within each electrode) differed
> across Destrieux parcels (19 parcels sampled in at least three participants,
> 370 electrodes; *F* = 1.69, label-exchange permutation *p* = 0.047;
> *p* = 0.017–0.097 with each participant left out), although no single parcel
> differed from zero after FDR correction (all *q* ≥ 0.17). It also varied
> with position (MNI coordinates: block *F* = 3.32, *p* = 0.018): relative to
> LWPS, LWPC was weaker dorsally (*z* slope −0.0087 SD/mm, 95 % CI [−0.0140,
> −0.0039] from an electrode bootstrap within participant, *p* = 0.003;
> Bonferroni-corrected over three axes, *p* = 0.008). The predicted
> anterior–posterior axis showed no effect (*p* = 0.44), and height explained
> about 2 % of the balance's variance within participants. In the
> task-responsive subset neither the parcel test (*F* = 0.74, *p* = 0.42;
> 13 parcels, 141 electrodes) nor the coordinate test (*F* = 1.34, *p* = 0.34)
> was significant, with a *z* slope of similar size (−0.0080 SD/mm,
> *p* = 0.22).

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
> **Participants as the unit.** *(Moved from the main text on 2026-10-06; N4
> §19.8.)* The main-text tests take electrodes as the units, with participant
> as a fixed effect. With participants as the units, the LWPC–LWPS overlap held
> when each participant's own correlation was weighted by its electrode count
> (*r* = 0.10, 95 % CI [0.03, 0.16], sign-flip *p* = 0.031; 20 participants
> with at least four electrodes) but not when every participant counted
> equally (*r* = 0.04, *p* = 0.35; 11 of 20 positive), so it describes the
> electrode population rather than every participant. The height gradient
> held, though less strongly and unevenly across participants: the
> participants' own slopes, weighted by their electrodes' spread in height,
> averaged −0.0077 SD/mm (95 % CI −0.0147 to −0.0013; sign-flip *p* = 0.041),
> and a mixed model with a random height slope gave −0.0083 SD/mm
> (*p* = 0.037; between-participant SD of the slope 0.010 SD/mm). With each
> participant left out, *p* ranged from 0.002 to 0.099. **[Local similarity
> with participant sign flips: N4 §19.8 table, raw high gamma.]**
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

*Copied from [`archive/methods.md` › A4](archive/methods.md#results-draft), Results (draft). It
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
> The two contrasts thus engage the same electrodes along partly shared
> population codes, sharing a component that appears only late in the trial.
>
> *(2026-10-16: "largely distinct" changed to "partly shared" here; the
> coding axes put the noise-corrected alignment at about 0.7 (§3.4).
> [`archive/methods.md`](archive/methods.md) and [`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01)
> still have the old wording. The RT sentence for the main text is now §3.4's;
> the bracketed RT choice below is for S5, once A4's own seeds are in.)*
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

*Copied from §14.2 of [`archive/a6_brain_behavior.md`](archive/a6_brain_behavior.md#142-results-supplement-s-bb).*

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

**Base-effect codes (F5 d–e, S5).** *Rewritten 2026-10-16 from the coding
axes ([`coding_axes.md`](coding_axes.md) §10.7). The 2026-10-01 version, from
[`decoding.md` › A4 §13.8.4](decoding.md#138-results-2026-10-01), said "largely
distinct population codes", which the noise-corrected alignment does not
support. It goes next to the N4 overlap result.*

> The same lPFC electrodes carried both base effects, congruency and switch
> type, each in essentially one population pattern. The two patterns were
> partly aligned, at about 0.7 of what noise allows, and the codes, aligned
> part included, were tied to RT (Fig. 5d, e; Supplementary S5). **[After the
> response-locked run: whether they persist before the response.]**

**Geometry limits to state** ([`coding_axes.md`](coding_axes.md) §10.8):

- No mean-difference code is reliable before 0.5 s, so the geometry says
  nothing about early separation.
- RT matching removes the codes; regression does not. Matching selects on the
  outcome, so it cannot tell response timing from a process that produces the
  slowing.
- The c–s plane holds about half of the reproducible condition-dependent
  activity.
- The occipital comparison has not been run, so nothing is lPFC-specific.

**Brain–behavior (S-BB).** *Copied from §14.3 of
[`archive/a6_brain_behavior.md`](archive/a6_brain_behavior.md#143-discussion-sentence).*

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
  the two base-effect maps are equally reliable (0.39 and 0.39 within
  participant, shared splits) and LWPC's gap is the same size (0.09 against 0.10), which
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

*2026-11-12: draft for the new F4a, the scatter alone (§1.5). It replaces
panel b of the old caption below; the old caption's panels a and c are out of
the main text, and d–e (now F4b–c) need the "task seq." name and the
2026-11-09 restyle (shuffle band, labelled bars, no grey background, Wong
colours). The old caption's counts are updated to the good set (371 / 20;
370 / 19 in the overlap; 362 / 19 in the trajectories; 200 splits).*

> **a**, LWPC against LWPS in every lPFC electrode (370 electrodes,
> 19 participants). Each point is one electrode's two scores from all trials
> (Cohen's *d*, responsiveness regressed out and each participant's mean
> subtracted, overall mean added back; SD units). The test (*p*, top right) is
> pre-specified: the Spearman correlation of one score from one half of the
> trials with the other score from the other half, averaged over 200 random
> splits, against a within-participant permutation null; *r* = 0.11 (text).
> Because the two full-data scores share trials and their noise, the points
> correlate more strongly (*r* = 0.23) than the test, so no line is drawn.

*Redrafted 2026-10-16 for the layout in §1.4: a–c are the anatomy as laid out
since 2026-10-10 (rendered brain, scatter with the centroid zoom, the gradient
in the test's units), and d–e are the coding axes, which replaced the A4
cross-decoding. The title follows §1.1's 2026-10-16 candidate: "intermixed" is
out (local similarity had no power at these reliabilities), and the RT clause
waits for the response-locked run. Numbers in bold brackets come from the §19
run or from the run named. The 2026-10-02 caption is in git history; the
2026-10-01 caption (overlap at both levels, and the bars) is kept below for
S-N4.*

> **Fig. 5 | LWPC and LWPS adaptation share lPFC electrodes, with a shallow
> dorsal–ventral bias in their balance; over the same electrodes, congruency
> and switch type are coded by partly aligned population patterns.** All lPFC
> electrodes (371 electrodes, 20 participants). **a**, Electrodes on the MNI
> template brain, coloured by tertile of height (MNI *z*; cuts at **[…]** and
> **[…]** mm, drawn as lines); each tertile's centroid in each hemisphere is
> drawn larger and darker. **b**, LWPC against LWPS: each point is one
> electrode's score from all trials, after regressing out overall
> responsiveness and subtracting each participant's mean, with each score's
> overall mean added back (370 electrodes from the 19 participants with at
> least three; SD units). Colour as in **a**; large markers, tertile centroids.
> Points above the identity line lean toward LWPS, points below it toward
> LWPC. *r* is the pre-specified test: one score from one half of the trials
> against the other score from the other half, averaged over 200 random
> splits (Spearman; within-participant permutation null). Because each half
> has half the trials, *r* is smaller than the correlation among the plotted
> points. Right, the centroids enlarged from the dashed box, each with its
> 95 % region from an electrode bootstrap within participant. The two effects'
> shared variation runs along the identity line; the height gradient is the
> centroids' spread across it. **c**, Each electrode's LWPC − LWPS against its
> height, with participant, responsiveness and MNI *x* and *y* removed
> (component plus residual). The line is the pre-specified coordinate model's
> *z* slope (within-electrode label-swap null), with **[its 95 %
> electrode-bootstrap interval]**; dashed lines, the tertile cuts.
> Participant-level versions are in Supplementary **[S-N4]**. **d–e** concern
> the base effects that the two adaptations act on. **d**, The four
> congruency × switch-type conditions on held-out trials (blue, incongruent;
> orange, congruent; solid, switch; dashed, repeat), each the equal-weight mean
> of its four proportion blocks with the mean of the four conditions
> subtracted, projected onto the congruency axis (*x*) and the part of the
> switch-type axis orthogonal to it (*y*). Both axes were fit on the other half
> of the trials over 0–1.5 s (362 electrodes from the 19 participants with
> every cell in each half). Open circle, first window; arrow, last. Below, *x*
> and *y* over time, mean and 95 % participant-bootstrap CI. Where the codes
> are shared, the switch conditions also move along *x*. The plane holds 0.53
> [0.39, 0.64] of the reproducible condition-dependent activity. **e**, Cosine
> between the congruency axis from one half of the trials and the switch-type
> axis from the other (cross; blue, with its Pearson version dashed) and the
> ceiling set by noise (orange), in 250-ms windows every 62.5 ms. Grey band,
> label-shuffle null (5th–95th percentile; its top edge is the threshold under
> the stricter of the two shuffles). Bars: windows where the cross cosine
> exceeds both nulls (blue; cluster-mass corrected) and where the 95 % CI of
> ceiling − cross excludes 0 (orange; pointwise). Grey background: windows
> where either ceiling is unreliable. Below, cross / ceiling with its 95 %
> participant-bootstrap CI.

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

*Copied from §14.4 of [`archive/a6_brain_behavior.md`](archive/a6_brain_behavior.md#144-figure-caption-supplementary-fig-s-bb).
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

F1, F2 and F4 captions, Supplementary S5 (the coding-axes controls and the RT
seeds table; the cross-decoding traces with ceilings, transfer and control
runs), and the rest of S-N4.
