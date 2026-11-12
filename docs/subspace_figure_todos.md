# Subspace figures (and friends): open items

Working list for the figure discussion. Started 2026-11-09. Each item says
what is open, what it depends on, and where it lives in the code; how to make
the change yourself is in [`figure_editing_guide.md`](figure_editing_guide.md)
(§4 for the subspace figures).

Figures referred to below:

- **Fig A**, "the codes are partly shared": schematic (to draw), cosine over
  time (`coding_axis_cosines`), trajectory plane with x(t), y(t)
  (`trajectories_0.0-1.5s`).
- **Fig B**, "the codes adapt to proportion": LWPC–LWPS scatter
  (`fig5_height_scatter`), proportion trajectories
  (`proportion_trajectories_0.0-1.5s`), block separations
  (`block_separations_0.0-1.5s`).
- **Supplement**: alignment (`coding_axis_cosines_alignment`), subspace test
  (`subspace_test`), separation changes (`block_separations_0.0-1.5s_changes`)
  and `block_contrasts.csv`.

The current figures are in the good-subject run's `figures_2026-11-09/`:

```
dcc_scripts/subspace/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/coding_axes/windows_0.0to1.5s_w64s16_all_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good/figures_2026-11-09/
```

The PNGs one folder up are the **old** previews (2026-10-30: blue/orange,
Pearson r line, 9-12 in wide). Don't judge the new design from those.

---

## 1. Items

Status: **done**, **next** (decided; code to be drafted and shown before
writing), **open** (needs a decision).

### 1.1 Cosine panel: shuffle trace, "cross", labelled bars, "25% I" (**done** 2026-11-09)

The shuffle is drawn as in the decoding figures (band + dotted centre line,
"shuffle" in the legend); the legend reads "shuffle, ceiling, cross"; the two
significance bars are labelled beside them ("cross > shuffle", "ceiling >
cross"; `SIG_BARS`). Redrawn in `figures_2026-11-09/`.

### 1.2 Colours for the four trajectory cells (Fig A) (**done**: black / grey, 2026-11-09)

Rendered both options (temporary preview:
`/tmp/claude-1491160/-hpc-home-jz421/285c9664-8ceb-475a-8097-1fff6b39ab8e/scratchpad/cells/compare.png`):
black / grey, and Wong reddish purple (incongruent) / bluish green (congruent).
Both with the power-trace line convention (see 1.12): switch dashed, repeat
solid. If purple / green: Fig 2's "Analyzed LPFC" (`#009E73`) and
"Task-significant LPFC" (`#CC79A7`) brain sets move to grey / black
(`dcc_scripts/vis/condition_plot_specs.py:215, 221`), as agreed 2026-11-09.

Wong colours and what they already mean:

| Wong colour | Hex | Already used for |
|---|---|---|
| black | `#000000` | cross line; cells (now) |
| orange | `#E69F00` | close to the 25 % I salmon `#FF7E79` |
| vermillion | `#D55E00` | close to the 75 % I dark red `#A32319`; LWPS-only orange |
| sky blue | `#56B4E9` | close to the 25 % S sky blue `#05B0F0` |
| blue | `#0072B2` | close to the 75 % S navy `#0B4F8A`; LWPC-only blue |
| bluish green | `#009E73` | Fig 2 analyzed LPFC electrodes (movable) |
| reddish purple | `#CC79A7` | Fig 2 task-significant LPFC electrodes (movable) |
| yellow | `#F0E442` | too pale for lines on white |

### 1.3 Both significance bars (**done** in 1.1; correction in 1.6)

### 1.4 Proportion trajectories: back to colour + line style, four legend entries (**done** 2026-11-09)

As the power traces: colour = block (25 % / 75 %), line style = level, with
i and s dashed, c and r solid (`plotting_parameters.py`: `Stimulus_i25` '--',
`Stimulus_c25` '-', ...); all four in the legend ("i, 25% I", "c, 25% I", ...);
no letters at the path ends. Note: the power traces use **dashed**, not dotted.
Also: y and x units become **z** (HG is z-scored to the pre-stimulus baseline,
`make_epoched_data.py:317`), in every subspace figure that says "(HG)".

### 1.5 Unreliable windows: caption, not shading (**done** 2026-11-09)

Drop the grey span from the cosine panel and the alignment figure; don't draw
the ceiling in unreliable windows (it means nothing there). Caption: "no
reliable code before 0.5 s"; a little more in the Results.

### 1.6 Cluster-correct "ceiling > cross" (**decided**: leave pointwise, say so in the caption; revisit only if a reviewer asks)

The "not one code" claim in the text rests on one test (the alignment averaged
over the reliable windows, `alignment_reliable_ci` in `summary.json`), so the
bar is descriptive. Notes from the discussion:

Why it is pointwise now: "cross > shuffle" has a permutation null (labels
shuffled), and cluster correction needs a null distribution of cluster sizes.
"Ceiling > cross" has no label-shuffle null: there is no relabelling that makes
"one code" true. The plan (§3.2) used the participant bootstrap of ceiling −
cross and asked whether its CI excludes 0, window by window.

Proposed method, consistent with the other cluster tests (cluster mass against
the maximum-cluster null): centre the bootstrap draws of the gap on 0 (each
draw minus the observed gap), which is the gap's sampling distribution under
"gap = 0"; per window, threshold at that distribution's 95th percentile; form
clusters of windows where the observed gap exceeds it; compare each cluster's
mass with the 95th percentile of the largest cluster mass across the centred
draws. Computable from the saved `per_split_windows.csv` (no job rerun).
Decide: this method, or another.

### 1.7 Error bands on cross and ceiling (**done** 2026-11-09: saved by `compare_coding_axes`; `add_cosine_cis.py` adds them to older runs)

They exist in the computation but are not saved: `compare_coding_axes`
bootstraps cross and ceiling (participant resampling) and only keeps the CIs of
their ratio (alignment), of their difference (the gap, for 1.6) and of the two
separate ceilings. Saving `cross_lo/hi` and `ceiling_lo/hi` (from the same
draws, from the saved `per_split_windows.csv`) lets the cosine panel show bands
as the decoding figures do. As in the decoding figures, the bands (bootstrap
CI: how precise the curve is) and the bars (tests) are different things; the
shuffle band is the null, not an error bar. Goes with 1.6 (same recompute).

### 1.8 Alignment figure to the supplement (**done** 2026-11-09)

y label "cross / ceiling"; "95% CI" out of the legend (to the caption).

### 1.9 Subspace test (supplement) (**decided**: keep as is, explain in caption and Methods)

Elements, for reference (and for its caption):

- **k**: how many directions (patterns over electrodes) each code is allowed.
  k = 1 is the single-axis comparison of the cosine figure, on the
  time-averaged pattern.
- **θ1, θ2, ...** (principal angles): the angles between the two k-direction
  codes, smallest first. cos θ1 = the best-matching pair of directions
  (anything shared?); cos θk = the worst (are they the same?). Plotted as
  cosines: 1 = same direction, 0 = perpendicular.
- **C→S, S→C** (alignment index): the share of the congruency effect that lies
  in the switch directions (and back), out of the most any k directions could
  capture.
- **dot** = cross (congruency from one half against switch from the other);
  **error bar** = participant-bootstrap 95 % CI; **filled** = above both
  shuffle nulls (= "shared"); **faded** = the ceiling is not reliable.
- **grey tick** = ceiling. **grey bar from 0** = the shuffle null's 95th
  percentile; "top = threshold": a dot must be above the bar's top to count.
- **shaded column** = the primary k (k = 2), the pre-specified test; the other
  columns are descriptive.
- **Bottom row**: "dimension" j = the j-th direction of a code (the j-th
  singular vector of its electrodes × windows effect matrix, from one half).
  **Held-out variance** = how much of the other half's effects that direction
  captures (cross-validated, so noise does not count); units z² summed over
  electrodes and windows. **Line** = the shuffle null's 95th percentile;
  **filled = reproducible** = above it. k per code = its number of leading
  reproducible directions (congruency 3, switch 2); primary k = the smaller.
  Congruency's first direction holds 28 against ~1 for the others; switch's
  14: each code is essentially one direction.
- **Log y** in the bottom row because 28 and 1 do not fit on one linear axis;
  on a linear axis dimensions 2-3 sit at the bottom, which is the message.
  Alternative: plot `heldout_fraction` (0.23, 0.01, 0.01) on a linear axis.

Units in the y labels: cosine and alignment index have none; held-out
variance is z².

Decide: a simpler figure (e.g. only the k = 2 column of the top row plus the
bottom row on a linear scale), or keep and caption.

### 1.10 Fig A schematic (**decided**: code kept as `dcc_scripts/subspace/schematic_figure.py`, not in the paper)

Draft (temporary):
`/tmp/claude-1491160/-hpc-home-jz421/285c9664-8ceb-475a-8097-1fff6b39ab8e/scratchpad/schematic/fig_a_schematic.png`
(and `.pdf`). Left: trials split into halves, per-electrode congruency and
switch effects per half, cross and ceiling. Right: separate / partly shared /
one code, with each half's wobble faint. Colours follow 1.2.

### 1.11 Significance-bar style (**done**)

Keep the current bars; label them when there is more than one (as in 1.1).

### 1.12 Line styles and block colours across the paper (**being done in another session**, 2026-11-09)

Another Claude session is making this change in `plotting_parameters.py`,
`condition_registry.py`, `paper_panels.py` and the brain-behaviour scatter: one
shared palette, `plotting_parameters.PROPORTION_COLORS` (Wong: inc. prop.
orange `#E69F00` / vermillion `#D55E00`, swi. prop. sky blue `#56B4E9` / blue
`#0072B2`); four-trace power plots with colour = block, c / r solid, i / s
dashed, legend "r, 25% S"; block shorthand "25% I" / "25% S"; units "(z)". The
subspace figures follow it and read `PROPORTION_COLORS` rather than keeping
their own copy. Don't edit those four files from this list.

Notes from before, for the record:

- **Line style**: the power traces use dashed for i and s, solid for c and r.
  Fig A's cells used dashed for *repeat*; the 1.2 renders follow the power
  traces (switch dashed). Adopt everywhere (Fig A, Fig B, schematic)?
- **Block colours differ between figure families**: the power traces
  (`plotting_parameters.py`) use pink / orange for 25 / 75 % I and sky blue /
  purple for 25 / 75 % S; the decoding figures (`condition_registry.py`) and
  the subspace figures use salmon `#FF7E79` / dark red `#A32319` and sky blue
  `#05B0F0` / navy `#0B4F8A`. Pick one set for the paper.


### 1.13 Cosine-panel colours (**done** 2026-11-09)

Cross Wong reddish purple `#CC79A7`, ceiling Wong bluish green `#009E73`
(`CROSS`, `CEILING`), shuffle band paler. These were Fig 2's LPFC electrode
colours, so Fig 2 changes (1.14).

### 1.14 Fig 2 electrode colours (**done** 2026-11-09 in `condition_plot_specs.py`: Wong yellow `#F0E442` analyzed, dark olive `#857300` task-significant; Fig 2 not re-rendered yet)

The LPFC sets ("Analyzed LPFC", "Task-significant LPFC";
`dcc_scripts/vis/condition_plot_specs.py:215, 221`) need new colours: the
whole-brain sets already use grey and black, and green / purple are now cross
/ ceiling. Idea (2026-11-09): light and dark yellow. Rendered three pairs
(temporary: `/tmp/claude-1491160/-hpc-home-jz421/285c9664-8ceb-475a-8097-1fff6b39ab8e/scratchpad/yellow/yellow_pairs.png`):
Wong yellow `#F0E442` + dark olive `#857300` is the most distinct and stays
clear of the 25 % I orange; yellow + ochre leans orange; pale yellow + gold is
too close to orange. Not checked with a colour-blindness simulator
(`colorspacious` is not installed). On a light-grey surface yellow may need a
thin dark edge.

### 1.15 Decoding legends "25% I" / "25% S" (**done** 2026-11-09; existing decoding PNGs keep the old text until regenerated)

`condition_registry.py` `trace_labels`: 20 labels, e.g. '25% incongruent' ->
'25% I'. That file is also being edited by the other session (1.12).

### 1.16 "switch" -> "task seq." in the subspace figures (**done** 2026-11-09: `NAMES` in `subspace_figures.py`; subspace-test ticks C→TS / TS→C)

Variable / axis names, panel titles ("task seq. × swi. prop."), cosine y label.
Levels (switch / repeat, s / r) stay.

---

## 2. Running / waiting on the cluster

- **Both nulls with 200 splits, electrode bootstrap (submitted 2026-11-12):**
  jobs 61697413 (all lPFC) and 61697414 (sig lPFC), `SUBJECT_SET=good
  ELECTRODES=all|sig ROIS=lpfc,dlpfc RT_CONTROL=none N_NULL_SPLITS=200
  N_BLOCK_PERM=500 N_BLOCK_NULL_SPLITS=200`, `BOOT_UNIT=electrode` (now the
  default), 36 h limit. Folders end
  `_perm1000x200_seed0_blocknull500x200_bootelectrode_subjects-good`. When
  done: redraw with `redraw_figures.py` (adds the cluster bars, change-vector
  panel and single panels), update the numbers in `paper_draft.md` §3.4 and
  the p-values quoted in §4 below. (Earlier submissions 61682547/48,
  61683794/95 and 61689260 were cancelled while settings changed; partial
  folders deleted.)
- **Kept: the sig run with participant bootstrap and 200-split nulls**
  (job 61689261, finished before the cancel):
  `windows_0.0to1.5s_w64s16_sig_lpfc-dlpfc_mean_diff_splits200_perm1000x200_seed0_blocknull500x200_subjects-good`.
  The participant-level version for a reviewer. It shows the 200-split null
  barely changes the p-values (congruency × inc. prop. 0.082 → 0.076; task
  seq. × swi. prop. 0.006 → 0.006; shared windows identical), so the 20-split
  mismatch was not why the CI and p disagree.

- **Occipital (submitted 2026-11-09), same settings as the lPFC runs:**
  - coding axes, all occ electrodes: job 61442719 (rewrites the 10-30
    `..._all_occ_..._subjects-good` folder; same seed, now with 500 block
    permutations); task-significant occ electrodes: job 61442720
    (`..._sig_occ_..._subjects-good`). Then `add_cosine_cis.py` is not needed
    (new runs save the CIs); redraw with `redraw_figures.py`.
  - window-mean bars and curves, ROI `occ`, for congruency, switch type, task
    and big letter: matched 61442721-24, unmatched 61442725-28 (output in each
    run's `<comparison>/occ/` folder).

- **Sig-electrode coding-axes run**, job 61433366: **done** (42 min). Folder
  `windows_0.0to1.5s_w64s16_sig_lpfc-dlpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good`
  (ROIs in the order given, so `lpfc-dlpfc`); CIs added and redrawn in its
  `figures_2026-11-09/`. 152 electrodes, 19 participants; alignment 0.63
  [0.49, 0.85]; shared from window 8 (about 0.6 s) to the end. Originally:
  task-significant lPFC + DLPFC electrodes, good-subject set, otherwise the
  main run's settings. When done, redraw:
  `python dcc_scripts/subspace/redraw_figures.py --run-dir <run>/windows_0.0to1.5s_w64s16_sig_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good --out-dir <that>/figures_<date>`.
  Open: main text, supplement, or a sentence?

## 3. Other open items from these sessions

- **Results log**: the task and big-letter window-mean runs (jobs
  61245177-80, finished 2026-11-07) have no entry in `results_log.md` yet. Task
  in lPFC: sig 0.58, all 0.55, nonsig 0.52 at 152 electrodes (chance 0.50;
  sig vs nonsig Holm p = 0.04).
- **Paper draft**: `paper_draft.md` §1.4 still describes F5 as panels a-e
  (anatomy + coding axes). Update to the Fig A / Fig B split and the scatter's
  move.
- **Controls overlay** (`plot_coding_axes_controls`) and the **electrode-count
  curves** (`electrode_curve.py`, `_SERIES`) still use the old blue/orange,
  which is close to the LWPC/LWPS colours.
- **Figure 5 anatomy** figures (`fig5_height*`) were not restyled; only the
  scatter is used now.

## 4. Tabled discussions (2026-11-12; pick up after the figures are arranged)

Numbers below are all lPFC on the old 20-split null unless noted; re-read them
from the new runs (§2).

1. **Bands on the block-separation changes: bootstrap CI or shuffle-null
   band?** The CI and the test can disagree (congruency × inc. prop.: CI
   [0.06, 0.56] excludes 0, block-null p ≈ 0.08–0.13). They answer different
   questions: the CI is precision across electrodes (no trial resampling;
   unmatched trial counts between proportions make the 25 % blocks noisier,
   not biased), the null asks whether block labels matter. The electrode
   bootstrap narrows the CIs, so this will happen more. Option: draw the 95 %
   of the shuffle null as a grey band (as the decoding figures show chance)
   instead of bootstrap bands; draws are in `null_block.npz`, a redraw only.
   Claims rest on the shuffle p and the cluster bars either way. Note the
   decoding bands are trial / pseudopopulation resampling, so they are a
   loose precedent for electrode-bootstrap bands. **Not SEM over splits:**
   splits re-divide the same trials (any two share about half their halves),
   so their spread reflects only the partition, and SEM = SD / √n_splits
   shrinks with however many splits were run (2,000 splits would give a band
   about 3× thinner on the same data). SD across splits is honest only as
   "split-to-split variability", not as an interval. Ranking: shuffle-null
   band > electrode-bootstrap CI > SD over splits > SEM over splits (avoid).
   With the null band, the traces are plain lines (no shading): grey band =
   2.5–97.5 percentiles per window of the 500 block-null draws (trial block
   labels shuffled within participant × cell, same shuffle for all of a
   participant's electrodes, full analysis rerun on fresh splits), centred
   near 0; cluster bars where a trace leaves it. Each panel has two
   contrasts with near-identical nulls: draw one grey band (their union) and
   say so in the caption. Lost: any picture of the estimate's precision.
   **Done 2026-11-12 (changes figure redrawn with the grey block-null band
   and dashed null median; `contrast_clusters` adds `null_lo/_mid/_hi`):
   decoding / cosine style everywhere.**
   Traces keep their bootstrap band (precision; lets the traces be compared
   by eye); wherever a shuffle test exists, the grey shuffle band is drawn
   too (chance); bars are the tests. Cosine: unchanged (label-shuffle band +
   bootstrap bands; "ceiling > cross" and "reliable windows" are themselves
   bootstrap tests, no shuffle). Changes: bootstrap bands + grey block-null
   band (one, the union of the two contrasts' nulls) + cluster bars; own −
   other is the test that compares the two traces. Trajectory strips:
   bootstrap bands only (no test; caption "descriptive"). A trace whose CI
   excludes 0 inside the grey band then reads as "precise but not beyond
   chance".
2. **Trajectory strips:** keep the bootstrap bands (descriptive, captioned)
   or drop them; there is no test on the strips.
3. **Separations vs per-electrode scores.** Averaging each electrode's LWPC /
   LWPS (the anatomy scores) gives about 0 (mean d 0.00 / 0.03; 47 % / 55 %
   positive) because electrodes with negative congruency effects adapt the
   other way and cancel. The separation weights each electrode by its
   held-out congruency (task-seq.) effect, so signs align and non-coding
   electrodes drop out. Plain wording for the paper: "each electrode's change
   in congruency effect, weighted by how strongly and in which direction it
   codes congruency (held-out trials), summed across electrodes, as a linear
   readout of the code would see it". Simpler variant: weight by the sign
   only (noisier). Decide wording or variant.
4. **Is F4d needed at all?** F3 (power traces) already shows LWPC and LWPS
   with per-electrode scores. F4d adds only "the shared population code
   adapts, and along which axis". Keep F4d if that claim is in the story;
   otherwise drop it and let F3 carry adaptation.
5. **Which F4d panel.** Candidates: the changes figure (time-resolved, with
   cluster bars: task seq. 25 % S > 75 % S, about 0.65–1.1 s), and the new
   change-vector panel (`proportion_change_vectors_*`: s − r by swi. prop.
   points along its own axis, 0.42 [0.18, 0.67], p = 0.03; i − c by inc.
   prop. diagonal, 0.22 along congruency and 0.27 along task seq.; controls
   near 0). Per-coordinate CIs, not a joint region (caption). Its average
   includes the six unreliable windows before 0.5 s; a reliable-window
   version would be clearer, but decide before looking and justify it as the
   alignment's convention. The four-path proportion trajectories and the
   difference paths (`proportion_differences_*`): leaning out, or supplement
   (no uncertainty; off-diagonal noise looks like effects; oblique numbers
   differ from the projections).
6. **Own − other test.** The block null shuffles both proportions together
   ("neither matters"). A null specific to "own changes more than other"
   would exchange the two proportions (per participant, or trial labels
   between them), as the anatomy exchanges LWPC and LWPS. Add or not.
7. **Subspace test: main or supplement.** Each code is one dimension (23 % and
   13 % of held-out variance; the rest about 1 %), so the test reduces to the
   cosine pooled over time. Shaded column = primary k (rule pre-specified:
   the smaller code's number of reproducible dimensions; k = 2 here, but
   dimension 2 only just passes; k = 1 in sig). Evidence for "not one code"
   is modest (0.75 [0.67, 0.94] of ceiling; cosine figure's alignment 0.72
   [0.59, 0.97]; ceiling > cross only about 0.8–1.0 s, pointwise). Options:
   one Results sentence + supplement figure, or a stripped panel (k = 2 θ1
   dot, ceiling, null + the dimensions row). "Alignment index" is
   established (Elsayed et al. 2016); define it in the caption; the plotted
   value is raw, cross / ceiling is the normalized number for the text.
8. **Trajectory y label.** The trajectories use orthogonalized axes (y = task
   seq. beyond congruency; true distances, plane fractions); the proportion
   figures use oblique coordinates (y = task-seq. axis; each change
   attributed to its own code). On orthogonalized axes a pure task-seq.
   change of 0.42 would show 0.16 along congruency. Labels are correct as
   they are; switching the trajectories to oblique coordinates would let both
   say "task-seq. axis" at the cost of true distances.
9. **Factor name in the prose:** "switch type" (paper text) vs "task seq."
   (figures). Pick one.

## 5. Done (for reference)

- 2026-11-12, **bootstrap unit decided (final):** electrode level everywhere,
  including the coding axes, subspace test, trajectories and block
  separations: electrodes resampled within participant (each keeps its
  count), 2,000 draws, percentile CIs. The bootstrap also decides "reliable
  windows" and "ceiling > cross", which pass more easily at this level; say
  so in Methods. Methods sentence: "Electrodes were the unit of analysis;
  permutation nulls were built within participant; intervals came from
  resampling electrodes within participant." `BOOT_UNIT=participant` gives
  the participant version (shared draw: `src/analysis/utils/resampling.py`,
  also used by the anatomy's centroid ellipses).
- 2026-11-12: every subspace figure and the overlap scatter also written as
  450-dpi LZW TIFF (`panel_layout.save_figure`); `figures_2026-11-12/`.

- 2026-11-06: task and letter window-mean curves and bars.
- 2026-11-06: `fig5_height_scatter` (LWPC–LWPS, no height colouring), now in
  Fig B.
- 2026-11-09: subspace figures restyled (print size, 7 pt, achromatic stats,
  black/grey cells, split block separations, letters on the proportion
  planes, supplementary `_alignment` and `_changes`), all as PNG + PDF.
- 2026-11-09: `figure_editing_guide.md` §4 and §10 (notebook recipe).
- 2026-11-11: task-seq. block separations were empty in every
  `figures_2026-11-09/10` render (the 1.16 rename turned the display name into
  the own-axis filter key). Fixed (`_OWN_AXIS` in `subspace_figures.py`, test
  `test_block_separations_draw_both_variables`); all lPFC, sig lPFC, all occ and
  sig occ redrawn to `figures_2026-11-11/`. Use those, not `figures_2026-11-09/`.
- 2026-11-11: overlap scatter (`fig5_scatter.py --seg-dir`): identity line
  dropped, separate-half r and p printed in the panel; drawn for both pairs
  (`fig5_height_scatter` in `section19_paper_2026-10-30/`,
  `fig5_height_main_effects_scatter` in `continuous/`).
