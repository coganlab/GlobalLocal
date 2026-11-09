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

### 1.2 Colours for the four trajectory cells (Fig A) (**open**)

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

### 1.4 Proportion trajectories: back to colour + line style, four legend entries (**next**)

As the power traces: colour = block (25 % / 75 %), line style = level, with
i and s dashed, c and r solid (`plotting_parameters.py`: `Stimulus_i25` '--',
`Stimulus_c25` '-', ...); all four in the legend ("i, 25% I", "c, 25% I", ...);
no letters at the path ends. Note: the power traces use **dashed**, not dotted.
Also: y and x units become **z** (HG is z-scored to the pre-stimulus baseline,
`make_epoched_data.py:317`), in every subspace figure that says "(HG)".

### 1.5 Unreliable windows: caption, not shading (**next**)

Drop the grey span from the cosine panel and the alignment figure; don't draw
the ceiling in unreliable windows (it means nothing there). Caption: "no
reliable code before 0.5 s"; a little more in the Results.

### 1.6 Cluster-correct "ceiling > cross" (**open**: method)

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

### 1.7 Error bands on cross and ceiling (**open**)

They exist in the computation but are not saved: `compare_coding_axes`
bootstraps cross and ceiling (participant resampling) and only keeps the CIs of
their ratio (alignment), of their difference (the gap, for 1.6) and of the two
separate ceilings. Saving `cross_lo/hi` and `ceiling_lo/hi` (from the same
draws, from the saved `per_split_windows.csv`) lets the cosine panel show bands
as the decoding figures do. As in the decoding figures, the bands (bootstrap
CI: how precise the curve is) and the bars (tests) are different things; the
shuffle band is the null, not an error bar. Goes with 1.6 (same recompute).

### 1.8 Alignment figure to the supplement (**next**)

y label "cross / ceiling"; "95% CI" out of the legend (to the caption).

### 1.9 Subspace test (supplement) (**open**: redesign)

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

### 1.10 Fig A schematic (**open**: review draft)

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


---

## 2. Running / waiting on the cluster

- **Sig-electrode coding-axes run**, job 61433366 (submitted 2026-11-09):
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
- **Failing test**: `test_coding_axes.py::test_time_axis_label_follows_the_epochs_lock`
  expects "(s from response)"; the function returns "Time from response (s)".
  Was failing before 2026-11-09. Fix the test or the label.
- **Figure 5 anatomy** figures (`fig5_height*`) were not restyled; only the
  scatter is used now.

## 4. Done (for reference)

- 2026-11-06: task and letter window-mean curves and bars.
- 2026-11-06: `fig5_height_scatter` (LWPC–LWPS, no height colouring), now in
  Fig B.
- 2026-11-09: subspace figures restyled (print size, 7 pt, achromatic stats,
  black/grey cells, split block separations, letters on the proportion
  planes, supplementary `_alignment` and `_changes`), all as PNG + PDF.
- 2026-11-09: `figure_editing_guide.md` §4 and §10 (notebook recipe).
