# Editing the paper figures yourself

How to change the look of the subspace (coding-axes), Figure 5 anatomy and
window-mean decoding figures, and redraw them **without rerunning any analysis**.
Written 2026-11-07. Line numbers are as of that date; if one has moved, search
for the name instead.

Contents:

1. [The loop: edit, redraw, look](#1-the-loop-edit-redraw-look)
2. [Where each figure is drawn](#2-where-each-figure-is-drawn)
3. [Changes that apply to every figure](#3-changes-that-apply-to-every-figure)
4. [Subspace figures, one by one](#4-subspace-figures-one-by-one)
5. [Figure 5 anatomy figures](#5-figure-5-anatomy-figures)
6. [Window-mean decoding: bars and electrode-count curves](#6-window-mean-decoding-bars-and-electrode-count-curves)
7. [Colours already in use (avoid recycling)](#7-colours-already-in-use-avoid-recycling)
8. [Decisions from 2026-11-06 and the edits they need](#8-decisions-from-2026-11-06-and-the-edits-they-need)
9. [Checking you didn't break anything](#9-checking-you-didnt-break-anything)

---

## 1. The loop: edit, redraw, look

Every figure here is drawn by a plotting-only function that reads the run's
saved tables (CSV/JSON). So the loop is always:

1. Edit the plotting function (or the colour constants at the top of its file).
2. Run the redraw command for that figure (§2). It takes seconds.
3. Open the PNG. When happy, use the PDF for Illustrator.

Redraw into a scratch folder while experimenting, so the run's own figures stay
as they were until you are happy:

```bash
cd /hpc/group/coganlab/jz421/GlobalLocal
conda activate ieeg
RUN=dcc_scripts/subspace/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/coding_axes/windows_0.0to1.5s_w64s16_all_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good
python dcc_scripts/subspace/redraw_figures.py --run-dir $RUN --out-dir ~/fig_tests
```

`--out-dir` defaults to the run folder itself, which **overwrites** the run's
figures. Leave it off only for the final version.

The figure functions take the data as arguments, so you can also call them
from a notebook (`dcc_scripts/subspace/subspace_notebook.ipynb` already
imports them) and iterate there.

## 2. Where each figure is drawn

| Figure (file it writes) | Drawn by | Redraw with |
|---|---|---|
| Cosine over time + alignment (`coding_axis_cosines.png`) | `plot_coding_axes`, [coding_axes_dcc.py:162](../dcc_scripts/subspace/coding_axes_dcc.py#L162) | `redraw_figures.py` |
| Subspace test: principal angles, alignment index, dimensions (`subspace_test.png`) | `plot_subspace_test`, [subspace_figures.py:91](../dcc_scripts/subspace/subspace_figures.py#L91) | `redraw_figures.py` |
| Trajectory plane + x(t), y(t) (`trajectories_<window>[_uncentred][_raw].png`) | `plot_trajectories`, [subspace_figures.py:115](../dcc_scripts/subspace/subspace_figures.py#L115) | `redraw_figures.py` |
| Block separations + changes (`block_separations_<window>[_unique].png`) | `plot_block_separations`, [subspace_figures.py:178](../dcc_scripts/subspace/subspace_figures.py#L178) | `redraw_figures.py` |
| Proportion trajectories, 2×2 (`proportion_trajectories_<window>.png`) | `plot_proportion_trajectories`, [subspace_figures.py:240](../dcc_scripts/subspace/subspace_figures.py#L240) | `redraw_figures.py` |
| Controls overlay: RT-adjusted, matched, random (`coding_axes_controls.png`) | `plot_coding_axes_controls`, [coding_axes_dcc.py:247](../dcc_scripts/subspace/coding_axes_dcc.py#L247) | `plot_coding_axes_controls.py` |
| Figure 5 anatomy, full (`fig5_height.png`) | `plot_figure5_height`, [stability_flexibility_anatomy.py:2486](../src/analysis/stats/stability_flexibility_anatomy.py#L2486) | `n4_section19_followups.py --sections 4` (recomputes the bootstraps, minutes) |
| LWPC–LWPS scatter alone (`fig5_height_scatter.png`) | `plot_figure5_scatter`, [stability_flexibility_anatomy.py:2455](../src/analysis/stats/stability_flexibility_anatomy.py#L2455) | `fig5_scatter.py` (seconds) |
| Window-mean group bars (`group_bars.png`) | `plot_group_accuracy_bars`, [group_bars.py](../src/analysis/decoding/plots/group_bars.py) | snippet in §6 |
| Accuracy by electrode count (`electrode_curve.png`) | `plot_accuracy_by_electrode_count`, [electrode_curve.py](../src/analysis/decoding/plots/electrode_curve.py) | snippet in §6 |

`redraw_figures.py` always redraws every subspace figure of a run; there is no
flag for just one. It is fast, so that is fine.

## 3. Changes that apply to every figure

### Size

Each function sets its own `figsize` in **inches**. Nature-family widths are
89 mm (single column) and 183 mm (double column), so use `figsize=(183 / 25.4, h)`.

**Draw at the final print size.** The current subspace figures are drawn 9–12
in wide with 8–9 pt text. Scaled down to 183 mm (7.2 in), that text ends up
around 5 pt, which is why it looks tiny. Set `figsize` to the size the panel
will have on the page and the font sizes come out as written. Don't scale
the figure in Illustrator.

Current sizes, for reference:

| Figure | `figsize` | Where |
|---|---|---|
| cosine | (9.0, 6.2) | [coding_axes_dcc.py:174](../dcc_scripts/subspace/coding_axes_dcc.py#L174) |
| subspace test | (10.5, 7.0) | [subspace_figures.py:98](../dcc_scripts/subspace/subspace_figures.py#L98) |
| trajectories | (10.5, 5.0) | [subspace_figures.py:127](../dcc_scripts/subspace/subspace_figures.py#L127) |
| block separations | (12.0, 6.0) | [subspace_figures.py:196](../dcc_scripts/subspace/subspace_figures.py#L196) |
| proportion trajectories | (10.0, 9.0) | [subspace_figures.py:252](../dcc_scripts/subspace/subspace_figures.py#L252) |
| Figure 5 anatomy | (7.2, 5.0) | `plot_figure5_height` |
| Figure 5 scatter | (2.3, 2.3) | `plot_figure5_scatter` |

### Font sizes

Nature asks for 5–7 pt text and 8 pt bold lowercase panel letters (check the
target journal's figure guide; they differ slightly). The code sets sizes in
three different ways:

- **Subspace figures** (`subspace_figures.py`, `coding_axes_dcc.py`): every call
  passes `fontsize=` explicitly (axis labels 9, ticks 8 via `_style`, legends
  7–7.5). Change them call by call, or better, wrap the function body in
  `with plt.rc_context(nature_style(7)):` and delete the `fontsize=` arguments,
  so one number sets them all.
- **Figure 5 anatomy**: one dict, `_FIG5_RC`
  ([stability_flexibility_anatomy.py:2405](../src/analysis/stats/stability_flexibility_anatomy.py#L2405)):
  6.5 pt labels, 6 pt ticks. Already Nature-sized.
- **Window-mean figures**: `nature_style(base_fontsize)` from
  [style.py:32](../src/analysis/decoding/plots/style.py#L32). Every size is
  derived from one number (labels = base, ticks = base − 1, legend = base − 1.5).
  The default is 9; pass `base_fontsize=7` for print.

`nature_style` is the shared one to converge on. It also sets a sans-serif
font (Arial, falling back to Nimbus Sans on the cluster), 0.8 pt axes and no
top/right spines.

### Fonts in the PDF

Add `'pdf.fonttype': 42` to the rc (as `_FIG5_RC` does) so text stays editable
in Illustrator instead of being converted to outlines.

### Less text

- Legends: the subspace figures put legends outside the axes
  (`bbox_to_anchor=(1.01, 1)`), which widens the figure. Direct labels at the end
  of each line (as `electrode_curve.py` does with `ax.annotate`) take no extra
  room and remove the need for a legend.
- Titles: delete the `set_title` calls and say it in the caption. Figure 5 and
  the cosine figure already have none.
- Axis labels: drop units that are the same everywhere ("(HG)") into the caption.

## 4. Subspace figures, one by one

All colours for these figures are module-level constants, so most colour
changes are a one-line edit:

- [subspace_figures.py:16-20](../dcc_scripts/subspace/subspace_figures.py#L16):
  `CROSS`, `CEILING` (blue/orange), `INK`, `MUTED`, `NULL_FILL`, `PRIMARY_FILL`,
  and `CELL_STYLE` (colour and line style of the four cells).
- [subspace_figures.py:164-170](../dcc_scripts/subspace/subspace_figures.py#L164):
  `OWN_COLOR` (block colours), `OTHER_LS`, `PCT`.
- [subspace_figures.py:233-237](../dcc_scripts/subspace/subspace_figures.py#L233):
  `_LEVEL_COLOR`, `_PCT_LS` (proportion-trajectory planes).
- [coding_axes_dcc.py:126-127](../dcc_scripts/subspace/coding_axes_dcc.py#L126):
  the cosine figure's own copies of `CROSS`, `CEILING`, `INK`, `MUTED`,
  `NULL_FILL`, `UNRELIABLE`. **These are separate from the ones in
  `subspace_figures.py`**, so change both, or import one from the other.

### Cosine over time (`plot_coding_axes`)

Two panels: (a) similarity, (b) alignment = cross / ceiling.

- **Remove the Pearson r line:** delete the `a.plot(t, table['cross_pearson'], ...)`
  call, [coding_axes_dcc.py:190-191](../dcc_scripts/subspace/coding_axes_dcc.py#L190).
- **Keep only the "shared" bar:** the loop at
  [line 195](../dcc_scripts/subspace/coding_axes_dcc.py#L195) draws two bars,
  `('shared', ...)` and `('not_one_code', ...)`. Delete the second tuple. The
  "not one code" windows are still in `coding_axis_cosines.csv`
  (`not_one_code` column) for the caption.
- **Drop panel b** (alignment): change `plt.subplots(2, 1, ...)` to one axis and
  delete the `b.` lines (207-214). The alignment numbers stay in the CSV
  (`alignment`, `alignment_lo`, `alignment_hi`).
- **Unreliable windows** are the pale grey spans (`UNRELIABLE`, drawn at line 178).
- The cross cosine has **no CI** in the saved table (only the ceilings, gap
  and alignment do), so there is nothing to draw around the cosine line without
  changing the analysis.

### Trajectories (`plot_trajectories`)

Left: the plane (x = congruency axis, y = switch axis beyond it). Right: x(t)
and y(t) with participant-bootstrap CIs.

- **Cell colours and dashes:** `CELL_STYLE` (line 20), a dict of
  `cell: (colour, line style)`.
- **Layout:** the gridspec right after `figsize` at line 127. To put x(t) and
  y(t) as thin strips **under** the plane, make it 3 rows × 1 column with
  `height_ratios=[3, 1, 1]`.
- Start of each path = open circle, end = arrow (lines ~139-143).
- `cells='uncentred'` and the `_raw` table give the variants `redraw_figures.py`
  also writes.

### Block separations (`plot_block_separations`)

2 × 2: left = separation per block, right = change (25 % − 75 %) with CI.

- Colour = the variable's own proportion (`OWN_COLOR`: red family for
  incongruent proportion, blue for switch proportion; light = 25 %, dark = 75 %),
  line style = the other proportion (`OTHER_LS`).
- The right column's lines are black/dark grey; their colours are set where the
  change lines are plotted (~line 206-213).
- `unique=True` (the `_unique` files) uses the switch axis with the congruency
  axis removed.
- The statistics (estimate, CI, p, `excludes_0` per window, for `incProp`,
  `switchProp` and `own_minus_other`) are in `block_contrasts.csv`
  (`block_contrasts_unique.csv` for the unique axis). That is the table for the
  supplement.

### Proportion trajectories (`plot_proportion_trajectories`)

2 × 2: rows = variable (congruency, switch), columns = proportion (incongruent,
switch); the diagonal panels are the matched ones.

- Colour = condition level (`_LEVEL_COLOR`), line style = 25/75 % (`_PCT_LS`).
- Legends are drawn inside each panel at line 273 (`loc='best'`).

### Subspace test (`plot_subspace_test`)

Supplement only (decided 2026-11-06). Top: principal-angle cosines and alignment
index per k with ceiling and null; bottom: held-out variance per dimension.
Shared markers come from `_comparison_panel` (line 44).

## 5. Figure 5 anatomy figures

All in [stability_flexibility_anatomy.py](../src/analysis/stats/stability_flexibility_anatomy.py).

- Colours: `_INK`, `_MUTED`, `_RULE` (line 1974); `HEIGHT_COLORS` (line 2198,
  the tertile colours); `HEIGHT_CENTROID_COLORS` (darker versions of them).
- Fonts and line widths: `_FIG5_RC` (line 2405).
- The scatter's axis limits, identity line and labels are in one helper,
  `_overlap_axes` (line 2441), used by **both** the full figure's panel b and the
  scatter-only figure. Removing the identity line there removes it from both.
- Point size/opacity of the scatter-only figure: the `ax.scatter` call in
  `plot_figure5_scatter` (`s=5`, `alpha=0.45`).

Redraw the scatter alone from a finished run:

```bash
python dcc_scripts/stats/fig5_scatter.py \
  --dir dcc_scripts/stats/results/<EPOCHS>/anatomy_a1_lpfc_window_0.0to1.5s_all/continuous/section19_paper_2026-10-30
# --pair main_effects for congruency vs switch
```

## 6. Window-mean decoding: bars and electrode-count curves

Outputs per condition, ROI and comparison:
`dcc_scripts/decoding/figs/<EPOCHS>_subjects-good/window_mean_groups/<condition>_0to1.5s_20bts_<n>perm_seed42[_unmatched]/<comparison>/<roi>/`.
Congruency, switch type, task and big letter are all there (task and letter
added 2026-11-06, jobs 61245177–80).

- **Bars:** [group_bars.py](../src/analysis/decoding/plots/group_bars.py).
  Bar colour `_BAR = '#0173B2'` at the top; the figure size is
  `(0.75 * n_groups + 1.2, 2.6)` in the `plt.subplots` call.
- **Curves:** [electrode_curve.py](../src/analysis/decoding/plots/electrode_curve.py).
  Group colours `_SERIES` (line 10), in the order sig, nonsig, all; figure size
  `(6.4, 2.6)`; the footnote is the `fig.text(...)` call near the end, delete it
  to move it to the caption.

The job writes these figures itself; there is no redraw script. This snippet
redraws both from one output folder (tested 2026-11-07):

```python
import json, os
import pandas as pd
from src.analysis.decoding.plots.electrode_curve import plot_accuracy_by_electrode_count
from src.analysis.decoding.plots.group_bars import plot_group_accuracy_bars

d = '<output folder>/<comparison>/<roi>'      # e.g. .../Task/lpfc
read = lambda name: pd.read_csv(os.path.join(d, name))
n_match = json.load(open(os.path.join(d, 'params.json')))['n_match']

plot_accuracy_by_electrode_count({'summary': read('curve_summary.csv')},
                                 os.path.expanduser('~/fig_tests/curve'), base_fontsize=7)
plot_group_accuracy_bars({'summary': read('summary.csv'), 'tests': read('tests.csv'),
                          'accuracies': read('accuracies.csv'), 'n_match': n_match},
                         os.path.expanduser('~/fig_tests/bars'), base_fontsize=7)
```

Run it from the repo root (`/hpc/group/coganlab/jz421/GlobalLocal`) so the
`src` imports resolve.

## 7. Colours already in use (avoid recycling)

| Colour(s) | Means | Where it's set |
|---|---|---|
| salmon `#FF7E79` / dark red `#A32319` | 25 % / 75 % incongruent blocks (the paradigm colours) | [condition_registry.py](../src/analysis/config/condition_registry.py) (decoding, power traces); `OWN_COLOR` in subspace_figures.py |
| sky blue `#05B0F0` / navy `#0B4F8A` | 25 % / 75 % switch blocks (paradigm). **Also** reused for task decoding (taskG vs taskL contrasts) in the registry | same |
| grey `#949494` | shuffle / chance traces | condition_registry.py |
| blue `#2c7fb8` / orange `#d95f0e` / green `#31a354` | LWPC only / LWPS only / both (A1 groups, top sets) | `GROUP_COLORS`, stability_flexibility_anatomy.py:125 |
| blue `#2a78d6` / orange `#eb6834` | incongruent / congruent cells; cross / ceiling in the cosine and subspace figures; first two series of the electrode curve and controls overlay | `CELL_STYLE`, `CROSS`, `CEILING`, `_SERIES`, `RUN_COLORS` |
| Wong blue `#0072b2` / bluish green `#009e73` / reddish purple `#cc79a7` | ventral / middle / dorsal height tertiles | `HEIGHT_COLORS` |
| green `(0, 158, 115)`, purple `(204, 121, 167)`, red, blue, black, yellow | electrode sets on the brain plots | [condition_plot_specs.py](../dcc_scripts/vis/condition_plot_specs.py) (around lines 200-245) |

The blue/orange in row 5 is close to the LWPC/LWPS blue/orange in row 4, which
is the recycling to fix. The paradigm colours are the ones in the registry, not
Wong orange/vermilion and Wong sky blue/blue.

Matplotlib in the `ieeg` env is 3.10, so two-colour dashed lines are available
(`ax.plot(..., color=A, gapcolor=B, linestyle=(0, (2.5, 2.5)))`) if you want to try
them again. In the 2026-11-06 mock-up they made the four block lines hard to tell
apart.

## 8. Decisions from 2026-11-06 and the edits they need

Decided in the figure discussion, not yet in the code:

| Decision | Edit |
|---|---|
| Split into **Fig A** "the codes are partly shared" and **Fig B** "the codes adapt to proportion"; the subspace test goes to the supplement | Assemble in Illustrator from the individual PDFs; no code change |
| Fig A gets a **schematic** panel: method pipeline (split halves → congruency and switch axes → cosine, same-variable cosine as ceiling) plus the three possible geometries (separate, partly shared, one code) | Draw by hand / Illustrator |
| Cosine panel: **no Pearson r line, one significance bar ("shared")**; "not one code" in the text/caption | §4, cosine: delete lines 190-191 and the `not_one_code` tuple |
| Cosine panel **all achromatic**: black cosine, mid-grey ceiling, light-grey null, black bar | `CROSS`/`CEILING` in coding_axes_dcc.py:126 |
| Trajectory cells **black (incongruent) / grey (congruent)**, solid switch / dashed repeat; legend text stays as it is | `CELL_STYLE`, subspace_figures.py:20 |
| **Keep x(t), y(t)** as thin strips under the plane (they carry the CIs) | gridspec in `plot_trajectories` |
| Block separations as **split panels**: colour = the relevant proportion in its paradigm light/dark, the other proportion becomes the panel; no dashes | Rewrite the left column of `plot_block_separations` (4 panels instead of 2) |
| Fig B proportion planes: **all four planes**, colour = block proportion (paradigm colours), all solid, **i / c, s / r letters at the line ends** (white halo so they don't sit on lines) | `plot_proportion_trajectories`: colour from `OWN_COLOR`-style dict keyed by proportion, replace legend with `ax.annotate` at each path end |
| Block-separation statistics to the **supplement** | `block_contrasts.csv` is the table |
| Everything at **Nature size** | §3 |

Still open when the discussion stopped:

- Where the electrode scatter goes. `fig5_height_scatter` is **LWPC vs LWPS** (the
  adaptations, r = 0.10), which fits Fig B; Fig A is about the congruency and
  switch base effects, whose scatter (r = 0.23) is `fig5_height_main_effects`'s
  pair (`fig5_scatter.py --pair main_effects`).
- Whether Fig B shows the change panels (25 % − 75 % with CI) in the main figure
  or only in the supplement.

## 9. Checking you didn't break anything

```bash
cd /hpc/group/coganlab/jz421/GlobalLocal
python -m pytest -q tests/analysis/stats/test_coding_axes.py          # subspace + cosine figures
python -m pytest -q tests/analysis/stats/test_section19_anatomy.py    # Figure 5 + scatter
python -m pytest -q tests/analysis/decoding/test_window_mean_decoding.py  # bars + curves
```

The tests check that the figures are written and the numbers behind them are
right, not how they look; look at the PNG for that.
