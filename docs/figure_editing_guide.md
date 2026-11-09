# Editing the paper figures yourself

How to change the look of the subspace (coding-axes), Figure 5 anatomy and
window-mean decoding figures, and redraw them **without rerunning any analysis**.
Written 2026-11-07; §2–4 and §8 updated 2026-11-09 for the restyled subspace
figures, and §10 (notebook) added. Line numbers are as of that date; if one has
moved, search for the name instead.

Contents:

1. [The loop: edit, redraw, look](#1-the-loop-edit-redraw-look)
2. [Where each figure is drawn](#2-where-each-figure-is-drawn)
3. [Changes that apply to every figure](#3-changes-that-apply-to-every-figure)
4. [Subspace figures, one by one](#4-subspace-figures-one-by-one)
5. [Figure 5 anatomy figures](#5-figure-5-anatomy-figures)
6. [Window-mean decoding: bars and electrode-count curves](#6-window-mean-decoding-bars-and-electrode-count-curves)
7. [Colours already in use (avoid recycling)](#7-colours-already-in-use-avoid-recycling)
8. [Decisions from 2026-11-06 and where they are in the code](#8-decisions-from-2026-11-06-and-where-they-are-in-the-code)
9. [Checking you didn't break anything](#9-checking-you-didnt-break-anything)
10. [Tweaking a subspace figure in a notebook](#10-tweaking-a-subspace-figure-in-a-notebook)

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
figures. Leave it off only for the final version. The restyled figures of
2026-11-09 are in `$RUN/figures_2026-11-09/`; the run folder itself still has
the old previews.

For changing one figure over and over, a notebook is faster: §10.

## 2. Where each figure is drawn

| Figure (file it writes) | Drawn by | Redraw with |
|---|---|---|
| Fig A: cosine over time (`coding_axis_cosines`) **+ supplement** alignment (`coding_axis_cosines_alignment`) | `plot_coding_axes`, [subspace_figures.py:109](../dcc_scripts/subspace/subspace_figures.py#L109) | `redraw_figures.py` |
| Supplement: subspace test, principal angles, alignment index, dimensions (`subspace_test`) | `plot_subspace_test`, [subspace_figures.py:212](../dcc_scripts/subspace/subspace_figures.py#L212) | `redraw_figures.py` |
| Fig A: trajectory plane + x(t), y(t) strips (`trajectories_<window>[_uncentred][_raw]`) | `plot_trajectories`, [subspace_figures.py:235](../dcc_scripts/subspace/subspace_figures.py#L235) | `redraw_figures.py` |
| Fig B: block separations, split panels (`block_separations_<window>[_unique]`) **+ supplement** changes 25 − 75 % (`..._changes`) | `plot_block_separations`, [subspace_figures.py:281](../dcc_scripts/subspace/subspace_figures.py#L281) | `redraw_figures.py` |
| Fig B: proportion trajectories, 2×2 (`proportion_trajectories_<window>`) | `plot_proportion_trajectories`, [subspace_figures.py:344](../dcc_scripts/subspace/subspace_figures.py#L344) | `redraw_figures.py` |
| Controls overlay: RT-adjusted, matched, random (`coding_axes_controls.png`) | `plot_coding_axes_controls`, [coding_axes_dcc.py:247](../dcc_scripts/subspace/coding_axes_dcc.py#L247) | `plot_coding_axes_controls.py` |
| Figure 5 anatomy, full (`fig5_height.png`) | `plot_figure5_height`, [stability_flexibility_anatomy.py:2486](../src/analysis/stats/stability_flexibility_anatomy.py#L2486) | `n4_section19_followups.py --sections 4` (recomputes the bootstraps, minutes) |
| LWPC–LWPS scatter alone (`fig5_height_scatter.png`) | `plot_figure5_scatter`, [stability_flexibility_anatomy.py:2455](../src/analysis/stats/stability_flexibility_anatomy.py#L2455) | `fig5_scatter.py` (seconds) |
| Window-mean group bars (`group_bars.png`) | `plot_group_accuracy_bars`, [group_bars.py](../src/analysis/decoding/plots/group_bars.py) | snippet in §6 |
| Accuracy by electrode count (`electrode_curve.png`) | `plot_accuracy_by_electrode_count`, [electrode_curve.py](../src/analysis/decoding/plots/electrode_curve.py) | snippet in §6 |

Every subspace figure is written as `.png` and `.pdf` (use the PDF in
Illustrator). A main-text figure with a supplementary companion writes both in
the same call. `redraw_figures.py` always redraws every subspace figure of a
run; there is no flag for just one. It takes seconds.

The coding-axes controls overlay (`plot_coding_axes_controls`) and the Figure 5
and window-mean figures were **not** restyled on 2026-11-09.

## 3. Changes that apply to every figure

### Size

Each function sets its own `figsize` in **inches**. Nature-family widths are
89 mm (single column) and 183 mm (double column), so use `figsize=(183 / 25.4, h)`.

**Draw at the final print size.** The current subspace figures are drawn 9–12
in wide with 8–9 pt text. Scaled down to 183 mm (7.2 in), that text ends up
around 5 pt, which is why it looks tiny. Set `figsize` to the size the panel
will have on the page and the font sizes come out as written. Don't scale
the figure in Illustrator.

The subspace figures are now drawn at print size; their sizes are all in one
dict, `SIZES` (§4). Elsewhere:

| Figure | `figsize` (in) | Where |
|---|---|---|
| all subspace figures | see `SIZES` | [subspace_figures.py:49](../dcc_scripts/subspace/subspace_figures.py#L49) |
| Figure 5 anatomy | (7.2, 5.0) | `plot_figure5_height` |
| Figure 5 scatter | (2.3, 2.3) | `plot_figure5_scatter` |
| window-mean bars | (0.75 × groups + 1.2, 2.6) | `plot_group_accuracy_bars` |
| electrode-count curves | (6.4, 2.6) | `plot_accuracy_by_electrode_count` |

### Font sizes

Nature asks for 5–7 pt text and 8 pt bold lowercase panel letters (check the
target journal's figure guide; they differ slightly). The code sets sizes in
three different ways:

- **Subspace figures** (`subspace_figures.py`): one number, `FONT_SIZE = 7`
  (line 28), through `nature_style`: labels 7 pt, ticks 6, legends 5.5. Every
  function draws inside `plt.rc_context(RC)`, so no call passes `fontsize=`.
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

Everything is in [dcc_scripts/subspace/subspace_figures.py](../dcc_scripts/subspace/subspace_figures.py)
(the cosine figure moved there from `coding_axes_dcc.py` on 2026-11-09).

### The style block (lines 26-54): change these first

Every subspace figure reads these constants, so one edit restyles all of them.

| Constant | Line | What it controls | Now |
|---|---|---|---|
| `FONT_SIZE` | 28 | axis-label size; ticks 1 pt and legends 1.5 pt smaller | 7 |
| `RC` | 30 | the matplotlib settings every figure is drawn with (`nature_style(FONT_SIZE)` + editable PDF fonts). Add any rcParam here, e.g. `'axes.linewidth': 0.6` | |
| `LW` | 31 | width of every data line (pt) | 1.2 |
| `INK`, `GREY`, `MUTED` | 34 | black (data), mid-grey (ceiling, congruent cells), dark grey (zero lines) | |
| `NULL_FILL`, `UNRELIABLE`, `PRIMARY_FILL` | 35-36 | shuffle-null band, unreliable windows, primary-k band (subspace test) | pale greys |
| `CROSS`, `CEILING` | 37 | colours of the cosine and the ceiling (cosine figure, subspace test) | ink, grey |
| `CELL_STYLE` | 40 | the four cells in the trajectories: `cell: (colour, line style)` | incongruent ink, congruent grey; switch `'-'`, repeat `'--'` |
| `PROP_COLOR` | 44 | the proportion blocks, `proportion: {'low': 25 %, 'high': 75 %}`; used by the block separations and the proportion trajectories | paradigm salmon/dark red, sky/navy |
| `SIZES` | 49 | every figure's width and height in inches, at print size | see the dict |
| `HALO` | 53 | white outline behind the i/c/s/r letters | 1.5 pt |

Rules of thumb:

- To make a figure bigger or smaller on the page, change its entry in `SIZES`,
  not `FONT_SIZE`. The text stays the right size because nothing is scaled
  afterwards.
- To change a colour everywhere it means the same thing, change the constant,
  not the call.

### Cosine over time (`plot_coding_axes`, line 109): Fig A

One panel: unreliable windows (pale), the shuffle-null band, the ceiling (grey),
the cosine (black) and one black bar over the windows where the cosine is
**shared**. No Pearson line and no "not one code" bar; both stay in
`coding_axis_cosines.csv` (`cross_pearson`, `not_one_code`) for the text.

- Bar height: `y = 1.1 * ...` (line 137), 10 % above the highest curve. Bar
  thickness: `lw=2.5` (line 139).
- Legend position: `ax.legend(loc='upper left')` (line 142).
- y-axis label: line 141.
- The **supplementary** alignment figure (`coding_axis_cosines_alignment`) is the
  second half of the same function (lines 145-157): cosine / ceiling with its
  95 % CI, dotted line at 1.
- There is no CI on the cosine itself: the saved table doesn't have one.

### Trajectories (`plot_trajectories`, line 235): Fig A

Plane on top, x(t) and y(t) as thin strips beneath.

- Relative heights: `height_ratios=[2.6, 2]` (plane : both strips) and the
  strips' `hspace` (lines 250-251).
- Cell colours / dashes: `CELL_STYLE`. Legend text comes from the cell names
  (`'incongruent, switch'` ...); position `loc='lower right'` (line 273).
- Start marker `ms=3` (line 261); end arrow `mutation_scale=7` (line 265).
- Axis names (`'switch beyond\ncongruency'`; `\n` is a line break): lines 245-247.
- CI opacity in the strips: `alpha=0.12` (line 270).

### Block separations (`plot_block_separations`, line 281): Fig B

Four panels: congruency separation in 25 % and 75 % switch blocks, then switch
separation in 25 % and 75 % incongruent blocks. Inside each panel the
variable's own proportion is light (25 %) vs dark (75 %) in its paradigm
colours; the adaptation is the gap between the two lines.

- Panel titles ("25% switch blocks"): line 320. Legend: line 323. One shared
  x-label for the whole figure: `fig.supxlabel` (line 324; `y=-0.08` sets its
  distance below the panels).
- Space between the two pairs: `wspace=0.25` (line 306); within a pair:
  `wspace=0.08` (line 308).
- The **supplementary** change figure (`..._changes`) is the second half (lines
  327-340): 25 % − 75 % change with the own proportion (black, solid) and the
  other (grey, dashed) with CIs. Its statistics are `block_contrasts.csv`
  (`estimate`, `lo`, `hi`, `p`, `excludes_0` per window, for `incProp`,
  `switchProp` and `own_minus_other`), which is the supplementary table.
- `unique=True` (the `_unique` files) is the switch axis with the congruency
  axis removed, drawn the same way.

### Proportion trajectories (`plot_proportion_trajectories`, line 344): Fig B

All four planes (matched on the diagonal, crossed off it). Colour = the block,
in the column's proportion's paradigm colours; all lines solid; the level is
the bold letter at the end of each path (i / c, s / r) with a white halo.

- Letter position: `xytext=(2, 2)` points right and up of the path's end
  (line 377). Letter style: `fontweight='bold'`, `HALO`.
- Legend (one entry per block): line 381, `loc='best'`.
- Panel titles: line 380. Spacing: `hspace`, `wspace` (line 361).

### Subspace test (`plot_subspace_test`, line 212): supplement

Unchanged logic, at print size and in the achromatic palette. Marker size
`ms=4` in `_comparison_panel` and `_dims_panel`.

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
| salmon `#FF7E79` / dark red `#A32319` | 25 % / 75 % incongruent blocks (the paradigm colours) | [condition_registry.py](../src/analysis/config/condition_registry.py) (decoding, power traces); `PROP_COLOR` in subspace_figures.py |
| sky blue `#05B0F0` / navy `#0B4F8A` | 25 % / 75 % switch blocks (paradigm). **Also** reused for task decoding (taskG vs taskL contrasts) in the registry | same |
| grey `#949494` | shuffle / chance traces | condition_registry.py |
| blue `#2c7fb8` / orange `#d95f0e` / green `#31a354` | LWPC only / LWPS only / both (A1 groups, top sets) | `GROUP_COLORS`, stability_flexibility_anatomy.py:125 |
| black `#0b0b0b` / grey `#8c8c8c` | subspace figures: incongruent / congruent cells; cosine / ceiling (since 2026-11-09) | `INK`, `GREY`, `CELL_STYLE`, `CROSS`, `CEILING` in subspace_figures.py |
| blue `#2a78d6` / orange `#eb6834` | first two series of the electrode-count curve and of the coding-axes controls overlay (the subspace figures no longer use them) | `_SERIES`, `RUN_COLORS` |
| Wong blue `#0072b2` / bluish green `#009e73` / reddish purple `#cc79a7` | ventral / middle / dorsal height tertiles | `HEIGHT_COLORS` |
| green `(0, 158, 115)`, purple `(204, 121, 167)`, red, blue, black, yellow | electrode sets on the brain plots | [condition_plot_specs.py](../dcc_scripts/vis/condition_plot_specs.py) (around lines 200-245) |

The blue/orange in row 6 is close to the LWPC/LWPS blue/orange in row 4; it is
now only in the electrode curve and the controls overlay, so change `_SERIES`
and `RUN_COLORS` if those figures end up near Figure 5. The paradigm colours are
the ones in the registry, not Wong orange/vermilion and Wong sky blue/blue.

Matplotlib in the `ieeg` env is 3.10, so two-colour dashed lines are available
(`ax.plot(..., color=A, gapcolor=B, linestyle=(0, (2.5, 2.5)))`) if you want to try
them again. In the 2026-11-06 mock-up they made the four block lines hard to tell
apart.

## 8. Decisions from 2026-11-06 and where they are in the code

All of these were made in code on 2026-11-09 (§4 says where to tweak each):

| Decision | Where it lives now |
|---|---|
| Two main figures: **Fig A** "the codes are partly shared" (schematic, cosine, trajectories), **Fig B** "the codes adapt to proportion" (LWPC–LWPS scatter, proportion trajectories, block separations); the subspace test to the supplement | Assembled in Illustrator from the PDFs. The scatter is `fig5_height_scatter.pdf` (§5) |
| Fig A **schematic**: method pipeline + the three geometries (separate, partly shared, one code) | To draw by hand / in Illustrator |
| Cosine panel: no Pearson r, one significance bar ("shared"); "not one code" in the text | `plot_coding_axes` |
| Cosine panel **achromatic** | `CROSS`, `CEILING` |
| Cells **black / grey**, solid switch / dashed repeat, legend text unchanged | `CELL_STYLE` |
| **x(t), y(t) kept** as thin strips under the plane | `plot_trajectories` |
| Block separations as **split panels** in the paradigm colours | `plot_block_separations`, `PROP_COLOR` |
| Fig B planes: **all four**, colour = block, letters at the ends | `plot_proportion_trajectories` |
| **Change panels and block-separation stats to the supplement** | `..._changes` figure; `block_contrasts.csv` |
| Alignment panel to the supplement | `coding_axis_cosines_alignment` |
| Everything at **Nature size**, 7 pt | `FONT_SIZE`, `SIZES` |

Not done: the schematic, and the controls overlay (`plot_coding_axes_controls`),
which still uses the old blue/orange series colours and sizes.

## 9. Checking you didn't break anything

```bash
cd /hpc/group/coganlab/jz421/GlobalLocal
python -m pytest -q tests/analysis/stats/test_coding_axes.py          # subspace + cosine figures
python -m pytest -q tests/analysis/stats/test_section19_anatomy.py    # Figure 5 + scatter
python -m pytest -q tests/analysis/decoding/test_window_mean_decoding.py  # bars + curves
```

The tests check that the figures are written and the numbers behind them are
right, not how they look; look at the PNG for that.

`test_coding_axes.py::test_time_axis_label_follows_the_epochs_lock` was already
failing before 2026-11-09: it expects labels like "(s from response)", while
`time_axis_label` returns "Time from response (s)".

## 10. Tweaking a subspace figure in a notebook

Faster than the redraw script when you are changing one figure over and over:
load the run's saved tables once, then edit `subspace_figures.py`, reload it and
redraw just that figure. Nothing is recomputed. (The current
`dcc_scripts/subspace/subspace_notebook.ipynb` steps through the pipeline that
*computes* the axes; you don't need any of that to change a figure.)

Cell 1, once:

```python
import importlib, json, os, sys
import pandas as pd
from IPython.display import Image, display

ROOT = '/hpc/group/coganlab/jz421/GlobalLocal'
sys.path.insert(0, ROOT)
RUN = os.path.join(ROOT, 'dcc_scripts/subspace/results/Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20/coding_axes/windows_0.0to1.5s_w64s16_all_dlpfc-lpfc_mean_diff_splits200_perm1000x20_seed0_subjects-good')
OUT = os.path.expanduser('~/fig_tests')
os.makedirs(OUT, exist_ok=True)

read = lambda name: pd.read_csv(os.path.join(RUN, name))
cosines = read('coding_axis_cosines.csv')
traj = read('trajectories.csv')
seps, cons = read('block_separations.csv'), read('block_contrasts.csv')
ptraj, pchanges = read('proportion_trajectories.csv'), read('proportion_changes.csv')
config = json.load(open(os.path.join(RUN, 'run_config.json')))
traj_summary = json.load(open(os.path.join(RUN, 'trajectory_summary.json')))

import dcc_scripts.subspace.subspace_figures as sf
```

Cell 2, after every edit to `subspace_figures.py` (change the call to the
figure you are working on):

```python
importlib.reload(sf)          # picks up your edit, including the style block
tl = sf.time_axis_label(config['epochs_root_file'])
paths = sf.plot_block_separations(seps, cons, '0.0-1.5s', f'{OUT}/block_separations', time_label=tl)
for p in paths:
    if p.endswith('.png'):
        display(Image(p))
```

The other calls, with the same tables:

```python
sf.plot_coding_axes(cosines, f'{OUT}/cosines', time_label=tl)
sf.plot_trajectories(traj, None, traj_summary['axes'].get('0.0-1.5s'), '0.0-1.5s',
                     f'{OUT}/trajectories', first=config.get('traj_first'), time_label=tl)
sf.plot_proportion_trajectories(ptraj, pchanges, '0.0-1.5s', f'{OUT}/proportion_trajectories')
```

To try a value without editing the file, set it on the module before calling,
e.g. `sf.SIZES['separations'] = (7.2, 2.2)` or `sf.LW = 1.0`; the next call
uses it, and `importlib.reload(sf)` puts the file's values back. (`RC` is built
from `FONT_SIZE` at import, so to try another font size set
`sf.RC = {**sf.nature_style(8), 'pdf.fonttype': 42}`.)

The figures are drawn with the non-interactive Agg backend (the module sets it
so it works on the cluster), so they don't appear inline by themselves; that is
why the cell displays the saved PNG.
