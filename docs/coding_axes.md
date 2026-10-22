# Coding axes, subspace test and trajectories: runbook

How to run, check and read the coding-axes job. It computes the congruency
and switch-type coding axes in every A4 decoding window and the cosine between
them, against a label-shuffle null and a split-half ceiling. On the same axes,
halves and shuffles it also runs the subspace test (principal angles and the
alignment index) and draws the condition trajectories. The design and the
reasons behind each choice are in
[`analysis_plans.md` › Subspace trajectory plan §3–4](analysis_plans.md#3-coding-axes-and-the-angle-between-them).
The math is worked through with small examples in
[`subspace_math_walkthrough.md`](subspace_math_walkthrough.md). This page
only says how to produce and read the numbers.

**Status (2026-10-15):** §3.1–3.3 (coding axes and cosine) run on real data;
results in [§10](#10-results-2026-10-15-run). The subspace test (§3.4) and
the trajectories (§4) were added to the job after that run and are tested on
synthetic data only: rerun the job to get them. Not built: the
lPFC-subsampled-to-occipital control, dPCA (§5) and the block-split adaptation
(§6).

**Update (2026-10-22):** everything above has run on real data. The block-split
adaptation (plan §6, §1.3 here) ran stimulus- and response-locked on
2026-10-22; results in [§12](#12-results-2026-10-22-runs-block-split-adaptation).
Still not built: the lPFC-subsampled control and dPCA.

## Contents

1. [What it computes](#1-what-it-computes)
2. [Quick start](#2-quick-start)
3. [The runs: main result and controls](#3-the-runs-main-result-and-controls)
4. [Settings](#4-settings)
5. [Outputs](#5-outputs)
6. [Reading the results](#6-reading-the-results)
7. [Checks before trusting a run](#7-checks-before-trusting-a-run)
8. [Code map and tests](#8-code-map-and-tests)
9. [Deviations from the plan, and limits](#9-deviations-from-the-plan-and-limits)
10. [Results (2026-10-15 run)](#10-results-2026-10-15-run)
11. [Results (2026-10-16 run): response-locked](#11-results-2026-10-16-run-response-locked)
12. [Results (2026-10-22 runs): block-split adaptation](#12-results-2026-10-22-runs-block-split-adaptation)

## 1. What it computes

In each window (64 samples every 16 at 256 Hz, i.e. 250 ms every 62.5 ms, over
0–1.5 s) and for every lPFC electrode, pooled across participants:

- **Congruency axis c:** mean HG on incongruent minus congruent trials, the two
  incongruent-proportion blocks weighted equally. **Switch axis s:** the same
  for switch minus repeat, over the switch-proportion blocks. Mean differences
  in HG units (`effect_measure='mean_diff'`), not Cohen's d, which would
  re-weight electrodes by their own SD.
- Each participant's trials are split into halves A and B, stratified on all
  four design factors, 200 times. **One split per participant, shared by all of
  its electrodes**, so electrode i's half A shares no trials with electrode j's
  half B.

Then, per window, across electrodes, averaged over the 200 splits:

| Quantity | Definition | Plot as |
|---|---|---|
| **cross** | ½ [cos(c_A, s_B) + cos(c_B, s_A)] | the "transfer" |
| **ceilings** | cos(c_A, c_B), cos(s_A, s_B) | – |
| **ceiling** | √(congruency ceiling × switch ceiling), NaN unless both > 0 | the within-contrast ceiling |
| **alignment** | cross / ceiling: 0 separate, 1 as shared as noise allows | the share of the ceiling |
| `*_pearson` | the same on electrode-centred vectors (Pearson r, the anatomy's number) | the dashed line |

Two tests, both needed for "partly shared":

- **Anything shared?** The congruency labels are shuffled within each
  participant's (switch type × incProp × switchProp) cells, one permutation per
  participant applied to all of its electrodes, the data rescored, and the raw
  cross cosine recorded; then the same for switch type. A window is **shared**
  where the cross cosine beats **both** nulls, cluster-corrected over windows.
- **Not one code?** Participants are resampled with replacement and everything
  recomputed. A window is **not one code** where the 95% CI of
  ceiling − cross excludes 0.

A window is **reliable** where both ceilings' 95% CIs exclude 0. Elsewhere
there is no measurable code, so its cosines mean nothing, and the figure greys
it out.

### 1.1 The subspace test (plan §3.4)

On each split, stack a half's per-window axes side by side: C_A, C_B (congruency)
and S_A, S_B (switch type), each electrodes × windows over 0–1.5 s. Then:

| Quantity | Definition |
|---|---|
| **subspace** of a matrix | its top k left singular vectors, from an SVD without centring (`coding_subspace`) |
| **principal angles** θ₁ ≤ … ≤ θ_k | cosines = singular values of U_Cᵀ U_S. θ₁ small: some direction is shared; all small: the same subspace |
| **alignment index** A(C → S) | the share of the congruency effect inside the switch subspace, out of the most any k-dim subspace could hold: ‖U_Sᵀ C‖² / (σ₁² + … + σ_k² of C). Also S → C |
| **cross** | congruency from one half against switch from the other, both pairings averaged |
| **ceilings** | the same between the two halves of one variable |
| **normalized** | cross / √(congruency ceiling × switch ceiling) |
| **k** per variable | its number of leading dimensions whose held-out variance (‖u_jᵀ M_B‖² with u_j from half A) beats its shuffle null |
| **primary k** | the smaller of the two k |

All are averaged over the 200 splits and reported for k = 1 to `K_MAX` (3).
The tests mirror §1: **shared** where the raw cross value beats both shuffle
nulls (the same shuffles as the cosine null, scored once for both); **not one
code** where the participant-bootstrap CI of ceiling − cross excludes 0. A row
is **reliable** where each ceiling beats its own variable's shuffle null.
These numbers are never negative, so "CI excludes 0" would pass even for
noise. The pre-specified tests are θ₁ and the two alignment indices at the
primary k; the other rows are descriptive.

### 1.2 The trajectories (plan §4)

On each split: the axes c and s from half A, averaged over an axis window
(0–1.5 s for the main figure); x = c/|c| and y = the part of s perpendicular to
c, unit length (Gram–Schmidt, congruency first). On half B: each electrode's
mean HG in the four congruency × switch-type cells in every window, each the
equal-weight mean of its four proportion blocks, centred by subtracting the
mean of the four cells. The cells' coordinates are their dot products with x
and y over the pooled electrodes. Then the halves are swapped, and everything
is averaged over both directions and the 200 splits, with a participant-bootstrap
CI. The halves are the coding axes' own.

The job also reports how much of the centred cells' activity the plane holds:
raw (a lower bound, since noise fills all ~400 directions) and noise-corrected
(cross-half products, which remove independent noise). Axis windows 0.1–0.5 s
and 0.6–1.4 s give the plan's early-vs-late variant.

### 1.3 Adaptation as block-split separations (plan §6, supplement)

On the same axes, halves and electrodes as the trajectories: within each of
the four proportion blocks (incProp × switchProp), the held-out congruency
separation, ĉ · [mean(IS, IR) − mean(CS, CR)], and the switch separation,
ŝ · [mean(IS, CS) − mean(IR, CR)], each on its own unit axis from the other
half, averaged over both directions and the 200 splits. From these: the change
in each separation with incProp and with switchProp (low − high, the other
proportion weighted equally), and own − other (congruency: incProp −
switchProp; switch: switchProp − incProp). The double dissociation predicts
each variable's own change away from 0 and the other at 0. Participant-bootstrap
CIs only; no shuffle null.

**The same as trajectories** (`proportion_trajectories`). Each variable crossed
with each proportion gives four conditions: the variable's two levels in the
25 % and 75 % blocks of that proportion (i25, c25, i75, c75 for congruency ×
incProp; i/c in 25 % and 75 % switch blocks for congruency × switchProp; and
likewise s/r for switch type), every other factor weighted equally. Their
means on the held-out half are drawn on half A's congruency and switch axes,
centred on the mean of all conditions, then swapped and averaged over the same
200 splits. The coordinates are **oblique**: each condition's activity in the
plane is written as a·ĉ + b·ŝ, so a change along the congruency axis moves only
x and one along the switch axis only y, even though the axes are at cos ≈ 0.4.
The number per set is the change in separation, (level 1 − level 0) at 25 %
minus at 75 %, along each axis. Prediction: in the own sets (congruency ×
incProp, switch × switchProp) it changes along the own axis only; in the cross
sets not at all.

## 2. Quick start

All commands run from `dcc_scripts/subspace/` on the DCC.

```bash
cd ~/coganlab/$USER/GlobalLocal/dcc_scripts/subspace

# 1. dry run on synthetic data (a minute or two; checks the pipeline and paths)
DATA_SOURCE=synthetic N_SPLITS=20 N_PERM=50 N_BOOT=200 bash submit_coding_axes_dcc.sh

# 2. the main result and every control, six jobs
bash submit_coding_axes_all.sh

# 3. when all six have finished, the comparison figure and table
python plot_coding_axes_controls.py \
    results/<EPOCHS_ROOT_FILE>/coding_axes/windows_0.0to1.5s_w64s16_all_lpfc_mean_diff_splits200_perm1000x20_seed0 \
    results/<EPOCHS_ROOT_FILE>/coding_axes/*_seed0_remove_mean \
    results/<EPOCHS_ROOT_FILE>/coding_axes/*_seed0_rt_adjust \
    results/<EPOCHS_ROOT_FILE>/coding_axes/*_seed0_rt_match10 \
    results/<EPOCHS_ROOT_FILE>/coding_axes/*_seed0_rt_random10 \
    results/<EPOCHS_ROOT_FILE>/coding_axes/windows_0.0to1.5s_w64s16_all_occ_mean_diff_splits200_perm1000x20_seed0
```

List the main run first: runs are drawn in the order given. The figure and
`coding_axes_controls.csv` are written next to the run directories (`--out DIR`
to put them elsewhere).

Each job is one `sbatch` (8 cores, 64 GB, 12 h limit). Expected run time on the
real lPFC data: loading the epochs, then about 20 min for the two
1000-permutation nulls (about 0.5 s per permutation), plus roughly 10–15 min
for the subspace test's participant bootstrap (2000 draws of 800 SVDs each;
timed at about 0.3 s per draw on a 4-core machine, not yet on the DCC). The
trajectories add a minute or two. Logs go to `out/`.

## 3. The runs: main result and controls

| Run | Command | Question | Read against |
|---|---|---|---|
| Main | `bash submit_coding_axes_dcc.sh` | Are the two codes shared, and when? | – |
| Overall activity | `REMOVE_MEAN=1 bash submit_coding_axes_dcc.sh` | Is the sharing a shift that moves all of a participant's electrodes together? Subtracts each participant's mean across its electrodes, per trial and window (A4's `remove_mean`). **The plan says to run this first**: the early-separation claim depends on it | Main |
| RT, regression | `RT_CONTROL=adjust bash submit_coding_axes_dcc.sh` | Is the sharing RT? Removes each electrode's pooled within-cell HG–RT slope from every window (as the segregation run's `RT_ADJUST_HG`) | Main |
| RT, matching | `RT_CONTROL=match bash submit_coding_axes_dcc.sh` | The same, by keeping only trials whose RT distributions match across the four congruency × switch-type cells, per participant (A4's `RT_MATCH=rt`, 10 quantile bins). Keeps about half the trials | **Random subset**, not Main |
| RT, random subset | `RT_CONTROL=random bash submit_coding_axes_dcc.sh` | Matching's control: as many trials per participant and cell, drawn without regard to RT | RT matching |
| Region | `ROIS=occ bash submit_coding_axes_dcc.sh` | Is the shared part lPFC-specific? A4 found that occipital cortex also transfers | Main |

`submit_coding_axes_all.sh` submits all six. Settings combine
(e.g. `REMOVE_MEAN=1 RT_CONTROL=adjust`), and anything you put in front of
`submit_coding_axes_all.sh` applies to all six jobs. Every control applies to
the subspace test and the trajectories too: the RT control and `remove_mean`
act on the trials before anything is scored.

Not built yet: the plan's lPFC-subsampled-to-occipital's-54-electrodes control.

## 4. Settings

Environment variables read by `run_coding_axes_dcc.py`; defaults in brackets.
The output directory name records every one that changes the coding-axis
numbers. `K_MAX`, `TRAJ_AXES` and `TRAJ_FIRST` change only the subspace and
trajectory outputs, and are recorded in `run_config.json` only.

| Variable | Default | What it does |
|---|---|---|
| `EPOCHS_ROOT_FILE` | set in the submit script | which HG epochs to load |
| `DATA_SOURCE` | `real` | `synthetic` for a dry run (planted shared code, RTs) |
| `WINDOWS_TMIN`, `WINDOWS_TMAX` | `0.0`, `1.5` | the stretch the windows tile (s). Keep `WINDOWS_TMIN` a multiple of 62.5 ms so the windows are A4's |
| `WINDOW_SIZE`, `STEP_SIZE` | `64`, `16` | samples per window, samples between window starts |
| `REMOVE_MEAN` | `0` | `1` = the overall-activity control |
| `RT_CONTROL` | `none` | `adjust`, `match` or `random` |
| `RT_MATCH_BINS` | `10` | quantile RT bins per participant for `match` / `random` |
| `ELECTRODES` | `all` | `all` or `sig` (task-significant) electrodes |
| `ROIS` | `lpfc` | config ROI name(s), comma-separated; `occ` for occipital |
| `N_SPLITS` | `200` | split halves for the real labels |
| `N_PERM` | `1000` | shuffles per factor (the null) |
| `N_NULL_SPLITS` | `20` | split halves per shuffle |
| `N_BOOT` | `2000` | participant resamples |
| `ALPHA` | `0.05` | test level; CIs are the central 1 − α |
| `SEED` | `0` | every random draw derives from it (`run_config.json` → `seeds`) |
| `K_MAX` | `3` | subspace dimensions compared (k = 1 … `K_MAX`) and tested for reproducibility |
| `TRAJ_AXES` | `0.0:1.5,0.1:0.5,0.6:1.4` | trajectory axis windows (s), `tmin:tmax`, comma-separated; one figure each. The first is the main figure |
| `TRAJ_FIRST` | `congruency` | the axis Gram–Schmidt keeps whole (x); `switch` swaps the roles |

## 5. Outputs

Each run writes to
`dcc_scripts/subspace/results/<EPOCHS_ROOT_FILE>/coding_axes/windows_<tmin>to<tmax>s_w<size>s<step>_<electrodes>_<rois>_mean_diff_splits<N>_perm<N_PERM>x<N_NULL_SPLITS>_seed<SEED>[_remove_mean][_rt_adjust|_rt_match<bins>|_rt_random<bins>]/`:

| File | Contents |
|---|---|
| `coding_axis_cosines.png` | the figure (§6) |
| `coding_axis_cosines.csv` | one row per window: everything in §1, the bootstrap CIs (`*_lo`, `*_hi`), the null band (`null_lo_<factor>`, `null_hi_<factor>`), pointwise p (`p_<factor>`) and the marks (`reliable`, `shared_<factor>`, `shared`, `not_one_code`) |
| `summary.json` | the number for the text: the alignment averaged over the reliable windows with its bootstrap CI; the shared and not-one-code windows |
| `per_split_windows.csv` | the coding axes themselves: `mxA`/`mxB` (congruency), `myA`/`myB` (switch type) per electrode, split and window. Enough to recompute every cosine without reloading epochs (`coding_axes.compare_coding_axes`) |
| `null_cross.npz` | the null: the raw cross cosine per shuffle and window, one array per factor |
| `run_config.json` | every setting and seed, electrode and participant counts, the window grid |
| `rt_adjust_slopes.csv` | (`adjust`) each electrode's HG-on-RT slope and within-cell correlation, per window |
| `rt_match_contrasts.csv`, `rt_match_summary.csv` | (`match`; `rt_random_*` for `random`) per-participant RT differences (incongruent − congruent, switch − repeat) before and after selection, and their t-test across participants |
| `subspace_test.png` | the subspace figure (§6.1) |
| `subspace_test.csv` | one row per `measure` (`angle`, `alignment`), `k` and `component` (angle index, or `congruency_to_switch` / `switch_to_congruency`): `cross`, `congruency_ceiling`, `switch_ceiling`, `ceiling`, `normalized`, the angles in degrees (`*_deg`), bootstrap CIs (`*_lo`, `*_hi`), the null thresholds (`null_hi_<factor>`), pointwise p (`p_<factor>`), the marks (`shared_<factor>`, `shared`, `reliable_congruency`, `reliable_switch`, `reliable`, `not_one_code`) and `primary` |
| `subspace_dimensions.csv` | per variable and dimension: held-out variance (`heldout`, and as a fraction of the held-out total), the null threshold, p, `reproducible`, `kept` |
| `subspace_summary.json` | k per variable, the primary k, and at the primary k: θ₁ and both alignment indices with their CIs and marks |
| `null_subspace.npz` | per factor (`<factor>_cross`, `<factor>_ceiling`, `<factor>_dims`): every shuffle's cross values, the shuffled variable's ceilings and its per-dimension held-out variance |
| `trajectories_<axes>.png`, `trajectories_<axes>_uncentred.png` | the trajectory figure for each axis window (§6.2), e.g. `trajectories_0.0-1.5s.png` |
| `trajectories.csv` | one row per axis window (`axes`), `cells` (`centred` / `uncentred`), `cell` and window: `x`, `y`, and for centred cells the bootstrap CI (`x_lo` … `y_hi`), with the window grid |
| `trajectory_fractions.csv` | per axis window and window: `plane_fraction` (raw) and `plane_fraction_signal` (noise-corrected); `window = -1` is all windows together, with bootstrap CIs |
| `trajectory_summary.json` | electrodes and participants used, electrodes dropped (per participant), and per axis window: the cosine between its two axes before Gram–Schmidt, the overall fractions with CIs, and the raw fraction's chance level 2 / n_electrodes |
| `block_separations_<axes>.png` | the §6 figure for each axis window (§6.3) |
| `block_separations.csv` | one row per axis window, `variable` (`congruency`, `switchType`), block (`inc_prop`, `switch_prop`: `low` / `high`) and window: `separation` (HG along the variable's unit axis) with bootstrap CI (`lo`, `hi`); `window = -1` is the mean over windows |
| `block_contrasts.csv` | one row per axis window, variable, `contrast` (`incProp`, `switchProp`: low − high; `own_minus_other`) and window: `estimate`, `lo`, `hi`, `excludes_0`; `window = -1` is the mean over windows |
| `block_summary.json` | the overall (window −1) contrasts with CIs per axis window and variable, electrodes and participants used |
| `proportion_trajectories_<axes>.png` | the 2 × 2 proportion-trajectory figure for each axis window (§6.3) |
| `proportion_trajectories.csv` | one row per axis window, `set` (`congruency x incProp`, `congruency x switchProp`, `switchType x incProp`, `switchType x switchProp`), `own`, `condition` (e.g. `i25`; the number is the set's proportion) and window: `congruency`, `switch` (oblique coordinates, HG) with bootstrap CIs |
| `proportion_changes.csv` | one row per axis window, set, `axis` (`congruency`, `switch`) and window: the change in separation (25 % − 75 %) along that axis, `estimate`, `lo`, `hi`, `excludes_0`; `window = -1` is the mean over windows |
| `proportion_summary.json` | per axis window: cos(c, s) and, per set, `own` and the overall changes with CIs |

`plot_coding_axes_controls.py` writes `coding_axes_controls.png` and
`coding_axes_controls.csv` (one row per run: alignment over its reliable
windows with CI, the stretch it is shared and not one code, electrode counts).

## 6. Reading the results

**A run's figure** (`coding_axis_cosines.png`) uses the same layout as the A4
cross-decoding panel:

- top panel: **cross** (solid blue) against the **ceiling** (orange). The grey
  band is the shuffle null, and its top edge is the test's threshold under
  whichever of the two shuffles is stricter. The dashed blue line is the Pearson
  version of cross. Bars above the curves: blue where something is **shared**,
  orange where it is **not one code**. Grey background: unreliable windows.
- bottom panel: **alignment** (cross / ceiling) with its participant-bootstrap
  CI, drawn on reliable windows only.

| What you see | Reading |
|---|---|
| cross above the band and below the ceiling, both bars | **partly shared**: the expected outcome |
| cross above the band, at the ceiling (no orange bar) | one shared code at this noise level. Not expected; A4's ceilings sit above its transfers |
| cross inside the band, ceilings reliable | separate codes in those windows: no shared direction detectable |
| ceilings unreliable (grey) | no code to compare; say nothing about those windows |
| solid cross well above the dashed Pearson cross | much of the sharing is a shift common to all electrodes; check it against `REMOVE_MEAN` |

The number for the text is `summary.json` → `alignment_reliable_mean` with
`alignment_reliable_ci`: the share of the ceiling, averaged over the reliable
windows. Report the cross cosine and ceiling alongside it, not the ratio alone.

**The controls figure** (`coding_axes_controls.png`) has three panels: the
cross cosine per run, the alignment per run (its share of its own ceiling, so
runs with different ceilings compare fairly), and a strip showing where each
run is reliable (thin line) and shared (thick line). Readings:

| Control | If the sharing… | it means |
|---|---|---|
| `REMOVE_MEAN` | disappears | it is a shift common to all of a participant's electrodes, not a pattern |
| `RT_CONTROL=adjust` | shrinks | part of it is RT-linked. Conservative: if a neural effect reaches RT through the same trial-by-trial coupling, it is removed too |
| `match` vs `random` | drops under `match` but not under `random` | that part is RT-linked; a drop under both is just the halved sample |
| `ROIS=occ` | is also there | the shared component is not lPFC-specific (as A4 found) |

Compare timing too: the plan expects the codes to be separable early and to
converge after about 0.6 s, as in A4. Report the first shared window from the
strip, or `shared_from_s` in the controls table.

### 6.1 The subspace test

`subspace_test.png`:

- **a, principal angles; b, alignment index.** One column per row of
  `subspace_test.csv`, labelled by k and component. Blue dot: the raw cross
  value with its bootstrap CI, filled where it is **shared** (above both shuffle
  nulls). Orange tick: the ceiling. Grey bar: the stricter null's threshold.
  Light band: the primary k. Faded: a ceiling not above its own null, so that
  row is not interpretable.
- **c, d, how many dimensions.** Held-out variance per dimension (log scale),
  filled where it is kept; the grey tick is the null's 95th percentile. The
  panel title gives k.

| What you see at the primary k | Reading |
|---|---|
| θ₁ shared and not one code; both alignment indices low-to-middling | **partly shared**: at least one direction in common, carrying that share of each effect. The expected outcome (plan §3.4) |
| θ₁ not above the nulls, ceilings reliable | separate subspaces at this noise level. Required before saying "orthogonal" (plan §7), and only with high ceilings |
| everything at the ceiling (no "not one code") | one shared code. Not expected |
| k = 0 for a variable (summary `primary_k` null) | that variable's code is not reproducible across halves; there is no subspace to compare |

For the text: `subspace_summary.json` → `theta_1` (`cross_deg`, `normalized`
with CI) and the two `alignment_*` entries. Quote the raw value, the ceiling
and the normalized value together, as for the cosine. Then read the controls'
runs: if θ₁ stops being shared under `REMOVE_MEAN=1` or RT matching, the shared
direction is the overall-activity or slowing component (plan §3.4).

### 6.2 The trajectories

`trajectories_0.0-1.5s.png`:

- **a:** the four centred condition means in the plane, time as the path (open
  circle = first window, arrow = last). Colour = congruency, line style = switch
  type. x is the congruency axis; y is the part of the switch axis
  perpendicular to it.
- **b, c:** x(t) and y(t) with the participant-bootstrap CI.
- The footer gives the plane fractions and cos(c, s) of the two axes.

Incongruent and congruent separate along x, and switch and repeat along y.
**Where the codes are shared, the switch cells also move along x**: the
solid lines (switch) sit to the right of the dashed ones (repeat) of the same
colour. Compare `trajectories_0.1-0.5s.png` and `trajectories_0.6-1.4s.png`:
if the codes converge late, the late axes' cosine is higher and the x offset
larger. The uncentred figures keep the evoked response (supplement).

The trajectories are a picture of the subspace test's k = 1 case, not a test.
Positions shrink toward 0 when the axes are noisy (held-out projections onto a
noisy axis are attenuated), so compare shapes and timing, not absolute sizes
across runs with different ceilings.

### 6.3 The block-split separations

`block_separations_0.0-1.5s.png`:

- **a, c:** each variable's separation in the four proportion blocks. Colour =
  its own proportion (blue low, orange high), line style = the other (solid
  low, dashed high). The prediction: the colours apart, the line styles
  together.
- **b, d:** the change with the own proportion (black) and with the other
  (grey dashed), low − high, with the participant-bootstrap CI. Positive = a
  larger separation in the low-proportion blocks, the behavioural direction of
  LWPC and LWPS.
- The footer gives the means over windows with CIs; the number to quote is
  `own_minus_other` at `window = -1` in `block_contrasts.csv`.

Expect noise: mean single-electrode LWPC and LWPS are near zero (d = 0.01 and
0.05), and each block × cell has a few trials per half. Keep it in the
supplement unless the own change clearly beats the other.

`proportion_trajectories_0.0-1.5s.png`: rows = variable, columns = proportion,
own sets on the diagonal (a, d), cross sets off it (b, c). x = congruency axis,
y = switch axis, both oblique. Colour = the variable's level (blue i / s,
orange c / r), line style = the proportion (solid 25 %, dashed 75 %); time runs
along each path from the open circle to the arrow. Read the **gap** between
the blue and orange paths of the same line style: the prediction is that in
(a) the solid gap and the dashed gap differ along x only, in (d) along y only,
and in (b, c) not at all. A shift of a whole solid pair against the dashed pair
is the block's overall activity difference, not adaptation. The footer gives
the change in separation along each axis, mean over windows, with CIs.

## 7. Checks before trusting a run

1. **Ceilings.** Most of 0.1–1.4 s should be reliable. If the ceilings are
   near 0 everywhere, the codes are not measured well enough and nothing else
   in the run can be read.
2. **RT matching worked.** In `rt_match_summary.csv`, `mean_after` should be a
   few ms, down from about 150–200 ms (`mean_before`). In
   `rt_random_summary.csv` the RT difference should still be there, which is
   the point of that control.
3. **Electrode counts.** `n_electrodes` in `summary.json` should be close to
   the lPFC count (398). Electrodes are dropped when a cell has fewer than 2
   trials in some half, which RT matching makes more likely; if many are
   dropped, lower `RT_MATCH_BINS`.
4. **Null resolution.** With `N_PERM=1000` the smallest pointwise p is 1/1001.
   A dry run with `N_PERM=50` can't get below about 0.02; don't read its p
   values.
5. **k.** `subspace_summary.json` → `k`. If a variable has k = 0, its code does
   not replicate across halves and the subspace rows mean nothing for it. If
   `primary_k_capped` is true, every tested dimension replicated; raise `K_MAX`
   to see where it stops.
6. **Trajectory electrodes.** `trajectory_summary.json` →
   `n_electrodes_dropped`. The block-balanced cells need at least one trial in
   each of the 16 congruency × switch × block cells in each half; the rarest
   (incongruent switch trials in the 25 %-incongruent, 25 %-switch block) has
   about 5 per participant. A participant with a cell of 0–1 trials loses all of
   its electrodes from the trajectories (not from the tests).
7. **Plane fractions.** The raw fraction should be well above its chance level
   (`plane_fraction_chance`); the noise-corrected one says what share of the
   condition-dependent signal the picture shows. A low corrected fraction means
   much of the condition-dependent activity is outside span{c, s} (an
   interaction, or patterns that change over time; see k).

## 8. Code map and tests

| Piece | Where |
|---|---|
| Window means, the A4 grid | `window_means` in [`src/analysis/stats/stability_flexibility_segregation.py`](../src/analysis/stats/stability_flexibility_segregation.py) |
| Coding axes (fast, mean differences, shared halves, `remove_mean`) | `main_effect_mean_diffs` (same module). Gives exactly the numbers of the general per-window path `compute_sensitivities_per_window`, which handles any scalar measure |
| `mean_diff` effect measure | `_mean_diff`, `_interaction_mean_diff` (same module); also selectable in the segregation job (`EFFECT_MEASURE=mean_diff`) |
| Cosines, ceilings, alignment | `coding_axis_cosines` in [`src/analysis/stats/coding_axes.py`](../src/analysis/stats/coding_axes.py). For a single window, `map_reliability(method='cosine')` gives the same numbers |
| RT control | `rt_control` (reuses `rt_adjust_hg`, `rt_match`, `count_matched_random`) |
| Shuffle null | `permute_within_cells`, `cross_cosine_null` |
| Cluster correction | `shared_windows` |
| Bootstrap, marks, summary | `compare_coding_axes` |
| Shuffle draws shared by both nulls | `null_draws` (`coding_axes.py`); `cross_cosine` and `coding_subspaces.subspace_null_stats` score each draw |
| Subspace primitives | `coding_subspace`, `principal_angle_cosines`, `alignment_index`, `heldout_variance` in [`src/analysis/stats/coding_subspaces.py`](../src/analysis/stats/coding_subspaces.py) |
| Subspace point estimates, k, null, bootstrap, marks | `subspace_stats`, `dimension_test`, `subspace_null`, `compare_subspaces` (same module) |
| Shared halves and cell means | `shared_halves`, `half_cell_means`, and `main_effect_mean_diffs(halves=...)` in `stability_flexibility_segregation.py` |
| Trajectories | `axis_window_means`, `cell_block_means`, `trajectory_sums`, `condition_trajectories` in [`src/analysis/stats/coding_trajectories.py`](../src/analysis/stats/coding_trajectories.py) |
| Block-split separations (plan §6) | `block_separations` (same module) |
| Proportion trajectories (plan §6) | `proportion_trajectories` (same module); figure `plot_proportion_trajectories` |
| Job, per-run figure, controls figure | [`dcc_scripts/subspace/coding_axes_dcc.py`](../dcc_scripts/subspace/coding_axes_dcc.py), `run_coding_axes_dcc.py`, `plot_coding_axes_controls.py` |
| Subspace and trajectory figures | [`dcc_scripts/subspace/subspace_figures.py`](../dcc_scripts/subspace/subspace_figures.py) |

`coding_axes.py`, `coding_subspaces.py` and `coding_trajectories.py` only
compute (in-memory tables in, tables out); loading, saving and plotting are in
the `dcc_scripts/subspace` files.

Tests (run from the repo root in the `ieeg` environment):

```bash
python -m pytest -q tests/analysis/stats/test_coding_axes.py \
                    tests/analysis/stats/test_per_window_sensitivities.py \
                    tests/analysis/stats/test_coding_subspaces.py \
                    tests/analysis/stats/test_coding_trajectories.py
```

(`python -m pytest`, not bare `pytest`, unless the repo is pip-installed: bare
`pytest` does not put the repo root on the import path.)

They pin:
- the fast scorer against the loop;
- `remove_mean`;
- the cosine against planted geometry and the §3.1 worked example, and the ceiling guard;
- agreement with `map_reliability`;
- the shuffle's invariants and the null's reproducibility;
- the RT control;
- the cluster test against a correlated null;
- the bootstrap and marks on a planted shared code;
- the controls figure;
- the subspace primitives against SciPy and planted geometry, and the plan's
  two worked examples (PCA drops a sustained pattern; Spearman of PC1s
  misreads swapped PC order);
- k = 1 reducing to the cosine; the ceiling normalization through noise;
- the dimension test finding planted ranks; a shared direction marked shared
  and not one code; orthogonal codes not marked shared;
- the trajectories against a direct one-split computation; cells at half the
  planted effects; equal block weights; the plane fractions;
- the block separations against a direct one-split computation; a planted
  adaptation of each code landing on its own proportion only;
- the proportion trajectories putting each planted change on its own axis in
  oblique coordinates, with partly shared axes;
- the whole job end to end on synthetic data (marked `slow`).

## 9. Deviations from the plan, and limits

- **Cluster correction by mass, not A4's run length.** Each null draw is one
  shuffle scored in every window, so a high draw is high in every window. The
  null's longest runs then span the whole stretch, and A4's length test can
  never pass: on a synthetic run it marked nothing where the effect was
  obvious, and its answer changed with its random seed. The cluster-mass test
  (Maris & Oostenveld 2007) also counts how far a run sits above the threshold,
  which a high null draw does not match. Same pointwise step as A4 (the null's
  95th percentile); say "cluster-mass corrected" in the Methods.
- **The null band in the figure is the 5th–95th percentile, not a 95% range.**
  Its top edge is then exactly the one-sided threshold the test uses.
- **"Shared" needs both nulls:** the cross cosine has to beat both the "no
  congruency code" and the "no switch code" null.
- **Null draws average 20 splits; the real cross cosine averages 200.** Fewer
  splits only widen the null, so the test is conservative.
- **Trials with a missing sample.** The windows need each trial's time course,
  so an electrode's trial is dropped if any sample in 0–1.5 s is missing. The
  0–1.5 s segregation runs average around missing samples, so this analysis
  uses slightly fewer trials.
- **RT matching is over the four congruency × switch-type cells per
  participant, not within proportion blocks** (A4's default). The coding axes
  weight the proportion blocks equally, so the matched subset can leave some
  block × condition cells thin. Check the electrode counts (§7).
- **Mean differences only.** The job scores no other measure: only mean
  differences have the fast scorer the null needs. For per-window Cohen's d,
  use `compute_sensitivities_per_window`.

Subspace test and trajectories:

- **The ceiling normalization is exact at k = 1 and a heuristic above it.**
  At k = 1 the normalized angle cosine is |§3.2's alignment| and the
  normalized index its square. For k > 1 the geometric mean of the two
  variables' ceilings is applied per angle and per index. On a planted shared
  2-D subspace it averages 0.92–1.06 across noise levels, but single estimates
  scatter (about ±0.05 at a raw value of 0.5, ±0.15 near 0.1). Lean on the
  raw values and the shuffle test at k > 1.
- **"Reliable" for subspace rows is a ceiling above its own shuffle null,**
  not §3.2's "CI excludes 0". Principal-angle cosines and alignment indices
  are never negative, so the CI rule would pass for noise.
- **The primary k is the smaller of the two variables' k,** so both subspaces
  are reproducible at it. The plan says "the largest at which half B's effects
  keep more variance in half A's subspace than under the null" without saying
  whose. Each dimension is tested on its own held-out variance, in order,
  stopping at the first failure.
- **No correction across rows.** The rows are not a time series. θ₁ and the
  two alignment indices at the primary k are the pre-specified tests; report
  the rest as descriptive.
- **The bootstrap uses all 200 splits per draw;** the null draws average 20
  (`N_NULL_SPLITS`), which only widens the null (conservative, as in §3.2).
- **Trajectory cells weight the four proportion blocks equally,** which needs
  a trial in each of the 16 congruency × switch × block cells in each half
  (check 6). Electrodes without one are dropped from the trajectories only.
- **The plane fraction comes in two versions.** The plan asks for "the
  fraction of the centred cells' variance in the plane". The raw version is
  that, but noise fills every electrode direction, so it is mostly a measure
  of signal-to-noise. The noise-corrected version (cross-half products) is the
  one to quote for "the 2-D view is not a thin slice".
- **Block separations (§6) have bootstrap CIs, no shuffle null,** and use the
  trajectories' electrodes (check 6). Each is projected on its own unit axis,
  not on the Gram–Schmidt y, so the switch separation is the same whichever
  axis comes first. `own_minus_other` compares the two proportions within the
  same electrodes, which cancels overall responsiveness (both adaptations
  scale with it).
- **Proportion trajectories are in oblique coordinates.** Distances on the
  plot are not Euclidean distances in electrode space; the coordinates answer
  "how far along the congruency axis and how far along the switch axis".
  Bootstrap CIs only, no shuffle null. They use the trajectories' electrodes.

## 10. Results (2026-10-15 runs)

*Drafted by Claude from the run outputs; not yet reviewed.*

### 10.1 What was run

Every `coding_axes_real` job used the defaults of §4 (0–1.5 s, 64/16-sample
windows, 21 windows, mean differences, 200 splits, 1000 × 20 null, 2000
participant resamples, seed 0) unless a seed or bin count is named. Three
batches, all COMPLETED:

1. **Morning**, 11 jobs, 10–20 min each (58726994, 58727006, 58727013,
   58727094, 58727099, 58727101, 58727134, 58727136, 58727140, 58727164,
   58727165). These ran before the subspace test and the trajectories were
   added to the job, so their folders hold the cosine outputs only.
2. **Afternoon**, 20 jobs, 14–25 min (58740222–58740241): RT matching and its
   random-subset control with `SEED` 1–4 (`rt_match10`, `rt_random10`), and
   with `RT_MATCH_BINS=5` at seed 0 (`rt_match5`, `rt_random5`), for all and
   task-significant lPFC. Then 58741497–58741498 reran the two main runs with
   the subspace test and the trajectories.
3. **Evening**: 58758317–58758318 reran the seed-0 `rt_match10` and
   `rt_random10` for all lPFC with the current code, so they also have the
   subspace and trajectory outputs. Their cosine numbers came out identical to
   the morning runs. 58759949 (29 min) did the same for `remove_mean` (all
   lPFC), again with identical cosine numbers.

Outputs are in
`dcc_scripts/subspace/results/<EPOCHS>/coding_axes/windows_0.0to1.5s_w64s16_<run>_mean_diff_splits200_perm1000x20_seed<SEED>[_control]/`.
† marks folders that have the subspace test and the trajectories:

| `<run>` (folder) | Electrodes / participants | Runs |
|---|---|---|
| `all_lpfc` | 398 / 22 | main only |
| `all_dlpfc-lpfc` | 398 / 22 (DLPFC ⊂ lPFC, so the union is the lPFC set: `all_lpfc` is the same main run) | main†, `remove_mean`†, `rt_adjust`, `rt_match10`† and `rt_random10`† at seeds 0–4, `rt_match5`†, `rt_random5`† |
| `all_dlpfc` | 277 / 21 | main only |
| `sig_dlpfc-lpfc` | 171 / 21 (task-significant lPFC) | main†, `remove_mean`, `rt_adjust`, `rt_match10` and `rt_random10` at seeds 0–4 (seeds 1–4†), `rt_match5`†, `rt_random5`† |

Not run: the occipital control (`ROIS=occ`), and `plot_coding_axes_controls.py`
(no `coding_axes_controls.png`/`.csv` yet). The numbers below come from each
run's `summary.json`, `coding_axis_cosines.csv`, `subspace_summary.json`,
`subspace_dimensions.csv`, `trajectory_summary.json`, `trajectories.csv` and
`rt_*_summary.csv`.

### 10.2 Key numbers: the coding-axis cosines

"Reliable" windows are those where both ceilings' 95 % CIs exclude 0;
alignment is `alignment_reliable_mean` (cross / ceiling, averaged over reliable
windows) with its participant-bootstrap 95 % CI; cross and ceiling are averaged
over the same windows. Window times are window centres.

| Run | Reliable windows | Cross | Ceiling | Alignment [95 % CI] | Shared (cluster-mass, both nulls) | Not one code (centres) |
|---|---|---|---|---|---|---|
| all lPFC, main | 15 / 21 (0.50–1.38 s) | 0.27 | 0.37 | **0.69 [0.57, 0.89]** | 0.56–1.38 s | 0.50, 0.81–1.00 s |
| all lPFC, `remove_mean` | 16 | 0.24 | 0.35 | 0.67 [0.48, 0.89] | 0.56–1.38 s | 0.50, 0.75–1.19 s |
| all lPFC, `rt_adjust` | 13 | 0.15 | 0.25 | 0.58 [0.37, 0.96] | 0.44–1.19 s | 0.50, 0.94–1.00 s |
| all lPFC, `rt_random10`, seed 0 (seeds 1–4: §10.3) | 9 | 0.10 | 0.28 | 0.36 [0.06, 0.90] | 1.06–1.38 s | 0.75–1.13 s |
| all lPFC, `rt_match10`, seed 0 (seeds 1–4: §10.3) | **1** | – | – | not interpretable (one window) | none | none |
| DLPFC, main | 15 | 0.27 | 0.40 | 0.66 [0.52, 0.85] | 0.63–1.38 s | 0.50–0.56, 0.75–1.13 s |
| sig lPFC, main | 16 | 0.32 | 0.50 | 0.61 [0.48, 0.78] | 0.56–1.38 s | 0.44–1.25 s (14 windows) |
| sig lPFC, `remove_mean` | 16 | 0.24 | 0.37 | 0.62 [0.46, 0.92] | 0.63–1.38 s | 0.44–0.50, 0.75–1.13 s |
| sig lPFC, `rt_adjust` | 13 | 0.16 | 0.36 | 0.45 [0.17, 0.83] | 0.44–1.13 s | 0.38–0.75, 1.00–1.06 s |
| sig lPFC, `rt_random10`, seed 0 | 11 | 0.13 | 0.36 | 0.36 [−0.02, 0.80] | 1.13–1.38 s | 0.75–1.19 s |
| sig lPFC, `rt_match10`, seed 0 | **0** | – | – | – | none | none |

Main all-lPFC run, per window (`coding_axis_cosines.csv`): the ceiling rises
from 0.19 at 0.50 s to 0.50 at 1.00 s and falls to 0.36 at 1.38 s; the cross
cosine rises from 0.05 to 0.36 (1.13 s) and stays there. The Pearson cross is
0.00–0.06 below the cosine. The main run has no reliable window before
0.50 s (the congruency ceiling is 0.05 or less up to 0.38 s).

RT matching did what it should (§7.2): the incongruent − congruent RT gap went
from 163 ms to 0.2 ms and switch − repeat from 188 ms to 0.7 ms across
participants (`rt_match_summary.csv`), while the random subset kept 166 and
189 ms. The same holds at every seed: −1.4 to 1.9 ms after matching with 10
bins. With 5 bins, 6–8 ms are left (t = 1.8–2.4 across participants), so
10 bins is the setting to use.

### 10.3 RT matching over five seeds

One random draw of each subset was not enough to read (morning batch, §10.7).
These are seeds 0–4. Ceiling and alignment are averaged over each run's
reliable windows. "θ₁" is the subspace test at the primary k (§10.4), given as
normalized [CI] with "sh" when it beats both shuffle nulls; "k = 0" means a
variable had no reproducible dimension, so there was no subspace to compare.

All lPFC (368–398 electrodes, 21–22 participants):

| Seed | Matched: reliable windows | Ceiling | Alignment [95 % CI] | Shared | k (C/S) | Random: reliable windows | Ceiling | Alignment [95 % CI] | Shared | θ₁ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.10 | (one window) | none | 1/0 | 9 | 0.28 | 0.36 [0.06, 0.90] | 1.06–1.38 s | 0.31 [0.24, 0.70] |
| 1 | 3 | 0.16 | 0.52 [−0.01, 1.01] | 0.75–1.06 s | 1/0 | 11 | 0.24 | 0.88 [0.73, 1.48] | 0.56–1.38 s | 0.78 [0.69, 0.97] sh |
| 2 | 3 | 0.13 | −0.04 [−1.42, 0.66] | none | 0/1 | 11 | 0.26 | 0.63 [0.47, 1.21] | 0.94–1.38 s | 0.60 [0.36, 0.90] sh |
| 3 | 6 | 0.13 | 0.50 [−0.21, 1.26] | 0.75–1.00 s | 1/0 | 12 | 0.28 | 0.79 [0.66, 1.24] | 0.62–1.38 s | 0.76 [0.68, 0.96] sh |
| 4 | 3 | 0.15 | 0.72 [0.02, 1.38] | none | 1/0 | 10 | 0.22 | 0.87 [0.69, 1.47] | 0.69–1.38 s | 0.76 [0.51, 1.18] sh |
| 0, 5 bins | 0 | – | – | none | 1/1 | 11 | 0.27 | 0.64 [0.37, 1.08] | 0.56–1.38 s | 0.67 [0.60, 0.96] sh |

Task-significant lPFC (152–171 electrodes, 20–21 participants; the seed-0
folders are from the morning batch and have no subspace output):

| Seed | Matched: reliable windows | Ceiling | Alignment [95 % CI] | Shared | θ₁ (k = 1 for both) | Random: reliable windows | Ceiling | Alignment [95 % CI] | Shared | θ₁ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | – | – | none | – | 11 | 0.36 | 0.36 [−0.02, 0.80] | 1.13–1.38 s | – |
| 1 | 3 | 0.22 | 0.49 [−0.11, 0.94] | 0.75–0.94 s | 0.76 [0.45, 1.07] sh | 8 | 0.35 | 0.81 [0.69, 1.28] | 0.56–1.38 s | 0.74 [0.62, 0.93] sh |
| 2 | 0 | – | – | none | 0.66 [0.47, 1.24] | 11 | 0.36 | 0.58 [0.43, 1.08] | 0.88–1.38 s | 0.63 [0.49, 0.94] sh |
| 3 | 2 | 0.18 | 0.38 [−0.35, 1.92] | none | 0.36 [0.29, 0.93] | 12 | 0.38 | 0.66 [0.45, 1.14] | 0.75–1.38 s | 0.71 [0.55, 0.96] sh |
| 4 | 4 | 0.19 | 0.74 [0.09, 1.55] | none | 0.54 [0.38, 1.12] | 9 | 0.33 | 0.64 [0.49, 1.00] | 0.75–1.38 s | 0.68 [0.49, 0.90] sh |
| 0, 5 bins | 7 | 0.16 | 0.31 [−0.23, 1.09] | none | 0.51 [0.35, 1.18] | 11 | 0.42 | 0.59 [0.45, 0.94] | 0.50–1.38 s | 0.54 [0.42, 0.84] sh |

Summary across the five 10-bin seeds:

- **Random subsets keep the codes and the sharing.** All lPFC: 9–12 reliable
  windows, ceilings 0.22–0.28, shared in every seed, alignment 0.36–0.88
  (median 0.79), θ₁ shared in 4 of 5 seeds (seed 0 fails, at 0.31 of its
  ceiling). Sig lPFC: 8–12 reliable windows,
  alignment 0.36–0.81 (median 0.64), θ₁ shared in all 4 seeds with a subspace
  output. **Seed 0 was the low draw in both electrode sets**, so the morning
  note that the random subset "weakens the sharing" was one draw (§10.7).
- **Matched subsets lose the codes.** All lPFC: 1–6 reliable windows (median
  3), ceilings 0.10–0.16, about half the random subsets'. In every seed one
  variable has k = 0, switch type in 4 of 5, so there is no subspace test. Sig
  lPFC: 0–4 reliable windows. Both variables keep one reproducible trial-wide
  dimension (k = 1 in seeds 1–4), but θ₁ beats the nulls in only 1 of 4 seeds,
  at θ₁ ceilings of 0.24–0.34 against 0.51–0.60 for the random subsets.
- **Trajectories under either subset are not readable.** The block-balanced
  cells need a trial in all 16 congruency × switch × block cells in each half
  (§7.6). Halving the trials drops between a quarter and three quarters of
  the electrodes (all lPFC: 175–221 of 398 kept under matching, 109–282 under
  random). Under matching the noise-corrected plane fractions have CIs as wide
  as ±6.

### 10.4 The subspace test (main runs)

From `subspace_summary.json` and `subspace_dimensions.csv`. Angle and
alignment-index rows are cross / ceiling = normalized [95 % CI]. All
primary-k rows are shared (p = 0.001 against both shuffles) and "not one
code" (CI of ceiling − cross excludes 0).

| Run | k (congruency / switch) | Primary k | θ₁: cos (deg), ceiling, normalized | Alignment index C → S | S → C |
|---|---|---|---|---|---|
| all lPFC | 3 / 2 | 2 | 0.45 (63°), 0.61, **0.74 [0.66, 0.93]** | 0.165 / 0.284 = 0.58 [0.46, 0.89] | 0.154 / 0.284 = 0.54 [0.45, 0.86] |
| all lPFC, `remove_mean` | 3 / 1 | 1 | 0.41 (66°), 0.59, **0.69 [0.51, 0.87]** | 0.174 / 0.355 = 0.49 [0.31, 0.76] | 0.177 / 0.355 = 0.50 [0.32, 0.77] |
| sig lPFC | 1 / 2 | 1 | 0.51 (59°), 0.74, **0.69 [0.57, 0.86]** | 0.270 / 0.558 = 0.48 [0.36, 0.75] | 0.271 / 0.558 = 0.49 [0.36, 0.75] |

`remove_mean` lowers the primary k to 1 because switch's second dimension
just misses (p = 0.054). Compare it with the main run at the same k = 1:
θ₁ 0.42 (65°), normalized 0.72 [0.55, 0.94]. Removing the mean activity
leaves the shared direction essentially unchanged.

- **Each code is dominated by one pattern.** In all lPFC, dimension 1 carries
  21 % (congruency) and 16 % (switch) of a half's total sum of squares, 18 and
  12 times its shuffle threshold. Dimensions 2–3 carry about 1 % each and pass
  only narrowly (p = 0.01–0.03). The total is mostly noise. The signal,
  estimated as ⟨M_A, M_B⟩ (independent noise averages out), is 35 % and
  30 % of it. So dimension 1 carries roughly 60 % (congruency) and 52 %
  (switch) of the reproducible signal. That is a lower bound, since half A's
  direction is itself noisy, and it assumes isotropic noise in the held-out
  projection. The rest is spread over directions that each replicate weakly
  or not at all.
- **The second principal angle is not interpretable.** At k = 2 (all lPFC),
  θ₂ fails the reliability check: congruency's own θ₂ ceiling (0.06) is not
  above its shuffle null. So do θ₂ and θ₃ at k = 3. All the evidence for
  sharing is in θ₁.
- The alignment index is on a variance scale: 0.58 normalized corresponds to
  about √0.58 ≈ 0.76 on the cosine scale, the same picture as θ₁.

### 10.5 The trajectories (main runs)

From `trajectory_summary.json`. The raw fraction is the share of the centred
cells' variance in the plane, with chance at 2 / electrodes. The signal
fraction is the noise-corrected version (§9).

| Run (electrodes / participants used) | Axis window | cos(c, s) | Raw fraction (chance) | Signal fraction [95 % CI] |
|---|---|---|---|---|
| all lPFC (389 / 21) | 0–1.5 s | 0.38 | 0.110 (0.005) | **0.52 [0.40, 0.63]** |
| | 0.1–0.5 s | −0.03 | 0.017 | 0.01 [−0.01, 0.07] |
| | 0.6–1.4 s | 0.43 | 0.126 | 0.61 [0.51, 0.70] |
| all lPFC, `remove_mean` (389 / 21) | 0–1.5 s | 0.37 | 0.102 (0.005) | 0.54 [0.45, 0.62] |
| | 0.1–0.5 s | −0.04 | 0.014 | 0.03 [−0.00, 0.07] |
| | 0.6–1.4 s | 0.41 | 0.112 | 0.60 [0.50, 0.66] |
| sig lPFC (165 / 20) | 0–1.5 s | 0.48 | 0.216 (0.012) | **0.69 [0.58, 0.77]** |
| | 0.1–0.5 s | −0.12 | 0.031 | 0.03 [−0.02, 0.12] |
| | 0.6–1.4 s | 0.51 | 0.230 | 0.74 [0.63, 0.81] |

All lPFC, 0–1.5 s axes, centred cells at 1.0 s (`trajectories.csv`, x = the
congruency axis, y = switch beyond it):

| Cell | x | y |
|---|---|---|
| incongruent-switch | 1.22 | 0.68 |
| incongruent-repeat | 0.37 | −0.30 |
| congruent-switch | −0.26 | 0.20 |
| congruent-repeat | −1.33 | −0.57 |

Along x the four cells line up incongruent-switch > incongruent-repeat >
congruent-switch > congruent-repeat. Switch moves along the congruency axis as
well, by about 0.96 against congruency's 1.60 (differences of the two-cell
means). Along y, switch separates from repeat by 0.88. Congruency also
separates by 0.38 on y, because one time-averaged congruency axis does not
capture the congruency pattern at every time. Sig lPFC gives the same order
and similar values. The uncentred figures show all four cells riding the
evoked response up and to the right (both axes load positively on it) and
back. Incongruent-switch reaches furthest and stays up longest.

### 10.6 How to read the figures

`coding_axis_cosines.png` in each folder: x is the window centre (s from
stimulus onset). Top panel, y is the cosine across electrodes: the solid blue
line is the cross cosine (does half A's congruency axis point where half B's
switch axis points?), orange the ceiling (how well each axis reproduces itself
across halves), dashed blue the Pearson version. The grey band is the 5th–95th
percentile of the label-shuffle null, so a cross cosine above its top edge
beats the "no congruency code" or "no switch code" null in that window; the
blue bar marks windows that do so against both nulls and survive cluster-mass
correction over windows. Orange bars mark windows where the participant
bootstrap CI of ceiling − cross excludes 0 ("not one code"). Grey background:
unreliable windows, where nothing should be read. Bottom panel: alignment with
its 95 % participant-bootstrap CI, 0 = separate, 1 = as shared as noise allows.

`subspace_test.png` and `trajectories_*.png`: see §6.1 and §6.2.

### 10.7 Reading

- **Partly shared, in the late part of the trial.** In all lPFC the cross
  cosine beats both shuffle nulls from 0.56 s to the end of the window, and
  sits at about 0.7 of the ceiling, with the CI excluding 1 in 5 windows
  (0.50 s and 0.81–1.00 s; pointwise, not corrected across windows). This is
  the "partly shared" outcome §6 calls expected. It agrees in kind with A4
  (transfer from +0.62 s, 26–47 % kept), and the alignment is close to the
  plan's noise-corrected anatomy estimate (~0.57, `analysis_plans.md` ›
  Subspace plan §1). DLPFC and the task-significant subset give the same
  picture (0.66, 0.61).
- **One shared direction, carried by each code's dominant pattern** (§10.4).
  Over the whole trial the subspace test gives θ₁ = 63° at 0.74 [0.66, 0.93] of
  its ceiling: shared, and not one code. Each code is essentially one electrode
  pattern whose strength changes over time, and θ₂ cannot be measured. So
  "partly shared" means the dominant congruency pattern and the dominant switch
  pattern overlap at about 0.7–0.75 of what noise allows. Nothing points to a
  second shared direction. This is plan §7's first or second row, depending on
  the controls.
- **The trajectories show the same thing** (§10.5). On held-out cells, switch
  moves along the congruency axis by about 60 % as much as congruency does,
  and the c–s plane holds about half of the reproducible condition-dependent
  signal (0.52 [0.40, 0.63]; 0.61 with late axes; 0.69–0.74 in sig lPFC). Axes
  from 0.1–0.5 s carry no signal (≈1 %), which agrees with there being no
  reliable code before 0.5 s.
- **Not a common shift.** `remove_mean` barely changes it (0.69 → 0.67, all
  lPFC; 0.61 → 0.62, sig), and the Pearson cross is close to the cosine. The
  subspace test agrees: θ₁ is 0.72 of its ceiling without the control and 0.69
  with it (both at k = 1), and the plane still holds 0.54 of the signal. As in
  A4, the shared component is a pattern, not every electrode rising together.
  `remove_mean` subtracts each trial's mean across electrodes, so it does not
  rule out a gain on the evoked pattern. The uncentred trajectories show both
  axes loading on the evoked response (§10.5).
- **Early separation cannot be assessed here.** The plan expected separable
  codes early, converging after ~0.6 s. With mean-difference axes there is no
  reliable window before 0.50 s in the main run (0.375 s at the earliest, under
  `rt_adjust`), so "separate early" is not testable
  with this measure; A4's decoders found congruency from +0.12 s. Do not
  report "no sharing early" from these runs.
- **RT matching removes the codes, and five seeds confirm it is not the
  smaller sample** (§10.3). Matched subsets keep 1–6 reliable windows (all
  lPFC) with ceilings of 0.10–0.16, and one variable has no reproducible
  dimension in every seed. Random subsets of the same size keep 9–12
  reliable windows, ceilings of 0.22–0.28 and the sharing (alignment median
  0.79). Read literally: when RT is equated across the four congruency ×
  switch-type cells, the congruency and switch mean differences in lPFC HG
  mostly disappear, and with them anything to share. That is stronger than
  plan §7's first row ("what they share is a common slowing component"): it is
  the late codes themselves, not only their shared part, that go with RT.
- **What is left after matching is too weak to call shared or separate.** In
  sig lPFC each variable keeps one reproducible trial-wide dimension under
  matching, but θ₁ beats the nulls in 1 of 4 seeds, at ceilings about half the
  random subsets'. At that ceiling the test has little power, so this is not
  evidence that an RT-independent part is separate.
- **The morning reading of the random subset was one draw.** Seed 0 gave 0.36
  in both electrode sets; seeds 1–4 give 0.63–0.88 (all) and 0.58–0.81 (sig).
  The subsetting itself does not weaken the sharing.
- **Still unreconciled: regression and A4.** `rt_adjust` keeps the sharing
  (0.58 [0.37, 0.96], shared 0.44–1.19 s), and A4's matched decoders still
  decoded. Three possible reasons, none tested:
  1. Regression removes only a linear within-cell HG–RT slope, while matching
     removes the condition differences in RT whatever their shape. If late HG
     lasts until the response, the condition effect in late stimulus-locked
     windows partly *is* the RT difference, and the slope removes only part of
     it.
  2. A decoder can use differences that a block-weighted mean difference
     cannot.
  3. Matching leaves some block × condition cells thin (§9; under matching
     the block-balanced trajectory cells lose about half the electrodes).

  A response-locked run would address the first; a trial count per
  participant × block × condition cell in the matched subset, the third.

### 10.8 Caveats

- RT matching and its random control have five seeds. Everything else is seed
  0 only.
- 22 participants. The null treats trials within participant as exchangeable
  and the bootstrap resamples participants, so the late sharing is a
  participant-level claim, but with 22 participants the CIs are wide
  (alignment 0.57–0.89, θ₁ 0.66–0.93).
- "Not one code" is a pointwise test over windows. "Shared" is
  cluster-corrected.
- The occipital control and the controls figure are missing, so
  "lPFC-specific" cannot be addressed yet.
- The sharing is strongest at 0.9–1.4 s, near or after many responses (A4's
  median RT 1.17 s). Stimulus-locked windows that late mix in response-related
  activity. Given §10.3, a response-locked version is now the most useful next
  run.
- The ceiling normalization of the subspace rows is a heuristic for k > 1
  (§9). The primary k is 2 in all lPFC, so quote θ₁'s raw cosine and ceiling
  alongside the ratio.

## 11. Results (2026-10-16 run): response-locked

*Drafted by Claude from the run outputs; not yet reviewed.*

The question from §10.7–10.8: the stimulus-locked sharing is strongest at
0.9–1.4 s, near or after many responses (median RT 1.17 s; 35 % of trials
answered by 1.0 s), RT matching removes the codes, and `rt_adjust` keeps the
sharing. Aligning every trial to its response asks where the codes and their
sharing sit relative to the button press.

### 11.1 What was run

Six `coding_axes_real` jobs from `bash submit_coding_axes_all.sh`, all
COMPLETED, exit 0: 58783103 (main, 57 min), 58783104 (`REMOVE_MEAN=1`, 58 min),
58783105 (`RT_CONTROL=adjust`, 57 min), 58783106 (`RT_CONTROL=match`, 10 bins,
49 min), 58783107 (`RT_CONTROL=random`, 10 bins, 48 min), 58783108
(`ROIS=occ`, 17 min). Settings as §4 except:

- `EPOCHS_ROOT_FILE=Response_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20`:
  0 s is the response.
- `WINDOWS_TMIN=-1.0`, `WINDOWS_TMAX=1.5`: 37 windows of 250 ms every 62.5 ms,
  centres −0.875 to +1.375 s. The window centred at −0.125 s ends at the
  response. Windows after 0 s include feedback and the next trial's
  preparation. On trials answered within 1.0 s, the first windows reach back
  before the stimulus.
- `TRAJ_AXES=-1.0:0.0,-1.0:-0.5,-0.5:0.0`: the main trajectory figure uses
  axes from the second before the response; the other two are the early and
  late halves of that second.
- `ROIS=dlpfc,lpfc` (the all-lPFC set, 398 electrodes / 22 participants, as in
  §10); the region control is all occipital electrodes (83 / 14).
- Seed 0, 200 splits, 1000 × 20 shuffles, 2000 participant resamples,
  `K_MAX=3`.

Outputs are in
`dcc_scripts/subspace/results/Response_…_zmax_20/coding_axes/windows_-1.0to1.5s_w64s16_all_<rois>_mean_diff_splits200_perm1000x20_seed0[_remove_mean|_rt_adjust|_rt_match10|_rt_random10]/`,
with `<rois>` = `dlpfc-lpfc` or `occ`. The controls figure and table
(`coding_axes_controls.png`, `coding_axes_controls.csv`) were made on
2026-10-16 with `plot_coding_axes_controls.py` (main run first) and written to
the `Response_…/coding_axes/` folder. Their x axis reads "s from response".

Three differences from the stimulus-locked runs matter when comparing:

1. **The subspace test spans the whole epoch** (all 37 windows, −1.0 to
   1.5 s), not 0–1.5 s. No window after +0.125 s has a reliable code, so the
   subspaces are dominated by the pre-response windows, but the rows mix
   pre- and post-response time.
2. **Code versions.** 58783103–05 started at 08:18. Edits to
   `coding_trajectories.py` and `coding_axes_dcc.py` (block separations, §1.3)
   were saved at 09:12–09:15, and 58783106–08 started at 09:15–09:17, so only
   those three folders have `block_separations*` and `block_*` files.
   `coding_axes.py` and `coding_subspaces.py` were last changed on 2026-10-15,
   so the cosine and subspace numbers of all six runs come from the same code.
   The trajectory electrode rule is the same in both versions (an electrode is
   dropped if any cell of any block is empty in a half). The block separations
   exist only for runs whose trajectories are unreadable (below) and are not
   reported.
3. **One seed.** The stimulus-locked RT runs have seeds 0–4 (§10.3). These have
   seed 0 only, and seed 0 was the low random-subset draw there.

### 11.2 Key numbers: the coding-axis cosines

Columns as in §10.2. Times are window centres in s from the response. The
stimulus-locked counterparts (§10.2) are in the last column.

| Run | Electrodes / participants | Reliable windows | Cross | Ceiling | Alignment [95 % CI] | Shared | Not one code | Stimulus-locked alignment |
|---|---|---|---|---|---|---|---|---|
| all lPFC, main | 398 / 22 | 16 / 37 (−0.875 to +0.06) | 0.25 | 0.41 | **0.59 [0.42, 0.83]** | −0.875 to −0.06 | −0.81 to −0.125; 0.0–0.06 | 0.69 [0.57, 0.89] |
| `remove_mean` | 398 / 22 | 20 (−0.875 to +0.125; 0.69–0.81) | 0.17 | 0.32 | 0.45 [0.25, 0.67] | −0.875 to −0.06 | −0.875 to +0.125; 0.75–0.81 | 0.67 [0.48, 0.89] |
| `rt_adjust` | 398 / 22 | 13 (−0.75 to 0.0) | 0.14 | 0.26 | 0.52 [0.30, 0.80] | −0.75 to −0.25 | −0.375 to −0.125; 0.0 | 0.58 [0.37, 0.96] |
| `rt_random10`, seed 0 | 398 / 22 | 8 (−0.56 to −0.125) | 0.09 | 0.24 | 0.36 [−0.04, 0.93] | −0.75 to −0.375 | −0.375 to −0.19 | 0.36 [0.06, 0.90] |
| `rt_match10`, seed 0 | 397 / 22 | **3** (−0.5, −0.375, −0.31) | 0.10 | 0.14 | not interpretable (three windows) | none | none | one window |
| occipital | 83 / 14 | **0** | – | – | – | none | none | not run |

Main run, per window (`coding_axis_cosines.csv`): the ceiling is 0.42 at
−0.875 s, 0.47–0.49 from −0.81 to −0.375 s, then falls to 0.26 at −0.06 s and
0.23 at +0.06 s. From +0.125 s on no window is reliable (switch ceiling ≤ 0.15,
congruency ceiling ≤ 0.30). The cross cosine peaks at 0.37 at −0.69 to
−0.56 s and falls to 0.13 at −0.06 s and 0.08 at 0 s. The alignment is
0.75–0.78 at −0.69 to −0.50 s, 0.65 at −0.31 s, 0.47 at −0.06 s and 0.32
[−0.02, 0.78] at 0 s (not shared there; p = 0.06 and 0.05 against the two
shuffles). The Pearson cross is 0.04–0.07 below the cosine before the
response, a larger gap than stimulus-locked (0.00–0.06).

**RT matching worked** (§7.2): incongruent − congruent went from 162.9 ms to
−0.35 ms and switch − repeat from 187.6 ms to −0.38 ms across participants
(`p_after` = 0.86 for both; `rt_match_summary.csv`). The random subset kept
169.7 and 181.8 ms (p < 10⁻⁴; `rt_random_summary.csv`). Both kept 47 % of the
electrode-trials (73,061 and 73,000 of 154,334).

### 11.3 The subspace test

From `subspace_summary.json`, `subspace_test.csv` and
`subspace_dimensions.csv`. Rows are raw cross / ceiling = normalized [95 % CI],
at the primary k; "sh" = beats both shuffle nulls.

| Run | k (C / S) | Primary k | θ₁: cos (deg), ceiling, normalized | Alignment index C → S | S → C |
|---|---|---|---|---|---|
| all lPFC, main | 3 / 3 (capped) | 3 | 0.48 (61°), 0.64, **0.76 [0.70, 0.93]**, sh (p = 0.001), not one code | 0.143 / 0.253 = 0.56 [0.50, 0.85] sh | 0.148 / 0.253 = 0.58 [0.47, 0.91] sh |
| `remove_mean` | 3 / 3 (capped) | 3 | 0.44 (64°), 0.59, 0.74 [0.67, 0.89], sh, not one code | 0.46 [0.41, 0.73] sh | 0.57 [0.50, 0.84] sh |
| `rt_adjust` | 3 / 3 (capped) | 3 | 0.37 (68°), 0.46, 0.79 [0.72, 1.00], sh (p = 0.013, 0.002), not one code | 0.67 [0.56, 0.96] sh | 0.61 [0.53, 1.00] sh |
| `rt_random10`, seed 0 | 3 / 2 | 2 | 0.22 (77°), 0.43, 0.51 [0.47, 0.94], **not shared** (p = 0.31, 0.16) | 0.39 [0.34, 0.94], not shared | 0.41 [0.35, 0.90], not shared |
| `rt_match10`, seed 0 | **0 / 0** | – | no reproducible dimension for either variable | – | – |
| occipital | **0 / 0** | – | – | – | – |

- **θ₁ matches the stimulus-locked value.** At k = 1, for a like-for-like
  comparison: θ₁ 0.42 (65°) against a ceiling of 0.57, normalized 0.735
  [0.46, 1.00]; stimulus-locked, 0.42 (65°), 0.72 [0.55, 0.94] (§10.4).
- **More reproducible dimensions.** Dimension 1 carries 14 % (congruency) and
  10 % (switch) of a half's held-out sum of squares, 11 and 7 times the
  shuffle threshold; dimensions 2–3 carry 1.3–2.4 % each and all pass
  (p ≤ 0.003), so k is capped at 3 for both (raise `K_MAX` to see where it
  stops). That is expected when the windows cover a time course with a rise,
  a peak and a fall (§11.4), so the pattern is not one fixed pattern scaled
  over time.
- **θ₂ is measurable here** (unlike §10.4): at k = 3, θ₂ = 79° (0.18 against
  a ceiling of 0.30), reliable and above both nulls (p = 0.040, 0.004), but its
  normalized CI includes 1 and the normalization is a heuristic above k = 1
  (§9). Descriptive only.
- **Random subset, seed 0:** θ₁ is shared at k = 1 (p = 0.013 for both, 0.52
  of its ceiling) but not at the primary k = 2. Stimulus-locked seed 0 failed
  the same way (§10.3), so this is the same weak draw, not a new result.

### 11.4 The trajectories

From `trajectory_summary.json`; columns as in §10.5.

| Run (electrodes / participants used) | Axis window | cos(c, s) | Raw fraction (chance) | Signal fraction [95 % CI] |
|---|---|---|---|---|
| all lPFC (391 / 21) | −1.0 to 0 s | 0.45 | 0.085 (0.005) | **0.44 [0.33, 0.59]** |
| | −1.0 to −0.5 s | 0.37 | 0.075 | 0.38 [0.32, 0.51] |
| | −0.5 to 0 s | 0.30 | 0.069 | 0.36 [0.24, 0.51] |
| `remove_mean` (391 / 21) | −1.0 to 0 s | 0.41 | 0.062 | 0.39 [0.31, 0.49] |
| | −1.0 to −0.5 s | 0.35 | 0.054 | 0.33 [0.26, 0.42] |
| | −0.5 to 0 s | 0.23 | 0.050 | 0.31 [0.24, 0.40] |
| `rt_adjust` (391 / 21) | −1.0 to 0 s | 0.25 | 0.036 | 0.22 [0.14, 0.38] |
| | −1.0 to −0.5 s | 0.14 | 0.029 | 0.14 [0.07, 0.28] |
| | −0.5 to 0 s | 0.14 | 0.032 | 0.20 [0.13, 0.36] |
| `rt_match10` (182 / 11), `rt_random10` (201 / 10), occipital (77 / 13) | all | – | – | not readable: signal-fraction CIs about ±0.9 (match), [0.08, 0.99] (random), [−3.8, 9.9] (occipital) |

All lPFC, axes from −1.0 to 0 s, centred cells (`trajectories.csv`;
separations are differences of two-cell means):

| Window centre | Congruency along x | Switch along x (share of congruency's) | Switch along y |
|---|---|---|---|
| −0.625 s | 1.45 | 0.93 (64 %) | 0.82 |
| −0.31 s | 1.32 | 1.00 (76 %) | 0.93 |
| −0.06 s | 0.69 | 0.36 (52 %) | 0.61 |
| +0.25 s | 0.61 | 0.31 | 0.26 |

Up to about −0.3 s the cells line up along x as incongruent-switch >
incongruent-repeat > congruent-switch > congruent-repeat, as stimulus-locked.
Incongruent-switch, the slowest cell, stays furthest out and collapses toward
the others from −0.3 s to the response. The uncentred figure
(`trajectories_-1.0-0.0s_uncentred.png`) shows all four cells on one ramp that
peaks about 0.3 s before the response and is back at baseline by about
+0.4 s. The condition differences stay about the same size up to the peak
(congruency along x 1.53 at −0.875 s, 1.32 at −0.31 s) and shrink over the
0.3 s from the peak to the response. The peak heights are in the same order
as the cells (incongruent-switch highest).

The early-half axes are more aligned than the late-half axes (cos 0.37 vs
0.30). Stimulus-locked it was the other way round (−0.03 early, 0.43 late).
In response time the sharing is earliest, furthest from the response.

### 11.5 Descriptive check: is the shared pattern the within-cell RT pattern?

Not part of the job. It was computed for this write-up from existing outputs,
in both frames: per window, the cosine across the 398 electrodes between each
electrode's pooled within-cell HG-on-RT slope (`rt_adjust_slopes.csv` of the
`rt_adjust` run) and the full-data congruency and switch axes (mean of both
halves over the 200 splits, `per_split_windows.csv` of the main run). The
slope is estimated within the congruency × switch cells, so it is not the
condition difference itself. Raw cosines only: no null and no ceiling, and
noise in either vector pulls them toward 0.

| Frame | Windows (centres) | cos(c, RT slope) | cos(s, RT slope) | cos(c, s), same full-data axes |
|---|---|---|---|---|
| Response | −0.875 to −0.50 s | 0.58–0.66 | 0.55–0.62 | 0.26–0.51 |
| Response | −0.25 to 0.0 s | 0.26–0.37 | 0.34–0.52 | 0.13–0.35 |
| Response | +0.375 s on | −0.04 to 0.18 | −0.07 to 0.19 | −0.09 to 0.13 |
| Stimulus | 0.125–0.56 s | −0.23 to 0.01 | −0.06 to 0.07 | −0.09 to 0.18 |
| Stimulus | 0.94–1.375 s | 0.56–0.63 | 0.57–0.68 | 0.42–0.49 |

In both frames, where the codes are shared, each code's electrode pattern
points at least as much toward the "slow trial within a cell" pattern as
toward the other code. That is what the shared direction would look like if
it were the pattern by which slower trials differ from faster ones. Two
caveats: an uncentred cosine is positive for any two patterns that both scale
with each electrode's responsiveness, and this does not say *why* slow trials
differ (timing or effort). A proper version would be a split-half RT axis
(fast vs slow trials within each cell) tested against a shuffle null and its
own ceiling, in both frames.

### 11.6 How to read the figures

- `coding_axis_cosines.png` (each run folder): as §10.6, but x is the window
  centre in s from the response. The blue bar (shared) runs from −0.875 to
  −0.06 s in the main run; the grey background starts at +0.125 s, where
  neither ceiling is reliable any more.
- `coding_axes_controls.png` (`Response_…/coding_axes/`): top, each run's
  cross cosine; middle, each run's alignment with its participant-bootstrap
  CI, on its reliable windows only; bottom strip, thin line = reliable
  windows, thick line = shared (cluster-mass corrected, both nulls). Read
  `rt_match10` against `rt_random10`, not against the main run. The occipital
  row is empty because no window is reliable.
- `subspace_test.png`: as §6.1. The grey band is at k = 3 because k was capped.
- `trajectories_-1.0-0.0s.png`: as §6.2, with axes fit on the second before
  the response and b, c in s from the response. Compare
  `trajectories_-1.0--0.5s.png` and `trajectories_-0.5-0.0s.png` for early vs
  late axes.

### 11.7 Reading

[`decoding.md` §13.4](decoding.md#134-reading-each-control) gives three
outcomes for a response-locked run. In terms of the coding axes:

| Where the response-locked sharing sits | Reading | Here |
|---|---|---|
| Far before the response, fading toward it, with the ceilings | The latency signature: at a fixed time before the response, slow trials are further past their stimulus | **Closest match** |
| Around the response, with both ceilings there | Something shared in the response period that is not a timing shift | Not seen: at 0 s the alignment is 0.32 and not shared, and no code is reliable after +0.06 s |
| Nowhere | The shared component is locked to the stimulus | Not seen |

- **The codes are present before the response and gone after it.** Response
  locking makes them clearer, not weaker: 16 reliable windows, all before
  +0.125 s, mean ceiling 0.41 against 0.37 stimulus-locked, 0.47–0.49 from
  −0.81 to −0.375 s. They do not continue into feedback or the next trial.
- **The sharing has the latency signature's shape.** It is strongest 0.4–0.75 s
  before the response (alignment 0.75–0.78), falls toward the response (0.47
  at −0.06 s; not shared at 0 s), and the axes from the early half of the
  second are more aligned than those from the late half. In that period the
  slow cells, above all incongruent-switch, sit higher on a common ramp.
  On the 35 % of trials answered within 1.0 s, the first windows also reach
  before the stimulus. All of this is what a shift in timing produces.
- **It is not a clean case.** The sharing lasts to the last pre-response window
  (−0.06 s, ceilings 0.26). And the peak heights of the ramp differ by cell at
  the same time before the response. A pure shift of one fixed waveform would
  not change the peak height. If the slower cells also have more variable RTs
  (usual, but not checked here), smearing would lower their averaged peak,
  while here the slowest cell peaks highest. Activity that grows with time on
  task fits that too. This run
  cannot separate "timing" from "a process that takes longer on slow trials".
- **RT matching again removes the codes** (3 reliable windows against 8 in the
  random subset; ceilings 0.14 against 0.24; k = 0 / 0 against 3 / 2), with RT
  equated to −0.4 ms. This is the stimulus-locked result (§10.3) in the other
  frame. The random subset's *sharing* is weak at seed 0 (0.36, θ₁ not shared
  at the primary k), exactly as stimulus-locked seed 0 was, while stimulus-locked
  seeds 1–4 gave 0.63–0.88. So the codes going with RT holds in this frame too.
  Whether matching removes the sharing beyond what the subsetting does can't
  be read from seed 0.
- **Regression and matching still disagree.** `rt_adjust` keeps the sharing
  (0.52 [0.30, 0.80], −0.75 to −0.25 s; θ₁ 0.79 of its ceiling, shared) while
  it removes the two earliest windows and roughly halves the ceilings. §10.7's
  first explanation was late stimulus-locked HG lasting until the response.
  Since the disagreement survives response locking, it more likely comes from
  how the two controls work: a linear within-cell slope removes only part of a
  nonlinear timing effect, while matching removes the condition RT difference
  whatever its form. §10.7's third reason (thin block × condition cells after
  matching) is still untested.
- **The shared pattern resembles the within-cell RT pattern** (§11.5,
  descriptive): cosines of 0.55–0.68 between each code and the HG-on-RT slope
  wherever the codes are shared, in both frames.
- **Not a common shift, but partly one near the response.** `remove_mean`
  lowers the alignment from 0.59 to 0.45 [0.25, 0.67], more than
  stimulus-locked (0.69 → 0.67), mostly from −0.4 s to the response (0.42–0.53
  against 0.57–0.69). θ₁ barely moves (0.76 → 0.74). The pattern survives; a
  shift common to a participant's electrodes adds to it near the response.
- **Occipital: no measurable code.** In 83 occipital electrodes from 14
  participants no window has both ceilings reliable, and neither variable has a
  reproducible dimension. The raw cross cosine reaches 0.13–0.21 before the
  response and beats the switch shuffle in several windows, but with
  unreliable ceilings that is not readable. So this says "no code
  measured", not "no sharing", and gives no answer on lPFC specificity. There
  is no stimulus-locked occipital run, and this is all occipital, not A4's 54
  task-significant electrodes.

**Overall.** The congruency and switch codes in lPFC HG, and their shared part,
are present before the response, strongest 0.4–0.75 s before it, fading as it
approaches, and gone after it. They still disappear when RT is matched. On
the §13.4 scheme this is closest to the latency signature, and the matching
result and §11.5 point the same way. It does not establish a pure timing
shift: sharing persists up to the response, and the ramp's peak heights
differ by cell. The wording this supports is "tied to RT, present before the
response", not "response-related" and not "a timing artifact".

Three checks would settle it:

1. `rt_match10` / `rt_random10` response-locked at seeds 1–4 (8 jobs, about
   50 min each at 8 cores / 64 GB): whether matching removes the sharing beyond
   the subsetting.
2. A latency-only model: each participant's mean stimulus-locked HG shifted by
   each trial's RT, re-epoched to the response, and run through the same job.
   If it reproduces the ceilings, the sharing and their timing, timing is
   enough. Not built.
3. The split-half RT-axis test of §11.5, with a null and a ceiling, in both
   frames.

### 11.8 Caveats

- Seed 0 only. The RT control pair has one draw, and seed 0 was the low
  random draw stimulus-locked.
- 22 participants, 398 electrodes, the same set as §10; CIs are wide
  (alignment 0.42–0.83).
- The subspace test spans the whole epoch, including post-response windows
  with no reliable code. Primary k is capped at 3; quote θ₁ raw with its
  ceiling, and treat θ₂ as descriptive (§9).
- "Not one code" is pointwise over windows; "shared" is cluster-corrected.
- The first windows mix in pre-stimulus time on fast trials. The windows after
  0 s include feedback and the next trial.
- Trajectories under matching, random subsets and in occipital cortex are
  unreadable: halving the trials leaves 10–11 participants with every
  block × cell in each half, and occipital cortex has no reliable code to
  project on (77 electrodes). In those runs the raw plane fraction also falls
  below its own bootstrap CI (match 0.035 vs [0.037, 0.116]; occipital 0.070
  vs [0.073, 0.208]), a small-sample bias of the bootstrap. Not interpreted.
- §11.5 is descriptive, computed outside the job, with no null.

## 12. Results (2026-10-22 runs): block-split adaptation

*Drafted by Claude from the run outputs; not yet reviewed.*

The question (plan §6): projected onto the coding axes, does the congruency
separation change with incongruent proportion but not switch proportion, and
the switch separation with switch proportion but not incongruent proportion?
How it is computed: §1.3; how to read the figure: §6.3.

### 12.1 What was run

Twelve `coding_axes_real` jobs, all COMPLETED, exit 0:

- **Stimulus-locked** (`bash submit_coding_axes_all.sh`, settings of §10.1):
  59283148 (main, 34 min), 59283149 (`REMOVE_MEAN=1`, 34 min), 59283150
  (`RT_CONTROL=adjust`, 34 min), 59283151 (`match`, 27 min), 59283152
  (`random`, 27 min), 59283153 (`ROIS=occ`, 16 min).
- **Response-locked** (settings of §11.1): 59283438 (main, 61 min), 59283439
  (`REMOVE_MEAN=1`, 62 min), 59283440 (`adjust`, 60 min), 59283441 (`match`,
  54 min), 59283442 (`random`, 53 min), 59283443 (`ROIS=occ`, 21 min).

All with `ROIS=dlpfc,lpfc` (the all-lPFC set) or `occ`, seed 0, 200 splits,
1000 × 20 shuffles, 2000 participant resamples. Submitted with
`RT_CONTROL=none` set explicitly, because commit ff91524 changed the default in
`submit_coding_axes_dcc.sh` to `adjust`; without it the main, `remove_mean` and
occipital jobs would have been RT-adjusted.

They overwrote the seed-0 folders of §10 and §11. **Every file those folders
already had came out byte-identical** (cosines, subspace test, trajectories,
and the three response-locked `block_*` files of §11.1), except
`run_config.json` (new timestamp, plus the `block_bootstrap` seed, 7). So
§10–11 stand as written. New in these folders:

- `block_separations.csv`, `block_contrasts.csv`, `block_summary.json`,
  `block_separations_<axes>.png` in all twelve.
- Stimulus-locked `rt_adjust`: the subspace test and trajectories (its §10
  folder had the cosines only).
- Stimulus-locked occipital: the whole run (§10 did not have it).

Outputs: `dcc_scripts/subspace/results/<EPOCHS>/coding_axes/windows_<tmin>to1.5s_w64s16_all_<rois>_mean_diff_splits200_perm1000x20_seed0[_control]/`.

### 12.2 Key numbers

From `block_summary.json`. Mean over all windows of the run, main axis window
(stimulus-locked 0–1.5 s; response-locked −1.0 to 0 s), 95 % participant-bootstrap
CI. Positive = a larger separation in the low-proportion blocks, the
behavioural direction of LWPC and LWPS. **Bold**: CI excludes 0.

| Run (electrodes / participants) | Congruency: own (incProp) | other (switchProp) | own − other | Switch: own (switchProp) | other (incProp) | own − other |
|---|---|---|---|---|---|---|
| **Stimulus-locked** | | | | | | |
| main (389 / 21) | **+0.33 [0.08, 0.57]** | +0.12 [−0.10, 0.35] | +0.21 [−0.17, 0.62] | **+0.71 [0.21, 1.39]** | −0.09 [−0.28, 0.07] | **+0.80 [0.31, 1.47]** |
| `remove_mean` (389 / 21) | **+0.30 [0.02, 0.54]** | +0.06 [−0.12, 0.26] | +0.24 [−0.18, 0.61] | **+0.46 [0.15, 0.77]** | −0.12 [−0.30, 0.05] | **+0.58 [0.22, 0.96]** |
| `rt_adjust` (389 / 21) | +0.19 [−0.01, 0.36] | +0.17 [−0.02, 0.36] | +0.03 [−0.28, 0.30] | **+0.47 [0.09, 0.94]** | −0.05 [−0.24, 0.13] | **+0.52 [0.13, 1.04]** |
| occipital (80 / 13) | +0.13 [−0.02, 0.22] | −0.10 [−0.22, 0.04] | +0.23 [−0.02, 0.43] | +0.30 [−0.04, 0.53] | −0.09 [−0.23, 0.07] | **+0.39 [0.03, 0.68]** |
| **Response-locked** | | | | | | |
| main (391 / 21) | **+0.56 [0.26, 0.90]** | −0.05 [−0.33, 0.22] | **+0.61 [0.19, 1.12]** | **+0.62 [0.15, 1.23]** | −0.07 [−0.42, 0.28] | **+0.69 [0.10, 1.33]** |
| `remove_mean` (391 / 21) | **+0.44 [0.14, 0.78]** | +0.01 [−0.12, 0.15] | **+0.43 [0.07, 0.82]** | **+0.39 [0.16, 0.62]** | −0.01 [−0.16, 0.12] | **+0.41 [0.18, 0.61]** |
| `rt_adjust` (391 / 21) | **+0.27 [0.11, 0.42]** | −0.03 [−0.28, 0.19] | **+0.31 [0.10, 0.58]** | **+0.32 [0.07, 0.58]** | −0.05 [−0.35, 0.28] | +0.37 [−0.02, 0.76] |

`rt_match10` (stimulus 175 / 10, response 182 / 11), `rt_random10` (199 / 10,
201 / 10) and response-locked occipital (77 / 13) are not interpreted: their
trajectory signal fractions are unreadable (CIs from about [−6, 6] to
[−3.8, 9.9]; §11.8 explains why), and their block numbers disagree between
match and random in ways no single reading explains.

**Per-block separations**, stimulus-locked main, 0–1.5 s axes
(`block_separations.csv`, `window = -1`; incProp, switchProp):

| Block | Congruency separation | Switch separation |
|---|---|---|
| low, low | 1.07 | 1.10 |
| low, high | 1.03 | **0.12** |
| high, low | 0.82 | 0.91 |
| high, high | 0.62 | 0.48 |

Response-locked main gives the same pattern (switch: 0.87, **0.12**, 0.81,
0.32; congruency: 0.70, 0.94, 0.34, 0.19).

**Where in time** (`block_contrasts.csv`, pointwise CIs, uncorrected):

- Stimulus-locked main: the switch own change excludes 0 from 0.625 s to the
  end (13 windows), and own − other in 10 of those 13. Congruency excludes 0 only
  in the last one or two windows (1.31–1.375 s), past the median RT (1.17 s).
- Response-locked main: switch own − other is strongest before the response
  (pre-response mean +1.01; CI excludes 0 in 5 of 14 pre-response windows,
  3 of 23 after). **Congruency own − other sits after the response** (CI
  excludes 0 in 1 of 14 pre-response windows, 12 of 23 after; pre-response
  mean +0.42). The same holds under `remove_mean` (0 of 14 before, 6 of 23
  after) and `rt_adjust` (pre-response mean −0.04).

**New stimulus-locked control outputs** (not §6):

- `rt_adjust` subspace and trajectories: θ₁ 76° (cosine 0.24), 0.63 of its
  ceiling, shared; k = 1 for both variables (main run: k = 3 / 2, θ₁ 63°,
  0.74). Trajectory signal fraction 0.31 [0.15, 0.50] for 0–1.5 s axes (main
  0.52), cos(c, s) 0.24 (main 0.38). Cosine alignment 0.58 [0.37, 0.96] over 13
  reliable windows, shared in windows 5–17 (the cosine outputs are
  byte-identical to §10's `rt_adjust` run).
- Occipital (83 electrodes / 14 participants): **no reliable window** in the
  cosine test, so no alignment. The time-pooled subspace test still finds one
  reproducible dimension per variable, θ₁ 77°, 0.69 of its ceiling, shared.
  Trajectory signal fraction 0.46 [0.18, 0.92].

### 12.3 Reading

- **Switch: the predicted pattern, and it holds up.** In lPFC the switch
  separation is larger in low- than in high-switch-proportion blocks and does
  not change with incongruent proportion. Own − other excludes 0 in both
  frames, survives `remove_mean`, and survives RT adjustment stimulus-locked
  (+0.52 [0.13, 1.04]; response-locked +0.37 [−0.02, 0.76]). Stimulus-locked
  it starts at about 0.6 s, after the switch code itself appears.
- **Congruency: an incProp effect, but no clean specificity.** The congruency
  separation is larger in low-incProp blocks in every lPFC run. Stimulus-locked
  its change with switchProp is not reliably smaller (own − other includes 0),
  and under RT adjustment the two are the same size. Response-locked own −
  other does exclude 0, but it comes from the windows after the response,
  where there is no reliable congruency code (§11) and the windows include
  feedback and the next trial. Before the response it is not there. So the
  congruency half of the dissociation is not shown.
- **The switch effect leans on one block.** The low-incProp, high-switchProp
  block has almost no switch separation (0.12 in both frames, against
  0.32–1.10 elsewhere). Its absence alone makes up much of the switchProp
  change, and it also implies an incProp × switchProp interaction on the switch
  separation. Check before leaning on it: trial counts per cell in that block,
  and whether behavioural switch costs are also smallest there.
- **Occipital shows the switch pattern too** (stimulus-locked own − other
  +0.39 [0.03, 0.68]; response-locked +0.24 [0.00, 0.43]), with weaker codes.
  The switch result is therefore not lPFC-specific. That fits §10–11's finding
  that the shared code is not lPFC-specific either.
- **Relation to Test 1.** This is the trajectory form of Test 1 (each
  adaptation tracks its own base effect), at the population level. It agrees
  for switch and is silent for congruency. It uses the same per-electrode
  effects, so it is not independent of Test 1 (plan §1).

### 12.4 Caveats

- Bootstrap CIs only; no label-shuffle null, no correction across windows or
  across the three axis windows and two frames. The main-axis, mean-over-windows
  numbers are the pre-specified ones; the per-window marks are descriptive.
- The response-locked mean over windows includes 23 post-response windows
  (to +1.4 s) with no reliable code; it is not a pre-response number.
- Seed 0 only. Each block × cell has a few trials per half (the rarest about
  2–3), and nine lPFC electrodes (seven response-locked) are dropped for an
  empty cell.
- Positions along a noisy held-out axis shrink toward 0 (§6.2), so compare the
  own and other changes within a run, not absolute sizes across runs.
