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
| Trajectories | `axis_window_means`, `balanced_cell_means`, `trajectory_sums`, `condition_trajectories` in [`src/analysis/stats/coding_trajectories.py`](../src/analysis/stats/coding_trajectories.py) |
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

- **Each code is essentially one pattern.** In all lPFC, dimension 1 carries
  21 % (congruency) and 16 % (switch) of the held-out sum of squares, 18 and 12
  times its shuffle threshold. Dimensions 2–3 carry about 1 % each and pass
  only narrowly (p = 0.01–0.03). The rest of the held-out sum of squares is
  noise.
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
