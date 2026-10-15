# Coding axes: runbook

How to run, check and read the coding-axes analysis: the congruency and
switch-type coding axes in every A4 decoding window, and the cosine between
them against a label-shuffle null and a split-half ceiling. The design and the
reasons behind each choice are in
[`analysis_plans.md` › Subspace trajectory plan §3](analysis_plans.md#3-coding-axes-and-the-angle-between-them);
this page only says how to produce and read the numbers.

**Status (2026-10-15):** §3.1–3.3 built and tested on synthetic data; not yet
run on real data. The subspace test (§3.4) and the trajectories (§4) are not
built.

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
real lPFC data: loading the epochs, then about 20 min, almost all of it the two
1000-permutation nulls (about 0.5 s per permutation). Logs go to `out/`.

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
`submit_coding_axes_all.sh` applies to all six jobs.

Not built yet: the plan's lPFC-subsampled-to-occipital's-54-electrodes control.

## 4. Settings

Environment variables read by `run_coding_axes_dcc.py`; defaults in brackets.
The output directory name records every one that changes the numbers.

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
| Job, per-run figure, controls figure | [`dcc_scripts/subspace/coding_axes_dcc.py`](../dcc_scripts/subspace/coding_axes_dcc.py), `run_coding_axes_dcc.py`, `plot_coding_axes_controls.py` |

`coding_axes.py` only computes (in-memory tables in, tables out); loading,
saving and plotting are in the `dcc_scripts/subspace` files.

Tests (run from the repo root in the `ieeg` environment):

```bash
python -m pytest -q tests/analysis/stats/test_coding_axes.py \
                    tests/analysis/stats/test_per_window_sensitivities.py
```

They pin:
- the fast scorer against the loop;
- `remove_mean`;
- the cosine against planted geometry and the §3.1 worked example, and the ceiling guard;
- agreement with `map_reliability`;
- the shuffle's invariants and the null's reproducibility;
- the RT control;
- the cluster test against a correlated null;
- the bootstrap and marks on a planted shared code;
- the controls figure.

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
