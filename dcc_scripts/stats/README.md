# Stability vs. Flexibility Segregation — DCC scripts

> **Method & rationale:** see the merged guide
> [`docs/stability_flexibility_guide.md`](../../docs/stability_flexibility_guide.md)
> — §3 (the four-interaction electrode definition), §4 (window-mean vs
> per-timepoint-cluster ANOVA), §5 (conjunction), and §9 (run order + how to run).


Runs the joint-distribution analysis in
`src/analysis/stats/stability_flexibility_segregation.py` on the cluster: do
distinct iEEG subpopulations uniquely support **stability** (congruency / LWPC)
vs. **flexibility** (switch / LWPS), or does a shared core do both?

See `stability_flexibility_segregation_tutorial.ipynb` (next to the analysis
module) for a step-by-step walk-through of the statistics.

## Files

| File | Role |
|---|---|
| `stability_flexibility_segregation_dcc.py` | Core: assembles the long-format single-trial HG table from epoched data, runs the analysis, writes results + a summary figure. Exposes `main(args)`. |
| `run_stability_flexibility_segregation_dcc.py` | Entrypoint: sets parameters (many overridable via env vars) and calls `main`. |
| `sbatch_stability_flexibility_segregation_dcc.sh` | SLURM job wrapper (`conda activate ieeg` → run entrypoint). |
| `submit_stability_flexibility_segregation_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/window/etc. and `sbatch`-submits the job. |

## Quick start

```bash
cd dcc_scripts/stats

# 1) Validate the pipeline + paths in seconds on synthetic data (no data load):
DATA_SOURCE=synthetic bash submit_stability_flexibility_segregation_dcc.sh

# 2) Real run — edit EPOCHS_ROOT_FILE in the submit script to a file you have, then:
bash submit_stability_flexibility_segregation_dcc.sh
```

You can also run the entrypoint directly (e.g. on a login/compute node) without
SLURM, which is handy for a fast local sanity check:

```bash
DATA_SOURCE=synthetic N_SPLITS=40 N_PERM_CORR=1000 N_PERM_LABEL=300 \
    python run_stability_flexibility_segregation_dcc.py
```

## How the data is assembled

For each subject, `load_HG_ev1_rescaled_per_subject` returns one
accuracy-filtered `HG_ev1_rescaled` Epochs object. We window-average HG over
`[WINDOW_TMIN, WINDOW_TMAX]` seconds and read the per-trial `congruency`
(`c`/`i`) and `task_sequence` (`s`/`r`, first-of-block `n` dropped) from the
epochs metadata, plus the block proportions `incongruent_proportion` and
`switch_proportion`, producing the long table the analysis expects:

```
subject | electrode (= subject-channel) | hg | congruency | switchType | incongruent_proportion | switch_proportion
```

With `EFFECT_MEASURE=cluster` the `hg` column instead holds each trial's HG
*time course* over the window (not the window mean), so each contrast can be
scored by its aggregate cluster-mass statistic rather than a difference of means.

## Key knobs (env vars, read by the entrypoint)

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS_ROOT_FILE` | — (required for real runs) | Which epoched HG file to load. |
| `DATA_SOURCE` | `real` | `real` = epoched data; `synthetic` = ground-truth dry run. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | Analysis window (s from stimulus onset). |
| `ELECTRODES` | `all` | `all` or `sig` (significant channels). |
| `CONTRAST_MODE` | `condition` | `condition` = stability from congruency (i vs c), flexibility from switchType (s vs r); `proportion` = stability from the **LWPC** congruency×`incongruent_proportion` interaction and flexibility from the **LWPS** switchType×`switch_proportion` interaction (each a 2×2 difference-of-differences, high vs low block). |
| `EFFECT_MEASURE` | `cohens_d` | `cohens_d` = standardized mean difference on window-mean HG; `cluster` = aggregate cluster-mass statistic on the windowed HG time course. |
| `N_SPLITS` | `200` | Disjoint trial-half resamples for sensitivity estimation. |
| `N_PERM_CORR` | `10000` | Permutations for the continuous test. |
| `N_PERM_LABEL` | `2000` | Permutations per electrode for S/F labeling. |

`CONTRAST_MODE` and `EFFECT_MEASURE` are independent — any of the four
combinations is valid, and both default to the original behaviour. Results are
written under a `..._<CONTRAST_MODE>_<EFFECT_MEASURE>` sub-folder so runs don't
collide. By default the `cluster` mass is a fast deterministic parametric
statistic; set `USE_TIME_PERM_CLUSTER = True` in
`src/analysis/stats/stability_flexibility_segregation.py` to instead use the
real `ieeg.calc.stats.time_perm_cluster` mask (much slower — it permutes on
every call).

To restrict to ROIs, set `ROIS_DICT` in `run_stability_flexibility_segregation_dcc.py`
(a commented LPFC/occipital example is included).

Use a precomputed responsiveness statistic for better gain control by setting
`RESPONSIVENESS` to a `{electrode: baseline-vs-signal cluster stat}` dict in the
entrypoint (defaults to the `mean|HG|` fallback).

## Outputs

Written to `results/<epochs_or_synthetic_tag>/window_<tmin>to<tmax>s_<electrodes>/`:

- `long_df.csv` — the assembled single-trial table.
- `electrodes.csv`, `labels.csv`, `continuous.csv` — per-electrode `x`/`y`,
  responsiveness, S/F labels, and residualized values.
- `correlation.json` — continuous test (corr, p, n).
- `conjunction.json`, `conjunction_per_subject.csv` — CMH odds ratio, p-values,
  pooled 2×2, per-subject tables.
- `segregation_summary.png` — 6-panel figure (joint scatter, residualized
  scatter, within-subject null, selectivity classes, pooled 2×2, per-subject).
- `summary.txt` — printed verdicts.

**Reading:** continuous `corr ≤ 0` / CMH `OR < 1` → **segregation**;
`corr > 0` / `OR > 1` → **shared core**.

---

# A3 — Anatomy of the stability/flexibility subpopulations

Sits on the A1 electrode definition and asks: *are the distinct subpopulations in
different **places**?* — while conditioning every claim on iEEG **coverage**
(clinically-placed electrodes are the main confound here). Analysis code lives in
`src/analysis/stats/stability_flexibility_anatomy.py`; a step-by-step walk-through
is in `stability_flexibility_anatomy_tutorial.ipynb` (next to that module).

## Files

| File | Role |
|---|---|
| `stability_flexibility_anatomy_dcc.py` | Core: assembles the long df, runs A1 to get S/F flags, maps electrodes → ROIs, builds the coverage matrix, runs the coverage-conditioned enrichment test, writes figures + `summary.txt`. Exposes `main(args)`. |
| `run_stability_flexibility_anatomy_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_stability_flexibility_anatomy_dcc.sh` | SLURM wrapper (`conda activate ieeg` → entrypoint). |
| `submit_stability_flexibility_anatomy_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/window/etc. and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/stats
# validate the whole path in seconds with a PLANTED group×ROI association:
DATA_SOURCE=synthetic bash submit_stability_flexibility_anatomy_dcc.sh
# the NULL version (no association — the test must come back n.s.):
DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0.0 bash submit_stability_flexibility_anatomy_dcc.sh
# real run — set EPOCHS_ROOT_FILE in the submit script, then:
bash submit_stability_flexibility_anatomy_dcc.sh
```

## What it does

1. Assembles the same long-format single-trial HG table as the A1/A2 job.
2. **A1** (`per_electrode_anova_labels`, `contrast_mode='proportion'`) → per-electrode `S`/`F` flags → 4-way group (`both`/`S_only`/`F_only`/`neither`).
3. Maps each electrode to a coarse ROI (`build_electrode_roi_map` over the shared `subjects_electrodes_to_ROIs_dict` + `config/rois.py`).
4. **Coverage**: subject × ROI boolean matrix (does subject *s* have any electrode in ROI *r*?).
5. **Coverage-conditioned enrichment test** (`roi_group_enrichment_test`): Pearson χ² on the group × ROI table with a **within-subject permutation null** (shuffle the group label within each subject, so the null respects nesting *and* coverage), restricted to ROIs sampled in ≥ `MIN_SUBJECTS` subjects.
6. Figures: ROI-group histograms (annotated with per-ROI coverage), the coverage heatmap + enrichment null, and a Glasser brain-surface figure via the existing `vis/` renderer (falls back to the ROI histogram off-cluster).

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = epoched data + ROI atlas; `synthetic` = ground-truth dry run. |
| `SYNTHETIC_ENRICHMENT` | `0.6` | synthetic only: strength of the planted group×ROI association (`0.0` = null). |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | analysis window (s from stimulus onset). |
| `ELECTRODES` | `all` | `all` or `sig`. |
| `ALPHA` | `0.05` | A1 FDR threshold for the S/F flags. |
| `MIN_SUBJECTS` | `3` | keep only ROIs sampled in ≥ this many subjects (the coverage condition). |
| `N_PERM` | `10000` | within-subject permutations for the enrichment null. |

## Outputs

Written to `results/<epochs_or_synthetic_tag>/anatomy_window_<tmin>to<tmax>s_<electrodes>/`:

- `anatomy_labels_roi.csv` — per-electrode S/F, ROI, group.
- `coverage_matrix.csv` — subject × ROI coverage.
- `group_roi_contingency.csv`, `roi_group_histogram.csv` — the tables behind the test/figure.
- `roi_enrichment.json` (+ `roi_enrichment_null.npy`) — ROIs tested, χ², permutation p, per-ROI coverage.
- `roi_group_histogram.png`, `anatomy_coverage_enrichment.png`, `selectivity_groups_on_brain.svg` (or `..._roi_hist.png` fallback).
- `summary.txt` — printed verdict.

**Reading:** a significant test means selectivity-group membership is associated
with ROI *beyond* what electrode placement forces; per-ROI coverage is reported so
no claim rests on where the grid happens to be.

---

# A5 — Timing: relative onset of stability vs. flexibility

Does the **LWPC** (stability) interaction arise *earlier* in the trial than the
**LWPS** (flexibility) interaction, or later? A *sequence* question neither the
conjunction (A2) nor the cross-decoding (A4) layer speaks to. Analysis code lives
in `src/analysis/stats/stability_flexibility_timing.py`; a step-by-step
walk-through is in `stability_flexibility_a5_a6_tutorial.ipynb` (next to that
module).

## Files

| File | Role |
|---|---|
| `stability_flexibility_timing_dcc.py` | Core: assembles the long df in **time-course** mode, builds the per-bin interaction traces, measures onsets/peaks, runs the jackknife, writes figures + `summary.txt`. Exposes `main(args)`. |
| `run_stability_flexibility_timing_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_stability_flexibility_timing_dcc.sh` | SLURM wrapper (`conda activate ieeg` → entrypoint). |
| `submit_stability_flexibility_timing_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/window/etc. and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/stats
# validate the whole path in seconds against a PLANTED onset ordering
# (stability at 0.20 s, flexibility at 0.40 s — the job should recover both):
DATA_SOURCE=synthetic bash submit_stability_flexibility_timing_dcc.sh
# the falsification: plant the REVERSE ordering; the reported sign must flip:
DATA_SOURCE=synthetic SYNTHETIC_STAB_ONSET=0.40 SYNTHETIC_FLEX_ONSET=0.20 \
    bash submit_stability_flexibility_timing_dcc.sh
# real run — set EPOCHS_ROOT_FILE in the submit script, then:
bash submit_stability_flexibility_timing_dcc.sh
```

## What it does

0. Runs `_assert_amplitude_invariance` **first** — the latency–amplitude guard as a
   live assertion (scaling a waveform by `k` must not move its 50 %-of-peak onset).
   A failure there would invalidate every onset the job goes on to report.
1. Assembles the long table with `effect_measure='cluster'`, so `hg` holds each
   trial's **time course** over the window, plus the window time axis
   (`window_times`, which also verifies every subject shares one axis — bin-by-bin
   grand-averaging is meaningless otherwise).
2. `interaction_time_course` per process: the **equal-cell-weight
   difference-of-differences** of the four `(cond, mod)` cell means per time bin,
   combined across electrodes. Equal cell weighting keeps the estimate orthogonal
   to both main effects, so the ~75/25 proportion imbalance can't leak a main
   effect in as a fake interaction.
3. `onset_50pct_peak` (onset) and `peak_latency` (shape cross-check) on each trace.
   Normalizing to each effect's **own** peak is what defeats the latency–amplitude
   confound: a bigger effect crosses any *absolute* threshold sooner, so without it
   "earlier" would just rename "larger".
4. `jackknife_onset_difference`: onsets read off **smooth leave-one-subject-out
   grand averages**, jackknife SE, and the Ulrich–Miller `(N−1)`-corrected paired
   *t* on the LWPC − LWPS difference.

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = epoched HG time courses; `synthetic` = planted-onset dry run. |
| `SYNTHETIC_STAB_ONSET` / `SYNTHETIC_FLEX_ONSET` | `0.20` / `0.40` | synthetic only: the planted onsets (s). Swap them for the falsification run. |
| `SYNTHETIC_N_SUBJ` | `12` | synthetic only: number of subjects. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `-0.2` / `0.8` | analysis window (s from stimulus onset) — **wider than the A1/A2 default on purpose**: A5 reads a rising flank, so the window must include the baseline and enough post-stimulus time for both effects to turn over. |
| `ELECTRODES` | `all` | `all` or `sig`. |
| `STATISTIC` | `mean` | `mean` = grand-average the per-electrode d-o-d(t); `t` = t across electrodes (noise-normalized, often a cleaner flank). |
| `ALPHA` | `0.05` | significance threshold for the reported verdict. |

## Outputs

Written to `results/<epochs_or_synthetic_tag>/timing_window_<tmin>to<tmax>s_<electrodes>_<statistic>/`:

- `interaction_time_courses.csv` — `time`, `lwpc`, `lwps`: the per-bin
  difference-of-differences behind every onset (the reusable artifact; the
  time-course long table itself is far too large to serialize).
- `jackknife_leave_one_out.csv` — the `N` leave-one-out onset pairs and differences.
- `onset_difference.json` — full-sample onsets/peaks, the jackknife SE, `t_raw`,
  `t_corrected`, `p`, and the 95 % CI.
- `timing_summary.png` — 3 panels: the two traces with onset/peak markers, the
  leave-one-out onsets, and the distribution of leave-one-out differences.
- `summary.txt` — printed verdict.

**Reading:** the **sign** of the onset difference says which process's information
arises first (negative = stability leads); the CI / `(N−1)`-corrected *t* say
whether that ordering is reliable. Claim an ordering only when **onset and peak
latency agree** — `summary.txt` checks that for you, and warns instead when an
effect is still at its ceiling at `WINDOW_TMAX` (its "peak" is then just the last
bin and carries no latency information — widen the window).

---

# A6 — Brain–behavior correlation

Does the A1 neural selectivity predict the **actual behavioral control
adjustment**? Analysis code lives in
`src/analysis/stats/stability_flexibility_brain_behavior.py`; a step-by-step
walk-through is in `stability_flexibility_a5_a6_tutorial.ipynb`.

## Files

| File | Role |
|---|---|
| `stability_flexibility_brain_behavior_dcc.py` | Core: runs A1 for the S/F flags, derives the per-subject behavioral LWPC/LWPS RT magnitudes, runs both correlation levels with their cross-pairing controls, writes figures + `summary.txt`. Exposes `main(args)`. |
| `run_stability_flexibility_brain_behavior_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_stability_flexibility_brain_behavior_dcc.sh` | SLURM wrapper (`conda activate ieeg` → entrypoint). |
| `submit_stability_flexibility_brain_behavior_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/`BEHAVIOR_CSV`/window/etc. and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/stats
# validate the whole path in seconds — a planted matched coupling that beats its
# cross control at BOTH levels:
DATA_SOURCE=synthetic bash submit_stability_flexibility_brain_behavior_dcc.sh
# the falsification: make each neural group drive BOTH adjustments equally;
# `specificity_ok` must stop holding:
DATA_SOURCE=synthetic SYNTHETIC_CROSS_FRAC=1.0 \
    bash submit_stability_flexibility_brain_behavior_dcc.sh
# real run — set EPOCHS_ROOT_FILE (and BEHAVIOR_CSV if not the repo-root copy):
bash submit_stability_flexibility_brain_behavior_dcc.sh
```

## What it does

1. Assembles the same window-mean long table as the A1/A2/A3 jobs and runs the A1
   electrode definition (`per_electrode_anova_labels`, `contrast_mode='proportion'`)
   → per-electrode `S`/`F` flags.
2. **Behavior**: per-subject LWPC/LWPS RT magnitudes from the raw trial table
   (`combinedData.csv`; `subject_ID` is renamed to `subject` on load) via
   `behavioral_lwpc_lwps_magnitudes` — the **same** equal-cell-weight
   difference-of-differences used for the neural interaction, so brain and behavior
   are measured on the identical contrast.
3. **(1) Across subjects** (`subject_level_brain_behavior`) for all three neural
   summaries — `count` (`n_S`/`n_F`), `frac`, and `effect` (mean interaction F) —
   each with its **cross-pairing** control. Underpowered at *n* = subjects by
   design; reported with that caveat attached.
4. **(2) Within subject, single trial** (`trialwise_brain_behavior`) — the powered
   test. `assemble_trial_table` builds a per-(subject, trial) table with RT and the
   window-mean HG averaged over the LWPC and LWPS electrode groups; the mixed model
   `adjustment ~ group HG` with a subject random intercept is then fit for the
   **matched** and the **cross** adjustment.

### How the trial-level adjustment columns are defined

`trialwise_brain_behavior` deliberately takes the adjustment columns as *input* —
the operationalization is a design choice, so this job makes it explicit
(`add_adjustment_columns`). Each adjustment is the trial's **signed contribution to
the very difference-of-differences the rest of the battery is built on**:

```
adj_congruency(t) = w(t) * (RT_t − mean RT of that subject)
w(t) = +1 for (i, high-incongruent) and (c, low-incongruent)
       −1 for (c, high-incongruent) and (i, low-incongruent)
```

those being exactly the four cell weights of the LWPC d-o-d — so a subject's mean
`adj_congruency` *is* their (trial-count-weighted) behavioral LWPC / 4, and a
positive slope means "trials with more HG in this electrode group push the
behavioral interaction harder". `adj_switch` is the same construction on
switchType × switch_proportion. **RT and the group HG are both centered within
subject**, so the slope is a purely within-subject quantity — with an uncentered
predictor, between-subject differences in mean HG would leak into the common slope,
which is exactly what the "within subject" framing is meant to exclude.

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = epoched data + behavioral CSV; `synthetic` = ground-truth dry run. |
| `SYNTHETIC_CROSS_FRAC` | `0.25` | synthetic only: how much of each link leaks into the WRONG pairing. `1.0` destroys specificity (the falsification run). |
| `SYNTHETIC_ACROSS_BETA` / `SYNTHETIC_WITHIN_BETA` | `1.2` / `0.6` | synthetic only: planted coupling strengths. |
| `BEHAVIOR_CSV` | repo-root `combinedData.csv` | raw trial-level behavior. |
| `BEHAVIOR_RT_COL` | `RT` | RT column in that table. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | analysis window (s from stimulus onset). |
| `ELECTRODES` | `all` | `all` or `sig`. |
| `ALPHA` | `0.05` | A1 FDR threshold for the S/F flags. |
| `NEURAL_SUMMARY` | `count` | which per-subject neural summary headlines the across-subject level (`count`/`frac`/`effect`); all three are computed. |
| `RUN_TRIALWISE` | `1` | set `0` for the across-subject level only (the single-trial level needs per-trial RT in the epochs metadata). |

## Outputs

Written to `results/<epochs_or_synthetic_tag>/brain_behavior_window_<tmin>to<tmax>s_<electrodes>_<neural_summary>/`:

- `electrode_labels.csv` — the A1 per-electrode S/F labels A6 sits on.
- `behavioral_magnitudes.csv` — per-subject `lwpc`/`lwps` RT d-o-d (signed, ms).
- `subject_table_<mode>.csv` — the merged neural × behavioral table per neural summary.
- `across_subject.json` — matched and cross correlations, `n`, and the caveat.
- `trial_df.csv` (real runs) — the single-trial table with group HG and both adjustments.
- `trialwise.json` — matched/cross slopes, p, z, and `specificity_ok` per group.
- `brain_behavior_summary.png` — 4 panels: both matched scatters, the across-subject
  specificity bars, and the within-subject slopes with 95 % CIs.
- `summary.txt` — printed verdict.

**Reading:** the headline is the **specificity gap**, not a p-value. With thousands
of trials every slope is "significant", so the claim rests on the matched pairing
being *stronger* than the cross pairing (`specificity_ok`) at both levels. The
across-subject correlation is reported with its *n* and an honest underpowered
caveat — a null there is uninformative; the within-subject mixed model is the real
test.
