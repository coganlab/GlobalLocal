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

Sits on a per-electrode S/F definition and asks: *are the distinct subpopulations
in different **places**?* — while conditioning every claim on iEEG **coverage**
(clinically-placed electrodes are the main confound here). Analysis code lives in
`src/analysis/stats/stability_flexibility_anatomy.py`; a step-by-step walk-through
is in `stability_flexibility_anatomy_tutorial.ipynb` (next to that module).

## Files

| File | Role |
|---|---|
| `stability_flexibility_anatomy_dcc.py` | Core: gets S/F flags (A1 **or** `power_traces`), maps electrodes → ROI group **and** raw Destrieux label, applies the optional ROI restriction, builds the coverage matrix, runs the coverage-conditioned enrichment test, writes figures (incl. the electrodes on the brain) + `summary.txt`. Exposes `main(args)`. |
| `run_stability_flexibility_anatomy_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_stability_flexibility_anatomy_dcc.sh` | SLURM wrapper (`conda activate ieeg` → `xvfb-run python` so the brain render has a display). |
| `submit_stability_flexibility_anatomy_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/window/label source/ROI filter and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/stats
# validate the whole path in seconds with a PLANTED group×ROI association:
DATA_SOURCE=synthetic bash submit_stability_flexibility_anatomy_dcc.sh
# the NULL version (no association — the test must come back n.s.):
DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0.0 bash submit_stability_flexibility_anatomy_dcc.sh
# real run on the A1 electrodes — set EPOCHS_ROOT_FILE in the submit script:
bash submit_stability_flexibility_anatomy_dcc.sh

# real run on the POWER_TRACES electrodes (cluster-corrected), lpfc only,
# counted by raw Destrieux label. Loads no epoched data — just the finished run:
POWER_FIGS=/hpc/home/$USER/coganlab/$USER/GlobalLocal/dcc_scripts/power/figs
EPOCHS_ROOT=Stimulus_-1.0to1.5sec_..._nan_policy_omit   # the ANOVA's dir name
LABEL_SOURCE=power_traces ROI_FILTER=lpfc PT_ROI=lpfc \
  PT_RUN_DIR="$POWER_FIGS/$EPOCHS_ROOT/anova_within_electrode/stimulus_experiment_conditions_24_subjects" \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

## What it does

1. **Electrode definition** (`LABEL_SOURCE`):
   * `a1` — assembles the long-format single-trial HG table and fits
     `per_electrode_anova_labels` (`contrast_mode='proportion'`) → `S`/`F` flags.
   * `power_traces` — reads finished **within-electrode windowed ANOVA with
     cluster correction** runs through `power_traces_conjunction.electrode_labels`
     (BH across electrodes by default). Same `S`/`F` contract, no epoched data.
   Either way → the 4-way group (`both`/`S_only`/`F_only`/`neither`).
2. Maps each electrode to **both** anatomical levels: the coarse ROI group
   (`build_electrode_roi_map` + `config/rois.py`) and the **raw Destrieux label**
   (`build_electrode_anat_map`, white-matter/unknown dropped).
3. Optional **ROI restriction** (`ROI_FILTER=lpfc`). The ROI groups overlap
   (`dlpfc`/`lpfc` share `G_front_middle`, `S_front_inf`, …), so the job subsets
   `rois_dict` to the requested group *before* building the map — otherwise
   `dlpfc` claims the shared labels and the lpfc filter would silently keep only
   lpfc's exclusive ones.
4. Picks the level to count/test on (`ANAT_LEVEL`, default `auto`): raw Destrieux
   labels whenever the analysis sits inside a single ROI group — inside an
   lpfc-only analysis every electrode's ROI *is* `lpfc`, so a group-level
   histogram is one bar and the group × ROI test has one column.
5. **Coverage**: subject × ROI boolean matrix at that level (does subject *s* have
   any electrode in ROI *r*?).
6. **Coverage-conditioned enrichment test** (`roi_group_enrichment_test`): Pearson χ² on the group × ROI table with a **within-subject permutation null** (shuffle the group label within each subject, so the null respects nesting *and* coverage), restricted to ROIs sampled in ≥ `MIN_SUBJECTS` subjects.
7. Figures: ROI-group **and** Destrieux histograms (annotated with per-ROI coverage), the coverage heatmap + enrichment null, and the selective electrodes themselves on the fsaverage brain — one colour per group, rendered through the same `vis/jim_mri.plot_on_average` path as `plot_sig_electrodes_dcc.py` (falls back to the ROI histogram when the surface stack isn't available).

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = real electrodes + ROI atlas; `synthetic` = ground-truth dry run. |
| `SYNTHETIC_ENRICHMENT` | `0.6` | synthetic only: strength of the planted group×ROI association (`0.0` = null). |
| `LABEL_SOURCE` | `a1` | `a1` = fit the window-mean ANOVA here; `power_traces` = read finished cluster-corrected runs. |
| `PT_RUN_DIR` | — | `power_traces` only: one 4-factor run dir (containing `summary.csv`). |
| `PT_RUN_CPC` / `PT_RUN_SPS` / `PT_RUN_CPS` / `PT_RUN_SPC` | — | …or one run dir per interaction (CPC + SPS required). |
| `PT_CORRECTION` | `fdr_bh` | `fdr_bh` (BH across electrodes — the right family for a count test), `cluster` (raw cluster p, the existing lab convention), `none`. |
| `PT_ALPHA` | = `ALPHA` | selection cutoff for the `power_traces` labels. |
| `PT_ROI` | — | restrict to one ROI **of the ANOVA run** (e.g. `lpfc`). |
| `ROI_FILTER` | — | restrict the **anatomy** to one/several ROI groups of `config/rois.py` (e.g. `lpfc`). Empty = whole brain. |
| `ANAT_LEVEL` | `auto` | `auto` \| `group` \| `destrieux` — level for the histogram + enrichment test. |
| `HIST_TOP_N` | — | cap the Destrieux histogram at the N most-populated labels. |
| `MAKE_BRAIN` / `BRAIN_HEMI` | `1` / `both` | render the brain figure; hemisphere(s) to draw. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | analysis window (s from stimulus onset). A1 route only. |
| `ELECTRODES` | `all` | `all` or `sig`. A1 route only. |
| `ALPHA` | `0.05` | A1 FDR threshold for the S/F flags. |
| `MIN_SUBJECTS` | `3` | keep only ROIs sampled in ≥ this many subjects (the coverage condition). |
| `N_PERM` | `10000` | within-subject permutations for the enrichment null. |

## Outputs

Written to `results/<epochs_or_pt_or_synthetic_tag>/anatomy_<label_source>_<roi_or_wholebrain>_window_<tmin>to<tmax>s_<electrodes>/`:

- `electrode_labels.csv` — the raw upstream labels (per-effect p/q + S/F), before anatomy.
- `anatomy_labels_roi.csv` — per-electrode S/F, ROI group, Destrieux label, group.
- `coverage_matrix.csv` — subject × ROI coverage, at the level being tested.
- `group_roi_contingency.csv` — the table the χ² is computed on.
- `roi_group_histogram.csv` / `.png` — counts per **ROI group**.
- `destrieux_group_histogram.csv` / `.png` — counts per **raw Destrieux label** (the one to read for an lpfc-only run).
- `roi_enrichment.json` (+ `roi_enrichment_null.npy`) — level tested, ROIs tested, χ², permutation p, per-ROI coverage.
- `anatomy_coverage_enrichment.png` — coverage heatmap + the permutation null with the observed χ².
- `selectivity_groups_on_brain.png` (+ `..._both.png` / `..._S_only.png` / `..._F_only.png`), or `..._roi_hist.png` when the surface stack is unavailable.
- `summary.txt` — printed verdict.

**Reading:** a significant test means selectivity-group membership is associated
with ROI (or Destrieux label) *beyond* what electrode placement forces; per-ROI
coverage is reported so no claim rests on where the grid happens to be. The
histograms are raw counts, not the test — read them alongside the coverage row.

## A3 vs `dcc_scripts/vis/plot_sig_electrodes_dcc.py`

They overlap in *output* (both can draw significant electrodes on fsaverage and
histogram their Destrieux labels) but answer different questions, and A3 now
**calls** the vis stack rather than duplicating it (`plot_on_average` +
`build_global_index_map` / `electrodes_to_global_indices`), so the two figures are
directly comparable.

| | `plot_sig_electrodes_dcc.py` | A3 anatomy |
|---|---|---|
| Unit | one colour per **condition/effect** (any `sig_chans` contrast or any ANOVA effect, from the registry in `condition_plot_specs.py`) | one colour per **selectivity group** (`both` / `S_only` / `F_only`), i.e. the S×F conjunction |
| Output | figures + `sig_electrodes_<condition>.json` | figures **+ the coverage-conditioned enrichment statistic** (χ², permutation p, coverage matrix, contingency) |
| Overlap handling | electrodes significant in >1 condition drawn in `overlap_color` | overlap *is* the `both` group — mutually exclusive by construction |

Use `plot_sig_electrodes` when you want a picture of "where are the electrodes
significant for effect X (and Y)"; use A3 when the claim is "the S and F
subpopulations sit in different places, and that isn't just coverage".

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

---

# A1′/A2 — Conjunction on `power_traces` cluster-corrected electrodes

Same conjunction as the A1/A2 job above, with the **electrode definition
swapped**. Instead of one ANOVA on window-mean HG, the S/F flags are read back
from a finished **within-electrode windowed ANOVA + cluster-correction** run
(`src/analysis/power/windowed_anova.py`, launched by
`dcc_scripts/power/run_power_traces_dcc.py` with `ANOVA_UNIT='electrode'`), which
fits the ANOVA at every window and cluster-corrects across time. Bridge code:
`src/analysis/stats/power_traces_conjunction.py`; rationale in guide §4/§4a/§5.1.

Nothing here re-fits an ANOVA. The count test is a **pure read** of a finished
run, so it costs seconds, needs no epoched data, and can be re-run at several
alphas / corrections for free.

## Files

| File | Role |
|---|---|
| `power_traces_conjunction_dcc.py` | Core: resolves the run dir(s), builds labels, runs the count battery, optionally the continuous confound control, writes results + figures. Exposes `main(args)`. |
| `run_power_traces_conjunction_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_power_traces_conjunction_dcc.sh` | SLURM wrapper (`conda activate ieeg` → entrypoint). |
| `submit_power_traces_conjunction_dcc.sh` | Sets `PT_RUN`/ROI/correction and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/stats

# validate the whole path in seconds against a KNOWN planted overlap:
DATA_SOURCE=synthetic bash submit_power_traces_conjunction_dcc.sh
# the NULL version (overlap == base rate — MH OR must come back ≈ 1):
DATA_SOURCE=synthetic SYNTHETIC_OVERLAP=0.25 bash submit_power_traces_conjunction_dcc.sh

# real: point PT_RUN at a finished within-electrode ANOVA run, then
bash submit_power_traces_conjunction_dcc.sh
RUN_CONTINUOUS=1 bash submit_power_traces_conjunction_dcc.sh   # + confound control
```

A **run directory** is the one holding `summary.csv` + `run_config.json`:
`dcc_scripts/power/figs/<EPOCHS_ROOT_FILE>/anova_within_electrode/<conditions_save_name>`.

## What it does

1. **Labels.** `ptc.electrode_labels` reads the run's `summary.csv`, pivots the
   four interactions (CPC/SPS/CPS/SPC) onto one row per electrode, and corrects
   **across electrodes** — the family a test that *counts electrodes* needs.
   `summary.csv` carries every electrode that was **tested**, not just the
   winners, which is what makes the 2×2 denominator honest.
2. **Counts.** `ptc.run_power_traces_conjunction`: CMH (subject-stratified),
   within-subject permutation null on the joint count, shared-vs-distinct, the
   two cross-interaction specificity controls, and a threshold sweep.
3. **Confound control** (optional, `RUN_CONTINUOUS=1` — the only step that loads
   epochs). Re-estimates each electrode's two sensitivities on **disjoint trial
   halves**, residualised on responsiveness, over the **same window the run
   tiled** (read from `run_config.json`, not from `WINDOW_TMIN/TMAX`), and under
   **all three** effect measures. It is the control on the counts, not a second
   headline (guide §5.1).

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `PT_RUN` | — | One run whose ANOVA held all four factors (`stimulus_experiment_conditions`). Preferred: all four interactions then come from the same electrodes and trials. |
| `PT_RUN_CPC` / `_SPS` / `_CPS` / `_SPC` | — | Or four separate two-factor runs. Only CPC and SPS are required. |
| `ROIS` | `lpfc` | Comma-separated ROI names, or `all` for one analysis pooled over ROIs. Each ROI gets its own output subdirectory. |
| `CORRECTION` | `fdr_bh` | `fdr_bh` = BH across electrodes within (roi, effect); `cluster` = raw cluster p < α, no across-electrode correction (the like-for-like port of what `load_significant_electrodes` does today); `none` = any surviving cluster. |
| `ALPHA` | `0.05` | Selection cutoff. |
| `REQUIRE_ALL` | `1` | Keep only electrodes present in every requested run — different runs can end up with different electrode sets (`min_trials_per_cell`), and a 2×2 over inconsistent denominators is not interpretable. |
| `USE_NPZ` | `1` | Legacy runs only (no `best_cluster_p` column): recompute a graded cluster p per electrode from the saved `.npz` null. |
| `N_PERM_NULL` / `THRESHOLDS` | `10000` / `0.01,0.025,0.05,0.10` | Permutation count and sweep cutoffs. |
| `RUN_CONTINUOUS` | `0` | `1` also runs the continuous confound control (needs `EPOCHS_ROOT_FILE`). |
| `EFFECT_MEASURES` | `peak_t,cluster,cohens_d` | Which scalarisations the control runs. Run all three: divergence in sign is itself the finding. |
| `N_SPLITS` / `N_PERM_CORR` / `MIN_ELEC` / `ELECTRODES` | `200` / `10000` / `3` / `all` | Control-only knobs (as in the segregation job). |

## Outputs

Written to `power_traces_conjunction_results/<run_tag>/<correction>_alpha<α>/<roi>/`:

- `labels.csv` — one row per electrode: `p_/q_/<g>_sign/<g>_extent` per group, the
  binary CPC/SPS/CPS/SPC flags, and the `S`/`F` aliases.
- `conjunction.json`, `conjunction_per_subject.csv` — MH OR, CMH p, homogeneity p,
  pooled 2×2, per-subject tables.
- `counts.json` — the four cells plus both permutation tests.
- `joint_count_null.npy`, `shared_minus_distinct_null.npy` — the raw nulls.
- `cross_controls.csv`, `threshold_sweep.csv`.
- `power_traces_conjunction_summary.png` (6 panels) and
  `power_traces_conjunction_evidence.png` (q-values + cluster extents).
- `continuous_confound_control.json` — ρ and p per effect measure, when run.
- `summary.txt` — printed verdicts.

**Reading:** MH `OR < 1` / overlap below null → **segregation**; `OR > 1` /
overlap above null → **shared core**; ≈1 → independent.

Two things the summary flags, because both read as findings if you skip them:

- **`shared − distinct` is the same test as the joint count.** With the marginals
  fixed by the within-subject shuffle, `D = 3·both − n_S − n_F`, so it is a
  monotone function of the `both` count and returns an identical p. Both are
  printed so the equivalence is visible — never report them as two lines of
  evidence.
- **Thin sweep rows.** `cmh_conjunction` now drops subject strata that carry no
  information about the S–F association and returns `OR = nan` when none are
  left, so the strict end of a sweep no longer fabricates evidence (see the note
  below). What remains is a judgement call it can't make for you: a row resting
  on one or two informative subjects has a finite OR that still says nothing
  about threshold stability. Rows with an undefined OR, fewer than three
  informative subjects, or `n_both = 0` are named in `summary.txt` and drawn
  hollow (outside the trend line) in the sweep panel.

---

## Note: uninformative subject strata in the CMH (fixed)

A subject with no S electrodes (or no F electrodes) carries **no information**
about whether S predicts F, but `StratifiedTable(..., shift_zeros=True)` used to
turn its empty table into a positive association anyway — biasing every pooled OR
toward "shared", most severely at the strict end of a threshold sweep. The
measured size of that bias, and why the shift is right for a sparse-but-real
table and wrong for an empty-marginal one, is in
[guide §5](../../docs/stability_flexibility_guide.md) (the "uninformative subject
strata" box).

What changed here, for both the A1/A2 and the A1′/A2 job:

- Strata with a zero marginal are **dropped before pooling** (a no-op on an
  unshifted analysis — such a table contributes nothing to either side of the MH
  ratio); when none are informative the odds ratio is **NaN**, not a number.
- `cmh_conjunction` gained `n_strata` / `n_informative_strata` /
  `n_dropped_strata`, `per_subject` gained an `informative` column, and the
  threshold sweep gained `n_informative_strata` — read it with every sweep row.
- Pass `drop_uninformative_strata=False` to reproduce the old numbers. Pinned by
  `tests/analysis/stats/test_cmh_uninformative_strata.py`.

**If you have already recorded CMH numbers, re-run them** — any run where some
subject had no S electrodes or no F electrodes was biased toward "shared".
