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
