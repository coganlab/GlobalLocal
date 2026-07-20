# Stability vs. Flexibility Segregation — DCC scripts

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
epochs metadata, producing the long table the analysis expects:

```
subject | electrode (= subject-channel) | hg | congruency | switchType
```

## Key knobs (env vars, read by the entrypoint)

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS_ROOT_FILE` | — (required for real runs) | Which epoched HG file to load. |
| `DATA_SOURCE` | `real` | `real` = epoched data; `synthetic` = ground-truth dry run. |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | Analysis window (s from stimulus onset). |
| `ELECTRODES` | `all` | `all` or `sig` (significant channels). |
| `N_SPLITS` | `200` | Disjoint trial-half resamples for sensitivity estimation. |
| `N_PERM_CORR` | `10000` | Permutations for the continuous test. |
| `N_PERM_LABEL` | `2000` | Permutations per electrode for S/F labeling. |

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
