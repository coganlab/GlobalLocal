# A4 — Cross-decoding of the stability/flexibility subpopulations

> **Method & rationale:** see the merged guide
> [`docs/stability_flexibility_guide.md`](../../docs/stability_flexibility_guide.md)
> — §7.1 (cross-decoding), and §3.2 for the four interaction-defined electrode
> groups and the **ignore-the-diagonal** double-dipping rule this job now applies
> (`within_block_by_group` in the outputs).


Runs the representation-level test in `src/analysis/decoding/cross_decoding.py` on
the cluster: co-localization (A1/A2) shows the *same electrodes* are selective for
both **stability** (LWPC) and **flexibility** (LWPS), but does an electrode carry
**one shared code** or **two orthogonal codes**? Counting can't tell them apart;
**cross-decoding can** — train a classifier on one contrast, test whether its
decision axis transfers to the other.

A step-by-step walk-through is in `cross_decoding_tutorial.ipynb` (next to the
analysis module).

## Files

| File | Role |
|---|---|
| `stability_flexibility_cross_decoding_dcc.py` | Core: assembles the cluster-mode HG table, derives the A1 `both`/`S_only`/`F_only` electrode groups, runs the four designs, writes results + a summary figure. Exposes `main(args)`. |
| `run_stability_flexibility_cross_decoding_dcc.py` | Entrypoint: sets parameters (env-overridable) and calls `main`. |
| `sbatch_stability_flexibility_cross_decoding_dcc.sh` | SLURM wrapper (`conda activate ieeg` → entrypoint). |
| `submit_stability_flexibility_cross_decoding_dcc.sh` | Sets `EPOCHS_ROOT_FILE`/window/etc. and `sbatch`-submits. |

## Quick start

```bash
cd dcc_scripts/decoding
# validate the discrimination in ~a minute with a PLANTED shared code
# (should cross-decode):
DATA_SOURCE=synthetic SYNTHETIC_CODE=shared bash submit_stability_flexibility_cross_decoding_dcc.sh
# the ORTHOGONAL code (each contrast decodable, but should NOT cross-decode):
DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal bash submit_stability_flexibility_cross_decoding_dcc.sh
# real run — set EPOCHS_ROOT_FILE in the submit script, then:
bash submit_stability_flexibility_cross_decoding_dcc.sh
```

You can also run the entrypoint directly (no SLURM) for a fast local sanity check:

```bash
DATA_SOURCE=synthetic SYNTHETIC_CODE=shared N_PSEUDO=40 N_FOLDS=3 N_PERM=200 \
    python run_stability_flexibility_cross_decoding_dcc.py
```

## The data model

A4 runs on the **same** long-format single-trial table as A1–A3, assembled with
`effect_measure='cluster'` so each row's `hg` is that trial's HG **time course**
over the window (not the window mean). Subjects don't share trials, so electrodes
are pooled into a **pseudopopulation** and **pseudo-trials** are synthesized by
matching on the full condition cell (congruency × inc_prop × switchType ×
switch_prop). Train and test pseudo-trials are drawn from **disjoint reservoirs**
of the underlying single trials (the circularity guard).

## What it does — the four designs

- **(0) within-block decoding baseline (Fig 9):** decode congruency within
  low/high incongruent-proportion blocks and switchType within low/high
  switch-proportion blocks; the block difference is a neural cross-effect (the
  decoding analog of the univariate LWPC/LWPS effects).
- **(a) label transfer:** train on stability, test on flexibility (and vice
  versa), **separately** on the A1 `both`/`S_only`/`F_only` electrode groups.
  Prediction: only the `both` group cross-decodes. Run **raw and
  per-condition-mean-removed**.
- **(b) set comparison:** the same label decoded within each electrode group (so
  you can compare where a code lives).
- **(c) temporal generalization (Fig 10):** train-time × test-time accuracy
  matrix, within a contrast and across contrasts. Off-diagonal generalization →
  sustained/stable code; a narrow diagonal → moving/phasic code.

## The confound controls (plan §0.8 — non-negotiable)

- **Circularity guard** — train/test pseudo-trials come from disjoint reservoirs;
  the electrodes/trials used to *define* a group are never the ones a transferred
  accuracy is computed on.
- **Trial-count matched** — fixed `N_PER_CELL`/`N_PSEUDO` per condition cell, so
  class/block counts are equal by construction.
- **Survives per-condition mean removal** — `strip_condition_means=True` subtracts
  each condition's per-feature mean; a transfer that collapses to chance after
  removal was a univariate offset, not a genuine multivariate code. Every transfer
  is reported both raw and mean-removed.
- **Null centred at chance** — chance is estimated by permuting the transferred
  (test) labels, not assumed to be 0.5.

## Classifier

The classifier is the same scaler → PCA → LDA pipeline the project `Decoder`
wraps (`ieeg.decoding.models.PcaEstimateDecoder`). `cross_decoding.make_classifier`
reuses `Decoder` when `ieeg` imports (on the cluster) and otherwise falls back to
an equivalent scikit-learn `Pipeline`, so the pseudo-trial / transfer logic runs
anywhere. The backend is resolved once and cached.

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = epoched data; `synthetic` = ground-truth dry run. |
| `SYNTHETIC_CODE` | `shared` | synthetic only: `shared` (should cross-decode) or `orthogonal` (should not). |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | analysis window (s from stimulus onset). |
| `ELECTRODES` | `all` | `all` or `sig`. |
| `ALPHA` | `0.05` | A1 FDR threshold for the electrode groups. |
| `N_PER_CELL` | `5` | trials averaged per electrode to form one pseudo-trial. |
| `N_PSEUDO` | `80` | pseudo-trials per class-labelled condition cell. |
| `N_FOLDS` | `5` | disjoint pseudo-trial folds (CV). |
| `N_PERM` | `500` | label-permutation null draws. |
| `MIN_GROUP_SIZE` | `5` | skip electrode groups smaller than this. |

## Outputs

Written to `results/<epochs_or_synthetic_tag>/cross_decoding_window_<tmin>to<tmax>s_<electrodes>/`:

- `cross_decoding.json` — accuracies, nulls, and p-values for every design/group
  (bulky arrays stripped).
- `tempgen_*.npy` — the temporal-generalization matrices.
- `anova_labels.csv` — the A1 electrode groups (real runs).
- `cross_decoding_summary.png` — within-block bars, label-transfer-by-group,
  temporal-generalization matrices.
- `summary.txt` — printed verdicts.

**Reading:** cross-decoding above chance (and surviving per-condition mean
removal) on the `both` group = a **shared** code; chance on `both` while each
process is individually decodable = **orthogonal** codes (segregation at the
representational level).
