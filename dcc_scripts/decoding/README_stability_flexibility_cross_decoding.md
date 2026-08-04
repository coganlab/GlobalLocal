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
DATA_SOURCE=synthetic SYNTHETIC_CODE=shared \
    WINDOW_SIZE=16 STEP_SIZE=16 N_SPLITS=3 N_REPEATS=2 N_PERM=50 \
    python run_stability_flexibility_cross_decoding_dcc.py
```

## The data model — this is the ordinary decoding pipeline

A4 does **not** have its own decoding machinery. It runs on the same stack as the
main decoding job, which already supplies everything a transfer needs:

| Requirement | Where it comes from |
|---|---|
| cross-subject **pseudopopulation** | the ROI LabeledArray — `put_data_in_labeled_array_per_roi_subject` NaN-pads each subject to the per-condition max and concatenates subjects along the **channel** axis; `mixup2` fills the padding |
| **disjoint train/test** (circularity guard) | the CV split inside `cv_cm_jim_window_shuffle` |
| **null centred at chance** | `shuffle=True` permutes the TRAIN labels and **refits**, so the null carries the variance of the whole pipeline (scaler → PCA → LDA, mixup, folds) |
| **multiple comparisons** | `time_perm_cluster` over the time-resolved accuracy trace |
| classifier | the project `Decoder` (scaler → PCA → LDA) |

The only thing A4 adds is a **second label vector**: train on one labelling of
the trials, score against another.

```python
decoder.cv_cm_jim_window_shuffle(data, labels_train,
                                 labels_test=labels_test,   # <- the whole of A4
                                 stratify_labels=strata)
```

`cross_decoding.build_cross_decoding_arrays` produces those three arrays from an
ROI LabeledArray. A condition only enters if **both** contrasts can label it —
the two factors have to cross, or a transfer is not identifiable. `strata` is the
source condition index; stratifying the folds on it keeps every fold balanced on
the label you *score*, not just the one you train on.

The A1 electrode definition still needs the long single-trial table
(`effect_measure='cluster'`), so a real run assembles both.

## What it does — the designs

- **(0) within-block decoding baseline (Fig 9):** decode congruency within
  low/high incongruent-proportion blocks and switchType within low/high
  switch-proportion blocks; the block difference is a neural cross-effect (the
  decoding analogue of the univariate LWPC/LWPS effects). This is an ordinary
  decode over a restricted condition set — `cd.filter_conditions(...)`, then the
  same contrast for train and test.
- **(0b) the per-group within-block 2×2**, restricted to each interaction-defined
  electrode group (CPC/SPS/CPS/SPC), **skipping the diagonal** cell that would
  double-dip — see `cd.is_circular_decode`. Only the off-diagonal cells are kept.
- **(a) label transfer:** train on stability, test on flexibility (and vice
  versa), **separately** on the A1 `both`/`S_only`/`F_only` electrode groups,
  plus the unselected reference group (`REFERENCE_GROUP`, default `all`).
  Prediction: only the `both` group cross-decodes.
- **(c) temporal generalization (Fig 10):** train-window × test-window accuracy
  matrix, within a contrast and across contrasts. Off-diagonal generalization →
  sustained/stable code; a narrow diagonal → moving/phasic code.

Design **(b) "set comparison"** — the same label decoded within each electrode
group — is just an ordinary decode with the electrodes restricted, which (0b)
already covers per group, so it no longer has its own code path.

## Circularity: what CV does and does not fix

Cross-validation makes the *decode* honest: the trials a transferred accuracy is
scored on were never trained on. It does **not** fix selection bias from the
electrode definition, because that selection happened *before* the split, using
every trial. Two options for the diagonal (define == decode) cell:

1. **Skip it** — `cd.is_circular_decode(group, contrast, block_col)` names it; the
   job omits it and keeps the three off-diagonal cells.
2. **Earn it** — define the electrodes on a disjoint set of trials
   (`trial_splitting.apply_electrode_definition_split`, `FRAC_DEF` env var).

## Which electrodes are decoded

Two independent choices, easy to conflate:

1. **Which electrodes are loaded at all** — `ELECTRODES`. `sig` keeps the
   baseline task-significant electrodes in the ROIs, `all` keeps every electrode
   in them. (With `ROIS_DICT = None`, the current default, no ROI/significance
   filter is applied and every channel is used.)
2. **How those loaded electrodes are split for the decodes** — the groups.
   `both`/`S_only`/`F_only` come from the interaction labels, and
   `REFERENCE_GROUP` (default `all`) adds the **unselected** set: every channel
   in the decoded ROI array.

The reference group matters because `both`, `S_only` and `F_only` were each
*chosen* for carrying an interaction, so none of them is a baseline for "does
this ROI cross-decode at all" — the selection is exactly what inflates
within-contrast decodability. The reference group is defined by nothing the
decode is about. Set `REFERENCE_GROUP=''` to drop it.

Temporal generalization costs `n_windows²` decodes per matrix, so it runs only on
`TEMPGEN_GROUPS` (default `both`); use `TEMPGEN_GROUPS=both,all` to get the
unselected comparison matrix too.

## Electrode definition: `anova` vs `power_traces`

`ELECTRODE_DEFINITION` picks how the S/F labels are derived. Both routes emit the
same table (`CPC`/`SPS`/`CPS`/`SPC` + `S`/`F` aliases), so everything downstream
is unchanged.

| Route | What it fits | Trade-off |
|---|---|---|
| `anova` (default) | one two-way ANOVA per electrode on the **window-mean** HG over `[WINDOW_TMIN, WINDOW_TMAX]`, BH-FDR'd across electrodes | self-contained — it only needs the epochs this job already loads, which is why it is the default. A strong but **transient** interaction is diluted by the window mean. |
| `power_traces` | reads the finished **within-electrode windowed ANOVA** runs and their permutation cluster correction (`power_traces_conjunction.electrode_labels`) | strictly more sensitive to transient interactions, and the decoded sets become literally the electrodes the power-trace figures call significant. Needs finished run directories. |

For the `power_traces` route, point at either one run whose ANOVA carried all
four interactions:

```bash
ELECTRODE_DEFINITION=power_traces POWER_TRACES_RUN_DIR=/path/to/run \
  bash submit_stability_flexibility_cross_decoding_dcc.sh
```

or one directory per interaction (`POWER_TRACES_CPC`, `POWER_TRACES_SPS`,
`POWER_TRACES_CPS`, `POWER_TRACES_SPC`). `POWER_TRACES_CORRECTION` chooses
`fdr_bh` (BH across electrodes — the family a test that *counts* electrodes
needs), `cluster` (raw cluster p, matching the existing lab convention), or
`none`.

## Key knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_SOURCE` | `real` | `real` = epoched data; `synthetic` = ground-truth dry run. |
| `SYNTHETIC_CODE` | `shared` | synthetic only: `shared` (should cross-decode) or `orthogonal` (should not). |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `0.5` | analysis window (s from stimulus onset) for the A1 definition. |
| `ELECTRODES` | `all` | `all` or `sig` (baseline task-significant) — which electrodes are loaded. |
| `ROI` | `all` | which ROI's LabeledArray to decode. |
| `ALPHA` | `0.05` | A1 FDR threshold for the electrode groups. |
| `ELECTRODE_DEFINITION` | `anova` | `anova` (in-job window-mean ANOVA) or `power_traces` (finished cluster-corrected runs). |
| `POWER_TRACES_RUN_DIR` | unset | `power_traces` only: one run carrying all four interactions. |
| `POWER_TRACES_CPC` / `_SPS` / `_CPS` / `_SPC` | unset | `power_traces` only: one run directory per interaction (overrides the single-run form). |
| `POWER_TRACES_CORRECTION` | `fdr_bh` | `fdr_bh`, `cluster`, or `none`. |
| `POWER_TRACES_ROI` | unset | `power_traces` only: restrict the labels to one ROI. |
| `REFERENCE_GROUP` | `all` | name of the unselected all-electrode group; `''` drops it. |
| `TEMPGEN_GROUPS` | `both` | comma-separated groups to run temporal generalization on; `''` skips it. |
| `WINDOW_SIZE` | `20` | decoding window, in samples. |
| `STEP_SIZE` | `10` | window stride, in samples. |
| `N_SPLITS` | `5` | CV folds — or random resamples per repeat when `FRAC_TRAIN` is set. |
| `N_REPEATS` | `10` | CV repeats. |
| `FRAC_TRAIN` | unset | **proportion of trials used for training.** Unset keeps `StratifiedKFold` at `(N_SPLITS-1)/N_SPLITS`; setting it switches to `StratifiedShuffleSplit` at exactly this fraction. |
| `EXPLAINED_VARIANCE` | `0.8` | PCA variance retained. |
| `N_PERM` | `500` | permutations for the cluster test over windows. |
| `MIN_GROUP_SIZE` | `5` | skip electrode groups smaller than this. |

## Outputs

Written to `results/<epochs_or_synthetic_tag>/cross_decoding_window_<tmin>to<tmax>s_<electrodes>/`:

- `cross_decoding.json` — per design/group: mean and peak accuracy, shuffle mean,
  number of cluster-significant windows (bulky arrays stripped).
- `accuracy_traces.npz` — the true and shuffle accuracy traces, for re-plotting.
- `tempgen_*.npy` — the temporal-generalization matrices.
- `anova_labels.csv` — the per-electrode definition table (real runs; written by
  whichever `ELECTRODE_DEFINITION` route ran).
- `cross_decoding_summary.png` — within-block bars, label-transfer traces by
  group, temporal-generalization matrices.
- `summary.txt` — printed verdicts.

**Reading:** cross-decoding above chance on the `both` group = a **shared** code;
chance on `both` while each process is individually decodable = **orthogonal**
codes (segregation at the representational level). Read `n_sig_windows` rather
than any single window's accuracy — the verdict is cluster-corrected across time,
and chance is the refit shuffle null rather than an assumed 0.5.
