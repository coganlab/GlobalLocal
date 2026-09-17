# N4 — continuous electrode scores → anatomy, brain maps, and descriptive centres

**What this document is.** A standalone, end-to-end walkthrough of beats **5–7**
in [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md):

1. the primary coverage-conditioned anatomical test of continuous LWPC and LWPS
   scores;
2. the five continuous-score brain maps; and
3. the per-subject, per-hemisphere weighted centres (descriptive only).

It explains what the analysis asks, where the code lives, which inputs are safe
to use, how data move through the code, exact cluster commands, and how to read
every output. It deliberately does **not** cover the older categorical S/F-group
anatomy arm except where necessary to distinguish it from this analysis.

If you read only three things, read [the estimand](#2-the-estimand-and-sign-convention),
[the primary test](#4-the-primary-anatomical-test), and
[the interpretation checklist](#11-interpretation-checklist).

---

## 0. The short version

N4 asks:

> **Does the balance between an electrode's LWPC and LWPS effects vary with its
> anatomical location?**

For every anatomically eligible electrode, the pipeline estimates an LWPC score
and an LWPS score on disjoint trial halves. Each score is a signed,
equal-cell-weighted, standardized difference-of-differences. After one pooled
scale factor per effect, it forms

```text
delta = lwpc_s - lwps_s
```

and asks whether `delta` differs by ROI/Destrieux parcel. This is an explicit
**effect type × anatomy interaction**. It does not infer a dissociation from
“LWPC significant here, LWPS not significant there.”

The primary null independently swaps the LWPC and LWPS labels within every
electrode. A swap negates `delta`, so the implementation is a sign-flip
permutation. It holds the subject, electrode, anatomy, coverage, responsiveness,
and the two marginal score distributions fixed. The primary result is the
omnibus permutation `F` and `p` in `summary.txt` / `score_anatomy.json`.

The brain maps visualize the two signed scores, their magnitudes, and their
difference. They do not add five new tests. The coordinate regression is a
secondary inferential description of spatial gradients. The weighted medoids
are descriptive and must not replace either anatomical test.

---

## 1. Where the code lives

| Role | File / function |
|---|---|
| Analysis specification | `docs/analysis_plan_concurrent_regulation.md`, §§5–7 |
| Score estimation | `src/analysis/stats/stability_flexibility_segregation.py`: `compute_sensitivities_per_split`, `average_over_splits`, `add_responsiveness` |
| Core N4 statistics | `src/analysis/stats/stability_flexibility_anatomy.py`: `attach_scores`, `relative_score_roi_test`, `relative_score_coordinate_test`, `map_reliability`, `leave_one_subject_out` |
| Brain maps | same file: `SCORE_MAPS`, `plot_score_by_roi`, `plot_scores_on_brain`, `plot_score_maps` |
| Descriptive centres | same file: `score_centers_per_subject` |
| DCC orchestration and output writing | `dcc_scripts/stats/stability_flexibility_anatomy_dcc.py`: `load_scores`, `run_score_anatomy`, `write_score_summary` |
| Environment-variable entry point | `dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py` |
| Slurm submitter / display wrapper | `dcc_scripts/stats/submit_stability_flexibility_anatomy_dcc.sh`, `sbatch_stability_flexibility_anatomy_dcc.sh` |
| Recommended upstream score-producing run | `dcc_scripts/stats/submit_stability_flexibility_segregation_dcc.sh` |
| Ground-truth regression tests | `tests/analysis/stats/test_stability_flexibility_anatomy.py` |

### Do not choose the wrong arm

`run_stability_flexibility_anatomy_dcc.py` defaults to `ARM=categorical`. That is
the older thresholded S/F-group enrichment analysis. **For this document, always
set `ARM=continuous`**. `ARM=both` runs categorical first and then writes N4 into
a `continuous/` subdirectory, but it costs more and can obscure which result is
primary.

`LABEL_SOURCE` is relevant to the categorical arm. It does not define the N4
continuous electrode set when `SCORES_CSV` is supplied. The continuous arm uses
the electrodes present in that score CSV and then applies the anatomical
`ROI_FILTER`.

---

## 2. The estimand and sign convention

### 2.1 What the two scores are

With the required `CONTRAST_MODE=proportion`, the two scores are:

```text
LWPC = (incongruent - congruent | low incongruent proportion)
     - (incongruent - congruent | high incongruent proportion)

LWPS = (switch - repeat | low switch proportion)
     - (switch - repeat | high switch proportion)
```

The implementation uses four cell means with equal weights and divides their
difference-of-differences by the pooled within-cell SD. It therefore avoids
letting the frequent cells dominate an unbalanced proportion design.

> **Positive LWPC or LWPS means adaptation:** the condition effect is smaller in
> the high-proportion block. Negative means the condition effect grew rather
> than shrank.

For each split, both effects are evaluated on both halves (`xA`, `xB`, `yA`,
`yB`). `average_over_splits` averages the two half-estimates and all requested
splits to give `x` (LWPC) and `y` (LWPS) per electrode. Cross-effect reliability
calculations continue to use the split-resolved values, so same-trial noise
cannot masquerade as co-localization.

### 2.2 Scaling and the relative score

`attach_scores` renames `x/y` to `lwpc_score/lwps_score`, then divides each
entire column by that effect's SD across all pooled electrodes:

```text
lwpc_s = lwpc_score / SD(lwpc_score across all electrodes)
lwps_s = lwps_score / SD(lwps_score across all electrodes)
delta  = lwpc_s - lwps_s
```

There is no subtraction of the pooled mean because the interaction asks about
relative spatial variation; a global offset cannot create between-ROI spread
after the nuisance intercept. There is deliberately no within-subject z-score.
With two electrodes it would force values to ±0.707 regardless of magnitude,
and with one electrode it would silently lose the subject. The scores are
already standardized, so one pooled spread adjustment per effect is sufficient.

Interpret `delta` as:

| Value | Meaning |
|---|---|
| `delta > 0` | relatively more LWPC-dominant |
| `delta < 0` | relatively more LWPS-dominant |
| `delta ≈ 0` | similar scaled LWPC and LWPS scores; it does **not** imply both effects are absent |

Always inspect the signed and magnitude maps alongside `delta`. For example,
`delta > 0` could mean strong positive LWPC, weak LWPS, negative LWPS, or both.

### 2.3 The electrode set is anatomical, not effect-selected

The defensible N4 population is all electrodes in the predeclared anatomical
scope (`ROIS=all` upstream for whole-brain N4, or a predeclared ROI scope), not
electrodes selected for an LWPC or LWPS effect. Selecting significant effects
before asking where those effects occur would make the anatomical answer partly
a consequence of the selection rule and would discard the many weak but
informative continuous observations.

For that reason, use `ELECTRODES=all`, `ROIS=all`, and
`CONTRAST_MODE=proportion` in the upstream score run for the whole-brain primary
analysis. A predeclared `ROI_FILTER=lpfc` is a legitimate separate lPFC analysis,
but is not a substitute for whole-brain N4.

---

## 3. Data flow through the code

The recommended path computes expensive scores once in the segregation job and
reuses its CSVs in the anatomy job:

```text
high-gamma Epochs on disk
  EPOCHS_ROOT_FILE
       │
       │ load_HG_ev1_rescaled_per_subject(..., acc_trials_only=True)
       ▼
per-subject MNE epochs + trial metadata
       │
       │ assemble_long_df(window_tmin, window_tmax,
       │                  effect_measure='cohens_d')
       ▼
long trial table
  subject, electrode, congruency, switchType,
  incongruent_proportion, switch_proportion, hg
       │
       │ compute_sensitivities_per_split(...,
       │     contrast_mode='proportion', n_splits=200)
       │ each electrode's trials are divided into disjoint halves
       ▼
per_split.csv
  one row / electrode / split: xA, xB, yA, yB
       │
       ├─ average_over_splits() + add_responsiveness()
       ▼
electrodes.csv
  subject, electrode, x, y, resp, ...
       │
       │ N4 anatomy job: load_scores()
       │ load atlas maps and optional fsaverage/MNI coordinates
       ▼
attach_scores()
  join ROI + Destrieux + coordinates
  pooled scaling; abs_lwpc, abs_lwps; delta = lwpc_s - lwps_s
       │
       ├─ restrict_to_roi(ROI_FILTER)
       ├─ build_coverage_matrix()
       │
       ├─ PRIMARY: relative_score_roi_test()
       │      delta ~ anatomy + responsiveness + subject dummies
       │      within-electrode score-label-swap null
       │
       ├─ LEVERAGE: leave_one_subject_out()
       │
       ├─ SECONDARY: relative_score_coordinate_test()
       │      delta ~ y + z + x + responsiveness + subject dummies
       │      all electrodes and each hemisphere
       │
       ├─ RELIABILITY: map_reliability()
       │      electrode and parcel levels
       │
       ├─ DESCRIPTIVE: score_centers_per_subject()
       │      subject × hemisphere weighted medoids
       │
       └─ FIGURES + CSV + JSON + summary.txt
```

### 3.1 Anatomy joins

The atlas cache supplies both:

- `roi`: a coarse group defined by `src/analysis/config/rois.py`; and
- `anat`: the raw Destrieux label.

`ANAT_LEVEL=group` tests `roi`; `ANAT_LEVEL=destrieux` tests `anat`.
`ANAT_LEVEL=auto` uses coarse groups for whole brain, but automatically switches
to Destrieux labels when `ROI_FILTER` contains one coarse group—otherwise every
electrode would have the same uninformative coarse label.

Coordinates follow the same reconstruction path used for brain figures:
`jim_mri.subject_to_info` → channel montage → forced MRI frame → subject
Talairach transform → fsaverage/MNI millimetres. Missing reconstruction files
produce warnings and partial coordinates rather than killing the primary ROI
test.

### 3.2 Responsiveness and subject nuisance terms

When the score CSV is produced by the full segregation job, `resp` is carried
into N4. In the current pipeline it defaults to mean absolute HG unless an
external responsiveness dictionary is supplied in code. It controls general
electrode gain so “large response everywhere” is not mistaken for preferential
LWPC or LWPS anatomy.

The implementation uses subject dummy variables as the fixed-effect,
no-shrinkage equivalent of the plan's `(1 | subject)` nuisance intercept. The
permutation supplies inference, so a parametric mixed-model degrees-of-freedom
approximation is not used.

---

## 4. The primary anatomical test

### 4.1 The question

At the selected anatomical level, the conceptual model is:

```text
delta_ij ~ anatomy_j + responsiveness_ij + subject_i
```

The omnibus statistic is a one-way `F`: between-anatomy variation in
nuisance-adjusted `delta` divided by within-anatomy variation. The `F` is only a
scale-free statistic; its reported significance comes from permutations, not a
parametric F distribution.

Because `delta` is formed within each electrode before the model, anatomy is
being tested against the difference between effect types. That is the required
**effect type × anatomy interaction**.

### 4.2 Coverage conditioning

iEEG sampling is clinically determined. `coverage_matrix.csv` has one row per
subject and one column per anatomical unit; a `1` means that subject has at least
one electrode in the unit. Anatomical units covered in fewer than
`MIN_SUBJECTS` are excluded before testing (default `3`).

This does not make implant coverage random. It prevents a label represented by
only one or two people from driving the tested ROI family. Report the retained
units and their subject counts with every anatomical claim.

### 4.3 The null

For each permutation, every electrode independently keeps its subject,
coordinates, anatomy, responsiveness, and pair of observed scores, but LWPC and
LWPS are randomly exchanged. Algebraically this is a sign flip of `delta`.

The Monte Carlo p-value is `(extreme + 1) / (N_PERM + 1)`. With the default
10,000 permutations its smallest possible value is about `0.0001`. Never report
`p = 0`.

### 4.4 Omnibus and per-anatomy rows

The omnibus permutation `p` answers whether relative LWPC/LWPS balance varies
anywhere across the retained anatomical units. `delta_per_roi.csv` then provides
unit-level descriptions:

- `mean_delta`: raw mean `delta`;
- `mean_delta_adj`: mean after removing subject and responsiveness nuisance
  effects—the quantity used by the statistic;
- `p`: two-sided within-electrode-swap permutation p for that adjusted mean;
- `q`: Benjamini–Hochberg value across the retained units;
- `n_electrodes`, `n_subjects`: the sampling behind the row.

Lead with the omnibus test. Use `q`, not an isolated raw row-level `p`, when
identifying the units that explain a significant omnibus result. A positive
adjusted mean is LWPC-dominant; a negative adjusted mean is LWPS-dominant.

### 4.5 Leave-one-subject-out leverage

The same ROI test is rerun after dropping each subject. These folds use fewer
permutations (`max(1000, N_PERM/10)`) because this is an estimate-leverage check,
not a second inferential family.

Read shifts in `observed_stat`, not only shifts in `p`: permutation p-values can
saturate at their resolution floor. A large F collapse when one subject is
removed means the pooled result is fragile even if every fold still prints a
small p-value.

---

## 5. Secondary coordinate test

When coordinates are available, the pipeline also fits, separately by
hemisphere as well as in the pooled coordinate table:

```text
delta ~ mni_y + mni_z + mni_x + responsiveness + subject
```

The block `F/p` tests whether the three-coordinate block predicts relative
effect dominance. Per-axis slopes and permutation p-values describe direction:

| Axis | Positive slope means |
|---|---|
| `mni_y` | LWPC dominance increases anteriorly |
| `mni_z` | LWPC dominance increases superiorly |
| `mni_x` | LWPC dominance increases toward more positive/rightward x; interpret only within hemisphere |

The code reports an `all` fit and, when possible, `lh` and `rh`. Prefer the
hemisphere-specific x slopes because mirrored bilateral x coordinates can
cancel. Treat axis p-values as follow-ups to the coordinate-block test and be
explicit about any multiplicity policy used in the manuscript.

The coordinate model is secondary to the coverage-conditioned categorical
anatomy test, but it is the preferred analysis for a directional claim such as
“LWPC dominance is more anterior.” It uses all positions rather than reducing a
spatial distribution to one point.

---

## 6. Brain maps of continuous scores

The job writes five maps from the same `scores_with_anatomy.csv`:

| Base filename | Values | Colour scale | What it shows |
|---|---|---|---|
| `score_map_lwpc_s.png` | signed pooled-scaled LWPC | symmetric `coolwarm` | direction and location of LWPC adaptation |
| `score_map_lwps_s.png` | signed pooled-scaled LWPS | symmetric `coolwarm` | direction and location of LWPS adaptation |
| `score_map_abs_lwpc.png` | `abs(lwpc_s)` | sequential `viridis` | LWPC magnitude regardless of sign |
| `score_map_abs_lwps.png` | `abs(lwps_s)` | sequential `viridis` | LWPS magnitude regardless of sign |
| `score_map_delta.png` | `lwpc_s - lwps_s` | symmetric `coolwarm` | the relative map tested by N4 |

Each map bins a continuous value into nine colours because the shared brain
renderer accepts one colour per electrode set. Its scale is clipped at the 98th
percentile so a few extremes do not flatten the remaining electrodes. The
adjacent `*_colorbar.png` records the actual range; do not compare hues between
figures without their own colourbars.

Signed and delta scales are centred on zero. Magnitude scales are sequential.
The maps pool electrodes for display, so dense implants are visually prominent.
They are **visualizations, not independent evidence**, and should be read with
`coverage_matrix.csv`, the ROI test, and the leave-one-subject-out table.

If PyVista/MNE/reconstruction/display dependencies are unavailable, the code
writes the colourbar and a `*_by_roi.png` fallback instead. That is a successful
statistical run, but it is not a rendered cortical surface. The Slurm wrapper
uses `xvfb-run` to provide a virtual display; missing recon/template data can
still trigger the fallback.

### `delta_by_roi.png`

This flat companion figure is always attempted. Bars are mean `delta` ± SEM,
dots are electrodes, and labels report electrode and covered-subject counts.
The zero line separates relative LWPC dominance from relative LWPS dominance.
The error bars are descriptive; significance comes from the swap test and the
subject/responsiveness-adjusted statistics, not whether a bar's SEM crosses zero.

### `joint_scatter.png`

Every point is an electrode (`lwpc_s` on x, `lwps_s` on y). It reports both the
pooled and within-subject correlation and, when `per_split.csv` is supplied,
annotates the split-half ceiling. This is useful context for whether the effects
share electrodes, but it is not the primary anatomy interaction.

---

## 7. Reliability and the noise ceiling

A low LWPC–LWPS spatial correlation is uninterpretable unless each map is
reliable enough to correlate with anything. `map_reliability` uses the
split-resolved table:

```text
between = mean over splits of ½[corr(xA, yB) + corr(xB, yA)]
LWPC reliability = mean corr(xA, xB)
LWPS reliability = mean corr(yA, yB)
noise-corrected between = between / sqrt(rel_LWPC × rel_LWPS)
```

It computes these at both electrode and parcel level. The cross-half pairing
prevents shared trial noise from inflating similarity.

Read the result comparatively:

- `between` near the two within-effect reliabilities: maps are about as similar
  as measurement permits;
- `between` well below healthy positive reliabilities: evidence for different
  maps is more credible;
- either reliability near zero or negative: the between-map correlation and
  attenuation correction are not interpretable; collect/aggregate better or
  soften the claim;
- corrected values can exceed ±1 under noisy finite-sample estimates; treat them
  as an instability warning, not a literal correlation.

The anatomy job also reruns the disjoint-half electrode correlation with
`min_elec` equal to 1, 2, and 3. `min_elec` drops whole subjects that have too few
eligible electrodes, so changes across `min_elec_sweep.csv` reveal an important
sample-definition sensitivity.

**Always pass `PER_SPLIT_CSV`.** The job is allowed to run without it, but then
the ceiling and min-electrode sweep are absent and the summary explicitly marks
the spatial correlation as unreportable.

---

## 8. Centres / medoids — descriptive only

The centre panel exists to describe the maps in millimetres, not to establish
anatomical separation. A single pooled centroid is invalid here because subjects
with more electrodes move the location estimate, clinical coverage can mimic a
physiological centre, bilateral distributions can put a centroid in the
midline/white matter, and signed weights can cancel.

The implemented version avoids the worst failures:

- one LWPC centre and one LWPS centre per **subject × hemisphere**;
- the same electrodes for both effects;
- non-negative weights `abs_lwpc` and `abs_lwps`;
- at least three electrodes in a subject × hemisphere group;
- a weighted **medoid** by default—the observed electrode minimizing weighted
  distance to the other electrodes;
- LWPC-minus-LWPS displacement within group; and
- the same within-electrode score-label-swap null.

`score_centers.csv` contains:

| Column | Meaning |
|---|---|
| `subject`, `hemi` | independent descriptive group |
| `n_electrodes` | sites available to locate both medoids |
| `dx`, `dy`, `dz` | LWPC medoid minus LWPS medoid in mm |
| `distance` | Euclidean separation in mm |
| `p_anterior` | within-group swap p for `dy`; descriptive diagnostic |

The summary averages displacement over the eligible subject × hemisphere
groups. Positive `dy` means the LWPC medoid is anterior to the LWPS medoid;
positive `dz` means superior; positive `dx` means more positive x. Lead with the
coordinate regression for an anterior/posterior claim and present the medoids as
an intuitive secondary panel. Do not call their p-values the primary anatomical
test.

---

## 9. How to run it

Run commands from the repository root unless a command explicitly changes
directory.

### 9.1 First validate the entire path with synthetic data

This exercises scores, atlas joins, primary and coordinate tests, reliability,
maps, centres, persistence, and summary writing without accessing real epochs:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

The synthetic default plants a spatial gradient. Also run the null:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0 N_PERM=1000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Use 1,000 permutations only as a path check. Use the default 10,000 for the
reported result.

### 9.2 Produce reusable real-data scores

Edit the hard-coded `EPOCHS_ROOT_FILE`, `WINDOW_TMIN`, and `WINDOW_TMAX` near the
top of `submit_stability_flexibility_segregation_dcc.sh` so they match the
predeclared analysis. Then run the anatomically defined whole-brain score set:

```bash
cd dcc_scripts/stats
ROIS=all ELECTRODES=all CONTRAST_MODE=proportion EFFECT_MEASURE=cohens_d \
N_SPLITS=200 N_PERM_CORR=10000 N_PERM_LABEL=2000 \
  bash submit_stability_flexibility_segregation_dcc.sh
```

Why these values matter:

- `ROIS=all`, `ELECTRODES=all`: no effect-based or lPFC-only selection before
  the whole-brain anatomy question;
- `CONTRAST_MODE=proportion`: score the LWPC/LWPS interactions, not congruency
  and switch main effects;
- `EFFECT_MEASURE=cohens_d`: the signed, balanced, unit-free measure specified
  for N4;
- `N_SPLITS=200`: stable disjoint-half averages and a split-resolved ceiling.

Do **not** use `SCATTER_ONLY=1` as the N4 input. That path is diagnostic and does
not write the complete full-run product set expected here.

The upstream directory is printed as `Save dir:` in the Slurm log and normally
has this shape:

```text
dcc_scripts/stats/results/<epochs>/segregation_results/
  window_<tmin>to<tmax>s_all_all_rois_proportion_cohens_d_fdr_bh/
```

Verify that it contains at least `electrodes.csv` and `per_split.csv`.

### 9.3 Run N4 from those finished scores — recommended route

Set both paths to the files from the **same** segregation run:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/segregation_results/window_..."

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER='' ANAT_LEVEL=group \
MIN_SUBJECTS=3 N_PERM=10000 SEED=0 \
MAKE_BRAIN=1 BRAIN_HEMI=both USE_COORDS=1 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Environment variables not explicitly repeated in the submitter's
`--export=...` list still reach Slurm through `--export=ALL`; the shell-prefix
form above is therefore intentional. Use absolute CSV paths because the batch
job's working directory may differ.

The score-CSV route does not load epochs. It still needs the electrode atlas and,
for coordinate/centre panels, reconstruction files. Set `ROI_DICT_DIR` if the
atlas JSON cache is not in its normal lab location.

### 9.4 Restricted lPFC / Destrieux follow-up

For a predeclared lPFC analysis using raw labels inside lPFC:

```bash
cd dcc_scripts/stats
SEG_DIR="/absolute/path/to/the/same/wholebrain/segregation/run"

ARM=continuous \
SCORES_CSV="$SEG_DIR/electrodes.csv" \
PER_SPLIT_CSV="$SEG_DIR/per_split.csv" \
ROI_FILTER=lpfc ANAT_LEVEL=destrieux HIST_TOP_N=20 \
MIN_SUBJECTS=3 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

`ANAT_LEVEL=auto` would also select Destrieux here. `HIST_TOP_N` belongs to the
categorical histogram path and does not change the continuous primary test or
`delta_by_roi.png`; setting it here is harmless but not necessary.

### 9.5 Compute scores inside the anatomy job — supported but expensive

If no upstream score run exists, omit `SCORES_CSV` and `PER_SPLIT_CSV` and let
the anatomy job load epochs and score them:

```bash
cd dcc_scripts/stats
ARM=continuous DATA_SOURCE=real ROI_FILTER='' ANAT_LEVEL=group \
CONTRAST_MODE=proportion ELECTRODES=all N_SPLITS=200 N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

Before doing this, edit the anatomy submitter's hard-coded `EPOCHS_ROOT_FILE`,
window, and electrode setting. The submitter currently sets `ELECTRODES=sig`, so
the command-line prefix alone cannot override that assignment; change it to
`all` in the file for the anatomically defined N4 population. The DCC core pins
the in-job estimator to `EFFECT_MEASURE='cohens_d'`; it is not an environment
variable exposed by the anatomy runner. Reusing a verified segregation
`electrodes.csv` is nevertheless clearer and safer.

### 9.6 Run without reconstruction/display support

The primary ROI test needs anatomy labels but not coordinates or 3-D rendering:

```bash
cd dcc_scripts/stats
ARM=continuous SCORES_CSV="/abs/path/electrodes.csv" \
PER_SPLIT_CSV="/abs/path/per_split.csv" USE_COORDS=0 MAKE_BRAIN=0 \
ANAT_LEVEL=group N_PERM=10000 \
  bash submit_stability_flexibility_anatomy_dcc.sh
```

This intentionally omits the coordinate test, cortical surfaces, and centres.
It still produces the primary test, coverage table, ROI figure, reliability,
scatter, leverage sweep, CSVs, JSON, and text summary.

### 9.7 Local invocation for debugging

The entry point can be run directly from the repository root, although real
data/recon path discovery is lab-specific:

```bash
ARM=continuous DATA_SOURCE=synthetic N_PERM=200 MAKE_BRAIN=0 \
python dcc_scripts/stats/run_stability_flexibility_anatomy_dcc.py
```

The Slurm wrapper is preferable for real maps because it launches Python under
`xvfb-run`.

---

## 10. Output directory and every output

The runner constructs:

```text
dcc_scripts/stats/results/<tag>/
  anatomy_<label_source>_<scope>_window_<tmin>to<tmax>s_<electrodes>/
    continuous/
```

On the CSV route `<tag>` is not derived from the score filename. With the default
`LABEL_SOURCE=a1` and no epoch directory it currently resolves to the synthetic
fallback tag; with `LABEL_SOURCE=power_traces` it resolves to `power_traces`.
This is only an output-path naming quirk—the loaded CSV remains the data source.
Trust the `Save dir:` line in the job log and archive the exact command/CSV paths
with the result. All files below live in `continuous/`.

### 10.1 Read these first

| File | Contents | How to use it |
|---|---|---|
| `summary.txt` | human-readable primary F/p, per-anatomy rows, LOSO, coordinates, centres, ceiling, correlations | Start here; it states skipped components explicitly. |
| `score_anatomy.json` | machine-readable compact summary of the same analysis | Manuscript/table automation and provenance. It excludes the full permutation arrays. |
| `delta_by_roi.png` | mean ± SEM and electrode dots for the tested relative score | Flat visual companion to the primary model. |
| `coverage_matrix.csv` | subject × tested anatomy presence/absence | Required context for every anatomical conclusion. |

### 10.2 Input/intermediate audit tables

| File | Important columns / interpretation |
|---|---|
| `scores.csv` | loaded or freshly computed per-electrode input (`x`, `y`, `resp`). This is a copy inside the run for provenance. |
| `per_split.csv` | `xA`, `xB`, `yA`, `yB` for every split/electrode; source of reliability and `min_elec` sensitivity. |
| `scores_with_anatomy.csv` | canonical audit table: raw scores, scaled scores, magnitudes, `delta`, ROI, Destrieux label, and coordinates. Use this to reproduce plots or inspect exclusions. |

In `scores_with_anatomy.csv`, check:

- unique `electrode_id` and sensible `subject` counts;
- fraction missing `roi` and `anat`;
- fraction missing MNI coordinates (coordinates may be missing without harming
  the primary ROI test);
- finite `lwpc_s`, `lwps_s`, and `delta`;
- that the requested `ROI_FILTER` was actually applied; and
- extreme values before accepting a percentile-clipped display.

### 10.3 Primary test and robustness tables

| File | Columns | Interpretation |
|---|---|---|
| `delta_per_roi.csv` | anatomy label, electrode/subject N, raw and adjusted means, p, q | Follow-up anatomy rows after the omnibus test. Sign gives relative dominance. |
| `delta_roi_loso.csv` | dropped subject, observed F, p, electrode N | Subject leverage; compare F against `(none)`. |
| `min_elec_sweep.csv` | threshold, cross-effect correlation/p, N, reliabilities, corrected r | Sensitivity to dropping subjects with fewer than 1/2/3 electrodes. Not the primary ROI test. |

### 10.4 Map and scatter figures

| Files | Interpretation |
|---|---|
| `joint_scatter.png` | electrode-level LWPC–LWPS relationship with pooled/within-subject diagnostics and ceiling annotation |
| `score_map_<value>.png` | rendered surface for each of the five values |
| `score_map_<value>_colorbar.png` | required scale for that map |
| `score_map_<value>_by_roi.png` | fallback only when the surface cannot render |

Depending on `BRAIN_HEMI` and the rendering helper, additional view-specific
files may accompany the combined map. `score_anatomy.json` records the combined
path actually returned, or the fallback path.

### 10.5 Centre table

`score_centers.csv` is written only if at least one subject × hemisphere group
has coordinates and at least three eligible electrodes. Its absence can simply
mean insufficient coordinates/coverage; read `summary.txt` and the job log
before treating absence as a pipeline error.

---

## 11. Interpretation checklist

Use this order. It prevents an attractive surface or centroid from outrunning
the actual test.

### A. Validate the population and coverage

- [ ] `ARM=continuous`, `CONTRAST_MODE=proportion`, and the intended time window.
- [ ] Scores came from an anatomically defined set (`all` for whole brain), not
      LWPC/LWPS-significant electrodes.
- [ ] `SCORES_CSV` and `PER_SPLIT_CSV` came from the same upstream run.
- [ ] Mapped electrode and subject counts are plausible.
- [ ] Retained anatomical units meet `MIN_SUBJECTS`; report their coverage.
- [ ] Missing coordinates affect only coordinate/maps/centres, not the primary
      label-based ROI test.

### B. Read the primary interaction

- [ ] Read the omnibus `F` and permutation `p` in `summary.txt`.
- [ ] If significant, use adjusted means and BH `q` in `delta_per_roi.csv` to
      describe where relative dominance lies.
- [ ] Say “relative LWPC/LWPS balance varies by anatomy,” not “LWPC is present
      here and LWPS absent there.”
- [ ] Do not interpret `delta` without checking both signed component maps.

### C. Check robustness

- [ ] Inspect LOSO changes in **F**, especially the largest shift.
- [ ] Compare pooled and within-subject LWPC–LWPS correlations.
- [ ] Inspect `min_elec=1,2,3` rather than reporting only the default survivor
      set.
- [ ] Interpret between-map similarity only relative to both split-half
      reliabilities.
- [ ] Flag undefined/unstable noise correction when reliability is non-positive.

### D. Add secondary spatial descriptions

- [ ] Coordinate block test first, then directional slopes.
- [ ] Prefer hemisphere-specific interpretation of `mni_x`.
- [ ] Use maps to show the data and coverage, not as independent tests.
- [ ] Describe weighted medoids in millimetres as convergent/descriptive only.

### Suggested result language

If the omnibus test is significant and robust:

> The within-electrode balance of pooled-scaled LWPC and LWPS scores varied
> across coverage-eligible anatomical units (swap-permutation omnibus F = …,
> p = …). Adjusted relative scores were LWPC-dominant in … and LWPS-dominant in
> … (BH q = …). The statistic remained similar in leave-one-subject-out fits,
> and the spatial comparison was interpretable given split-half reliabilities of
> … and … .

If it is null with a healthy ceiling:

> Relative LWPC/LWPS balance did not vary detectably across coverage-eligible
> anatomical units (F = …, swap-permutation p = …). Split-half spatial
> reliabilities were … and …, and cross-effect similarity was …; thus the null
> is not readily explained by wholly unreliable maps.

If it is null with a poor ceiling:

> The anatomy interaction was not significant, but the LWPC and/or LWPS map had
> low split-half reliability. The analysis therefore cannot distinguish a
> shared anatomical organization from insufficient precision, and the null is
> inconclusive.

---

## 12. Parameters that change the answer

| Variable | Default in runner | Recommendation / effect |
|---|---:|---|
| `ARM` | `categorical` | **Set `continuous`.** |
| `SCORES_CSV` | unset | Recommended: full segregation run's `electrodes.csv`. |
| `PER_SPLIT_CSV` | unset | Recommended/needed for reportable ceiling and min-electrode sweep. |
| `N_SPLITS` | `200` | Used only if scoring inside anatomy; more splits stabilize estimates but cost time. |
| `CONTRAST_MODE` | `proportion` | Must remain `proportion` for LWPC/LWPS N4. |
| `ROI_FILTER` | whole brain | Changes the tested population; predeclare whole-brain primary vs regional follow-up. |
| `ANAT_LEVEL` | `auto` | Whole brain → coarse groups; one-group restriction → Destrieux. Explicit values improve provenance. |
| `MIN_SUBJECTS` | `3` | Coverage threshold; higher is more conservative but drops anatomical units. |
| `N_PERM` | `10000` | Primary swap-null resolution; lower only for dry runs. |
| `SEED` | `0` | Reproducibility of permutations and plot jitter. |
| `USE_COORDS` | `1` | Controls coordinate test and centres, not the primary ROI test. |
| `MAKE_BRAIN` | `1` | Controls five surface/fallback maps, not statistics. |
| `BRAIN_HEMI` | `both` | Display choice (`both`, `lh`, `rh`, `split`). |
| `ELECTRODES` | runner `all`, submitter `sig` | CSV route inherits its input population; in-job scoring must be changed to `all` for N4. |

Two non-obvious distinctions:

1. `ALPHA` labels significance in the text and is used by some score estimators;
   the N4 primary decision still comes from the reported permutation `p`.
2. `FDR_CORRECTION` and `LABEL_SOURCE` largely describe the categorical arm;
   do not mistake them for filtering continuous scores loaded from CSV.

---

## 13. Common failure modes

### “My job ran categorical anatomy”

`ARM` was omitted. Rerun with `ARM=continuous`; do not interpret the categorical
S/F table as beats 5–7.

### “The anatomy job says the ceiling was not computed”

`PER_SPLIT_CSV` was omitted or unreadable. Point it to `per_split.csv` from the
same segregation run as `SCORES_CSV`.

### “Only lPFC electrodes are present in my whole-brain run”

The upstream segregation submitter defaults to `ROIS=lpfc`. Recompute scores
with `ROIS=all`; changing only `ROI_FILTER` cannot restore electrodes that never
entered `electrodes.csv`.

### “The in-job run used significant electrodes”

The anatomy submitter assigns `ELECTRODES=sig`. Edit it to `all`, or—preferably—
provide a verified all-electrode score CSV.

### “No cortical surfaces appeared”

Read the Slurm log for the renderer exception and look for `*_by_roi.png` files.
Statistics may be complete. Confirm `xvfb-run`, PyVista/MNE, fsaverage/template
surfaces, and subject reconstructions are available.

### “No coordinate result or `score_centers.csv`”

Either `USE_COORDS=0`, reconstruction lookup failed, coordinates were missing,
or no subject × hemisphere had three eligible electrodes. This does not
invalidate the label-based primary test.

### “The omnibus is significant but no row has q < .05”

The omnibus asks whether any between-unit variation exists collectively;
row-level BH tests ask which adjusted unit means differ from zero. These are not
identical questions. Report the omnibus and avoid naming a specific unit as the
driver unless its corrected follow-up supports that claim.

### “The delta map looks dramatic but the primary test is null”

The display pools electrodes, clips extremes, and does not account visually for
subject leverage or coverage. Trust the prespecified swap test and LOSO result;
describe the map as exploratory/descriptive.

### “A medoid p-value is smaller than the ROI-test p-value”

Do not promote it. The centre statistic answers a narrower, coverage-sensitive
location-summary question. The plan explicitly makes it descriptive; the ROI
interaction and coordinate regression remain the inferential analyses.

---

## 14. Reproducibility and tests

Before a production run:

```bash
pytest tests/analysis/stats/test_stability_flexibility_anatomy.py -v
```

The continuous-arm tests verify pooled scaling without losing tiny subjects,
electrode-ID/anatomy/coordinate joins, recovery of planted and null ROI effects,
coverage filtering, sign invariance of the swap test, coordinate-gradient
recovery, reliability degradation with noise, real-electrode medoids, LOSO
coverage, and surface-render fallback behavior.

For every reported run archive:

- the git commit;
- the exact submission command and Slurm log;
- absolute upstream CSV paths;
- epoch directory, window, electrode and ROI scope;
- `ANAT_LEVEL`, `MIN_SUBJECTS`, `N_SPLITS`, `N_PERM`, and seed;
- `summary.txt`, `score_anatomy.json`, all CSVs, and all figures; and
- warnings about missing atlas labels, coordinates, or render fallback.

Related reading:

- [`n2_direction_tests.md`](n2_direction_tests.md) — the direction test that
  should be settled before N4;
- [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md)
  §§5–7 — the statistical rationale and reporting hierarchy;
- [`stability_flexibility_data_flow.md`](stability_flexibility_data_flow.md) —
  broader A1–A6 data flow; and
- [`stability_flexibility_outputs_guide.md`](stability_flexibility_outputs_guide.md)
  — the wider segregation/anatomy output family.
