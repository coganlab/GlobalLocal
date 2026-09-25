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

**Units.** `lwpc_score`/`lwps_score` are Cohen's *d* — a difference-of-differences
of cell means over the pooled within-cell SD of single-trial high-gamma — so the
raw scores are already unit-free. `lwpc_s`/`lwps_s`/`delta` are that *d* divided
by its own across-electrode SD, so their unit is **"SDs of this effect across
this dataset's electrodes"**: `lwpc_s = 1.5` means an electrode 1.5 cross-electrode
SDs above the mean LWPC *d*, not *d* = 1.5 and not a z-score (the mean is not
removed, and nothing is standardized within subject). They are a *relative*,
sample-dependent scale: the same electrode rescored against a different electrode
set gets a different number, so compare them within a run, never across runs, and
quote `lwpc_score`/`lwps_score` when an absolute effect size is wanted.

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

Three things to know before quoting any of these numbers (worked through on
real data in [§15.4](#154-how-reliable-are-the-maps)):

- `between_noise_corrected` is computed from **Pearson** correlations whatever
  `method` is. The attenuation formula is Pearson algebra, and a Spearman ratio
  can sit far above 1.
- `between_noise_corrected_ci` bootstraps the **splits**. The splits all
  re-divide the same trials and electrodes, so it only measures which trials fell
  in which half; it is not a confidence interval. Resample participants for
  sampling uncertainty.
- The within-participant reliabilities that `split_resolved_corr` reports (the
  `min_elec` sweep) are biased low, because `compute_sensitivities_per_split`
  draws a new trial split for each electrode. Treat a zero or negative value
  there as unusable, not as a map with no signal.

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

Before doing this, edit the anatomy submitter's hard-coded `EPOCHS_ROOT_FILE`
and window. The submitter also sets `SCORES_CSV=${SCORES_CSV:-"$SEG_RUN/electrodes.csv"}`,
which treats an empty value as unset, so blank the `SEG_RUN` default in the file
rather than on the command line. The DCC core pins the in-job estimator to
`EFFECT_MEASURE='cohens_d'`; it is not an environment variable exposed by the
anatomy runner. Reusing a verified segregation `electrodes.csv` is nevertheless
clearer and safer.

**`ELECTRODES` does not select electrodes in the anatomy job, on either route.**
On the CSV route the electrode set is whatever the CSV contains
(`load_scores`, `stability_flexibility_anatomy_dcc.py`). On this in-job route,
`run_stability_flexibility_anatomy_dcc.py` hard-codes `ROIS_DICT = None`, and
`resolve_electrodes_to_keep` keeps every channel when that is `None`. The only
visible effect of `ELECTRODES=sig` is the `_sig` suffix on the output directory.
To analyse task-significant electrodes, produce their scores upstream: set
`ELECTRODES=sig` in `submit_stability_flexibility_segregation_dcc.sh` (it is
hard-coded there too, so edit the line), run it, and point `SCORES_CSV` and
`PER_SPLIT_CSV` at the resulting `window_<tmin>to<tmax>s_sig_<rois>_...`
directory.

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

For the lPFC runs, the worked version of this language, covering both the
all-electrode and task-significant sets, is in
[§15.12](#1512-reporting).

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
| `BRAIN_HEMI` | `both` | Display choice (`both`, `lh`, `rh`, `split`). `split` draws one hemisphere per panel in a window twice as wide. |
| `BRAIN_ZOOM` | renderer default | Per-panel camera zoom (`<1` zooms out). Display choice only; lower it to widen the gap between the `split` hemispheres. |
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

### “I set `ELECTRODES=sig` but the run used every electrode”

`ELECTRODES` has no effect on which electrodes the anatomy job analyses. With
`SCORES_CSV` set, the electrodes are the ones in that CSV; computing scores in
the job keeps every channel because the runner hard-codes `ROIS_DICT = None`.
The output directory still ends in `_sig`, so rename or delete it. To analyse
task-significant electrodes, run the segregation job with `ELECTRODES=sig` and
point `SCORES_CSV` / `PER_SPLIT_CSV` at that run ([§9.5](#95-compute-scores-inside-the-anatomy-job--supported-but-expensive)).

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

---

## 15. Findings from the lPFC runs

Two runs of the continuous arm on lPFC (`roi == 'lpfc'`), both scored on 200
disjoint half-splits with `CONTRAST_MODE=proportion` and
`EFFECT_MEASURE=cohens_d`:

| run | electrodes | participants | upstream segregation run |
|---|---|---|---|
| **all lPFC** | 398 (254 lh / 144 rh) | 22 | `window_0.0to1.5s_all_lpfc_proportion_cohens_d_fdr_bh` |
| **task-significant lPFC** | 171 (120 lh / 51 rh) | 21 | `window_0.0to1.5s_sig_lpfc_proportion_cohens_d_fdr_bh` |

"Task-significant" is `ELECTRODES=sig`: electrodes significant against baseline
in the epochs file's stimulus test, the set the power-trace and decoding
analyses use. It is not selected on LWPC or LWPS. Every task-significant
electrode is also in the all-lPFC run with identical scores, so the smaller run
is a subset of the larger.

Every number below is reproduced from the runs' `scores_with_anatomy.csv` and
`per_split.csv` by

```bash
python dcc_scripts/stats/n4_section15_followups.py \
    --scores        <all-lPFC run>/scores_with_anatomy.csv \
    --per-split     <all-lPFC run>/per_split.csv \
    --subset-scores <task-significant run>/scores_with_anatomy.csv
```

and by the same command on the task-significant run without
`--subset-scores`. Each block of output is labelled `[pipeline: <function>]` or
`[follow-up]`. Swap-null p-values use 20,000 permutations and coordinate-shuffle
p-values 2,000. The co-localization test in §15.5 is quoted from the segregation
runs' own `summary.txt`. The few numbers that were computed by hand and are not
in the script are marked as such.

### 15.1 Takeaway

1. **LWPC and LWPS are carried by one intermixed lPFC population, not separate
   modules.** Across all lPFC electrodes the two effects are positively
   correlated (pre-specified test: r = +0.097, p ≤ 0.001), and electrodes
   positive for each effect sit in the same place (centroids 1.4 mm apart,
   p = 0.95). The task-significant set shows the same pattern at lower power
   (r = +0.077, p = 0.057; centroids p = 0.44).
2. **The balance between them tilts along the dorsoventral axis.** Ventral lPFC
   is roughly balanced and dorsal lPFC leans LWPS. Across all lPFC the z slope of
   delta is −0.0077 SD/mm (p = 0.0075), and it replicates across disjoint trial
   halves (p = 0.005). Task-significant electrodes have the same slope
   (−0.0075 SD/mm) but too few electrodes for significance (p = 0.24), and their
   slope does not differ from the other electrodes' (p = 0.67).
3. **The tilt is an effect-type × height interaction, not segregation.**
   Electrodes positive for each effect occur at every height, the correlation
   between the effects is positive in every height band, and height explains 2 %
   of delta's variance. Among task-significant electrodes both effects are
   positive at every height; LWPC is simply lower dorsally.
4. **The dorsoventral axis was not predicted.** The predicted anterior–posterior
   axis is null in both sets (p = 0.58 and 0.30). z is one of three axes, so the
   three-axis block test (p = 0.032) is the protected result, and the
   Bonferroni-corrected z p is 0.022.

Single electrodes are mostly noise: full-data split-half reliability is
0.26–0.39, and an electrode's sign agrees between its two trial halves only
51–59 % of the time. Interpret population summaries only.

### 15.2 Corrections to the previous revision

The previous version of this section made five claims that do not hold.

| previous claim | correction |
|---|---|
| The two maps are "correlated at their noise ceiling" (noise-corrected r = 1.02, 95 % interval [0.998, 1.046]) | That interval resamples the 200 splits, which re-divide the same trials and electrodes; it measures only which trials fell in which half. Resampling participants gives [0.54, 4.71], with 9 % of resamples undefined. The ratio is not estimable at these reliabilities (§15.4). |
| On the split-averaged maps the disjoint-half within-subject correlation is r = +0.243 | Averaging each half over 200 splits rebuilds the full-data map (corr(mean xA, mean xB) = 0.987), so this correlation shares trials. It is the same quantity as `summary.txt`'s within-subject r = +0.224. |
| The §5.1 sweep's negative `reliability_y` comes from within-subject centring removing variance | Centring pushes a reliability toward zero, not below it, and cannot explain a cross-map r larger than √(rel_x · rel_y). The likely cause is the per-electrode split (§15.4). |
| Testing the separation of the sign-defined centres would be circular | A test is valid when the two sets are re-formed inside every permutation (§15.10). It restates the slope rather than adding evidence. |
| LWPC dominance increases ventrally | delta is negative on average in every height band, because LWPS is larger overall. Ventral lPFC is roughly balanced (mean delta −0.02) and dorsal lPFC leans LWPS (−0.24). |

The magnitude null was also called "settled". It is better described as
uninformative, because absolute values cannot register a small shift of scores
centred near zero (§15.9).

### 15.3 Vocabulary used in this section

Each electrode carries two **signed** scores: positive means the effect runs in
the direction behaviour predicts, negative means it runs the other way.

```text
delta = lwpc_s - lwps_s        # positive = LWPC-dominant, negative = LWPS-dominant
```

Electrodes are grouped by whether their two scores point the same way:

- **concordant**: both scores have the **same** sign (`++` or `--`).
- **discordant**: the scores have **opposite** signs (`+-` or `-+`).

This matters because `delta` mixes two things a reader might not want mixed:

| electrode | `lwpc_s` | `lwps_s` | `delta` | group | bigger in magnitude |
|---|---|---|---|---|---|
| A | +1.5 | +0.5 | **+1.0** | concordant `++` | LWPC |
| B | −1.5 | −0.5 | **−1.0** | concordant `--` | LWPC |
| C | +1.0 | −1.0 | **+2.0** | discordant `+-` | neither (tied) |
| D | +1.0 | +1.0 | **0.0** | concordant `++` | neither (tied) |

A and B have identical magnitude relationships but **opposite deltas**; C has the
largest `delta` despite its two effects being equal in size. None of this needs
resolving to read the results, because the tests are about how `delta` changes
across electrodes, not about any one electrode's value (§15.7).

### 15.4 How reliable are the maps?

Split-half reliability is computed within each split, where `xA` and `xB` are
disjoint trial halves, and averaged over splits (`map_reliability`):

| | all lPFC: LWPC | all lPFC: LWPS | task-sig: LWPC | task-sig: LWPS |
|---|---|---|---|---|
| Spearman, half data | +0.162 | +0.097 | +0.145 | +0.200 |
| Pearson, half data | +0.178 | +0.163 | +0.148 | +0.241 |
| Pearson, full data (Spearman–Brown) | 0.30 | 0.28 | 0.26 | 0.39 |
| sign agreement between halves | 55.5 % | 51.0 % | 56.7 % | 59.4 % |

⚠️ **Do not compute reliability after averaging over splits.** The A-half of one
split overlaps the B-half of another, so `corr(x̄A, x̄B)` across split-averaged
maps returns **+0.987**, an artefact, not a ceiling.

⚠️ **Do not Spearman-Brown the reliabilities for the attenuation correction.**
Both sides of `map_reliability` are half-length (`between` correlates two
half-trial estimates, and so do the reliabilities), so the ratio is already at
matched trial counts. The full-data row above is only for reading the averaged
map.

⚠️ **The +1.374 in the archived `summary.txt` was a rank-transform artefact.**
The attenuation formula is algebra for **Pearson** correlations, while `method`
defaults to Spearman, and ranking deflates the self-reliabilities much more than
the cross term:

| all lPFC | `between` | ceiling √(rel·rel) | ratio |
|---|---|---|---|
| Spearman (archived) | +0.172 | 0.125 | +1.374 |
| Pearson | +0.174 | 0.170 | +1.022 |

`map_reliability` now computes `between_noise_corrected` from Pearson whatever
`method` is, and sets `note` when the ratio is out of range or undefined.

⚠️ **The noise-corrected ratio is not estimable at these reliabilities.** The
Pearson ratio is 1.022 in all lPFC and 0.684 in the task-significant run. The
`between_noise_corrected_ci` that `map_reliability` returns bootstraps the
splits, so it only reflects which trials fell in which half (all lPFC
[0.998, 1.046]; task-significant [0.654, 0.716]). Resampling participants
gives the sampling uncertainty:

| | participant bootstrap, 95 % | resamples undefined |
|---|---|---|
| all lPFC | [0.54, 4.71] | 9 % |
| task-significant | [0.25, 2.64] | 10 % |

Report `between` and the two reliabilities, never the ratio or the split
interval. The parcel-level ratio is undefined in both runs, because the LWPC
parcel map's Pearson reliability is negative (−0.026 all lPFC, −0.201
task-significant).

⚠️ **Within-participant reliabilities are biased low by the split design.**
`split_resolved_corr`, the pre-specified co-localization test (§15.5), centres
each score within participant before computing reliabilities. There they come
out at +0.069 / −0.094 for LWPC / LWPS in all lPFC (Pearson: +0.095 / −0.079)
and +0.014 / −0.000 in the task-significant run. Two
independent halves of a real map cannot have a negative expected correlation,
and a cross-map correlation cannot exceed √(rel_x · rel_y). In all lPFC the
cross-map r (+0.105, Pearson) breaks that bound in 99.75 % of participant
resamples, so this is not sampling noise.

The likely cause: `compute_sensitivities_per_split` draws a new trial split for
**each electrode**. Within a participant, electrode i's half A therefore shares
about a quarter of the trials with electrode j's half B. Trial-level noise shared
across a participant's electrodes then correlates one electrode's half A with its
neighbours' half B, and within-participant centring turns that into a negative
correlation between each electrode's own two halves. In a simulation with
common-mode noise correlation 0.2–0.4, within-participant reliabilities that are
+0.06 to +0.10 with one split per participant fall to −0.04 to −0.15 with one
split per electrode.

This does not affect any reported result:

| result | affected by per-electrode splits? | why |
|---|---|---|
| parcel test, coordinate slopes, sign subsets, \|score\| | no | these use scores averaged over all splits, which equal the full-data scores however the splits were drawn |
| cross-validated μ² | no | it multiplies an electrode's own two halves, which never share trials |
| co-localization r | negligibly | it compares two different contrasts; same-half and separate-half correlations agree (§15.5) |
| cross-validated slope | negligibly, toward zero | simulation: 0.0041 vs 0.0040 for a true 0.004 |
| pooled (uncentred) reliabilities | negligibly | the bias is about 18 times smaller than within participant |
| within-participant reliabilities | yes, pushed down | not reported |

One split per participant, shared by all its electrodes, is needed only to report
a within-participant reliability or ceiling, or to confirm this explanation on
the real trials. It has not been implemented. (The bound check and both
simulations were run by hand and are not in the script.)

**What low reliability does and does not invalidate.** A reliability of ~0.3
means individual electrode scores are mostly noise, so no single electrode
should be interpreted and any dot map is largely noise. It does **not**
invalidate a slope or a correlation across electrodes: those summaries average
over hundreds of noisy electrodes, and §15.7 checks the slope directly on
separate trial halves.

### 15.5 Do the two effects share electrodes?

**Pre-specified test.** `split_resolved_corr` in the segregation runs: LWPC on one
trial half against LWPS on the other, residualised on responsiveness, centred
within participant, Spearman, participants with at least three electrodes
(`min_elec = 3`). From each segregation run's `summary.txt`:

| | all lPFC | task-significant |
|---|---|---|
| r | **+0.097** | +0.077 |
| p (1,000 permutations) | **≤ 0.001** | 0.057 |
| electrodes / participants | 397 / 21 | 167 / 18 |
| participants dropped (< 3 electrodes) | D0069 | D0065, D0110, D0145 |

With 1,000 permutations p cannot go below 0.001; the anatomy run's `min_elec`
sweep (2,000 permutations) gives p = 0.0005 for the same all-lPFC value. For an
exact p, rerun the segregation job with `N_PERM_CORR=10000`. Report the Spearman
values: the Pearson version (task-significant r = +0.091, p = 0.034) is not the
pre-specified test.

Reading: across all lPFC the two effects share signal. In the task-significant
set the association is not significant, and because its within-participant
reliabilities are about zero, that null is not evidence of independence either.
The two estimates are close (0.08 vs 0.10).

**Supporting checks** (follow-ups, section 4 of the script):

| | all lPFC | task-significant |
|---|---|---|
| electrodes positive on both effects | 135 vs 121.6 ± 4.2 expected, p = 0.001 | 96 vs 90.0 ± 2.3 expected, p = 0.01 |
| responsiveness vs LWPC / LWPS, within participant | +0.14 / +0.24 | +0.16 / +0.33 |

The expected counts come from shuffling the LWPS signs among each participant's
electrodes. The count uses only the signs of full-data scores, so it is weaker
than the pre-specified test and should not outrank it. More responsive
electrodes adapt more on both effects, but controlling for responsiveness only
trims the all-lPFC separate-half correlation from about +0.12 (participant only)
to +0.105 (participant and responsiveness, Pearson).

**Not shared-trial noise.** Correlations per split, averaged over splits:

| | all lPFC: pooled r | all lPFC: within participant | task-sig: within participant |
|---|---|---|---|
| `xA` vs `yA` (same trials) | +0.169 | +0.116 | +0.097 |
| `xB` vs `yB` (same trials) | +0.200 | +0.146 | +0.126 |
| `xA` vs `yB` (separate trials) | +0.174 | +0.126 | +0.118 |
| `xB` vs `yA` (separate trials) | +0.174 | +0.114 | +0.115 |

Same-trial and separate-trial correlations agree, so sharing trials does not
inflate the association.

**The categorical (CMH) test is not computable in either run.** After FDR, no
electrode is individually significant for LWPC in either run (all lPFC: 0 LWPC,
0 LWPS; task-significant: 0 LWPC, 8 LWPS), so there are no groups to compare and
the odds ratio is `nan`. The segregation summary's "segregated (n.s.)" label
comes from the missing odds ratio, not from evidence of segregation. Report the
test as not computable (see `stability_flexibility_outputs_guide.md`).

**Thresholded maps look disjoint whatever the truth.** With 49 % of all-lPFC
electrodes LWPC-positive and 56 % LWPS-positive, independence alone would leave
only 28 % in both maps. Apparent segregation in a thresholded dot map is not
evidence of segregation; §15.6 tests location directly.

### 15.6 Are LWPC-positive and LWPS-positive electrodes in different places?

`centroid_shuffle_test` (section 10 of the script). Only electrodes positive on
at least one score enter. The statistic is the distance between the centroid of
the LWPC-positive electrodes and the centroid of the LWPS-positive ones; an
electrode positive on both counts toward both. The null shuffles the electrode
types among electrodes of the same participant, which keeps every position and
each participant's mix of types.

| | all lPFC | task-significant |
|---|---|---|
| electrodes (LWPC+ / LWPS+ / both) | 284 (195 / 224 / 135) | 147 (114 / 129 / 96) |
| centroid distance | 1.4 mm | 3.3 mm |
| shuffle within participant | **p = 0.95** | **p = 0.44** |
| shuffle within participant × hemisphere | p = 0.78 | p = 0.45 |
| left hemisphere only (within participant) | 3.4 mm, p = 0.063 | 2.4 mm, p = 0.65 |
| right hemisphere only (within participant) | 3.8 mm, p = 0.41 | 0.4 mm, p = 0.99 |

No separation in either set. Design notes:

- **Shuffle within participant.** The mix of types differs a lot between
  participants, and participants cover different parts of lPFC. A shuffle across
  all electrodes would let "this participant is LWPS-heavy and happens to have
  dorsal electrodes" pass for anatomy. Here it happens not to change the answer
  (across all electrodes: p = 0.79 all lPFC, 0.17 task-significant).
- **Read the within-hemisphere rows.** The pooled 3-D distance in all lPFC is
  almost all x (1.39 of 1.40 mm), so it mostly reflects how the two sets split
  between hemispheres.
- **It tests location, not balance.** Only signs enter, electrodes negative on
  both effects are left out, and electrodes positive on both pull the two
  centroids together.

**Why it cannot see the dorsoventral tilt.** Take a dorsal electrode with LWPS 0.5
and LWPC 0.2, and a ventral one with LWPC 0.5 and LWPS 0.2. Both are in both sets,
so the centroids coincide, yet the balance flips from dorsal to ventral. "Positive
electrodes of each kind sit in the same place" (this section) and "the balance
shifts with height" (§15.7) are both true.

### 15.7 The balance tilts along the dorsoventral axis

`relative_score_coordinate_test(value_col='delta')`:
`delta ~ mni_y + mni_z + mni_x + responsiveness + participant`, with the swap
null. The parcel test (`relative_score_roi_test(roi_col='anat')`) is the plan's
primary anatomical test and is shown alongside.

| | all lPFC | task-significant |
|---|---|---|
| parcel omnibus (primary) | F = 1.90, **p = 0.010** (19 parcels) | F = 0.91, p = 0.29 (15 parcels) |
| three-axis block F | 2.87, **p = 0.032** | 1.44, p = 0.29 |
| z slope (SD/mm) | **−0.0077, p = 0.0075** | −0.0075, p = 0.24 |
| y slope (SD/mm) | +0.0024, p = 0.58 | +0.0084, p = 0.30 |
| x slope (SD/mm) | +0.0013, p = 0.62 | −0.0002, p = 0.97 |

Units: each score is divided by its SD across the run's electrodes, so a slope is
in those SDs per mm. Across the central 90 % of the all-lPFC z range (67 mm)
delta changes by about 0.44 SD of delta.

**Read it as an interaction.** The z slope of delta is exactly the LWPC slope minus
the LWPS slope (all lPFC: −0.0037 − 0.0040 = −0.0077). So the test asks one
question: do the two effects change differently with height? That is an
effect-type × height interaction, tested by swapping the effect labels within
electrodes. Subtracting within an electrode is the paired version of it: it
cancels anything an electrode contributes to both scores (gain, responsiveness,
participant). Neither effect's own slope is significant (all lPFC: LWPC −0.0037,
p = 0.12; LWPS +0.0040, p = 0.06; task-significant: −0.0046, p = 0.29 and
+0.0029, p = 0.41), so describe the result as LWPC **relative to** LWPS, not as
either effect changing on its own.

**What it looks like.** Height bands (tertiles of the all-lPFC z; Cohen's d
adjusted for participant and responsiveness; section 12 of the script):

| band (MNI z, mm) | all lPFC: n | LWPC d | LWPS d | mean delta | LWPC–LWPS r | task-sig: n | LWPC d | LWPS d |
|---|---|---|---|---|---|---|---|---|
| ventral (−22 to 16) | 133 | 0.01 | 0.02 | −0.02 | +0.10 | 52 | 0.15 | 0.17 |
| middle (16 to 38) | 132 | 0.06 | 0.08 | −0.18 | +0.10 | 72 | 0.16 | 0.17 |
| dorsal (38 to 76) | 133 | −0.04 | 0.05 | −0.24 | +0.13 | 47 | 0.08 | 0.20 |

LWPC–LWPS r is computed on separate trial halves, within participant. The band
means are descriptive; the test is the slope.

- **Negative scores are not needed for the tilt.** Among task-significant
  electrodes both effects are positive at every height; LWPC is lower dorsally
  and LWPS is not.
- **A positive correlation and a tilt coexist.** In every band the effects are
  positively correlated while the balance shifts: a shared component plus a
  small height-dependent shift. Height explains 2.1 % of delta's variance within
  participant.

**Replication across trial halves.** The slope refitted separately on each half of
every split (all lPFC; task-significant in brackets):

| | result |
|---|---|
| mean z slope, half A / half B | −0.00780 / −0.00759 (−0.00784 / −0.00710) |
| same sign in both halves | 98.5 % of splits (79.5 %) |
| E[slope_A × slope_B] | 4.96 × 10⁻⁵ (2.34 × 10⁻⁵) |
| share of the observed slope that is signal | 91 % (65 %) |
| within-participant coordinate shuffle | **p = 0.005** (p = 0.15) |

Because the halves share no trials, E[slope_A · slope_B] is unbiased for the
squared slope. This controls trial noise, not participant sampling.

**Robustness (all lPFC).**

| check | result |
|---|---|
| add a hemisphere term | z slope −0.0074, p = 0.009 |
| z × hemisphere interaction | +0.0031, p = 0.65 (one shared slope) |
| drop the responsiveness covariate | z p = 0.007 |
| leave one participant out | z p < 0.05 in 21/22 folds (worst: drop D0146, p = 0.09) |

The separate hemisphere fits (left z p = 0.21, right p = 0.98) are underpowered,
and the interaction finds no difference to explain. In the task-significant run
no leave-one-out fold reaches p < 0.05 (worst p = 0.45), as expected at its
power.

**Parcels (all lPFC).** The extremes order dorsoventrally, matching the slope:

| Destrieux label | n | adj. mean `delta` | p | q |
|---|---|---|---|---|
| `lh_S_front_sup` | 33 | −0.54 | 0.009 | 0.126 |
| `lh_G_front_sup` | 47 | −0.30 | 0.013 | 0.126 |
| `rh_G_front_middle` | 34 | +0.41 | 0.020 | 0.129 |
| `lh_G_front_inf-Triangul` | 26 | +0.31 | 0.124 | 0.337 |

No label survives FDR, so the omnibus is the claim. In the task-significant run
one parcel (`rh_S_front_sup`, 7 electrodes from 4 participants) has q = 0.012,
but a parcel row means something only after a significant omnibus, and that
omnibus is p = 0.29. Do not report it.

**Task-significant electrodes vs the rest** (section 11, fitted in the all-lPFC
run's units):

| | electrodes | z slope (SD/mm) | p |
|---|---|---|---|
| task-significant | 171 | −0.0076 | 0.24 |
| other lPFC | 227 | −0.0103 | 0.001 |
| difference | | +0.0027 | 0.67 |

Task significance itself is not organised by height (within participant
r = +0.005, p = 0.93), so restricting to it changes the number of electrodes, not
the spatial sampling. Planting the all-lPFC slope on the task-significant layout
at the observed reliability, p < 0.05 is reached 26 % of the time, against 66 %
on the all-lPFC layout (section 9). The task-significant null is what that power
predicts.

### 15.8 The anterior–posterior hypothesis is null

`delta ~ y`: all lPFC p = 0.58 (left 0.77, right 0.22); task-significant
p = 0.30 (left 0.62, right 0.69). The §8 centre machinery is built around this
axis (`p_anterior`). Report it as a null.

### 15.9 Why unsigned and subset versions do not show it

| value regressed on z | all lPFC | task-significant | null |
|---|---|---|---|
| `delta` (signed) | −0.0077, p = 0.0075 | −0.0075, p = 0.24 | swap |
| `abs_lwpc − abs_lwps` | +0.0023, p = 0.32 | −0.0068, p = 0.17 | swap |
| cross-validated μ²: E[xA·xB] − E[yA·yB] | +0.0046, p = 0.40 | −0.0163, p = 0.22 | swap |
| `lwpc_s` alone | −0.0037, p = 0.12 | −0.0046, p = 0.29 | coordinate shuffle |
| `lwps_s` alone | +0.0040, p = 0.06 | +0.0029, p = 0.41 | coordinate shuffle |
| `delta`, positive-on-both electrodes only | −0.0059, p = 0.26 (n = 135) | −0.0047, p = 0.54 (n = 96) | swap |

**Absolute values cannot register a small shift around zero.** Across all lPFC
both scores are centred near zero (51 % of LWPC and 44 % of LWPS scores are
negative), and across the covered z range each score shifts by only about
0.25 SD. For a score distributed N(μ, 1), E|x| ≈ 0.80 + 0.4μ², which is nearly
flat near zero: moving μ from 0 to 0.25 changes E|x| by about 0.025. In the
task-significant set, where most scores are positive (67 % LWPC, 75 % LWPS; mean
d 0.14 and 0.18), |score| ≈ score, and the absolute version gives nearly the
signed slope (−0.0068 vs −0.0075). So the all-lPFC magnitude null reflects the
transform, not evidence that negative scores drive the tilt.

The cross-validated μ² (the mean over splits of xA·xB) removes the noise floor
that biases |score| upward (E|score| ≈ 0.8σ for an electrode with no effect), but
it is also quadratic in μ and just as insensitive near zero; its point estimate
points one way in all lPFC and the other in the task-significant set. Neither
unsigned version is informative. Say that unsigned measures did not detect the
tilt, not that "neither effect's magnitude changes".

**Fitting only the electrodes positive on both.** The test is valid, because
swapping the labels keeps an electrode in the `++` set, but selecting on noisy
signs shrinks the slope and discards most of the data. Planting the observed
gradient on the all-lPFC layout at the observed reliability (section 9):

| analysis | mean slope (true −0.0077) | P(p < 0.05) |
|---|---|---|
| all electrodes | −0.0079 | 66 % |
| `++` electrodes, selected on the same data | −0.0033 | 10 % |
| `++` selected on half A, fitted on half B | −0.0073 | 13 % |

With no gradient planted, all three give p < 0.05 about 5 % of the time. Only
about 68 % of electrodes that look `++` are truly `++`. The observed `++` result
(p = 0.26, same direction) is what a real gradient would produce. Selecting on
one score alone (for example LWPC > 0) is different: the swap moves electrodes in
and out of the set, so the swap null no longer applies.

**Is `delta` just picking up sign disagreement?** Concordance partitions are
invariant under the label swap, so the null stays valid inside each (all lPFC):

| subset | n | `delta` z slope | p |
|---|---|---|---|
| all | 398 | −0.0077 | 0.0075 |
| concordant (`++` or `--`) | 249 | −0.0082 | 0.003 |
| discordant (`+-` or `-+`) | 149 | −0.0149 | 0.029 |
| `++` only | 135 | −0.0059 | 0.26 |
| `--` only | 114 | −0.0079 | 0.023 |

No: the tilt is present with the same sign in every subset. In the
task-significant set every subset is non-significant (concordant p = 0.32), as
expected at its size.

**Which effect dominates moves with height; neither effect's sign does**
(within-participant correlation with z, coordinate shuffle):

| | all lPFC | task-significant |
|---|---|---|
| P(`lwpc_s` > 0) | r = −0.021, p = 0.67 | r = +0.012, p = 0.89 |
| P(`lwps_s` > 0) | r = +0.043, p = 0.42 | r = +0.131, p = 0.12 |
| P(`delta` > 0) | **r = −0.144, p = 0.006** | r = −0.096, p = 0.21 |

⚠️ **Null validity.** The sign-flip null in `_swap_null` is valid only for a
paired difference, where negation equals the label swap. It is valid for
`delta`, `abs_lwpc − abs_lwps` and the μ² difference, and **invalid** for any
single score. Passing `value_col='lwpc_s'` or `'abs_lwpc'` to
`relative_score_coordinate_test` tests nothing; the single-score rows use a
within-participant coordinate shuffle instead.

### 15.10 Weighted centres

`score_centers_per_subject` is null on every axis under every weighting:

| weighting / centre | all lPFC: dx (p) | dy (p) | dz (p) | task-sig: dz (p) |
|---|---|---|---|---|
| `abs`, `medoid=True` | +0.69 (0.59) | +0.46 (0.80) | −3.26 (0.15) | −2.35 (0.44) |
| `abs`, `medoid=False` | +0.61 (0.35) | +0.89 (0.33) | +1.42 (0.26) | −1.22 (0.35) |
| positive-clipped, `medoid=True` | +2.08 (0.29) | −0.86 (0.76) | −1.95 (0.56) | +0.09 (0.98) |

This is expected. The centres weight by |score|, which cannot see a shift of
signed scores around zero (§15.9). A null centre does not qualify §15.7.
Implementation hazards:

- `medoid=True` returns **exactly zero** displacement for 6 of 25 participant ×
  hemisphere groups in all lPFC (7 of 20 in the task-significant run). The
  medoid takes only *n* discrete values, and on clustered depth shafts both
  labels snap to the same contact. `medoid=False` produces none.
- The two centre definitions disagree in sign on dz in all lPFC (−3.26 vs
  +1.42).
- Positive clipping leaves some groups with all-zero weights for one effect;
  those centres are undefined and returned as zero.

**Centres of the sign-defined sets** describe the tilt directly:

| all lPFC | LWPC-dominant (`delta > 0`) | LWPS-dominant (`delta < 0`) | Δz |
|---|---|---|---|
| pooled | n = 186, z̄ = 24.6 | n = 212, z̄ = 29.3 | −4.6 mm |
| lh | n = 118, z̄ = 21.0 | n = 136, z̄ = 28.3 | −7.3 mm |
| rh | n = 68, z̄ = 30.9 | n = 76, z̄ = 31.1 | −0.1 mm |

Averaged over participant × hemisphere groups (both sets ≥ 2 electrodes, 21
groups) Δz is −1.83 mm, 14 of 21 in the expected direction; the pooled rows are
coverage-inflated. As a test, re-forming the two sets inside every swap and
centring heights within participant, Δz = −5.1 mm, p = 0.009 (task-significant:
−2.7 mm, p = 0.29). The two Δz values differ because one averages groups and the
other averages electrodes. The test is valid, but it restates §15.7 rather than
adding independent evidence.

### 15.11 Which electrode set to report

The choice of primary population is open. The facts that bear on it:

- **Consistency.** The power-trace and decoding analyses use the
  task-significant electrodes.
- **The plan.** §5.1 of `analysis_plan_concurrent_regulation.md` and §2.3 of
  this document specify an anatomical electrode set, "not effect-selected". The
  task-significant selection is not made on LWPC or LWPS, so it does not create
  the circularity those sections warn about, but "anatomical" most naturally
  means every electrode in the region.
- **Power, not a different result.** Both positive findings (co-localization and
  the tilt) are significant only in all lPFC. Their estimates are the same in the
  task-significant set (r 0.08 vs 0.10; slope −0.0075 vs −0.0077), the two
  groups' slopes do not differ (p = 0.67), and task significance is unrelated to
  height (p = 0.93).

Options:

1. **Task-significant as primary.** Consistent with the rest of the paper. The
   anatomy section then claims no evidence of segregation and a non-significant
   trend toward overlap, with the all-lPFC results as the larger-sample check.
2. **All lPFC as primary.** Justified by the plan's anatomical population, stated
   in Methods. The task-significant set becomes the consistency check. Without
   that justification written down, switching populations for one section will
   read as choosing the set that gives significance.

Whichever is primary, report the other in full.

### 15.12 Reporting

**Tests to cite.**

| claim | test | code |
|---|---|---|
| the effects share signal | pre-specified continuous correlation on separate halves | `split_resolved_corr` (segregation `summary.txt`) |
| no anatomical separation | centroid shuffle within participant | `centroid_shuffle_test` (script section 10) |
| balance differs by parcel | parcel omnibus, swap null | `relative_score_roi_test` |
| balance tilts with height | coordinate regression, swap null; Bonferroni over 3 axes | `relative_score_coordinate_test` |
| the tilt is not trial noise | slope refitted on each half, coordinate shuffle | script section 5 |
| same tilt in both electrode sets | subset × z interaction, swap null | script section 11 |
| anterior–posterior null | coordinate regression, y | `relative_score_coordinate_test` |

**Figures.**

- LWPC and LWPS against height: band or binned means ± SEM across participants,
  two lines, one per effect. This is the figure for the tilt; the pipeline does
  not make it yet.
- `joint_scatter.png`: LWPC against LWPS. Re-annotate it with the pre-specified
  r and remove the noise-corrected value it currently prints.
- `delta_by_roi.png`: adjusted mean delta per parcel, reordered by mean z.
- Per-electrode dot maps only as coverage or illustration, with a legend line
  saying single electrodes are not interpretable. Never as evidence.

**Draft Results paragraph** (task-significant electrodes as primary; for all lPFC
as primary, swap the order of the two paragraphs and drop "Because this subset
was small"):

> To test whether LWPC and LWPS adaptation are carried by separate lPFC
> populations, we scored each task-responsive lPFC electrode (171 electrodes, 21
> participants) for both effects as a signed, standardized difference-of-differences
> in high-gamma power. Both effects were positive on average (mean Cohen's
> d = 0.14 for LWPC and 0.18 for LWPS). Single-electrode estimates were noisy
> (split-half reliability 0.26–0.39), so we tested only population-level
> summaries. We found no evidence that the two effects occupy different
> electrodes or regions. LWPC and LWPS scores measured on separate halves of the
> trials were weakly and non-significantly correlated (Spearman r = 0.08,
> p = 0.057; 167 electrodes from 18 participants with at least three
> electrodes). Electrodes positive for LWPC and for LWPS did not differ in
> location (centroid distance 3.3 mm, p = 0.44, electrode labels shuffled within
> participant), and the balance between the two effects (LWPC − LWPS) did not
> differ across Destrieux parcels (F = 0.91, p = 0.29).
>
> Because this subset was small, we repeated the analyses on all lPFC electrodes
> (398 electrodes, 22 participants). LWPC and LWPS were positively correlated
> (r = 0.10, p ≤ 0.001), and electrodes positive for each were again not
> spatially separated (centroid distance 1.4 mm, p = 0.95). The balance between
> the two effects, however, differed across parcels (F = 1.90, p = 0.010) and
> varied along the dorsoventral axis (three-axis F = 2.87, p = 0.03; z slope
> −0.0077 SD/mm, p = 0.007, Bonferroni-corrected p = 0.022). We had not
> predicted this axis, and the predicted anterior–posterior axis showed no effect
> (p = 0.58). Relative to LWPS, LWPC adaptation was weaker in dorsal lPFC. Among
> task-responsive electrodes, both effects were positive at every height, but
> LWPC fell from d = 0.15 ventrally to 0.08 dorsally while LWPS stayed at
> 0.17–0.20. This tilt replicated across independent halves of the trials
> (p = 0.005) and was the same size among task-responsive electrodes
> (−0.0075 SD/mm, p = 0.24; difference from the remaining electrodes, p = 0.67).
> It did not appear in the unsigned magnitude of either effect (p ≥ 0.32).
> Together, these results suggest that LWPC and LWPS adaptation are carried by an
> overlapping, intermixed lPFC population whose balance shifts modestly along the
> dorsoventral axis.

### 15.13 Open items

- [ ] Choose the primary electrode set (§15.11) and state the reason in Methods.
- [ ] Rerun both segregation jobs with `N_PERM_CORR=10000` for exact
      co-localization p-values.
- [ ] Rerun both anatomy jobs so `summary.txt` carries the Pearson-based
      noise-corrected value; the archived all-lPFC summary still shows +1.374 /
      +1.369.
- [ ] Make the LWPC-and-LWPS-by-height figure.
- [ ] Re-annotate `joint_scatter.png` without the noise-corrected value.
- [ ] Change `map_reliability`'s `between_noise_corrected_ci` from a split
      bootstrap to a participant bootstrap (code not yet changed).
- [ ] Optional: one trial split per participant in
      `compute_sensitivities_per_split`, only to report within-participant
      reliabilities or confirm §15.4's explanation.
- [ ] Deferred: confirmation across held-out participants. The trial-half
      replication controls trial noise only; leave-one-participant-out is
      reassuring but is not a held-out test.

---

## 16. Main effects as the reference for the tilt

The question from [`closing_figure_plan.md`](closing_figure_plan.md): is the
dorsoventral tilt in delta (LWPC − LWPS) inherited from how the base effects,
congruency and switch type, are organized?

### 16.1 How the scores are made

`compute_sensitivities_per_split(..., contrast_mode='proportion',
main_effects=True)` also scores each process's main effect, from the same four
cells as its adaptation effect with equal weight over the proportion levels
(`W_MAIN`), on the same trial halves:

```text
congruency = ½[(i − c | 25 %) + (i − c | 75 %)]      per_split: mxA, mxB   electrodes: mx
switch     = ½[(s − r | 25 %) + (s − r | 75 %)]      per_split: myA, myB   electrodes: my
```

LWPC and LWPS come out identical with or without it; the split draws do not
change. Do not use a separate `CONTRAST_MODE=condition` run instead. Its halves
do not line up with these. It also weights congruency by trial count over the
proportion levels, so block effects leak into it (see "Two traps in a separate
condition-mode run" in the plan).

`attach_scores` turns `mx`/`my` into `cong_s`, `switch_s` and
`dm = cong_s − switch_s` (positive = relatively congruency-dominant), with the
same pooled scaling as `lwpc_s`/`lwps_s`. The swap null is valid for dm because
it is a paired difference, so every test in this document takes
`value_col='dm'`.

### 16.2 How to run it

1. Segregation, once per electrode set. `MAIN_EFFECTS=1` is the default in
   the submitter. The output directory gets a `_main_effects` suffix, so the
   archived LWPC/LWPS-only runs are never overwritten. Scoring takes about
   twice as long.

   ```bash
   cd dcc_scripts/stats
   ROIS=lpfc CONTRAST_MODE=proportion EFFECT_MEASURE=cohens_d N_SPLITS=200 \
   N_PERM_CORR=10000 bash submit_stability_flexibility_segregation_dcc.sh
   ```

   For the task-significant set, edit `ELECTRODES=sig` in the submitter as
   before (§9.5).

2. Anatomy. `SEG_RUN` in `submit_stability_flexibility_anatomy_dcc.sh` now
   points at the all-lPFC `_main_effects` directory; change `all` to `sig` in it
   for the task-significant set. When the scores carry `mx`/`my`, the
   continuous arm runs the main-effect block after the usual N4 outputs. With
   older CSVs it runs exactly as before.

   ```bash
   ARM=continuous ROI_FILTER=lpfc ANAT_LEVEL=destrieux N_PERM=10000 \
     bash submit_stability_flexibility_anatomy_dcc.sh
   ```

### 16.3 What it writes (in `continuous/`)

| Output | Contents |
|---|---|
| `summary.txt`, block `MAIN EFFECTS` | dm parcel test, main-effect reliabilities, Test 1 and Test 2 tables |
| `score_anatomy.json`, key `main_effects` | the same numbers |
| `dm_per_roi.csv`, `dm_by_roi.png` | the parcel test on dm (column names as in `delta_per_roi.csv`) |
| `score_map_cong_s.png`, `score_map_switch_s.png`, `score_map_dm.png` | main-effect maps (with `MAKE_BRAIN=1`) |
| `delta_tracking.csv` | Test 1 |
| `tilt_with_dm.csv` | Test 2 |

The segregation run's own `summary.txt` and `correlation_main_effects.json`
report the congruency–switch co-localization on the same halves.

### 16.4 The two tests

**Test 1, `delta_tracking_test`: does dm track delta?** This is
`split_resolved_corr`, the pre-specified co-localization test, applied to dm
and delta: dm from one half against delta from the other within every split,
residualized on responsiveness, centred within participant, Spearman,
participants with at least three electrodes, within-participant permutation
null. Rows in `delta_tracking.csv`:

- `dm vs delta`: the primary number. A shared smooth gradient counts here,
  because that is the "inherited" hypothesis.
- `dm vs delta, + MNI covariates`: the same with coordinates partialled out
  within participant (`split_resolved_corr(covariates=...)`). Does the tracking
  go beyond shared geography?
- `congruency vs LWPC`, `switch vs LWPS`, and the two crossed pairings as
  controls.

The reliabilities in each row are within participant, so they are biased low
(§15.4). Compare them, and never divide by them.

**Test 2, `tilt_with_main_effect_covariate`: does delta's z tilt survive dm?**
Rows in `tilt_with_dm.csv`, all `relative_score_coordinate_test`'s fit with the
swap null:

| fit | question |
|---|---|
| `dm` | Do the main effects tilt? An inherited tilt needs a negative z slope: congruency weaker than switch dorsally. |
| `delta` | The §15.7 tilt, on the same electrodes. |
| `delta + dm` | The tilt with dm as a covariate, on full-data scores. dm and delta share trials here, with a noise correlation of about +0.05. |
| `delta, split halves` / `delta + dm from the opposite half` | Each half of every split refitted with dm from the other half, then averaged. This is the clean version. No p-value; read the full-data rows for it. |

`shrinkage = 1 − with/without`. Near 1, the tilt is carried by the main effects.
Near 0, it survives them. It is only meaningful when the `delta` row itself is
significant. The swap null flips only the adaptation labels, which also breaks
delta's link with dm, so the null is conservative.

### 16.5 Reading the outcome

| Result | Ending (from the plan) |
|---|---|
| Test 1 positive, dm tilts the same way, shrinkage near 1 | Each adaptation scales with the local strength of the demand it regulates. |
| dm has no matching tilt, or shrinkage near 0 with `delta + dm` still significant | Adaptation has spatial structure of its own. |
| dm organized (parcel or coordinate test) but Test 1 null | The demands are organized; their adaptation is shared. |

The main-effect maps will be far more reliable than the adaptation maps. Put the
two levels' correlations next to their reliabilities, never as "the main effects
are more segregated", and never as a noise-corrected ratio.

The synthetic check is in `tests/analysis/stats/test_main_effect_anatomy.py`.
It uses `_synthetic_scores(main_effects='inherited' | 'independent')`: the
inherited world must shrink the tilt and show tracking, and the independent
world must do neither. `DATA_SOURCE=synthetic` runs the inherited world through
the job; its planted layout is anterior–posterior, so its z rows are nulls.
