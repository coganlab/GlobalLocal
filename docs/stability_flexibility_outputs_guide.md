# Stability/Flexibility analysis outputs guide

This guide explains the files written by the stability/flexibility stats scripts, how to read the figures, and what to check when counts look unexpectedly low.

## Why your example output looked sparse

The low counts in the examples are not a plotting problem by themselves: they reflect the electrode definition being used.

- `electrodes=sig` first restricts the analysis to electrodes that passed the upstream significant-channel filter. In your segregation example this reduced the table to 231 electrodes before any stability/flexibility definition was applied.
- The A1 anatomy/conjunction route then applies a per-electrode Type III two-way interaction ANOVA and Benjamini-Hochberg FDR across electrodes separately for the LWPC/stability and LWPS/flexibility interactions. In the shown run, only 4 electrodes passed the stability definition and 8 passed the flexibility definition out of 4412 electrodes.
- `both=0` means no electrode survived both definitions at the selected alpha. CMH odds ratios become `nan` when no subject has an informative 2 x 2 stratum containing both S and F variation; that is an underpowered/undefined categorical test, not evidence for a strong anatomical segregation.
- `thresholds: [0.01]` is more stringent than `alpha=0.05`. If the threshold sweep is only at `q<0.01`, it can show even fewer electrodes than the main A1 counts.
- The `power_traces_conjunction` route with `require_all=True` keeps only electrodes present in every requested run. Different runs/ROIs/corrections can therefore drop many electrodes before counts are made.

Recommended sensitivity checks before interpreting a null: rerun with `ELECTRODES=all`, include threshold sweeps such as `0.01,0.05,0.10`, compare A1 vs `LABEL_SOURCE=power_traces`, and inspect the p/q-value scatter/histograms to see whether there is a broad near-threshold signal or genuinely no interaction evidence.

## A1/A2 ANOVA conjunction output directory

Typical directory name: `anova_conjunction_window_<tmin>to<tmax>s_<electrodes>`.

### `anova_conjunction_summary.png`

Purpose: a compact visual summary of the A1 electrode definition and A2 overlap/conjunction test.

How to read common panels:

- Electrode group/count panels report how many electrodes are `S_only` (LWPC/stability), `F_only` (LWPS/flexibility), `both`, or `neither`.
- P/q-value panels show evidence for the stability interaction on one axis and the flexibility interaction on the other. Points near the lower-left are electrodes with evidence for both. Dashed threshold lines mark the selected alpha/q cutoff.
- The CMH/threshold panel shows whether S and F labels co-occur within subjects more than expected. Odds ratio > 1 suggests shared-core overlap; odds ratio < 1 suggests segregation; `nan` means there were too few informative subject strata.
- The permutation/null panel compares the observed overlap count with a within-subject null. Observed overlap above the null supports shared overlap; below the null supports segregation; no informative overlap gives `p=1`/`z=nan`.

### `electrode_labels.csv`

One row per electrode. Key columns:

- `subject`, `electrode`: electrode identity.
- `p_cpc`, `q_cpc`, `F_cpc`, `CPC`: stability/LWPC ANOVA interaction statistics and binary flag.
- `p_sps`, `q_sps`, `F_sps`, `SPS`: flexibility/LWPS interaction statistics and binary flag.
- `S`, `F`: backward-compatible aliases for `CPC` and `SPS`.
- `p_cps`/`q_cps` and `p_spc`/`q_spc`: cross-control interactions.

### `summary.txt`

Text version of the main results. Treat `MH odds ratio = nan`, `CMH p = nan`, or “fewer than 3 informative subjects” as “the categorical overlap test is undefined/thin,” not as a positive result.

## A3 anatomy output directory

Typical directory name: `anatomy_<label_source>_<scope>_window_<tmin>to<tmax>s_<electrodes>`.

### Brain images

- `selectivity_groups_on_brain.png`: combined surface rendering of selective electrodes on the average brain.
- `selectivity_groups_on_brain_S_only.png`: stability-only electrodes.
- `selectivity_groups_on_brain_F_only.png`: flexibility-only electrodes.
- `selectivity_groups_on_brain_both.png`: electrodes significant for both definitions, when any exist.
- `selectivity_groups_on_brain_roi_hist.png`: fallback histogram written only if surface rendering cannot proceed.

Colors are consistent across A3 plots: green = `both`, blue = `S_only`, orange = `F_only`, gray = `neither` when shown.

Surface brain plots need the same recon/electrode-location files used by the older `dcc_scripts/vis/plot_sig_electrodes_dcc.py` path. In practice that means `subject_to_info()` must be able to find each subject's `elec_recon/*_elec_locations_RAS_brainshifted.txt` (or equivalent configured recon products). The summary/anatomy statistics do **not** need `elec_recon`, but actual 3-D electrode placement does; without those coordinates, the script cannot know where to draw the contacts on the fsaverage brain. If only some subjects are missing recon files, the renderer skips those subjects and still renders electrodes from subjects with usable recon data. It falls back to a histogram only if no usable plotting subjects/electrodes remain or the whole surface stack is unavailable.

### `roi_group_histogram.png`

Grouped bar chart of coarse ROI group membership.

- x-axis: ROI group, optionally annotated as `(n=<subjects> subj)` when coverage is supplied.
- y-axis: number of selective electrodes.
- legend: selectivity group (`both`, `S_only`, `F_only`).

### `destrieux_group_histogram.png`

Grouped bar chart at the raw Destrieux-label level.

- x-axis: Destrieux anatomical label, optionally annotated with subject coverage.
- y-axis: number of selective electrodes.
- Use this plot when the analysis is restricted to one coarse ROI, because the coarse ROI column is then constant.

### `anatomy_coverage_enrichment.png`

Two-panel diagnostic.

- Left panel: subject x anatomical-label coverage matrix. x-axis is ROI group or Destrieux label; y-axis is subject; green means the subject had at least one electrode in that anatomical bin.
- Right panel: within-subject permutation null for the group x anatomy chi-square statistic. x-axis is permuted chi-square; y-axis is number of permutations. The vertical red line is the observed statistic.

### CSV/JSON files

- `electrode_labels.csv`: upstream S/F labels before anatomy attachment.
- `label_funnel.csv`: written for `LABEL_SOURCE=power_traces`; counts how many electrodes were tested in the power-traces run, how many had raw cluster `p<alpha`, how many remained after aligning runs, and how many were finally flagged after the requested correction.
- `anatomy_labels_roi.csv`: one row per electrode after anatomy attachment. Key columns include `roi`, `anat`, and `group`.
- `coverage_matrix.csv`: subject x ROI/Destrieux boolean coverage used by the enrichment test.
- `group_roi_contingency.csv`: group x anatomical-bin counts restricted to bins passing `min_subjects`.
- `roi_group_histogram.csv`: uncorrected group x coarse-ROI counts.
- `destrieux_group_histogram.csv`: uncorrected group x raw-Destrieux counts.
- `roi_enrichment.json`: machine-readable test summary (`rois_tested`, `observed_stat`, permutation `p`, `n_electrodes`, coverage per bin).
- `roi_enrichment_null.npy`: saved permutation null distribution.
- `summary.txt`: human-readable anatomy summary.

Interpretation: A significant enrichment result means selectivity-group membership is associated with anatomy beyond what electrode coverage alone forces. Non-significant enrichment with only a handful of selective electrodes should be reported as underpowered/descriptive.

### The `continuous/` subdirectory (`ARM=continuous` or `ARM=both`)

The continuous arm of [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md) §5–§7. Same anatomy join and same coverage bookkeeping as above, but the response is the per-electrode **score** rather than a binary flag, so no electrode has to survive a threshold and the effect sizes are not discarded. Everything turns on `delta = lwpc_s - lwps_s`, a within-electrode contrast between the two effects — which is why the test is an effect-type x anatomy interaction rather than the difference-of-significance fallacy.

- `summary.txt`: read this first. Primary ROI test, leave-one-subject-out leverage, coordinate regression per hemisphere, medoid displacement, noise ceiling, `min_elec` sweep.
- `scores.csv` / `per_split.csv`: the disjoint-half LWPC/LWPS scores and the per-split table they were averaged from. Point a later run at these (`SCORES_CSV`, `PER_SPLIT_CSV`) to skip re-scoring.
- `scores_with_anatomy.csv`: one row per electrode with `lwpc_s`, `lwps_s`, `delta`, `roi`, `anat`, `mni_x/y/z`, `hemi`, `resp`.
- `delta_per_roi.csv`: per-ROI mean `delta` (raw and nuisance-adjusted), permutation p and BH q. `delta > 0` = LWPC-dominant.
- `delta_roi_loso.csv`: the same statistic with each subject dropped. Read leverage off `observed_stat`, not `p` — p saturates at the permutation floor.
- `min_elec_sweep.csv`: the LWPC-LWPS correlation at `min_elec` 1/2/3. That filter drops **whole subjects**, so it moves the effective N more than any scaling choice.
- `delta_by_roi.png`: mean +/- SEM of `delta` per ROI with every electrode drawn on. Also the fallback figure when the surface stack is missing.
- `joint_scatter.png`: the per-electrode LWPC-vs-LWPS scatter on the pooled-scaled scores, with the split-half ceiling annotated.
- `score_map_{lwpc_s,lwps_s,abs_lwpc,abs_lwps,delta}.png` (+ `_colorbar.png`): the five §6 surfaces. `delta` carries the argument; the other four are what a reader needs to check it is not driven by one effect's magnitude alone. The renderer takes one colour per call, so the scalar is drawn as nine colour bins — hence the separate colourbar.
- `score_centers.csv`: per subject x hemisphere weighted medoid displacement (LWPC minus LWPS). Descriptive only.
- `score_anatomy.json`: machine-readable version of the summary.

Interpretation: the ROI test asks whether the *relative* score varies with anatomy once subject and responsiveness are conditioned on, against a null that swaps the two effect labels within each electrode (which preserves subject, coverage, location and responsiveness exactly). Report the noise ceiling next to every spatial number — without it, "the two maps do not correlate" cannot be told apart from "neither map is measured well enough to correlate with anything".

## Segregation output directory

Typical directory name: `window_<tmin>to<tmax>s_<electrodes>_<rois>_<contrast_mode>_<effect_measure>_<fdr_correction>`, with `_scatter_only_splits<N>` appended for a `SCATTER_ONLY` run so its scatter does not overwrite a full run's.

### `segregation_joint_scatter.png`

Purpose: the descriptive joint distribution — each electrode's stability sensitivity against its own flexibility sensitivity, coloured by subject, with a marginal histogram on each axis and a per-subject correlation panel. Look at this before the headline number: positive diagonal = shared, spread on both axes with no correspondence = independent, spread on one axis only = one mechanism, negative diagonal = opponent, and all the structure in one colour or a few points = artifact.

It is descriptive only. There is no permutation, no responsiveness residualisation and no within-subject centring, and by default the two sensitivities are scored on the same trials — so its correlation is an upper bound on the pipeline's, which is annotated on the figure for comparison whenever a full run produced one. Written by every run, and produced on its own (minutes, no inference) with `SCATTER_ONLY=1`.

- `segregation_joint_scatter_diagnostics.json`: the leverage numbers behind the "artifact" reading — `corr`, `corr_within_subject`, `loso_min`/`loso_max` with `most_influential_subject`, `corr_drop_top` (correlation without the most influential electrodes) with `most_influential_electrode`, `max_subject_share`, and `flags`. A non-empty `flags` list means the apparent structure is carried by one subject, a handful of electrodes, or a between-subject offset; investigate before reporting the correlation. Flags are only raised once |corr| >= 0.1 — below that there is no apparent structure to attribute.
- `segregation_joint_scatter_per_subject.csv`: per subject, the electrode count, that subject's own correlation, and the pooled correlation with that subject held out.
- `scatter_sensitivities.csv`: written by `SCATTER_ONLY` runs only — the per-electrode `x`/`y` the scatter was drawn from.

### `segregation_summary.png`

Purpose: compares stability and flexibility sensitivity as continuous electrode scores and categorical thresholded labels.

- Continuous panels ask whether stability and flexibility effect sizes are correlated across electrodes after controls. Positive correlation suggests shared/core sensitivity; zero or negative correlation suggests segregation.
- Categorical panels ask whether thresholded S and F labels overlap within subjects. Odds ratio > 1 suggests shared overlap; odds ratio < 1 suggests segregation; `nan` means the strata are too sparse.

### `segregation_diagnostics.png`

Diagnostic panels for trial counts, electrode counts, p-values/q-values, split-half stability, and controls. Use this before interpreting the headline figure; a significant continuous correlation with undefined categorical CMH usually means there is graded shared sensitivity but too few thresholded electrodes to support an overlap-count claim.

### Common files

- `summary.txt`: headline continuous and categorical statistics.
- `labels.csv` or similarly named label table: per-electrode continuous scores, p/q values, and thresholded S/F calls.
- Permutation/null arrays or JSON summaries: support the plotted null distributions.

## Power-traces conjunction output directory

Typical directory name: `power_traces_conjunction_results/<run>/<correction>_alpha<alpha>/<roi>`.

### `power_traces_conjunction_summary.png`

Purpose: conjunction/overlap analysis using electrodes detected by the within-electrode time-resolved ANOVA pipeline rather than the A1 window-mean ANOVA.

- Count panels report `both`, `S-only`, `F-only`, and `neither` after the requested correction.
- Cross-control rows (`CPS`, `SPC`) should be near null. Large cross-control counts mean the selectivity definition is not specific.
- Threshold sweep shows robustness across q cutoffs when q-values exist. Ignore rows flagged as too thin/undefined.

### Key interpretation knobs

- `correction=fdr_bh`: flags q-values after FDR, usually appropriate for electrode-count claims.
- `correction=cluster`: uses the cluster-corrected run decision; q-value sweeps may not be available.
- `require_all=True`: drops electrodes not present in all requested runs; this can drastically reduce counts.
- `roi=<name>`: limits counts to that ANOVA ROI.

### `label_funnel.csv`

This table is the quickest way to answer “where did my power-traces electrodes go?”

- `run_tested_electrodes`: denominator in the source power-traces summary for that interaction and ROI.
- `run_raw_p_lt_alpha`: like-for-like raw cluster count from the power-traces run before across-electrode FDR.
- `present_after_alignment`: electrodes left after the conjunction script aligns CPC/SPS/(optional CPS/SPC) onto one common electrode universe. With `require_all=True`, this is the intersection across requested runs.
- `dropped_by_alignment`: electrodes lost because they were not present in every requested run.
- `aligned_raw_p_lt_alpha`: raw cluster-significant count after alignment but before the final correction.
- `final_flagged`: count actually used in `labels.csv`, summary figures, CMH, and anatomy. Under `correction=fdr_bh`, this can be much lower than `run_raw_p_lt_alpha`; under `correction=cluster`, it should match the aligned raw cluster count.

## Quick cluster commands to count the drop-off

After rerunning `power_traces_conjunction` or A3 with `LABEL_SOURCE=power_traces`, inspect the newly written `label_funnel.csv`:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
run = Path('/path/to/power_traces_conjunction_or_anatomy_output')
funnel = pd.read_csv(run / 'label_funnel.csv')
print(funnel.to_string(index=False))
labels = pd.read_csv(run / ('labels.csv' if (run / 'labels.csv').exists() else 'electrode_labels.csv'))
print('\nfinal labels:')
print('electrodes =', labels[['subject', 'electrode']].drop_duplicates().shape[0])
for col in ['CPC', 'SPS', 'CPS', 'SPC', 'S', 'F']:
    if col in labels:
        print(f'{col}:', int(labels[col].sum()))
PY
```

If you have not rerun yet, you can still count directly from a source `power_traces` `summary.csv` for one effect/ROI:

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv('/path/to/power_traces_run/summary.csv')
roi = 'lpfc'
alpha = 0.05
effect = 'C(congruency):C(incongruentProportion)'  # CPC/LWPC; change for SPS
s = summary[(summary['roi'] == roi) & (summary['effect'] == effect)]
pcol = 'best_cluster_p' if 'best_cluster_p' in s else 'cluster_p_value'
best = (s.sort_values([pcol, 'extent_windows'], ascending=[True, False])
          .groupby(['subject', 'electrode', 'roi'], as_index=False).first())
print('tested:', len(best))
print(f'raw {pcol}<alpha:', int((best[pcol] < alpha).sum()))
print(best.groupby('subject')['electrode'].nunique().describe())
PY
```

## How to report sparse results

A safe wording for the example outputs would be: “The continuous segregation analysis showed a small positive stability-flexibility association, but thresholded A1 and power-traces conjunction analyses yielded very few stability/flexibility selective electrodes and no informative overlap strata. Therefore the categorical shared-core/segregation and anatomy-enrichment analyses are underpowered/undefined for this window and electrode definition.”
