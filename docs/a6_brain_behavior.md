# A6 — brain–behavior: one score per participant, the RT confound, and how to read it

**What this document is.** The runbook and reading guide for A6 as rebuilt on
2026-09-27. It covers how each participant gets one neural and one behavioral
LWPC / LWPS score, what was wrong before, how to run the job (synthetic and
real), what every output file holds, and how to interpret the numbers. The
brain–behavior row of [`analysis_plans.md` › Closing figure plan](analysis_plans.md#closing-figure-plan) and §19
of [`analysis_guide.md`](analysis_guide.md) point here.

| Role | File |
|---|---|
| Scores, RT adjustment, correlations | `src/analysis/stats/stability_flexibility_brain_behavior.py` |
| Cluster job (all three levels) | `dcc_scripts/stats/stability_flexibility_brain_behavior_dcc.py` |
| Knobs | `dcc_scripts/stats/run_stability_flexibility_brain_behavior_dcc.py` |
| Submitter | `dcc_scripts/stats/submit_stability_flexibility_brain_behavior_dcc.sh` |
| Long table (now with `trial`, `rt`, `acc`) | `assemble_long_df` in `dcc_scripts/stats/stability_flexibility_segregation_dcc.py` |
| Behavioral LWPC, LWPS and the other per-subject effects | `src/config/ieeg_behavioral_subject_level_effects.csv` |
| Where that CSV comes from | [`analysis/iEEGBehavioralAnalysis.ipynb`](https://github.com/jimzhang629/TSF-fMRI-python_Jim/blob/HEAD/analysis/iEEGBehavioralAnalysis.ipynb) in `jimzhang629/TSF-fMRI-python_Jim` |

---

## 0. The short version

- **Report level (1)**, the per-participant continuous scores: each
  participant's mean signed per-electrode d against its behavioral RT
  difference-of-differences, **RT-adjusted**, with its reliability ceiling and the
  |r| needed at this n. The other two levels are context.
- **Rerun anything A6 produced before 2026-09-27.** The behavioral "LWPC" was a
  congruency × *switch*-proportion contrast (§1.1).
- **Expect a null you cannot interpret.** At n ≈ 18–21, significance needs
  |r| ≥ 0.43–0.47, and the reliabilities cap what any correlation can reach
  (§6). Write the supplement text around the ceiling, not the p-value.
- **First real run: null and uninformative** (§13). Neural LWPC has no
  measurable reliability, and the LWPS ceiling (0.46) is below the 0.48 needed
  at n = 17.
- **Paper: supplement S-BB only**, framed as "could not be tested at this n".
  The Results paragraph, Discussion sentence, figure caption and what not to
  report are in §14; the Methods are in
  [`methods.md` › A6](methods.md#a6-brainbehavior-supplement-s-bb).

```bash
cd dcc_scripts/stats
# 1. the RT confound on synthetic data: raw r positive, RT-adjusted r ~0
DATA_SOURCE=synthetic SYNTHETIC_LINK=0 SYNTHETIC_RT_COUPLING=0.4 SYNTHETIC_N_SUBJ=24 \
    bash submit_stability_flexibility_brain_behavior_dcc.sh
# 2. the real run: task-significant lPFC, 0-1.5 s
bash submit_stability_flexibility_brain_behavior_dcc.sh
```

---

## 1. What changed and why

### 1.1 The behavioral LWPC was the wrong contrast

`_BLOCK_PROPORTION_MAP` swapped the incongruent proportion of blocks A and D. The
task builds the blocks like this (`src/task/mainTask.m`: `createCongruencyArr`
makes A and B 75 % incongruent, `createTaskArr` makes A and C 25 % switch), and
`combinedData.csv` agrees:

| Block | Incongruent | Switch | Old map (incongruent) |
|---|---|---|---|
| A | 75 % | 25 % | 25 % ✗ |
| B | 75 % | 75 % | 75 % |
| C | 25 % | 25 % | 25 % |
| D | 25 % | 75 % | 75 % ✗ |

The two proportions are **fully crossed**. The old map made them look collinear
(the module's docstring said so), and its "LWPC" compared A+C with B+D, which
differ in switch proportion, not incongruent proportion. The result was a
congruency × switch-proportion contrast mixed with block-level RT differences. On
`combinedData.csv`, correct trials, 23 participants:

| | Old map (archived A6) | Correct map |
|---|---|---|
| Behavioral LWPC, mean / SD across participants | 46 / 203 ms | 123 / 147 ms |
| LWPC split-half reliability (full length) | 0.88 (spurious) | 0.69 |
| LWPS split-half reliability | 0.37 | 0.37 |
| r(LWPC, LWPS) across participants | 0.02 | 0.44 |
| LWPC > 0 | — | 20 / 23, t(22) = 4.0, p = 0.0006 |
| LWPS > 0 | — | 20 / 23, t(22) = 4.5, p = 0.0002 |

Where the bad map lived, and what was done:

| Place | Fix |
|---|---|
| `_BLOCK_PROPORTION_MAP` (A6 module) | corrected; pinned by tests against `mainTask.m` and the CSV |
| `stats/erin_linear_mixed_effects_model.py` / `.ipynb` | corrected; the swap made `CongruentProp` collinear with `SwitchProp`, and its saved fit (SE ≈ 3 × 10⁹) was cleared from the notebook |
| `stability_flexibility_timing_brain_behavior_correlation_tutorial.ipynb` | the "collinear" caveat and the map text corrected; the stale output of the `combinedData.csv` cell cleared (rerun it) |

**Affected:** every behavioral LWPC computed from `blockType` with the old map,
including the archived A6 across-subject numbers
(`results/<epochs>/brain_behavior_window_0.0to1.5s_sig_count/summary.txt`).
**Not affected:** every neural analysis. The epochs metadata take the
proportions from the event names (`Stimulus/i25.0/r75.0/...`), and
`general_utils.map_block_type` was right. Two notebooks (`erin_stats.ipynb`,
`power/power_traces.ipynb`) assign `map_block_type`'s output to swapped column
names; neither uses the columns afterwards.

> **N1 check.** `analysis_plans.md` › Concurrent-regulation plan lists the behavioral
> LWPC/LWPS as coming from `erin_linear_mixed_effects_model.py`. That script is a
> post-error model with no congruency × proportion term, and it carried the swapped
> map. Confirm which script produced the reported N1 numbers. The corrected values
> are in the table above.

### 1.2 Behavior comes from the subject-level effects table

**Where the table comes from.** `src/config/ieeg_behavioral_subject_level_effects.csv`
was generated with
[`analysis/iEEGBehavioralAnalysis.ipynb`](https://github.com/jimzhang629/TSF-fMRI-python_Jim/blob/HEAD/analysis/iEEGBehavioralAnalysis.ipynb)
in the `jimzhang629/TSF-fMRI-python_Jim` repository. It is not built by
anything in this repository: to change the behavioral scores, rerun that
notebook and replace the CSV. It holds one row per subject and measure
(`key_RT_mean`, `acc_mean`, `error_mean`) with the overall mean, the congruency
effect, the switch cost, each split by block proportion, `LWPC_effect`,
`LWPS_effect`, and the other interaction terms.

**The behavioral LWPC, LWPS and related scores all come from this CSV.** A6 does
not compute them itself. The behavioral LWPC / LWPS that levels (1) and (2)
correlate against are read with `load_subject_level_behavior`: the
`LWPC_effect` / `LWPS_effect` of its `key_RT_mean` rows, one per subject.
`behavioral_magnitudes.csv` carries the table's other columns alongside them. Both are LOW minus HIGH proportion
(`congruency_effect_25_inc − congruency_effect_75_inc`,
`switch_cost_25_switch − switch_cost_75_switch`), the orientation of the neural
scores; a test pins this against the file. The table's IDs are stems (`D0107`),
so they are matched to the epochs' IDs (`D0107A`) with `subject_stem`; a
participant missing from the table (e.g. `D0144`) drops out and `summary.txt`
says so. Point `BEHAVIOR_CSV` elsewhere to use another table with the same
columns.

The same contrast is still scored on the long table's own trials, but only as a
cross-check (§9, the `behavior cross-check` line) and as the estimate of the
behavioral reliability behind the level-(1) ceiling: the table has no trials, so
it has no split-half reliability of its own, and the same- / disjoint-half
correlations (§4.3) are n/a. Level (3) is unchanged: its adjustments are built
from each trial's RT.

### 1.3 The electrode set now filters

The runner had `ROIS_DICT = None`, so `resolve_electrodes_to_keep` kept every
channel and `ELECTRODES=sig` did nothing (the archived run used all 4412
channels). `ROIS` now defaults to `lpfc`, as in the segregation runner, and the
output folder name carries the ROI.

### 1.4 New: per-participant continuous scores

`participant_scores`, `participant_brain_behavior` and `rt_adjust_hg` in the
module, run as level (1) of the job. The long table gained `trial` (the epoch
index, shared by all of a participant's electrodes), `rt` and `acc`, which they
need. The submitter's default epochs file is now the one the N4 scores use, so a
participant's neural score is the mean of the same electrode scores N4 maps.

---

## 2. The three levels and which to report

| Level | Question | n | Report? |
|---|---|---|---|
| **(1) Across participants, continuous** | Does a participant whose lPFC adapts more also adapt more in behavior? | participants (≈ 18–21) | **yes** — supplement S-BB, RT-adjusted, with ceiling |
| (2) Across subjects, label-based | Same question with counts / fractions of "selective" electrodes, or their mean F | participants | no: counts need thresholded labels (no lPFC electrode passes FDR for LWPC) and 'effect' averages an unsigned F |
| (3) Within subject, single trial | Does trial HG in a selected group predict the trial's signed contribution to the RT d-o-d? | trials | not as built: see §4.1 and §11 |

---

## 3. How a participant's score is built

**Neural.** For each electrode, the equal-cell-weight difference-of-differences
of window-mean HG over the four cells, divided by the pooled within-cell SD: the
same d as N4's `lwpc_score` / `lwps_score` (`sfs._interaction_cohens_d`). The
participant's score is the **plain mean over its electrodes**. Equal weights are
right because a participant's electrodes share its trials, so their sampling
errors are nearly equal. Participants with fewer than `MIN_ELEC` usable
electrodes (default 3) get no neural score.

**Behavioral.** The `LWPC_effect` / `LWPS_effect` (RT, ms) of the participant's
`key_RT_mean` row in `ieeg_behavioral_subject_level_effects.csv` (§1.2). They
are the same four-cell difference-of-differences. The job does not score them
from the HG trials; it scores those only to cross-check the table and to
estimate its reliability.

**Sign.** Both are LOW minus HIGH proportion. Positive means the condition effect
shrinks in the high-proportion block, the direction behavior shows.

**Which electrodes.** Never electrodes selected on LWPC or LWPS. Use the
task-significant set (`ELECTRODES=sig`, the default) as primary, and all lPFC
(`ELECTRODES=all`) as the check. By the closing plan's own logic, "how strong is
the effect in this participant" is a question for responsive electrodes. All-lPFC
LWPC is near zero on average (d = 0.01 / 0.06 / −0.04 by height band), so each
participant's mean gets diluted by however many unresponsive contacts it has.

**Signed averaging.** A shrinking effect reads as negative LWPC on an electrode
whose congruency effect is itself negative (i < c). Such electrodes pull the mean
toward zero. In task-significant lPFC both simple effects are positive (N2), so
the signed mean is appropriate there. To check, compare the main-effect signs
(`MAIN_EFFECTS=1` segregation run).

**Exploratory alternatives to the signed mean.** Level (1) also scores each
participant two other ways, each raw and RT-adjusted, with its own reliability,
correlations and scatter figure:

| Variant | Participant's neural score | Columns |
|---|---|---|
| `abs`, `abs_rtadj` | mean \|d\| over its usable electrodes, so electrodes adapting in opposite directions no longer cancel | `*_neural_abs`, `*_neural_rtadj_abs` |
| `pos`, `pos_rtadj` | mean d over only the electrodes whose own score is positive, chosen per effect (LWPC and LWPS separately); needs `MIN_ELEC` such electrodes, else NaN | `*_neural_pos`, `*_neural_rtadj_pos`; `*_pos_n` counts them |

Treat both as exploratory. Each folds or selects on the same noisy d it
averages, so noise alone makes them positive even with no true effect, and more
so for a participant with fewer trials: a participant's score partly measures
its noise level. `summary.txt` prints `r(neural score, trials per participant)`
for each of these variants; a clearly negative value means that noise, not
adaptation, is driving the score. `pos` also loses participants: a participant
counts only if both its LWPC and its LWPS have `MIN_ELEC` positive electrodes.

---

## 4. The three traps, and what the code does about each

### 4.1 RT coupling fakes a brain–behavior correlation

If single-trial HG tracks RT inside each cell, `hg = a_cell + b · rt + noise`,
then every cell mean carries `b ×` that cell's mean RT, and so:

```text
neural d-o-d  =  true neural d-o-d  +  b × behavioral d-o-d
```

In d units the RT part is ρ × (behavioral d-o-d) / σ_RT, where ρ is the
within-cell HG–RT correlation. With this dataset's numbers (LWPC 123 ms, LWPS
97 ms, within-participant RT SD ≈ 380 ms) and ρ = 0.2, that is about 0.065 d for
LWPC and 0.05 d for LWPS. Task-significant electrodes average 0.14 and 0.18. A
0–1.5 s stimulus-locked window with a median RT of 1.19 s makes such coupling
likely. What it does:

- **Across participants it builds a positive matched correlation.** The term
  scales with each participant's own behavioral effect.
- **It passes the specificity checks.** It builds a cross correlation of about
  0.44 × the matched one, the behavioral LWPC–LWPS correlation, so "matched beats
  cross" is exactly what it predicts. The joint regression calls it specific too:
  matched β large, cross β ≈ 0.
- **Disjoint trial halves only remove part of it.** The term carries the *true*
  behavioral effect, which is present in both halves.

Synthetic check, with no brain–behavior link at all and RT coupling 0.4
(n = 24):

| | raw | RT-adjusted |
|---|---|---|
| matched r, LWPC | **+0.56, p = 0.004** | −0.07 |
| joint β: matched / cross | +0.56 / +0.02 | −0.07 / +0.03 |
| half-length r, same half / disjoint halves | +0.55 / +0.25 | −0.03 / −0.05 |

**The fix: `rt_adjust_hg`.** For each electrode it fits the *pooled within-cell*
slope of HG on RT, over all 16 congruency × switch type × incongruent proportion
× switch proportion cells (the ANCOVA slope), and subtracts `b · (rt − mean rt)`.
It must be the within-cell slope: a plain `hg ~ rt` regression also soaks up the
condition effects, which move both HG and RT (a test fails if you swap it in). The
adjusted d-o-d is exactly the raw one minus `b ×` the behavioral one, so an
RT-linked component is removed exactly, not just on average.

The adjustment is **conservative**. If adaptation in HG reaches behavior through
the same trial-by-trial coupling, the adjustment removes that part too. So the
adjusted score is the neural adaptation not carried by RT, and the raw score is
an upper bound. Report both.

The same term reaches the other analyses. N2's sign match with behavior is
partly what RT coupling alone predicts. At level (3), a plain HG–RT correlation
leaks into both slopes (§11).

### 4.2 Reliability needs one trial split per participant

`compute_sensitivities_per_split` draws a new split for every electrode. For a
*participant mean*, that lets electrode i's half A share trials with electrode
j's half B. Noise common to a participant's electrodes then makes the two
half-means agree. With a common-mode correlation of 0.2–0.4 and 10 electrodes, a
participant mean with no real between-participant differences shows a split-half
reliability of 0.47–0.64 (the tests plant this and see it). This is the flip side
of the within-participant bias in `n4_continuous_anatomy.md` §15.4.

`participant_scores` draws **one split per participant**, stratified on the 16
design cells, and applies it to all of its electrodes and to its behavior. The
reliability of each participant-level score is the across-participant
correlation of half-A and half-B values, averaged over `PARTICIPANT_N_SPLITS`
splits (`r_half`). It is then taken to full length with Spearman–Brown
(`reliability`, NaN when `r_half` ≤ 0). The **ceiling** on the observed
brain–behavior r is √(reliability_neural × reliability_behavior).

### 4.3 Brain and behavior share trials

The same split gives a **disjoint-half** brain–behavior r: neural from half A
against behavior from half B, and the reverse, which shared trial noise cannot
inflate. Compare it with the same-half r. Both are half-length, so they sit below
the full-data r.

---

## 5. Data flow

```text
HG epochs (.fif) + metadata (event names -> congruency, proportions, reaction_time, accuracy)
        |
        |  assemble_long_df(window, electrodes = ROIS x ELECTRODES, effect_measure='cohens_d')
        v
long table: subject, electrode, trial, hg, congruency, switchType,
            incongruent_proportion, switch_proportion, rt, acc      -> long_df.csv
        |
        +--> participant_scores(df)                                   LEVEL (1)
        |       correct trials with an RT only (brain and behavior alike)
        |       rt_adjust_hg: per-electrode within-cell HG~RT slope -> hg_adj
        |       per electrode: d-o-d / pooled SD on hg and hg_adj    -> participant_electrode_scores.csv
        |       per participant: mean over usable electrodes; RT d-o-d on the same
        |         trials (cross-check and reliability estimate only)
        |       PARTICIPANT_N_SPLITS shared splits -> half values -> reliabilities
        |           -> participant_scores.csv, participant_reliability.csv
        |     participant_brain_behavior(ps, 'rtadj' | 'raw', behavior=<the CSV>)
        |       matched / cross r, Spearman, CI, joint regression, ceiling,
        |       |r| needed, same- vs disjoint-half r
        |           -> participant_brain_behavior.json / .png, summary.txt (1)
        |
        +--> per_electrode_anova_labels -> S/F flags -> level (2) (label-based)
        +--> load_subject_level_behavior -> behavioral_magnitudes.csv
        |       (levels 1 and 2; behavior_from_long_df only cross-checks it)
        +--> assemble_trial_table + group HG -> level (3) mixed models
```

---

## 6. What to expect at this n

Significance needs |r| ≥ 0.47 at n = 18 (task-significant lPFC with ≥ 3
electrodes) and ≥ 0.43 at n = 21. The observed r is the true r times the ceiling.
Behavioral reliability is 0.69 (LWPC) and 0.37 (LWPS). Neural reliability is
unknown until the run, so two values are shown. Power at α = .05 is simulated:

| True r | Neural reliability | LWPC: expected r, power n = 18 / 21 | LWPS: expected r, power n = 18 / 21 |
|---|---|---|---|
| 0.3 | 0.3 | 0.14, 0.08 / 0.09 | 0.10, 0.07 / 0.07 |
| 0.3 | 0.6 | 0.19, 0.12 / 0.13 | 0.14, 0.09 / 0.09 |
| 0.5 | 0.3 | 0.23, 0.15 / 0.17 | 0.17, 0.10 / 0.11 |
| 0.5 | 0.6 | 0.32, 0.26 / 0.30 | 0.24, 0.16 / 0.18 |
| 0.7 | 0.3 | 0.32, 0.26 / 0.30 | 0.23, 0.15 / 0.18 |
| 0.7 | 0.6 | 0.45, 0.49 / 0.56 | 0.33, 0.27 / 0.32 |

Even a strong true link is more likely missed than found, and LWPS is worse than
LWPC because its behavioral score is noisier. **A null is uninformative; a
positive result is fragile.** That is why this belongs in the supplement unless
it is striking.

---

## 7. Parameters

All are environment variables read by the runner; the submitter sets the
defaults shown.

| Variable | Submitter default | What it does |
|---|---|---|
| `EPOCHS_ROOT_FILE` | the N4 epochs file (`…drop_and_nan…ttest_zmax_20`) | which HG epochs to load |
| `WINDOW_TMIN` / `WINDOW_TMAX` | `0.0` / `1.5` | window over which HG is averaged per trial |
| `ELECTRODES` | `sig` | `sig` = task-significant, `all` = every electrode in the ROI |
| `ROIS` | `lpfc` | comma-separated names from `src/analysis/config/rois.py`, or `all` (then `ELECTRODES` has no effect) |
| `MIN_ELEC` | `3` | fewer usable electrodes → no neural score for that participant |
| `PARTICIPANT_N_SPLITS` | `200` | shared trial splits behind the reliabilities |
| `SEED` | `0` | split seed |
| `BEHAVIOR_CSV` | `src/config/ieeg_behavioral_subject_level_effects.csv` | the subject-level table the behavioral LWPC / LWPS come from (`LWPC_effect` / `LWPS_effect`, `key_RT_mean` rows) |
| `RUN_TRIALWISE` | `1` | `0` skips level (3) |
| `CONTRAST_MODE` / `FDR_CORRECTION` / `ALPHA` / `NEURAL_SUMMARY` | `proportion` / `none` / `0.05` / `count` | level (2)'s electrode labels and which summary it stars; they do not touch level (1) |
| `DATA_SOURCE` | `real` | `synthetic` runs every level on planted data |
| `SYNTHETIC_N_SUBJ` | `16` | synthetic participants |
| `SYNTHETIC_LINK` | `0.6` | planted across-participant brain–behavior r (level 1) |
| `SYNTHETIC_RT_COUPLING` | `0.3` | HG noise SDs added per RT SD (level 1) |
| `SYNTHETIC_CROSS_FRAC` / `_ACROSS_BETA` / `_WITHIN_BETA` | `0.25` / `1.2` / `0.6` | levels (2) and (3) planted couplings |

---

## 8. How to run

### 8.1 Synthetic checks (seconds, no data)

```bash
cd dcc_scripts/stats
# a planted link (0.6) with RT coupling on top: every level has signal
DATA_SOURCE=synthetic bash submit_stability_flexibility_brain_behavior_dcc.sh
# the RT confound alone: level (1) raw r positive, RT-adjusted r ~0
DATA_SOURCE=synthetic SYNTHETIC_LINK=0 SYNTHETIC_RT_COUPLING=0.4 SYNTHETIC_N_SUBJ=24 \
    bash submit_stability_flexibility_brain_behavior_dcc.sh
# levels (2)/(3) falsification: `specificity_ok` must stop holding
DATA_SOURCE=synthetic SYNTHETIC_CROSS_FRAC=1.0 \
    bash submit_stability_flexibility_brain_behavior_dcc.sh
```

To run without the cluster, call the runner directly with the same variables:
`DATA_SOURCE=synthetic python run_stability_flexibility_brain_behavior_dcc.py`.
Synthetic runs write to `results/synthetic_…/`.

### 8.2 The real runs

```bash
cd dcc_scripts/stats
# primary: task-significant lPFC, 0-1.5 s
bash submit_stability_flexibility_brain_behavior_dcc.sh
# check: every lPFC electrode
ELECTRODES=all bash submit_stability_flexibility_brain_behavior_dcc.sh
# robustness: an early window that ends before almost every response (median RT 1.19 s)
WINDOW_TMIN=0.0 WINDOW_TMAX=0.5 bash submit_stability_flexibility_brain_behavior_dcc.sh
```

Output lands in
`dcc_scripts/stats/results/<EPOCHS_ROOT_FILE>/brain_behavior_window_<tmin>to<tmax>s_<electrodes>_<rois>_<neural_summary>/`.
Level (1) takes about 5 s for lPFC at 200 splits (about 3 minutes if `ROIS=all`
keeps every channel); the mixed models and the A1 labels dominate the runtime.

### 8.3 Offline, from a saved long table

`long_df.csv` now carries `trial`, `rt` and `acc`, so level (1) can be rerun
without the epochs: for example on a different electrode subset, or joined to
coordinates.

```python
import pandas as pd
from src.analysis.stats import stability_flexibility_brain_behavior as sbb

df = pd.read_csv('<save_dir>/long_df.csv')
# optional: restrict to a subset, e.g. the N4 electrodes
# keep = pd.read_csv('<n4 run>/scores_with_anatomy.csv')['electrode']
# df = df[df['electrode'].isin(keep)]
ps = sbb.participant_scores(df, n_splits=200, min_elec=3)
res = sbb.participant_brain_behavior(ps, variant='rtadj')   # or 'raw'
print(ps['reliability'], res['corr_lwpc'], res['ceiling_lwpc'], res['caveat'])
```

To redraw only the Nature-style scatter from a finished run, without rerunning
anything:

```python
import pandas as pd
from dcc_scripts.stats.stability_flexibility_brain_behavior_dcc import make_participant_scatter

scores = pd.read_csv('<save_dir>/participant_scores.csv')
make_participant_scatter(scores, '<save_dir>', variant='rtadj')   # or 'raw'
```

A long table from before this change has no `rt` and no `trial`. Level (1) then
returns the neural scores only and says so in `ps['notes']`. It refuses a
participant whose electrodes list different numbers of trials, because without
`trial` they cannot be aligned. **Coverage adjustment** (a
robustness check, not built into the job): merge `participant_electrode_scores.csv`
with the N4 coordinates, fit `d ~ C(subject) + resp + mni_z`, and use the subject
coefficients as participant scores at a common responsiveness and height.

### 8.4 Tests

```bash
pytest tests/analysis/stats/test_participant_brain_behavior.py \
       tests/analysis/stats/test_stability_flexibility_brain_behavior.py \
       tests/analysis/stats/test_assemble_long_df.py \
       tests/analysis/stats/test_effect_sign_conventions.py -o addopts=""
```

---

## 9. Outputs

| File | Contents |
|---|---|
| `summary.txt` | everything below in words; **read this first** |
| `participant_scores.csv` | one row per participant: `n_elec`, `n_trials`, `lwpc_neural`, `lwps_neural`, `lwpc_neural_rtadj`, `lwps_neural_rtadj`, `lwpc_behav`, `lwps_behav` (ms, from `ieeg_behavioral_subject_level_effects.csv`), `lwpc_behav_trials`, `lwps_behav_trials` (ms, scored on the HG trials, for the cross-check), `mean_rt`, `resp` (mean \|HG\|), `rt_hg_r` (median within-cell HG–RT correlation of its electrodes) |
| `participant_electrode_scores.csv` | one row per electrode: the four neural d's, `usable`, `resp`, `rt_slope`, `rt_r` |
| `participant_reliability.csv` | one row per score: `r_half`, `sd_half`, `reliability` (full length), `n_participants`, `n_splits` |
| `participant_brain_behavior.json` | for `rtadj` and `raw`: every number in §10; plus the settings, notes and the CSV cross-check |
| `participant_brain_behavior.png` | LWPC and LWPS (rows) × RT-adjusted and raw (columns); one dot per participant, fitted line, r with CI, ceiling, disjoint-half r |
| `participant_brain_behavior_scatter_<rtadj\|raw\|abs_rtadj\|abs\|pos_rtadj\|pos>.{png,pdf,eps}` | the figure version, in the decoding figures' Nature style: LWPC and LWPS side by side, behavioral score (ms) on x, neural score on y, one dot per participant, least-squares line, R² and p top right. R² and p are that variant's r² and p |
| `behavioral_magnitudes.csv` | per-participant behavioral `lwpc` / `lwps` (and the table's other effect columns) from the subject-level table, under the epochs' subject IDs (levels 1 and 2 use these) |
| `long_df.csv` | the single-trial long table (real runs), including `trial`, `rt`, `acc` |
| `electrode_labels.csv`, `subject_table_<mode>.csv`, `across_subject.json` | level (2) |
| `trial_df.csv`, `trialwise.json` | level (3) |
| `brain_behavior_summary.png` | levels (2) and (3) |

---

## 10. How to read the result

Work down `summary.txt` in this order.

1. **Behavior sanity.** The `behavioral magnitudes` line should show positive
   group means, near +120 ms (LWPC) and +100 ms (LWPS). Individual participants
   scatter widely. The `behavior cross-check` line compares
   `ieeg_behavioral_subject_level_effects.csv` with the same contrast scored on
   the long table's trials, for the same participants. Expect a high r: both
   score the same sessions and share most trials (the long table holds only
   correct trials that survived preprocessing). A low or negative r means the
   notebook and this job are not scoring the same contrast.
2. **Coverage.** How many participants have ≥ `MIN_ELEC` usable electrodes, and
   the median electrode and trial counts. Those dropped still have behavior in
   `participant_scores.csv`.
3. **Reliability and ceiling.** The CSV has no trials, so the behavioral
   reliability (tagged `(iEEG trials)`) is measured on the long table's trials
   and stands in for the CSV's. It should come out near 0.69 (LWPC) and 0.37
   (LWPS), the values on `combinedData.csv`, or somewhat lower if preprocessing
   dropped many trials. A neural reliability near zero means participants do not differ
   measurably in that score; no correlation with anything is then possible. If
   the ceiling is below the |r| needed, even a perfect true link could not reach
   significance, and the text should say so rather than report a null.
4. **The claim: RT-adjusted matched r**, with its 95 % CI and Spearman ρ.
   Compare it with both the |r| needed and the ceiling.
5. **Raw vs adjusted.** A raw r that shrinks after adjustment was carried by RT.
   `within-cell HG-RT correlation` shows how strong the coupling is. Near zero,
   raw and adjusted agree and the question is moot.
6. **Specificity: the JOINT lines.** Each behavioral score regressed on both
   neural scores (z-scored): the matched β should exceed the cross β. Do not read
   specificity off matched |r| > cross |r|. Behavioral LWPC and LWPS correlate
   (r ≈ 0.44), so cross r's are not expected to be zero, and RT coupling alone
   makes matched beat cross.
7. **Same vs disjoint half.** A same-half r clearly above the disjoint-half r
   means shared trial noise inflates the same-trial correlation. After the RT
   adjustment the two should agree.
8. **Levels (2) and (3)** are context. Level (2) inherits the thresholding
   problems in §2. For level (3), see §11.

What to write, by outcome (supplement S-BB):

| Outcome | Sentence |
|---|---|
| Adjusted r n.s., ceiling below the \|r\| needed | "At n = …, reliabilities of … and … cap any observable correlation at …, below the … needed for significance; the across-participant test cannot detect a brain–behavior link of plausible size." |
| Adjusted r n.s., ceiling above the \|r\| needed | "Participants' neural and behavioral adaptation were not detectably related (r = … [CI]); the ceiling was …." Report the CI; do not call it evidence of absence. |
| Adjusted r significant, joint matched β > cross β | "Participants whose lPFC adapted more also adapted more behaviorally (r = …, RT-adjusted; ceiling …), specifically for the matched effect (β = … vs …)." Show both variants. |
| Raw r significant, adjusted not | "The raw correlation (r = …) did not survive removing the RT-linked part of high gamma (r = …), consistent with HG tracking RT rather than adaptation per se." |

---

## 11. Limitations and open items

- **Level (3) as built.** Its adjustment is `w(t) · RT` with `w = +1` on the rare
  cells, so `w` averages about −0.5. Any plain HG–RT correlation leaks into both
  the matched and the cross slope. In the archived run they are nearly equal
  (117 vs 130; 163 vs 154). The closing plan's replacement, `RT ~ congruency ×
  incongruent proportion × HG` with a participant random effect, is not built yet.
- **N2.** The direction tests' sign match with behavior is partly what RT
  coupling alone predicts. `participant_electrode_scores.csv` has RT-adjusted
  per-electrode scores for a re-check.
- **Block order.** 15 of 23 participants ran order DACB, which puts the
  75 %-incongruent blocks later on average. Practice that shrinks the congruency
  effect would then feed LWPC in both brain and behavior. The long table does not
  carry block order; a check needs it joined from the behavioral CSV
  (`absBlockN`).
- **Coverage.** The participant mean depends on where its electrodes sit (the
  dorsoventral tilt) and how responsive they are. See the offline recipe in §8.3.

---

## 12. Tests and what they pin

| Test file | Pins |
|---|---|
| `test_stability_flexibility_brain_behavior.py` | the block map against `mainTask.m` and `combinedData.csv`; full crossing; a planted LWPC is recovered from `blockType` alone, not the cross effect |
| `test_assemble_long_df.py` | `trial` is the epoch index shared across electrodes and survives a per-channel dropped row; `rt` / `acc` come from the metadata and are NaN, not an error, when missing |
| `test_participant_brain_behavior.py` | per-electrode d equals `naive_sensitivities` (raw and adjusted); behavior equals `_dod_rt`; the participant score is the electrode mean; the adjustment removes an RT-linked component exactly and leaves other effects alone; the RT confound passes the specificity checks but not the adjustment; shared-split reliability is ~0 under the null where per-electrode splits are not; a planted link and its ceiling are recovered; missing `rt` / `trial` / error trials; the synthetic job end to end |
| `test_effect_sign_conventions.py` | LOW − HIGH on both sides of every correlation |

---

## 13. First real run (2026-09-30): methods and results

A record of what A6 found on the real data, for understanding, not for the
paper. Source:
`dcc_scripts/stats/results/<epochs>/brain_behavior_window_0.0to1.5s_sig_lpfc_count/`
(`summary.txt`, `participant_brain_behavior.json`), run with the submitter's
defaults (`MIN_ELEC=3`, `PARTICIPANT_N_SPLITS=200`, `SEED=0`,
`FDR_CORRECTION=none`).

### 13.1 Methods

**Neural data.** High-gamma epochs
`Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20`
(70–150 Hz, filterbank Hilbert, stimulus-locked). Each trial's high gamma was
averaged over 0–1.5 s after stimulus onset. Electrodes were the task-significant
electrodes in lateral prefrontal cortex (`ELECTRODES=sig`, `ROIS=lpfc`; 171
electrodes). Only correct trials with a recorded RT were used.

**Neural score per electrode.** For each electrode, the four cell means of
window-mean high gamma were combined into an equal-cell-weight difference of
differences, low minus high proportion, pooling over the other two factors:

- LWPC = (incongruent − congruent) in 25 %-incongruent blocks − (incongruent − congruent) in 75 %-incongruent blocks
- LWPS = (switch − repeat) in 25 %-switch blocks − (switch − repeat) in 75 %-switch blocks

Each was divided by the pooled within-cell SD (Cohen's d); every cell needed
≥ 2 trials. Positive means the condition effect is smaller in the
high-proportion block, the direction behavior shows.

**Neural score per participant.** The mean of its usable electrodes' d (usable =
finite on all four d's: LWPC and LWPS, raw and RT-adjusted). Participants with
fewer than 3 usable electrodes got no score.

**RT adjustment.** Per electrode, the slope b of high gamma on RT was fitted on
deviations from each of the 16 condition-cell means (congruency × switch type ×
incongruent proportion × switch proportion), pooled over cells. Each trial's
high gamma was then replaced by HG − b × (RT − the electrode's mean RT), and
the d's were recomputed. This removes from each electrode's contrast exactly
b × the same contrast of cell-mean RT (§4.1). RTs came from the epochs
metadata. Behavior was not adjusted.

**Behavioral score.** `LWPC_effect` and `LWPS_effect` from the `key_RT_mean`
rows of `src/config/ieeg_behavioral_subject_level_effects.csv` (generated by
`iEEGBehavioralAnalysis.ipynb`, §1.2), in ms, with the same low-minus-high
orientation. Participants were matched on their ID stem (`D0107A` → `D0107`).

**Across-participant statistics (level 1).**
- Pearson r between the neural and behavioral scores, two-sided, with a 95 %
  CI from Fisher's z, and Spearman ρ.
- Cross pairings: neural LWPC against behavioral LWPS, and the reverse.
- Joint regression, all variables z-scored: behavioral LWPC ~ neural LWPC +
  neural LWPS (OLS), and the same for behavioral LWPS. Specificity means the
  matched β exceeds the cross β.
- p-values are not corrected across the 12 level-1 correlations.

**Reliability and ceiling.**
- Each participant's trials were split in half 200 times at random, stratified
  on the 16 cells. One split per participant was shared by all its electrodes.
- Every score was recomputed in each half. `r_half` is the across-participant
  correlation of half-A and half-B values, averaged over splits.
- Full-length reliability = 2 × r_half / (1 + r_half) (Spearman–Brown);
  undefined when r_half ≤ 0.
- The CSV has no trials, so its reliability was estimated by that of the same
  contrast computed on the iEEG trials' RT.
- Ceiling = √(neural reliability × behavioral reliability): the largest r
  expected even for a perfect true link.

**Exploratory neural summaries.**
- Mean |d|: electrodes adapting in opposite directions no longer cancel.
- Mean d over only the electrodes with d > 0, chosen separately for LWPC and
  LWPS. A participant needed ≥ 3 such electrodes for both effects.
- Both average the same noisy d they fold or select on, so noise alone raises
  them. As a noise check, each was correlated with the participant's trial
  count.

**Level 2 (label-based).** A two-way ANOVA (Type III) on each electrode's
window-mean high gamma, one for each interaction, at α = 0.05 uncorrected.
This flagged each electrode as LWPC-selective (S) and/or LWPS-selective (F).
Per participant: the number of S / F electrodes (`count`), their share of its
electrodes (`frac`), and its mean interaction F (`effect`), each correlated
with behavior.

**Level 3 (single trial).**
- Per trial, high gamma was averaged over the participant's S electrodes (or
  its F electrodes) and centered within participant.
- The trial's "adjustment" = w × (RT − participant's mean RT), where w = ±1 is
  the trial's cell sign in the difference of differences.
- A linear mixed model, adjustment ~ high gamma with a participant random
  intercept (ML), was fitted for the matched and for the cross adjustment.

### 13.2 Results

**Sample.**
- 21 participants had electrodes in the set; 18 had ≥ 3 usable electrodes
  (median 7 electrodes and 397 trials each).
- 17 of those 18 are in the CSV; D0144 is not. So n = 17, and |r| ≥ 0.48 was
  needed for p < .05.
- The positive-only summaries kept 11 (RT-adjusted, |r| ≥ 0.60 needed) and 12
  (raw, |r| ≥ 0.58) participants.

**Behavior.**
- Across the 20 participants with high gamma and a CSV row, mean behavioral
  LWPC was 117 ms and mean LWPS 75 ms.
- The CSV agreed with the same contrasts scored on the iEEG trials: r = 0.90
  (LWPC) and 0.76 (LWPS), mean absolute difference 59 and 75 ms.
- The median within-cell high gamma–RT correlation, over electrodes and then
  participants, was +0.13.

**Reliability**, full length (half-split r in brackets); "—" means r_half ≤ 0:

| Score | LWPC | LWPS |
|---|---|---|
| neural, signed mean, raw | — (−0.15) | 0.38 (+0.23) |
| neural, signed mean, RT-adjusted | — (−0.25) | 0.38 (+0.23) |
| neural, mean \|d\|, raw | 0.20 (+0.11) | 0.45 (+0.29) |
| neural, mean \|d\|, RT-adjusted | 0.16 (+0.09) | 0.49 (+0.33) |
| neural, d > 0 electrodes, raw | 0.29 (+0.17) | 0.52 (+0.35) |
| neural, d > 0 electrodes, RT-adjusted | 0.41 (+0.26) | 0.48 (+0.32) |
| behavioral (iEEG-trial RT, stand-in for the CSV) | 0.72 (+0.57) | 0.56 (+0.39) |

**Level 1, matched correlations** (neural against behavioral, same effect):

| Neural summary | Effect | n | r [95 % CI] | p | Spearman ρ | ceiling | r(score, trials) |
|---|---|---|---|---|---|---|---|
| signed mean, RT-adjusted | LWPC | 17 | +0.10 [−0.40, +0.55] | 0.70 | −0.02 | — | |
| | LWPS | 17 | −0.06 [−0.52, +0.44] | 0.83 | +0.10 | 0.46 | |
| signed mean, raw | LWPC | 17 | +0.24 [−0.27, +0.65] | 0.35 | +0.07 | — | |
| | LWPS | 17 | +0.19 [−0.32, +0.61] | 0.47 | +0.31 | 0.46 | |
| mean \|d\|, RT-adjusted | LWPC | 17 | +0.32 [−0.19, +0.69] | 0.21 | +0.07 | 0.34 | −0.15 |
| | LWPS | 17 | −0.45 [−0.77, +0.03] | 0.068 | −0.38 | 0.53 | −0.61 |
| mean \|d\|, raw | LWPC | 17 | +0.23 [−0.28, +0.64] | 0.37 | −0.00 | 0.38 | −0.05 |
| | LWPS | 17 | −0.26 [−0.66, +0.25] | 0.31 | −0.18 | 0.50 | −0.46 |
| d > 0 electrodes, RT-adjusted | LWPC | 11 | +0.56 [−0.06, +0.87] | 0.074 | +0.34 | 0.54 | +0.21 |
| | LWPS | 11 | +0.44 [−0.22, +0.82] | 0.18 | +0.32 | 0.52 | −0.02 |
| d > 0 electrodes, raw | LWPC | 12 | +0.53 [−0.07, +0.85] | 0.078 | +0.49 | 0.46 | +0.27 |
| | LWPS | 12 | +0.55 [−0.04, +0.85] | 0.065 | +0.55 | 0.54 | +0.07 |

**Level 1, cross pairings and joint regressions** (β with p):

| Neural summary | cross r: neural LWPC ↔ behav. LWPS | cross r: neural LWPS ↔ behav. LWPC | behav. LWPC: β matched / β cross | behav. LWPS: β matched / β cross |
|---|---|---|---|---|
| signed mean, RT-adjusted | −0.13 | +0.15 | +0.05 (.87) / +0.13 (.66) | −0.00 (.99) / −0.13 (.67) |
| signed mean, raw | +0.01 | +0.15 | +0.22 (.45) / +0.07 (.81) | +0.22 (.46) / −0.08 (.78) |
| mean \|d\|, RT-adjusted | −0.38 | −0.39 | +0.51 (.038) / −0.57 (.024) | −0.37 (.16) / −0.26 (.31) |
| mean \|d\|, raw | −0.32 | −0.41 | +0.40 (.11) / −0.53 (.039) | −0.18 (.50) / −0.26 (.34) |
| d > 0 electrodes, RT-adjusted | +0.02 | −0.06 | +0.56 (.095) / −0.02 (.95) | +0.44 (.20) / +0.05 (.87) |
| d > 0 electrodes, raw | +0.00 | −0.17 | +0.51 (.11) / −0.06 (.83) | +0.57 (.073) / +0.13 (.67) |

**Level 2** (n = 20). Of 171 electrodes, 22 were flagged S (LWPC), 27 F (LWPS),
and 5 both.

| Neural summary | matched LWPC | matched LWPS | cross LWPC ↔ LWPS | cross LWPS ↔ LWPC |
|---|---|---|---|---|
| `count` | +0.47 (p = .035) | +0.14 (.57) | +0.14 (.56) | +0.00 (.995) |
| `frac` | +0.29 (.21) | −0.21 (.38) | −0.13 (.60) | −0.36 (.11) |
| `effect` (mean F) | +0.27 (.25) | −0.06 (.81) | −0.11 (.64) | −0.19 (.42) |

**Level 3.** Slopes are in ms of adjustment per unit of high gamma. The printed
`n_subjects` = 24 counts every participant in the trial table; only those with
an electrode in the group enter the fit.

| Electrode group | trials | matched slope (z, p) | cross slope (z, p) |
|---|---|---|---|
| LWPC (S) | 3660 | −89 (z = −3.8, p = 1e−4) | −140 (z = −6.0, p = 2e−9) |
| LWPS (F) | 4522 | −161 (z = −8.6, p = 1e−17) | −149 (z = −7.9, p = 2e−15) |

### 13.3 Interpretation

- **The primary test is null, and could not have been otherwise.** The signed
  neural LWPC score has no measurable between-participant reliability (half-split
  r below zero), so it cannot correlate with anything. The LWPS ceiling (0.46)
  is below the 0.48 needed at n = 17, so even a perfect true link could not
  reach significance.
- **Raw vs RT-adjusted.** The raw correlations (+0.24, +0.19) shrink after
  adjustment (+0.10, −0.06), and high gamma tracks RT within cells (r ≈ +0.13).
  This fits part of the raw correlation being RT coupling, but neither variant
  is significant.
- **Mean |d| is not interpretable.**
  - The LWPS score is noise-driven: it falls with trial count (r = −0.61
    adjusted, −0.46 raw).
  - LWPC reliability is 0.16–0.20.
  - The one "significant" joint regression (β +0.51 vs −0.57) is a suppression
    pattern, not specificity. The two |d| scores share a noise-magnitude
    component (they correlate +0.34 RT-adjusted and +0.31 raw across
    participants, as implied by the β's and simple r's), and
    opposite-signed simple correlations with behavior then inflate both β's.
- **The positive-only trend (r ≈ 0.53–0.56, p ≈ 0.07) is not evidence.**
  - Only 11–12 participants remain.
  - The observed r meets or exceeds its own ceiling, which would imply a true
    correlation near 1; sampling noise has most likely inflated it.
  - For RT-adjusted LWPC, Spearman ρ (0.34) is well below Pearson r (0.56),
    so a few participants carry it.
  - It is one of 12 uncorrected level-1 tests, chosen after the others were
    null.
  - Electrodes were selected on the sign of the same d that is averaged.
  - Its noise check is fine, so trial count isn't driving it.
  - A non-circular follow-up would pick each participant's positive electrodes
    on half the trials and score them on the other half.
- **Level 2's one nominal hit** (`count`, LWPC, r = 0.47, p = .035) is
  uncorrected across six tests and drops to 0.29 as a fraction of the
  participant's electrodes. That points to coverage (more electrodes, more
  significant ones) rather than adaptation.
- **Level 3 measures the high gamma–RT link, not adaptation.** All four slopes
  are negative and of similar size, matched and cross alike. That is the
  signature §11 predicts: w averages −0.5, so any positive high gamma–RT
  correlation yields a negative slope for every adjustment. A valid single-trial
  test would be RT ~ congruency × incongruent proportion × high gamma with a
  participant random effect, where the three-way term carries the question.

**If this were reported** (supplement): "At n = 17, the reliability of the
participant-level neural scores (LWPC: none measurable; LWPS: 0.38) capped any
observable brain–behavior correlation below the |r| = 0.48 needed for
significance; the across-participant test could not detect a link of plausible
size." The full manuscript text is in §14.

### 13.4 Power at the observed ceiling

Simulated from the LWPS ceiling of the RT-adjusted run (0.46). 40,000 draws of
n = 17 from a bivariate normal with correlation = ceiling × true r; "n for 80 %"
is from Fisher's z. Both treat the estimated reliabilities as known.

| True r | Expected observed r | Power at n = 17 | n for 80 % power |
|---|---|---|---|
| 0.3 | 0.14 | 0.08 | 407 |
| 0.5 | 0.23 | 0.14 | 146 |
| 0.7 | 0.32 | 0.25 | 73 |
| 1.0 | 0.46 | 0.49 | 35 |

For LWPC no table is possible: the neural score has no measurable reliability.
More participants would not fix that; only more trials per participant (or a
more reliable neural score) would.

---

## 14. Manuscript text and placement (2026-10-01)

### 14.1 Placement

**Not in the main text. Keep a short supplementary note (S-BB) and one
Discussion sentence. Do not omit it entirely.**

- **Why not the main text.** The test was uninformative before it ran: the LWPS
  ceiling is below the |r| needed, and neural LWPC has no measurable
  between-participant reliability. A null that could not have been otherwise
  carries no evidence either way.
- **Why not drop it.** Reviewers will ask whether neural adaptation tracks
  behavioral adaptation (the closing figure plan expected this). The reliability
  analysis already answers that it cannot be tested at this n, with numbers. One
  paragraph and one figure pre-empt the question, and leaving the analysis out
  after running it is harder to defend than reporting it.
- **What the null is not.** It is not evidence of no link: the CIs run to
  +0.55 (LWPC) and +0.44 (LWPS). "Underpowered" is also only half the story.
  For LWPS, more participants would help but would take about 146 for a true
  r = 0.5 (§13.4). For LWPC, no number of participants would help at these trial
  counts. Write "could not be tested", not "no relationship" and not "a trend".
- **Report:** the RT-adjusted matched correlations (primary), the unadjusted ones
  as an upper bound, the reliabilities, the ceiling, and the power figure.
  Methods: [`methods.md` › A6](methods.md#a6-brainbehavior-supplement-s-bb).
- **Do not report as results:**

  | Result | Why not |
  |---|---|
  | Positive-electrode summary, r ≈ 0.53–0.56, p ≈ 0.07 | Post hoc, after the primary test was null. Only 11–12 participants remain. r is at or above its own ceiling in 3 of 4 cases, so it is inflated. It selects on the sign of the same d it averages (§13.3). |
  | Mean \|d\| joint regression, p = .02–.04 | A suppression pattern from a shared noise component, not specificity. The LWPS score falls with trial count, r = −0.61 (§13.3). |
  | Level 2 `count`, LWPC r = 0.47, p = .035 | Uncorrected across six tests. It drops to 0.29 as a fraction of electrodes, which points to coverage. |
  | Level 3 slopes | They measure the high gamma–RT link: matched and cross slopes are equal (§11). |

  If you want the exploratory summaries on record, use the optional last
  paragraph of the Methods and one sentence of Results saying they were not
  interpretable and why. Do not use them as a figure. The positive-only trend
  can only be tested out of sample: select each participant's positive
  electrodes on half of the trials and score them on the other half.
- **Remaining checks.** All lPFC and the 0–0.5 s window (§8.2) are not needed for
  the supplement. All lPFC dilutes the participant mean with unresponsive
  electrodes (§3), and the RT-confound question is moot when the adjusted and
  unadjusted results are both null. Run them only if a reviewer asks. The one
  version with the power to say something is the within-participant three-way
  model (RT ~ congruency × incongruent proportion × HG, participant random
  effect; §11), which is not built.

### 14.2 Results (supplement S-BB)

> **Brain–behavior.** We asked whether participants whose lateral prefrontal
> high gamma adapted more also adapted more in behavior. Of the 21 participants
> with task-significant lPFC electrodes, 17 had at least three usable
> electrodes and behavioral scores (median 7 electrodes and 397 trials per
> participant). Behavioral adaptation was reliable across participants
> (split-half reliability, estimated on the recorded trials: 0.72 for LWPC,
> 0.56 for LWPS). Participant-level neural adaptation was not: neural LWPC had
> no measurable between-participant reliability, and neural LWPS had a
> reliability of 0.38. These reliabilities cap the observable LWPS correlation
> at 0.46, below the |r| = 0.48 needed for significance at n = 17. Even a perfect
> underlying relationship would have reached significance in fewer than half of
> samples of this size (simulated power 0.49), and a moderate one (r = 0.5) in
> 14 %. For LWPC, no correlation was detectable at all. Neither correlation
> differed from zero after removing the RT-linked component of high gamma (LWPC:
> r = 0.10, 95 % CI [−0.40, 0.55], p = 0.70; LWPS: r = −0.06 [−0.52, 0.44],
> p = 0.83; Supplementary Fig. S-BB) or without that adjustment (LWPC: r = 0.24
> [−0.27, 0.65], p = 0.35; LWPS: r = 0.19 [−0.32, 0.61], p = 0.47). The
> unadjusted correlations were somewhat larger, as expected if part of each
> neural score reflects trial-by-trial coupling between high gamma and RT
> (median within-cell r = 0.13). With these intervals, the data neither support
> nor rule out a relationship between individual differences in neural and
> behavioral adaptation.

Optional sentence, if the exploratory summaries are in the Methods:

> Two alternative participant summaries, the mean absolute score and the mean
> over positively scoring electrodes, gave no interpretable result: the first
> tracked participants' trial counts (LWPS: r = −0.61), as expected of a score
> driven by noise, and the second retained only 11 participants.

### 14.3 Discussion sentence

> We could not test whether individual differences in lPFC adaptation track
> individual differences in behavioral adaptation. With 17 participants, the
> split-half reliability of participant-level difference-of-differences scores
> left any across-participant correlation undetectable, even for a strong
> underlying relationship (Supplementary Note S-BB). Difference scores are
> characteristically unreliable across individuals (Hedge et al., 2018), so such
> a test needs many more participants, many more trials per participant, or
> both.

### 14.4 Figure caption (Supplementary Fig. S-BB)

Use `participant_brain_behavior_scatter_rtadj` (or the 2 × 2
`participant_brain_behavior.png` if the unadjusted panels go in too), with one
change. The scatter prints R², which drops the sign: LWPS is r = −0.06. Annotate
each panel with r, its 95 % CI and the ceiling instead. `make_participant_scatter`
(§8.3) redraws it from `participant_scores.csv` without a rerun.

> **Supplementary Fig. S-BB | Neural against behavioral adaptation across
> participants.** Each point is one participant (n = 17). *x*: behavioral LWPC
> (left) and LWPS (right), the difference of differences in mean RT (ms), low-
> minus high-proportion blocks. *y*: the same contrast in high gamma (0–1.5 s
> after stimulus onset), as Cohen's *d* averaged over the participant's
> task-significant lPFC electrodes, after removing the component of high gamma
> linearly related to RT within condition cells. Positive values on both axes
> indicate the predicted adaptation. Lines are least-squares fits. Each panel
> gives Pearson's r with its 95 % CI and p. The ceiling (LWPS: 0.46) is the
> largest correlation expected for a perfect underlying relationship, given the
> split-half reliabilities of the two scores. Neural LWPC had no measurable
> reliability, so no ceiling is given. Significance at α = .05 required
> |r| ≥ 0.48.
