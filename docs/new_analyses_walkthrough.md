# The 2026-10-13 analyses: how to run them, the code, and the statistics

Five analyses were added or changed on 2026-10-13 (commit `80ad353`, with the
cosine option in `map_reliability` added on 2026-10-14). For each one this doc
gives the commands to run it on the DCC, walks through the code that does the
work, and explains the statistics and why they are set up that way. The short
runbook entries stay where they are: [`decoding.md`](decoding.md) §2.4, §3.6
and §3.7, and [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)
§19.4–§19.5. This doc is the long version.

| § | Analysis | Code status | Run with | Short entry |
|---|---|---|---|---|
| [1](#1-decoding-with-channels-as-the-features) | Decoding with the channels as the features (one sample = one trial at one time point), and PCA trajectories in channel space | Built and tested | `FEATURE_MODE=channels` on any decoding submitter | [`decoding.md`](decoding.md) §2.4 |
| [2](#2-window-mean-decoding-in-sig-nonsig-and-all-electrodes) | Window-mean decoding in sig, non-sig and all electrodes at a matched electrode count (the bar plot) | Built and tested | `submit_window_mean_groups_dcc.sh` | [`decoding.md`](decoding.md) §3.6 |
| [3](#3-the-top-20--electrodes-by-each-score-on-held-out-trials) | The top 20 % of electrodes by LWPC, LWPS, congruency or switch type, chosen on 30 % of the trials; decoding and power traces on the other 70 % | Built and tested | `sbatch_score_electrode_selection_dcc.sh`, then `submit_decoding_with_score_electrode_sets_dcc.sh` and `submit_specific_conditions_power_traces_dcc.sh` | [`decoding.md`](decoding.md) §3.7 |
| [4](#4-figure-5-the-brain-the-scatter-and-every-electrode-along-z) | Figure 5 redrawn: a, the brain; b, LWPC against LWPS with a zoom on the centroids; c, every electrode along MNI z | Built and tested; preview drawn 2026-10-10 | The anatomy job, or `n4_section19_followups.py --sections 1,4 --brain` | [N4 §19.4](n4_continuous_anatomy.md#194-the-combined-figure-5) |
| [5](#5-the-state-space--subspace-figure-partly-built) | State-space (subspace) figure | **Partly built.** The coding axes and the cosine between them (§3.1-3.3) were built on 2026-10-15, after this doc; see [`coding_axes.md`](coding_axes.md). The subspace test, trajectories and dPCA are not built | – | [Subspace trajectory plan](analysis_plans.md#subspace-trajectory-plan) |

"Built and tested" means the code is in `main` and its tests pass
([§6](#6-tests)). It does not mean the analysis has been run on the real data;
check `dcc_scripts/<area>/figs/` on the DCC for that.

---

## 0. Before you run anything

**Where to run.** Follow [`CLAUDE.md`](../CLAUDE.md) and
[`dcc_ai_agents.md`](dcc_ai_agents.md): submit from the DCC, never run Python on
a login node. Every command below assumes this preamble:

```bash
cd /hpc/home/$USER/coganlab/$USER/GlobalLocal
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ieeg
```

**`cd` into the submitter's folder first.** Each `sbatch_*.sh` writes its log
to `out/` relative to where you submit, and the submitters call their
`sbatch_*.sh` by relative path.

**Check the ROIs.** As of 2026-10-08 both `run_decoding_dcc.py` and
`run_power_traces_dcc.py` ship with `ROIS_DICT = {'dlpfc': ...}`.

- Decoding reads `ROIS` from the environment (`ROIS=lpfc`, or `ROIS=lpfc,dlpfc`).
- Power traces has no override. To run lPFC there, edit `ROIS_DICT` in
  `dcc_scripts/power/run_power_traces_dcc.py` in your local clone, push, and pull
  on the DCC.
- The score selection (§3) must cover every ROI the later decoding and power
  jobs use. They refuse to run otherwise.

**What a call submits.**

| Job | Batch script | Jobs per call | Each job asks for |
|---|---|---|---|
| Decoding (§1, §2, §3 step 2) | `dcc_scripts/decoding/sbatch_decoding_dcc.sh` | one per condition set in `CONDITIONS` | 5 CPUs, 225 GB, up to 48 h |
| Score selection (§3 step 1) | `dcc_scripts/decoding/sbatch_score_electrode_selection_dcc.sh` | 1 | 2 CPUs, 64 GB, up to 4 h |
| Power traces (§3 step 3) | `dcc_scripts/power/sbatch_power_traces_dcc.sh` | 4: the four block-balanced condition sets, hard-coded in the submitter | 8 CPUs, 100 GB, up to 10 h |
| Anatomy (§4) | `dcc_scripts/stats/sbatch_stability_flexibility_anatomy_dcc.sh` | 1 | 8 CPUs, 64 GB, up to 12 h |

---

## 1. Decoding with channels as the features

### 1.1 The question

The ordinary decoder treats one trial as one sample, and every channel × time
point in the window as a feature. It can separate the classes by **where** the
response is (which channels) or by **when** it happens within the window (the
time course). With `FEATURE_MODE=channels`, each trial at each time point is a
sample and only the channels are features. The decoder then asks a narrower
question: is there **one spatial pattern** across electrodes that separates the
classes throughout the window? A difference in the time course within the window
no longer counts.

### 1.2 How to run it

```bash
cd dcc_scripts/decoding
FEATURE_MODE=channels ROIS=lpfc \
CONDITIONS="stimulus_congruency_conditions stimulus_switch_type_conditions" \
    bash submit_specific_conditions_decoding_dcc.sh
```

- Without `CONDITIONS` the submitter runs its four block-balanced sets (LWPC,
  LWPS, and congruency and switch type by the other proportion), one job each.
- `FEATURE_MODE` is a property of the decoder, so it combines with the other
  electrode routes: `ELECTRODE_SETS_JSON` (§3), `ANOVA_LABELS_CSV`, the
  responsiveness split.
- The classifier needs a `decision_function` (LDA, the default, or linear SVC).

**The channel-space trajectories** are drawn by the visualisation step, which is
off by default. It is a constant, not an environment variable: set
`RUN_VISUALIZATION_DEBUG = True` near the top of
`dcc_scripts/decoding/run_decoding_dcc.py` (line 138) in your local clone, push,
pull on the DCC, then submit as above.

**Outputs.** The usual decoding folders (`figs/<EPOCHS>/<condition>/<roi>/`),
with `_feat-channels` in every file name. With the visualisation step on, each
comparison and ROI also gets:

- `channel_pca_3d_trajectory_<roi>.png`/`.pdf`
- `channel_pca_3d_trajectory_<roi>_condition_mean_removed.png`/`.pdf`

### 1.3 What it changes, and what it leaves alone

`feature_mode` reaches the sliding-window decode and its pooled shuffle:

```
run_decoding_dcc.py      FEATURE_MODE (env) -> args.feature_mode
decoding_dcc.main        adds '_feat-channels' to the file names
process_bootstrap        -> get_confusion_matrices_for_rois_time_window_decoding_jim(feature_mode=...)
                         -> run_pooled_shuffle_for_roi -> make_pooled_shuffle_distribution(feature_mode=...)
roi_confusion            Decoder(..., feature_mode=...) for the true and the shuffled decode
decoder.Decoder          the switch itself (§1.4)
```

So the accuracy traces, the cluster test against the pooled shuffle, and the
LWPC/LWPS context comparisons all use channel features, because they are built
from those confusion matrices. Three things do **not** change:

- the time-averaged confusion matrices (`get_time_averaged_confusion_matrix`),
  which still flatten channels × time over the whole epoch. Their file names
  get `_feat-channels` too, so don't read them as a channel-feature result;
- every job that builds its own `Decoder` (A4 cross-decoding, TFR-cluster
  decoding, window-mean decoding in §2);
- the older `pca_3d_trajectory` plot, which the visualisation step still draws
  next to the new one.

### 1.4 The code, step by step

All of it is in `Decoder._window_and_predict_minimal`
(`src/analysis/decoding/decoder.py:439–486`), which runs once per CV fold.
Steps 1–5 are unchanged: the fold's train and test trials are stacked,
`sample_fold` fills NaNs (mixup for training trials, noise for test trials), and
`windower` cuts the windows. Each window arrives as `(trials, channels, w)`.

1. **Reshape** (`decoder.py:452–457`). `np.moveaxis` puts time next to trials,
   `(trials, w, channels)`, and the reshape makes it `(trials × w, channels)`.
   Trial k's time points are rows `k·w` to `k·w + w − 1`.
2. **Split by trial** (`decoder.py:461`). `sample_fold` put the training trials
   first, so cutting at row `n_train × w` cuts between trials. The fold was
   chosen over trials before the reshape, so all of a trial's time points are on
   one side.
3. **Repeat the labels** (`decoder.py:462`). `y_fit = np.repeat(y_train, w)`:
   each training trial's label goes to each of its `w` rows.
4. **Fit.** The `PcaEstimateDecoder` pipeline: PCA keeping `EXPLAINED_VARIANCE`
   (0.90) of the variance, then LDA. PCA now runs in channel space, so it has
   at most `n_channels` components.
5. **Predict a trial, not a row** (`_predict_trials`, `decoder.py:54–68`). The
   decision score of every test row is averaged over the trial's `w` rows (a
   soft vote), and the trial gets the class with the best mean score: the sign
   for two classes, the argmax for more.
6. **Count trials.** `confusion_matrix(y_test, preds)` therefore counts test
   trials, exactly as in the default mode, so the statistics and figures
   downstream need no change.

The label shuffle for the null (`decoder.py:293–296`) permutes `y_train` per
trial, before step 3, so a shuffled trial's time points share one wrong label.

### 1.5 The statistics, and why it is set up this way

**Why the folds are cut by trial.** Neighbouring time points of one trial are
strongly autocorrelated: the HG envelope is smooth, and at 256 Hz one 250 ms
window (`WINDOW_SIZE = 64`) holds 64 samples of it. If the folds were cut over
the `trials × w` rows, the classifier would
be tested on time points whose neighbours it was trained on, and accuracy would
measure that memory, not the class. Cutting by trial keeps train and test
independent, as in the default mode.

**What the soft vote does.** PCA and LDA are linear, so a row's decision score
is an affine function of its channel vector. The mean of the `w` scores of a
trial is then exactly the score of the trial's **window-mean** channel vector.
At test time, channel mode classifies each trial's mean pattern over the window.
What differs from simply decoding window means is training: the classifier is
fitted to single time points, so its covariance estimate includes the
fluctuation of the pattern within the window, and it has `w` times as many rows.
Those rows are not independent, so the effective sample size is still the
number of training trials. This is also why a time-course difference inside the
window cannot be decoded: averaging erases it.

**The null.** Unchanged in form: the same decoder fitted to shuffled training
labels, pooled over the comparison's conditions (`make_pooled_shuffle_distribution`),
with the time-cluster test of [`decoding.md`](decoding.md) §2.3. The shuffled
decode uses the same `feature_mode` as the true one; a null built from a
different decoder would answer a different question.

**Pseudo-trials.** Electrodes from different participants were recorded on
different trials. The decoding arrays pair them by condition into pseudo-trials
([`decoding.md`](decoding.md) §2.2). Because every trial is time-locked to the
stimulus, a (pseudo-trial, time point) vector is a pseudo-population state at
that time.

### 1.6 The channel-space trajectories

`channel_pca_trajectories` (`src/analysis/decoding/plots/trajectories.py:396`),
drawn by `plot_channel_pca_3d_trajectory` (`:433`):

1. One balanced bootstrap of the ROI's data (`concatenate_and_balance_data_for_decoding`,
   `balance_method='subsample'`).
2. Each condition's mean over trials: `(conditions, channels, time)`.
3. Each mean averaged within the decoder's windows: one channel vector per
   condition and window (the population state).
4. With `remove_condition_mean`, the across-condition mean state of each window is
   subtracted.
5. PCA is fitted with the (condition, window) states as samples and the channels
   as features, so each PC is a pattern across channels. Each condition's path is
   its states projected onto the top 3 PCs.
6. The window centres are read from the data's own time labels, so the time
   axis is right for any epoch start.

**Why two versions.** In high gamma most of the variance across states is the
response that every condition shares, so PC1 of the plain version is mostly
"activity rises after the stimulus" (the test
`test_pc1_is_the_shared_response_unless_the_condition_mean_is_removed` plants
exactly this). The condition-mean-removed version takes that out, so its PCs
describe how the conditions differ.

**How to read them, and their limits.**

- **They are descriptive.** The PCs and the projections come from the same
  trials, with no cross-validation. Two condition means always differ by some
  noise, and with many channels that noise alone separates the paths. Nothing in
  the plot is tested; the held-out version is the subspace plan's §4 (fit the
  axes on half the trials, project the other half).
- **With two conditions the mean-removed paths are mirror images.** After the
  mean of two states is subtracted they are +d/2 and −d/2, so the two paths are
  reflections through the origin. The plot then shows the path of the difference
  vector d over time, drawn twice.
- **The older `pca_3d_trajectory`** fits PCA on channels × time features: each
  PC is a whole spatiotemporal pattern, which is why its first PCs explain
  little of the variance. It also takes its time axis from `FIRST_TIME_POINT`
  (−1.5), which is 0.5 s off for the −1.0 s epochs.

### 1.7 Run log (2026-10-15): first real runs failed; rerun outstanding

*Drafted by Claude from the job logs; not yet reviewed. No results yet.*

`submit_specific_conditions_decoding_dcc.sh` with `FEATURE_MODE=channels`, no
ANOVA table (job names `dec_a0e0_<condition>_feat-channels`), all electrodes,
six condition sets (congruency, switch type, LWPC and LWPS block-balanced,
congruency by switch proportion, switch type by incongruent proportion). Three
attempts, none finished:

| Jobs | ROIs | Request | Outcome | Cause |
|---|---|---|---|---|
| 58724121–58724126 | dlpfc | 5 CPUs, 225 GB | 5 FAILED, 1 CANCELLED (58724121) after 17–35 min | `numpy.linalg.LinAlgError: SVD did not converge`, raised in a loky worker while fitting a window's decoder for the pooled label-shuffle null (`make_pooled_shuffle_distribution` → `Decoder._window_and_predict_minimal` → `fit`) |
| 58725544–58725549 | dlpfc | 5 CPUs, 225 GB | all 6 OUT_OF_MEMORY after 15–35 min | OOM kill at 225 GB, right after loading (19–20 GB RSS) in the first PCA fit, or (58725549) while building the 2D-visualization debug arrays |
| 58725697–58725702 | dlpfc, lpfc | 5 CPUs, 225 GB | all 6 OUT_OF_MEMORY after 15–32 min | OOM kill at 225 GB while building the first bootstrap's arrays or the visualization debug arrays |

The same six conditions were resubmitted as 58738829–58738834 (dlpfc, lpfc;
50 CPUs, 500 GB) and were running or pending on 2026-10-15. Whether the SVD
failure of the first attempt recurs there is not known yet: the second and
third attempts died of memory before reaching that step.

---

## 2. Window-mean decoding in sig, nonsig and all electrodes

### 2.1 The question

Do the task-significant electrodes (`sig`), the rest (`nonsig`) and all of them
(`all`) decode a contrast equally well? It decides whether the anatomy can use
all lPFC electrodes rather than only the task-significant ones: if the nonsig
electrodes decode as well as the sig ones, restricting to sig throws data away.

### 2.2 How to run it

```bash
cd dcc_scripts/decoding
ROIS=lpfc bash submit_window_mean_groups_dcc.sh                    # congruency and switch type
ROIS=lpfc,dlpfc WINDOW_MEAN_N_PERM=2000 bash submit_window_mean_groups_dcc.sh
CONDITIONS="stimulus_lwpc_block_balanced_conditions" ROIS=lpfc \
    bash submit_window_mean_groups_dcc.sh
WINDOW_MEAN_MATCH_ELECTRODES=false ROIS=lpfc bash submit_window_mean_groups_dcc.sh   # each group whole
```

One job per condition set; the default sets are congruency and switch type. The
job runs only this analysis, in place of the decoding battery, and refuses to
combine with any other electrode route (`ANOVA_LABELS_CSV`, the splits,
coupling sets, `ELECTRODE_SETS_JSON`).

| Variable | Default | What it does |
|---|---|---|
| `WINDOW_MEAN_TMIN` / `WINDOW_MEAN_TMAX` | 0 / 1.5 s | The window each electrode's HG is averaged over |
| `WINDOW_MEAN_BOOTSTRAPS` | 20 | Pseudo-trial draws |
| `WINDOW_MEAN_N_PERM` | 1000 | Membership permutations per pair of groups. The smallest possible p is 1/(n_perm + 1) |
| `WINDOW_MEAN_N_MATCH` | smallest group | Electrodes decoded per group |
| `WINDOW_MEAN_MATCH_ELECTRODES` | true | `false` decodes each group on all its electrodes, with no size matching (§2.4). `WINDOW_MEAN_N_MATCH` must then be empty |

The classifier, `EXPLAINED_VARIANCE`, `N_SPLITS` and `N_REPEATS` are the
decoding job's (LDA, 0.90, 5, 5).

**Cost.** About 0.16 s per decode. Each permutation decodes both groups in every
bootstrap, so 1000 permutations × 20 bootstraps × 2 groups × 3 pairs is 120,000
decodes, about an hour per contrast and ROI on 5 cores. Unmatched, the decodes
are the same in number but larger (`all` uses every electrode), so slower.

**Outputs**, in
`figs/<EPOCHS>/window_mean_groups/<condition>_<tmin>to<tmax>s_<n>bts_<n>perm_seed<seed>/<comparison>/<roi>/`
(the tag ends in `_unmatched` with `WINDOW_MEAN_MATCH_ELECTRODES=false`):

| File | What |
|---|---|
| `accuracies.csv` | One row per group, bootstrap and CV repeat: true accuracy and shuffled-label accuracy |
| `summary.csv` | Per group: mean accuracy, its 2.5–97.5 percentiles over bootstraps × repeats, mean chance, the group's size, `n_match` (empty unmatched) and `n_decoded` (electrodes it was decoded on) |
| `tests.csv` | Per pair: the accuracy difference, permutation p, Holm-adjusted p, the null's 2.5–97.5 percentiles |
| `params.json` | Every setting, including group sizes and subjects |
| `group_bars.png`/`.pdf` | The bar plot (§2.5) |

### 2.3 The code, step by step

Entry: `decoding_dcc.main` (`dcc_scripts/decoding/decoding_dcc.py:421`) hands
over to `run_window_mean_group_comparison`
(`dcc_scripts/decoding/window_mean_groups_dcc.py:29`), which loads, saves and
plots. The computation is `src/analysis/decoding/window_mean_decoding.py`.

1. **Groups** (`electrode_groups`, `window_mean_groups_dcc.py:21`). `all` is
   every ROI electrode in the loaded epochs; `sig` is the task-significant ones
   among them (from `sig_chans_<subject>_<EPOCHS>.json`, HG above its
   pre-stimulus baseline); `nonsig` is `all` minus `sig`. The `ELECTRODES`
   constant plays no part.
2. **Data, once per bootstrap.** One pseudo-trial draw over **all** electrodes
   (`make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel`),
   so every group is a subset of the same columns and sees the same trials.
   `window_mean_dataset` (`window_mean_decoding.py:41`) balances the classes as
   the time-resolved decode does and averages each electrode over the window:
   `X` is `(trials, electrodes)`.
3. **Decode every group at the matched size** (`compare_electrode_groups`,
   `:98`). In each bootstrap each group gets `n_match` electrodes drawn at random
   without replacement, a fresh draw each time (`_group_columns`, `:74`).
   Unmatched, each group is decoded on all its electrodes instead.
   `decode_accuracy` (`:63`) runs 5-fold CV × 5 repeats and returns one accuracy
   per repeat. The same columns decoded with shuffled training labels give
   chance.
4. **The observed difference** for a pair is mean accuracy of group 1 minus
   group 2, over bootstraps and repeats, from the draws in step 3.
5. **The null** (`compare_electrode_groups`, the `for g1, g2 in pairs` loop). The
   two groups' electrodes are pooled and shuffled, and the shuffled pool is dealt
   back: its first `n1` electrodes become pseudo-group 1, its last `n2` become
   pseudo-group 2. `_group_diff` (`:82`) then repeats steps 3–4 on the
   pseudo-groups, with new electrode draws and new CV seeds. This is repeated
   `n_perm` times.
6. **p and Holm.** p = (1 + #{|null| ≥ |observed|}) / (1 + n_perm), two-sided.
   Unmatched, p = 2 × (1 + the smaller of #{null ≥ observed} and
   #{null ≤ observed}) / (1 + n_perm), capped at 1 (§2.4). Holm's correction is
   applied across the pairs (three by default).

### 2.4 The statistics, and why it is set up this way

**Why the groups are matched in size.** Decoding accuracy grows with the number
of electrodes. `all` contains `sig` and `nonsig`, so it would win on size alone.
Decoding every group with the same `n_match` electrodes compares what an
electrode of each group carries, not how many there are. Drawing the `n_match`
afresh in every bootstrap averages over which electrodes happen to be picked.

**Without the matching** (`WINDOW_MEAN_MATCH_ELECTRODES=false`). Each group is
decoded on all its electrodes, so the bars mix what an electrode carries with
how many there are, and `all` is expected to come out on top. The test still
accounts for size: each permutation deals pseudo-groups of the real groups'
sizes and decodes them whole, so the null asks whether the real group does
better or worse than a random group **of its own size** from the pool. That null
is centred on the size difference, not on zero, so the p doubles the smaller
tail instead of comparing absolute differences
(`test_unmatched_null_carries_the_size_difference`). Its floor is
2/(n_perm + 1). For `sig` vs `all` the pseudo-group 2 is `all` in every
permutation, so the null is a random `n_sig` of all against all of it.

**Why a membership permutation and not an ANOVA over bootstraps.** The 20
bootstraps and the 5 CV repeats re-use the same trials. They measure the
variability of the procedure, not independent replications. An ANOVA or t-test
across them gets more significant the more of them you run, with no change in
the data. The permutation asks the question at the right level: **does it matter
which group an electrode belongs to?** Under that null, a random set of
electrodes from the pool decodes as well as the real group.

**How the deal handles overlap and nesting.** Dealing the shuffled pool from the
front (group 1) and the back (group 2) keeps the overlap the two real groups
have:

| Pair | Pool | Pseudo-group 1 | Pseudo-group 2 |
|---|---|---|---|
| sig vs nonsig (disjoint) | all | a random `n_sig` of all | the remaining `n_nonsig` |
| sig vs all (nested) | all | a random `n_sig` of all (a pseudo-sig) | all of it |
| nonsig vs all (nested) | all | a random `n_nonsig` of all | all of it |

So for a nested pair the superset stays whole and the null draws a
pseudo-subset from it (`test_nested_null_keeps_the_superset_whole`).

**What the null carries.** Each permutation redraws the matched electrodes and
the CV splits, so the null includes the noise of those draws as well as the
membership. The trial draws are fixed across permutations: the test is
conditional on these pseudo-trials and treats electrodes as the unit. It says
whether group membership matters for these electrodes, not whether the result
would replicate in new participants.

**Selection is label-blind.** `sig` is defined by HG against baseline, which
ignores the condition labels. Selecting on it cannot by itself make a group
decode the contrast better under the null of no condition effect, so the sig
group needs no held-out trials here.

**Reading it.**

- For the anatomy, the claim needed is "nonsig and all decode **at least as
  well as** sig". A non-significant p is not evidence of equivalence. Report the
  difference next to the null's 95 % range (`tests.csv`, `null_lo`/`null_hi`)
  and look at how much the bars overlap. A difference inside the null's range
  and close to zero is the useful result; a wide null with a large negative
  difference is not reassuring even when p > 0.05.
- If sig decodes better than nonsig, the anatomy can still use all electrodes,
  but say that the extra electrodes carry less of the contrast.

### 2.5 The bar plot

`plot_group_accuracy_bars` (`src/analysis/decoding/plots/group_bars.py`):

- one bar per group: mean accuracy at `n_match` electrodes;
- whiskers: the 2.5–97.5 percentiles over bootstraps × repeats. They show the
  procedure's spread, not a confidence interval for the group;
- dots: one per bootstrap (the mean over its CV repeats);
- a white dashed line across each bar: chance, from the shuffled-label decode
  of the same electrodes;
- brackets: each tested pair with its Holm-adjusted permutation p;
- x labels give each group's full size; the y label gives `n_match`, or "all
  electrodes per group" unmatched.

### 2.6 Results (2026-10-15 run)

*Drafted by Claude from the job logs and output tables; not yet reviewed.*

**What was run.** `submit_window_mean_groups_dcc.sh` with `ROIS=lpfc,dlpfc`
and the defaults (0–1.5 s window means, 20 bootstraps, 1000 membership
permutations, `n_match` = smallest group, seed 42; LDA, PCA to 90 % variance,
5 folds × 5 repeats). Two jobs, both COMPLETED:

| Job | Condition set | Contrast | Output folder (under `dcc_scripts/decoding/figs/<EPOCHS>/window_mean_groups/`) |
|---|---|---|---|
| 58726163 | `stimulus_congruency_conditions` | incongruent vs congruent | `stimulus_congruency_conditions_0to1.5s_20bts_1000perm_seed42/congruency/{lpfc,dlpfc}/` |
| 58726164 | `stimulus_switch_type_conditions` | switch vs repeat | `stimulus_switch_type_conditions_0to1.5s_20bts_1000perm_seed42/switchType/{lpfc,dlpfc}/` |

`<EPOCHS>` is `Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20`.
24 participants were listed; 22 contribute lPFC electrodes. Group sizes: lPFC
sig 171, nonsig 227, all 398 (`n_match` = 171); DLPFC sig 121, nonsig 156, all
277 (`n_match` = 121). Pseudo-trials per class after balancing: 105
(congruency; the incongruent class loses 31 % of its pseudo-trials to the
per-channel NaN filter) and 120 (switch type).

**Key numbers** (`summary.csv`, `tests.csv`; mean accuracy over 20 bootstraps
× 5 repeats, [2.5, 97.5] percentiles of the same; chance is the
shuffled-label decode of the same electrodes, 0.50–0.51 everywhere):

| Contrast | ROI | sig | nonsig | all | sig − nonsig | sig − all | nonsig − all |
|---|---|---|---|---|---|---|---|
| Congruency | lPFC | 0.859 [0.82, 0.90] | 0.667 [0.57, 0.75] | 0.786 [0.70, 0.86] | +0.192 (null ±0.08) | +0.074 (null −0.05/+0.04) | −0.119 (null ±0.04) |
| Congruency | DLPFC | 0.819 [0.76, 0.87] | 0.637 [0.56, 0.72] | 0.735 [0.68, 0.78] | +0.182 (null ±0.09) | +0.084 (null ±0.05) | −0.098 (null ±0.04) |
| Switch type | lPFC | 0.833 [0.78, 0.88] | 0.662 [0.55, 0.73] | 0.761 [0.69, 0.83] | +0.172 (null ±0.08) | +0.072 (null ±0.05) | −0.099 (null ±0.04) |
| Switch type | DLPFC | 0.812 [0.75, 0.87] | 0.576 [0.51, 0.64] | 0.704 [0.64, 0.78] | +0.236 (null ±0.09) | +0.108 (null ±0.05) | −0.128 (null ±0.05) |

Every one of the 12 pairwise differences has permutation p = 0.001 (the floor,
1/1001) or 0.004, Holm p ≤ 0.004. The "null" columns give the 2.5–97.5 % range
of the membership-permutation null for that difference.

**How to read the figure** (`group_bars.png` in each `<roi>/` folder). y axis:
decoding accuracy with `n_match` electrodes per group; x axis: the three groups,
with their full sizes. Bars are means, whiskers the 2.5–97.5 % spread over
bootstraps × repeats (the procedure's spread, not a CI for the population),
dots one per bootstrap, white dashes chance. Brackets give the Holm-adjusted
membership-permutation p: the null is "which group an electrode belongs to does
not matter", so a significant bracket means electrodes of the two groups carry
different amounts of the contrast at equal electrode count.

**Reading.**
- The answer to §2.1 is **no, not equally well**: in both ROIs and both
  contrasts, sig electrodes decode clearly better than nonsig at matched count
  (by 0.17–0.24 accuracy, two to three times the null's half-width), and
  `all` sits in between. The equivalence reading ("nonsig decodes at least as
  well as sig") is ruled out, not just unsupported.
- Nonsig electrodes still decode well above chance (0.58–0.67), so they are
  not empty: the "Reading it" case of §2.4 applies. All lPFC can stay the
  primary anatomy set, but the text should say that the electrodes added by not
  restricting to task-significant ones carry less of each base effect per
  electrode.
- The permutation conditions on these pseudo-trials and treats electrodes as
  the unit (§2.4). It does not say the gap would replicate in new participants;
  sig and nonsig electrodes are also not spread evenly over participants, which
  the test does not model.

**Caveats.**
- Window-mean features, base effects only (congruency, switch type). Nothing
  here speaks to LWPC or LWPS, which are the anatomy's main scores; a run with
  `CONDITIONS="stimulus_lwpc_block_balanced_conditions ..."` would.
- The 0.86 / 0.83 accuracies are pooled pseudo-population accuracies with 171
  electrodes; they are not comparable to single-participant decoding.
- The congruency decode lost 31 % of the incongruent pseudo-trials to NaN
  filtering before balancing; both classes were then subsampled to 105, so
  balance is intact, but the effective trial count is lower than for switch
  type.

---

## 3. The top 20 % electrodes by each score, on held-out trials

### 3.1 The question

Pick the electrodes with the largest LWPC, the largest LWPS, the largest
congruency effect and the largest switch effect, and follow each set through
decoding and power. For example: do the top LWPC electrodes carry LWPC but not
LWPS? Do the most switch-selective electrodes decode congruency? Selection and
measurement use disjoint trials, so a set's effect on its own score is not
inflated by having been picked for it.

### 3.2 How to run it

**Step 1: select (minutes).**

```bash
cd dcc_scripts/decoding
mkdir -p out
sbatch sbatch_score_electrode_selection_dcc.sh --rois lpfc,dlpfc                    # top 20 %, positive
sbatch sbatch_score_electrode_selection_dcc.sh --rois lpfc,dlpfc --sign negative    # most negative 20 %
sbatch sbatch_score_electrode_selection_dcc.sh --rois lpfc,dlpfc --sign absolute    # largest |score|
```

Use `--rois` to cover every ROI the later jobs use (§0): `lpfc` for decoding
with `ROIS=lpfc`, and `dlpfc` for power traces as shipped. Each ROI is ranked on
its own.

| Option | Default | What it does |
|---|---|---|
| `--rois` | `lpfc` | ROIs from `src/analysis/config/rois.py`, comma separated |
| `--electrodes` | `all` | Candidates: every ROI electrode, or only the task-significant ones |
| `--frac-select` | 0.3 | Share of each participant's correct trials used to select |
| `--top-frac` | 0.2 | Share of each ROI's electrodes kept per score |
| `--sign` | `positive` | `positive`: the largest scores; `negative`: the most negative; `absolute`: the largest magnitude |
| `--tmin` / `--tmax` | 0 / 1.5 s | The window the scores are computed on |
| `--seed` | 0 | Seeds the trial split |

It writes to
`figs/<EPOCHS>/score_selections/<rois>_<electrodes>_sel<frac>_top<frac>_<sign>_<tmin>to<tmax>s_seed<seed>/`
(for example `lpfc-dlpfc_all_sel0.3_top0.2_positive_0to1.5s_seed0/`):

- `score_selection.json`: the sets, the trial split (the `trial_count` ids on
  each side for every participant) and the parameters;
- `selection_scores.csv`: every candidate's four scores and its set memberships.

The log prints each set's electrode and participant counts. Read them before
step 2.

**Step 2: decode the other 70 %.**

```bash
cd dcc_scripts/decoding
SEL=$PWD/figs/<EPOCHS>/score_selections/<the folder from step 1>/score_selection.json
ROIS=lpfc ELECTRODE_SETS_JSON=$SEL bash submit_decoding_with_score_electrode_sets_dcc.sh
ROIS=lpfc ELECTRODE_SETS_JSON=$SEL ELECTRODE_SETS=lwpc_only,lwps_only \
    CONDITIONS="stimulus_lwpc_block_balanced_conditions stimulus_lwps_block_balanced_conditions" \
    bash submit_decoding_with_score_electrode_sets_dcc.sh
```

One job per condition set (default: congruency, switch type, and the
block-balanced LWPC and LWPS sets: four jobs). Each job runs the whole decoding
battery once per electrode set, into
`figs/<EPOCHS>/score_selections/<selection>/<CONDITION>/elecset_<set>/`. File
names get `_scoresel-top0.2positive`. `ELECTRODE_SETS` limits the sets; the
default is all of them (10 when all four scores are present, §3.4), so a full
call is 4 jobs × 10 battery runs.

**Step 3: power traces on the same 70 %.** One set per call; each call submits
the four block-balanced condition sets hard-coded in the submitter:

```bash
cd ../power
for SET in lwpc lwps congruency switch_type; do
  ELECTRODE_SETS_JSON=$SEL ELECTRODE_SET=$SET ELECTRODES=all \
      bash submit_specific_conditions_power_traces_dcc.sh
done
```

That loop is 16 jobs. Give `ELECTRODE_SETS_JSON` as an **absolute** path: this
submitter passes it on unchanged, and the job resolves a relative path against
`dcc_scripts/power/`. `ELECTRODES=all` only sets the `all_elecs`/`sig_elecs`
label in the file names (the submitter's default is `sig`); the electrodes come
from the set. Output:
`figs/<EPOCHS>/anova_within_roi/score_selections/<selection>/<set>/<condition>/`.

### 3.3 The code, step by step

**Selection** (`dcc_scripts/decoding/score_electrode_selection_dcc.py:77`):

1. **Candidates.** Every electrode of each ROI (`--electrodes all`) or the
   task-significant ones, from the electrode-to-ROI map and the sig-channel
   files.
2. **The split** (`collect_subject_trial_strata` and `assign_trial_partitions`,
   `src/analysis/decoding/anova_electrode_selection.py:156`). For each
   participant, its correct trials are split 30/70, stratified on
   `congruency`, `task_sequence` and `block_type` (`DEFAULT_STRATA_COLS`), and
   keyed on `trial_count`. Each participant's seed is
   `seed + crc32(subject) % 100000`.
3. **Scores on the 30 % only.** `assemble_long_df` builds the window-mean HG
   table for the selection trials; `naive_sensitivities(contrast_mode='proportion',
   main_effects=True)` (`src/analysis/stats/stability_flexibility_segregation.py:1197`)
   scores every electrode (§3.4).
4. **Sets** (`score_electrode_sets`,
   `src/analysis/decoding/score_electrode_selection.py:49`). Within each ROI, with
   the electrodes of all participants pooled, `top_fraction` (`:34`) marks each
   score's top fraction. Pairs of sets get their unique and shared parts.
5. **Save** (`save_score_selection`, `:84`): the sets, the split and the
   parameters in one JSON, so every later job uses the same split.

**Measurement** (`held_out_electrode_sets`, `score_electrode_selection.py:106`),
called by `decoding_dcc.main` (`decoding_dcc.py:493`) and by
`power_traces_dcc.main` (`dcc_scripts/power/power_traces_dcc.py:188`):

1. Load the file. Refuse a set name the file lacks, a job ROI the file did not
   select in, or a participant with no split.
2. `apply_trial_partition(..., which='decode')`: keep only the 70 % of trials
   in every participant's epochs.
3. Keep each set's electrodes that are in the loaded epochs.

The decoding job then runs the battery once per set
(`run_decoding_for_one_electrode_set`); the power job measures its one set.

### 3.4 The statistics, and why it is set up this way

**The four scores.** Each is a Cohen's d on window-mean HG (0–1.5 s), computed
per electrode on the selection trials from the four cells of a 2 × 2 design
(`_interaction_effect`, with equal weight on each cell). For congruency the
cells are congruency × the block's incongruent proportion; for switch type they
are switch type × the block's switch proportion. With m the cell means and
s the pooled within-cell SD:

| Score | Contrast over the four cell means, ÷ s | Positive means |
|---|---|---|
| `lwpc` | (m(I, low) − m(C, low)) − (m(I, high) − m(C, high)) | The congruency effect is larger in the low-incongruent block: the behavioural adaptation direction |
| `lwps` | (m(S, low) − m(R, low)) − (m(S, high) − m(R, high)) | The switch effect is larger in the low-switch block |
| `congruency` | ½ [(m(I, low) − m(C, low)) + (m(I, high) − m(C, high))] | Incongruent > congruent, averaged over the two blocks |
| `switch_type` | ½ [(m(S, low) − m(R, low)) + (m(S, high) − m(R, high))] | Switch > repeat, averaged over the two blocks |

The main effects are averages of within-block differences, not pooled
incongruent-vs-congruent comparisons. Pooling would weight each block by its
trial count, and since incongruent trials are mostly in the high-incongruent
block, a block-level offset would leak into the congruency score
(`BALANCE_MAIN_EFFECTS` in the segregation module explains this).

**Why held-out trials.** If the same trials chose the electrodes and measured
them, the top LWPC electrodes would show a large LWPC partly because their trial
noise happened to point that way, and they would show it on any contrast that
shares those trials. With disjoint trials the noise that picked an electrode is
independent of the noise it is measured with, so the 70 % estimate is unbiased.
Expect it to be smaller than the 30 % score: that is regression to the mean, and
it is what the split is for.

**Why the top fraction, and the ranking.**

- `top_fraction` keeps ceil(0.2 × n) of an ROI's n electrodes with a finite
  score, so the four main sets have the same size and their decoding accuracies
  can be compared without a size confound. Ties at the cut go to the
  electrode listed first.
- Electrodes are ranked pooled over participants, on raw d. Unlike the A3
  scores, responsiveness and participant offsets are not removed first, so
  participants with a larger signal can contribute more electrodes. Check the
  per-set participant counts in the log.
- Each score is ranked on its own: the top 20 % of LWPC and the top 20 % of LWPS
  are separate selections that may overlap.

**The ten sets.**

| Set | Electrodes |
|---|---|
| `lwpc`, `lwps`, `congruency`, `switch_type` | Each score's top 20 % |
| `lwpc_only`, `lwps_only` | In one of the two adaptation sets but not the other |
| `adaptation_overlap` | In both `lwpc` and `lwps` |
| `congruency_only`, `switch_type_only` | In one of the two main-effect sets but not the other |
| `main_effects_overlap` | In both `congruency` and `switch_type` |

If the two scores were independent, about 0.2 × 0.2 = 4 % of an ROI's
electrodes would be in an `*_overlap` set by chance. The observed overlap is
larger when the scores correlate across electrodes (LWPC–LWPS r ≈ 0.10,
congruency–switch r ≈ 0.23 in lPFC). The `*_only` and `*_overlap` sets are not
matched in size, so don't compare their accuracies directly; compare them with
the matched main sets in mind.

**Reading it.**

- The claim "the top LWPC electrodes carry LWPC but not LWPS" is a comparison:
  LWPC in the `lwpc` set against LWPS in the `lwpc` set, and against LWPC in the
  `lwps` set. One effect significant and the other not is not a test of their
  difference (Nieuwenhuis et al. 2011). The pipeline tests each set and contrast
  on its own and has no test across sets yet. Until it does, report each
  effect's size and interval side by side.
- Only 30 % of the trials pick the electrodes, and single-electrode scores are
  noisy (split-half reliability ~0.3 in lPFC, N4 §19), so part of each top set
  is chosen by noise. That weakens the contrast between sets; it does not bias
  it.
- The power traces' windowed ANOVA permutes observations as independent though
  every electrode contributes to both conditions
  ([concurrent-regulation plan §9.4](analysis_plans.md#94-the-power-traces-run-an-independent-permutation-on-a-paired-design)).
  That is conservative: it can miss an effect but not invent one.

**Fixed alongside (2026-10-13).** `assign_trial_partitions` used to seed each
participant with `hash(subject)`. Python salts string hashes per process, so the
same `SEED` gave a different 30/70 split in every job, including the ANOVA
electrode-selection route ([`decoding.md`](decoding.md) §3.3). It now uses
`zlib.crc32`, and `test_trial_split_does_not_depend_on_the_python_hash_seed`
checks it. ANOVA-route runs made before the fix cannot be reproduced exactly.

### 3.5 Results (2026-10-15 run): step 1, the selections

*Drafted by Claude from the job logs and `selection_scores.csv`; not yet
reviewed. Steps 2 and 3 have not finished: the decoding jobs on these sets
(`dec_scoresel_*`, 58739251–54) were still pending at the time of writing, and
no power-trace job on them had been submitted.*

**What was run.** `sbatch_score_electrode_selection_dcc.sh --rois lpfc,dlpfc`
with `--electrodes all --frac-select 0.3 --top-frac 0.2 --tmin 0 --tmax 1.5
--seed 0`, once per sign. All three COMPLETED in about 10 min:

| Job | `--sign` | Folder (under `dcc_scripts/decoding/figs/<EPOCHS>/score_selections/`) |
|---|---|---|
| 58726651 | positive | `lpfc-dlpfc_all_sel0.3_top0.2_positive_0to1.5s_seed0/` |
| 58726655 | negative | `lpfc-dlpfc_all_sel0.3_top0.2_negative_0to1.5s_seed0/` |
| 58726659 | absolute | `lpfc-dlpfc_all_sel0.3_top0.2_absolute_0to1.5s_seed0/` |

Candidates: 398 lPFC electrodes (22 participants) and 277 DLPFC electrodes (21
participants; every DLPFC electrode is also an lPFC electrode). Selection used
86–132 correct trials per participant, measurement keeps 197–307. The scores
are identical across the three files (same split, seed 0); only the set
membership differs.

**Set sizes** (electrodes / participants). Each main set is ceil(0.2 × n): 80
in lPFC, 56 in DLPFC.

| Set | lPFC positive | lPFC negative | lPFC absolute | DLPFC positive | DLPFC negative | DLPFC absolute |
|---|---|---|---|---|---|---|
| `lwpc` | 80 / 19 | 80 / 14 | 80 / 18 | 56 / 11 | 56 / 12 | 56 / 13 |
| `lwps` | 80 / 17 | 80 / 15 | 80 / 18 | 56 / 14 | 56 / 13 | 56 / 15 |
| `congruency` | 80 / 14 | 80 / 19 | 80 / 14 | 56 / 11 | 56 / 14 | 56 / 11 |
| `switch_type` | 80 / 18 | 80 / 18 | 80 / 20 | 56 / 16 | 56 / 13 | 56 / 15 |
| `adaptation_overlap` | 13 / 3 | 18 / 7 | 13 / 6 | 11 / 4 | 8 / 4 | 11 / 6 |
| `main_effects_overlap` | 23 / 9 | 19 / 11 | 15 / 8 | 18 / 8 | 14 / 8 | 10 / 5 |
| `lwpc_only` = `lwps_only` | 67 | 62 | 67 | 45 | 48 | 45 |
| `congruency_only` = `switch_type_only` | 57 | 61 | 65 | 38 | 42 | 46 |

**Checks on the selection.**
- **Participant concentration.** One participant supplies 14–41 % of a set
  (largest: DLPFC negative `lwps`, 23 of 56 electrodes from D0133; lPFC
  positive `congruency`, 20 of 80 from D0134). This is what ranking raw d
  pooled over participants does (§3.4), and it means a set's decoding result
  can rest on two or three participants. Report the per-set participant counts
  with the step-2 results.
- **Overlap.** Positive `adaptation_overlap` is 13 / 80 in lPFC (16 %) and
  11 / 56 in DLPFC (20 %), against ~20 % expected by chance for two independent
  top-20 % sets of the same electrodes (0.2 × 80 = 16). On these 30 % trials
  the LWPC–LWPS correlation across electrodes is r = 0.05 (lPFC) and 0.00
  (DLPFC), lower than the full-data r ≈ 0.10 quoted in §3.4.
- **The adaptation scores are mostly noise at this trial count.** Across lPFC
  electrodes the LWPC and LWPS scores have SD 0.48 and 0.47, and mean 0.07 and
  0.04. With ~115 selection trials over the four cells of each 2 × 2, an
  interaction d has a sampling SD of at least √(4/29) ≈ 0.37 from noise alone,
  and about 0.43 with 25 % / 75 % cells (about 14 and 43 trials). So the
  observed spread is close to what noise alone would give: very roughly a fifth
  of the LWPC variance is signal. The main-effect scores (SD 0.27 and 0.25,
  against a noise floor of about 0.22) have a larger signal share, roughly a
  third. These are back-of-envelope figures, not computed from the actual cell
  counts; they agree in order with the full-data shared-split reliabilities
  of N4 §19.8 (LWPC 0.21, LWPS 0.04, congruency 0.46, switch 0.35).
  Expect strong regression to the mean on the 70 % for `lwpc`/`lwps`, and
  little separation between `lwpc_only` and `lwps_only`. To settle how much
  signal the sets carry before step 2 is read: correlate each score on the 30 %
  with the same score on the 70 % (the held-out reliability).
- No electrode is in both the positive and the negative set of any score, and
  the absolute sets are the union of the most extreme of each (e.g. lPFC
  `lwpc`: 49 from the positive tail, 31 from the negative).

---

## 4. Figure 5: the brain, the scatter, and every electrode along z

### 4.1 What changed

| Panel | Before (2026-10-06) | Now (2026-10-10) |
|---|---|---|
| a | Sagittal projection of the electrodes (MNI y against z), coloured by height tertile; the brain was a separate file | **The rendered brain** (`fig5_height_brain.png`, white margins trimmed) across the top row, with the tertile cuts in the title. The sagittal projection is now only the fallback when no render exists |
| b | LWPC against LWPS, coloured by tertile, with a box marking c | LWPC against LWPS with the tertile centroids, and beside it a zoom on the centroids with their 95 % bootstrap ellipses, tied to the scatter by connector lines (old b and c merged) |
| c | The centroids enlarged | **Every electrode's LWPC − LWPS against its MNI z**, with participant, responsiveness and MNI x and y removed, and the coordinate test's z slope as the line (§4.4) |
| d | The tertile means ± SEM of the adjusted balance | Removed from the figure. The numbers stay in `fig5_height_balance.csv` |

The layout is now 7.2 × 5.0 in, two rows. `paper_draft.md`'s F5 table describes
the same panels (it calls the cross-decoding panel d).

### 4.2 How to run it

**On a new anatomy run** (Figure 5 is part of §19, which runs when `FOLLOWUPS`
contains 19, the default):

```bash
cd dcc_scripts/stats
ARM=continuous ROI_FILTER=lpfc ANAT_LEVEL=destrieux N_PERM=10000 MAKE_BRAIN=1 BRAIN_HEMI=split \
    bash submit_stability_flexibility_anatomy_dcc.sh
```

- `ELECTRODES` and `SEG_RUN` are plain assignments near the top of the
  submitter, not environment variables. The current Figure 5 uses all lPFC
  (397 electrodes), so set `ELECTRODES=all` there (line 39) and check that
  `SEG_RUN` points at the `_main_effects` segregation run you want.
- With `CONTRAST_MODE=condition` (the submitter's current default) and scores
  from a `MAIN_EFFECTS=1` segregation run, the job also draws the same figure on
  congruency against switch type: `fig5_height_main_effects*`. With
  `CONTRAST_MODE=proportion` it draws only LWPC/LWPS.
- The batch script runs under `xvfb-run`, which the brain render needs.

**From an existing anatomy run** (no epochs, atlases or recon files except for
the brain):

```bash
xvfb-run -a python dcc_scripts/stats/n4_section19_followups.py \
    --anatomy-dir <anatomy run>/continuous \
    --seg-dir     <the _main_effects segregation run it read> \
    --sections 1,4 --brain --brain-hemi split
```

Run this in a Slurm job, not on a login node. Keep section 1 with section 4: it
supplies the electrode-bootstrap interval printed in `fig5_height.png` and the
participant-level p printed in `fig5_height_participants.png`. The script draws
only the LWPC/LWPS pair. Without `--brain`, panel a reuses a
`fig5_height_brain.png` already in the output folder, or falls back to the
sagittal projection.

**Outputs** (in `continuous/`, or `<anatomy run>/continuous/section19/` for the
script):

| File | What |
|---|---|
| `fig5_height.png`/`.pdf` | The figure, electrodes as the unit (main text) |
| `fig5_height_participants.png`/`.pdf` | The same with participant-bootstrap ellipses (supplement) |
| `fig5_height_points.csv` | Panel b's points |
| `fig5_height_centroids.csv` | Each tertile's centroid, its balance with a 95 % interval, and the bootstrap covariance for the ellipse |
| `fig5_height_partials.csv` | Panel c's points: `subject`, `electrode`, `mni_z`, `band`, `partial` |
| `fig5_height_balance.csv` | The tertile means ± SEM that the old panel d drew |
| `fig5_height_brain.png`, `_legend.png`, `fig5_height_brain_centroids.csv` | Panel a and the centroid positions (N4 §19.4) |
| `summary_section19.txt` | The numbers; the line `c  line through the N electrodes: ...` checks panel c's slope against the test's |

### 4.3 The code, step by step

`figure5_height` (`src/analysis/stats/stability_flexibility_anatomy.py:2788`),
called by `section19` (`:3601`) once per pair and level:

1. `figure5_height_points` (`:2236`): panel b's points. LWPC and LWPS with
   responsiveness regressed out and participant-centred (`prepare_continuous`),
   on `delta`'s pooled scale with each score's mean added back, so the identity
   line means LWPC = LWPS. Tertile edges of MNI z over every electrode with
   coordinates.
2. `_figure5_test`: panel b's r, the pre-specified separate-half LWPC–LWPS
   correlation, from the segregation run's `correlation.json`.
3. `height_centroids` (`:2269`): each tertile's centroid and a bootstrap (2000
   draws; electrodes within participant for `unit='electrode'`, whole
   participants for `'participant'`) for the 95 % ellipse and the balance
   interval.
4. `height_partial_residuals` (`:2357`): panel c's points (§4.4).
5. `relative_score_coordinate_test` (`:1239`), unless the anatomy job passes its
   result in: the slope and p printed in panel c.
6. **The brain first, then the figure.** `plot_height_bands_on_brain` (`:2705`)
   renders `fig5_height_brain.png` when `make_brain` is on. The render now
   happens before the figure is drawn, so panel a can use it. Without a new
   render, panel a looks for one from an earlier run in the same folder.
7. `plot_figure5_height` (`:2432`) draws the panels; panel a goes through
   `trim_whitespace` (`src/analysis/utils/image_utils.py:6`), which crops the
   render to the box around its non-white, non-transparent pixels plus 4 px.

### 4.4 The statistics behind panel c

**The test.** The gradient is the pre-specified coordinate test
(`relative_score_coordinate_test`):

```
delta ~ mni_y + mni_z + mni_x + responsiveness + participant
```

over all electrodes with coordinates, where `delta = LWPC_s − LWPS_s` (each
score divided by its SD across all electrodes, `attach_scores`), and participant
is a fixed effect (dummies). The z slope's p comes from the within-electrode
swap null: under "LWPC and LWPS are exchangeable at each electrode", swapping
them flips the sign of delta, so the null is built from sign flips of delta,
refitted each time (10,000 permutations). The interval in the figure is the
electrode bootstrap of `coordinate_slope_by_electrode`. On the all-lPFC run:
z slope −0.0077 SD/mm, 95 % CI [−0.0124, −0.0032], p = 0.0074
([`paper_draft.md`](paper_draft.md), F5).

**The points.** Plotting raw delta against z would mix the z effect with
everything else the model removes (participant offsets, responsiveness, the y
and x gradients), and a line fitted through them would not be the test. Panel c
plots each electrode's **component plus residual** instead:

```
partial_i = residual_i + β_z (z_i − mean z) + mean(delta)
```

where β_z and the residuals come from the fit above. The residuals of an OLS fit
are orthogonal to every column of its design, including the centred z, so a
straight line fitted through (z_i, partial_i) has slope exactly β_z
(Frisch–Waugh–Lovell). **The line in panel c is the test's slope, not a separate
fit**, and the vertical scatter around it is the scatter the test sees. The
summary line `c  line through the N electrodes` checks this and flags any
difference, which would mean the coordinate test was fitted on other electrodes.
`test_height_partial_residuals_carry_the_coordinate_tests_slope` checks it on
synthetic data.

**What it shows and what it doesn't.** The slope uses every electrode, so no
choice of bins affects it. The tertile colours and the dashed cut lines are for
display; the test does not use them. Points below zero lean LWPS, above zero
LWPC. The y axis is in SD units of the pooled scores, the x axis in mm. At about
2 % of variance explained, the points scatter widely around the line; panel c
shows that honestly, which the old tertile means ± SEM did not.

**Why the brain moved to panel a.** The advisors read the anatomy on the brain,
not on an abstract y–z projection. The brain shows the same tertiles in the
same colours, with each tertile's centroid per hemisphere (N4 §19.4 has the
centroid conventions).

---

## 5. The state-space / subspace figure: partly built

The plan is the [subspace trajectory plan](analysis_plans.md#subspace-trajectory-plan)
(2026-10-13). **Update, 2026-10-15:** its §3.1-3.3 now exist: the coding axes
per A4 window, the coding cosine over time against a shuffle null and a
split-half ceiling, and the `remove_mean`, RT and occipital controls. How to
run and read them is in [`coding_axes.md`](coding_axes.md). Still not built:
the principal angles and alignment index (the subspace test, §3.4), the
held-out four-cell trajectories (§4) and dPCA.

What existed when this doc was written (2026-10-14):

- **`map_reliability(..., method='cosine')`**
  (`src/analysis/stats/stability_flexibility_anatomy.py:1276`, added
  2026-10-14). It compares the LWPC and LWPS maps by the uncentred cosine
  instead of a correlation, the plan's §3.1. A correlation subtracts each map's
  mean; the cosine keeps it, so it also asks whether the two effects move
  activity in the same direction overall. Its `between_noise_corrected` is built
  from the cosines themselves, which is the plan's alignment (§3.2), and the
  `*_pearson` keys still report Pearson. This is a map-level number on the
  per-split scores, not the subspace test.
- **`dcc_scripts/subspace/subspace_notebook.ipynb`**: a scratch notebook that
  steps through `map_reliability` on one segregation run's `per_split.csv`
  (the path is hard-coded to one user's DCC folder).
- **§1's channel-space trajectories**, which are descriptive and not
  cross-validated. They are not the plan's trajectories (§4 of the plan fits the
  axes on half A and projects half B).

---

## 6. Tests

Every analysis above has tests on synthetic data (no real data needed):

| File | What it checks |
|---|---|
| `tests/analysis/decoding/test_decoder_feature_mode.py` | Channel mode decodes a spatial pattern, is at chance on noise, works without windows, and leaves the default mode unchanged |
| `tests/analysis/decoding/test_channel_trajectories.py` | PC1 is the shared response unless the condition mean is removed; shapes and padding; the figure is written |
| `tests/analysis/decoding/test_window_mean_decoding.py` | Window means; matched group sizes and a planted difference found; no difference when membership does not matter; the nested null keeps the superset whole; unmatched groups decoded whole, the unmatched null centred on the size difference, the DCC run's `_unmatched` output |
| `tests/analysis/decoding/test_score_electrode_selection.py` | `top_fraction` signs and rounding; the sets and pairs; held-out sets keep only decode trials; the split does not depend on Python's hash seed; `naive_sensitivities(main_effects=True)` |
| `tests/analysis/utils/test_image_utils.py` | `trim_whitespace` keeps the ink and its padding |
| `tests/analysis/stats/test_section19_anatomy.py` | Includes `test_height_partial_residuals_carry_the_coordinate_tests_slope` |

```bash
pytest tests/analysis/decoding/test_decoder_feature_mode.py \
       tests/analysis/decoding/test_channel_trajectories.py \
       tests/analysis/decoding/test_window_mean_decoding.py \
       tests/analysis/decoding/test_score_electrode_selection.py \
       tests/analysis/utils/test_image_utils.py \
       tests/analysis/stats/test_section19_anatomy.py
```

All 58 passed on 2026-10-15 (about 3 minutes).

## References

- Nieuwenhuis, S., Forstmann, B. U. & Wagenmakers, E.-J. (2011). Erroneous
  analyses of interactions in neuroscience: a problem of significance. *Nature
  Neuroscience* 14, 1105–1107.
