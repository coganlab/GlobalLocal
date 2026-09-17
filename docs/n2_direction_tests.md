# N2 — direction tests on the adaptation effects

**What this document is.** A standalone walkthrough of the analysis that answers
beat **N2** in [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md)
§2: *which way do the LWPC and LWPS effects go in lPFC high gamma?* It covers the
data flow end to end, every parameter that changes the answer, the exact commands
to run it, and how to read what comes back.

It assumes nothing about what you remember. If you only read one section, read
[§2 The sign convention](#2-the-sign-convention-read-this-first) — every other
number in this document is meaningless without it.

---

## 0. Where the code actually lives

**The N2 direction tests are not in `stability_flexibility_segregation_dcc.py` or
`stability_flexibility_anatomy_dcc.py`.** Those two scripts are the **N4 /
anatomy** pipeline — per-electrode continuous scores mapped onto cortex. They are
a *different beat of the paper* and they answer a different question ("are the
two effects organized differently across cortex?").

N2 lives here:

| Role | File |
|---|---|
| **The test itself** | `dcc_scripts/power/power_traces_dcc.py` — the `statistical_method == 'time_perm_cluster_interaction'` branch |
| **The per-test figures** | `src/analysis/power/plots.py` — `plot_direction_test_traces` |
| **The knobs** | `dcc_scripts/power/run_power_traces_dcc.py` — plain Python constants at the top |
| **Cluster wrapper** | `dcc_scripts/power/sbatch_power_traces_dcc.sh` |
| **Job submitter** | `dcc_scripts/power/submit_specific_conditions_power_traces_dcc.sh` |
| **Which conditions get contrasted** | `src/analysis/config/condition_registry.py:99` (LWPC), `:153` (LWPS) — the `subtraction_pairs` key |
| **Evoked assembly + the cluster test** | `src/analysis/power/evoked_builders.py` (`:133`, `:325`, `:358`, `:386`) |

The confusion is understandable: the segregation module computes per-electrode
LWPC/LWPS scores on **the same sign convention**, so it is a second read on the
same direction question. That relationship is spelled out in
[§8](#8-the-second-read-the-two-files-you-thought-this-was-in).

---

## 1. What N2 asks and why it exists

The power traces run a windowed ANOVA. `anova_lm` reports **F**, and for a 1-df
term `F = t²`. So the ANOVA establishes that a congruency × incongruent-proportion
interaction **exists** in lPFC, but it cannot say **which way it goes** — the sign
is squared away.

"Concurrent regulation" is a claim about direction. A list-wide manipulation that
recruits more control should **shrink** the effect it acts on:

- more incongruent trials in a block → **smaller** congruency effect (LWPC)
- more switch trials in a block → **smaller** switch cost (LWPS)

N2 recovers that sign. Per effect it runs **three** cluster-corrected tests:

```
LWPC:  simple effect in the low block    (i − c | 25% incongruent)
       simple effect in the high block   (i − c | 75% incongruent)
       the interaction                   (i − c | 25%) − (i − c | 75%)

LWPS:  simple effect in the low block    (s − r | 25% switch)
       simple effect in the high block   (s − r | 75% switch)
       the interaction                   (s − r | 25%) − (s − r | 75%)
```

**It is a kill switch, not a headline result.** Per the plan: if the neural
directions disagree with the behavioral ones, stop and re-read the epoch metadata
before running anything in N3–N4. It costs an afternoon and protects the rest of
the analysis.

---

## 2. The sign convention (read this first)

Everything is oriented **LOW proportion minus HIGH proportion**:

> ### A **positive** effect means the condition effect **SHRINKS** in the high-proportion block — the predicted adaptation direction.

This is pinned in code, not by convention-in-your-head. `power_traces_dcc.py:263-270`
selects the subtraction pairs **by name** rather than by list order:

```python
low_pair  = next(p for p in usable if '25' in p[0])   # ('Stimulus_i25', 'Stimulus_c25')
high_pair = next(p for p in usable if '75' in p[0])   # ('Stimulus_i75', 'Stimulus_c75')
```

The registry happens to list the 75% pair first, so relying on order would have
silently flipped the sign. It doesn't.

### The two places that are deliberately the opposite sign

Both are documented where they are defined, and **neither is a bug** — but if you
compare numbers across them without converting, you will read the result backwards.

| Where | Orientation | Why it differs |
|---|---|---|
| `power_traces_dcc.py` interaction follow-up (**this analysis**) | **low − high** | positive = adaptation |
| `stability_flexibility_segregation.W_INTERACTION` (`:519`) | **low − high** | same convention — agrees with N2 |
| `windowed_anova._signed_contrast_per_window` (`:37`) | **high − low** | orders factor levels *alphabetically*; only used to split clusters at sign flips and colour pos/neg bars, neither of which depends on absolute orientation |
| `power_traces_dcc.py:531` `signed_contrast` in the saved ANOVA npz | **high − low** | inherits the above. The inline comment says "Flip it when reading." |
| cross-decoding `block_difference` | high − low on *accuracy* | not a condition effect; has no adaptation direction |

The convention is pinned by
`tests/analysis/stats/test_effect_sign_conventions.py`, which plants an effect
with a known direction and asserts the sign that comes back — on the neural
scores *and* the behavioral difference-of-differences. Run it if you ever doubt
which way is up:

```bash
pytest tests/analysis/stats/test_effect_sign_conventions.py -v
```

---

## 3. Data flow

```
BIDS epochs on disk
  EPOCHS_ROOT_FILE, e.g.
  Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_..._filterbank_hilbert_stat_func_ttest_zmax_20
        │
        │  utils.create_subjects_mne_objects_dict(..., just_HG_ev1_rescaled=True,
        │                                          acc_trials_only=ACC_TRIALS_ONLY)
        ▼
  subjects_mne_objects[subject][condition]['HG_ev1_power_rescaled_avg']
  one TRIAL-AVERAGED evoked per (subject, condition)      <- trials collapse HERE
        │
        │  ELECTRODE SELECTION, three filters in order:
        │    1. ROI membership            ROIS_DICT (Destrieux labels -> 'lpfc'/'occ')
        │    2. ELECTRODES = all | sig    sig = passed the baseline-vs-signal filter
        │    3. ANOVA_LABELS_CSV          optional: a saved ANOVA subpopulation
        │       EXCLUDE_ELECTRODES        optional: named electrodes dropped outright
        ▼
  make_multi_channel_evokeds_for_all_conditions_and_rois()
        │  evoked_builders.py:133 loops subjects, extracts one single-channel
        │  evoked per electrode, and `.extend`s them into one flat list
        │
        │  >>> SUBJECT IDENTITY IS DISCARDED AT THIS LINE <<<
        │
        │  combine_single_channel_evokeds() stacks them
        ▼
  evks_dict_elecs[condition][roi]  ->  (n_electrodes, n_times)
        │                               e.g. (~174, ~640) for lpfc
        │
        ├─── create_subtracted_evokeds_dict()   evoked_builders.py:358
        │      d_low  = Stimulus_i25 − Stimulus_c25     per electrode
        │      d_high = Stimulus_i75 − Stimulus_c75     per electrode
        ▼
  THREE CALLS to time_perm_cluster_between_two_evokeds()   evoked_builders.py:386
        │   -> ieeg.calc.stats.time_perm_cluster(axis=0, permutation_type='samples')
        │
        │   test 1  simple_low     : i25   vs c25
        │   test 2  simple_high    : i75   vs c75
        │   test 3  interaction    : d_low vs d_high
        ▼
  (mask over time, cluster p-values)  +  a printed mean signed delta
        │
        ├──> stdout / slurm .out             the printed deltas (see §6)
        ├──> n2_direction_tests/<roi>/...    one figure PER TEST, each with its
        │                                    own cluster bar
        ├──> <roi>/..._n2_direction_<test>_cluster.npz   mask, cluster p-values,
        │                                    signed delta
        └──> the interaction mask is also drawn on the subtraction figure
```

### What the unit of inference is, and what that costs

**The unit is the electrode, pooled across subjects. There is no subject term in
the null.** The permutation exchanges observations along `axis=0`, which is the
electrode axis.

This is deliberate and it matches the power traces the test exists to
interrogate — the same `create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition`
call flattens subjects there too. Holding the kill switch to a stricter null than
the traces it validates would let it fail a direction those traces already
reported.

The cost, stated plainly: electrodes within a subject are correlated, so the
effective N sits below the electrode count by roughly the design effect
`1 + (m̄ − 1)·ICC`. At ~174 lPFC electrodes across 12 subjects (m̄ ≈ 14.5), an ICC
of 0.1 puts effective N near 74; an ICC of 0.3 near 34.

> **The consequence is an optimistically small p-value, not a wrong sign.** N2's
> output is a *direction*, cross-checked against behavior and against each simple
> effect's own sign, so an inflated p changes nothing about the verdict.

This tolerance is specific to N2, the traces and the decoding. It is **not**
acceptable on the LWPC–LWPS correlation (plan §5.1), where between-subject SNR
offsets can reverse the sign and manufacture the effect outright.

### Why the test is paired

Every electrode contributes an evoked to **both** sides of every contrast, so the
design is paired. Under `permutation_type='samples'`, scipy randomly swaps the
paired observations — which for a difference statistic is exactly a **sign-flip
permutation on the per-electrode difference**, the test plan §2 specifies.

Pairing cancels between-electrode variance in overall HG amplitude, which is
enormous. Running `'independent'` here would pool and re-partition, carrying that
variance into the null and **losing sensitivity**. That is conservative rather
than wrong, but there is no reason to accept it: the interaction branch forces
`ttest_rel` / `'samples'` automatically (`run_power_traces_dcc.py:97-98`).

`time_perm_cluster_between_two_evokeds` raises a `ValueError` if the two evokeds
carry different channels or the same channels in a different order, because
`'samples'` pairs by **index** and would otherwise silently pair unrelated
electrodes. That guard is at `evoked_builders.py:399-403`.

---

## 4. Parameters

### 4.1 The ones that decide whether N2 runs at all

| Parameter | Where | Required value for N2 | Notes |
|---|---|---|---|
| `STATISTICAL_METHOD` | `run_power_traces_dcc.py:72` | `'time_perm_cluster_interaction'` | **Ships as `'anova'`.** You must edit this line. There is no env var for it. |
| `CONDITION_LABEL` | env var, via `--export` | `stimulus_lwpc_conditions` or `stimulus_lwps_conditions` | **Only these two work.** See §4.2. |
| `STAT_FUNC_CHOICE` | `run_power_traces_dcc.py:96-98` | `'ttest_rel'` | Set **automatically** when the method is the interaction. Don't override it. |
| `EPOCHS_ROOT_FILE` | env var | the epochs dir name | No default; raises if unset. |
| `ANOVA_UNIT` | env var | `'roi'` or `'electrode'` | Raises if unset, but the interaction path **ignores its value** — it only lands in the output path name. Pass `roi`. |

### 4.2 Which `CONDITION_LABEL` values actually work

The interaction follow-up needs `subtraction_pairs` in the registry. Verified by
resolving the registry directly:

| `CONDITION_LABEL` | `subtraction_pairs` | Runs N2? |
|---|---|---|
| `stimulus_lwpc_conditions` | `[('Stimulus_i75','Stimulus_c75'), ('Stimulus_i25','Stimulus_c25')]` | ✅ |
| `stimulus_lwps_conditions` | `[('Stimulus_s75','Stimulus_r75'), ('Stimulus_s25','Stimulus_r25')]` | ✅ |
| `stimulus_lwpc_block_balanced_conditions` | `[]` | ❌ `ValueError` |
| `stimulus_lwps_block_balanced_conditions` | `[]` | ❌ `ValueError` |
| `stimulus_congruency_by_switch_proportion_conditions` | `[]` | ❌ `ValueError` |
| `stimulus_switch_type_by_incongruent_proportion_conditions` | `[]` | ❌ `ValueError` |

> **Gotcha.** The `CONDITIONS` array in `submit_specific_conditions_power_traces_dcc.sh`
> ships with all four of the first group's labels **plus** the two cross-effect
> labels. If you flip `STATISTICAL_METHOD` and submit with that array unchanged,
> **two of the four jobs crash** with:
> ```
> ValueError: condition_label '...' has no 'subtraction_pairs';
> the interaction follow-up needs two simple-effect pairs.
> ```
> Trim the array to the two LWPC/LWPS labels before submitting.

> **Gotcha.** The block-balanced 8-cell sets are the ones the submit script
> recommends for the *ANOVA* (they make the 2-way terms equal-weight contrasts, so
> a block-level tonic offset cancels). They cannot run N2 as the code stands,
> because no one added `subtraction_pairs` for them. So the direction test runs on
> the 4-cell sets, which split by the **tested** proportion factor but leave the
> **other** proportion factor uncontrolled. Worth a per-cell count check; see
> [§7](#7-known-gaps-against-the-plan).

### 4.3 The statistical knobs

All in `run_power_traces_dcc.py`, all plain Python constants.

| Parameter | Line | Default | What it does |
|---|---|---|---|
| `N_PERM` | `:136` | `500` | Permutations. 500 floors any p at ~0.002. Fine for a direction check; raise for a reported p. |
| `P_THRESH_FOR_TIME_PERM_CLUSTER_STATS` | `:134` | `0.05` | Per-timepoint threshold for *forming* a cluster. |
| `P_CLUSTER` | `:135` | `0.05` | Threshold on the cluster statistic itself. |
| `TAILS` | `:137` | `2` | Two-sided. Keep it — a sign flip is a real possible outcome and you want to see it. |
| `PERMUTATION_TYPE` | `:121` | `'samples'` | Derived from `ttest_rel`. Paired. |
| `WINDOW_SIZE` / `STEP_SIZE` | `:90-92` | forced to `None` | ANOVA-only; the interaction path is full-resolution in time. |
| `N_JOBS` | `:69` | `-1` | All cores. |

### 4.4 Electrode selection

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `ROIS_DICT` | `run_power_traces_dcc.py:184` | `lpfc` + `occ` | lPFC is the target; `occ` rides along free as a negative control. |
| `ELECTRODES` | env var, `:193` | `'sig'` in the submit script, `'all'` in the runner | `sig` = passed the baseline-vs-signal filter. **This changes n substantially.** |
| `EXCLUDE_ELECTRODES` | env var, `:202` | empty | Comma-separated, `CHAN` or `SUB:CHAN`. Must be `export`ed, **not** passed through `sbatch --export` (commas truncate the list). |
| `ANOVA_LABELS_CSV` | env var, `:84` | `None` | Optional narrowing to a saved ANOVA subpopulation. **Leave unset for N2** — selecting on the effect before testing its direction makes the answer partly a property of the selection rule. |
| `SUBJECTS` | `:59` | 24 subjects | |
| `ACC_TRIALS_ONLY` | `:66` | `True` | Correct trials only. |

---

## 5. How to run it

### 5.1 On the cluster (the real run)

**Step 1 — switch the method.** Edit `dcc_scripts/power/run_power_traces_dcc.py:72`:

```python
STATISTICAL_METHOD = 'time_perm_cluster_interaction'
```

`WINDOW_SIZE`, `STEP_SIZE`, `STAT_FUNC_CHOICE` and `PERMUTATION_TYPE` all
reconfigure themselves from that one line.

**Step 2 — trim the condition list.** In
`dcc_scripts/power/submit_specific_conditions_power_traces_dcc.sh`:

```bash
CONDITIONS=(
    stimulus_lwpc_conditions
    stimulus_lwps_conditions
)
```

**Step 3 — submit.**

```bash
cd dcc_scripts/power
ELECTRODES=sig bash submit_specific_conditions_power_traces_dcc.sh
```

That submits one job per condition label (two jobs), each covering both ROIs and
emitting all three tests per ROI. Runtime is dominated by `N_PERM × n_times`;
the sbatch wrapper asks for 8 cores / 100 GB / 10 h, which is generous for this.

**Step 4 — run it again with `ELECTRODES=all`.** The direction should not depend
on the responsiveness filter. If it does, that is the result, and it is a problem.

```bash
ELECTRODES=all bash submit_specific_conditions_power_traces_dcc.sh
```

### 5.2 A single job by hand

```bash
cd dcc_scripts/power
sbatch --job-name=n2_lwpc \
  --export=ALL,CONDITION_LABEL=stimulus_lwpc_conditions,\
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20",\
ANOVA_UNIT=roi,ELECTRODES=sig \
  sbatch_power_traces_dcc.sh
```

### 5.3 Smoke test before burning cluster time

Per plan §9.4, `ttest_rel` through the vectorized path had not been exercised at
the time the branch was written. Uncomment the testing block at
`run_power_traces_dcc.py:253-259` (one subject, `N_PERM=2`, `N_JOBS=1`, lPFC only)
and run one condition locally first. You are checking that it completes and prints
three `[test] mean delta` lines per ROI — not the values.

### 5.4 Check the sign convention hasn't drifted

```bash
pytest tests/analysis/stats/test_effect_sign_conventions.py -v
```

---

## 6. Outputs

### Each of the three tests gets its own figure and its own npz

All three masks are persisted, and all three are plotted over the traces they
were computed from:

```
<save_dir>/n2_direction_tests/<roi>/
    <roi>_<label>_n2_direction_simple_low_<low_pair>_<n>_subjects_<elec>_sem_shading.{png,pdf}
    <roi>_<label>_n2_direction_simple_high_<high_pair>_<n>_subjects_<elec>_sem_shading.{png,pdf}
    <roi>_<label>_n2_direction_interaction_low_minus_high_<n>_subjects_<elec>_sem_shading.{png,pdf}
```

- The two **simple-effect** figures draw the raw condition traces (`i25` vs
  `c25`, `i75` vs `c75`) on the same y scale as the main power-trace figure, so
  they can be read against it directly. Their bar is that block's own simple
  effect — which is what §7 step 1 asks for and what the subtraction figure
  cannot show you.
- The **interaction** figure draws the two difference waves with the
  interaction bar. It drops the raw-trace `ylim`/`yticks` and auto-places the
  bar, because difference waves are roughly an order of magnitude smaller than
  the traces they come from: on the raw-trace axis they flatten onto zero and
  the bar sits at a height with no data near it.

The masks themselves land next to the evoked npz files, one file per ROI per
test:

```
<roi>/<conditions_save_name>_<roi>_n2_direction_{simple_low,simple_high,interaction}_cluster.npz
    mask              sample-level boolean, the cluster bar
    cluster_p_values  from time_perm_cluster
    delta             signed, averaged over electrodes, per timepoint
                      (positive = effect shrinks in the high-proportion block)
    traces            the two names the test contrasted
```

`delta` is the number `F` throws away, at full time resolution — so the
direction can be re-read, re-plotted or checked against the segregation scores
(§9) without re-running anything.

### Still keep the slurm log

The printed per-test deltas (below) are the fastest read on the result and are
the only place the "in sig cluster" vs "over epoch (n.s.)" distinction is
spelled out for you. Archive `out/aligned_svm_ncv/slurm_<jobid>.out`.

### What the log looks like

```
Interaction follow-up: (Stimulus_i25-Stimulus_c25) vs (Stimulus_i75-Stimulus_c75)
  positive delta = condition effect SHRINKS in the 75% block
-- Processing ROI: lpfc --
   [simple_Stimulus_i25-Stimulus_c25]  mean delta in sig cluster: +0.04812
   [simple_Stimulus_i75-Stimulus_c75]  mean delta in sig cluster: +0.01277
   [interaction]                       mean delta in sig cluster: +0.03535
```

The bracketed names are unchanged, so old logs still grep. The figure and npz
filenames use the shorter `simple_low` / `simple_high` / `interaction` keys.

Reading the delta line:

- `delta = (e1.data - e2.data).mean(axis=0)` — the signed effect averaged over
  electrodes, per timepoint. This is the number `F` throws away.
- **`in sig cluster`** → averaged over the significant timepoints only. Meaningful.
- **`over epoch (n.s.)`** → **no significant cluster was found**, so the number is
  the average over the whole epoch including the baseline. It is a descriptive of
  a null result. Do not report it as an effect.

### Files on disk

Output root:

```
dcc_scripts/power/figs/<EPOCHS_ROOT_FILE>/anova_within_<ANOVA_UNIT>/
```

(Yes, `anova_within_` even for the interaction method — the path name is built
from `ANOVA_UNIT` unconditionally at `run_power_traces_dcc.py:261`.)

| File | Contains | Useful for N2? |
|---|---|---|
| `n2_direction_tests/<roi>/..._n2_direction_simple_low_<low_pair>_...{png,pdf}` | `i25` vs `c25` with the **low-block simple-effect** bar | ✅ **§7 step 1** |
| `n2_direction_tests/<roi>/..._n2_direction_simple_high_<high_pair>_...{png,pdf}` | `i75` vs `c75` with the **high-block simple-effect** bar | ✅ **§7 step 1** |
| `n2_direction_tests/<roi>/..._n2_direction_interaction_low_minus_high_...{png,pdf}` | the two difference waves, auto-scaled, with the **interaction** bar | ✅ **§7 step 2** |
| `<roi>/..._n2_direction_<test>_cluster.npz` | `mask`, `cluster_p_values`, `delta`, `traces` — one per ROI per test | ✅ the result, in numbers |
| `<roi>/<roi>_<label>_subtractions_<n>_subjects_<elec>_sem_shading.{png,pdf}` | both difference waves on the raw-trace y scale, with the **interaction** cluster bar | the older single N2 figure |
| `<roi>/<roi>_<label>_<n>_subjects_<elec>_sem_shading.{png,pdf}` | the four raw condition traces, **no** cluster bar | context |
| `<roi>/..._electrode_deviations.txt` | the most deviant electrodes in the baseline window | leverage check |
| `<roi>/<save_name>_<condition>_<roi>_evoked.npz` | `data`, `times`, `ch_names` — the `(n_elec, n_times)` matrices | ✅ lets you recompute anything |
| `<save_name>_metadata.json` | condition label, ROIs, `statistical_method` | confirms which method ran |

The `_evoked.npz` files are the escape hatch: since every per-condition
`(n_electrodes, n_times)` matrix is saved with its channel names, you can
reconstruct the difference waves and rerun the test offline without touching the
epochs.

> **Note.** `.png`, `.pdf`, `.npz`, `.csv` and `.json` are all in `.gitignore`.
> None of these outputs come back with the repo — they live on the cluster only.

---

## 7. How to read the result

Read the **three tests together**, in this order.

### Step 1 — the two simple effects

Each should be **positive** in lPFC: incongruent > congruent, switch > repeat.
That is the base effect and it should be there.

- Same sign, different magnitude → the expected adaptation pattern. Go to step 2.
- **A sign flip between blocks** → more interesting, and more suspicious. The
  condition effect *reversed* under the block manipulation rather than shrinking.
  Check the epoch metadata before believing it.
- Neither simple effect significant → there is no effect to adapt. The interaction,
  if any, is not interpretable as adaptation.

### Step 2 — the interaction

| Interaction sign | Reading | Verdict |
|---|---|---|
| **positive**, both simple effects positive | effect shrinks in the high-proportion block | ✅ **adaptation — the predicted direction.** N2 passes. |
| **negative**, both simple effects positive | effect *grows* in the high-proportion block | ⚠️ opposite to behavior. **Kill switch fires.** |
| not significant | no cluster survives | ⚠️ the ANOVA said the interaction exists; this says its direction is not resolvable at this threshold. Reconcile before proceeding. |

### Step 3 — the kill switch

> Compare against the **behavioral** LWPC/LWPS directions from
> `stats/erin_linear_mixed_effects_model.py` / `combinedData.csv`. Both are on the
> same low-minus-high convention (pinned by the sign-convention test), so they are
> directly comparable.
>
> **If the neural directions disagree with the behavioral ones, stop.** Re-read the
> epoch metadata before running anything in N3–N4.

### Step 4 — report `n_electrodes` and `n_subjects` together

Never one without the other. `n_electrodes` is in the evoked's `ch_names`;
`n_subjects` is `len(SUBJECTS)` after the dropped-electrode summary the script
prints. The pooled p-value is optimistic (§3), so the leverage descriptives are
what actually protect the result.

### Step 5 — the `occ` control

`occ` rides along in `ROIS_DICT` for free. A congruency direction that looks the
same in occipital cortex as in lPFC is a warning about the baseline, not a finding.

---

## 8. Known gaps against the plan

Two things plan §2 asks for that the code does **not** currently do. Neither
blocks the direction read; both are what you would need for a manuscript.

1. **No per-subject direction tally.** The plan asks how many subjects' electrode
   averages point the expected way. Subject identity is discarded at
   `evoked_builders.py:133`, so this is not recoverable from the run — but it *is*
   recoverable from the saved `_evoked.npz` files, whose `ch_names` carry the
   subject prefix.

2. **No leave-one-subject-out sweep.** `leave_one_subject_out` exists at
   `stability_flexibility_anatomy.py:1316` and is wired into the anatomy DCC, but
   there is no equivalent on the N2 path. lPFC coverage is skewed, and LOSO is what
   rules out a one-subject result. Same workaround: rebuild from the npz files.

Two design-level caveats:

3. **The direction test runs on the 4-cell sets, not the block-balanced 8-cell
   sets**, because only the former have `subtraction_pairs`. The 4-cell sets do
   split by the tested proportion factor — `Stimulus_c25` is a single BIDS event
   (`Stimulus/c25.0`), congruent trials in 25%-incongruent blocks — so the LWPC
   block factor is explicit. What rides along uncontrolled is the **other**
   proportion factor (switch proportion, for LWPC). Check the per-cell counts
   before treating a marginal interaction as real.

4. **`N_PERM = 500`** floors any p-value at about 0.002. Adequate for a direction
   check; raise it if the number goes in a paper.

---

## 9. The second read — the two files you thought this was in

`stability_flexibility_segregation_dcc.py` and `stability_flexibility_anatomy_dcc.py`
are the **N4** pipeline. They are not N2 — but the segregation module gives you a
genuinely independent second read on the same direction question, which is
probably why they came to mind.

### What segregation computes that is relevant

`compute_sensitivities_per_split(df, contrast_mode='proportion')` scores, **per
electrode**, an LWPC value (`x`) and an LWPS value (`y`). Each is an equal-cell-weight
difference-of-differences divided by the pooled within-cell SD — a signed,
*d*-like effect size — computed on **disjoint trial halves** so the score is not
read at a peak selected by the same contrast.

`W_INTERACTION` (`stability_flexibility_segregation.py:519`) is the **same
low-minus-high orientation as N2**. So:

> **The mean sign of the `x` column is a windowed, per-electrode read on exactly
> the direction N2 tests in time.** If N2 says the LWPC interaction is positive in
> lPFC, `x` should average positive over the same electrodes. If they disagree,
> one of them is mis-oriented and the sign-convention test will tell you which.

The relevant columns land in `continuous.csv` and `per_split.csv` in the
segregation output directory.

### Running it for that purpose

```bash
cd dcc_scripts/stats
EPOCHS_ROOT_FILE="<same epochs dir as the N2 run>" \
WINDOW_TMIN=0.0 WINDOW_TMAX=0.5 \
CONTRAST_MODE=proportion \
ELECTRODES=sig ROIS=lpfc \
bash submit_stability_flexibility_segregation_dcc.sh
```

`CONTRAST_MODE=proportion` is what makes `x`/`y` the LWPC/LWPS **interactions**
rather than the congruency/switchType main effects — with `CONTRAST_MODE=condition`
you get base effects and the comparison to N2 is meaningless.

There is also an alignment helper: setting `ALIGN_TO_POWER_TRACES_RUN` to a power
traces run directory reads `WINDOW_TMIN`/`WINDOW_TMAX` off the extent that run's
windows actually tiled, so the two analyses cover the same window.

### What each of those two scripts is actually for

| Script | Beat | Question |
|---|---|---|
| `stability_flexibility_segregation_dcc.py` | N4 input | per-electrode LWPC/LWPS scores on disjoint halves; their joint distribution, correlation, and split-half noise ceiling |
| `stability_flexibility_anatomy_dcc.py` | N4 | are those scores organized differently across cortex? ROI/Destrieux enrichment, coverage-conditioned, with a within-electrode effect-label-swap null |

Their outputs are documented separately in
[`stability_flexibility_outputs_guide.md`](stability_flexibility_outputs_guide.md)
(§"Segregation output directory" and §"A3 anatomy output directory").

---

## 10. Checklist

```
[ ] pytest tests/analysis/stats/test_effect_sign_conventions.py   (sign convention intact)
[ ] run_power_traces_dcc.py:72  ->  'time_perm_cluster_interaction'
[ ] CONDITIONS array trimmed to the two LWPC/LWPS labels
[ ] smoke test: 1 subject, N_PERM=2, lpfc only, completes
[ ] submit with ELECTRODES=sig
[ ] submit with ELECTRODES=all      (direction must not depend on the filter)
[ ] archive the slurm .out files    (fastest read on the three deltas)
[ ] record n_electrodes AND n_subjects for each ROI
[ ] both simple effects positive in lpfc?   <- n2_direction_tests/<roi>/*simple_{low,high}*
[ ] interaction positive  =  adaptation, N2 passes
                                            <- n2_direction_tests/<roi>/*interaction*
[ ] compare against the behavioral LWPC/LWPS direction  <- KILL SWITCH
[ ] occ does not show the same pattern
```

---

## Related documents

- [`analysis_plan_concurrent_regulation.md`](analysis_plan_concurrent_regulation.md) §2 — the N2 spec this implements; §9.2 and §9.4 for the leverage and pairing arguments
- [`stability_flexibility_data_flow.md`](stability_flexibility_data_flow.md) — the A1–A7 walkthrough
- [`analysis_guide.md`](analysis_guide.md) — the pipelines as built
- [`stability_flexibility_outputs_guide.md`](stability_flexibility_outputs_guide.md) — how to read the N4 segregation/anatomy outputs
- [`n4_continuous_anatomy.md`](n4_continuous_anatomy.md) — how to run and interpret the N4 continuous-score anatomy tests, maps, and descriptive medoids
- [`analysis_simplification_plan.md`](analysis_simplification_plan.md) §2.2b — why the contrasts are cell-balanced
