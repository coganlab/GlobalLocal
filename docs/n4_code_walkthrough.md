# N4 code walkthrough: from trials to the §19 results

**What this is.** A line-by-line guide to the code behind N4, written for
someone who knows the results ([`n4_continuous_anatomy.md`](n4_continuous_anatomy.md)
§0) but not the code. Most of it is about §19: participants as the unit,
shared trial splits, local similarity, the overlap controls and the combined
Figure 5. The building blocks those reuse come first.

**The companion notebook**, `dcc_scripts/stats/n4_code_walkthrough.ipynb`,
follows the same sections and runs each step on the real all-lPFC outputs, so
you can print every intermediate array. Open it on the DCC (or anywhere the
run folders are reachable); without the real files it falls back to planted
data with the same columns. It also has the active-learning parts: questions
answered with `ask('q1', 'a')`, "predict before you run" prompts, and exercises
checked against the pipeline. The questions, with folded answers, are also in
[§7](#7-check-yourself) here.

**References** are `file:line` at the commit that added this guide. Files:

| Short name | File |
|---|---|
| `sfs` | `src/analysis/stats/stability_flexibility_segregation.py` (scoring, the overlap test) |
| `sfa` | `src/analysis/stats/stability_flexibility_anatomy.py` (anatomy, §16, §19) |
| script | `dcc_scripts/stats/n4_section19_followups.py` |
| tests | `tests/analysis/stats/test_section19_anatomy.py` (planted worlds with known answers) |

---

## 1. The map

```text
long table (long_df.csv): one row per electrode × trial
  subject, electrode, trial, congruency, switchType,
  incongruent_proportion, switch_proportion, hg, rt
        │
        │ sfs.compute_sensitivities_per_split          §3.1
        ▼
per_split.csv: one row per electrode × split
  xA xB   LWPC on half A / half B
  yA yB   LWPS
  mxA mxB congruency          (MAIN_EFFECTS=1)
  myA myB switch type
  split_scheme = 'participant' (only with shared splits)
        │
        ├── sfs.average_over_splits + add_responsiveness → electrodes.csv (x, y, mx, my, resp)
        │        │
        │        │ sfa.attach_scores                    §3.5
        │        ▼
        │   scores_with_anatomy.csv: lwpc_s, lwps_s, cong_s, switch_s,
        │        delta, dm, resp, anat, mni_x/y/z, hemi
        │
        └── sfs.split_resolved_corr (the overlap test)  §3.3
                 built on _residualised_split_matrices  §3.2

sfa.section19(scores, per_split, ...)                    §4.6
  1  coordinate_slope_by_participant, coordinate_slope_loso   §4.1
  2  sfs.participant_split_corr                               §4.2
  3  local_similarity (+ _shared_split_reliabilities)         §4.3
  4  figure5_height                                           §4.4
  5  overlap_controls                                         §4.5
```

Two conventions hold everywhere:

- **x is LWPC and y is LWPS** in the per-split table and in `electrodes.csv`
  (`mx`, `my` for congruency and switch). `attach_scores` renames them to
  `lwpc_*`, `lwps_*`, `cong_*`, `switch_*`.
- **A and B are the two halves of one random split** of an electrode's trials.
  Anything that relates two scores takes them from opposite halves.

---

## 2. The three tables you will meet

**The long table** (`long_df.csv`, written by `assemble_long_df`): one row per
trial per electrode. `hg` is window-mean high gamma (0–1.5 s); `trial` is the
trial id shared by all of a participant's electrodes (added 2026-09-27, needed
for shared splits); `rt` is the reaction time. `RT_ADJUST_HG=1` replaces `hg`
with its RT-adjusted version (`rt_adjust_hg`, `stability_flexibility_brain_behavior.py:499`).

**The per-split table** (`per_split.csv`): for each electrode and split
*k* = 0 … K−1, the four scores on half A and on half B. 398 electrodes × 1,000
splits = 398,000 rows in the all-lPFC run.

**The score table** (`scores_with_anatomy.csv`): one row per electrode, with the
split-averaged scores, their pooled-scaled versions, the balances, the
covariate `resp` and the anatomy.

---

## 3. Building blocks

### 3.1 Scoring: `compute_sensitivities_per_split` (`sfs:745`)

The score itself is `_interaction_cohens_d` (`sfs:539`): the four cell means
combined with fixed weights, over the pooled within-cell SD.

```python
W_INTERACTION = {(1, 1): +1, (0, 1): -1, (1, 0): -1, (0, 0): +1}   # sfs:519
W_MAIN        = {(1, 1): +.5, (0, 1): -.5, (1, 0): +.5, (0, 0): -.5}  # sfs:521
```

The key is `(cond, mod)`: `cond` = 1 for incongruent (switch), `mod` = 1 for the
low-proportion (25 %) block (`_CONTRAST_PRESETS`, `sfs:158`). So W_INTERACTION
gives (I − C | 25 %) − (I − C | 75 %) = LWPC, and W_MAIN gives the mean of the
two simple effects = congruency. Equal weights mean the frequent cells cannot
dominate.

The loop, `sfs:795–819`:

```python
for (subj, elec), sub in work.groupby(['subject', 'electrode']):
    if shared is not None:                                    # shared split
        halves = shared[subj].loc[sub['trial'].to_numpy()].to_numpy()
    for k in range(n_splits):
        if shared is not None:
            g1, g2 = sub[halves[:, k] == 0], sub[halves[:, k] == 1]
        else:                                                 # the default
            h1, h2 = _stratified_half_split(sub, rng, strata_cols=strata)
            g1, g2 = sub.loc[h1], sub.loc[h2]
        row = dict(..., xA=_effect_for(g1, 'stability', ...), xB=_effect_for(g2, ...),
                        yA=_effect_for(g1, 'flexibility', ...), yB=...)
```

- **Default (per-electrode split).** `_stratified_half_split` (`sfs:674`)
  shuffles each electrode's own trials within each design cell and cuts them in
  half. Every electrode gets its own random halves.
- **Shared split** (`shared_split=True`). `_shared_trial_halves` (`sfs:722`)
  draws the halves once per participant and split, over trial ids, and every
  electrode of that participant looks up its trials in that one table
  (`halves[:, k]`, 0 = A, 1 = B). The output gets `split_scheme = 'participant'`.

**Why the shared split matters.** Within one electrode both schemes give
disjoint halves. Across electrodes they do not: with per-electrode splits,
electrode *i*'s half A and electrode *j*'s half B share about half their
trials. Neighbouring contacts share trial-to-trial noise, so anything that
compares one electrode's half with *another* electrode's half picks up that
shared noise. Two §19 analyses do exactly that: local similarity (pairs of
electrodes) and within-participant reliabilities (centring within participant
mixes every electrode's half into every other's). The test
`test_per_electrode_splits_turn_shared_trial_noise_into_local_similarity`
plants trial noise correlated between neighbours and no real local structure;
per-electrode splits make neighbours look similar, shared splits do not.

### 3.2 The matrices behind every overlap number: `_residualised_split_matrices` (`sfs:953`)

Turns the per-split table into one array per score, shape (splits × electrodes),
already cleaned the way the pre-specified test needs.

1. `sfs:973–980`: attach `resp` (and any `covariates`, e.g. MNI coordinates) to
   each row; drop rows with missing values.
2. `sfs:987–990`: keep only electrodes present in **every** split (so each split
   is a vector over the same electrodes), then drop participants with fewer than
   `min_elec` electrodes.
3. `sfs:1008–1020`: build `elecs` (sorted ids), `subj`, `groups` (the column
   indices of each participant) and `mats[k]` for each key (`xA`, `xB`, …).
4. `sfs:1032–1046`, the cleaning, **per split and per key**:

   ```python
   r = mats[k][si] - X @ lstsq(X, mats[k][si])     # regress out intercept + resp (+ covariates)
   for g in groups:
       r[g] -= r[g].mean()                          # centre within participant
   ```

   Covariates are centred within participant before entering `X`, so the
   residuals are orthogonal to them within participant. `half_covariates`
   (used by the overlap controls, §4.5) adds per-split columns from the **same
   half** to one key only: `xA` is cleaned of `mxA` and `myA`, `yB` of `mxB` and
   `myB`, so the two sides of a correlation still come from different trials.

Returns `(elecs, subj, groups, splits, mats, n_elec_in)`. The pooled overlap
test, its participant-level version and local similarity all call this, so they
see identical values.

### 3.3 The overlap test: `split_resolved_corr` (`sfs:852`)

```python
U = {k: vstack([_unit_vector(mats[k][si], method) for si in splits]) for k in mats}   # sfs:904
M = 0.5 * (U['xA'].T @ U['yB'] + U['xB'].T @ U['yA']) / len(splits)                 # sfs:913
obs = trace(M)                                                                        # sfs:915
rel_x = (U['xA'] * U['xB']).sum(1).mean()                                             # sfs:916
```

- `_unit_vector` (`sfs:841`) ranks (Spearman), centres and scales each split's
  vector to unit length, so **the correlation of two such vectors is their dot
  product**.
- `M[i, j]` is the average over splits of LWPC of electrode *i* (one half)
  times LWPS of electrode *j* (other half). Its diagonal summed is the observed
  correlation averaged over the two directions and all splits: that is `corr`.
- `rel_x`: LWPC half A against LWPC half B, the split-half reliability. Within
  participant, because the values were centred within participant.
- **The null** (`sfs:921–925`) permutes electrodes within each participant,
  the same permutation for every split, and reads the permuted statistic off
  the precomputed `M` as `M[identity, perm].sum()`: each permutation costs
  O(electrodes) rather than a full recomputation.
- `p = (#{|null| ≥ |obs|} + 1) / (n_perm + 1)`, two-sided.

### 3.4 The anatomy tests: `_nuisance_design`, `_swap_null`, `_coordinate_fit`

- `_nuisance_design` (`sfa:1019`): the design matrix every anatomy model
  conditions on: intercept, participant dummies (fixed effects), centred `resp`.
- `_swap_null` (`sfa:1047`): the effect-label swap. Swapping LWPC and LWPS within
  an electrode turns delta into −delta, so the null multiplies delta by random
  ±1 per electrode, re-residualises on the nuisance design, and recomputes the
  statistic, in chunks of 512.
- `_coordinate_fit` (`sfa:1188`): delta ~ coordinates + nuisance. By
  Frisch–Waugh, residualising the coordinates on the nuisance design once
  (`Z = Z0 − X B Z0`, `sfa:1194`) lets each permutation be a 3-column regression:
  `R @ Zp.T` gives the three slopes, and the block *F* is the explained over the
  residual sum of squares (`sfa:1209–1212`).

### 3.5 Scaling: `attach_scores` (`sfa:896`)

Renames `x`, `y`, `mx`, `my`, joins anatomy and coordinates, then
(`sfa:959–969`):

```python
out['lwpc_s'] = out['lwpc_score'] / out['lwpc_score'].std(ddof=1)   # one factor per effect
out['delta']  = out['lwpc_s'] - out['lwps_s']
out['dm']     = out['cong_s'] - out['switch_s']
```

One SD across all electrodes, no centring, no within-participant z-score (with
two electrodes a within-participant z-score is ±0.707 whatever the data). The
units of every slope and balance in §19 are these "SD units".

---

## 4. §19, part by part

### 4.1 Section 1: the height slope with participants as the unit

**`coordinate_slope_by_participant`** (`sfa:1670`).

*Step 1, residualise* (`_partial_axis_residuals`, `sfa:1608`):

```python
X, _ = _nuisance_design(d, covariates)              # intercept, participant dummies, resp
N = column_stack([X, d[['mni_y', 'mni_x']]])        # plus the OTHER two axes
v = delta - N @ pinv(N) @ delta                     # delta, cleaned
a = z     - N @ pinv(N) @ z                         # height, cleaned
```

By Frisch–Waugh, the slope of `v` on `a` is exactly the coordinate test's *z*
slope. Because participant dummies are in `N`, `v` and `a` are centred within
each participant.

*Step 2, split the slope by participant* (`sfa:1704–1711`):

```python
for s in participants:
    w_s = sum(a[s] ** 2)                 # participant's spread along z (after cleaning)
    b_s = sum(a[s] * v[s]) / w_s         # participant's own slope
pooled = sum(a * v) / sum(a * a)         # = sum_s w_s b_s / sum_s w_s
```

The last line is an identity: the pooled numerator and denominator are sums
over participants. So **the coordinate test's slope is a weighted average of
participants' own slopes, weighted by how much each participant's electrodes
spread in height** (`test_participant_slopes_decompose_the_coordinate_test_slope`
checks it to 1e-9). `top3_weight_share` is how much of that average three
participants carry (38 % in all lPFC).

*Step 3, test across participants* (`sfa:1713–1723`):

```python
flips = rng.choice((-1, 1), size=(n_perm, n_participants))
null  = (flips * (w * b)).sum(1) / w.sum()          # flip whole participants' slopes
p_signflip = (#{|null| >= |pooled|} + 1) / (n_perm + 1)
boot = resample participants with replacement, recompute the weighted mean
```

Under the null that participants' slopes are symmetric around zero, each
participant's sign is a coin flip. Participants, not electrodes, are now the
units.

*Step 4, unweighted* (`sfa:1725–1736`): the plain mean of `b_s` over
participants with ≥ 3 electrodes and ≥ 5 mm spread, one-sample *t*-test, and a
sign test on how many are negative. Every participant counts once.

*Step 5, mixed models* (`_mixed_slope_fits`, `sfa:1622`): `statsmodels.mixedlm`,
every predictor centred within participant and coordinates in cm for the
optimiser (`sfa:1636–1641`), REML. `random_intercept` reproduces the
electrode-level slope and precision (centred predictors: the intercept only
absorbs participant means), so it is *not* a participant-level test.
`random_slope` (`re_formula='~mni_z_c'`) lets each participant have its own
height slope; its fixed slope's Wald *p* and the random-slope SD
(`sqrt(cov_re)`, `sfa:1664`) are the numbers to quote.

**`coordinate_slope_loso`** (`sfa:1748`): `_coordinate_fit` with each
participant dropped, via `leave_one_subject_out` (`sfa:1472`); the first row,
`(none)`, is the full fit.

### 4.2 Section 2: the overlap r with participants as the unit

**`participant_split_corr`** (`sfs:1051`). Same matrices as the pooled test
(`_residualised_split_matrices`, `min_elec=4`), then for each participant
(`sfs:1089–1096`):

```python
U = {k: unit_rows(mats[k][:, g]) for k in mats}     # rank + centre + unit-norm within participant
corr = mean over splits of 0.5 * (sum(U['xA'] * U['yB']) + sum(U['xB'] * U['yA']))
```

That is `split_resolved_corr` run on one participant's electrodes (with one
participant the two agree exactly; tested). Then (`sfs:1098–1112`):

```python
z = arctanh(corr)                       # Fisher z per participant
w = max(n_electrodes - 3, 1)            # inverse-variance weight of a correlation
obs = sum(w * z) / sum(w)
null: flip the sign of each participant's z;  CI: participant bootstrap, back to r with tanh
unweighted: t-test of z against 0
```

### 4.3 Section 3: local similarity

**`local_similarity`** (`sfa:2744`). The question: are nearby electrodes more
alike in their LWPC − LWPS balance than distant ones?

*Guard* (`sfa:2795`): refuses a per-split table that is not
`split_scheme == 'participant'` (§3.1 explains why).

*Scores* (`_local_score_halves`, `sfa:2717`): builds the balance per half,
`balanceA = xA / SD_lwpc − yA / SD_lwps`, with the same pooled SDs as `delta`
(read back from `scores` as `lwpc_score / lwpc_s`). The five scores are the
balance, LWPC, LWPS, congruency and switch; each is a pair of (half A, half B)
columns.

*Clean* (`sfa:2811–2813`): `_residualised_split_matrices` with the MNI
coordinates as covariates, so the linear gradient is removed (the question is
about structure *beyond* the gradient), plus `resp`, then centred within
participant.

*Standardise* (`standardise`, `sfa:2816`): within each participant, rank across
its electrodes (Spearman) and re-centre; then scale each split's vector to unit
mean square.

*Pairs and distances* (`sfa:2832–2842`): for each participant, every pair
`i < j` (`np.triu_indices`), their Euclidean distance in MNI mm, and, for bipolar
channels, drop pairs that share a contact (`_channel_poles`, `sfa:2709`; none in
this data set, which is monopolar).

*The similarity matrix* (`sfa:2866`), per score and participant:

```python
C = 0.5 * (A[:, g].T @ B[:, g] + B[:, g].T @ A[:, g]) / n_splits
```

`C[i, j]` = electrode *i*'s half A times electrode *j*'s half B, plus the
reverse, averaged over splits. Because the halves are shared, *i*'s A and *j*'s
B never share a trial. `C[i, i]` is electrode *i*'s own split-half
reliability; `trace(C) / n` is the score's reliability (`sfa:2867`, `2876`).

*Bins and baseline* (`sfa:2849–2871`):

```python
obs[g], cnt[g] = bin_sums(C[iu, ju], D[iu, ju])              # sum of C per distance bin
base[g] = mean over 5,000 shuffles p of bin_sums(C[iu, ju], D[p[iu], p[ju]])
dev = obs - base                                              # each participant's excess
```

Shuffling the electrodes' positions within a participant keeps the same pairs'
similarities but scrambles which distance bin they fall in, so `base` is what
each bin would hold with no relation between distance and similarity.
Within-participant centring makes pairs slightly anti-correlated on average,
and the baseline carries the same bias, so subtracting it removes it.

*Test* (`sfa:2873–2900`): `excess = sum_g dev / n_pairs` per bin. The null
flips the sign of each participant's whole excess (`flips @ dev`), one-sided
(more similar than baseline). Intervals come from resampling participants.
Pairs are not used as units because one data set's estimation noise is itself
spatially smooth: a pair-level null gave 13 % false positives on simulated data,
the participant sign-flip 2.5 %.

*Ratios only to a reliable map* (`sfa:2879–2885`): the excess is also expressed
as a share of the score's reliability (`relative`), but only when the
reliability is positive in ≥ 90 % of bootstrap draws. Otherwise `notes` says
why it is missing.

*Contrasts and comparison* (`sfa:2906–2933`): nearest bin minus farthest bin per
score, with the same sign-flip null; and the balance's relative near-range
excess against each single score's, with a paired participant bootstrap.

**How to read it.** The single scores are the positive control: they should
show a near-range excess if the analysis can see local structure at all. Then
a balance with no excess while the single scores show one is "intermixed at the
recorded scale"; a balance with an excess is "patchy"; no score with an excess
is "no power". On the real data a fourth case occurred: the balance's own
reliability is about zero, so there is nothing reliable to be patchy or
intermixed (§19.8.6 of the N4 doc).

**`_shared_split_reliabilities`** (`sfa:3084`): runs `split_resolved_corr`
(with `n_perm=1`; only the reliabilities and *r* are wanted) on the
per-electrode and the shared table, restricted to the electrodes both have,
and on the main effects via `main_effect_view` (`sfs:667`, which puts `mx*`/`my*`
into the `x*`/`y*` slots). Output: `reliability_by_split_scheme.csv`.

### 4.4 Section 4: the combined Figure 5

**`figure5_height`** (`sfa:2571`) assembles four pieces:

1. `figure5_height_points` (`sfa:2144`): runs `prepare_continuous` (`sfs:1173`:
   regress each score on `resp`, centre within participant) on `lwpc_s` and
   `lwps_s`, then adds each score's overall mean back (`x_plot`, `y_plot`), so
   the identity line still means LWPC = LWPS. `balance = x_plot − y_plot`.
   `height_bands` (`sfa:2133`) cuts MNI *z* at its 33rd and 67th percentiles.
2. `_figure5_test` (`sfa:1920`): panel b's *r*, read from the segregation run's
   `correlation.json` if it tested the same electrodes, otherwise recomputed.
3. `height_centroids` (`sfa:2172`): each tertile's mean of `x_plot`, `y_plot`.
   The bootstrap resamples participants (one draw shared by all bands): per
   draw, `bincount` sums each participant's points, `n[draws].sum(1)` adds up
   the resampled participants' counts, and the ratio is the resampled centroid.
   The 2 × 2 covariance of those draws gives the ellipse (`_ellipse`,
   `sfa:2231`, χ² with 2 df) and the 2.5–97.5 % of `x − y` the balance interval.
4. `balance_by_height` (`sfa:2208`): delta with participant and `resp` removed
   (`_nuisance_design` + least squares, mean added back), averaged within each
   participant and band, then mean ± SEM across participants.

`sfa:2618–2620` refits the coordinate test on the plotted balance and prints
its slope next to the test's: if the figure's scaling drifted from the test's,
the two would differ (on the real data both are −0.00770/mm).

**The tertiles on the brain** (added 2026-10-06). `figure5_height` also writes
`fig5_height_brain_centroids.csv` on every run, from
`height_band_brain_centroids` (`sfa:2393`): for each tertile and hemisphere (the
sign of *x*), the mean MNI position of its electrodes, skipped if fewer than
three. Per hemisphere because a bilateral mean of *x* would sit near the
midline, in neither hemisphere. With `make_brain=True` (the anatomy job's
`MAKE_BRAIN`, the script's `--brain`), `plot_height_bands_on_brain`
(`sfa:2494`) draws the electrodes in their tertile's colour at 40 % opacity and
each centroid as an opaque, darker sphere, moved onto the lateral cortical
surface at the same *y* and *z* so it is not hidden under the translucent brain.
It needs the recon files and a display; without them it writes a sagittal
fallback. Like panel c, these centroids describe where the sampled electrodes
are and test nothing.

### 4.5 Section 5: the overlap controls

**`overlap_controls`** (`sfa:1783`). One inner function, `run(name, table,
covariates, half)`, calls `split_resolved_corr` with extra covariates and
appends a row. In order (`sfa:1839–1865`):

| Row | What is passed | Effect |
|---|---|---|
| `pre-specified` | nothing extra | the test as specified |
| `+ responsiveness, nonlinear` | `log(resp)`, `resp²` as `covariates` | signal-to-noise beyond linear mean \|HG\| |
| `+ MNI coordinates` | `mni_y, mni_z, mni_x` | a shared smooth gradient |
| `+ base effects, same half` | `half=_SAME_HALF_BASE` (`sfa:1779`) | each half's LWPC and LWPS cleaned of that half's congruency and switch |
| `+ RT coupling` | `rt_r` per electrode, on the electrodes that have one (and the `pre-specified` reference row on the same electrodes) | RT coupling |
| `all of the above` | every covariate together, plus the same-half base effects | |
| `responsiveness tertile …` | the pre-specified test within each third of `resp` | descriptive |

`_SAME_HALF_BASE = {'xA': ('mxA', 'myA'), 'xB': ('mxB', 'myB'), 'yA': ('mxA', 'myA'), 'yB': ('mxB', 'myB')}`:
the half-A scores are cleaned of half-A base effects and the half-B scores of
half-B base effects, so `xA` (cleaned with A) is still correlated with `yB`
(cleaned with B) across disjoint trials. `test_same_half_covariates_keep_the_halves_apart`
checks that this control does not touch an overlap the base effects do not
carry; `test_overlap_controls_remove_the_confound_that_made_the_overlap`
plants a base-driven and an RT-driven overlap and checks that each control
removes its own confound and not the other.

Leave-one-participant-out (`sfa:1869–1880`): the pre-specified test with each
participant dropped, at a tenth of the permutations.

### 4.6 The dispatcher and the script

**`section19`** (`sfa:3108`) runs the five parts, each inside its own
`try`, so one failure does not stop the rest:

- section 1 only if coordinates exist;
- sections 2, 3, 5 only with a per-split table;
- section 3 uses `per_split_shared` if given, else the run's own table **only if**
  that table is itself shared; otherwise it prints "local similarity: skipped";
- with a shared table it also prints the overlap *r* on the rescored table and
  `reliability_by_split_scheme.csv`.

It returns `(lines, out)`: the text that becomes `summary_section19.txt` and a
dict that becomes `section19.json`. The anatomy job calls it on every new run
(`stability_flexibility_anatomy_dcc.py:714`) without a shared table, so new
runs skip section 3.

**The script** (`n4_section19_followups.py:81`) is a thin wrapper:

1. reads `scores_with_anatomy.csv` and `per_split.csv` from `--anatomy-dir`
   (or `--seg-dir`);
2. if the `--long-df` or `--per-split-shared` file has an
   `rt_adjustment_slopes.csv` beside it, labels the run RT-adjusted and writes
   to `section19_rt_adjusted/` (`:123–131`);
3. with `--long-df`, rescores with `compute_sensitivities_per_split(...,
   main_effects=True, shared_split=True)` (`:147`), 200 splits by default, and
   saves `per_split_shared.csv`; a table without `trial` prints how to get one
   and skips section 3;
4. reads `--rt-coupling` (needs `electrode`, `rt_r`);
5. calls `sfa.section19` and writes `summary_section19.txt` and `section19.json`.
   `--brain` (with `--brain-hemi`, `--brain-zoom`) passes `make_brain=True` to
   it, which draws the tertile brain in section 4; run that under `xvfb-run`.

---

## 5. Where to look when a number surprises you

| Number | Recompute it with | Check |
|---|---|---|
| Overlap *r* | `sfs.split_resolved_corr(per_split, resp)` | Same electrodes and `min_elec` as the run? |
| Within-participant reliability | `split_resolved_corr(...)['reliability_x']` | Which split scheme? Per-electrode reliabilities are biased low. |
| A participant's slope | `coordinate_slope_by_participant(scores)['per_participant']` | Its `spread_mm`: a participant with little spread has a noisy slope and little weight. |
| A participant's overlap *r* | `participant_split_corr(per_split, resp)['per_participant']` | Its `n_electrodes` and reliabilities. |
| Local-similarity excess | `local_similarity(per_split_shared, scores)['table']` | `split_scheme` is `participant`; the "same electrode" row's reliability. |
| An overlap-control row | `overlap_controls(scores, per_split, rt)` | `n_electrodes` of the row (the RT rows use only electrodes with an RT value). |

## 6. Glossary

| Term | Meaning |
|---|---|
| balance, delta | `lwpc_s − lwps_s` per electrode; positive = relatively LWPC-dominant |
| dm | `cong_s − switch_s`; positive = relatively congruency-dominant |
| SD units | a score divided by its SD across all electrodes of the run |
| `resp`, responsiveness | mean \|high gamma\| of the electrode, the gain covariate |
| per-electrode split / shared split | trial halves drawn for each electrode separately / once per participant and given to all its electrodes |
| separate-half (cross-half) | one score from half A, the other from half B of the same split |
| sign-flip test | null made by multiplying each unit's statistic by random ±1; the unit is a participant in §19, an electrode in the swap null |
| swap null | exchanging LWPC and LWPS within an electrode, i.e. a sign flip of delta |
| Frisch–Waugh | regressing *y* and *x* on the other predictors first gives the same slope of *y* on *x* as the full model |
| Fisher *z* | `arctanh(r)`; averages correlations on a scale where their sampling variance is about 1/(*n* − 3) |

## 7. Check yourself

The same questions are in the notebook, where `ask('q6', 'b')` tells you whether
you are right, and three exercises (E1–E3) are checked against the pipeline's
own numbers. Here, try each before opening the answer.

**q1 (§2).** In `per_split.csv`, what is `yB`?

<details><summary>Answer</summary>

LWPS scored on half B of that split. x = LWPC, mx = congruency, my = switch;
A and B are the two halves of one random split.
</details>

**q2 (§3.1).** Why does the score weight the four cell means equally instead of
pooling trials?

<details><summary>Answer</summary>

With unequal trial counts per cell (75 % incongruent blocks have few congruent
trials), a trial-weighted contrast lets the frequent cells dominate and leaks
main effects into the interaction. Equal cell weights make LWPC and the
congruency main effect orthogonal contrasts of the four cell means.
</details>

**q3 (§3.1).** With per-electrode splits, about what fraction of electrode *i*'s
half-A trials are also in electrode *j*'s half B? And with a shared split?

<details><summary>Answer</summary>

About 50 % (a quarter of all trials): *j*'s half B is an independent random
half of the same trials. With a shared split, 0 %.
</details>

**q4 (§3.3).** Why does the overlap test pair LWPC from half A with LWPS from
half B, not both from the same half?

<details><summary>Answer</summary>

Trial-to-trial noise in one half enters every score computed from that half, so
two scores from the same trials can correlate through noise alone. Opposite
halves share no trials.
</details>

**q5 (§3.3).** Why does the null permute LWPS within participant, with the same
permutation in every split?

<details><summary>Answer</summary>

Within participant: the test is about electrodes within a participant (the
scores are also centred within participant), so participant-wide differences
cannot create the correlation. The same permutation in every split: it breaks
the LWPC–LWPS pairing of electrodes while keeping each split's structure.
</details>

**q6 (§4.1).** Participant A has 40 electrodes spread over 5 mm in height; B
has 8 electrodes spread over 40 mm. Who weighs more in the pooled slope?

<details><summary>Answer</summary>

B. A participant's weight is the sum of its squared (cleaned) heights, about
*n* × spread²: A ≈ 40 × 25 = 1,000, B ≈ 8 × 1,600 = 12,800.
</details>

**q7 (§4.1).** Why is the random-intercept model's *p* = 0.005 not a
participant-level test?

<details><summary>Answer</summary>

With every predictor centred within participant, the random intercept only
absorbs participant means; the slope's standard error still treats electrodes
as independent, so the model reproduces the electrode-level test. Only the
random slope lets participants' slopes differ (*p* = 0.037).
</details>

**q8 (§4.2).** The weighted participant-level overlap is significant
(*p* = 0.031), the unweighted one is not (*p* = 0.35). What should the paper
say?

<details><summary>Answer</summary>

That the overlap is a property of the electrode population, carried by
electrode-rich participants, not something every participant shows.
</details>

**q9 (§4.3).** In local similarity, what does the diagonal `C[i, i]` estimate?

<details><summary>Answer</summary>

Electrode *i*'s split-half reliability (its own half A against its own half B).
</details>

**q10 (§4.3).** The balance shows no near-range excess. Why can't we conclude
it is intermixed?

<details><summary>Answer</summary>

Because the balance has about zero reliable within-participant variation
(−0.04, interval up to 0.03): there is nothing reliable to be patchy or
intermixed. "Intermixed" would need a reliable balance with no near-range
excess while the single scores show one.
</details>

**q11 (§4.3).** Why flip the signs of whole participants' excesses instead of
shuffling pairs?

<details><summary>Answer</summary>

One data set's estimation noise is spatially smooth, so pairs are not
exchangeable; a pair-level null gave 13 % false positives at α = 0.05 on
simulated data, the participant sign-flip 2.5 %.
</details>

**q12 (§4.4).** In Fig. 5b (LWPC on *x*, LWPS on *y*), what does a point above
the identity line mean?

<details><summary>Answer</summary>

The electrode leans LWPS: its LWPS exceeds its LWPC in SD units. LWPC − LWPS is
the point's signed distance below the line, so the gradient is a shift across
the line and the overlap a spread along it.
</details>

**q13 (§4.5).** Partialling the same-half base effects drops the overlap from
0.097 to 0.028. Which reading does this row alone not rule out?

<details><summary>Answer</summary>

That both adaptations scale with residual signal-to-noise: the base effects are
also the best available proxy for it. The process-specific matched > crossed
pattern (Test 1) argues against signal-to-noise being all of it.
</details>

**q14 (§4.5).** Why take the base effects from the same half as the adaptation
score they clean?

<details><summary>Answer</summary>

So the two sides of the correlation still share no trials: LWPC half A is
cleaned with half-A base effects, LWPS half B with half-B base effects.
</details>
