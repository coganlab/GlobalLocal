# Segregation analysis, line by line

**What this is.** A line-by-line guide to
`src/analysis/stats/stability_flexibility_segregation.py` (the A1/A2 statistics:
do LWPC and LWPS live on the same electrodes?), written for someone who wants to
know what every step computes and *why it is there*. How to run the job on the
cluster, its knobs and its outputs are in [`analysis_guide.md`](analysis_guide.md)
§14; the manuscript text is in [`methods.md`](methods.md).

**The companion notebook**, `src/analysis/stats/stability_flexibility_segregation_tutorial.ipynb`,
has the same section numbers. It plants a small data set with known answers
(5 subjects × 8 electrodes), picks one demo electrode, and runs the body of
each function one statement at a time, printing every intermediate array. After
each unrolled function it calls the real one and asserts the two agree, so if
the module changes the notebook says which step is stale. The numbers quoted
below come from that notebook.

**References** are `sfs:line` into `stability_flexibility_segregation.py` at the
commit that added this guide. Search for the function name if lines have moved.

---

## 1. The question, and the map

Two constructs, each a 2×2 **interaction** on single-trial high gamma (HG):

- **LWPC (stability)** = congruency × incongruent proportion.
- **LWPS (flexibility)** = switch type × switch proportion.

Each is scored per electrode as a **difference of differences** (d-o-d),
LOW-proportion block minus HIGH:

```
LWPC = (i − c | 25 % incongruent) − (i − c | 75 % incongruent)
LWPS = (s − r | 25 % switch)      − (s − r | 75 % switch)
```

Positive means the condition effect shrinks in the high-proportion block, the
direction behaviour shows. This is a sign *convention* (set in one place,
`_CONTRAST_PRESETS`, `sfs:158`); every test is two-sided and the ANOVA flags are
unsigned.

**Two tests, which fail differently:**

| | Continuous | Categorical |
|---|---|---|
| Per electrode | signed LWPC score *x*, LWPS score *y* | flags S (LWPC-selective), F (LWPS-selective) |
| Across electrodes | within-subject correlation of *x* and *y* | per-subject 2×2 of S × F, pooled by CMH |
| Segregation | *r* ≤ 0 | MH odds ratio < 1 |
| Shared core | *r* > 0 | MH odds ratio > 1 |
| Weakness | a null *r* means nothing unless both maps are reliable | thresholding loses power; per-electrode SNR and shared trial noise inflate "both" |

**Four ways to fake a correlation between *x* and *y*, and the fix for each:**

| Fake | Fix | § |
|---|---|---|
| shared trial noise (x and y from the same trials) | disjoint trial halves, correlation taken within a split | 5, 8, 10 |
| shared gain / SNR (a strong contact is big on everything) | regress both on responsiveness, mean\|HG\| | 6, 7 |
| design imbalance (unequal cells leak a main effect) | equal cell weights | 4 |
| subject nesting (patients differ in level) | centre and permute within subject; stratify the CMH | 7, 8, 12 |

**The call tree** of `run_joint_distribution_analysis` (`sfs:1788`):

```text
long table: one row per (electrode, trial)                         §2
│
├── continuous arm
│   compute_sensitivities_per_split → xA xB yA yB per electrode × split    §5
│   │   ├── finalize_contrasts, _canonical_labels   trial → 0/1 labels     §3
│   │   ├── _stratified_half_split                  disjoint halves        §5
│   │   └── _effect_for → _interaction_effect       balanced d-o-d score   §4
│   average_over_splits + add_responsiveness → x, y, resp (for plots)     §5–6
│   split_resolved_corr → PRIMARY r, p, reliabilities                    §7–8
│   prepare_continuous + subject_clustered_corr → diagnostic only         §10
│
└── categorical arm
    per_electrode_labels → S/F: within-electrode permutation + FDR         §11
    cmh_conjunction → per-subject 2×2 → MH odds ratio                     §12
```

`per_electrode_anova_labels` (§11b) is the parametric A1 definition used by the
`anova_conjunction` job, A3 anatomy and A4 decoding; the conjunction functions
(§12–13) accept its output unchanged.

---

## 2. The input table

One row per (electrode, trial):

| column | meaning |
|---|---|
| `subject` | subject id |
| `electrode` | `"{subject}-{channel}"`, unique across subjects |
| `trial` | epoch index within the subject, shared by all its electrodes (needed by `shared_split`) |
| `congruency` | `'c'` / `'i'` |
| `switchType` | `'s'` / `'r'` (first-of-block `'n'` trials are dropped when the table is built) |
| `incongruent_proportion`, `switch_proportion` | the block's proportions, 25.0 / 75.0 |
| `hg` | window-mean HG (a float), or with `effect_measure='cluster'`/`'peak_t'` the trial's HG time course over the window (a 1-D array in an object column) |

Built by `assemble_long_df` in `dcc_scripts/stats/stability_flexibility_segregation_dcc.py`
(notebook §15). Within a block the share of incongruent (switch) trials equals
the block's proportion, so the four cells of each interaction are deliberately
unequal: per electrode in the notebook's design, 144 trials in (i, 75 %) and
(c, 25 %) and 48 in (i, 25 %) and (c, 75 %). That imbalance is why §4 exists.

---

## 3. Step 1: trial labels from the contrast spec

The contrasts are data, not code. `_CONTRAST_PRESETS` (`sfs:158`) holds two
presets:

```python
'condition':  stability   = congruency  i vs c
              flexibility = switchType  s vs r
'proportion': stability   = interaction(cond = congruency i vs c,
                                        mod  = incongruent_proportion low vs high)
              flexibility = interaction(cond = switchType s vs r,
                                        mod  = switch_proportion low vs high)
```

The battery uses `'proportion'` (analysis_guide §14.1 explains why the constructs
are the interactions, not the main effects).

**`resolve_contrasts`** (`sfs:184`) returns a deep copy of a preset (or of an
explicit `contrasts` dict). The copy matters: the next step writes into the spec,
and without it the module-level preset would be mutated for every later call.

**`finalize_contrasts` → `_finalize_simple`** (`sfs:214`, `sfs:197`) resolves the
`'low'`/`'high'` sentinels to numbers. It converts the column to numeric, takes
the finite values from the **whole table**, and stores `_hi = max`, `_lo = min`
(75.0, 25.0). Computing them once, from all electrodes, means every electrode is
split at the same thresholds even if one lacks a level. Nothing hard-codes 25/75.

**`_group_masks`** (`sfs:227`) turns one simple spec into two boolean masks over
trials. Its `resolve(target)`:

- `'high'` → `num >= _hi`; `'low'` → `num <= _lo`;
- a list/tuple/set → `np.isin`;
- a scalar → numeric closeness if the column parses as numbers, else `==`
  (so `'i'` on the congruency column is a plain equality).

**`_contrast_membership`** (`sfs:254`) combines the cond and mod masks into the
±1 "super-groups" of a 2×2 interaction:

```python
pos = (cp & mp) | (cn & mn)      # (i, low) and (c, high): +1
neg = (cp & mn) | (cn & mp)      # (i, high) and (c, low): −1
```

These are attached as `_slab`/`_flab`, but the interaction is **not** scored by
pooling `pos` against `neg` (§4.5 shows why).

**`_canonical_labels`** (`sfs:305`) is what every entry point calls. It adds one
float column per label, `1.0` = pos, `0.0` = neg, `NaN` = excluded:

| column | for | 1.0 means |
|---|---|---|
| `_slab`, `_flab` | stability / flexibility super-group | +1 cell |
| `_scond`, `_smod` | LWPC sub-factors | incongruent; 25 %-incongruent block |
| `_fcond`, `_fmod` | LWPS sub-factors | switch; 25 %-switch block |

From here on nothing needs to know column names or level values. The two cross
interactions of §11b (CPS, SPC) are just these sub-labels recombined
(`_scond` × `_fmod`, `_fcond` × `_smod`).

**`_strata_columns`** (`sfs:274`) lists the raw factor columns of both contrasts
(congruency, incongruent_proportion, switchType, switch_proportion). §5 splits
trials within each of their 16 combinations.

---

## 4. Step 2: scoring one electrode

`_effect_for(frame, key, labcol, contrasts, effect_measure, alpha, w=None)`
(`sfs:635`) is the dispatcher every caller uses:

- interaction spec → `_interaction_effect(hg, cond, mod, ...)` on the two
  sub-label columns (`_scond`/`_smod` or `_fcond`/`_fmod`);
- simple spec with `BALANCE_MAIN_EFFECTS` (`sfs:632`) → `_interaction_effect`
  with `w=W_MAIN`, balanced over the *other* contrast's levels (§4.6);
- otherwise the old two-group `_contrast_effect`.

**`_interaction_effect`** (`sfs:590`) drops trials with a NaN label, builds the
cells, refuses time courses under Cohen's *d* (`_require_scalar_hg`, `sfs:457`),
and dispatches on `effect_measure`.

### 4.1 `_dod_cells` (`sfs:505`)

The four `(cond, mod)` cells as arrays: `(1,1)` = incongruent in a 25 % block,
`(0,1)` = congruent in a 25 % block, `(1,0)` and `(0,0)` the same in 75 % blocks.
Returns `None` if any cell is empty (the score is then NaN).

### 4.2–4.3 `_cell_stats`, the weights, `_combine` (`sfs:525`, `sfs:519`, `sfs:535`)

`_cell_stats` returns each cell's mean, variance and *n*, or `None` if any cell
has fewer than 2 trials. The contrast is a weighted sum of the four **cell
means**:

```python
W_INTERACTION = {(1,1): +1, (0,1): −1, (1,0): −1, (0,0): +1}
#   = (m11 − m01) − (m10 − m00) = (i − c | low) − (i − c | high)        → LWPC
W_MAIN        = {(1,1): +.5, (0,1): −.5, (1,0): +.5, (0,0): −.5}
#   = ½ [(m11 − m01) + (m10 − m00)]                                     → congruency
```

Each cell counts once, however many trials it holds. The two weight vectors are
orthogonal, so the interaction score is unaffected by the main effect and vice
versa, whatever the cell counts.

### 4.4 `_interaction_cohens_d` (`sfs:539`)

```python
ssq = Σ_k (n_k − 1) var_k                    # pooled within-cell sum of squares
sp  = sqrt(ssq / Σ_k (n_k − 1))              # pooled within-cell SD
d   = _combine(means, w) / sp
```

Standardising by the within-cell SD puts every electrode on a noise-SD scale,
so an electrode with large raw HG variance does not dominate the correlation.
On the notebook's demo electrode: d-o-d = 1.289, pooled SD = 1.089, score =
1.184 (planted d-o-d 1.475 HG units, standard error 0.26).

### 4.5 Why equal cell weights

A pooled contrast (all +1 trials vs all −1 trials, one Cohen's *d*) weights each
cell by its trial count. If the +1 group is, say, 16 incongruent + 48 congruent
trials and the −1 group 144 incongruent + 48 congruent, the pooled difference is
about −½ × (i − c): the congruency **main effect** leaks into the
"interaction". In the notebook, cutting the 25 %-incongruent blocks to a third
gives a pooled "interaction" of −0.39 on average on 20 electrodes whose true
interaction is 0; the balanced d-o-d gives −0.03.

A leak that scales with a per-electrode property (each electrode's own main
effect) turns into an across-electrode correlation, which is the quantity under
test. The price of equal weights is variance: the rare cell counts as much as
the frequent ones. That is the right trade, because noise is measured by the
reliabilities (§8) and leakage is not. The ANOVA route (§11b) gets the same
protection from sum coding + Type III sums of squares.

### 4.6 Main effects (`W_MAIN`, `BALANCE_MAIN_EFFECTS`, `main_effects=True`)

- `contrast_mode='condition'`: stability *is* the congruency main effect. With
  `BALANCE_MAIN_EFFECTS = True` it is scored as `W_MAIN` over the four
  congruency × switch-type cells, i.e. the mean of the congruency effect on
  switch trials and on repeat trials. Pooling instead would let a switch effect
  leak into the congruency score whenever incongruent trials are
  disproportionately switch trials. The comment at `sfs:610` gives simulated
  correlations under a true null: pooled +0.58 (naive) / +0.44 (within-split),
  balanced −0.21 / −0.07.
- `contrast_mode='proportion', main_effects=True`: also scores congruency and
  switch with `W_MAIN` on the **same** four cells and halves as LWPC/LWPS
  (`mxA`…`myB`), so the main-effect correlation is directly comparable.

### 4.7 Time-resolved scores (`effect_measure='cluster'` / `'peak_t'`)

`hg` holds each trial's time course; the cells are (n, T) arrays and the same
weights apply per time bin.

- **`_cell_weighted_t`** (`sfs:552`): per bin, `t = Σ w·mean / sqrt(Σ w²·var/n)`.
- **`_interaction_cluster`** (`sfs:564`): keep bins with |t| above the two-sided
  `alpha` critical *t* (df = N − 4) and sum their signed *t*. There is **no
  contiguity requirement** and no cluster-level correction: it is a thresholded
  signed integral used as a per-electrode *statistic*. Inference happens one
  level up (within-electrode permutation, then FDR across electrodes).
- **`_interaction_peak_t`** (`sfs:576`): the signed *t* at the bin with the
  largest |t|. Amplitude only, so an effect is not rewarded for lasting longer;
  run it next to `cluster` to check the verdict does not depend on duration.

The two-group versions (`_cluster_effect`, `_peak_t_effect`, `sfs:364`, `sfs:415`)
serve condition mode with `BALANCE_MAIN_EFFECTS = False`; only `_cluster_effect`
can use the real `time_perm_cluster` (`USE_TIME_PERM_CLUSTER`), which is a
two-condition test and does not apply to a four-cell interaction.

---

## 5. Step 3: disjoint trial halves

### 5.1 `_stratified_half_split` (`sfs:674`)

```python
for _, cell in sub.groupby(strata_cols, dropna=False):   # each of the 16 strata
    idx = cell.index.to_numpy().copy()
    rng.shuffle(idx)
    cut = len(idx) // 2
    h1.append(idx[:cut]); h2.append(idx[cut:])
```

Splitting inside each stratum keeps all eight LWPC/LWPS cells populated in both
halves, in the same proportions (an odd count leaves the extra trial in half B).
The rarest strata in the demo have 4–6 trials. A d-o-d needs ≥ 2 trials per
cell per half, so ≥ 4 per cell before splitting; electrodes below that get NaN
scores on some split and are dropped in §7 (with a warning above 10 %).

### 5.2 `compute_sensitivities_per_split` (`sfs:745`)

For every electrode and split *k*, score **both** contrasts on **both** halves:

```python
xA = _effect_for(g1, 'stability',   ...)     xB = _effect_for(g2, 'stability',   ...)
yA = _effect_for(g1, 'flexibility', ...)     yB = _effect_for(g2, 'flexibility', ...)
```

One `rng = default_rng(seed)` drives the whole loop, so each electrode and split
gets its own halves and the table is reproducible from `seed`.

Why both halves and not one (as the older `compute_sensitivities`, `sfs:689`,
does with a coin flip): with all four, the correlation can pair `xA` with `yB`
*and* `xB` with `yA` (symmetric use of the data), and `xA` vs `xB` gives the
split-half reliability for free (§8).

### 5.3 Shared halves (`shared_split=True`, `_shared_trial_halves`, `sfs:722`)

By default electrode *i*'s half A shares about half its trials with electrode
*j*'s half B (53 % in the demo). That is fine for the primary test, which only
pairs halves of the **same** electrode. Analyses that pair one electrode's half
with **another's** (local similarity, within-subject reliabilities in N4) need
`shared_split=True`: one split per subject and repetition, keyed on `trial`,
used by all of the subject's electrodes, so every half A is disjoint from every
half B. The table then carries `split_scheme = 'participant'`.

### 5.4 `average_over_splits` (`sfs:826`)

`x = mean over splits of (xA + xB)/2`, same for *y*. This is what the scatter and
`electrodes.csv` show. It is **not** the inference (§10).

---

## 6. Step 4: responsiveness (`add_responsiveness`, `sfs:1212`)

A contact that sees the signal strongly scores high on every contrast. With
independent tuning but varying gain, *x* and *y* still rise together. In the
notebook's gain world (independent amplitudes, planted *r* = −0.04): subject-
centred scores give *r* = +0.43; residualised on responsiveness, +0.03.

`resp` is either the caller's `{electrode: value}` (the cluster job's
`RESPONSIVENESS`, ideally a baseline-vs-task statistic) or **mean |HG|** over
all trials (and time bins). It must measure gain, not the effects: |mean HG|
moves with the condition effects themselves, so regressing *x* and *y* on it
regresses them on their own sum and pushes the residuals apart, a fake
segregation (docstring: −0.12 to −0.21 under a true null, vs −0.07 to +0.03 for
mean|HG|).

The regression is linear and gain multiplies, so a very wide gain spread leaves
a small residual bias; a better drive measure helps more than a fancier model.

---

## 7. Step 5: residualise and centre (`_residualised_split_matrices`, `sfs:953`)

Turns `per_split` into one **(splits × electrodes)** matrix per score. Shared by
`split_resolved_corr`, `participant_split_corr` and `electrode_split_corr`, so
all three see the same numbers.

1. **Merge `resp`, drop rows with a missing score** (`sfs:973–980`).
2. **Keep electrodes defined on every split** (`sfs:987–989`). The permutation
   in §8 needs a fixed electrode set. An electrode whose score is NaN on even
   one split (a cell emptied by that split) is dropped; `n_electrodes_dropped`
   reports how many.
3. **Drop subjects with fewer than `min_elec` electrodes** (`sfs:990`). A
   within-subject correlation over 1–2 electrodes is meaningless.
4. **Bookkeeping** (`sfs:1008–1020`): sorted electrode ids, each one's subject,
   `groups` = the column indices of each subject, and the matrices filled split
   by split.
5. **Residualise and centre** (`sfs:1022–1046`), per split and per score:

   ```python
   r = mats[k][si] - fitted(OLS of mats[k][si] on [1, resp])   # _ols_resid, one fit over ALL electrodes
   for g in groups:
       r[g] -= r[g].mean()                                     # centre within subject
   ```

   - **One OLS over all electrodes**: responsiveness is an electrode property and
     one slope estimated from everyone is the most stable. `covariates=` (e.g.
     MNI coordinates, centred within subject first) join the same design;
     `half_covariates=` adds per-split columns from the *same half* as the
     score they adjust, so the two sides of the correlation still share no
     trials.
   - **Per split and per score**: the halves must never mix.
   - **Centre within subject**: the estimand is "within a patient, are
     electrodes high on LWPC also high on LWPS?". A patient whose electrodes
     are all strong must not create the correlation alone. This matches the
     within-subject null of §8.

---

## 8. Step 6: the split-resolved correlation (`split_resolved_corr`, `sfs:852`) — primary

```
S     = mean_k ½ [ corr(xA_k, yB_k) + corr(xB_k, yA_k) ]     co-localisation
rel_x = mean_k     corr(xA_k, xB_k)                          LWPC map reliability
rel_y = mean_k     corr(yA_k, yB_k)                          LWPS map reliability
```

### 8.1 `_unit_vector` (`sfs:841`): a correlation is a dot product

Rank (Spearman), centre, scale to unit length. The correlation of two such
vectors is their dot product.

### 8.2 The matrix trick (`sfs:904–917`)

Stack each score's unit vectors into `U[k]` (splits × electrodes) and form

```python
M = 0.5 * (U['xA'].T @ U['yB'] + U['xB'].T @ U['yA']) / n_splits
S = trace(M)
```

`M[i, j]` is the split-averaged cross-half product of electrode *i*'s LWPC with
electrode *j*'s LWPS. Permuting which electrode's *y* sits next to which *x*
reorders the columns of `U_y`, and ranking, centring and normalising are all
permutation-invariant, so a permuted statistic is `Σ_i M[i, perm[i]]`: one
lookup per electrode instead of recomputing every split's correlation.
`rel_x = mean_k (U_xA[k] · U_xB[k])`, likewise `rel_y`.

### 8.3 The null (`sfs:919–926`)

```python
for i in range(n_perm):
    perm = identity.copy()
    for g in groups:
        perm[g] = g[rng.permutation(len(g))]    # shuffle y within each subject
    null[i] = M[identity, perm].sum()
p = (#{|null| >= |obs|} + 1) / (n_perm + 1)     # two-sided; never exactly 0
```

The same permutation is used in every split, so it breaks the x–y electrode
pairing while leaving each split's internal structure intact. Within-subject
shuffling keeps between-subject structure, matching the within-subject
centring.

### 8.4 The noise ceiling and how to read the output

- **Read the reliabilities first.** They say how well each map agrees with
  itself across halves. If both are clearly positive, *r* ≈ 0 is evidence the
  two effects load on different electrodes. If either is ≈ 0, *r* ≈ 0 says only
  that the map was not measured well enough to correlate with anything.
  When a reliability is ≤ 0 the function returns `reliability_note` instead of
  a corrected value (often because within-subject centring at small electrode
  counts removed most of the between-electrode variance).
- `corr_noise_corrected = S / sqrt(rel_x · rel_y)`: the correlation of
  noiseless maps. Numerator and denominator are both half-length estimates, so
  this is the right attenuation correction; do not Spearman–Brown the
  reliabilities first. `p` applies to it too (the denominator is a fixed
  positive constant).

Notebook: *r* = −0.40, *p* = .006, reliabilities 0.82 / 0.80, noise-corrected
*r* = −0.50 against the planted −0.5.

---

## 9. Step 7: participants as the unit

The pooled test of §8 treats electrodes as the units: a subject with many
electrodes counts more.

**`participant_split_corr`** (`sfs:1125`): one split-resolved correlation per
subject over its own electrodes (ranked within the subject), Fisher-*z*
transformed, combined with weights *n* − 3 (at least 1), tested by flipping the
sign of whole subjects; also an unweighted one-sample *t* on *z*. `min_elec`
defaults to 4. With *S* subjects there are only 2^S sign patterns, so the
smallest possible two-sided sign-flip *p* is about 2/2^S: with 5 subjects it is
≈ 0.06 however consistent they are (notebook: all 5 negative, *p* = 0.064).
`n_positive` is often the more honest summary at small *S*.

**`electrode_split_corr`** (`sfs:1051`): the pooled *r* and *p* of §8 plus 95 %
intervals from resampling electrodes within subject (each subject keeps its
count), re-centring and re-ranking per draw, for *r*, both reliabilities, their
difference, and the noise-corrected *r* (only when both reliabilities are
positive in ≥ 90 % of draws).

---

## 10. Step 8: the diagnostic estimators

**`prepare_continuous`** (`sfs:1247`) + **`subject_clustered_corr`** (`sfs:1260`):
the same residualise → centre → within-subject permutation, applied once to a
single (x, y) per electrode. The orchestrator feeds them the split-averaged
scores and returns the result as `correlation_split_averaged`.
**`naive_sensitivities`** (`sfs:1197`) scores *x* and *y* on all trials.

**Why the halves must stay separate through the correlation.** A score is a
weighted sum over trials, `x = Σ_t w_x(t) hg(t)` with `w_x(t) = W[cell(t)] /
n_cell(t)`. Its error is `Σ_t w_x(t) noise(t)`, so for iid noise the error
correlation of *x* and *y* from the same trials is `cos(w_x, w_y)`. That cosine
depends only on the subject's trial labels, which all of its electrodes share,
so every electrode's error pair is tilted the same way and the across-electrode
correlation inherits it. On disjoint halves the cosine is exactly 0. Averaging
the half-scores over splits *before* correlating rebuilds *x* and *y* from
(almost) all trials and brings the tilt back.

In the notebook's crossed, balanced design the cosines are small (−0.06 to
+0.14), so naive (−0.48) and split-averaged (−0.48) are close to the truth and
the split-resolved *r* (−0.40) is the smallest: it uses half the trials per
score and is attenuated accordingly, which the noise ceiling measures and the
noise-corrected *r* (−0.50) undoes. Real designs (dropped trials, condition
sequences correlated with block type, condition mode with congruency
correlated with switch type) need not be so kind. The split-resolved estimator
trades a measurable, correctable attenuation for the removal of a bias you
cannot measure.

`mixedlm_check` (`sfs:1281`) fits `y1 ~ x1` with a subject random intercept as a
subject-weighted cross-check.

---

## 11. Step 9: categorical arm — electrode labels

### 11a `per_electrode_labels` (`sfs:1291`) — used by `run_joint_distribution_analysis`

Per electrode, a within-electrode permutation *p* for each contrast, then BH-FDR
across electrodes, then `S = q_cong < alpha`, `F = q_switch < alpha`.

For an interaction, `perm_p_interaction` (`sfs:1320`):

```python
obs = _interaction_effect(h, cond, mod, ...)               # the §4 score
blocks = [np.where(cond == cv)[0] for cv in (1.0, 0.0)]    # trials of each condition level
for _ in range(n_perm):
    mp = mod.copy()
    for idx in blocks:
        mp[idx] = mod[rng.permutation(idx)]                # shuffle block label WITHIN condition
    cnt += abs(_interaction_effect(h, cond, mp, ...)) >= abs(obs)
p = (cnt + 1) / (n_perm + 1)
```

Shuffling the modulator within each condition level keeps the four cell counts
and the condition main effect fixed and nulls only the interaction; a free
shuffle would let a main effect pose as an interaction under unequal cells. The
source comment records two caveats: the block main effect is not held fixed
(the null is a little wide: conservative), and block proportion is a block-level
variable, so trials within a block are not truly exchangeable in it
(anticonservative on that axis; permuting the condition within block would be
the stricter scheme).

The smallest attainable *p* is 1/(n_perm + 1); with few permutations and many
electrodes, BH can only pass electrodes if many share that minimum. Use the
cluster default (2,000).

**BH-FDR** (`multipletests(..., 'fdr_bh')`): sort the *p*-values, multiply the
*k*-th smallest by *m*/*k*, take the running minimum from the top, cap at 1.
NaN *p*-values are filled with 1 first, so a degenerate electrode stays in the
denominator rather than inflating everyone else's significance.
`fdr_correction='none'` flags on raw *p* for threshold-sensitivity runs.

### 11b `per_electrode_anova_labels` (`sfs:1480`) — the parametric A1 definition

Per electrode and per interaction, `_anova_interaction_stats` (`sfs:1413`) fits

```python
smf.ols('hg ~ C(cond, Sum) * C(mod, Sum)', data).fit()  →  anova_lm(model, typ=3)
```

and reads the interaction row's *F* and *p*. Sum coding + Type III make the
interaction row orthogonal to both main effects under unequal cells. In a full
2×2 model the interaction *F* is exactly *t*² of the balanced d-o-d of §4 with
standard error `sqrt(MSE · Σ 1/n_cell)` (the notebook checks this), so the two
routes test the same contrast, one parametrically and one by permutation.

Four interactions per electrode, FDR'd separately across electrodes:

| flag | interaction | alias |
|---|---|---|
| `CPC` | congruency × incongruent proportion | `S`, LWPC |
| `SPS` | switch type × switch proportion | `F`, LWPS |
| `CPS` | congruency × switch proportion | cross |
| `SPC` | switch type × incongruent proportion | cross |

The cross groups should be near-empty in univariate HG (specificity control) and
are named groups so A4 can identify the circular decode cell (analysis_guide
§14.1). The flag is two-sided; `<g>_sign` (from the balanced Cohen's *d*) records
direction for reporting only. Time-course tables are reduced to window means
(`_scalar_hg`, `sfs:1395`) so the *F* and the sign describe the same statistic.
`S`/`F` and the old `p_cong`/`q_switch`/… columns are aliases, so the table drops
into `cmh_conjunction` unchanged. With `contrast_mode='condition'` the same
function labels main effects (`anova_model='twoway'` balances each over the other
and adds the congruency × switch interaction as `CXS`).

---

## 12. Step 10: the conjunction (`cmh_conjunction`, `sfs:1642`)

Per subject, the 2×2 of S × F:

```python
a = both;  b = S only;  c = F only;  e = neither
```

**Stratify by subject.** Pooling all electrodes into one table lets a subject
with many S *and* many F electrodes create an association on its own (Simpson's
paradox). Cochran–Mantel–Haenszel keeps one table per subject.

**Drop uninformative strata** (`_informative_stratum`, `sfs:1637`): a subject
with no S electrodes (or no F, or all S, or all F) has a zero margin and cannot
speak to the association. With `shift_zeros=True`, statsmodels adds 0.5 to every
cell of any table containing a zero, which would turn `[[0,0],[c,e]]` into a
spurious positive association (analysis_guide §14.5 gives the size: up to
OR = 51 at a threshold where nothing was selected). If no stratum is informative
the OR is NaN, not a number.

**The statistics** (after the 0.5 shift on tables with a zero cell):

```
MH OR    = Σ_k (a_k d_k / n_k) / Σ_k (b_k c_k / n_k)
E[a_k]   = (a_k + b_k)(a_k + c_k) / n_k
Var(a_k) = (a+b)(c+d)(a+c)(b+d) / (n² (n − 1))
CMH χ²   = (Σ_k (a_k − E[a_k]))² / Σ_k Var(a_k),   1 df
```

OR < 1: fewer "both" electrodes than each subject's margins predict
(segregation); OR > 1: more (shared core). `homogeneity` (Breslow–Day) asks
whether the OR differs between subjects. `pooled_table` and the pooled Fisher
test ignore subjects and are descriptive only. Notebook: OR = 0.12 (planted
0.11), CMH *p* = .003.

---

## 13. Step 11: count null and threshold sweep

**`conjunction_permutation_null`** (`sfs:1724`): shuffle F within each subject
(each subject's S and F counts fixed, only the pairing random) and count "both".
It is the CMH's own null made explicit. *p* is two-sided around the null mean,
with *z* for scale. Notebook: 6 observed vs 11.0 expected, *z* = −3.0.

**`conjunction_threshold_sweep`** (`sfs:1754`): the stronger effect recruits more
electrodes at a fixed alpha, so a verdict should hold across cutoffs. The labels
come from a callable `threshold → labels`, so any selector works (ANOVA *q*,
permutation *q*, effect-size percentiles). Read `n_informative_strata` on every
row: at the strict end selection runs out and the OR becomes NaN by design.

---

## 14. The orchestrator (`run_joint_distribution_analysis`, `sfs:1788`)

```python
per_split = compute_sensitivities_per_split(df, n_splits, ...)            # §5   seed 0
elec = add_responsiveness(average_over_splits(per_split), df, resp)      # §5–6
corr = split_resolved_corr(per_split, resp, ...)                         # §7–8 seed 1, PRIMARY
cont = prepare_continuous(elec); corr_avg = subject_clustered_corr(cont) # §10  diagnostic
labels = per_electrode_labels(df, ...)                                   # §11a seed 2
conj = cmh_conjunction(labels)                                           # §12
```

| key | contents | § |
|---|---|---|
| `correlation` | primary: *r*, *p*, reliabilities, noise-corrected *r*, `n_electrodes_dropped` | 8 |
| `correlation_split_averaged` | diagnostic | 10 |
| `per_split` | xA, xB, yA, yB (and mxA… with `main_effects=True`) per electrode × split | 5 |
| `electrodes` | split-averaged x, y, `resp`, and S/F | 5–6, 11 |
| `continuous` | residualised, centred x, y behind the diagnostic | 10 |
| `labels`, `conjunction` | categorical arm | 11–12 |
| `main_effect_correlation` | with `main_effects=True`: §8 on congruency vs switch, same halves | 4.6 |

**Falsification.** The notebook plants the opposite world (six "both"
electrodes per subject) and the verdict flips: *r* = +0.65, OR = 30.

---

## 15. On real data

`dcc_scripts/stats/stability_flexibility_segregation_dcc.py: main` builds the
long table (`assemble_long_df`: cut the window from each subject's
`HG_ev1_rescaled` epochs, average it or keep the time course, drop
first-of-block trials, prefix channels with the subject), optionally removes the
RT-linked part of HG (`RT_ADJUST_HG`), calls `run_joint_distribution_analysis`,
and writes the tables, three figures and `summary.txt`. Knobs:
analysis_guide §14.6. Output files: `stability_flexibility_battery.md` ›
Outputs guide.

---

## 16. Check yourself

The notebook ends with ten questions with folded answers (why LOW − HIGH, why
equal cell weights, what `xB` is, why `xA` pairs with `yB`, why correlate before
averaging, what responsiveness removes, why within-subject, how to read a null
*r* with one low reliability, why the CMH drops uninformative strata, and how a
strong continuous result can sit beside a weak CMH).
