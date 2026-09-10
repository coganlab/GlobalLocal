# Analysis simplification plan — stability/flexibility shared vs. independent mechanisms

Working plan produced from a diagnostic pass over the decoding and power-trace
pipelines (2026-09). Records what the diagnosis found, what to do instead, and
why. Companion to `analysis_guide.md` (which describes the pipelines as built);
this document argues for changing which of them is primary.

**Scientific question.** Do stability adaptation (LWPC) and flexibility
adaptation (LWPS) rely on shared or independent neural mechanisms in lPFC?

---

## TL;DR

1. **The decoding pipeline, as constructed, cannot answer the question.** It
   builds synthetic pseudotrials by sampling each channel's trials
   independently, which destroys the trial-level cross-electrode covariance that
   would make it multivariate. What remains is close to a weighted univariate
   analysis wearing MVPA clothing.
2. **The analysis that does answer the question already exists in this repo**:
   `src/analysis/stats/stability_flexibility_segregation.py`, run via
   `run_joint_distribution_analysis(..., contrast_mode='proportion')`. Make it
   primary.
3. **Four changes were needed before it was trustworthy; all four are now
   implemented.** Two were planned: the split aggregation is fixed (§2.2 — the
   disjoint-half correction was being forfeited by averaging the estimates
   before correlating them) and a split-half noise ceiling is reported (§2.3 —
   without it a null result is uninterpretable; with it, a null becomes positive
   evidence for independence). Two more turned up while testing those, both
   invisible to the split because they bias the *signal* rather than the noise:
   main-effect contrasts are now cell-balanced (§2.2b), and the default
   responsiveness proxy no longer double-counts the effects it is meant to
   control for (§2.2c). Under a simulated true null the estimator moved from
   −0.25 … +0.62 to roughly ±0.07. One item remains open: the categorical arm
   still scores main effects the old way (§2.2b, "What is still open").
4. **Leave the power traces alone.** Their structure is fine; the pain we found
   is specific to the pseudopopulation, which they don't use.
5. **Retire the block-context accuracy comparisons.** They are confounded by
   trial count and SNR, and they were never a direct test of shared mechanism.

---

## Part 1 — Diagnostic findings

Each finding is tagged with how confident to be: **[verified]** = read in the
code and traced; **[derived]** = arithmetic from verified values; **[open]** =
needs checking before relying on it.

### 1.1 The pseudopopulation destroys cross-electrode covariance — [verified]

`labeled_array_utils.py:822-826`, inside the constructor that
`process_bootstrap` actually calls:

```python
for channel in all_channels_in_roi:
    channel_data = nan_removed_data_dict[condition_name][channel]
    sample_indices = rng.choice(len(channel_data), size=n_samples, replace=False)
    resampled_channels_for_condition.append(channel_data[sample_indices])
concatenated_chans = np.stack(resampled_channels_for_condition, axis=chans_axs)
```

Each channel draws its own trial indices. Pseudotrial row *i* is therefore a
chimera — channel A's trial 17 beside channel B's trial 4 — **even for two
electrodes in the same patient recorded simultaneously**.

What survives: each channel's condition-dependent mean and variance, and the
temporal structure within that channel's selected trial.

What is destroyed: simultaneous trial-by-trial covariance between electrodes,
coordinated population fluctuations, trial-level network states.

**Why this is the decisive finding.** A linear classifier on data with no
cross-channel covariance can only exploit each channel's marginal tuning. Its
information content is therefore close to that of a weighted sum of
per-electrode effects — which is what the power traces already measure, minus
the weighting. It is a legitimate form of pseudopopulation decoding, but it
answers "do electrodes collectively have condition-dependent tuning?" rather
than "does the trial-level population state carry the information?" Only the
second question justifies MVPA over a univariate contrast.

This also explains, retrospectively, why the block-split decoding accuracies
tracked trial count and SNR so closely: there was little genuine multivariate
signal for them to track instead.

### 1.2 Trial counts are set by the worst channel, and differ across the cells being compared — [verified/derived]

`subsample_to_min_trials_per_condition` (`labeled_array_utils.py:761`) takes the
minimum valid trial count **across all channels in the ROI** for each condition.
One bad electrode caps the trial count for the whole ROI in that cell.
Conditions then land at different heights, `LabeledArray.from_dict` pads them to
a common height with NaN, and `data_prep.py` strips the padding — which is why
the `[NaN filter]` log lines report "79.4% dropped" for rare cells. That figure
is padding removal, not artifact rejection.

Because the surviving count per stratum is a property of the dataset rather than
of the run, the final n for any block-restricted split can be reconstructed by
replaying the balancing (`data_prep.py` steps 3–4) over the per-stratum counts
printed in the run logs. Doing so reproduces four actual runs exactly, so the
two unobserved cells below are trustworthy:

| contrast | block | final n/class | limiting cell |
|---|---|---|---|
| c vs i | MC (25% I) | 22 | `i_MC_MS` (11) |
| c vs i | MI (75% I) | 30 | `c_MI_MR` (15) |
| c vs i | MR (25% S) | 26 | `i_MC_MR` (13) |
| c vs i | MS (75% S) | 22 | `i_MC_MS` (11) |

Consequences for the four decoding panels:

- **Congruency × switchProportion** — MR = 26 vs MS = 22. The observed effect
  (25% S decodes better) runs *the same direction as the n difference*. That
  panel also carries a significant **pre-stimulus** cluster, in a window where
  congruency information cannot exist. Treat as a trial-count artifact.
- **Congruency × incongruentProportion** — MC = 22 vs MI = 30. The observed
  effect (25% I decodes better) runs *against* the n difference, so it survives
  this confound and is if anything understated.

### 1.3 The parameter surface — [verified]

~20 parameters materially change the result before counting the optional
electrode-selection modes; ~45 with them. Two specific problems:

- `N_SHUFFLE_PERMS = 50` bounds the smallest resolvable p at 1/51 ≈ 0.02, while
  inference runs at one-tailed α = 0.025. That is at the resolution floor.
- `PERCENTILE`/`CLUSTER_PERCENTILE` (both 95) and `P_THRESH`/`P_CLUSTER` (both
  0.025) express two decisions in four parameters.

Feature count is also extreme: 174 electrodes × 64 samples = **11,136 features**
per pseudotrial against 22–30 observations per class. PCA is not optional at
that ratio, which makes `EXPLAINED_VARIANCE` consequential rather than cosmetic.

### 1.4 The baseline carries a block-level confound — [verified]

`make_epoched_data.py:195` draws a random 0.5 s baseline window from within
−1.0–0 s; line 322 rescales with `mode='zscore'` using statistics **pooled
across all trials** per channel. So tonic block-level differences in HG survive
into the pre-stimulus window by construction. Since `incongruentProportion` *is*
the block in a blocked design, a sustained pre-stimulus separation is the
expected result, not a finding.

The asymmetry in the power traces confirms it: the switchType panel (varies
*within* block) shows no pre-stimulus cluster; the incongruentProportion panel
(block-level) shows one spanning the entire baseline. A single bad electrode
would contaminate both.

**Recommended control, not replacement:** per-trial baseline mean subtraction
with a pooled SD (see §2.4). Run it as a robustness check, because you cannot
determine from these data whether the tonic block difference is nuisance or is
itself the proactive-control effect — that is undecidable in a blocked design.

### 1.5 Corrections — do NOT act on these earlier suggestions

Recorded so they aren't repeated:

- **"Uncomment the trial shuffle at `labeled_array_utils.py:236-240`."** Wrong.
  That code sits in `process_single_subject_labeled_array`, which
  `process_bootstrap` does not call. Trial sampling in the active path is
  already random. The fix would be a no-op.
- **"Surviving trials are each subject's chronologically first *k*."** Wrong,
  and it followed from the same misidentified path. Sampling is random
  (`rng.choice`).
- **"The minimum is across subjects."** It is across *channels* — stricter.
- **"Bootstraps are redundant with `N_REPEATS`."** Wrong. `N_REPEATS` re-splits
  the same trials into folds; `BOOTSTRAPS` changes which trials enter. They
  measure different variance components.
- **"Decode per subject instead of pooling."** Too aggressive for this dataset.
  lPFC electrode counts are heavily skewed and a per-subject decoder on 1–2
  electrodes sits at chance. The plan below keeps electrodes pooled and moves
  the *inference* unit to subject instead.

---

## Part 2 — The plan

### 2.1 Primary analysis: joint-distribution segregation (already implemented)

`src/analysis/stats/stability_flexibility_segregation.py`, entry point
`run_joint_distribution_analysis`, with **`contrast_mode='proportion'`**.

This is **A2-continuous** in the existing battery. What it computes and why the
estimator is shaped the way it is are already documented — do not re-derive them
here:

| For | Read |
|---|---|
| what A2 does, step by step | `stability_flexibility_data_flow.md` §3b, §8 |
| how it relates to (and differs from) RSA | `stability_flexibility_data_flow.md` §10 |
| manuscript-ready Methods prose | `stability_flexibility_segregation_methods.md` |
| where it sits among A1–A7 | `stability_flexibility_data_flow.md` §11 |

This document adds only the argument for making it **primary**, plus two defects
found in the implementation (§2.2, §2.3).

Why it is the right primary test:

- **It targets the adaptation question directly.** LWPC and LWPS are
  interactions — how the base effect changes with block proportion. That is what
  "stability/flexibility adaptation" means. Accuracy-in-context-A vs.
  accuracy-in-context-B is an indirect proxy for the same thing, mediated by
  signal, noise, trial count, and estimator behaviour.
- **It recovers what the ROI mean throws away — without needing trial-level
  covariance.** The ROI mean cancels when some electrodes push positive and
  others negative; asking whether the two effects land on the *same electrodes,
  in the same direction* does not. Note the vocabulary: this is **not**
  multivariate — each electrode's sensitivity is a scalar and there is no
  pattern dimension (`stability_flexibility_data_flow.md` §10 is right about this, and the
  representational-geometry question lives in A4). Its virtue here is precisely
  that it needs no within-trial cross-electrode structure, so §1.1 does not
  touch it.
- **It already handles two confounds that would otherwise dominate** — shared
  trial noise via disjoint halves, shared gain/SNR via responsiveness
  residualisation. *(As originally implemented the shared-noise correction did
  not survive the aggregation; fixed — see §2.2.)*
- **Inference already respects subjects** — within-subject centering plus
  within-subject permutation (`subject_clustered_corr`), and CMH stratification
  for the categorical conjunction. This is the piece the decoding pipeline never
  had: CV folds, repeats and bootstraps are resampling replicates, not
  independent biological observations, and treating ~25 accuracy curves as n=25
  overstates evidence badly.

Interpretation:

| result | reading |
|---|---|
| corr > 0 | shared / domain-general core |
| corr ≈ 0 | no evidence of a common spatial pattern |
| corr < 0 | segregated subpopulations |
| both effects reliable, corr ≈ 0 | adaptation happens in lPFC via spatially distinct electrode sets |

That last row is precisely what an ROI-mean power trace cannot reveal, and it is
the scientifically interesting outcome.

**Verified in code** (`compute_sensitivities_per_split`, `_stratified_half_split`,
`split_resolved_corr`): the half-split is stratified on the contrast cells and
genuinely disjoint; both cross directions are used, so the halves enter
symmetrically; the permutation null shuffles y within subject, preserving
between-subject structure. All as documented. The aggregation problem found in
the original implementation is fixed — see §2.2, and §2.2b for a further bias
found while testing that fix.

### 2.2 The split aggregation — **fixed**

**Status: implemented.** `compute_sensitivities_per_split` + `split_resolved_corr`,
wired into `run_joint_distribution_analysis` as the primary `correlation`.

**The bug.** `compute_sensitivities` averaged x over the 200 splits and y over
the 200 splits, then returned one `(x, y)` per electrode which
`subject_clustered_corr` correlated. That aggregation forfeited the
disjoint-half correction entirely.

Within one split *k*, `x_k` and `y_k` come from disjoint trial sets, so their
sampling noise is independent — which is the whole design. But the average
contains K·(K−1) cross terms `cov(x_j, y_k)` with *j ≠ k*, computed on trial
sets that overlap by ~50%. Those terms do not vanish and they swamp the K
disjoint ones.

How completely they swamp them is the striking part. Across three simulated
regimes, the split-averaged estimator and the naive same-trial estimator agree
to three decimal places:

| regime | naive (all trials) | split-averaged (old) | within-split (new) |
|---|---|---|---|
| A | −0.2777 | −0.2780 | −0.2074 |
| B | +0.6153 | +0.6157 | +0.4866 |
| C | +0.0591 | +0.0609 | +0.0535 |

200 splits bought nothing at all. (Regimes defined in §2.2b; truth is 0 in all
three.)

**The fix.** Correlate *within* each split, then average the correlations:

```
S = mean_k  ½ [ corr(x_A,k , y_B,k) + corr(x_B,k , y_A,k) ]
```

Both cross directions are used, so the halves enter symmetrically and the coin
flip in the old code is unnecessary. Residualisation on responsiveness and
within-subject centring are applied *per split*, matching `prepare_continuous`.
The permutation null uses one within-subject electrode permutation applied
identically across all splits. Because each split's vectors are centred and
unit-normed, a correlation is a dot product, so the whole thing collapses to a
lookup in one precomputed n_elec × n_elec matrix — 10,000 permutations cost
O(n_elec) each rather than O(n_splits × n_elec).

**What the fix does and does not buy.** Isolated in simulation (`bx = by = 0`,
so the *only* coupling is shared sampling noise, with a non-proportional 2×2
cross-tab so that noise actually covaries):

| | naive | within-split |
|---|---|---|
| pure shared-trial noise | −0.46 … +0.40 | +0.002 … +0.068 |

That is the mechanism the disjoint split was designed for, and it is removed
essentially completely. It is *not* the only mechanism — see §2.2b.

**A correction to an earlier version of this section.** It said the bias
"appears when the cells are unbalanced." That is imprecise. Unequal *marginals*
(25%/75%) are harmless on their own: what matters is whether the 2×2 cross-tab
is **proportional**. Writing the per-trial contrast weights

```
w_x(t) = +1/n_i if incongruent else −1/n_c
w_y(t) = +1/n_s if switch      else −1/n_r
```

the shared-noise covariance of the two effects is `σ² · Σ_t w_x(t) w_y(t)`,
which is exactly zero when `n_is = n_i·n_s/N` and so on, however lopsided the
marginals are. Simulated with independent 25%/75% factors, all estimators sit at
0 and there is nothing to fix. The bias needs congruency and switchType to be
*correlated in the trial table*. **Check this in your own data before assuming
either way** — compute the four cell counts per subject and test the cross-tab
for proportionality. That number, not the marginals, says how much of this
mattered.

### 2.2b Cell-balanced main effects — a third bias the split cannot remove

**Status: implemented** (`BALANCE_MAIN_EFFECTS`, `W_MAIN`). Found while testing
§2.2; not in the original plan.

Testing the §2.2 fix against a true null in three regimes turned up two further
couplings, both of which survive the disjoint split untouched, because both are
**signal** confounds rather than noise confounds — they are present identically
in every trial and therefore in every half.

- **Regime B — design non-orthogonality.** If congruency and switchType are
  correlated in the trial table, an electrode with a purely congruency-driven
  response still scores a switch effect, because its incongruent trials are
  disproportionately switch trials. Every electrode inherits the same leakage,
  which is precisely what a spurious across-electrode correlation is made of.
- **Regime A — the shared pooled SD.** Cohen's *d* divides by an SD estimated
  from the same trials. A large stability effect inflates the within-group
  variance that the flexibility contrast divides by, deflating it — a *negative*
  coupling, which is the direction that would masquerade as segregation. (Regime
  A turned out to have a second, larger contributor as well; see §2.2c.)

Both are fixed at the contrast level, not the split level. Scoring a main effect
as the equal-weight mean of the within-cell differences makes the two contrasts
orthogonal in cell-mean space *by construction*, whatever the cell counts:

```
x = ½[(m_is − m_cs) + (m_ir − m_cr)]        # congruency, equal weight over switch
y = ½[(m_cs − m_cr) + (m_is − m_ir)]        # switch, equal weight over congruency
```

Simulated under a true null, ~330 electrodes, 12 subjects:

| regime | scoring | naive | within-split |
|---|---|---|---|
| B (non-orthogonal design) | pooled two-group (old) | +0.58 | +0.44 |
| B | cell-weighted (new) | −0.21 | **−0.07** |
| A (shared pooled SD) | pooled two-group (old) | −0.23 | −0.18 |
| A | cell-weighted (new) | −0.09 | **−0.09** |
| C (clean control) | either | ≈ 0 | ≈ 0 |

Neither change alone suffices: cell-weighting removes the signal-mediated
coupling, the split removes the noise-mediated coupling, and only together do
all four regimes land near zero.

**Note this only applies to `contrast_mode='condition'`.** The interaction path
(`contrast_mode='proportion'`, the manuscript's primary mode) has always used an
equal-cell-weight difference-of-differences, and is protected already. This
extends the same treatment one level down, to main effects.

Also tested and **rejected**: dropping the denominator entirely (a raw
cell-balanced mean difference). It fixes regime B equally well but is markedly
*worse* in regime A (−0.25 … −0.29 vs −0.09), because without standardisation
the sensitivities scale with per-electrode gain and the linear responsiveness
residualisation does not fully remove a multiplicative gain term. Keep the
pooled within-cell SD.

### 2.2c The responsiveness proxy was a function of the effects — **fixed**

**Status: implemented** (`add_responsiveness`). A plain bug, found by chasing the
residual regime-A bias after §2.2b.

`prepare_continuous` residualises *x* and *y* on `resp` to remove the shared
gain/SNR confound. That only works if `resp` measures **gain** and nothing else.
The scalar-HG branch computed `|mean HG|` — the absolute value of the
electrode's mean — while the docstring, and the time-resolved branch, said
`mean |HG|`. Those are very different quantities. When both contrasts push the
electrode's mean the same way (which is what a population-level effect *means*),
`|mean HG|` behaves like `x + y`. Regressing *x* and *y* on their own sum drives
the two residuals apart, so the correction manufactures a **negative**
correlation — spurious *segregation*, the more publishable direction.

Regime A, naive estimator, three seeds:

| responsiveness proxy | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `\|mean HG\|` (scalar branch, as written) | −0.120 | −0.209 | −0.161 |
| `mean \|HG\|` (as documented; now both branches) | +0.025 | −0.066 | −0.020 |
| true per-electrode gain (oracle) | +0.058 | −0.052 | +0.012 |

The documented proxy lands level with the oracle. Both branches now compute
`mean |HG|`.

This only affects runs that used the **default** proxy. `add_responsiveness`
takes an explicit `responsiveness=` argument and the guide already recommends
passing a baseline-vs-signal cluster statistic; runs that did so are unaffected.

Taken together over regime A, the three fixes compose:

```
−0.25   original
−0.12   + cell-weighted main effects  (§2.2b)
+0.03   + corrected responsiveness proxy  (§2.2c)
```

**What is still open.** `per_electrode_labels` — the categorical (A3) arm —
scores simple contrasts through `_effect_from_arrays`, not `_effect_for`, so it
still uses the old pooled two-group form and is still exposed to the regime-B
leakage. It was left alone deliberately: changing it would change the S/F label
counts and the conjunction table, i.e. published numbers, and its
within-electrode permutation null is a separate design with its own documented
caveats. The consequence is an asymmetry worth knowing about when reading
`electrodes.csv`: `x`/`y` are now cell-balanced, `S`/`F` are not. Resolve before
the categorical arm is used for anything load-bearing in `condition` mode.

### 2.3 Split-half noise ceiling — **added**

**Status: implemented.** `split_resolved_corr` returns `reliability_x`,
`reliability_y` and `corr_noise_corrected`; the launcher writes them to
`correlation.json` and prints them in `summary.txt` with an explicit
"a null corr is NOT evidence of independence" warning when either reliability is
low.

The trials are already split into disjoint halves A and B to estimate the two
sensitivities. From those same halves, also compute:

```
r_LWPC = corr(β_LWPC,A , β_LWPC,B)     # is the stability pattern reliably measured?
r_LWPS = corr(β_LWPS,A , β_LWPS,B)     # is the flexibility pattern?
S_corrected = S / sqrt(r_LWPC × r_LWPS)
```

**Why it is essential.** Without it, `S ≈ 0` cannot be distinguished from
"neither pattern is measured well enough to correlate with anything." With it,
high `r_LWPC` and `r_LWPS` alongside `S ≈ 0` is *positive evidence* that the
patterns are reliable **and** unrelated — which is the only honest route to an
"independent mechanisms" claim (see Part 6).

This matters more, not less, under the recommended anatomical electrode set
(§2.6), because including mostly-unresponsive electrodes attenuates `S` toward
zero and the ceiling is what makes the attenuated number readable.

Cost: it came free. Computing each contrast on *both* halves rather than one
(four effect evaluations per split instead of two) yields the reliabilities
directly and removes the need for the coin flip. The only price is 2× the effect
evaluations, which matters solely for `effect_measure='cluster'`, where each
evaluation runs its own permutation test — halve `n_splits` there.

`p` is reported for `S`; it applies to `S_corrected` unchanged, since the two
differ only by a fixed positive denominator and so order the null identically.

One operational note: an electrode whose effect is undefined on even a single
split (a contrast cell emptied by that split) is dropped from the correlation
entirely, so that the per-split vectors are comparable and the permutation
applies to a fixed electrode set. `n_electrodes_dropped` reports how many. With
few trials per cell this can bite; check it before interpreting `n_electrodes`.

### 2.4 Supporting: power traces — keep as they are

No structural change needed. The pseudopopulation is a *decoding-specific*
requirement: decoding needs one trial carrying a value on all 174 channels
simultaneously, so it must synthesise them. Power traces average within each
electrode first, so heterogeneous trial counts never have to line up and none of
§1.1–1.2 applies.

Their role in the paper: establish that each adaptation exists in mean activity.
That is a different and complementary claim from whether they share a pattern.

Two changes worth making anyway:

1. **Re-run with per-trial baseline** as a robustness check (§1.4). Subtract
   each trial's own baseline mean, keep a pooled per-channel SD:

   ```python
   base = HG_base._data                                       # (trials, ch, base_times)
   per_trial_mean = np.nanmean(base, axis=-1, keepdims=True)
   pooled_sd = np.nanstd(base - per_trial_mean, axis=(0, -1), keepdims=True)
   HG_ev1_rescaled._data = (HG_ev1._data - per_trial_mean) / pooled_sd
   ```

   Mean-centre before taking the SD so the denominator describes the same
   quantity the numerator now contains (within-trial fluctuation), otherwise
   each channel's normalised baseline lands on a different scale set by how much
   tonic drift it happened to carry. Note Response epochs have fewer trials than
   the baseline array (434 vs 448 for D0121), so align on `TrialCount` metadata
   rather than position.

   Report the post-stimulus effect *with and without*. Do not read the flattened
   pre-stimulus window as evidence — the baseline is drawn from inside it, so
   flatness there is near-tautological.

2. **Report effect sizes alongside cluster presence.** The current
   double-dissociation claim rests on two *non-significant* interaction
   clusters, and non-significance is not evidence of specificity.

### 2.5 Supporting: the scatterplot — make this first

Before any inference machinery:

- x-axis: each electrode's LWPC interaction
- y-axis: the same electrode's LWPS interaction
- colour by subject; marginal histograms on both axes

Read it as: positive diagonal → shared; spread on both axes with no
correspondence → independent; spread on one axis only → one mechanism; negative
diagonal → opponent; **all the structure in one colour or a few points →
artifact**.

This is an afternoon's work and will tell you most of the answer before you
commit to anything. It also gives reviewers a direct view of the data instead of
asking them to trust a pipeline.

### 2.6 Electrode set: anatomical, not condition-selected

Use one anatomically defined lPFC set with recording-quality exclusions only.

Do **not** pre-select LWPC-significant, LWPS-significant, their union, or their
intersection before measuring co-localization — selecting on the effects whose
overlap you are about to test makes the overlap partly a property of the
selection rule. The current decoding runner defaults to `ELECTRODES='sig'` with
`ELECTRODE_DEFINITION_SPLIT` off, and offers several further selection modes;
none of them belong upstream of this analysis.

The bias this avoids, and the nested-selection machinery for cases where you
*must* select, are worked out in `nested_electrode_selection.md` — see "Where
the bias sits" and "The null must run selection too". The recommendation here is
simply to sidestep it: an anatomical set needs no nested selection at all.

Trade-off: including unresponsive electrodes attenuates `S`. That is
conservative and acceptable — and it is exactly why §2.3 is not optional.

### 2.7 Timing

You can keep timing without the decoding machinery. `S` is a scalar per time
window, so compute it in sliding windows and plot a time-resolved
shared-pattern curve. Correcting over time for one scalar is far simpler than
cluster-correcting differences between accuracy curves.

For the primary *inferential* claim, use one pre-specified window and show the
time course descriptively: "we show the full time course for completeness;
statistical inference was performed on a pre-specified window." If you want an
actual timing claim ("stability adapts earlier than flexibility"), onset latency
with a bootstrap CI on the difference is a sharper instrument than a cluster bar.

### 2.8 Optional confirmatory: minimal cross-decoding

This is **A4** in the existing battery, and it is already specified in depth —
designs, the double-dipping guard, the within-block 2×2, temporal generalization
— in `stability_flexibility_data_flow.md` §5, implemented in
`dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py`. Read those
first; this section only says what to strip out and one framing problem.

Only if reviewers expect MVPA. Keep it small: per-subject, one mean value per
electrode in the fixed window (no time samples as separate features), shrinkage
LDA (`solver='lsqr', shrinkage='auto'`, no PCA, no hyperparameter search),
**leave-one-block-out CV**, subject as the unit of inference, 1000+ permutations
within valid exchangeability strata.

Leave-one-block-out matters specifically: random trial splits within a block let
the classifier exploit block-level drift — the same tonic confound as §1.4.

**Caveat on framing.** Training i-vs-c and testing s-vs-r asks whether
*congruency* and *switch type* share a coding axis. That is **not** the
adaptation question — a region could code conflict and switching on a shared
axis while their block-level adaptations are independent. Cross-decoding
adaptation directly is awkward in principle, because an interaction is not a
trial-level class label; that awkwardness is itself a reason the
pattern-correlation approach is the better primary instrument. Label this
analysis honestly as the base-effect question, or omit it.

---

## Part 3 — What to retire

From the primary result:

- cross-subject synthetic pseudotrials and independent per-channel sampling
- PCA explained-variance threshold
- 250 ms sliding windows at 62.5 ms steps (75% overlap) as an inferential object
- block-context accuracy comparisons (`C/I (25% S) > C/I (75% S)` etc.)
- folds / repeats / bootstraps as inferential sample size
- the multiple electrode-selection variants
- two-stage percentile + cluster-percentile procedures
- accuracy plots carrying several different significance bars

Remaining major decisions, all statable in a short Methods paragraph:
anatomical ROI · time window · baseline definition · minimum trial/electrode
criterion · similarity metric · permutation scheme.

---

## Part 4 — Open questions to resolve first

1. **What does `ELECTRODES='sig'` actually mean?** [open] If significance was
   defined on the same trials, contrast, or window subsequently analysed, any
   downstream effect is partly circular. If it is a condition-independent
   responsiveness criterion, the problem is much smaller. Resolve before
   reporting anything conditioned on it. (§2.6 sidesteps this for the primary
   analysis by using anatomical electrodes.)
2. **Which subjects cap the trial counts, and how many electrodes do they
   contribute?** A subject contributing 1–2 of 174 electrodes while capping a
   cell at 11 is worth excluding; one contributing 8 is not. Needs a
   leave-one-subject-out sweep on the real per-subject counts (the electrode
   distribution in `sig_electrodes_per_subject_roi.json` is stale — 44
   electrodes / 12 subjects vs. the 174 actually used).
3. **Does the incongruentProportion decoding crossover survive per-trial
   baselining?** It is the one decoding result that ran against its trial-count
   confound, so it is worth one clean check before discarding the decoding
   results wholesale.

---

## Part 5 — Order of operations

Within the A1–A7 sequence of `stability_flexibility_data_flow.md` §11 this is a
re-prioritisation, not a new pipeline: A2-continuous is promoted to primary, A4
demoted to optional confirmation.

1. **Make the §2.5 scatterplot.** Cheapest, most informative, no new machinery,
   and it tells you most of the answer before any inference.
2. ~~Fix the §2.2 split aggregation~~ — **done.** Correlation is now computed
   per split and averaged.
3. ~~Add the §2.3 noise ceiling~~ — **done.** Reliabilities and the
   noise-corrected estimate come back in `correlation.json` and `summary.txt`.
4. **Check the 2×2 cross-tab in your own data** (§2.2, end). Per subject, the
   four congruency × switchType cell counts, tested for proportionality. This is
   a few lines and it tells you how much §2.2b was actually buying — cheap, and
   it should be in the Methods either way.
5. **Run** `run_joint_distribution_analysis(..., contrast_mode='proportion')` on
   anatomical lPFC electrodes. Read `reliability_x`/`reliability_y` *before*
   reading `corr`: if either is near zero, the correlation is uninterpretable
   whatever it says, and the electrode set or the effect measure is the problem
   to fix first. Also check `n_electrodes_dropped` (§2.3).
6. Re-run power traces with per-trial baseline as a robustness check (§2.4).
7. Add the time-resolved `S` curve if the timing claim is wanted (§2.7).
8. Only then, if desired, the §2.8 minimal cross-decoder.

Steps 1 and 4–5 are the paper. Everything after is support.

Before the categorical arm is used for anything load-bearing in `condition`
mode, resolve the §2.2b open item (`per_electrode_labels` still scores main
effects the old way).

---

## Part 6 — What not to claim

If pattern similarity or cross-decoding comes out null, the honest statement is:

> We found no evidence that the measured linear HG pattern was shared between
> stability and flexibility adaptation.

**Not:** "the mechanisms are independent." Independence requires positive
evidence — reliable within-domain patterns for *both* effects (this is what the
§2.3 noise ceiling supplies, now reported as `reliability_x`/`reliability_y`),
adequate measurement quality, and a confidence interval excluding a
theoretically meaningful shared-pattern effect. Two significant main effects
plus a non-significant correlation is not a dissociation.

One further caution now that §2.2b is known: a *negative* correlation is the
easiest result to over-read, because two of the three biases documented there
push in that direction. Before reading `corr < 0` as segregation, confirm it
survives on the cell-weighted contrast (it now does by default) and check the
cross-tab proportionality per §2.2 step 4.

The same caution applies to the existing power-trace double dissociation, which
currently rests on two non-significant interaction clusters (§2.4).
