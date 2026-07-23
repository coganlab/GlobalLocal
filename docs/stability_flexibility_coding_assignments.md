# Stability vs. Flexibility — Coding Assignments

A staged implementation plan for the analyses in
`stability_flexibility_analysis_plan.md`. Each assignment is self-contained:
it states **why** it exists, **where** the code goes, the **tasks**, the
**acceptance criteria** (how you know you're done), and **hints** you can
reveal if you get stuck.

> **How to use the hints.** Each hint is a collapsible `<details>` block —
> on GitHub (or any Markdown viewer) click the ▸ triangle to expand it. They
> are ordered from gentle nudge → concrete approach → near-solution. Try the
> task first; only open the next hint when you're genuinely stuck.

**Work order.** A1 → A2 unlock the population-organization layer (§1–§2 of the
plan) and are the natural first pieces because most of the scaffolding already
exists in `src/analysis/stats/stability_flexibility_segregation.py`. A3 (anatomy)
and A6 (behavior) are independent and can slot in any time. A4 (decoding) and A5
(timing) are the two larger, mostly-greenfield pieces; do them once A1/A2 have
given you a trustworthy electrode definition.

| # | Plan § | Deliverable | New or existing code |
|---|---|---|---|
| **A0** | — | Get the existing pipeline running; read the module | existing |
| **A1** | §1 | Per-electrode two-way **ANOVA** electrode definition | new fn in segregation module |
| **A2** | §2 | Conjunction **permutation null** + **threshold sweep** | new wrappers around `cmh_conjunction` |
| **A3** | §3 | Anatomy: brain maps + ROI histograms + coverage-conditioned test | `src/analysis/vis` + new stats fn |
| **A4** | §4 | **Cross-decoding**: pseudo-trials + label/set/temporal transfer | `src/analysis/decoding/decoding.py` |
| **A5** | §5 | **Timing**: interaction time course, 50%-of-peak onset, jackknife | new `stability_flexibility_timing.py` |
| **A6** | §6 | **Brain–behavior** correlation (across- and within-subject) | new fn in stats |

Throughout, keep the **§0 cross-cutting principles** from the plan in view —
double-dipping, disjoint halves, power matching, FDR, coverage bias,
latency–amplitude confound, the tonic baseline, and decoding confounds. Several
acceptance criteria below are just those principles made concrete.

---

## A quick recap of what "segregation" means (so the assignments make sense)

The whole battery answers one question: **are stability (LWPC) and flexibility
(LWPS) carried by the *same* neural machinery or by *distinct* machinery?**
"Segregation" is the *distinct* answer. The segregation module
(`stability_flexibility_segregation.py`) attacks the **population-organization**
layer of that question — are the same *electrodes* selective for both processes,
more or less than you'd expect by chance? — in two complementary ways:

- **Continuous:** correlate each electrode's stability effect size against its
  flexibility effect size. Correlation ≈ 0 (or negative) → the two selectivities
  live on different channels → **segregation**. Positive → a **shared** core.
- **Categorical:** label each electrode stability-selective (S) and/or
  flexibility-selective (F), build the 2×2 (`both / S-only / F-only / neither`)
  per subject, and pool with Cochran–Mantel–Haenszel. Odds ratio **< 1** →
  fewer "both" electrodes than chance → **segregation**; **> 1** → shared.

The full "why do I need this" writeup is in the chat message that accompanies
this doc and in `stability_flexibility_segregation_tutorial.ipynb`. The
one-line version: **the conjunction is the only test in the battery that can
give *positive evidence for distinctness* (OR < 1). Decoding can only fail to
find a shared code, which is weak; segregation can affirmatively show the
populations don't overlap.** A0 has you run it end-to-end before you extend it.

---

## A0 — Warm-up: run the pipeline and read the module

**Why.** Every later assignment either calls into or mirrors
`stability_flexibility_segregation.py`. Get it running and internalize its
shape before you add to it.

**Tasks.**
1. Run the module's synthetic smoke test:
   `python src/analysis/stats/stability_flexibility_segregation.py`. Confirm you
   get a partial correlation and an MH odds ratio printed for both
   `condition` and `proportion` modes and for the `cluster` effect measure.
2. Run the DCC dry run: from `dcc_scripts/stats`,
   `DATA_SOURCE=synthetic bash submit_stability_flexibility_segregation_dcc.sh`
   (or the direct `python run_...` invocation in that folder's README). Find the
   outputs it writes and open `segregation_summary.png`.
3. Read `stability_flexibility_segregation_tutorial.ipynb` alongside the code.
   Write yourself a 5-line summary of what `compute_sensitivities`,
   `per_electrode_labels`, and `cmh_conjunction` each return.

**Acceptance criteria.**
- You can state, for one electrode, exactly how `x` (stability) and `y`
  (flexibility) are computed and why they come from **disjoint trial halves**.
- You can point to where FDR correction across electrodes happens
  (`per_electrode_labels`, `multipletests(..., method='fdr_bh')`).

<details><summary>Hint 1 — where the outputs land</summary>

Results go under
`dcc_scripts/stats/results/<epochs_or_synthetic_tag>/window_<tmin>to<tmax>s_<electrodes>/`
in a `..._<CONTRAST_MODE>_<EFFECT_MEASURE>` subfolder. The DCC README's
"Outputs" section lists every file.
</details>

<details><summary>Hint 2 — the disjoint-half trick in one place</summary>

Look at `compute_sensitivities`: for each electrode it calls
`_stratified_half_split` to cut the trials into two balanced halves, estimates
stability on one half and flexibility on the *other*, then averages over
`n_splits` random splits. That's the whole point — shared trial noise can't
inflate the x–y correlation if x and y never see the same trials.
</details>

---

## A1 — Per-electrode two-way ANOVA electrode definition (plan §1)

**Why.** The plan names the **ANOVA interaction** as the *primary* electrode
definition, and flags that it is "**not yet wired into the pipeline**." The
existing `per_electrode_labels` gives a nonparametric cross-check, but the
parametric Type III ANOVA is the headline definition and the one reviewers will
expect. You need it to produce the S/F flags and the four groups (LWPC-only,
LWPS-only, both, neither).

**Where.** A new function in
`src/analysis/stats/stability_flexibility_segregation.py`, e.g.
`per_electrode_anova_labels(df, alpha=0.05, ...)`, returning a table with the
same shape as `per_electrode_labels` (so it's a drop-in alternative into
`cmh_conjunction`).

**Tasks.**
1. For each electrode, fit a two-way ANOVA on window-mean HG:
   - **LWPC model:** `hg ~ C(congruency) * C(incongruent_proportion)`.
   - **LWPS model:** `hg ~ C(switchType) * C(switch_proportion)`.
2. Use **Type III** sums of squares (`anova_lm(model, typ=3)`), which requires
   sum/effect coding of the factors so the interaction is orthogonal to the main
   effects — this is the parametric analogue of the equal-cell-weight
   difference-of-differences the module already uses (see `_interaction_effect`).
3. Extract, per electrode: the interaction F and p, and the **sign** of the
   difference-of-differences (the F-test is unsigned — you need the sign to keep
   only electrodes whose effect grows in the predicted direction, and to feed the
   §2 continuous correlation).
4. FDR (Benjamini–Hochberg) the interaction p-values **across electrodes**; set
   `S`/`F` from the q-values at `alpha`.
5. Also fit the two **cross** interactions
   (`congruency × switch_proportion`, `switchType × incongruent_proportion`) and
   return their stats as **specificity controls** — report, never select on them.

**Acceptance criteria.**
- Output columns match `per_electrode_labels` where they overlap
  (`subject, electrode, S, F` plus p/q), so `cmh_conjunction(labels)` accepts it
  unchanged.
- On synthetic data (`_synthetic_df(contrast_mode='proportion'...)` ground truth)
  the ANOVA S/F flags **agree closely** with `per_electrode_labels` — they test
  the same balanced interaction. Add an assertion/notebook cell showing the
  agreement (e.g. Cohen's κ or a 2×2 of the two label sets).
- The cross-interaction terms are **near-null** on synthetic data (there is no
  true cross-effect in the generator) — a good sanity check that Type III coding
  killed the main-effect leak.

<details><summary>Hint 1 — the leakage trap this assignment is really about</summary>

The proportion cells are deliberately unequal (~75/25). With default
*treatment* coding, Type III interaction SS is **not** orthogonal to the main
effects, so a pure congruency/switch main effect leaks into the "interaction"
(the plan quotes ~0.8 SD of fake effect in a zero-interaction sim). Effect/sum
coding (`C(congruency, Sum)`) fixes this. This is the exact same problem
`_interaction_effect` solves by weighting the four cells equally.
</details>

<details><summary>Hint 2 — statsmodels skeleton</summary>

```python
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

model = smf.ols(
    "hg ~ C(congruency, Sum) * C(incongruent_proportion, Sum)",
    data=elec_df,
).fit()
aov = anova_lm(model, typ=3)          # Type III
F   = aov.loc["C(congruency, Sum):C(incongruent_proportion, Sum)", "F"]
p   = aov.loc["C(congruency, Sum):C(incongruent_proportion, Sum)", "PR(>F)"]
```
Guard against singular fits (an electrode missing a cell) with try/except →
NaN, exactly like the effect helpers return NaN for too-few trials.
</details>

<details><summary>Hint 3 — getting the sign</summary>

The ANOVA F is unsigned. Reuse the module's own signed estimate: call
`_interaction_effect(hg, cond, mod, 'cohens_d', alpha)` (after building the
`_scond`/`_smod` labels via `_canonical_labels`) and take its sign. That keeps
your sign definition identical to the one the continuous correlation consumes,
so the two analyses can't disagree on direction.
</details>

---

## A2 — Conjunction permutation null + threshold sweep (plan §2)

**Why.** `cmh_conjunction` returns the MH odds ratio and its parametric tests,
but the plan calls for (a) a **within-subject permutation null** on the overlap
count (respects the subject nesting better than the hypergeometric) and (b) a
**threshold sweep** so a segregation claim is shown to be stable across α, not
an artifact of one cutoff.

**Where.** New wrappers in the segregation module, e.g.
`conjunction_permutation_null(labels, n_perm=10000, seed=...)` and
`conjunction_threshold_sweep(df, thresholds, ...)`.

**Tasks.**
1. **Permutation null.** Shuffle the `S` and `F` labels **within each subject**
   independently and recompute the observed "both" count each time → empirical
   null distribution. Report observed overlap vs. null with a two-sided p-value.
2. **Threshold sweep.** Recompute the MH odds ratio (and the overlap count)
   across a range of selection thresholds (q-value cutoffs, or effect-size
   percentiles) and return a tidy table you can plot OR-vs-threshold.
3. Wire the ANOVA labels from A1 in as the label source, and confirm the
   continuous correlation (`subject_clustered_corr`) tells the same story as the
   categorical OR at the sweep's midpoint.

**Acceptance criteria.**
- The permutation null preserves each subject's marginal S and F counts (only the
  *pairing* is shuffled) — verify by asserting per-subject `S.sum()` and
  `F.sum()` are invariant across permutations.
- On synthetic data with **independent** bx/by (the generator draws them
  independently), the null-based p is non-significant and the OR ≈ 1 across the
  whole sweep — i.e. the test is not manufacturing segregation or overlap.
- The sweep plot is monotone-ish and doesn't flip sign of the conclusion across
  reasonable thresholds; if it does, that's a finding to report, not to hide.

<details><summary>Hint 1 — why within-subject shuffling</summary>

Global label shuffling would break the subject structure and inflate
significance (Simpson-type pooling). Permuting S vs F pairing *within* each
subject holds each subject's selectivity rates fixed and only tests whether S
and F **co-occur** on the same electrodes more/less than chance — the exact null
CMH assumes. It mirrors the within-subject permutation already used in
`subject_clustered_corr`.
</details>

<details><summary>Hint 2 — reuse, don't rebuild</summary>

You already have `cmh_conjunction(labels)`; the sweep is just a loop that
re-derives `S`/`F` at each threshold and calls it. For the null, permute
`labels['F']` within `groupby('subject')` and recount
`((S==1)&(F==1)).sum()` — no need to touch the CMH machinery for the count-based
null (though you can also recompute the pooled OR per permutation for a second,
OR-based null).
</details>

---

## A3 — Anatomy: brain maps, ROI histograms, coverage-conditioned test (plan §3)

**Why.** "Are the distinct subpopulations in *different places*?" This is the
descriptive anatomical layer (Greg's plots) — necessary support, and the piece
most exposed to the **coverage-bias** trap: iEEG coverage is clinical, so an ROI
difference can just reflect *where electrodes are*.

**Where.** Plotting in `src/analysis/vis/` (reuse the existing brain-figure
machinery, e.g.
`brain_figure_glasser_separate_svgs_lateral_medial_view_less_bold.py`); the
coverage-conditioned test as a small stats function (in the segregation module
or a new `stability_flexibility_anatomy.py`).

**Tasks.**
1. Join the A1 labels to each electrode's ROI (use
   `make_or_load_subjects_electrodes_to_ROIs_dict` from `utils/general_utils.py`
   and `config/rois.py`).
2. Plot LWPC-only, LWPS-only, and both electrodes on the brain, color-coded by
   group. Produce ROI-membership histograms per group.
3. **Coverage-conditioned test.** Restrict to ROIs sampled in ≥ *k* subjects,
   then test whether group membership (S-only vs F-only vs both) is associated
   with ROI via a permutation / χ² on the counts — reporting per-subject coverage
   alongside so the reader can see the difference isn't pure placement.

**Acceptance criteria.**
- Every anatomical claim is conditioned on coverage: the test excludes ROIs
  below the *k*-subject threshold and you report how many subjects cover each ROI.
- The brain figure legend distinguishes all four groups (incl. neither, or note
  its omission).

<details><summary>Hint 1 — don't reinvent the brain plot</summary>

The vis module already renders ROI-highlighted Glasser surfaces to SVG via MNE +
PyVista. Feed it your per-group electrode lists as the highlight sets rather than
writing new surface code. `analysis_paths.md` §10 points at the exact file.
</details>

<details><summary>Hint 2 — the coverage condition in practice</summary>

Build a `subject × ROI` coverage matrix (does subject *s* have any electrode in
ROI *r*?). Keep ROIs with column-sum ≥ *k*. Run the χ²/permutation only on
electrodes in those ROIs, and permute group labels *within subject* so the null
respects nesting and coverage simultaneously.
</details>

---

## A4 — Cross-decoding: pseudo-trials + label/set/temporal transfer (plan §4)

**Why.** Co-localization ≠ shared *code*. This is the representation-level test
that disambiguates the "both" group: a shared representational geometry vs.
mixed selectivity with orthogonal codes. It's the largest piece and the one the
counting analyses can't do.

**Where.** Build on `src/analysis/decoding/decoding.py` (the `Decoder` class and
its CV machinery). Add the cross-condition train/test split and the pseudo-trial
construction; a thin orchestrator can live in a new
`dcc_scripts/decoding/` runner or a notebook.

**Tasks (do in this order — each is a checkpoint).**
1. **(0) Within-block decoding baseline (Fig 9).** Decode inc/con within
   mostly-congruent vs mostly-incongruent blocks (and switch/repeat within
   mostly-repeat vs mostly-switch blocks); compare accuracies. This is the
   decoding analog of the univariate LWPC/LWPS effects and where the **neural
   cross-effects** surface.
2. **Pseudo-trials.** Since subjects don't share trials, build pseudo-trials by
   aligning within condition cells (match on
   `congruency × inc_prop × switchType × switch_prop`) across the
   pseudopopulation, keeping **train/test trials disjoint**.
3. **(a) Label-transfer:** train on the stability contrast, test on the
   flexibility contrast (and vice versa) on the *same* electrodes; run separately
   on LWPC-only, LWPS-only, and both groups. Prediction: the *both* group
   cross-decodes; the distinct groups don't.
4. **(b) Electrode-set-transfer:** train on one electrode set, test on the other
   for the *same* label.
5. **(c) Temporal generalization (Fig 10):** train at time *t*, test at *t′*,
   both within and across contrasts; render the train×test matrix.
6. Null by permuting the transferred labels; CV over pseudo-trial folds.

**Acceptance criteria (these are the §0.8 confound controls — non-negotiable).**
- Trial counts matched across blocks before comparing accuracies; RT regressed
  out or matched; the cross-effect confirmed to survive **per-condition mean
  removal** (so it's not a univariate offset the classifier is exploiting).
- **Circularity:** the electrodes/trials used to *define* a group are never the
  ones the transferred (test) accuracy is computed on. Group definition and
  decoding-test-trial selection are independent.
- The label-permutation null is centered at chance for a within-block sanity
  decode you *know* is real.

<details><summary>Hint 1 — reuse the Decoder, add only the split</summary>

`Decoder.cv_cm_jim_window_shuffle` already gives you time-resolved accuracy with
a shuffle distribution and NaN/PCA handling. The genuinely new code is (i) the
pseudo-trial builder and (ii) letting train labels come from one contrast and
test labels from another. Don't rewrite the classifier — wrap it.
</details>

<details><summary>Hint 2 — pseudo-trial construction</summary>

Group trials by the full condition cell (`congruency × inc_prop × switchType ×
switch_prop`). Within a cell, average or sample *m* trials per electrode to form
one pseudo-trial across the pseudopopulation; repeat to make a pseudo-trial pool;
split that pool into disjoint train/test folds. `labeled_array_utils.py` already
has the bootstrapping/downsampling helpers to equalize trial counts — lean on
them.
</details>

<details><summary>Hint 3 — proving it's not a univariate offset</summary>

Before trusting any cross-decode, re-run it after subtracting each condition's
per-electrode mean (per time bin) from every trial. If the transfer accuracy
survives, it's multivariate structure, not a mean shift. Do the same after
regressing HG on RT. Report both the raw and the confound-controlled accuracy.
</details>

---

## A5 — Timing: interaction time course, 50%-of-peak onset, jackknife (plan §5)

**Why.** A *sequence* claim — does stability information arise earlier than
flexibility (or vice versa)? — is an axis neither overlap nor decoding speaks
to. The plan marks §5 as fully specified, so this is a clean, well-bounded
build.

**Where.** A new module `src/analysis/stats/stability_flexibility_timing.py`.
Reuse the time-resolved contrast the segregation module already produces via the
`effect_measure='cluster'` path (per-trial HG time courses →
`_interaction_effect` per bin).

**Tasks.**
1. **Information time course.** For each process, compute the interaction
   magnitude **over time** — the LWPC and LWPS difference-of-differences as
   functions of time (grand average over the relevant electrodes, or a
   t-statistic time course).
2. **Onset = 50%-of-peak.** Find the peak of the effect in the expected
   direction within the window, then take the **first upward crossing of 50% of
   that peak** on the rising flank. Also report **peak latency** as a cross-check.
3. **Jackknife (Miller/Ulrich).** For each leave-one-subject-out subaverage,
   measure LWPC and LWPS onset; compute the jackknife SE from the pseudovalues
   (variance inflated by `(N−1)`) and compare the two onsets with the
   **Ulrich–Miller corrected paired t** (`t_c = t / (N−1)`). Report the paired
   onset *difference* with a CI, not just a p.

**Acceptance criteria.**
- Onset is measured on **smooth leave-one-out grand-averages**, never per-subject
  single-trace crossings (that's the whole reason for the jackknife).
- The 50%-of-peak measure is verified to neutralize a pure amplitude difference:
  in a test where `stab(t) = k·flex(t)`, the two onsets come out **equal**
  (this is the latency–amplitude confound guard — bake it into a unit test).
- You report onset **and** peak latency; a claim rests on both agreeing.

<details><summary>Hint 1 — the confound this defeats, and the test for it</summary>

A bigger effect crosses any *absolute* threshold sooner — that would make
"earlier" just mean "larger." Normalizing to each process's own peak removes a
pure multiplicative amplitude difference. The unit test writes itself: take one
waveform, scale it by `k`, confirm both cross 50%-of-peak at the same sample.
</details>

<details><summary>Hint 2 — Ulrich–Miller in code</summary>

Compute onset on each of the N leave-one-out averages → `onset_lwpc[i]`,
`onset_lwps[i]`. The jackknife SE of a latency is
`sqrt((N-1)/N * sum((d_i - d_bar)**2))` where `d_i` is the per-leaveout onset
*difference*. Run an ordinary paired t on the N difference values, then divide
the t by `(N-1)` (equivalently multiply the SE). That correction is the entire
Miller/Ulrich trick — the raw jackknife t is inflated by `(N-1)`.
</details>

<details><summary>Hint 3 — reuse the cluster path for the time course</summary>

The segregation module's `effect_measure='cluster'` route already scores the
interaction per time bin (`_interaction_cluster` / the per-bin
difference-of-differences t). You can lift that per-bin d-o-d computation to get
the interaction-vs-time trace instead of writing a new time-resolved estimator.
</details>

---

## A6 — Brain–behavior correlation (plan §6)

**Why.** Ties the neural selectivity to the actual behavioral control
adjustment, so the substrates are shown to be *functional*, not incidental.

**Where.** New function in the stats area; behavioral effects come from the
existing behavioral pipeline (`combinedData.csv` /
`erin_linear_mixed_effects_model.py`).

**Tasks.**
1. **Across subjects (low power, n = subjects):** correlate a subject's LWPC/LWPS
   electrode count (or mean effect) with their behavioral LWPC/LWPS magnitude
   (the congruency-sequence and switch-proportion RT effects).
2. **Within subject, single-trial (preferred):** does trial-by-trial HG in the
   LWPC group predict the trial-by-trial congruency-sequence RT adjustment (and
   the LWPS group ↔ switch adjustment)? Use a mixed model with subject random
   effects.

**Acceptance criteria.**
- The across-subject correlation is reported with its n and an honest
  "underpowered at n = subjects" caveat.
- The within-subject model links the *matching* neural group to the *matching*
  behavioral adjustment (LWPC group ↔ congruency-sequence RT; LWPS ↔ switch), and
  you check the *cross* pairing is weaker (a specificity control).

<details><summary>Hint 1 — where the behavioral effects already live</summary>

`stats/erin_linear_mixed_effects_model.py` already fits
`PostErrorRT ~ ... IncongruentProportion + SwitchProportion + (1|Subject)`. The
LWPC/LWPS behavioral magnitudes are contrasts within that same design — extract
per-subject effects from it rather than recomputing RT effects from scratch.
</details>

---

## Cross-cutting checklist (apply to every assignment)

Lift straight from §0 of the plan — tick these before calling any result real:

- [ ] **Double-dipping:** anything reported on the *selection* contrast comes
      from held-out trials (disjoint half) or is labeled descriptive-only. The
      cross-contrast direction (define on LWPC, test LWPS) is the clean workhorse.
- [ ] **Disjoint halves:** selection and test contrasts estimated on disjoint
      trial halves where they could share trial noise.
- [ ] **Power matching:** report counts/effects *as a function of threshold*, not
      one α snapshot (A2 sweep).
- [ ] **FDR** across electrodes for per-electrode selection tests.
- [ ] **Coverage bias:** every anatomical claim conditioned on coverage (A3).
- [ ] **Latency–amplitude:** onset defended by 50%-of-peak + peak latency (A5).
- [ ] **Tonic baseline:** don't baseline-correct across the pre-trial block state
      you're studying; report the tonic vs phasic split.
- [ ] **Decoding confounds:** trial-count/RT matched, survives per-condition mean
      removal (A4).
