# Decoding & Electrode-Definition — Methods Decisions and Troubleshooting

Companion notes to `stability_flexibility_analysis_plan.md`. This file collects
the *design decisions* and *gotchas* that come up when wiring the electrode
definition (§1) into the segregation stats (§2), the anatomy plots (§3), and the
decoding (§4) — the reasoning that is too discursive for the plan's terse
section bodies but that reviewers (and future-you) will ask about.

Cross-references point back to the plan sections they elaborate.

---

## A. Use **one** electrode definition, not several (plan §1)

**The worry.** Defining "significant electrodes" with more than one method is
annoying to write up and invites reviewer comments ("why two definitions? which
is primary? do they agree?"). The instinct to consolidate onto a single
definition is correct.

**Why you can't just point the segregation stats at the current
`power_traces` output.** This is a shape problem, not a style preference:

- `load_significant_electrodes` (in `src/analysis/power/windowed_anova.py`)
  returns a **flat `[(subject, electrode)]` pass/fail list, per effect**.
- The conjunction (`cmh_conjunction`, §2) needs **S and F co-registered on the
  same electrode row**.
- The continuous correlation (§2 headline, Fig 7) needs a **signed, graded
  per-electrode effect size** — you cannot correlate two boolean/F pass-fail
  lists.

`power_traces` emits neither shape. So "reuse it directly" already implies
running it twice (once per effect) and bolting on a signed graded effect — i.e.
writing new glue regardless.

**So the real choice** is *which single definition, and where the code that
emits the segregation-shaped table lives* — not "A1 vs power_traces":

| Option | Single definition | What it costs | Methods sentence |
|---|---|---|---|
| **A (recommended)** | **A1** static window-mean two-way interaction ANOVA (`per_electrode_anova_labels`) | re-point the brain plots at A1's electrode lists — this is assignment **A3**, a small wiring change | "Electrodes were labeled LWPC/LWPS-selective by a per-electrode two-way ANOVA on window-mean high-gamma (Type III, FDR across electrodes)." |
| **B** | **power_traces** windowed/cluster ANOVA | refactor its ANOVA core to emit the `per_electrode_labels`-shaped S/F table **with a signed graded effect**; commit to the cluster-over-time definition everywhere | "…defined by a cluster-corrected windowed two-way ANOVA," + you must describe the cluster params |

**Recommendation: Option A.** `per_electrode_anova_labels` already produces the
S/F + signed-effect table the segregation stats require; the only remaining step
to full consolidation is pointing the brain plots at it (A3, currently
`dcc_scripts/vis/plot_sig_electrodes_dcc.py` still reads the `power_traces`
sig-chans — see plan §1 "which plotting consumes which"). Then `power_traces`
stops being a competing *electrode definition* and becomes your
**temporal-profile figure** (the F-traces — "when does the effect emerge") that
answers a *different* question, so it raises no reviewer conflict.

**What about `per_electrode_labels` (the nonparametric route)?** Don't report it
as a second definition. Keep it — if at all — as a single robustness sentence
("results held under a distribution-free within-electrode permutation test"),
which reviewers reward rather than question.

**Bottom line.** A1 is still necessary, but *not in addition to* a second
electrode definition — it is the thing that lets you have exactly **one**
definition feeding both the plots and the stats.

See also: plan §1 subsection "A1 vs. the `power_traces` windowed ANOVA — same
machinery, different jobs."

---

## B. Cross-decoding baseline leakage — why Figs 3–4 look weird (plan §4)

**The observation.** In the within-block decoding 2×2 (plan §4, step 0), the two
**matched** decodes (LWPC: congruency split by inc-proportion; LWPS: switch type
split by switch-proportion) have **chance baselines** and rise ~0.4–0.5 s after
stimulus onset — correct. The two **cross** decodes (switch type split by
inc-proportion; congruency split by switch-proportion) show significant clusters
that extend **into and before** the pre-stimulus baseline.

**The diagnostic tell.** Congruency is a property of the **current stimulus** —
the subject cannot know whether *this* trial is incongruent until they see it.
So any significant *congruency* decoding at `t < 0` **cannot be a real evoked
code**; it is leakage. Use the congruency baseline (`t < 0`) as an **artifact
meter**: whatever change drives it back to chance is the right fix. The clean
matched baselines vs. contaminated cross baselines localize the problem to the
**rare cross cells**, not the pipeline globally.

**Why the cross cells specifically break** (grounded in the current code):

1. **Random CV folds + slow drift (most likely).** `decoder.py:234` uses
   `StratifiedKFold(shuffle=True)` — folds assigned randomly, ignoring trial
   time / run order. iEEG has strong slow drift and trial-to-trial
   autocorrelation. The minority class of a cross decode (incongruent trials in
   a mostly-congruent block, switch trials in a mostly-repeat block — ~25% by
   design, rarer once the *other* block factor is also conditioned on) tends to
   be **temporally clustered**, so the label correlates with drift / session
   position. Random folds let the classifier learn that drift axis and leak it
   across train/test → **flat, all-timepoint** above-chance accuracy, baseline
   included. The label-shuffle null destroys the label↔drift correlation, so the
   null sits at chance while the true trace rides the drift → spurious
   "significance," even pre-stimulus.
2. **Tiny, min-balanced samples.** `data_prep.py`
   (`concatenate_and_balance_data_for_decoding`) balances by subsampling every
   class down to the **minimum** cell count. The rare cross cell sets that floor,
   so cross decodes run on very few trials → high-variance accuracy against a
   too-tight pooled null.
3. **Sequence carryover.** Switch/repeat and congruency-sequence are defined
   relative to trial *n−1*; the prior trial's conflict/switch leaves a tonic
   trace in the current baseline. This is *legitimate* pre-stimulus information
   for **switch type** (so some of the switch-type cross panel's baseline effect
   may be real cue/carryover), but a **confound** for current-trial
   **congruency** (often via stimulus/response feature repetition).

**Fixes, in priority order.**

- **Time-/run-aware folds** — leave-one-run(block)-out or `GroupKFold` on run id,
  so autocorrelated neighbors never straddle train/test. Highest-value test;
  prediction: the baseline collapses to chance.
- **Baseline-correct the accuracy trace** (or restrict the cluster test to
  `t > 0`) so a sustained flat offset cannot form clusters.
- **Match trial counts** across the two block versions of each cross decode and
  report *n* per cell.
- **Per-condition mean removal** — `remove_condition_means` already exists in
  `src/analysis/decoding/cross_decoding.py`; if the cross decode survives it, the
  effect is multivariate structure; if it vanishes, it was a drift/offset.

If a cross decode still shows a **post-stimulus** effect with a **chance
baseline** after all of this, *that* is a genuine neural cross-effect worth
reporting (the effect §4 step 0 is looking for).

See also: plan §4 step (0) "Observed status / caveat."

---

## C. Circularity between electrode definition and decoding (plan §0.1, §0.2)

**The question.** Should trials be split into one set for *defining significant
electrodes* and a disjoint set for *decoding*, to avoid double-dipping between
electrode selection and decoding accuracy?

**Short answer: yes, whenever decoding is restricted to a selected electrode set
that was selected using the trials you then decode.** This is the §0
"double-dipping" / "disjoint halves" principle made concrete for the decoding
path.

**Is it implemented today? No.** The decoding driver
(`dcc_scripts/decoding/decoding_dcc.py`) selects electrodes like this:

```python
# decoding_dcc.py (~L115–128)
sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, ...)
all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = \
    make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)

if args.electrodes == 'all':
    raw_electrodes = all_electrodes_per_subject_roi     # whole ROI — no selection
elif args.electrodes == 'sig':
    raw_electrodes = sig_electrodes_per_subject_roi     # selected electrodes
```

- `get_sig_chans_per_subject` reads the **precomputed**
  `sig_chans_{subject}_{epochs_root_file}.json`, produced by the upstream stats
  pipeline on the **full trial set**.
- Decoding then runs (CV) on **overlapping** trials.
- There is **no partition** separating the trials used to define electrodes from
  the trials used to decode.

**Severity depends on the selection contrast.**

- **`electrodes='all'`** — no electrode selection at all, so **no
  selection-induced circularity** (at the cost of diluting the signal with
  non-responsive channels). This is the currently-safe option.
- **`electrodes='sig'`, selection contrast *orthogonal* to the decode** — the
  `sig_chans` are a **task-responsiveness** selection (e.g. stimulus vs
  baseline), which is *approximately* orthogonal to the inc-vs-con /
  switch-vs-repeat decode. Selecting on an orthogonal contrast is the standard
  Kriegeskorte-style defense — but it is only *strictly* unbiased when the
  selection statistic is independent of the test statistic under the null.
  Because both share the same trials and noise, high-SNR selection can still
  **modestly inflate** decoding. Defensible if reported as an orthogonal-contrast
  selection, but a reviewer can still object.
- **Selecting on the decode contrast itself** (e.g. the A1 LWPC/LWPS
  interaction electrodes, then decoding the *same* LWPC/LWPS contrast on the same
  trials) — **full double-dipping**. This *must* use disjoint trials (or the
  cross-contrast "define on LWPC, test LWPS" workhorse from §0/§4).

**Recommended fix: a disjoint, within-subject, stratified trial split.**

1. Within each subject, partition trials into `P_def` and `P_dec`, **stratified**
   by the full condition cell (`congruency × inc_prop × switchType ×
   switch_prop`) and by run/block so both halves stay balanced and neither is
   temporally lopsided.
2. **Define electrodes on `P_def` only** — recompute the per-electrode selection
   (A1's `per_electrode_anova_labels`, or the sig-chan test) on `P_def`.
3. **Decode on `P_dec` only**, restricted to those electrodes.
4. Optionally average over repeated random splits (as the segregation module
   already does via `_stratified_half_split`) to stabilize the estimate.

**Why this is a real change, not a flag.** The current sig-chans are a
**precomputed artifact** (`sig_chans_*.json`) built on all trials *outside* the
decoding run. Truly breaking the circularity requires **recomputing electrode
significance on `P_def` inside the decoding pipeline** (or emitting held-out
sig-chan artifacts) — merely holding out decode trials while keeping the
all-trials electrode selection does **not** remove the leak, because selection
already saw the decode trials. So the honest options are:

- **(i)** keep `electrodes='all'` for the decoding figures (no circularity;
  simplest to defend), and/or
- **(ii)** keep `electrodes='sig'` but state explicitly that selection is on an
  **orthogonal** task-responsiveness contrast, and/or
- **(iii)** implement the disjoint split above (recompute selection on `P_def`,
  decode on `P_dec`) for the electrode-set-restricted decodes and the
  define-on-one-effect / test-the-other cross-decodes.

**Implementation sketch for (iii)** (the piece to build):

- A reusable `stratified_trial_split(metadata, strata_cols, frac_def=0.5,
  n_splits, seed)` returning disjoint `(idx_def, idx_dec)` per subject —
  unit-testable in isolation (assert disjoint, assert per-stratum balance
  preserved). Mirror the existing `_stratified_half_split` in
  `stability_flexibility_segregation.py` and the `_split_reservoirs` disjoint
  logic in `cross_decoding.py` rather than writing a third splitter.
- A selection hook in `decoding_dcc.py` that, when a new
  `args.electrode_definition_split` flag is set, computes the electrode set from
  `P_def` (calling the chosen per-electrode test on that subset) and passes only
  `P_dec` trials into the decoder.
- Default the flag **off** so existing runs reproduce.

> **Status: not yet implemented.** This section is the design; the split code and
> the `decoding_dcc.py` wiring are a follow-up (they touch a DCC pipeline that
> can't be validated outside the cluster, so they should be built behind the
> off-by-default flag and checked on a subject before a full re-run).
