# A7 (self-check) — Bootstrap OR CI + a reconciled segregation verdict

A little **build-a-feature** assignment to check that you understand how the
stability/flexibility **segregation battery** actually fits together — not any
one function, but how its two inference layers relate and how the subject-nesting
principle propagates through the whole thing. It is deliberately *not* one of the
official A1–A6 items; it composes code you've already got (A1/A2 included) into
something new.

> **Why this shape.** The segregation analysis is self-contained on purpose — it
> runs on `_synthetic_df` with **no real iEEG data, no MNE**. That's exactly what
> makes it checkable: the synthetic generator draws each electrode's stability
> and flexibility sensitivities (`bx`, `by`) **independently**, so the *true*
> answer is "no real overlap" (OR ≈ 1). A correct implementation must recover
> that; a subtly wrong one manufactures a signal, and a test catches it. A
> feature that genuinely needed several *file* modules (anatomy joins in
> `general_utils`, brain plots in `vis/`) can't be self-graded because it needs
> real data — so the "touches several parts" here is the two **analysis layers**,
> which is the part that's verifiable. (An optional anatomy extension is at the
> bottom if you want the cross-module reach too.)

## The task

You're adding a **subject-clustered bootstrap** confidence interval for the
conjunction odds ratio, and a single **verdict** function that reconciles what
the continuous and categorical layers say. Three functions in
[`a7_segregation_verdict.py`](./a7_segregation_verdict.py), each with numbered
steps in its docstring:

| # | Function | What it checks you understand |
|---|---|---|
| 1 | `bootstrap_conjunction_or` | Why inference resamples **subjects, not electrodes** — and why a subject drawn twice must become two separate CMH strata. |
| 2 | `classify_segregation` | The **meaning** of the two tests: OR < 1 / corr < 0 → segregation, OR > 1 / corr > 0 → shared, and why a genuine disagreement is its own outcome. |
| 3 | `segregation_verdict` | The **pipeline order** — sensitivities → responsiveness → residualise → correlate, in parallel with labels → CMH → null → bootstrap. |

## How to run the self-check

From the **repo root**:

```bash
python -m pytest docs/learning_assignments/segregation_bootstrap/test_a7_segregation_verdict.py -q
```

It is **red now** (the stubs `raise NotImplementedError`). Implement the three
functions until it's **green**. The `classify_*` tests are instant; the
`bootstrap_*` and `verdict` tests do real permutation work, so give them a
minute. There is **no solution file** — the helpers you're composing are the
reference; the new logic is yours.

## Acceptance criteria (what "green" is really testing)

- **Bootstrap point estimate** equals `cmh_conjunction(labels)['mh_odds_ratio']`
  exactly, and the CI brackets it.
- On synthetic data the **OR CI covers 1** and the **verdict is `inconclusive`** —
  the honest read of independent `bx`/`by`. If you get `segregated` or `shared`
  here, you've invented an effect.
- The bootstrap CI is **not artificially narrow**: resampling whole subjects must
  give a CI no tighter than one that (wrongly) treats each electrode as its own
  stratum. This is the nesting principle, made into a test.

## Before you start — read these three functions in the module

Everything you write leans on
`src/analysis/stats/stability_flexibility_segregation.py`. Skim, in order:

1. `compute_sensitivities` — note it estimates `x` and `y` on **disjoint trial
   halves** (`_stratified_half_split`). Ask yourself *why* before reading task 3.
2. `per_electrode_labels` → `cmh_conjunction` — the categorical layer that
   produces the OR you're about to put a CI on.
3. `conjunction_permutation_null` — the A2 null. Your bootstrap is its CI-shaped
   cousin; both shuffle/resample **within/at the subject level**.

## Reflect-back questions (answer these; they're the real check)

Write yourself 2–3 sentences on each. They have definite answers in the code and
docstrings — if one is fuzzy, that's the part to go re-read.

1. `compute_sensitivities` estimates stability on one trial-half and flexibility
   on the **disjoint** other half. What specific bias would you get if it used
   **all** trials for both (see `naive_sensitivities`)? Why does that bias push
   the x–y correlation in a particular direction?
2. `conjunction_permutation_null` shuffles `F` **within each subject**. What goes
   wrong — and in which direction — if you shuffle `F` **globally** across all
   electrodes instead?
3. Your bootstrap resamples **subjects**. Why not electrodes? What assumption
   about electrodes-within-a-subject would electrode resampling violate, and
   would it make the CI too wide or too narrow?
4. The continuous test can report `corr ≈ 0` while the categorical test reports
   `OR < 1`. Are those in conflict, or can both be true at once? (Hint: one is a
   *gradient* across all electrodes, the other is about *double-selective*
   electrodes specifically.) This is why `classify_segregation` has a
   `'conflicting'` outcome.

## Optional extension (touches other modules; not auto-graded)

Once green, wire an **anatomy** view of your verdict, the A3 way: join each
electrode's S/F labels to its ROI with
`make_or_load_subjects_electrodes_to_ROIs_dict` (`utils/general_utils.py`) and
`config/rois.py`, and report the "both" electrodes' ROI histogram
**conditioned on coverage** (≥ *k* subjects per ROI). This needs the real
`subjects_electrodes_to_ROIs_dict`, so there's no synthetic self-check — but it's
the natural next question ("are the double-selective electrodes *somewhere
specific*?") and it's how you'd make this reach across `stats/`, `utils/`, and
`vis/`.
