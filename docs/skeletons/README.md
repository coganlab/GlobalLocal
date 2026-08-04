# Skeleton stubs for the stability/flexibility coding assignments

Runnable-but-unimplemented stubs for the A1–A6 assignments. Each `aN_*.py` file
is the strongest hint for assignment `AN`: real function signatures, docstrings
with numbered implementation steps, and `raise NotImplementedError` bodies to
fill in. The assignment order and each assignment's method live in
[`../stability_flexibility_guide.md`](../stability_flexibility_guide.md) (§9.0
for the order; A1 → §3, A2 → §5, A3 → §6, A4 → §7.1, A5 → §7.2, A6 → §7.3), and
the cross-cutting acceptance checklist is guide §2.

- Every file's **module docstring** names its **drop-in target** — the
  production module the finished code belongs in — and the exact existing
  helpers to import and reuse.
- They live here under `docs/` (not `src/`) on purpose: out of the import path
  and out of pytest collection until you move them. All pass
  `python -m py_compile docs/skeletons/*.py`.

## Files, drop-in targets, and implementation status

Every assignment below is **finished in production code** — the stubs are kept as
reading aids, not as open work. Implement one yourself first if you are using it
as an exercise, then diff against the real module.

| File | Assignment | Implemented in (drop-in target) | DCC job | Tutorial |
|---|---|---|---|---|
| `a1_anova_labels.py` | A1 — ANOVA electrode definition | `src/analysis/stats/stability_flexibility_segregation.py` | `dcc_scripts/stats/*_anova_conjunction_dcc.*` | `.../stability_flexibility_segregation_tutorial.ipynb` |
| `a2_conjunction_null_sweep.py` | A2 — permutation null + threshold sweep | same | same | same |
| `a3_anatomy.py` | A3 — anatomy + coverage-conditioned test | `src/analysis/stats/stability_flexibility_anatomy.py` | `dcc_scripts/stats/*_anatomy_dcc.*` | `.../stability_flexibility_anatomy_tutorial.ipynb` |
| `a4_cross_decoding.py` | A4 — cross-decoding | `src/analysis/decoding/cross_decoding.py` | `dcc_scripts/decoding/*_cross_decoding_dcc.*` | `.../cross_decoding_tutorial.ipynb` |
| `a5_stability_flexibility_timing.py` | A5 — timing | `src/analysis/stats/stability_flexibility_timing.py` | `dcc_scripts/stats/*_timing_dcc.*` | `.../stability_flexibility_a5_a6_tutorial.ipynb` |
| `a6_brain_behavior.py` | A6 — brain–behavior | `src/analysis/stats/stability_flexibility_brain_behavior.py` | `dcc_scripts/stats/*_brain_behavior_dcc.*` | same |

Each production module is validated on synthetic ground truth (A3: a planted vs.
null ROI enrichment; A4: a shared vs. orthogonal code; A5: a planted onset
ordering and its reversal; A6: a planted matched-beats-cross coupling).

The one assignment with **no** reference solution is **A7**, the self-check in
[`../learning_assignments/segregation_bootstrap/`](../learning_assignments/segregation_bootstrap/)
— its pytest grader stays red until you implement the stubs.

## Workflow

1. Open the stub for your assignment and read the module docstring (drop-in
   target + imports to reuse).
2. Implement each function, following the numbered steps in its docstring.
3. Test against the synthetic generator (`_synthetic_df` in the segregation
   module) and the acceptance criteria in guide §2.
4. Compare against the production module named above.
