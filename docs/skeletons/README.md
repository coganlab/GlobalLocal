# Skeleton stubs for the stability/flexibility coding assignments

Runnable-but-unimplemented stubs for the assignments in
`../stability_flexibility_coding_assignments.md`. Each `aN_*.py` file is the
strongest hint for assignment `AN`: real function signatures, docstrings with
numbered implementation steps, and `raise NotImplementedError` bodies to fill in.

- Every file's **module docstring** names its **drop-in target** — the
  production module the finished code belongs in — and the exact existing
  helpers to import and reuse.
- They live here under `docs/` (not `src/`) on purpose: out of the import path
  and out of pytest collection until you move them. All pass
  `python -m py_compile docs/skeletons/*.py`.

## Files

| File | Assignment | Drop-in target |
|---|---|---|
| `a1_anova_labels.py` | A1 — ANOVA electrode definition | `src/analysis/stats/stability_flexibility_segregation.py` |
| `a2_conjunction_null_sweep.py` | A2 — permutation null + threshold sweep | `src/analysis/stats/stability_flexibility_segregation.py` |
| `a3_anatomy.py` | A3 — anatomy + coverage-conditioned test | new `stability_flexibility_anatomy.py` + `src/analysis/vis/` |
| `a4_cross_decoding.py` | A4 — cross-decoding | new `src/analysis/decoding/cross_decoding.py` |
| `a5_stability_flexibility_timing.py` | A5 — timing | new `src/analysis/stats/stability_flexibility_timing.py` |
| `a6_brain_behavior.py` | A6 — brain–behavior | new `stability_flexibility_brain_behavior.py` |

## Workflow

1. Open the stub for your assignment and read the module docstring (drop-in
   target + imports to reuse).
2. Implement each function, following the numbered steps in its docstring.
3. Test against the synthetic generator (`_synthetic_df` in the segregation
   module) and the acceptance criteria in the assignments doc.
4. Move the finished code into its drop-in target and delete the stub.
