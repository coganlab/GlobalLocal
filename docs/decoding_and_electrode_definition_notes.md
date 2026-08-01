# Decoding & Electrode-Definition — Methods Notes (merged)

> **This document has been merged into a single guide:
> [`stability_flexibility_guide.md`](stability_flexibility_guide.md).**

Where the old design-decision notes went:

| Old section (this file) | New home in the guide |
|---|---|
| A. Use one electrode definition, not several | §4 (window-mean vs per-timepoint-cluster ANOVA, answered in full) |
| B. Cross-decoding baseline leakage | §7.1 "Observed status / caveat" |
| C. Circularity between definition and decoding | §8 (disjoint trial split) |

New in the guide (not in the old notes): the **four**-interaction electrode
definition (S/F/CS/SI) and the **ignore-the-diagonal** double-dipping rule for
decoding — guide §3.2 — plus line-by-line walk-throughs of
`per_electrode_anova_labels`, `_interaction_cohens_d`, `attach_roi`, and the
double-dip guard.
