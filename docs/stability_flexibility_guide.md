# Stability vs. Flexibility — Complete Analysis Guide

**This is the single, merged document for the stability/flexibility iEEG battery.**
It supersedes and folds together what used to be four separate files:

| Merged-in source | What it contributed |
|---|---|
| `stability_flexibility_analysis_plan.md` | the scientific plan, figure map, and the §0 statistical-rigor checklist |
| `stability_flexibility_coding_assignments.md` | the staged build plan (A0–A6) and acceptance criteria |
| `decoding_and_electrode_definition_notes.md` | the discursive design decisions (one definition, cross-decoding leakage, circularity) |
| `analysis_paths.md` §§12–13 | where each analysis lives and how to launch it |

The operational per-job READMEs (`dcc_scripts/stats/README.md`,
`dcc_scripts/decoding/README_stability_flexibility_cross_decoding.md`) stay next
to their scripts as launch references; §10 here tells you which to run when.

**How to read this guide.**
- §1–§2: the question and the statistical principles everything obeys.
- §3: **electrode definition** — the four interaction groups and the
  double-dipping rule (this is the part that changed most recently; read it in
  full).
- §4: the **window-mean vs. per-timepoint-cluster ANOVA** decision, answered
  honestly and at the level of the actual code lines.
- §5–§7: the rest of the battery, the anatomy line-by-line worked example
  (`attach_roi`), and the circularity control.
- §8: **what order to walk the tutorials and run the code**, and how to run each.
- §9: a function/file map you can grep from.

Where a design decision hinges on a specific line of code, this guide quotes the
line and says **what it does** and **why it is written that way rather than the
obvious alternative** — that is the level of detail the tutorials aim for too.

---

## 1. The question and the figure plan

**The question.** Do **stability** (LWPC / proactive control) and **flexibility**
(LWPS / reactive control) rely on **shared** or **distinct** iEEG substrates?
Concretely: are there *distinct subpopulations* supporting one process but not the
other, or only *shared populations* carrying both — and if shared, is it the same
*code*, at the same *sites*, arising at the same *time*?

Two constructs, each a **two-way interaction** on single-trial high-gamma (HG):

- **LWPC (stability)** = `congruency × incongruent_proportion` — the congruency
  effect grows in high-incongruent-proportion blocks.
- **LWPS (flexibility)** = `switchType × switch_proportion` — the switch effect
  grows in high-switch-proportion blocks.

"Shared vs distinct" is **three questions, not one**, and the answer can differ at
each level:

1. **Anatomical / electrode overlap** — are the same *sites* selective for both? → §5 (conjunction), §6 (anatomy)
2. **Single-channel tuning** — does the same channel carry both signals? → §5 continuous correlation
3. **Representational format** — is it the same *code*? → §7 (cross-decoding)

Figure sequence:

| Fig | Content | Role | Section |
|---|---|---|---|
| 1 | Behavior: LWPC + LWPS effects, **no behavioral cross-effects** | the puzzle | motivation |
| 2 | Time–frequency: congruency (inc−con), switch cost (switch−repeat) | signal validation | §3 setup |
| 3 | High-gamma rises after stimulus onset | signal validation | §3 setup |
| 4 | HG power traces: LWPC & LWPS within-trial; **pre-trial cross-effects** | effects + tonic/baseline issue | §3, §7 |
| 5 | 2×2 conjunction (electrode counts) + stats | same sites selective for both? | §5 |
| 6 | Onset latency (jackknife, 50%-of-peak) | does one precede the other? | §7 timing |
| 7 | Segregation: conjunction **+ continuous effect-size correlation** | core anatomical answer | §5 |
| 8 | Orthogonal power traces (define on LWPC → LWPS trace, vice versa) | cross-contrast confirmation | §5 |
| 9 | Within-block decoding (the 2×2), incl. neural cross-effects | readable info + dissociation | §7 |
| 10 | Cross-decoding (label transfer) + temporal-generalization matrices | shared code vs co-located | §7 |

**The headline dissociation.** Fig 1 shows *no behavioral crossover*, yet Fig 9
shows *neural* cross-effects (congruency decoding differs by switch-proportion
block, and vice versa). This **behavior-independent / neural-interacting**
pattern is a result, not a nuisance — *provided* it survives the decoding
confounds in §2.8. Treat the *behavioral* cross-interactions as specificity
controls (they should be null); treat the *neural* cross-effects as a finding to
confound-proof.

> **Why not one four-way ANOVA?** `congruency × switchType × inc_prop ×
> switch_prop` has uninterpretable, underpowered high-order terms. Two focused
> two-way interactions map onto the constructs; the two *cross* interactions are
> specificity controls in univariate HG (should be null) but become real
> electrode-definition groups for the decoding double-dip bookkeeping — see §3.

> **Frequency scope.** Constructs are defined on HG (proxy for local activity).
> Conflict (theta) and switching (beta) have low-frequency signatures; HG is
> primary, and the conjunction/decoding are re-run in low bands as a robustness
> check.

---

## 2. Cross-cutting statistical principles (read before any result is "real")

These are the difference between a real result and an artifact. Every assignment
below has acceptance criteria that are just these made concrete.

1. **Double-dipping / selection bias.** Defining electrodes on contrast A and then
   reporting A's effect (or A's decoding) in that group is circular. The clean
   direction is **cross-contrast**: define on LWPC, test LWPS (and vice versa).
   Anything reported *on the selection contrast* must come from **held-out
   trials** (disjoint half; `_stratified_half_split`) or be labeled
   descriptive-only. **§3.2 turns this principle into the concrete "ignore the
   diagonal decode cell" rule.**
2. **Disjoint trial halves.** Even the cross-contrast test couples through shared
   trial noise (LWPC and LWPS are estimated from the same trials). Estimate the
   selection and test contrasts on disjoint halves.
3. **Power matching.** LWPC and LWPS almost certainly differ in effect size, so
   the stronger recruits more electrodes at fixed α. Report counts/effects **as a
   function of threshold**, not one α snapshot (§5 sweep).
4. **Multiple comparisons.** FDR (Benjamini–Hochberg) across electrodes for the
   per-electrode selection tests.
5. **Coverage bias.** iEEG coverage is clinically determined. Any anatomical claim
   must be conditioned on coverage (§6), or it reflects *where electrodes are*.
6. **Latency–amplitude confound.** A larger effect crosses any onset threshold
   sooner. Any "X earlier than Y" claim must guard against X simply being bigger
   (§7 timing, 50%-of-peak).
7. **Tonic / pre-trial baseline.** List-wide manipulations induce a *sustained*
   block-level state present **before** stimulus onset. Pre-trial "cross-effects"
   (Fig 4) may be genuine tonic proactive-control signals — but they poison any
   baseline correction spanning them. Use a baseline that predates the block
   context, report the pre-trial effect, and separate tonic (sustained) from
   phasic (evoked). This is a result about proactive control, not a cleanup step.
8. **Decoding confounds.** Blocks differ in difficulty and RT, so a classifier can
   exploit RT-correlated power or a univariate mean offset instead of a control
   code. Before interpreting any decode — especially the neural cross-effects —
   match trial counts, regress/match RT, and confirm survival of per-condition
   mean removal.

---

## 3. Electrode definition — the four interaction groups

> **Goal.** Label each electrode by which of the **four two-way interactions** it
> is selective for, so both the conjunction (§5) and the non-circular decoding
> (§7) can consume co-registered labels.

### 3.1 Why interactions, not main effects

An earlier framing selected electrodes on the **main effects** — congruency
(i vs c) and switchType (s vs r), i.e. `contrast_mode='condition'`. That is the
wrong selector for this paper: a congruency *main effect* means "this electrode
responds to conflict," not "this electrode implements the *list-wide adjustment*."
The constructs of interest **are the interactions** — the congruency effect
*growing with incongruent-proportion* (LWPC) and the switch effect *growing with
switch-proportion* (LWPS). So selection uses `contrast_mode='proportion'`, and the
selected quantity is a balanced 2×2 **difference-of-differences**, not a two-group
mean difference.

### 3.2 The four groups and the double-dipping rule

We now define **all four** two-way interactions as electrode-selection groups, not
just the two constructs of interest:

| Flag | Interaction | Meaning |
|---|---|---|
| `S`  | congruency × incongruent_proportion | **LWPC** (stability) |
| `F`  | switchType × switch_proportion | **LWPS** (flexibility) |
| `CS` | congruency × switch_proportion | cross (a *flexibility* manipulation moving a *stability* readout) |
| `SI` | switchType × incongruent_proportion | cross (a *stability* manipulation moving a *flexibility* readout) |

In **univariate HG** the two cross groups (`CS`, `SI`) are expected to be
near-null — that is their long-standing role as *specificity controls*. So why
promote them from report-only p-values to full FDR'd, flagged electrode groups?

**Because the decoding battery (§7) decodes a 2×2 of `{contrast} × {block
modulator}`, and each of those four decode cells is the multivariate readout
analog of exactly one of these four interactions:**

| Decode cell (what × split-by) | Readout analog of |
|---|---|
| congruency × inc-prop | `S` (LWPC) |
| switchType × switch-prop | `F` (LWPS) |
| congruency × switch-prop | `CS` |
| switchType × inc-prop | `SI` |

**The rule (bullet 1's "ignore the diagonal").** When a decode cell is restricted
to the electrode set that *the same interaction* defined, its accuracy is
guaranteed to be inflated — the electrodes were chosen for having that very
difference-of-differences. **Ignore that result.** Keep only the **off-diagonal**
cells: define on one interaction, decode a *different* cell. This is principle
§2.1 (double-dipping) made mechanical, generalized from the "define on LWPC, test
LWPS" workhorse to all four groups. Each defined group therefore yields **three**
usable (non-circular) decode cells and **one** ignored (circular) one.

The diagonal map lives in code as a single table, so nothing hand-tracks it:

```python
# src/analysis/decoding/cross_decoding.py
DEFINITION_DECODE_DIAGONAL = {
    "S":  ("congruency", "incongruent_proportion"),   # LWPC  = congruency x inc_prop
    "F":  ("switchType", "switch_proportion"),         # LWPS  = switchType x switch_prop
    "CS": ("congruency", "switch_proportion"),         # cross = congruency x switch_prop
    "SI": ("switchType", "incongruent_proportion"),    # cross = switchType x inc_prop
}
```

- **What each row is.** `flag -> (decode_contrast, block_modulator)`. The value is
  the *one* within-block decode cell that would double-dip on electrodes selected
  by that flag's interaction.
- **Why a dict keyed by the flag** rather than, say, hard-coding the skip inside
  the decode loop: the mapping is the *definition* of circularity for this design,
  so it belongs in one named, testable place; the loop just asks it. If a future
  contrast is added, you extend one table, not scattered `if` branches.

The predicates that consume it:

```python
def circular_decode_for_group(definition_group):
    return DEFINITION_DECODE_DIAGONAL.get(definition_group)   # None for 'both'/'all'

def is_circular_decode(definition_group, contrast, block_col):
    diag = circular_decode_for_group(definition_group)
    return diag is not None and diag == (contrast, block_col)
```

- **`.get(...)` returns `None`** for composite groups like `both` or `all`, which
  are not a single interaction — so `is_circular_decode` is `False` for them and
  nothing is skipped. This is deliberate: the "both" group's cross-decodes (train
  LWPC → test LWPS) are *already* cross-contrast and non-circular, so we must not
  accidentally suppress them.
- **`diag == (contrast, block_col)`** is an exact tuple match, not a
  `contrast in diag` membership test, because a cell is only circular when *both*
  the decoded contrast **and** the block modulator match the defining interaction.
  Decoding congruency split by *switch*-prop on `S` electrodes is off-diagonal
  (clean) even though the contrast `congruency` appears in `S`'s diagonal.

The DCC orchestrator (`stability_flexibility_cross_decoding_dcc.py`) builds the
four groups and runs the per-group 2×2, skipping the diagonal:

```python
for gflag, elset in interaction_groups.items():         # S, F, CS, SI
    ...
    for contrast, block_col in decode_cells:            # the four decode cells
        if cd.is_circular_decode(gflag, contrast, block_col):
            continue                                    # double-dipping: ignore
        cells[...] = cd.within_block_decoding_baseline(df, contrast=contrast,
                                                       block_col=block_col,
                                                       electrodes=elset, ...)
```

> Every result kept here is a decode of one interaction's electrodes on a
> *different* interaction's cell — exactly the clean cross-contrast evidence §2.1
> asks for.

### 3.3 Line-by-line: `per_electrode_anova_labels` (the definition itself)

Lives in `src/analysis/stats/stability_flexibility_segregation.py`. This is the
parametric primary definition (assignment **A1**). Walking the body:

```python
contrasts = finalize_contrasts(df, resolve_contrasts(contrast_mode, contrasts))
work = _canonical_labels(df, contrasts)      # attaches _scond/_smod/_fcond/_fmod
```
- **`resolve_contrasts('proportion', None)`** returns the preset that says
  stability = congruency×inc_prop and flexibility = switchType×switch_prop.
  `finalize_contrasts` resolves any `'high'/'low'` proportion sentinels to the
  df's actual numeric extremes (so `75.0`/`25.0` need not be hard-coded).
- **`_canonical_labels`** attaches four `{0.0, 1.0, NaN}` sub-factor columns —
  `_scond` (congruency), `_smod` (inc-prop), `_fcond` (switchType), `_fmod`
  (switch-prop). *Why pre-attach them once here* rather than re-derive per
  electrode: the sign step (below) and all four interactions reuse them, and the
  same encoding must be identical everywhere or the sign and the F-test could
  disagree on direction.

```python
for (subj, elec), g in work.groupby(['subject', 'electrode']):
    hg = g['hg'].to_numpy()
    scond, smod = g['_scond'].to_numpy(), g['_smod'].to_numpy()
    fcond, fmod = g['_fcond'].to_numpy(), g['_fmod'].to_numpy()
```
- One ANOVA **per electrode** (grouped by `(subject, electrode)`). Electrode ids
  are subject-scoped (`f'{subject}_{elec}'`) so two subjects' "channel 5" never
  collide in the groupby.

```python
    st = _anova_interaction_stats(g, 'congruency', 'incongruent_proportion')
    fl = _anova_interaction_stats(g, 'switchType', 'switch_proportion')
```
- **`_anova_interaction_stats`** fits `hg ~ C(a, Sum) * C(b, Sum)` and pulls the
  interaction row's `F` and `PR(>F)` from `anova_lm(model, typ=3)`. The
  **`Sum` (effect) coding + Type III** is the whole subtlety — see §4's coding
  note. It is wrapped in try/except → `NaN` for a singular fit (an electrode
  missing a 2×2 cell), mirroring how the effect helpers return `NaN` for too-few
  trials, so one degenerate electrode never crashes the sweep.

```python
    s_sign = np.sign(_interaction_effect(hg, scond, smod, 'cohens_d', alpha))
    f_sign = np.sign(_interaction_effect(hg, fcond, fmod, 'cohens_d', alpha))
```
- **The ANOVA F is unsigned**; the difference-of-differences can grow the
  *predicted* way or the opposite. We take the sign from the module's own
  equal-cell-weight estimator `_interaction_effect(..., 'cohens_d')` — *the very
  quantity the §5 continuous correlation uses* — so the sign the labels carry and
  the sign the correlation sees can never disagree. (Using `np.sign` of the F, or
  of a treatment-coded contrast, would risk exactly that disagreement.)

```python
    if include_cross_controls:
        cs = _anova_interaction_stats(g, 'congruency', 'switch_proportion')
        si = _anova_interaction_stats(g, 'switchType', 'incongruent_proportion')
        rec['p_cross_cs'] = cs['p']; rec['F_cross_cs'] = cs['F']
        rec['p_cross_si'] = si['p']; rec['F_cross_si'] = si['F']
        rec['cs_sign'] = np.sign(_interaction_effect(hg, scond, fmod, 'cohens_d', alpha))
        rec['si_sign'] = np.sign(_interaction_effect(hg, fcond, smod, 'cohens_d', alpha))
```
- **The two cross interactions**, now recorded with F, p, and sign — the same
  three quantities as S/F. *Why the sign reuses the already-attached sub-factors*:
  `CS` = congruency (`scond`) × switch-prop (`fmod`); `SI` = switchType (`fcond`)
  × inc-prop (`smod`). No new label construction is needed — the four cross
  sub-factors are just the S and F sub-factors recombined, which is exactly why
  `_canonical_labels` attaches all four up front.

```python
out['q_cong'] = multipletests(out['p_cong'].fillna(1), method='fdr_bh')[1]
out['q_switch'] = multipletests(out['p_switch'].fillna(1), method='fdr_bh')[1]
```
- **FDR across electrodes**, per interaction, on the p-values. `fillna(1)` makes a
  singular-fit electrode (p = NaN) count as "not significant" rather than dropping
  it, so the number of tests in the FDR denominator stays honest (dropping NaNs
  would inflate every other electrode's significance).

```python
S = out['q_cong'] < alpha
F = out['q_switch'] < alpha
if require_sign:
    S = S & (out['s_sign'] > 0)
    F = F & (out['f_sign'] > 0)
out['S'] = S.astype(int); out['F'] = F.astype(int)
```
- Flag at `alpha` on the FDR q-value. **`require_sign`** optionally keeps only
  electrodes whose effect grows in the *predicted* (positive) direction — an
  electrode whose congruency effect *shrinks* with inc-proportion is a significant
  interaction but not an LWPC electrode. It is a parameter, not hard-coded on,
  because the conjunction sometimes wants the symmetric "any interaction" set.

```python
if include_cross_controls:
    out['q_cross_cs'] = multipletests(out['p_cross_cs'].fillna(1), method='fdr_bh')[1]
    out['q_cross_si'] = multipletests(out['p_cross_si'].fillna(1), method='fdr_bh')[1]
    CS = out['q_cross_cs'] < alpha
    SI = out['q_cross_si'] < alpha
    if require_sign:
        CS = CS & (out['cs_sign'] > 0); SI = SI & (out['si_sign'] > 0)
    out['CS'] = CS.astype(int); out['SI'] = SI.astype(int)
```
- The **new** block: the cross interactions get their own FDR sweep and flags,
  turning them into full selection groups. *Why a separate FDR per interaction*
  rather than one FDR over all four pooled: each interaction is a distinct family
  of hypotheses with its own null rate; pooling them would let a strong S effect
  borrow significance for a weak CS effect. Keeping `CS`/`SI` under
  `include_cross_controls` means the pure S/F conjunction path
  (`include_cross_controls=False`) is byte-for-byte unchanged — backward
  compatibility the tests pin down.

**Output contract.** One row per electrode with `subject, electrode, S, F` (and,
under `include_cross_controls`, `CS, SI`), plus each interaction's `F`, `p`, `q`,
and sign. Because `subject, S, F` are present and column-compatible with
`per_electrode_labels`, the table drops straight into `cmh_conjunction` unchanged.

### 3.4 Line-by-line: the balanced difference-of-differences (`_interaction_cohens_d`)

This function is *why* the interaction is trustworthy under the deliberately
unequal (~75/25) proportion cells.

```python
def _interaction_cohens_d(cells):
    num, dfree, means = 0.0, 0, {}
    for k, v in cells.items():
        n = len(v)
        if n < 2:
            return np.nan                       # a cell with <2 trials -> undefined
        num += (n - 1) * v.var(ddof=1)          # pooled within-cell SS
        dfree += n - 1
        means[k] = v.mean(0)
    dod = ((means[(1.0, 1.0)] - means[(0.0, 1.0)])
           - (means[(1.0, 0.0)] - means[(0.0, 0.0)]))
    sp = np.sqrt(num / dfree)
    return np.nan if sp == 0 else dod / sp
```
- **`cells`** is the four `(cond, mod)` cells as separate arrays (from
  `_dod_cells`). The estimator averages the four cell means with **equal weight**,
  not trial-count weight.
- **`dod`** is the difference-of-differences: (effect of congruency in high-prop) −
  (effect of congruency in low-prop). *Why equal cell weights matter:* the naive
  "+1 diagonal vs −1 diagonal pooled mean difference" is trial-count weighted, and
  in a 75/25 design the +1 super-group is dominated by the frequent cells. Under
  that imbalance a pure congruency **main effect** leaks into the "interaction"
  (~0.8 SD of fake effect in a zero-interaction simulation). The equal-cell d-o-d
  is orthogonal to *both* main effects, so it isolates the interaction — this is
  the nonparametric twin of the Type III + sum-coding trick in the ANOVA.
- **`sp`** is the pooled within-cell SD (standardizes the d-o-d into a Cohen's-d
  scale so effect sizes are comparable across electrodes with different HG
  variance). **`return NaN if sp == 0`** guards a flat electrode.
- **`n < 2 -> NaN`** for any cell: a difference-of-differences needs all four
  cells populated; returning NaN (rather than 0) keeps a degenerate electrode out
  of the FDR count instead of pretending it has a null effect.

The time-resolved sibling `_interaction_cluster(cells, alpha)` computes the same
d-o-d **per time bin**, converts to a per-bin t, thresholds at the parametric
`alpha` critical t, and returns the **signed cluster mass** (sum of supra-threshold
t within contiguous runs). That single function is the bridge to §4: it is the
temporal, cluster-corrected interaction test, emitting a signed graded scalar.

### 3.5 Where the definition is wired

- **Segregation / conjunction (A1/A2):**
  `dcc_scripts/stats/submit_stability_flexibility_anova_conjunction_dcc.sh`.
- **Anatomy (A3):** `stability_flexibility_anatomy.py` calls
  `per_electrode_anova_labels(contrast_mode='proportion')` for its S/F groups.
- **Cross-decoding (A4):** `_electrode_groups` builds `both/S_only/F_only` and
  `_interaction_groups` builds the four `S/F/CS/SI` selection sets used by the
  per-group 2×2 with the diagonal skipped (§3.2).
- **Brain plots (A3 vis):** re-pointing the brain maps at A1's groups (rather than
  the `power_traces` sig-chans) is the remaining wiring task — see §4's "which
  plotting consumes which."

---

## 4. Window-mean ANOVA vs. per-timepoint cluster ANOVA — the honest answer

This addresses the recurring disagreement directly, because it deserves a precise
answer rather than a restated preference.

**Your position (as I understand it).** A1 runs a *single* ANOVA on a
time-window-**averaged** HG value, which drops the time dimension and can wash out
a strong-but-transient interaction. Your `power_traces` method instead runs the
ANOVA **at each time point** and **cluster-corrects across time**, keeping any
electrode with a surviving cluster. You think that is more correct. You want me to
either justify the window-mean or concede.

**The honest concession first.** On the narrow question *"does this electrode carry
an interaction at all,"* **you are right that the window-mean is not more correct —
and is often less sensitive.** An interaction present for 100 ms inside a 500 ms
window is diluted ~5× by averaging; a per-bin test with cluster correction over
time recovers it. So I will not claim the time-window average is a *statistically
superior detector*. It isn't. If detection sensitivity to transient interactions
were the only criterion, the per-timepoint cluster test wins.

**What the earlier pushback was actually about — and it was a real point, just
mis-labeled.** The objection was never "keep the time dimension is wrong." It was
**"you cannot reuse the *current* `power_traces` output"** —
`load_significant_electrodes` in `src/analysis/power/windowed_anova.py`. That
function returns a **flat, per-effect, unsigned pass/fail list**
(`[(subject, electrode)]` for *one* effect, ROI-scoped). Three concrete downstream
needs it cannot meet:

1. **A signed, graded scalar per electrode.** The Fig-7 continuous correlation
   correlates each electrode's LWPC effect size against its LWPS effect size. You
   cannot correlate two "has a surviving cluster" booleans, nor two unsigned F
   values (wrong sign, wrong scale). You need one number with a **sign** and a
   **scale** per electrode.
2. **Both processes co-registered on the same row.** The 2×2 conjunction
   (`both / S-only / F-only / neither`) needs S and F — and now the two cross
   controls — on the *same electrode record*. A flat per-effect list gives one
   verdict per electrode per effect, with no paired labeling, no sign, no cross
   controls. You cannot assemble the conjunction table from it.
3. **Legible multiple-comparison bookkeeping.** `load_significant_electrodes`'s
   pipeline FDRs across **clusters within `(roi, effect)`**; the definition needs
   FDR across **electrodes**, per process. These are different correction
   families answering different questions.

That is a genuine objection — **but it is about the output contract, not about the
temporal model.** Conflating the two ("window-mean is more correct") would be
wrong, and if the pushback ever read that way, this section is the correction.

**The resolution that gives you what you want.** You do **not** have to drop the
time dimension to satisfy 1–3. The segregation module already implements the
temporal, cluster-corrected interaction test *and* emits it in the right shape:

- **`effect_measure='cluster'`** scores each interaction by its **signed cluster
  mass over time** (`_interaction_cluster`, §3.4) — a per-bin difference-of-
  differences t, thresholded and summed over contiguous supra-threshold runs. That
  is a per-timepoint ANOVA-style test, cluster-corrected across time, returning a
  **signed graded scalar per electrode** (need #1), co-registered for S, F, CS, SI
  on the same row (need #2), FDR'd across electrodes (need #3).
- **`USE_TIME_PERM_CLUSTER=True`** swaps the fast parametric threshold for the real
  `ieeg.calc.stats.time_perm_cluster` permutation mask.
- **"Keep electrodes with any surviving cluster"** is then simply thresholding that
  cluster mass at its permutation-significant value → the same S/F/CS/SI flags,
  computed from the *temporal* statistic.

So the principled recommendation is: **run A1 with `effect_measure='cluster'` as
the primary electrode definition** (you keep the temporal information you rightly
care about, *and* you get the co-registered signed scalar the conjunction and the
correlation require), and use `effect_measure='cohens_d'` on the window mean as the
**simpler robustness cross-check**, not the other way around. That is the opposite
of "drop the time dimension" — it selects electrodes exactly by your
per-timepoint, cluster-corrected criterion, delivered in the shape the rest of the
battery consumes.

**One technical caveat that also favors the module over raw `power_traces`.** A 2×2
interaction is a **difference-of-differences (four cells)**, not a two-sample
contrast. `ieeg`'s `time_perm_cluster` is a **two-condition** permutation test, so
it does not apply *directly* to the interaction — you would be permuting the wrong
thing. The correct null permutes the **modulator within each condition level**,
holding both main effects fixed so only the interaction is nulled — which is
exactly what the segregation module's permutation path
(`per_electrode_labels` / `_interaction_cluster`) implements, and what
`_interaction_cohens_d`/`_interaction_cluster` compute for the point estimate.
`power_traces`' generic per-window ANOVA + extent-cluster correction
(`run_within_electrode_windowed_anova_cluster_correction`) is fine for selecting on
*a factor's own* significance, but its interaction handling and its flat
`load_significant_electrodes` output are not the interaction-co-registered signed
scalar this design needs.

**Coding note (why Type III + sum coding, and why it's *not* "power_traces is
biased").** For the *top-order interaction* term, Type II (treatment) and Type III
(sum) coding yield the **same** SS — coding only changes the lower-order main-effect
SS. So this is not "power_traces is biased, A1 fixes it": `power_traces` already
computes an equal-cell-weight signed contrast (`_signed_contrast_per_window`) for
its sign trace. A1 adopts sum/Type III as a documented convention so its
interaction estimate is orthogonal to the main effects *by construction* and matches
the equal-cell difference-of-differences the permutation route uses. The ~0.8 SD
imbalance leak is a property of a *pooled super-group* effect-size estimator, which
both routes now avoid.

**Which plotting consumes which (the remaining inconsistency to fix).** Today the
brain-map / ROI-histogram / F-trace visualizers read **power_traces**
(`dcc_scripts/vis/plot_sig_electrodes_dcc.py` imports
`load_significant_electrodes`; `power_traces_anova_f_traces_vis.py` reads the
F-trace `.npz`). A1's labels currently feed only the segregation statistics.
Re-pointing the anatomical brain plots at A1's S/F/CS/SI groups is assignment **A3**
and is not fully wired. Once done, `power_traces` stops being a *competing
electrode definition* and becomes your **temporal-profile figure** (the F-traces:
"when does the effect emerge") — a different question, so it raises no reviewer
conflict about "which definition is primary."

**Bottom line.** Use **one** definition (A1), run it with the **cluster**
effect-measure so it *is* your per-timepoint, cluster-corrected ANOVA; keep the
window mean as a robustness check; and let `power_traces` be the temporal-profile
figure. That satisfies your methodological point and the downstream contract at the
same time.

---

## 5. Overlap / conjunction and continuous correlation (§2 → Figs 5, 7)

**Goal.** The core "distinct vs shared populations" test.

- Build the per-subject 2×2 (`both / S-only / F-only / neither`) and pool with
  **Cochran–Mantel–Haenszel** (subject-stratified). MH OR **<1** → segregation;
  **>1** → shared core; **≈1** → independent. (`cmh_conjunction`.)
- **Permutation null** (`conjunction_permutation_null`): shuffle F **within each
  subject** so every subject's S- and F-marginals stay fixed and only the *pairing*
  is randomized — the exact null CMH assumes; a global shuffle would break the
  nesting and manufacture significance.
- **Threshold sweep** (`conjunction_threshold_sweep`): recompute OR across cutoffs;
  a real claim is stable across α (principle §2.3).
- **Continuous, threshold-free (Fig 7 headline):** correlate each electrode's LWPC
  effect size against its LWPS effect size across all electrodes
  (`subject_clustered_corr`), estimated on **disjoint trial halves** so shared
  trial noise cannot inflate it; null by within-subject permutation. Positive →
  shared tuning; ≈0 → segregation.

**Why the conjunction matters most:** it is the only test in the battery that can
give **positive evidence for distinctness** (OR < 1). Decoding can only *fail* to
find a shared code, which is weaker.

**Limitation → why §7 exists.** Co-localization ≠ shared code. A "both" electrode
can be a genuinely shared representation *or* mixed selectivity with orthogonal
codes. Counting cannot tell them apart.

---

## 6. Anatomy — brain maps, ROI histograms, coverage-conditioned test (§3 → Fig 5)

**Goal.** Are the distinct subpopulations in *different places*? — with the catch
that **iEEG coverage is clinical**, so a raw ROI difference can just reflect where
electrodes are. Every claim is conditioned on coverage.

### 6.1 Line-by-line worked example: `attach_roi`

This is the function you flagged as used-but-not-understood. Full body from
`src/analysis/stats/stability_flexibility_anatomy.py`:

```python
def attach_roi(labels, electrodes_to_rois):
    out = labels.copy()
    e2r = dict(electrodes_to_rois)
    out['roi'] = out['electrode'].map(e2r)
    out['group'] = out.apply(_derive_group, axis=1)
    return out
```

- **`out = labels.copy()`** — work on a copy so the caller's A1 labels table is
  never mutated in place. *Why this matters here specifically:* the anatomy job
  reuses `labels` for the histogram and the coverage matrix; a silent in-place add
  of `roi`/`group` would make those later steps depend on call order. A copy makes
  `attach_roi` a pure function (same input → same output, no side effects).
- **`e2r = dict(electrodes_to_rois)`** — normalize the mapping to a plain dict. The
  argument may arrive as the flat map from `build_electrode_roi_map`, a pandas
  `Series`, or a dict. Wrapping in `dict(...)` gives one type with one lookup
  semantics, so the next line's `.map` behaves identically regardless of what the
  caller passed. *Alternative rejected:* calling `.map` directly on a `Series`
  works too, but then a `Series` with a non-unique or differently-ordered index
  could align by index instead of by value and silently mis-map — `dict(...)`
  removes that footgun.
- **`out['roi'] = out['electrode'].map(e2r)`** — the actual join, done as a
  **vectorized `.map`** (electrode id → ROI) rather than a Python loop or a
  `merge`. `.map` is O(n) with a dict lookup per row and, crucially, yields
  **`NaN` for any electrode not in `e2r`** instead of raising. That NaN is
  load-bearing: an electrode with no atlas ROI is *kept* (with `roi=NaN`) so the
  caller can report how many selective electrodes fall outside the atlas; the
  coverage-conditioned test drops them later, on purpose, rather than here. *A
  `merge(how='inner')` would silently delete those electrodes — losing exactly the
  count you want to report.*
- **`out['group'] = out.apply(_derive_group, axis=1)`** — derive the 4-way
  selectivity group (`both / S_only / F_only / neither`) from each row's `(S, F)`.
  `axis=1` applies `_derive_group` **per row** (it needs both S and F together), so
  a column-wise vectorized expression won't do; `_derive_group` is a small
  readable function (`S and F → 'both'`, `S and not F → 'S_only'`, …) rather than a
  nested `np.where`, because the four-way branch reads more clearly as explicit
  cases and this runs once per electrode, not per trial — so the `.apply` overhead
  is negligible and clarity wins.
- **`return out`** — the labels table plus `roi` and `group`, ready for
  `build_coverage_matrix` and `roi_group_enrichment_test`.

*One-sentence mental model:* `attach_roi` is a **pure, NaN-preserving left join**
of the ROI atlas onto the A1 labels, plus a per-row S/F → group derivation —
NaN-preserving because "this selective electrode has no atlas ROI" is information
the coverage step needs, not an error.

### 6.2 The coverage-conditioned test

- **`build_coverage_matrix`** — a subject × ROI boolean (does subject *s* have any
  electrode in ROI *r*?).
- **`roi_group_enrichment_test`** — Pearson χ² on the group × ROI table with a
  **within-subject permutation null** (shuffle the group label inside each
  subject, so the null respects nesting *and* coverage), restricted to ROIs
  sampled in ≥ `MIN_SUBJECTS` subjects. A significant permutation p means group
  membership is associated with ROI **beyond** what placement forces; per-ROI
  coverage is reported alongside so no claim rests on where the grid happens to be.

---

## 7. Cross-decoding and timing — shared code? earlier?

### 7.1 Cross-decoding (§4 → Figs 9, 10)

Co-localization (§5–§6) shows the *same electrodes* are selective for both, but not
whether they carry **one shared code** or **two orthogonal codes**. Cross-decoding
trains a classifier on one contrast and tests whether its decision axis transfers.

**(0) Within-block decoding baseline (Fig 9) — the 2×2.** Decode `{congruency,
switchType}` × split-by `{inc-prop, switch-prop}`. Diagonal = matched LWPC/LWPS;
off-diagonal = the two neural cross-effects (the decoding analog of §3's cross
interactions). **This is where §3.2's ignore-the-diagonal rule applies once you
restrict a cell to a defined electrode group.**

> **Observed status / caveat.** The two **matched** decodes behave as expected:
> baseline at chance, rising ~0.4–0.5 s post-stimulus, matched-block ordering
> correct. The two **cross** decodes currently show significant clusters extending
> *into and before* the pre-stimulus baseline. For current-trial *congruency* that
> is diagnostically impossible (you cannot know this trial's congruency before the
> stimulus), so treat the cross panels as **baseline-leakage artifacts pending the
> §2.8 confound controls**. Use the congruency `t<0` baseline as an artifact meter:
> whatever drives it back to chance is the right fix. Leading suspects: (i)
> `StratifiedKFold(shuffle=True)` random folds ignoring trial time/run order, so
> slow drift correlated with a temporally-clustered rare label leaks across folds;
> (ii) tiny min-balanced samples on the rare cross cell; (iii) sequence carryover
> (legitimate for switch type, a confound for congruency). Fixes, in order:
> time-/run-aware folds (leave-one-run-out / `GroupKFold`), baseline-correct the
> accuracy trace before cluster-forming, match trial counts, re-run after
> `remove_condition_means`.

**(a) Label transfer.** Train on stability, test on flexibility (and vice versa),
on the *same* electrodes, separately per group. Prediction: only the `both` group
cross-decodes. Run raw **and** per-condition-mean-removed (a transfer that
collapses after mean removal was a univariate offset, not a code).

**(b) Set transfer.** The same label decoded within each electrode set (compare
where a code lives).

**(c) Temporal generalization (Fig 10).** Train at *t*, test at *t′*; off-diagonal
generalization → sustained/stable code, narrow diagonal → moving/phasic code.

**Pseudopopulation.** Subjects don't share trials, so pool electrodes and build
**pseudo-trials** by matching on the full condition cell, with train/test drawn
from **disjoint reservoirs** (the circularity guard).

**The payoff 2×2** (reading §5 and §7 together):

| | Cross-decodes | Doesn't cross-decode |
|---|---|---|
| **Co-localized (OR>1)** | shared substrate / shared code | mixed selectivity, orthogonal codes |
| **Not co-localized (OR<1)** | (rare) shared low-D code across sites | **distinct substrates** |

### 7.2 Timing (§5 → Fig 6)

Does stability information arise earlier than flexibility? Interaction magnitude
**over time** (the `effect_measure='cluster'` per-bin d-o-d), then **onset =
first upward crossing of 50% of that effect's own peak**, compared with the
**Ulrich–Miller jackknife** (onsets on leave-one-subject-out grand-averages,
`(N−1)`-corrected paired t). Peak-normalization neutralizes the latency–amplitude
confound: if `stab(t) = k·flex(t)`, both cross 50%-of-peak at the same time (a unit
test pins this). Report onset **and** peak latency; a claim rests on both agreeing.

### 7.3 Brain–behavior (§6)

Tie neural selectivity to the behavioral adjustment. Across-subjects (n = subjects,
underpowered) and within-subject single-trial (preferred): does trial-by-trial HG
in the LWPC group predict the trial-by-trial congruency-sequence RT adjustment
(LWPS ↔ switch)? The *matched* pairing should beat the *cross* pairing — that gap
is the specificity result.

---

## 8. Circularity between definition and decoding (the disjoint trial split)

When decoding is restricted to a *selected* electrode set chosen on the **same
trials** the decoder then scores, selection biases accuracy upward — double-dipping.
Two independent guards, for two different leaks:

- **§3.2's ignore-the-diagonal rule** removes the *contrast-level* leak: never
  report a cell decoded on the electrodes its own interaction defined.
- **The disjoint trial split** (`src/analysis/decoding/trial_splitting.py`) removes
  the *trial-level* leak even for off-diagonal cells: define electrodes on `P_def`,
  decode on the disjoint `P_dec`.

Primitives (unit-tested in `tests/analysis/decoding/test_trial_splitting.py`):
`stratified_trial_split` (disjoint, stratified, deterministic),
`strata_key_from_metadata`, `select_responsive_channels` (held-out selector, FDR
across channels), and the orchestrator `apply_electrode_definition_split`. Wiring is
**off by default** (`ELECTRODE_DEFINITION_SPLIT = True` to enable); outputs get a
`_defsplit` tag. It is I/O glue — **smoke-test on one subject** (trial counts drop
~`frac_def`, a plausible electrode set survives, the decode still runs) before a
full re-run.

**Which guard when.** Selecting on an *orthogonal* contrast (task-responsiveness,
`electrodes='sig'`) is the standard Kriegeskorte defense and only modestly inflates.
Selecting on the *decode contrast itself* (the diagonal) is full double-dipping and
must use the disjoint split **or** be dropped by §3.2. `electrodes='all'` has no
selection and no circularity — the currently-safe default for the decoding figures.

---

## 9. How to run everything, and in what order (start here to *do* the analysis)

Every module has a **synthetic dry run** that validates the whole path in seconds
with ground-truth data — always run that first to confirm your environment before
pointing `EPOCHS_ROOT_FILE` at real data. Every module is also directly runnable
(`python src/analysis/stats/<module>.py`) for a synthetic smoke test.

### 9.0 Order of the analysis (and why this order)

The dependency chain is **A0 → A1 → {A2, A3, A6} → A4 → A5**:

1. **A0 — get the pipeline running and read the segregation module.** Everything
   either calls into or mirrors `stability_flexibility_segregation.py`.
2. **A1 — electrode definition** (§3). Produces the S/F/CS/SI labels every later
   step consumes. Nothing downstream is trustworthy until A1 is.
3. **A2 — conjunction** (§5). Needs A1's labels. Natural next step: it and A1 share
   most scaffolding.
4. **A3 — anatomy** (§6) and **A6 — brain–behavior** (§7.3) are **independent** of
   each other and can slot in any time after A1.
5. **A4 — cross-decoding** (§7.1) and **A5 — timing** (§7.2) are the larger,
   mostly-greenfield pieces; do them once A1/A2 give a trustworthy definition.

### 9.1 Order to walk the tutorial notebooks

Walk them in the same dependency order; each is synthetic and runs anywhere:

| # | Tutorial notebook | Covers | Read alongside |
|---|---|---|---|
| 1 | `src/analysis/stats/stability_flexibility_assignments_sandbox.ipynb` | A1→A6 end to end, fill-in-the-blank, with `reveal("aN_solution")` | this guide, all sections |
| 2 | `src/analysis/stats/stability_flexibility_segregation_tutorial.ipynb` | A1 definition + A2 conjunction/correlation | §3, §5 |
| 3 | `src/analysis/stats/stability_flexibility_anatomy_tutorial.ipynb` | A3 coverage-conditioned ROI enrichment (incl. `attach_roi` line-by-line) | §6 |
| 4 | `src/analysis/decoding/trial_splitting_tutorial.ipynb` | the disjoint def/decode split + double-dip demo | §8 |
| 5 | `src/analysis/decoding/cross_decoding_tutorial.ipynb` | A4 pseudo-trials + label/set/temporal transfer | §7.1, §3.2 |
| 6 | `src/analysis/stats/stability_flexibility_a5_a6_tutorial.ipynb` | A5 timing + A6 brain–behavior | §7.2, §7.3 |

Start with the **sandbox** for the whole arc, then take the per-analysis tutorials
in order for depth. The tutorials aim to explain each function **line by line**
(the `attach_roi` walk-through in §6.1 is the template).

### 9.2 How to run each analysis

**A1 / A2 — definition + conjunction** (from `dcc_scripts/stats`):
```bash
DATA_SOURCE=synthetic bash submit_stability_flexibility_anova_conjunction_dcc.sh   # dry run
bash submit_stability_flexibility_anova_conjunction_dcc.sh                          # real (set EPOCHS_ROOT_FILE first)
# continuous correlation + CMH on their own:
DATA_SOURCE=synthetic bash submit_stability_flexibility_segregation_dcc.sh
```
Key knobs (env vars): `EPOCHS_ROOT_FILE`, `WINDOW_TMIN/TMAX` (default 0.0/0.5),
`CONTRAST_MODE` (**use `proportion`** for the interactions, §3.1), `EFFECT_MEASURE`
(**use `cluster`** to keep the time dimension, §4), `N_SPLITS`, `N_PERM_CORR`,
`N_PERM_LABEL`. Set `USE_TIME_PERM_CLUSTER = True` in the segregation module for the
real permutation cluster mask (slower). Outputs land in
`results/<tag>/window_<tmin>to<tmax>s_<electrodes>/<CONTRAST_MODE>_<EFFECT_MEASURE>/`
(`labels.csv`, `conjunction.json`, `correlation.json`, `segregation_summary.png`).

**A3 — anatomy** (from `dcc_scripts/stats`):
```bash
DATA_SOURCE=synthetic bash submit_stability_flexibility_anatomy_dcc.sh                        # planted enrichment
DATA_SOURCE=synthetic SYNTHETIC_ENRICHMENT=0.0 bash submit_stability_flexibility_anatomy_dcc.sh  # null -> n.s.
bash submit_stability_flexibility_anatomy_dcc.sh                                              # real
```

**A4 — cross-decoding** (from `dcc_scripts/decoding`):
```bash
DATA_SOURCE=synthetic SYNTHETIC_CODE=shared     bash submit_stability_flexibility_cross_decoding_dcc.sh  # should transfer
DATA_SOURCE=synthetic SYNTHETIC_CODE=orthogonal bash submit_stability_flexibility_cross_decoding_dcc.sh  # should NOT
bash submit_stability_flexibility_cross_decoding_dcc.sh                                                   # real
```
On real data this now also emits `within_block_by_group` — the per-group 2×2 with
the diagonal (define==decode) cell ignored (§3.2). `ALPHA` sets the A1 FDR
threshold for the four groups; `MIN_GROUP_SIZE` skips groups too small to decode.

**Disjoint def/decode split** (from `dcc_scripts/decoding`):
```bash
bash submit_decoding_with_electrode_definition_split_dcc.sh
FRAC_DEF=0.6 SEED=1 ALPHA=0.05 STRATA=congruency,switchType,blockType \
    bash submit_decoding_with_electrode_definition_split_dcc.sh
```

**A5 / A6 — timing + brain–behavior** (no DCC launcher; notebook + smoke test):
```bash
python src/analysis/stats/stability_flexibility_timing.py          # synthetic smoke test
python src/analysis/stats/stability_flexibility_brain_behavior.py  # synthetic smoke test
```

**Preprocessing (produces the shared HG epochs everything consumes):**
```bash
python src/analysis/preproc/make_epoched_data.py --passband 70 150 --subjects D0057
```
The `<name>` it writes (e.g. `Stimulus_1sec_preStimulusBase_decFactor_10`) is the
`EPOCHS_ROOT_FILE` the analyses load back.

---

## 10. Function / file map (grep targets)

| Concept | Symbol | File |
|---|---|---|
| Four-interaction electrode definition | `per_electrode_anova_labels` | `src/analysis/stats/stability_flexibility_segregation.py` |
| Nonparametric definition (cross-check) | `per_electrode_labels` | same |
| Balanced d-o-d effect (window mean) | `_interaction_cohens_d` | same |
| Balanced d-o-d effect (time-resolved, cluster) | `_interaction_cluster` | same |
| Signed interaction estimator (sign source) | `_interaction_effect` | same |
| Conjunction (CMH) | `cmh_conjunction` | same |
| Permutation null / threshold sweep | `conjunction_permutation_null`, `conjunction_threshold_sweep` | same |
| Continuous correlation | `subject_clustered_corr` | same |
| Double-dip diagonal map + predicates | `DEFINITION_DECODE_DIAGONAL`, `is_circular_decode`, `circular_decode_for_group` | `src/analysis/decoding/cross_decoding.py` |
| Within-block 2×2 decode | `within_block_decoding_baseline` | same |
| Cross-decode (label/set transfer) | `cross_decode` | same |
| Temporal generalization | `temporal_generalization` | same |
| Four-group derivation (DCC) | `_interaction_groups`, `_electrode_groups` | `dcc_scripts/decoding/stability_flexibility_cross_decoding_dcc.py` |
| Anatomy join | `attach_roi`, `_derive_group` | `src/analysis/stats/stability_flexibility_anatomy.py` |
| Coverage + enrichment | `build_coverage_matrix`, `roi_group_enrichment_test` | same |
| Disjoint trial split | `stratified_trial_split`, `apply_electrode_definition_split` | `src/analysis/decoding/trial_splitting.py` |
| power_traces windowed ANOVA (temporal-profile figure) | `run_within_electrode_windowed_anova_cluster_correction`, `load_significant_electrodes` | `src/analysis/power/windowed_anova.py` |
| Timing | `interaction_time_course`, `onset_50pct_peak`, `jackknife_onset_difference` | `src/analysis/stats/stability_flexibility_timing.py` |
| Brain–behavior | `subject_level_brain_behavior`, `trialwise_brain_behavior` | `src/analysis/stats/stability_flexibility_brain_behavior.py` |

**Tests** (run with `pytest`): `tests/analysis/stats/test_stability_flexibility_anova_labels.py`
(four-group definition), `tests/analysis/decoding/test_cross_decoding_circularity.py`
(diagonal guard), `tests/analysis/decoding/test_trial_splitting.py` (disjoint split),
plus the A5/A6 tests.
