# N4 §19 with electrodes as the unit

*Added 2026-10-06. Companion to §19 of
[`n4_continuous_anatomy.md`](n4_continuous_anatomy.md). Code is in place and
tested; the electrode-level numbers below marked **[run]** need one rerun of
`n4_section19_followups.py` (§5).*

## 1. The decision

§19 added participant-level versions of the anatomy tests after the advisors
asked whether participant is a random effect. The pre-specified tests treat
participant as a **fixed** effect and electrodes as the units of inference.
That is the level the paper reports at: **electrodes as the unit in the main
text, participants as the unit in the supplement** (S-N4), for a reviewer who
asks how far the results generalise across participants.

Every §19 part now reports both, electrode level first. Nothing that was
computed before has changed: the participant-level numbers in §19.8 of the N4
doc stand, and the electrode-level estimates are the same estimates with a
different inference.

What "electrodes as the unit" means here: participant enters as a fixed effect
(centring or dummies), and the uncertainty comes from which electrodes were
sampled and from trial noise, given these participants. It does not say how
the result would vary in new participants; that is what the supplement's
participant-level tests are for.

## 2. Each part at the two levels

| Part | Electrodes as the unit (main text) | Participants as the unit (supplement) |
|---|---|---|
| §19.1 height slope | `sfa.coordinate_slope_by_electrode`: the coordinate test's slope and swap-null p (each electrode's LWPC and LWPS swapped), plus a 95 % interval from an electrode bootstrap within participant (each participant keeps its electrode count; residualised values recentred per draw) | `coordinate_slope_by_participant` (weighted sign-flip, unweighted t, mixed models), `coordinate_slope_loso` |
| §19.2 overlap r | `sfs.electrode_split_corr`: `split_resolved_corr`'s r and permutation p, plus electrode-bootstrap intervals for r, both within-participant reliabilities and the noise-corrected r (100 evenly spaced splits; the noise-corrected interval only when both reliabilities are positive in ≥ 90 % of draws) | `participant_split_corr` |
| §19.3 local similarity | `sfa.local_similarity`'s `*_electrode` columns: SE, ± 1.96 SE interval and one-sided p for every bin, the contrast and the reliability; the share-of-reliability comparison by the delta method (§3) | the same function's participant sign-flip p and participant-bootstrap intervals |
| §19.4 Figure 5 | `sfa.figure5_height(unit='electrode')` → `fig5_height.png`: c, 95 % electrode-bootstrap ellipses; d, electrode means ± SEM, annotated with the slope, its electrode-bootstrap CI and p | `unit='participant'` → `fig5_height_participants.png`: participant-bootstrap ellipses, participant means ± SEM, participant and random-slope p |
| §19.7 overlap controls | already electrode level (within-participant permutation) | leave one participant out (`overlap_loso.csv`) |

On the shared-split table (section 3 of the script), `electrode_split_corr` runs
on LWPC–LWPS and, from a main-effects table, on congruency–switch. That gives
the within-participant reliabilities the paper quotes (LWPC 0.21, LWPS 0.04,
congruency 0.46, switch 0.35 on raw high gamma) their electrode-level
intervals, an interval on congruency − switch (the paper's "switch type is the
less reliable map"), and the noise-corrected overlap its interval, which §19.8
said it needed before it could be quoted. On the pipeline's per-electrode split
the reliabilities are biased (§19.3 of the N4 doc), so section 2 prints r and p
there but not the reliabilities.

## 3. Local similarity: why it needed new inference, and what it uses

The obvious electrode-level test, shuffling electrode positions within
participant, is invalid here. Neighbouring contacts share trial noise, so one
dataset's estimation noise is spatially smooth, near pairs are not
exchangeable with far ones, and the shuffle's null is too narrow: 13 % false
positives at α = 0.05 (§19.3 of the N4 doc). That is why §19.3 used participant
sign flips. An electrode-level test has to carry the shared noise explicitly.

**The method** (`_LocalElectrodeCov`). Write each split's halves as
S ± D: S = (A + B)/2 is the full-data score, D = (A − B)/2 the half
difference. With one split per participant, D is the electrodes' trial noise
with random signs, so `mean_k D_k D_k'` estimates the noise covariance between
every pair of a participant's electrodes, including what neighbours share.

1. Every local-similarity estimate is a quadratic form in S. A participant's
   excess in a bin is Σ W_ij C_ij with W_ij = 1[pair in bin] − (the bin's share
   of the participant's pairs), the second term being the position-shuffle
   baseline's exact expectation; the reliability is the trace of C; the
   contrast is a difference of two bins' forms.
2. Under no local structure, the covariance of S across a participant's
   electrodes is Γ = σ² P + N: exchangeable signal (σ² from the same-electrode
   cross-half similarity, P the within-participant centring) plus the noise
   covariance N from the half differences.
3. The variance of a Gaussian quadratic form is tr(WΓWΓ)/2 (Isserlis), summed
   over participants. The per-split standardisation divides every estimate by
   the scores' mean square; the delta method carries that through. It matters
   for the reliability (its SE shrinks by 1 − reliability), hardly for an
   excess near zero.
4. p is one-sided normal (more similar than baseline), intervals ± 1.96 SE.

The point estimates are unchanged: `excess` already pooled every electrode
pair. Only the inference is new.

**Calibration.** Null worlds built like the real data: 21 participants, 2–4
shafts of 4–7 contacts 3.5 mm apart, trial noise correlated between contacts
as exp(−(d/ℓ)²), halves that are disjoint trial sets of one dataset, 40 shared
splits, no local structure. Near bin (< 10 mm), α = 0.05, 300 datasets per
row, range over LWPC, LWPS and the balance:

| World | Reliability | Electrode level | Participant sign-flip | SD of excess / mean SE |
|---|---|---|---|---|
| ℓ = 5 mm | 0.24 | 4.0–4.3 % | 1.7–3.0 % | 0.94–1.01 |
| ℓ = 8 mm | 0.26 | 4.0–4.7 % | 1.7–3.0 % | 0.89–0.98 |
| ℓ = 5 mm | 0.49 | 4.3–5.7 % | 2.3–4.3 % | 1.02–1.10 |
| ℓ = 5 mm | 0.80 | 5.0–6.7 % | 1.0–4.0 % | 1.10–1.19 |

The nearest-minus-farthest contrast behaves the same (3.3–7.0 %). The
reliability's SE matches its spread to within about 10 % at reliabilities up to
0.5.
At reliability 0.8 the test runs slightly liberal; with Pearson instead of
Spearman it does not (SD/SE 1.00–1.13), so the excess comes from ranking. The
real reliabilities are 0.0–0.46, the calibrated range.

**Power.** With a smooth field planted in LWPC only (field SD 0.6, length
6 mm): near-bin detection at α = 0.05 in 97 % of datasets at the electrode
level against 93 % with participant sign flips; for the balance, 56 % against
40 %. LWPS, which had no field, stayed at 3.7 % and 2.0 %.

**What the interval is.** The SE is computed under no local structure, so it
is the right yardstick for the p-value. When an excess is real, the true spread
is larger: in the planted world the excess's SD was about 1.3 times the SE.
Report the electrode-level p; treat the ± 1.96 SE interval as approximate when
the excess is clearly non-zero.

**What it does not change.** The reading table in §19.3 of the N4 doc holds at
either level. With reliabilities near zero (the balance's −0.04 on all lPFC) the
null still cannot show intermixing; what the electrode level adds is power for
the positive control and a tighter interval on the balance's own reliability.

Tests: `tests/analysis/stats/test_section19_anatomy.py`, section "electrodes as
the unit". One of them repeats the calibration on 40 smaller null datasets
(excess SD / SE within 0.7–1.35; reliability within 0.6–1.4).

## 4. Outputs

New runs of the anatomy job and `n4_section19_followups.py` write, in addition
to the §19.5 files:

| File | What |
|---|---|
| `summary_section19.txt` | every part twice: `..., ELECTRODES AS THE UNIT` first, `..., PARTICIPANTS AS THE UNIT` second |
| `section19.json` | new keys `slope_by_electrode`, `electrode_corr`, `main_effects_shared`, `figure5_height_participants`; `overlap_shared` now carries the intervals |
| `local_similarity.csv`, `_contrasts.csv`, `_comparison.csv` | the participant-level columns as before, plus `se_electrode`, `excess_lo_electrode`/`_hi_electrode`, `p_greater_electrode`, `similarity_lo_electrode`/`_hi_electrode` (same-electrode row), `relative_*_electrode`; comparison `*_electrode` and `note_electrode` |
| `local_similarity.png` | **now electrode level** (± 1.96 SE) |
| `local_similarity_by_participant.png` | the participant-bootstrap version (was `local_similarity.png`) |
| `fig5_height.png/.pdf`, `fig5_height_centroids.csv`, `fig5_height_balance.csv` | **now electrode level**; the balance table has `n_electrodes` |
| `fig5_height_participants.png/.pdf`, `_participants_centroids.csv`, `_participants_balance.csv`, `fig5_height_balance_by_participant.csv` | the participant-level figure (was `fig5_height.*`) |
| `fig5_height_brain.png`, `fig5_height_brain_centroids.csv` | the bands and their centroids on the brain (`--brain` / `MAKE_BRAIN`; N4 §19.4). The same at both levels, so drawn once, with the electrode-level figure |

Two file names changed meaning: `local_similarity.png` and `fig5_height.*`
are electrode level now. Older folders hold the participant-level versions
under those names.

## 5. Getting the electrode-level numbers

Same commands as the 2026-10-05 runs (§19.5 and §19.8 of the N4 doc), all
sections. The variables are the ones used there.

```bash
git pull
# primary: all lPFC, raw high gamma, shared splits for section 3
python dcc_scripts/stats/n4_section19_followups.py \
  --anatomy-dir $R/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous \
  --seg-dir $SEG \
  --long-df $RAW/long_df.csv
# RT-adjusted companion (writes section19_rt_adjusted/)
python dcc_scripts/stats/n4_section19_followups.py \
  --anatomy-dir $R/anatomy_a1_lpfc_window_0.0to1.5s_sig/continuous \
  --seg-dir $SEG \
  --long-df $RTRUN/long_df.csv --rt-coupling $RTRUN/rt_adjustment_slopes.csv
```

Each takes the rescoring time (5–10 minutes) plus about 2 minutes. The
participant-level numbers come out the same as on 2026-10-05 (same seeds).

Numbers to collect for the main text **[run]**: the z slope's
electrode-bootstrap CI; the overlap r's CI; the shared-split reliabilities'
CIs, the congruency − switch reliability difference with its CI, and the
noise-corrected r with its CI (or "not estimable", likely with LWPS at 0.04);
every local-similarity p and interval at the electrode level.

## 6. What to report where

Supersedes the reporting order of §19.6 of the N4 doc.

| Result | Main text (electrode level) | Supplement (participant level) |
|---|---|---|
| Height slope | −0.0077 SD/mm, p = 0.0074, 95 % CI **[run]** | weighted sign-flip p = 0.041, random slope p = 0.037, unweighted p = 0.078, leave-one-out p 0.002–0.099 |
| Overlap r | 0.097, p = 0.0002, 95 % CI **[run]**; on shared splits 0.111, p = 0.0002 | weighted r = 0.098, p = 0.031; unweighted 0.043, p = 0.35 |
| Reliabilities (shared splits) | LWPC 0.21, LWPS 0.04, congruency 0.46, switch 0.35, each with CI **[run]** | participant-bootstrap CIs in `local_similarity.csv` |
| Local similarity | electrode-level p and interval per score **[run]**, with the positive control | §19.8's participant-level table |
| Fig. 5 c, d | `fig5_height.png` | `fig5_height_participants.png` |
| Overlap controls | as in §19.8 | leave-one-out range |

Methods and Results wording: [`paper_draft.md`](paper_draft.md) §2.5 and §3.4.
