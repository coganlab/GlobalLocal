# The subspace test and the trajectories: the math, worked through

This doc explains the math behind the subspace test and the trajectories of the
[subspace trajectory plan](analysis_plans.md#subspace-trajectory-plan) (§3.4,
§4), step by step with small examples. It starts from "what is a vector" and
ends at the numbers the job writes. Each section answers one of the questions
in the margin notes on the plan.

- **The design and the reasons for each choice:**
  [`analysis_plans.md` › Subspace trajectory plan](analysis_plans.md#subspace-trajectory-plan).
- **How to run the job and read its outputs:** [`coding_axes.md`](coding_axes.md).
- **The code:** `src/analysis/stats/coding_subspaces.py` (§3.4) and
  `src/analysis/stats/coding_trajectories.py` (§4).

Every number below was computed with numpy; [§11](#11-reproduce-the-numbers)
has the code.

## Contents

0. [From your notes to the code](#0-from-your-notes-to-the-code)
1. [The objects: an electrode pattern is a vector](#1-the-objects-an-electrode-pattern-is-a-vector)
2. [Dot product, cosine, projection](#2-dot-product-cosine-projection)
3. [What "the subspace of C" means](#3-what-the-subspace-of-c-means)
4. [The SVD, and how U spans the patterns](#4-the-svd-and-how-u-spans-the-patterns)
5. [Why uncentred SVD and not PCA](#5-why-uncentred-svd-and-not-pca)
6. [Choosing k: "does k capture variance?"](#6-choosing-k-does-k-capture-variance)
7. [Principal angles](#7-principal-angles)
8. [The alignment index](#8-the-alignment-index)
9. [Noise: halves, ceiling, null, bootstrap](#9-noise-halves-ceiling-null-bootstrap)
10. [The trajectories](#10-the-trajectories)
11. [Reproduce the numbers](#11-reproduce-the-numbers)

---

## 0. From your notes to the code

| Your note | What it is | Where |
|---|---|---|
| "Effects per electrode over time … the first part of get_similarity or something" | The coding axes: each electrode's incongruent − congruent (and switch − repeat) mean HG in every A4 window, on two trial halves | `main_effect_mean_diffs` in `stability_flexibility_segregation.py`; saved as `per_split_windows.csv` (`mxA`, `mxB` = congruency on halves A, B; `myA`, `myB` = switch type) |
| "Can use regression if I add a trial RT covariate" | The RT control | `RT_CONTROL=adjust` (removes each electrode's within-cell HG-on-RT slope before scoring), or `match` / `random` |
| `uncentered_SVD(matrix, k=3)` | The top-k left singular vectors of an effect matrix | `coding_subspace(effects, k)` |
| "keep k dimensions that replicate across halves" | Held-out variance per dimension against the shuffle null | `dimension_test` |
| `principal_angles_between_two_subspaces(...)` | Cosines of the principal angles | `principal_angle_cosines(basis_1, basis_2)` |
| `alignment_index(matrix1, matrix2)` | The share of one variable's effect in the other's subspace | `alignment_index(effects, basis)`: the matrix whose effect is projected, and the other variable's basis |
| "Split-half ceiling and label-shuffle null … incorporate these into the above functions" | Everything computed across halves, the same between halves of one variable, and under shuffles | `compare_subspaces` (with `coding_axes.null_draws`) |
| Trajectories | Held-out condition means in the plane of the two axes | `condition_trajectories` |

The functions take numpy arrays, so you can call them on toy matrices like the
ones below.

## 1. The objects: an electrode pattern is a vector

**A vector is a list of numbers, one per electrode.** With ~400 lPFC electrodes
pooled across participants, an electrode pattern is 400 numbers. Picture a
space with 400 axes, one per electrode. A pattern is an arrow in that space,
and its direction says which electrodes go up and which go down, by how much
relative to each other. Two patterns point the same way if they raise the same
electrodes in the same proportions, whatever their overall size.

**The number for each electrode is the score.** You asked how the congruency
effect "is operationalized and measured. The score, right?" Yes. For electrode
e and window t:

```
c_e(t) = 1/2 [ (mean HG incongruent − mean HG congruent | 25 % incongruent blocks)
             + (mean HG incongruent − mean HG congruent | 75 % incongruent blocks) ]
```

It is in HG units (not Cohen's d), and each proportion block counts equally, so
the block a trial came from cannot leak in. The switch-type score s_e(t) is the
same for switch − repeat over the switch-proportion blocks. It is a regression
beta too (plan §3.1): regress HG on congruency, the incongruent-proportion
block and their interaction, all effect-coded (±½). That model fits the four
cell means exactly, so its congruency beta is the equal-weight average of the
two within-block differences, which is the score. Adding trial RT as a
covariate with one slope per electrode and window is what `RT_CONTROL=adjust`
does: it removes the pooled within-cell HG-on-RT slope from every trial before
scoring.

**Pooling across participants.** The electrodes were not recorded at the same
time. That does not matter: each entry is a condition-mean difference, so the
geometry only needs one number per electrode per condition contrast.

**The effect matrix C.** Stack the windows side by side. C is electrodes ×
windows: column t is the congruency pattern in window t (a 400-vector), and
row e is electrode e's congruency effect over time. S is the same for switch
type. Over 0–1.5 s with 250 ms windows every 62.5 ms there are 21 windows, so C
is about 400 × 21.

## 2. Dot product, cosine, projection

Take three electrodes, a congruency pattern c and a switch pattern s:

```
c = (2, 1, 0)        s = (1, 1, 1)

dot product   c · s = 2·1 + 1·1 + 0·1 = 3
lengths       |c| = √(4 + 1 + 0) = 2.236     |s| = √3 = 1.732
cosine        cos θ = c · s / (|c| |s|) = 3 / (2.236 · 1.732) = 0.775
angle         θ = 39.2°
```

- **The cosine** is the dot product with the sizes divided out: 1 = same
  direction, 0 = perpendicular (90°), −1 = opposite. Plan §3.1 explains why
  this, and not a correlation, is the measure of shared direction.
- **Projection.** ŝ = s / |s| is s scaled to length 1, a unit vector.
  c · ŝ = 3 / 1.732 = 1.732 is how far c reaches along s's direction, and
  equals |c| cos θ = 2.236 × 0.775.
- **Fraction captured.** The share of c's squared length that lies along s is
  (c · ŝ)² / |c|² = 3 / 5 = 0.6 = cos² θ. The alignment index (§8) is this
  number, generalized to many windows and many dimensions.

Lengths add up in squares: c's squared length 5 splits into 3 along s and 2
perpendicular to s (Pythagoras). Squared lengths are the "variance" everything
below counts, uncentred: a sum of squares, with nothing subtracted.

## 3. What "the subspace of C" means

This is the passage in your notes that ends "Huh..??? How does this work????".
Two small examples, 4 electrodes × 3 windows.

**One fixed pattern that grows and shrinks.** p = (1, 2, 0, −1), with strength
0.5, 1.0, 0.5 in the three windows:

```
        win1  win2  win3
C =  [  0.5   1.0   0.5 ]     column t = (strength in window t) × p
     [  1.0   2.0   1.0 ]
     [  0.0   0.0   0.0 ]
     [ −0.5  −1.0  −0.5 ]
```

Every column is a multiple of p. Knowing p and one number per window
reconstructs C entirely. The effect uses **one direction** in electrode space.
C has rank 1, and its subspace is the line through p.

**One pattern early, another late.** p in windows 1–2, q = (0, 1, 2, 1) in
window 3:

```
        win1  win2  win3
C =  [   1     1     0  ]
     [   2     2     1  ]
     [   0     0     2  ]
     [  −1    −1     1  ]
```

No single direction reconstructs all three columns: windows 1–2 point along p,
window 3 along q, and p · q = 1 (nearly perpendicular; cos = 1/6). C has rank
2. Its subspace is the plane of all combinations a·p + b·q, and every window's
pattern lies in that plane.

**So "the subspace congruency uses" is the set of electrode patterns its
effect takes over the trial**: a line if the pattern is fixed, a plane if it
changes between two patterns, and so on. The question "do congruency and switch
share a direction?" is then a question about how these subspaces sit relative
to each other.

**With noise**, every column gets a little random pattern added, and the matrix
technically uses all 21 directions. The real question becomes which directions
carry more than noise. That is §6.

## 4. The SVD, and how U spans the patterns

The SVD writes any matrix as

```
C = U Σ Vᵀ
```

| Piece | Shape | What it is |
|---|---|---|
| U | electrodes × r | r unit-length electrode patterns, mutually perpendicular ("orthonormal"): u₁, u₂, … |
| Σ | r × r, diagonal | strengths σ₁ ≥ σ₂ ≥ … ≥ 0, one per pattern |
| V | windows × r | for each pattern, a unit-length time course: how much each window uses it |

Read it column by column. Window t of C is

```
c(t) = σ₁ V[t,1] u₁ + σ₂ V[t,2] u₂ + …
```

**Every window's pattern is a weighted mix of the u's.** That is how U's
columns "span the patterns congruency takes": by construction, every column of
C is built out of them.

**The rank-1 example.** The SVD gives σ = (3, 0, 0). u₁ = ±p/|p| =
±(0.41, 0.82, 0, −0.41), and v₁ = ±(0.41, 0.82, 0.41), the strength profile
(0.5, 1, 0.5) scaled to unit length. One nonzero σ: one direction. (3 = |p| ×
|strengths| = √6 × √1.5.)

**The rank-2 example.** σ = (3.51, 2.38, 0), so σ² = (12.3, 5.7, 0). These add
up to 18, C's total sum of squares (each column has squared length 6). Two
nonzero σ: two directions. u₁ and u₂ lie in the plane of p and q but are not p
and q themselves. The SVD picks perpendicular directions ordered by strength.
Only the subspace matters, not the particular basis.

**Why the top k.** σ_j² is the sum of squares of C along u_j. A theorem
(Eckart–Young) says the top k u's capture the most of C's sum of squares that
any k-dimensional subspace can: σ₁² + … + σ_k². In the rank-2 example, the best
single direction keeps 12.3 of 18 (68 %), and the best plane keeps all 18.

**The sign is arbitrary.** −u₁ with −v₁ gives the same product, so software can
return either. Anything that depends on the sign of a singular vector (like a
correlation between two PC1s) has an arbitrary sign. Subspaces have no sign:
the line through u₁ is also the line through −u₁.

**Which library.** `np.linalg.svd`. scikit-learn's `PCA` centres first, which
is the wrong thing here (§5). Its `TruncatedSVD` does not centre, but it is a
randomized, approximate solver for huge sparse matrices; 400 × 21 is tiny, and
the exact SVD takes milliseconds.

## 5. Why uncentred SVD and not PCA

**PCA is the SVD after subtracting each row's mean.** Here a row is one
electrode's effect across windows. Subtracting it removes each electrode's
time-averaged effect before looking for patterns.

The plan's example, with 3 electrodes so it can be done by hand: a sustained
pattern u = (1, 0, 0) in all six windows, plus a second pattern v = (0, 1, 0)
only in the last three.

```
            early (u)     late (u + v)
C =  [  1   1   1   |   1   1   1  ]     electrode 1: always on
     [  0   0   0   |   1   1   1  ]     electrode 2: on late only
     [  0   0   0   |   0   0   0  ]

PCA centres each row:
     [  0   0   0   |   0   0   0  ]     electrode 1's constant effect is gone
     [−.5  −.5  −.5 |  .5  .5  .5  ]
     [  0   0   0   |   0   0   0  ]
```

- **PCA:** one nonzero σ (1.22), PC1 = (0, 1, 0) = v. It finds what changed
  between early and late, and loses u entirely, even though u is in every window
  and is the biggest part of the effect.
- **Uncentred SVD:** σ = (2.80, 1.07, 0), and U spans the plane of (1, 0, 0) and
  (0, 1, 0), that is {u, v}. (Its basis vectors are (0.85, 0.53, 0) and
  (−0.53, 0.85, 0): the same plane, rotated.)

PCA answers "how does the code change over time". The subspace test asks "what
is the code", so it must not subtract the sustained part. Over 0–1.5 s much of
the congruency effect is probably sustained, so this would bite.
`tests/analysis/stats/test_coding_subspaces.py::test_uncentred_svd_keeps_the_sustained_pattern_that_pca_drops`
pins it.

## 6. Choosing k: "does k capture variance?"

Yes, but **held-out** variance, measured against noise, and not the variance of
the matrix the SVD was fitted to.

**Why in-sample variance cannot choose k.** The SVD orders directions by how
much of *that matrix* they capture, and noise has structure too. The σ's of a
pure-noise matrix still fall off smoothly, so "variance explained" always looks
like a few big components and a tail. A pure-noise version of our matrix (400
electrodes, the 21 overlapping A4 windows):

| Dimension | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| In-sample share of the sum of squares | 19.5 % | 18.2 % | 14.6 % | 11.4 % | 10.0 % |
| In-sample σ_j² | 24.4 | 22.8 | 18.3 | 14.2 | 12.4 |
| Held-out: captured of an independent noise matrix | 0.14 | 0.73 | 0.17 | 0.21 | 0.47 |

A "95 % of variance" rule keeps 12 of the 21 dimensions here, and every one of
them is noise. With signal present, the cutoff still depends on how noisy and
how smooth the matrix is, not on whether a dimension is real.

**Held-out variance.** Fit u_j on half A. Then measure how much of half B's
effects it captures:

```
h_j = ||u_jᵀ C_B||² = Σ_t (u_j · c_B(t))²
```

That is the projection of §2, squared and summed over windows. A real
dimension of the congruency code is there in both halves, so it captures B's
effect. A noise dimension of half A is a random direction as far as half B's
noise is concerned. It captures only about 1/400 of B's sum of squares, and
in-sample it looked like 20 % (the table's 24.4 vs 0.14).

**Against what?** Compare h_j with the same number when congruency's labels are
shuffled (§9). Then there is no congruency code, but the noise has exactly the
data's structure: the same trials, the same correlations between a
participant's electrodes, the same design. Dimension j **replicates** if h_j is
above the shuffle null's 95th percentile. k for congruency is the number of
leading dimensions that replicate, stopping at the first failure. (Testing in
order and stopping keeps the false-positive rate at 5 % overall.) The same goes
for switch, and the **primary k** is the smaller of the two, so both subspaces
are real at it.

Symmetry: h_j is averaged over fitting on A and testing on B and the reverse,
and over the 200 splits. Outputs: `subspace_dimensions.csv`, and panels c–d of
`subspace_test.png` (filled dots = kept).

## 7. Principal angles

Your note: "Uh still don't get the math dude." Build it up one dimension at a
time.

**Between two lines (k = 1).** A line through the origin has no arrowhead: the
line through u is the line through −u. The angle between two lines is
therefore between 0° and 90°, with cos θ = |u₁ · u₂| for unit vectors.

**Between two planes (k = 2).** In ordinary 3-D, take the floor (spanned by
e₁ = (1, 0, 0) and e₂ = (0, 1, 0)) and a plane tilted 45° along the e₁ axis
(spanned by e₁ and (0, 1, 1)/√2).

- θ₁: the smallest angle between any line in one plane and any line in the
  other. Both planes contain the e₁ line, so θ₁ = 0°.
- θ₂: the smallest angle between lines that are perpendicular to that first
  pair. In the floor that is e₂; in the tilted plane, (0, 1, 1)/√2. Their
  angle is 45°, so θ₂ = 45°.

A small θ₁ means the subspaces share at least one direction. A small θ_k (all
angles small) means they are the same subspace.

**Why the cosines are the singular values of U_Cᵀ U_S.** U_C's columns are
orthonormal, so every unit vector in its subspace is U_C a for some unit vector
a (a holds the coordinates), and likewise U_S b. Their dot product is

```
(U_C a) · (U_S b) = aᵀ (U_Cᵀ U_S) b
```

M = U_Cᵀ U_S is a small k × k matrix. Its entry (i, j) is the cosine between
basis vector i of one subspace and basis vector j of the other. The largest
value aᵀ M b can take over unit a and b is, by definition, M's largest singular
value. So cos θ₁ = σ₁(M). Searching again perpendicular to the best pair gives
σ₂(M), and so on. In the planes example:

```
U_floor = [1  0]     U_tilted = [1   0  ]     M = U_floorᵀ U_tilted = [1   0  ]
          [0  1]                [0  .707]                             [0  .707]
          [0  0]                [0  .707]

singular values of M: 1, 0.707  →  θ₁ = 0°, θ₂ = 45°
```

`principal_angle_cosines` is exactly this, one line. It matches
`scipy.linalg.subspace_angles`, and a test pins that.

**The intuition about dimensions, with numbers.** In 3-D, two planes always
share a line, so θ₁ = 0 means nothing. In 400-D, random directions are nearly
perpendicular:

| | mean | 95th percentile |
|---|---|---|
| \|cos\| between two random lines in 400-D | 0.04 (88°) | 0.10 (84°) |
| cos θ₁ between two random 3-D subspaces of 400-D | 0.13 (83°) | 0.18 (80°) |

So in 400-D, a θ₁ well below 80° is far from random geometry. The reference the
test uses is the shuffle null, not random geometry, because noise and the
design both add structure (§9).

**Why not Weber's Spearman r between PC1 loadings.** Three problems:

1. **Sign.** u and −u are equally valid PC1s (§4), so the sign of r is
   arbitrary: flip one PC and r flips too. Principal angles use the subspaces,
   which contain both u and −u, so there is no sign to get wrong.
2. **Only PC1, and PC order is fragile.** Suppose both variables use patterns
   u and v, congruency a bit more u and switch a bit more v. Congruency's PC1
   is then u and switch's PC1 is v. The PC1s are nearly perpendicular, which
   reads as "orthogonal", yet the 2-D subspaces are identical (principal angles
   0°, 0°). When two components have similar strength, noise alone can swap
   their order. A test plants exactly this.
3. **Centring and ranks.** Spearman centres each loading vector, so it has the
   cosine-versus-correlation problem of plan §3.1, and ranking throws away the
   magnitudes.

With k = 1, cos θ₁ = |cos(u_C, u_S)|: Weber's comparison with these problems
removed.

## 8. The alignment index

Your note: "I honestly have no idea how this math works and how to implement
this." It is §2's "fraction captured", done for all windows at once and for a
k-dimensional subspace. Three steps.

1. **Project each window's congruency pattern onto the switch subspace.**
   U_Sᵀ c(t) gives the k coordinates of c(t) inside switch's subspace, and its
   squared length ||U_Sᵀ c(t)||² is how much of that window's pattern lies
   there.
2. **Add up over windows.** Σ_t ||U_Sᵀ c(t)||² = ||U_Sᵀ C||², the sum of the
   squares of every entry of the k × windows matrix U_Sᵀ C. The plan writes it
   as trace(U_Sᵀ C Cᵀ U_S); that is the same number. This is the congruency
   effect captured by the switch subspace.
3. **Divide by the most any k-dimensional subspace could capture.** That is
   C's own top k, σ₁² + … + σ_k² of C (Eckart–Young, §4).

```
A(C → S) = ||U_Sᵀ C||² / (σ₁² + … + σ_k²)        0 = perpendicular, 1 = as good as C's own
```

The implementation is these three lines (`alignment_index`).

**Example, 3 electrodes × 2 windows:** the windows' patterns are (2, 1, 0) and
(2, 0, 1).

```
C = [2  2]        total sum of squares = 4+4+1+0+0+1 = 10
    [1  0]        CᵀC = [5 4; 4 5], eigenvalues 9 and 1  →  σ₁² = 9, σ₂² = 1
    [0  1]        the best single direction is (4, 1, 1)/√18 and keeps 9 of 10

switch axis (1, 0, 0):  captures 2² + 2² = 8   →  A = 8/9 = 0.89
switch axis (0, 1, 0):  captures 1² + 0² = 1   →  A = 1/9 = 0.11
```

The first switch axis is nearly as good at capturing the congruency effect as
the best possible line; the second captures almost none of it.

**Facts to keep in mind:**

- **k = 1 with one window gives cos² θ**, §2's 0.6. So for one axis, the index
  is the squared cosine of §3.2, and the angle and the index carry the same
  information.
- **It is not symmetric.** A(C → S) uses C's effect and S's subspace;
  A(S → C) the reverse. The job reports both.
- **Angles and index answer different questions.** Principal angles treat
  every dimension alike: a shared direction carrying 2 % of the congruency
  effect gives as small a θ₁ as one carrying 80 %. The index weights each
  direction by how much effect it carries. "One shared direction (small θ₁)
  that carries 15 % of the congruency effect (low index)" says more than either
  alone.
- **Relation to Weber.** Weber projected the history betas onto the context
  PCs and took the Euclidean distance between the trajectories. That is the
  numerator here, without the normalization, so it grows with effect size. The
  index divides by the best possible, which makes it scale-free.

## 9. Noise: halves, ceiling, null, bootstrap

Your plan in the notes ("repeat using congruency on half 1 vs congruency on
half 2, and same for switch, and do the permutation null by shuffling the
congruency labels while keeping the switch and prop ones constant") is what is
built. The details:

**Halves.** Each participant's trials are split into halves A and B (stratified
on all four design factors, one split per participant shared by all its
electrodes, 200 times). Every comparison crosses the halves: U_C from A against
U_S from B, and U_C from B against U_S from A, averaged. No two numbers that
are compared come from the same trials. Otherwise, noise on a trial that moves
both the congruency and the switch estimate would look like shared coding.

**Ceiling.** The same quantity between the halves of one variable: U_C from A
against U_C from B, and likewise for S. If congruency and switch used the
identical code, the cross value could not beat these, because only noise
separates them. The reported **normalized** value is

```
normalized = cross / √(ceiling_C × ceiling_S)       0 = separate, 1 = as shared as noise allows
```

**Why the geometric mean, with k = 1.** Write each half's axis as the true one
plus noise, c_A = c + n_A. Noise makes vectors point in random extra
directions, which shrinks cosines by a predictable factor:
cos(c_A, s_B) ≈ cos(c, s) × √rel_C × √rel_S, and cos(c_A, c_B) ≈ rel_C, where
rel is the share of the vector that is signal. Dividing by √(ceiling_C ×
ceiling_S) cancels the shrinkage. Simulation: true cos(c, s) = 0.6, both
reliabilities 0.5. The raw cross cosine comes out at 0.30, both ceilings at
0.50, and the normalized value at 0.30 / 0.50 = 0.60, the true value back. For
k > 1 it is a heuristic: on a planted shared 2-D subspace it averages 0.92–1.06
across noise levels, but single draws scatter (about ±0.05 when the raw value
is 0.5, more when it is near 0).

**The null.** One iteration for congruency:

1. For each participant, shuffle the congruency labels **only among trials
   with the same switch type, incongruent-proportion block and
   switch-proportion block**. This is what "keeping the switch and prop ones
   constant" has to mean: the design (cell counts, congruency's correlation
   with the incongruent proportion) stays exactly as it was, and only the link
   between congruency and HG is broken.
2. Use **one permutation per participant for all of its electrodes**. A
   participant's electrodes share trials, and their trial-to-trial noise is
   correlated (often an "everything up on this trial" direction). A separate
   permutation per electrode would make that noise independent and the null too
   easy to beat.
3. Rescore everything on fresh halves, rebuild C (S stays real), and record the
   cross values, congruency's ceilings and its per-dimension held-out variance.

Repeat 1000 times, then do the same shuffling switch type. Each shuffle is
scored once and feeds both the §3.2 cosine null and this one.

**What the null answers.** "Is anything shared beyond what noise and the
design produce?" It is the null of no congruency code (or no switch code), not
of orthogonal codes. A cross value above both nulls is marked **shared**.

**Why the null is not centred on 0.** Three reasons, all of which it is right
to include:

- Principal-angle cosines and alignment indices are never negative.
- Noise has structure: the top noise direction leans toward the noisiest
  electrodes, and so may the real effects.
- Keeping the cell counts keeps the design's chance imbalances, for example
  slightly more switch trials among incongruent than congruent trials in a
  block. That leaks a little of the switch effect into the shuffled congruency
  effect, in the real data and in the null alike. In the synthetic dry run
  this put the null's mean cross cosine at +0.2.

This is also why, for the subspace numbers, a ceiling counts as **reliable**
when it beats its own variable's shuffle null, not when its CI excludes 0. A
cosine of principal angles is above 0 even for pure noise, so a CI that
excludes 0 says nothing.

**Not one code.** Participants are resampled with replacement (a participant
drawn twice contributes its electrodes twice), and everything is recomputed
2000 times. **Not one code** means the 95 % CI of ceiling − cross excludes 0:
the codes are less alike than noise alone would make them.

**"Partly shared"** = shared (above both nulls) and not one code (below the
ceiling), at the primary k. That mirrors A4's "above chance, below the
ceiling".

## 10. The trajectories

The subspace test is the statistic; the trajectories are the picture.

**Step 1, axes from half A.** c and s are each electrode's congruency and
switch effect averaged over 0–1.5 s: one 400-vector each.

**Step 2, make them perpendicular (Gram–Schmidt).** A plot's x and y axes are
perpendicular, and c and s are not. Projecting onto both directly would count
the shared part twice.

```
x = c / |c|                             congruency, unit length
s_perp = s − (s · x) x                  remove the part of s along x
y = s_perp / |s_perp|                   switch beyond what it shares with congruency
```

The plan's toy example: c = (1, 0), s = (0.6, 0.8) gives x = (1, 0) and
y = (0, 1). Put the four cells at ±½c ± ½s (each cell is the grand mean plus
or minus half of each effect) and read off their coordinates:

| Cell | position | x | y |
|---|---|---|---|
| incongruent-switch | ½c + ½s = (0.8, 0.4) | 0.8 | 0.4 |
| incongruent-repeat | ½c − ½s = (0.2, −0.4) | 0.2 | −0.4 |
| congruent-switch | −½c + ½s = (−0.2, 0.4) | −0.2 | 0.4 |
| congruent-repeat | −½c − ½s = (−0.8, −0.4) | −0.8 | −0.4 |

Congruency separates the cells along x by |c| = 1. Switch separates them along
y by 0.8 = |s_perp|, and also along x by 0.6 = s · x: the switch cells sit 0.3
to the right of their repeat partners. **That x offset is the shared part,
drawn.** The order matters: with c first, x is pure congruency, and y is
"switch beyond congruency". `TRAJ_FIRST=switch` swaps them. Say which in the
caption.

**Step 3, cells from half B.** For each electrode and window, average HG in the
four congruency × switch cells. Each cell is the equal-weight mean of its four
proportion blocks, so a block offset cannot move one cell more than another.
Subtract the mean of the four cells at each window: the evoked response common
to all four would otherwise move every trajectory together. The uncentred
version is also written, for the supplement.

**Step 4, project.** A cell's coordinates in window t are p(t) · x and
p(t) · y, summed over all 400 electrodes. Then swap the halves, and average
over both directions and the 200 splits.

**Why the cells must come from the other half.** c from half A contains A's
noise: on a trial where an electrode happened to be high and the trial was
incongruent, it pushed c_A up. A's incongruent cells contain the same noisy
trial. Projecting A's cells onto A's axis therefore separates incongruent from
congruent even with no effect at all. B's noise is independent of A's axis, so
it adds scatter but no separation.

**How much of the activity the plane holds.** The plane is only 2 of 400
directions. The job reports the share of the centred cells' sum of squares
inside it, in two versions:

- **Raw** (`plane_fraction`): Σ (x² + y²) / Σ |p|². Noise fills every one of
  the 400 directions, and the plane gets only 2 of them, so this is a lower
  bound. It mostly says how big the signal is relative to the noise. A random
  plane would get about 2/400.
- **Noise-corrected** (`plane_fraction_signal`): multiply quantities from
  different halves instead of squaring one. For a number a measured twice with
  independent noise, (a + n_A)(a + n_B) averages a², because the noise terms
  average out of the product, whereas (a + n_B)² averages a² + noise. So
  Σ p_A · p_B estimates the signal's sum of squares, and the product of the two
  held-out trajectories (A's axes with B's cells, times B's axes with A's cells)
  estimates the signal inside the plane. Their ratio is the share of the
  condition-dependent signal that the plane holds. The plane comes from noisy
  axes, which point partly into noise directions, so the ratio underestimates
  the share in the true plane, by more the noisier the axes. It can leave
  [0, 1] where there is little signal.

**Early vs late axes.** The job also draws the trajectories with axes from
0.1–0.5 s and from 0.6–1.4 s (`TRAJ_AXES`). If the codes converge late, the
late axes' cos(c, s) (printed under each figure) is higher, and the switch
cells move further along x.

## 11. Reproduce the numbers

Every example above, in numpy (run from the repo root):

```python
import numpy as np
from src.analysis.stats import coding_subspaces as cs

# §2: cosine, projection, fraction captured
c, s = np.array([2., 1, 0]), np.array([1., 1, 1])
cos = c @ s / np.linalg.norm(c) / np.linalg.norm(s)            # 0.775
cs.alignment_index(c[:, None], (s / np.linalg.norm(s))[:, None])  # 0.6 = cos**2

# §3-4: rank 1 and rank 2
p, q = np.array([1., 2, 0, -1]), np.array([0., 1, 2, 1])
np.linalg.svd(np.outer(p, [.5, 1, .5]), compute_uv=False)     # [3, 0, 0]
np.linalg.svd(np.column_stack([p, p, q]), compute_uv=False)    # [3.51, 2.38, 0]

# §5: PCA drops the sustained pattern
C = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0.]])
cs.coding_subspace(C, 2)                                       # spans e1, e2
cs.coding_subspace(C - C.mean(1, keepdims=True), 1)            # (0, ±1, 0): only v

# §7: the two planes
floor = np.array([[1, 0], [0, 1], [0, 0.]])
tilted = np.array([[1, 0], [0, 2 ** -.5], [0, 2 ** -.5]])
np.degrees(np.arccos(cs.principal_angle_cosines(floor, tilted)))  # [0, 45]

# §8: the alignment index example
C = np.array([[2., 2], [1, 0], [0, 1]])
cs.alignment_index(C, np.array([[1.], [0], [0]]))              # 8/9
cs.alignment_index(C, np.array([[0.], [1], [0]]))              # 1/9
```

On the real data, after the job has run, everything is in its output
directory: `per_split_windows.csv` holds C and S for every split, so
`coding_subspaces.subspace_stats(pd.read_csv(...))` recomputes the point
estimates without reloading epochs.
