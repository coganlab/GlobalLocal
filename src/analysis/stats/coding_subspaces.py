"""The subspace test (docs/analysis_plans.md, subspace plan §3.4): do the
congruency and switch-type codes share any direction over the trial?

The math, worked through with numbers: docs/subspace_math_walkthrough.md.

Input is the per-split, per-window table of `coding_axes` (congruency in
`mxA`/`mxB`, switch type in `myA`/`myB`). For each split, stacking a half's
per-window effects gives four electrode x window effect matrices, C_A, C_B
(congruency, halves A and B) and S_A, S_B (switch type). Column t of C is the
congruency coding axis c of window t.

1. **Subspaces.** U_C = the top k left singular vectors of C, from an
   uncentred SVD (`coding_subspace`): an orthonormal basis (electrodes x k) for
   the k electrode patterns that carry most of the congruency effect across the
   trial. Uncentred, because centring each electrode across windows (PCA)
   would remove its time-averaged effect, which is the main coding direction.

2. **Principal angles** (`principal_angle_cosines`): the cosines of the angles
   between two k-dimensional subspaces are the singular values of U_1ᵀ U_2.
   cos θ₁ (the largest) says whether ANY direction is shared; cos θ_k whether
   the subspaces are the same.

3. **Alignment index** (`alignment_index`, Elsayed et al. 2016): the share of
   M's effect that lands in a subspace U, out of the most any k-dimensional
   subspace could capture: ||Uᵀ M||² / (σ₁² + ... + σ_k²), with σ the singular
   values of M. ||Uᵀ M||² = trace(Uᵀ M Mᵀ U). 0 = orthogonal, 1 = M's own top k.

Every comparison is across halves, as in §3.2, so no two numbers that are
compared come from the same trials:

    cross   angles  1/2 [pa(U_C,A, U_S,B) + pa(U_C,B, U_S,A)]
            C -> S  1/2 [AI(C_A, U_S,B) + AI(C_B, U_S,A)]      (and S -> C)
    ceiling angles  pa(U_C,A, U_C,B) and pa(U_S,A, U_S,B)
                    AI(C_A, U_C,B) and AI(S_A, U_S,B), symmetrised the same way

each averaged over splits; then `ceiling` = sqrt(congruency ceiling x switch
ceiling) and `normalized` = cross / ceiling: 0 separate, 1 as shared as noise
allows. With k = 1 the angle cosine is |cos(c, s)| of the time-averaged
pattern and the alignment index is its square, so both reduce to §3.2's
numbers; the geometric-mean ceiling is exact there and a heuristic for k > 1.
On a planted shared 2-D subspace it averages 0.92-1.06 across noise levels
(spread growing as the raw values fall: about ±0.05 at a raw θ2 cosine of
0.5, ±0.15 at 0.1), and near 0 for orthogonal subspaces; the test file pins
both.

**Choosing k** (`dimension_test`): dimension j of a variable is reproducible
when the j-th singular vector of half A captures more of half B's effects
(||u_jᵀ M_B||², symmetrised) than it does with that variable's labels
shuffled. k for a variable is its number of leading reproducible dimensions;
the primary k is the smaller of the two, so both subspaces are signal at it.

**The tests**, as in §3.2:
  * anything shared? The raw cross values against the label-shuffle nulls,
    one factor shuffled at a time (`coding_axes.null_draws`); "shared" needs
    both.
  * not one code? A participant bootstrap of ceiling - cross: its CI excludes 0.
  * reliable? Each variable's own ceiling above its shuffle null. Cosines of
    principal angles and alignment indices are never negative, so "CI
    excludes 0" (§3.2's rule) would pass even for noise; the null is the
    right reference here.

`compare_subspaces` computes all of it from the in-memory tables. One
deviation from Elsayed et al.: they compared the alignment index against
random subspaces drawn to match the data covariance; this uses the
label-shuffle null so it lines up with A4 and §3.2.
"""

import warnings

import numpy as np
import pandas as pd

from . import coding_axes as ca

ALIGNMENT_DIRECTIONS = ('congruency_to_switch', 'switch_to_congruency')
VARIABLES = {'congruency': ('mxA', 'mxB'), 'switchType': ('myA', 'myB')}


# ---------------------------------------------------------------------------
# the three primitives; each works on one matrix or on a stack (..., e, w)
# ---------------------------------------------------------------------------
def coding_subspace(effects, k):
    """An orthonormal basis for the subspace of an electrode x window effect
    matrix: its top `k` left singular vectors, from an uncentred SVD.

    effects = U Σ Vᵀ. Column j of U is an electrode pattern (unit length), σ_j
    how much of the effect it carries, and row j of Vᵀ how strongly each window
    uses it. U[:, :k] spans the k patterns that best reconstruct the matrix:
    no k-dimensional subspace captures more of its sum of squares (Eckart-Young).
    Nothing is subtracted first, so a pattern present in every window counts.

    `effects` is (electrodes, windows) or a stack (..., electrodes, windows);
    returns (..., electrodes, k). A singular vector's sign is arbitrary, which
    is why everything below depends only on the subspace, not on the signs.
    """
    effects = np.asarray(effects, float)
    if k > min(effects.shape[-2:]):
        raise ValueError(f"k={k} exceeds the matrix rank bound {min(effects.shape[-2:])} "
                         f"(electrodes x windows = {effects.shape[-2:]})")
    u, _, _ = np.linalg.svd(effects, full_matrices=False)
    return u[..., :k]


def principal_angle_cosines(basis_1, basis_2):
    """cos θ₁ >= ... >= cos θ_k between two subspaces with orthonormal bases
    (electrodes x k): the singular values of basis_1ᵀ basis_2.

    θ₁ is the smallest angle between any unit vector in one subspace and any in
    the other; θ₂ the smallest using vectors orthogonal to that first pair; and
    so on. Equal to `np.cos(scipy.linalg.subspace_angles(basis_1, basis_2))`
    sorted descending. Takes stacks (..., electrodes, k); returns (..., k).
    """
    m = np.swapaxes(basis_1, -1, -2) @ basis_2
    return np.clip(np.linalg.svd(m, compute_uv=False), 0.0, 1.0)


def alignment_index(effects, basis, singular_values=None):
    """The share of an effect matrix that lands in a k-dimensional subspace, out
    of the most any k-dimensional subspace could capture (Elsayed et al. 2016):

        trace(Uᵀ M Mᵀ U) / (σ₁² + ... + σ_k²)

    The numerator projects every column of M (one window's pattern) onto the
    subspace and adds up the squared lengths, ||Uᵀ M||²; the denominator is the
    same for M's own top k singular vectors. 1 = the subspace is as good as M's
    own; 0 = orthogonal to every column of M.

    `effects` (..., electrodes, windows), `basis` (..., electrodes, k);
    `singular_values` of `effects`, descending, if already computed. Returns
    (...).
    """
    k = basis.shape[-1]
    captured = ((np.swapaxes(basis, -1, -2) @ effects) ** 2).sum((-2, -1))
    if singular_values is None:
        singular_values = np.linalg.svd(effects, compute_uv=False)
    return captured / (singular_values[..., :k] ** 2).sum(-1)


def heldout_variance(basis, effects):
    """||u_jᵀ M||² for each column u_j of `basis`: how much of M (e.g. the other
    half's effects) each dimension captures, separately. (..., electrodes, k)
    and (..., electrodes, windows) -> (..., k)."""
    return ((np.swapaxes(basis, -1, -2) @ effects) ** 2).sum(-1)


# ---------------------------------------------------------------------------
# per-split statistics
# ---------------------------------------------------------------------------
def _rows(k_max):
    """The (measure, k, component) of every reported number, in table order:
    each principal angle of each k, then both alignment directions of each k."""
    rows = [('angle', k, str(i)) for k in range(1, k_max + 1) for i in range(1, k + 1)]
    rows += [('alignment', k, d) for k in range(1, k_max + 1) for d in ALIGNMENT_DIRECTIONS]
    return rows


def _matrices(arrays):
    """{col: (splits, electrodes, windows)} from `axis_arrays`' (windows, splits,
    electrodes)."""
    return {c: np.transpose(a, (1, 2, 0)) for c, a in arrays.items()}


def _split_stats(arrays, k_max):
    """Split-averaged cross, ceilings and dimension statistics.

    Returns ``(cross, ceil_c, ceil_s, dims)``: the first three (rows,) in
    `_rows(k_max)` order, `dims` a dict with each variable's per-dimension
    held-out variance (k_max,) and the total held-out sum of squares, symmetric
    over the halves."""
    m = _matrices(arrays)
    svd = {c: np.linalg.svd(v, full_matrices=False) for c, v in m.items()}
    u = {c: s[0] for c, s in svd.items()}                    # (splits, e, windows)
    sv = {c: s[1] for c, s in svd.items()}                   # (splits, windows)

    def pa(a, b, k):
        return principal_angle_cosines(u[a][..., :k], u[b][..., :k])

    def ai(effects, basis, k):
        return alignment_index(m[effects], u[basis][..., :k], sv[effects])

    cross, cc, cs = [], [], []
    for k in range(1, k_max + 1):
        cross.append(0.5 * (pa('mxA', 'myB', k) + pa('mxB', 'myA', k)))
        cc.append(pa('mxA', 'mxB', k))
        cs.append(pa('myA', 'myB', k))
    ai_cc = [0.5 * (ai('mxA', 'mxB', k) + ai('mxB', 'mxA', k)) for k in range(1, k_max + 1)]
    ai_ss = [0.5 * (ai('myA', 'myB', k) + ai('myB', 'myA', k)) for k in range(1, k_max + 1)]
    for k in range(1, k_max + 1):
        cross += [0.5 * (ai('mxA', 'myB', k) + ai('mxB', 'myA', k))[:, None],    # C -> S
                  0.5 * (ai('myA', 'mxB', k) + ai('myB', 'mxA', k))[:, None]]    # S -> C
        cc += [ai_cc[k - 1][:, None]] * 2
        cs += [ai_ss[k - 1][:, None]] * 2

    dims = {}
    for var, (a, b) in VARIABLES.items():
        dims[var] = 0.5 * (heldout_variance(u[a][..., :k_max], m[b])
                           + heldout_variance(u[b][..., :k_max], m[a]))
        dims[f'{var}_total'] = 0.5 * ((m[a] ** 2).sum((-2, -1)) + (m[b] ** 2).sum((-2, -1)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = [np.nanmean(np.concatenate(v, axis=1), axis=0) for v in (cross, cc, cs)]
        dims = {key: np.nanmean(v, axis=0) for key, v in dims.items()}
    return (*out, dims)


def _checked_arrays(per_split, k_max):
    subjects, arrays = ca.axis_arrays(per_split)
    n_windows = next(iter(arrays.values())).shape[0]
    if k_max > n_windows:
        raise ValueError(f"k_max={k_max} needs at least as many windows; the table has "
                         f"{n_windows}")
    return subjects, arrays


def subspace_stats(per_split, k_max=3):
    """The point estimates: one row per (measure, k, component) with the cross
    value, both ceilings, `ceiling` and `normalized`, plus the angles in
    degrees; and the per-dimension table (`dimension_test` without the null).

    Returns ``(table, dims)``. Electrodes are those `axis_arrays` keeps."""
    subjects, arrays = _checked_arrays(per_split, k_max)
    cross, cc, cs, dims = _split_stats(arrays, k_max)
    table = pd.DataFrame(_rows(k_max), columns=['measure', 'k', 'component'])
    table['n_electrodes'] = len(subjects)
    table['n_subjects'] = len(set(subjects))
    table['cross'], table['congruency_ceiling'], table['switch_ceiling'] = cross, cc, cs
    table['ceiling'] = ca._ceiling(cc, cs)
    table['normalized'] = cross / table['ceiling']
    is_angle = table['measure'] == 'angle'
    for col in ('cross', 'congruency_ceiling', 'switch_ceiling'):
        table[f'{col}_deg'] = np.where(is_angle, np.degrees(np.arccos(np.clip(table[col], 0, 1))),
                                       np.nan)
    return table, _dims_table(dims, k_max)


def _dims_table(dims, k_max):
    rows = []
    for var in VARIABLES:
        for j in range(k_max):
            rows.append(dict(variable=var, dimension=j + 1, heldout=dims[var][j],
                             heldout_fraction=dims[var][j] / dims[f'{var}_total']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------
def subspace_null_stats(per_split, factor, k_max=3):
    """What one label-shuffle draw records (`per_split` rescored with `factor`
    shuffled, from `coding_axes.null_draws`): the cross values of every row
    (is anything shared?), the shuffled variable's own ceilings (is its code
    reliable?) and its per-dimension held-out variance (how many dimensions
    are reproducible?). A dict of arrays; stack draws with `stack_null`."""
    if factor not in VARIABLES:
        raise ValueError(f"factor must be one of {tuple(VARIABLES)}; got {factor!r}")
    _, arrays = _checked_arrays(per_split, k_max)
    cross, cc, cs, dims = _split_stats(arrays, k_max)
    return dict(cross=cross, ceiling=cc if factor == 'congruency' else cs, dims=dims[factor])


def stack_null(records):
    """A list of `subspace_null_stats` dicts -> one dict of (n_perm, ...) arrays."""
    return {key: np.array([r[key] for r in records]) for key in records[0]}


def subspace_null(df, hg_windows, factor, k_max=3, n_perm=1000, n_splits=20, seed=0,
                  remove_mean=False):
    """`subspace_null_stats` of `n_perm` draws of `coding_axes.null_draws`, stacked.
    With the same arguments the draws are those of `coding_axes.cross_cosine_null`."""
    return stack_null([subspace_null_stats(per, factor, k_max) for per in ca.null_draws(
        df, hg_windows, factor, n_perm=n_perm, n_splits=n_splits, seed=seed,
        remove_mean=remove_mean)])


# ---------------------------------------------------------------------------
# choosing k
# ---------------------------------------------------------------------------
def dimension_test(dims, null, alpha=0.05):
    """Which dimensions of each variable are reproducible, and its k.

    `dims` is the per-dimension table of `subspace_stats`; `null` maps a factor
    to its stacked null (`stack_null`). Dimension j is `reproducible` when its
    held-out variance is above the (1 - `alpha`) percentile of the same number
    with the variable's labels shuffled; `kept` marks the leading run of
    reproducible dimensions, and k for the variable is its length. Testing in
    order and stopping at the first failure keeps the family-wise error at
    `alpha` (a fixed-sequence test).

    Returns ``(dims, k)`` with `dims` gaining `null_hi`, `p`, `reproducible`
    and `kept`, and `k` = {variable: k}.
    """
    dims = dims.copy()
    for col in ('null_hi', 'p'):
        dims[col] = np.nan
    dims['reproducible'] = False
    k = {}
    for var in VARIABLES:
        rows = dims['variable'] == var
        if var not in null:
            k[var] = None
            continue
        draws = np.asarray(null[var]['dims'], float)            # (n_perm, k_max)
        observed = dims.loc[rows, 'heldout'].to_numpy()
        dims.loc[rows, 'null_hi'] = np.percentile(draws, 100 * (1 - alpha), axis=0)
        dims.loc[rows, 'p'] = ((draws >= observed).sum(0) + 1) / (len(draws) + 1)
        dims.loc[rows, 'reproducible'] = observed > dims.loc[rows, 'null_hi'].to_numpy()
        passed = dims.loc[rows, 'reproducible'].to_numpy(bool)
        k[var] = int(np.argmin(passed)) if not passed.all() else int(passed.size)
    dims['kept'] = [(k.get(v) is not None and d <= k[v])
                    for v, d in zip(dims['variable'], dims['dimension'])]
    return dims, k


# ---------------------------------------------------------------------------
# bootstrap and the full comparison
# ---------------------------------------------------------------------------
def _bootstrap(arrays, subjects, k_max, n_boot, rng):
    """Participant-bootstrap draws of (cross, ceil_c, ceil_s), each (n_boot, rows):
    participants resampled with replacement, keeping their real labels; a
    participant drawn twice contributes its electrodes twice."""
    codes = pd.factorize(subjects)[0]
    members = [np.flatnonzero(codes == p) for p in range(codes.max() + 1)]
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(len(members), size=len(members))
        idx = np.concatenate([members[p] for p in pick])
        draws.append(_split_stats({c: a[..., idx] for c, a in arrays.items()}, k_max)[:3])
    return tuple(np.array(d) for d in zip(*draws))


def compare_subspaces(per_split, null=None, k_max=3, n_boot=2000, seed=0, alpha=0.05):
    """Everything §3.4 reports.

    `per_split` is the real-label table (module docstring); `null` maps a factor
    to its stacked null (`subspace_null` or `stack_null`); without it the
    shuffle tests and the choice of k are skipped. Returns
    ``(table, dims, summary)``.

    `table` has the `subspace_stats` columns plus, per row:

        cross_lo/_hi, normalized_lo/_hi, congruency_ceiling_lo/_hi,
        switch_ceiling_lo/_hi       bootstrap CIs (central 1 - `alpha`)
        gap_lo/_hi                  CI of ceiling - cross
        null_hi_<f>                 the (1 - `alpha`) percentile of the cross
                                    value with factor f shuffled: the threshold
        p_<f>                       one-sided permutation p of the raw cross value
        shared_<f>, shared          above f's null; above both
        reliable_congruency/_switch each ceiling above its own variable's null
        reliable                    both (with no null: both ceilings' CIs > 0)
        not_one_code                reliable and gap_lo > 0
        primary                     k is the primary k

    `dims` is `dimension_test`'s table (held-out variance only, without a
    null). `summary` holds k per variable, the primary k, the primary rows'
    numbers, and the settings. The rows are not a time series, so nothing is
    cluster-corrected; the primary k's θ₁ and alignment indices are the
    pre-specified tests, the rest descriptive.
    """
    subjects, arrays = _checked_arrays(per_split, k_max)
    table, dims = subspace_stats(per_split, k_max)

    rng = np.random.default_rng(seed)
    b_cross, b_cc, b_cs = _bootstrap(arrays, subjects, k_max, n_boot, rng)
    b_ceiling = ca._ceiling(b_cc, b_cs)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for name, draws in (('cross', b_cross), ('congruency_ceiling', b_cc),
                            ('switch_ceiling', b_cs), ('gap', b_ceiling - b_cross),
                            ('normalized', b_cross / b_ceiling)):
            table[f'{name}_lo'], table[f'{name}_hi'] = np.nanpercentile(draws, q, axis=0)

    for factor, draws in (null or {}).items():
        cross_null = np.asarray(draws['cross'], float)
        table[f'null_hi_{factor}'] = np.percentile(cross_null, 100 * (1 - alpha), axis=0)
        table[f'p_{factor}'] = (((cross_null >= table['cross'].to_numpy()).sum(0) + 1)
                                / (len(cross_null) + 1))
        table[f'shared_{factor}'] = table['cross'] > table[f'null_hi_{factor}']
        own = 'congruency_ceiling' if factor == 'congruency' else 'switch_ceiling'
        ceiling_null = np.percentile(np.asarray(draws['ceiling'], float),
                                     100 * (1 - alpha), axis=0)
        table[f'reliable_{"congruency" if factor == "congruency" else "switch"}'] = \
            table[own] > ceiling_null
    if null and set(null) == set(VARIABLES):
        table['shared'] = table[[f'shared_{f}' for f in VARIABLES]].all(axis=1)
        table['reliable'] = table['reliable_congruency'] & table['reliable_switch']
    else:
        table['reliable'] = (table['congruency_ceiling_lo'] > 0) & (table['switch_ceiling_lo'] > 0)
    table['not_one_code'] = table['reliable'] & (table['gap_lo'] > 0)

    k = {}
    if null:
        dims, k = dimension_test(dims, null, alpha=alpha)
    known = [v for v in k.values() if v is not None]
    primary_k = min(known) if len(known) == len(VARIABLES) else None
    table['primary'] = table['k'] == primary_k

    def row(measure, component):
        sel = table[(table['measure'] == measure) & table['primary']
                    & (table['component'] == component)]
        if sel.empty:
            return None
        keep = ['cross', 'cross_lo', 'cross_hi', 'ceiling', 'normalized', 'normalized_lo',
                'normalized_hi', 'reliable', 'not_one_code', 'shared', 'cross_deg',
                *[c for c in table if c.startswith('p_')]]
        r = sel.iloc[0]
        return {c: (bool(r[c]) if isinstance(r[c], (bool, np.bool_)) else float(r[c]))
                for c in keep if c in table}

    summary = dict(
        n_electrodes=int(table['n_electrodes'].iloc[0]),
        n_subjects=int(table['n_subjects'].iloc[0]),
        n_windows=int(next(iter(arrays.values())).shape[0]),
        k=k or None, primary_k=primary_k,
        primary_k_capped=bool(primary_k == k_max) if primary_k is not None else None,
        theta_1=row('angle', '1'),
        alignment_congruency_to_switch=row('alignment', 'congruency_to_switch'),
        alignment_switch_to_congruency=row('alignment', 'switch_to_congruency'),
        k_max=k_max, n_boot=n_boot, seed=seed, alpha=alpha,
        n_perm={f: int(len(d['cross'])) for f, d in (null or {}).items()})
    return table, dims, summary
