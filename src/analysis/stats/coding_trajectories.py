"""Trajectories of the four condition means in the coding plane
(docs/analysis_plans.md, subspace plan §4), and adaptation as block-split
projections onto the same axes (§6). The worked math is in
docs/subspace_math_walkthrough.md §6.

For every split of each participant's trials into halves A and B:

1. **Axes on half A.** The congruency axis c and the switch axis s, each
   electrode's main effect on HG averaged over an axis window (0-1.5 s for the
   main figure), scored exactly as the coding axes are
   (`main_effect_mean_diffs`, proportion blocks weighted equally).
2. **An orthonormal plane** (Gram-Schmidt, congruency first by default):

       x = c / |c|                          pure congruency
       y = s_perp / |s_perp|,  s_perp = s - (s . x) x
                                            switch beyond what it shares with c

   `first=None` skips this: x = c_hat . p and y = s_hat . p, on axes that are
   not perpendicular (cos(c, s) is in the summary).

3. **Cells on half B.** Each electrode's mean HG in the four congruency x
   switch-type cells at every time bin, each the equal-weight mean of its four
   proportion blocks (incProp x switchProp), so no block difference leaks in.
   Centred: the mean of the four cells is subtracted at each time bin, which
   removes the evoked response they all share.
4. **Project:** a cell's coordinates at time t are p(t) . x and p(t) . y,
   summed over the pooled electrodes.

Then the same with the halves swapped, and everything averaged over splits.
Axes and cells never come from the same trials, so noise in the axes cannot
push the cells apart.

Every quantity is a ratio of sums over electrodes, so it is computed from
per-participant sums, and a participant bootstrap only re-weights them.

**How much of the condition-dependent activity the plane holds** (§4 step 5),
per time bin and overall:

    plane_fraction          sum (x^2 + y^2) / sum |p|^2  over the four centred cells
                            on the held-out half. Noise fills every one of the
                            ~400 electrode directions, so this is a lower bound;
                            a random plane would get about 2 / n_electrodes.
    plane_fraction_signal   the same with noise removed by crossing the halves:
                            sum [x_AB x_BA + y_AB y_BA] / sum p_A . p_B, where
                            x_AB is x from A's axes and B's cells. Independent
                            noise averages out of both sums, so this estimates
                            the share of the condition-dependent SIGNAL in the
                            plane (as estimated from noisy axes). Noisy; can
                            leave [0, 1] where there is little signal.

**Adaptation as block-split separations** (plan §6, `block_separations`): on
the same axes and halves, within each of the four proportion blocks
(incProp x switchProp), the held-out congruency separation projected on the
unit congruency axis,

    c_hat . [mean(IS, IR) - mean(CS, CR)]     (switch types weighted equally)

and the switch separation on the unit switch axis, s_hat . [mean(IS, CS) -
mean(IR, CR)]. Each is projected on its own axis, not on the Gram-Schmidt y,
so the switch separation does not depend on which axis came first. The double
dissociation predicts that the congruency separation changes with incProp but
not switchProp, and the switch separation with switchProp but not incProp.
With `unique=True` each separation is instead projected on its own axis's part
orthogonal to the other axis (c_perp = c - (c.s / s.s) s, s_perp likewise;
symmetric, so the order does not matter): a robustness check of whether the
adaptation sits in what each code does not share. A shared direction that
adapts counts on the raw axes (the default) but not on the unique ones.

**Adaptation as trajectories** (plan §6, `proportion_trajectories`): each
variable crossed with each proportion gives four conditions, the variable's two
levels in the 25 % and 75 % blocks of that proportion (e.g. i25, c25, i75,
c75), every other factor weighted equally. Their held-out means are drawn on
the congruency and switch axes themselves, in oblique coordinates: a
condition's activity in the plane written as a c_hat + b s_hat. The axes are
not perpendicular, so this is not the projection; it makes a change along
c_hat move only a, and one along s_hat only b, which is what the prediction is
about.
"""

import warnings

import numpy as np
import pandas as pd

from . import stability_flexibility_segregation as sfs

# The four cells in (`_scond`, `_fcond`) levels (1 = incongruent / switch), and
# the four proportion blocks, (incProp, switchProp), in (`_smod`, `_fmod`)
# levels (1 = low proportion).
CELLS = {'incongruent-switch': (1.0, 1.0), 'incongruent-repeat': (1.0, 0.0),
         'congruent-switch': (0.0, 1.0), 'congruent-repeat': (0.0, 0.0)}
BLOCKS = {('low', 'low'): (1.0, 1.0), ('low', 'high'): (1.0, 0.0),
          ('high', 'low'): (0.0, 1.0), ('high', 'high'): (0.0, 0.0)}
_SYSTEM = {'cells': (('_scond', '_fcond', '_smod', '_fmod'),
                     tuple((*cell, *block) for cell in CELLS.values() for block in BLOCKS.values()))}
FIRST_AXES = ('congruency', 'switch')
# Each variable's separation as weights over CELLS (the other variable's
# levels weighted equally), and the two proportion effects (low - high) as
# weights over BLOCKS (the other proportion weighted equally).
_SEPARATION = {'congruency': np.array([0.5, 0.5, -0.5, -0.5]),
               'switchType': np.array([0.5, -0.5, 0.5, -0.5])}
_PROPORTION = {'incProp': np.array([0.5, 0.5, -0.5, -0.5]),
               'switchProp': np.array([0.5, -0.5, 0.5, -0.5])}
# The proportion each variable's own adaptation runs over (LWPC, LWPS).
OWN_PROPORTION = {'congruency': 'incProp', 'switchType': 'switchProp'}
_TRAJECTORY_SUMS = ('cc', 'ss', 'cs', 'cQ', 'sQ', 'PP', 'PAPB')
_PROJECTION_SUMS = ('cc', 'ss', 'cs', 'cQ', 'sQ')
# The block-balanced cells as weights over (cell, block): cell n, blocks equal.
_CELL_WEIGHTS = np.repeat(np.eye(len(CELLS))[:, :, None], len(BLOCKS), -1) / len(BLOCKS)
# Each variable's index into the CELLS levels and the names of its levels 1 and
# 0, and each proportion's index into the BLOCKS levels (level 1 = low = 25 %).
_LEVELS = {'congruency': (0, ('i', 'c')), 'switchType': (1, ('s', 'r'))}
_PROPS = {'incProp': 0, 'switchProp': 1}


def axis_window_means(hg, times, windows):
    """Each row's mean HG over each axis window: (rows, len(windows)).

    `hg` holds equal-length time courses (the `hg` column of a long table built
    with ``assemble_long_df(..., time_courses=True)``), `times` their sample
    times (s), `windows` a list of (tmin, tmax) in s, both ends included."""
    x = np.vstack(np.asarray(hg))
    times = np.asarray(times, float)
    cols = []
    for tmin, tmax in windows:
        sel = (times >= tmin - 1e-9) & (times <= tmax + 1e-9)
        if not sel.any():
            raise ValueError(f"axis window [{tmin}, {tmax}] s has no samples in "
                             f"[{times[0]:.3f}, {times[-1]:.3f}] s")
        cols.append(x[:, sel].mean(axis=1))
    return np.column_stack(cols)


def cell_block_means(df, hg_windows, halves, remove_mean=False):
    """Per participant, each electrode's mean HG in the four congruency x
    switch-type cells within each of the four proportion blocks, every time
    bin, split and half. Yields ``(subject, electrodes, means)`` with `means`
    (electrodes, time bins, splits, 2 halves, 4 cells, 4 blocks), in `CELLS`
    and `BLOCKS` order; NaN where a cell has no trial in that block and half.
    The mean over the last axis is the block-balanced cell mean."""
    for subj, elecs, cm in sfs.half_cell_means(df, hg_windows, _SYSTEM, halves, remove_mean):
        means, _ = cm['cells']                          # (..., 16) cell-major
        yield subj, elecs, means.reshape(*means.shape[:-1], len(CELLS), len(BLOCKS))


def _participant_axes(axes, subj, elecs):
    """{col: (electrodes, axis windows, splits)} for one participant, from the
    `main_effect_mean_diffs` table (electrode-major, then window, then split)."""
    a = axes[axes['subject'] == subj]
    n_ax, n_sp = a['window'].nunique(), a['split'].nunique()
    if not np.array_equal(a['electrode'].to_numpy()[::n_ax * n_sp], elecs):
        raise RuntimeError(f"axis and cell electrodes are not in the same order for {subj}")
    return {c: a[c].to_numpy(float).reshape(len(elecs), n_ax, n_sp)
            for c in ('mxA', 'mxB', 'myA', 'myB')}


def trajectory_sums(df, hg_windows, axis_hg, n_splits=200, seed=0, remove_mean=False):
    """Per-participant sums everything in the trajectories is built from.

    `df` is the long table (one row per electrode x trial, with `trial` and the
    four design columns); `hg_windows` (rows of `df`, time bins) the HG the
    cells are averaged from; `axis_hg` (rows of `df`, n_axes) each trial's HG
    averaged over each axis window. Both go through `remove_mean` if set.

    Directions: d = 0 takes the axes from half A and the cells from half B,
    d = 1 the reverse. Returns ``(sums, info)``: `sums` maps a name to a
    (participants, ...) array,

        cc, ss, cs      c.c, s.s, c.s                 (n_axes, splits, 2)
        cQ, sQ          c.p, s.p of every cell in every proportion block,
                        uncentred      (n_axes, splits, 2, time, 4 cells, 4 blocks)
        PP              |p|^2 summed over the cells   (splits, 2, time)
        PAPB            p_A . p_B summed over cells   (splits, time)

    with p a centred block-balanced cell mean in PP and PAPB. A projection is
    linear, so the projection of any combination of cell means (a
    block-balanced cell, a separation, a proportion condition) is the same
    combination of cQ (sQ); `_projections` forms it.

    over each participant's electrodes defined everywhere (axes on every split
    and axis window; every cell of every block on every split, half and time
    bin). `info` has the subjects, and the electrodes kept and dropped per
    participant.
    """
    axis_hg = np.asarray(axis_hg, float)
    if axis_hg.ndim == 1:
        axis_hg = axis_hg[:, None]
    halves = sfs.shared_halves(df, n_splits=n_splits, seed=seed)
    axes = sfs.main_effect_mean_diffs(df, axis_hg, remove_mean=remove_mean, halves=halves)
    parts = {k: [] for k in _TRAJECTORY_SUMS}
    info = dict(subjects=[], n_kept=[], n_dropped=[])
    for subj, elecs, blocks in cell_block_means(df, hg_windows, halves, remove_mean):
        ax = _participant_axes(axes, subj, elecs)
        c = np.stack([ax['mxA'], ax['mxB']], -1)        # (e, axes, splits, d): d=0 -> A
        s = np.stack([ax['myA'], ax['myB']], -1)
        blocks = blocks[:, :, :, ::-1]                  # (e, t, splits, d, 4, 4): d=0 -> B
        centred = blocks.mean(-1)                       # block-balanced cells, centred
        centred = centred - centred.mean(-1, keepdims=True)
        ok = (np.isfinite(c).all((1, 2, 3)) & np.isfinite(s).all((1, 2, 3))
              & np.isfinite(blocks).all((1, 2, 3, 4, 5)))
        info['subjects'].append(subj)
        info['n_kept'].append(int(ok.sum()))
        info['n_dropped'].append(int((~ok).sum()))
        c, s, centred, blocks = c[ok], s[ok], centred[ok], blocks[ok]
        parts['cc'].append(np.einsum('eakd,eakd->akd', c, c))
        parts['ss'].append(np.einsum('eakd,eakd->akd', s, s))
        parts['cs'].append(np.einsum('eakd,eakd->akd', c, s))
        for name, axis in (('c', c), ('s', s)):
            parts[f'{name}Q'].append(np.einsum('eakd,etkdxb->akdtxb', axis, blocks))
        parts['PP'].append(np.einsum('etkdx,etkdx->kdt', centred, centred))
        parts['PAPB'].append(np.einsum('etkx,etkx->kt', centred[:, :, :, 1], centred[:, :, :, 0]))
    if not sum(info['n_kept']):
        raise ValueError("no electrode has every cell in every half; the trajectories need "
                         "at least one trial per cell and proportion block in each half")
    return {k: np.stack(v) for k, v in parts.items()}, info


def _projections(sums, weights):
    """(c.p, s.p) of the conditions `weights` (n, 4 cells, 4 blocks) defines
    as combinations of cell means, each (..., n_axes, splits, 2, time, n).
    `sums` combined over participants, possibly with leading (draw) axes."""
    return tuple(np.einsum('...xb,nxb->...n', sums[k], weights) for k in ('cQ', 'sQ'))


def _coordinates(sums, first='congruency', cells='centred'):
    """(x, y), each (n_axes, splits, 2, time, 4), of the block-balanced cells
    in the Gram-Schmidt plane, from sums combined over participants.
    `first=None`: no Gram-Schmidt, x and y the projections on the unit c and
    unit s axes."""
    if first not in FIRST_AXES + (None,):
        raise ValueError(f"first must be one of {FIRST_AXES} or None; got {first!r}")
    ap, bp = _projections(sums, _CELL_WEIGHTS)
    if cells == 'centred':                               # centring commutes with projecting
        ap, bp = ap - ap.mean(-1, keepdims=True), bp - bp.mean(-1, keepdims=True)
    aa, bb = sums['cc'], sums['ss']
    if first == 'switch':
        aa, bb, ap, bp = bb, aa, bp, ap
    ab = sums['cs']
    expand = (Ellipsis, None, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = ap / np.sqrt(aa)[expand]
        if first is None:
            return x, bp / np.sqrt(bb)[expand]
        perp = bb - ab ** 2 / aa                         # |s_perp|^2
        y = (bp - (ab / aa)[expand] * ap) / np.sqrt(perp)[expand]
    return x, y


def _fractions(x, y, sums):
    """Per axis window, (plane_fraction, plane_fraction_signal) per time bin
    (n_axes, time) and overall (n_axes,). Ratios of sums over splits."""
    num_raw = (x ** 2 + y ** 2).sum(-1).sum((1, 2))               # (axes, time)
    den_raw = sums['PP'].sum((0, 1))                              # (time,)
    num_sig = (x[:, :, 0] * x[:, :, 1] + y[:, :, 0] * y[:, :, 1]).sum(-1).sum(1)
    den_sig = sums['PAPB'].sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (num_raw / den_raw, num_sig / den_sig,
                num_raw.sum(-1) / den_raw.sum(), num_sig.sum(-1) / den_sig.sum())


def _weighted(sums, weights):
    """Sums over participants with weights (participants,) or (participants, n)
    -> leading axis n (or none)."""
    return {k: np.tensordot(weights, v, axes=([0], [0])) for k, v in sums.items()}


def _present_sums(df, hg_windows, axis_hg, axis_labels, n_splits, seed, remove_mean, keys):
    """``(sums, info, axis_labels)``: `trajectory_sums` kept to `keys` and to the
    participants with an electrode left (one with none would enter a bootstrap
    as zeros), and the axis labels checked against the axis windows (default
    '0', '1', ...)."""
    sums, info = trajectory_sums(df, hg_windows, axis_hg, n_splits=n_splits, seed=seed,
                                 remove_mean=remove_mean)
    n_ax = sums['cc'].shape[1]
    axis_labels = [str(i) for i in range(n_ax)] if axis_labels is None else list(axis_labels)
    if len(axis_labels) != n_ax:
        raise ValueError(f"{len(axis_labels)} axis labels for {n_ax} axis windows")
    present = np.array(info['n_kept']) > 0
    return {k: sums[k][present] for k in keys}, info, axis_labels


def condition_trajectories(df, hg_windows, axis_hg, axis_labels=None, n_splits=200, seed=0,
                           remove_mean=False, first='congruency', n_boot=1000, alpha=0.05,
                           boot_seed=None, chunk=25):
    """The four condition-mean trajectories in the coding plane (§4).

    Arguments as for `trajectory_sums`; with the coding axes' `n_splits` and
    `seed` the halves are theirs. `axis_labels` names each column of `axis_hg`
    (default '0', '1', ...); `first` is the axis kept whole by Gram-Schmidt
    ('congruency': x = congruency, y = switch beyond it). State the order in the
    caption. `first=None` skips Gram-Schmidt: x and y are the projections on
    the unit c and unit s axes. Those axes are oblique (`axis_cosine`), so
    distances in a perpendicular plot are not distances in electrode space. The
    fractions are always those of the plane, whatever `first` is. `boot_seed`
    seeds the bootstrap (default `seed`).

    Returns ``(trajectories, fractions, summary)``:

    trajectories  one row per axis window, cells ('centred' or 'uncentred'),
                  cell and time bin (`window`, 0-based): x, y, averaged over
                  splits and both directions; for centred cells also
                  x_lo/x_hi, y_lo/y_hi, the participant-bootstrap CI.
    fractions     one row per axis window and time bin, plus `window` = -1
                  for all bins together: plane_fraction, plane_fraction_signal
                  (module docstring), with bootstrap CIs on the overall row.
    summary       per axis window: electrodes and participants used, the
                  cosine between its c and s axes (split-averaged, before
                  Gram-Schmidt; the plot's axes are perpendicular, the codes
                  are not), the overall fractions with CIs, and the chance
                  level 2 / n_electrodes of the raw fraction.
    """
    sums, info, axis_labels = _present_sums(df, hg_windows, axis_hg, axis_labels, n_splits,
                                            seed, remove_mean, _TRAJECTORY_SUMS)
    n_part = len(sums['cc'])
    total = _weighted(sums, np.ones(n_part))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    plane = first or 'congruency'

    # participant bootstrap of the centred coordinates and the fractions
    rng = np.random.default_rng(seed if boot_seed is None else boot_seed)
    b_x, b_y, b_raw, b_sig = [], [], [], []
    for start in range(0, n_boot, chunk):
        w = rng.multinomial(n_part, np.full(n_part, 1 / n_part),
                            size=min(chunk, n_boot - start)).T.astype(float)
        weighted = _weighted(sums, w)                    # each (draws in chunk, ...)
        for j in range(w.shape[1]):
            one = {k: v[j] for k, v in weighted.items()}
            x, y = _coordinates(one, first)
            b_x.append(np.nanmean(x, (1, 2)))
            b_y.append(np.nanmean(y, (1, 2)))
            fx, fy = (x, y) if first else _coordinates(one, plane)
            _, _, raw, sig = _fractions(fx, fy, one)
            b_raw.append(raw)
            b_sig.append(sig)

    rows, frac_rows = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ci = {name: np.nanpercentile(np.array(d), q, axis=0)
              for name, d in (('x', b_x), ('y', b_y), ('raw', b_raw), ('sig', b_sig))}
        for cells in ('centred', 'uncentred'):
            x, y = _coordinates(total, first, cells)
            xm, ym = np.nanmean(x, (1, 2)), np.nanmean(y, (1, 2))     # (axes, time, 4)
            for a, label in enumerate(axis_labels):
                for ci_cell, cell in enumerate(CELLS):
                    for t in range(xm.shape[1]):
                        r = dict(axes=label, cells=cells, cell=cell, window=t,
                                 x=xm[a, t, ci_cell], y=ym[a, t, ci_cell])
                        if cells == 'centred':
                            r.update(x_lo=ci['x'][0][a, t, ci_cell], x_hi=ci['x'][1][a, t, ci_cell],
                                     y_lo=ci['y'][0][a, t, ci_cell], y_hi=ci['y'][1][a, t, ci_cell])
                        rows.append(r)
        raw_t, sig_t, raw, sig = _fractions(*_coordinates(total, plane), total)
        cos_axes = np.nanmean(total['cs'] / np.sqrt(total['cc'] * total['ss']), (1, 2))
    n_e = int(sum(info['n_kept']))
    summary = {}
    for a, label in enumerate(axis_labels):
        for t in range(raw_t.shape[1]):
            frac_rows.append(dict(axes=label, window=t, plane_fraction=raw_t[a, t],
                                  plane_fraction_signal=sig_t[a, t]))
        frac_rows.append(dict(axes=label, window=-1, plane_fraction=raw[a],
                              plane_fraction_signal=sig[a],
                              plane_fraction_lo=ci['raw'][0][a], plane_fraction_hi=ci['raw'][1][a],
                              plane_fraction_signal_lo=ci['sig'][0][a],
                              plane_fraction_signal_hi=ci['sig'][1][a]))
        summary[label] = dict(
            axis_cosine=float(cos_axes[a]),
            plane_fraction=float(raw[a]),
            plane_fraction_ci=[float(v) for v in ci['raw'][:, a]],
            plane_fraction_chance=2 / n_e,
            plane_fraction_signal=float(sig[a]),
            plane_fraction_signal_ci=[float(v) for v in ci['sig'][:, a]])
    summary = dict(axes=summary, first=first, n_electrodes=n_e,
                   n_subjects=int(sum(k > 0 for k in info['n_kept'])),
                   n_electrodes_dropped=int(sum(info['n_dropped'])),
                   dropped_by_subject={s: d for s, d in zip(info['subjects'], info['n_dropped'])
                                       if d},
                   n_splits=n_splits, n_boot=n_boot, seed=seed, boot_seed=boot_seed,
                   alpha=alpha, remove_mean=bool(remove_mean))
    return pd.DataFrame(rows), pd.DataFrame(frac_rows), summary


VARIABLES = tuple(_SEPARATION)                               # ('congruency', 'switchType')
CONTRASTS = ('incProp', 'switchProp', 'own_minus_other')


def _separations(sums, unique=False):
    """Each variable's held-out separation per block on its own unit axis
    (`unique`: on its own axis's part orthogonal to the other axis), averaged
    over splits and directions: (..., variables, n_axes, time, blocks), in
    `VARIABLES` and `BLOCKS` order. `sums` may carry leading (draw) axes. The
    axes are per split and direction, from the pooled c and s."""
    def sep(q, var):                                 # axis . separation, per block
        return np.einsum('...xb,x->...b', sums[q], _SEPARATION[var])
    cc, ss, cs = sums['cc'], sums['ss'], sums['cs']
    cd, sd = sep('cQ', 'congruency'), sep('sQ', 'congruency')     # c.d_c, s.d_c
    cw, sw = sep('cQ', 'switchType'), sep('sQ', 'switchType')     # c.d_s, s.d_s
    e = (Ellipsis, None, None)
    if unique:   # c_perp.d = c.d - (c.s/s.s) s.d;  |c_perp|^2 = c.c - (c.s)^2/s.s;  same for s
        proj = [cd - (cs / ss)[e] * sd, sw - (cs / cc)[e] * cw]
        norm2 = [cc - cs ** 2 / ss, ss - cs ** 2 / cc]
    else:
        proj, norm2 = [cd, sw], [cc, ss]
    return np.stack([(p / np.sqrt(n)[e]).mean((-4, -3)) for p, n in zip(proj, norm2)], -4)


def _contrasts(seps):
    """(..., variables, n_axes, time, contrasts) from `_separations`, in
    `CONTRASTS` order: the change with incProp and with switchProp (low - high,
    the other proportion weighted equally), and own - other."""
    change = {p: seps @ w for p, w in _PROPORTION.items()}        # (..., variables, axes, time)
    own = np.stack([change[OWN_PROPORTION[v]][..., i, :, :] for i, v in enumerate(VARIABLES)], -3)
    other = change['incProp'] + change['switchProp'] - own
    return np.stack([change['incProp'], change['switchProp'], own - other], -1)


def _with_overall(a, axis=-2):
    """Append the mean over time bins (`axis`) as a last bin, read as window -1."""
    return np.concatenate([a, a.mean(axis, keepdims=True)], axis)


def block_separations(df, hg_windows, axis_hg, axis_labels=None, n_splits=200, seed=0,
                      remove_mean=False, unique=False, n_boot=1000, alpha=0.05, boot_seed=None,
                      chunk=25):
    """Adaptation in the trajectories (plan §6): the held-out congruency and
    switch separations within each proportion block, each projected on its own
    unit axis (module docstring).

    Arguments as for `condition_trajectories`; with the same `n_splits` and
    `seed` the axes, halves and electrodes are the trajectories' own.
    `unique=True` projects each separation on its own axis's part orthogonal
    to the other axis (c_perp = c - (c.s / s.s) s, s_perp likewise; symmetric,
    so the order does not matter): does the adaptation sit in what each code
    does not share? A shared direction that adapts counts on the raw axes (the
    default, the main analysis) but not here.

    Returns ``(separations, contrasts, summary)``:

    separations  one row per axis window (`axes`), variable ('congruency',
                 'switchType'), block (`inc_prop`, `switch_prop`: 'low' /
                 'high') and time bin (`window`, 0-based; -1 = the mean over
                 all bins): separation (HG units along the unit axis), and its
                 participant-bootstrap CI (lo, hi).
    contrasts    one row per axis window, variable, contrast and window:
                 estimate, lo, hi, excludes_0 (the CI excludes 0).
                     incProp          separation in the low- minus the high-
                                      incProp blocks, switchProp weighted equally
                     switchProp       the same for switchProp
                     own_minus_other  the change with the variable's own
                                      proportion (congruency: incProp; switch:
                                      switchProp) minus the change with the other
                 Positive = a larger separation in the low-proportion blocks,
                 the behavioural direction of LWPC and LWPS. The double
                 dissociation predicts the own change away from 0 and the other
                 at 0. own_minus_other compares them within the same electrodes
                 and trials, so overall responsiveness, which scales both,
                 cancels.
    summary      per axis window and variable, the overall (window -1)
                 contrasts with CIs; electrodes and participants used; settings.
    """
    sums, info, axis_labels = _present_sums(df, hg_windows, axis_hg, axis_labels, n_splits,
                                            seed, remove_mean, _PROJECTION_SUMS)
    n_part = len(sums['cc'])
    total = _separations(_weighted(sums, np.ones(n_part)), unique)
    seps, cons = _with_overall(total), _with_overall(_contrasts(total))

    # participant bootstrap; everything is a ratio of participant sums, so a
    # draw only re-weights them
    rng = np.random.default_rng(seed if boot_seed is None else boot_seed)
    b_seps, b_cons = [], []
    for start in range(0, n_boot, chunk):
        w = rng.multinomial(n_part, np.full(n_part, 1 / n_part),
                            size=min(chunk, n_boot - start)).T.astype(float)
        draws = _separations(_weighted(sums, w), unique)   # (draws, variables, axes, time, blocks)
        b_seps.append(_with_overall(draws))
        b_cons.append(_with_overall(_contrasts(draws)))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    seps_lo, seps_hi = np.percentile(np.concatenate(b_seps), q, axis=0)
    cons_lo, cons_hi = np.percentile(np.concatenate(b_cons), q, axis=0)

    n_t = seps.shape[-2]
    windows = [*range(n_t - 1), -1]
    sep_rows, con_rows, summary = [], [], {}
    for a, label in enumerate(axis_labels):
        summary[label] = {}
        for v, var in enumerate(VARIABLES):
            for t, window in enumerate(windows):
                for b, (inc, sw) in enumerate(BLOCKS):
                    sep_rows.append(dict(axes=label, variable=var, inc_prop=inc, switch_prop=sw,
                                         window=window, separation=seps[v, a, t, b],
                                         lo=seps_lo[v, a, t, b], hi=seps_hi[v, a, t, b]))
                for j, con in enumerate(CONTRASTS):
                    lo, hi = cons_lo[v, a, t, j], cons_hi[v, a, t, j]
                    con_rows.append(dict(axes=label, variable=var, contrast=con, window=window,
                                         estimate=cons[v, a, t, j], lo=lo, hi=hi,
                                         excludes_0=bool(lo > 0 or hi < 0)))
            summary[label][var] = {
                con: dict(estimate=float(cons[v, a, -1, j]),
                          ci=[float(cons_lo[v, a, -1, j]), float(cons_hi[v, a, -1, j])])
                for j, con in enumerate(CONTRASTS)}
            summary[label][var]['own_proportion'] = OWN_PROPORTION[var]
    summary = dict(axes=summary, n_electrodes=int(sum(info['n_kept'])), n_subjects=n_part,
                   n_electrodes_dropped=int(sum(info['n_dropped'])),
                   n_splits=n_splits, n_boot=n_boot, seed=seed, boot_seed=boot_seed,
                   alpha=alpha, remove_mean=bool(remove_mean), unique=bool(unique))
    return pd.DataFrame(sep_rows), pd.DataFrame(con_rows), summary


PROPORTION_SETS = tuple((v, p) for v in _LEVELS for p in _PROPS)
_CHANGE = np.array([1.0, -1.0, -1.0, 1.0])      # (level 1 - level 0) at 25 % minus at 75 %
_AXIS_NAMES = ('congruency', 'switch')


def _set_weights(variable, proportion):
    """The four conditions of `variable` x `proportion`: names (e.g. 'i25') and
    weights (4, 4 cells, 4 blocks), in the order level 1 at 25 %, level 0 at
    25 %, level 1 at 75 %, level 0 at 75 %. Each is the equal-weight mean of
    its cells over the other variable and its blocks over the other
    proportion."""
    v, level_names = _LEVELS[variable]
    p = _PROPS[proportion]
    names, weights = [], []
    for block_level, pct in ((1.0, '25'), (0.0, '75')):
        for level, name in zip((1.0, 0.0), level_names):
            cell = np.array([c[v] == level for c in CELLS.values()], float)
            block = np.array([b[p] == block_level for b in BLOCKS.values()], float)
            names.append(name + pct)
            weights.append(np.outer(cell, block) / (cell.sum() * block.sum()))
    return names, np.stack(weights)


def _oblique(sums, cp, sp):
    """(a, b): the in-plane part of each condition written as a c_hat + b s_hat,
    from its projections `cp`, `sp` (..., n_axes, splits, 2, time, n). Solves
    p.c_hat = a + b k and p.s_hat = a k + b, with k = cos(c, s)."""
    nc = np.sqrt(sums['cc'])[..., None, None]
    ns = np.sqrt(sums['ss'])[..., None, None]
    k = sums['cs'][..., None, None] / (nc * ns)
    pc, ps = cp / nc, sp / ns
    return (pc - k * ps) / (1 - k ** 2), (ps - k * pc) / (1 - k ** 2)


def _proportion_coordinates(sums, weights):
    """(a, b) of every condition in `weights`, centred on the grand mean of the
    sixteen cell-block means and averaged over splits and directions:
    (..., n_axes, time, n, 2)."""
    cp, sp = _projections(sums, np.concatenate([weights, np.full((1, 4, 4), 1 / 16)]))
    a, b = _oblique(sums, cp[..., :-1] - cp[..., -1:], sp[..., :-1] - sp[..., -1:])
    return np.stack([a, b], -1).mean((-5, -4))


def proportion_trajectories(df, hg_windows, axis_hg, axis_labels=None, n_splits=200, seed=0,
                            remove_mean=False, n_boot=1000, alpha=0.05, boot_seed=None,
                            chunk=25):
    """Adaptation as trajectories (plan §6): for each variable crossed with each
    proportion, the held-out means of the variable's two levels in the 25 % and
    75 % blocks, on the congruency and switch axes (oblique coordinates, module
    docstring). Centred on the mean of all conditions at each time bin, the
    same centre as `condition_trajectories`.

    Arguments as for `condition_trajectories`; with the same `n_splits` and
    `seed` the axes, halves and electrodes are the trajectories' own.

    The prediction: in the own sets (congruency x incProp, switchType x
    switchProp) the separation between the two levels differs between the 25 %
    and 75 % blocks along the variable's own axis only; in the cross sets it
    does not differ.

    Returns ``(trajectories, changes, summary)``:

    trajectories  one row per axis window (`axes`), set ('congruency x
                  incProp', ...), condition ('i25', 'c25', 'i75', 'c75'; 's',
                  'r' for switch type; the number is the set's proportion) and
                  time bin: `congruency` and `switch` coordinates (HG), with
                  participant-bootstrap CIs (`*_lo`, `*_hi`); `own` marks the
                  own sets.
    changes       one row per axis window, set, `axis` ('congruency',
                  'switch') and time bin (-1 = mean over bins): the change in
                  separation, (level 1 - level 0) at 25 % minus at 75 %, along
                  that axis: estimate, lo, hi, excludes_0. Positive = a larger
                  separation in the 25 % blocks (the sign of
                  `block_separations`).
    summary       per axis window: the axes' cosine and, per set, `own` and
                  the overall change along each axis with its CI; electrodes
                  and participants used; settings.
    """
    sums, info, axis_labels = _present_sums(df, hg_windows, axis_hg, axis_labels, n_splits,
                                            seed, remove_mean, _PROJECTION_SUMS)
    n_part = len(sums['cc'])
    sets = [_set_weights(v, p) for v, p in PROPORTION_SETS]
    weights = np.concatenate([w for _, w in sets])                # (16, 4 cells, 4 blocks)

    def coordinates_and_changes(s):
        coords = _proportion_coordinates(s, weights)              # (..., axes, time, 16, 2)
        by_set = coords.reshape(*coords.shape[:-2], len(sets), 4, 2)
        return coords, _with_overall(np.einsum('...sxc,x->...sc', by_set, _CHANGE), -3)

    total = _weighted(sums, np.ones(n_part))
    coords, changes = coordinates_and_changes(total)
    cos_axes = (total['cs'] / np.sqrt(total['cc'] * total['ss'])).mean((1, 2))

    # participant bootstrap; a draw only re-weights the participant sums
    rng = np.random.default_rng(seed if boot_seed is None else boot_seed)
    b_coords, b_changes = [], []
    for start in range(0, n_boot, chunk):
        w = rng.multinomial(n_part, np.full(n_part, 1 / n_part),
                            size=min(chunk, n_boot - start)).T.astype(float)
        bc, bch = coordinates_and_changes(_weighted(sums, w))
        b_coords.append(bc)
        b_changes.append(bch)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    co_lo, co_hi = np.percentile(np.concatenate(b_coords), q, axis=0)
    ch_lo, ch_hi = np.percentile(np.concatenate(b_changes), q, axis=0)

    n_t = coords.shape[1]
    traj_rows, change_rows, summary = [], [], {}
    for a, label in enumerate(axis_labels):
        summary[label] = dict(axis_cosine=float(cos_axes[a]), sets={})
        for s, ((var, prop), (names, _)) in enumerate(zip(PROPORTION_SETS, sets)):
            name, own = f'{var} x {prop}', OWN_PROPORTION[var] == prop
            for t in range(n_t):
                for n, cond in enumerate(names):
                    i = 4 * s + n
                    traj_rows.append(dict(
                        axes=label, set=name, own=own, condition=cond, window=t,
                        **{f'{ax}{suffix}': v[a, t, i, j]
                           for j, ax in enumerate(_AXIS_NAMES)
                           for suffix, v in (('', coords), ('_lo', co_lo), ('_hi', co_hi))}))
            for t, window in enumerate([*range(n_t), -1]):
                for j, ax in enumerate(_AXIS_NAMES):
                    lo, hi = ch_lo[a, t, s, j], ch_hi[a, t, s, j]
                    change_rows.append(dict(axes=label, set=name, own=own, axis=ax, window=window,
                                            estimate=changes[a, t, s, j], lo=lo, hi=hi,
                                            excludes_0=bool(lo > 0 or hi < 0)))
            summary[label]['sets'][name] = dict(own=own, **{
                ax: dict(estimate=float(changes[a, -1, s, j]),
                         ci=[float(ch_lo[a, -1, s, j]), float(ch_hi[a, -1, s, j])])
                for j, ax in enumerate(_AXIS_NAMES)})
    summary = dict(axes=summary, n_electrodes=int(sum(info['n_kept'])), n_subjects=n_part,
                   n_electrodes_dropped=int(sum(info['n_dropped'])),
                   n_splits=n_splits, n_boot=n_boot, seed=seed, boot_seed=boot_seed,
                   alpha=alpha, remove_mean=bool(remove_mean))
    return pd.DataFrame(traj_rows), pd.DataFrame(change_rows), summary
