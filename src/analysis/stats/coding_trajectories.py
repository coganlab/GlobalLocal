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
_TRAJECTORY_SUMS = ('cc', 'ss', 'cs', 'cP', 'sP', 'cU', 'sU', 'PP', 'PAPB')
_BLOCK_SUMS = ('cc', 'ss', 'cB', 'sB')


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
        cP, sP          c.p, s.p per cell, centred    (n_axes, splits, 2, time, 4)
        cU, sU          the same, cells uncentred
        PP              |p|^2 summed over the cells   (splits, 2, time)
        PAPB            p_A . p_B summed over cells   (splits, time)
        cB              c . (congruency separation)   (n_axes, splits, 2, time, 4)
                        within each proportion block (`BLOCKS` order)
        sB              s . (switch separation), the same

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
    parts = {k: [] for k in _TRAJECTORY_SUMS + ('cB', 'sB')}
    info = dict(subjects=[], n_kept=[], n_dropped=[])
    for subj, elecs, blocks in cell_block_means(df, hg_windows, halves, remove_mean):
        ax = _participant_axes(axes, subj, elecs)
        c = np.stack([ax['mxA'], ax['mxB']], -1)        # (e, axes, splits, d): d=0 -> A
        s = np.stack([ax['myA'], ax['myB']], -1)
        blocks = blocks[:, :, :, ::-1]                  # (e, t, splits, d, 4, 4): d=0 -> B
        uncentred = blocks.mean(-1)                     # block-balanced cells
        centred = uncentred - uncentred.mean(-1, keepdims=True)
        ok = (np.isfinite(c).all((1, 2, 3)) & np.isfinite(s).all((1, 2, 3))
              & np.isfinite(blocks).all((1, 2, 3, 4, 5)))
        info['subjects'].append(subj)
        info['n_kept'].append(int(ok.sum()))
        info['n_dropped'].append(int((~ok).sum()))
        c, s, centred, uncentred, blocks = c[ok], s[ok], centred[ok], uncentred[ok], blocks[ok]
        parts['cc'].append(np.einsum('eakd,eakd->akd', c, c))
        parts['ss'].append(np.einsum('eakd,eakd->akd', s, s))
        parts['cs'].append(np.einsum('eakd,eakd->akd', c, s))
        for name, axis in (('c', c), ('s', s)):
            parts[f'{name}P'].append(np.einsum('eakd,etkdx->akdtx', axis, centred))
            parts[f'{name}U'].append(np.einsum('eakd,etkdx->akdtx', axis, uncentred))
        parts['PP'].append(np.einsum('etkdx,etkdx->kdt', centred, centred))
        parts['PAPB'].append(np.einsum('etkx,etkx->kt', centred[:, :, :, 1], centred[:, :, :, 0]))
        for name, axis, var in (('c', c, 'congruency'), ('s', s, 'switchType')):
            separation = np.einsum('etkdxb,x->etkdb', blocks, _SEPARATION[var])
            parts[f'{name}B'].append(np.einsum('eakd,etkdb->akdtb', axis, separation))
    if not sum(info['n_kept']):
        raise ValueError("no electrode has every cell in every half; the trajectories need "
                         "at least one trial per cell and proportion block in each half")
    return {k: np.stack(v) for k, v in parts.items()}, info


def _coordinates(sums, first='congruency', cells='centred'):
    """(x, y), each (n_axes, splits, 2, time, 4), from sums already combined
    over participants (leading participant axis removed)."""
    if first not in FIRST_AXES:
        raise ValueError(f"first must be one of {FIRST_AXES}; got {first!r}")
    p = 'P' if cells == 'centred' else 'U'
    (aa, bb), (ap, bp) = ((sums['cc'], sums['ss']), (sums[f'c{p}'], sums[f's{p}']))
    if first == 'switch':
        aa, bb, ap, bp = bb, aa, bp, ap
    ab = sums['cs']
    expand = (Ellipsis, None, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = ap / np.sqrt(aa)[expand]
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
    caption. `boot_seed` seeds the bootstrap (default `seed`).

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
            _, _, raw, sig = _fractions(x, y, one)
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
        x, y = _coordinates(total, first)
        raw_t, sig_t, raw, sig = _fractions(x, y, total)
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


def _separations(sums):
    """Each variable's held-out separation per block on its own unit axis,
    averaged over splits and directions: (..., variables, n_axes, time,
    blocks), in `VARIABLES` and `BLOCKS` order. `sums` may carry leading (draw)
    axes. The unit axis is per split and direction, from the pooled |c| (|s|)."""
    out = [sums[sep] / np.sqrt(sums[sq])[..., None, None] for sep, sq in (('cB', 'cc'), ('sB', 'ss'))]
    return np.stack([o.mean((-4, -3)) for o in out], -4)


def _contrasts(seps):
    """(..., variables, n_axes, time, contrasts) from `_separations`, in
    `CONTRASTS` order: the change with incProp and with switchProp (low - high,
    the other proportion weighted equally), and own - other."""
    change = {p: seps @ w for p, w in _PROPORTION.items()}        # (..., variables, axes, time)
    own = np.stack([change[OWN_PROPORTION[v]][..., i, :, :] for i, v in enumerate(VARIABLES)], -3)
    other = change['incProp'] + change['switchProp'] - own
    return np.stack([change['incProp'], change['switchProp'], own - other], -1)


def _with_overall(a):
    """Append the mean over time bins (axis -2) as a last bin, read as window -1."""
    return np.concatenate([a, a.mean(-2, keepdims=True)], -2)


def block_separations(df, hg_windows, axis_hg, axis_labels=None, n_splits=200, seed=0,
                      remove_mean=False, n_boot=1000, alpha=0.05, boot_seed=None, chunk=25):
    """Adaptation in the trajectories (plan §6): the held-out congruency and
    switch separations within each proportion block, each projected on its own
    unit axis (module docstring).

    Arguments as for `condition_trajectories`; with the same `n_splits` and
    `seed` the axes, halves and electrodes are the trajectories' own.

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
                                            seed, remove_mean, _BLOCK_SUMS)
    n_part = len(sums['cc'])
    total = _separations(_weighted(sums, np.ones(n_part)))
    seps, cons = _with_overall(total), _with_overall(_contrasts(total))

    # participant bootstrap; everything is a ratio of participant sums, so a
    # draw only re-weights them
    rng = np.random.default_rng(seed if boot_seed is None else boot_seed)
    b_seps, b_cons = [], []
    for start in range(0, n_boot, chunk):
        w = rng.multinomial(n_part, np.full(n_part, 1 / n_part),
                            size=min(chunk, n_boot - start)).T.astype(float)
        draws = _separations(_weighted(sums, w))           # (draws, variables, axes, time, blocks)
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
                   alpha=alpha, remove_mean=bool(remove_mean))
    return pd.DataFrame(sep_rows), pd.DataFrame(con_rows), summary
