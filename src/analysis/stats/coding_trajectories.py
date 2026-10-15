"""Trajectories of the four condition means in the coding plane
(docs/analysis_plans.md, subspace plan §4). The worked math is in
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
"""

import warnings

import numpy as np
import pandas as pd

from . import stability_flexibility_segregation as sfs

# The four cells in (`_scond`, `_fcond`) levels (1 = incongruent / switch), and
# the four proportion blocks in (`_smod`, `_fmod`) levels (1 = low proportion).
CELLS = {'incongruent-switch': (1.0, 1.0), 'incongruent-repeat': (1.0, 0.0),
         'congruent-switch': (0.0, 1.0), 'congruent-repeat': (0.0, 0.0)}
_BLOCKS = ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
_SYSTEM = {'cells': (('_scond', '_fcond', '_smod', '_fmod'),
                     tuple((*cell, *block) for cell in CELLS.values() for block in _BLOCKS))}
FIRST_AXES = ('congruency', 'switch')


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


def balanced_cell_means(df, hg_windows, halves, remove_mean=False):
    """Per participant, each electrode's mean HG in the four congruency x
    switch-type cells, every time bin, split and half: the equal-weight mean of
    the cell's four proportion blocks. Yields ``(subject, electrodes, means)``
    with `means` (electrodes, time bins, splits, 2 halves, 4 cells), cells in
    `CELLS` order; NaN where a block of the cell has no trial in that half."""
    for subj, elecs, cm in sfs.half_cell_means(df, hg_windows, _SYSTEM, halves, remove_mean):
        means, _ = cm['cells']                          # (..., 16) cell-major
        yield subj, elecs, means.reshape(*means.shape[:-1], len(CELLS), len(_BLOCKS)).mean(-1)


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

    over each participant's electrodes defined everywhere (axes on every split
    and axis window; every cell on every split, half and time bin). `info`
    has the subjects, and the electrodes kept and dropped per participant.
    """
    axis_hg = np.asarray(axis_hg, float)
    if axis_hg.ndim == 1:
        axis_hg = axis_hg[:, None]
    halves = sfs.shared_halves(df, n_splits=n_splits, seed=seed)
    axes = sfs.main_effect_mean_diffs(df, axis_hg, remove_mean=remove_mean, halves=halves)
    parts = {k: [] for k in ('cc', 'ss', 'cs', 'cP', 'sP', 'cU', 'sU', 'PP', 'PAPB')}
    info = dict(subjects=[], n_kept=[], n_dropped=[])
    for subj, elecs, cells in balanced_cell_means(df, hg_windows, halves, remove_mean):
        ax = _participant_axes(axes, subj, elecs)
        c = np.stack([ax['mxA'], ax['mxB']], -1)        # (e, axes, splits, d): d=0 -> A
        s = np.stack([ax['myA'], ax['myB']], -1)
        uncentred = cells[:, :, :, ::-1, :]             # (e, t, splits, d, 4): d=0 -> B
        centred = uncentred - uncentred.mean(-1, keepdims=True)
        ok = (np.isfinite(c).all((1, 2, 3)) & np.isfinite(s).all((1, 2, 3))
              & np.isfinite(cells).all((1, 2, 3, 4)))
        info['subjects'].append(subj)
        info['n_kept'].append(int(ok.sum()))
        info['n_dropped'].append(int((~ok).sum()))
        c, s, centred, uncentred = c[ok], s[ok], centred[ok], uncentred[ok]
        parts['cc'].append(np.einsum('eakd,eakd->akd', c, c))
        parts['ss'].append(np.einsum('eakd,eakd->akd', s, s))
        parts['cs'].append(np.einsum('eakd,eakd->akd', c, s))
        for name, axis in (('c', c), ('s', s)):
            parts[f'{name}P'].append(np.einsum('eakd,etkdx->akdtx', axis, centred))
            parts[f'{name}U'].append(np.einsum('eakd,etkdx->akdtx', axis, uncentred))
        parts['PP'].append(np.einsum('etkdx,etkdx->kdt', centred, centred))
        parts['PAPB'].append(np.einsum('etkx,etkx->kt', centred[:, :, :, 1], centred[:, :, :, 0]))
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
    sums, info = trajectory_sums(df, hg_windows, axis_hg, n_splits=n_splits, seed=seed,
                                 remove_mean=remove_mean)
    n_ax = sums['cc'].shape[1]
    axis_labels = [str(i) for i in range(n_ax)] if axis_labels is None else list(axis_labels)
    if len(axis_labels) != n_ax:
        raise ValueError(f"{len(axis_labels)} axis labels for {n_ax} axis windows")
    # A participant with no electrode left would enter the bootstrap as zeros.
    present = np.array(info['n_kept']) > 0
    sums = {k: v[present] for k, v in sums.items()}
    n_part = int(present.sum())
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
