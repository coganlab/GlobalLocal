"""
The joint scatter: each electrode's stability sensitivity against its own
flexibility sensitivity, coloured by subject, with marginal histograms.

This is §2.5 of `docs/analysis_simplification_plan.md` -- the descriptive figure
that is meant to come BEFORE any inference machinery, because it shows the
reader the joint distribution the whole segregation question is about instead of
asking them to trust a pipeline.

How to read it (from the plan):

    positive diagonal ................ shared mechanism
    spread on both axes, no
      correspondence ................. independent mechanisms
    spread on one axis only .......... one mechanism (the other isn't measured)
    negative diagonal ................ opponent / segregated subpopulations
    all the structure in one colour
      or a few points ................ artifact

The last line is the reason this module does more than draw dots. "One colour or
a few points" is checkable rather than eyeballable, so `joint_scatter_diagnostics`
computes it: the per-subject correlations, the leave-one-subject-out range, the
influence of the most extreme electrodes, and the correlation after within-subject
centring (which separates structure that lives BETWEEN subjects -- a subject-level
confound -- from structure that lives within them). Those numbers are drawn on the
figure and returned for serialisation, and any that trip a threshold are raised as
explicit flags.

Nothing here is inferential. There is no permutation null and no p-value: the
sensitivities are correlated as given, so this is a picture of the data and its
leverage, not a test. The test is `stability_flexibility_segregation.
run_joint_distribution_analysis`, whose `correlation` entry corrects for shared
trial noise (disjoint halves), gain (responsiveness residualisation) and subject
nesting -- none of which this figure does. Expect the number here to differ from
the pipeline's, and expect it to be the more optimistic of the two.

Typical use, cheapest first::

    from src.analysis.stats import segregation_scatter as scat
    elec = scat.sensitivities_for_scatter(df, contrast_mode='proportion')
    fig, diag = scat.plot_joint_scatter(elec, save_path='joint_scatter.png')

or, after a full run, straight off its per-electrode table::

    fig, diag = scat.plot_joint_scatter(out['electrodes'], contrast_mode='proportion')
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt


# Default axis labels per contrast mode. 'proportion' (LWPC/LWPS) is the
# manuscript's primary mode; 'condition' scores the two main effects instead.
_AXIS_LABELS = {
    'proportion': ("x = LWPC  (congruency x incongruent-proportion interaction)",
                   "y = LWPS  (switchType x switch-proportion interaction)"),
    'condition': ("x = stability  (congruency: i - c)",
                  "y = flexibility  (switchType: s - r)"),
}

_READING_KEY = ("Reading:  positive diagonal -> shared   |   spread on both axes, no correspondence -> independent   |   "
                "spread on one axis only -> one mechanism   |   negative diagonal -> opponent   |   "
                "all structure in one colour or a few points -> artifact")


# ----------------------------------------------------------------------------
# sensitivities (the cheap route to something plottable)
# ----------------------------------------------------------------------------
def sensitivities_for_scatter(df, contrast_mode='proportion', contrasts=None,
                              effect_measure='cohens_d', alpha=0.05,
                              n_splits=0, seed=0):
    """Per-electrode (x, y) for the scatter, from the long single-trial table.

    `n_splits=0` (default) uses `naive_sensitivities`: both effects on ALL of the
    electrode's trials. That is the cheap "afternoon's work" route -- no
    resampling, no permutation -- and it is the right one for a first look,
    provided the tilt it shows is read as an upper bound. x and y share trials
    there, so shared trial noise inflates the correlation whenever the 2x2
    congruency x switchType cross-tab is non-proportional (plan §2.2).

    `n_splits>0` instead averages the disjoint-half estimates over that many
    splits (`compute_sensitivities_per_split` + `average_over_splits`), which is
    what the pipeline plots. Same trial cost per split as the real analysis, so
    it is markedly slower; use it when the naive scatter looks tilted and you
    want to see how much of the tilt was shared noise.

    Returns a DataFrame with subject, electrode, x, y.
    """
    from src.analysis.stats import stability_flexibility_segregation as sfs

    if n_splits and n_splits > 0:
        per_split = sfs.compute_sensitivities_per_split(
            df, n_splits=n_splits, seed=seed, contrast_mode=contrast_mode,
            contrasts=contrasts, effect_measure=effect_measure, alpha=alpha)
        return sfs.average_over_splits(per_split)
    return sfs.naive_sensitivities(df, contrast_mode=contrast_mode,
                                   contrasts=contrasts,
                                   effect_measure=effect_measure, alpha=alpha)


# ----------------------------------------------------------------------------
# diagnostics: is the structure real, or one colour / a few points?
# ----------------------------------------------------------------------------
def _corr(x, y, method='spearman'):
    """Correlation of two vectors; NaN when there is too little to correlate."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    fn = spearmanr if method == 'spearman' else pearsonr
    with np.errstate(invalid='ignore'):
        r = fn(x[m], y[m])[0]
    return float(r) if np.isfinite(r) else np.nan


def joint_scatter_diagnostics(elec, method='spearman', value_cols=('x', 'y'),
                              n_top=None, drop_frac=0.02, min_corr_to_flag=0.1):
    """Leverage summary for the joint scatter -- the numbers behind "artifact".

    A pooled across-electrode correlation can be produced by three things that
    have nothing to do with co-localization, and each has its own entry here:

      * one subject. `per_subject` gives each subject's own correlation and
        electrode count; `loso_min`/`loso_max` give the pooled correlation with
        each subject held out, and `most_influential_subject` names the one whose
        removal moves it furthest.
      * a few electrodes. `corr_drop_top` recomputes the correlation without the
        `n_top` most influential electrodes (default: 2% of them, at least one),
        chosen by their leave-one-out effect on the correlation. Electrodes are
        dropped only in the direction that SUPPORTS the observed correlation, so
        this is a fair "how much survives without the extreme points" and not a
        search for the smallest achievable number.
      * subject-level structure. `corr_within_subject` centres both axes within
        subject before correlating. Between-subject structure -- subjects who
        happen to be high on both axes -- cannot contribute to it, so a pooled
        correlation much larger than this one is a subject-level effect, not
        an electrode-level one.

    `flags` collects the ones that crossed a threshold, as plain sentences. The
    leverage flags are relative to the observed correlation, so they are only
    raised once |corr| >= `min_corr_to_flag`: below that there is no apparent
    structure to attribute to a subject or a handful of points, and every
    perturbation is large relative to nothing. A flat cloud is a result (the
    plan's "independent mechanisms" reading), not a suspect figure -- what makes
    it interpretable is the noise ceiling from the pipeline, not this panel.
    """
    xc, yc = value_cols
    d = elec.dropna(subset=[xc, yc]).copy()
    x = d[xc].to_numpy(float); y = d[yc].to_numpy(float)
    subj = d['subject'].astype(str).to_numpy()
    n = len(d)

    out = dict(method=method, n_electrodes=int(n),
               n_subjects=int(pd.unique(subj).size),
               corr=_corr(x, y, method))

    # -- within-subject centred correlation (kills between-subject structure) --
    if n >= 3:
        xw, yw = x.copy(), y.copy()
        for s in np.unique(subj):
            m = subj == s
            xw[m] -= xw[m].mean()
            yw[m] -= yw[m].mean()
        out['corr_within_subject'] = _corr(xw, yw, method)
    else:
        out['corr_within_subject'] = np.nan

    # -- per subject: own correlation, share of the electrodes, LOSO ----------
    rows = []
    for s in sorted(np.unique(subj)):
        m = subj == s
        rows.append(dict(subject=s, n_electrodes=int(m.sum()),
                         corr=_corr(x[m], y[m], method),
                         corr_leave_out=_corr(x[~m], y[~m], method)))
    per_subject = pd.DataFrame(rows)
    out['per_subject'] = per_subject
    out['max_subject_share'] = (float(per_subject.n_electrodes.max() / n)
                                if n else np.nan)

    loso = per_subject['corr_leave_out'].to_numpy(float)
    if np.isfinite(loso).any() and np.isfinite(out['corr']):
        out['loso_min'] = float(np.nanmin(loso))
        out['loso_max'] = float(np.nanmax(loso))
        delta = loso - out['corr']
        i = int(np.nanargmax(np.abs(delta)))
        out['most_influential_subject'] = per_subject.subject.iloc[i]
        out['most_influential_subject_delta'] = float(delta[i])
    else:
        out.update(loso_min=np.nan, loso_max=np.nan,
                   most_influential_subject=None,
                   most_influential_subject_delta=np.nan)

    # -- a few points: leave-one-electrode-out influence ----------------------
    if n_top is None:
        n_top = max(1, int(round(drop_frac * n)))
    n_top = int(min(n_top, max(0, n - 3)))
    if n >= 5 and np.isfinite(out['corr']) and n_top > 0:
        keep = np.ones(n, bool)
        delta = np.empty(n)
        for i in range(n):
            keep[i] = False
            delta[i] = _corr(x[keep], y[keep], method) - out['corr']
            keep[i] = True
        # An electrode SUPPORTS the observed correlation when removing it moves
        # the correlation toward zero, i.e. delta has the opposite sign to corr.
        support = -np.sign(out['corr']) * delta
        drop = np.argsort(support)[::-1][:n_top]
        m = np.ones(n, bool); m[drop] = False
        out['n_top_dropped'] = n_top
        out['corr_drop_top'] = _corr(x[m], y[m], method)
        j = int(np.nanargmax(np.abs(delta)))
        out['most_influential_electrode'] = str(d['electrode'].iloc[j]) \
            if 'electrode' in d.columns else None
        out['most_influential_electrode_delta'] = float(delta[j])
    else:
        out.update(n_top_dropped=0, corr_drop_top=np.nan,
                   most_influential_electrode=None,
                   most_influential_electrode_delta=np.nan)

    out['flags'] = _diagnostic_flags(out, min_corr=min_corr_to_flag)
    return out


def _diagnostic_flags(d, min_corr=0.1):
    """Turn the leverage numbers into the warnings the plan's last reading asks
    for. Thresholds are deliberately loose -- these mark a figure as needing a
    second look, they do not adjudicate anything."""
    flags = []
    r = d.get('corr', np.nan)
    if not np.isfinite(r):
        return ["correlation undefined (too few electrodes)"]

    share = d.get('max_subject_share', np.nan)
    if np.isfinite(share) and share > 0.5:
        flags.append(f"one subject contributes {share:.0%} of the electrodes")

    if abs(r) < min_corr:
        # Nothing to attribute. The leverage flags below all ask "is this
        # apparent structure carried by one thing?", which is not a question
        # about a cloud that has no tilt to begin with.
        return flags

    ds = d.get('most_influential_subject_delta', np.nan)
    if np.isfinite(ds) and abs(ds) > 0.5 * abs(r):
        flags.append(
            f"dropping subject {d['most_influential_subject']} moves the "
            f"correlation by {ds:+.3f} (observed {r:+.3f})")

    dt = d.get('corr_drop_top', np.nan)
    if np.isfinite(dt) and (abs(dt) < 0.5 * abs(r) or np.sign(dt) != np.sign(r)):
        flags.append(
            f"without the {d['n_top_dropped']} most influential electrodes the "
            f"correlation is {dt:+.3f} (observed {r:+.3f})")

    rw = d.get('corr_within_subject', np.nan)
    if np.isfinite(rw) and abs(rw) < 0.5 * abs(r):
        flags.append(
            f"within-subject correlation is {rw:+.3f} vs {r:+.3f} pooled -- the "
            "structure is largely BETWEEN subjects")
    return flags


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------
def subject_palette(subjects):
    """Distinct colours per subject, stable in the sorted subject order.

    tab20 + tab20b give 40 categorical colours before anything has to repeat,
    which covers this dataset's 24 subjects; beyond that it falls back to
    sampling hsv so nothing silently collides with a neighbour."""
    subjects = list(subjects)
    n = len(subjects)
    if n <= 40:
        base = (list(plt.get_cmap('tab20').colors)
                + list(plt.get_cmap('tab20b').colors))
        colors = base[:n]
    else:
        colors = [plt.get_cmap('hsv')(i / n) for i in range(n)]
    return dict(zip(subjects, colors))


def _fit_line(ax, x, y):
    """Least-squares line through the cloud, for reading the tilt only."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return
    b1, b0 = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(xs, b0 + b1 * xs, color='k', lw=1.2, ls='--', zorder=5,
            label=f"fit (slope {b1:+.3f})")


def _diagnostics_text(d):
    lines = [
        f"r ({d['method']}) = {d['corr']:+.3f}"
        f"    n = {d['n_electrodes']} electrodes, {d['n_subjects']} subjects",
        f"within-subject r = {d['corr_within_subject']:+.3f}"
        f"    (pooled r above includes between-subject structure)",
    ]
    if np.isfinite(d.get('loso_min', np.nan)):
        lines.append(
            f"leave-one-subject-out r: {d['loso_min']:+.3f} to {d['loso_max']:+.3f}"
            f"    (largest shift: {d['most_influential_subject']}, "
            f"{d['most_influential_subject_delta']:+.3f})")
    if d.get('n_top_dropped'):
        lines.append(
            f"without the {d['n_top_dropped']} most influential electrodes: "
            f"r = {d['corr_drop_top']:+.3f}"
            f"    (largest single: {d['most_influential_electrode']}, "
            f"{d['most_influential_electrode_delta']:+.3f})")
    lines.append(f"largest subject share of electrodes: {d['max_subject_share']:.0%}")
    for f in d['flags']:
        lines.append(f"!  {f}")
    return "\n".join(lines)


def plot_joint_scatter(elec, save_path=None, contrast_mode='proportion',
                       effect_measure=None, method='spearman',
                       value_cols=('x', 'y'), xlabel=None, ylabel=None,
                       title=None, bins=30, figsize=(15, 7.0), dpi=140,
                       diagnostics=None, annotate=None, max_legend_subjects=12,
                       text_size=8.5):
    """Draw the §2.5 joint scatter and return `(fig, diagnostics)`.

    `elec` needs `subject`, `x`, `y` (rename via `value_cols`) and, for the
    single-electrode influence line, `electrode`. Pass `out['electrodes']` from
    `run_joint_distribution_analysis`, or build it with
    `sensitivities_for_scatter`.

    Panels: the joint scatter with a stacked marginal histogram on each axis
    (stacked BY SUBJECT, so a spread carried by one subject is visible as one
    colour dominating a tail), and a per-subject correlation bar chart on the
    right, which doubles as the colour key past `max_legend_subjects`.
    `annotate` adds a line of text under the diagnostics -- use it to show the
    pipeline's corrected estimate beside the descriptive one, since they answer
    different questions.

    `figsize` is the size of the PLOTTING area; the figure grows downward to fit
    the diagnostics block, whose height depends on how many flags fired. Sizing
    the figure to the text (rather than writing the text at a fixed offset) is
    what keeps a heavily flagged figure -- the one you most need to read -- from
    printing its flags on top of the reading key.
    """
    xc, yc = value_cols
    d = elec.dropna(subset=[xc, yc]).copy()
    if d.empty:
        raise ValueError(f"no electrode has finite {xc} and {yc}")
    d['subject'] = d['subject'].astype(str)
    diag = diagnostics or joint_scatter_diagnostics(d, method=method,
                                                    value_cols=value_cols)

    subjects = sorted(d['subject'].unique())
    colors = subject_palette(subjects)
    x = d[xc].to_numpy(float); y = d[yc].to_numpy(float)

    if xlabel is None or ylabel is None:
        dx, dy = _AXIS_LABELS.get(contrast_mode, (xc, yc))
        xlabel = xlabel or dx
        ylabel = ylabel or dy
    if effect_measure:
        xlabel = f"{xlabel}   [{effect_measure}]"
        ylabel = f"{ylabel}   [{effect_measure}]"

    # The text block is written before the figure exists, because its height
    # decides the figure's: reserve one line per diagnostic line, one for the
    # reading key, and two of slack.
    text = _diagnostics_text(diag)
    if isinstance(annotate, str) and annotate:
        text += "\n" + annotate
    line_in = text_size * 1.6 / 72.0                       # inches per line
    text_in = (len(text.splitlines()) + 3) * line_in
    fig_h = figsize[1] + text_in
    bottom = text_in / fig_h

    fig = plt.figure(figsize=(figsize[0], fig_h))
    gs = fig.add_gridspec(2, 3, width_ratios=[4.0, 1.0, 2.4],
                          height_ratios=[1.0, 4.0],
                          wspace=0.06, hspace=0.06,
                          left=0.06, right=0.985,
                          top=1 - 0.5 / fig_h, bottom=bottom + 0.5 / fig_h)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_side = fig.add_subplot(gs[:, 2])

    # -- main scatter, one colour per subject --------------------------------
    for s in subjects:
        m = (d['subject'] == s).to_numpy()
        ax.scatter(x[m], y[m], s=22, alpha=.75, color=colors[s],
                   edgecolors='none', label=f"{s} (n={int(m.sum())})")
    ax.axhline(0, color='k', lw=.6, zorder=1)
    ax.axvline(0, color='k', lw=.6, zorder=1)
    _fit_line(ax, x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # -- marginals, stacked by subject ---------------------------------------
    for a, vals, orient in ((ax_top, x, 'vertical'), (ax_right, y, 'horizontal')):
        edges = np.histogram_bin_edges(vals[np.isfinite(vals)], bins=bins)
        stacks = [vals[(d['subject'] == s).to_numpy()] for s in subjects]
        a.hist(stacks, bins=edges, stacked=True,
               color=[colors[s] for s in subjects], orientation=orient)
        if orient == 'vertical':
            a.axvline(0, color='k', lw=.6)
            a.set_ylabel("# elec", fontsize=8)
            a.tick_params(axis='x', labelbottom=False)
        else:
            a.axhline(0, color='k', lw=.6)
            a.set_xlabel("# elec", fontsize=8)
            a.tick_params(axis='y', labelleft=False)
        a.tick_params(labelsize=7)

    ttl = title or ("Joint distribution of per-electrode stability and "
                    "flexibility sensitivity")
    ax_top.set_title(ttl, fontsize=12, pad=8)

    # -- per-subject correlations (is it all one colour?) --------------------
    ps = diag['per_subject']
    ypos = np.arange(len(ps))
    per_sub_r = ps['corr'].to_numpy(float)
    ax_side.barh(ypos, per_sub_r,
                 color=[colors.get(s, '#888') for s in ps['subject']],
                 height=.75)
    # A subject with < 3 electrodes has no correlation, so barh draws nothing and
    # its colour would drop out of the key entirely. Mark the row so the colour
    # is still identifiable in the scatter.
    for i, (s, r) in enumerate(zip(ps['subject'], per_sub_r)):
        if not np.isfinite(r):
            ax_side.plot(0, i, marker='o', ms=5, color=colors.get(s, '#888'))
            ax_side.text(0.04, i, "n < 3", fontsize=6.5, va='center', color='#666')
    ax_side.axvline(0, color='k', lw=.6)
    if np.isfinite(diag['corr']):
        ax_side.axvline(diag['corr'], color='#d7191c', lw=1.6,
                        label=f"pooled {diag['corr']:+.3f}")
        ax_side.legend(fontsize=8, loc='lower right')
    ax_side.set_yticks(ypos)
    ax_side.set_yticklabels([f"{s}  (n={n})" for s, n
                             in zip(ps['subject'], ps['n_electrodes'])],
                            fontsize=7)
    ax_side.invert_yaxis()
    ax_side.set_xlim(-1.05, 1.05)
    ax_side.set(title="Correlation within each subject", xlabel=f"r ({method})")
    ax_side.tick_params(axis='x', labelsize=8)

    # -- legend + the numbers ------------------------------------------------
    # Past a dozen subjects an in-plot legend covers more data than it explains,
    # and the per-subject panel is already a colour key (same colours, subject
    # and n on every row), so only the fit line stays.
    handles, labels = ax.get_legend_handles_labels()
    if len(subjects) > max_legend_subjects:
        handles = [h for h, l in zip(handles, labels) if l.startswith('fit')]
        labels = [l for l in labels if l.startswith('fit')]
    if handles:
        ncol = max(1, min(6, int(np.ceil(len(handles) / 4))))
        ax.legend(handles, labels, fontsize=6.5, ncol=ncol, loc='best',
                  framealpha=.85, markerscale=.9, handletextpad=.2,
                  columnspacing=.8, labelspacing=.25)

    fig.text(0.06, bottom - 0.3 * line_in / fig_h, text, fontsize=text_size,
             va='top', family='monospace')
    fig.text(0.06, 0.4 * line_in / fig_h, _READING_KEY, fontsize=text_size - 0.5,
             va='bottom', color='#444')

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"saved figure: {save_path}")
    return fig, diag


def diagnostics_to_json(diag):
    """`joint_scatter_diagnostics` output as plain JSON-serialisable types
    (the per-subject table becomes a list of records)."""
    out = {}
    for k, v in diag.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient='records')
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
