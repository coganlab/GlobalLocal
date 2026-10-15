"""Decoding from each electrode's window-mean high gamma, and comparing
electrode groups on it.

Each trial is one vector of per-electrode mean HG over a window (0-1.5 s by
default): channels are the features and time is not. The question this answers
is whether electrode groups (task-significant, non-significant, all) carry the
decoded contrast equally well, so it is a comparison between groups, and two
things make a naive comparison misleading:

* Size. Accuracy grows with the number of electrodes, so every group is
  decoded at one matched size, ``n_match`` electrodes drawn at random from the
  group, afresh for each bootstrap.
* Pseudo-replication. The bootstraps and CV repeats re-use the same trials, so
  a test across them (an ANOVA on bootstrap accuracies) becomes significant
  just by running more of them. Two groups are compared instead with a
  membership permutation: the electrodes of both groups are pooled and dealt
  back into groups of the original sizes (keeping any overlap the two groups
  have), and the whole matched decode is repeated. The null is "group
  membership does not matter", at the electrodes' level.

``match_electrodes=False`` turns the size matching off: every group is decoded
on all of its electrodes. The accuracies then mix what each electrode carries
with how many there are, but the permutation test still compares each group
with random groups of its own size.

Computation only: callers load the data, save the results and plot them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from src.analysis.utils.labeled_array_utils import get_data_in_time_range
from .data_prep import concatenate_and_balance_data_for_decoding
from .decoder import Decoder


def window_mean_dataset(roi_labeled_array, strings_to_find, time_range=(0.0, 1.5),
                        balance_method='subsample', balance_strata=True, random_state=None):
    """One bootstrap's decoding data as window means.

    ``roi_labeled_array``: one ROI's LabeledArray (condition, trial, channel,
    time), as the decoding bootstrap builds it. The trials are balanced as in
    the time-resolved decode, then each electrode's HG is averaged over
    ``time_range`` (s, inclusive).

    Returns ``(X, y, cats, channels)``: X is (n_trials, n_channels), y the class
    labels, cats the class coding, channels the ``'<subject>-<electrode>'``
    names of X's columns.
    """
    cropped = get_data_in_time_range(roi_labeled_array, time_range)
    data, y, cats = concatenate_and_balance_data_for_decoding(
        {'roi': cropped}, 'roi', strings_to_find, obs_axs=0, balance_method=balance_method,
        balance_strata=balance_strata, random_state=random_state)
    X = np.nanmean(np.asarray(data, float), axis=-1)
    channels = np.asarray(cropped.labels[2]).astype(str)
    return X, np.asarray(y), cats, channels


def decode_accuracy(X, y, cats, columns, clf, explained_variance=0.9, n_splits=5,
                    n_repeats=5, random_state=0, shuffle=False):
    """Cross-validated accuracy of decoding ``y`` from columns ``columns`` of X,
    one value per CV repeat. ``shuffle`` permutes the training labels (chance)."""
    dec = Decoder(cats, explained_variance, oversample=True, n_splits=n_splits,
                  n_repeats=n_repeats, clf=clf, clf_params={}, random_state=random_state)
    cm = dec.cv_cm_jim_window_shuffle(X[:, columns, None], y, normalize=None, obs_axs=0,
                                      time_axs=-1, window=None, shuffle=shuffle)
    return np.trace(cm, axis1=-2, axis2=-1) / cm.sum(axis=(-2, -1))   # (n_repeats,)


def _group_columns(members, n_match, rng):
    """The columns a group is decoded on, sorted: ``n_match`` drawn without
    replacement from ``members``, or all of ``members`` when ``n_match`` is None."""
    if n_match is None:
        return np.sort(members)
    return np.sort(rng.choice(members, size=n_match, replace=False))


def _group_diff(datasets, cols1, cols2, n_match, seed, decode_kw):
    """Mean accuracy of group 1 minus group 2 over bootstraps and repeats, each
    group decoded on its own ``n_match`` random members in every bootstrap (all
    of them when ``n_match`` is None): the statistic of one membership
    permutation."""
    rng = np.random.default_rng(seed)
    diffs = []
    for b, (X, y, cats) in enumerate(datasets):
        a1 = decode_accuracy(X, y, cats, _group_columns(cols1, n_match, rng),
                             random_state=seed + b, **decode_kw)
        a2 = decode_accuracy(X, y, cats, _group_columns(cols2, n_match, rng),
                             random_state=seed + b, **decode_kw)
        diffs.append(a1.mean() - a2.mean())
    return float(np.mean(diffs))


def compare_electrode_groups(datasets, channels, groups, pairs=None, n_match=None,
                             match_electrodes=True, n_perm=1000, seed=0, n_jobs=1, clf=None,
                             explained_variance=0.9, n_splits=5, n_repeats=5):
    """Decode every group at a matched size (or at its own size) and test pairs
    of groups by membership permutation.

    Parameters
    ----------
    datasets : list of (X, y, cats)
        One per bootstrap (pseudo-trial draw), from :func:`window_mean_dataset`;
        every X has the same columns, named by ``channels``.
    groups : dict
        ``{name: collection of channel names}``; channels not in ``channels``
        are ignored. Groups may overlap or nest (all contains sig).
    pairs : list of (name, name), optional
        Which groups to test; default every pair.
    n_match : int, optional
        Electrodes decoded per group; default the smallest group's size. Only
        with ``match_electrodes``.
    match_electrodes : bool
        True (default) decodes every group at ``n_match`` electrodes. False
        decodes every group on all its electrodes; each permutation then deals
        pseudo-groups of the real groups' sizes and decodes them whole, so the
        null carries the size difference. That null is not centred on zero, so
        p is two-sided by doubling the smaller tail rather than from absolute
        differences (the matched null is symmetric about zero).
    n_perm : int
        Membership permutations per pair. The p-value cannot go below
        1 / (n_perm + 1) (2 / (n_perm + 1) unmatched).
    seed : int
        Seeds the electrode draws, the CV splits and the permutations.

    Returns
    -------
    dict
        ``accuracies``: one row per group, bootstrap and repeat (true and
        shuffled-label accuracy); ``summary``: per group, mean accuracy, its
        2.5-97.5 percentiles over bootstraps and repeats, mean chance;
        ``tests``: per pair, the observed accuracy difference, permutation p,
        Holm-adjusted p and the null's 2.5-97.5 percentiles; ``n_match``
        (None unmatched); ``group_sizes``; ``null`` ({pair: null differences}).
        The summary's ``n_decoded`` is the electrodes each group was decoded on.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis() if clf is None else clf
    decode_kw = dict(clf=clf, explained_variance=explained_variance, n_splits=n_splits,
                     n_repeats=n_repeats)
    index = {c: i for i, c in enumerate(channels)}
    members = {g: np.array(sorted(index[c] for c in chans if c in index))
               for g, chans in groups.items()}
    sizes = {g: len(m) for g, m in members.items()}
    if min(sizes.values()) < 2:
        raise ValueError(f"every group needs at least 2 electrodes; sizes: {sizes}")
    if not match_electrodes:
        if n_match is not None:
            raise ValueError(f"n_match={n_match} needs match_electrodes=True; unmatched, "
                             f"every group is decoded on all its electrodes")
    else:
        n_match = int(n_match or min(sizes.values()))
        if n_match > min(sizes.values()):
            raise ValueError(f"n_match={n_match} exceeds the smallest group ({sizes})")
    names = list(groups)
    pairs = list(pairs) if pairs is not None else [
        (a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    # every group at the matched size (or whole); the shuffled-label decode
    # gives chance
    rng = np.random.default_rng(seed)
    rows = []
    for b, (X, y, cats) in enumerate(datasets):
        for g in names:
            cols = _group_columns(members[g], n_match, rng)
            true = decode_accuracy(X, y, cats, cols, random_state=seed + b, **decode_kw)
            null = decode_accuracy(X, y, cats, cols, random_state=seed + b, shuffle=True,
                                   **decode_kw)
            rows += [dict(group=g, bootstrap=b, repeat=r, accuracy=t, chance=s)
                     for r, (t, s) in enumerate(zip(true, null))]
    acc = pd.DataFrame(rows)
    summary = (acc.groupby('group', sort=False)
               .agg(mean=('accuracy', 'mean'),
                    lo=('accuracy', lambda a: np.percentile(a, 2.5)),
                    hi=('accuracy', lambda a: np.percentile(a, 97.5)),
                    chance=('chance', 'mean'))
               .reindex(names).reset_index())
    summary['n_electrodes'] = summary['group'].map(sizes)
    summary['n_match'] = n_match
    summary['n_decoded'] = n_match if match_electrodes else summary['n_electrodes']

    # pairs: the observed difference against membership permutations
    tests, nulls = [], {}
    perm_rng = np.random.default_rng(seed + 1)
    mean_acc = acc.groupby('group')['accuracy'].mean()
    for g1, g2 in pairs:
        # observed: the same matched draws the summary reports
        observed = float(mean_acc[g1] - mean_acc[g2])
        pool = np.union1d(members[g1], members[g2])
        n1, n2 = sizes[g1], sizes[g2]
        # dealing the shuffled pool front (g1) and back (g2) keeps the overlap
        # size: disjoint groups split the pool, a group equal to the pool stays it
        deals = [perm_rng.permutation(pool) for _ in range(int(n_perm))]
        null = np.array(Parallel(n_jobs=n_jobs)(
            delayed(_group_diff)(datasets, d[:n1], d[len(pool) - n2:], n_match,
                                 seed + 2 + k, decode_kw)
            for k, d in enumerate(deals)))
        if match_electrodes:
            p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
        else:
            # unmatched, the null is centred on the size difference
            tail = min(np.sum(null >= observed), np.sum(null <= observed))
            p = min(1.0, 2 * (1 + tail) / (1 + len(null)))
        tests.append(dict(group_1=g1, group_2=g2, difference=observed, p=float(p),
                          null_lo=float(np.percentile(null, 2.5)),
                          null_hi=float(np.percentile(null, 97.5)), n_perm=int(n_perm)))
        nulls[(g1, g2)] = null
    tests = pd.DataFrame(tests)
    if len(tests):
        tests['p_holm'] = multipletests(tests['p'], method='holm')[1]
    return dict(accuracies=acc, summary=summary, tests=tests, n_match=n_match,
                group_sizes=sizes, null=nulls)
