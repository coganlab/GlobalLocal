# Decoding script, takes a GroupData set and uses Linear Discriminant Analysis
# to decode trial conditions or word tokens from neural data

import numpy as np
import os

from aaron_grouping import GroupData
from analysis.utils.plotting import plot_horizontal_bars
from ieeg.calc.stats import time_perm_cluster
from aaron_decoding_init import (Decoder, get_scores, plot_all_scores, plot_dist_bound)


def dict_to_structured_array(dict_matrices, filename='structured_array.npy'):
    # Get the keys and shapes
    keys = list(dict_matrices.keys())
    shape = dict_matrices[keys[0]].shape

    # Create a data type for the structured array
    dt = np.dtype([(key, dict_matrices[key].dtype, shape) for key in keys])

    # Create the structured array
    structured_array = np.zeros((1,), dtype=dt)

    # Fill the structured array
    for key in keys:
        structured_array[key] = dict_matrices[key]

    # Save the structured array to a file
    np.save(filename, structured_array)


def score(categories, test_size, method, n_splits, n_repeats, sub, idxs,
          conds, window_kwargs, scores_dict, shuffle=False):
    decoder = Decoder(categories, test_size, method, n_splits=n_splits, n_repeats=n_repeats)
    names = list(scores_dict.keys())
    while len(scores_dict) > 0:
        scores_dict.popitem()
    for key, values in get_scores(sub, decoder, idxs, conds, names, shuffle=shuffle, **window_kwargs):
        print(key)
        scores_dict[key] = values
    return scores_dict


if __name__ == '__main__':

    # %% Imports
    box = os.path.expanduser(os.path.join("~","Box"))
    fpath = os.path.join(box, "CoganLab")
    subjects_dir = os.path.join(box, "ECoG_Recon")
    sub = GroupData.from_intermediates(
        "SentenceRep", fpath, folder='stats', subjects_dir=subjects_dir)
    all_data = []
    colors = [[1, 0, 0]]
    scores = {'Sensory-Motor': None}
    scores2 = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
    idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
    idxs = [list(idx) for idx in idxs]
    names = list(scores.keys())
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -2,
                    'average_repetitions': False}

    # %% Time Sliding decoding for word tokens

    scores = score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}, 0.8, 'lda', 5, 10, sub, idxs, conds,
                                window_kwargs, scores,
                                shuffle=False)
    # dict_to_structured_array(scores, 'true_scores.npy')
    # scores2 = score({'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4},
    #                                 0.8, 'lda', 5, 250, sub, idxs, conds,
    #                                 window_kwargs, scores2,
    #                                 shuffle=True)
    # dict_to_structured_array(scores2, 'shuffle_score.npy')

    # %% Plotting
    data_dir = ''
    # scores = np.load(data_dir + '../true_scores.npy', allow_pickle=True)[0]
    scores = {name: scores[name] for name in scores.keys()}

    plots = {}
    for key, values in scores.items():
        if values is None:
            continue
        plots[key] = np.mean(values.T[np.eye(4).astype(bool)].T, axis=2)
    fig, axs = plot_all_scores(plots, conds, {n: i for n, i in zip(names, idxs)}, colors, "Word Decoding")

    # %% Time Sliding decoding significance

    shuffle_score = np.load(data_dir + '../shuffle_score_2009.npy', allow_pickle=True)[0]
    shuffle_score = {name: shuffle_score[name] for name in shuffle_score.dtype.names}
    signif = {}
    for cond, score in scores.items():
        true = np.mean(score.T[np.eye(4).astype(bool)].T, axis=2)
        shuffle = np.mean(shuffle_score[cond].T[np.eye(4).astype(bool)].T, axis=2)
        signif[cond] = time_perm_cluster(true.T, shuffle.T, 0.005, stat_func=lambda x, y, axis: np.mean(x, axis=axis))[0]

    # %% Plot significance
    for cond, ax in zip(conds, axs):
        bars = []
        if isinstance(cond, list):
            cond = "-".join(cond)
        for i, idx in enumerate(idxs):
            name = "-".join([names[i], cond])
            if name.endswith('resp'):
                times = (-1, 1)
            else:
                times = (-0.5, 1.5)
            shuffle = np.mean(shuffle_score[name].T[np.eye(4).astype(bool)].T, axis=2)
            # smooth the shuffle using a window
            window = np.lib.stride_tricks.sliding_window_view(shuffle, 20, 0)
            shuffle = np.mean(window, axis=-1)
            plot_dist_bound(shuffle, 'std', 'both', times, 0, ax=ax, color=colors[i], alpha=0.3)
            bars.append(signif[name])
        plot_horizontal_bars(ax, bars, 0.025, 'below')

    for ax in fig.axes:
        ax.axhline(0.25, color='k', linestyle='--')