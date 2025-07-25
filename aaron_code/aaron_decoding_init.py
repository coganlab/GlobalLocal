from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# from grouping import GroupData
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from ieeg.calc.oversample import MinimumNaNSplit, mixup
from numpy.lib.stride_tricks import as_strided
from ieeg.calc import stats
import numpy as np
import matplotlib.pyplot as plt
from ieeg.viz.ensemble import plot_dist
from joblib import Parallel, delayed
import itertools
from tqdm import tqdm


class Decoder(PcaLdaClassification, MinimumNaNSplit):


    # jim decoding init that has oversample in it
    # def __init__(self, categories: dict, *args, n_splits: int = 5, n_repeats: int = 10,
    #              oversample: bool = True, max_features: int = float("inf"), **kwargs):
    #     PcaLdaClassification.__init__(self, *args, **kwargs)
    #     MinimumNaNSplit.__init__(self, n_splits, n_repeats)
    #     if not oversample:
    #         self.oversample = lambda x, func, axis: x
    #     self.categories = categories
    #     self.max_features = max_features

    def __init__(self, categories: dict, *args,
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 min_samples: int = 1,
                 which: str = 'test',
                 **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats,
                                 None, min_samples, which)
        self.categories = categories

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2, n_jobs: int = 1,
              average_repetitions: bool = True, window: int = None,
              shuffle: bool = False, oversample: bool = True) -> np.ndarray:
        """Cross-validated confusion matrix"""
        n_cats = len(set(labels))
        out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)
        if window is not None:
            out_shape = (x_data.shape[-1] - window + 1,) + out_shape
        mats = np.zeros(out_shape, dtype=np.uint8)
        data = x_data.swapaxes(0, obs_axs)

        if shuffle:
            # shuffled label pool
            label_stack = []
            for i in range(self.n_repeats):
                label_stack.append(labels.copy())
                self.shuffle_labels(data, label_stack[-1], 0)

            # build the test/train indices from the shuffled labels for each
            # repetition, then chain together the repetitions
            # splits = (train, test)
            idxs = ((self.split(data, l), l) for l in label_stack)
            idxs = ((itertools.islice(s, self.n_splits),
                     itertools.repeat(l, self.n_splits))
                    for s, l in idxs)
            splits, label = zip(*idxs)
            splits = itertools.chain.from_iterable(splits)
            label = itertools.chain.from_iterable(label)
            idxs = zip(splits, label)

        else:
            idxs = ((splits, labels) for splits in self.split(data, labels))

        def proc(train_idx, test_idx, l):
            x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
            windowed = windower(x_stacked, window, axis=-1)
            out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
            for i, x_window in enumerate(windowed):
                x_flat = x_window.reshape(x_window.shape[0], -1)
                x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
                out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
            return out

        # loop over folds and repetitions
        if n_jobs == 1:
            idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
            results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
        else:
            results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
                delayed(proc)(train_idx, test_idx, l)
                for (train_idx, test_idx), l in idxs)

        # Collect the results
        for i, result in enumerate(results):
            rep, fold = divmod(i, self.n_splits)
            mats[:, rep, fold] = result

        # average the repetitions
        if average_repetitions:
            mats = np.mean(mats, axis=1)

        # normalize, sum the folds
        mats = np.sum(mats, axis=-3)
        if normalize == 'true':
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return mats / divisor

    def fit_predict(self, x_train, x_test, y_train, y_test):

        # fit model and score results
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)
        return confusion_matrix(y_test, pred)


def sample_fold(train_idx: np.ndarray, test_idx: np.ndarray,
                x_data: np.ndarray, labels: np.ndarray,
                axis: int, oversample: bool = True):

    # make first and only copy of x_data
    idx_stacked = np.concatenate((train_idx, test_idx))
    x_stacked = np.take(x_data, idx_stacked, axis)
    y_stacked = labels[idx_stacked]

    # define train and test as views of x_stacked
    sep = train_idx.shape[0]
    y_train, y_test = np.split(y_stacked, [sep])

    if not oversample:
        return x_stacked, y_train, y_test

    x_train, x_test = np.split(x_stacked, [sep], axis=axis)
    idx = [slice(None) for _ in range(x_data.ndim)]
    unique = np.unique(labels)
    for i in unique:
        # fill in train data nans with random combinations of
        # existing train data trials (mixup)
        idx[axis] = y_train == i
        out = x_train[tuple(idx)]
        mixup(out, axis)
        x_train[tuple(idx)] = out

    # fill in test data nans with noise from distribution
    is_nan = np.isnan(x_test)
    x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

    return x_stacked, y_train, y_test


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    out = arr.swapaxes(0, obs_axs)
    return out.reshape(out.shape[0], -1)


def windower(x_data: np.ndarray, window_size: int, axis: int = -1, insert_at: int = 0):
    """Create a sliding window view of the array with the given window size."""
    # Compute the shape and strides for the sliding window view
    shape = list(x_data.shape)
    shape[axis] = x_data.shape[axis] - window_size + 1
    shape.insert(axis, window_size)
    strides = list(x_data.strides)
    strides.insert(axis, x_data.strides[axis])

    # Create the sliding window view
    out = as_strided(x_data, shape=shape, strides=strides)

    # Move the window size dimension to the front
    out = np.moveaxis(out, axis, insert_at)

    return out


def classes_from_labels(labels: np.ndarray, delim: str = '-', which: int = 0,
                        crop: slice = slice(None)) -> tuple[dict, np.ndarray]:
    class_ids = np.array([k.split(delim, )[which][crop] for k in labels])
    classes = {k: i for i, k in enumerate(np.unique(class_ids))}
    return classes, np.array([classes[k] for k in class_ids])


# def extract(sub: GroupData, conds: list[str], idx: list[int] = slice(None), common: int = 5,
#             datatype: str = 'zscore', crop_nan: bool = False) -> LabeledArray:
#     """Extract data from GroupData object"""
#     reduced = sub[:, conds][:, :, :, idx]
#     reduced.array = reduced.array.dropna()
#     # also sorts the trials by nan or not
#     reduced = reduced.nan_common_denom(True, common, crop_nan)
#     comb = reduced.combine(('epoch', 'trial'))[datatype]
#     return (comb.array.dropna()).combine((0, 2))


def scale(X, xmax: float, xmin: float):
    return (X - xmin) / (xmax - xmin)


def flatten_list(nested_list: list[list[str] | str]) -> list[str]:
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list


def concatenate_conditions(data, conditions, axis=1):
    """Concatenate data for all conditions"""
    concatenated_data = data[:, conditions[0]]
    for condition in conditions[1:]:
        concatenated_data = concatenated_data.concatenate(data[:, condition], axis=axis)
    return concatenated_data


def decode_and_score(decoder, data, labels, scorer='acc', **decoder_kwargs):
    """Perform decoding and scoring"""
    mats = decoder.cv_cm(data.__array__(), labels, **decoder_kwargs)
    if scorer == 'acc':
        score = np.mean(mats.T[np.eye(len(decoder.categories)).astype(bool)].T, axis=-1)
    else:
        raise NotImplementedError("Only accuracy is implemented")
    return score


def get_scores(subjects, decoder, idxs: list[list[int]], conds: list[str],
               names: list[str], weights: list[list[int]] = None, **decoder_kwargs) -> dict[str, np.ndarray]:
    for i, idx in enumerate(idxs):
        all_conds = flatten_list(conds)
        x_data = extract(subjects, all_conds, idx, decoder.n_splits, 'zscore',
                         False)

        for cond in conds:
            if isinstance(cond, list):
                X = concatenate_conditions(x_data, cond)
                cond = "-".join(cond)
            else:
                X = x_data[:, cond]

            cats, labels = classes_from_labels(X.labels[1], crop=slice(0, 4))

            # Decoding
            if weights is None:
                score = decoder.cv_cm(X.__array__(), labels, **decoder_kwargs)
                yield "-".join([names[i], cond]), score
            else:
                for j, weight in enumerate(weights):
                    score = decoder.cv_cm(X.__array__() * weight[X.labels[0], None, None].__array__(), labels, **decoder_kwargs)
                    yield "-".join([names[i], cond, str(j)]), score


def plot_all_scores(all_scores: dict[str, np.ndarray],
                    conds: list[str], idxs: dict[str, list[int]],
                    colors: list[list[float]], suptitle: str = None,
                    fig: plt.Figure = None, axs: plt.Axes = None,
                    **plot_kwargs) -> tuple[plt.Figure, plt.Axes]:
    names = list(idxs.keys())
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, len(conds))
    elif axs is None:
        axs = fig.get_axes()
    if len(conds) == 1:
        axs = [axs]
    for color, name, idx in zip(colors, names, idxs.values()):
        for cond, ax in zip(conds, axs):
            if isinstance(cond, list):
                cond = "-".join(cond)
            ax.set_title(cond)
            if cond == 'resp':
                times = (-0.9, 0.9)
                ax.set_xlabel("Time from response (s)")
            else:
                times = (-0.4, 1.4)
                if 'aud' in cond:
                    ax.set_xlabel("Time from stim (s)")
                elif 'go' in cond:
                    ax.set_xlabel("Time from go (s)")
                else:
                    raise ValueError("Condition not recognized")
            pl_sc = np.reshape(all_scores["-".join([name, cond])],
                               (all_scores["-".join([name, cond])].shape[0],
                                -1)).T
            plot_dist(pl_sc, mode='std', times=times,
                      color=color, label=name, ax=ax,
                      **plot_kwargs)
            if name is names[-1]:
                ax.legend()
                ax.set_title(cond)
                ax.set_ylim(0.1, 0.7)

    axs[0].set_ylabel("Accuracy (%)")
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axs


def plot_dist_bound(data: np.ndarray, mode: str = 'sem', which: str = 'both',
                    times: tuple[float, float] = None, axis: int = None,
                    color: list[list[float]] = None,
                    ax: plt.Axes = None, **plot_kwargs) -> plt.Axes:
    assert which in ['both', 'upper', 'lower']
    mean, std = stats.dist(data, axis=axis, mode=mode)
    if times is None:
        tscale = range(mean.shape[0])
    else:
        tscale = np.linspace(times[0], times[1], len(mean))
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if color is None:
        color = ax.get_color()
    if which == 'upper':
        ax.plot(tscale, mean + std, color=color, **plot_kwargs)
    elif which == 'lower':
        ax.plot(tscale, mean - std, color=color, **plot_kwargs)
    elif which == 'both':
        ax.fill_between(tscale, mean - std, mean + std, color=color, **plot_kwargs)
    return ax
