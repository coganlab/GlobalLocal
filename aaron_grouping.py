import os
import mne
import numpy as np
from ieeg import PathLike, Doubles
from ieeg.io import get_data
from ieeg.viz.mri import plot_on_average
from ieeg.viz.ensemble import plot_dist
from ieeg.calc.mat import LabeledArray, combine
from collections.abc import Sequence
from analysis.utils.mat_load import load_dict
from ieeg.viz.mri import subject_to_info, gen_labels, get_sub, pick_no_wm
import matplotlib.pyplot as plt

import nimfa
from scipy import sparse


mne.set_log_level("ERROR")


class GroupData:
    array = LabeledArray([])
    signif = LabeledArray([])

    @classmethod
    def from_intermediates(cls, task: str, root: PathLike,
                           conds: dict[str, Doubles] = None,
                           folder: str = 'stats', **kwargs):
        layout = get_data(task, root=root)
        conds = cls._set_conditions(conds)
        sig = load_dict(layout, conds, "significance", True, folder)
        sig = combine(sig, (0, 2))
        pwr = load_dict(layout, conds, "power", False, folder)
        zscore = load_dict(layout, conds, "zscore", False, folder)
        data = combine(dict(power=pwr, zscore=zscore), (1, 4))
        # subjects = tuple(data['power'].keys())
        pvals = load_dict(layout, conds, "pval", True, folder)
        if pvals:
            kwargs['pvals'] = combine(pvals, (0, 2))

        out = cls(data, sig, **kwargs)
        # out.subjects = subjects
        out.task = task
        out._root = root
        return out

    def __init__(self, data: dict | LabeledArray,
                 mask: dict[str, np.ndarray] | LabeledArray = None,
                 categories: Sequence[str] = ('dtype', 'epoch', 'stim',
                                              'channel', 'trial', 'time'),
                 pvals: dict[str, np.ndarray] | LabeledArray = None,
                 fdr: bool = False, pval: float = 0.05,
                 wide: bool = False, subjects_dir: PathLike = None
                 , per_subject: bool = False):
        self._set_data(data, 'array')
        self._categories = categories
        self.subjects_dir = subjects_dir
        if mask is not None:
            self._set_data(mask, 'signif')
            if not ((self.signif == 0) | (self.signif == 1)).all():
                self.p_vals = self.signif.copy()
                if 'epoch' in categories:
                    for i, arr in enumerate(self.signif):
                        self.signif[i] = self.correction(arr, fdr, pval, per_subject)
                else:
                    self.signif = self.correction(self.signif, fdr, pval, per_subject)
            elif pvals:
                self._set_data(pvals, 'p_vals')
            keys = self.signif.labels
            if all(cond in keys[0] for cond in
                   ["aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "resp"]):

                self.AUD, self.SM, self.PROD, self.sig_chans = group_elecs(
                    self.signif, keys[1], keys[0], wide=wide)
            else:
                self.sig_chans = self._find_sig_chans(self.signif)

    def correction(self, p_vals, fdr: bool, thresh: float,
                   per_subject: bool = False, verbose=False):

        if per_subject:
            sig = np.zeros_like(p_vals, dtype=bool)
            for sub in self.subjects:
                idx = self.keys['channel'].astype('U5') == sub
                if fdr:
                    temp = p_vals[idx]
                    sig[idx] = mne.stats.fdr_correction(temp, thresh)[0]
                    if verbose:
                        passed = np.logical_and(sig[:, 0], idx)
                        new_thsh = np.max(p_vals[(passed,)]) if np.any(passed) else 0
                        print(f"FDR correction applied, new threshold: {new_thsh}")
                else:
                    sig[idx] = p_vals[idx] < thresh
        elif fdr:
            sig = mne.stats.fdr_correction(p_vals, thresh)[0]
            if verbose:
                new_thresh = np.max(p_vals[(sig[:, 0],)])
                print(f"FDR correction applied, new threshold: {new_thresh}")
        else:
            sig = p_vals < thresh

        return sig

    def _set_data(self, data: dict | LabeledArray, attr: str):
        if isinstance(data, dict):
            setattr(self, attr, LabeledArray.from_dict(data))
        elif isinstance(data, LabeledArray):
            setattr(self, attr, data)
        else:
            raise TypeError(f"input has to be dict or LabeledArray, not "
                            f"{type(data)}")

    @property
    def shape(self):
        return self.array.shape

    @property
    def keys(self):
        keys = self.array.labels
        return {self._categories[i]: k for i, k in enumerate(keys)}

    @property
    def subjects(self):
        return set(f"{ch[:5]}" for ch in self.keys['channel'])

    @property
    def grey_matter(self):
        if not hasattr(self, 'atlas'):
            self.atlas = ".a2009s"
        wm = get_grey_matter(self.subjects, self.subjects_dir, self.atlas)
        return {i for i, ch in enumerate(self.keys['channel']) if ch in wm}

    @staticmethod
    def _set_conditions(conditions: dict[str, Doubles]):
        if conditions is None:
            return {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}
        else:
            return conditions

    @staticmethod
    def _find_sig_chans(sig: np.ndarray) -> list[int]:
        return np.where(np.any(np.atleast_2d(sig) == 1, axis=1))[0].tolist()

    def combine(self, levels: tuple[str, str]):
        assert all(lev in self._categories for lev in levels), "Invalid level"
        lev_nums = tuple(self._categories.index(lev) for lev in levels)
        new_data = self.array.combine(lev_nums)
        new_cats = list(self._categories)
        new_cats.pop(lev_nums[0])
        if not hasattr(self, 'signif'):
            new_sig = None
        elif np.any([np.array_equal(self.keys[levels[0]], l) for l in self.signif.labels])\
                and np.any([np.array_equal(self.keys[levels[1]], l) for l in self.signif.labels]):
            new_sig = self.signif.combine(lev_nums)
        else:
            new_sig = self.signif
        return type(self)(new_data, new_sig, new_cats)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        while len(item) < self.array.ndim:
            item += (slice(None),)

        sig = getattr(self, 'signif', LabeledArray([]))
        if sig.size > 0:
            sig_keys = []
            for i, key in enumerate(item):
                labels = self.array.labels[i]
                if np.any([np.array_equal(labels, l) for l in sig.labels]):
                    if isinstance(key, list):
                        key = tuple(key)
                    sig_keys.append(key)
            if sig_keys:
                sig = sig[tuple(sig_keys)]

        cats = tuple(self._categories[i] for i, key in enumerate(item) if
                   isinstance(key, (Sequence, slice)) and not isinstance(key, str))

        return type(self)(self.array[item], sig, cats)

    def nan_common_denom(self, sort: bool = True, min_trials: int = 0,
                         crop_trials: bool = True, verbose: bool = False):
        """Remove trials with NaNs from all channels"""
        trials_idx = self._categories.index('trial')
        ch_idx = self._categories.index('channel')
        others = [i for i in range(len(self._categories)) if ch_idx != i != trials_idx]
        isn = np.isnan(self.array)
        nan_trials = np.any(isn, axis=tuple(others))

        # Sort the trials by whether they are nan or not
        if sort:
            order = np.argsort(nan_trials, axis=1)
            old_shape = list(order.shape)
            new_shape = [1 if ch_idx != i != trials_idx else old_shape.pop(0)
                         for i in range(len(self._categories))]
            order = np.reshape(order, new_shape)
            data = np.take_along_axis(self.array.__array__(), order, axis=trials_idx)
            data = LabeledArray(data, self.array.labels.copy())
        else:
            data = self.array

        ch_tnum = self.shape[trials_idx] - np.sum(nan_trials, axis=1)
        ch_min = ch_tnum.min()
        if verbose:
            print(f"Lowest trials {ch_min} at "
                  f"{self.keys['channel'][ch_tnum.argmin()]}")

        ntrials = max(ch_min, min_trials)
        if ch_min < min_trials:
            # data = data.take(np.where(ch_tnum >= ntrials)[0], ch_idx)
            ch = np.array(self.keys['channel'])[ch_tnum < ntrials].tolist()
            if verbose:
                print(f"Channels excluded (too few trials): {ch}")

        # data = data.take(np.arange(ntrials), trials_idx)
        idx = [np.arange(ntrials) if i == trials_idx and crop_trials
               else np.arange(s) for i, s in enumerate(self.shape)]
        idx[ch_idx] = np.where([ch_tnum >= ntrials])[1]

        sig = getattr(self, '_significance', None)
        if sig is not None:
            sig_ch_idx = sig.labels.index(data.labels[ch_idx])
            sig = sig.take(idx[ch_idx], sig_ch_idx)

        return type(self)(data[np.ix_(*idx)], sig, self._categories)

    def plot_each(self, axis: tuple[int, ...] | int = None, n_cols: int = 10,
                  n_rows: int = 6, times: tuple[float, float] = None,
                  size: tuple[int, int] = (16, 12), metric: str = 'zscore'
                  ) -> list[plt.Figure]:
        """Plot each channel separately"""
        if axis is None:
            axis = tuple(self._categories.index('channel'))
        elif isinstance(axis, int):
            axis = (axis,)
        if times is None:
            times = (self.array.labels[-1][0], self.array.labels[-1][-1])
        if 'dtype' in self._categories:
            arr = self[metric].array.copy()
        else:
            arr = self.array.copy()

        while len(axis) > 1:
            arr = arr.combine(axis[:2])
            axis = tuple(i - 1 for i in axis[1:])

        per_fig = n_cols * n_rows
        chans = arr.labels[axis[-1]]
        numfigs = int(np.ceil(len(chans) / per_fig))
        figs = []
        indices = np.linspace(times[0], times[-1], arr.shape[-1])
        for i in range(numfigs):
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, frameon=False,
                                    figsize=size)
            for j, ch in enumerate(chans[i * per_fig:(i + 1) * per_fig]):
                ax = axs.flatten()[j]
                idx = tuple(slice(None) if k != axis[-1]
                            else ch for k in range(arr.ndim))
                plot_dist(arr[idx].__array__(), ax=ax, times=times)
                bvect = vect2ind(self.signif[ch].__array__())
                loc = [(indices[b[0]], indices[b[1]]) for b in bvect]
                ax.broken_barh(loc, (-0.3, 0.3), color='black')
                ax.set_title(ch)
            ymax = max(ax.get_ylim()[1] for ax in axs.flatten())
            ymin = min(ax.get_ylim()[0] for ax in axs.flatten())
            for ax in axs.flatten():
                ax.set_ylim(ymin, ymax)

            if i == numfigs - 1:
                while j + 1 < n_cols * n_rows:
                    j += 1
                    ax = axs.flatten()[j]
                    ax.axis("off")

            fig.tight_layout()
            figs.append(fig)
        return figs

    def filter(self, item: str):
        """Filter data by key

        Takes the underlying self.array nested dictionary, finds the first
        level with a key that matches the item, and returns a new SubjectData
        object with the all other keys removed at that level. """

        new_categories = list(self._categories)

        def inner(data, lvl=0):
            if isinstance(data, LabeledArray):
                if item in data.keys():
                    new_categories.pop(lvl)
                    return data[item]
                else:
                    return ...