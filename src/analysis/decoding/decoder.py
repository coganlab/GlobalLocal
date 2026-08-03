"""The `Decoder` estimator and its cross-validated confusion-matrix methods."""

import numpy as np
from scipy.stats import norm, t
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     StratifiedShuffleSplit)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator
from ieeg.decoding.models import PcaLdaClassification, PcaEstimateDecoder
from ieeg.calc.oversample import MinimumNaNSplit
from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict, windower

from .data_prep import flatten_features, mixup2, sample_fold

class Decoder(PcaEstimateDecoder, MinimumNaNSplit):
    '''
    Decoder class that inherits from two parents: PcaEstimateDecoder and MinimumNaNSplit.
    PcaEstimateDecoder (from ieeg.decoding.models) sets up self.model as a sklearn Pipeline([pca, clf]).
    decoder.fit(X, y) calls self.model.fit(X, y), which fits pca -> clf in sequence.
    decoder.predict(X) calls self.model.predict
    MinimumNaNSplit (from ieeg.calc.oversample) provides self.split(x, y). This is a stratified K-fold that's NaN-aware.
    It ensures each test fold has at least N non-NaN trials per class, so there's no folds where one class is entirely missing on some channel.
    
    '''
    
    def __init__(self, categories: dict,
                 explained_variance: float = 0.8,
                 n_splits: int = 5, n_repeats: int = 10,
                 oversample: bool = True, max_features: int = float("inf"), 
                 clf: BaseEstimator = LinearDiscriminantAnalysis(),
                 clf_params: dict = None,
                 random_state: int = None):
        # two-stage constructor b/c inherits from two classes
        PcaEstimateDecoder.__init__(self, 
                                    explained_variance=explained_variance,
                                    clf=clf,
                                    clf_params=clf_params)
        
        MinimumNaNSplit.__init__(self, n_splits, n_repeats)
        if not oversample:
            self.oversample = lambda x, func, axis: x
        self.categories = categories
        self.max_features = max_features
        self.random_state = random_state

    def cv_cm_jim(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        '''
        This produces a cross-validated confusion matrix, such that there is one confusion matrix per (repeat, fold).
        '''
        
        n_cats = len(set(labels)) # Number of classes. 2 for binary encoding.
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats)) # output container: one confusion matrix per (repeat, fold).
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs # obs axs is the trial axis that gets normalized to a positive index so that np.take is unambiguous. 
        idx = [slice(None) for _ in range(x_data.ndim)] # create a list of slice(None, None, None) values, with one for each dim of x_data - so probably 3, resulting in [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(0, obs_axs), labels)): # self.split is from the MinimumNaNSplit class, and is a stratified K-fold that ensures each test fold has at least N non-NaN trials per class. The x_data.swapaxes(0, obs_axs) is a view to satisfy self.split's expectation that trials are on axis 0. f is a flat index 0...n_repeats*n_splits-1, corresponding to all folds across all repeats.
            x_train = np.take(x_data, train_idx, obs_axs) # np.take selects along whatever axis trials really live on. x_train keeps the original shape just fewer trials
            x_test = np.take(x_data, test_idx, obs_axs) 
            
            y_train = labels[train_idx]
            mixup2(arr=x_train, labels=y_train, obs_axs=obs_axs, alpha=1., seed=None) # replace trials with NaN timepoints with a random combination of two non-NaN trials: these trials can be from the same class (50% or higher chance for each trial) or a different class.
            y_test = labels[test_idx]
            # for i in set(labels):
            #     # fill in train data nans with random combinations of existing train data trials (mixup)
            #     idx[obs_axs] = y_train == i
            #     x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)], axis=obs_axs, func=mixup)

            # fill in test data nans with noise from distribution
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan)) # NaNs are filled with independent and identically distributed noise, which is intentionally non-informative, approximately matching the scaled-feature distribution after the scaler maps everything to unit variance. This is so test imputation doesn't leak class info.

            # feature selection
            train_in = flatten_features(x_train, obs_axs) # flatten features collapses everything except the trial axis into one feature dimension so end up with (n_trials, n_channels * n_timepoints)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features: # if the resulting feature count exceeds self.max_features, pick a random subset of features, with the same indices for train and test so the slicing is consistent. Usually will not be triggered since  default max_features = inf.
                tidx = np.random.choice(train_in.shape[1], self.max_features, replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]

            # fit model and score results
            self.fit(train_in, y_train) # fits scaler -> pca -> clf. Called fresh every iteration, so decoder.model.named_steps['pca'] is overwritten on every fold.
            pred = self.predict(test_in) # transforms with the same scaler/pca, then predicts
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1) # sum across folds within each repeat, resulting in matk of shape (n_repeats, n_cats, n_cats)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True) # row-normalize
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True) # col-normalize
        elif normalize == 'all':
            divisor = self.n_repeats # average over repeats
        else:
            divisor = 1
        return matk / divisor # returns accuracy per repeat of shape (n_repeats, n_cats, n_cats), with the n_repeats distribution being used later for significance testing
    
    # untested 11/30
    # def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray,
    #             normalize: str = None, obs_axs: int = -2, time_axs: int = -1, n_jobs: int = 1,
    #             window: int = None, step_size: int = 1,
    #                 shuffle: bool = False, oversample: bool = True) -> np.ndarray:
    #     """Cross-validated confusion matrix with windowing and optional shuffling. REPLACING THIS, DEPRECATED"""
    #     n_cats = len(set(labels))
    #     time_axs_positive = time_axs % x_data.ndim

    #     out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)

    #     if window is not None:
    #         # Include the step size in the windowed output shape
    #         steps = (x_data.shape[time_axs_positive] - window) // step_size + 1
    #         out_shape = (steps,) + out_shape
                
    #     mats = np.zeros(out_shape, dtype=np.uint8)
    #     data = x_data.swapaxes(0, obs_axs)

    #     if shuffle:
    #         # shuffled label pool
    #         label_stack = []
    #         for i in range(self.n_repeats):
    #             label_stack.append(labels.copy())
    #             self.shuffle_labels(data, label_stack[-1], 0)

    #         # build the test/train indices from the shuffled labels for each
    #         # repetition, then chain together the repetitions
    #         # splits = (train, test)

    #         print("Shuffle validation:")
    #         for i, labels in enumerate(label_stack):
    #             # Compare with the first repetition to ensure variety in shuffles
    #             if i > 0:
    #                 diff = np.sum(label_stack[0] != labels)

    #         idxs = ((self.split(data, l), l) for l in label_stack)
    #         idxs = ((itertools.islice(s, self.n_splits),
    #                  itertools.repeat(l, self.n_splits))
    #                 for s, l in idxs)
    #         splits, label = zip(*idxs)
    #         splits = itertools.chain.from_iterable(splits)
    #         label = itertools.chain.from_iterable(label)
    #         idxs = zip(splits, label)

    #     else:
    #         idxs = ((splits, labels) for splits in self.split(data, labels))
    
    #     # 11/1 below is aaron's code for windowing. 
    #     def proc(train_idx, test_idx, l):
    #         x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
    #         print(f"x_stacked shape: {x_stacked.shape}")

    #         # Use the updated windower function with step_size
    #         windowed = windower(x_stacked, window, axis=time_axs, step_size=step_size)
    #         print(f"windowed shape: {windowed.shape}")

    #         out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
    #         for i, x_window in enumerate(windowed):
    #             x_flat = x_window.reshape(x_window.shape[0], -1)
    #             x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
    #             out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
    #         return out

    #     # # loop over folds and repetitions
    #     if n_jobs == 1:
    #         idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
    #         results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
    #     else:
    #         results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
    #             delayed(proc)(train_idx, test_idx, l)
    #             for (train_idx, test_idx), l in idxs)

    #     # # Collect the results
    #     for i, result in enumerate(results):
    #         rep, fold = divmod(i, self.n_splits)
    #         mats[:, rep, fold] = result

    #     # normalize, sum the folds
    #     mats = np.sum(mats, axis=-3)
    #     if normalize == 'true':
    #         divisor = np.sum(mats, axis=-1, keepdims=True)
    #     elif normalize == 'pred':
    #         divisor = np.sum(mats, axis=-2, keepdims=True)
    #     elif normalize == 'all':
    #         divisor = self.n_repeats
    #     else:
    #         divisor = 1
    #     return mats / divisor

    def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray, normalize: str = None,
        obs_axs : int = -2, time_axs: int = -1, window: int = None, step_size: int = 1,
        shuffle: bool = False, oversample: bool = True, folds_as_samples: bool = False,
        labels_test: np.ndarray = None, stratify_labels: np.ndarray = None,
        frac_train: float = None, temporal_generalization: bool = False) -> np.ndarray:
        """
        Cross-validated confusion matrix with windowing, optional shuffling, and an option to treat folds as independent samples.

        This function performs cross-validated decoding with optional sliding windows over time.
        It can shuffle labels (for permutation testing) and handles missing data via mixup.

        Cross-decoding (`labels_test`)
        ------------------------------
        Pass `labels_test` to train on one labelling of the trials and score against
        a DIFFERENT one — e.g. fit on congruency (stability) and test whether that
        decision axis predicts switchType (flexibility). Both vectors index the same
        trials, so the CV split still guarantees train and test trials are disjoint;
        that is the circularity guard, no extra machinery needed. `labels_test=None`
        (default) keeps the ordinary same-label behaviour.

        When cross-decoding, pass `stratify_labels` = the joint condition cell
        (e.g. congruency x switchType) so each fold stays balanced on the label you
        SCORE, not just the one you train on. Stratifying on the train labels alone
        can leave a test fold with a lopsided (or absent) test class.

        `shuffle=True` permutes the TRAIN labels and refits, so the null carries the
        variance of the whole estimation pipeline (scaler -> PCA -> LDA, mixup, fold
        structure). For a cross-decode this is exactly the right null: "does an axis
        trained on real congruency labels predict switchType better than an axis
        trained on scrambled ones?"

        Train/test proportion (`frac_train`)
        ------------------------------------
        By default the split is `StratifiedKFold(n_splits)`, i.e. a fixed
        (n_splits-1)/n_splits train fraction. Pass `frac_train` (0 < f < 1) to use
        `StratifiedShuffleSplit` instead and set the proportion directly; `n_splits`
        then means "number of random resamples per repeat" rather than "folds", and
        the resamples are no longer a partition of the data.

        Temporal generalization (`temporal_generalization`)
        --------------------------------------------------
        With `True`, fit at each train window and predict at EVERY test window,
        returning a (n_train_windows, n_test_windows, ...) matrix instead of the
        diagonal only. Cost scales with n_windows^2 predictions (but only n_windows
        fits), so start with a coarse `step_size`.

        Returns
        -------
        Without windowing              : (n_repeats, n_cats, n_cats)
        With `window`                  : (n_windows, n_samples, n_cats, n_cats)
        With `temporal_generalization` : (n_train_windows, n_test_windows, n_samples, n_cats, n_cats)

        where `n_samples` is `n_repeats` (default) or `n_repeats * n_splits`
        (`folds_as_samples=True`).
        """

        # Step 1: Setup basic parameters
        # Count unique classes. With a separate test labelling both vectors must
        # use the same class coding, so the count comes from their union.
        if labels_test is not None:
            labels_test = np.asarray(labels_test)
            if len(labels_test) != len(labels):
                raise ValueError(
                    f"labels_test has {len(labels_test)} entries but labels has "
                    f"{len(labels)}; both must label the SAME trials")
            n_cats = len(set(labels) | set(labels_test))
        else:
            n_cats = len(set(labels))
        cm_labels = np.arange(n_cats)   # pin the CM axes so an absent class can't reshape it

        # What the fold split is stratified on. Defaults to the train labels
        # (the historical behaviour); for a cross-decode pass the joint cell.
        strat = labels if stratify_labels is None else np.asarray(stratify_labels)
        if len(strat) != len(labels):
            raise ValueError(
                f"stratify_labels has {len(strat)} entries but labels has {len(labels)}")

        # Convert negative time axis to positive (e.g., -1 becomes 3 for 4D array)
        time_axs_positive = time_axs % x_data.ndim

        # Step 2: Determine output shape based on windowing
        # Base shape without windows: (repeats, splits, classes, classes)
        base_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)

        if window is not None:
            # Calculate how many windows fit with the given step size
            # E.g., 256 samples, window=64, step=32 → (256-64)/32 + 1 = 7 windows
            steps = (x_data.shape[time_axs_positive] - window) // step_size + 1

            if temporal_generalization:
                # (repeats, splits, train_windows, test_windows, classes, classes)
                out_shape = (self.n_repeats, self.n_splits, steps, steps, n_cats, n_cats)
            else:
                # Add windows dimension: (repeats, splits, windows, classes, classes)
                out_shape = (self.n_repeats, self.n_splits, steps, n_cats, n_cats)
        else:
            if temporal_generalization:
                raise ValueError("temporal_generalization=True requires `window`")
            # No windowing - use base shape
            out_shape = base_shape

        # Step 3: Initialize output array and prepare data
        # Create array to store all confusion matrices
        mats = np.zeros(out_shape, dtype=np.float32)

        # Move observations/trials to first axis for easier indexing
        # E.g., (trials, channels, freqs, time) stays same if obs_axs=0
        data = x_data.swapaxes(0, obs_axs)

        # Initialize random state for reproducibility
        rng = np.random.RandomState(seed=self.random_state if hasattr(self, 'random_state') else None)

        if frac_train is not None and not 0 < frac_train < 1:
            raise ValueError(f"frac_train must be in (0, 1); got {frac_train}")

        # Step 4: Main cross-validation loop
        for i in range(self.n_repeats):
            # Each repeat gets a different random split of the data. StratifiedKFold
            # partitions the trials; StratifiedShuffleSplit instead draws n_splits
            # independent train/test resamples at the requested proportion.
            if frac_train is None:
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                           random_state=rng)
            else:
                splitter = StratifiedShuffleSplit(n_splits=self.n_splits,
                                                  train_size=frac_train,
                                                  random_state=rng)

            # Iterate through each fold
            for f, (train_idx, test_idx) in enumerate(splitter.split(data, strat)):
                # Extract train/test data for this fold
                x_train = data[train_idx]
                y_train = labels[train_idx].copy()  # Copy to avoid modifying original
                x_test = data[test_idx]
                # Cross-decoding scores against the OTHER labelling of the same trials
                y_test = (labels if labels_test is None else labels_test)[test_idx]

                # Step 5: Optional label shuffling (for permutation testing)
                if shuffle:
                    # Randomly permute training labels to break label-data relationship.
                    # The model is refit below, so the null reflects the full pipeline.
                    rng.shuffle(y_train)

                # Step 6: Window and predict
                # This returns confusion matrix(es) for this fold
                cm_windowed = self._window_and_predict_minimal(
                    x_train, y_train, x_test, y_test,
                    window, step_size, time_axs_positive, oversample,
                    temporal_generalization=temporal_generalization,
                    cm_labels=cm_labels
                )

                # Step 7: Store results
                if window is not None:
                    # cm_windowed shape: (n_windows, n_cats, n_cats), or
                    # (n_train_windows, n_test_windows, n_cats, n_cats) for tempgen
                    mats[i, f, :] = cm_windowed
                else:
                    # cm_windowed shape: (n_cats, n_cats)
                    mats[i, f] = cm_windowed

        # Step 8: Reorganize dimensions for output
        # n_win_axes is how many window axes sit between (repeats, splits) and the
        # two class axes: 0 without windowing, 1 normally, 2 for temporal generalization.
        n_win_axes = 0 if window is None else (2 if temporal_generalization else 1)
        win_axes = tuple(range(2, 2 + n_win_axes))

        if folds_as_samples:
            # (n_repeats, n_splits, *win, n_cats, n_cats)
            #   -> (*win, n_repeats, n_splits, n_cats, n_cats)
            mats = np.transpose(mats, win_axes + (0, 1, mats.ndim - 2, mats.ndim - 1))
            # then collapse repeats x splits into one 'samples' dimension
            win_shape = mats.shape[:n_win_axes]
            n_repeats, n_splits = mats.shape[n_win_axes:n_win_axes + 2]
            mats = mats.reshape(win_shape + (n_repeats * n_splits, n_cats, n_cats))
            # final: (*win, n_repeats * n_splits, n_cats, n_cats)
        else:
            # sum over splits, then bring the window axes to the front
            mats = np.sum(mats, axis=1)   # -> (n_repeats, *win, n_cats, n_cats)
            win_axes_after_sum = tuple(range(1, 1 + n_win_axes))
            mats = np.transpose(
                mats, win_axes_after_sum + (0, mats.ndim - 2, mats.ndim - 1))
            # final: (*win, n_repeats, n_cats, n_cats)

        # Step 9: Apply normalization
        if normalize == 'true':
            # Normalize by row sums (true class totals)
            divisor = np.sum(mats, axis=-1, keepdims=True)
        elif normalize == 'pred':
            # Normalize by column sums (predicted class totals)
            divisor = np.sum(mats, axis=-2, keepdims=True)
        elif normalize == 'all':
            # Normalize by total sum
            divisor = np.sum(mats, axis=(-2,-1), keepdims=True)
        else:
            # No normalization
            divisor = 1
        
        # Step 10: Safe division and return
        with np.errstate(divide='ignore', invalid='ignore'):
            result = mats / divisor
            # Replace any inf/nan from division by zero with 0
            result[~np.isfinite(result)] = 0
        
        return result

    def _window_and_predict_minimal(self, x_train, y_train, x_test, y_test,
                                window, step_size, time_axs, oversample,
                                temporal_generalization=False, cm_labels=None):
        """
        helper function that handles windowing and prediction for a single CV fold

        EXAMPLE FLOW:

        Initial data:
        - x_train: (70, 10, 256) - 70 training trials, some with NaNs
        - x_test: (30, 10, 256) - 30 test trials, some with NaNs
        - window=64, step_size=32

        1. _window_and_predict_minimal combines data:
        x_stacked = (100, 10, 256)

        2. sample_fold is called:
        - Reorders data to put train first, test second
        - Applies mixup to fill training NaNs with smart combinations
        - Fills test NaNs with random noise
        - Returns processed (100, 10, 256) with no NaNs

        3. Windowing applied:
        windowed = (7, 100, 10, 64) - 7 time windows

        4. For each window:
        - Flatten: (100, 10, 64) → (100, 640)
        - Split: train (70, 640), test (30, 640)
        - Decode and get confusion matrix

        5. Return: (7, 2, 2) for binary classification with 7 windows

        With `temporal_generalization=True`, step 4 becomes a double loop: fit once
        per train window, then predict at every test window, returning (7, 7, 2, 2).

        `cm_labels` pins the confusion-matrix axes (e.g. `np.arange(n_cats)`) so a
        fold in which one class never appears still returns a full-size matrix
        instead of a smaller one that would fail to broadcast into the output.
        """
        # step 1: get number of classes from decoder configuration
        n_cats = len(self.categories)
        if cm_labels is None:
            cm_labels = np.arange(n_cats) if n_cats else None
        elif n_cats == 0:
            n_cats = len(cm_labels)

        # step 2: combine combine train and test data for consistent windowing
        # this ensures windows align properly across train/test boundary
        x_stacked = np.concatenate((x_train, x_test), axis=0)

        # step 3: create index arrays for sample_fold
        # these tell sample_fold which samples are train vs test
        train_idx = np.arange(len(y_train)) # [0,1,2,...,n_train-1]
        test_idx = np.arange(len(y_train), len(y_train) + len(y_test)) # [n_train, ..., n_total-1]
        
        # Step 4: Use sample_fold for preprocessing. 
        # This handles:
        # - Mixup augmentation for training NaNs
        # - Random noise filling for test NaNs
        # - Proper data splitting
        x_processed, y_train_proc, y_test_proc = sample_fold(
            train_idx, test_idx, x_stacked, 
            np.concatenate([y_train, y_test]), # combine labels for sample_fold
            axis=0, # trials are on axis 0
            oversample=oversample
        )
        
        # Step 5: Apply sliding window if specified
        if window is not None:
            # windower creates overlapping windows
            # E.g., (100, 10, 256) -> (7, 100, 10, 64)
            # where 7 windows of size 64 with step size 32
            windowed = windower(x_processed, window, axis=time_axs, step_size=step_size)
        else:
            # no windowing - add fake window dimension for consistency
            # (100, 10, 256) -> (1, 100, 10, 256)
            windowed = x_processed[np.newaxis, ...]
        
        # Step 6: Flatten each window and split it back into train / test.
        # Done up front so temporal generalization can pair any train window with
        # any test window without re-windowing.
        # E.g., (100, 10, 64) -> (100, 640): one feature vector per trial.
        n_train = len(y_train_proc)
        per_window = []
        for x_window in windowed:
            x_flat = x_window.reshape(x_window.shape[0], -1)
            # We know the first n_train samples are training (sample_fold ordered them)
            per_window.append(np.split(x_flat, [n_train], axis=0))

        # Step 7: Fit and predict
        if temporal_generalization:
            # Fit ONCE per train window, then predict at every test window.
            # n_windows fits, n_windows^2 predictions.
            n_win = len(per_window)
            out_cm = np.zeros((n_win, n_win, n_cats, n_cats))
            for i, (x_train_w, _) in enumerate(per_window):
                self.fit(x_train_w, y_train_proc)
                for j, (_, x_test_w) in enumerate(per_window):
                    preds = self.predict(x_test_w)
                    out_cm[i, j] = confusion_matrix(y_test_proc, preds, labels=cm_labels)
            return out_cm   # Shape: (n_train_windows, n_test_windows, n_cats, n_cats)

        out_cm = []  # list to collect confusion matrices
        for x_train_w, x_test_w in per_window:
            self.fit(x_train_w, y_train_proc)   # train on this window's features
            preds = self.predict(x_test_w)      # predict test labels
            # Compares true test labels with predictions. With a separate test
            # labelling (cross-decoding) y_test_proc is the OTHER label vector.
            out_cm.append(confusion_matrix(y_test_proc, preds, labels=cm_labels))

        # Step 8: Format output
        # If only one window, remove the window dimension
        # otherwise, return array of confusion matrices
        if len(out_cm) == 1:
            return np.squeeze(np.array(out_cm)) # remove window dimension
        else:
            return np.array(out_cm) # Shape: (n_windows, n_cats, n_cats)
    
    def fit_predict(self, x_train, x_test, y_train, y_test):
        # fit model and score results
        self.model.fit(x_train, y_train)
        pred = self.model.predict(x_test)
        return confusion_matrix(y_test, pred)
    
    def cv_cm_return_scores(self, x_data: np.ndarray, labels: np.ndarray,
                            normalize: str = None, obs_axs: int = -2):
        '''
        trying to get the scores manually from cv cm but i realize that in decoders.py, PcaLdaClassification already has a get_scores function. Try get_scores with shuffle=True to get fake, permuted scores.
        '''
        # Get the confusion matrix by calling `cv_cm`
        cm = self.cv_cm_jim(x_data, labels, normalize, obs_axs)

        # Average the confusion matrices across the repetitions
        cm_avg = np.mean(cm, axis=0)  # Now cm_avg will be of shape (2, 2)

        # Calculate the individual decoding scores (Accuracy, Precision, etc.)
        scores = self.calculate_scores(cm_avg)

        return cm_avg, scores

    def calculate_scores(self, cm):
        """
        Calculate the individual decoding scores from the confusion matrix. 10/27 Ugh Aaron already does this directly in the PcaLdaClassification class... 

        Parameters:
        - cm: The confusion matrix (averaged over folds).

        Returns:
        - scores: A dictionary containing the scores (accuracy, precision, recall, f1, d-prime) for each class.
        """
        scores = {}
        tp = np.diag(cm)  # True Positives
        fp = np.sum(cm, axis=0) - tp  # False Positives
        fn = np.sum(cm, axis=1) - tp  # False Negatives
        tn = np.sum(cm) - (fp + fn + tp)  # True Negatives

        # Calculate accuracy, precision, recall, and f1 score
        accuracy = np.sum(tp) / np.sum(cm)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Store the basic scores
        scores['accuracy'] = accuracy
        scores['precision'] = precision
        scores['recall'] = recall
        scores['f1'] = f1

        # Calculate hit rate and false alarm rate
        hit_rate = recall  # Hit rate is the same as recall (TP / (TP + FN))
        false_alarm_rate = fp / (fp + tn + 1e-8)  # False alarm rate (FP / (FP + TN))

        # Ensure hit_rate and false_alarm_rate are in valid range [0, 1] for Z-transform
        hit_rate = np.clip(hit_rate, 1e-8, 1 - 1e-8)
        false_alarm_rate = np.clip(false_alarm_rate, 1e-8, 1 - 1e-8)

        # Z-transform to calculate d-prime
        z_hit_rate = norm.ppf(hit_rate)  # Z-transform for hit rate
        z_false_alarm_rate = norm.ppf(false_alarm_rate)  # Z-transform for false alarm rate

        # Calculate d-prime
        d_prime = z_hit_rate - z_false_alarm_rate

        # Store d-prime in the scores dictionary
        scores['d_prime'] = d_prime

        return scores
