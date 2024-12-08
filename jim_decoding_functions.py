# copy this back into jim_decoding_functions.py once it's working! 12/6
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from scipy.stats import norm
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
import itertools
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view

# largely stolen from aaron's ieeg plot_decoding.py

def mixup2(arr: np.ndarray, labels: np.ndarray, obs_axs: int, alpha: float = 1.,
          seed: int = None) -> None:
    """Mixup the data using the labels

    Parameters
    ----------
    arr : array
        The data to mixup.
    labels : array
        The labels to use for mixing.
    obs_axs : int
        The axis along which to apply func.
    alpha : float
        The alpha value for the beta distribution.
    seed : int
        The seed for the random number generator.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> mixup2(arr, labels, 0)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [6.03943491, 7.03943491]])
           """
    if arr.ndim > 2:
        arr = arr.swapaxes(obs_axs, -2)
        for i in range(arr.shape[0]):
            mixup2(arr=arr[i], labels=labels, obs_axs=obs_axs, alpha=alpha, seed=seed)
    else:
        if seed is not None:
            np.random.seed(seed)
        if obs_axs == 1:
            arr = arr.T

        n_nan = np.where(np.isnan(arr).any(axis=1))[0]
        n_non_nan = np.where(~np.isnan(arr).any(axis=1))[0]

        for i in n_nan:
            l_class = labels[i]
            possible_choices = np.nonzero(np.logical_and(~np.isnan(arr).any(axis=1), labels == l_class))[0]
            choice1 = np.random.choice(possible_choices)
            choice2 = np.random.choice(n_non_nan)
            l = np.random.beta(alpha, alpha)
            if l < .5:
                l = 1 - l
            arr[i] = l * arr[choice1] + (1 - l) * arr[choice2]

class Decoder(PcaLdaClassification, MinimumNaNSplit):
    def __init__(self, categories: dict, *args, n_splits: int = 5, n_repeats: int = 10,
                 oversample: bool = True, max_features: int = float("inf"), **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats)
        if not oversample:
            self.oversample = lambda x, func, axis: x
        self.categories = categories
        self.max_features = max_features

    def cv_cm_jim(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            
            y_train = labels[train_idx]
            mixup2(arr=x_train, labels=y_train, obs_axs=obs_axs, alpha=1., seed=None)
            y_test = labels[test_idx]
            # for i in set(labels):
            #     # fill in train data nans with random combinations of existing train data trials (mixup)
            #     idx[obs_axs] = y_train == i
            #     x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)], axis=obs_axs, func=mixup)

            # fill in test data nans with noise from distribution
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

            # feature selection
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features, replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]

            # fit model and score results
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return matk / divisor
    
    # untested 11/30
    def cv_cm_jim_window_shuffle(self, x_data: np.ndarray, labels: np.ndarray,
                normalize: str = None, obs_axs: int = -2, time_axs: int = -1, n_jobs: int = 1,
                window: int = None, step_size: int = 1,
                    shuffle: bool = False, oversample: bool = True) -> np.ndarray:
        """Cross-validated confusion matrix with windowing and optional shuffling."""
        n_cats = len(set(labels))
        time_axs_positive = time_axs % x_data.ndim
        out_shape = (self.n_repeats, self.n_splits, n_cats, n_cats)

        if window is not None:
            # Include the step size in the windowed output shape
            steps = (x_data.shape[time_axs_positive] - window) // step_size + 1
            out_shape = (steps,) + out_shape
                
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

            print("Shuffle validation:")
            for i, labels in enumerate(label_stack):
                # Compare with the first repetition to ensure variety in shuffles
                if i > 0:
                    diff = np.sum(label_stack[0] != labels)

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

        # 11/1 below is aaron's code for windowing. 
        def proc(train_idx, test_idx, l):
            x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
            print(f"x_stacked shape: {x_stacked.shape}")

            # Use the updated windower function with step_size
            windowed = windower(x_stacked, window, axis=time_axs, step_size=step_size)
            print(f"windowed shape: {windowed.shape}")

            out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
            for i, x_window in enumerate(windowed):
                x_flat = x_window.reshape(x_window.shape[0], -1)
                x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
                out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
            return out

        # # loop over folds and repetitions
        if n_jobs == 1:
            idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
            results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
        else:
            results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
                delayed(proc)(train_idx, test_idx, l)
                for (train_idx, test_idx), l in idxs)

        # # Collect the results
        for i, result in enumerate(results):
            rep, fold = divmod(i, self.n_splits)
            mats[:, rep, fold] = result

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

    
def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)

def windower(x_data: np.ndarray, window_size: int = None, axis: int = -1,
             step_size: int = 1, insert_at: int = 0):
    if window_size is None:
        return x_data[np.newaxis, ...]  # Add a new axis for compatibility

    axis = axis % x_data.ndim
    data_length = x_data.shape[axis]
    
    # Compute the number of full steps (exclude remainder)
    full_steps = (data_length - window_size) // step_size + 1

    # Create the sliding window view for full windows
    windowed = sliding_window_view(x_data, window_shape=window_size, axis=axis)
    
    # Apply step_size by slicing
    if step_size > 1:
        slicing = [slice(None)] * windowed.ndim
        slicing[axis] = slice(0, None, step_size) # try 0, None, step_size for now, I think this should exclude the remainder..? Keep debugging with the bottom cell.
        windowed = windowed[tuple(slicing)]
    
    # Move the window dimension to the desired location
    if insert_at != axis:
        windowed = np.moveaxis(windowed, axis, insert_at)
    
    return windowed




# this is aaron's windower function. Replace with my windower that accounts for step size.
# def windower(x_data: np.ndarray, window_size: int, axis: int = -1, insert_at: int = 0):
#     """Create a sliding window view of the array with a given window size."""
#     # Compute the shape and strides for the sliding window view
#     shape = list(x_data.shape)
#     shape[axis] = x_data.shape[axis] - window_size + 1
#     shape.insert(axis, window_size)
#     strides = list(x_data.strides)
#     strides.insert(axis, x_data.strides[axis])

#     # Create the sliding window view
#     out = as_strided(x_data, shape=shape, strides=strides)

#     # Move the window size dimension to the front
#     out = np.moveaxis(out, axis, insert_at)

#     return out

# modified by jim 11/23, check aaron_decoding_init.py for original.
def sample_fold(train_idx: np.ndarray, test_idx: np.ndarray,
                x_data: np.ndarray, labels: np.ndarray,
                axis: int, oversample: bool = True):

    # Combine train and test indices
    idx_stacked = np.concatenate((train_idx, test_idx))
    x_stacked = np.take(x_data, idx_stacked, axis)
    y_stacked = labels[idx_stacked]

    # Split into training and testing sets
    sep = train_idx.shape[0]
    y_train, y_test = np.split(y_stacked, [sep])
    x_train, x_test = np.split(x_stacked, [sep], axis=axis)

    if oversample:
        # Apply mixup2 to x_train
        mixup2(arr=x_train, labels=y_train, obs_axs=axis, alpha=1., seed=None)

    # Fill in test data nans with noise from distribution
    is_nan = np.isnan(x_test)
    x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

    # Recombine x_train and x_test
    x_stacked = np.concatenate((x_train, x_test), axis=axis)

    return x_stacked, y_train, y_test
