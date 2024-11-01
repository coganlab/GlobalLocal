import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from scipy.stats import norm
from tqdm import tqdm

# largely stolen from aaron's ieeg plot_decoding.py

def mixup2(arr: np.ndarray, obs_axis: int, alpha: float = 1.,
          seed: int = None, labels: np.ndarray) -> None:
    arr = arr.swapaxes(obs_axis, -2)
    if arr.ndim > 2:
        for i in range(arr.shape[0]):
            mixup2(arr[i], obs_axis, alpha, seed)
    else:
        if seed is None:
            seed = np.random.randint(0, 2 ** 16 - 1)
        if obs_axis == 0:
            arr = arr.swapaxes(1, obs_axis)

        n_nan = np.where(np.isnan(arr).any(axis=1))
        n_non_nan = np.where(~np.isnan(arr).any(axis=1))
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, size=n_nan)
        else:
            lam = np.ones(n_nan)

        for i, l in zip(n_nan, lam):
            l_class = labels[i]
            possible_choices = np.where(np.logical_and(~np.isnan(arr).any(axis=1), labels == l_class))
            choice1 = np.choice(possible_choices)
            choice2 = np.choice(n_non_nan)
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

    def cv_cm_old(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            
            y_train = labels[train_idx]
            mixup2(x_train, obs_axs, 1., None, y_train)
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
    
    def cv_cm_aaron(self, x_data: np.ndarray, labels: np.ndarray,
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

            print("Shuffle validation:")
            for i, labels in enumerate(label_stack):
                print(f"Shuffle {i+1}: Labels preview - {labels[:10]}")
                # Compare with the first repetition to ensure variety in shuffles
                if i > 0:
                    diff = np.sum(label_stack[0] != labels)
                    print(f"Difference with first shuffle: {diff} labels differ")

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
        # 11/1 put in the actual cv cm logic from the cv cm old here. Note this uses idxs instead of idx, so will be a little different since idxs includes splits and labels already

        # 11/1 below is aaron's code for windowing. 
        # def proc(train_idx, test_idx, l):
        #     x_stacked, y_train, y_test = sample_fold(train_idx, test_idx, data, l, 0, oversample)
        #     windowed = windower(x_stacked, window, axis=-1)
        #     out = np.zeros((windowed.shape[0], n_cats, n_cats), dtype=np.uint8)
        #     for i, x_window in enumerate(windowed):
        #         x_flat = x_window.reshape(x_window.shape[0], -1)
        #         x_train, x_test = np.split(x_flat, [train_idx.shape[0]], 0)
        #         out[i] = self.fit_predict(x_train, x_test, y_train, y_test)
        #     return out

        # # loop over folds and repetitions
        # if n_jobs == 1:
        #     idxs = tqdm(idxs, total=self.n_splits * self.n_repeats)
        #     results = (proc(train_idx, test_idx, l) for (train_idx, test_idx), l in idxs)
        # else:
        #     results = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
        #         delayed(proc)(train_idx, test_idx, l)
        #         for (train_idx, test_idx), l in idxs)

        # # Collect the results
        # for i, result in enumerate(results):
        #     rep, fold = divmod(i, self.n_splits)
        #     mats[:, rep, fold] = result

        # # average the repetitions
        # if average_repetitions:
        #     mats = np.mean(mats, axis=1)

        # # normalize, sum the folds
        # mats = np.sum(mats, axis=-3)
        # if normalize == 'true':
        #     divisor = np.sum(mats, axis=-1, keepdims=True)
        # elif normalize == 'pred':
        #     divisor = np.sum(mats, axis=-2, keepdims=True)
        # elif normalize == 'all':
        #     divisor = self.n_repeats
        # else:
        #     divisor = 1
        # return mats / divisor
    
    def cv_cm_return_scores(self, x_data: np.ndarray, labels: np.ndarray,
                            normalize: str = None, obs_axs: int = -2):
        '''
        trying to get the scores manually from cv cm but i realize that in decoders.py, PcaLdaClassification already has a get_scores function. Try get_scores with shuffle=True to get fake, permuted scores.
        '''
        # Get the confusion matrix by calling `cv_cm`
        cm = self.cv_cm_old(x_data, labels, normalize, obs_axs)

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
