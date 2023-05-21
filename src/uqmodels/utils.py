"""
This module implements utility functions.
"""

import sys
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import numpy as np
from typing import Iterable

EPSILON = sys.float_info.min  # small value to avoid underflow


def extract_from_dict(dictionaire, list_str):
    list_extract = []
    for str_name in list_str:
        if str_name in list(dictionaire.keys()):
            list_extract.append(dictionaire[str_name])
        else:
            list_extract.append(None)

    if len(list_extract) == 1:
        list_extract = list_extract[0]
    return list_extract


def get_coeff(alpha):
    return scipy.stats.norm.ppf(alpha, 0, 1)


def flatten(y):
    if not (y is None):
        if (len(y.shape) == 2) & (y.shape[1] == 1):
            y = y[:, 0]
    return y


# Folding function


def get_fold_nstep(size_window, size_subseq, padding):
    # Compute the number of step according to size of window, size of subseq and padding.
    return int(np.floor((size_window - size_subseq) / (padding))) + 1


def stack_and_roll(array, horizon, lag=0, seq_idx=None, padding=1):
    # Perform a stack "horizon" time an array and roll in temporal stairs.
    # a n+1 dimensional array with value rolled k times for the kth line
    shape = (1, horizon, 1)
    new_array = np.tile(array[:, None], shape)
    if seq_idx is None:
        for i in np.arange(horizon):
            lag_current = horizon - lag - i - 1
            new_array[:, i] = np.roll(new_array[:, i], lag_current, axis=0)

        # Erase procedure
        for i in np.arange(horizon):
            lag_current = horizon - lag - i - 1
            if lag_current > 0:
                new_array[i, :lag_current] = 0

        # by blocks :

    else:
        _, idx = np.unique(seq_idx, return_index=True)
        for j in seq_idx[np.sort(idx)]:
            flag = seq_idx == j
            list_idx = np.nonzero(flag)[0]
            for i in np.arange(horizon):
                lag_current = horizon - lag - i - 1
                new_array[flag, i] = np.roll(new_array[flag, i], lag_current, axis=0)
                # A optimize : get first non zeros values.
            for i in np.arange(horizon):
                lag_current = horizon - lag - i - 1
                new_array[list_idx[i], :lag_current] = 0
    return new_array


def stack_and_roll_layer(inputs, size_window, size_subseq, padding, name=""):
    slide_tensor = []
    n_step = get_fold_nstep(size_window, size_subseq, padding)
    # Implementation numpy
    if False:
        for i in range(n_step):
            slide_tensor.append(
                inputs[:, (i * padding) : (i * padding) + size_subseq, :][:, None]
            )
        return Lambda(lambda x: K.concatenate(x, axis=1), name=name + "_rollstack")(
            slide_tensor
        )
    else:
        x = tf.map_fn(
            lambda i: inputs[:, (i * padding) : (i * padding) + size_subseq, :],
            tf.range(n_step),
            fn_output_signature=tf.float32,
        )
        x = tf.transpose(x, [1, 0, 2, 3])
        return x


def apply_mask(list_or_array, mask):
    if type(list_or_array) is list:
        return [i[mask] for i in list_or_array]
    else:
        return list_or_array[mask]


# Preprocessing function


def cut(var, cut_min, cut_max):
    vpmin = np.quantile(var, cut_min, axis=0)
    vpmax = np.quantile(var, cut_max, axis=0)
    var = np.minimum(var, vpmax)
    var = np.maximum(var, vpmin)
    return var


def format(self, X, y, fit=False, mode=None, flag_inverse=False):
    """Feature and target Formatting"""
    if self.rescale:
        flag_reshape = False
        if not (y is None):
            if len(y.shape) == 1:
                flag_reshape = True
                y = y[:, None]

        if fit:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            X = scalerX.fit_transform(X)
            scalerY = StandardScaler(with_mean=True, with_std=True)
            y = scalerY.fit_transform(y)
            self.scaler = [scalerX, scalerY]

        elif not (flag_inverse):
            X = self.scaler[0].transform(X)
            if not (y is None):
                y = self.scaler[1].transform(y)
        else:
            X_transformer = self.scaler[0].inverse_transform
            Y_transformer = self.scaler[1].inverse_transform
            if not (X is None) and not (len(X) == 0):
                X = X_transformer(X)

            if not (y is None) and not (len(y) == 0):
                sigma = np.sqrt(self.scaler[1].var_)
                if mode == "sigma":
                    y = y * sigma

                elif mode == "2sigma":
                    y_reshape = np.moveaxis(y, -1, 0)
                    if len(y.shape) == 3:
                        y = np.concatenate(
                            [
                                np.expand_dims(i, -1)
                                for i in [y_reshape[0] * sigma, y_reshape[1] * sigma]
                            ],
                            axis=-1,
                        )
                    else:
                        y = np.concatenate(
                            [
                                y_reshape[0][:, None] * sigma,
                                y_reshape[1][:, None] * sigma,
                            ],
                            axis=-1,
                        )

                else:
                    y = Y_transformer(y)

        if flag_reshape:
            y = y[:, 0]
    return (X, y)


# PIs basics computation functionalities


def compute_born(y_pred, sigma, alpha, mode="sigma"):
    """Compute y_upper and y_lower boundary from gaussian hypothesis (sigma or 2sigma)

    Args:
        y_pred (array) : Mean prediction
        sigma (array) : Variance estimation
        alpha (float) : Misscoverage ratio
        mode (str) : Distribution hypothesis (sigma : gaussian residual hypothesis, 2sigma : gaussian positive and negative residual hypothesis)

    Returns:
       (y_lower,y_upper): Lower and upper bondary of Predictive interval
    """

    # Case sigma
    if mode == "sigma":
        y_lower = y_pred + scipy.stats.norm.ppf((alpha / 2), 0, sigma)
        y_upper = y_pred + scipy.stats.norm.ppf((1 - (alpha / 2)), 0, sigma)
    # Case 2 sigma
    elif mode == "2sigma":
        sigma = np.moveaxis(sigma, -1, 0)
        y_lower = y_pred + scipy.stats.norm.ppf((alpha / 2), 0, sigma[0])
        y_upper = y_pred + scipy.stats.norm.ppf((1 - (alpha / 2)), 0, sigma[1])
    return y_lower, y_upper


# Gaussian mixture quantile estimation :

from joblib import Parallel, delayed


def mixture_quantile(pred, var_A, quantiles, n_jobs=5):
    def aux_mixture_quantile(pred, var_A, quantiles):
        list_q = []
        n_data = pred.shape[1]
        n_mixture = pred.shape[0]
        for n in range(n_data):
            mean_law = scipy.stats.norm(
                pred[:, n, 0].mean(), np.sqrt(var_A[:, n, 0].mean())
            )
            xmin = mean_law.ppf(0.0000001)
            xmax = mean_law.ppf(0.9999999)
            scale = np.arange(xmin, xmax, (xmax - xmin) / 300)
            Mixture_cdf = np.zeros(len(scale))
            for i in range(n_mixture):
                cur_law = scipy.stats.norm(pred[i, n, 0], np.sqrt(var_A[i, n, 0]))
                Mixture_cdf += cur_law.cdf(scale) / n_mixture
            q_val = []
            for q in quantiles:
                q_val.append(scale[np.abs(Mixture_cdf - q).argmin()])
            list_q.append(q_val)
        return np.array(list_q)

    list_q = []
    n_data = pred.shape[1]
    n_mixture = pred.shape[0]
    parallel_partition = np.array_split(np.arange(n_data), 3)
    # Split inputs of auxillar parralel tree statistics extraction
    parallel_input = []
    for partition in parallel_partition:
        parallel_input.append((pred[:, partition], var_A[:, partition], quantiles))
    list_q = Parallel(n_jobs=n_jobs)(
        delayed(aux_mixture_quantile)(*inputs) for inputs in parallel_input
    )
    return np.concatenate(list_q, axis=0)


# Scikit tuning function


def aux_tuning(
    model,
    X,
    Y,
    params=None,
    score="neg_mean_squared_error",
    n_esti=100,
    folds=4,
    verbose=0,
):
    """Random_search with sequential k-split

    Args:
        model (scikit model): Estimator
        X ([type]): Features
        Y ([type]): Target
        params ([type], optional): parameter_grid. Defaults to None.
        score (str, optional): score. Defaults to 'neg_mean_squared_error'.
        n_esti (int, optional): Number of grid try . Defaults to 100.
        folds (int, optional): Number of sequential fold. Defaults to 4.
        verbose (int, optional): [description]. Defaults to 0.
    """
    if type(params) == type(None):
        return model
    else:
        tscv = TimeSeriesSplit(n_splits=folds)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=n_esti,
            scoring=score,
            n_jobs=8,
            cv=tscv.split(X),
            verbose=verbose,
        )
        random_search.fit(X, Y)
        return random_search.best_estimator_


def agg_list(l: Iterable):
    try:
        return np.concatenate(l, axis=0)
    except ValueError:
        return None


def agg_func(l: Iterable):
    try:
        return np.mean(l, axis=0)
    except TypeError:
        return None


class GenericCalibrator:
    """Generic calibrator implementing several calibration
    type_res : "no_calib : No calibration
               "res" : Calibration based on mean residuals
               "w_res" : Calibration based on weigthed mean residuals
               "cqr" : Calibration based on quantile residuals

    mode :
            if "symetric" : symetric calibration ()
            else perform  calibration independently on positive and negative residuals.
    """

    def __init__(self, type_res="res", mode="symetric", name=None, alpha=0.1):
        super().__init__()
        self._residuals = None
        self.name = name
        if name is None:
            self.name = type_res + "_" + mode
        self.mode = mode
        self.alpha = alpha
        self.type_res = type_res

        if mode == "symetric":
            self.fcorr = 1
        else:
            self.fcorr_lower = 1
            self.fcorr_upper = 1

    def estimate(
        self, y_true, y_pred, y_pred_lower, y_pred_upper, sigma_pred, **kwargs
    ):

        flag_res_lower = np.zeros(y_true.shape)
        flag_res_upper = np.zeros(y_true.shape)

        if (self.type_res == "res") | (self.type_res == "w_res"):
            if self.type_res == "res":
                residuals = y_true - y_pred
                sigma_pred = y_true * 0 + 1

            if self.type_res == "w_res":
                # sigma_pred = np.maximum(y_pred_upper - y_pred_lower, EPSILON) / 2
                residuals = y_true - y_pred

            if len(residuals.shape) == 1:
                flag_res_lower = residuals <= 0
                flag_res_upper = residuals >= 0

            else:
                flag_res_lower = np.concatenate(
                    [
                        np.expand_dims(residuals[:, i] <= 0, -1)
                        for i in range(0, y_true.shape[1])
                    ]
                )
                flag_res_upper = np.concatenate(
                    [
                        np.expand_dims(residuals[:, i] >= 0, -1)
                        for i in range(0, y_true.shape[1])
                    ]
                )

            if y_pred.shape != sigma_pred.shape:
                sigma_pred = np.moveaxis(sigma_pred, -1, 0)
                residuals[flag_res_lower] = (
                    np.abs(residuals)[flag_res_lower] / (sigma_pred[0])[flag_res_lower]
                )
                residuals[flag_res_upper] = (
                    np.abs(residuals)[flag_res_upper] / (sigma_pred[1])[flag_res_upper]
                )
            else:
                residuals = np.abs(residuals) / (sigma_pred)

        elif self.type_res == "cqr":
            residuals = np.maximum(y_pred_lower - y_true, y_true - y_pred_upper)
            flag_res_lower = (y_pred_lower - y_true) >= (y_true - y_pred_upper)
            flag_res_upper = (y_pred_lower - y_true) <= (y_true - y_pred_upper)

        elif self.type_res == "no_calib":
            return

        else:
            print("Unknown type_res")
            return

        if self.mode == "symetric":
            self.fcorr = np.quantile(
                residuals, (1 - self.alpha) * (1 + 1 / len(residuals))
            )

        else:
            self.fcorr_lower = np.quantile(
                residuals[flag_res_lower],
                np.minimum((1 - self.alpha) * (1 + 1 / flag_res_lower.sum()), 1),
            )

            self.fcorr_upper = np.quantile(
                residuals[flag_res_upper],
                np.minimum((1 - self.alpha) * (1 + 1 / flag_res_upper.sum()), 1),
            )
        return

    def calibrate(self, y_pred, y_pred_lower, y_pred_upper, sigma_pred, **kwargs):

        if self.type_res == "res":
            sigma_pred = y_pred * 0 + 1

        if self.mode == "symetric":
            fcorr_lower = self.fcorr
            fcorr_upper = self.fcorr
        else:
            fcorr_lower = self.fcorr_lower
            fcorr_upper = self.fcorr_upper

        if self.type_res in ["res", "w_res", "no_calib"]:

            if y_pred.shape != sigma_pred.shape:
                sigma_pred = np.moveaxis(sigma_pred, -1, 0)
                y_pred_lower = y_pred - sigma_pred[0] * fcorr_lower
                y_pred_upper = y_pred + sigma_pred[1] * fcorr_upper
            else:
                y_pred_lower = y_pred - sigma_pred * fcorr_lower
                y_pred_upper = y_pred + sigma_pred * fcorr_upper

        elif self.type_res in ["cqr"]:
            y_pred_upper = y_pred_upper + fcorr_upper
            y_pred_lower = y_pred_lower - fcorr_lower
        else:
            print("Unknown type_res")
            return

        return (y_pred_lower, y_pred_upper)
