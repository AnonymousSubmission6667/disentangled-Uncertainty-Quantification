"""
This module provides wrappings for ML models.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy
from uqmodels.utils import aux_tuning, compute_born
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class BaseUQPredictor(BaseEstimator):
    """Abstract structure of a base predictor class."""

    def __init__(self):
        self.estimator = "None"
        self.rescale = False
        self.is_trained = False

    @abstractmethod
    def _format(self, X: np.array, y: np.array, type_tranform: str):
        """Format data to be consistent with the the fit and predict methods.
        Args:
            X: features
            y: labels
            type_tranform: action : {fit_transform,tranform,inverse_transform}
        Returns:
            formated_X, formated_y
        """
        if self.rescale:
            flag_reshape = False
            if not (y is None):
                if len(y.shape) == 1:
                    flag_reshape = True
                    y = y[:, None]

            if type_tranform == "fit_transform":  # Fit X&Y Scaler
                scalerX = StandardScaler(with_mean=True, with_std=True)
                X = scalerX.fit_transform(X)
                scalerY = StandardScaler(with_mean=True, with_std=True)
                y = scalerY.fit_transform(y)
                self.scaler = [scalerX, scalerY]

            elif type_tranform == "tranform":  # Transform X
                X = self.scaler[0].transform(X)
                if not (y is None):  # Transform Y if given
                    print("Atypical call : why not_fit_transform ?")
                    y = self.scaler[1].transform(y)

            elif type_tranform == "inverse_transform":  # Inverse Transform predY
                Y_transformer = self.scaler[1].inverse_transform
                if not (X is None):  # X inverse transform why ?
                    print("Atypical call : why inverse transform X ?")
                # Y inverse transform
                y = Y_transformer(y)
            else:
                print("Atypical call : type incorrect : ", type)

        if flag_reshape:
            y = y[:, 0]
        return X, y

    @abstractmethod
    def fit(self, X: np.array, y: np.array, **kwargs) -> None:
        """Fit model to the training data.
        Args:
            X: train features
            y: train labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.array, **kwargs):
        """Compute predictions on new examples.
        Args:
            X: new examples' features
        Returns:
            y_pred, y_lower, y_upper, sigma_pred
        """
        pass

    @abstractmethod
    def _tuning(self, X: np.array, y: np.array, **kwargs):
        """Fine-tune the model's hyperparameters.
        Args:
            X: features from the validation dataset
            y: labels from the validation dataset
        """
        pass


class BaseDUQPredictor(BaseUQPredictor):
    def _format(self, X: np.array, y: np.array, type_tranform: str, type_y="y"):
        """Format data to be consistent with the the fit and predict methods.
        Args:
            X: features
            y: labels
            type_transform: action : "fit_transform,tranform,inverse_transform"}
            type_y : "y,sigma"
        Returns:
            formated_X, formated_y
        """
        if self.rescale:
            # Transform X & Y
            if type_y == "y":
                X, y = super()._format(X, y, type_tranform)

            # Transform sigma pred
            elif (type_y == "UQ") & (type_tranform == "inverse_tranform"):
                flag_reshape = False
                if not (y is None):
                    if len(y.shape) == 1:
                        flag_reshape = True
                        y = y[:, None]

                sigma = np.sqrt(self.scaler[1].var_)
                y = y * sigma
                if flag_reshape:
                    y = y[:, 0]
        return (X, y)


class NNDUQPredictor(BaseDUQPredictor):
    """Class for neural network with disentengled uncertainty quantification
    INprogess
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator


class MeanPredictor(BaseUQPredictor):
    def __init__(self, estimator_mu):
        super().__init__()
        self.name = "MeanPredictor"
        self.estimator_mu = estimator_mu

    def _format(self, X, y):
        return X, y

    def fit(self, X, y, **kwargs):
        self.estimator_mu.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        y_pred = self.estimator_mu.predict(X)
        return y_pred, None, None, None


class MeanVarPredictor(BaseUQPredictor):
    def __init__(self, estimator_mu, estimator_sigma, gaussian=True):
        super().__init__()
        self.name = "MeanVarPredictor"
        self.estimator_mu = estimator_mu
        self.estimator_sigma = estimator_sigma
        self.gaussian = gaussian

    def fit(self, X, y, **kwargs):
        self.estimator_mu.fit(X, y)
        y_pred = self.estimator_mu.predict(X)
        residual = np.abs(y - y_pred)
        self.estimator_sigma.fit(X, residual)
        self.is_trained = True

    def predict(self, X, **kwargs):
        y_pred = self.estimator_mu.predict(X)
        sigma_pred = self.estimator_sigma.predict(X)
        y_pred_lower, y_pred_upper = None, None
        if self.gaussian:
            if "beta" in kwargs.keys() and kwargs["beta"] is not None:
                y_pred_lower, y_pred_upper = compute_born(
                    y_pred, sigma_pred, alpha=kwargs["beta"], mode="sigma"
                )
            else:
                raise Exception("Alpha argument needed if gaussian is True.")
        return y_pred, y_pred_lower, y_pred_upper, sigma_pred

    def _format(
        self, X: np.array, y: np.array, type_tranform: str, type_y="y", mode_UQ="sigma"
    ):
        """Format data to be consistent with the the fit and predict methods.
        Args:
            X: features
            y: labels
            type_transform: action : "fit_transform,tranform,inverse_transform"}
            type_y : {"y" or "sigma"} : Transform target or UQestimation
            UQ_mode : {"sigma" or "2sigma"} : UQestimation in sigma or 2sigma hypothesis
        Returns:
            formated_X, formated_y"""
        if self.rescale:
            # Transform X & Y
            if type_y == "y":
                X, y = super()._format(X, y, type_tranform)

            # Transform sigma pred
            elif (type_y == "UQ") & (type_tranform == "inverse_tranform"):
                flag_reshape = False
                if not (y is None):
                    if len(y.shape) == 1:
                        flag_reshape = True
                        y = y[:, None]

                sigma = np.sqrt(self.scaler[1].var_)
                if mode_UQ == "sigma":
                    y = y * sigma

                elif mode_UQ == "2sigma":
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

                if flag_reshape:
                    y = y[:, 0]
        return (X, y)


class QuantilePredictor(BaseUQPredictor):
    def __init__(self, estimator_qbot, estimator_qtop, estimator_qmid="None"):
        """
        Args:
            q_lo_model: lower quantile model
            q_hi_model: upper quantile model
            q_mid_model: mid quantile model (can be None)
        """
        super().__init__()
        self.name = "QuantilePredictor"
        self.estimator_qmid = estimator_qmid
        self.estimator_qbot = estimator_qbot
        self.estimator_qtop = estimator_qtop

    def fit(self, X, y, **kwargs):
        if self.estimator_qmid != "None":
            self.estimator_qmid.fit(X, y)
        self.estimator_qbot.fit(X, y)
        self.estimator_qtop.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        y_pred = None
        if self.estimator_qmid != "None":
            y_pred = self.estimator_qmid.predict(X)
        y_pred_lower = self.estimator_qbot.predict(X)
        y_pred_upper = self.estimator_qtop.predict(X)
        return y_pred, y_pred_lower, y_pred_upper, None

    def _format(self, X: np.array, y: np.array, type_tranform: str):
        """Format data to be consistent with the the fit and predict methods.
        Args:
            X: features
            y: labels
        Returns:
            formated_X, formated_y"""
        X, y = super()._format(X, y, type_tranform)
        return (X, y)
