Predictive Uncertainty Quantification and Calibration

# Work in progress : UQ for neural network 





# warning LSTM-ED model need numpy==1.19.5


# Examples (Air Liquide Demand Forecast)

* [**Gradient Boosting Regressor Quantile UQ-Predictor**](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/uc_air_liquide/demo/uq_models/GBRQ_predictor.ipynb) 

* [**Gaussian Process UQ-Predictor**](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/uc_air_liquide/demo/uq_models/GP_predictor.ipynb) 

* [**Regression ML As UQ-Predictor **](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/uc_air_liquide/demo/uq_models/REGML_predictor.ipynb) 

* [**Random forest Uncertainty Quantification UQ-Predictor**](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/uc_air_liquide/demo/uq_models/RFUQ_predictor.ipynb) 

* [**Loop test of All UQ-Predictors**](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/uc_air_liquide/demo/uq_models/test_predictor.ipynb)


## Predictor

The `uq_models.predictor.BasePredictor` class is a wrapper for regression models that aims to standardize their interface and force compliance with the previously presented requirements:
* The models have to implement the `fit` and `predict` methods
* The models have to operate on datasets formated as numpy arrays (in `_format`)

Any specific preprocessing should be included in a **subclass** of `predictor.BasePredictor`.

A special attention is to be directed towards the `predict` method.

```python
# Args:
#   X: new example features
# Returns:
#   A tuple composed of (y_pred, y_pred_lower, y_pred_upper, sigma_pred)
predict(X: numpy.array) -> (numpy.array, numpy.array, numpy.array, numpy.array)
```
The tuple returned by the method contains four elements:
*   `y_pred`: point predictions 
*   `y_pred_lower`: lower bound of the prediction intervals 
*   `y_pred_upper`: upper bound of the prediction intervals
*   `sigma_pred`: variability estimations

If the model does not estimate some of these values, they are substituted by `None` at the right position. For example, a quantile regressor that returns upper and lower interval bounds is wrapped as follows:

```python
class QuantilePredictor(BasePredictor):
    def __init__(self, q_lo_model, q_hi_model):
        """
        Args:
            q_lo_model: lower quantile model
            q_hi_model: upper quantile model
        """
        super().__init__()
        self.q_lo_model = q_lo_model
        self.q_hi_model = q_hi_model

    def fit(self, X, y, **kwargs):
        """Fit model to the training data.
        Args:
            X: train features
            y: train labels
        """
        self.q_lo_model.fit(X, y)
        self.q_hi_model.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        """Compute predictions on new examples.
        Args:
            X: new examples' features
        Returns:
            y_pred, y_lower, y_upper, sigma_pred
        """
        y_pred_lower = self.q_lo_model.predict(X)
        y_pred_upper = self.q_hi_model.predict(X)
        return None, y_pred_lower, y_pred_upper, None
``` 

To cover a large range of conformal prediction methods, three subclasses of `puncc.predictor.BasePredictor` have been implemented: 

* `uq_models.predictor.MeanPredictor`: wrapper of point-based models   
* `uq_models.predictor.MeanVarPredictor`: wrapper of point-based models that also estimates variability statistics (e.g., standard deviation)  
* `uq_models.predictor.QuantilePredictor`: wrapper of specific interval-based models that estimate upper and lower quantiles of the data generating distribution 

User-defined predictors have to subclass `puncc.predictor.BasePredictor` and redefine its methods.
Some standars UQ predictors are already implemented using subclass BasePredictor wrapper :

* `uq_models.common.predictors.GP_predictor` : Gaussian process
* `uq_models.common.predictors.GP_predictor` : Gradient bossting regressor quantile
* `uq_models.common.predictors.GP_predictor` : Regresion ML UQ based
*`uq_models.common.predictors.GP_predictor`  : Random Forest for Uncertainty Quantification.

