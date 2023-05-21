Agnostic benchmark tool for ML performance evaluation on a wrapper paradigm.

The idea is to specify upstream a model wrapper dedicated to the task to be solved, associated with a metrics wrapper capable of interpreting the model wrapper's output to produce performance.

We also have to provide an iterable data handler that loads/contains/generates different datasets used to perform some training (data handler can realize cross-validation). We can attach contextual information to the subsets of data in order to compute contextual performance based on external contextual information.

Then, we specify and provide to "abench.benchmark" function the models and metrics to be evaluated using wrappers. It stores, in a file tree, the different subsets of data, their contextual information and the associated results of each model.

Another "abench.evaluate" function can then post-compute the performance associated with the wrapper metrics for each data-set, and store the per-dataset and the aggregated performance results in the tree file.

Metric performances can then be retrieved to be visualized through graphical. Model outputs can also be retrieved to feed more specific visualizations that have to be wrapped in a specific format.


# Quickstart

Agnostic benchmark module implement an benchmark loop that aim to facilitate models comparison :

Agnostics benchmark requiere to specify :
- Model encapsultor paradigme (implicite task specification)
- Metrics compatible with the outputs of model encapsulation
- Visualisation compatible with the outputs of model encapsulation

# Architecture Overview

![plot](./doc/Abench_core.jpg)

![plot](./doc/Abench_example.jpg)

## Model encapsulators

`abench.benchmark.Encapsulated_model` is the canvas of model wrappers. An object instance can be constructed by subpart.

`abench.benchmark.Encapsulated_model` have to implements three methods:

* A `init` method that initialise Encaspulated model from the subparts elements
```python
# subparts : components of model to encaspulate
encapsulated_model(subparts)
```

* A `fit` method that fits the model for a dedicated task
```python
# X_train and y_train are the full training dataset used to train some subpart of the model
# context_train are additional informal that may be used during training by some subpart of the model
# The splitter passed as argument to ConformalPredictor assigns data 
# to the fit and calibration sets based on the provided splitting strategy
encapsulated_model.fit(X_train, y_train, context_train, **kwarg)
```

* a `predict` method that fits the model for a dedicated task
```python
# X are the feature used to perform model inference.
# predict method return a arbitrary object 'Output' (specify by the wrapper) that contain model results.
output = encapsulated_model.predict(X, context, **kwarg)
```

* a `factory` method can also be implemented in order to preprocess data.
```python
# X and y are the raw data that will be transform in Inputs and Targets ojects in order to feed 
# fit and predict model. Factory is more relevant to be perform in the model wrapper definition
# y can be None in order  
Inputs, Targets = encapsulated_model.meta_model.factory(X, y, **kwarg)
```

## Model Wrapper 

Le modèle wrapper est une spécification cruciale permettant à abench de manipuler de manière agnostique des modèles d'implémentations différentes par wrapping qui permettant d'homogénéiser leurs méthodes (Init, Fit, Predict) et leurs sorties (output). Elle peut être implémentée par l'utilisateur ou reprise sur des templates spécifier.

```python
# Abstract encapsulator class :

class Encapsulated_model(ABC):
    """Abstract Encapsulated Model class :
    Allow generic manipulation of models"""

    def __init__(self, subpart_1=None, subpart_n=None, **kwarg):
        """ Stockage / initialization o
        
        self.subpart_1 = subpart_1
        self.subpart_n = subpart_n

    def fit(self, X, y, **kwarg):
        """Fitting procedure

        Args:
            X (array): Inputs
            y (array): Targets
        """
        pass

    def predict(self, X, **kwarg):
        """Predict procedure

        Args:
            X (array): Inputs

        Returns:
            output : Encapsulated results format
        """
        output = None
        return output

        def factory(self, X, y=none,mask=none, **kwarg):
        """Predict procedure

        Args:
            X (array): Inputs
            y (array): Targets
            mask (array): Mask array

        Returns:
            Inputs : XEncapsulated results format
            Targets : y for fit function
            mask :  Mask array
        """
        Inputs,Targets = X,y
        return Inputs,Targets,mask

```

# Metrics encapsulators

`abench.benchmark.Encapsulated_metrics` is the canvas of metrics wrappers. Encapsulated_metrics manipulated output of Encapsulated_model

`abench.benchmark.Encapsulated_model` implements three methods:

* A `init` method that initialise Encaspulated metric paramaters and name
```python
encapsulated_metrics()
```

* A `compute` method that compute metrics using Encaspulated_model output and additional information
```python
#output (array): Model results
#y (array): Targets
#sets (array list): Sub-set (train,test)
#context (array): Additional information
#objective (array) : ground truth for unsupervised task evaluation
metric_performance = encapsulated_model.predict(X, y, context=none, ojective=none,**kwarg)
```

```python
# Encapsulated metrics class :
class Encapsulated_metrics(ABC):
    """Abstract Encapsulated Metrics class :
    Allow generic manipulation of metrics with output specifyied format"""

    def __init__(self):
        self.name = "metrics"

    def compute(self, output, y, sets, context, **kwarg):
        """Compute metrics

        Args:
            output (array): Model results
            y (array): Targets
            sets (array list): Sub-set (train,test)
            context (array): Additional information
        """
        pass

```

# Examples (Air Liquide Demand Forecast)

* [**Benchmark apply on regression task**](https://git.irt-systemx.fr/confianceai/ec_3/n5_uncertainty_calibration/-/blob/master/demo/demo_benchmark.ipynb) 

To do visualisation + Split benchmark strategy