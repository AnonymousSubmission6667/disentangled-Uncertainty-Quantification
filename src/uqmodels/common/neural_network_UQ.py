from uqmodels.utils import cut
import numpy as np
from uqmodels.utils import apply_mask, compute_born, format, flatten, EPSILON
from tensorflow.keras import callbacks
from tensorflow.keras.models import clone_model
from keras import metrics, optimizers, regularizers
from sklearn.model_selection import KFold
import copy
from uqmodels.predictor import NNDUQPredictor


class NN_var(NNDUQPredictor):
    def __init__(
        self,
        model_initializer,
        model_parameters,
        factory_parameters=dict(),
        training_parameters=dict(),
        type_var=None,
        factory_function=None,
        rescale=False,
        n_ech=8,
        train_ratio=0.9,
        name="NN",
        native_procedure=False,
    ):

        self.name = name
        self.model_initializer = model_initializer
        self.model_parameters = model_parameters
        self.native_procedure = native_procedure
        self.factory_parameters = factory_parameters
        self.training_parameters = training_parameters
        self.type_var = type_var
        self.initialized = False
        self.rescale = rescale
        self.history = []
        self.n_ech = n_ech
        self.train_ratio = train_ratio

        # Additional deep ensemble parameter
        self.ddof = 1
        if "ddof" in model_parameters.keys():
            self.ddof = model_parameters["ddof"]

        if "train_ratio" in model_parameters.keys():
            self.train_ratio = model_parameters["train_ratio"]

        if "n_ech" in model_parameters.keys():
            self.n_ech = model_parameters["n_ech"]

        self.snapshot = False
        if "snapshot" in model_parameters.keys():
            self.snapshot = model_parameters["snapshot"]

        self.data_drop = 0
        if "data_drop" in model_parameters.keys():
            self.data_drop = model_parameters["data_drop"]

        if "k_fold" in model_parameters.keys():
            self.k_fold = model_parameters["k_fold"]

        if not (factory_function is None):
            self.factory_function = factory_function

    def _format(self, X, y, fit=False, mode=None, flag_inverse=False, **kwargs):
        X, y = format(self, X, y, fit, mode, flag_inverse)
        return (X, y)

    def factory(self, X, y, mask=None, cut_param=None):
        shape = y.shape
        if cut_param is None:
            y = y.reshape(shape)
        else:
            print("cuting_target")
            min_cut, max_cut = cut_param
            y = cut(y, min_cut, max_cut).reshape(shape)

        if self.rescale:
            X, y = self._format(X, y, fit=True, mode=None, flag_inverse=False)
        Inputs, Targets = [X], y

        if not (self.initialized) or not hasattr(self, "model"):
            self.init_neural_network()

        if hasattr(self, "factory_function"):
            Inputs, Targets, mask = self.factory_function(
                X, y, **self.factory_parameters
            )

        else:
            if hasattr(self, "model"):
                if hasattr(self.model, "factory"):
                    Inputs, Targets, mask = self.model.factory(
                        X, y, **self.factory_parameters
                    )

        return (Inputs, Targets, mask)

    def save(self):
        self.model.save_weights(self.name)

    def load(self):
        self.model.load_weights(self.name)

    def modify_dropout(self, dp):
        self.model.save_weights(self.name)
        self.model_parameters["dp"] = dp
        self.model = self.model_initializer(**self.model_parameters)
        self.initialized = True
        self.model.load_weights(self.name)

    def reset(self):
        del self.model
        self.initialized = False

    def init_neural_network(self):
        "apply model_initializer function with model_parameters and store in self.model"
        if self.type_var == "Deep_ensemble":
            self.model = []
            for i in range(self.n_ech):
                self.model.append(self.model_initializer(**self.model_parameters))
        else:
            self.model = self.model_initializer(**self.model_parameters)
        self.initialized = True

    def fit(
        self,
        Inputs,
        Targets,
        train=None,
        test=None,
        training_parameters=None,
        verbose=0,
        **kwargs
    ):
        print("start_fit")

        if training_parameters is None:
            training_parameters = self.training_parameters

        if not (self.initialized) or not hasattr(self, "model"):
            self.init_neural_network()

        if train is None:
            if type(Inputs) is list:
                train = np.random.rand(len(Inputs[0])) < self.train_ratio
            else:
                train = np.random.rand(len(Inputs)) < self.train_ratio
            test = np.invert(train)

        if self.native_procedure:
            history = self.model.fit(
                Inputs,
                Targets,
                train,
                test,
                verbose=verbose,
                sample_w=None,
                **self.training_parameters
            )
            self.history.append(history)

        else:
            history = self.basic_fit(
                Inputs,
                Targets,
                train,
                test,
                verbose=verbose,
                **self.training_parameters
            )

            for i in history:
                self.history.append(history)

    # Basic_predict function

    def basic_fit(
        self,
        Inputs,
        Targets,
        train=None,
        test=None,
        epochs=[1000, 1000],
        b_s=[100, 20],
        l_r=[0.01, 0.005],
        sample_w=None,
        verbose=0,
        list_loss=["mse"],
        metrics=None,
        callbacks=[],
        param_loss=None,
    ):

        # Training function
        history = []

        if train is None:
            np.random.seed(0)
            train = np.random.rand(len(Inputs[0])) < 0.9
            test = np.invert(train)

        list_history = []

        n_model = 1
        if self.type_var == "Deep_ensemble":
            n_model = self.n_ech

            list_sampletoremove = []
            if not (self.k_fold is None):
                if self.k_fold < self.n_ech:
                    print("Warning kfold lesser than model number")
                # Drop data using Kfold + random drop ratio to add variability to deep ensemble
                for n_fold, (keep, removed) in enumerate(
                    KFold(self.k_fold, shuffle=True).split(train)
                ):
                    if self.data_drop > 0:
                        sampletoremove = np.random.choice(
                            keep, int(len(keep) * self.data_drop), replace=False
                        )
                        sampletoremove = np.concatenate([removed, sampletoremove])
                        sampletoremove.sort()
                        list_sampletoremove.append(sampletoremove)
            else:
                list_sampletoremove = [[] for i in range(self.n_ech)]
                if self.data_drop > 0:
                    for n, i in enumerate(list_sampletoremove):
                        sampletoremove = np.random.choice(
                            np.arange(len(train)),
                            int(len(train) * self.data_drop),
                            replace=False,
                        )
                        list_sampletoremove[n] = sampletoremove

        for n_model in range(n_model):
            train_ = np.copy(train)
            test_ = np.copy(test)

            # Deep_ensemble : Submodel dataset differentiation if kfold activated
            if self.type_var == "Deep_ensemble":
                train_[list_sampletoremove[n_model]] = False
                test_[list_sampletoremove[n_model]] = True
                print(train_.mean(), len(self.model))

            for n in range(len(list_loss)):
                for i in range(len(b_s)):
                    if not (param_loss is None):
                        if type(param_loss[n]) is dict:
                            loss_ = list_loss[n](**param_loss[n])
                            print(param_loss[n])

                        else:
                            loss_ = list_loss[n](param_loss[n])
                    else:
                        loss_ = list_loss[n]

                    if self.type_var == "Deep_ensemble":
                        if (self.snapshot) & (n_model > 0):
                            self.model[n_model] = clone_model(self.model[0])

                        self.model[n_model].compile(
                            optimizer=optimizers.Nadam(lr=l_r[i]),
                            loss=loss_,
                            metrics=metrics,
                        )
                        history = self.model[n_model].fit(
                            x=apply_mask(Inputs, train_),
                            y=apply_mask(Targets, train_),
                            validation_data=(
                                apply_mask(Inputs, test_),
                                apply_mask(Targets, test_),
                            ),
                            epochs=epochs[i],
                            batch_size=b_s[i],
                            sample_weight=sample_w,
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=verbose,
                        )

                    else:
                        self.model.compile(
                            optimizer=optimizers.Nadam(lr=l_r[i]),
                            loss=loss_,
                            metrics=metrics,
                        )

                        history = self.model.fit(
                            x=apply_mask(Inputs, train_),
                            y=apply_mask(Targets, train_),
                            validation_data=(
                                apply_mask(Inputs, test_),
                                apply_mask(Targets, test_),
                            ),
                            epochs=epochs[i],
                            batch_size=b_s[i],
                            sample_weight=sample_w,
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=verbose,
                        )
                    list_history.append(history)

        return list_history

    def predict(self, X, beta=None, type_var=None, **kwargs):
        if type_var == None:
            type_var = self.type_var

        if self.native_procedure:
            output = self.model.predict(X, n_ech=self.n_ech)
        else:
            output = self.basic_predict(X, n_ech=self.n_ech, type_var=type_var)

        if type_var == "BNN_raw":
            pred, var_A, var_E = output
            if self.rescale:
                pred = np.concatenate(
                    [
                        self._format(None, pred_i, flag_inverse=True)[1][None]
                        for pred_i in pred
                    ],
                    axis=0,
                )
                var_A = np.concatenate(
                    [
                        np.power(
                            self._format(
                                None, np.sqrt(var_A_i), flag_inverse=True, mode="sigma"
                            )[1],
                            2,
                        )[None]
                        for var_A_i in var_A
                    ],
                    axis=0,
                )

        else:
            pred, var_A, var_E = output
            if self.rescale:
                _, pred = self._format(None, pred, flag_inverse=True)

                var_A = np.power(
                    self._format(None, np.sqrt(var_A), flag_inverse=True, mode="sigma")[
                        1
                    ],
                    2,
                )

                var_E = np.power(
                    self._format(None, np.sqrt(var_E), flag_inverse=True, mode="sigma")[
                        1
                    ],
                    2,
                )

        return (pred, var_A, var_E)

    def basic_predict(self, Inputs, n_ech=20, type_var="MC_Dropout", s_min=0.000001):
        # Variational prediction + variance estimation for step T+1 et T+4(lag)
        model = self.model
        output = []
        if type_var == "MC_Dropout":
            for i in range(n_ech):
                output.append(model.predict(Inputs))
            pred, logvar = np.split(np.array(output), 2, -1)

            var_A = np.exp(logvar).mean(axis=0)
            var_E = np.var(pred, axis=0, ddof=self.ddof)
            pred = pred.mean(axis=0)

        elif type_var == "Deep_ensemble":
            for submodel in model:
                output.append(submodel.predict(Inputs))
            pred, logvar = np.split(np.array(output), 2, -1)
            var_A = np.exp(logvar).mean(axis=0)
            var_E = np.var(pred, axis=0, ddof=self.ddof)
            pred = pred.mean(axis=0)

        elif type_var == "MCDP":
            for i in range(n_ech):
                output.append(model.predict(Inputs))
            pred = np.array(output)
            var_E = np.var(pred, axis=0, ddof=self.ddof)
            pred = pred.mean(axis=0)
            var_A = var_E * 0

        elif type_var == "EDL":
            gamma, vu, alpha, beta = np.split(model.predict(Inputs), 4, -1)
            alpha = alpha + 10e-6
            pred = gamma
            s_min = 0.0001
            var_A = beta / (alpha - 1)
            # WARNING sqrt or not sqrt ?
            var_E = beta / (vu * (alpha - 1))
            if (var_E == np.inf).sum() > 0:
                print("Warning inf values in var_E replace by s-min")
            if (var_A == np.inf).sum() > 0:
                print("Warning inf values in var_E replace by s-min")
            var_E[var_E == np.inf] = 0
            var_A[var_A == np.inf] = 0

        elif type_var == "BNN_raw":
            for i in range(n_ech):
                output.append(model.predict(Inputs))
            pred, logvar = np.split(np.array(output), 2, -1)
            var_A = np.exp(logvar)
            var_E = np.power(pred, 2).mean(axis=0) - np.power(pred.mean(axis=0), 2)
            pred = pred

        elif type_var is None:
            pred = model.predict(Inputs)
            var_A = np.ones(pred.shape)
            var_E = np.ones(pred.shape)

        else:
            raise Exception(
                "Unknown type_var : choose 'MC_Dropout' or 'Deep_esemble' or 'EDL' or None"
            )

        var_E[var_E < s_min] = s_min
        var_A[var_A < s_min] = s_min

        return (pred, var_A, var_E)
