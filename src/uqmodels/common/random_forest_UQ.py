import pickle
import numpy as np
from joblib import Parallel, delayed
from uqmodels.predictor import BaseUQPredictor, BaseDUQPredictor
from uqmodels.utils import aux_tuning, compute_born, format, flatten, EPSILON
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _generate_sample_indices


class PredictorRF_UQ(BaseUQPredictor):
    """Uncertainty quantification approch based on "local" sub-sampling UQ estimation from Random forest neighboorhood extraction"""

    def __init__(
        self,
        estimator=RandomForestRegressor(random_state=0),
        pretuned=False,
        mode="sigma",
        use_biais=True,
        rescale=True,
        n_jobs=8,
        beta=0.1,
        list_statistics=[
            "pred",
            "n_obs",
            "aleatoric",
            "epistemic",
            "oob_aleatoric",
        ],
        var_min=0.00001,
    ):

        super().__init__()
        self.name = "PredictorRF_UQ"
        self.rescale = rescale
        self.mode = mode
        self.beta = beta
        self.use_biais = use_biais
        self.var_min = var_min
        if self.use_biais:
            list_statistics.append("biais")

        self.pretuned = pretuned
        self.estimator_mean = estimator
        self.dict_leaves_statistics = dict()
        self.n_jobs = n_jobs
        if self.mode == "sigma":
            list_statistics.append("var")

        elif self.mode == "2sigma":
            list_statistics.append("var_bot")
            list_statistics.append("var_top")

        elif self.mode == "quantile":
            list_statistics.append("Q_bot")
            list_statistics.append("Q_top")

        self.list_statistics = list_statistics

    def _format(self, X, y, fit=False, mode=None, flag_inverse=False, **kwargs):
        if self.rescale:
            X, y = format(self, X, y, fit, mode, flag_inverse)
            if fit:
                self.var_min = self.var_min / np.power(self.scaler[1].scale_, 2)
        return X, y

    def fit(self, X, y, beta=None, **kwargs):
        """Train scikit RF model and perform subsample uncertainty quantification stati(X)
        Args:
            X ([array]): Features of the training set
            y ([attay]): Target of the training set
            beta ([float]): miss-coverage target (uncertainty quantification)
        """

        def aux_leaf_statistics(Y_leaf, Y_pred_leaf, Y_oob_leaf, list_statistics, beta):
            """Auxiliar function : Extract statistics of a leaf:

            Args:
                Y_leaf ([float np.array (?,dim)]): Target of train leaf's elements
                Y_pred_leaf ([float np.array (?,dim)]): Predict of train leaf's elements
                Y_oob_leaf ([float np.array (?,dim)]): Target of oob(out-of-bound) leaf's elements
                list_statistics ([list]) List of statistics to compute.
                beta ([float]): miss-coverage target (used for quantile estimation)

            Ouput:
                dict_statistics : dict of extracted statistics statistics.
            """

            dict_statistics = dict()

            n_obs = len(Y_leaf)
            n_obs_oob = len(Y_oob_leaf)
            if "n_obs" in list_statistics:
                dict_statistics["n_obs"] = n_obs

            # Compute biais : Error on out of "bag" sample (Part of train sample leave beside for the Tree)
            biais = 0
            if "biais" in list_statistics:  # Moyenne du biais
                if len(Y_oob_leaf) > 4:
                    biais = (Y_oob_leaf - Y_leaf.mean(axis=0)).mean(axis=0)
                dict_statistics["biais"] = biais

            if "oob_aleatoric" in list_statistics:  # Variance du biais
                dict_statistics["oob_aleatoric"] = 0
                if len(Y_oob_leaf) > 1:
                    dict_statistics["oob_aleatoric"] = (Y_oob_leaf).var(axis=0, ddof=1)

            # Compute : the whole bias : Mean of leafs erros on both Used train sample and Non-Used train sample.
            # If val_pred is the tree forecast values (Y_train_leaf - val_pred) = 0 so biais = biais

            # Debias forecast values and compute residuals in order to perform variance estimation
            Residual = Y_leaf - Y_leaf.mean(axis=0)

            if "pred" in list_statistics:
                dict_statistics["pred"] = Y_leaf.mean(axis=0)

            # Estimation of leaf residuals variance.
            if "var" in list_statistics:
                dict_statistics["var"] = (np.power(Y_leaf, 2) / (n_obs)).sum(axis=0)

            # Estimation of exploratory statistics
            if "aleatoric" in list_statistics:
                dict_statistics["aleatoric"] = Y_leaf.var(
                    axis=0, ddof=1
                )  # partial E[Var[X]]

            # partial E[X] No biais because reducing with mean that doesn't take account biais (native RF predict function)
            if "epistemic" in list_statistics:
                dict_statistics["epistemic"] = np.power(Y_leaf.mean(axis=0), 2)

            # Identify negative and positive residuals (for '2sigma' or 'quantile' estimation)
            Residual = Y_leaf - (Y_leaf.mean(axis=0) - biais)
            flag_res_bot = Residual <= 0
            flag_res_top = Residual >= 0

            # Estimation of positive and negative leaf residuals variance.
            if "var_bot" in list_statistics:
                # Identify negative and positive residuals
                flag_res_bot = Residual <= 0
                flag_res_top = Residual >= 0
                dict_statistics["var_bot"] = EPSILON
                dict_statistics["var_top"] = EPSILON
                if (flag_res_bot).sum() > 2:
                    dict_statistics["var_bot"] = Residual[flag_res_bot].var(
                        axis=0, ddof=1
                    )

                if (flag_res_top).sum() > 2:
                    dict_statistics["var_top"] = Residual[flag_res_top].var(
                        axis=0, ddof=1
                    )

            if "Q_bot" in list_statistics:
                # Identify negative and positive residuals
                flag_res_bot = Residual <= 0
                flag_res_top = Residual >= 0
                dict_statistics["Q_bot"] = EPSILON
                dict_statistics["Q_top"] = EPSILON
                if (flag_res_bot).sum() > 2:
                    dict_statistics["Q_bot"] = np.quantile(
                        Residual[flag_res_bot],
                        beta * (1 + 1 / flag_res_bot.sum()),
                        axis=0,
                    )
                if (flag_res_top).sum() > 2:
                    dict_statistics["Q_top"] = np.quantile(
                        Residual[flag_res_top],
                        np.minimum((1 - beta) * (1 + 1 / flag_res_top.sum()), 0.995),
                        axis=0,
                    )

            return dict_statistics

        def aux_tree_statistics(
            num_tree,
            n_samples,
            max_samples,
            y,
            y_pred,
            tree_affectation,
            list_statistics,
            beta,
            simple_m,
            bootstrap,
        ):
            """Extraction of statistics for each leaves of a tree

            Args:
                num_tree ([int]): ID of the tree
                n_samples ([float]): Random forest paramater (used to reproduce the trainning set)
                max_samples ([int]): Random forest parameter (used to reproduce the trainning set)
                y ([array]): Target of training set values
                y_pred ([array]): Forecast of training set values
                tree_affectation ([érray]): Array of element leaves affectations
                pred_true ([2D Array of float]): Forecast values !!! Non-used !!!
                list_statistics ([list]) List of statistics to compute.
                beta ([float]): miss-coverage target (used for quantile estimation)
                simple_m ([object]): scikit learn decision tree model

            Output:
                Side effect on dict_statistics
            """

            # Regenerate the bootstrat training sample thank to scikit function

            leaves = list(set(tree_affectation))
            if bootstrap:
                n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, max_samples)
                re_sample = _generate_sample_indices(
                    simple_m.random_state, n_samples, n_samples_bootstrap
                )
                inv_draw = np.ones(n_samples)
                inv_draw[re_sample] = 0
                oob_sample = np.repeat(np.arange(inv_draw.size), inv_draw.astype(int))
            else:
                # Compute leaves affectation for the tree on bootstrap sample.

                # Identify non-used training data : oob data
                re_sample = np.arange(n_samples)
                inv_draw = np.ones(n_samples)
                inv_draw[re_sample] = 0
                oob_sample = np.repeat(np.arange(inv_draw.size), inv_draw.astype(int))
            # Compute leaves affectation for the oob sample.

            leaves_interest = []
            # add key : (num_tree,num_leaf) to save leaf values.

            # For each (visited) leaves :
            tree_statistics = dict()
            for num_leaf in leaves:

                # Identify concerned bootstrap and oob observations.
                Y_leaf = y[re_sample[tree_affectation[re_sample] == num_leaf]]
                Y_pred_leaf = y_pred[re_sample[tree_affectation[re_sample] == num_leaf]]
                Y_oob_leaf = y[oob_sample[tree_affectation[oob_sample] == num_leaf]]

                # Extract leaf statistics
                tree_statistics[num_leaf] = aux_leaf_statistics(
                    Y_leaf, Y_pred_leaf, Y_oob_leaf, list_statistics, beta
                )
                if (num_tree, num_leaf) in leaves_interest:
                    tree_statistics[num_leaf]["Y_leaf"] = Y_leaf
            return (num_tree, tree_statistics)

        if beta is None:
            beta = self.beta

        X, y = self._format(X, y, fit=True)
        self.X_train = X
        self.Y_train = y
        list_statistics = self.list_statistics
        model_rf = self.estimator_mean
        # Fit the model using scikit method
        model_rf.fit(X, y)
        RF_affectation = model_rf.apply(X)
        n_estimators = int(model_rf.n_estimators)
        y_pred = model_rf.predict(X)

        # Extract subsample statistics
        parrallel_inputs = [
            (
                num_tree,
                len(y),
                model_rf.max_samples,
                y,
                y_pred,
                RF_affectation[:, num_tree],
                list_statistics,
                beta,
                model_rf.estimators_[num_tree],
                model_rf.bootstrap,
            )
            for num_tree in np.arange(n_estimators)
        ]

        Rf_leaves_statistics = Parallel(n_jobs=self.n_jobs)(
            delayed(aux_tree_statistics)(*inputs) for inputs in parrallel_inputs
        )
        # Store leaves statistics of each tree in a dict
        dict_leaves_statistics = dict()
        for num_tree, dict_tree_statistics in Rf_leaves_statistics:
            dict_leaves_statistics[num_tree] = dict_tree_statistics
        self.dict_leaves_statistics = dict_leaves_statistics
        return

    def predict(self, X, beta=None, **kwargs):
        """Predict both forecast and UQ estimations values
        Args:
            X ([type]): Features of the data to forecast
            beta ([type]): Miss-coverage target

        Returns:
        y_pred ([array]): Forecast values
        y_pred_lower ([type]): Lower bound of Predictive interval
        y_pred_upper ([type]): Upper bound of Predictive interval
        sigma_pred ([type]): Uncertainty quantification values.
        """

        if beta is None:
            beta = self.beta

        X, _ = self._format(X, None, fit=False)

        # Call auxiliaire function that compute RF statistics from leaf subsampling.
        y_pred, biais, sigma_pred, var_A, var_E, _ = self.RF_extraction(X)

        if self.use_biais:
            y_pred = y_pred - biais

        # Compute (top,bot) boundaries from (1 or 2)-sigma estimation
        if (self.mode == "sigma") | (self.mode == "2sigma"):
            y_pred_lower, y_pred_upper = compute_born(
                y_pred, sigma_pred, alpha=beta, mode=self.mode
            )
            _, sigma_pred = self._format(None, sigma_pred, mode=self.mode)

        # Estimate pseudo-sigma from quantile (bot,top) boundaries
        elif self.mode == "quantile":
            y_pred_lower, y_pred_upper = (
                np.moveaxis(sigma_pred, -1, 0)[0],
                np.moveaxis(sigma_pred, -1, 0)[1],
            )
            sigma_pred = y_pred_upper - y_pred_lower

        _, y_pred = self._format(None, y_pred, flag_inverse=True)
        _, y_pred_lower = self._format(None, y_pred_lower, flag_inverse=True)
        _, y_pred_upper = self._format(None, y_pred_upper, flag_inverse=True)

        return (y_pred, y_pred_lower, y_pred_upper, sigma_pred)

    def RF_extraction(self, X):
        """Random-forest subsampling statistics extraction "

        Args:
            X ([array]): Features of elements.

        Output:
            Statistics array of shape (n_obs,dim):
            'Pred' : Random forest forecast values
            'Biais' : RF Biais computed as the sum of Oob Tree biais

            'UQ' : Shape of UQ depends of the mode !!!
            IF mode=SIGMA:
                UQ = Var  RF variance computed as the sum of esiduals' variance of the leaves
            IF mode=2-SIGMA:
                UQ = (Var_bot,Var_top), !!! 2-uple of 2D array !!!
                Var_bot : bot variance stimation  (partial sum) as sum of negative residuals' variance of the leaves
                Var_top : top variance stimation  (partial sum) as sum of positive residuals' variance of the leaves
            IF mode=quantile:
                UQ = (Q_bot,Q_top), !!! 2-uple of 2D array !!!
                'Q_bot' :  Bot quantile estimation (partial sum) as ponderation of leaves' bot quantile
                'Q_top' :  Top quantile estimatuon (partial sum) as ponderation of leaves' top quantile

            Other advanced statistics.
            'Biais_oob' :  part of Biais
            'Var_E' : Part of total variance (Law of total variance)
            'E_Var' : Part of total variance (Law of total variance)
            'Var_oob' : Part of total variance (related to biais)
        """

        def aux_predict(
            shape,
            list_statistics,
            list_RF_affectation,
            list_dict_tree_statistics,
            prediction,
        ):
            """Aggregate statistics (partial sum) of serveral trees.
            Args:
                shape ([tupple]): shape of statistics
                list_statistics ([list]): list of statistcs to extract
                list_RF_affectation ([array]): array of elements affectations for serveral trees
                list_dict_tree_statistics ([dict]): pre-computed leaves statistics for the trees

            Returns:
                agg_statistics ([array]): Partials aggregated statistics (for serveral tree) of elmments to forecast
            """
            agg_statistics = np.zeros((shape))
            for num, tree_affectation in enumerate(list_RF_affectation):
                tree_statistic = list_dict_tree_statistics[num]
                tree_statistics = tree_predict(
                    shape, list_statistics, tree_affectation, tree_statistic, prediction
                )
                agg_statistics += np.array(tree_statistics)

            return agg_statistics

        def tree_predict(
            shape, list_statistics, tree_affectation, tree_statistic, prediction
        ):
            """Compute extracted statistcs of a tree for the elements to forecast.

            Args:
                shape ([tupple]): shape of statistics
                list_statistics ([type]): [description]
                tree_affectation ([type]): array of elements affectations for the tree
                tree_statistic ([type]): pre-computed leaves statistics for the tree

            Returns:
                statistics ([array]): Partials statistics for (a tree) of elmments to forecast
            """
            leaves = list(set(tree_affectation))
            statistics = []
            for n, key in enumerate(list_statistics):
                statistics.append(np.zeros(shape[1::]))

            for num_leaf in leaves:
                mask = tree_affectation == num_leaf
                for n, key in enumerate(list_statistics):
                    statistics[n][mask] = tree_statistic[num_leaf][key]
            return statistics

        n_trees = self.estimator_mean.n_estimators
        # Compute Leaves affectation array
        RF_affectation = self.estimator_mean.apply(X)
        prediction = self.estimator_mean.predict(X)
        list_statistics = self.list_statistics

        # Define shape of the statistic array.
        if len(self.Y_train.shape) == 1:
            shape = (len(list_statistics), len(X), 1)

        else:
            shape = (len(list_statistics), len(X), self.Y_train.shape[1])

        parallel_partition = np.array_split(range(n_trees), self.n_jobs * 2)

        # Split inputs of auxillar parralel tree statistics extraction
        parallel_input = []
        for partition in parallel_partition:
            parallel_input.append(
                (
                    shape,
                    list_statistics,
                    [RF_affectation[:, i] for i in partition],
                    [self.dict_leaves_statistics[i] for i in partition],
                    prediction,
                )
            )

        # Extract statistcs for each tree in RF in a parralel way :

        Predicted_statistics = Parallel(n_jobs=self.n_jobs)(
            delayed(aux_predict)(*inputs) for inputs in parallel_input
        )

        # Final aggregation and normalisation for each statistics
        Predicted_statistics = np.stack(Predicted_statistics).sum(axis=0)

        Pred = Predicted_statistics[list_statistics.index("pred")] / n_trees
        Biais = Pred * 0
        if "biais" in list_statistics:
            Biais = Predicted_statistics[list_statistics.index("biais")] / n_trees

        if self.mode == "sigma":
            var = (
                Predicted_statistics[list_statistics.index("var")] / (n_trees)
            ) - np.power(Pred, 2)

            var = np.maximum(var, self.var_min)
            UQ = np.sqrt(var)

        if self.mode == "2sigma":

            var_bot = Predicted_statistics[list_statistics.index("var_bot")] / n_trees
            Sigma_bot = np.sqrt(np.maximum(var_bot, self.var_min / 2))
            var_top = Predicted_statistics[list_statistics.index("var_top")] / n_trees
            Sigma_top = np.sqrt(np.maximum(var_top, self.var_min / 2))

            if len(self.Y_train.shape) == 1:
                UQ = np.concatenate([Sigma_bot, Sigma_top], axis=-1)
            else:
                UQ = np.concatenate(
                    [np.expand_dims(i, -1) for i in [Sigma_bot, Sigma_top]], axis=-1
                )

        if self.mode == "quantile":
            Q_bot = (
                Pred + Predicted_statistics[list_statistics.index("Q_bot")] / n_trees
            )
            Q_top = (
                Pred + Predicted_statistics[list_statistics.index("Q_top")] / n_trees
            )
            if len(self.Y_train.shape) == 1:
                UQ = np.concatenate([Q_bot, Q_top], axis=-1)

            else:
                UQ = np.concatenate(
                    [np.expand_dims(i, -1) for i in [Q_bot, Q_top]], axis=-1
                )

        if "aleatoric" in list_statistics:
            var_aleatoric = (
                Predicted_statistics[list_statistics.index("aleatoric")] / n_trees
            )
            var_aleatoric = np.maximum(var_aleatoric, self.var_min / 2)

        if "epistemic" in list_statistics:
            var_epistemic = Predicted_statistics[list_statistics.index("epistemic")] / (
                n_trees
            ) - np.power(Pred, 2)
            var_epistemic = np.maximum(var_epistemic, self.var_min / 2)

        if "oob_aleatoric" in list_statistics:
            var_aleatoric_oob = (
                Predicted_statistics[list_statistics.index("oob_aleatoric")] / n_trees
            )
            var_aleatoric_oob = np.maximum(var_aleatoric_oob, self.var_min / 2)

        Pred, Biais, UQ = [array for array in [Pred, Biais, UQ]]

        return (
            Pred,
            Biais,
            UQ,
            var_aleatoric,
            var_epistemic,
            var_aleatoric_oob,
        )

    def _tuning(self, X, y, n_esti=100, folds=4, params=None, **kwarg):
        """Perform random search tuning using a given grid parameter"""
        if not (self.pretuned):
            if type(params) != type(None):
                X, y = self._format(X, y, fit=True)
                reg = RandomForestRegressor(random_state=0)
                score = "neg_mean_squared_error"
                self.estimator_mean = aux_tuning(
                    reg, X, y, params, score, n_esti, folds
                )


# CHANGE TO DO : swap both inherence and better hold forma
class PredictorRF_UQ_distangle(PredictorRF_UQ, BaseDUQPredictor):
    def factory(self, X, y, mask=None, **kwargs):
        return (X, y, mask)

    def predict(self, X, beta=None, **kwargs):
        """Predict both forecast and UQ estimations values
        Args:
            X ([type]): Features of the data to forecast
            beta ([type]): Miss-coverage target

        Returns:
        y_pred ([array]): Forecast values
        y_pred_lower ([type]): Lower bound of Predictive interval
        y_pred_upper ([type]): Upper bound of Predictive interval
        sigma_pred ([type]): Uncertainty quantification values.
        """

        if beta is None:
            beta = self.beta

        X, _ = self._format(X, None, fit=False)

        # Call auxiliaire function that compute RF statistics from leaf subsampling.
        y_pred, biais, sigma_pred, var_A, var_E, _ = self.RF_extraction(X)
        if self.use_biais:
            y_pred = y_pred - biais

        _, y_pred = self._format(None, y_pred, flag_inverse=True)
        _, sigma_A = self._format(
            None, np.sqrt(var_A), flag_inverse=True, mode=self.mode
        )
        _, sigma_E = self._format(
            None, np.sqrt(var_E), flag_inverse=True, mode=self.mode
        )
        return y_pred, np.power(sigma_A, 2), np.power(sigma_E, 2)
