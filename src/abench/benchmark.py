"""
This module aim to performs a agnostics task benchmark using Encaspulated object
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pickle
from pyparsing import nested_expr
from sklearn.model_selection import PredefinedSplit, KFold, TimeSeriesSplit
from abench.store import (
    Extract_dict,
    write,
    read,
    get_data_generator,
    store_data_generator,
    store_model_parameters,
    get_cv_list,
    store_cv_list,
    get_model_result,
    get_dataset,
)
from abc import ABC, abstractmethod
from matplotlib.ticker import FormatStrFormatter

# Dictionary reading function for multiple keys
def apply_mask(list_tuple_array, mask):
    if type(list_tuple_array) is list:
        return [i[mask] for i in list_tuple_array]
    elif type(list_tuple_array) is tuple:
        array = np.array(list_tuple_array)[:, mask]
        return tuple(map(np.array, array))
    else:
        return list_tuple_array[mask]


def init_subpart(subpart):
    """Init subpart from stored in dict_exp format"""
    if subpart["parameters"] is None:
        subpart = subpart["initializer"]
    else:
        subpart = subpart["initializer"](**subpart["parameters"])
    return subpart


# Encapsulated model format :
class Encapsulated_model(ABC):
    """Abstract Encapsulated Model class :
    Allow generic manipulation of models"""

    def __init__(self, **kwarg):
        """init procedure

        Warning : abench store subpart model in a dict{'initializer':initializer,'paramaters':paramaters}
        In order to build subpart model in the meta model, use such procedure :

        for each subpart do :
        __init__(self,subpart_1= stored_subpart_1) where stored_subpart_1 is provided by abench
        subpart_1 = init_subpart(subpart_1)


        """

        pass

    def _tuning(self, X, y, context=None, **kwarg):
        """Tunning procedure

        Args:
            X (array): Inputs
            y (array): Targets
            context (array): Contextual complementary informations
        """
        pass

    def fit(self, X, y, context=None, **kwarg):
        """Fitting procedure

        Args:
            X (array): Inputs
            y (array): Targets
            context (array): Additional information
        """
        pass
        pass

    def predict(self, X, context=None, **kwarg):
        """Predict procedure

        Args:
            X (array): Inputs
             context (array): Contextual complementary information

        Returns:
            output : Encapsulated results format
        """
        output = None
        return output


# Encapsulated metrics format :
class Encapsulated_metrics(ABC):
    """Abstract Encapsulated Metrics class :
    Allow generic manipulation of metrics with output specifyied format"""

    def __init__(self):
        self.name = "metrics"

    def compute(self, y, output, sets, context, **kwarg):
        """Compute metrics

        Args:
            output (array): Model results
            y (array): Targets
            sets (array list): Sub-set (train,test)
            context (array): Additional information
        """
        pass


def build_ctx_mask(context, list_ctx_constraint):
    meta_flag = []
    for (ctx, min_, max_) in list_ctx_constraint:
        if not (min_ is None):
            meta_flag.append(context[:, ctx] > min_)
        if not (max_ is None):
            meta_flag.append(context[:, ctx] < max_)
    ctx_flag = np.array(meta_flag).mean(axis=0) == 1
    return ctx_flag


class Generic_metric(Encapsulated_metrics):
    def __init__(
        self,
        ABmetric,
        name="Metric",
        mask=None,
        list_ctx_constraint=None,
        reduce=True,
        **kwarg
    ):
        self.ABmetric = ABmetric
        self.name = name
        self.reduce = reduce
        self.mask = mask
        self.list_ctx_constraint = list_ctx_constraint
        self.kwarg = kwarg

    def compute(self, y, output, sets, context, **kwarg):
        perf_res = []
        if self.kwarg != dict():
            kwarg = self.kwarg

        if not (self.list_ctx_constraint is None):
            ctx_mask = build_ctx_mask(context, self.list_ctx_constraint)
        for set_ in sets:
            set_ = set_
            if not (self.list_ctx_constraint is None):
                set_ = set_ & ctx_mask
            perf_res.append(
                self.ABmetric(y, output, set_, self.mask, self.reduce, **kwarg)
            )
        return perf_res


# Encapsulated visualitation format :
def meta_plot(y, output, train, test, size, **kwarg):
    """Abstract visualisation function

    Args:
        y (array): Targets
        output (array): Model results
        train (boolean array): Identify training sample
        test (boolean array): Identify testing sample
        size (int,int): size of plot
    """
    pass


# Encapsulated data from stored dict file :
class splitter:
    """Generic data-set provider (Iterable)"""

    def __init__(self, X_split):
        self.X_split = X_split

    def split(self, X):
        def cv_split(X_split, i):
            train = np.arange(len(X))[X_split < i]
            test = np.arange(len(X))[X_split == i]
            return (train, test)

        return [
            cv_split(self.X_split, i) for i in range(1, 1 + int(self.X_split.max()))
        ]


def dataset_generator_from_stored_dict(list_file):
    """Produce data_generator (iterable [X, y, context, objective, train, X_split,cv_name]) from pickle stored dict (link)"""

    def load_data(file):
        dict_dataset = pickle.load(open(file, "rb"))
        list_str = ["X", "Y", "context", "objective", "train", "X_split", "cv_name"]
        X, y, context, objective, train, X_split, cv_name = Extract_dict(
            dict_dataset, list_str
        )
        return (X, y, context, objective, train, X_split, cv_name)

    dataset_generator = []
    for file in list_file:
        for i in list_file:
            load_data(file)
            yield
    return dataset_generator


# Encapsulated data from array :
def dataset_generator_from_array(
    X,
    y,
    context=None,
    objective=None,
    sk_split=TimeSeriesSplit(5),
    repetition=1,
    remove_from_train=None,
    attack_name="",
    cv_list_name=None,
):
    """Produce data_generator (iterable [X, y, context, objective, train, X_split, name]) from arrays

    Args:
        X (array): Inputs.
        y (array or None): Targets.
        context (array or None): Additional information.
        objective (array or None): Ground truth (Unsupervised task).
        sk_split (split strategy): Sklearn split strategy."""

    def select_or_none(array, sample):
        if array is None:
            return None
        elif type(array) is str:
            return array
        else:
            return array[sample]

    if remove_from_train is None:
        remove_from_train = np.zeros(len(X))

    dataset_generator = []
    for n_repet in np.arange(repetition):
        cpt = 0
        if n_repet == 0:
            str_repet = ""
        else:
            str_repet = "_bis" + str(n_repet)

        for train_index, test_index in sk_split.split(X):
            train = np.zeros(len(X))
            train[train_index] = 1
            train[(remove_from_train == 1) & (train == 1)] = -1

            sample_cv = np.concatenate([train_index, test_index])
            sample_cv.sort()
            cv_name = "cv_" + str(cpt) + attack_name + str_repet
            if cv_list_name:
                cv_name = cv_list_name[cpt] + attack_name + str_repet
            dataset_generator.append(
                [
                    select_or_none(e, sample_cv)
                    for e in [X, y, train, context, objective, cv_name]
                ]
            )
            cpt += 1
    return dataset_generator


######################################################################################
# Task agnostics benchmark core function :


def benchmark(
    storing,
    dataset_generator,
    dict_exp,
    obj_params,
    list_metrics=None,
    tuning_kwarg=None,
    obj_param=None,
    name_exp=None,
    cv_list=[],
    verbose=1,
):
    """Run a Task-agnostic evaluation benchmark on Encapsulated Meta-model (specified in dict_exp) using the given dataset_generator (iterable)
    and store result and performance in dist_res using specified meta-metrics (list_metrics).

    Args:
        dataset_generator (benchmark_generator): Iterable composed of item : (X,y,train,Context,Objective)
            X: Learning features
            y: Learning target
            train: Flag between train and test
            Context: Contextual features (Structure information  about times & context structure)
            Objective: Objective (Ground truth) for unsupervised task evaluation

        dict_exp (dict): Experiments process store in a dictionary with [Scheme,tuning_scheme,Each Subpart_dicts, exp_design]
                scheme : 2-upple giving Meta-model structure  : ('Meta-model-Encapsulator_str_id', List of subpart_str_id of Meta_model_init argument)
                tuning_scheme :  Specifies the parts / sub-parts to be tuned during the benchmark
                *Subpart_dict : a dictionnary that contains for each subpart and for the meta_model candidates in a dict with two keys ;
                    - 'subpart_model_name' : subpart_model &
                    - 'params': dict of gridsearch paramater for tunning.
                exp_design : List of List of Meta model (specify by dict with name, Metamodel-encaspulated:str_link, *Subparts:str_links) Each sub list share tuning procedure

            Meta model encapsulator must following meta_model_class Encapsulator

            class meta_model(ABC):
                def __init__(self,subpart_1_str=subpart_1_model,...,,subpart_n_str=subpart_n_model,**kwarg):
                    pass

                def _tuning(self, X, y, context=None, **kwarg):
                    pass

                def fit(self, X, y, context=None, **kwarg):
                    pass

                def predict(self, X, context=None, **kwarg):
                    output = None
                    return output


        obj_params (dict): dict of objective parameter that have to be given to the meta_model
            For example {'alpha':0.1,'beta':0.1,} for Prediction Interval task using ConfortPredictor meta-model.

        dict_res (dict): Storage of experiments results.

             dict_res['Meta_model_name'] = {'cv_list': list of str 'cv_i' for each cv_step
                                            'param': Paramater of UQ_model submodels.
                                            'perf_agg': dict of cv_aggregated performance (stored for each meta_metrics of list_metrics)
                                            for each 'cv_i':
                                                'cv_i'= {'sample_cv': Boolean list of cv sample (size = len(X)),
                                                        'train_cv': Boolean list of train cv sample (size = cv_sample.sum()),
                                                        'test_cv': Boolean list of cv test cv sample (size = cv_sample.sum()),
                                                        'ouput': meta model_output
                                                        'perf' : dict of performance metrics evaluated on cv_i  (stored for each meta_metrics of list_metrics)


        list_metrics (list): list of meta_metrics used for evaluation
            class meta_metrics(ABC):
                def __init__(self):
                    pass

                def compute(self, y, output, sets, context, **kwarg):
                    return(metric_result)
        tuning_kwarg (fivy): dict of tuning parameter given to tunning function of meta_model/submodel if it specfied.

        name_exp (str): name of the pickle dump file used store dict_res "".

    Returns:
        dict_res : Dict of result (also stored in "name_exp" pickle file)
    """
    flag_all_cv = False
    if len(cv_list) == 0:
        flag_all_cv = True

    # Extract Models information from dict_exp
    if not (obj_param is None):
        print("depreciated parameter : used obj_params that handle list paramaters")

    list_extract = ["encapsulated_model", "tuning_scheme", "exp_design"]
    encapsulated_model, tuning_scheme, exp_design = Extract_dict(dict_exp, list_extract)
    sub_part_str_list = list(tuning_scheme.keys())

    if not (type(obj_params) == list):
        obj_params = [obj_params]

    for obj_param in obj_params:

        if len(obj_params) == 1:
            obj_param_name = ""

        else:
            obj_param_name = "|" + obj_param["name"]

        # For eachs group of meta-model to be test.
        for n_experiments, experiments in enumerate(exp_design):
            print("n_experiments : " + str(n_experiments))
            # If it specified, Apply a common tunning process of each subpart to be test in the group

            for n_subpart, subpart_str in enumerate(sub_part_str_list):

                # If tuning_scheme specified a tunning process :
                if not (tuning_scheme[subpart_str] is None):

                    # If we have tuning_parameter specified
                    if not (tuning_kwarg is None):
                        # Recovers tuning_data of subpart to be tuned.
                        X, y = tuning_scheme[subpart_str]

                        # Recovers each candidate_name of the subpart in the group
                        sub_models = set(
                            [
                                dict_subparts_txt[subpart_str]
                                for dict_subparts_txt in experiments
                            ]
                        )
                        for sub_model_name in list(sub_models):
                            # Recovers each candidate from dict_exp[subpart] storage
                            current_sub_part = dict_exp[subpart_str][sub_model_name]

                            if type(current_sub_part) == dict:
                                if "params" in current_sub_part.keys():
                                    print(
                                        "depreciated model storage replace key : 'params' by 'grid_params' in "
                                        + sub_model_name
                                        + " dict"
                                    )

                                if not ("parameters" in current_sub_part.keys()):
                                    current_sub_part["parameters"] = None

                                if not ("grid_params" in current_sub_part.keys()):
                                    current_sub_part["grid_params"] = None

                                sub_model = current_sub_part["subpart"]
                                grid_params = current_sub_part["grid_params"]

                            else:
                                print(
                                    "depreciated : old dict_exp structure : move to subpart_dict with 'subpart':subpart & 'params':params keys to replace tupple (subpart,params)"
                                )
                                sub_model = current_sub_part[0]
                                grid_params = current_sub_part[1]
                            if not (grid_params is None):
                                # Call subpart tuning procedure
                                sub_model._tuning(
                                    X,
                                    y,
                                    params=grid_params,
                                    **tuning_kwarg,
                                    **obj_param
                                )
                                print("End of tunning")
                                # Store tunned model
                                if type(current_sub_part) == dict:
                                    current_sub_part["subpart"] = sub_model.__class__
                                    current_sub_part[
                                        "parameters"
                                    ] = sub_model.get_params()

                                else:
                                    current_sub_part = (sub_model, grid_params)

            # Eachs meta-model (specified as a list of subpart_candidate_name)

            for nn_experiments, dict_subparts_txt in enumerate(experiments):

                dict_subpart = {}
                # Recovers subpart model from subpart_candidate_name.
                for n, subpart_str in enumerate(sub_part_str_list):
                    subpart_name = dict_subparts_txt[subpart_str]
                    subpart_storage = dict_exp[subpart_str][subpart_name]
                    if type(subpart_storage) == dict:
                        dict_subpart[subpart_str] = {
                            "initializer": subpart_storage["subpart"],
                            "parameters": subpart_storage["parameters"],
                        }
                    elif type(subpart_storage) == tuple:
                        dict_subpart[subpart_str] = {
                            "initializer": subpart_storage[0],
                            "parameters": None,
                        }

                    else:
                        dict_subpart[subpart_str] = {
                            "initializer": subpart_storage,
                            "parameters": None,
                        }

                # Recovers name of meta_model_candidate
                name = dict_subparts_txt["name"] + obj_param_name
                print(name)
                list_cv_name_to_write = []
                # Subsample/UC_air_liquide/data set of the dataset generator i.e a item [X,y,split,context objectif]
                flag_write_cv_list = True
                if dataset_generator is None:
                    flag_write_cv_list = False
                    print("load existant data_set")
                    dataset_generator = get_data_generator(storing)
                    if dataset_generator is None:
                        print("Erreur : no data-set stored in" + storing)
                else:
                    store_data_generator(storing, dataset_generator)

                store_model_parameters(storing, name, str(dict_subpart))

                meta_model = None
                for i, dataset in enumerate(dataset_generator):
                    if len(dataset) == 6:
                        (X, y, split, context, objective, cv_name) = dataset
                    else:
                        print(
                            "depreciated dataset have to be structured as (X, y, split, context, objective, cv_name)"
                        )
                        (X, y, split, context, objective) = dataset
                        cv_name = "cv_" + str(i)

                    list_cv_name_to_write.append(cv_name)

                    if flag_all_cv:
                        cv_list.append(cv_name)

                    if cv_name in cv_list:
                        print("start " + cv_name)
                        start_time = time.time()

                        train_cv = split == 1
                        test_cv = split == 0

                        # Instanciate Meta_model be giving subpart model to the meta_model_encaspulator
                        meta_model = encapsulated_model(**dict_subpart)
                        # Fit meta_model using training sample.

                        start_time = time.time()

                        if hasattr(meta_model, "factory"):
                            Inputs, Targets, train_cv = meta_model.factory(
                                X, y, train_cv
                            )
                        else:
                            Inputs, Targets = X, y

                        meta_model.fit(
                            apply_mask(Inputs, train_cv),
                            apply_mask(Targets, train_cv),
                            verbose=verbose,
                            **obj_param
                        )

                        time_fit = time.time() - start_time
                        start_time = time.time()
                        output = meta_model.predict(Inputs, **obj_param)
                        time_pred = time.time() - start_time
                        if hasattr(meta_model, "reset"):
                            meta_model.reset()

                        # Store data
                        list_to_store = [
                            (
                                ["result", name, cv_name, "dict_perf"],
                                {"time_fit": time_fit, "time_pred": time_pred},
                            ),
                            (["result", name, cv_name, "output"], output),
                            (["result", name, "cv_list"], cv_list),
                            (["result", name, "obj_param"], obj_param),
                        ]

                        for (keys, values) in list_to_store:
                            write(storing, keys, values)

                        storing_model = read(storing, keys=["result", name])

                        perf_evaluate(
                            [storing_model],
                            dataset_generator,
                            list_metrics,
                            obj_param=obj_param,
                            cv_list=[cv_name],
                            verbose=1,
                        )

                store_cv_list(storing, list_cv_name_to_write)

                if meta_model:
                    if hasattr(meta_model, "delete"):
                        meta_model.delete()

                    del meta_model
                # Store data
                # Store the number of cv_step in dict_res

                # Performance evaluation on each cv_set for given list of meta_metrics
                storing_model = read(storing, keys=["result", name])

                perf_evaluate(
                    [storing_model],
                    dataset_generator,
                    list_metrics,
                    obj_param=obj_param,
                    cv_list=cv_list,
                )

                # Performance aggregation for given metrics
                if verbose:
                    dict_perf = read(storing, keys=["result", name, "perf_agg"])
                    for metric in list_metrics:
                        metric_name = metric.name
                        mean_train, std_train, mean_test, std_test = dict_perf[
                            metric_name
                        ]

                        print(
                            metric_name,
                            "Train",
                            np.round(mean_train, 3),
                            "±",
                            np.round(std_train, 3),
                            "| TEST",
                            np.round(mean_test, 3),
                            "±",
                            np.round(std_test, 3),
                        )

            # Store current result in the 'name_exp' pickle file.
            if (not (name_exp is None)) & (type(storing) == dict):
                file = open(name_exp, "wb")
                pickle.dump(storing, file)
                file.close()
    return storing


def build_sets(train_cv, test_cv, context, list_ctx, list_ctx_name):
    # for contextual data
    list_sets = []
    list_ctx_name_ = []
    if list_ctx is None:
        list_sets = [[train_cv | test_cv]]
        list_ctx_name_ = [""]
    else:
        for n, n_ctx in enumerate(list_ctx):
            list_sets.append([])
            # Case of univariate
            if not (type(n_ctx) in [tuple, list]):
                list_ctx_name_.append(list_ctx_name[n])

                # for modality of contextual data
                for modality in np.sort(list(set(context[:, n_ctx]))):
                    list_sets[n].append((context[:, n_ctx] == modality))

            # Case of multivarite
            else:
                ctx_1, ctx_2 = n_ctx
                list_ctx_name_.append(list_ctx_name[ctx_1] + "_" + list_ctx_name[ctx_2])

                # for modality of ctx_1
                for modality_1 in np.sort(list(set(context[:, ctx_1]))):

                    # for modality of ctx_2
                    for modality_2 in np.sort(list(set(context[:, ctx_2]))):
                        flag_1 = context[:, ctx_1] == modality_1
                        flag_2 = context[:, ctx_2] == modality_2
                        list_sets[n].append((flag_1 & flag_2))

    list_sets_ = []
    for n, sets in enumerate(list_sets):
        list_sets_.append([])
        for subset in sets:
            list_sets_[n].append((subset & train_cv))
        for subset in sets:
            list_sets_[n].append((subset & test_cv))

    return list_sets_, list_ctx_name_


def perf_evaluate(
    list_storing_model,
    dataset_generator,
    list_metrics,
    list_ctx=None,
    list_name_ctx=None,
    obj_param=None,
    cv_list=[],
    verbose=0,
):
    """Perform evaluation for a meta_model candidate from stored data & results and store performance in the given dict_res

    Args:
        model_result (dict): benchmark dict_res result for a specified meta_model candidate (contain  and ground truth).
        list_metrics (list): List of meta_metrics used for evaluation
        obj_param (dict): dict of objective parameter that have to be given to the meta_model

    Returns:
        None : Performance are stored in model_result dict
    """
    # Identify number of sub-experiments.

    # For each sub-experiments
    flag_all_cv = False
    if len(cv_list) == 0:
        flag_all_cv = True

    for i, dataset in enumerate(dataset_generator):
        if len(dataset) == 6:
            (X, y, split, context, objective, cv_name) = dataset
        else:
            print(
                "depreciated dataset have to be structured as (X, y, split, context, objective, cv_name)"
            )
            (X, y, split, context, objective) = dataset
            cv_name = "cv_" + str(i)

        if flag_all_cv:
            cv_list.append(cv_name)
        if cv_name in cv_list:
            print(cv_name, "len_train", (split == 1).sum())
            train_cv = split == 1
            test_cv = split == 0
            for storing_model in list_storing_model:
                list_to_read = [
                    [cv_name, "output"],
                    ["obj_param"],
                    [cv_name, "dict_perf"],
                ]
                output, obj_param, dict_perf_cv = [
                    read(storing_model, keys) for keys in list_to_read
                ]

                if None in [output]:
                    print(storing_model, "no output stored")
                    break

                if dict_perf_cv is None:
                    dict_perf_cv = dict()

                if obj_param is None:
                    obj_param = {}

                # for each meta_metrics of the metrics_lists
                list_to_store = []
                list_metrics_ctx = []
                for metric in list_metrics:

                    list_sets, list_name_ctx_ = build_sets(
                        train_cv, test_cv, context, list_ctx, list_name_ctx
                    )

                    for n_sets, sets in enumerate(list_sets):
                        # Perform metric evaluation
                        perf_metric = metric.compute(
                            y,
                            output,
                            sets=sets,
                            context=context,
                            objective=objective,
                            **obj_param
                        )
                        metric_name = metric.name + list_name_ctx_[n_sets]
                        list_metrics_ctx.append(metric_name)
                        list_to_store.append(([metric_name], perf_metric))

                if verbose:
                    print(list_to_store)

                for (keys, values) in list_to_store:
                    write(dict_perf_cv, keys, values)

                write(storing_model, [cv_name, "dict_perf"], dict_perf_cv)

    # For each models
    for storing_model in list_storing_model:
        # Compute meta-perform for each meta-metrics by aggreagate sub-experiments
        dict_perf_agg = read(storing_model, ["perf_agg"])
        if dict_perf_agg is None:
            dict_perf_agg = {}

        list_dict_perf_cv = [
            read(storing_model, [cv_name, "dict_perf"]) for cv_name in cv_list
        ]

        # Time_fit
        if "time_fit" in dict_perf_cv.keys():
            time_fit_mean = np.array(
                [dict_perf_cv["time_fit"] for dict_perf_cv in list_dict_perf_cv]
            ).mean()
        else:
            time_fit_mean = -1
        dict_perf_agg["time_fit"] = time_fit_mean

        # Time_pred
        if "time_pred" in dict_perf_cv.keys():
            time_pred_mean = np.array(
                [dict_perf_cv["time_pred"] for dict_perf_cv in list_dict_perf_cv]
            ).mean()
        else:
            time_pred_mean = -1
        dict_perf_agg["time_pred"] = time_pred_mean

        # Compute meta-perform for each meta-metrics by aggreagate sub-experiments

        for metric_name in list_metrics_ctx:

            metrics_perfs = np.array(
                [dict_perf_cv[metric_name] for dict_perf_cv in list_dict_perf_cv]
            )

            means = metrics_perfs.mean(axis=0)
            stds = metrics_perfs.std(axis=0)

            pos_mid = int(len(means) / 2)

            dict_perf_agg[metric_name] = np.array(
                (
                    np.squeeze(means[:pos_mid]),
                    np.squeeze(stds[:pos_mid]),
                    np.squeeze(means[pos_mid:]),
                    np.squeeze(stds[pos_mid:]),
                )
            )

        write(storing_model, ["perf_agg"], dict_perf_agg)


def evaluate(
    storing,
    list_name,
    list_metrics,
    list_ctx=None,
    list_name_ctx=None,
    verbose=0,
    dict_perf=None,
    obj_param=None,
    suffixe_name="",
    cv_list=[],
):
    """Perform evaluation using precomputed-experiments (dict_res) and store performance in a dictionary of metrics performance (dict_perf).
    Args:
        model_result (dict): benchmark dict_res result for a specified meta_model candidate (contain  and ground truth).
        list_metrics (list): List of meta_metrics used for evaluation
        obj_params (dict): dict of objective parameter that have to be given to the meta_model

    Returns:
        dict_perf : dict of aggregated performance for each meta_metrics in list_metrics
    """
    if not (obj_param is None):
        print("predreciated parameter : recovered from dict_res")

    # Initiate storage performance dictionary
    if dict_perf is None:
        dict_perf = {}

    # Load data_generator
    dataset_generator = get_data_generator(storing)
    # for each meta-model candidate
    list_storing_model = []
    for name in list_name:
        storing_model = read(storing, keys=["result", name])
        list_storing_model.append(storing_model)
    # Compute metrics using auxiliary function.

    perf_evaluate(
        list_storing_model,
        dataset_generator,
        list_metrics,
        list_ctx,
        list_name_ctx,
        cv_list=cv_list,
        verbose=0,
    )

    for name in list_name:
        name_suffixed = name + suffixe_name
        # Store aggregate performances metrics in dict_perf
        dict_perf[name_suffixed] = read(storing, keys=["result", name, "perf_agg"])

        if verbose > 0:
            # Print performance metrics for each metrics.
            print(
                name,
                " |time_fit :",
                np.round(dict_perf[name]["time_fit"], 2),
                "time_pred :",
                np.round(dict_perf[name]["time_pred"], 2),
            )
            list_metrics_name = [i.name for i in list_metrics]
            for metric_name in list_metrics_name:
                mean_train, std_train, mean_test, std_test = dict_perf[name][
                    metric_name
                ]
                print(
                    metric_name,
                    "Train",
                    np.round(mean_train, 3),
                    "±",
                    np.round(std_train, 3),
                    "| TEST",
                    np.round(mean_test, 3),
                    "±",
                    np.round(std_test, 3),
                )
            print()

    return dict_perf


######################################################################################
# META visualisation


def plot_curve(
    storing, list_name, meta_plot, cv_name, plot_param, size=(18, 2), names=None
):
    """[summary]

    Args:
        dict_res ([type]): [description]
        list_name ([type]): [description]
        meta_plot ([type]): [description]
        cv_name ([type]): [description]
        plot_param ([type]): [description]
        size (tuple, optional): [description]. Defaults to (18, 2).
    """

    dataset_generator, cv_list = get_data_generator(storing), get_cv_list(storing)
    if cv_name is None:
        cv_name = cv_list[-1]

    cv_id = cv_list.index(cv_name)

    # data cross validation recovering
    dataset = dataset_generator[cv_id]
    y, split, context = dataset[1], dataset[2], dataset[3]
    train = split == 1
    test = split == 0

    if names is None:
        names = list_name

    for n, (name, title) in enumerate(zip(list_name, names)):
        output = get_model_result(storing, name, cv_name)
        meta_plot(
            y,
            output,
            train,
            test,
            context,
            size=size,
            name=title,
            show_legend=(n == 0),
            **plot_param
        )
    return


def scatter_result(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    figsize=(15, 15),
):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        dict_perf ([type]): [description]
        colors ([type]): [description]
    """

    if names is None:
        names = candidates

    X_loc = np.zeros((len(candidates), len(Metrics_performance) * 2))

    for n, candidate in enumerate(candidates):
        for nn, m in enumerate(Metrics_performance):
            perf = dict_perf[candidate][m]
            X_loc[n][nn * 2] = np.mean(perf[2])
            X_loc[n][nn * 2 + 1] = np.mean(perf[3])

    x = X_loc[:, 0]
    y = X_loc[:, 2]
    x_error = X_loc[:, 1] / 2
    y_error = X_loc[:, 3] / 2

    fig, (ax0) = plt.subplots(nrows=1, sharex=True, figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    for i in range(len(X_loc)):
        color = colors[i]
        ax0.scatter(x[i], y[i], label=names[i], color=color)
        ax0.errorbar(
            x[i],
            y[i],
            xerr=x_error[i],
            yerr=y_error[i],
            fmt="none",
            lw=1,
            color=color,
            capsize=2,
        )

    if xlim:
        plt.xlim(xlim[0], xlim[1])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    # for i in range(1, 20):
    #    plt.vlines(i * 5, min(y), max(y), color="black", ls="--")
    plt.xlabel(Metrics_performance[0])
    plt.ylabel(Metrics_performance[1])
    plt.legend(loc=0, ncol=2, fontsize=8)
    plt.show()


def barplot_result(
    dict_perf,
    Name_metrics,
    Name_candidates,
    Names_contextes=None,
    colors=None,
    xlim=None,
    ylim=None,
    names=None,
    target=(None, None),
    figsize=(15, 15),
    n_ctx=None,
    swap=None,
    save_path=None,
    link_lim=False,
    k_size=1):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        dict_perf ([type]): [description]
        colors ([type]): [description]
    """
    if Names_contextes is None:
        Names_contextes = [""]

    metric_numbers = len(Name_metrics)
    candidates_numbers = len(Name_candidates)
    contextes_numbers = len(Names_contextes)
    figshape = (contextes_numbers, metric_numbers)

    if colors is None:
        colors = [
            plt.get_cmap("jet", candidates_numbers)(i)
            for i in range(candidates_numbers)
        ]
    figsize=figsize[0]*k_size,figsize[1]*k_size
    if figshape is None:
        figshape = (1, metric_numbers)

    plt.figure(figsize=figsize)
    plt.style.use('seaborn-darkgrid')

    X_loc = np.zeros((contextes_numbers, candidates_numbers, metric_numbers, 2))
    for n_ctx, ctx in enumerate(Names_contextes):
        for n_cddt, candidate in enumerate(Name_candidates):
            for n_mtc, metric in enumerate(Name_metrics):
                perf = dict_perf[candidate][metric]
                if contextes_numbers == 1:
                    X_loc[n_ctx, n_cddt, n_mtc] = (np.mean(perf[2]), np.mean(perf[3]))
                else:
                    X_loc[n_ctx, n_cddt, n_mtc] = (perf[2, n_ctx], perf[3, n_ctx])
                # print(n_ctx, n_cddt, n_mtc, X_loc[n_ctx, n_cddt, n_mtc, 0])
    column_order = [Name_candidates, Names_contextes, Name_metrics]

    if not (swap is None):
        column_order = [column_order[i] for i in swap]
        X_loc = np.moveaxis(X_loc, (0, 1, 2), swap)

    Name_lignes = column_order[0]
    Name_subfigs = column_order[1]
    Name_figs = column_order[2]

    aux_baplot(
        X_loc,
        Name_lignes,
        Name_subfigs,
        Name_figs,
        colors,
        ylim,
        xlim,
        target,
        figshape,
        link_lim,
        k_size)
    if not (save_path is None):
        plt.savefig(save_path)
    plt.show()


def aux_baplot(
    X_loc,
    Name_ligne,
    Name_subfig,
    Name_fig,
    colors,
    ylim=None,
    xlim=None,
    target=(None, None),
    figshape=None,
    link_lim=False,
    k_size=1):

    y = np.arange(len(Name_ligne))[::-1]
    ligne_numbers = len(Name_ligne)
    fig_numbers = len(Name_fig)
    for n_subfig, name_subfig in enumerate(Name_subfig):
        for n_fig, name_fig in enumerate(Name_fig):
            if n_subfig == -1:
                n_subfig = 0
            ax = plt.subplot(
                figshape[0], figshape[1], n_subfig * fig_numbers + n_fig + 1
            )
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            
            if n_subfig == 0:
                plt.title(name_fig, fontsize=15*k_size)

            if n_fig == fig_numbers - 1:
                cbar = plt.colorbar(
                    mpl.cm.ScalarMappable(cmap=plt.get_cmap("jet")),
                    fraction=0.01,
                    aspect=0.01,
                )

                cbar.ax.set_yticklabels(labels=[])
                cbar.set_label(name_subfig, rotation=270, fontsize=16*k_size)

            if n_fig == 0:
                plt.yticks(y, Name_ligne, fontsize=14*k_size)
            else:
                plt.yticks(y, [])

            for n_ligne, name_ligne in enumerate(Name_ligne):
                plt.errorbar(
                    X_loc[n_subfig, n_ligne, n_fig, 0],
                    y[n_ligne],
                    xerr=X_loc[n_subfig, n_ligne, n_fig, 1],
                    fmt="ok",
                    lw=1.5*k_size,
                    marker="d",
                    markersize=6*k_size,
                    capsize=5*k_size,
                    color=colors[n_ligne],
                )

            if not (target[0] is None):
                plt.vlines(
                    target[0],
                    y.min() - 0.5,
                    y.max() + 0.5,
                    color="red",
                    ls="--",
                    label="target",
                )
            plt.xticks(fontsize=12*k_size*0.8)
            plt.ylim(-1 + 0.6, ligne_numbers - 0.6)
            if (link_lim) & (xlim[n_fig] is None):
                x_lim_min = np.min(X_loc[:, :, n_fig, 0] - X_loc[:, :, n_fig, 1])

                x_lim_max = np.max(X_loc[:, :, n_fig, 0] + X_loc[:, :, n_fig, 1])
                xlim[n_fig] = (
                    x_lim_min - 0.05 * np.abs(x_lim_min),
                    x_lim_max + 0.05 * np.abs(x_lim_max),
                )

            if not (xlim is None):
                if not (xlim[n_fig] is None):
                    plt.xlim(xlim[n_fig][0], xlim[n_fig][1])

            if not (ylim is None):
                if not (ylim[n_fig] is None):
                    plt.xlim(ylim[n_fig][0], ylim[n_fig][1])
            if n_subfig != (len(Name_subfig) - 1):
                plt.xticks(labels=None, fontsize=0)
        plt.subplots_adjust(wspace=0.03, hspace=0.06)
        # plt.tight_layout()
        plt.yticks(y, [], fontsize=12*k_size)


def barplot_ctx(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    list_names_ctx=None,
    target=(None, None),
    figshape=None,
    figsize=(15, 15),
    save_path=None,
):
    """Perform multi barplot visualisation"""

    if figshape == None:
        carre = int(np.ceil(np.sqrt(len(Metrics_performance))))
        figshape = (carre, carre)

    if names is None:
        names = candidates

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    for nn, metrics in enumerate(Metrics_performance):
        ax = plt.subplot(figshape[0], figshape[1], nn + 1)

        perf_res = dict_perf[candidates[0]][metrics][0]
        if type(perf_res) in [list, np.ndarray]:
            n_ctx = len(perf_res)
        else:
            n_ctx = 1

        y = np.arange(n_ctx)
        set_off_ctx = 0.5 / len(candidates)
        X_loc = np.zeros((len(candidates), 2, n_ctx))
        for n, candidate in enumerate(candidates):
            perf = dict_perf[candidate][metrics]
            X_loc[n][0] = perf[2]
            X_loc[n][1] = perf[3]
        for n, candidate in enumerate(candidates):
            for i_ctx in range(n_ctx):
                if i_ctx == 0:
                    plt.bar(
                        y[i_ctx] + set_off_ctx * (n),
                        X_loc[n, 0, i_ctx],
                        width=set_off_ctx * 0.9,
                        color=colors[n],
                        label=candidate,
                    )
                else:
                    plt.bar(
                        y[i_ctx] + set_off_ctx * (n),
                        X_loc[n, 0, i_ctx],
                        width=set_off_ctx * 0.9,
                        color=colors[n],
                    )

                plt.errorbar(
                    y[i_ctx] + set_off_ctx * (n),
                    X_loc[n, 0, i_ctx],
                    yerr=X_loc[n, 1, i_ctx] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    zorder=10,
                    color="black",
                )
        if list_names_ctx is None:
            names_ctx = y
        else:
            names_ctx = list_names_ctx[nn]
        if nn == 0:
            leg = ax.legend(loc=0, fontsize=16, framealpha=0.5)
            frame = leg.get_frame()
            frame.set_facecolor("gray")
        # plt.yticks(y, [], fontsize=20)
        ax.set_xticks(y + set_off_ctx, names_ctx, fontsize=14)
        ax.set_ylabel(metrics, fontsize=14)

        if not (ylim is None):
            if not (ylim[0] is None):
                ax.set_ylim(ylim[k][0], ylim[k][1])

    if not (target[0] is None):
        plt.vlines(
            target[0],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )

    if not (save_path is None):
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def barplot_result_ctx(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    target=(None, None),
    figsize=(15, 15),
    save_path=None,
):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        dict_perf ([type]): [description]
        colors ([type]): [description]
    """

    if names is None:
        names = candidates

    n_ctx = int(len(Metrics_performance) / 2)

    set_off_ctx = 0.5 / n_ctx

    X_loc = np.zeros((len(candidates), len(Metrics_performance) * 2))

    for n, candidate in enumerate(candidates):
        for nn, m in enumerate(Metrics_performance):
            perf = dict_perf[candidate][m]
            X_loc[n][nn * 2] = perf[2]
            X_loc[n][nn * 2 + 1] = perf[3]

    y = np.arange(len(candidates))[::-1]

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    plt.subplot(1, 2, 1)
    plt.title("PINAW (sharpness)", fontsize=28)
    name_ctx = ["All", "low-var", "mid-var", "high-var"]
    for n, candidate in enumerate(candidates):
        for m in range(n_ctx):
            if n == 0:
                plt.errorbar(
                    X_loc[:, m * 2][n],
                    y[n] + set_off_ctx * (m),
                    xerr=X_loc[:, m * 2 + 1][n] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    color=colors[n],
                    label=name_ctx[m],
                )
            else:
                plt.errorbar(
                    X_loc[:, m * 2][n],
                    y[n] + set_off_ctx * (m),
                    xerr=X_loc[:, m * 2 + 1][n] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    color=colors[n],
                )

    if not (target[0] is None):
        plt.vlines(
            target[0],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )
    plt.yticks(y, names, fontsize=24)
    plt.xlabel("← best", fontsize=24)
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.legend(loc=1, fontsize=15)
    plt.ylim(y.min() - 0.2, y.max() + 0.5)

    plt.subplot(1, 2, 2)
    plt.title("PICP (Coverage)", fontsize=24)
    for n, candidate in enumerate(candidates):
        for m in range(n_ctx):
            ind = m + n_ctx
            plt.errorbar(
                X_loc[n, ind * 2],
                y[n] + set_off_ctx * (m),
                xerr=X_loc[n, ind * 2 + 1] / 2,
                fmt="ok",
                lw=2,
                marker="d",
                markersize=6,
                capsize=6,
                color=colors[m],
            )
    if not (target[1] is None):
        plt.vlines(
            target[1],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )

    plt.xticks(fontsize=20)
    plt.xlabel("best →", fontsize=24)
    plt.tight_layout()
    plt.ylim(y.min() - 0.2, y.max() + 0.5)
    plt.yticks(y, [], fontsize=20)
    if not (save_path is None):
        plt.savefig(save_path)
    plt.show()

    # Data_processor for preprocessed data stored in a dict


class TimeSeries_from_dict:
    def __init__(self, path_file, **kwargs):
        """Read preprocessed  dataset stored in a dict with :
        'X' : Input
        'Y' : Target
        'context' : Additional information
        'train' : Boolean array for train split.
        'X_split' : Additional split information for cross-validation.
        """
        self.path_file = path_file

    def process(self, **kwargs):
        """Load dict at path_fill"""
        self.dict_dataset = pickle.load(open(self.path_file, "rb"))

    def get_data(self, **kwargs):
        """Provide dataset as list of array : [X,y,context,train,test,X_split]"""
        X = self.dict_dataset["X"]
        y = self.dict_dataset["Y"].reshape(len(X), -1)
        context = self.dict_dataset["context"]
        train = self.dict_dataset["train"]
        test = np.invert(train)
        X_split = self.dict_dataset["X_split"]
        return (X, y, context, train, test, X_split)

    def split_train_test(self, split=None):
        """Provide Train and Test data using predifine split or condition"""
        X = self.dict_dataset["X"]
        y = self.dict_dataset["Y"]
        context = self.dict_dataset["context"]
        X_split = self.dict_dataset["X_split"]
        if split == None:
            train = self.dict_dataset["train"]
            test = np.invert(train)
        else:
            train = X_split <= 1
            test = np.invert(train)

        return (X[train], X[test], y[train], y[test], context[train], context[test])

    def split_fit_calib(self, **kwargs):
        """None"""
        return ()


def revoring_models_outputs(storing, list_name_model, list_list_cv_attack, ind_ctx=-1,only_test=True):
    list_y = []
    list_set_model_outputs = []
    list_split = []
    list_context = []
    list_flag = []
    cv_list = get_cv_list(storing)
    for n, name_model in enumerate(list_name_model):
        list_model_outputs = []
        for cv, list_cv_attack in enumerate(list_list_cv_attack):
            if n == 0:  # Storage of dataset info the 1rst time.
                X, y, split, context, _, _ = get_dataset(
                    storing, list_cv_attack[0], cv_list
                )
                if(only_test):
                    flag = split != 1
                else:
                    flag = split < 10000
                list_split.append(split)   
                list_context.append(context[flag, ind_ctx])
                list_y.append(y[flag])
                list_flag.append(flag)
            else:
                flag = list_flag[cv]
            list_output = []
            for cv_attack in list_cv_attack:
                output = get_model_result(storing, name_model, cv_attack)
                selected_output = apply_mask(output, flag)
                list_output.append(selected_output)
            list_model_outputs.append(list_output)
        list_set_model_outputs.append(list_model_outputs)
    return (list_y, list_set_model_outputs, list_context,list_split)


def analyse_data_generator(dataset_generator1, n_ctx=-1):
    print("% train for each sub sample for each set")
    for n, (_, _, split_, context_, _, cv_name) in enumerate(dataset_generator1):
        train_val = []
        test_val = []
        drop_val = []
        for i in list(set(context_[:, n_ctx])):
            flag = context_[:, n_ctx] == i
            train_val.append((split_[flag] == 1).sum() / len(split_) * 100)
            test_val.append((split_[flag] == 0).sum() / len(split_) * 100)
            drop_val.append((split_[flag] == -1).sum() / len(split_) * 100)
        print(
            cv_name,
            " len",
            len(split_),
            "%train/test/drop:",
            np.round(train_val, 1),
            np.round(test_val, 1),
            np.round(drop_val, 1),
        )
