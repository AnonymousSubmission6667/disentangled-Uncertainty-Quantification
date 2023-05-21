import os
import pickle
import pandas
from pathlib import Path


def Extract_dict(dictionaire, list_keys):
    """Extract list of values of dictionaire from list_of_keys
    return None if keys isn't in dictionaire


    Args:
        dictionaire (dict): dictionary
        list_keys (str list): list of keys

    Returns:
        list_of_values or values: return list of values,
            if len(list)=1 return value
    """
    list_extract = []
    for keys in list_keys:
        if keys in list(dictionaire.keys()):
            list_extract.append(dictionaire[keys])
        else:
            list_extract.append(None)

    if len(list_extract) == 1:
        list_extract = list_extract[0]
    return list_extract


def write_function(value, filename):
    """Writing API : write values as  pickle or csv filename depending of value type.

    Args:
        values (_type_): values to store in a file
        filename (_type_): path of files

    Returns:
        None
    """
    if type(value) == pandas.core.frame.DataFrame:
        write_type = "pandas"
    else:
        write_type = "pickle"

    if write_type == "pickle":
        pickle.dump(value, open(str(filename) + ".p", "wb"))

    elif write_type == "pandas":
        file = open(str(filename) + ".csv", "wb")
        value.to_csv(file)
        file.close()
    return ()


def read_function(filename):
    """Reading file API, load from csv or pickle or folde

    Args:
        filename (_type_): file path

    Returns:
        object: - python object if store in .p
                - pandas if store in .csv
                - Path file if filename is folder
    """
    values = None
    filename_csv = Path(str(filename) + ".csv")
    filename_p = Path(str(filename) + ".p")
    flag_csv = False
    flag_p = False

    if filename_csv.is_file():
        read_type = "pandas"
        flag_csv = True

    if filename_p.is_file():
        read_type = "pickle"
        flag_p = True

    if flag_csv & flag_p:
        print(
            "warning csv and pickle with same name : "
            + filename
            + ". priority to pickle file"
        )

    if filename.is_file() or flag_p or flag_csv:
        if read_type == "pickle":
            file = open(str(filename) + ".p", "rb")
            values = pickle.load(file)
            file.close()
        elif read_type == "pandas":
            values = pandas.read_csv(open(str(filename) + ".csv", "rb"))
    if filename.is_dir():
        values = str(filename)
    return values


def write(storing, keys, values):
    """write python object stored in storing (str_path or dict)
    at keys ('tree path') location

    Args:
        storing (dict or str_path): Storage : str_path or dict
        keys (str list): ('tree path') location
        values (object): python object to write

    Returns:None
    """

    if type(storing) == dict:
        mode = "dict"
    elif type(storing) == str:
        mode = "file"
    else:
        print("storing have to be a 'dict' or 'str path_file'", type(storing))

    if mode == "dict":
        sub_dict = storing
        for k in keys[:-1]:
            if not (k in list(sub_dict.keys())):
                sub_dict[k] = {}
            sub_dict = sub_dict[k]
        sub_dict[keys[-1]] = values

    elif mode == "file":
        full_path = storing
        for k in keys[:-1]:
            full_path += "/" + k
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        filename = Path(full_path + "/" + keys[-1])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_function(values, filename)


def read(storing, keys):
    """read python object stored in storing (str_path or dict)
    at keys ('tree path') location

    Args:
        storing (dict or str_path): store object
        keys (str list): 'tree path' that give location

    Returns:
        objet : stored objet in storing at keys location
    """

    if type(storing) == dict:
        mode = "dict"
    elif type(storing) == str:
        mode = "file"
    else:
        print("storing have to be a 'dict' or 'str path_file', not :", type(storing))

    if mode == "dict":
        sub_dict = storing
        for n, k in enumerate(keys):
            if k in list(sub_dict.keys()):
                sub_dict = sub_dict[k]
                if n + 1 == len(keys):
                    return sub_dict
            else:
                return None

    elif mode == "file":
        if type(storing) != str:
            print("ERROR : storing is not a path")

        full_path = storing
        for n, k in enumerate(keys):
            full_path += "/" + k
        filename = Path(full_path)
        return read_function(filename)

    else:
        print("mode have to be 'dict' or 'file'")


def dict_to_folders(storing_dict):
    return ()


def dict_to_folders(storing_path):
    return ()


# Base functionality
def get_data_generator(storing):
    # Load data_generator stored in storing
    return read(storing, ["data", "generator"])


def store_data_generator(storing, data_generator):
    # Store data_generator in storing
    write(storing, ["data", "generator"], data_generator)


def store_model_parameters(storing, name_model, parameters):
    # Store data_generator in storing
    write(storing, ["result", name_model, "parameters"], parameters)


def get_cv_list(storing):
    # Load data_generator stored in storing
    return read(storing, ["data", "cv_list"])


def store_cv_list(storing, cv_list):
    # Load data_generator stored in storing
    write(storing, ["data", "cv_list"], cv_list)


def get_model_result(storing, name_model, cv):
    # Load model_result of a cros-val set "cv" stored in storing
    return read(storing, ["result", name_model, cv, "output"])


def get_dataset(storing, cv_name, cv_list=None):
    # Load cros-val data "cv" stored in storing
    if cv_list is None:
        cv_list = get_cv_list(storing)
    cv = cv_list.index(cv_name)
    dataset = get_data_generator(storing)[cv]
    return dataset
