import os
import pickle
import pandas
import numpy
from pathlib import Path


def Extract_dict(dictionaire, list_str):
    list_extract = []
    for str_name in list_str:
        if str_name in list(dictionaire.keys()):
            list_extract.append(dictionaire[str_name])
        else:
            list_extract.append(None)

    if len(list_extract) == 1:
        list_extract = list_extract[0]
    return list_extract


def write_function(values, filename):
    if type(values) == pandas.core.frame.DataFrame:
        write_type = "pandas"
    else:
        write_type = "pickle"

    if write_type == "pickle":
        pickle.dump(values, open(str(filename) + ".p", "wb"))

    elif write_type == "pandas":
        file = open(str(filename) + ".csv", "wb")
        values.to_csv(file)
        file.close()
    return ()


def read_function(filename):
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
        return None


def read(storing, keys):

    if type(storing) == dict:
        mode = "dict"
    elif type(storing) == str:
        mode = "file"
    else:
        print("storing have to be a 'dict' or 'str path_file'")

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
