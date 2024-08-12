

import typing
import typing as tp

import numpy as np

from processtransformer.util.types import EventValueList, EventValueDict, GeneralList


def dict_to_mat(dict_of_lists) -> typing.Tuple[typing.List[str], typing.List[str], np.ndarray]:
    """Returns x_labels, y_labels, mat"""
    # Get labels
    y_labels = dict_of_lists.keys()
    y_labels = list(y_labels)
    y_labels.sort()
    x_labels = set()
    for key in y_labels:
        dict_of_lists[key] = dict(dict_of_lists[key])
        x_labels = x_labels.union(set(dict_of_lists[key].keys()))
    x_labels = list(x_labels)
    x_labels.sort()

    # Construct matrix
    mat = np.zeros(shape=(len(y_labels), len(x_labels)))
    for i, key in enumerate(y_labels):
        for j, key_col in enumerate(x_labels):
            val = 0.0
            if key_col in dict_of_lists[key].keys():
                val = dict_of_lists[key][key_col]
            mat[i][j] = val

    return x_labels, y_labels, mat


def compare_list_of_keys(list_dict1: tp.Union[EventValueList, EventValueDict],
                         list_dict2: tp.Union[EventValueList, EventValueDict],
                         return_as_sorted_list=False):
    # To dict
    if isinstance(list_dict1, tp.List):
        list_dict1 = tuple_list_to_dict(list_dict1)
    if isinstance(list_dict2, tp.List):
        list_dict2 = tuple_list_to_dict(list_dict2)

    # Normalize to 1.0
    normalize_dict_to_one(list_dict1)
    normalize_dict_to_one(list_dict2)

    # Build difference
    diff_dict: EventValueDict = dict()
    all_keys = set(list_dict1.keys()).union(set(list_dict2.keys()))
    for key in all_keys:
        val1 = 0.0
        if key in list_dict1.keys():
            val1 = list_dict1[key]
        val2 = 0.0
        if key in list_dict2.keys():
            val2 = list_dict2[key]
        diff_dict[key] = val1 - val2

    if return_as_sorted_list:
        diff_list = list(diff_dict.items())
        diff_list.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return diff_list

    return diff_dict


def normalize_dict_to_one(list_dict: EventValueDict):
    """
    in-place operation
    """
    sum_list = sum(list(list_dict.values()))
    for key in list_dict.keys():
        list_dict[key] /= sum_list


def tuple_list_to_dict(tuple_list: GeneralList) -> tp.Dict[tp.Any, tp.Any]:
    return {key: value for key, value in tuple_list}


def dict_to_tuple_list(dct: tp.Dict[tp.Any, tp.Any]) -> GeneralList:
    return [(key, value) for key, value in dct.items()]
