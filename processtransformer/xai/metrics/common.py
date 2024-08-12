
import json
import os
import typing

import numpy as np
import pandas as pd

from processtransformer.data_models.explaining_model import ExplainingModel
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.util.types import Trace, RelationsDict


def get_jaccard_value(local_trace: Trace, dct1: RelationsDict, dct2: RelationsDict) -> float:
    keys1 = set(dct1.keys())
    keys2 = set(dct2.keys())
    common_keys = keys1.intersection(keys2)
    unmatched_keys = keys1.difference(keys2)
    false_keys = keys2.difference(keys1)

    # See https://en.wikipedia.org/wiki/Jaccard_index
    jaccard = 0
    for key in common_keys:
        right_side1 = set(dct1[key])
        right_side2 = set(dct2[key])
        # 1 if both match perfectly.
        right_side_len = len(right_side1.union(right_side2))
        if right_side_len == 0:
            continue
        jaccard += len(right_side1.intersection(right_side2)) / right_side_len

    if len(common_keys) == 0:
        return 0.0
    # Divide by count
    jaccard /= len(common_keys)
    # Decrease if unmatched/false keys exist
    jaccard *= len(common_keys) / (len(common_keys) + len(unmatched_keys) + len(false_keys))

    return jaccard


def get_relaxed_jaccard_value(dct1: RelationsDict, dct2: RelationsDict) -> float:
    left1 = set([event for tpl in dct1.keys() for event in tpl])
    left2 = set([event for tpl in dct2.keys() for event in tpl])
    right1 = set([event for tpl in dct1.values() for event in tpl])
    right2 = set([event for tpl in dct2.values() for event in tpl])

    len_left_side = len(left1.union(left2))
    if len_left_side == 0:
        left_jaccard = 0.0
    else:
        left_jaccard = len(left1.intersection(left2)) / len_left_side

    len_right_side = len(right1.union(right2))
    if len_right_side == 0:
        right_jaccard = 0.0
    else:
        right_jaccard = len(right1.intersection(right2)) / len_right_side

    return (left_jaccard + right_jaccard) / 2.0


def dct_to_vector(dct: typing.Dict[str, typing.Any]) -> typing.List[typing.Any]:
    # (event, importance)
    dct = [(key, value) for key, value in dct.items()]
    # Sort alphabetically
    dct.sort(key=lambda x: x[0])
    return [pair[1] for pair in dct]


def kullback_leibler_divergence(a, b):
    # See https://datascience.stackexchange.com/a/9264
    # Cross-checked with https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Basic_example
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    a += 1e-10
    b += 1e-10

    return np.sum(a * np.log(a / b))


def jensen_shannon_divergence(a, b):
    # See https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    c = (a + b) / 2.0

    return 0.5 * kullback_leibler_divergence(a, c) + 0.5 * kullback_leibler_divergence(b, c)


def total_variation_distance(y1, y2):
    # See paper "Attention is not Explanation"
    y1 = np.asarray(y1, dtype=np.float)
    y2 = np.asarray(y2, dtype=np.float)
    return 0.5 * np.sum(np.abs(y1 - y2))


def dct_to_ordered_list(dct: typing.Dict[str, float]) -> typing.List[float]:
    # Example: [(B, 1.0), (A, 0.5), (D, 0.0), (C, 0.0)]
    lst = [(key, value) for key, value in dct.items()]
    # Sort alphabetically -> [(A, 0.5), (B, 1.0), (C, 0.0), (D, 0.0)]
    lst.sort(key=lambda x: x[0], reverse=False)
    # --> [0.5, 1.0, 0.0, 0.0]
    return [pair[1] for pair in lst]


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    df_summary = df.quantile(quantiles)
    df_summary.index = [f'p-{q:.2f}' for q in quantiles]
    df_summary = df_summary.transpose()
    return df_summary


file_endings = ['.png', '.svg']


def save_df_dict_to_disk(df_dict, metrics_dir):
    os.makedirs(metrics_dir, exist_ok=True)
    for metric, metric_df_list in df_dict.items():
        if len(metric_df_list) == 0:
            continue
        df = pd.concat(metric_df_list)
        df.to_csv(os.path.join(metrics_dir, f'{metric}.csv'))
        df.to_latex(os.path.join(metrics_dir, f'{metric}.tex'), float_format="%.3f")


def save_run_config(run_config: typing.Union[TrainingConfiguration, ExplainingModel],
                    res_dir):
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, run_config.name + ".json"), 'w', encoding="utf8") as f:
        json.dump(run_config.to_dict(), f, indent=2)
