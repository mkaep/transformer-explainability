
import argparse
import datetime
import json
import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import pm4py

from main import dict_dir_name, df_dir_name
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.models.helper import Predictor
from processtransformer.util.trace_util import extract_traces_from_xes
from processtransformer.xai.main import load_model_and_dicts_via_dirs
from processtransformer.xai.metrics.attn_feature_importance import eval_attn_vs_bb_feature_importance
from processtransformer.xai.metrics.attn_masking import eval_attn_vs_trace_masking
from processtransformer.xai.metrics.attn_uniform_weights import eval_attn_uniform_weights
from processtransformer.xai.metrics.common import save_df_dict_to_disk, save_run_config
from train_model import TrainNextActivityModel
from processtransformer.util.compressor import decompress
from processtransformer.util.types import Trace
from processtransformer.xai.trace_series_generators import prefix_trace_series

plt.rcParams["figure.figsize"] = (5.4 * 0.9, 4)
plt.rcParams["svg.fonttype"] = 'none'
plt.rcParams["font.size"] = 11.0


def run_through_nn_metrics(nn_config: TrainingConfiguration):
    res_dir = nn_config.data_source.result_dir
    save_run_config(nn_config, res_dir)

    df_dict = {
        'attn_uniform_weights_jsd_tvd': [],
        'attn_uniform_weights_jsd_tvd_summary': [],
        'attn_uniform_weights_attn': [],
        'attn_uniform_weights_attn_summary': [],
        'attn_feature_importance': [],
        'attn_feature_importance_summary': [],
        'attn_masking': [],
        'attn_masking_summary': [],
    }

    nn_dir = os.path.join(nn_config.data_source.result_dir, TrainNextActivityModel.model_directory)
    dict_dir = os.path.join(nn_config.data_source.result_dir, dict_dir_name)
    df_dir = os.path.join(nn_config.data_source.result_dir, df_dir_name)
    processed_df = decompress(os.path.join(df_dir, 'processed_df'))
    train_processed_df = decompress(os.path.join(df_dir, 'train_processed_df'))
    model, x_dict, y_dict = load_model_and_dicts_via_dirs(nn_dir, dict_dir)
    predictor = Predictor(model, x_dict, y_dict)

    event_log = pm4py.read_xes(nn_config.prefix_and_y_true_log)
    traces = extract_traces_from_xes(nn_config.prefix_and_y_true_log)
    traces = [trace for trace_list in [list(prefix_trace_series(t)) for t in traces] for trace in trace_list]
    traces = _pick_random_traces(traces, 2000, 123)

    metrics_dir = os.path.join(nn_config.data_source.result_dir, 'metrics')

    # Attention masking
    mask_res_dir = os.path.join(metrics_dir, 'attn_masking')
    os.makedirs(mask_res_dir, exist_ok=True)
    attn_masking_metrics = eval_attn_vs_trace_masking(predictor, traces,
                                                      result_dir=mask_res_dir)
    df_dict['attn_masking'].append(attn_masking_metrics.df)
    df_dict['attn_masking_summary'].append(attn_masking_metrics.df_summary)

    # Attention uniform and seeds
    res_dir = os.path.join(metrics_dir, 'attn_uniform_weights')
    os.makedirs(res_dir, exist_ok=True)
    attn_uniform_metrics = eval_attn_uniform_weights(nn_config,
                                                     traces=traces,
                                                     result_dir=res_dir,
                                                     predictor_base=predictor,
                                                     processed_df=processed_df,
                                                     train_processed_df=train_processed_df)
    df_dict['attn_uniform_weights_jsd_tvd'].append(attn_uniform_metrics.jsd_tvd_df)
    df_dict['attn_uniform_weights_jsd_tvd_summary'].append(attn_uniform_metrics.jsd_tvd_df_summary)
    df_dict['attn_uniform_weights_attn'].append(attn_uniform_metrics.attn_df)
    df_dict['attn_uniform_weights_attn_summary'].append(attn_uniform_metrics.attn_df_summary)

    # Attention feature-importance
    attn_vs_bb_fi_metrics = eval_attn_vs_bb_feature_importance(traces, predictor, event_log)
    df_dict['attn_feature_importance'].append(attn_vs_bb_fi_metrics.df)
    df_dict['attn_feature_importance_summary'].append(attn_vs_bb_fi_metrics.df_summary)

    save_df_dict_to_disk(df_dict, metrics_dir)


def main():
    parser = argparse.ArgumentParser(description="Process Transformer - NN-Metrics.")
    parser.add_argument("--nn_config", type=str, required=True,
                        help='Path of the NN-file (data, parameters).')
    args = parser.parse_args()

    main_with_args(args.nn_config)


def main_with_args(nn_config_path):
    start = datetime.datetime.now()

    with open(nn_config_path, 'r', encoding="utf8") as f:
        nn_config = TrainingConfiguration.from_dict(json.load(f))
        run_through_nn_metrics(nn_config)

    end = datetime.datetime.now()
    time_str = f'Entire evaluation took {end - start} (hh:mm:ss).\n' \
               f'Started at {start.strftime("%c")}, finished at {end.strftime("%c")}.'

    os.makedirs(nn_config.data_source.result_dir, exist_ok=True)
    with open(os.path.join(nn_config.data_source.result_dir, "run_info.txt"), 'w', encoding="utf8") as f:
        f.write(time_str)

    print(time_str)


def _pick_random_traces(trace_pred_list: typing.List[Trace],
                        max_len: int, seed: int) -> typing.List[Trace]:
    # Unique traces only
    trace_pred_list = list(set([tuple(trace) for trace in trace_pred_list]))
    trace_pred_list = [list(trace) for trace in trace_pred_list]

    # Shuffle indices
    rng_traces = np.random.Generator(np.random.PCG64(seed))
    trace_indices = rng_traces.integers(low=0, high=len(trace_pred_list), size=min(len(trace_pred_list), max_len))

    # Pick first indices + restrict size
    trace_pred_list = [trace_pred_list[i] for i in trace_indices]

    return trace_pred_list


if __name__ == '__main__':
    main()
