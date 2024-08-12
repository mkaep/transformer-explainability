
import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import sys
import threading
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm

from main import df_dir_name
from processtransformer.data_models.explaining_model import ExplainingModel
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.models.helper import Predictor
from processtransformer.util.trace_util import extract_traces_and_ypred_from_xes
from processtransformer.util.types import Trace, RelationsDict, YEvent
from processtransformer.xai.explainer import Explainer
from processtransformer.xai.main import load_model_and_dicts, map_output_to_viz
from processtransformer.xai.metrics.common import save_df_dict_to_disk
from processtransformer.xai.metrics.common import save_run_config
from processtransformer.xai.metrics.common import summarize_df
from processtransformer.xai.metrics.compactness_co07 import CompactnessCo07Wrapper
from processtransformer.xai.metrics.completeness_co02 import CompletenessCo02Wrapper
from processtransformer.xai.metrics.consistency_co03 import eval_consistency, PackedOutput, ConsistencyMetrics
from processtransformer.xai.metrics.continuity_co04 import ContinuityCo04Wrapper
from processtransformer.xai.metrics.contrastivity_co05 import ContrastivityCo05Wrapper
from processtransformer.xai.metrics.correctness_co01 import CorrectnessCo01Wrapper
from processtransformer.xai.metrics.feature_importance import BlackboxFIWrapper
from processtransformer.xai.metrics.trace_generation import RealEnvWrapper, ArtEnvWrapper, MaybeGenWrapper
from processtransformer.xai.trace_series_generators import prefix_pred_and_event_series
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData
from processtransformer.util.compressor import compress, decompress
from train_model import TrainNextActivityModel

plt.rcParams["figure.figsize"] = (5.4 * 0.9, 4)
plt.rcParams["svg.fonttype"] = 'none'
plt.rcParams["font.size"] = 11.0

run_co01_flag = True
run_co02_flag = True
run_co03_flag = True
run_co04_flag = True
run_co05_flag = True
run_co07_flag = True

max_num_of_traces = 100
max_num_of_prefixes = 400
xai_prefix_count = 100


class PseudoExplainableOutput(ExplainableOutputData):
    def __init__(self, explainer: Explainer, event_log: EventLog):
        super().__init__()
        self.explainer = explainer
        self.event_log = event_log

    @classmethod
    def supports_explanation_for_trace(cls) -> bool:
        return True

    def _get_explanation_for_trace(self, trace: Trace, transformer_info,
                                   ) -> typing.List[RelationsDict]:
        # Fit XAI locally
        out = self.explainer.explain_trace(trace, None, self.event_log)
        rel = filter_out_data_to_relations(out)

        if len(rel) > 0:
            # Get explanation for input trace
            return [r.get_explanation_for_trace(trace, None) for r in rel]

        raise ValueError('Explainer does not support explanation of a single trace with ExplainableOutputData')


class Progress:
    def __init__(self, total, logger, name, print_all_n=20):
        self.total = total
        self.logger = logger
        self.name = name
        self.print_all_n = print_all_n

        self.start = time.time()
        self.last_update = 0
        self.done = 0
        self.lock = threading.Lock()

    def update(self, delta):
        with self.lock:
            self.done += delta

            if self.done - self.last_update >= self.print_all_n:
                time_diff = time.time() - self.start
                self.logger.info(f'{self.name}: iterated over {self.done}/{self.total} ({_pretty_time(time_diff)})')
                self.last_update = self.done


def run_through_xai_metrics(xai_config: ExplainingModel,
                            logger: logging.Logger,
                            ):
    # Save config
    xai_res_dir = xai_config.result_dir
    save_run_config(xai_config, xai_res_dir)

    avail_threads = max(os.cpu_count() - 1, 1)  # leave on thread free for user/whatever
    model, x_dict, y_dict = load_model_and_dicts(xai_config)
    pred = Predictor(model, x_dict, y_dict)
    predictors = [pred for _ in range(avail_threads)]

    xai_result_dir = os.path.join(xai_res_dir, 'xai')
    os.makedirs(xai_result_dir, exist_ok=True)
    explainer = xai_config.explainer(model, x_dict, y_dict, xai_result_dir, **xai_config.explainer_kwargs)

    metrics_dir = os.path.join(xai_res_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    prefixed_traces, traces_ypreds = _prepare_traces_and_prefixes(xai_config)

    compress(prefixed_traces, os.path.join(metrics_dir, 'prefixed_traces'))
    compress(traces_ypreds, os.path.join(metrics_dir, 'traces_ypreds'))

    event_log = pm4py.read_xes(xai_config.prefix_and_y_true_log)
    dfg = pm4py.discover_dfg(event_log)

    # Run XAI and get output-data
    trace_sup = explainer.get_trace_support()
    rel_data = None
    if trace_sup.multi_trace:
        logger.info('Using multi-trace XAI')
        out_data = explainer.explain_multiple_traces(prefixed_traces[:xai_prefix_count], event_log)
        rel_data = filter_out_data_to_relations(out_data)
        map_output_to_viz(out_data)

    # Run metrics on XAI and NN

    traces_and_relations = compute_local_xai(event_log, explainer, rel_data, traces_ypreds)
    prefixed_traces_and_relations = compute_local_xai(event_log, explainer, rel_data, prefixed_traces)

    logger.info('Prepared traces')

    # Caching setup - separated by wrappers as caching is (currently) not thread-safe!
    # Should be thread-safe now. Performance-impact of locks unclear though.
    co01_art_env_wrapper = ArtEnvWrapper(event_log, dfg)
    CorrectnessCo01Wrapper.set_global_bb_wrapper(BlackboxFIWrapper(x_dict, event_log, pred,
                                                                   art_env_wrapper=co01_art_env_wrapper))
    CorrectnessCo01Wrapper.set_global_art_env_wrapper(co01_art_env_wrapper)

    CompletenessCo02Wrapper.set_global_env_wrapper(RealEnvWrapper(event_log, dfg))

    co04_art_env_wrapper = ArtEnvWrapper(event_log, dfg)
    ContinuityCo04Wrapper.set_global_bb_fi_wrapper(BlackboxFIWrapper(x_dict, event_log, pred,
                                                                     art_env_wrapper=co04_art_env_wrapper))
    ContinuityCo04Wrapper.set_global_maybe_gen_wrapper(MaybeGenWrapper(event_log, dfg))
    ContinuityCo04Wrapper.set_global_art_env_wrapper(co04_art_env_wrapper)

    co07_art_env_wrapper = ArtEnvWrapper(event_log, dfg)
    CompactnessCo07Wrapper.set_global_bb_wrapper(BlackboxFIWrapper(x_dict, event_log, pred,
                                                                   art_env_wrapper=co07_art_env_wrapper))

    def args_func_co02(_progress, _i_start, _i_stop, _local_dict, _pred):
        return (event_log, _pred, rel_data,
                traces_and_relations[_i_start:_i_stop], y_dict,
                _local_dict, _progress)

    _run_multithreaded_metric(args_func_co02, logger, 'Co02', metrics_dir, predictors,
                              run_co02, traces_and_relations, run_co02_flag)

    def args_func_co01(_progress, _i_start, _i_stop, _local_dict, _pred):
        return (event_log, _pred, prefixed_traces_and_relations[_i_start:_i_stop],
                rel_data, x_dict, _local_dict, _progress)

    _run_multithreaded_metric(args_func_co01, logger, 'Co01', metrics_dir, predictors,
                              run_co01, prefixed_traces_and_relations, run_co01_flag)

    def args_func_co04(_progress, _i_start, _i_stop, _local_dict, _pred):
        return (event_log, _pred, prefixed_traces_and_relations[_i_start:_i_stop],
                rel_data, y_dict, _local_dict, _progress)

    _run_multithreaded_metric(args_func_co04, logger, 'Co04', metrics_dir, predictors,
                              run_co04, prefixed_traces_and_relations, run_co04_flag)

    def args_func_co05(_progress, _i_start, _i_stop, _local_dict, _pred):
        return (event_log, _pred, prefixed_traces_and_relations[_i_start:_i_stop],
                rel_data, _local_dict, _progress)

    _run_multithreaded_metric(args_func_co05, logger, 'Co05', metrics_dir, predictors,
                              run_co05, prefixed_traces_and_relations, run_co05_flag)

    def args_func_co07(_progress, _i_start, _i_stop, _local_dict, _pred):
        return (event_log, _pred, prefixed_traces_and_relations[_i_start:_i_stop],
                rel_data, x_dict, y_dict, _local_dict, _progress)

    _run_multithreaded_metric(args_func_co07, logger, 'Co07', metrics_dir, predictors,
                              run_co07, prefixed_traces_and_relations, run_co07_flag)

    run_co03(event_log, logger, metrics_dir, prefixed_traces, xai_config, xai_result_dir, predictors[0])


def _run_multithreaded_metric(args_func, logger, metric_name, metrics_dir, predictors,
                              target, list_to_progress, run_flag):
    if not run_flag:
        return

    work = _get_index_pairs_for_n_threads(len(predictors), len(list_to_progress))
    threads = []
    progress = Progress(len(list_to_progress), logger, metric_name)
    logger.info(f'{metric_name}: Started')
    start = time.time()
    for (i_start, i_stop), pred in zip(work, predictors):
        local_dict = dict()
        # All arguments are either read-only or thread-safe
        t = threading.Thread(target=target,
                             args=args_func(progress, i_start, i_stop, local_dict, pred))
        t.start()
        threads.append((t, local_dict))
    for t, _ in threads:
        t.join()
    metric_dict = threads[0][1]
    for _, local_dict in threads[1:]:
        for k, v in local_dict.items():
            metric_dict[k] += v
    _summarize_dict(metric_dict)
    save_df_dict_to_disk(metric_dict, metrics_dir)
    logger.info(f'{metric_name}: Finished ({_pretty_time(time.time() - start)})')


def run_co03(event_log, logger, metrics_dir, prefixed_traces, xai_config, xai_result_dir, predictor):
    if not run_co03_flag:
        return

    start = time.time()
    logger.info('Co03: Started')
    train_dir = os.path.join(xai_config.neural_network_model_dir, os.path.pardir)
    training_configuration = None
    for file_path in glob.glob(os.path.join(train_dir, '*.json')):
        try:
            with open(file_path, 'r', encoding="utf8") as file:
                training_configuration = TrainingConfiguration.from_dict(json.load(file))
                break
        except Exception as e:
            continue
    temp_xai_dir = os.path.join(xai_result_dir, 'temp')

    nn_res_dir = training_configuration.data_source.result_dir
    df_dir = os.path.join(nn_res_dir, df_dir_name)
    processed_df = decompress(os.path.join(df_dir, 'processed_df'))
    train_processed_df = decompress(os.path.join(df_dir, 'train_processed_df'))

    def get_relations_output(_model, x_dct, y_dct) -> PackedOutput:
        os.makedirs(temp_xai_dir, exist_ok=True)
        local_xai = xai_config.explainer(_model, x_dct, y_dct, temp_xai_dir, **xai_config.explainer_kwargs)
        if local_xai.get_trace_support().multi_trace:
            local_out = local_xai.explain_multiple_traces(prefixed_traces, event_log)
            local_filtered = filter_out_data_to_relations(local_out)

            class LocalClass(PackedOutput):
                def explain_trace(self, _trace) -> typing.List[RelationsDict]:
                    return [out.get_explanation_for_trace(_trace, None) for out in local_filtered]

            return LocalClass()
        else:
            class LocalClass(PackedOutput):

                def explain_trace(self, _trace) -> typing.List[RelationsDict]:
                    # Fit XAI locally
                    out = local_xai.explain_trace(_trace, None, event_log)
                    rel = filter_out_data_to_relations(out)

                    if len(rel) > 0:
                        # Get explanation for input trace
                        return [r.get_explanation_for_trace(_trace, None) for r in rel]

                    raise ValueError(
                        'Explainer does not support explanation of a single trace with ExplainableOutputData')

            return LocalClass()

    if training_configuration is not None:
        diff_mean_path = os.path.join(nn_res_dir, 'co03', 'diff_mean.xz')
        diff_std_path = os.path.join(nn_res_dir, 'co03', 'diff_std.xz')
        model2_path = os.path.join(nn_res_dir, 'model2_co03')
        if os.path.exists(diff_mean_path) and os.path.exists(diff_std_path) and os.path.exists(model2_path):
            diff_mean = decompress(diff_mean_path)
            diff_std = decompress(diff_std_path)
            model2 = TrainNextActivityModel.load_model(model2_path)
            previous_output = ConsistencyMetrics(None, None, diff_mean, diff_std, None, model2)
        else:
            previous_output = None

        co03_metrics = eval_consistency(training_configuration, get_relations_output, event_log,
                                        prefixed_traces, logger=logger,
                                        predictor=predictor,
                                        processed_df=processed_df,
                                        train_processed_df=train_processed_df,
                                        previous_output=previous_output,
                                        )
        shutil.rmtree(temp_xai_dir)
        co03_dict = {'consistency_co03': [co03_metrics.df],
                     'consistency_co03_complete': [co03_metrics.complete_df]}
        save_df_dict_to_disk(co03_dict, metrics_dir)

        os.makedirs(os.path.dirname(diff_mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(diff_std_path), exist_ok=True)
        os.makedirs(model2_path, exist_ok=True)
        compress(co03_metrics.diff_mean, diff_mean_path)
        compress(co03_metrics.diff_std, diff_std_path)
        TrainNextActivityModel.save_tf_model(model2_path, co03_metrics.model2)

    logger.info(f'Co03: Finished ({time.time() - start} seconds)')


def run_co02(event_log, predictor, rel_data, traces_and_relations, y_dict, co02_dict, progress):
    co02_dict['completeness_co02_single'] = []
    co02_dict['completeness_co02_combined'] = []
    if rel_data is not None:
        co02_wrappers = [CompletenessCo02Wrapper(event_log, y_dict, predictor, rel) for rel in rel_data]

    for i, (trace, local_rel_data) in enumerate(traces_and_relations):
        if local_rel_data is not None:
            co02_wrappers = [CompletenessCo02Wrapper(event_log, y_dict, predictor, rel) for rel in local_rel_data]
        # noinspection PyUnboundLocalVariable
        for wrapper in co02_wrappers:
            _, _, single_df, combined_df = wrapper.eval(trace)
            co02_dict['completeness_co02_single'].append(single_df)
            co02_dict['completeness_co02_combined'].append(combined_df)

        progress.update(1)


def _prepare_traces_and_prefixes(xai_config, trace_max=max_num_of_traces, prefix_max=max_num_of_prefixes,
                                 trace_seed=372, prefix_seed=327):
    traces_ypreds = extract_traces_and_ypred_from_xes(xai_config.prefix_and_y_true_log)
    prefixed_traces = list(prefix_pred_and_event_series(traces_ypreds))

    traces_ypreds = _pick_random_traces(traces_ypreds, trace_max, trace_seed)
    prefixed_traces = _pick_random_traces(prefixed_traces, prefix_max, prefix_seed)

    return prefixed_traces, traces_ypreds


def _pick_random_traces(trace_pred_list: typing.List[typing.Tuple[Trace, YEvent]],
                        max_len: int, seed: int) -> typing.List[typing.Tuple[Trace, YEvent]]:
    # Unique traces only
    trace_pred_list = list(set([(tuple(trace), pred) for trace, pred in trace_pred_list]))
    trace_pred_list = [(list(trace), pred) for trace, pred in trace_pred_list]

    # Shuffle indices
    rng_traces = np.random.Generator(np.random.PCG64(seed))
    trace_indices = rng_traces.integers(low=0, high=len(trace_pred_list), size=min(len(trace_pred_list), max_len))

    # Pick first indices + restrict size
    trace_pred_list = [trace_pred_list[i] for i in trace_indices]

    return trace_pred_list


def _summarize_dict(dict_to_summarize):
    dict_appendix = dict()
    for metric, df_list in dict_to_summarize.items():
        if len(df_list) == 0:
            continue

        summary_df = pd.concat(df_list)
        summary_df = summarize_df(summary_df)

        dict_appendix[f'{metric}_summary'] = [summary_df]
    dict_to_summarize.update(dict_appendix)


def compute_local_xai(event_log, explainer, rel_data, traces_ypreds):
    traces_and_relations = []
    if rel_data is not None:
        traces_and_relations = list(zip([trace for trace, _ in traces_ypreds], [None] * len(traces_ypreds)))
    else:
        for trace, y_pred in tqdm(traces_ypreds, 'Running local XAI over traces'):
            out_data = explainer.explain_trace(trace, y_pred, event_log)
            local_rel_data = filter_out_data_to_relations(out_data)
            traces_and_relations.append((trace, local_rel_data))
    return traces_and_relations


def run_co07(event_log, predictor, traces_and_relations, rel_data, x_dict, y_dict, co07_dict, progress):
    co07_dict['compactness_co07'] = []
    if rel_data is not None:
        co07_wrappers = [CompactnessCo07Wrapper(predictor, x_dict, y_dict, event_log, rel) for rel in rel_data]
    for i, (trace, local_rel_data) in enumerate(traces_and_relations):
        if local_rel_data is not None:
            co07_wrappers = [CompactnessCo07Wrapper(predictor, x_dict, y_dict, event_log, rel)
                             for rel in local_rel_data]
        # noinspection PyUnboundLocalVariable
        for wrapper in co07_wrappers:
            compactness_metric = wrapper.eval(trace)
            co07_dict['compactness_co07'].append(compactness_metric.get_as_df())

        progress.update(1)


def run_co05(event_log, predictor, traces_and_relations, rel_data, co05_dict, progress):
    co05_dict['contrastivity_co05'] = []
    if rel_data is not None:
        co05_wrappers = [ContrastivityCo05Wrapper(event_log, predictor, rel) for rel in rel_data]
    for i, (trace, local_rel_data) in enumerate(traces_and_relations):
        if local_rel_data is not None:
            co05_wrappers = [ContrastivityCo05Wrapper(event_log, predictor, rel) for rel in local_rel_data]
        # noinspection PyUnboundLocalVariable
        for wrapper in co05_wrappers:
            contrastivity_metric = wrapper.eval(trace)
            co05_dict['contrastivity_co05'].append(contrastivity_metric.get_as_df())

        progress.update(1)


def run_co04(event_log, predictor, traces_and_relations, rel_data, y_dict, co04_dict, progress):
    co04_dict['continuity_co04'] = []
    if rel_data is not None:
        co04_wrappers = [ContinuityCo04Wrapper(event_log, predictor, rel, y_dict) for rel in rel_data]
    for i, (trace, local_rel_data) in enumerate(traces_and_relations):
        if local_rel_data is not None:
            co04_wrappers = [ContinuityCo04Wrapper(event_log, predictor, rel, y_dict) for rel in local_rel_data]
        # noinspection PyUnboundLocalVariable
        for wrapper in co04_wrappers:
            continuity_metric = wrapper.eval(trace)
            co04_dict['continuity_co04'].append(continuity_metric.get_as_df())

        progress.update(1)


def run_co01(event_log, predictor, traces_and_relations, rel_data, x_dict, co01_dict, progress):
    co01_dict['correctness_co01'] = []
    if rel_data is not None:
        co01_wrappers = [CorrectnessCo01Wrapper(x_dict, event_log, predictor, rel) for rel in rel_data]
    for i, (trace, local_rel_data) in enumerate(traces_and_relations):
        if local_rel_data is not None:
            co01_wrappers = [CorrectnessCo01Wrapper(x_dict, event_log, predictor, rel) for rel in local_rel_data]

        if len(trace) > 0:
            # noinspection PyUnboundLocalVariable
            for wrapper in co01_wrappers:
                correctness_metric = wrapper.eval(trace)
                co01_dict['correctness_co01'].append(correctness_metric.df)

        progress.update(1)


def filter_out_data_to_relations(out_data) -> typing.List[ExplainableOutputData]:
    return [out for out in out_data if isinstance(out, ExplainableOutputData)]


def main():
    parser = argparse.ArgumentParser(description="Process Transformer - XAI-Metrics.")
    parser.add_argument("--xai_config", type=str, required=True,
                        help='Path of the XAI-file (data, parameters).')
    args = parser.parse_args()
    main_with_args(args.xai_config)


def main_with_args(xai_config: str):
    with open(xai_config, 'r', encoding="utf8") as f:
        xai_config = ExplainingModel.from_dict(json.load(f))
        os.makedirs(xai_config.result_dir, exist_ok=True)

        logger = _get_logger(xai_config)

        start = datetime.datetime.now()
        logger.info(f'Started at {start}')

        logger.info(f'Running config: {xai_config.to_dict()}')
        run_through_xai_metrics(xai_config, logger)

    end = datetime.datetime.now()
    logger.info(f'Finished at {end}')

    logger.info(f'Entire evaluation took {end - start} (hh:mm:ss).')


def _get_logger(xai_config: ExplainingModel):
    log_file = os.path.join(xai_config.result_dir, 'log.log')
    logger = logging.getLogger(f'XAI-metrics - {xai_config.name}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger


def _get_index_pairs_for_n_threads(n: int, length: int):
    """
    Fairly distribute indices among n threads for a list with length-elements.
    Returns a list of pairs that indicate [start, stop).
    E.g. n=3, length=8 returns [(0, 3), (3, 6), (6, 8)]
    """
    work_per_thread = length // n
    remainder = length % n
    if remainder == 0:
        return [(i * work_per_thread, (i + 1) * work_per_thread) for i in range(n)]

    work = []
    last = 0
    for i in range(n):
        if i < remainder:
            end = work_per_thread + 1
        else:
            end = work_per_thread
        work.append((last, last + end))
        last += end
    return work


def _pretty_time(seconds: float):
    s = int(seconds % 60)
    m = int((seconds / 60) % 60)
    h = int(seconds / 3600)

    def two_digits(t):
        return f'0{t}' if t < 10 else f'{t}'

    s = two_digits(s)
    m = two_digits(m)
    h = two_digits(h)

    return f'{h}h{m}m{s}s'


if __name__ == '__main__':
    main()
