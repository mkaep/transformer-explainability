
import copy
import os
import typing

import numpy as np
import pm4py
from tqdm import tqdm

from processtransformer.models.transformer import Transformer
from processtransformer.util import compressor
from processtransformer.util.attn_funcs import attn_transform
from processtransformer.util.misc_algs import dict_to_mat
from processtransformer.util.softmax_funcs import filter_softmax_vector
from processtransformer.util.subclassing import EvaluatorSubclasses
from processtransformer.util.types import RelationsDict
from processtransformer.xai.event_eval.evaluator import BasePredictionInfo, Evaluator
from processtransformer.xai.explainer import Explainer, TraceSupport
from processtransformer.xai.trace_series_generators import prefix_trace_series
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.mat_output import MatOutput
from processtransformer.xai.visualization.output_models.output_data import OutputData, TransformerInfoForExplanation, \
    ExplainableOutputData
from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from processtransformer.util.nx_graph import mat_to_graph
from processtransformer.xai.visualization.output_models.graph_output import GraphOutput


class AttentionExplorationExplainer(Explainer):
    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 result_dir: str,
                 show_pad: bool = False,
                 evaluators: typing.List[typing.Tuple[typing.Dict, typing.Type[Evaluator]]] = None,
                 normal_to_abs_thr: float = 0.6,
                 dependency_threshold: float = 0.1,
                 attention_score_threshold: float = 0.15,
                 prediction_threshold: float = 0.1,
                 bpmn_model_path = None,
                 ):
        super().__init__(model, x_word_dict, y_word_dict, result_dir, show_pad)
        self.normal_to_abs_thr = normal_to_abs_thr
        self.dependency_threshold = dependency_threshold
        self.attention_score_threshold = attention_score_threshold
        self.prediction_threshold = prediction_threshold
        self.bpmn = None
        if bpmn_model_path is not None:
            self.bpmn = pm4py.read_bpmn(bpmn_model_path)

        # Use all evaluators by default
        if evaluators is None or len(evaluators) == 0:
            evaluators = EvaluatorSubclasses().get_all_subclasses()
            evaluators = [(dict(), e) for e in evaluators]
        evaluators = [evaluator(self.predictor, **dct) for dct, evaluator in evaluators]

        self.evaluators: typing.List[Evaluator] = evaluators

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_trace_support():
        return TraceSupport(single_trace=False, multi_trace=True, multi_with_single_trace=True)

    def explain_trace(self, trace: typing.List[str], y_true: str, log) -> typing.List[OutputData]:
        raise NotImplementedError()

    def explain_multiple_traces(self, traces: typing.List[typing.Tuple[typing.List[str], str]], log,
                                trace_to_explain=None) -> typing.List[OutputData]:
        # Per predicted event we have a dict that holds attention-scores per event
        checks = [evaluator.get_name() for evaluator in self.evaluators]
        score_dicts = dict()
        for check in checks:
            single_dict = {y_event: {x_event: 0.0 for x_event in self.x_word_dict.keys()} for y_event in
                           self.y_word_dict.keys()}
            score_dicts[check] = single_dict

        prediction_dict = self.collect_traces_by_prediction(traces)

        for pred, trace_set in prediction_dict.items():
            trace_lst = list(trace_set)
            results = self.predictor.make_multi_predictions(trace_lst, return_softmax=True)
            results = list(zip(results, trace_lst))[:500]

            for (_, _, base_attn, _, base_softmax, softmax_val), trace in tqdm(results, desc=f'Working on "{pred}"'):

                base_softmax = filter_softmax_vector(base_softmax)
                # An event may occur more than once (more correctly: one activity may occur several times)
                relevant_predictions = [(event, pred_val, index) for index, (event, pred_val) in enumerate(base_softmax)
                                        if pred_val > self.prediction_threshold and event == pred]
                base_attn = attn_transform(trace, base_attn)
                relevant_attn_indices = [index for index, (event, score) in enumerate(base_attn)
                                         if score > self.attention_score_threshold]
                if len(relevant_attn_indices) == 0:
                    continue

                info = BasePredictionInfo(trace, base_attn, base_softmax,
                                          relevant_predictions, relevant_attn_indices)

                local_score_dict = dict()
                for check in checks:
                    single_local_score_dict = {y_event: {x_event: [] for x_event in self.x_word_dict.keys()} for y_event
                                               in self.y_word_dict.keys()}
                    local_score_dict[check] = single_local_score_dict

                for evaluator in self.evaluators:
                    evaluator.eval(info, local_score_dict[evaluator.get_name()])

                self.transfer_local_to_global_score_dict(local_score_dict, score_dicts, self.normal_to_abs_thr)

        # Boolean dict telling on what events a prediction depends
        dependency_dict = dict()
        # Normalize all rows to one (i.e. for each y_event)
        for check, score_dict in score_dicts.items():
            local_dep_dict = dict()
            dependency_dict[check] = local_dep_dict
            for y_event, y_dict in score_dict.items():
                sum_y = sum(list(y_dict.values()))
                y_dep_dict = dict()
                local_dep_dict[y_event] = y_dep_dict
                for key in y_dict.keys():
                    if sum_y != 0.0:
                        new_val = y_dict[key] / sum_y
                    else:
                        new_val = 0.0
                    y_dict[key] = new_val
                    y_dep_dict[key] = True if new_val > self.dependency_threshold else False

        output_list = []
        output_list_with_checks = []
        for check, dep_dict in dependency_dict.items():
            x_labels, y_labels, mat = dict_to_mat(dep_dict)
            file_path = os.path.join(self.result_dir, f'{check}_matrix.png')
            output = MatOutput(mat, x_labels, y_labels, FigureData(file_path, f'Check: {check}'),
                               self.prediction_threshold)
            compressor.compress(dep_dict, os.path.join(self.result_dir, f'{check}_matrix'))
            output_list.append(output)
            output_list_with_checks.append((output, check))

        # Does not make sense to "combine" one dict with itself
        if len(score_dicts) > 1:
            # logical_and_output = self._combine_dicts(dependency_dict, True, lambda x, y: x and y, 'logical_and')
            # output_list.append(logical_and_output)

            logical_or_output = self._combine_dicts(dependency_dict, False, lambda x, y: x or y, 'logical_or')
            # Or gives the best results -> only output this and none of the others
            output_list = [logical_or_output]

        for check, score_dict in score_dicts.items():
            compressor.compress(score_dict, os.path.join(self.result_dir, check + '_score_dicts_mult'))

        # self._explain_single_trace(output_list, output_list_with_checks, trace_to_explain)

        return output_list

    def _combine_dicts(self, dependency_dict, init_value: bool, join_op: typing.Callable[[bool, bool], bool],
                       title: str) -> MatOutput:
        dep_dict = list(dependency_dict.values())[0]
        joined_dict = {y_event: {x_event: init_value for x_event in dep_dict[y_event]} for
                       y_event in dep_dict}
        for check, dep_dict in dependency_dict.items():
            for y_key, y_dict in dep_dict.items():
                for x_key, x_val in y_dict.items():
                    joined_dict[y_key][x_key] = join_op(joined_dict[y_key][x_key], x_val)

        x_labels, y_labels, mat = dict_to_mat(dep_dict)
        file_path = os.path.join(self.result_dir, f'{title}_matrix.png')
        output = MatOutput(mat, x_labels, y_labels, FigureData(file_path, f'Combined: {title}'),
                           self.prediction_threshold, self.attention_score_threshold)
        compressor.compress(joined_dict, os.path.join(self.result_dir, f'{title}_matrix'))
        # graph = mat_to_graph(dep_dict)
        # output = GraphOutput(graph, self.bpmn, FigureData(os.path.join(self.result_dir, 'graph.svg'), ''))

        return output

    def _explain_single_trace(self, output_list,
                              output_list_with_checks: typing.List[typing.Tuple[OutputData, str]],
                              trace_to_explain):
        output_list_with_checks = [out for out in output_list_with_checks if isinstance(out, ExplainableOutputData)]

        if trace_to_explain is not None:
            trace_to_explain_dict: RelationsDict = dict()
            for output, check in output_list_with_checks:
                if not output.supports_explanation_for_trace():
                    continue

                _, _, attn, _, softmax, _ = self.predictor.make_prediction(trace_to_explain, return_softmax=True)
                attn = attn_transform(trace_to_explain, attn)
                softmax = filter_softmax_vector(softmax, self.prediction_threshold)

                dct = output.get_explanation_for_trace(trace_to_explain,
                                                       TransformerInfoForExplanation(softmax, attn))
                file_path = os.path.join(self.result_dir, f'{check}_relations.txt')
                output_list.append(RelationsOutput(copy.deepcopy(dct),
                                                   FigureData(file_path, check),
                                                   f'{self.get_name()}: {check}',
                                                   [trace_to_explain[0]]))
                # Update existing keys, add new ones
                same_keys = set(dct.keys()).intersection(trace_to_explain_dict.keys())
                for same_key in same_keys:
                    trace_to_explain_dict[same_key] += dct.pop(same_key)
                trace_to_explain_dict.update(dct)
                for key, value in trace_to_explain_dict.items():
                    trace_to_explain_dict[key] = tuple(set(value))
            file_path = os.path.join(self.result_dir, 'overall_relations.txt')
            output_list.append(RelationsOutput(trace_to_explain_dict,
                                               FigureData(file_path, f'Overall Relations'),
                                               f'{self.get_name()}: overall',
                                               [trace_to_explain[0]]))

    @staticmethod
    def transfer_local_to_global_score_dict(local_score_dict, score_dicts, normal_to_abs_thr=0.6):
        for check, check_dict in local_score_dict.items():
            for y_event, y_dict in check_dict.items():
                for x_event, x_scores in y_dict.items():
                    len_x_scores = len(x_scores)
                    if len_x_scores == 0:
                        continue

                    normal_sum = sum(x_scores)
                    abs_sum = np.abs(x_scores).sum()
                    if abs_sum == 0.0:
                        continue

                    # E.g. abs-sum = 25, normal-sum = 20.
                    normal_sum /= abs_sum * len_x_scores
                    # Divide by abs-sum for clarity. Of course, 1.0 / len_x_scores would give the same result.
                    abs_sum /= abs_sum * len_x_scores

                    overall_score = normal_sum
                    if normal_sum / abs_sum < normal_to_abs_thr:
                        # Lots of negative values
                        overall_score = 0.0
                    score_dicts[check][y_event][x_event] += overall_score

    def collect_traces_by_prediction(self, traces):
        # Collect traces by prediction, i.e. what traces predict event 'A' amongst others
        prediction_dict = {pred: set() for pred in self.y_word_dict.keys()}
        for trace in tqdm(traces, desc="Iterating over traces"):
            trace = trace[0]

            prefixes = list(prefix_trace_series(trace))
            results = self.predictor.make_multi_predictions(prefixes, return_softmax=True)
            for (_, _, attn, _, softmax, _), prefix in zip(results, prefixes):
                # _, _, attn, _, softmax, _ = self.predictor.make_prediction(prefix, return_softmax=True)
                relevant_pred = [(event, pred_val, index) for index, (event, pred_val) in enumerate(softmax)
                                 if pred_val > self.prediction_threshold]
                for pred in relevant_pred:
                    prediction_dict[pred[0]].add(tuple(prefix))

        return prediction_dict
