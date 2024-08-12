
import abc
import os
import typing

import networkx as nx
import pm4py
from scipy.spatial import distance

from processtransformer.models.transformer import Transformer
from processtransformer.xai.explainer import Explainer, TraceSupport
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.graph_output import GraphOutput
from processtransformer.xai.visualization.output_models.output_data import OutputData


class TraceModificationExplainer(Explainer, metaclass=abc.ABCMeta):
    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 result_dir: str,
                 show_pad: bool = False,
                 softmax_closeness_threshold=0.05,
                 event_relevance_threshold=0.1,
                 prediction_threshold=0.2,
                 bpmn_model_path=None,
                 ):
        super().__init__(model, x_word_dict, y_word_dict, result_dir, show_pad)

        # The lower, the closer both softmax-vectors have to be. Range is [0..1].
        self.softmax_closeness_threshold = softmax_closeness_threshold
        # If an event's cumulated attention-score is above this,
        # it is considered relevant for the final graph. Range is [0..1].
        self.event_relevance_threshold = event_relevance_threshold
        # Minimum value in softmax-output for an event to count as a prediction.
        self.prediction_threshold = prediction_threshold

        try:
            self.bpmn_model = pm4py.read_bpmn(bpmn_model_path)
        except Exception as e:
            self.bpmn_model = None

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_trace_support():
        return TraceSupport(single_trace=True, multi_trace=False, multi_with_single_trace=False)

    def explain_multiple_traces(self, traces: typing.List[typing.Tuple[typing.List[str], str]], log,
                                trace_to_explain=None) -> typing.List[OutputData]:
        raise NotImplementedError()

    def explain_trace(self, trace: typing.List[str], y_true: str, log) -> typing.List[OutputData]:
        graph = nx.DiGraph()
        # Go over all prefixes of this trace
        for t in self._trace_series(trace):
            # Get 'normal' prediction from Transformer
            g = self._get_graph_for_trace(t)
            graph.add_nodes_from(g.nodes)
            graph.add_edges_from(g.edges)

        # graph_df.drop_duplicates()
        res = self.predictor.make_prediction(trace, return_softmax=True)
        pred = res[0]
        pred_val = {k: v for k, v in res[4]}[pred] * 100
        title = f'Trace: {trace}, ({pred}: {pred_val:.0f}%)'
        return [GraphOutput(graph, self.bpmn_model,
                            FigureData(os.path.join(self.result_dir, 'graph.svg'), title))]

    def _get_graph_for_trace(self, trace):
        _, _, init_attn_scores, _, init_event_softmax, init_softmax = self.predictor.make_prediction(trace, True)
        total_attn_score_per_event = {key: 0.0 for key in self.x_word_dict.keys()}
        predictions = [pair[0] for pair in init_event_softmax if pair[1] > self.prediction_threshold]

        self._explain_single_trace(trace, total_attn_score_per_event, init_softmax, self._trace_modification)

        return self._construct_graph_network(total_attn_score_per_event, predictions)

    @staticmethod
    @abc.abstractmethod
    def _trace_series(trace):
        """
        Override this generator-function.
        Provides the entire series of traces to be considered.
        E.g. yield only the original trace or all of its prefixes.
        """
        raise NotImplementedError("Do not use this class directly, inherit from it.")

    @abc.abstractmethod
    def _trace_modification(self, trace):
        """
        Override this generator-function.
        Provides all variation/modification of a single trace.
        E.g. mask out some positions.
        This generator gets fully iterated for every trace of _trace_series!
        """
        raise NotImplementedError("Do not use this class directly, inherit from it.")

    def _explain_single_trace(self, trace: typing.List[str], total_attn_score_per_event: typing.Dict[str, float],
                              init_softmax, trace_generator):
        gen_traces = list(trace_generator(trace))
        res = self.predictor.make_multi_predictions(gen_traces, return_softmax=True)

        for (_, _, attn_scores, _, event_softmax, softmax), gen_trace in zip(res, gen_traces):

            if not self._are_softmax_vectors_close(init_softmax, softmax):
                # Ignore these attention matrices as this prediction differs from the original prediction
                continue

            attn_scores = attn_scores.squeeze(axis=0)  # only have a batch of size 1
            attn_scores = attn_scores.sum(axis=(0, 1))  # sum along heads and rows -> columns persist

            for index, event in enumerate(gen_trace):
                if event in ['[PAD]', '[UNK]'] or event not in total_attn_score_per_event.keys():
                    continue
                # Add score to each token
                total_attn_score_per_event[event] += attn_scores[index]

    def _are_softmax_vectors_close(self, init_softmax, softmax) -> bool:
        return distance.cosine(init_softmax, softmax) < self.softmax_closeness_threshold

    def _construct_graph_network(self, total_attn_score_per_event: typing.Dict[str, float],
                                 predictions: typing.List[str]):
        total_value = sum(list(total_attn_score_per_event.values()))
        graph = nx.DiGraph()

        if total_value == 0:
            return graph

        for event in total_attn_score_per_event.keys():
            relevance = total_attn_score_per_event[event] / total_value

            if relevance < self.event_relevance_threshold:
                continue

            graph.add_node(event)
            graph.add_nodes_from(predictions)
            # Add edge from event to all predictions
            graph.add_edges_from([(event, pred) for pred in predictions])

        return graph
