import os
import typing
from dataclasses import dataclass, field

import networkx as nx
import pm4py
from tqdm import tqdm

from processtransformer.models.transformer import Transformer
from processtransformer.util.nx_graph import clean_up_graph
from processtransformer.xai.explainer import TraceSupport
from processtransformer.xai.trace_mod_generators import trace_partial_masking_modification
from processtransformer.xai.trace_modification_explainer import TraceModificationExplainer
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.graph_output import GraphOutput
from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclass
class TraceInfo:
    graph: nx.DiGraph
    edges_to_remove: typing.Set[typing.Tuple[str, str]]
    explored_traces: typing.Set[typing.Tuple[str]]
    # prefix_trace + to_predict + suffix_trace == original trace
    prefix_trace: typing.Tuple[str]  # "prefix", used for prediction
    suffix_trace: typing.Tuple[str]  # Remainder of trace
    children: list = field(default_factory=list)  # list of TraceInfo's


# Note: Tuples are used at most places as they can be used within a set whereas lists cannot.
class TraceBackwardExplainer(TraceModificationExplainer):
    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 result_dir: str,
                 show_pad: bool = False,
                 softmax_closeness_threshold=0.05,
                 event_relevance_threshold=0.1,
                 prediction_threshold=0.2,
                 mask_count=20,
                 bpmn_model_path=None,
                 ):
        super().__init__(model, x_word_dict, y_word_dict, result_dir, show_pad, softmax_closeness_threshold,
                         event_relevance_threshold, prediction_threshold)
        self.mask_count = mask_count
        try:
            self.bpmn_model = pm4py.read_bpmn(bpmn_model_path)
        except Exception as e:
            self.bpmn_model = None

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_trace_support():
        return TraceSupport(single_trace=False, multi_trace=True, multi_with_single_trace=False)

    @staticmethod
    def _trace_series(trace):
        pass

    def _trace_modification(self, trace):
        return trace_partial_masking_modification(trace, self.mask_count)

    def explore_trace_info(self, trace_info: TraceInfo):
        if len(trace_info.prefix_trace) == 0:
            # Base case, we looked at the entire trace
            return

        if trace_info.prefix_trace in trace_info.explored_traces:
            return
        trace_info.explored_traces.add(trace_info.prefix_trace)

        # Construct graph for this prefix-trace
        g_trace = self._get_graph_for_trace(list(trace_info.prefix_trace))
        overall_graph = trace_info.graph
        overall_graph.add_nodes_from(g_trace.nodes)
        overall_graph.add_edges_from(g_trace.edges)

        # Collect edges for removal
        for edge_AB in g_trace.edges:
            for suffix_event in trace_info.suffix_trace:
                # E.g. edge=A->B, suffix_event=C. If A->C and B->C exist, then remove A->C, as
                # we can go via A->B, B->C from A to C.
                edge_AC = (edge_AB[0], suffix_event)
                edge_BC = (edge_AB[1], suffix_event)
                if overall_graph.has_edge(*edge_AC) and overall_graph.has_edge(*edge_BC):
                    trace_info.edges_to_remove.add(edge_AC)

        # Add children
        for edge in g_trace.edges:
            index = -1
            for i, event in enumerate(trace_info.prefix_trace):
                if edge[0] == event:
                    index = i

            if index <= 0:
                # no more prefix possible
                continue

            # Everything before index
            prefix_trace = trace_info.prefix_trace[:index]
            # Everything after index
            suffix_trace = trace_info.prefix_trace[index + 1:] + trace_info.suffix_trace

            child = TraceInfo(trace_info.graph, trace_info.edges_to_remove, trace_info.explored_traces,
                              prefix_trace, suffix_trace)
            trace_info.children.append(child)

        # And redo the same procedure until the base case
        for child in trace_info.children:
            self.explore_trace_info(child)

    def explain_multiple_traces(self, traces: typing.List[typing.Tuple[typing.List[str], str]], log,
                                trace_to_explain=None) -> typing.List[OutputData]:
        graph = nx.DiGraph()
        edges_to_remove = set()
        explored_traces = set()

        for trace_tuple in tqdm(traces):
            trace_info = TraceInfo(graph, edges_to_remove, explored_traces,
                                   tuple(trace_tuple[0]), tuple())
            self.explore_trace_info(trace_info)

        # dfg = pm4py.discover_dfg(log)
        # edges_to_remove = edges_to_remove.difference(set(dfg[0].keys()))

        for edge in edges_to_remove:
            graph.remove_edge(*edge)

        clean_up_graph(graph)

        return [GraphOutput(graph, self.bpmn_model,
                            FigureData(os.path.join(self.result_dir, 'graph.svg'), None))]

    def explain_trace(self, trace: typing.List[str], y_true: str, log):
        self.explain_multiple_traces([(trace, y_true)], log)

    @staticmethod
    def _add_prefixes_to_explore(already_explored, new_to_explore, start_events, t):
        # Add prefixes to "to explore", unless already explored
        # Enumerate over trace as start events may occur more often (loop)
        # This is a bit imprecise as not all events of the same activity may be relevant,
        # i.e., <A, B, A, B, A>: Only the first B and not the second B may be of relevance. We use both here.
        for index, event in enumerate(t):
            if event not in start_events:
                continue

            # exclude the event in question, i.e. <A, B> instead of <A, B, C> if C = event
            prefix = t[:index]

            if prefix in already_explored:
                continue

            new_to_explore.append(prefix)
