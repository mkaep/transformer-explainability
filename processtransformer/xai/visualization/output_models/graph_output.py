
from __future__ import annotations

import dataclasses
import typing

import networkx as nx
from pm4py import BPMN

from processtransformer.util.types import Trace, RelationsDict
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import TransformerInfoForExplanation, \
    ExplainableOutputData


@dataclasses.dataclass
class GraphOutput(ExplainableOutputData):
    graph: nx.DiGraph
    bpmn_model: typing.Optional[BPMN]
    figure_data: FigureData

    def __post_init__(self):
        super().__init__()

    @classmethod
    def supports_explanation_for_trace(cls) -> bool:
        return True

    def _get_explanation_for_trace(self, trace: Trace,
                                   transformer_info: TransformerInfoForExplanation,
                                   ) -> RelationsDict:
        result_dict: RelationsDict = dict()

        for event in set(trace):
            if not self.graph.has_node(event):
                continue

            successors = tuple(self.graph.successors(event))
            result_dict[(event,)] = successors

        return result_dict
