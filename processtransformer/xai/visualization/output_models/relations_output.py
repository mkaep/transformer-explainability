
import dataclasses
import typing

from processtransformer.util.types import Trace, RelationsDict, YEvent
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import TransformerInfoForExplanation, \
    ExplainableOutputData


@dataclasses.dataclass
class RelationsOutput(ExplainableOutputData):
    relations_dict: RelationsDict
    figure_data: FigureData
    name: str
    start_events: typing.List[YEvent]

    def __post_init__(self):
        super().__init__()

    @classmethod
    def supports_explanation_for_trace(cls) -> bool:
        return True

    def _get_explanation_for_trace(self, trace: Trace,
                                   transformer_info: TransformerInfoForExplanation,
                                   ) -> RelationsDict:
        if len(trace) == 0:
            return {('',): tuple(self.start_events)}

        relations_dict = {event: set() for event in trace}

        for i, event in enumerate(trace):
            if not (event,) in self.relations_dict.keys():
                # Might be masked out
                continue
            right_side = self.relations_dict[(event,)]
            # Remove all following events from trace
            right_side = set(right_side).difference(set(trace[i + 1:]))
            relations_dict[event] = relations_dict[event].union(right_side)

        # Remove empty right sides
        relations_dict = {tuple(key): tuple(value) for key, value in relations_dict.items() if len(value) > 0}
        return relations_dict
