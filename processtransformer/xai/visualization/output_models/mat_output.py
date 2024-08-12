
from __future__ import annotations

import dataclasses
import typing

import numpy as np

from processtransformer.util.types import Trace, RelationsDict, XEvent, YEvent
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import TransformerInfoForExplanation, \
    ExplainableOutputData


@dataclasses.dataclass
class MatOutput(ExplainableOutputData):
    def __post_init__(self) -> None:
        super().__init__()
        shape = self.mat.shape
        assert len(shape) == 2, 'shape has to be two-dimensional'
        assert shape[0] == len(self.y_labels), 'first dimension (rows) has to match number of y-labels'
        assert shape[1] == len(self.x_labels), 'second dimension (columns) has to match number of x-labels'

    mat: np.ndarray
    x_labels: typing.List[XEvent]
    y_labels: typing.List[YEvent]
    figure_data: FigureData
    pred_thr: float = 0.1
    attn_thr: float = 0.1

    @classmethod
    def supports_explanation_for_trace(cls) -> bool:
        return True

    def _get_explanation_for_trace(self, trace: Trace,
                                   transformer_info: TransformerInfoForExplanation,
                                   ) -> RelationsDict:
        result_dict: RelationsDict = dict()

        for event in set(trace):
            # Check if event is considered by this matrix at all
            if event not in self.x_labels:
                continue

            index = self.x_labels.index(event)

            # Get all relevant predictions for this event
            prediction_values = self.mat[:, index]
            actual_predictions = [self.y_labels[i] for i, val in enumerate(prediction_values) if val > self.pred_thr]

            if len(actual_predictions) > 0:
                result_dict[(event,)] = tuple(actual_predictions)

        return result_dict
