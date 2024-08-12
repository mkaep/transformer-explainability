
import dataclasses

import numpy

from processtransformer.util.types import Trace, YEvent
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclasses.dataclass
class AttentionOutput(OutputData):
    attention: numpy.ndarray
    trace: Trace
    predicted_event: YEvent
    figure_data: FigureData
