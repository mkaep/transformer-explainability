
import dataclasses

from processtransformer.util.types import SoftmaxVector
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclasses.dataclass
class SoftmaxOutput(OutputData):
    softmax_vector: SoftmaxVector
    figure_data: FigureData
