
import dataclasses

from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclasses.dataclass
class TextOutput(OutputData):
    text: str
    file_path: str
