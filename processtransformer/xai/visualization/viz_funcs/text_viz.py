
import typing

from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from processtransformer.xai.visualization.output_models.text_output import TextOutput
from processtransformer.xai.visualization.viz_funcs import BaseViz
from processtransformer.xai.visualization.viz_funcs.base_viz import TextOutput as NextTextOutput
from processtransformer.xai.visualization.viz_funcs.base_viz import VizOutput


class TextViz(BaseViz):
    accepts = [TextOutput, RelationsOutput]

    def __init__(self, text_output: TextOutput) -> None:
        super().__init__()
        self.text_output = text_output

    def visualize(self) -> typing.List[VizOutput]:
        if isinstance(self.text_output, TextOutput):
            return self._viz_normal_text()
        elif isinstance(self.text_output, RelationsOutput):
            return self._viz_relations()

    def _viz_relations(self) -> typing.List[VizOutput]:
        if not isinstance(self.text_output, RelationsOutput):
            return []

        dct = self.text_output.relations_dict
        str_output = [f'Relations of {self.text_output.name}']
        for left_side, right_side in dct.items():
            # E.g. gives A, B -> X, Y
            str_output.append(f"{', '.join(left_side)} -> {', '.join(right_side)}")
        str_output = '\n'.join(str_output)

        with open(self.text_output.figure_data.file_path, 'w') as text_file:
            text_file.write(str_output)
        return [NextTextOutput(str_output)]

    def _viz_normal_text(self) -> typing.List[VizOutput]:
        with open(self.text_output.file_path, 'w') as text_file:
            text_file.write(self.text_output.text)
        return [NextTextOutput(self.text_output.text)]
