
import numpy as np
from matplotlib import pyplot as plt
from pm4py.util import typing

from processtransformer.xai.visualization.output_models.softmax_output import SoftmaxOutput
from processtransformer.xai.visualization.viz_funcs import BaseViz
from processtransformer.xai.visualization.viz_funcs.base_viz import VizOutput, FigureOutput


class SoftmaxViz(BaseViz):
    accepts = [SoftmaxOutput]

    def __init__(self, softmax_output: SoftmaxOutput) -> None:
        super().__init__()
        self.softmax_output = softmax_output

    def visualize(self) -> typing.List[VizOutput]:
        softmax = self.softmax_output.softmax_vector
        cm = 1 / 2.54
        fig, ax = plt.subplots(figsize=(18 * cm, 5 * cm))
        # Map to tuples to numbers
        labels = [pair[0] for pair in softmax]
        values = [pair[1] for pair in softmax]
        im = ax.matshow(np.asarray(values).reshape(1, len(values)), cmap="gray")

        ax.set_xticks(range(len(labels)), labels=labels)
        ax.xaxis.set_ticks_position('top')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
                 rotation_mode="anchor")
        ax.yaxis.set_visible(False)

        argmax = np.argmax(values)
        for i, value in enumerate(values):
            weight = 'normal'
            if i == argmax:
                weight = 'bold'

            ax.text(i, 0, '{:0.2f}'.format(values[i]), ha='center', va='center',
                    fontweight=weight,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        fig.suptitle(f'Softmax-distribution for predictions. Prediction: {labels[argmax]}.')
        fig.tight_layout()
        plt.savefig(self.softmax_output.figure_data.file_path)

        return [FigureOutput(self.softmax_output.figure_data.file_path)]
