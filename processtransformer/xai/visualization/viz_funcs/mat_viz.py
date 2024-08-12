
import numpy as np
import typing
from matplotlib import pyplot as plt

from processtransformer.xai.visualization.output_models.mat_output import MatOutput
from processtransformer.xai.visualization.viz_funcs.base_viz import BaseViz, VizOutput, FigureOutput


class MatrixViz(BaseViz):
    accepts = [MatOutput]

    def __init__(self, mat_output: MatOutput) -> None:
        super().__init__()
        self.mat_output = mat_output

    def visualize(self) -> typing.List[VizOutput]:
        fig, ax = plt.subplots()
        # Colormap; try to restrict range to [0..1]
        v_min = min(np.min(self.mat_output.mat), 0.0)
        v_max = max(np.max(self.mat_output.mat), 1.0)
        ax_mat = ax.matshow(self.mat_output.mat, cmap='gray', vmin=v_min, vmax=v_max)

        # Ticks and labels
        ax.set_xticks(range(len(self.mat_output.x_labels)), labels=self.mat_output.x_labels)
        ax.xaxis.set_ticks_position('top')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="left",
                 rotation_mode="anchor")
        ax.set_yticks(range(len(self.mat_output.y_labels)), labels=self.mat_output.y_labels)
        plt.xlabel('Predecessors', va='top')
        plt.ylabel('Prediction')

        # Title and rest
        ax.set_title('Relevance of events')
        if self.mat_output.figure_data.title is not None:
            ax.set_title(self.mat_output.figure_data.title)
        fig.colorbar(ax_mat, location='bottom')
        plt.tight_layout()

        if self.mat_output.figure_data.file_path is not None:
            plt.savefig(self.mat_output.figure_data.file_path)

        # plt.show()
        return [FigureOutput(self.mat_output.figure_data.file_path)]
