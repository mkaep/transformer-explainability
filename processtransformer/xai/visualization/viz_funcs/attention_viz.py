
import typing
from matplotlib import pyplot as plt

from processtransformer.xai.visualization.output_models.attention_output import AttentionOutput
from processtransformer.xai.visualization.viz_funcs import BaseViz
from processtransformer.xai.visualization.viz_funcs.base_viz import VizOutput, FigureOutput


class AttentionViz(BaseViz):
    accepts = [AttentionOutput]

    def __init__(self, attention_output: AttentionOutput) -> None:
        super().__init__()
        self.attention_output = attention_output

    def visualize(self) -> typing.List[VizOutput]:
        """
        Taken from https://www.tensorflow.org/text/tutorials/transformer#create_attention_plots
        """
        attention = self.attention_output.attention
        trace = self.attention_output.trace
        trace = [event[:5] for event in trace]

        num_layers = len(attention)
        num_heads = len(attention[0])  # number of heads is the same for all layers (or should be)
        cm = 1 / 2.54
        num_cols = 2
        num_head_rows = ((num_heads + num_cols - 1) // num_cols)
        num_total_rows = num_layers * num_head_rows
        fig, axs = plt.subplots(num_total_rows, num_cols,
                                sharex=False, sharey=False, squeeze=False)
        for i_layer in range(num_layers):
            pass

            attn_layer = attention[i_layer]
            for i_head_row in range(num_head_rows):
                for i in range(num_cols):
                    i_head = i_head_row * num_cols + i
                    if i_head >= num_heads:
                        # Went through all heads, last row is only partially filled
                        break

                    head = attn_layer[i_head]
                    ax = axs[i_layer + i_head_row, i]

                    im = ax.matshow(head, cmap="gray", vmin=0.0, vmax=1.0)

                    if i_layer == 0 and i_head_row == 0:
                        # Only create horizontal labels for first row
                        ax.set_xticks(range(len(trace)), labels=trace)
                        ax.xaxis.set_ticks_position('top')
                        plt.setp(ax.get_xticklabels(), rotation=30, ha="left",
                                 rotation_mode="anchor")
                    else:
                        ax.xaxis.set_visible(False)

                    if i == 0:
                        # Only create vertical labels for first column
                        ax.set_yticks(range(len(trace)), labels=trace)
                    else:
                        ax.yaxis.set_visible(False)

                    # Create colorbar
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("attention score", rotation=-90, va="bottom")

                    ax.set_title(f'Layer {i_layer + 1}/{num_layers}. '
                                 f'Head {i_head + 1}/{num_heads}.')

        title_prefix = self.attention_output.figure_data.title
        # fig.suptitle(f'{title_prefix}. Next event: {self.attention_output.predicted_event}.')
        fig.tight_layout()

        file_path = self.attention_output.figure_data.file_path
        if file_path is not None:
            plt.savefig(file_path)

        return [FigureOutput(file_path)]
