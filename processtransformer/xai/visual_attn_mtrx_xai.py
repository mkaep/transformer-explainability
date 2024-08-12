

import os.path
import typing

from processtransformer.models.transformer import Transformer
from processtransformer.xai.explainer import Explainer, TraceSupport
from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.attention_output import AttentionOutput
from processtransformer.xai.visualization.output_models.output_data import OutputData
from processtransformer.xai.visualization.output_models.softmax_output import SoftmaxOutput


class VisualAttentionMatrixExplainer(Explainer):
    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 result_dir: str,
                 show_pad: bool = False):
        super().__init__(model, x_word_dict, y_word_dict, result_dir, show_pad)

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_trace_support():
        return TraceSupport(single_trace=True, multi_trace=False, multi_with_single_trace=False)

    def explain_multiple_traces(self, traces: typing.List[typing.Tuple[typing.List[str], str]], log,
                                trace_to_explain=None) -> typing.List[OutputData]:
        raise NotImplementedError()

    def explain_trace(self, trace: typing.List[str], y_true: str, log) -> typing.List[OutputData]:
        output = []

        # Predict next event ("Pre prediction")
        file_path_pre = os.path.join(self.result_dir, "pre_prediction.svg")
        pre_y_pred_event, pre_y_pred_token, pre_attn_scores, pre_event_trace, y_softmax, _ = \
            self.predictor.make_prediction(trace, return_softmax=True)

        output.append(SoftmaxOutput(y_softmax,
                                    FigureData(os.path.join(self.result_dir, "softmax_distribution.svg"), None)))
        output.append(AttentionOutput(pre_attn_scores, pre_event_trace, y_true,
                                      FigureData(file_path_pre, 'Attention Scores before Prediction')))

        return output
