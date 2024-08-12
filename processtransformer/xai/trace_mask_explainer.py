

import typing

from processtransformer.models.transformer import Transformer
from processtransformer.xai.trace_modification_explainer import TraceModificationExplainer
from processtransformer.xai.trace_mod_generators import trace_partial_masking_modification
from processtransformer.xai.trace_series_generators import prefix_trace_series, prefix_trace_series_n_times


class TraceMaskExplainer(TraceModificationExplainer):
    def __init__(self, model: Transformer, x_word_dict: typing.Dict[str, int], y_word_dict: typing.Dict[str, int],
                 result_dir: str, show_pad: bool = False,
                 softmax_closeness_threshold=0.05,
                 event_relevance_threshold=0.1,
                 prediction_threshold=0.2,
                 mask_count=20,
                 bpmn_model_path=None,
                 ):
        super().__init__(model, x_word_dict, y_word_dict, result_dir, show_pad, softmax_closeness_threshold,
                         event_relevance_threshold, prediction_threshold, bpmn_model_path)
        self.mask_count = mask_count

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def _trace_series(trace):
        return prefix_trace_series_n_times(trace, 20)

    def _trace_modification(self, trace):
        return trace_partial_masking_modification(trace, count=self.mask_count)
