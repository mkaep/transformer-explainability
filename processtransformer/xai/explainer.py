
from __future__ import annotations

import abc
import dataclasses
import typing

from processtransformer.models.helper import Predictor
from processtransformer.models.transformer import Transformer
from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclasses.dataclass
class TraceSupport:
    single_trace: bool
    multi_trace: bool
    multi_with_single_trace: bool


class Explainer(abc.ABC):
    """
    Important: Import your subclass in the __init__.py of this module!
    """

    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 result_dir: str,
                 show_pad: bool = False):
        self.model = model
        self.predictor = Predictor(model, x_word_dict, y_word_dict, show_pad)
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.result_dir = result_dir

    @staticmethod
    @abc.abstractmethod
    def get_name():
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_trace_support():
        return TraceSupport(False, False, False)

    @abc.abstractmethod
    def explain_trace(self, trace: typing.List[str], y_true: str, log) -> typing.List[OutputData]:
        raise NotImplementedError()

    @abc.abstractmethod
    def explain_multiple_traces(self, traces: typing.List[typing.Tuple[typing.List[str], str]], log,
                                trace_to_explain=None) -> typing.List[OutputData]:
        raise NotImplementedError()

    @staticmethod
    def reduce_attn_to_mult_row(attn):
        # Remove head
        attn = attn.sum(axis=0)

        heads = []
        for head in attn:
            head = head.sum(axis=0)
            head = head / sum(head)
            head.append(head)
        return heads
