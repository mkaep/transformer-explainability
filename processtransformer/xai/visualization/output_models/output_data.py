
from __future__ import annotations

import abc
import dataclasses
import threading

from processtransformer.util.types import Trace, RelationsDict, SoftmaxVector, AttentionVector


@dataclasses.dataclass
class TransformerInfoForExplanation:
    softmax_vec: SoftmaxVector
    attn_vec: AttentionVector


class OutputData(abc.ABC):
    pass


class ExplainableOutputData(OutputData):
    def __init__(self) -> None:
        super().__init__()
        self.cache = dict()
        self.lock = threading.Lock()

    @classmethod
    def supports_explanation_for_trace(cls) -> bool:
        return False

    def get_explanation_for_trace(self, trace: Trace,
                                  transformer_info: TransformerInfoForExplanation,
                                  ) -> RelationsDict:
        key = tuple(trace)
        with self.lock:
            if key in self.cache.keys():
                return self.cache[key]

        result = self._get_explanation_for_trace(trace, transformer_info)

        with self.lock:
            self.cache[key] = result
        return result

    def _get_explanation_for_trace(self, trace: Trace,
                                   transformer_info: TransformerInfoForExplanation,
                                   ) -> RelationsDict:
        pass
