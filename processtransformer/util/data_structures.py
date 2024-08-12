
import dataclasses
from typing import TypeVar

from pm4py.util import typing

T = TypeVar('T')


class Wrapper:
    def __init__(self, value: T):
        self.value: T = value


@dataclasses.dataclass
class TraceInfo:
    trace: typing.List[typing.Tuple[str, float]]
    prediction: typing.List[typing.Tuple[str, float]]
    prediction_for_event: float


class DictInfo:
    def __init__(self):
        self.trace_infos: typing.List[TraceInfo] = []
        self.event_set: typing.Set[str] = set()

    def add_trace_info(self, trace_info: TraceInfo):
        self.trace_infos.append(trace_info)

    def add_event(self, event: str):
        self.event_set.add(event)
