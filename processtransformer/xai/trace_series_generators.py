
import typing

import numpy as np

from processtransformer.util.types import YEvent, Trace


def prefix_trace_series(trace: typing.List[str]):
    # Go over all prefixes of this trace
    for i in range(1, len(trace)):
        cut_trace = trace[0: i]
        yield cut_trace


def prefix_trace_series_n_times(trace: Trace, count):
    rng = np.random.Generator(np.random.PCG64(1234))
    if count >= len(trace) - 1:
        return list(prefix_trace_series(trace))

    indices = rng.integers(low=1, high=len(trace), size=count)
    return [trace[0:i] for i in indices]


def prefix_pred_and_event_series(trace_ypred_tuples: typing.List[typing.Tuple[Trace, YEvent]]):
    for trace, y_pred in trace_ypred_tuples:
        for i in range(0, len(trace)):
            yield trace[0:i], trace[i]
        yield trace, y_pred


def extend_trace_series(trace: typing.List[str], predictions: typing.List[str]):
    for pred in predictions:
        yield trace + [pred]


def single_event_series(trace: typing.List[str], lengths=3):
    for length in range(1, lengths + 1):
        for i in range(0, len(trace)):
            yield trace[i:i + length]
