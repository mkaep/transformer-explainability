
import threading
import typing

import networkx as nx
import pm4py
from pm4py.objects.log.obj import EventLog

from processtransformer.util.types import Trace


class RealEnvWrapper:
    def __init__(self, event_log, dfg=None) -> None:
        super().__init__()
        self.event_log = event_log
        self.dfg = dfg

        self.lock = threading.Lock()
        self.cache = dict()

    def get_env(self,
                base_trace: Trace,
                return_base_trace=False
                ) -> typing.List[Trace]:
        key = (tuple(base_trace), return_base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = generate_real_local_env(base_trace, self.event_log, return_base_trace, self.dfg)
        with self.lock:
            self.cache[key] = result
        return result


def generate_real_local_env(base_trace: Trace,
                            event_log: EventLog,
                            return_base_trace=False,
                            dfg=None,
                            ) -> typing.List[Trace]:
    if dfg is None:
        dfg = pm4py.discover_dfg(event_log)
    graph = dfg_to_graph(dfg)

    # Cutoff limits the length of path; we still want to be somewhat local
    cutoff = 2

    new_traces = []
    if return_base_trace:
        new_traces.append(base_trace)

    # Check if switching neighbor events is possible, i.e., occurs in DFG.
    # E.g. ABC, does BAC or ACB occur?
    for i, ev1 in enumerate(base_trace[:-1]):
        ev2 = base_trace[i + 1]
        if ev1 == ev2:
            continue
        # Can we change event order?
        if graph.has_edge(ev2, ev1):
            # E.g. i = 1, trace = [A, B, C, D]
            # [A] + [C] + [B] + [D]
            new_traces.append(base_trace[:i] + [ev2] + [ev1] + base_trace[i + 2:])

        # Can we squeeze in other sub-traces? E.g. [A, D] --> [A, B, C, D]?
        successors = set(graph.successors(ev1)).difference(ev2)
        if len(successors) > 0:
            for successor in successors:
                paths = list(nx.all_simple_paths(graph, successor, ev2, cutoff=cutoff))
                for path in paths:
                    new_traces.append(base_trace[:i + 1] + path + base_trace[i + 2:])

    skip_dist = 3
    # Example: base_trace = [A, B, C, D, E, F]
    for i, ev1 in enumerate(base_trace):
        ev2_start_i = i + 2
        for j, ev2 in enumerate(base_trace[ev2_start_i:ev2_start_i + skip_dist]):
            # 1st: [ev1=A, B, ev2=C].
            # 2nd: [ev1=A, B, C, ev2=D].
            # 3rd: [ev1=A, B, C, D, ev2=E]
            base_path = tuple(base_trace[i:ev2_start_i + j + 1])
            paths = list(nx.all_simple_paths(graph, ev1, ev2, cutoff=cutoff + j))
            paths = set(tuple(path) for path in paths)
            paths = paths.difference({tuple(base_path)})

            for path in paths:
                new_traces.append(base_trace[:i] +
                                  list(path) +
                                  base_trace[ev2_start_i + j + 1:])

    # Filter out duplicates
    new_traces = set(tuple(trace) for trace in new_traces)
    new_traces = [list(trace) for trace in new_traces]

    return new_traces


def dfg_to_graph(dfg) -> nx.DiGraph:
    graph = nx.DiGraph()
    for pair in dfg[0]:
        graph.add_node(pair[0])
        graph.add_node(pair[1])
        graph.add_edge(*pair)
    return graph


def replace_single_event(base_trace: Trace,
                         index: int,
                         event_log: EventLog,
                         return_base_trace=False,
                         dfg=None,
                         ) -> typing.List[Trace]:
    assert 0 <= index < len(base_trace), 'Index out of trace'

    new_traces = [[event if i != index else f'M-{event}' for i, event in enumerate(base_trace)]]

    # Mask out event at index
    if return_base_trace:
        new_traces.append(base_trace)

    if dfg is None:
        dfg = pm4py.discover_dfg(event_log)

    # Case 1: If only length one -> we are at the start. Add each start event except the given event
    if len(base_trace) == 1 or index == 0:
        start_events = set(dfg[1].keys()).difference({base_trace[0]})
        for start_event in start_events:
            new_traces.append([start_event] + base_trace[index + 1:])
        return new_traces

    graph = dfg_to_graph(dfg)

    # Case 2: Last event; insert possible follow-up events
    if index == len(base_trace) - 1:
        if graph.has_node(base_trace[-2]):
            # E.g. if we have [A, B, C] and index=2, we want successors of B (not C!)
            successors = graph.successors(base_trace[-2])
            successors = set(successors).difference({base_trace[-1]})
            for successor in successors:
                new_traces.append(base_trace[:-1] + [successor])
        return new_traces

    # Case 3: We are somewhere in between, i.e. there is definitely an event before and after the index!
    event_before = base_trace[index - 1]
    event_after = base_trace[index + 1]
    if graph.has_node(event_before) and graph.has_node(event_after):
        # Only consider paths of length 2, i.e. [event_before, event_x, event_after]
        alternative_paths = nx.all_simple_paths(graph, event_before, event_after, cutoff=2)
        alternative_paths = set(tuple(path) for path in alternative_paths)
        base_path = base_trace[index - 1:index + 2]
        alternative_paths = alternative_paths.difference({tuple(base_path)})

        new_traces += [base_trace[:index - 1] + list(path) + base_trace[index + 2:] for path in alternative_paths]

    return new_traces


class ArtEnvWrapper:
    def __init__(self, event_log, dfg=None) -> None:
        super().__init__()
        self.event_log = event_log
        self.dfg = dfg

        self.lock = threading.Lock()
        self.cache = dict()

    def get_env(self, trace):
        key = tuple(trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = list(generate_artificial_local_env(trace, self.event_log, self.dfg))
        with self.lock:
            self.cache[key] = result
        return result


def generate_artificial_local_env(trace, event_log, dfg=None):
    for i, event in enumerate(trace):
        local_traces = replace_single_event(trace, i, event_log, dfg=dfg)
        for local_trace in local_traces:
            yield event, local_trace


class MaybeGenWrapper:
    def __init__(self, event_log, dfg=None) -> None:
        super().__init__()
        self.event_log = event_log
        self.dfg = dfg

        self.lock = threading.Lock()
        self.cache = dict()

    def eval(self, trace, return_base_trace=False):
        key = (tuple(trace), return_base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = maybe_gen_real_local_env(trace, self.event_log, return_base_trace, self.dfg)
        with self.lock:
            self.cache[key] = result
        return result


def maybe_gen_real_local_env(trace, event_log, return_base_trace=False, dfg=None):
    local_traces = generate_real_local_env(trace, event_log, return_base_trace, dfg)
    if len(local_traces) == 0:
        # It may be that no real local environment can be found that differs from the base-trace. E.g. sequence.
        local_traces = list(generate_artificial_local_env(trace, event_log))
        local_traces = [trace for event, trace in local_traces]
    return local_traces
