
import argparse
import math
import typing
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.bpmn.obj import BPMN
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Misc - BPMN Traversal.")
parser.add_argument("--bpmn_path", type=str, required=True, help='Path to the BPMN file.')
parser.add_argument("--output_path", type=str, required=True, help='Path to the output XES log-file.')
parser.add_argument('--balanced', default=False, action='store_true')
parser.add_argument('--hard_restrict', default=False, action='store_true',
                    help='If balanced, the trace count might be exceeded. This flag cuts hard after the'
                         'trace count, possibly limiting balancing/variety.')
parser.add_argument("--trace_count", type=int, required=False, default=1000,
                    help='Number of traces to generate')
args = parser.parse_args()


class Node:
    likelihood_change = 0.9

    def __init__(self, node: typing.Union[BPMN.Activity, BPMN.Gateway, BPMN.Event]) -> None:
        super().__init__()
        self.node = node
        self.children: typing.List[Node] = []
        self.hit_count = 0

        self.is_reset = False
        self.likelihood = 1.0
        self.reset()

    def reset(self):
        self.hit_count = 0
        # Normal distribution with mean and standard-deviation
        self.likelihood = max(0.1, np.random.normal(1.0, 0.5))
        if self.is_reset is False:
            self.is_reset = True
            for child in self.children:
                child.reset()

    def get_value(self):
        self.is_reset = False
        self.hit_count += 1

        if isinstance(self.node, BPMN.Activity):
            return self.node.get_name()

        return None

    def get_next_nodes(self):
        if len(self.children) == 0:
            return []

        if isinstance(self.node, BPMN.ExclusiveGateway):
            # Only choose one child!
            probabilities = np.asarray([child.likelihood for child in self.children])
            s = sum(probabilities)
            if s == 0:
                return [self.children[0]]

            probabilities = probabilities / s  # make equal to one
            child = np.random.choice(self.children, 1, p=probabilities)[0]
            child.likelihood *= self.likelihood_change
            return [child]

        # As long as not all branches have reached this merging gateway -> do not proceed
        if isinstance(self.node, BPMN.ParallelGateway) and len(self.node.get_in_arcs()) > 1:
            # Did we reach the hit count?
            if self.hit_count < len(self.node.get_in_arcs()):
                return []
            else:
                # reset once we proceed!
                self.hit_count = 0

        for child in self.children:
            child.likelihood *= self.likelihood_change
        return self.children


def build_bpmn_graph(file_path: str):
    model = pm4py.read_bpmn(file_path)
    start = None
    for n in model.get_nodes():
        if isinstance(n, BPMN.StartEvent):
            start = n
            break

    if start is None:
        raise Exception('Could not find start event!')
    out = start.get_out_arcs()
    if len(out) != 1:
        raise Exception('Start event points to more than one activity!')

    node_list: typing.List[Node] = list()
    root = Node(start.get_out_arcs()[0].target)
    add_nodes(root, node_list)
    return root, model


def add_nodes(node: Node, node_list: typing.List[Node]):
    out_arcs = node.node.get_out_arcs()
    for out in out_arcs:
        target = out.get_target()

        new_node = None
        for n in node_list:
            if n.node == target:
                new_node = n
                break

        if new_node is None:
            new_node = Node(target)
            node_list.append(new_node)
            add_nodes(new_node, node_list)

        node.children.append(new_node)


def generate_trace(root: Node):
    root.reset()
    trace = []
    nodes_to_advance = [root]

    while len(nodes_to_advance) > 0:
        # Choose random node
        index = np.random.randint(0, len(nodes_to_advance))
        node = nodes_to_advance[index]

        # Get value from node
        val = node.get_value()
        if val is not None:
            trace.append(val)

        # Get next nodes
        next_ones = node.get_next_nodes()

        # Update list, i.e. remove old one, insert new one
        nodes_to_advance = nodes_to_advance[:index] + next_ones + nodes_to_advance[index + 1:]

        if len(trace) > 1000:
            # "Emergency stop", e.g. if stuck in a loop
            break

    root.reset()
    return trace


if __name__ == "__main__":
    root, bpmn_model = build_bpmn_graph(args.bpmn_path)

    traces = []
    trace_df = pd.DataFrame()

    trace_num = args.trace_count
    trace_set: typing.Set[typing.Tuple[str]] = set()
    for i in tqdm(range(max(10000, trace_num)), desc="Generating traces"):
        trace: typing.List[str] = generate_trace(root)
        trace_set.add(tuple(trace))
        traces.append(trace)

    # Balance traces
    if args.balanced:
        trace_num_per_case = math.ceil(trace_num / len(trace_set))
        traces = []
        for trace in trace_set:
            for i in range(trace_num_per_case):
                traces.append(trace)

    # Or if not balanced or hard restrict -> just use the start
    if not args.balanced or args.hard_restrict:
        traces = traces[:trace_num]
    # And shuffle
    np.random.shuffle(traces)

    date_now = datetime.utcnow()
    process_id = bpmn_model.get_process_id()

    # Transform into Dataframe
    for i, trace in enumerate(tqdm(traces, desc="Transforming into Dataframe")):
        case_id = [str(i + 1) for _ in range(len(trace))]
        org_group = [process_id for _ in range(len(trace))]

        # Timestamps need this exact format, especially milliseconds, but not more!
        # Example date: 2023-04-21T12:01:21.374000+00:00
        timestamps = [pd.to_datetime((date_now + timedelta(seconds=i * 10)).strftime('%FT%T.%f')[:-3], utc=True)
                      for i in range(len(trace))]

        trace_df = pd.concat([trace_df, pd.DataFrame(
            {'concept:name': trace,
             'case:concept:name': case_id,
             'org:group': org_group,
             'time:timestamp': timestamps})])

    print(f'Unique traces: {len(trace_set)}')
    pm4py.write_xes(trace_df, args.output_path)
