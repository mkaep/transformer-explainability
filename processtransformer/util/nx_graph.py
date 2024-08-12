

from __future__ import annotations

import typing
from enum import Enum

import networkx as nx

from processtransformer.util.types import XEvent, YEvent


class NodeRole(Enum):
    # Only choose colors from Matplotlib! https://matplotlib.org/stable/gallery/color/named_colors.html
    PREDICTION = 'orange'
    ATTENTION = 'green'
    UNINVOLVED = 'lightblue'
    IN_TRACE = 'deepskyblue'


def clean_up_graph(graph: nx.DiGraph):
    # Remove nodes without edges
    nodes_to_remove: typing.Set[str] = set()
    for node in graph.nodes:
        # graph.edges is semantically the same as out-edges (I guess)
        if len(graph.in_edges(node)) == 0 and len(graph.edges(node)) == 0:
            nodes_to_remove.add(node)
    for r in nodes_to_remove:
        graph.remove_node(r)
    # Remove self-loops
    edges_to_remove = set()
    for edge in graph.edges:
        if edge[0] == edge[1]:
            edges_to_remove.add(edge)
    for edge in edges_to_remove:
        graph.remove_edge(*edge)


def mat_to_graph(mat: typing.Dict[YEvent, typing.Dict[XEvent, bool]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(list(mat.values())[0].keys())
    for v, us in mat.items():
        for u, draw_edge in us.items():
            if draw_edge:
                graph.add_edge(u, v)

    return graph
