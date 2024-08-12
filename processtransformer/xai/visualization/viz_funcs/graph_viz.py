
from __future__ import annotations

import typing

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pm4py import BPMN

from processtransformer.util.nx_graph import NodeRole
from processtransformer.xai.visualization.output_models.graph_output import GraphOutput
from processtransformer.xai.visualization.viz_funcs.base_viz import BaseViz, VizOutput, FigureOutput
from processtransformer.xai.visualization.common.figure_data import FigureData


class GraphViz(BaseViz):
    accepts = [GraphOutput]

    def __init__(self, graph_output: GraphOutput) -> None:
        super().__init__()
        self.graph_output = graph_output

    def visualize(self) -> typing.List[VizOutput]:
        graph = self.graph_output.graph
        bpmn_model = self.graph_output.bpmn_model
        pos = nx.fruchterman_reingold_layout(graph, seed=7, k=0.5)

        if bpmn_model is not None:
            for node in bpmn_model.get_nodes():
                if not isinstance(node, BPMN.Activity):
                    continue
                layout_node = bpmn_model.get_layout().get(node)
                x = layout_node.get_x()
                y = layout_node.get_y()
                if node.get_name() in pos.keys():
                    # Have to invert y - BPMN uses it differently than networkx
                    pos[node.get_name()] = np.asarray([x, -y])

        # if edge_label_list is None:
        #     edge_label_list = []

        font_size = 12
        node_size = 200

        cm = 1 / 2.54
        plt.figure(figsize=(20 * cm, 12 * cm))

        # nodes
        node_dict = {}
        for role in NodeRole:
            node_dict[role] = []

        for node in graph.nodes:
            color = NodeRole.UNINVOLVED
            if 'color' in graph.nodes[node].keys():
                color = graph.nodes[node]['color']
            node_dict[color].append(node)
        for key, value in node_dict.items():
            # nodes - only drawn to get the legend right
            nx.draw_networkx_nodes(graph, pos, nodelist=value, label=key.name,
                                   node_color=key.value, node_size=node_size, node_shape='s')
            # node labels
            value_dict = dict()
            all_NaN = True
            for v in value:
                try:
                    # Get attention score and round to two digits
                    attn = f'{nx.get_node_attributes(graph, "attn_score")[v]:.2f}'
                    all_NaN = False
                except KeyError:
                    attn = 'NaN'  # if event is not in trace
                value_dict[v] = f'{v}:\n{attn}'

            if all_NaN:
                value_dict = {v: v for v in value}
            # value = {v: v + str(v.attn_score) for v in value}
            nx.draw_networkx_labels(graph, pos, labels=value_dict, font_size=font_size, font_family="sans-serif",
                                    bbox=dict(boxstyle='round', pad=0.3, alpha=1.0, color=key.value,
                                              fill=False, linewidth=2.0))

        # Draw over previously drawn, colored nodes
        nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=node_size * 2, node_shape='o')

        # edges with spacing
        try:
            alpha_values = [max(0.0, min(1.0, graph.get_edge_data(*e)['weight'])) for e in
                            graph.edges()]  # restrict to 0..1
        except KeyError:
            alpha_values = None
        nx.draw_networkx_edges(graph, pos, node_size=node_size * 4.25, min_source_margin=10, min_target_margin=10,
                               arrowsize=30, arrowstyle='->',
                               alpha=alpha_values)

        # labels on edges
        # for label in edge_label_list:
        #     edge_label = nx.get_edge_attributes(graph, label)
        #     edge_label = {key: f'{value:.2f}' for key, value in edge_label.items()}
        #     nx.draw_networkx_edge_labels(graph, pos, edge_label, font_size=font_size)

        plt.legend()
        title = self.graph_output.figure_data.title
        if title is not None:
            plt.title(title)
        plt.tight_layout()

        file_path = self.graph_output.figure_data.file_path
        if file_path is not None:
            plt.savefig(file_path)

        plt.show()
        return [FigureOutput(file_path)]


def plot_graph_with_bpmn(graph: nx.DiGraph, bpmn: BPMN, path=None, title=None):
    g_out = GraphOutput(graph, bpmn, FigureData(path, title))
    g_viz = GraphViz(g_out)
    g_viz.visualize()
