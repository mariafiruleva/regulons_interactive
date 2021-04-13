"""
This module contains the controller that connects view and models.
"""
import networkx as nx
import traitlets

from .views import NetworkAnalyzerView
from .models import NodesEdgesFilter, GraphLayout


def _make_graph(nodes, edges):
    """
    Creates a graph from nodes and edges.

    Args:
        nodes (pandas.DataFrame): Nodes.
        edges (pandas.DataFrame): Edges.

    Returns:
        networkx.Graph: Graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from((key, value) for key, value in nodes.to_dict("index").items())
    graph.add_edges_from(
        (key[0], key[1], value) for key, value in edges.to_dict("index").items()
    )
    return graph


# pylint: disable=too-many-ancestors
class NetworkAnalyzer(NetworkAnalyzerView):
    """
    The controller for NetworkAnalyzer that bring together views and models.

    Args:
        nodes (pandas.DataFrame): Nodes.
        edges (pandas.DataFrame): Edges.

    Attributes:
        filter_model (models.NodesEdgesFilter): Filter model.
        layout_model (models.GraphLayout): layout model.
    """

    def __init__(self, nodes, edges):

        super().__init__(nodes, edges)

        self.filter_model = NodesEdgesFilter(nodes, edges)

        self.layout_model = GraphLayout(_make_graph(nodes, edges))

        traitlets.link(
            (self.filter_model, "nodes_range"),
            (self.filter_view.nodes_range.range, "value"),
        )
        traitlets.link(
            (self.filter_model, "edges_range"),
            (self.filter_view.edges_range.range, "value"),
        )
        traitlets.link(
            (self.filter_model, "nodes_ids"), (self.graph_view, "_nodes_ids")
        )
        traitlets.link(
            (self.filter_model, "edges_ids"), (self.graph_view, "_edges_ids")
        )
        traitlets.link(
            (self.filter_model, "nodes_ids"), (self.focus_view.node_id, "options")
        )

        traitlets.link(
            (self.layout_model, "layout_method"),
            (self.layout_view.layout_method, "value"),
        )
        traitlets.link((self.layout_model, "k"), (self.layout_view.k, "value"))
        traitlets.link(
            (self.layout_model, "iterations"), (self.layout_view.iterations, "value")
        )
        traitlets.link(
            (self.layout_model, "threshold"), (self.layout_view.threshold, "value")
        )
        traitlets.link((self.layout_model, "layout"), (self.graph_view, "_layout_data"))

        traitlets.link(
            (self.focus_view.node_id, "value"), (self.graph_view, "_node_focus")
        )
