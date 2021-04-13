"""
This module contains views for controllers.NetworkAnalyzer.
"""

from ipywidgets import widgets
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import traitlets
import random


# pylint: disable=too-many-ancestors
class LayoutView(widgets.GridBox):
    """
    A View for defining a layout method and the corresponding hyper parameters.

    Attributes:
        layout_method (widgets.Dropwdown): Dropdown for the layout method.
        k (widgets.BoundedFloatText): k parameter for layout method.
        iterations (widgets.BoundedIntText): interations parameter for layout method.
        threshold (widgets.BoundedFloatText): threshold parameter for layout method.
    """

    def __init__(self):
        super().__init__()

        self.layout_method = widgets.Dropdown(
            options=[
                "fruchterman_reingold_layout",
                "kamada_kawai_layout",
                "spectral_layout",
                "spring_layout",
            ],
            value="fruchterman_reingold_layout",
        )

        k_label = widgets.Label("k")
        self.k = widgets.BoundedFloatText(value=0.1, min=0, max=1, step=0.05)

        iterations_label = widgets.Label("iterations")
        self.iterations = widgets.BoundedIntText(value=50, min=0, max=500, step=50)

        threshold_label = widgets.Label("threshold")
        self.threshold = widgets.BoundedFloatText(
            value=0.0001, min=0, max=0.1, step=0.0001
        )

        self.children = [
            self.layout_method,
            widgets.VBox([k_label, iterations_label, threshold_label]),
            widgets.VBox([self.k, self.iterations, self.threshold]),
        ]

        self.layout_method.observe(self._update)

        self.layout = widgets.Layout(grid_template_columns="repeat(3, 400px)")

    # pylint: disable=unused-argument
    def _update(self, change=None):
        """
        Disables and enables the hyperparamters, depending on the layout_method.

        Args:
            change (dict, optional): Change.
        """
        if self.layout_method.value in ("spectral_layout", "spring_layout"):
            self.k.disabled = True
            self.iterations.disabled = True
            self.threshold.disabled = True
        else:
            self.k.disabled = False
            self.iterations.disabled = False
            self.threshold.disabled = False

    # pylint: enable=unused-argument


class RangeView(widgets.HBox):
    """
    A generic view with a range slider.
    """

    def __init__(self):
        super().__init__()

        self.range = widgets.FloatRangeSlider(
            value=(0.5, 1),
            min=0,
            max=1,
            step=0.01,
            readout=True,
            readout_format=".2f",
            continuous_update=False,
        )

        self.children = [self.range]


class FilterView(widgets.GridBox):
    """
    A filter view with two range sliders for node size and edge weight.

    Attributes:
        nodes_range (RangeView): Filter for the node size.
        edges_range (RangeView): Filter for the edge weight.
    """

    def __init__(self):

        super().__init__()

        nodes_label = widgets.Label(value="Node Size:")
        self.nodes_range = RangeView()

        edges_label = widgets.Label(value="Edge Weight:")
        self.edges_range = RangeView()

        self.children = [nodes_label, self.nodes_range, edges_label, self.edges_range]

        self.layout = widgets.Layout(grid_template_columns="repeat(2, 400px)")

    @property
    def node_size_range(self):
        """
        Node size range.
        """
        return self.nodes_range.range.value

    @property
    def edge_weight_range(self):
        """
        Edge weight range.
        """
        return self.edges_range.range.value


class NodeFocusView(widgets.GridBox):
    """
    View to focus on specific nodes.

    Args:
        node_ids (list): Valid node ids.

    Attributes:
        node_ids (list): Valid node ids.
        node_ids (widgets.Dropdown): Dropdown with node_ids.
    """

    def __init__(self, node_ids):
        super().__init__()
        self.node_ids = node_ids

        node_id_label = widgets.Label("Node ID:")
        #self.node_id = widgets.Dropdown(value="data science", options=node_ids)
        self.node_id = widgets.Dropdown(value=random.choice(node_ids), options=node_ids)

        self.children = [node_id_label, self.node_id]

        self.layout = widgets.Layout(grid_template_columns="repeat(2, 400px)")


class NetworkGraph(traitlets.HasTraits):
    """
    View component with plotly figure.

    Args:
        nodes (pandas.DataFrame): Nodes.
        edges (pandas.DataFrame): Edges.

    Attributes:
        nodes (pandas.DataFrame): Nodes.
        edges (pandsa.DataFrame): Edges.
        _nodes_ids (traitlets.List): Selected nodes ids.
        _edges_ids (traitlets.List): Selected edges ids.
        _layout_data (traitlets.Dict): Layout data.
        _node_focus (traitlets.Unicode): Focus node.
        nodes_plottable (plotly.graph_objs.Scatter): Nodes to plot.
        edges_plottable (list): Edges to plot.
        fig (ploty.graph_obs.FigureWidget): Figure.
    """

    _nodes_ids = traitlets.List()
    _edges_ids = traitlets.List()
    _layout_data = traitlets.Dict()
    _node_focus = traitlets.Unicode()

    def __init__(self, nodes, edges):

        super().__init__()

        self.nodes = nodes
        self.edges = edges

        self._nodes_ids = list(self.nodes.index)
        self._edges_ids = []

        self.nodes_plottable = None
        self.edges_plottable = None

        self.fig = go.FigureWidget()
        self.fig.layout.update(showlegend=False)
        self.fig.layout.coloraxis.update(cmin=0, cmax=1)
        self.fig.layout.xaxis.update(
            showgrid=False, zeroline=False, showticklabels=False
        )
        self.fig.layout.yaxis.update(
            showgrid=False, zeroline=False, showticklabels=False
        )

    @property
    def nodes_ids(self):
        """
        Nodes IDs.
        """
        return pd.Series(self._nodes_ids)

    @property
    def edges_ids(self):
        """
        Edges IDs.
        """
        return pd.Series(self._edges_ids)

    @property
    def layout_data(self):
        """
        Layout data.
        """
        if self._layout_data:
            layout_data = pd.DataFrame(self._layout_data).T
            layout_data.columns = ["x", "y"]
            return layout_data
        return None

    @traitlets.validate("_nodes_ids")
    def validate_nodes_ids(self, proposal):
        """"
        Makes sure that the proposed value for _nodes_ids is valid.
        Is executed automatically everytime _nodes_ids attempts to change.

        Args:
            proposal (dict): Proposed value.

        Return:
            list: New value for _nodes_ids.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        if not proposal["value"]:
            return None
        nodes_ids = pd.Series(proposal["value"])
        if not all(nodes_ids.isin(self.nodes.index)):
            raise traitlets.TraitError("_nodes_ids must be in self.nodes.index.")
        return proposal["value"]

    @traitlets.validate("_edges_ids")
    def validate_edges_ids(self, proposal):
        """"
        Makes sure that the proposed value for _edges_ids is valid.
        Is executed automatically everytime _edges_ids attempts to change.

        Args:
            proposal (dict): Proposed value.

        Return:
            list: New value for _edges_ids.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        if not proposal["value"]:
            return None
        edges_ids = pd.Series(proposal["value"])
        if not all(edges_ids.isin(self.edges.index)):
            raise traitlets.TraitError("_edges_ids must be in self.edges.index.")
        return proposal["value"]

    @traitlets.validate("_layout_data")
    def validate_layout_data(self, proposal):
        """"
        Makes sure that the proposed value for _layout_data is valid.
        Is executed automatically everytime _layout_data attempts to change.

        Args:
            proposal (dict): Proposed value.

        Return:
            dict: New value for _layout_data.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        if not proposal["value"]:
            return None
        layout_data_ids = set(proposal["value"].keys())
        if not layout_data_ids == set(self.nodes.index):
            raise traitlets.TraitError(
                "_layout_data.keys() must equal self.nodes.index."
            )
        return proposal["value"]

    @traitlets.validate("_node_focus")
    def validate_node_focus(self, proposal):
        """"
        Makes sure that the proposed value for _node_focus is valid.
        Is executed automatically everytime _node_focus attempts to change.

        Args:
            proposal (dict): Proposed value.

        Return:
            str: New value for _node_focus.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        node_focus = proposal["value"]
        if node_focus not in self._nodes_ids:
            raise traitlets.TraitError("_node_focus must be in self.node_ids.")
        return node_focus

    # pylint: disable=unused-argument
    @traitlets.observe("_node_focus")
    def update_fig_range(self, change=None):
        """
        Updates the figure range.
        Is executed automatically when _node_focus changes.

        Args:
            change (dict, optional): Change.
        """
        x, y = self._layout_data[self._node_focus]
        self.fig.update_xaxes(range=[x - 0.05, x + 0.05])
        self.fig.update_yaxes(range=[y - 0.05, y + 0.05])

    # pylint: enable=unused-argument

    def _update_nodes_plottable(self):
        """
        Updates the plottable nodes.
        """
        nodes_plottable = px.scatter(
            self.nodes.merge(
                self.layout_data, how="left", left_index=True, right_index=True
            )
            .loc[self.nodes_ids]
            .reset_index(),
            x="x",
            y="y",
            size="size",
            color="size",
            hover_name="id",
        ).data[0]
        nodes_plottable.marker = nodes_plottable.marker.update({"cmin": 0, "cmax": 1})
        self.nodes_plottable = nodes_plottable

    def _update_edges_plottable(self):
        """
        Updates the plottable edges.
        """
        self.edges_plottable = (
            self.edges.loc[self.edges_ids]
            .reset_index()
            .rename({"level_0": "from", "level_1": "to"}, axis=1)
            .merge(self.layout_data, left_on="from", right_index=True)
            .merge(
                self.layout_data,
                left_on="to",
                right_index=True,
                suffixes=("_from", "_to"),
            )
            .set_index(["from", "to"])
            .apply(lambda x: self._make_edge(**x), axis=1)
        )

    # pylint: disable=unused-argument
    @traitlets.observe("_layout_data")
    def _update_layout(self, changes=None):
        """
        Updates the layout in the figure.
        Is executed automatically every time _layout_data changes.

        Args:
            changes (dict, optional): Changes.
        """

        if self._nodes_ids and self._edges_ids:
            self._update_nodes_plottable()
            self._update_edges_plottable()

            self.fig.data = []

            self.fig.add_traces(list(self.edges_plottable))
            self.fig.add_traces(self.nodes_plottable)

    # pylint: enable=unused-argument

    @staticmethod
    def _make_edge(x_from, x_to, y_from, y_to, weight):
        """
        Creates an edge scatter.

        Args:
            x_from (float): Start x coordinate.
            x_to (float): End x coordinate.
            y_from (float): Start y coordinate.
            y_to (float): End y coordinate.
            weight (float): Weight of the edge.

        Returns:
            plotly.graph_objs.Scatter: An edge scatter.
        """
        return go.Scatter(
            x=[x_from, x_to],
            y=[y_from, y_to],
            mode="lines",
            line=dict(color="#888", width=weight),
            opacity=weight,
            hoverinfo="none",
        )

    # pylint: disable=unused-argument
    @traitlets.observe("_nodes_ids", "_edges_ids")
    def update_fig_nodes_edges(self, changes=None):
        """
        Updates the nodes and edges in the figure.
        Is executed automatically whenever _nodes_ids or edges_ids change.

        Args:
            changes (dict, optional): Changes.
        """
        if self.layout_data is not None:
            self._update_nodes_plottable()
            self._update_edges_plottable()

            self.fig.data = []
            self.fig.add_traces(list(self.edges_plottable.values))
            self.fig.add_trace(self.nodes_plottable)

    # pylint: enable=unused-argument


class NetworkAnalyzerView(widgets.VBox):
    """
    The final view for the NetworkAnalyzer that contains all components.

    Args:
        nodes (pandas.DataFrame): Nodes.
        edges (pandas.DataFrame): Edges.

    Attributes:
        layout_view (LayoutView): Layout view.
        filter_view (FilterView): Filter view.
        focus_view (NodeFocusView): Focus view
        graph_view (NetworkGraph): Graph view.
    """

    def __init__(self, nodes, edges):

        super().__init__()

        self.layout_view = LayoutView()
        self.filter_view = FilterView()
        self.focus_view = NodeFocusView(nodes.index)
        self.graph_view = NetworkGraph(nodes, edges)

        accordion = widgets.Accordion(
            [self.layout_view, self.filter_view, self.focus_view]
        )

        accordion.set_title(0, "Layout")
        accordion.set_title(1, "Filter")
        accordion.set_title(2, "Focus")

        self.children = [accordion, self.graph_view.fig]
