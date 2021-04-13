"""
This module contains model classes for the controllers.NetworkAnalyzer.
"""
import traitlets

from networkx.drawing.layout import (
    fruchterman_reingold_layout,
    kamada_kawai_layout,
    spectral_layout,
)


class RangeFilter(traitlets.HasTraits):
    """
    Generic filter that returns all indexes of a DataFrame that are within a defined
    range.

    Args:
        data_frame (pandas.DataFrame): DataFrame to be subject to filtering.
        col (str): Name of column in data_frame that will be filtered by.

    Attributes:
        data_frame (pandas.DataFrame): DataFrame to be subject to filtering.
        col (str): Name of column in data_frame that will be filtered by.
        val_range (traitlets.Tuple): Range tuple with minimum value and maximum value.
        val_ids (traitlets.List): List of indexes in data_frame that passed the filter.
    """

    val_range = traitlets.Tuple()
    val_ids = traitlets.List()

    def __init__(self, data_frame, col):
        super().__init__()
        self.data_frame = data_frame
        self.col = col

    # pylint: disable=unused-argument, no-self-use
    @traitlets.validate("val_range")
    def _validate_val_range(self, proposal):
        """
        Makes sure that proposed value for val_range in valid.
        Is executed automagically everytime self.val_range changes.

        Args:
            proposal (dict): The proposed value.

        Returns:
            tuple: The new value for val_range.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        val_range = proposal["value"]
        if len(val_range) != 2:
            raise traitlets.TraitError("val_range must be of length 2.")
        if val_range[0] > val_range[1]:
            raise traitlets.TraitError(
                "val_range[0] must be smaller than val_range[1]."
            )
        return val_range

    # pylint: enable=unused-argument, no-self-use

    # pylint: disable=unused-argument
    @traitlets.observe("val_range")
    def _update_val_ids(self, change=None):
        """
        Updates the val_ids attribute.
        Is executed automagically everytime self.val_range changes.

        Args:
            change (dict, optional): Change.
        """

        val_min, val_max = self.val_range
        self.val_ids = list(
            self.data_frame[
                (self.data_frame[self.col] >= val_min)
                & (self.data_frame[self.col] <= val_max)
            ].index
        )
        # pylint: enable=unused-argument


class NodesEdgesFilter(traitlets.HasTraits):
    """
    Filters nodes and edges based on defined ranges.

    Args:
        nodes (pandas.DataFrame): DataFrame with nodes.
        edges (pandas.DataFrame): DataFrame with edges.

    Attributes:
        nodes (pandas.DataFrame): DataFrame with nodes.
        edges (pandas.DataFrame): DataFrame with edges.
        nodes_range (traitlets.Tuple): Range for node size.
        edges_range (traitlets.Tuple): Range for edge weight.
        nodes_ids (traitlets.List): List of node indexes that passed the filter.
        edges_ids (traitlets.List): List of edge indexes that passed the filter.
        _nodes_filter (RangeFilter): Filter for the nodes.
        _edges_filter (RangeFilter): Filter for the edges.
    """

    nodes_range = traitlets.Tuple(default_value=(0.25, 1.0))
    edges_range = traitlets.Tuple(default_value=(0.75, 1.0))
    nodes_ids = traitlets.List()
    edges_ids = traitlets.List()

    def __init__(self, nodes, edges):
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self._nodes_filter = RangeFilter(nodes, "size")
        self._edges_filter = RangeFilter(edges, "weight")
        self._update_nodes_ids()

    # pylint: disable=unused-argument
    @traitlets.observe("nodes_range")
    def _update_nodes_ids(self, change=None):
        """
        Updates the nodes_ids attribute.
        Is executed automagically everytime self.nodes_range changes.

        Args:
            change (dict, optional): Change.
        """
        self._nodes_filter.val_range = self.nodes_range
        self.nodes_ids = self._nodes_filter.val_ids
        self._update_edges_filtered(change)

    # pylint: enable=unused-argument

    def _filter_edges(self, edges_filtered):
        """
        Filters edges, so that edges only connect available nodes.

        Args:
            edges_filtered (list): Pre-filtered edges.

        Returns:
            list: Post-filtered edges.
        """
        return [
            edge
            for edge in edges_filtered
            if edge[0] in self.nodes_ids and edge[1] in self.nodes_ids
        ]

    # pylint: disable=unused-argument
    @traitlets.observe("edges_range")
    def _update_edges_filtered(self, change=None):
        """
        Updates the edges_ids attribute.
        Is executed automagically everytime self.edges_range changes.

        Args:
            change (dict, optional): Change.
        """
        self._edges_filter.val_range = self.edges_range
        edges_ids = self._edges_filter.val_ids
        self.edges_ids = self._filter_edges(edges_ids)

    # pylint: enable=unused-argument


class GraphLayout(traitlets.HasTraits):
    """
    Calculates layouts for a network graph.

    Args:
        graph (networkx.Graph): Graph.
        seed (int): Random number seed for stochastic layout methods.

    Attributes:
        graph (networkx.Graph): Graph.
        seed (int): Random number seed for stochastic layout methods.
        layout_method (traitlets.Unicode): Name of the layout method.
        k (traitlets.Float): Optimal distance between nodes.
        iterations (traitlets.Integer): Number of iterations.
        threshold (traitlets.Float): Threshold for relative error in node position
            changes.
        layout (traitlets.Dict): Layout with node ids as keys and (x, y) coordinates
            as values.
    """

    layout_method = traitlets.Unicode(default_value="fruchterman_reingold_layout")
    k = traitlets.Float(default_value=0.1)
    iterations = traitlets.Integer(default_value=50)
    threshold = traitlets.Float(default_value=0.0001)
    layout = traitlets.Dict()

    def __init__(self, graph, seed=42):
        super().__init__()
        self.graph = graph
        self.seed = seed
        self.update_layout()

    @property
    def layout_method_mapper(self):
        """
        Dictionary that maps layout method names to the methods.

        Returns:
            callable: A networkx.drawing.layout method.
        """
        return {
            "kamada_kawai_layout": kamada_kawai_layout,
            "fruchterman_reingold_layout": fruchterman_reingold_layout,
            "spectral_layout": spectral_layout,
        }

    @traitlets.validate("layout_method")
    def _validate_layout_method(self, proposal):
        """
        Makes sure that the proposed value for layout_method is valid.
        Is executed automagically everytime self.layout_method changes.

        Args:
            proposal (dict): Proposed value.

        Returns:
            str: New value for layout_method.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        layout_method = proposal["value"]
        if layout_method not in self.layout_method_mapper.keys():
            raise traitlets.TraitError(
                "layout_method must be one of {}".format(
                    self.layout_method_mapper.keys()
                )
            )
        return layout_method

    # pylint: disable=no-self-use
    @traitlets.validate("k")
    def _validate_k(self, proposal):
        """
        Makes sure that the proposed value for k is valid.
        Is executed automagically everytime self.k changes.

        Args:
            proposal (dict): Proposed value.

        Returns:
            int: New value for k.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        k = proposal["value"]
        if k <= 0:
            raise traitlets.TraitError("k must be greater than 0.")
        return k

    # pylint: enable=no-self-use

    # pylint: disable=no-self-use
    @traitlets.validate("iterations")
    def _validate_iterations(self, proposal):
        """
        Makes sure that the proposed value for iterations is valid.
        Is executed automagically everytime self.iterations changes.

        Args:
            proposal (dict): Proposed value.

        Returns:
            int: The new value for iterations.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        iterations = proposal["value"]
        if iterations <= 0:
            raise traitlets.TraitError("iterations must be greater than 0.")
        return iterations

    # pylint: enable=no-self-use

    # pylint: disable=no-self-use
    @traitlets.validate("threshold")
    def _validate_threshold(self, proposal):
        """
        Makes sure that the proposed value for threshold is valid.
        Is executed automagically everytime self.threshold changes.

        Args:
            proposal (dict): Proposed value.

        Returns:
            int: The new value for threshold.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        threshold = proposal["value"]
        if threshold <= 0:
            raise traitlets.TraitError("threshold must be greater than 0.")
        return threshold

    # pylint: enable=no-self-use

    # pylint: disable=unused-argument
    @traitlets.observe("layout_method", "k", "iterations", "threshold")
    def update_layout(self, change=None):
        """
        Updates the layout attribute.
        Is executed if any of self.layout_method, self.k, self.iterations or
        self.threshold changes.

        Args:
            change (dict, optional): Change.
        """
        layout_method = self.layout_method_mapper[self.layout_method]
        if self.layout_method in "fruchterman_reingold_layout":
            self.layout = layout_method(
                self.graph,
                k=self.k,
                iterations=self.iterations,
                threshold=self.threshold,
                seed=self.seed,
            )
        else:
            self.layout = layout_method(self.graph)

    # pylint: enable=unused-argument


class NodesFocus(traitlets.HasTraits):

    """
    Generates a focus zoom in for a provided node_id.

    Args:
        nodes (pandas.DataFrame): DataFrame with nodes.
        layout (traitlets.Dict): Initial layout of nodes.
        dist (float): Distance from node in all directions in zoom.

    Attributes:
        nodes (pandas.DataFrame): DataFrame with nodes.
        dist (float): Distance from node in all directions in zoom.
        node_id (traitlets.Unicode):  ID of focus node.
        layout (traitlets.Dict): Layout of nodes.
        coords (traitlets.Tuple): Coords for plot in the format x0, x1, y0, y1.
    """

    layout = traitlets.Dict()
    node_id = traitlets.Unicode()
    coords = traitlets.Tuple()

    def __init__(self, nodes, layout, dist=0.05):
        super().__init__()
        self.nodes = nodes
        self.layout = layout
        self.dist = dist

    @traitlets.validate("layout")
    def _validate_layout(self, proposal):
        """
        Makes sure that the proposed value for layout is valid.
        Is executed automagically everytime self.layout changes.

        Args:
            proposal (dict): The proposed value.

        Returns:
            dict: The new value for layout.

        Raises:
            traitlets.TraitError: If proposed value is not valid.
        """
        layout = proposal["value"]
        layout_ids = set(layout.keys())
        if layout_ids != set(self.nodes.index):
            raise traitlets.TraitError("layout.keys() must equal self.nodex.index.")
        return layout

    @traitlets.validate("node_id")
    def _validate_node_id_(self, proposal):
        """
        Makes sure that the proposed value for node_id is valid.
        Is executed automagically everytime self.node_id changes.

        Args:
            proposal (dict): The proposed value.

        Returns:
            str: The new value for node_id.

        Raises:
            traitlets.TraitError: If the proposed value is not valid.
        """
        node_id = proposal["value"]
        if not node_id or node_id not in self.nodes.index:
            raise traitlets.TraitError("node_id must be in {}".format(self.nodes.index))
        return node_id

    # pylint: disable=unused-argument
    @traitlets.observe("node_id")
    def _update_coords(self, change=None):
        """
        Updates the coords attribute.
        Is executed automagically everytime self.node_id changes.

        Args:
            change (dict, optional): Change.
        """
        if self.node_id:
            x, y = self.layout[self.node_id]
            self.coords = (x - self.dist, x + self.dist, y - self.dist, y + self.dist)

    # pylint: enable=unused-argument
