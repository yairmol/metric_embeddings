from collections import OrderedDict
from typing import TypeVar

import networkx as nx
import retworkx as rx
from metric_embedding.core.metric_space import FiniteMetricSpace

T = TypeVar("T")

def nx_graph_to_rx_graph(G: nx.Graph):
    Gr = rx.PyGraph()  # type: ignore
    Gr.add_nodes_from(G.nodes)
    Gr.add_edges_from(list((u, v, G[u][v]) for u, v in G.edges))
    return Gr


def unit_weight(_):
    return 1


class GraphMetricSpace(FiniteMetricSpace[T]):
    def __init__(self, G: nx.Graph, weight=None):
        super().__init__(set(G.nodes), self.shortest_path_metric)
        self.G = G
        self.__G = nx_graph_to_rx_graph(self.G)
        self.__sp_dict = OrderedDict()
        self.__max_sp_buffer_size = 1
        if weight is None:
            self.__weight = unit_weight
        elif callable(weight):
            self.__weight = weight
        else:
            assert weight is str
            self.weight_str = weight
            self.__weight = self.__edge_weight
    
    def __edge_weight(self, e):
        return self.G.edges[e][self.weight_str]

    def shortest_path_metric(self, u: T, v: T):
        if u == v:
            return 0
        if u in self.__sp_dict:
            return self.__sp_dict[u][v]
        elif v in self.__sp_dict:
            return self.__sp_dict[v][u]
        if len(self.__sp_dict) == self.__max_sp_buffer_size:
            self.__sp_dict.popitem(last=False)
        self.__sp_dict[u] = rx.dijkstra_shortest_path_lengths(
            self.__G, u, self.__weight
        )
        return self.__sp_dict[u][v]
