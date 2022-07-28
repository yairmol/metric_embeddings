from collections import OrderedDict
from typing import Dict, TypeVar

import networkx as nx
import retworkx as rx
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.utils.rx_utils import (RxGraphWrapper,
                                             dijkstra_shortest_path_lengths)

T = TypeVar("T")


def __rx_unit_weight(_):
    return 1


class GraphMetricSpace(FiniteMetricSpace[T]):
    def __init__(self, G: nx.Graph, weight=None):
        super().__init__(set(G.nodes), self.shortest_path_metric)
        self.G = G
        self.__G = RxGraphWrapper.from_networkx_graph(G)
        self.__sp_dict = OrderedDict()
        self.__max_sp_buffer_size = 1
        if weight is None:
            self.__weight = __rx_unit_weight
        elif callable(weight):
            self.__weight = weight
        else:
            if not isinstance(weight, str):
                raise ValueError(
                    "weight parameter must be None, callable or string"
                )
            self.weight_str = weight
            self.__weight = self.__edge_weight
    
    def __edge_weight(self, e_data: Dict[str, float]):
        return e_data[self.weight_str]

    def shortest_path_metric(self, u: T, v: T):
        if u == v:
            return 0
        if u in self.__sp_dict:
            return self.__sp_dict[u][v]
        elif v in self.__sp_dict:
            return self.__sp_dict[v][u]
        if len(self.__sp_dict) == self.__max_sp_buffer_size:
            self.__sp_dict.popitem(last=False)
        self.__sp_dict[u] = dijkstra_shortest_path_lengths(
            self.__G, u, self.__weight
        )
        return self.__sp_dict[u][v]
