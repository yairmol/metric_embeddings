from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, TypeVar)

import networkx as nx
import retworkx as rx

T = TypeVar("T")


class RxGraphWrapper(Generic[T]):
    def __init__(self) -> None:
        self.G = rx.PyGraph()  # type: ignore
        self.rev_map: Dict[T, int] = dict()
    
    @classmethod
    def from_networkx_graph(cls, G: nx.Graph):
        Gr = cls()
        Gr.add_nodes_from(G.nodes)
        Gr.add_edges_from(G.edges(data=True))  # type: ignore
        return Gr
    
    def add_node(self, n: T):
        i: int = self.G.add_node(n)
        self.rev_map[n] = i
    
    def add_nodes_from(self, nodes: Iterable[T]):
        nodes_lst: List[T] = list(filter(lambda x: x not in self.rev_map, nodes))
        indices = self.G.add_nodes_from(nodes_lst)
        self.rev_map.update(dict(zip(nodes_lst, indices)))
    
    def add_edge(self, node_a: T, node_b: T, data: Any):
        a, b = self.rev_map[node_a], self.rev_map[node_b]
        self.G.add_edge(a, b, data)
    
    def add_edges_from(self, edges: Iterable[Tuple[T, T, Any]]):
        iedges = [(self.rev_map[u], self.rev_map[v], data) for u, v, data in edges]
        self.G.add_edges_from(iedges)
    
    def add_edges_from_no_data(self, edges: Iterable[Tuple[T, T]]):
        iedges = [(self.rev_map[u], self.rev_map[v]) for u, v in edges]
        self.G.add_edges_from(iedges)
    
    def remove_node(self, node: T):
        n = self.rev_map[node]
        self.G.remove_node(n)

    def remove_edge(self, node_a: T, node_b: T):
        a, b = self.rev_map[node_a], self.rev_map[node_b]
        self.G.remove_edge(a, b)

    def to_networkx_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.G.nodes())
        G.add_edges_from(
            (self.G[u], self.G[v], self.G.get_edge_data(u, v))
            for u, v in self.G.edge_list()
        )
        return G
    
    def degree(self, node):
        return self.G.degree(self.rev_map[node])
    
    def edges(self, data=False):
        if data:
            return ((self.G[u], self.G[v], self.G.get_edge_data(u, v))
                    for u, v in self.G.edge_list())
        return ((self.G[u], self.G[v]) for u, v in self.G.edge_list())
    

WeightFunc = Callable[[Any], float]

def dijkstra_shortest_path_lengths(
    G: RxGraphWrapper[T], 
    s: T, 
    weight: WeightFunc,
    t: Optional[T] = None
):
    ti = G.rev_map[t] if t is not None else None
    d: Dict[int, float] = rx.dijkstra_shortest_path_lengths(
        G.G, G.rev_map[s], weight, ti
    )
    node_map = G.G
    return {node_map[u]: distance for u, distance in d.items()}
