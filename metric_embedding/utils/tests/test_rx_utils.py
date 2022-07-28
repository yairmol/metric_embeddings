from time import time

import networkx as nx
from metric_embedding.utils.rx_utils import (RxGraphWrapper,
                                             dijkstra_shortest_path_lengths)


class TestRxGraphWrapper:

    N = 10

    def test_simple_functions(self):
        G = RxGraphWrapper()
        G.add_node("A")
        G.add_node("B")
        assert set(G.G.nodes()) == {"A", "B"}

        G.add_edge("A", "B", dict())
        assert set(G.edges()) == {("A", "B")}

        G.add_nodes_from(["C", "D", "E"])
        assert set(G.G.nodes()) == {"A", "B", "C", "D", "E"}

        G.add_edges_from([("A", "C", 1), ("A", "D", 1), ("B", "E", 1)])
        assert set(G.edges()) == {("A", "B"), ("A", "C"), ("A", "D"), ("B", "E")}
        assert G.degree("A") == 3
        assert G.degree("B") == 2
        assert G.degree("C") == 1

    def test_networkx_to_rx(self):
        G: nx.Graph = nx.path_graph(self.N)  # type: ignore
        Gr = RxGraphWrapper.from_networkx_graph(G)
        assert set(Gr.G.nodes()) == set(G.nodes)
        assert set(Gr.edges()) == set(G.edges())
    
    def test_retworkx_to_nx(self):
        G: nx.Graph = nx.path_graph(self.N)  # type: ignore
        Gr = RxGraphWrapper.from_networkx_graph(G)
        G1 = Gr.to_networkx_graph()
        assert G1.nodes() == G.nodes()
        assert G1.edges() == G.edges()
        assert list(G1.edges(data=True)) == list(G.edges(data=True))


class TestRxWrappedFunctions:
    
    N = 10

    def test_dijkstra_sp_length(self):
        N = 10
        G = RxGraphWrapper()
        G.add_nodes_from(chr(i) for i in range(ord("A"), ord("A") + N))
        G.add_edges_from(
            (chr(i), chr(i + 1), {"weight": 1})
            for i in range(ord("A"), ord("A") + N - 1)
        )
        assert dijkstra_shortest_path_lengths(G, "A", lambda x: x["weight"]) == {
            chr(i + ord("A")): i for i in range(1, N)
        }
    
    def test_dijkstra_sp_length_runtime(self):
        N = 10000
        offset = 100
        G = RxGraphWrapper()
        G.add_nodes_from(range(offset, offset + N))
        G.add_edges_from(
            (i, i + 1, {"weight": 1})
            for i in range(offset, offset + N - 1)
        )
        Gn = G.to_networkx_graph()
        start_1 = time()
        dijkstra_shortest_path_lengths(G, offset, lambda x: x["weight"])
        t1 = time() - start_1
        start_2 = time()
        nx.single_source_shortest_path_length(Gn, offset)
        t2 = time() - start_2
        assert t1 < 0.5 * t2
