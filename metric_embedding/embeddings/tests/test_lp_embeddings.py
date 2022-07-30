from itertools import combinations

import networkx as nx
import numpy as np
from metric_embedding.core.embedding_analysis import embedding_distortion
from metric_embedding.embeddings.lp_embeddings import (
    _1_2_elimination, l_infinity_tree_embedding, tree_separator)
from metric_embedding.metrics.graph_metric import GraphMetricSpace
from metric_embedding.metrics.lp_metrics import LpMetric


class TestTreeEmbedding:

    epsilon = 10 ** -5

    def test_tree_separator_custom_tree(self):
        T = nx.Graph()
        T.add_edges_from(
            [("D", "C"), ("C", "B"), ("C", "A"),
             ("D", "E"), ("E", "F"), ("E", "G")]
        )
        s, C = tree_separator(T)
        assert s == "D"
        assert C == {"A", "B", "C"} or C == {"E", "F", "G"}

    def test_tree_separator(self):
        N = 10
        T: nx.Graph = nx.random_tree(N)  # type: ignore
        s, C = tree_separator(T)
        assert s in T.nodes
        assert C.issubset(T.nodes)
        T.remove_node(s)
        CCs = list(nx.connected_components(T))
        C_max = CCs[np.argmax([len(c) for c in CCs])]
        assert C_max == C
        assert len(C_max) <= 2 * N / 3
    
    def test_tree_embedding_small_tree(self):
        T = nx.Graph()
        T.add_edges_from([(0, 1), (1, 2)])
        f = l_infinity_tree_embedding(GraphMetricSpace(T))
        opt1 = {0: np.array([-1]), 1: np.array([0]), 2: np.array([1])}
        opt2 = {0: np.array([1]), 1: np.array([0]), 2: np.array([-1])}
        assert f == opt1 or f == opt2
    
    def test_tree_embedding_small_weighted_tree(self):
        T = nx.Graph()
        T.add_edges_from([(0, 1, {"weight": 0.2}), (1, 2, {"weight": 1.3})])
        f = l_infinity_tree_embedding(GraphMetricSpace(T, weight="weight"))
        opt1 = {0: np.array([-0.2]), 1: np.array([0]), 2: np.array([1.3])}
        opt2 = {0: np.array([0.2]), 1: np.array([0]), 2: np.array([-1.3])}
        assert f == opt1 or f == opt2
    
    def test_tree_embedding(self):
        N = 128
        T: nx.Graph = nx.random_tree(N)  # type: ignore
        dT = GraphMetricSpace(T)
        f = l_infinity_tree_embedding(dT)
        dim = len(f[next(iter(T.nodes))])
        assert all(len(f[u]) == dim for u in T.nodes)
        linf = LpMetric(float('inf'))
        assert embedding_distortion(dT, linf.d, lambda x: f[x]) == 1
        assert dim <= int((1 / (np.log2(3) - 1)) * np.log2(len(dT)))
    
    def test_tree_embedding_weighted(self):
        N = 128
        T: nx.Graph = nx.random_tree(N)  # type: ignore
        edge_weights = np.random.rand(len(T.edges))
        for i, (u, v) in enumerate(T.edges):
            T[u][v]["weight"] = edge_weights[i]
        dT = GraphMetricSpace(T, weight="weight")
        f = l_infinity_tree_embedding(dT)
        dim = len(f[next(iter(T.nodes))])
        assert all(len(f[u]) == dim for u in T.nodes)
        linf = LpMetric(float('inf'))
        assert embedding_distortion(dT, linf.d, lambda x: f[x]) - 1 < self.epsilon
        assert dim <= int((1 / (np.log2(3) - 1)) * np.log2(len(dT)))


class TestSparseGraphEmbedding:
    def test_1_2_elimination(self):
        G: nx.Graph = nx.complete_graph(4)  # type: ignore
        tree_edges = [(0, 4), (0, 5), (4, 6), (4, 7)]
        G.add_edges_from(tree_edges) # a tree hanged from 0
        G.add_edges_from([(1, 8), (8, 9), (9, 3)]) # an isolated path
        for u, v in G.edges:
            G[u][v]["weight"] = 1
        G1, F, Ps = _1_2_elimination(G, "weight")
        assert set(G1.edges) == {(u, v, 0) for u, v in combinations(range(4), 2)}.union({(1, 3, 1)})  # type: ignore
        assert {frozenset(e) for e in F.edges} == {frozenset(e) for e in tree_edges}
        assert Ps == [[1, 8, 9, 3]] or Ps == [[3, 8, 9, 1]]

