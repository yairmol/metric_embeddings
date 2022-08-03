import networkx as nx
from metric_embedding.generators.sparse_graphs import (
    random_connected_expected_sparse_graph, random_connected_sparse_graph)


class TestSparseGraphGen:
    
    n_tries = 10

    def test_random_sparse_graph(self):
        n, m = 100, 102
        for _ in range(self.n_tries):
            G = random_connected_sparse_graph(n, m)
            assert len(G.nodes) == n
            assert len(G.edges) == m
            assert nx.is_connected(G)
    
    def test_random_sparse_graph_with_chi(self):
        n, chi = 100, 3
        for _ in range(self.n_tries):
            G = random_connected_sparse_graph(n, chi=chi)
            assert len(G.nodes) == n
            assert len(G.edges) == n - 1 + chi
            assert nx.is_connected(G)
        
    def test_random_expected_sparse_graph(self):
        n, chi = 256, 10
        m = n * (n - 1) // 2 - (n - 1)
        p = chi / m
        std = (m * p * (1 - p)) ** 0.5
        for _ in range(self.n_tries):
            G = random_connected_expected_sparse_graph(n, chi)
            assert len(G.nodes) == n
            assert nx.is_connected(G)
            assert n - 1 + max(chi - 3 * std, 0) <= len(G.edges) <= n - 1 + chi + 3 * std
