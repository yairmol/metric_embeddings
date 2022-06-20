from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np


def random_connected_sparse_graph(
    n: int,
    m: Optional[int] = None,
    chi: Optional[int] = None
):
    """
    Generate a random connected sparse graph G with n vertices and m edges
    or if chi is given instead of m then m = n - 1 + chi

    Parameters
    ----------
    n: number of vertices
    m: If given, this will be the number of edges
    chi: The euler characteristic of the output graph, If given.
        either m is given or chi, exactly one of them.

    Notes
    -----
    The euler characteristic of a graph G = (V, E) is defined as the number
    of edges G has more then a tree, i.e. chi(G) = |E| - |V| + 1
    """
    G: nx.Graph = nx.random_tree(n)  # type: ignore
    if m is None and chi is None:
        raise ValueError("either m or chi must be given")
    if m is None and chi is not None:
        m = chi + n - 1
    while G.number_of_edges() < m:  # type: ignore
        G.add_edge(*np.random.randint(0, n, 2))
    return G


def connected_expected_sparse_graph(n: int, chi: int):
    """
    returns a random graph with expected chi(G) = chi

    Parameters
    ----------
    n: number of vertices
    chi: The euler characteristic of the output graph, If given.
        either m is given or chi, exactly one of them.
    
    Notes
    -----
    The euler characteristic of a graph G = (V, E) is defined as the number
    of edges G has more then a tree, i.e. chi(G) = |E| - |V| + 1
    """
    G: nx.Graph = nx.random_tree(n)  # type: ignore
    p = chi / ((n * (n - 1) / 2) - (n - 1))
    for u, v in combinations(range(n), 2):
        if G.has_edge(u, v):
            continue
        if np.random.rand(1) <= p:
            G.add_edge(u, v)
    return G
