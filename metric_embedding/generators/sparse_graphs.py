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
    The euler characteristic of a graph G = (V, E) is defined
    as the number of edges G has more then a tree, i.e. 
    chi(G) = |E| - |V| + 1
    """
    G: nx.Graph = nx.random_tree(n)  # type: ignore
    if m is None and chi is None:
        raise ValueError("either m or chi must be given")
    if m is None and chi is not None:
        m = chi + n - 1
    while G.number_of_edges() < m:  # type: ignore
        G.add_edge(*np.random.randint(0, n, 2))
    return G


def random_connected_expected_sparse_graph(n: int, chi: int):
    """
    returns a random graph with expected chi(G) = chi

    Parameters
    ----------
    n: number of vertices
    chi: The expected euler characteristic of the output graph

    Returns
    -------
    a networkx connected graph with n vertices and
    (n - 1 + chi) expected number of edges
    
    Notes
    -----
    The euler characteristic of a graph G = (V, E) is defined
    as the number of edges G has more then a tree, i.e. 
    chi(G) = |E| - (|V| - 1)
    """
    G: nx.Graph = nx.random_tree(n)  # type: ignore
    p = chi / ((n * (n - 1) / 2) - (n - 1))
    coins = np.random.rand(n * (n - 1) // 2 - (n - 1))
    possible_edges = (
        (u, v) for u, v in combinations(range(n), 2) 
        if not G.has_edge(u, v)
    )
    for i, (u, v) in enumerate(possible_edges):
        if coins[i] <= p:
            G.add_edge(u, v)
    return G


def random_graph_weighing(
    G: nx.Graph,
    min_weight: float,
    max_weight: float,
    weight: str
):
    """
    Sets the weights of the edges (in-place) to be randomly
    chosen numbers within the given range of [min_weight, max_weight]

    Parameters
    ----------
    G: a networkx graph
    min_weight: the minimum weight an edge can have
    max_weight: the maximum weight an edge can have
    weight: a string to be used as an attribute key for the weight

    Returns
    -------
    The same graph G that was given, with weights on it
    """
    if min_weight > max_weight:
        raise ValueError("min weight must be smaller the max_weight")
    weights = min_weight + (max_weight - min_weight) * np.random.rand(len(G.edges))
    
    for i, (u, v) in enumerate(G.edges):
        G[u][v][weight] = weights[i]
    
    return G
