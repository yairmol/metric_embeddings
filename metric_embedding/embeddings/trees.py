from typing import Dict, List, Set, Tuple, TypeVar, Union

import networkx as nx
import numpy as np
import rustworkx as rx
from numpy.typing import NDArray

from metric_embedding.metrics.graph_metric import GraphMetricSpace
from metric_embedding.utils.rx_utils import RxGraphWrapper

T = TypeVar("T")


def _separate_tree(
        rx_tree: RxGraphWrapper[T], separator_vertex: int # type: ignore
) -> Tuple[List[Set[int]], int, Dict[int, dict]]:
    sep_neighbors: Dict[int, dict] = rx_tree.G.adj(separator_vertex)
    sep_node: T = rx_tree.G[separator_vertex]
    rx_tree.G.remove_node(separator_vertex)
    CCs: List[Set[int]] = rx.connected_components(rx_tree.G)  # type: ignore
    separator_vertex: int = rx_tree.G.add_node(sep_node)
    rx_tree.G.add_edges_from([(u, separator_vertex, None) for u in sep_neighbors])
    return CCs, separator_vertex, sep_neighbors


def tree_separator(Tree: nx.Graph) -> Tuple[T, Set[T]]:
    """
    Finds a single vertex v that separates the tree well, i.e. every
    connected component of T - {v} is of size at most 2/3|T|
    
    Parameters
    ----------
    T: a networkx graph which is a tree

    Returns
    -------
    a single vertex from T that best separates T
    """
    n = len(Tree.nodes)
    rx_tree: RxGraphWrapper[T] = RxGraphWrapper.from_networkx_graph(Tree) # type: ignore
    v: int = rx_tree.G.node_indices()[0] # start from arbitrary initial guess
    best_cc_size: int = n
    best_cc: Set[int] = rx_tree.G.node_indices()
    best_separator: int = v
    while True:
        CCs, v, v_neighbors = _separate_tree(rx_tree, v)
        C_max = CCs[np.argmax([len(C) for C in CCs])]
        if len(C_max) < best_cc_size:
            best_cc_size, best_cc, best_separator = len(C_max), C_max, v
        else:
            return rx_tree.G[best_separator], {rx_tree.G[u] for u in best_cc}
        # v has a single neighbor in each connected component since otherwise
        # there would be a cycle. This neighbor is a better separator
        v = next(iter(set(C_max).intersection(v_neighbors)))


def _translate_embedding(
        f: Dict[T, NDArray[np.float64]],
        t: NDArray[np.float64],
) -> Dict[T, NDArray[np.float64]]:
    for u in f:
        f[u] -= t
    return f


def _find_separator_and_separate(S: Set[T], dT: GraphMetricSpace[T]):
    v, L = tree_separator(nx.induced_subgraph(dT.G, S)) # type: ignore
    R = set(S).difference(L)
    L.add(v)
    return L, R, v


def _rec_linf_tree_embedding(
        S: Set[T], dT: GraphMetricSpace[T], dim: int, idx: int
) -> Dict[T, NDArray[np.float64]]:
    if len(S) <= 2:
        return {v: np.zeros(dim) for v in S}
    L, R, separator = _find_separator_and_separate(S, dT)
    f = dict()
    for sign, side in [(-1, L), (1, R)]:
        f_side = _rec_linf_tree_embedding(side, dT, dim, idx + 1)
        f_side = _translate_embedding(f_side, f_side[separator].copy())
        for u in side:
            f_side[u][idx] = sign * dT.d(separator, u)
        f.update(f_side)
    return f


def l_infinity_tree_embedding(
    dT: GraphMetricSpace[T],
    return_dim=False
) -> Union[Dict[T, NDArray[np.float64]], Tuple[Dict[T, NDArray[np.float64]], int]]:
    """
    Computes an isometric embedding of the tree T to l infinity with dimension
    d <= c * log(n) where c = 1/(log3 - 1)

    Parameters
    ----------
    T: a graph metric space for a graph that is a tree

    Returns
    -------
    an embedding from the tree vertices to numpy arrays

    Notes
    -----
    This is an implementation of the algorithm by [1]. It is also
    asymptotically tight, stars require an Omega(log(n)) dimension to
    be isometrically embedded in any normed space

    References
    ----------
    [1] N. Linial, E. London, and Y. Rabinovich. The geometry of 
    graphs and some of its algorithmic applications. Combinatorica,
    15(2):215=245, 1995.
    """
    if not nx.tree.is_tree(dT.G):
        raise ValueError("Given graph is not a tree")
    
    dim = int(np.ceil(np.log(len(dT)) / np.log(3 / 2)))
    f = _rec_linf_tree_embedding(dT.points, dT, dim, idx=0)
    if return_dim:
        return f, dim
    return f