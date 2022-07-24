import os
from itertools import compress, zip_longest
from multiprocessing import Pool
from typing import (Any, Dict, Generator, Generic, List, Optional, Set,
                    TypeVar, Union)

import networkx as nx
import numpy as np
import retworkx as rx
from metric_embedding.core.embedding_analysis import DictEmbedding, Embedding
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.metrics.graph_metric import (GraphMetricSpace,
                                                   nx_graph_to_rx_graph)
from tqdm import tqdm

T = TypeVar("T")

class FrechetEmbedding(Generic[T]):
    """
    Given a finite metric space (V, d_V) and
    a list of sets As of length k, compute an
    embedding f of V into R^k defined by
    f(v)[i] = d_V(v, As[i])
    """

    def __init__(
        self,
        V: FiniteMetricSpace[T],
        As: List[Set[T]],
        lazy: bool = True,
        n_workers: int = 0,
        progress_bar: bool = False
    ):
        """
        Parameters
        ----------
        V: A finite metric space
        As: A list of subsets of V to be used as coordinates
        lazy: If lazy is false, the vectors of points in V
            are precomputed when the embedding object is
            created. If true they are computed when needed
            and cached
        n_workers: should not be given if lazy is True. If lazy
            is false this is the number of parallel processes
            that precompute the vectors
        progress_bar: whether or not to display a progress bar 
            when precomputing the vectors. value is disregarded
            when lazy is True.
        """
        self.V = V
        self.As = As
        self.cache: Dict[T, np.ndarray] = dict()
        if not lazy:
            self.__fill_table(n_workers, progress_bar)
    
    def __fill_table(self, n_workers: int, progress_bar: bool):
        V = list(self.V)
        Vt = V
        if progress_bar:
            Vt = tqdm(V)
        if n_workers == 0:
            res = map(self, Vt)
        else:
            n_workers = min(os.cpu_count() or 1, n_workers)
            with Pool(n_workers) as p:
                res = p.map(self, Vt)
        for v, fv in zip(V, res):
            self.cache[v] = fv
    
    def __len__(self):
        return len(self.As)

    def __call__(self, v: T):
        if v not in self.cache:
            self.cache[v] = np.array(
                [self.V.set_distance({v}, A) for A in self.As]
            )
        return self.cache[v]
    
    def dim(self):
        return len(self)
    
    def __getitem__(self, v):
        return self.__call__(v)
    
    def as_dict_embedding(self):
        for u in self.V:
            self(u)
        return self.cache


class GraphFrechetEmbedding(FrechetEmbedding[T]):
    """
    A Frechet Embedding that acts specifically on graph
    metrics. It implements a more efficient computation
    of the vectors than the general computation for a
    general metric space.
    """
    def __init__(
        self,
        V: GraphMetricSpace[T],
        As: List[Set[T]],
        lazy: bool = True,
        n_workers: int = 0,
        progress_bar: bool = False
    ):
        """
        Parameters
        ----------
        V: A finite metric space
        As: A list of subsets of V to be used as coordinates
        lazy: If lazy is false, the vectors of vertices in V
            are precomputed when the embedding object is 
            created, otherwise they are computed when needed
            and cached
        n_workers: should not be given if lazy is True. If lazy
            is false this is the number of parallel processes
            that precompute the vectors
        progress_bar: whether or not to display a progress bar 
            when precomputing the vectors. value is disregarded
            when lazy is True.
        """
        super().__init__(V, As, lazy, n_workers, progress_bar)
        self.V = V
    
    def __embedding(self, source):
        fv = np.zeros(len(self.As))
        sets_dict = dict(enumerate(self.As))
        seen = set()
        level = 0  # the current level
        nextlevel = {source}  # set of nodes to check at next level
        adj = self.V.G.adj
        while nextlevel and sets_dict:
            thislevel = nextlevel  # advance to next level
            nextlevel = set()  # and start a new set (fringe)
            found = []
            for v in thislevel:
                if v not in seen:
                    seen.add(v)
                    found.append(v)
                    to_remove = []
                    for i, Ai in sets_dict.items():
                        if v in Ai:
                            fv[i] = level
                            to_remove.append(i)
                    for i in to_remove:
                        sets_dict.pop(i)
            for v in found:
                nextlevel.update(adj[v])
            level += 1
        del seen
        return fv
    
    def __call__(self, v: T):
        if v not in self.cache:
            self.cache[v] = self.__embedding(v)
        return self.cache[v]


def __compute_embedding_dimension(
    n: int, q: int, p: float, p_succ: Optional[float] = None
):
    c1 = (p / (1 - (np.e ** (-p)))) * (1 / ((1 - p) ** (1 / p)))
    if p_succ is not None:
        p_fail = 1 - p_succ
        c = ((np.log(1 / p_fail) / np.log(n)) + 2) * c1
        print(c1, np.log(1 / p_fail), c)
    else:
        c = 2 * c1
    return int(np.ceil(c * np.log(n) * (n ** (1 / q))))


def __random_subset(V: Set[T], p: float) -> Set[T]:
    coins = np.random.rand(len(V)) <= p
    return set(compress(V, coins))


def l_infinity_embedding(
    V: FiniteMetricSpace[T], q: int,
    success_probability: Optional[float] = None
) -> FrechetEmbedding[T]:
    """
    Given any q = 1, 2, ..., Computes a (2q - 1)-embedding of V
    to the l_inf normed space with dimension d = O(q n^(1/q) ln(n))
    where |V| = n.
    
    Parameters
    ----------
    V: A finite metric space to be embedded to l infinity
    q: Controls the dimension/distortion tradeoff. 
        Given the function outputs a (2q - 1)-embedding with 
        dimension O(n^(1/q))
    success_probability: The embedding is random and this parameter
        is the probability of the embedding to be a (2q-1)-embedding
        This may increase the dimension (the higher the success probability
        the bigger the dimension). Value must be smaller then 1
    
    Returns
    -------
    An embedding f:V -> l_{inf} with dimension d = O(n^(1/q)ln(n))

    Notes
    -----
    This function implements Matousek embedding [1].
    It is a form of a Frechet embedding, where the coordinates
    are distances to randomly chosen subsets of V with suitable
    densities. The embedding is a (2q - 1)-embedding to l_inf
    with dimension O(n^(1/q)ln(n))

    When setting q = log(n) this embedding may also be used 
    as a O(log(n)^2)-embedding with dimension O(log(n)) to l2
    As opposed to Bourgain's embedding which is an O(log(n))-
    embedding with dimension O(log(n)^2)

    References
    ----------
    [1] Matousek, J. (1996). On the distortion required for 
    embedding finite metric spaces into normed spaces.
    Israel J. Math.,93:333-344.
    """
    if q == 1:
        return FrechetEmbedding(V, [{v} for v in V.points])
    if success_probability is not None and success_probability >= 1:
        raise ValueError("success probability must be smaller then 1")
    n = len(V.points)
    q = min(q, int(np.log2(n)))
    p = n ** (-1 / q)  # q is at most log_2(n) so p <= 0.5
    m = __compute_embedding_dimension(n, q, p, success_probability)
    if m * q >= n:
        return FrechetEmbedding(V, [{v} for v in V.points])
    ps = [p ** j for j in range(1, q + 1)]
    A = [__random_subset(V.points, ps[j]) for j in range(q) for _ in range(m)]
    A = [Aij for Aij in A if len(Aij) != 0]
    return FrechetEmbedding(V, A)


def l2_embedding(V: FiniteMetricSpace[T]) -> FrechetEmbedding[T]:
    """
    Given any finite metric space V, Computes
    an O(log(n))-embedding to euclidean space
    (l_2) with dimension O(log^2(n))
    
    Parameters
    ----------
    V: A finite metric space to be embedded to l infinity
    
    Returns
    -------
    An embedding f:V -> l_2 with dimension d = O(log^2(n))

    Notes
    -----
    This function implements Bourgain's embedding [1].
    It is a form of a Frechet embedding, where the coordinates
    are distances to randomly chosen subsets of V with suitable
    densities. The embedding is an O(log(n))-embedding to l_2
    with dimension O(log(n)^2)

    References
    ----------
    [1] J. Bourgain. On Lipschitz embedding of finite metric spaces 
    in Hilbert space. Israel J. Math., 52(1-2):46-52, 1985.
    """
    n = len(V)
    q = int(np.floor(np.log2(n))) + 1
    m = int(np.floor(2 * np.log2(n)))
    A = [__random_subset(V.points, 2 ** (-j)) for j in range(q) for _ in range(m)]
    A = [Aij for Aij in A if len(Aij) != 0]
    return FrechetEmbedding(V, A)


def tree_separator(T: nx.Graph):
    """
    Finds a single vertex v that separates the tree well, i.e. every
    connected component of T - {v} is of size at most 2/3|T|
    
    Parameters
    ----------
    T: a networkx graph which is a tree

    Returns
    -------
    a single vertex from T
    """
    v = next(iter(T.nodes))
    n = len(T.nodes)
    T_copy = nx_graph_to_rx_graph(T)
    while True:
        v_neighbors = T[v]
        T_copy.remove_node(v)
        CCs: List[set] = rx.connected_components(T_copy)  # type: ignore
        C_max = CCs[np.argmax([len(C) for C in CCs])]
        if len(C_max) <= 2 * n / 3:
            C_max = {T_copy[u] for u in C_max}
            return v, C_max, set(T.nodes).difference(C_max)
        # v has a single neighbor in each connected component since otherwise
        # there would be a cycle. This neighbor is a better separator
        T_copy.add_node(v)
        T_copy.add_edges_from([(u, v, None) for u in v_neighbors])
        v = next(iter(set(v_neighbors).intersection(C_max)))


def l_infinity_tree_embedding(dT: GraphMetricSpace[T]) -> Dict[T, np.ndarray]:
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
    if not nx.is_tree(dT.G):
        raise ValueError("Given graph is not a tree")
    if len(dT) == 2:
        u, v = tuple(dT.G.nodes)
        return {u: np.array([0]), v: np.array([dT.d(u, v)])}
    v, L, R = tree_separator(dT.G)
    f = dict()
    for S in [L, R]:
        S.add(v)
        f_s = l_infinity_tree_embedding(dT.induced_subgraph_metric(S))
        fv = f_s[v]
        for u in f_s:
            fu = f[u]
            fu -= fv
            fu.resize(len(fu) + 1, refcheck=False)
            fu[-1] = dT.d(v, u)
        f.update(f_s)
    return f


Path = List


def find_isolated_path(G: nx.MultiGraph) -> Optional[Path]:
    if not all(G.degree[u] >= 2 for u in G.nodes):
        raise ValueError("Graph input for isolated path finding"
                         "must have only vertices of degree >= 2")
    deg_2_node = next((u for u in G if G.degree[u] == 2), None)
    if deg_2_node is None:
        return None
    prev = deg_2_node
    while G.degree[deg_2_node] == 2:
        tmp = deg_2_node
        deg_2_node = next(u for u in G[deg_2_node] if u != prev)
        prev = tmp
    path = [deg_2_node, prev]
    deg_2_node = prev
    while G.degree[deg_2_node] == 2:
        u = next(u for u in G[deg_2_node] if u != path[-2])
        deg_2_node = u
        path.append(u)
    return path


def __path_weight(G: Union[nx.Graph, nx.MultiGraph], P: list, weight: str):
    if isinstance(G, nx.MultiGraph):
        return sum(
            min(data[weight] for data in G[u][v].values())
            for u, v in zip(P[:-1], P[1:])
        )
    return sum(G[u][v][weight] for u, v in zip(P[:-1], P[1:]))


def _1_2_elimination(G: nx.Graph, weight: str):
    """
    remove all 1 degree vertices from the graph and replace all isolated paths
    with equally weighted edges

    Parameters
    ----------
    G: a weighted undirected graph

    Returns
    -------
    A triple of
    1. A weighted undirected graph which is the result of the 1-2 elimination 
        process
    2. A forest of all the degree 1 vertices removed
    3. A list of all isolated paths removed
    """
    G1 = nx.MultiGraph(G)
    degree_one_nodes = [u for u in G1.nodes if G1.degree[u] == 1]
    F = nx.Graph()
    while degree_one_nodes:
        u = degree_one_nodes.pop(0)
        v = next(G1.neighbors(u))
        F.add_edge(u, v, **G[u][v])
        G1.remove_node(u)
        if G1.degree[v] == 1:
            degree_one_nodes.append(v)
    
    isolated_paths: List[Path] = []
    P = find_isolated_path(G1)
    while P is not None:
        isolated_paths.append(P)
        w = __path_weight(G1, P, weight)
        for u in P[1:-1]:
            G1.remove_node(u)
        G1.add_edge(P[0], P[-1], **{weight: w})
        P = find_isolated_path(G1)
    return G1, F, isolated_paths


def multigraph_to_graph(G: nx.MultiGraph, weight: str):
    """
    Returns a graph from a multigraph, ambiguities in edge weights
    are solved by taking the minimum weight to preserve the metric structure

    Parameters
    ----------
    G: a multigraph
    weight: a string which is the key in the attr dictionary of the edges
        representing the weight 
    
    Returns
    -------
    A simple graph with the same shortest path metric as the given multigraph
    """
    G1 = nx.Graph(G)
    for u, v in G1.edges:
        G1[u][v][weight] = min(data[weight] for data in G1[u][v].values())
    return G1


def _degree_one_embedding_extension(
    G: GraphMetricSpace[T],
    f: DictEmbedding[T, np.ndarray],
    F: nx.Graph
):
    if not nx.is_forest(F):
        raise ValueError("F must be a forest")
    trees: Generator[Set[T], None, None] = nx.connected_components(F)
    root_to_tree = [(next(iter(T.intersection(G.G.nodes))), T) for T in trees]
    d_F = G.induced_subgraph_metric(F.nodes)
    
    def rec_ext(_root_to_tree):
        if len(_root_to_tree) == 1:
            return []
        
        root_to_tree_1 = _root_to_tree[:len(_root_to_tree) // 2]
        root_to_tree_2 = _root_to_tree[len(_root_to_tree) // 2:]
        f_l = {u: fu.copy() for u, fu in f.items()}
        for r, Tree in root_to_tree_1:
            flr = f_l[r]
            f_l.update({u: flr + d_F.d(r, u) for u in Tree if u != r})
        for r, Tree in root_to_tree_2:
            flr = f_l[r]
            f_l.update({u: flr - d_F.d(r, u) for u in Tree if u != r})
        fs1 = rec_ext(root_to_tree_1)
        fs2 = rec_ext(root_to_tree_2)
        fs: List[DictEmbedding[T, np.ndarray]] = [f_l]
        for d1, d2 in zip_longest(fs1, fs2):
            if d1 is None:
                d1, d2 = d2, d1
            if d2 is not None:
                d1.update(d2)
            fs.append(d1)
        return fs
    
    fs = rec_ext(root_to_tree)
    # for r, Tree in root_to_tree:
    #     f_T = l_infinity_tree_embedding(G.induced_subgraph_metric(Tree))
    #     for u in f_T - 
    # fs.append()
    return {u: np.concatenate([f_i[u] for f_i in fs]) for u in G.points}


def _degree_two_embedding_extension(
    G: GraphMetricSpace[T],
    f: DictEmbedding[T, np.ndarray],
    isolated_paths: List[Path[T]]
):
    return f


def sparse_graph_embedding(G: nx.Graph, weight: str, q: int):
    dG = GraphMetricSpace(G, weight=weight)
    G1, F, isolated_paths = _1_2_elimination(G, weight)
    G1 = multigraph_to_graph(G1, weight)
    dG1 = GraphMetricSpace(G1, weight=weight)
    f = l_infinity_embedding(dG1, q)
    f = GraphFrechetEmbedding(dG1, f.As, lazy=False)
    g = _degree_one_embedding_extension(dG, f.as_dict_embedding(), F)
    h = _degree_two_embedding_extension(dG, g, isolated_paths)
    return h
