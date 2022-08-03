import os
from collections import defaultdict
from functools import cache
from itertools import compress, count, zip_longest
from multiprocessing import Pool
from typing import (Dict, FrozenSet, Generator, Generic, List, Optional, Set,
                    Tuple, TypeVar, Union)

import networkx as nx
import numpy as np
import retworkx as rx
from metric_embedding.core.embedding_analysis import DictEmbedding
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.metrics.graph_metric import GraphMetricSpace
from metric_embedding.utils.rx_utils import RxGraphWrapper
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
    a single vertex from T that best separates T
    """
    n = len(T.nodes)
    Tr = RxGraphWrapper.from_networkx_graph(T).G
    v = Tr.node_indices()[0]
    best_size, best_c, best_v = n, Tr.node_indices(), Tr[v]
    while True:
        v_neighbors = Tr.adj(v)
        v_node = Tr[v]
        Tr.remove_node(v)
        CCs: List[set] = rx.connected_components(Tr)  # type: ignore
        C_max = CCs[np.argmax([len(C) for C in CCs])]
        v = Tr.add_node(v_node)
        if len(C_max) < best_size:
            best_size, best_c, best_v = len(C_max), C_max, v_node
        else:
            return best_v, {Tr[u] for u in best_c}
        Tr.add_edges_from([(u, v, None) for u in v_neighbors])
        # v has a single neighbor in each connected component since otherwise
        # there would be a cycle. This neighbor is a better separator
        v = next(iter(set(v_neighbors).intersection(C_max)))


def l_infinity_tree_embedding(
    dT: GraphMetricSpace[T],
    return_dim=False
) -> Union[Dict[T, np.ndarray], Tuple[Dict[T, np.ndarray], int]]:
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
    
    def run(S):
        if len(S) <= 2:
            return {v: np.array([]) for v in S}, 0
        v, L = tree_separator(nx.induced_subgraph(dT.G, S))
        R = set(S).difference(L)
        L.add(v)
        f_l, dim_l = run(L)
        f_r, dim_r = run(R)
        dim = max(dim_l, dim_r) + 1
        f = dict()
        for f_s, sign in [(f_l, -1), (f_r, 1)]:
            for u in f_s:
                f_s[u].resize(dim, refcheck=False)
            fv = f_s[v].copy()
            for u in f_s:
                fu = f_s[u]
                fu -= fv
                fu[-1] = sign * dT.d(v, u)
            f.update(f_s)
        return f, dim
    
    f, dim = run(dT)
    if return_dim:
        return f, dim
    return f


Path = List


def find_isolated_path(G: nx.Graph) -> Optional[Path]:
    """
    If there exists an isolated path in G, returns an isolated path
    otherwise returns None

    Parameters
    ----------
    G: a networkx graph (or multigraph)

    Returns
    -------
    A path as a list of vertices if there is an isolated path in G,
    otherwise returns None

    Notes
    -----
    An isolated path is a path with at least 3 vertices such that all inner 
    vertices are of degree 2
    """
    deg_2_node = next((u for u in G if G.degree[u] == 2), None)
    if len(G[deg_2_node]) <= 1: # this deg_2_node has self loop
        return None
    if deg_2_node is None:
        return None
    
    first = deg_2_node
    prev = first
    while G.degree[deg_2_node] == 2:
        prev, deg_2_node = deg_2_node, next(u for u in G[deg_2_node] if u != prev)
        if deg_2_node == first:
            break

    path = [deg_2_node, prev]
    deg_2_node = prev
    while G.degree[deg_2_node] == 2 and deg_2_node != path[0]:
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
    G1 = nx.Graph()
    G1.add_nodes_from(G.nodes)
    G1.add_edges_from({(u, v) for u, v, i in G.edges if u != v})
    for u, v in G1.edges:
        G1[u][v][weight] = min(data[weight] for data in G[u][v].values())
    return G1


def deg_one_embedding_extension(
    G: GraphMetricSpace[T],
    f: DictEmbedding[T, np.ndarray],
    F: nx.Graph
):
    if len(F.nodes) != 0 and not nx.is_forest(F):
        raise ValueError("F must be a forest")
    trees: Generator[Set[T], None, None] = nx.connected_components(F)
    root_to_tree = {next(iter(T.intersection(G.G.nodes))): T for T in trees}
    roots = list(root_to_tree.keys())
    if len(roots) == 0:
        return f
    d_F = G.induced_subgraph_metric(F.nodes)

    @cache
    def embed_tree(r, sign):
        Tree = root_to_tree[r]
        f_r = f[r]
        return {u: f_r + sign * d_F.d(r, u) for u in Tree}
    
    def rec_ext(_roots):
        """
        returns a list of embeddings of all vertices,
        to later be concatenated
        """
        if len(_roots) == 1:
            return []
        
        roots_l, roots_r = _roots[:len(_roots) // 2],_roots[len(_roots) // 2:]
        fi_l, fi_r = f.copy(), f.copy()
        for roots_s, sign in [(roots_l, -1), (roots_r, +1)]:
            for r in roots_s:
                fi_l.update(embed_tree(r, sign))
                fi_r.update(embed_tree(r, -sign))

        fs_l, fs_r = rec_ext(roots_l), rec_ext(roots_r)
        fs: List[DictEmbedding[T, np.ndarray]] = [fi_l, fi_r]
        for d1, d2 in zip_longest(fs_l, fs_r):
            if d1 is None:
                d1 = fi_l.copy()
            if d2 is None:
                d2 = fi_r
            d1.update(d2)
            fs.append(d1)
        return fs
    
    max_dim = 0
    g: Dict[T, np.ndarray] = dict()
    for r in roots:
        g_r, dim = l_infinity_tree_embedding(
            d_F.induced_subgraph_metric(root_to_tree[r]), return_dim=True
        )
        g.update(g_r)
        max_dim = max(max_dim, dim)
    for u in g:
        g[u].resize(max_dim, refcheck=False)
    for v in f:
        if v in g:
            continue
        g[v] = np.zeros(max_dim)
        
    fs = rec_ext(roots)
    fs.append(g)
    return {u: np.concatenate([f_i[u] for f_i in fs]) for u in G.points}


def generate_path(u, v, w: float, v_gen: count, eps: float, weight: str):

    def run(w: float):
        if w <= 1:
            v = next(v_gen)
            return nx.Graph(), v, v
        num_vs = int(np.ceil(1 / eps))
        vs = [next(v_gen) for _ in range(num_vs)]
        P, s, t = run(w / 2)
        P.add_edges_from(zip(vs[:-1], vs[1:]), **{weight: eps * w / 2})
        missing_weight = w / 2 - (eps * w / 2) * (num_vs - 1)
        P.add_edge(vs[-1], s, **{weight: missing_weight})
        return P, vs[0], t
    
    P1, s1, t1 = run(w / 2)
    P2, s2, t2 = run(w / 2)
    for n in P2[s2]:
        P1.add_edge(s1, n, **P2[s2][n])
    P2.remove_node(s2)
    missing_weight_1 = w / 2  - sum(P1[u][v][weight] for u, v in P1.edges)
    missing_weight_2 = w / 2  - sum(P2[u][v][weight] for u, v in P2.edges)
    P1.add_edge(u, t1, **{weight: missing_weight_1})
    P2.add_edge(t2, v, **{weight: missing_weight_2})
    P1.add_edges_from(P2.edges(data=True))
    return P1


_EndpointsToPaths = Dict[FrozenSet[T], Dict[int, List[Path[T]]]]


def extend_metric(
    G: nx.Graph,
    G1: nx.Graph,
    weight: str,
    isolated_paths: List[Path[int]],
    q: int,
    eps: float = 0.5
):
    """
    Extends the graph G1 with epsilon spaces virtual vertices
    in place of the isolated paths

    Parameters
    ----------
    G: The original graph
    G1: The graph G after 1_2_elimination
    weight: key for the weight attribute of edges
    isolated_paths: a list of isolated paths obtained from 1_2_elimination
    q: The first embedding distortion parameter
    eps: The second embedding distortion parameter

    Returns
    -------
    The extension of the graph G1

    Notes
    -----
    If n = |V(G1)| and W = max{length(P)} where the maximum is taken over
    all isolated paths P, then the output graph will have at most
    O((n * log(W) / eps) ^ 2) vertices
    """
    # eps = eps / (4 * q)
    endpoints_to_paths: _EndpointsToPaths[int] = defaultdict(lambda: defaultdict(list))
    for P in isolated_paths:
        uv = frozenset({P[0], P[-1]})
        w = __path_weight(G, P, weight)
        scale = int(np.log(w) / np.log(1 + eps))
        endpoints_to_paths[uv][scale].append(P)
    
    vertex_gen = count(max(G.nodes) + 1, step=1)
    G1 = G1.copy()  # type: ignore
    
    generated_nodes_to_scale_endpoints = dict()
    for uv, scale_to_paths in endpoints_to_paths.items():
        if len(uv) == 1:
            u, = tuple(uv)
            v = u
        else:
            u, v = tuple(uv)
        for scale in scale_to_paths:
            w = (1 + eps) ** scale
            P = generate_path(u, v, w, vertex_gen, eps, weight)
            for n in P.nodes:
                if n in {u, v}:
                    continue
                generated_nodes_to_scale_endpoints[n] = (u, v, scale)
            G1.add_edges_from(P.edges(data=True))
    
    return G1, generated_nodes_to_scale_endpoints


def deg_2_embedding_extension(
    G1: nx.Graph,
    f: GraphFrechetEmbedding[T],
    Ps: List[Path[T]],
    endpoints_to_paths: _EndpointsToPaths[T],
    eps: float, weight: str
    # node_to_ep_scale: Dict[int, Tuple[T, T, int]],
):
    """
    Takes a frechet embedding of the graph with virtual nodes
    and returns the embedding of the isolated paths vertices
    with respect to the sets sampled in the frechet embedding
    """
    # paths_dists = dict()
    # for P in Ps:
    #     total = 0
    #     for i in range(1, len(P) - 1):
    #         total += G1[P[i - 1]][P[i]][weight]
    #         paths_dists[P[i]] = total
    
    # f_dict = f.as_dict_embedding()
    # for P in Ps:
    #     u, v = P[0], P[-1]
    #     scale = int(np.log(w) / np.log(1 + eps))
    #     for n in P[1:-1]:
            
    #     for A in f.As:
            
    #         for n in P[1:-1]:
    return f.as_dict_embedding()


def l_infinity_sparse_graph_embedding(G: nx.Graph, weight: str, q: int, eps=0.5):
    """
    Calculates a (1 + eps)(2q - 1)-embedding of G to l_{inf} with dimension
    d = O((x * log(x) / eps)^(2/q)) where x = chi(G) = |E| - (|V| - 1)

    Parameters
    ----------
    G: a networkx graph. G should be sparse meaning that chi(G) << |V|
    weight: a string that is the weight attribute key
    q: embedding distortion parameter (the distortion will be <= (1+eps)(2q-1))
    eps: embedding distortion parameter (the distortion will be <= (1+eps)(2q-1))

    Returns
    -------
    an embedding in the form of a dictionary, mapping from vertices to numpy
    arrays
    """
    dG = GraphMetricSpace(G, weight=weight)
    G1, F, isolated_paths = _1_2_elimination(G, weight)
    G1 = multigraph_to_graph(G1, weight)
    G1_ext, gen_to_ep_scale = extend_metric(G, G1, weight, isolated_paths, q, eps)
    dG1_ext = GraphMetricSpace(G1_ext, weight=weight)
    f = l_infinity_embedding(dG1_ext, q)
    f = GraphFrechetEmbedding(dG1_ext, f.As, lazy=False)
    g = deg_2_embedding_extension(G1, f, isolated_paths, gen_to_ep_scale, eps, weight)
    return deg_one_embedding_extension(dG, g, F)


# def _degree_two_embedding_extension(
#     dG: GraphMetricSpace(G),
#     weight: str,
#     f: DictEmbedding[T, np.ndarray],
#     isolated_paths: List[Path[T]]
# ):
#     paths_dists = dict()
#     for P in isolated_paths:
#         total = 0
#         for i in range(1, len(P) - 1):
#             total += G[P[i - 1]][P[i]]
#             paths_dists[P[i]] = total
    
#     dim_f = len(f[next(iter(f.keys()))])

#     def run(Ps: List[Path]):
#         f_i = f.copy()
#         for P in Ps:
#             for i in range(1, len(P) - 1):
#                 f_i[P[i]] = np.zeros(dim_f)

#         for l in range(dim_f):
#             Ps.sort(key=lambda P: min(f[P[0]][l], f[P[-1]][l]))
#             Ps_l, Ps_r = Ps[:len(Ps) // 2], Ps[len(Ps) // 2:]
#             for P in Ps_l:
#                 w = __path_weight(G, P, weight)
#                 d = 
#                 for u in P[1:-1]:
