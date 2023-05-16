from collections import defaultdict
from functools import cache
from itertools import count, zip_longest
from typing import (Dict, FrozenSet, Generator, List, Optional, Set, TypeVar,
                    Union)

import networkx as nx
import numpy as np

from metric_embedding.core.embedding_analysis import DictEmbedding
from metric_embedding.embeddings.frechet_embedding import GraphFrechetEmbedding
from metric_embedding.embeddings.trees import l_infinity_tree_embedding
from metric_embedding.metrics.graph_metric import GraphMetricSpace

T = TypeVar("T")
Path = List[T]

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
    deg_2_node = next((u for u in G if G.degree[u] == 2), None) # type: ignore
    if deg_2_node is None:
        return None
    if len(G[deg_2_node]) <= 1: # this deg_2_node has self loop
        return None
    
    first = deg_2_node
    prev = first
    while G.degree[deg_2_node] == 2: # type: ignore
        prev, deg_2_node = deg_2_node, next(u for u in G[deg_2_node] if u != prev)
        if deg_2_node == first:
            break

    path = [deg_2_node, prev]
    deg_2_node = prev
    while G.degree[deg_2_node] == 2 and deg_2_node != path[0]: # type: ignore
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
    degree_one_nodes = [u for u in G1.nodes if G1.degree[u] == 1] # type: ignore
    F = nx.Graph()
    while degree_one_nodes:
        u = degree_one_nodes.pop(0)
        v = next(G1.neighbors(u))
        F.add_edge(u, v, **G[u][v])
        G1.remove_node(u)
        if G1.degree[v] == 1: # type: ignore
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
    if len(F.nodes) != 0 and not nx.is_forest(F): # type: ignore
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
    missing_weight_1 = max(0, w / 2  - sum(P1[u][v][weight] for u, v in P1.edges))
    missing_weight_2 = max(0, w / 2  - sum(P2[u][v][weight] for u, v in P2.edges))
    P1.add_edge(u, t1, **{weight: missing_weight_1})
    P2.add_edge(t2, v, **{weight: missing_weight_2})
    P1.add_edges_from(P2.edges(data=True))
    return P1


def path_graph_to_list(P: nx.Graph, u):
    path = [u]
    first = u
    prev = first
    while P.degree[u] == 2: # type: ignore
        prev, u = u, next(v for v in P[u] if v != prev)
        path.append(u)
        if u == first:
            break
    return path


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
    
    eps_to_gen: Dict[FrozenSet[int], Dict[int, Set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    generated_paths = list()
    for uv, scale_to_paths in endpoints_to_paths.items():
        if len(uv) == 1:
            u, = tuple(uv)
            v = u
        else:
            u, v = tuple(uv)
        for scale in scale_to_paths:
            w = (1 + eps) ** scale
            P = generate_path(u, v, w, vertex_gen, eps, weight)
            generated_paths.append(path_graph_to_list(P, u))
            eps_to_gen[uv][scale].update({n for n in P.nodes if n not in uv})
            for n in P.nodes:
                if n in {u, v}:
                    continue
            G1.add_edges_from(P.edges(data=True))
    
    return G1, eps_to_gen, generated_paths


def deg_2_embedding_extension(
    G: nx.Graph,
    G1_ext: nx.Graph,
    f: GraphFrechetEmbedding[T],
    Ps: List[Path[T]],
    eps_to_gen: Dict[FrozenSet[T], Dict[int, Set[int]]],
    gen_paths: List[Path[int]],
    eps: float, weight: str
):
    """
    Takes a frechet embedding of the graph with virtual nodes
    and returns the embedding of the isolated paths vertices
    with respect to the sets sampled in the frechet embedding
    """
    paths_dists = defaultdict(dict)
    for P in Ps:
        total = 0
        w = __path_weight(G, P, weight)
        for i in range(1, len(P) - 1):
            total += G[P[i - 1]][P[i]][weight]
            paths_dists[P[0]][P[i]] = total
            paths_dists[P[-1]][P[i]] = w - total
    
    for P in gen_paths:
        total = 0
        w = __path_weight(G1_ext, P, weight)
        for i in range(1, len(P) - 1):
            total += G1_ext[P[i - 1]][P[i]][weight]
            paths_dists[P[0]][P[i]] = total
            paths_dists[P[-1]][P[i]] = w - total
    
    f_dict = f.as_dict_embedding()
    for P in Ps:
        u, v = P[0], P[-1]
        uv = frozenset({u, v})
        w = __path_weight(G, P, weight)
        scale = int(np.log(w) / np.log(1 + eps))
        for n in P[1:-1]:
            f_dict[n] = np.zeros(len(f))
        for i, A in enumerate(f.As):
            inter = A.intersection(eps_to_gen[uv][scale])
            if not inter:
                for n in P[1:-1]:
                    f_dict[n][i] = min(f_dict[u][i] + paths_dists[u][n], f_dict[v][i] + paths_dists[v][n])
            else:
                for n in P[1: -1]:
                    f_dict[n][i] = min(
                        min(abs(paths_dists[u][n] - paths_dists[u][n2]) for n2 in inter),
                        min(f_dict[u][i] + paths_dists[u][n], f_dict[v][i] + paths_dists[v][n])
                    )
                    
    return f_dict