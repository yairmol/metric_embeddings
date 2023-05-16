from itertools import compress
from typing import Optional, Set, TypeVar

import networkx as nx
import numpy as np

from metric_embedding.core.embedding_analysis import embedding_distortion
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.embeddings.frechet_embedding import (
    FrechetEmbedding, GraphFrechetEmbedding)
from metric_embedding.embeddings.sparse_graphs import (
    _1_2_elimination, deg_2_embedding_extension, deg_one_embedding_extension,
    extend_metric, multigraph_to_graph)
from metric_embedding.metrics.graph_metric import GraphMetricSpace
from metric_embedding.metrics.lp_metrics import LpMetric

T = TypeVar("T")

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
    G1_ext, eps_to_gen, gen_paths = extend_metric(G, G1, weight, isolated_paths, q, eps)
    dG1_ext = GraphMetricSpace(G1_ext, weight=weight)
    f = l_infinity_embedding(dG1_ext, q)
    f = GraphFrechetEmbedding(dG1_ext, f.As, lazy=False)
    print("distortion1", embedding_distortion(dG1_ext, LpMetric(float("inf")).d, f))
    g = deg_2_embedding_extension(G, G1_ext, f, isolated_paths, eps_to_gen, gen_paths, eps, weight)
    return deg_one_embedding_extension(dG, g, F)


