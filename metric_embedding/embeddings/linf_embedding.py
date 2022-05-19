from functools import cache
from itertools import compress
from typing import Generic, List, Optional, Set, TypeVar

import numpy as np
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.metrics.graph_metric import GraphMetricSpace

T = TypeVar("T")


class FrechetEmbedding(Generic[T]):
    """
    Given a finite metric space (V, d_V) and a list of sets As of length k,
    compute an embedding f of V into R^k defined by f(v)[i] = d_V(v, As[i])
    """

    def __init__(self, V: FiniteMetricSpace[T], As: List[Set[T]]):
        self.V = V
        self.As = As
        self.cache = dict()

    def __call__(self, v: T):
        if v not in self.cache:
            self.cache[v] = np.array([self.V.set_distance({v}, A) for A in self.As])
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
    V: FiniteMetricSpace, q: int,
    success_probability: Optional[float] = None
):
    """
    Given any q = 1, 2, ..., Computes a (2q - 1)-embedding of V
    to the l infinity normed space with dimension d = O(q n^(1/q) ln(n))
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
