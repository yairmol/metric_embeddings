import os
from functools import cache
from itertools import compress
from multiprocessing import Pool
from typing import Generic, List, Optional, Set, TypeVar

import numpy as np
from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.metrics.graph_metric import GraphMetricSpace
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
        self.cache = dict()
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
            self.cache[v] = np.array([self.V.set_distance({v}, A) for A in self.As])
        return self.cache[v]
    
    def dim(self):
        return len(self)


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
    V: FiniteMetricSpace, q: int,
    success_probability: Optional[float] = None
):
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


def l2_embedding(V: FiniteMetricSpace):
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
    densities. The embedding is an O(log(n)-embedding to l_2
    with dimension O(log^2(n))

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

