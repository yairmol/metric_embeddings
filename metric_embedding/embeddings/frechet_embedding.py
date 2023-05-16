import os
from multiprocessing import Pool
from typing import Dict, Generic, List, Set, TypeVar

import numpy as np
from tqdm import tqdm

from metric_embedding.core.metric_space import FiniteMetricSpace
from metric_embedding.metrics.graph_metric import GraphMetricSpace

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