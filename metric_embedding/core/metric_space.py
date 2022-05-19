from itertools import combinations, product
from typing import Callable, Generic, Iterable, Set, TypeVar

T = TypeVar("T")
Metric = Callable[[T, T], float]


class MetricSpace(Generic[T]):
    def __init__(self, d: Metric[T]):
        self.d = d


class FiniteMetricSpace(MetricSpace[T]):
    def __init__(self, points: Set[T], d: Metric[T]):
        self.points: Set[T] = points
        super().__init__(d)
    
    def __iter__(self):
        return iter(self.points)
    
    def pairs(self):
        return combinations(self.points, 2)
    
    def set_distance(self, A: Iterable[T], B: Iterable[T]):
        return min(self.d(u, v) for u, v in product(A, B))
