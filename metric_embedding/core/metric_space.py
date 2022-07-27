from itertools import combinations, product
from typing import Callable, Generic, Iterable, Set, TypeVar

T = TypeVar("T")
Metric = Callable[[T, T], float]


class MetricSpace(Generic[T]):
    def __init__(self, d: Metric[T]):
        self.d = d
    
    @classmethod
    def is_metric(cls, d_S: Metric[T], S: Set[T]):
        """
        validates if the function d_S is in fact a metric on S, i.e. satisfies
        1. d_S(x, y) >= 0 and d_S(x, y) == 0 iff x == y for all x, y in S
        2. d_S(x, y) <= d_S(x, z) + d_S(z, y) for all x, y, z in S
        """
        return all(
            (d_S(x, y) >= 0 and (x != y or d_S(x, y) == 0)) and
            all(d_S(x, y) <= d_S(x, z) + d_S(z, y) for z in S)
            for x, y in combinations(S, 2)
        )


class FiniteMetricSpace(MetricSpace[T]):
    def __init__(self, points: Set[T], d: Metric[T]):
        self.points: Set[T] = points
        super().__init__(d)
    
    def __iter__(self):
        return iter(self.points)
    
    def __len__(self):
        return len(self.points)
    
    def pairs(self):
        return combinations(self.points, 2)
    
    def set_distance(self, A: Iterable[T], B: Iterable[T]):
        return min(self.d(u, v) for u, v in product(A, B))
