from functools import partial
from typing import Any, Callable

import numpy as np
from metric_embedding.core.metric_space import MetricSpace


class NormedMetricSpace(MetricSpace):
    def __init__(self, norm: Callable[[Any], float]):
        super().__init__(self.norm_metric)
        self.norm = norm
    
    def norm_metric(self, x, y):
        return self.norm(x - y)


class LpMetric(NormedMetricSpace):
    """
    A class representing an lp metric space. Given two vectors
    x, y in R^k, the distance between x and y according to the
    lp norm is ||x - y||_p = sum(abs(x[i] - y[i]) ** p for i in range(k)) ** 1/p.
    This forms a metric for all p >= 1. We can also consider the
    case of p = inf in which ||x - y||_inf = max(abs(x[i] - y[i]) for i in range(k))
    """
    
    def __init__(self, p: float):
        """
        Parameters
        ----------
        p: a float between zero and infinity (including infinity)

        Examples
        --------
        For p = 2 we get the euclidean norm, and for p = float("inf") we
        get the maximum norm
        """
        super().__init__(partial(np.linalg.norm, ord=p))  # type: ignore
        self.p = p
