from typing import Dict, Tuple

from metric_embedding.core.embedding_analysis import (embedding_contraction,
                                                      embedding_distortion,
                                                      embedding_expansion)
from metric_embedding.core.metric_space import FiniteMetricSpace


class TestExpansionAnalysis:
    
    N = 10

    def test_expansion(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 2 * abs(y - x))
        assert embedding_expansion(X, Y.d, lambda x: x) == 2
    
    def test_expansion_2(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        assert embedding_expansion(X, Y.d, lambda x: x) == self.N - 1
    
    def test_expansion_with_contraction(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_expansion(X, Y.d, lambda x: x) == 1.4
    
    def test_expansion_isometric_embedding(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        assert embedding_expansion(X, X.d, lambda x: x) == 1
    
    def test_expansion_no_expansion(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.5 * abs(y - x))
        assert embedding_expansion(X, Y.d, lambda x: x) == 1


class TestContractionAnalysis:

    N = 10

    def test_contraction(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.25 * abs(y - x))
        assert embedding_contraction(X, Y.d, lambda x: x) == 4
    
    def test_contraction_with_expansion(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_contraction(X, Y.d, lambda x: x) == 2
    
    def test_contraction_isometric_embedding(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        assert embedding_contraction(X, X.d, lambda x: x) == 1
    
    def test_contraction_no_contraction(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 2 * abs(y - x))
        assert embedding_contraction(X, Y.d, lambda x: x) == 1


class TestDistortionAnalysis:

    N = 10

    def test_distortion_isometric_embedding(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(1, self.N + 1)), lambda x, y: abs(y - x))
        assert embedding_distortion(X, Y.d, lambda x: x + 1) == 1
    
    def test_distortion_contraction_only(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.25 * abs(y - x))
        assert embedding_distortion(X, Y.d, lambda x: x) == 4
    
    def test_distortion_expansion_only(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 4 * abs(y - x))
        assert embedding_distortion(X, Y.d, lambda x: x) == 4
    
    def test_distortion_with_contraction_and_expansion(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_distortion(X, Y.d, lambda x: x) == 2 * 1.4
    

