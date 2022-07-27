from itertools import combinations, product
from typing import Dict, Tuple

from metric_embedding.core.embedding_analysis import (contracted_pairs,
                                                      embedding_contraction,
                                                      embedding_distortion,
                                                      embedding_expansion,
                                                      expanded_pairs)
from metric_embedding.core.metric_space import FiniteMetricSpace, MetricSpace


class TestExpansionAnalysis:
    
    N = 6

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
        dy.update({(u, u): 0 for u in S})
        assert MetricSpace.is_metric(lambda x, y: dy[(x, y)], S)
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_expansion(X, Y.d, lambda x: x) == 1.4
    
    def test_expansion_isometric_embedding(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        assert embedding_expansion(X, X.d, lambda x: x) == 1
        assert embedding_expansion(X, X.d, lambda x: x, set(range(self.N // 2))) == 1
        assert embedding_expansion(
            X, X.d, lambda x: x,
            S=set(range(self.N // 2)),
            T=set(range(self.N // 2))
        ) == 1
    
    def test_expansion_no_expansion(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.5 * abs(y - x))
        assert embedding_expansion(X, Y.d, lambda x: x) == 1
    
    def test_expansion_with_subsets(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(
            set(range(self.N)), 
            lambda x, y: 2 * abs(y - x) if max(x, y) >= self.N // 2 
            else 4 * abs(y - x)
        )

        assert embedding_expansion(
            X, Y.d, lambda x: x, S=set(range(self.N // 2))
        ) == 4

        assert embedding_expansion(
            X, Y.d, lambda x: x, S=set(range(self.N // 2, self.N))
        ) == 2

        assert embedding_expansion(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N // 2, self.N))
        ) == 2
    
    def test_expansion_with_contraction_and_subsets(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_expansion(X, Y.d, lambda x: x, S={1, 2}) == 1
        assert embedding_expansion(X, Y.d, lambda x: x, S={1, 3}) == 1
        assert embedding_expansion(X, Y.d, lambda x: x, S={2, 3}) == 1.4
        assert embedding_expansion(X, Y.d, lambda x: x, S={1, 2}, T={1, 3}) == 1.4
    
    def test_expanded_pairs(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 2 * abs(y - x))
        assert set(expanded_pairs(X, Y.d, lambda x: x)) == set(X.pairs())
        assert set(expanded_pairs(X, Y.d, lambda x: x, threshold=3)) == set()
    
    def test_expanded_pairs_with_subsets(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 2 * abs(y - x))
        
        assert set(expanded_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2))
        )) == set(combinations(range(self.N // 2), 2))

        assert set(expanded_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)), threshold=3
        )) == set()

        assert set(expanded_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N // 2, self.N))
        )) == set(product(range(self.N // 2), range(self.N // 2, self.N)))

        assert set(expanded_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N))
        )) == set(filter(
            lambda x: x[0] < x[1], product(range(self.N // 2), range(self.N))
        ))
    
    def test_expanded_pairs_with_contraction(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert set(expanded_pairs(X, Y.d, lambda x: x)) == {(2, 3)}
    
    def test_expanded_pairs_with_contraction_and_subsets(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert set(expanded_pairs(X, Y.d, lambda x: x, S={1, 2})) == set()
        assert set(expanded_pairs(X, Y.d, lambda x: x, S={2, 3})) == {(2, 3)}
        assert set(expanded_pairs(X, Y.d, lambda x: x, S={1, 2}, T=S)) == {(2, 3)}
        assert set(expanded_pairs(X, Y.d, lambda x: x, S={1, 2}, T={2, 3})) == {(2, 3)}
        assert set(expanded_pairs(X, Y.d, lambda x: x, S={1}, T={2, 3})) == set()


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
        assert embedding_contraction(
            X, X.d, lambda x: x, set(range(self.N // 2))
        ) == 1
        assert embedding_contraction(
            X, X.d, lambda x: x,
            S=set(range(self.N // 2)),
            T=set(range(self.N // 2, self.N))
        ) == 1
    
    def test_contraction_no_contraction(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 2 * abs(y - x))
        assert embedding_contraction(X, Y.d, lambda x: x) == 1
    
    def test_contraction_with_subsets(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(
            set(range(self.N)), 
            lambda x, y: 0.5 * abs(y - x) if max(x, y) >= self.N // 2
            else 0.25 * abs(y - x)
        )
        assert embedding_contraction(
            X, Y.d, lambda x: x, S=set(range(self.N // 2))
        ) == 4
        assert embedding_contraction(
            X, Y.d, lambda x: x, S=set(range(self.N // 2, self.N))
        ) == 2
        assert embedding_contraction(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N // 2, self.N))
        ) == 2
    
    def test_contraction_with_expansion_and_subsets(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_contraction(X, Y.d, lambda x: x, S={2, 3}) == 1
        assert embedding_contraction(X, Y.d, lambda x: x, S={1, 3}) == 1 / 0.9
        assert embedding_contraction(X, Y.d, lambda x: x, S={2, 3}, T={1}) == 2
    
    def test_contracted_pairs(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.5 * abs(y - x))
        assert set(contracted_pairs(X, Y.d, lambda x: x)) == set(X.pairs())
        assert set(contracted_pairs(X, Y.d, lambda x: x, threshold=3)) == set()
    
    def test_contracted_pairs_with_subsets(self):
        X = FiniteMetricSpace(set(range(self.N)), lambda x, y: abs(y - x))
        Y = FiniteMetricSpace(set(range(self.N)), lambda x, y: 0.5 * abs(y - x))
        
        assert set(contracted_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2))
        )) == set(combinations(range(self.N // 2), 2))

        assert set(contracted_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)), threshold=3
        )) == set()

        assert set(contracted_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N // 2, self.N))
        )) == set(product(range(self.N // 2), range(self.N // 2, self.N)))

        assert set(contracted_pairs(
            X, Y.d, lambda x: x, S=set(range(self.N // 2)),
            T=set(range(self.N))
        )) == set(filter(
            lambda x: x[0] < x[1], product(range(self.N // 2), range(self.N))
        ))
    
    def test_contracted_pairs_with_expansion(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert set(contracted_pairs(X, Y.d, lambda x: x)) == {(1, 2), (1, 3)}
    
    def test_contracted_pairs_with_expansion_and_subsets(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert set(contracted_pairs(X, Y.d, lambda x: x, S={1, 2})) == {(1, 2)}
        assert set(contracted_pairs(X, Y.d, lambda x: x, S={2, 3})) == set()
        assert set(contracted_pairs(X, Y.d, lambda x: x, S={1, 2}, T=S)) == {(1, 2), (1, 3)}
        assert set(contracted_pairs(X, Y.d, lambda x: x, S={1, 2}, T={2, 3})) == {(1, 2), (1, 3)}
        assert set(contracted_pairs(X, Y.d, lambda x: x, S={1, 2}, T={1, 3})) == {(1, 3), (2, 1)}


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
    
    def test_distortion_with_subsets(self):
        S = {1, 2, 3}
        dy: Dict[Tuple[int, int], float] = {(1, 2): 0.5 , (1, 3): 0.9, (2, 3): 1.4}
        dy.update({(v, u): d for (u, v), d in dy.items()})
        X = FiniteMetricSpace(S, lambda x, y: 0 if x == y else 1)
        Y = FiniteMetricSpace(S, lambda x, y: dy[(x, y)])
        assert embedding_distortion(X, Y.d, lambda x: x, S={1, 2}) == 2
        assert embedding_distortion(X, Y.d, lambda x: x, S={1}, T={2, 3}) == 2
        assert embedding_distortion(X, Y.d, lambda x: x, S={2, 3}) == 1.4
    

