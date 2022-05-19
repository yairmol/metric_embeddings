from itertools import chain, combinations, product
from typing import Callable, Iterable, Optional, TypeVar

from metric_embedding.core.metric_space import FiniteMetricSpace, Metric

T1 = TypeVar("T1")
T2 = TypeVar("T2")
Embedding = Callable[[T1], T2]


def expansion(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    u: T1, v: T1
):
    return d_Y(f(u), f(v)) / d_X(u, v)


def expansion_curried(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
):
    return lambda u, v: d_Y(f(u), f(v)) / d_X(u, v)


def contraction(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    u: T1, v: T1
):
    return d_X(u, v) / d_Y(f(u), f(v))


def contraction_curried(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
):
    return lambda u, v: d_X(u, v) / d_Y(f(u), f(v))


def __preprocess(X: FiniteMetricSpace[T1], S: Optional[Iterable[T1]], T: Optional[Iterable[T1]]):
    if T is not None and S is None:
        raise ValueError("S must be given when T is")
    if S is None:
        S = X.points
    if T is None:
        T = X.points
    S = set(S)
    I = S.intersection(T)
    S.difference_update(I)
    return S, T, I

def embedding_contraction(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,

) -> float:
    """
    Calculate the contraction of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T

    Returns
    -------
    The contraction of the embedding, defined as the maximum over all u,v 
    of d_X(u, v) / d_Y(f(u), f(v))
    """
    S, T, I = __preprocess(X, S, T)
    c = contraction_curried(X.d, d_Y, f)
    ctag = lambda t: c(*t)
    return max(map(ctag, chain(product(S, T), combinations(I, 2))))


def embedding_expansion(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,

) -> float:
    """
    Calculate the expansion of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T

    Returns
    -------
    The expansion of the embedding, defined as the maximum over all u,v 
    of  d_Y(f(u), f(v)) / d_X(u, v)
    """
    S, T, I = __preprocess(X, S, T)
    e = expansion_curried(X.d, d_Y, f)
    etag = lambda t: e(*t)
    return max(map(etag, chain(product(S, T), combinations(I, 2))))


def embedding_distortion(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
) -> float:
    """
    Calculate the distortion of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T

    Returns
    -------
    The distortion of the embedding, defined as the expansion * distortion
    """
    return (
        embedding_contraction(X, d_Y, f, S, T) 
        * embedding_expansion(X, d_Y, f, S, T)
    )


def expanded_pairs(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    threshold: float = 1
):
    """
    return the set of all pairs whose distance was expended 
    by at least threshold

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T
    threshold: A number which is at least 1. pairs who are expanded by a factor
        larger then threshold are returned
    
    Returns
    -------
    A set of pairs that were expanded by a factor larger then threshold
    """
    S, T, I = __preprocess(X, S, T)
    e = expansion_curried(X.d, d_Y, f)
    return filter(
        lambda t: e(*t) > threshold, 
        chain(product(S, T), combinations(I, 2))
    )


def contracted_pairs(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    threshold: float = 1
):
    """
    return the set of all pairs whose distance was contracted
    by at least threshold

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T
    threshold: A number which is at least 1. pairs who are contracted by a factor
        larger then threshold are returned
    
    Returns
    -------
    A set of pairs that were contracted by a factor larger then threshold
    """
    S, T, I = __preprocess(X, S, T)
    e = expansion_curried(X.d, d_Y, f)
    return filter(
        lambda t: e(*t) > threshold, 
        chain(product(S, T), combinations(I, 2))
    )


def distorted_pairs(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    threshold: float = 1
):
    """
    return the set of all pairs whose distance was distorted
    by at least threshold

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs intersecting S
    T: A subset of X. If given the function calculates only with pairs in S x T
    threshold: A number which is at least 1. pairs who are expanded or contracted
        by a factor larger then threshold are returned
    
    Returns
    -------
    A set of pairs that were expanded or contracted by a factor larger then threshold
    """
    return chain(
        expanded_pairs(X, d_Y, f, S, T, threshold),
        contracted_pairs(X, d_Y, f, S, T, threshold)
    )
