import os
from functools import partial
from itertools import chain, combinations, product
from multiprocessing import Pool
from typing import Callable, Iterable, Optional, Set, Tuple, TypeVar

from metric_embedding.core.metric_space import FiniteMetricSpace, Metric
from tqdm import tqdm

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


def expansion1(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    uv: Tuple[T1, T1]
):
    u, v = uv
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


def contraction1(
    d_X: Metric[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    uv: Tuple[T1, T1]
):
    u, v = uv
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
        T = S
    S, T = set(S), set(T)
    I = S.intersection(T)
    S.difference_update(I)
    return S, T, I


def __pairs(S: Set[T1], T: Set[T1], I: Set[T1]):
    return chain(product(S, T), product(I, T.difference(I)), combinations(I, 2))


def embedding_contraction(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    progress_bar: bool = False,
    n_workers: int = 0
) -> float:
    """
    Calculate the contraction of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs contained in S
    T: A subset of X. If given the function calculates only with pairs in S x T
    progress_bar: If True, display a progress bar for the contraction
        calculation
    n_workers: The number of parallel subprocesses. 0 or 1 will means no
        parallelism and above 1 is the number of worker processes


    Returns
    -------
    The contraction of the embedding, defined as the maximum over all u,v 
    of d_X(u, v) / d_Y(f(u), f(v))
    """
    S, T, I = __preprocess(X, S, T)
    pairs = __pairs(S, T, I)
    if progress_bar:
        pairs = tqdm(pairs, total=len(S) * len(T) + len(I) * (len(I) - 1) // 2)
    if n_workers == 0:
        c = contraction_curried(X.d, d_Y, f)
        ctag = lambda t: c(*t)
        return max(1, max(map(ctag, pairs)))
    c = partial(contraction1, X.d, d_Y, f)
    n_workers = min(n_workers, os.cpu_count() or 1)
    with Pool(n_workers) as p:
        return max(1, max(p.map(c, pairs)))
    

def embedding_expansion(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    progress_bar: bool = False,
    n_workers: int = 0
) -> float:
    """
    Calculate the expansion of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs contained in S
    T: A subset of X. If given the function calculates only with pairs in S x T
    progress_bar: If True, display a progress bar for the contraction
        calculation
    n_workers: The number of parallel subprocesses. 0 or 1 will means no
        parallelism and above 1 is the number of worker processes

    Returns
    -------
    The expansion of the embedding, defined as the maximum over all u,v 
    of  d_Y(f(u), f(v)) / d_X(u, v)
    """
    S, T, I = __preprocess(X, S, T)
    pairs = __pairs(S, T, I)
    if progress_bar:
        pairs = tqdm(pairs, total=len(S) * len(T) + len(I) * (len(I) - 1) // 2)
    if n_workers == 0:
        e = expansion_curried(X.d, d_Y, f)
        etag = lambda t: e(*t)
        return max(1, max(map(etag, pairs)))
    n_workers = min(n_workers, os.cpu_count() or 1)
    e = partial(expansion1, X.d, d_Y, f)
    with Pool(n_workers) as p:
        return max(1, max(p.map(e, pairs)))


def embedding_distortion(
    X: FiniteMetricSpace[T1],
    d_Y: Metric[T2],
    f: Embedding[T1, T2],
    S: Optional[Iterable[T1]] = None,
    T: Optional[Iterable[T1]] = None,
    progress_bar: bool = False,
    n_workers: int = 0
) -> float:
    """
    Calculate the distortion of an embedding f: (X, d_X) -> (Y, d_Y)

    Parameters
    ----------
    X: A finite metric space being embedded
    d_Y: A metric that acts on the image of the embedding f
    f: A function mapping defined on the points of x, mapping to the domain of d_Y
    S: A subset of X. If given, the function calculates only with pairs contained in S
    T: A subset of X. If given the function calculates only with pairs in S x T
    progress_bar: If True, display a progress bar for the contraction
        calculation
    n_workers: The number of parallel subprocesses. 0 or 1 will means no
        parallelism and above 1 is the number of worker processes

    Returns
    -------
    The distortion of the embedding, defined as the expansion * distortion
    """
    return (
        embedding_contraction(X, d_Y, f, S, T, progress_bar=progress_bar, n_workers=n_workers) 
        * embedding_expansion(X, d_Y, f, S, T, progress_bar=progress_bar, n_workers=n_workers)
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
    S: A subset of X. If given, the function calculates only with pairs contained in S
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
        __pairs(S, T, I)
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
    S: A subset of X. If given, the function calculates only with pairs contained in S
    T: A subset of X. If given the function calculates only with pairs in S x T
    threshold: A number which is at least 1. pairs who are contracted by a factor
        larger then threshold are returned
    
    Returns
    -------
    A set of pairs that were contracted by a factor larger then threshold
    """
    S, T, I = __preprocess(X, S, T)
    e = contraction_curried(X.d, d_Y, f)
    return filter(
        lambda t: e(*t) > threshold, 
        __pairs(S, T, I)
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
    S: A subset of X. If given, the function calculates only with pairs contained in S
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
