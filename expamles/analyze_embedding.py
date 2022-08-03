from typing import Iterable

import networkx as nx
from metric_embedding.core.embedding_analysis import (analyze_embedding,
                                                      plot_results)
from metric_embedding.embeddings.lp_embeddings import (GraphFrechetEmbedding,
                                                       l2_embedding,
                                                       l_infinity_embedding)
from metric_embedding.metrics.graph_metric import GraphMetricSpace
from metric_embedding.metrics.lp_metrics import LpMetric


def l2_embedding_generator(sizes: Iterable[int]):
    dY = LpMetric(2).d
    for size in sizes:
        G: nx.Graph = nx.cycle_graph(size)  # type: ignore
        dG = GraphMetricSpace(G)
        f = l2_embedding(dG)
        f = GraphFrechetEmbedding(dG, f.As)
        yield dG, dY, f


def analyze_l2_embedding():
    xs, yss = analyze_embedding(
        l2_embedding_generator(range(100, 1001, 100)),
        len,
        ["contraction", "distortion", lambda _, __, f: len(f)],  # type: ignore
        progress_bar=True
    )
    print(xs, yss)
    plot_results(xs, yss)

if __name__ == "__main__":
    analyze_l2_embedding()
