from collections import Counter
import numpy as np
import networkx as nx
from .dynamics import encode_permutation


def largest_cluster_fraction(perms: np.ndarray) -> float:
    codes = [encode_permutation(p) for p in perms]
    counts = Counter(codes)
    return max(counts.values()) / len(perms)


def mean_pair_agreement(perms: np.ndarray, graph: nx.Graph) -> float:
    edges = graph.number_of_edges()
    if edges == 0:
        return 1.0
    codes = [encode_permutation(p) for p in perms]
    agree = 0
    for u, v in graph.edges():
        if codes[u] == codes[v]:
            agree += 1
    return agree / edges
