from collections import Counter
import numpy as np
import networkx as nx
from .dynamics import encode_permutation, footrule_distance


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


def plurality_winner(perms: np.ndarray) -> int:
    """Returns the item that is most often ranked first (plurality winner)."""
    n, m = perms.shape
    first_choices = perms[:, 0]  # First element of each permutation
    counts = Counter(first_choices)
    return counts.most_common(1)[0][0]


def consensus_fraction(perms: np.ndarray) -> float:
    """Fraction of agents whose top choice equals the plurality winner."""
    winner = plurality_winner(perms)
    first_choices = perms[:, 0]
    return np.mean(first_choices == winner)


def disagreement_components(perms: np.ndarray, graph: nx.Graph) -> int:
    """Number of connected components in the disagreement graph."""
    n = len(perms)
    disagreement_graph = nx.Graph()
    disagreement_graph.add_nodes_from(range(n))
    
    codes = [encode_permutation(p) for p in perms]
    for u, v in graph.edges():
        if codes[u] != codes[v]:  # Disagreement edge
            disagreement_graph.add_edge(u, v)
    
    return nx.number_connected_components(disagreement_graph)


def winner_flipped(initial_perms: np.ndarray, final_perms: np.ndarray) -> bool:
    """Check if plurality winner changed from initial to final state."""
    initial_winner = plurality_winner(initial_perms)
    final_winner = plurality_winner(final_perms)
    return initial_winner != final_winner


def footrule_shift(initial_perms: np.ndarray, final_perms: np.ndarray) -> float:
    """Average Footrule distance between initial and final permutations."""
    n = len(initial_perms)
    total_shift = 0
    for i in range(n):
        total_shift += footrule_distance(initial_perms[i], final_perms[i])
    return total_shift / n
