import networkx as nx
import numpy as np


def erdos_renyi_connected(n: int, p: float, rng: np.random.Generator):
    """Generate a connected Erdős–Rényi graph G(n,p).

    Returns the graph and the number of resampling attempts before success.
    """
    attempts = 0
    while True:
        attempts += 1
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    G.add_edge(i, j)
        if n == 1 or nx.is_connected(G):
            break
    return G, attempts - 1
