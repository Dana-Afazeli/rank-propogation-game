import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import networkx as nx
from footrule_sync.dynamics import run_sync_dynamics



def test_three_by_three_dynamics():
    G = nx.complete_graph(3)
    initial = np.array([
        [0, 1, 2],  # 123
        [2, 0, 1],  # 312
        [1, 2, 0],  # 231
    ], dtype=int)
    res = run_sync_dynamics(G, initial, max_steps=20, self_inclusive=False)
    assert res["converged"]
    assert res["steps_run"] == 2


def test_single_node_converges():
    G = nx.Graph()
    G.add_node(0)
    initial = np.array([[0, 1, 2]], dtype=int)
    res = run_sync_dynamics(G, initial, max_steps=10, self_inclusive=False)
    assert res["converged"]
    assert res["steps_run"] == 0
    assert res["phi_history"] == [0]
