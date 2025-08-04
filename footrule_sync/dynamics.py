import math
from typing import List, Sequence, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx


def footrule_distance(p: Sequence[int], q: Sequence[int]) -> int:
    p = np.asarray(p)
    q = np.asarray(q)
    m = len(p)
    pos_p = np.empty(m, dtype=int)
    pos_q = np.empty(m, dtype=int)
    pos_p[p] = np.arange(m)
    pos_q[q] = np.arange(m)
    return int(np.abs(pos_p - pos_q).sum())


def encode_permutation(perm: Sequence[int]) -> int:
    perm = list(perm)
    m = len(perm)
    code = 0
    factorial = math.factorial
    for i in range(m):
        x = perm[i]
        smaller = sum(1 for y in perm[i + 1 :] if y < x)
        code += smaller * factorial(m - i - 1)
    return code


def potential(perms: np.ndarray, graph: nx.Graph) -> int:
    total = 0
    for u, v in graph.edges():
        total += footrule_distance(perms[u], perms[v])
    return int(total)


def normalised_potential(perms: np.ndarray, graph: nx.Graph) -> float:
    edges = graph.number_of_edges()
    if edges == 0:
        return 0.0
    m = perms.shape[1]
    return potential(perms, graph) / (edges * (m * m / 2))


def footrule_median(perms: Sequence[Sequence[int]]) -> List[int]:
    perms = np.asarray(perms, dtype=int)
    k, m = perms.shape
    positions = np.argsort(perms, axis=1)  # alt -> position
    ranks = np.arange(m)
    cost = np.abs(positions[:, :, None] - ranks[None, None, :]).sum(axis=0)

    row_ind, col_ind = linear_sum_assignment(cost)
    base_cost = cost[row_ind, col_ind].sum()

    def recurse(prefix, remaining_alts, remaining_ranks, remaining_cost):
        if not remaining_ranks:
            return prefix
        rank = remaining_ranks[0]
        for alt in sorted(remaining_alts):
            next_alts = [a for a in remaining_alts if a != alt]
            next_ranks = remaining_ranks[1:]
            if next_ranks:
                sub_cost = cost[np.ix_(next_alts, next_ranks)]
                ri, ci = linear_sum_assignment(sub_cost)
                rest = sub_cost[ri, ci].sum()
            else:
                rest = 0
            total = cost[alt, rank] + rest
            if total == remaining_cost:
                result = recurse(prefix + [alt], next_alts, next_ranks, remaining_cost - cost[alt, rank])
                if result is not None:
                    return result
        return None

    perm = recurse([], list(range(m)), list(range(m)), base_cost)
    return perm


def run_sync_dynamics(
    graph: nx.Graph,
    initial_perms: np.ndarray,
    *,
    max_steps: int = 500,
    self_inclusive: bool = False,
) -> Dict:
    perms = np.array(initial_perms, dtype=int)
    n, m = perms.shape

    phi_history = [potential(perms, graph)]
    state = tuple(encode_permutation(perms[i]) for i in range(n))
    seen = {state: 0}

    for step in range(1, max_steps + 1):
        next_perms = np.empty_like(perms)
        for u in range(n):
            neigh = list(graph.neighbors(u))
            perms_list = [perms[v] for v in neigh]
            if self_inclusive or not perms_list:
                perms_list.append(perms[u])
            next_perms[u] = footrule_median(perms_list)

        if np.array_equal(next_perms, perms):
            converged = True
            cycle_length = 0
            steps_run = step - 1
            break

        phi_history.append(potential(next_perms, graph))
        state = tuple(encode_permutation(next_perms[i]) for i in range(n))
        if state in seen:
            converged = False
            cycle_length = step - seen[state]
            steps_run = step
            perms = next_perms
            break
        seen[state] = step
        perms = next_perms
    else:
        converged = False
        cycle_length = 0
        steps_run = max_steps
        perms = next_perms

    return {
        "perms": perms,
        "phi_history": phi_history,
        "converged": converged,
        "cycle_length": cycle_length,
        "steps_run": steps_run,
    }
