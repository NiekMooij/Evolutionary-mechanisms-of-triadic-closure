import numpy as np
import random
import networkx as nx

from .get_first_bifurcation import get_first_bifurcation

def rewire_random_edges(G):
    """
    Rewires two randomly chosen edges in the graph G.

    Returns:
        networkx.Graph or bool: Rewired graph if successful, False otherwise.
    """
    edges = list(G.edges())
    if len(edges) < 2:
        return False  # Cannot rewire with less than two edges

    edge1, edge2 = random.sample(edges, 2)
    u1, v1 = edge1
    u2, v2 = edge2

    # Disallow shared nodes (keeps degree sequence unchanged by a simple swap)
    if len(set([u1, v1, u2, v2])) < 4:
        return False

    if random.random() < 0.5:
        new_edge1 = (u1, u2)
        new_edge2 = (v1, v2)
    else:
        new_edge1 = (u1, v2)
        new_edge2 = (v1, u2)

    if not G.has_edge(*new_edge1) and not G.has_edge(*new_edge2):
        G_copy = G.copy()
        G_copy.remove_edges_from([edge1, edge2])
        G_copy.add_edges_from([new_edge1, new_edge2])
        return G_copy

    return False

def rewired_graph(G, max_attempts=1000):
    """
    Attempts to rewire edges in the graph G.

    Returns:
        networkx.Graph or bool: Rewired graph if successful, False otherwise.
    """
    for _ in range(max_attempts):
        G_rewired = rewire_random_edges(G)
        if G_rewired:
            return G_rewired
    print('No potential rewiring found!')
    return False

def objective_value(G, tau_initial, tolerance, regular=False):
    """
    Computes the scalar objective to optimize: the first element returned by get_first_bifurcation.
    Add a small guard for numerical issues; if computation fails, return np.nan so it's auto-rejected.
    """
    try:
        val = get_first_bifurcation(G, tau_initial, tolerance, regular=regular)[0]
        return float(val)
    except Exception as e:
        # If evaluation fails for a candidate, treat as invalid (worse).
        return np.nan

def accept_rewire(value_old, G_rewired, T, tau_initial, tolerance, regular=False, optimise=True):
    """
    Metropolis acceptance step based on the objective value from get_first_bifurcation(...)[0].

    Args:
        value_old (float): Objective for the current graph.
        G_rewired (nx.Graph): Candidate graph.
        T (float): Temperature.
        optimise (bool): If True, maximize; if False, minimize.

    Returns:
        (bool, float): (accept?, value_new_if_accepted_else_value_old)
    """
    value_new = objective_value(G_rewired, tau_initial, tolerance, regular=regular)

    # Reject NaN/invalid evaluations outright
    if not np.isfinite(value_new):
        return False, value_old

    diff = value_new - value_old

    if optimise:  # maximize
        if diff > 0:
            return True, value_new
        if T == 0:
            return False, value_old
        return (np.random.uniform(0, 1) < np.exp(diff / T), value_new if np.random.uniform(0, 1) < np.exp(diff / T) else value_old)
    else:         # minimize
        if diff < 0:
            return True, value_new
        if T == 0:
            return False, value_old
        prob = np.exp(-diff / T)
        accept = np.random.uniform(0, 1) < prob
        return (accept, value_new if accept else value_old)

def rewire_iteration(G, value_old, T=2.1, tau_initial=1.0, tolerance=1e-6, regular=False, optimise=True):
    """
    Perform one iteration of the rewiring process for the bifurcation objective.

    Returns:
        (bool, nx.Graph, float): (accepted?, graph, objective_value_for_returned_graph)
    """
    G_rewired = rewired_graph(G)
    if not G_rewired:
        # No valid rewiring found; keep current
        return False, G, value_old

    accepted, value_out = accept_rewire(
        value_old=value_old,
        G_rewired=G_rewired,
        T=T,
        tau_initial=tau_initial,
        tolerance=tolerance,
        regular=regular,
        optimise=optimise
    )

    if accepted:
        return True, G_rewired, value_out
    else:
        return False, G, value_old

def optimise_coupling(
    G,
    rewire_count=100,
    T=0.0025,
    tau_initial=1.0,
    tolerance=1e-6,
    regular=False,
    optimise=True
):
    """
    Simulated annealing-style optimization of get_first_bifurcation(G, ...)[0].

    Args:
        optimise (bool): True -> maximize the objective. False -> minimize it.
    """
    G_arr = []
    # Initial objective
    value0 = objective_value(G, tau_initial, tolerance, regular=regular)
    if not np.isfinite(value0):
        raise ValueError("Initial objective value is not finite. Check get_first_bifurcation inputs or graph.")
    value_arr = [value0]

    for index in range(rewire_count):
        accepted, G, value = rewire_iteration(
            G,
            value_old=value_arr[-1],
            T=T,
            tau_initial=tau_initial,
            tolerance=tolerance,
            regular=regular,
            optimise=optimise
        )
        G_arr.append(G)
        value_arr.append(value)

        percentage = (index + 1) / rewire_count * 100
        print(f'{percentage:.2f}% done', end='\r')

    cycle = {
        'nodes': G.nodes(),
        'edges': [G.edges() for G in G_arr],
        'objective_arr': value_arr,
    }
    return cycle