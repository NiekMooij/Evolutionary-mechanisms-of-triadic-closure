import numpy as np
import random

from .get_first_bifurcation import get_first_bifurcation

def rewire_random_edges(G):
    """
    Rewires two randomly chosen edges in the graph G.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        networkx.Graph or bool: Rewired graph if successful, False otherwise.
    """
    # Get a list of edges for rewiring
    edges = list(G.edges())

    # Choose two random edges
    if len(edges) < 2:
        return False  # Cannot rewire with less than two edges
    edge1, edge2 = random.sample(edges, 2)

    # Get the nodes of the chosen edges
    u1, v1 = edge1
    u2, v2 = edge2

    # Check if the chosen edges share any nodes
    if len(set([u1, v1, u2, v2])) < 4:
        return False  # Cannot rewire edges sharing nodes

    # Define potential new edges
    if random.random() < 1/2:
        new_edge1 = (u1, u2)
        new_edge2 = (v1, v2)
    else:
        new_edge1 = (u1, v2)
        new_edge2 = (v1, u2)    

    # Check if potential new edges already exist
    if not G.has_edge(*new_edge1) and not G.has_edge(*new_edge2):
        # If not, create a copy of the graph to perform rewiring
        G_copy = G.copy()

        # Rewire the edges
        G_copy.remove_edges_from([edge1, edge2])
        G_copy.add_edges_from([new_edge1, new_edge2])
        return G_copy

    return False

def rewired_graph(G, max_attempts=1000):
    """
    Attempts to rewire edges in the graph G.

    Parameters:
        G (networkx.Graph): The input graph.
        max_attempts (int): Maximum number of attempts to rewire edges.

    Returns:
        networkx.Graph or bool: Rewired graph if successful, False otherwise.
    """
    for _ in range(max_attempts):
        G_rewired = rewire_random_edges(G)
        if G_rewired:
            return G_rewired

    print('No potential rewiring found!')
    return False

def accept_rewire(tau_old, G_rewired, T, optimise=True):
    """
    Determines whether to accept the rewiring based on the Metropolis criterion.

    Parameters:
        G (networkx.Graph): Original graph.
        G_rewired (networkx.Graph): Rewired graph.
        T (float): Temperature parameter.

    Returns:
        bool: True if the rewiring is accepted, False otherwise.
    """
    tolerance = 1e-8
    tau_initial = 1e-20
    
    tau_rewired, _ = get_first_bifurcation(G=G_rewired, tau_initial=tau_initial, tolerance=tolerance)

    diff = tau_rewired - tau_old

    if optimise:
        if diff > 0:
            return True, tau_rewired
        
        else:
            if T == 0:
                return False, tau_old
            elif np.random.uniform(0, 1) < np.exp(diff / T):
                return True, tau_rewired
            else:
                return False, tau_old
            
    else:
        if diff < 0:
            return True, tau_rewired
        
        else:
            if T == 0:
                return False, tau_old
            elif np.random.uniform(0, 1) < np.exp(-diff / T):
                return True, tau_rewired
            else:
                return False, tau_old

def rewire_iteration(G, tau_old, T=2.1, optimise=True):
    """
    Perform one iteration of the rewiring process.

    Parameters:
        G (networkx.Graph): The input graph.
        T (float): Temperature parameter.

    Returns:
        networkx.Graph: Rewired graph.
    """
    G_rewired = rewired_graph(G)

    if G_rewired:
        # print(accept_rewire(G, G_rewired, T))
        flag, tau = accept_rewire(tau_old=tau_old, G_rewired=G_rewired, T=T, optimise=optimise)
    
        if flag:
            return True, G_rewired, tau
            
        else: 
            return False, G, tau_old
        
def network_cycle(G, rewire_count=100, T=0.0025, seek_maximum=False, optimise=True):
    G_arr = []
    tau_arr = [get_first_bifurcation(G=G, tau_initial=1e-8, tolerance=1e-20)[0]]

    for index in range(rewire_count):
        rewired_flag, G, tau = rewire_iteration(G, tau_old=tau_arr[-1], T=T, optimise=optimise)
        G_arr.append(G)
        tau_arr.append(tau)
        
        percentage = (index+1) / rewire_count * 100
        print(f'{percentage:.2f}% done', end='\r')

        if seek_maximum and abs(tau-1) < 1e-7:
            break
        
    cycle = { 'nodes': G.nodes(), 'edges': [ G.edges() for G in G_arr ], 'tau_arr': tau_arr }

    return cycle