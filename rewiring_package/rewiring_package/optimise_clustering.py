import numpy as np
import random
import networkx as nx

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

def accept_rewire(clustering_old, G_rewired, T, optimise=True):
    """
    Determines whether to accept the rewiring based on the Metropolis criterion.

    Parameters:
        G (networkx.Graph): Original graph.
        G_rewired (networkx.Graph): Rewired graph.
        T (float): Temperature parameter.

    Returns:
        bool: True if the rewiring is accepted, False otherwise.
    """
    
    clustering_rewired = nx.average_clustering(G_rewired)

    diff = clustering_rewired - clustering_old

    if optimise:
        if diff > 0:
            return True, clustering_rewired
        
        else:
            if T == 0:
                return False, clustering_old
            elif np.random.uniform(0, 1) < np.exp(diff / T):
                return True, clustering_rewired
            else:
                return False, clustering_old
            
    else:
        if diff < 0:
            return True, clustering_rewired
        
        else:
            if T == 0:
                return False, clustering_old
            elif np.random.uniform(0, 1) < np.exp(-diff / T):
                return True, clustering_rewired
            else:
                return False, clustering_old

def rewire_iteration(G, clustering_old, T=2.1, optimise=True):
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
        flag, clustering = accept_rewire(clustering_old=clustering_old, G_rewired=G_rewired, T=T, optimise=optimise)
    
        if flag:
            return True, G_rewired, clustering
            
        else: 
            return False, G, clustering_old 
        
def optimise_clustering(G, rewire_count=100, T=0.0025, optimise=True):
    G_arr = []
    clustering_arr = [nx.average_clustering(G)]

    for index in range(rewire_count):
        rewired_flag, G, clustering = rewire_iteration(G, clustering_old=clustering_arr[-1], T=T, optimise=optimise)
        G_arr.append(G)
        clustering_arr.append(clustering)
        
        percentage = (index+1) / rewire_count * 100
        print(f'{percentage:.2f}% done', end='\r')
        
    cycle = { 'nodes': G.nodes(), 'edges': [ G.edges() for G in G_arr ], 'clustering_arr': clustering_arr }

    return cycle