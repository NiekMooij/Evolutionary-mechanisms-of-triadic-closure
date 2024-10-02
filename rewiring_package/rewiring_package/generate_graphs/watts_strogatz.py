import networkx as nx
import random

def watts_strogatz(n: int, k: int, p: float) -> nx.Graph:
    """
    Generate a Watts-Strogatz small-world graph.

    Parameters:
    n (int): The number of nodes in the graph.
    k (int): Each node is connected to its k nearest neighbors in a ring topology.
    p (float): The probability of rewiring each edge. Should be between 0 and 1.

    Returns:
    nx.Graph: A NetworkX graph object representing the generated small-world network.
    
    The algorithm starts with a ring lattice where each node is connected to its k nearest neighbors.
    Then it randomly rewires the edges with a probability p, creating shortcuts between nodes.
    This results in a small-world network with properties of both regular lattices and random graphs.
    """
    G = nx.Graph()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n)
            G.add_edge(i, (i - j) % n)

    # Rewire edges with probability p
    for u, v in list(G.edges()):
        if random.random() < p:
            # Choose a new node that is not connected to u
            new_v = u
            while new_v == u or G.has_edge(u, new_v):
                new_v = random.randint(0, n - 1)
            G.add_edge(u, new_v)
            G.remove_edge(u, v)

    return G