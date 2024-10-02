import networkx as nx
import random

def erdos_renyi(size: int, p: float, connected: bool = False) -> nx.Graph:
    """
    Generate an Erdős-Rényi random graph.

    Parameters:
    - size (int): The number of nodes in the graph.
    - p (float): The probability of an edge between any two nodes.
    - connected (bool): Whether the graph should be connected.

    Returns:
    - nx.Graph: The generated Erdős-Rényi graph.
    """
    while True:
        G = nx.Graph()
        G.add_nodes_from(range(1, size+1))

        # Create a list of all possible edges
        edge_arr = [(i, j) for i in range(1, size+1) for j in range(1, size+1) if i < j]

        # Add edges with probability p
        for edge in edge_arr:
            if random.random() < p:
                G.add_edge(*edge)

        if not connected:
            # Stop the process when the graph does not have to be connected
            break

        if nx.is_connected(G):
            # Break whenever the graph is connected
            break

    return G