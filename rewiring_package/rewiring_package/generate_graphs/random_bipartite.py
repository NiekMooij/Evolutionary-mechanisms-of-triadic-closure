import networkx as nx
import random

def random_bipartite(sizeA: int, sizeB: int, p: float, connected: bool = False) -> nx.Graph:
    """
    Generate a random bipartite graph.

    Parameters:
    - sizeA (int): Number of vertices in set A.
    - sizeB (int): Number of vertices in set B.
    - p (float): Probability of an edge between vertices in set A and set B.
    - connected (bool): Whether the graph should be connected.

    Returns:
    - nx.Graph: Bipartite graph with nodes labeled from 1 to sizeA+sizeB.
    """
    while True:
        # Re-initialize the graph and nodes
        G = nx.Graph()
        A_vertices = list(range(1, sizeA + 1))
        B_vertices = list(range(sizeA + 1, sizeA + sizeB + 1))
        G.add_nodes_from(A_vertices, bipartite=0)
        G.add_nodes_from(B_vertices, bipartite=1)

        # Add edges with probability p
        for vertexA in A_vertices:
            for vertexB in B_vertices:
                if random.random() < p:
                    G.add_edge(vertexA, vertexB)

        if not connected:
            # Stop the process when the graph does not have to be connected
            break

        if nx.is_connected(G):
            # Break whenever the graph is connected
            break

    return G