import networkx as nx
import random

def random_geometric(size: int, k: float, connected: bool = False) -> nx.Graph:
    """
    Generate a geometric random graph.

    Parameters:
    - size (int): The number of nodes in the graph.
    - k (float): The threshold distance for edges between nodes.
    - connected (bool): Whether the graph should be connected.

    Returns:
    - nx.Graph: The generated geometric graph.
    """

    # Continue adding nodes until the graph becomes connected
    while True:
        G = nx.Graph()

        # Add nodes with random positions in the unit square
        for i in range(1, size + 1):
            x, y = random.random(), random.random()
            G.add_node(i, pos=(x, y))

        # Add edges based on the distance between nodes
        for i in range(1, size + 1):
            for j in range(i + 1, size + 1):
                x1, y1 = G.nodes[i]['pos']
                x2, y2 = G.nodes[j]['pos']
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if distance < k:
                    G.add_edge(i, j)

        if not connected:
            # Stop the process when the graph does not have to be connected
            break

        if nx.is_connected(G):
            # Break whenever the graph is connected
            break

    return G