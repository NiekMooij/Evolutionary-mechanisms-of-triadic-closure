import networkx as nx

def random_regular(size: int, d: int, connected: bool = False) -> nx.Graph:
    """
    Generate random regular graph.

    Parameters:
    - size (int): The number of nodes in the graph.
    - d (int): constant degree
    - connected (bool): Whether the graph should be connected.

    Returns:
    - nx.Graph: The generated random regular graph.
    """
    G = nx.random_regular_graph(d, size)
    
    if connected:
        while not nx.is_connected(G):
            G = nx.random_regular_graph(d, size)

    return G