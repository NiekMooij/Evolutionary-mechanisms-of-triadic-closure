import networkx as nx
import random

def barabasi_albert(size: int, m: int) -> nx.Graph:
    """
    Generates a Barabási-Albert (BA) scale-free network.
    
    Parameters:
    - size (int): The total number of nodes in the graph.
    - m (int): Number of edges to attach from a new node to existing nodes.
    
    Returns:
    - G (networkx.Graph): A NetworkX graph representing the BA model.
    """
    # Initialize the graph with a complete graph of m+1 nodes
    G = nx.complete_graph(range(1, m+2))
    
    # List of existing nodes to target for new edges
    target_nodes = list(G.nodes)
    
    for i in range(m+2, size+1):
        # Add the new node
        G.add_node(i)
        
        # Select m unique target nodes with probability proportional to their degree
        targets = set()
        while len(targets) < m:
            target = random.choice(target_nodes)
            targets.add(target)
        
        # Add edges from the new node to the chosen target nodes
        G.add_edges_from((i, target) for target in targets)
        
        # Update the list of target nodes: add the new node m times
        target_nodes.extend(targets)
        target_nodes.extend([i] * m)
    
    return G