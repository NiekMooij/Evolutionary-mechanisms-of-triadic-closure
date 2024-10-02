import networkx as nx
import random

def uniform_graph(N, d_max, d_min, d_rest):
    # Create the degree sequence
    degrees = [random.randint(d_min, d_rest) for _ in range(N)]
    degrees[0] = d_max  # Set the degree of the first node to d_max
    
    # Ensure the sum of degrees is even
    if sum(degrees) % 2 != 0:
        # Adjust one of the degrees to make the sum even
        for i in range(1, N):
            if degrees[i] > d_min:
                degrees[i] -= 1
                break
            elif degrees[i] < d_rest:
                degrees[i] += 1
                break
    
    # Ensure the degree sequence is valid
    if sum(degrees) % 2 != 0:
        raise ValueError("The sum of degrees must be even for a valid graph.")
    
    # Generate the graph
    G = nx.configuration_model(degrees)
    
    # Convert to a simple graph (no parallel edges or self-loops)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G