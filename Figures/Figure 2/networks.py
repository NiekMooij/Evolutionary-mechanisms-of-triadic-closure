import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

import rewiring_package as rp
                
def save_data(folder, data_dict):
    with open(folder + "data_dict" + '.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

def create_geometric_graph(num_nodes, radius):
    """
    Create a 2D geometric graph with periodic (cyclic) boundary conditions.
    
    Parameters:
        num_nodes (int): Number of nodes in the graph.
        radius (float): Maximum distance to connect nodes.
        space_size (float): The width and height of the square space.
        
    Returns:
        G (networkx.Graph): The generated geometric graph with cyclic boundary conditions.
        pos (dict): Dictionary of node positions.
    """
    # Initialize graph and node positions
    G = nx.Graph()
    pos = {}

    # Randomly place nodes in a square space [0, space_size] x [0, space_size]
    positions = np.random.rand(num_nodes, 2)
    for i in range(num_nodes):
        pos[i] = positions[i]
        G.add_node(i)

    # Add edges with cyclic boundary conditions
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Calculate Euclidean distance considering cyclic boundaries
            dx = np.abs(pos[i][0] - pos[j][0])
            dy = np.abs(pos[i][1] - pos[j][1])

            # Wrap distances if nodes are across boundaries
            dx = min(dx, 1 - dx)
            dy = min(dy, 1 - dy)

            distance = np.sqrt(dx**2 + dy**2)

            # Add an edge if distance is within the radius
            if distance <= radius:
                G.add_edge(i, j)

    return G, pos

if __name__ == "__main__":
    size = 100
    d_mean = 4

    G, pos = create_geometric_graph(size, radius=np.sqrt(d_mean / (np.pi*(size-1))))
    G = rp.random_regular(size, d_mean)
    G = rp.erdos_renyi(size, p=d_mean / (size - 1))
    G = rp.watts_strogatz(n=size, k=d_mean, p=0.1)

    # Save adjacency matrices as numpy arrays
    random_geometric = nx.adjacency_matrix(G).todense()
    np.save(os.path.join(sys.path[0], "networks/random_geometric.npy"), random_geometric)

    random_regular = nx.adjacency_matrix(rp.random_regular(size, d_mean)).todense()
    np.save(os.path.join(sys.path[0], "networks/random_regular.npy"), random_regular)

    erdos_renyi = nx.adjacency_matrix(rp.erdos_renyi(size, p=d_mean / (size - 1))).todense()
    np.save(os.path.join(sys.path[0], "networks/erdos_renyi.npy"), erdos_renyi)

    watts_strogatz = nx.adjacency_matrix(rp.watts_strogatz(n=size, k=d_mean, p=0.1)).todense()
    np.save(os.path.join(sys.path[0], "networks/watts_strogatz.npy"), watts_strogatz)