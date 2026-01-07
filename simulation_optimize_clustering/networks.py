import os
import sys
import pickle
from typing import Dict, Any, Tuple

import numpy as np
import networkx as nx
import rewiring_package as rp


def load_data(path: str) -> Dict[str, Any]:
    """
    Load a pickled data dictionary from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data(path: str, data: Dict[str, Any]) -> None:
    """
    Ensure directory exists and save a pickled data dictionary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_adjacency(path: str, adjacency: np.ndarray) -> None:
    """
    Ensure directory exists and save adjacency matrix as a NumPy .npy file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, adjacency)


def create_geometric_graph(
    num_nodes: int,
    radius: float
) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Create a 2D geometric graph with periodic boundary conditions.

    Returns:
        G: the generated graph
        pos: node positions in [0,1]x[0,1]
    """
    G = nx.Graph()
    positions = np.random.rand(num_nodes, 2)
    pos = {i: tuple(positions[i]) for i in range(num_nodes)}
    G.add_nodes_from(pos)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dx = abs(pos[i][0] - pos[j][0])
            dy = abs(pos[i][1] - pos[j][1])
            dx = min(dx, 1 - dx)
            dy = min(dy, 1 - dy)
            if np.hypot(dx, dy) <= radius:
                G.add_edge(i, j)

    return G, pos


def generate_and_save_networks(
    size: int,
    mean_degree: float,
    output_dir: str
) -> None:
    """
    Generate various network types and save their adjacency matrices.
    """
    types = {
        'random_geometric': lambda: create_geometric_graph(
            num_nodes=size,
            radius=np.sqrt(mean_degree / (np.pi * (size - 1)))
        )[0],
        'random_regular': lambda: rp.random_regular(n=size, k=int(mean_degree)),
        'erdos_renyi': lambda: rp.erdos_renyi(n=size, p=mean_degree / (size - 1)),
        'watts_strogatz': lambda: rp.watts_strogatz(n=size, k=int(mean_degree), p=0.1),
        'barabasi_albert': lambda: rp.barabasi_albert(size=size, m=int(mean_degree // 2))
    }

    base = output_dir or sys.path[0]
    net_dir = os.path.join(base, 'networks')

    for name, constructor in types.items():
        G = constructor()
        adj = nx.adjacency_matrix(G).toarray()
        file_path = os.path.join(net_dir, f"{name}.npy")
        save_adjacency(file_path, adj)
        print(f"Saved {name} adjacency to {file_path}")


def main():
    """
    Entry point: configure parameters and generate networks.
    """
    size = 100
    mean_degree = 4
    output_dir = sys.path[0]

    generate_and_save_networks(size, mean_degree, output_dir)


if __name__ == '__main__':
    main()
