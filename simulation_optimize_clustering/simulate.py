import os
import sys
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
import networkx as nx
import rewiring_package as rp

def save_data(path: str, data: dict) -> None:
    """
    Ensure directory exists and pickle the data to the given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_adjacency(file_path: str) -> np.ndarray:
    """
    Load an adjacency matrix stored as a NumPy .npy file.
    """
    return np.load(file_path)

def generate_random_graph(adjacency: np.ndarray) -> nx.Graph:
    """
    Create a random graph with the same degree sequence as the given adjacency.
    """
    G = nx.from_numpy_array(adjacency)
    degrees = [deg for _, deg in G.degree()]
    return nx.random_degree_sequence_graph(degrees, tries=1000)

def optimize_orbit(
    G: nx.Graph,
    rewire_count: int,
    temperature: float,
    forward: bool
) -> Dict[str, list]:
    """
    Run clustering optimization in the forward or backward direction.

    Args:
        G: initial graph
        rewire_count: number of edges to rewire per step
        temperature: annealing parameter T
        forward: if True, optimize clustering; else, optimize anti-clustering

    Returns:
        A dictionary of orbit data for the run
    """
    return rp.optimise_clustering(
        G,
        rewire_count=rewire_count,
        T=temperature,
        optimise=forward
    )

def combine_orbits(
    nodes: list,
    orbit_back: Dict[str, list],
    orbit_front: Dict[str, list]
) -> Dict[str, list]:
    """
    Merge backward and forward orbit data into a single chronology.
    """
    combined = {'nodes': nodes}
    for key, back_vals in orbit_back.items():
        if key == 'nodes':
            continue
        combined[key] = back_vals[::-1] + orbit_front.get(key, [])
    return combined

def simulate_one(
    trial_type: str,
    adjacency: np.ndarray,
    rewire_count: int,
    temperature: float,
    trial_index: int,
    output_dir: str
) -> None:
    """
    Run a single trial: generate graph, optimize clustering, and save orbit data.
    """
    G_random = generate_random_graph(adjacency)
    orbit_front = optimize_orbit(G_random, rewire_count, temperature, forward=True)
    orbit_back = optimize_orbit(G_random, rewire_count, temperature, forward=False)

    orbit = combine_orbits(
        nodes=list(G_random.nodes()),
        orbit_back=orbit_back,
        orbit_front=orbit_front
    )

    save_path = os.path.join(output_dir, trial_type, f"trial_{trial_index}_data_dict.pkl")
    save_data(save_path, orbit)

def simulate_all(
    trial_type: str,
    adjacency: np.ndarray,
    iterations: int,
    rewire_count: int,
    temperature: float,
    base_dir: Optional[str] = None
) -> None:
    """
    Perform multiple trials of clustering optimization for a given network type.
    """
    base_dir = base_dir or sys.path[0]
    output_dir = os.path.join(base_dir, trial_type)
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(1, iterations + 1):
        simulate_one(
            trial_type=trial_type,
            adjacency=adjacency,
            rewire_count=rewire_count,
            temperature=temperature,
            trial_index=idx,
            output_dir=base_dir
        )
        percent = 100 * idx / iterations
        print(f"{trial_type}: {percent:.2f}%", end='\r')
    print(f"\nFinished {trial_type} trials.")

def main():
    """
    Load each network adjacency and run simulations.
    """
    T = 0.001
    iterations = 100
    rewire_count = 1000
    types = [
        "erdos_renyi",
        "random_regular",
        "random_geometric",
        "watts_strogatz",
        "barabasi_albert"
    ]
    base = sys.path[0]

    for t in types:
        adj_path = os.path.join(base, f"networks/{t}.npy")
        adjacency = load_adjacency(adj_path)
        simulate_all(
            trial_type=t,
            adjacency=adjacency,
            iterations=iterations,
            rewire_count=rewire_count,
            temperature=T,
            base_dir=os.path.join(base, t)
        )


if __name__ == '__main__':
    main()