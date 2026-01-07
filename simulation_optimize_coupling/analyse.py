import os
import sys
import pickle
from typing import List, Dict, Any, Callable

import numpy as np
import networkx as nx
import rewiring_package as rp  # still used for other tasks; not needed for clustering specifically
from tqdm.auto import tqdm  # NEW: progress bars


def load_data(path: str) -> Dict[str, Any]:
    """Load a pickled data dictionary from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data(path: str, data: Dict[str, Any]) -> None:
    """Ensure directory exists and save a pickled data dictionary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def _build_graph(nodes: List[Any], edges: List[tuple]) -> nx.Graph:
    """Helper: build a Graph from nodes + edges, robust to edge view types."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # Ensure each edge is a 2-tuple (u, v)
    G.add_edges_from((tuple(e) if not isinstance(e, tuple) else e) for e in edges)
    return G


def compute_clustering(nodes: List[Any], edges_list: List[List[tuple]]) -> List[float]:
    """
    Compute NetworkX average clustering for each snapshot.
    """
    vals = []
    for edges in edges_list:
        G = _build_graph(nodes, edges)
        vals.append(nx.average_clustering(G))
    return vals


def compute_assortativity(nodes: List[Any], edges_list: List[List[tuple]]) -> List[float]:
    """
    Compute degree assortativity for each snapshot of edges.
    """
    vals = []
    for edges in edges_list:
        G = _build_graph(nodes, edges)
        vals.append(nx.degree_assortativity_coefficient(G))
    return vals


def process_trial_metrics(
    trial_type: str,
    trial_index: int,
    compute_fn: Callable[[List[Any], List[List[tuple]]], List[float]],
    key_name: str
) -> None:
    """
    Load a single trial data file, compute metrics with compute_fn, and save under key_name.
    """
    base = sys.path[0]
    file_name = os.path.join(base, trial_type, f"trial_{trial_index}_data_dict.pkl")
    data = load_data(file_name)
    nodes = list(data.get('nodes', []))
    edges_list = data.get('edges', [])

    metric_values = compute_fn(nodes, edges_list)
    data[key_name] = metric_values
    save_data(file_name, data)
    # Progress is handled by tqdm in the caller; no print here.


def batch_process(
    trial_type: str,
    iterations: int,
    compute_fn: Callable[[List[Any], List[List[tuple]]], List[float]],
    key_name: str
) -> None:
    """Run process_trial_metrics across multiple trials with a tqdm progress bar."""
    with tqdm(total=iterations, desc=f"{trial_type} · {key_name}", unit="trial", leave=False, position=1) as pbar:
        for idx in range(1, iterations + 1):
            process_trial_metrics(trial_type, idx, compute_fn, key_name)
            pbar.update(1)
    tqdm.write(f"Finished {key_name} for {trial_type} ({iterations} trials).")


def main():
    """
    Add clustering (instead of critical tau) and assortativity to each trial for given network types.
    Shows nested tqdm bars: one over network types and one over trials per metric.
    """
    iterations = 100
    trial_types = [
        'erdos_renyi',
        'random_geometric',
        'random_regular',
        'watts_strogatz',
        'barabasi_albert'
    ]

    for t in tqdm(trial_types, desc="Network types", unit="type", position=0):
        # Store average clustering time series
        batch_process(t, iterations, compute_clustering, 'clustering')
        # Keep assortativity as before
        batch_process(t, iterations, compute_assortativity, 'assortativity')


if __name__ == '__main__':
    main()
