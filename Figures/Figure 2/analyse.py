import os
import sys
import pickle
from typing import List, Dict, Any, Callable

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


def compute_critical_tau(nodes: List[Any], edges_list: List[List[tuple]]) -> List[float]:
    """
    Given node list and list of edge sets, return critical tau for each snapshot.
    """
    tau_values = []
    for edges in edges_list:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        tau = rp.get_first_bifurcation(
            G=G,
            tau_initial=1e-8,
            tolerance=1e-8,
            regular=False
        )
        tau_values.append(tau)
    return tau_values


def compute_assortativity(nodes: List[Any], edges_list: List[List[tuple]]) -> List[float]:
    """
    Compute degree assortativity for each snapshot of edges.
    """
    assort_values = []
    for edges in edges_list:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        assort_values.append(nx.degree_assortativity_coefficient(G))
    return assort_values


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
    nodes = data.get('nodes', [])
    edges_list = data.get('edges', [])

    metric_values = compute_fn(nodes, edges_list)
    data[key_name] = metric_values
    save_data(file_name, data)
    print(f"{trial_type} trial {trial_index}: computed {key_name}")


def batch_process(
    trial_type: str,
    iterations: int,
    compute_fn: Callable[[List[Any], List[List[tuple]]], List[float]],
    key_name: str
) -> None:
    """
    Run process_trial_metrics across multiple trials.
    """
    for idx in range(1, iterations + 1):
        process_trial_metrics(trial_type, idx, compute_fn, key_name)
    print(f"Finished computing {key_name} for {trial_type} ({iterations} trials)")


def main():
    """
    Add critical tau and assortativity to each trial for given network types.
    """
    iterations = 100
    trial_types = [
        'erdos_renyi',
        'random_geometric',
        'random_regular',
        'watts_strogatz',
        'barabasi_albert'
    ]

    for t in trial_types:
        batch_process(t, iterations, compute_critical_tau, 'tau')
        batch_process(t, iterations, compute_assortativity, 'assortativity_standard')


if __name__ == '__main__':
    main()
