#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import List, Sequence, Optional

import networkx as nx
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample vascular plant networks for one habitat."
    )
    p.add_argument("--habitat-code", "-H", help="Override habitat code in CONFIG")

    return p.parse_args()

# ============================================================
# ===================== USER PARAMETERS =======================
# ============================================================

# type = "R3_NP"
args = parse_args()
type = args.habitat_code + "_NP"

# Directory containing your adjacency CSV files
INPUT_DIR = Path(f"sim_results/{type}/networks")

# Where you want to save degree sequences + generated adjacency matrices
OUTPUT_DIR = Path(f"sim_results/{type}/degree_sequences")

# Only process the adjacency CSVs, not interaction / full
FILE_PATTERN = "*_adjacency.csv"

# How many random graphs (with the same degree sequence) per input file
NUM_GRAPHS_PER_FILE = 500

# Base seed for reproducibility (or None for non-reproducible runs)
SEED = None

# Parameters for random_degree_sequence_graph
RDS_TRIES_PER_CALL = 500      # internal tries per call
RDS_MAX_RESTARTS = 1000       # how many times to restart before fallback
# ============================================================


def load_graph_from_file(path: Path) -> nx.Graph:
    """
    Load a graph from a CSV adjacency matrix with:
      - header row = column labels
      - first column = row labels
      - numerical entries (0/1 or weights) for edges.
    Returns a simple, undirected NetworkX graph.
    """
    ext = path.suffix.lower()

    if ext != ".csv":
        raise ValueError(f"Expected .csv adjacency file, got: {path}")

    # Read CSV with row labels as index
    df = pd.read_csv(path, index_col=0)
    adj = df.to_numpy()

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"CSV adjacency is not square: {path} with shape {adj.shape}"
        )

    G = nx.from_numpy_array(adj)

    # Make sure it's a simple undirected graph
    if G.is_directed():
        G = G.to_undirected()

    G = nx.Graph(G)  # collapse multiedges
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def get_degree_sequence(G: nx.Graph) -> List[int]:
    """Return degree sequence as a list."""
    return [d for _, d in G.degree()]


def generate_exact_degree_graph(
    degree_sequence: Sequence[int],
    base_seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a simple undirected graph whose degree sequence matches
    `degree_sequence` EXACTLY.

    Strategy:
    1. Try nx.random_degree_sequence_graph (simple graph with given degrees).
    2. If that fails too many times, fall back to nx.havel_hakimi_graph,
       which always constructs a simple realization of a graphical sequence.
    """
    deg_seq = list(map(int, degree_sequence))
    rng = np.random.default_rng(base_seed)

    # Try random_degree_sequence_graph with several random seeds
    for _ in range(RDS_MAX_RESTARTS):
        local_seed = int(rng.integers(0, 2**32 - 1))
        try:
            G = nx.random_degree_sequence_graph(
                deg_seq,
                seed=local_seed,
                tries=RDS_TRIES_PER_CALL,
            )
            return G
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # Retry with a different seed
            continue

    # Last resort: deterministic Havel–Hakimi
    return nx.havel_hakimi_graph(deg_seq)


def save_degree_sequence(deg_seq: Sequence[int], out_dir: Path) -> None:
    """Save original degree sequence as JSON and NPY in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "degree_sequence.json").open("w") as f:
        json.dump({"degree_sequence": list(map(int, deg_seq))}, f, indent=2)

    np.save(out_dir / "degree_sequence.npy", np.array(deg_seq, dtype=int))


def save_adjacency_matrix(G: nx.Graph, path: Path) -> None:
    """
    Save adjacency matrix of G as a CSV file at `path`.
    Nodes are assumed to be indexed 0..n-1; matrix is n x n.
    """
    adj = nx.to_numpy_array(G, dtype=int)
    np.savetxt(path, adj, fmt="%d", delimiter=",")


def process_file(path: Path) -> None:
    """
    For one input adjacency file:
      - load original graph
      - extract degree sequence
      - create ONE folder named after the file stem
      - save original degree sequence there
      - generate NUM_GRAPHS_PER_FILE graphs with the same exact degree sequence
      - save each generated graph as an adjacency matrix CSV in that folder
    """
    print(f"Processing: {path}")

    G = load_graph_from_file(path)
    deg_seq = get_degree_sequence(G)

    # ONE folder per input file
    out_dir = OUTPUT_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original degree sequence (once per input)
    save_degree_sequence(deg_seq, out_dir)

    # Generate graphs with EXACT same degree sequence
    for i in range(NUM_GRAPHS_PER_FILE):
        seed = None if SEED is None else SEED + i

        H = generate_exact_degree_graph(deg_seq, base_seed=seed)

        out_file = out_dir / f"config_model_{i:03d}.csv"
        save_adjacency_matrix(H, out_file)

        print(f"  Generated adjacency matrix {i+1}/{NUM_GRAPHS_PER_FILE}: {out_file}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input dir does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in INPUT_DIR.glob(FILE_PATTERN) if f.is_file())
    if not files:
        raise SystemExit(f"No files found in {INPUT_DIR} matching '{FILE_PATTERN}'")

    print(f"Found {len(files)} adjacency file(s) matching '{FILE_PATTERN}'")
    print(f"Saving output to: {OUTPUT_DIR}")

    for f in files:
        process_file(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
