#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import rewiring_package as rp  # <-- your package

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample vascular plant networks for one habitat."
    )
    p.add_argument("--habitat-code", "-H", help="Override habitat code in CONFIG")

    return p.parse_args()

# ============================================================
# ===================== USER PARAMETERS =======================
# ============================================================

# Folder where the generated adjacency matrices are stored:
# e.g. sim_results/R3_NP/degree_sequences/R3_NP_001_adjacency/
args = parse_args()
type = args.habitat_code + "_NP"

GENERATED_ROOT = Path(f"sim_results/{type}/degree_sequences")

# Folder containing original adjacency matrices:
# e.g. sim_results/R3_NP/networks/R3_NP_001_adjacency.csv
ORIGINAL_ROOT = Path(f"sim_results/{type}/networks")
# Pattern of subfolders to consider:
FOLDER_PATTERN = "*_adjacency"

# Output files
OUTPUT_CSV = Path(f"sim_results/{type}/{type}_measure_summary.csv")
OUTPUT_PKL = Path(f"sim_results/{type}/{type}_measure_summary.pkl")

# Parameters for tau computation
TAU_INITIAL = 1e-8
TAU_TOLERANCE = 1e-8
TAU_REGULAR = False

# ============================================================


def load_adj_csv_as_graph(path: Path) -> nx.Graph:
    """
    Load adjacency CSV into a simple undirected graph.

    Assumes:
      - header row = column labels
      - first column = row labels
      - matrix entries = adjacency values (0/1 or weights)
    """
    df = pd.read_csv(path, index_col=0)
    adj = df.to_numpy()

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Non-square adjacency matrix at {path}")

    G = nx.from_numpy_array(adj)

    if G.is_directed():
        G = G.to_undirected()

    G = nx.Graph(G)  # collapse multiedges if any
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def analyze_graph_from_csv(path: Path) -> tuple[float, float]:
    """
    Load a graph from an adjacency CSV and compute:
      - mean clustering coefficient
      - tau via rp.get_first_bifurcation

    Returns (clustering, tau).
    If tau computation fails, returns tau = np.nan.
    """
    G = load_adj_csv_as_graph(path)

    # Mean clustering
    clustering = nx.average_clustering(G)

    # Tau
    try:
        tau = rp.get_first_bifurcation(
            G=G,
            tau_initial=TAU_INITIAL,
            tolerance=TAU_TOLERANCE,
            regular=TAU_REGULAR,
        )[0]
    except Exception as e:
        print(f"⚠ tau computation failed for {path}: {e}")
        tau = np.nan

    return clustering, float(tau)


def main():
    rows = []  # rows for the final DataFrame

    folders = sorted(GENERATED_ROOT.glob(FOLDER_PATTERN))

    if not folders:
        raise SystemExit(f"No folders found in {GENERATED_ROOT} matching {FOLDER_PATTERN}")

    print(f"Found {len(folders)} network types.\n")

    for folder in folders:
        name = folder.name  # e.g. "R2_NP_001_adjacency"
        base_name = name.replace("_adjacency", "")  # e.g. "R2_NP_001"

        # ----- Original network -----
        original_file = ORIGINAL_ROOT / f"{name}.csv"
        if not original_file.exists():
            print(f"⚠ Original adjacency file not found for {name}, skipping.")
            continue

        orig_clust, orig_tau = analyze_graph_from_csv(original_file)

        # ----- Generated networks -----
        generated_files = sorted(folder.glob("config_model_*.csv"))
        if not generated_files:
            print(f"⚠ No generated matrices found in {folder}, skipping.")
            continue

        gen_clust_list: list[float] = []
        gen_tau_list: list[float] = []

        for gf in generated_files:
            c, t = analyze_graph_from_csv(gf)
            gen_clust_list.append(c)
            gen_tau_list.append(t)

        gen_mean_clust = float(np.mean(gen_clust_list))
        gen_mean_tau = float(np.nanmean(gen_tau_list))  # ignore NaNs if any

        clust_ratio = orig_clust / gen_mean_clust if gen_mean_clust != 0 else np.nan
        tau_ratio = orig_tau / gen_mean_tau if gen_mean_tau != 0 else np.nan

        # ----- Store in table -----
        rows.append({
            "network_type": base_name,
            # clustering
            "original_clustering": orig_clust,
            "generated_mean_clustering": gen_mean_clust,
            "clustering_ratio": clust_ratio,
            "generated_clustering_values": gen_clust_list,
            # tau
            "original_tau": orig_tau,
            "generated_mean_tau": gen_mean_tau,
            "tau_ratio": tau_ratio,
            "generated_tau_values": gen_tau_list,
        })

        print(
            f"{base_name}: "
            f"clust_ratio = {clust_ratio:.4f} "
            f"(orig={orig_clust:.4f}, gen_avg={gen_mean_clust:.4f}); "
            f"tau_ratio = {tau_ratio:.4f} "
            f"(orig={orig_tau:.4e}, gen_avg={gen_mean_tau:.4e})"
        )

    # ----- Build DataFrame -----
    df = pd.DataFrame(rows)

    # ----- Save results -----
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_pickle(OUTPUT_PKL)

    print(f"\nSaved summary to:\n  CSV: {OUTPUT_CSV}\n  PKL: {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
