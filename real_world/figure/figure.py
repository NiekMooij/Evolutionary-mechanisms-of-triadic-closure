#!/usr/bin/env python3
"""
Three-panel figure for publication (Nature-style):

Panel a: Histogram of interaction strengths (standard Matplotlib blue)
Panel b: Realized network from adjacency matrix (nodes coloured & sized by degree)
Panel c: ΔC vs Δτ scatter for multiple environments

Outputs:
  - combined_figure.png
  - combined_figure.pdf
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable

# ============================================================
# ===================== USER PARAMETERS =======================
# ============================================================

# Files for panels a & b
ADJACENCY_FNAME = "R1_NP_001_adjacency.csv"
INTERACTION_FNAME = "R1_NP_001_interaction_full.csv"

# Map variable labels to CSV paths (panel c)
DATA_FILES = {
    "N1": Path("data/N1_NP_measure_summary.csv"),
    "Q2": Path("data/Q2_NP_measure_summary.csv"),
    "Q5": Path("data/Q5_NP_measure_summary.csv"),
    "R1": Path("data/R1_NP_measure_summary.csv"),
    "R2": Path("data/R2_NP_measure_summary.csv"),
    "R3": Path("data/R3_NP_measure_summary.csv"),
    "S9": Path("data/S9_NP_measure_summary.csv"),
}

# Output for the combined 3-panel figure
COMBINED_OUTPUT = Path("combined_figure.png")

POINT_SIZE = 18         # marker area for panel c
ALPHA = 0.85
CI_Z = 1.96
DPI = 600

# Colour palette for panel c
PALETTE = [
    "#4C72B0",  # N1
    "#DD8452",  # Q2
    "#55A868",  # Q5
    "#C44E52",  # R1
    "#8172B3",  # R2
    "#937860",  # R3
    "#64B5CD",  # S9
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

ERROR_COLOR = "0.6"

# Histogram colour: standard Matplotlib blue
HIST_COLOR = "#1f77b4"
HIST_EDGE_COLOR = "white"

# Network edge colour
NET_EDGE_COLOR = "0.6"

# ============================================================
# ==================== GLOBAL STYLE SETUP ====================
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

# ============================================================
# ================= PANEL a: HISTOGRAM =======================
# ============================================================

def plot_interaction_histogram(ax, fname: str) -> None:
    df = pd.read_csv(fname)
    numeric_df = df.select_dtypes(include="number")
    values = numeric_df.to_numpy().ravel()
    values = values[~np.isnan(values)]

    # Draw histogram and get patches
    n, bins, patches = ax.hist(
        values,
        bins=25,
        color=HIST_COLOR,
        edgecolor=HIST_EDGE_COLOR,
        linewidth=0.2,
    )

    # Give all bins with centers < 0.9 a lower opacity
    for left, right, patch in zip(bins[:-1], bins[1:], patches):
        center = 0.5 * (left + right)
        if center < 0.9:
            patch.set_alpha(0.35)
        else:
            patch.set_alpha(1.0)

    ax.set_xlabel("Interaction strength")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)


# ============================================================
# ================= PANEL b: NETWORK =========================
# ============================================================

def plot_realized_network(ax, fname: str) -> None:
    """
    Plot realized network (from adjacency matrix) on the given axes.

    - Layout: Kamada–Kawai (spreads in 2D) + strong jitter to avoid “line” artefacts.
    - Nodes coloured & sized by degree (viridis).
    """
    df = pd.read_csv(fname, header=None)

    labels = df.iloc[0, 1:].tolist()
    matrix = df.iloc[1:, 1:].astype(float).to_numpy()

    G = nx.Graph()
    for label in labels:
        G.add_node(label)

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i, j]
            if w != 0:
                G.add_edge(labels[i], labels[j], weight=w)

    if G.number_of_nodes() == 0:
        ax.set_axis_off()
        return

    # Kamada–Kawai layout: good global spacing
    pos = nx.kamada_kawai_layout(G)

    # Strong jitter to make nodes clearly 2D, not collinear
    jitter = 0.25
    for node in pos:
        pos[node] = pos[node] + jitter * np.random.randn(2)

    # Degree-based colour and size
    nodes = list(G.nodes())
    degrees = np.array([G.degree(node) for node in nodes], dtype=float)
    if degrees.size == 0:
        ax.set_axis_off()
        return

    dmin, dmax = degrees.min(), degrees.max()
    if dmax == dmin:
        dmax = dmin + 1.0  # avoid zero-range

    norm = plt.Normalize(vmin=dmin, vmax=dmax)
    cmap = plt.cm.viridis
    node_colors = cmap(norm(degrees))

    # Node sizes: modest, narrow range to keep overlap low
    node_sizes = 12 + 24 * (degrees - dmin) / (dmax - dmin)

    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=0.35, width=0.35, edge_color=NET_EDGE_COLOR
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.5,
        edgecolors="black",
    )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# ================= PANEL c: ΔC vs Δτ ========================
# ============================================================

def parse_list_column(col: pd.Series):
    return col.apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])


def load_and_process(label: str, csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV for {label}: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "original_clustering", "generated_mean_clustering",
        "original_tau", "generated_mean_tau",
        "clustering_ratio", "tau_ratio",
        "generated_clustering_values", "generated_tau_values",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} is missing columns: {missing}")

    df["gen_clust_list"] = parse_list_column(df["generated_clustering_values"])
    df["gen_tau_list"] = parse_list_column(df["generated_tau_values"])

    df["clustering_diff"] = df["original_clustering"] - df["generated_mean_clustering"]
    df["tau_diff"] = df["original_tau"] - df["generated_mean_tau"]

    clust_xerr = []
    tau_yerr = []

    for _, row in df.iterrows():
        cl_vals = np.asarray(row["gen_clust_list"], dtype=float)
        tau_vals = np.asarray(row["gen_tau_list"], dtype=float)

        if cl_vals.size > 1:
            cl_std = np.std(cl_vals, ddof=1)
            cl_se = cl_std / np.sqrt(cl_vals.size)
            cl_ci = CI_Z * cl_se
        else:
            cl_ci = 0.0

        tau_vals = tau_vals[~np.isnan(tau_vals)]
        if tau_vals.size > 1:
            tau_std = np.std(tau_vals, ddof=1)
            tau_se = tau_std / np.sqrt(tau_vals.size)
            tau_ci = CI_Z * tau_se
        else:
            tau_ci = 0.0

        clust_xerr.append(cl_ci)
        tau_yerr.append(tau_ci)

    df["clust_ci"] = clust_xerr
    df["tau_ci"] = tau_yerr
    df["variable"] = label
    return df


def plot_clustering_tau_panel(ax):
    dfs = []
    var_colors = {}
    var_markers = {}

    for (label, path), color, marker in zip(DATA_FILES.items(), PALETTE, MARKERS):
        df_label = load_and_process(label, path)
        df_label["color"] = color
        dfs.append(df_label)
        var_colors[label] = color
        var_markers[label] = marker

    df_all = pd.concat(dfs, ignore_index=True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    for label, df_sub in df_all.groupby("variable"):
        color = var_colors[label]
        marker = var_markers[label]
        ax.scatter(
            df_sub["clustering_diff"],
            df_sub["tau_diff"],
            s=POINT_SIZE,
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=ALPHA,
            marker=marker,
            label=label,
        )

    ax.axvline(0, color=ERROR_COLOR, linestyle="--", linewidth=0.5, zorder=0)
    ax.axhline(0, color=ERROR_COLOR, linestyle="--", linewidth=0.5, zorder=0)

    ax.set_xlabel(r"$\Delta C$")
    ax.set_ylabel(r"$\Delta \tau_c$")
    ax.set_xlim(-0.06, 0.6)
    ax.set_ylim(-0.02, 0.14)

    # Legend in bottom-left region, lifted above zero lines
    ax.legend(
        title=None,
        frameon=True,
        fontsize=5.5,
        handletextpad=0.4,
        borderpad=0.2,
        loc="lower left",
        bbox_to_anchor=(0.09, 0.124),
        ncol=2,
        edgecolor="black",
        fancybox=False,
        framealpha=0.9
    )

    ax.tick_params(axis="both", pad=2)


# ============================================================
# =================== PANEL LABEL HELPER =====================
# ============================================================

def add_panel_label(ax, label):
    # Panel label inside axes, upper-left corner (same for all panels)
    ax.text(
        0.035, 0.98, label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


# ============================================================
# =========================== MAIN ===========================
# ============================================================

def main():
    # Three equal-sized panels in one row
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.4),
        constrained_layout=False,
    )

    plt.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.22,
        wspace=0.35,
    )

    # Panel a
    plot_interaction_histogram(ax1, INTERACTION_FNAME)
    add_panel_label(ax1, "a")

    # Panel b (same label position as a and c)
    plot_realized_network(ax2, ADJACENCY_FNAME)
    add_panel_label(ax2, "b")

    # Panel c
    plot_clustering_tau_panel(ax3)
    add_panel_label(ax3, "c")

    COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = COMBINED_OUTPUT.with_suffix(".pdf")
    fig.savefig(COMBINED_OUTPUT, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved combined figure to: {COMBINED_OUTPUT} and {pdf_path}")


if __name__ == "__main__":
    main()
