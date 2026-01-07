#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure: τ_c vs clustering / assortativity across network models.

Notes:
- Raw values
- Shared y-label on the left
- Shared x-label per row
- One shared legend at the top
- Panel letters inside axes (upper-left)
- Panel f (Regular assortativity) shown as 'NA'
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import rewiring_package as rp


# ----------------------------
# Styling
# ----------------------------
def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": True,
        }
    )


# ----------------------------
# Helpers
# ----------------------------
def load_data(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _finite_xy(x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray] = None):
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        mask &= np.isfinite(yerr)
        return x[mask], y[mask], yerr[mask], mask
    return x[mask], y[mask], mask


def _panel_letter(ax: mpl.axes.Axes, letter: str) -> None:
    ax.text(
        0.09,
        0.97,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.0, pad=0.2),
    )


def _annotate_r2(ax: mpl.axes.Axes, r2: float) -> None:
    if np.isfinite(r2):
        ax.text(
            0.98,
            0.05,
            rf"$R^2={r2:.3f}$",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="right",
        )


def _plot_baseline_guides(ax: mpl.axes.Axes, x0: float, y0: float) -> None:
    if np.isfinite(x0):
        ax.axvline(x0, color="0.5", linestyle="--", linewidth=0.8, zorder=400)
    if np.isfinite(y0):
        ax.axhline(y0, color="0.5", linestyle="--", linewidth=0.8, zorder=400)
    if np.isfinite(x0) and np.isfinite(y0):
        ax.scatter(
            [x0],
            [y0],
            color="#6A1B9A",
            s=140,
            marker="*",
            zorder=1000,
            edgecolor="black",
            linewidth=0.8,
        )


# ----------------------------
# Main
# ----------------------------
def save_figures(output_dir: str) -> None:
    set_nature_style()

    fig_w, fig_h = 7.2, 3.6
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.90,
        bottom=0.16,
        wspace=0.35,
        hspace=0.4,
    )

    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        _panel_letter(ax, chr(ord("a") + i))
        ax.minorticks_on()
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.margins(x=0.05, y=0.08)

    network_info = [
        ("random_regular", "Regular"),
        ("erdos_renyi", "Poisson"),
        ("random_geometric", "Geometric"),
        ("watts_strogatz", "Watts–Strogatz"),
        ("barabasi_albert", "Barabási–Albert"),
    ]

    clustering_marker = dict(
        fmt="o",
        markeredgecolor="black",
        markerfacecolor="#1f77b4",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        markersize=6,
        zorder=600,
    )
    assort_marker = dict(
        fmt="^",
        markeredgecolor="black",
        markerfacecolor="#ff7f0e",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        markersize=7,
        zorder=600,
    )
    fit_line_style = dict(color="#d62728", linewidth=1.6, zorder=500)

    for idx, (net_key, title) in enumerate(network_info):
        ax_c = axes[0, idx]
        ax_a = axes[1, idx]

        ax_c.set_title(title, pad=2)

        network_path = os.path.join(output_dir, f"networks/{net_key}.npy")
        G = nx.from_numpy_array(np.load(network_path))

        tau_orig = rp.get_first_bifurcation(
            G=G, tau_initial=1e-8, tolerance=1e-8, regular=False
        )[0]
        c_orig = nx.average_clustering(G)
        ass_orig = nx.degree_assortativity_coefficient(G)

        data_path = os.path.join(output_dir, f"data/data_dict_{net_key}.pkl")
        ddict = load_data(data_path)

        # --- clustering ---
        d_c = ddict["clustering"]
        x_c, y_c, yerr_c, _ = _finite_xy(
            np.array(d_c["bin_centers"], float),
            np.array(d_c["bin_means"], float),
            3 * np.array(d_c["bin_std_error"], float),
        )
        if x_c.size:
            ax_c.errorbar(x_c, y_c, yerr=yerr_c, **clustering_marker)
            if net_key != "barabasi_albert":
                m = float(d_c["slope"])
                b = float(d_c["intercept"])
                xs = np.linspace(np.nanmin(x_c), np.nanmax(x_c), 200)
                ax_c.plot(xs, m * xs + b, **fit_line_style)
                _annotate_r2(ax_c, float(d_c.get("r_squared", np.nan)))
            _plot_baseline_guides(ax_c, c_orig, tau_orig)

        # --- assortativity ---
        if net_key == "random_regular":
            ax_a.text(
                0.5,
                0.5,
                "NA",
                transform=ax_a.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="center",
            )
            ax_a.set_xticks([])
            ax_a.set_yticks([])
        else:
            d_a = ddict["assortativity"]
            x_a, y_a, yerr_a, _ = _finite_xy(
                np.array(d_a["bin_centers"], float),
                np.array(d_a["bin_means"], float),
                3 * np.array(d_a["bin_std_error"], float),
            )
            if x_a.size:
                ax_a.errorbar(x_a, y_a, yerr=yerr_a, **assort_marker)
                m = float(d_a["slope"])
                b = float(d_a["intercept"])
                xs = np.linspace(np.nanmin(x_a), np.nanmax(x_a), 200)
                ax_a.plot(xs, m * xs + b, **fit_line_style)
                _annotate_r2(ax_a, float(d_a.get("r_squared", np.nan)))
                _plot_baseline_guides(ax_a, ass_orig, tau_orig)

    # Shared y-label
    fig.text(
        0.025,
        0.5,
        r"Critical coupling, $\tau_c$",
        va="center",
        ha="center",
        rotation="vertical",
    )

    # Shared x-labels per row
    fig.canvas.draw()
    top_row_y0 = axes[0, 0].get_position().y0
    bottom_row_y1 = axes[1, 0].get_position().y1
    between_rows_y = (bottom_row_y1 + top_row_y0) / 2.0
    fig.text(0.5, between_rows_y, "Clustering coefficient", ha="center", va="center")
    fig.text(0.5, 0.06, "Degree assortativity", ha="center", va="center")

    # Shared legend
    labels = ["Data (±3 SE)", "Linear fit", "Baseline (original network)"]
    handles = [
        plt.Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            label=labels[0],
        ),
        plt.Line2D([], [], color="#d62728", label=labels[1]),
        plt.Line2D(
            [],
            [],
            color="none",
            marker="*",
            markerfacecolor="#6A1B9A",
            markeredgecolor="black",
            markersize=10,
            label=labels[2],
        ),
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Save
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    base = "fig_tau_vs_clustering_assortativity"
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(fig_dir, f"{base}.{ext}"), format=ext)

    plt.show()


def main():
    save_figures(sys.path[0])


if __name__ == "__main__":
    main()
