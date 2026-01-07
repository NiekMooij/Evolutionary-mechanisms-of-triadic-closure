#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure: τ_c vs clustering / assortativity across network models.

Features:
- Raw values (no transforms)
- Shared y-label on the left margin
- Shared x-label per row
- One shared legend at the top
- Panel letters inside axes (upper-left)
- Two-decimal y-ticks
- Bottom-left panel (Regular assortativity) shown as 'NA'
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
# Plot style
# ----------------------------
def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            # Lines & markers
            "lines.linewidth": 1.5,
            # Axes/spines
            "axes.linewidth": 0.8,
            "axes.spines.right": True,
            "axes.spines.top": True,
            # Ticks
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
            # Saving
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": True,
        }
    )


# ----------------------------
# IO
# ----------------------------
def load_data(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------
# Plot utilities
# ----------------------------
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
# Figure builder
# ----------------------------
def save_figures(output_dir: str) -> None:
    """
    Create and save plots:
      - Top row: clustering coefficient (x) vs τ_c (y)
      - Bottom row: degree assortativity (x) vs τ_c (y)

    Regression lines use y = m x + b from the stored fit parameters.
    """
    set_plot_style()

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
        ax_clust = axes[0, idx]
        ax_assort = axes[1, idx]

        ax_clust.set_title(title, pad=2)

        network_path = os.path.join(output_dir, f"networks/{net_key}.npy")
        G = nx.from_numpy_array(np.load(network_path))

        tau_original = rp.get_first_bifurcation(
            G=G, tau_initial=1e-8, tolerance=1e-8, regular=False
        )[0]
        c_original = nx.average_clustering(G)
        ass_original = nx.degree_assortativity_coefficient(G)

        data_path = os.path.join(output_dir, f"data/data_dict_{net_key}.pkl")
        ddict = load_data(data_path)

        # ---- clustering vs tau ----
        data_c = ddict["clustering"]
        x_raw_c = np.array(data_c["bin_centers"], dtype=float)
        y_raw_c = np.array(data_c["bin_means"], dtype=float)
        y_err_c = 3.0 * np.array(data_c["bin_std_error"], dtype=float)

        x_c, y_c, yerr_c, _ = _finite_xy(x_raw_c, y_raw_c, y_err_c)
        if x_c.size:
            ax_clust.errorbar(x_c, y_c, yerr=yerr_c, **clustering_marker)

            m_raw_c = data_c.get("slope", None)
            b_raw_c = data_c.get("intercept", None)
            r2_raw_c = data_c.get("r_squared", None)

            if (net_key != "barabasi_albert") and (m_raw_c is not None) and (b_raw_c is not None):
                m_raw_c = float(m_raw_c)
                b_raw_c = float(b_raw_c)
                xspan = np.linspace(np.nanmin(x_c), np.nanmax(x_c), 200)
                ax_clust.plot(xspan, m_raw_c * xspan + b_raw_c, **fit_line_style)
                if r2_raw_c is not None:
                    _annotate_r2(ax_clust, float(r2_raw_c))

            _plot_baseline_guides(ax_clust, c_original, tau_original)

        # ---- assortativity vs tau ----
        if net_key == "random_regular":
            ax_assort.text(
                0.5,
                0.5,
                "NA",
                transform=ax_assort.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="center",
            )
            ax_assort.set_xticks([])
            ax_assort.set_yticks([])
        else:
            a_key = "assortativity_standard" if "assortativity_standard" in ddict else "assortativity"
            data_a = ddict[a_key]

            x_raw_a = np.array(data_a["bin_centers"], dtype=float)
            y_raw_a = np.array(data_a["bin_means"], dtype=float)
            y_err_a = 3.0 * np.array(data_a["bin_std_error"], dtype=float)

            x_a, y_a, yerr_a, _ = _finite_xy(x_raw_a, y_raw_a, y_err_a)
            if x_a.size:
                ax_assort.errorbar(x_a, y_a, yerr=yerr_a, **assort_marker)

                m_raw_a = data_a.get("slope", None)
                b_raw_a = data_a.get("intercept", None)
                r2_raw_a = data_a.get("r_squared", None)

                if (m_raw_a is not None) and (b_raw_a is not None):
                    m_raw_a = float(m_raw_a)
                    b_raw_a = float(b_raw_a)
                    xspan_a = np.linspace(np.nanmin(x_a), np.nanmax(x_a), 200)
                    ax_assort.plot(xspan_a, m_raw_a * xspan_a + b_raw_a, **fit_line_style)
                    if r2_raw_a is not None:
                        _annotate_r2(ax_assort, float(r2_raw_a))

                _plot_baseline_guides(ax_assort, ass_original, tau_original)

    # ---- shared labels & legend ----
    fig.text(
        0.025,
        0.5,
        r"Critical coupling, $\tau_c$",
        va="center",
        ha="center",
        rotation="vertical",
    )

    fig.canvas.draw()
    top_row_y0 = axes[0, 0].get_position().y0
    bottom_row_y1 = axes[1, 0].get_position().y1
    between_rows_y = (bottom_row_y1 + top_row_y0) / 2.0

    fig.text(0.5, between_rows_y, "Clustering coefficient", ha="center", va="center")
    fig.text(0.5, 0.06, "Degree assortativity", ha="center", va="center")

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

    # ---- save ----
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    base = "clustering_assortativity_vs_tau_style"
    for ext in ("pdf", "png"):
        out_path = os.path.join(fig_dir, f"{base}.{ext}")
        plt.savefig(out_path, format=ext)

    plt.show()


def main():
    save_figures(sys.path[0])


if __name__ == "__main__":
    main()
