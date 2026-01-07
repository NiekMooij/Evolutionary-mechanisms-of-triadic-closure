import ast
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# ===================== USER PARAMETERS =======================
# ============================================================

SUMMARY_CSV = Path("sim_results/R3_NP/measure_summary.csv")
OUTPUT_PLOT = Path("sim_results/R3_NP/clustering_tau_with_error.png")

POINT_SIZE = 28          # marker area in points^2
ALPHA = 0.8
FIGSIZE = (5.0, 2.3)     # ~two-column Nature-style width

CI_Z = 1.96              # ~95% CI (not used in plot, but kept for reference)
DPI = 900                # high resolution for publication

# Nature-like muted colour palette (similar to seaborn-deep / Nature figures)
COLOR_PANEL1 = "#4C72B0"  # muted blue
COLOR_PANEL2 = "#DD8452"  # muted orange
ERROR_COLOR = "0.5"       # medium grey for reference lines

# ============================================================
# ==================== GLOBAL STYLE SETUP ====================
# ============================================================

plt.rcParams.update({
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    # Axes / spines
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    # Ticks
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # Save
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    # Use all spines by default; we'll style them per-axis
    "axes.spines.top": True,
    "axes.spines.right": True,
})


# ============================================================
# ======================= FUNCTIONS ==========================
# ============================================================

def parse_list_column(col: pd.Series):
    """
    Parse a column that stores Python-like list strings, e.g.
    '[0.1, 0.2, 0.3]' -> [0.1, 0.2, 0.3]
    """
    return col.apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])


def main():
    # ---- Load table ----
    df = pd.read_csv(SUMMARY_CSV)

    required_cols = {
        "original_clustering", "generated_mean_clustering",
        "original_tau", "generated_mean_tau",
        "clustering_ratio", "tau_ratio",
        "generated_clustering_values", "generated_tau_values",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # ---- Parse list columns back into Python lists ----
    df["gen_clust_list"] = parse_list_column(df["generated_clustering_values"])
    df["gen_tau_list"] = parse_list_column(df["generated_tau_values"])

    # ---- Compute differences ----
    df["clustering_diff"] = df["original_clustering"] - df["generated_mean_clustering"]
    df["tau_diff"] = df["original_tau"] - df["generated_mean_tau"]

    # ---- Compute 95% CI half-widths for generated means (not plotted) ----
    clust_xerr = []
    tau_yerr = []

    for _, row in df.iterrows():
        cl_vals = np.array(row["gen_clust_list"], dtype=float)
        tau_vals = np.array(row["gen_tau_list"], dtype=float)

        # clustering
        if cl_vals.size > 1:
            cl_std = np.std(cl_vals, ddof=1)
            cl_se = cl_std / np.sqrt(cl_vals.size)
            cl_ci = CI_Z * cl_se
        else:
            cl_ci = 0.0

        # tau (ignore NaNs)
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

    # ============================================================
    # ========================= PLOTTING =========================
    # ============================================================

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    ax1, ax2 = axes

    # Ensure all four spines are on and styled
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    # ------------------------------------------------------------
    # Panel a: Ratios
    # ------------------------------------------------------------
    ax1.scatter(
        df["clustering_ratio"],
        df["tau_ratio"],
        s=POINT_SIZE,
        facecolor=COLOR_PANEL1,
        edgecolor="white",      # white marker boundary
        linewidth=0.6,
        alpha=ALPHA,
    )

    ax1.axvline(1, color=ERROR_COLOR, linestyle="--", linewidth=0.8, zorder=0)
    ax1.axhline(1, color=ERROR_COLOR, linestyle="--", linewidth=0.8, zorder=0)

    # Improved axis labels
    ax1.set_xlabel("Relative clustering (orig / gen)")
    ax1.set_ylabel(r"Relative $\tau$ (orig / gen)")

    # Panel label (upper-left corner)
    ax1.text(
        0.98, 0.98, "a",
        transform=ax1.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
    )

    # ------------------------------------------------------------
    # Panel b: Differences WITHOUT error bars
    # ------------------------------------------------------------
    ax2.scatter(
        df["clustering_diff"],
        df["tau_diff"],
        s=POINT_SIZE,
        facecolor=COLOR_PANEL2,
        edgecolor="white",      # white marker boundary
        linewidth=0.6,
        alpha=ALPHA,
    )

    ax2.axvline(0, color=ERROR_COLOR, linestyle="--", linewidth=0.8, zorder=0)
    ax2.axhline(0, color=ERROR_COLOR, linestyle="--", linewidth=0.8, zorder=0)

    # Improved axis labels
    ax2.set_xlabel("Δ clustering (orig − gen)")
    ax2.set_ylabel(r"Δ $\tau$ (orig − gen)")

    # Panel label (upper-left corner)
    ax2.text(
        0.98, 0.98, "b",
        transform=ax2.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="top",
    )

    # ------------------------------------------------------------
    # Final layout & save
    # ------------------------------------------------------------
    plt.tight_layout(w_pad=2.2)
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(OUTPUT_PLOT)
    pdf_path = OUTPUT_PLOT.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()

    print(f"Saved Nature-style coloured plot to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
