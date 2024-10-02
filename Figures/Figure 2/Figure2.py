import networkx as nx
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == "__main__":

    er_tau_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/er_tau_arr_mean.npy'))
    er_clustering_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/er_clustering_arr_mean.npy'))

    ba_tau_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/ba_tau_arr_mean.npy'))
    ba_clustering_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/ba_clustering_arr_mean.npy'))

    rg_tau_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/rg_tau_arr_mean.npy'))
    rg_clustering_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/rg_clustering_arr_mean.npy'))

    rr_tau_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/rr_tau_arr_mean.npy'))
    rr_clustering_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/rr_clustering_arr_mean.npy'))

    ws_tau_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/ws_tau_arr_mean.npy'))
    ws_clustering_arr_mean = np.load(os.path.join(sys.path[0], 'data_analysed/ws_clustering_arr_mean.npy'))

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.text(x=0, y=1.05, s='a)', ha='center', transform=ax1.transAxes, fontsize=12)
    ax2.text(x=0, y=1.05, s='b)', ha='center', transform=ax2.transAxes, fontsize=12)
    plt.rcParams['font.family'] = 'sans-serif'

    cud_palette = [
    '#0101fd',  # Blue
    '#E69F00',  # Orange
    '#7D7D7D',  # Medium Grey'
    '#ff0101',  # Red
    '#009B77',  # Teal
    ]
    markers = ['s', 'o', '^', 'X', 'p', 'P']

    x = np.arange(1001)
    linewidth = 1.2
    marker_size = 80
    marker_linewidth = 0.3
    
    # Axis 1
    ax1.plot(x[1:], er_tau_arr_mean[1:], linewidth=linewidth, color=cud_palette[0])
    ax1.scatter([1, 10e0, 10e1, 10e2], [ er_tau_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[0], marker=markers[0], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax1.plot(x[1:], ba_tau_arr_mean[1:], linewidth=linewidth, color=cud_palette[1])
    ax1.scatter([1, 10e0, 10e1, 10e2], [ ba_tau_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[1], marker=markers[1], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax1.plot(x[1:], rg_tau_arr_mean[1:], linewidth=linewidth, color=cud_palette[2])
    ax1.scatter([1, 10e0, 10e1, 10e2], [ rg_tau_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[2], marker=markers[2], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax1.plot(x[1:], rr_tau_arr_mean[1:], linewidth=linewidth, color=cud_palette[3])
    ax1.scatter([1, 10e0, 10e1, 10e2], [ rr_tau_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[3], marker=markers[3], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax1.plot(x[1:], ws_tau_arr_mean[1:], linewidth=linewidth, color=cud_palette[4])
    ax1.scatter([1, 10e0, 10e1, 10e2], [ ws_tau_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[4], marker=markers[4], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)

    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_title(r'Collapse measure $\tau_{c}$', pad=15)
    ax1.set_xticks([10e0, 10e1, 10e2, 10e3], ['10e0', '10e1', '10e2', '10e3'], fontsize=12)
    ax1.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], ['0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40'], fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    ax1.yaxis.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Axis 2
    ax2.plot(x[1:], er_clustering_arr_mean[1:], linewidth=linewidth, color=cud_palette[0])
    ax2.scatter([1, 10e0, 10e1, 10e2], [ er_clustering_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[0], marker=markers[0], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax2.plot(x[1:], ba_clustering_arr_mean[1:], linewidth=linewidth, color=cud_palette[1])
    ax2.scatter([1, 10e0, 10e1, 10e2], [ ba_clustering_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[1], marker=markers[1], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax2.plot(x[1:], rg_clustering_arr_mean[1:], linewidth=linewidth, color=cud_palette[2])
    ax2.scatter([1, 10e0, 10e1, 10e2], [ rg_clustering_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[2], marker=markers[2], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax2.plot(x[1:], rr_clustering_arr_mean[1:], linewidth=linewidth, color=cud_palette[3])
    ax2.scatter([1, 10e0, 10e1, 10e2], [ rr_clustering_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[3], marker=markers[3], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)
    ax2.plot(x[1:], ws_clustering_arr_mean[1:], linewidth=linewidth, color=cud_palette[4])
    ax2.scatter([1, 10e0, 10e1, 10e2], [ ws_clustering_arr_mean[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], color=cud_palette[4], marker=markers[4], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)

    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_title(r'Clustering coefficient', pad=15)
    ax2.set_xticks([10e0, 10e1, 10e2, 10e3], ['10e0', '10e1', '10e2', '10e3'], fontsize=12)
    ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontsize=12)
    ax2.set_xscale('log')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    ax2.yaxis.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Make legend
    ax3.scatter([], [], label='Random geometric', color=cud_palette[2], marker=markers[2], s=100, edgecolor='black', linewidth=0.3)
    ax3.scatter([], [], label='Barabási–Albert', color=cud_palette[1], marker=markers[1], s=100, edgecolor='black', linewidth=0.3)
    ax3.scatter([], [], label='Watts-Strogatz', color=cud_palette[4], marker=markers[4], s=100, edgecolor='black', linewidth=0.3)
    ax3.scatter([], [], label='Erdős–Rényi', color=cud_palette[0], marker=markers[0], s=100, edgecolor='black', linewidth=0.3)
    ax3.scatter([], [], label='Random regular', color=cud_palette[3], marker=markers[3], s=100, edgecolor='black', linewidth=0.3)

    ax3.legend(loc='center left', fontsize=14)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('')
    ax3.set_ylabel('')

    # Save the figure
    save_name = os.path.join(sys.path[0], 'Figure2.pdf')
    plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.show()