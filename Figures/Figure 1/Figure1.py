import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == "__main__":    
    rr_df = pd.read_csv(os.path.join(sys.path[0], 'data_analysed/regular_data.csv'))
    er_df = pd.read_csv(os.path.join(sys.path[0], 'data_analysed/erdos_renyi_data.csv'))
    rg_df = pd.read_csv(os.path.join(sys.path[0], 'data_analysed/random_geometric_data.csv'))

    cud_palette = [
    '#0101fd',  # Blue
    '#E69F00',  # Orange
    '#7D7D7D',  # Medium Grey'
    '#ff0101',  # Red
    '#009B77',  # Teal
    ]
    markers = ['s', 'o', '^', 'X', 'p', 'P']

    linewidth = 2.2
    marker_size = 100
    marker_linewidth = 0.3

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    gs.update(wspace=0.4)

    ax1.text(x=0, y=1.05, s='a)', ha='center', transform=ax1.transAxes, fontsize=12)
    ax2.text(x=0, y=1.05, s='b)', ha='center', transform=ax2.transAxes, fontsize=12)
    ax3.text(x=0, y=1.05, s='c)', ha='center', transform=ax3.transAxes, fontsize=12)
    plt.rcParams['font.family'] = 'sans-serif'

    # Axis 1
    ax1.plot( rg_df['degree'], rg_df['mean'], linewidth=linewidth, color=cud_palette[2])
    ax1.errorbar( [10, 20, 30, 40, 50], [ rg_df['mean'][i] for i in [9, 19, 29, 39, 49] ], yerr=([ rg_df['minimum'][i] for i in [9, 19, 29, 39, 49] ], [ rg_df['maximum'][i] for i in [9, 19, 29, 39, 49] ]), fmt='none', ecolor=cud_palette[2], capsize=5 )
    ax1.scatter([10, 20, 30, 40, 50], [ rg_df['mean'][i] for i in [9, 19, 29, 39, 49] ], color=cud_palette[2], marker=markers[2], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)

    # Plot S_k and K_n
    ax1.plot([-5, 60] , np.ones(2)*0.017543859649122816, linewidth=1, color='grey', alpha=0.8, linestyle='--')
    ax1.plot([-5, 60] , np.ones(2)*1, linewidth=1, color='grey', alpha=0.8, linestyle='--')

    ax1.set_xlabel('Average degree', fontsize=12)
    ax1.set_ylabel(r'Mean $\tau_{c}$', fontsize=12)
    ax1.set_title('Random geometric', fontsize=12)
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60], ['0', '10', '20', '30', '40', '50', '60'], fontsize=12)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    ax1.axvline(x=60, color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    ax1.yaxis.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(-2, 61)
    ax1.set_ylim(-0.05, 1.1)

    # Axis 2
    ax2.plot( er_df['degree'], er_df['mean'], linewidth=linewidth, color=cud_palette[0])
    ax2.errorbar( [10, 20, 30, 40, 50], [ er_df['mean'][i] for i in [9, 19, 29, 39, 49] ], yerr=([ er_df['minimum'][i] for i in [9, 19, 29, 39, 49] ], [ er_df['maximum'][i] for i in [9, 19, 29, 39, 49] ]), fmt='none', ecolor=cud_palette[0], capsize=5 )
    ax2.scatter([10, 20, 30, 40, 50], [ er_df['mean'][i] for i in [9, 19, 29, 39, 49] ], color=cud_palette[0], marker=markers[0], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)

    # Plot S_k and K_n
    ax2.plot([-5, 60] , np.ones(2)*0.017543859649122816, linewidth=1, color='grey', alpha=0.8, linestyle='--')
    ax2.plot([-5, 60] , np.ones(2)*1, linewidth=1, color='grey', alpha=0.8, linestyle='--')

    ax2.set_xlabel('Average degree', fontsize=12)
    ax2.set_ylabel(r'Mean $\tau_{c}$', fontsize=12)
    ax2.set_title('Erdős–Rényi', fontsize=12)
    ax2.set_xticks([0, 10, 20, 30, 40, 50, 60], ['0', '10', '20', '30', '40', '50', '60'], fontsize=12)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    ax2.axvline(x=60, color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    ax2.yaxis.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(-2, 61)
    ax2.set_ylim(-0.05, 1.1)

    # Axis 3
    ax3.plot( rr_df['degree'], rr_df['mean'], linewidth=linewidth, color=cud_palette[3])
    ax3.errorbar( [10, 20, 30, 40, 50], [ rr_df['mean'][i] for i in [9, 19, 29, 39, 49] ], yerr=([ rr_df['minimum'][i] for i in [9, 19, 29, 39, 49] ], [ rr_df['maximum'][i] for i in [9, 19, 29, 39, 49] ]), fmt='none', ecolor=cud_palette[3], capsize=5 )
    ax3.scatter([10, 20, 30, 40, 50], [ rr_df['mean'][i] for i in [9, 19, 29, 39, 49] ], color=cud_palette[3], marker=markers[3], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth)

    # Plot S_k and K_n
    ax3.plot([-5, 60] , np.ones(2)*0.017543859649122816, linewidth=1, color='grey', alpha=0.8, linestyle='--')
    ax3.plot([-5, 60] , np.ones(2)*1, linewidth=1, color='grey', alpha=0.8, linestyle='--')
    
    ax3.set_xlabel('Average degree', fontsize=12)
    ax3.set_ylabel(r'Mean $\tau_{c}$', fontsize=12)
    ax3.set_title('Random regular', fontsize=12)
    ax3.set_xticks([0, 10, 20, 30, 40, 50, 60], ['0', '10', '20', '30', '40', '50', '60'], fontsize=12)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
    ax3.axvline(x=60, color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
    ax3.yaxis.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim(-2, 61)
    ax3.set_ylim(-0.05, 1.1)

    ax3.annotate(r'$S_{n}$',
            xy=(60, 0.017543859649122816),  # Point to highlight
            xytext=(62 + 1, 0.017543859649122816),  # Text position
            fontsize=14, color='black', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    ax3.annotate(r'$K_{n}$',
            xy=(60, 1),
            xytext=(62 + 1, 1),
            fontsize=14, color='black', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))


    # Make legend
    # ax2.scatter([], [], label='Random geometric', color=cud_palette[2], marker=markers[2], s=100, edgecolor='black', linewidth=0.3)
    # ax2.scatter([], [], label='Barabási–Albert', color=cud_palette[1], marker=markers[1], s=100, edgecolor='black', linewidth=0.3)
    # ax2.scatter([], [], label='Watts-Strogatz', color=cud_palette[4], marker=markers[4], s=100, edgecolor='black', linewidth=0.3)
    # ax2.scatter([], [], label='Erdős–Rényi', color=cud_palette[0], marker=markers[0], s=100, edgecolor='black', linewidth=0.3)
    # ax2.scatter([], [], label='Random regular', color=cud_palette[3], marker=markers[3], s=100, edgecolor='black', linewidth=0.3)

    # ax2.legend(loc='center left', fontsize=14)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # ax2.set_xlabel('')
    # ax2.set_ylabel('')

    # Save the figure
    save_name = os.path.join(sys.path[0], 'Figure1.pdf')
    plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.show()