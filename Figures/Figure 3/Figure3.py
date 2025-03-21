import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import time
import sys
import os
import pickle
import rewiring_package as rp
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

def load_data(file_name):
    with open(file_name, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict

if __name__ == "__main__":

    markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', 'p', '8', 'h', 's', 'd', '>', '<', '4']
    
    colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown',
    'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow',
    'teal', 'navy', 'maroon', 'gold', 'lime', 'indigo',
    'coral', 'turquoise', 'violet', 'salmon', 'plum', 
    'tan', 'lavender', 'aqua', 'darkgreen', 'darkred',
    'orchid', 'slateblue'
    ]

    markers_ant = markers
    markers_elephant = markers
    markers_parkeet = markers
    markers_cattle = markers

    count_ant = 0
    count_elephant = 0
    count_parkeet = 0
    count_cattle = 0

    alpha = 0.8

    file_name = os.path.join(sys.path[0], f'data_analysed.pkl')
    data = load_data(file_name)

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    # Create the figure
    fig = plt.figure(figsize=(10, 3))

    # Define the grid layout
    gs = GridSpec(3, 5, figure=fig)  # 5 rows, 5 columns

    # Assign subplots to the grid
    ax1 = fig.add_subplot(gs[:, :2])  # ax1 occupies the first 2 columns of all 5 rows
    ax2 = fig.add_subplot(gs[0, 2:])  # ax2 occupies the first row of the last 3 columns

    # Additional axes as described
    ax3 = fig.add_subplot(gs[1:, 2])    # ax3 takes the next two rows of the last 3 columns
    ax4 = fig.add_subplot(gs[1:, 3])    # ax4 takes the fourth row of the last 3 columns
    ax5 = fig.add_subplot(gs[1:, 4])    # ax5 takes the fifth row of the last 3 columns

    fig.subplots_adjust(wspace=0.3)

    for i, network_dict in enumerate(data):
        name = network_dict['Name']
        tau = network_dict['Tau']
        clustering = network_dict['Clustering']
        config_clustering = network_dict['Configuration Clustering']
        config_tau = network_dict['Configuration Tau']

        x = clustering - config_clustering
        y = tau - config_tau

        if name == 'Geese':
            continue

        if name[:3] == 'Ant':
            color = colors[0]
            marker = markers_ant[count_ant]
            count_ant += 1
            ax3.scatter(x, y, label=name, marker=marker, s=90, edgecolors='black', linewidth=0.5, facecolors=color, alpha=alpha)

        if name[:8] == 'Elephant':
            color = colors[1]
            marker = markers_elephant[count_elephant]
            count_elephant += 1
            ax4.scatter(x, y, label=name, marker=marker, s=90, edgecolors='black', linewidth=0.5, facecolors=color, alpha=alpha)

        if name[:7] == 'Parkeet':
            color = colors[2]
            marker = markers_parkeet[count_parkeet]
            count_parkeet += 1
            ax5.scatter(x, y, label=name, marker=marker, s=90, edgecolors='black', linewidth=0.5, facecolors=color, alpha=alpha)

        if name == 'Cattle':
            color = colors[3]
            marker = markers_cattle[count_cattle]
            count_cattle += 1     
            ax5.scatter(x, y, label=name, marker=marker, s=90, edgecolors='black', linewidth=0.5, facecolors=color, alpha=alpha) 

        # ax1.scatter(x, y, label=name, marker='s', s=100, edgecolors='black', linewidth=0.5)
        ax1.scatter(x, y, label=name, marker=marker, s=100, edgecolors='black', linewidth=0.5, facecolors=color, alpha=alpha)
        # ax2.scatter([], [], label=name, marker=marker, s=60, edgecolors='black', linewidth=0.5, facecolors=color, alpha=1)
        # ax.annotate(name, (config_clustering, config_tau))

    ax2.scatter([], [], label='Ants', marker='o', s=60, edgecolors='black', linewidth=0.5, facecolors='blue', alpha=1)
    ax2.scatter([], [], label='Elephantseals', marker='o', s=60, edgecolors='black', linewidth=0.5, facecolors='orange', alpha=1)
    ax2.scatter([], [], label='Parkeet', marker='o', s=60, edgecolors='black', linewidth=0.5, facecolors='green', alpha=1)
    ax2.scatter([], [], label='Cattle', marker='o', s=60, edgecolors='black', linewidth=0.5, facecolors='red', alpha=1)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    rect = plt.Rectangle(
        (0.12, 0.01), 0.36, 0.53, transform=ax1.transAxes,
        linewidth=1, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax1.add_patch(rect)

    rect = plt.Rectangle(
        (0.125, 0.58), 0.26, 0.38, transform=ax1.transAxes,
        linewidth=1, edgecolor='orange', facecolor='none', linestyle='--'
    )
    ax1.add_patch(rect)

    rect = plt.Rectangle(
        (0.7, 0.44), 0.22, 0.31, transform=ax1.transAxes,
        linewidth=1, edgecolor='green', facecolor='none', linestyle='--'
    )
    ax1.add_patch(rect)

    ax1.set_xlabel(r'$\Delta C$', fontsize=12)
    ax1.set_ylabel(r'$\Delta \tau$', fontsize=12)
    ax1.set_ylim(-0.42, 0.1)

    ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], [ '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6', '0.8'], fontsize=11)
    ax1.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1], ['-0.4', '-0.3', '-0.2', '-0.1', '0', '0.1'], fontsize=11)

    ax2.legend(ncol=4, loc='center', fontsize=9, frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])

    # ax3.set_xlabel(r'$\Delta C$', fontsize=12)
    # ax3.set_ylabel(r'$\Delta \tau$', fontsize=12)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])
    for s in ['top', 'right', 'left', 'bottom']:
        ax3.spines[s].set_color('blue')
        ax3.spines[s].set_linestyle('--')

    ax3.set_xlim(-0.42, 0.05)
    ax3.set_ylim(-0.41, -0.14)

    # ax4.set_xlabel(r'$\Delta C$', fontsize=12)
    # ax4.set_ylabel(r'$\Delta \tau$', fontsize=12)
    ax4.set_xticks([], [])
    ax4.set_yticks([], [])
    for s in ['top', 'right', 'left', 'bottom']:
        ax4.spines[s].set_color('orange')
        ax4.spines[s].set_linestyle('--')

    ax4.set_xlim(-0.4, -0.05)
    ax4.set_ylim(-0.11, 0.06)

    # ax5.set_xlabel(r'$\Delta C$', fontsize=12)
    # ax5.set_ylabel(r'$\Delta \tau$', fontsize=12)
    ax5.set_xticks([], [])
    ax5.set_yticks([], [])
    for s in ['top', 'right', 'left', 'bottom']:
        ax5.spines[s].set_color('green')
        ax5.spines[s].set_linestyle('--')

    ax5.set_xlim(0.41, 0.66)
    ax5.set_ylim(-0.18, -0.04)

    fig.text(0.92, 0.35, r'$\Delta \tau$', va='center', rotation='vertical', fontsize=12)
    fig.text(0.67, 0.03, r'$\Delta C$', va='center', fontsize=12)

    ax4.axhline(y=-0.12, xmin=-1.3, xmax=2.3, color='black', linestyle='-', linewidth=1, clip_on=False)
    ax5.axvline(x=0.68, ymin=0.01, ymax=1, color='black', linestyle='-', linewidth=1, clip_on=False)

    ax1.text(-0.12, 1.12, 'a', transform=ax1.transAxes, fontsize=14, va='top', ha='right')
    ax3.text(-0.1, 1.1, 'b', transform=ax3.transAxes, fontsize=14, va='top', ha='right')
    ax4.text(-0.1, 1.1, 'c', transform=ax4.transAxes, fontsize=14, va='top', ha='right')
    ax5.text(-0.1, 1.1, 'd', transform=ax5.transAxes, fontsize=14, va='top', ha='right')








    
    # ax.set_title(r'Configuration Clustering Coefficient vs. $\tau$ (No data accept degree sequence)')
    save_name = os.path.join(sys.path[0], 'Figure5.pdf')
    plt.savefig(save_name, format='pdf', bbox_inches='tight', dpi=300, transparent=True)
    save_name = os.path.join(sys.path[0], 'Figure5.png')
    plt.savefig(save_name, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()