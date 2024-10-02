import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

# Create the figure
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(15, 20)

ax_left = fig.add_subplot(gs[:, :6])
ax_middle_up = fig.add_subplot(gs[:6, 7:13])
ax_middle_down = fig.add_subplot(gs[9:15, 7:13])
ax_right_1 = fig.add_subplot(gs[0:5, 14:17])
ax_right_2 = fig.add_subplot(gs[0:5, 17:18])
ax_right_3 = fig.add_subplot(gs[0:5, 17:20])
ax_right_4 = fig.add_subplot(gs[5:10, 14:15])
ax_right_5 = fig.add_subplot(gs[5:10, 15:19])
ax_right_6 = fig.add_subplot(gs[5:10, 19:20])
ax_right_7 = fig.add_subplot(gs[10:15, 14:17])
ax_right_8 = fig.add_subplot(gs[10:15, 17:18])
ax_right_9 = fig.add_subplot(gs[10:15, 17:20])

ax_left.text(x=0, y=1.05, s='a)', ha='center', transform=ax_left.transAxes, fontsize=12)
ax_middle_up.text(x=0, y=1.05, s='b)', ha='center', transform=ax_middle_up.transAxes, fontsize=12)
ax_right_1.text(x=0, y=1.05, s='c)', ha='center', transform=ax_right_1.transAxes, fontsize=12)

# Left figure
plt.rcParams['font.family'] = 'sans-serif'

cud_palette = [
'#0101fd',  # Blue
'#E69F00',  # Orange
'#7D7D7D',  # Medium Grey'
'#ff0101',  # Red
'#009B77',  # Teal
]
markers = ['s', 'o', '^', 'X', 'p', 'P']
edgecolor = 'black'
line_width = 1.5
x = np.arange(1001)
marker_size = 80
marker_linewidth = 0.3

load_name = os.path.join(sys.path[0], 'data.pkl')
with open(load_name, 'rb') as f:
    data = pickle.load(f)

edges_start = data['edges'][-1]
G = nx.Graph()
G.add_edges_from(edges_start)

# Set node colors based on degree
node_colors = [ cud_palette[0], cud_palette[0], cud_palette[0], cud_palette[1], cud_palette[1], cud_palette[1], cud_palette[1], cud_palette[2], cud_palette[2], cud_palette[2], cud_palette[2], cud_palette[2], cud_palette[3], cud_palette[3], cud_palette[3], cud_palette[3], cud_palette[3], cud_palette[3], cud_palette[4], cud_palette[4], cud_palette[4], cud_palette[4], cud_palette[4], cud_palette[4], cud_palette[4] ]
pos = nx.spring_layout(G)

nx.draw(G, pos=pos, ax=ax_left, with_labels=False, node_color=node_colors, node_size=400, edgecolors=edgecolor, width=line_width)

# Middle figure
tau_arr = data['tau_arr']
tau_arr = tau_arr[::-1]
ax_middle_up.plot(tau_arr, linewidth=line_width, color='black')
ax_middle_up.scatter([1, 10e0, 10e1, 10e2], [ tau_arr[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], marker=markers[0], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth, color='black')

ax_middle_up.set_xlabel('Iterations', fontsize=12)
ax_middle_up.set_ylabel(r'$\tau_{c}$', fontsize=12)
ax_middle_up.set_xscale('log')
ax_middle_up.set_xticks([10e0, 10e1, 10e2, 10e3], ['10e0', '10e1', '10e2', '10e3'], fontsize=12)
ax_middle_up.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
ax_middle_up.set_xscale('log')
ax_middle_up.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
ax_middle_up.yaxis.grid(False)
ax_middle_up.spines['top'].set_visible(False)
ax_middle_up.spines['right'].set_visible(False)

# Second axis
clustering_arr = data['clustering_arr']
clustering_arr = clustering_arr[::-1]
ax_middle_down.plot(clustering_arr, linewidth=line_width, color='black')
ax_middle_down.scatter([1, 10e0, 10e1, 10e2], [ clustering_arr[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], marker=markers[1], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth, color='black')

ax_middle_down.set_xlabel('Iterations', fontsize=12)
ax_middle_down.set_ylabel('Clustering', fontsize=12)
ax_middle_down.set_xscale('log')
ax_middle_down.set_xticks([10e0, 10e1, 10e2, 10e3], ['10e0', '10e1', '10e2', '10e3'], fontsize=12)
ax_middle_down.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
ax_middle_down.set_xscale('log')
ax_middle_down.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
ax_middle_down.yaxis.grid(False)
ax_middle_down.spines['top'].set_visible(False)
ax_middle_down.spines['right'].set_visible(False)

# Right figure
clique_2 = nx.complete_graph(3)
clique_3 = nx.complete_graph(4)
clique_4 = nx.complete_graph(5)
clique_5 = nx.complete_graph(6)
clique_6 = nx.complete_graph(7)

for ax in [ax_right_1, ax_right_2, ax_right_3, ax_right_4, ax_right_5, ax_right_6, ax_right_7, ax_right_8, ax_right_9]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

nx.draw(clique_2, ax=ax_right_1, with_labels=False, node_color=cud_palette[0], node_size=400, edgecolors=edgecolor, width=line_width)
nx.draw(clique_3, ax=ax_right_3, with_labels=False, node_color=cud_palette[1], node_size=400, edgecolors=edgecolor, width=line_width)
nx.draw(clique_4, ax=ax_right_7, with_labels=False, node_color=cud_palette[2], node_size=400, edgecolors=edgecolor, width=line_width)
nx.draw(clique_5, ax=ax_right_9, with_labels=False, node_color=cud_palette[3], node_size=400, edgecolors=edgecolor, width=line_width)
nx.draw(clique_6, ax=ax_right_5, with_labels=False, node_color=cud_palette[4], node_size=400, edgecolors=edgecolor, width=line_width)

save_name = os.path.join(sys.path[0], 'Figure3.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()