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
# fig = plt.figure(figsize=(5, 5))
# gs = gridspec.GridSpec(15, 6)

# ax_right = fig.add_subplot(gs[:, :])

# ax_right_1 = fig.add_subplot(gs[0:5, 0:3])
# ax_right_2 = fig.add_subplot(gs[0:5, 3:4])
# ax_right_3 = fig.add_subplot(gs[0:5, 3:6])
# ax_right_4 = fig.add_subplot(gs[5:10, 0:1])
# ax_right_5 = fig.add_subplot(gs[5:10, 1:5])
# ax_right_6 = fig.add_subplot(gs[5:10, 5:6])
# ax_right_7 = fig.add_subplot(gs[10:15, 0:3])
# ax_right_8 = fig.add_subplot(gs[10:15, 3:4])
# ax_right_9 = fig.add_subplot(gs[10:15, 3:6])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

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
line_width = 2.5
x = np.arange(1001)
marker_size = 80
marker_linewidth = 0.3

# load_name = os.path.join(sys.path[0], 'data.pkl')
# with open(load_name, 'rb') as f:
#     data = pickle.load(f)

# edges_start = data['edges'][-1]
# G = nx.Graph()
# G.add_edges_from(edges_start)

# node_colors = []
# for node in G:
#     if G.degree(node) == 1:
#         node_colors.append(cud_palette[0])
#     elif G.degree(node) == 2:
#         node_colors.append(cud_palette[1])
#     elif G.degree(node) == 3:
#         node_colors.append(cud_palette[2])
#     elif G.degree(node) == 4:
#         node_colors.append(cud_palette[3])
#     elif G.degree(node) == 5:
#         node_colors.append(cud_palette[4])

# pos = nx.spring_layout(G)

# Right figure
clique_2 = nx.complete_graph(2)
clique_3 = nx.complete_graph(3)
clique_4 = nx.complete_graph(4)
clique_5 = nx.complete_graph(5)
# clique_6 = nx.complete_graph(6)

for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

nx.draw(clique_2, ax=ax1, with_labels=False, node_color=cud_palette[0], node_size=500, edgecolors=edgecolor, width=line_width, linewidths=2)
nx.draw(clique_3, ax=ax2, with_labels=False, node_color=cud_palette[1], node_size=500, edgecolors=edgecolor, width=line_width, linewidths=2)
nx.draw(clique_4, ax=ax3, with_labels=False, node_color=cud_palette[2], node_size=500, edgecolors=edgecolor, width=line_width, linewidths=2)
nx.draw(clique_5, ax=ax4, with_labels=False, node_color=cud_palette[3], node_size=500, edgecolors=edgecolor, width=line_width, linewidths=2)
# nx.draw(clique_6, ax=ax_right_5, with_labels=False, node_color=cud_palette[4], node_size=400, edgecolors=edgecolor, width=line_width, linewidths=2)

save_name = os.path.join(sys.path[0], 'Figure_b.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
save_name = os.path.join(sys.path[0], 'Figure_b.png')
plt.savefig(save_name, dpi=600, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()