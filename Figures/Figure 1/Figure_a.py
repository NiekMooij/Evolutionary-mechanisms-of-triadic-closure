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
fig, ax = plt.subplots(figsize=(5, 5))

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

load_name = os.path.join(sys.path[0], 'data.pkl')
with open(load_name, 'rb') as f:
    data = pickle.load(f)

edges_start = data['edges'][-1]
G = nx.Graph()
G.add_edges_from(edges_start)

print(len(edges_start))
exit()

node_colors = []
for node in G:
    if G.degree(node) == 1:
        node_colors.append(cud_palette[0])
    elif G.degree(node) == 2:
        node_colors.append(cud_palette[1])
    elif G.degree(node) == 3:
        node_colors.append(cud_palette[2])
    elif G.degree(node) == 4:
        node_colors.append(cud_palette[3])
    elif G.degree(node) == 5:
        node_colors.append(cud_palette[4])
    # elif G.degree(node) == 6:
    #     node_colors.append(cud_palette[4])

pos = nx.spring_layout(G)

# --- Make triangle edges thicker ---
triangle_edges = set()
for u, v in G.edges():
    # If u and v have any common neighbor, edge (u,v) is part of a triangle
    if set(G.neighbors(u)).intersection(G.neighbors(v)):
        triangle_edges.add(frozenset((u, v)))

edges = list(G.edges())
triangle_line_width = 4.5  # thicker width for triangle edges
edge_widths = [
    triangle_line_width if frozenset(e) in triangle_edges else line_width
    for e in edges
]
# --- end triangle-thickening block ---
triangle_edge_color = 'black'
non_triangle_edge_color = (0, 0, 0, 0.65)  # RGBA with lower alpha for transparency
edge_colors = [
    triangle_edge_color if frozenset(e) in triangle_edges else non_triangle_edge_color
    for e in edges]

nx.draw(
    G,
    pos=pos,
    ax=ax,
    with_labels=False,
    node_color=node_colors,
    node_size=500,
    edgelist=edges,              # ensure width order matches these edges
    width=edge_widths,           # thicker for triangle edges
    edgecolors=edgecolor,
    edge_color=edge_colors       # set the color of the edges
)

save_name = os.path.join(sys.path[0], 'Figure_a.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.01, transparent=True)
save_name = os.path.join(sys.path[0], 'Figure_a.png')
plt.savefig(save_name, dpi=600, format='png', bbox_inches='tight', pad_inches=0.01, transparent=True)

plt.show()