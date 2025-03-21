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
        
pos = nx.spring_layout(G)

nx.draw(G, pos=pos, ax=ax, with_labels=False, node_color=node_colors, node_size=500, edgecolors=edgecolor, width=line_width)

save_name = os.path.join(sys.path[0], 'Figure_c.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
save_name = os.path.join(sys.path[0], 'Figure_c.png')
plt.savefig(save_name, dpi=600, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()