import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

load_name = os.path.join(sys.path[0], 'data.pkl')
with open(load_name, 'rb') as f:
    data = pickle.load(f)

tau_arr = data['tau_arr'][:30]
tau_arr = tau_arr[::-1]

clustering_arr = data['clustering_arr'][1:30]
clustering_arr.append(0)
clustering_arr = clustering_arr[::-1]
clustering_arr.append(1)

# fig, ax = plt.subplots(figsize=(9,4))
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
ax = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# Axis 1
ax.plot(range(len(clustering_arr)), clustering_arr, linewidth=1)
ax.scatter(range(len(clustering_arr)), clustering_arr, s=35, edgecolors='black', zorder=100)
ax.text(-0.03, 1.08, 'a', transform=ax.transAxes, fontsize=14, va='top', ha='right')

inset_ax = fig.add_axes([0.1, 0.182, 0.2, 0.2])
inset_ax.text(0, 1.0, 'c', transform=inset_ax.transAxes, fontsize=14, va='top', ha='right')
file_name = os.path.join(sys.path[0], f'Figure_a.png')
img = mpimg.imread(file_name)
inset_ax.imshow(img)
inset_ax.axis('off')

inset_ax = fig.add_axes([0.3, 0.25, 0.2, 0.2])
inset_ax.text(0, 1.0, 'd', transform=inset_ax.transAxes, fontsize=14, va='top', ha='right')
file_name = os.path.join(sys.path[0], f'Figure_c.png')
img = mpimg.imread(file_name)
inset_ax.imshow(img)
inset_ax.axis('off')

inset_ax = fig.add_axes([0.4, 0.65, 0.2, 0.2])
inset_ax.text(0, 1.0, 'e', transform=inset_ax.transAxes, fontsize=14, va='top', ha='right')
file_name = os.path.join(sys.path[0], f'Figure_b.png')
img = mpimg.imread(file_name)
inset_ax.imshow(img)
inset_ax.axis('off')

# Add arrows
arrowprops = dict(facecolor='black', arrowstyle='->')
ax.annotate('', xy=(0, clustering_arr[0]), xytext=(0.12, 0.17), textcoords='axes fraction', arrowprops=arrowprops)
ax.annotate('', xy=(15, clustering_arr[15]), xytext=(0.53, 0.23), textcoords='axes fraction', arrowprops=arrowprops)
ax.annotate('', xy=(30, clustering_arr[30]), xytext=(0.87, 0.84), textcoords='axes fraction', arrowprops=arrowprops)

ax.set_ylabel('Mean Clustering', fontsize=12)
ax.set_xlabel('Rewiring number', fontsize=12)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)
# ax.set_xticks([0, 5, 10, 15, 20, 25, 30], ['0', '5', '10', '15', '20', '25', '30'])
ax.set_xticks([0, 5, 10, 15, 20, 25, 30], ['-15', '-10', '-5', '0', '5', '10', '15'], fontsize=11)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)

# Add critical clustering plot
inset_ax = fig.add_axes([0.18, 0.55, 0.18, 0.23])
inset_ax.plot(range(len(tau_arr)), tau_arr, linewidth=1.5)
inset_ax.scatter(range(len(tau_arr)), tau_arr, s=20, edgecolors='black', zorder=100)
inset_ax.text(-0.1, 1.2, 'b', transform=inset_ax.transAxes, fontsize=14, va='top', ha='right')

inset_ax.set_xlabel('Rewiring number', fontsize=9)
inset_ax.set_ylabel('Critical coupling', fontsize=9)
# inset_ax.set_xticks([0, 5, 10, 15, 20, 25, 30], ['0', '5', '10', '15', '20', '25', '30'], fontsize=8)
inset_ax.set_xticks([0, 5, 10, 15, 20, 25, 30], ['-15', '-10', '-5', '0', '5', '10', '15'], fontsize=8)
inset_ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

# Axis 2
inset_ax2_1 = inset_axes(ax2, width="30%", height="30%", loc="upper center")
inset_ax2_2 = inset_axes(ax2, width="30%", height="30%", loc="lower left")
inset_ax2_3 = inset_axes(ax2, width="30%", height="30%", loc="lower right")
inset_ax2_1.text(-0.3, 1.2, 'f', transform=inset_ax2_1.transAxes, fontsize=14, va='top', ha='right')

pos = {
    1: (0, 0),
    2: (1/2, 0),
    3: (0, 1/2),
    4: (1/2, 1/2)
}
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (3, 4)])
nx.draw(G, pos, ax=inset_ax2_1, with_labels=True, node_size=250, font_color='white', node_color='blue', edge_color='black', width=3, edgecolors='black')
inset_ax2_1.set_xlim(-0.2, 0.7)
inset_ax2_1.set_ylim(-0.2, 0.7)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 4), (2, 3)])
nx.draw(G, pos, ax=inset_ax2_2, with_labels=True, node_size=250, font_color='white', node_color='blue', edge_color='black', width=3, edgecolors='black')
inset_ax2_2.set_xlim(-0.2, 0.7)
inset_ax2_2.set_ylim(-0.2, 0.7)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 3), (2, 4)])
nx.draw(G, pos, ax=inset_ax2_3, with_labels=True, node_size=250, font_color='white', node_color='blue', edge_color='black', width=3, edgecolors='black')
inset_ax2_3.set_xlim(-0.2, 0.7)
inset_ax2_3.set_ylim(-0.2, 0.7)

for spine in ax2.spines.values():
    spine.set_visible(False)

ax2.set_xticks([], [])
ax2.set_yticks([], [])

# Add arrow from inset_ax2_1 to inset_ax2_2
arrowprops = dict(facecolor='black', arrowstyle='->', linewidth=2)
ax2.annotate('', xy=(0.2, 0.32), xytext=(0.4, 0.65), textcoords='axes fraction', arrowprops=arrowprops)
ax2.annotate('', xy=(0.8, 0.32), xytext=(0.65, 0.65), textcoords='axes fraction', arrowprops=arrowprops)

save_name = os.path.join(sys.path[0], 'Figure3.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.01, transparent=True)
save_name = os.path.join(sys.path[0], 'Figure3.png')
plt.savefig(save_name, dpi=600, format='png', bbox_inches='tight', pad_inches=0.01, transparent=True)

plt.show()