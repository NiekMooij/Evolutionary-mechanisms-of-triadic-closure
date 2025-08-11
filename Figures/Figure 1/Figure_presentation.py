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

fig = plt.figure(figsize=(7.2, 4.6)) 

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
fig.subplots_adjust(wspace=0.4, hspace=0.8)

# Panel 1 (Top-left)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(range(len(clustering_arr)), clustering_arr, linewidth=1, alpha=0.7, linestyle='--')
ax1.scatter(range(len(clustering_arr)), clustering_arr, s=35, edgecolors='black', zorder=100)

ax1.set_title('n=14, m=20', fontsize=12, pad=5)
ax1.set_ylabel('Mean Clustering', fontsize=12)
ax1.set_xlabel('Rewiring number', fontsize=12)
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.set_xticks([0, 5, 10, 15, 20, 25, 30], ['-15', '-10', '-5', '0', '5', '10', '15'], fontsize=11)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)

# Add inset figures inside ax1
ax2 = plt.subplot(gs[0, 1])
file_names = [
    os.path.join(sys.path[0], f'Figure_a.png'),
    os.path.join(sys.path[0], f'Figure_c.png'),
    os.path.join(sys.path[0], f'Figure_b.png')
]

for i, file_name in enumerate(file_names):
    img = mpimg.imread(file_name)
    ax2.imshow(img, extent=[i * 1.5, (i + 1) * 1.5, 0.3, 0.42], aspect='auto', zorder=10)

for spine_name, spine in ax2.spines.items():
    if spine_name not in ['bottom']:
        spine.set_visible(False)

ax2.axvline(x=1.93, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax2.axvspan(1.93, len(file_names) * 1.5, color='green', alpha=0.1, zorder=0)
ax2.axvspan(-0.4, 1.93, color='purple', alpha=0.1, zorder=0)
ax2.set_xlabel('Image Index', fontsize=12)
width = 4.5/7
ax2.set_xticks([width*i for i in range(7)], ['-15', '-10', '-5', '0', '5', '10', '15'], fontsize=11)
ax2.set_xlabel('Rewiring number', fontsize=12)
ax2.set_xlim(-0.4, len(file_names) * 1.5)
ax2.set_ylim(0.3, 0.5)
ax2.set_yticks([], [])
# Add text indicating increased clustering in the middle of the red domain
ax2.text(3.25, 0.48, 'Increased', color='black', fontsize=8, 
         ha='center', va='center', transform=ax2.transData, zorder=20)
ax2.text(0.7, 0.48, 'Decreased', color='black', fontsize=8, 
         ha='center', va='center', transform=ax2.transData, zorder=20)

# Panel 4 (Bottom-right)
ax4 = plt.subplot(gs[1, 1])
inset_ax4_1 = inset_axes(ax4, width="40%", height="40%", loc="upper center")
inset_ax4_2 = inset_axes(ax4, width="40%", height="40%", loc="lower left")
inset_ax4_3 = inset_axes(ax4, width="40%", height="40%", loc="lower right")

pos = {
    1: (0, 0),
    2: (1, 0),
    3: (0, 1),
    4: (1, 1)
}
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (3, 4)])
nx.draw(G, pos, ax=inset_ax4_1, with_labels=True, node_size=200, font_color='white', node_color='blue', edge_color='black', width=2, edgecolors='black')
inset_ax4_1.set_xlim(-0.5, 1.5)
inset_ax4_1.set_ylim(-0.5, 1.5)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 4), (2, 3)])
nx.draw(G, pos, ax=inset_ax4_2, with_labels=True, node_size=200, font_color='white', node_color='blue', edge_color='black', width=2, edgecolors='black')
inset_ax4_2.set_xlim(-0.5, 1.5)
inset_ax4_2.set_ylim(-0.5, 1.5)

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 3), (2, 4)])
nx.draw(G, pos, ax=inset_ax4_3, with_labels=True, node_size=200, font_color='white', node_color='blue', edge_color='black', width=2, edgecolors='black')
inset_ax4_3.set_xlim(-0.5, 1.5)
inset_ax4_3.set_ylim(-0.5, 1.5)

for spine in ax4.spines.values():
    spine.set_visible(False)

ax4.set_xticks([], [])
ax4.set_yticks([], [])

# Add arrow from inset_ax4_1 to inset_ax4_2
arrowprops = dict(facecolor='black', arrowstyle='->', linewidth=1.5)
ax4.annotate('', xy=(0.25, 0.35), xytext=(0.45, 0.7), textcoords='axes fraction', arrowprops=arrowprops)
ax4.text(0.25, 0.55, 'crossed', transform=ax4.transAxes, fontsize=8, va='center', ha='center', rotation=45)
ax4.annotate('', xy=(0.75, 0.35), xytext=(0.55, 0.7), textcoords='axes fraction', arrowprops=arrowprops)
ax4.annotate('', xy=(0.75, 0.35), xytext=(0.55, 0.7), textcoords='axes fraction', arrowprops=arrowprops)
ax4.text(0.75, 0.55, 'switched', transform=ax4.transAxes, fontsize=8, va='center', ha='center', rotation=-45)

# Panel 3 (Bottom-left)
ax3 = plt.subplot(gs[1, 0])
ax3.plot(range(len(tau_arr)), tau_arr, linewidth=1.5, alpha=0.7, linestyle='--')
ax3.scatter(range(len(tau_arr)), tau_arr, s=30, edgecolors='black', zorder=100)

ax3.set_title('n=14, m=20', fontsize=12, pad=5)
ax3.set_xlabel('Rewiring number', fontsize=12)
ax3.set_ylabel('Critical coupling', fontsize=12)
ax3.set_xticks([0, 5, 10, 15, 20, 25, 30], ['-15', '-10', '-5', '0', '5', '10', '15'], fontsize=11)
ax3.set_yticks([0.4, 0.6, 0.8, 1.0], ['0.4', '0.6', '0.8', '1.0'], fontsize=11)
# ax3.set_yticks([0.38, 0.40, 0.42, 0.44], ['0.38', '0.40', '0.42', '0.44'], fontsize=11)
ax3.set_ylim(0.3, 1.05)

# Add labels a, b, c, d to the axes
ax1.text(-0.1, 1.15, 'a', transform=ax1.transAxes, fontsize=16, va='top', ha='right')
ax2.text(-0.1, 1.15, 'b', transform=ax2.transAxes, fontsize=16, va='top', ha='right')
ax3.text(-0.1, 1.15, 'c', transform=ax3.transAxes, fontsize=16, va='top', ha='right')
ax4.text(-0.1, 1.15, 'd', transform=ax4.transAxes, fontsize=16, va='top', ha='right')

save_name = os.path.join(sys.path[0], 'figure_rewiring_presentation.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.01, transparent=True)
save_name = os.path.join(sys.path[0], 'figure_rewiring_presentation.png')
plt.savefig(save_name, dpi=600, format='png', bbox_inches='tight', pad_inches=0.01, transparent=True)

plt.show()
