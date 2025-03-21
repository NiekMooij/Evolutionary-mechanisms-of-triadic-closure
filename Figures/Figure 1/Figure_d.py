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
line_width = 2
x = np.arange(1001)
marker_size = 80
marker_linewidth = 0.3

load_name = os.path.join(sys.path[0], 'data.pkl')
with open(load_name, 'rb') as f:
    data = pickle.load(f)

# Middle figure
tau_arr = data['tau_arr']
tau_arr = tau_arr[::-1]
ax.plot(tau_arr, linewidth=line_width, color='black')
# ax.scatter([1, 10e0, 10e1, 10e2], [ tau_arr[i] for i in [1, int(10e0), int(10e1), int(10e2)] ], marker=markers[0], s=marker_size, zorder=2, edgecolor='black', linewidth=marker_linewidth, color='black')

ax.set_xlabel('Iterations', fontsize=12)
ax.set_ylabel(r'$\tau_{c}$', fontsize=12)
ax.set_xscale('log')
ax.set_xticks([10e0, 10e1, 10e2, 10e3], ['10e0', '10e1', '10e2', '10e3'], fontsize=12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
ax.set_xscale('log')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.9)
ax.yaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_name = os.path.join(sys.path[0], 'Figure_d.pdf')
plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()