import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

import rewiring_package as rp
from sklearn.linear_model import LinearRegression
                
def load_data(load_name):
    with open(load_name, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_data(folder, data_dict):
    with open(folder, 'wb') as f:
        pickle.dump(data_dict, f)

if __name__ == "__main__":
    iterations = 100

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

    for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                fontsize=16, va='top', ha='right')

    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    for t in [ 'erdos_renyi', 'random_regular', 'random_geometric', 'watts_strogatz' ]:
        if t == 'erdos_renyi':
            ax = ax1
            # ax.set_title('Erdos-Renyi', fontsize=14)
            ax.set_title('Poisson', fontsize=14)
        elif t == 'random_regular':
            ax = ax2
            # ax.set_title('Random Regular', fontsize=14)
            ax.set_title('Regular', fontsize=14)
        elif t == 'random_geometric':
            ax = ax3
            ax.set_title('Geometric', fontsize=14)
        elif t == 'watts_strogatz':
            ax = ax4
            ax.set_title('Watts-Strogatz', fontsize=14)

        file_path = os.path.join(sys.path[0], f'data/data_dict_{t}.pkl')
        data_dict = load_data(file_path)

        # bin_centers = data_dict['']
        # bin_means = data_dict
        # slope = data_dict
        # intercept = data_dict

        # ax.scatter(data_dict['bin_centers'], data_dict['bin_means'], facecolor='blue', edgecolor='black', s=80, zorder=1000)
        ax.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
        label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
        ax.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
        
        ax.set_xlabel('Clustering', fontsize=12)
        ax.set_ylabel('Critical coupling', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(fancybox=False, fontsize=7, loc='upper left', facecolor='white', edgecolor='black')

    save_name = os.path.join(sys.path[0], 'figures/clustering_vs_tau_presentation.pdf')
    plt.savefig(save_name, format='pdf', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.01)
    save_name = os.path.join(sys.path[0], 'figures/clustering_vs_tau_presentation.png')
    plt.savefig(save_name, format='png', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.01)
    plt.show()
