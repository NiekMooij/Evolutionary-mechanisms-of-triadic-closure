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

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                fontsize=16, va='top', ha='right')

    # Axis 1
    ax1.set_title('Poisson', fontsize=14)
    file_path = os.path.join(sys.path[0], f'data/data_dict_erdos_renyi.pkl')
    data_dict = load_data(file_path)['clustering']
    ax1.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax1.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax1.set_xlabel('Clustering', fontsize=12)
    ax1.set_ylabel('Critical coupling', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 5
    file_path = os.path.join(sys.path[0], f'data/data_dict_erdos_renyi.pkl')
    data_dict = load_data(file_path)['assortativity_standard']
    ax5.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax5.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax5.set_xlabel('Assortativity', fontsize=12)
    ax5.set_ylabel('Critical coupling', fontsize=12)
    ax5.tick_params(axis='both', which='major', labelsize=11)
    ax5.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 2
    ax2.set_title('Regular', fontsize=14)
    file_path = os.path.join(sys.path[0], f'data/data_dict_random_regular.pkl')
    data_dict = load_data(file_path)['clustering']
    ax2.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax2.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax2.set_xlabel('Clustering', fontsize=12)
    ax2.set_ylabel('Critical coupling', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 6
    file_path = os.path.join(sys.path[0], f'data/data_dict_random_regular.pkl')
    data_dict = load_data(file_path)['assortativity']
    # ax6.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    # label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    # ax6.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax6.set_xlabel('Assortativity', fontsize=12)
    ax6.set_ylabel('Critical coupling', fontsize=12)
    ax6.set_xticks([], [])
    ax6.set_yticks([], [])
    ax6.tick_params(axis='both', which='major', labelsize=11)
    # ax6.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')
    ax6.text(0.5, 0.5, 'NA', transform=ax6.transAxes, fontsize=16, va='center', ha='center')

    # Axis 3
    ax3.set_title('Geometric', fontsize=14)
    file_path = os.path.join(sys.path[0], f'data/data_dict_random_geometric.pkl')
    data_dict = load_data(file_path)['clustering']
    ax3.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax3.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax3.set_xlabel('Clustering', fontsize=12)
    ax3.set_ylabel('Critical coupling', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=11)
    ax3.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 7
    file_path = os.path.join(sys.path[0], f'data/data_dict_random_geometric.pkl')
    data_dict = load_data(file_path)['assortativity_standard']
    ax7.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax7.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax7.set_xlabel('Assortativity', fontsize=12)
    ax7.set_ylabel('Critical coupling', fontsize=12)
    ax7.tick_params(axis='both', which='major', labelsize=11)
    ax7.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 4
    ax4.set_title('Watts-Strogatz', fontsize=14)
    file_path = os.path.join(sys.path[0], f'data/data_dict_watts_strogatz.pkl')
    data_dict = load_data(file_path)['clustering']
    ax4.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax4.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax4.set_xlabel('Clustering', fontsize=12)
    ax4.set_ylabel('Critical coupling', fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    # Axis 8
    file_path = os.path.join(sys.path[0], f'data/data_dict_watts_strogatz.pkl')
    data_dict = load_data(file_path)['assortativity_standard']
    ax8.errorbar(data_dict['bin_centers'], data_dict['bin_means'], yerr=data_dict['bin_stds'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, capthick=1, zorder=1000, markeredgecolor='black')
    label = f'y={np.round(data_dict['slope'],3)}x + {np.round(data_dict['intercept'],3)}'
    ax8.plot(data_dict['bin_centers'], data_dict['slope']*data_dict['bin_centers'] + data_dict['intercept'], color='red', linewidth=2, label=label)
    
    ax8.set_xlabel('Assortativity', fontsize=12)
    ax8.set_ylabel('Critical coupling', fontsize=12)
    ax8.tick_params(axis='both', which='major', labelsize=11)
    ax8.legend(fancybox=False, fontsize=8, loc='upper left', facecolor='white', edgecolor='black')

    save_name = os.path.join(sys.path[0], 'figures/clustering_vs_tau.pdf')
    plt.savefig(save_name, format='pdf', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.01)
    save_name = os.path.join(sys.path[0], 'figures/clustering_vs_tau.png')
    plt.savefig(save_name, format='png', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.01)
    plt.show()