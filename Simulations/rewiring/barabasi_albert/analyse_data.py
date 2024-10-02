import networkx as nx
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle

def load_data(load_name):
    with open(load_name, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

if __name__ == "__main__":
    
    tau_arr = []
    clustering_arr = []

    file_names = os.listdir(os.path.join(sys.path[0], 'data_raw'))
    file_names = [ file for file in file_names if file.startswith('trial_') ]
    for i, name in enumerate(file_names):
        orbit = load_data(os.path.join(sys.path[0], 'data_raw/' + name))
        tau_arr.append(orbit['tau_arr'])
        clustering_arr.append(orbit['clustering_arr'])

    tau_arr_mean = np.mean(tau_arr, axis=0)
    clustering_arr_mean = np.mean(clustering_arr, axis=0)

    np.save(os.path.join(sys.path[0], 'data_analysed/tau_arr_mean.npy'), tau_arr_mean)
    np.save(os.path.join(sys.path[0], 'data_analysed/clustering_arr_mean.npy'), clustering_arr_mean)

    fig, ax1 = plt.subplots()

    ax1.plot(tau_arr_mean, 'b-', label=r'$\tau_{c}$', linewidth=1.8)
    ax1.plot(clustering_arr_mean, 'r-', label='Clustering Coefficient', linewidth=1.8)

    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(r'Barabasi-albert networks -- Clustering Coefficient and $\tau_{c}$')
    ax1.legend()

    save_name = os.path.join(sys.path[0], 'rewiring_barabasi_albert.pdf')
    plt.savefig(save_name, dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.show()

    exit()

    