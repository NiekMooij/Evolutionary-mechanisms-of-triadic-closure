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