import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

import rewiring_package as rp
                
def save_data(folder, data_dict):
    with open(folder + "data_dict" + '.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

def simulate(type, adjacency, iterations, rewire_count, T=0.001):
    G = nx.from_numpy_array(adjacency)
    degrees = [G.degree(node) for node in G]

    for index in range(1, iterations+1):
        G = nx.random_degree_sequence_graph(degrees, tries=1000)
        orbit_front = rp.optimise_clustering(G, rewire_count=rewire_count, T=T, optimise=True)
        orbit_back = rp.optimise_clustering(G, rewire_count=rewire_count, T=T, optimise=False)

        orbit = {'nodes': [node for node in G]}
        for key in orbit_back.keys():
            if key == 'nodes':
                continue
            orbit[key] = orbit_back[key][::-1] + orbit_front[key]

        save_data(os.path.join(sys.path[0], f'{type}/trial_{index}_'), orbit)

        print('\n')
        percentage = np.round(index / iterations * 100, 3)
        print(f'{t}: {percentage} %', end='\r')
        print('\n')

if __name__ == "__main__":
    T = 0.001
    iterations = 100
    rewire_count = 1000

    # for t in [ "erdos_renyi", "random_regular", "random_geometric", "watts_strogatz" ]:
    for t in ["random_regular", "random_geometric", "watts_strogatz"]:
        file_path = os.path.join(sys.path[0], f"networks/{t}.npy")
        adjacency = np.load(file_path)
        simulate(t, adjacency, iterations, rewire_count, T=T)