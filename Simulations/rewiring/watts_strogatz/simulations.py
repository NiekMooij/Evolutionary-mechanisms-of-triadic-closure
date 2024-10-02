import networkx as nx
import numpy as np
import os
import sys
import pickle

import rewiring_package as rp
                
def save_data(folder, data_dict):
    with open(folder + "data_dict" + '.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

if __name__ == "__main__":
    size = 30
    T = 0.001
    iterations = 100
    rewire_count = 1000

    for index in range(1, iterations+1):
        G = rp.watts_strogatz(n=size, k=5, p=0.3)
        orbit = rp.network_cycle(G, rewire_count=rewire_count, T=T, seek_maximum=True)
        clustering_arr = [np.mean(list(nx.clustering(G).values()))]
        for edges in orbit['edges']:
            G = nx.Graph()
            G.add_nodes_from(orbit['nodes'])
            G.add_edges_from(edges)
            clustering = np.mean(list(nx.clustering(G).values()))
            clustering_arr.append(clustering)

        orbit['clustering_arr'] = clustering_arr
        save_data(os.path.join(sys.path[0], f'data_raw/trial_{index}_'), orbit)