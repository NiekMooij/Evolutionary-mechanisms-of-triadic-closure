import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

import rewiring_package as rp

def save_data(folder, data_dict):
    with open(folder, 'wb') as f:
        pickle.dump(data_dict, f)

def load_data(load_name):
    with open(load_name, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

if __name__ == "__main__":
    degree_sequence = [ 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6 ]
    A = np.load(os.path.join(sys.path[0], 'A.npy'))

    clique_2 = nx.complete_graph(3)
    clique_3 = nx.complete_graph(4)
    clique_4 = nx.complete_graph(5)
    clique_5 = nx.complete_graph(6)
    clique_6 = nx.complete_graph(7)

    cliques = [clique_2, clique_3, clique_4, clique_5, clique_6]
    G = nx.disjoint_union_all(cliques)
    
    T = 0
    rewire_count = int(1e3)

    orbit = rp.network_cycle(G, rewire_count=rewire_count, T=T, optimise=False)
    clustering_arr = [np.mean(list(nx.clustering(G).values()))]
    for edges in orbit['edges']:
        G = nx.Graph()
        G.add_nodes_from(orbit['nodes'])
        G.add_edges_from(edges)
        clustering = np.mean(list(nx.clustering(G).values()))
        clustering_arr.append(clustering)

    orbit['clustering_arr'] = clustering_arr
    save_data(os.path.join(sys.path[0], 'data.pkl'), orbit)