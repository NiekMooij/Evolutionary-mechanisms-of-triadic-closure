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

if __name__ == "__main__":
    G = nx.Graph()

    clique_2 = nx.complete_graph(2)
    G = nx.compose(G, clique_2)
    clique_3 = nx.complete_graph(3)
    mapping_3 = {i: i + 20 for i in clique_3.nodes()}
    clique_3 = nx.relabel_nodes(clique_3, mapping_3)
    G = nx.compose(G, clique_3)
    clique_4 = nx.complete_graph(4)
    mapping_4 = {i: i + 23 for i in clique_4.nodes()}
    clique_4 = nx.relabel_nodes(clique_4, mapping_4)
    G = nx.compose(G, clique_4)
    clique_5 = nx.complete_graph(5)
    mapping_5 = {i: i + 27 for i in clique_5.nodes()}
    clique_5 = nx.relabel_nodes(clique_5, mapping_5)
    G = nx.compose(G, clique_5)

    orbit_back = rp.optimise_clustering(G, rewire_count=100, T=0.000000000005, optimise=False)
    orbit_back['edges'] = [list(G.edges())] + orbit_back['edges']
    orbit_back['clustering_arr'] = [nx.average_clustering(G)] + orbit_back['clustering_arr']
    save_data(os.path.join(sys.path[0], f'data.pkl'), orbit_back)

    load_name = os.path.join(sys.path[0], 'data.pkl')
    with open(load_name, 'rb') as f:
        data = pickle.load(f)

    nodes = data['nodes']

    tau_arr = []
    for edges in data['edges']:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        tau = rp.get_first_bifurcation(G=G, tau_initial=1e-8, tolerance=1e08, regular=False)[0]

        tau_arr.append(tau)

    data['tau_arr'] = tau_arr

    # fig, ax = plt.subplots()
    # ax.plot(range(len(tau_arr)), tau_arr, linewidth=1)
    # ax.scatter(range(len(tau_arr)), tau_arr, s=15)

    # plt.show()

    save_data(os.path.join(sys.path[0], f'data.pkl'), data)