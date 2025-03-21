import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

import rewiring_package as rp
                
def load_data(load_name):
    with open(load_name, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_data(folder, data_dict):
    with open(folder, 'wb') as f:
        pickle.dump(data_dict, f)

# def assortativity(G):
#     value = 0
#     for e in G.edges:
#         d1 = G.degree(e[0])
#         d2 = G.degree(e[1])

#         value += d1 * d2

#     d_mean = np.mean([G.degree(node) for node in G])
#     value = value / (d_mean**2)
#     value = value / len(G.edges())
    
#     return value

def assortativity(G):
    return nx.degree_assortativity_coefficient(G)

def add_critical_tau(type, iterations):
    for index in range(1, iterations+1):
        file_name = os.path.join(sys.path[0], f'{type}/trial_{index}_data_dict.pkl')
        data_dict = load_data(file_name)
        nodes = data_dict['nodes']

        tau_arr = []
        for j, edges in enumerate(data_dict['edges']):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            # Calculate critical tau
            tau = rp.get_first_bifurcation(G=G, tau_initial=1e-8, tolerance=1e-8, regular=False)
            tau_arr.append(tau)
            print(f'{j+1}/2000')

        data_dict['tau'] = tau_arr

        # Save data
        save_data(file_name, data_dict)

        percentage = np.round((index / iterations) * 100, 3)
        print(f'{t} - {percentage}% complete')

def add_assortativity(type, iterations):
    for index in range(1, iterations+1):
        file_name = os.path.join(sys.path[0], f'{type}/trial_{index}_data_dict.pkl')
        data_dict = load_data(file_name)
        nodes = data_dict['nodes']

        ass_arr = []
        for j, edges in enumerate(data_dict['edges']):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            # Calculate critical tau
            ass = assortativity(G)
            ass_arr.append(ass)
            print(f'{j+1}/2000')

        data_dict['assortativity_standard'] = ass_arr

        # Save data
        save_data(file_name, data_dict)

        percentage = np.round((index / iterations) * 100, 3)
        print(f'{t} - {percentage}% complete')

if __name__ == "__main__":
    iterations = 100

    # for t in ['random_geometric', 'random_regular', 'erdos_renyi', 'watts_strogatz']:
    for t in ['random_geometric', 'erdos_renyi', 'watts_strogatz']:
        # add_critical_tau(t, iterations)
        add_assortativity(t, iterations)
    