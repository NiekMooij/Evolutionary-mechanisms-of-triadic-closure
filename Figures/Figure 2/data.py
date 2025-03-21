import networkx as nx
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from scipy import stats

import rewiring_package as rp
from sklearn.linear_model import LinearRegression
                
def load_data(load_name):
    with open(load_name, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def save_data(folder, data_dict):
    with open(folder, 'wb') as f:
        pickle.dump(data_dict, f)

def data(iterations, t):
    data_c = []
    data_t = []
    data_ass = []
    for index in range(1, iterations+1):
        file_name = os.path.join(sys.path[0], f'{t}/trial_{index}_data_dict.pkl')
        data_dict = load_data(file_name)

        clustering = data_dict['clustering_arr']
        tau = [ item[0] for item in data_dict['tau'] ]
        ass_arr = data_dict['assortativity_standard']
        clustering.pop(1000)
        clustering.pop(1001)

        lower_index = max([i for i, val in enumerate(clustering[:1000]) if val == 0], default=None)
        upper_index = max([i for i, val in enumerate(clustering[1000:]) if val == 0], default=None)

        clustering = clustering[lower_index:upper_index]
        tau = tau[lower_index:upper_index]
        ass_arr = ass_arr[lower_index:upper_index]

        data_c.extend(clustering)
        data_t.extend(tau)
        data_ass.extend(ass_arr)

    data_dict = {}

    # Clustering data
    clustering = np.array(data_c).reshape(-1, 1)
    tau = np.array(data_t)
    model = LinearRegression()
    model.fit(clustering, tau)

    data_c = np.array(data_c)
    data_t = np.array(data_t)

    num_bins = 15
    bins = np.linspace(min(data_c), max(data_c), num_bins+1)
    bin_indices = np.digitize(data_c, bins)
    bin_means = [data_t[bin_indices == i].mean() for i in range(1, len(bins))]
    bin_std_error = [stats.sem(data_t[bin_indices == i]) for i in range(1, len(bins))]

    bin_stds = [data_t[bin_indices == i].std() for i in range(1, len(bins))]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Predict values
    # tau_pred = model.predict(clustering)
    intercept = model.intercept_
    slope = model.coef_[0]
    
    data_dict['clustering'] = {'bin_centers': bin_centers, 'bin_means': bin_means, 'bin_stds': bin_stds, 'bin_std_error': bin_std_error, 'slope': slope, 'intercept': intercept}

    # Assortativity data
    ass_arr = np.array(data_ass).reshape(-1, 1)
    tau = np.array(data_t)
    model = LinearRegression()
    model.fit(ass_arr, tau)

    data_a = np.array(ass_arr)
    data_a = [ i[0] for i in data_a ]
    data_t = np.array(data_t)

    num_bins = 15
    bins = np.linspace(min(data_a), max(data_a), num_bins+1)
    bin_indices = np.digitize(data_a, bins)
    bin_means = [data_t[bin_indices == i].mean() for i in range(1, len(bins))]
    bin_std_error = [stats.sem(data_t[bin_indices == i]) for i in range(1, len(bins))]

    bin_stds = [data_t[bin_indices == i].std() for i in range(1, len(bins))]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Predict values
    intercept = model.intercept_
    slope = model.coef_[0]

    data_dict['assortativity_standard'] = {'bin_centers': bin_centers, 'bin_means': bin_means, 'bin_stds': bin_stds, 'bin_std_error': bin_std_error, 'slope': slope, 'intercept': intercept}

    return data_dict

if __name__ == "__main__":
    iterations = 100
    # for t in [ 'erdos_renyi', 'random_regular', 'random_geometric', 'watts_strogatz' ]:
    for t in [ 'erdos_renyi', 'random_geometric', 'watts_strogatz' ]:
        data_dict = data(100, t)
        file_path = os.path.join(sys.path[0], f'data/data_dict_{t}.pkl')
        save_data(file_path, data_dict)