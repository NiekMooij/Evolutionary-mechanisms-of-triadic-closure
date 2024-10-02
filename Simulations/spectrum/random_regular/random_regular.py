import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import time

import rewiring_package as rp

if __name__ == "__main__":
    start_time = time.time()
    size = 58
    trials = 100
    results = {}
    for index, degree in enumerate(range(1, size, 1)):
        results[degree] = []
        for i in range(trials):

            if degree > (size+1)/2:
                G = rp.random_regular(size=size, d=size-degree-1)
                G = nx.complement(G)
            else:
                G = rp.random_regular(size=size, d=degree)
            value, _ = rp.get_first_bifurcation(G, tau_initial=1e-20, tolerance=1e-8, regular=True)
            results[degree].append(float(value))

        percentage = np.round((index + 1) / size, 3)
        print(f'{percentage * 100}% Done', end='\r')

    # Convert results to a DataFrame
    df = pd.DataFrame(columns=['degree', 'values'])
    for key, val in results.items():
        new_row = pd.DataFrame([{'degree': key, 'values': val}])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(os.path.join(sys.path[0], 'data_raw/regular_degrees.csv'), index=False)

    end_time = time.time() - start_time
    print(end_time)