import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import time
import ast

import rewiring_package as rp

if __name__ == "__main__":

    df = pd.read_csv(os.path.join(sys.path[0], 'data_raw/regular_degrees.csv'))
    df['degree'] = pd.to_numeric(df['degree'], errors='coerce')
    df['values'] = df['values'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))

    df_data = pd.DataFrame(columns=['degree', 'mean', 'std'])

    degrees = df['degree'].unique()

    for degree in degrees:
        subset = df[df['degree'] == degree]
        mean_values = subset['values'].apply(np.mean)
        std_values = subset['values'].apply(np.std)
        mean = np.mean(mean_values)
        std = std_values.mean()
        minimum = min(np.array(subset['values'])[0])
        maximum = max(np.array(subset['values'])[0])
        new_row = pd.DataFrame([{'degree': degree, 'mean': mean, 'std': std, 'minimum': minimum, 'maximum': maximum}])
        df_data = pd.concat([df_data, new_row], ignore_index=True)

    df_data.to_csv(os.path.join(sys.path[0], 'data_analysed/regular_data.csv'), index=False)