import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def load_data(path: str) -> dict:
    """
    Load a pickled dictionary from the given file path.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data(path: str, data: dict) -> None:
    """
    Save a dictionary to the given file path using pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def aggregate_trials(
    iterations: int,
    trial_type: str,
    base_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and aggregate clustering, tau, and assortativity data
    across multiple trial pickles for a given trial type.
    """
    base = base_dir or sys.path[0]
    clustering_list, tau_list, assort_list = [], [], []

    for idx in range(1, iterations + 1):
        path = os.path.join(base, f"{trial_type}/trial_{idx}_data_dict.pkl")
        data = load_data(path)

        # Prepare raw arrays
        cluster = list(data['clustering_arr'][:-2])
        tau_vals = [t[0] for t in data['tau']]
        assort_vals = data.get('assortativity', [])

        # Determine valid segment where clustering resets to zero
        lower = max((i for i, v in enumerate(cluster[:1000]) if v == 0), default=0)
        upper = max((i for i, v in enumerate(cluster[1000:], start=1000) if v == 0), default=len(cluster))

        clustering_list.extend(cluster[lower:upper])
        tau_list.extend(tau_vals[lower:upper])
        assort_list.extend(assort_vals[lower:upper] if assort_vals else [])

    return (
        np.array(clustering_list),
        np.array(tau_list),
        np.array(assort_list) if assort_list else np.array([])
    )


def fit_linear_model(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit a simple linear regression y ~ x.

    Returns:
        intercept, slope, r_squared
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.intercept_, model.coef_[0], model.score(x.reshape(-1, 1), y)


def compute_binned_statistics(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int = 15
) -> Dict[str, np.ndarray]:
    """
    Bin x into intervals and compute mean, std, and stderr of y in each bin.
    Returns a dict with bin_centers, means, stds, stderrs.
    """
    bins = np.linspace(x.min(), x.max(), num_bins + 1)
    indices = np.digitize(x, bins)

    means, stds, stderrs = [], [], []
    for i in range(1, len(bins)):
        mask = indices == i
        if mask.any():
            y_slice = y[mask]
            means.append(y_slice.mean())
            stds.append(y_slice.std())
            stderrs.append(stats.sem(y_slice))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            stderrs.append(np.nan)

    centers = 0.5 * (bins[:-1] + bins[1:])
    return {
        'bin_centers': centers,
        'bin_means': np.array(means),
        'bin_stds': np.array(stds),
        'bin_std_error': np.array(stderrs)
    }


def analyze_feature(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, Optional[np.ndarray]]:
    """
    Perform linear regression and binned statistics for a feature.

    Returns a dict containing regression params and binned stats.
    """
    stats_dict = compute_binned_statistics(x, y)
    intercept, slope, r2 = fit_linear_model(x, y)
    stats_dict.update({'intercept': intercept, 'slope': slope, 'r_squared': r2})
    return stats_dict


def process_trials(
    iterations: int,
    trial_type: str,
    base_dir: Optional[str] = None
) -> Dict[str, dict]:
    """
    Aggregate trial data and compute stats for clustering and assortativity.
    """
    arr_c, arr_t, arr_a = aggregate_trials(iterations, trial_type, base_dir)
    clustering_stats = analyze_feature(arr_c, arr_t)

    if trial_type != 'random_regular' and arr_a.size:
        assort_stats = analyze_feature(arr_a, arr_t)
    else:
        # empty results for random_regular
        assort_stats = {
            'bin_centers': np.array([]),
            'bin_means': np.array([]),
            'bin_stds': np.array([]),
            'bin_std_error': np.array([]),
            'intercept': None,
            'slope': None,
            'r_squared': None
        }

    return {
        'clustering': clustering_stats,
        'assortativity': assort_stats
    }


def main():
    """
    Main execution: run analysis for predefined network types and save results.
    """
    iterations = 100
    types = [
        'erdos_renyi',
        'random_regular',
        'random_geometric',
        'watts_strogatz',
        'barabasi_albert'
    ]
    base = sys.path[0]
    out_dir = os.path.join(base, 'data')

    for t in types:
        results = process_trials(iterations, t, base)
        save_data(os.path.join(out_dir, f'data_dict_{t}.pkl'), results)
        print(f"Completed processing for {t}.")


if __name__ == '__main__':
    main()
