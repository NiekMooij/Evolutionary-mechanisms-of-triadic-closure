import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm  # progress bars

# NOTE: matplotlib was imported but unused; removed for clarity.


def load_data(path: str) -> dict:
    """Load a pickled dictionary from the given file path."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data(path: str, data: dict) -> None:
    """Save a dictionary to the given file path using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def _coerce_1d(a) -> np.ndarray:
    """Ensure we always get a 1D float array (empty if missing)."""
    if a is None:
        return np.array([], dtype=float)
    arr = np.asarray(list(a), dtype=float)
    return arr.ravel()


def aggregate_trials(
    iterations: int,
    trial_type: str,
    base_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and aggregate **exact** clustering, objective (tau), and assortativity across trials.
    No baseline subtraction, no zero-based segmentation, no dropping of tail entries.

    Returns:
        (clustering_all, objective_all, assortativity_all)
    """
    base = base_dir or sys.path[0]
    clustering_all, objective_all, assort_all = [], [], []

    with tqdm(total=iterations, desc=f"{trial_type} · load/aggregate", unit="trial", leave=False, position=1) as pbar:
        for idx in range(1, iterations + 1):
            path = os.path.join(base, f"{trial_type}/trial_{idx}_data_dict.pkl")
            data = load_data(path)

            # Prefer new keys; fall back to legacy keys if needed
            cluster = _coerce_1d(
                data.get('clustering', data.get('clustering_arr'))
            )
            objective = _coerce_1d(
                data.get('objective_arr', data.get('tau'))  # objective_arr is τ*(G) at each step
            )
            assort = _coerce_1d(
                data.get('assortativity_standard', data.get('assortativity'))
            )

            # Align lengths per trial so x and y pair up correctly
            if cluster.size == 0 or objective.size == 0:
                # Skip trials with missing core arrays
                pbar.update(1)
                continue

            L = int(min(cluster.size, objective.size))
            cluster = cluster[:L]
            objective = objective[:L]
            if assort.size:
                assort = assort[:L]  # may be empty for random_regular

            clustering_all.extend(cluster.tolist())
            objective_all.extend(objective.tolist())

            if assort.size:
                assort_all.extend(assort.tolist())

            pbar.update(1)

    return (
        np.array(clustering_all, dtype=float),
        np.array(objective_all, dtype=float),
        np.array(assort_all, dtype=float),
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
    return float(model.intercept_), float(model.coef_[0]), float(model.score(x.reshape(-1, 1), y))


def compute_binned_statistics(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int = 15
) -> Dict[str, np.ndarray]:
    """
    Bin x into intervals and compute mean, std, and stderr of y in each bin.
    Returns a dict with bin_centers, means, stds, stderrs.
    """
    # Clean NaNs/infs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return {
            'bin_centers': np.array([]),
            'bin_means': np.array([]),
            'bin_stds': np.array([]),
            'bin_std_error': np.array([]),
        }

    if np.allclose(x.min(), x.max()):
        bins = np.array([x.min(), x.max() + 1e-12])
    else:
        bins = np.linspace(x.min(), x.max(), num_bins + 1)

    indices = np.digitize(x, bins)
    means, stds, stderrs = [], [], []
    for i in range(1, len(bins)):
        sel = indices == i
        if np.any(sel):
            ys = y[sel]
            means.append(float(np.nanmean(ys)))
            stds.append(float(np.nanstd(ys)))
            stderrs.append(float(stats.sem(ys)) if ys.size > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(np.nan)
            stderrs.append(np.nan)

    centers = 0.5 * (bins[:-1] + bins[1:])
    return {
        'bin_centers': centers,
        'bin_means': np.array(means),
        'bin_stds': np.array(stds),
        'bin_std_error': np.array(stderrs),
    }


def analyze_feature(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, Optional[np.ndarray]]:
    """
    Perform linear regression and binned statistics for a feature.
    Uses exact values as provided.
    """
    stats_dict = compute_binned_statistics(x, y)

    # Only regress if we have data
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) >= 2:
        intercept, slope, r2 = fit_linear_model(x[mask], y[mask])
    else:
        intercept = slope = r2 = None

    stats_dict.update({'intercept': intercept, 'slope': slope, 'r_squared': r2})
    return stats_dict


def process_trials(
    iterations: int,
    trial_type: str,
    base_dir: Optional[str] = None
) -> Dict[str, dict]:
    """
    Aggregate trial data and compute stats for clustering and assortativity
    against the objective values (tau) — all using **exact stored values**.
    """
    arr_c, arr_t, arr_a = aggregate_trials(iterations, trial_type, base_dir)
    clustering_stats = analyze_feature(arr_c, arr_t)

    if trial_type != 'random_regular' and arr_a.size:
        assort_stats = analyze_feature(arr_a, arr_t)
    else:
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
    Run analysis for predefined network types and save results.
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

    for t in tqdm(types, desc="Network types", unit="type", position=0):
        results = process_trials(iterations, t, base)
        save_data(os.path.join(out_dir, f'data_dict_{t}.pkl'), results)
        tqdm.write(f"Completed processing for {t}.")


if __name__ == '__main__':
    main()
