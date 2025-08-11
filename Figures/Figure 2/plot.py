import os
import sys
import pickle
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Any

def load_data(path: str) -> Dict[str, Any]:
    """
    Load a pickled data dictionary from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_figures(output_dir: str) -> None:
    """
    Create and save the clustering vs tau and assortativity plots for various network types.
    """
    # Create a 2x5 grid of subplots with axes in increasing order
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axes_flat = axes.flatten()

    # Label panels a–j
    for ax, label in zip(axes_flat, [chr(ord('a') + i) for i in range(10)]):
        ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                fontsize=16, va='top', ha='right')

    network_info = [
        ('erdos_renyi', 'Poisson'),
        ('random_regular', 'Regular'),
        ('random_geometric', 'Geometric'),
        ('watts_strogatz', 'Watts-Strogatz'),
        ('barabasi_albert', 'Barabasi-Albert')
    ]

    for idx, (net_key, title) in enumerate(network_info):
        # clustering axis
        ax_clust = axes_flat[idx]
        ax_clust.set_title(title, fontsize=14)
        ax_clust.set_xlabel('Clustering', fontsize=12)
        ax_clust.set_ylabel(r'Critical coupling $\tau_c$', fontsize=12)
        ax_clust.tick_params(axis='both', which='major', labelsize=11)

        # assortativity axis
        ax_assort = axes_flat[idx + 5]
        ax_assort.set_xlabel('Assortativity', fontsize=12)
        ax_assort.set_ylabel(r'Critical coupling $\tau_c$', fontsize=12)
        ax_assort.tick_params(axis='both', which='major', labelsize=11)

        # Load data
        data = load_data(os.path.join(output_dir, f'data/data_dict_{net_key}.pkl'))['clustering']
        # Clustering plot
        ax_clust.errorbar(
            data['bin_centers'], data['bin_means'],
            yerr=3 * np.array(data['bin_std_error']),
            fmt='o', ecolor='black', elinewidth=1,
            capsize=2, capthick=1, zorder=1000,
            markeredgecolor='black', markersize=8, markerfacecolor='blue'
        )
        if net_key != 'barabasi_albert':
            label = f"y={data['slope']:.3f}x + {data['intercept']:.3f}"
            ax_clust.plot(
                data['bin_centers'],
                data['slope'] * data['bin_centers'] + data['intercept'],
                color='red', linewidth=2, label=label
            )
            ax_clust.text(
                0.95, 0.05, f"$R^2={data['r_squared']:.3f}$",
                transform=ax_clust.transAxes,
                fontsize=10, va='bottom', ha='right'
            )
            ax_clust.legend(fancybox=False, fontsize=8,
                            loc='upper left', facecolor='white',
                            edgecolor='black')

        # Assortativity
        if net_key == 'random_regular':
            # leave empty but keep axes and labels
            ax_assort.text(0.5, 0.5, 'NA', transform=ax_assort.transAxes,
                           fontsize=16, va='center', ha='center')
            continue
    
        data_a = load_data(os.path.join(output_dir, f'data/data_dict_{net_key}.pkl'))['assortativity_standard']
        ax_assort.errorbar(
            data_a['bin_centers'], data_a['bin_means'],
            yerr=3 * np.array(data_a['bin_std_error']),
            fmt='^', ecolor='black', elinewidth=1,
            capsize=2, capthick=1, zorder=1000,
            markeredgecolor='black', markersize=10, markerfacecolor='orange'
        )
        label_a = f"y={data_a['slope']:.3f}x + {data_a['intercept']:.3f}"
        ax_assort.plot(
            data_a['bin_centers'],
            data_a['slope'] * data_a['bin_centers'] + data_a['intercept'],
            color='red', linewidth=2, label=label_a
        )
        ax_assort.text(
            0.95, 0.05, f"$R^2={data_a['r_squared']:.3f}$",
            transform=ax_assort.transAxes,
            fontsize=10, va='bottom', ha='right'
        )
        ax_assort.legend(fancybox=False, fontsize=8,
                        loc='upper left', facecolor='white',
                        edgecolor='black')

    # Save
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = os.path.join(fig_dir, f'clustering_vs_tau.{ext}')
        plt.savefig(out_path, format=ext, bbox_inches='tight',
                    dpi=300, transparent=True, pad_inches=0.01)
    plt.show()()


def main():
    save_figures(sys.path[0])


if __name__ == '__main__':
    main()
