# __init__.py

# Import important classes, functions, or variables that you want to make available when the package is imported.
from .generate_graphs.erdos_renyi import erdos_renyi
from .generate_graphs.random_bipartite import random_bipartite
from .generate_graphs.random_geometric import random_geometric
from .generate_graphs.barabasi_albert import barabasi_albert
from .generate_graphs.random_regular import random_regular
from .generate_graphs.uniform_graph import uniform_graph
from .generate_graphs.watts_strogatz import watts_strogatz

from .get_first_bifurcation import get_first_bifurcation

from .rewire import rewire_random_edges
from .rewire import rewired_graph
from .rewire import accept_rewire
from .rewire import rewire_iteration
from .rewire import network_cycle
from .optimise_clustering import optimise_clustering

# Define the __all__ variable to specify what should be imported when using "from my_package import *".
__all__ = [
            'erdos_renyi',
            'random_bipartite',
            'random_geometric',
            'barabasi_albert',
            'random_regular',
            'uniform_graph',
            'watts_strogatz',
            'get_first_bifurcation',
            'rewire_random_edges',
            'rewired_graph',
            'accept_rewire',
            'rewire_iteration',
            'network_cycle',
            'optimise_clustering'
           ]