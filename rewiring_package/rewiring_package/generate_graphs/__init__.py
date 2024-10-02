# __init__.py

# Import important classes, functions, or variables that you want to make available when the package is imported.
from .erdos_renyi import erdos_renyi
from .random_bipartite import random_bipartite
from .random_geometric import random_geometric
from .barabasi_albert import barabasi_albert
from .random_regular import random_regular
from .uniform_graph import uniform_graph
from .watts_strogatz import watts_strogatz

# Define the __all__ variable to specify what should be imported when using "from my_package import *".
__all__ = ['erdos_renyi', 'random_bipartite', 'random_geometric', 'barabasi_albert', 'random_regular', 'uniform_graph', 'watts_strogatz']