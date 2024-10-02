import networkx as nx
import numpy as np
from scipy import linalg

def F(point: np.ndarray) -> float:
    """Calculate product of array elements.

    Parameters:
    - point (np.ndarray): Input array.

    Returns:
    - float: The product of array elements.
    """ 
    return np.prod(point)


def dxdtau(A: np.ndarray, tau: float) -> np.ndarray:
    """Calculate derivative of interior fixed point as a function of tau.

    Parameters:
    - A (np.ndarray): Input matrix.
    - tau (float): A parameter.

    Returns:
    - np.ndarray: The calculated derivative.
    """
    M = tau * A + np.identity(len(A))
    prod1 = linalg.solve(-M, np.ones(len(A)), assume_a='sym')
    prod2 = linalg.solve(M, prod1, assume_a='sym')
    return A @ prod2


def dFdtau(A: np.ndarray, tau: float, point: np.ndarray) -> float:
    """Calculate derivative of functional as a function of tau.

    Parameters:
    - A (np.ndarray): Input matrix.
    - tau (float): A parameter.
    - point (np.ndarray): Input array.

    Returns:
    - float: The calculated derivative.
    """
    derivatives = dxdtau(A, tau)
    total_product = np.prod(point)

    if min(np.abs(point)) > 1e-100:
        products = total_product / point
        return np.sum(derivatives * products)
    else:
        products = np.array([np.prod([point[j] for j in range(len(point)) if j != i]) for i in range(len(point))])
        return np.sum(derivatives * products)

def get_first_bifurcation(G: nx.Graph, tau_initial: float, tolerance: float, regular=False) -> float:
    """Find bifurcation tau.

    Parameters:
    - G (nx.Graph): The input graph.
    - tau_initial (float): Initial value of tau.
    - tolerance (float): Tolerance value.
    - regular (bool): Regular or not.

    Returns:
    - float: The bifurcation tau.
    """
    A = nx.to_numpy_array(G)
    tau_max = 1 / abs(np.linalg.eigvalsh(A)[0])
    dt = 1e200

    tau = tau_initial
    iteration = 0
    max_iterations = 500

    if regular:
        return tau_max, True

    while dt > tolerance and tau < tau_max and iteration < max_iterations:
        M = tau * A + np.identity(len(G))

        try:
            point = linalg.solve(M, np.ones(len(A)), assume_a='sym')
            dFdtau_value = dFdtau(A, tau, point)

            if abs(dFdtau_value) < 1e-200:
                break

            tau1 = tau - F(point) / dFdtau_value

        except:
            return tau_max, True

        dt = abs(tau1 - tau)
        tau = tau1
        iteration += 1

    if abs(tau - tau_max) < 1e-5:
        return tau_max, True
    else:
        if min(point) < 1e-3:
            return min(tau, tau_max), min(tau, tau_max) == tau_max
        else:
            return tau_max, True