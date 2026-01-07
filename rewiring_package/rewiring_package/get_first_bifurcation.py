import numpy as np
import networkx as nx
from scipy import linalg
from typing import Tuple

def F(point: np.ndarray) -> float:
    """Product of entries of x."""
    return float(np.prod(point))

def dxdtau(A: np.ndarray, tau: float) -> np.ndarray:
    """d x*/d tau where x* solves (I + tau A) x = 1."""
    n = A.shape[0]
    M = tau * A + np.eye(n)
    # Solve in two steps to avoid forming explicit inverses
    try:
        prod1 = linalg.solve(-M, np.ones(n), assume_a='sym')
        prod2 = linalg.solve(M, prod1, assume_a='sym')
    except linalg.LinAlgError:
        # Singular or ill-conditioned; propagate NaNs to trigger safe fallback
        return np.full(n, np.nan)
    return A @ prod2

def dFdtau(A: np.ndarray, tau: float, point: np.ndarray) -> float:
    """d/d tau of F(x*(tau)) with F(x)=∏ x_i and (I+tau A) x = 1."""
    deriv = dxdtau(A, tau)
    if not np.all(np.isfinite(deriv)):
        return np.nan

    # Use stable product logic
    total_prod = float(np.prod(point))
    # If all coordinates are safely away from zero, use division trick
    if np.min(np.abs(point)) > 1e-100:
        return float(np.sum(deriv * (total_prod / point)))
    # Otherwise compute leave-one-out products
    n = point.size
    loo = np.empty(n, dtype=float)
    for i in range(n):
        # product of all point[j] for j != i
        loo[i] = float(np.prod(point[np.arange(n) != i]))
    return float(np.sum(deriv * loo))

def _tau_max_from_spectrum(A: np.ndarray) -> float:
    """
    Original code used: tau_max = 1 / |lambda_min(A)| with lambda_min the most negative eigenvalue.
    If lambda_min == 0, tau_max = inf.
    """
    evals = np.linalg.eigvalsh(A)  # sorted ascending
    if evals.size == 0:
        return np.inf
    lam_min = float(evals[0])
    if lam_min == 0.0:
        return np.inf
    return 1.0 / abs(lam_min)

def get_first_bifurcation(
    G: nx.Graph, tau_initial: float, tolerance: float, regular: bool = False
) -> Tuple[float, bool]:
    """
    Find the first bifurcation tau for (I + tau A) x = 1 with functional F(x) = ∏ x_i.

    Returns:
        (tau*, hit_tau_max_flag)
        - tau*: estimated bifurcation parameter (finite number or +inf)
        - hit_tau_max_flag: True if we conclude tau=tau_max boundary was reached
    """
    A = nx.to_numpy_array(G, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.inf, True

    tau_max = _tau_max_from_spectrum(A)

    # "Regular" shortcut per original code
    if regular:
        return tau_max, True

    # If tau_max is infinite or tau_initial is already outside the domain, return boundary
    if not np.isfinite(tau_max) or tau_initial >= tau_max:
        return tau_max, True

    # Newton iteration on tau
    tau = float(tau_initial)
    max_iterations = 500
    last_point = None  # track last feasible point
    dt = np.inf

    # Optional damping to avoid wild steps; keep within [0, tau_max]
    def _newton_step(tau_cur: float, f_val: float, df_val: float) -> float:
        step = -f_val / df_val
        tau_next = tau_cur + step
        # clip to domain
        tau_next = max(0.0, min(tau_next, tau_max))
        # if numerically identical, add tiny nudge
        if tau_next == tau_cur:
            tau_next = min(tau_max, tau_cur + max(1e-12, abs(tau_cur) * 1e-12))
        return tau_next

    for it in range(max_iterations):
        # Build M and attempt solve
        M = tau * A + np.eye(n)
        try:
            point = linalg.solve(M, np.ones(n), assume_a='sym')
        except linalg.LinAlgError:
            # Singular/ill-conditioned at current tau: treat as boundary reached
            return tau_max, True

        last_point = point  # save for post-criteria

        dF = dFdtau(A, tau, point)
        if not np.isfinite(dF) or abs(dF) < 1e-200:
            # Derivative vanished or invalid -> stop; decide using current tau/point
            break

        # Newton update
        f_val = F(point)
        tau_next = _newton_step(tau, f_val, dF)

        dt = abs(tau_next - tau)
        tau = tau_next

        # Convergence or boundary checks
        if dt <= tolerance:
            break
        if tau >= tau_max - 1e-12:
            tau = tau_max
            break

    # Post-processing: classify the outcome
    if not np.isfinite(tau_max):
        return np.inf, True

    if abs(tau - tau_max) < 1e-5:
        return tau_max, True

    # If we never solved for 'point' (shouldn't happen with checks above), treat as boundary
    if last_point is None:
        return tau_max, True

    # Heuristic from original code: if any coordinate is small, consider interior bifurcation found
    if np.min(last_point) < 1e-3:
        tau_star = min(tau, tau_max)
        return tau_star, (tau_star == tau_max)

    # Default: we didn't detect an interior root; report boundary
    return tau_max, True
