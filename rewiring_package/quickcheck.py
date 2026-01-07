#!/usr/bin/env python3
# quickcheck.py
import sys
import math
import traceback
from pathlib import Path
import argparse

import numpy as np
import networkx as nx

# --- Try to import the package under test
try:
    import rewiring_package as rp
except Exception as e:
    print("ERROR: Could not import rewiring_package:", e)
    sys.exit(1)

# Prefer optimise_clustering if available; else fall back to optimise_coupling
OPT_FN_NAME = "optimise_clustering" if hasattr(rp, "optimise_clustering") else "optimise_coupling"
optimise_fn = getattr(rp, OPT_FN_NAME, None)
if optimise_fn is None:
    print("ERROR: Neither optimise_clustering nor optimise_coupling found in rewiring_package.")
    sys.exit(1)

# get_first_bifurcation must exist
if not hasattr(rp, "get_first_bifurcation"):
    # Sometimes it’s inside the package but not exported at top-level; try direct import
    try:
        from rewiring_package.get_first_bifurcation import get_first_bifurcation
    except Exception as e:
        print("ERROR: get_first_bifurcation not found:", e)
        sys.exit(1)
else:
    get_first_bifurcation = rp.get_first_bifurcation


def eig_diagnostics(G: nx.Graph):
    A = nx.to_numpy_array(G)
    evals = np.linalg.eigvalsh(A)
    lam_min = float(evals[0]) if evals.size else float("nan")
    lam_abs_min = float(np.min(np.abs(evals))) if evals.size else float("nan")
    # This matches the code path in get_first_bifurcation
    tau_max = (1.0 / abs(lam_min)) if (evals.size and lam_min != 0) else float("inf")
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
        "components": nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0,
        "min_degree": int(min(dict(G.degree()).values())) if G.number_of_nodes() else None,
        "max_degree": int(max(dict(G.degree()).values())) if G.number_of_nodes() else None,
        "eig_min": lam_min,
        "eig_abs_min": lam_abs_min,
        "tau_max": tau_max,
    }


def safe_bifurcation(G, tau_initial, tolerance, regular=False):
    # try:
    tau, flag = get_first_bifurcation(G, tau_initial, tolerance, regular)
    print(f'tau: {tau}')
    exit()
    return float(tau), bool(flag), None
    # except Exception as e:
    #     return float("nan"), False, e


def run_one(name: str, G: nx.Graph, args):
    print("\n" + "=" * 80)
    print(f"[{name}] Graph: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    diag = eig_diagnostics(G)
    print(
        f"  connected={diag['connected']} comps={diag['components']} "
        f"deg[min,max]=[{diag['min_degree']},{diag['max_degree']}] "
        f"eig_min={diag['eig_min']:.6g} eig_abs_min={diag['eig_abs_min']:.6g} "
        f"tau_max={diag['tau_max']}"
    )

    # Evaluate objective before optimizing
    tau0, flag0, err0 = safe_bifurcation(G, args.tau_initial, args.tolerance, args.regular)
    if err0:
        print("  get_first_bifurcation raised:", repr(err0))
    print(f"  initial tau={tau0} (finite={math.isfinite(tau0)}) regular_flag={flag0}")

    # Try optimize (max)
    print(f"  -> Optimizing ({OPT_FN_NAME}) maximise=True")
    try:
        out_max = optimise_fn(
            G,
            rewire_count=args.rewire_count,
            T=args.temperature,
            optimise=True,
            tau_initial=args.tau_initial,
            tolerance=args.tolerance,
            regular=args.regular,
        )
        obj_arr = out_max.get("objective_arr", [])
        if obj_arr:
            print(f"     max-run: start={obj_arr[0]} end={obj_arr[-1]} (finite_end={math.isfinite(obj_arr[-1])})")
        else:
            print("     max-run: no objective_arr returned")
    except Exception as e:
        print("     max-run ERROR:")
        traceback.print_exc()

    # Try optimize (min)
    print(f"  -> Optimizing ({OPT_FN_NAME}) maximise=False")
    try:
        out_min = optimise_fn(
            G,
            rewire_count=args.rewire_count,
            T=args.temperature,
            optimise=False,
            tau_initial=args.tau_initial,
            tolerance=args.tolerance,
            regular=args.regular,
        )
        obj_arr = out_min.get("objective_arr", [])
        if obj_arr:
            print(f"     min-run: start={obj_arr[0]} end={obj_arr[-1]} (finite_end={math.isfinite(obj_arr[-1])})")
        else:
            print("     min-run: no objective_arr returned")
    except Exception as e:
        print("     min-run ERROR:")
        traceback.print_exc()


def build_graphs(n: int):
    # Keep them small/fast for a smoke test
    graphs = {}
    p = min(0.1, 3.0 / max(n, 1))
    graphs["erdos_renyi"] = nx.erdos_renyi_graph(n, p, seed=1)
    graphs["watts_strogatz"] = nx.watts_strogatz_graph(n, k=max(2, n // 10), p=0.2, seed=2)
    graphs["barabasi_albert"] = nx.barabasi_albert_graph(n, max(1, n // 20), seed=3)
    graphs["random_regular"] = nx.random_regular_graph(d=max(2, n // 15), n=n, seed=4)
    # random_geometric tends to be sparse; radius chosen for moderate degree
    graphs["random_geometric"] = nx.random_geometric_graph(n, radius=0.25, seed=5)
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Quick package smoke test for rewiring_package")
    parser.add_argument("--n", type=int, default=80, help="nodes for synthetic graphs")
    parser.add_argument("--rewire_count", type=int, default=100, help="rewire iterations (small for quick test)")
    parser.add_argument("--temperature", type=float, default=0.001, help="annealing temperature T")
    parser.add_argument("--tau_initial", type=float, default=1.0, help="initial tau for bifurcation")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="tolerance for bifurcation solver")
    parser.add_argument("--regular", action="store_true", help="pass regular=True to get_first_bifurcation")
    args = parser.parse_args()

    print("Using rewiring_package from:", rp.__file__)
    print(f"Using optimiser function: {OPT_FN_NAME}")

    graphs = build_graphs(args.n)
    for name, G in graphs.items():
        run_one(name, G, args)

    print("\nAll done. If any run failed above, see the printed diagnostics.")


if __name__ == "__main__":
    main()
