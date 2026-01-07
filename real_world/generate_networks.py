#!/usr/bin/env python3
"""
Sample vascular plant species by habitat, build interaction matrices, sparsify, and
write adjacency networks. Files per sample (in sim_results/<prefix>/networks/):
- <base>_sampled_species.csv
- <base>_interaction_full.csv      (diagonal=0)
- <base>_interaction.csv           (after zeroing weakest edges)
- <base>_adjacency.csv             (0/1 derived from sparse interaction)
- <base>_meta.json

Dependencies: pandas, numpy, scipy, tqdm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Data loading
# =============================================================================

def load_inputs(
    habitat_csv_path: str,
    species_csv_path: str,
    niche_csv_path: str,
    species_csv_encoding: str = "latin1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three input tables."""
    df_habitat = pd.read_csv(habitat_csv_path)
    species_df = pd.read_csv(species_csv_path, encoding=species_csv_encoding)
    species_niche_df = pd.read_csv(niche_csv_path)
    return df_habitat, species_df, species_niche_df


# =============================================================================
# Sampling utilities
# =============================================================================

def _get_plot_ids_for_habitat_prefix(
    df_habitat: pd.DataFrame,
    habitat_code: str,
) -> Tuple[np.ndarray, Dict]:
    if "habitat" not in df_habitat.columns:
        raise KeyError("df_habitat must contain a 'habitat' column.")
    if "plot_ID" not in df_habitat.columns:
        raise KeyError("df_habitat must contain a 'plot_ID' column.")

    habitats = df_habitat["habitat"].astype(str)

    mask = habitats.eq(habitat_code)
    match_mode = "exact" if mask.any() else "prefix"
    if match_mode == "prefix":
        mask = habitats.str.startswith(str(habitat_code))

    plot_ids = df_habitat.loc[mask, "plot_ID"].dropna().unique()

    matched_habitats = df_habitat.loc[mask, "habitat"].dropna().astype(str).unique().tolist()

    meta = {
        "habitat_code_input": habitat_code,
        "habitat_match_mode": match_mode,
        "matched_habitats": matched_habitats,
        "num_matched_habitats": len(matched_habitats),
    }
    return plot_ids, meta


def sample_species_by_habitat_and_var(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,        # this is GlobNut1.0_species.csv
    species_niche_df: pd.DataFrame,
    habitat_code: str,
    var_name: str,
    n: int,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    def _empty(reason: str, available: int = 0):
        info = {"requested": n, "available": available, "returned": 0, "undersampled": True, "reason": reason}
        info.update(habitat_meta)
        return pd.DataFrame(), info

    plot_ids, habitat_meta = _get_plot_ids_for_habitat_prefix(df_habitat, habitat_code)
    if len(plot_ids) == 0:
        return _empty(f'No plots found for habitat code "{habitat_code}" (no exact or prefix matches).')

    if "vascular_plant" not in species_df.columns:
        raise KeyError("species_df must contain a 'vascular_plant' column.")
    vp = pd.to_numeric(species_df["vascular_plant"], errors="coerce").fillna(0).astype(int)

    species_in_habitat = (
        species_df.loc[species_df["plot_ID"].isin(plot_ids) & (vp == 1), "species_new"]
        .dropna()
        .drop_duplicates()
    )
    if species_in_habitat.empty:
        return _empty(f'No vascular-plant species found in plots for habitat code "{habitat_code}".')

    niche_var = species_niche_df.loc[species_niche_df["var_name"].eq(var_name)].copy()
    if niche_var.empty:
        return _empty(f'No niche data found for var_name "{var_name}".')

    niche_for_species = niche_var[niche_var["species_new"].isin(species_in_habitat)].copy()
    if niche_for_species.empty:
        return _empty("No overlap between habitat vascular species and niche data for the given var_name.")

    niche_for_species = niche_for_species.sort_values(by=list(niche_for_species.columns))
    niche_for_species = niche_for_species.drop_duplicates(subset=["species_new"], keep="first")

    available = len(niche_for_species)
    sampled = niche_for_species.sample(n=min(n, available), replace=False, random_state=random_state).reset_index(drop=True)

    info = {"requested": n, "available": available, "returned": len(sampled), "undersampled": len(sampled) < n}
    info.update(habitat_meta)
    return sampled, info


# =============================================================================
# Interaction matrix (vectorized)
# =============================================================================
def _pianka_overlap_matrix(means: np.ndarray, sds: np.ndarray) -> np.ndarray:
    """Pianka niche-overlap between all pairs of normal resource-use curves.

    For N(m1, s1^2) and N(m2, s2^2) interpreted as continuous resource-use pdfs,
    Pianka's overlap O_ij is:

        O_ij = ∫ f_i(x) f_j(x) dx
               ------------------------------------
               sqrt( ∫ f_i(x)^2 dx * ∫ f_j(x)^2 dx )

    which for normals simplifies to

        O_ij = sqrt(2) * sqrt( s1 * s2 / (s1^2 + s2^2) ) *
               exp( -(m1 - m2)^2 / (2 * (s1^2 + s2^2)) )

    Handles s1 = s2 = 0 specially: if means equal -> overlap = 1, else 0.
    """
    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)

    # Pairwise differences and SDs
    m1 = means[:, None]
    m2 = means[None, :]
    diff = m1 - m2

    s1 = sds[:, None]
    s2 = sds[None, :]

    var_sum = s1**2 + s2**2  # s1^2 + s2^2

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # sqrt(2) * sqrt(s1*s2 / (s1^2 + s2^2))
        prefactor = np.sqrt(2.0) * np.sqrt(s1 * s2 / var_sum)

        # exp( -(m1 - m2)^2 / (2 * (s1^2 + s2^2)) )
        exponent = -0.5 * (diff**2) / var_sum
        overlap = prefactor * np.exp(exponent)

    # Handle s1 = s2 = 0 specially
    mask_zero = (var_sum == 0.0)
    if np.any(mask_zero):
        # If both sds are zero:
        # - same mean -> overlap 1
        # - different mean -> overlap 0
        same_mean = (diff == 0.0).astype(float)
        overlap = overlap.copy()
        overlap[mask_zero] = same_mean[mask_zero]

    # Numerical safety: clip into [0, 1]
    overlap = np.clip(overlap, 0.0, 1.0)

    return overlap

def build_interaction_matrix(sampled: pd.DataFrame) -> pd.DataFrame:
    """Build a symmetric interaction (overlap) matrix with diagonal set to 0."""
    required = {"species_new", "mean", "sd"}
    missing = required - set(sampled.columns)
    if missing:
        raise KeyError(f"`sampled` missing required columns: {sorted(missing)}")

    species = sampled["species_new"].astype(str).tolist()
    means = sampled["mean"].to_numpy(dtype=float)
    sds = sampled["sd"].to_numpy(dtype=float)

    mat = _pianka_overlap_matrix(means, sds)
    np.fill_diagonal(mat, 0.0)
    return pd.DataFrame(mat, index=species, columns=species)


# =============================================================================
# Sparsification
# =============================================================================

def sparsify_by_smallest_abs_edges(
    interaction_df: pd.DataFrame,
    percentage: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zero out the smallest |edge weights| in the upper triangle by a given percentage,
    mirror to keep symmetry, and return (interaction_s, adjacency_01).
    """
    if not (0.0 <= percentage <= 1.0):
        raise ValueError("percentage must be between 0 and 1 inclusive.")

    mat = interaction_df.to_numpy(copy=True)
    n = mat.shape[0]
    if n != interaction_df.shape[1]:
        raise ValueError("interaction_df must be square.")

    triu_i, triu_j = np.triu_indices(n, k=1)
    weights = mat[triu_i, triu_j]

    num_edges = weights.size
    num_to_zero = int(num_edges * percentage)
    if num_to_zero > 0:
        order = np.argsort(np.abs(weights), kind="stable")
        kill_idx = order[:num_to_zero]
        i_kill = triu_i[kill_idx]
        j_kill = triu_j[kill_idx]
        mat[i_kill, j_kill] = 0.0
        mat[j_kill, i_kill] = 0.0

    interaction_s = pd.DataFrame(mat, index=interaction_df.index, columns=interaction_df.columns)
    adjacency_df = (interaction_s != 0).astype(int)
    return interaction_s, adjacency_df


# =============================================================================
# High-level workflows
# =============================================================================

def sample_interaction_matrix(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,
    species_niche_df: pd.DataFrame,
    habitat_code: str,
    var_name: str,
    n_species: int,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    sampled, info = sample_species_by_habitat_and_var(
        df_habitat=df_habitat,
        species_df=species_df,
        species_niche_df=species_niche_df,
        habitat_code=habitat_code,
        var_name=var_name,
        n=n_species,
        random_state=random_state,
    )
    return (pd.DataFrame(), sampled, info) if sampled.empty else (build_interaction_matrix(sampled), sampled, info)


def build_one_adjacency(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,
    species_niche_df: pd.DataFrame,
    *,
    habitat_code: str,
    var_name: str,
    n_species: int,
    percentage_zero: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    interaction_full, sampled, info = sample_interaction_matrix(
        df_habitat=df_habitat,
        species_df=species_df,
        species_niche_df=species_niche_df,
        habitat_code=habitat_code,
        var_name=var_name,
        n_species=n_species,
        random_state=random_state,
    )

    interaction_s = adjacency_df = pd.DataFrame()
    if not interaction_full.empty:
        interaction_s, adjacency_df = sparsify_by_smallest_abs_edges(interaction_full, percentage=percentage_zero)
    return interaction_full, interaction_s, adjacency_df, sampled, info


def _ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_safe(obj):
    """Best-effort conversion of arbitrary objects into JSON-safe structures."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        return str(obj)


def _build_meta(habitat_code, var_name, n_species, percentage_zero, seed, info, interaction_full):
    return {
        "habitat_code": habitat_code,
        "var_name": var_name,
        "n_species": n_species,
        "percentage_zero": percentage_zero,
        "random_state": seed,
        "info": _json_safe(info),
        "species_index": [] if interaction_full.empty else interaction_full.index.tolist(),
    }


def _write_outputs(
    base: str,
    out_dir: Path,
    adjacency_df: pd.DataFrame,
    interaction_full: pd.DataFrame,
    interaction_s: pd.DataFrame,
    sampled: pd.DataFrame,
    meta: Optional[dict],
    *,
    save_interaction: bool,
    save_sampled_species: bool,
    save_metadata: bool,
    print_paths: bool = False,
) -> Dict[str, str]:
    out_dir = _ensure_dir(out_dir)
    files: Dict[str, str] = {}

    adjacency_path = out_dir / f"{base}_adjacency.csv"
    adjacency_df.to_csv(adjacency_path)
    files["adjacency"] = str(adjacency_path)
    if print_paths:
        print(f"Saved adjacency: {adjacency_path}")

    if save_interaction and not interaction_full.empty:
        full_path = out_dir / f"{base}_interaction_full.csv"
        interaction_full.to_csv(full_path)
        files["interaction_full"] = str(full_path)
        if print_paths:
            print(f"Saved interaction (full): {full_path}")

        if not interaction_s.empty:
            sparse_path = out_dir / f"{base}_interaction.csv"
            interaction_s.to_csv(sparse_path)
            files["interaction"] = str(sparse_path)
            if print_paths:
                print(f"Saved interaction (sparse): {sparse_path}")

    if save_sampled_species and not sampled.empty:
        sampled_path = out_dir / f"{base}_sampled_species.csv"
        sampled.to_csv(sampled_path, index=False)
        files["sampled_species"] = str(sampled_path)
        if print_paths:
            print(f"Saved sampled species: {sampled_path}")

    if save_metadata and meta is not None:
        meta_path = out_dir / f"{base}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        files["meta"] = str(meta_path)
        if print_paths:
            print(f"Saved meta: {meta_path}")

    return files


def sample_many_adjacency_matrices(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,
    species_niche_df: pd.DataFrame,
    *,
    habitat_code: str,
    var_name: str,
    n_species: int,
    percentage_zero: float = 0.2,
    num_samples: int = 10,
    seeds: Optional[Iterable[Optional[int]]] = None,
    save_dir: Path | str = "outputs",
    prefix: str = "sample",
    save_interaction: bool = True,
    save_sampled_species: bool = True,
    save_metadata: bool = True,
) -> List[Dict[str, str]]:
    """
    Produce multiple samples and save:
      <base>_sampled_species.csv
      <base>_interaction_full.csv
      <base>_interaction.csv
      <base>_adjacency.csv
      <base>_meta.json
    """
    out_dir = _ensure_dir(save_dir)

    # default seeds: None for all (non-deterministic), or provided list
    if seeds is None:
        seeds = [None] * num_samples
    else:
        seeds = list(seeds)
        if len(seeds) != num_samples:
            raise ValueError("Length of `seeds` must equal `num_samples` if provided.")

    manifest: List[Dict[str, str]] = []

    for k, seed in enumerate(tqdm(seeds, total=num_samples, desc="Sampling networks")):
        interaction_full, interaction_s, adjacency_df, sampled, info = build_one_adjacency(
            df_habitat,
            species_df,
            species_niche_df,
            habitat_code=habitat_code,
            var_name=var_name,
            n_species=n_species,
            percentage_zero=percentage_zero,
            random_state=seed,
        )

        base = f"{prefix}_{k+1:03d}"
        meta = _build_meta(habitat_code, var_name, n_species, percentage_zero, seed, info, interaction_full)
        manifest.append(
            _write_outputs(
                base,
                out_dir,
                adjacency_df,
                interaction_full,
                interaction_s,
                sampled,
                meta,
                save_interaction=save_interaction,
                save_sampled_species=save_sampled_species,
                save_metadata=save_metadata,
            )
        )

    return manifest


# =============================================================================
# Run configuration (set values here instead of using CLI args)
# =============================================================================

# Edit these values to control what gets generated.
CONFIG = {
    "habitat_csv": "data/Globnut1.0_EUNIS.csv",
    "species_csv": "data/Globnut_spec_Niek.csv",
    "niche_csv": "data/Species_niche_Niek.csv",
    "encoding": "latin1",
    "habitat_code": "R1",
    "var_name": "NP",
    "n_species": 100,
    "percentage_zero": 0.9,
    "mode": "batch",          # "single" or "batch"
    "seed": None,             # used only in single mode
    "seeds": None,            # list/iterable for batch; None => random each run
    "num_samples": 30,
    "outdir": None,           # None => sim_results/<prefix>/networks next to this script
    "prefix": None,           # None => "<habitat_code>_<var_name>"
    "save_interaction": True,
    "save_sampled_species": True,
    "save_metadata": True,
}


def main(config: dict = CONFIG):
    df_habitat, species_df, species_niche_df = load_inputs(
        config["habitat_csv"], config["species_csv"], config["niche_csv"], species_csv_encoding=config["encoding"]
    )

    prefix = config["prefix"] or f'{config["habitat_code"]}_{config["var_name"]}'

    script_dir = Path(__file__).resolve().parent
    outdir = Path(config["outdir"]) if config["outdir"] else (script_dir / "sim_results" / prefix / "networks")
    outdir.mkdir(parents=True, exist_ok=True)

    if config["mode"] == "single":
        interaction_full, interaction_s, adjacency_df, sampled, info = build_one_adjacency(
            df_habitat,
            species_df,
            species_niche_df,
            habitat_code=config["habitat_code"],
            var_name=config["var_name"],
            n_species=config["n_species"],
            percentage_zero=config["percentage_zero"],
            random_state=config["seed"],
        )
        meta = _build_meta(
            config["habitat_code"],
            config["var_name"],
            config["n_species"],
            config["percentage_zero"],
            config["seed"],
            info,
            interaction_full,
        )
        _write_outputs(
            prefix,
            outdir,
            adjacency_df,
            interaction_full,
            interaction_s,
            sampled,
            meta,
            save_interaction=config["save_interaction"],
            save_sampled_species=config["save_sampled_species"],
            save_metadata=config["save_metadata"],
            print_paths=True,
        )

    else:  # batch
        manifest = sample_many_adjacency_matrices(
            df_habitat=df_habitat,
            species_df=species_df,
            species_niche_df=species_niche_df,
            habitat_code=config["habitat_code"],
            var_name=config["var_name"],
            n_species=config["n_species"],
            percentage_zero=config["percentage_zero"],
            num_samples=config["num_samples"],
            seeds=config["seeds"],
            save_dir=outdir,
            prefix=prefix,
            save_interaction=config["save_interaction"],
            save_sampled_species=config["save_sampled_species"],
            save_metadata=config["save_metadata"],
        )
        print(f"Wrote {len(manifest)} samples to: {outdir.resolve()}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample vascular plant networks for one habitat."
    )
    p.add_argument("--habitat-code", "-H", help="Override habitat code in CONFIG")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Start from default CONFIG
    cfg = CONFIG.copy()

    if args.habitat_code is not None:
        cfg["habitat_code"] = args.habitat_code

    main(cfg)