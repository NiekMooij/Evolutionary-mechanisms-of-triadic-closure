#!/usr/bin/env python3
"""
Count vascular plant species per environment (habitat_code + var_name).

Environment definition matches the logic in the sampling script, but we work
with *two-character* habitat codes only, e.g.:

    R21, R22, R23, ...  -> habitat_code "R2"
    S11, S12, ...       -> habitat_code "S1"

Steps per (habitat_code, var_name):

1. Determine plots whose habitat matches a given two-character habitat_code:
   - Use df_habitat["habitat"] values that start with that two-character code.
   - This uses the same helper as the sampling script, with prefix matching.

2. From species_df, select rows:
   - plot_ID is in the selected plots
   - vascular_plant == 1
   - Take distinct species_new -> "vascular species in habitat"

3. From species_niche_df, select rows:
   - var_name equals the given var_name
   - species_new is in the "vascular species in habitat"
   - Drop duplicate species_new
   -> "species with niche data for this habitat_code + var_name"

For all combinations of (two-char habitat_code, var_name), we count:
- n_vascular_species_in_habitat
- n_species_with_niche_for_var

Results are written to a CSV table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
# Core helpers (mirroring logic from the sampling script)
# =============================================================================


def _get_plot_ids_for_habitat_prefix(
    df_habitat: pd.DataFrame,
    habitat_code: str,
) -> Tuple[np.ndarray, Dict]:
    """
    Match plots by habitat code using the same logic as the sampling script:

    - First try exact matches on df_habitat["habitat"].
    - If none are found, use prefix-matching: startswith(habitat_code).
    """
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
    matched_habitats = (
        df_habitat.loc[mask, "habitat"].dropna().astype(str).unique().tolist()
    )

    meta = {
        "habitat_code_input": habitat_code,
        "habitat_match_mode": match_mode,
        "matched_habitats": matched_habitats,
        "num_matched_habitats": len(matched_habitats),
    }
    return plot_ids, meta


def count_species_for_habitat_and_var(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,
    species_niche_df: pd.DataFrame,
    habitat_code: str,
    var_name: str,
) -> Dict:
    """
    Count species for a single (two-char habitat_code, var_name) environment.

    Returns a dict with:
      - habitat_code_input
      - habitat_match_mode
      - matched_habitats
      - num_matched_habitats
      - var_name
      - n_vascular_species_in_habitat
      - n_species_with_niche_for_var
      - notes (if something is missing / zero)
    """
    plot_ids, habitat_meta = _get_plot_ids_for_habitat_prefix(
        df_habitat, habitat_code
    )

    result: Dict = {
        **habitat_meta,
        "var_name": var_name,
        "n_vascular_species_in_habitat": 0,
        "n_species_with_niche_for_var": 0,
        "notes": "",
    }

    if len(plot_ids) == 0:
        result["notes"] = (
            f'No plots found for habitat code "{habitat_code}" (no exact or prefix matches).'
        )
        return result

    if "vascular_plant" not in species_df.columns:
        raise KeyError("species_df must contain a 'vascular_plant' column.")
    if "plot_ID" not in species_df.columns:
        raise KeyError("species_df must contain a 'plot_ID' column.")
    if "species_new" not in species_df.columns:
        raise KeyError("species_df must contain a 'species_new' column.")

    # Vascular plants in the matched plots
    vp = pd.to_numeric(species_df["vascular_plant"], errors="coerce").fillna(0).astype(int)
    species_in_habitat = (
        species_df.loc[
            species_df["plot_ID"].isin(plot_ids) & (vp == 1),
            "species_new",
        ]
        .dropna()
        .drop_duplicates()
    )

    n_vascular = len(species_in_habitat)
    result["n_vascular_species_in_habitat"] = int(n_vascular)

    if n_vascular == 0:
        result["notes"] = (
            "No vascular-plant species found in plots for this habitat code."
        )
        return result

    # Niche data for this var_name
    if "var_name" not in species_niche_df.columns:
        raise KeyError("species_niche_df must contain a 'var_name' column.")
    if "species_new" not in species_niche_df.columns:
        raise KeyError("species_niche_df must contain a 'species_new' column.")

    niche_var = species_niche_df.loc[
        species_niche_df["var_name"].astype(str).eq(str(var_name))
    ].copy()

    if niche_var.empty:
        result["notes"] = f'No niche data found for var_name "{var_name}".'
        return result

    niche_for_species = niche_var[niche_var["species_new"].isin(species_in_habitat)].copy()

    if niche_for_species.empty:
        result["notes"] = (
            "No overlap between habitat vascular species and niche data for this var_name."
        )
        return result

    # Ensure one row per species_new (if needed)
    niche_for_species = niche_for_species.sort_values(
        by=list(niche_for_species.columns)
    ).drop_duplicates(subset=["species_new"], keep="first")

    n_with_niche = len(niche_for_species)
    result["n_species_with_niche_for_var"] = int(n_with_niche)

    return result


# =============================================================================
# High-level workflow
# =============================================================================


def _two_char_habitat_codes(df_habitat: pd.DataFrame) -> List[str]:
    """
    Derive unique two-character habitat codes from df_habitat["habitat"].

    Example:
        R21, R22, R23 -> "R2"
        S11, S12      -> "S1"
    """
    if "habitat" not in df_habitat.columns:
        raise KeyError("df_habitat must contain a 'habitat' column.")

    habitats = df_habitat["habitat"].dropna().astype(str)
    codes = (
        habitats.str.slice(0, 2)  # stop at the second character
        .dropna()
        .unique()
        .tolist()
    )
    codes = sorted(codes)
    return codes


def enumerate_environments_and_count(
    df_habitat: pd.DataFrame,
    species_df: pd.DataFrame,
    species_niche_df: pd.DataFrame,
    *,
    habitat_codes: Optional[List[str]] = None,
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For all combinations of (two-char habitat_code, var_name), count species.

    If habitat_codes is None, use all unique two-character prefixes derived
    from df_habitat["habitat"] (R21, R22, ... -> R2, etc).

    If var_names is None, use all unique species_niche_df["var_name"] values.
    """
    if habitat_codes is None:
        habitat_codes = _two_char_habitat_codes(df_habitat)

    if var_names is None:
        if "var_name" not in species_niche_df.columns:
            raise KeyError("species_niche_df must contain a 'var_name' column.")
        var_names = (
            species_niche_df["var_name"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

    records: List[Dict] = []
    for habitat_code in habitat_codes:
        for var_name in var_names:
            rec = count_species_for_habitat_and_var(
                df_habitat, species_df, species_niche_df, habitat_code, var_name
            )
            records.append(rec)

    return pd.DataFrame.from_records(records)


# =============================================================================
# Run configuration
# =============================================================================

CONFIG = {
    "habitat_csv": "data/Globnut1.0_EUNIS.csv",
    "species_csv": "data/Globnut_spec_Niek.csv",
    "niche_csv": "data/Species_niche_Niek.csv",
    "encoding": "latin1",
    # If you want to restrict to a subset, fill these; None => auto-detect all
    # habitat_codes should be two-character codes like "R2", "S1", ...
    "habitat_codes": None,  # e.g. ["R2", "R3"]
    "var_names": None,      # e.g. ["NP", "N", "P"]
    # Output CSV (None => inferred next to this script)
    "out_csv": None,
}


def main(config: dict = CONFIG) -> None:
    df_habitat, species_df, species_niche_df = load_inputs(
        config["habitat_csv"],
        config["species_csv"],
        config["niche_csv"],
        species_csv_encoding=config["encoding"],
    )

    counts_df = enumerate_environments_and_count(
        df_habitat=df_habitat,
        species_df=species_df,
        species_niche_df=species_niche_df,
        habitat_codes=config["habitat_codes"],
        var_names=config["var_names"],
    )

    script_dir = Path(__file__).resolve().parent
    out_csv = (
        script_dir / "environment_species_counts_twochar.csv"
        if config["out_csv"] is None
        else Path(config["out_csv"])
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(out_csv, index=False)

    print(f"Saved environment species counts (two-char habitats) to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
