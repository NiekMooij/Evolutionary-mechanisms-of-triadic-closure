#!/usr/bin/env bash
set -euo pipefail

# Python scripts to run for each habitat
SCRIPTS=(
  "generate_networks.py"
  "sample_degree_sequences.py"
  "compare_clustering.py"
)

# List of habitat codes you want to process
HABITATS=("R1" "R2" "R3" "N1" "Q2" "Q5" "S9")

for H in "${HABITATS[@]}"; do
  for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Running ${SCRIPT} for habitat ${H}..."
    python "${SCRIPT}" --habitat-code "${H}"
  done
done
