#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-${ROOT_DIR}/../CFX-GNN/cfx_gnn/dataset}"

files=(
  bail/bail_graph.bin bail/bail_index.bin
  pokec/pokec_n_graph.bin pokec/pokec_n_index.bin
  pokec/pokec_z_graph.bin pokec/pokec_z_index.bin
)
for file in "${files[@]}"; do
  [[ -f "${SOURCE_DIR}/${file}" ]] || { echo "Missing source cache: ${SOURCE_DIR}/${file}" >&2; exit 2; }
done
mkdir -p "${ROOT_DIR}/cfx_gnn/dataset/bail" "${ROOT_DIR}/cfx_gnn/dataset/pokec"
for file in "${files[@]}"; do cp "${SOURCE_DIR}/${file}" "${ROOT_DIR}/cfx_gnn/dataset/${file}"; done
echo "Imported six local caches. They remain excluded by .gitignore."
