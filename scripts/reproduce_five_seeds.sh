#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
datasets=(bail pokec_n pokec_z)
pids=()

wait_batch() {
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  pids=()
  [[ ${failed} -eq 0 ]] || exit 1
}

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  python "${ROOT_DIR}/run.py" --dataset "${dataset}" --gpu "${gpu}" &
  pids+=("$!")
  [[ ${#pids[@]} -lt ${#GPU_IDS[@]} ]] || wait_batch
done
wait_batch
