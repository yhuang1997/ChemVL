#!/usr/bin/env bash
# Shared setup for baseline-under-chemvl shell entrypoints (source, do not execute).
set -euo pipefail

external_baseline_repo_root() {
  local script_dir="$1"
  REPO_ROOT="${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
  cd "$REPO_ROOT"
}

external_baseline_export_data_root() {
  : "${CHEMVL_DATA_ROOT:?Set CHEMVL_DATA_ROOT to your data root directory}"
  export CHEMVL_DATA_ROOT
}

external_baseline_resolve_default_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
  else
    PY="python3"
  fi
}

external_baseline_resolve_molmcl_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
  elif [[ -x "${HOME}/miniconda3/envs/molmcl/bin/python" ]]; then
    PY="${HOME}/miniconda3/envs/molmcl/bin/python"
  elif [[ -x "${HOME}/anaconda3/envs/molmcl/bin/python" ]]; then
    PY="${HOME}/anaconda3/envs/molmcl/bin/python"
  else
    PY="python3"
  fi
}

external_baseline_check_molmcl_checkpoints() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  local missing=0
  for ckpt in \
    "${CHEMVL_DATA_ROOT}/checkpoints/external/zinc-gps_best.pt" \
    "${CHEMVL_DATA_ROOT}/checkpoints/external/zinc-gnn_best.pt"; do
    if [[ ! -f "$ckpt" ]]; then
      echo "Missing checkpoint: $ckpt" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Run: export CHEMVL_DATA_ROOT=... && python tools/hf_download.py download" >&2
    exit 1
  fi
}
