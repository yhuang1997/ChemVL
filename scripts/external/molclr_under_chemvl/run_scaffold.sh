#!/usr/bin/env bash
# MolCLR under ChemVL — scaffold split batch (4 graph methods × MoleculeNet 10 × runseeds).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${_SCRIPT_DIR}/../_common.sh"
external_baseline_repo_root "$_SCRIPT_DIR"
external_baseline_export_data_root
external_baseline_resolve_default_python

EXP_NAME="${EXP_NAME:-baseline-under-chemvl-scaffold}"
SPLIT="scaffold"
DATASET_LIST="${DATASET_LIST:-configs/external/moleculenet/dataset_list_moleculenet10.txt}"
TASK_DIR="${TASK_DIR:-configs/external/moleculenet/datasets}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
BATCH="${_SCRIPT_DIR}/batch_run_external.py"

CKPT_MOLCLR_GIN="${CHEMVL_DATA_ROOT}/checkpoints/external/MolCLR-GIN.ckpt"
CKPT_MOLCLR_GCN="${CHEMVL_DATA_ROOT}/checkpoints/external/MolCLR-GCN.ckpt"

_missing=0
for ckpt in "$CKPT_MOLCLR_GIN" "$CKPT_MOLCLR_GCN"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    _missing=1
  fi
done
if [[ "${DRY_RUN:-0}" != "1" && "$_missing" -ne 0 ]]; then
  echo "Download MolCLR checkpoints via: python tools/hf_download.py download" >&2
  exit 1
fi

DATASETS_CSV="${DATASETS:-}"
EXTRA=()
if [[ -n "$DATASETS_CSV" ]]; then
  EXTRA+=(--datasets "$DATASETS_CSV")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA+=(--dry-run)
fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then
  EXTRA+=(--no-skip-existing)
fi
if [[ "${SAVE_GIT_DIFF:-0}" == "1" ]]; then
  EXTRA+=(--save-git-diff)
fi

BASE_CONFIGS=(
  configs/external/molclr/molclr_gin_scratch_moleculenet_scaffold.external.json
  configs/external/molclr/molclr_gin_molclr_moleculenet_scaffold.external.json
  configs/external/molclr/molclr_gcn_scratch_moleculenet_scaffold.external.json
  configs/external/molclr/molclr_gcn_molclr_moleculenet_scaffold.external.json
)

echo "=== MolCLR under ChemVL: split=${SPLIT} exp=${EXP_NAME} ==="

_fail=0
for base in "${BASE_CONFIGS[@]}"; do
  echo "--- ${base} ---"
  if ! "$PY" "$BATCH" \
    --base-config "$base" \
    --dataset-list "$DATASET_LIST" \
    --task-dir "$TASK_DIR" \
    --runseed-start "$RUNSEED_START" \
    --runseed-end "$RUNSEED_END" \
    --exp-name "$EXP_NAME" \
    --python "$PY" \
    "${EXTRA[@]}" \
    "$@"; then
    _fail=1
  fi
done
exit "$_fail"
