#!/usr/bin/env bash
# MolMCL under ChemVL — MoleculeNet random_scaffold batch (GIN/GPS × cls/reg).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${_SCRIPT_DIR}/../_common.sh"
external_baseline_repo_root "$_SCRIPT_DIR"
external_baseline_export_data_root
external_baseline_resolve_molmcl_python
external_baseline_check_molmcl_checkpoints

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXP_NAME="${EXP_NAME:-molmcl_moleculenet_smoke_random_scaffold}"
else
  EXP_NAME="${EXP_NAME:-baseline-under-chemvl-random_scaffold}"
fi
DATASET_LIST_CLS="${DATASET_LIST_CLS:-configs/external/molmcl/dataset_list_moleculenet_cls6.example.txt}"
DATASET_LIST_REG="${DATASET_LIST_REG:-configs/external/molmcl/dataset_list_moleculenet_reg4.example.txt}"
TASK_DIR="${TASK_DIR:-configs/external/moleculenet/datasets}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
BATCH="${_SCRIPT_DIR}/batch_run_external.py"
RUN_GIN="${RUN_GIN:-1}"
RUN_GPS="${RUN_GPS:-1}"
RUN_CLS="${RUN_CLS:-1}"
RUN_REG="${RUN_REG:-1}"

EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA+=(--dry-run)
fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then
  EXTRA+=(--no-skip-existing)
fi
if [[ "${SAVE_GIT_DIFF:-0}" == "1" ]]; then
  EXTRA+=(--save-git-diff)
fi

_run_batch() {
  local base="$1"
  local list="$2"
  echo "--- ${base} (list=${list}) ---"
  if [[ -n "${DATASETS:-}" ]]; then
    "$PY" "$BATCH" \
      --base-config "$base" \
      --datasets "$DATASETS" \
      --task-dir "$TASK_DIR" \
      --runseed-start "$RUNSEED_START" \
      --runseed-end "$RUNSEED_END" \
      --exp-name "$EXP_NAME" \
      --python "$PY" \
      "${EXTRA[@]}"
  else
    "$PY" "$BATCH" \
      --base-config "$base" \
      --dataset-list "$list" \
      --task-dir "$TASK_DIR" \
      --runseed-start "$RUNSEED_START" \
      --runseed-end "$RUNSEED_END" \
      --exp-name "$EXP_NAME" \
      --python "$PY" \
      "${EXTRA[@]}"
  fi
}

echo "=== MolMCL under ChemVL: random_scaffold exp=${EXP_NAME} RUN_GIN=${RUN_GIN} RUN_GPS=${RUN_GPS} PY=${PY} ==="

_fail=0
if [[ "$RUN_GIN" == "1" ]]; then
  if [[ "$RUN_CLS" == "1" ]]; then
    _run_batch configs/external/molmcl/molmcl_gin_moleculenet_classification_random-scaffold.external.json "$DATASET_LIST_CLS" || _fail=1
  fi
  if [[ "$RUN_REG" == "1" ]]; then
    _run_batch configs/external/molmcl/molmcl_gin_moleculenet_regression_random-scaffold.external.json "$DATASET_LIST_REG" || _fail=1
  fi
fi
if [[ "$RUN_GPS" == "1" ]]; then
  if [[ "$RUN_CLS" == "1" ]]; then
    _run_batch configs/external/molmcl/molmcl_gps_moleculenet_classification_random-scaffold.external.json "$DATASET_LIST_CLS" || _fail=1
  fi
  if [[ "$RUN_REG" == "1" ]]; then
    _run_batch configs/external/molmcl/molmcl_gps_moleculenet_regression_random-scaffold.external.json "$DATASET_LIST_REG" || _fail=1
  fi
fi
if [[ "$RUN_GIN" != "1" && "$RUN_GPS" != "1" ]]; then
  echo "Both RUN_GIN and RUN_GPS are 0; nothing to run." >&2
  exit 1
fi
if [[ "$RUN_CLS" != "1" && "$RUN_REG" != "1" ]]; then
  echo "Both RUN_CLS and RUN_REG are 0; nothing to run." >&2
  exit 1
fi
exit "$_fail"
