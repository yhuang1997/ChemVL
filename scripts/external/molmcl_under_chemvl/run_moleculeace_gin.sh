#!/usr/bin/env bash
# Batch MoleculeACE and/or MoleculeNet under ChemVL (``utils/external/finetune_external.py``).
# Switch benchmark via BASE_CONFIG (see configs/external/molmcl/*.external.json).
#
# Examples:
#   export DATASET_LIST=configs/external/molmcl/dataset_list_moleculeace30.example.txt
#   export BASE_CONFIG=configs/external/molmcl/molmcl_gin_moleculeace.external.json
#   bash scripts/external/molmcl_under_chemvl/run_moleculeace_gin.sh
#
# Optional: DRY_RUN=1 SAVE_GIT_DIFF=1 NO_SKIP=1  (append extra Python flags via "$@").
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${_SCRIPT_DIR}/../_common.sh"
external_baseline_repo_root "$_SCRIPT_DIR"
external_baseline_export_data_root
external_baseline_resolve_molmcl_python
external_baseline_check_molmcl_checkpoints

BASE_CONFIG="${BASE_CONFIG:-configs/external/molmcl/molmcl_gin_moleculeace.external.json}"
EXP_NAME="${EXP_NAME:-molmcl_under_chemvl}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
DATASET_LIST="${DATASET_LIST:-configs/external/molmcl/dataset_list_moleculeace30.example.txt}"
BATCH="${_SCRIPT_DIR}/batch_run_external.py"

CMD=(
  "$PY" "$BATCH"
  --base-config "$BASE_CONFIG"
  --dataset-list "$DATASET_LIST"
  --runseed-start "$RUNSEED_START"
  --runseed-end "$RUNSEED_END"
  --exp-name "$EXP_NAME"
  --python "$PY"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${SAVE_GIT_DIFF:-0}" == "1" ]]; then
  CMD+=(--save-git-diff)
fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then
  CMD+=(--no-skip-existing)
fi

echo "=== MolMCL under ChemVL: gin moleculeace exp=${EXP_NAME} PY=${PY} ==="
exec "${CMD[@]}" "$@"
