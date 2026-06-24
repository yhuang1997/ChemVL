#!/usr/bin/env bash
# ImageMol under ChemVL — random_scaffold split batch.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${_SCRIPT_DIR}/../_common.sh"
external_baseline_repo_root "$_SCRIPT_DIR"
external_baseline_export_data_root
external_baseline_resolve_default_python

EXP_NAME="${EXP_NAME:-baseline-under-chemvl-random_scaffold}"
BASE_CONFIG="${BASE_CONFIG:-configs/external/imagemol/imagemol_moleculenet_random_scaffold.external.json}"
DATASET_LIST="${DATASET_LIST:-configs/external/moleculenet/dataset_list_moleculenet10.txt}"
TASK_DIR="${TASK_DIR:-configs/external/moleculenet/datasets}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
BATCH="${REPO_ROOT}/scripts/external/molclr_under_chemvl/batch_run_external.py"

CKPT_IMAGEMOL="${CHEMVL_DATA_ROOT}/checkpoints/external/ImageMol.pth.tar"
if [[ "${DRY_RUN:-0}" != "1" && ! -f "$CKPT_IMAGEMOL" ]]; then
  echo "Missing checkpoint: $CKPT_IMAGEMOL" >&2
  echo "Run: export CHEMVL_DATA_ROOT=... && python tools/hf_download.py download" >&2
  exit 1
fi

CMD=(
  "$PY" "$BATCH"
  --base-config "$BASE_CONFIG"
  --dataset-list "$DATASET_LIST"
  --task-dir "$TASK_DIR"
  --runseed-start "$RUNSEED_START"
  --runseed-end "$RUNSEED_END"
  --exp-name "$EXP_NAME"
  --python "$PY"
)

if [[ -n "${DATASETS:-}" ]]; then
  CMD+=(--datasets "$DATASETS")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${NO_SKIP:-0}" == "1" ]]; then
  CMD+=(--no-skip-existing)
fi
if [[ "${SAVE_GIT_DIFF:-0}" == "1" ]]; then
  CMD+=(--save-git-diff)
fi

echo "=== ImageMol under ChemVL: random_scaffold exp=${EXP_NAME} ==="
exec "${CMD[@]}" "$@"
