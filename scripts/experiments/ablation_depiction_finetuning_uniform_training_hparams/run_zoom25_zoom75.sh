#!/usr/bin/env bash
# zoom_25 + zoom_75 depiction finetuning only (3 seeds × 4 MPP classification datasets).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-depiction-finetuning-uniform-hparams}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"

python scripts/ablation_study_run.py \
  --registry configs/ablation_study/depiction_image_ft/group_registry.json \
  --task-dir configs/ablation_study/shared/datasets \
  --runseed-start "${RUNSEED_START}" \
  --runseed-end "${RUNSEED_END}" \
  --group image_chemvl_ft_zoom25 \
  --group image_chemvl_ft_zoom75 \
  --finetune-script finetune_moleculenet.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "$@"
