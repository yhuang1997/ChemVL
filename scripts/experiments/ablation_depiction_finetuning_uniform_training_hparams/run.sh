#!/usr/bin/env bash
# Depiction image finetuning grid: 4 presets + baseline with train-time multi-view (registry in image_chemvl_ft).
#
# Override paths on other machines:
#   REPO_ROOT, LOG_DIR_BASE (must match configs/ablation_study/depiction_image_ft/*.json basic.log_dir_base),
#   EXP_NAME (nested under LOG_DIR_BASE via ablation_study_run --exp-name).
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
  --group image_chemvl_ft_baseline \
  --group image_chemvl_ft_layoutVar \
  --group image_chemvl_ft_styleVar \
  --group image_chemvl_ft_zoom50 \
  --group image_chemvl_ft_baseline_multiview \
  --finetune-script finetune_moleculenet.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "$@"
