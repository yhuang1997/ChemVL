#!/usr/bin/env bash
# Misplacement + structure-mask attacks: image RN50 and graph GIN ChemVL FT on four MPP datasets.
#
# Override on other machines: REPO_ROOT, LOG_DIR_BASE (match configs),
# EXP_NAME (nested under log_dir_base via ablation_study_run --exp-name).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-misplacement-mask-ablation-uniform-hparams}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"

python scripts/ablation_study_run.py \
  --registry configs/ablation_study/misplacement_mask/group_registry.json \
  --task-dir configs/ablation_study/shared/datasets \
  --runseed-start "${RUNSEED_START}" \
  --runseed-end "${RUNSEED_END}" \
  --group image_chemvl_ft_misplacement \
  --group image_chemvl_ft_mask_attack_025 \
  --group image_chemvl_ft_mask_attack_050 \
  --group graph_gin_chemvl_ft_misplacement \
  --group graph_gin_chemvl_ft_mask_attack_025 \
  --group graph_gin_chemvl_ft_mask_attack_050 \
  --group image_chemvl_ft_baseline \
  --group graph_gin_chemvl_ft_baseline \
  --finetune-script finetune_moleculenet.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "$@"
