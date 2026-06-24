#!/usr/bin/env bash
# Aggregate test metrics for depiction finetuning runs (same layout as ablation_study_analyze expects).
#
# RESULT_ROOT must equal: join(basic.log_dir_base from your FT configs, EXP_NAME)
# Default matches image_chemvl_ft_* configs + EXP_NAME from run.sh.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR_BASE="${LOG_DIR_BASE:-/mnt/d/wsl-data/chemvl/results/pretraining_ablation}"
EXP_NAME="${EXP_NAME:-depiction-finetuning-uniform-hparams}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

python scripts/ablation_study_analyze.py \
  --root "${RESULT_ROOT}" \
  --group-prefix ft_depiction_image_chemvl \
  --datasets bace,bbbp,clintox,tox21 \
  --out-stem depiction_image_chemvl_ft \
  "$@"
