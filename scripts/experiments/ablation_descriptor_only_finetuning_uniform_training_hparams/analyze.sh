#!/usr/bin/env bash
# Aggregate test metrics for descriptor-only finetuning runs.
#
# RESULT_ROOT must equal: join(basic.log_dir_base from FT configs, EXP_NAME)
# Default matches descriptor_only_ft_base.json + EXP_NAME from run.sh.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"


LOG_DIR_BASE="${LOG_DIR_BASE:-/mnt/d/wsl-data/chemvl/results/ablation_descriptor_only}"
EXP_NAME="${EXP_NAME:-descriptor-only-ablation-uniform-hparams}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

python scripts/ablation_study_analyze.py \
  --root "${RESULT_ROOT}" \
  --group-prefix ft_descriptor_only_feature \
  --datasets bace,bbbp,clintox,tox21 \
  --out-stem ft_descriptor_only_feature \
  "$@"
