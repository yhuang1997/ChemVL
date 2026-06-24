#!/usr/bin/env bash
# Aggregate test metrics + bar chart for runs under this exp's result root.
#
# Other machines: set REPO_ROOT to your ChemVL repo clone path.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR_BASE="${LOG_DIR_BASE:-/mnt/d/wsl-data/chemvl/results/pretraining_ablation}"
EXP_NAME="${EXP_NAME:-pretraining-uniform-training-hparams}"
RESULT_ROOT="${LOG_DIR_BASE}/${EXP_NAME}"

python scripts/ablation_study_analyze.py \
  --root "${RESULT_ROOT}" \
  --group-prefix pretrain_ablation \
  --datasets bace,bbbp,clintox,tox21 \
  --out-stem pretrain_ablation \
  "$@"
