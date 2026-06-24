#!/usr/bin/env bash
# Plot finetuning trajectories for depiction ablation runs (requires train_val_test_history.csv per run).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR_BASE="${LOG_DIR_BASE:-/mnt/d/wsl-data/chemvl/results/pretraining_ablation}"
EXP_NAME="${EXP_NAME:-depiction-finetuning-uniform-hparams}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

# Strip group-prefix to short labels (directories are ft_depiction_image_chemvl_*).
GROUPS_FILTER="${GROUPS_FILTER:-baseline,layoutVar,styleVar,zoom50,baseline_multiview}"

python scripts/ablation_study_plot_trajectories.py \
  --root "${RESULT_ROOT}" \
  --group-prefix ft_depiction_image_chemvl \
  --datasets bace,bbbp,clintox,tox21 \
  --groups "${GROUPS_FILTER}" \
  --out-stem depiction_image_chemvl_ft \
  "$@"
