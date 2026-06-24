#!/usr/bin/env bash
# Plot finetuning trajectories (requires epoch history CSVs from extensive_finetune / similar).
#
# Other machines: set REPO_ROOT to your ChemVL repo clone path.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-pretraining-uniform-training-hparams}"
RESULT_ROOT="/mnt/d/wsl-data/chemvl/results/pretraining_ablation/${EXP_NAME}"

# Short names after stripping group-prefix (matches directories pretrain_ablation_*).
GROUPS_FILTER="${GROUPS_FILTER:-image_scratch,image_imagemol,image_chemvl}"

python scripts/ablation_study_plot_trajectories.py \
  --root "${RESULT_ROOT}" \
  --group-prefix pretrain_ablation \
  --datasets bace,bbbp,clintox,tox21 \
  --groups "${GROUPS_FILTER}" \
  --out-stem pretrain_ablation_image \
  "$@"
