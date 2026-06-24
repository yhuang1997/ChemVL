#!/usr/bin/env bash
# Aggregate dummy prompt learner KGPT runs (random fixed query variant only).
#
# Learnable baseline lives under ablation_dummy_structure; use aggregate presets for side-by-side.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR_BASE="${LOG_DIR_BASE:-/mnt/d/wsl-data/chemvl/results/ablation_dummy_prompt_learner}"
EXP_NAME="${EXP_NAME:-dummy-prompt-learner-kgpt-sota-hparams}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_DIR_BASE}/${EXP_NAME}}"

python scripts/ablation_study_analyze.py \
  --root "${RESULT_ROOT}" \
  --group-prefix kgpt_dummy_prompt \
  --datasets bace,bbbp,clintox,tox21 \
  --out-stem dummy_prompt_learner_kgpt \
  "$@"
