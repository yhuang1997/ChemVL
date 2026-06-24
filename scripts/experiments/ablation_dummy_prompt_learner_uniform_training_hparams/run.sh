#!/usr/bin/env bash
# Dummy prompt learner ablation: 6 local KGPT groups (2 prompt x 2 structure-freeze states).
# Learnable baseline is NOT rerun; compare via aggregate presets against
# kgpt-sota-hparams (run_kgpt_sota.sh).
#
# Override: REPO_ROOT, EXP_NAME, RUNSEED_*, PARAMS_YAML.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-dummy-prompt-learner-kgpt-sota-hparams}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"
PARAMS_YAML="${PARAMS_YAML:-${_SCRIPT_DIR}/params_best.yaml}"

EXTRA_ARGS=()
if [[ -f "${PARAMS_YAML}" ]]; then
  EXTRA_ARGS+=(--training-hparams-yaml "${PARAMS_YAML}")
fi

python scripts/ablation_study_run.py \
  --registry configs/ablation_study/dummy_prompt_learner/group_registry.json \
  --task-dir configs/ablation_study/shared/datasets \
  --runseed-start "${RUNSEED_START}" \
  --runseed-end "${RUNSEED_END}" \
  --group image_chemvl_kgpt_random_fixed_query \
  --group graph_gin_chemvl_kgpt_random_fixed_query \
  --group image_chemvl_kgpt_random_fixed_query_structure_frozen \
  --group graph_gin_chemvl_kgpt_random_fixed_query_structure_frozen \
  --group image_chemvl_kgpt_learnable_query_structure_frozen \
  --group graph_gin_chemvl_kgpt_learnable_query_structure_frozen \
  --finetune-script finetune_moleculenet.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
