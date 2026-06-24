#!/usr/bin/env bash
# Descriptor-only ablation: @feature (RDKit + MLP). Public registry has two groups only.
#
# Override: REPO_ROOT, EXP_NAME, RUNSEED_*; ensure CHEMVL_DATA_ROOT for descriptor cache.
#
# Group selection (explicit --group in "$@" wins; else PRIOR_DESCRIPTOR_VERSION):
#   all    -> descriptor_only_feature_all     (default)
#   v2_107 -> descriptor_only_feature_v2_107
#
# Examples:
#   ./run.sh
#   PRIOR_DESCRIPTOR_VERSION=v2_107 ./run.sh
#   ./run.sh --group descriptor_only_feature_all --datasets bace
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-descriptor-only-ablation-uniform-hparams}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-3}"

PRIOR_DESCRIPTOR_VERSION="${PRIOR_DESCRIPTOR_VERSION:-all}"

_resolve_prior_group() {
  case "${PRIOR_DESCRIPTOR_VERSION}" in
    all)    echo "descriptor_only_feature_all" ;;
    v2_107) echo "descriptor_only_feature_v2_107" ;;
    *)
      echo "Unknown PRIOR_DESCRIPTOR_VERSION=${PRIOR_DESCRIPTOR_VERSION} (use: all or v2_107)" >&2
      return 1
      ;;
  esac
}

GROUP="${GROUP:-$(_resolve_prior_group)}"

GROUP_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--group" ]]; then
    GROUP_ARGS=()
    break
  fi
done
if [[ ${#GROUP_ARGS[@]} -eq 0 ]]; then
  GROUP_ARGS=(--group "${GROUP}")
fi

python scripts/ablation_study_run.py \
  --registry configs/ablation_study/descriptor_only/group_registry.json \
  --task-dir configs/ablation_study/shared/datasets \
  --runseed-start "${RUNSEED_START}" \
  --runseed-end "${RUNSEED_END}" \
  "${GROUP_ARGS[@]}" \
  --finetune-script scripts/experiments/ablation_descriptor_only_finetuning_uniform_training_hparams/descriptor_only_finetune.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "$@"
