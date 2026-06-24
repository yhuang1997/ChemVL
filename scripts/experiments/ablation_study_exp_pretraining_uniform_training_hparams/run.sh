#!/usr/bin/env bash
# Uniform training hyperparameters grid (pretraining ablation registry).
# Repro metadata: results/finetuning/<EXP_NAME>/_repro/ (written by ablation_study_run.py).
#
# MODALITY selects which experimental groups to run:
#   image     — image_scratch, image_imagemol, image_chemvl
#   graph_gin — graph_gin_scratch, graph_gin_molclr, graph_gin_chemvl
#   graph_gcn — graph_gcn_scratch, graph_gcn_molclr, graph_gcn_chemvl
#   all       — all 9 groups (default)
#
# Examples (extend seeds 4–10 without saving checkpoints):
#   MODALITY=image RUNSEED_START=4 RUNSEED_END=10 bash run.sh --no-save-ckpt
#   MODALITY=graph_gin RUNSEED_START=4 RUNSEED_END=10 bash run.sh --no-save-ckpt
#
# Other machines: set REPO_ROOT to your ChemVL repo clone path.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO_ROOT"

EXP_NAME="${EXP_NAME:-pretraining-uniform-training-hparams}"
MODALITY="${MODALITY:-all}"
RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-10}"

case "$MODALITY" in
  image)
    GROUP_ARGS=(
      --group image_scratch
      --group image_imagemol
      --group image_chemvl
    )
    ;;
  graph_gin)
    GROUP_ARGS=(
      --group graph_gin_scratch
      --group graph_gin_molclr
      --group graph_gin_chemvl
    )
    ;;
  graph_gcn)
    GROUP_ARGS=(
      --group graph_gcn_scratch
      --group graph_gcn_molclr
      --group graph_gcn_chemvl
    )
    ;;
  all)
    GROUP_ARGS=(
      --group image_scratch
      --group image_imagemol
      --group image_chemvl
      --group graph_gin_scratch
      --group graph_gin_molclr
      --group graph_gin_chemvl
      --group graph_gcn_scratch
      --group graph_gcn_molclr
      --group graph_gcn_chemvl
    )
    ;;
  *)
    echo "Unknown MODALITY=${MODALITY!r} (expected: image, graph_gin, graph_gcn, all)" >&2
    exit 1
    ;;
esac

python scripts/ablation_study_run.py \
  --runseed-start "${RUNSEED_START}" \
  --runseed-end "${RUNSEED_END}" \
  "${GROUP_ARGS[@]}" \
  --finetune-script finetune_moleculenet.py \
  --uniform-training-hparams \
  --exp-name "${EXP_NAME}" \
  "$@"
