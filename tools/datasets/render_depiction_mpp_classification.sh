#!/usr/bin/env bash
# Pre-render non-default depiction presets for MPP classification datasets (demo / production).
# Requires RDKit and repo Python path. Set DATAROOT to .../MPP/classification.
#
# Usage:
#   export DATAROOT=/path/to/finetuning_datasets/MPP/classification
#   bash tools/datasets/render_depiction_mpp_classification_demo.sh
#
# Optional: CANVAS=224 PRESETS="layout_var style_var zoom_50" DATASETS="bbbp bace tox21 clintox"
set -euo pipefail

# Repo root: this script lives in tools/datasets/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHEMVL_DATA_ROOT:?Set CHEMVL_DATA_ROOT to your data root directory}"
DATAROOT="${DATAROOT:-${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification}"

cd "$REPO_ROOT"

CANVAS="${CANVAS:-224}"
PRESETS="${PRESETS:-layout_var style_var zoom_50}"
DATASETS="${DATASETS:-bbbp bace tox21 clintox}"

for ds in $DATASETS; do
  for preset in $PRESETS; do
    echo "==> dataset=${ds} preset=${preset}"
    python tools/datasets/render_depiction_dataset.py \
      --dataroot "$DATAROOT" \
      --dataset "$ds" \
      --render-canvas-px "$CANVAS" \
      --render-preset "$preset"
  done
done

echo "Done."
