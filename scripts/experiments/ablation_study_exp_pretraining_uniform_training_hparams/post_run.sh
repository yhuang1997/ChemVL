#!/usr/bin/env bash
# Refresh summaries + aggregate metrics + trajectory plots after training completes.
# Re-run when all 10 runseeds are done to update figures with full statistics.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${DIR}/run.sh"
EXP_DIR="${DIR}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate chemvl

RUNSEED_START="${RUNSEED_START:-1}"
RUNSEED_END="${RUNSEED_END:-10}"

echo "[post_run] rebuild summary (image)..."
MODALITY=image RUNSEED_START="${RUNSEED_START}" RUNSEED_END="${RUNSEED_END}" \
  bash "${SCRIPT}" --rebuild-summary --dry-run

echo "[post_run] rebuild summary (graph_gin)..."
MODALITY=graph_gin RUNSEED_START="${RUNSEED_START}" RUNSEED_END="${RUNSEED_END}" \
  bash "${SCRIPT}" --rebuild-summary --dry-run

echo "[post_run] aggregate bar chart..."
bash "${EXP_DIR}/analyze.sh"

echo "[post_run] trajectory plots..."
bash "${EXP_DIR}/plot_image_lines.sh"
bash "${EXP_DIR}/plot_graph-gin_lines.sh"

echo "[post_run] done."
