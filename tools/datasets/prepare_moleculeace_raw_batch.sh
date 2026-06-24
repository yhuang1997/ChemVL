#!/usr/bin/env bash
# Batch-build ChemVL ``*_processed_ac.csv`` + 224×224 default PNGs from MolMCL-style CSVs
# placed under ``${MOLECULEACE_RAW_DIR}/*.csv`` (columns: smiles, y).
#
# Output layout (per file ``${MOLECULEACE_RAW_DIR}/CHEMBLxxxx_EC50.csv``)::
#   ${MOLECULEACE_DATAROOT}/CHEMBLxxxx_EC50/processed/CHEMBLxxxx_EC50_processed_ac.csv
#   ${MOLECULEACE_DATAROOT}/CHEMBLxxxx_EC50/processed/224/{index}.png
#
# Usage::
#   export CHEMVL_DATA_ROOT=/path/to/your/chemvl-data
#   export MOLECULEACE_RAW_DIR="${CHEMVL_DATA_ROOT}/finetuning_datasets/MoleculeACE/raw"
#   export MOLECULEACE_DATAROOT="${CHEMVL_DATA_ROOT}/finetuning_datasets/MoleculeACE"
#   bash tools/datasets/prepare_moleculeace_raw_batch.sh
#
# Optional: CANVAS=224  SKIP_EXISTING=1  (skip PNG if file already exists — prepare script already skips PNG)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHEMVL_DATA_ROOT:?Set CHEMVL_DATA_ROOT to your data root directory}"
MOLECULEACE_RAW_DIR="${MOLECULEACE_RAW_DIR:-${CHEMVL_DATA_ROOT}/finetuning_datasets/MoleculeACE/raw}"
MOLECULEACE_DATAROOT="${MOLECULEACE_DATAROOT:-${CHEMVL_DATA_ROOT}/finetuning_datasets/MoleculeACE}"
CANVAS="${CANVAS:-224}"

cd "$REPO_ROOT"

shopt -s nullglob
csv_list=("${MOLECULEACE_RAW_DIR}"/*.csv)
if ((${#csv_list[@]} == 0)); then
  echo "No CSV files under: ${MOLECULEACE_RAW_DIR}" >&2
  exit 1
fi

for f in "${csv_list[@]}"; do
  base="$(basename "$f" .csv)"
  echo "==> ${base}  (from ${f})"
  python tools/datasets/prepare_moleculeace_chemvl.py \
    --input-csv "$f" \
    --dataroot "$MOLECULEACE_DATAROOT" \
    --dataset "$base" \
    --canvas "$CANVAS" \
    --render-images
done

echo "All tasks processed. DATAROOT=${MOLECULEACE_DATAROOT}"
